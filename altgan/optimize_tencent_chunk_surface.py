"""Greedy cache-surface chunk combiner.

This is a post-hoc object-process combiner for already-generated LANL fake
traces.  By default it keeps the base trace's timing and marks, then tries
synthetic donor object streams in contiguous chunks.  A replacement is accepted
only when it lowers the official cachesim mean against the reference.

`--swap-columns` can widen the disclosed synthetic-donor contract, e.g.
`--swap-columns obj_id,obj_size` for an object-ID-plus-size architecture scout.

No real object IDs or real-order chunks are copied.  The real reference is used
only as the cachesim target surface.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from llgan.cachesim_eval import (
    DEFAULT_POLICIES,
    DEFAULT_SIZES,
    _find_cachesim,
    _run_cachesim,
    print_report,
)


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_paths(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_columns(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Base LANL fake CSV.")
    p.add_argument("--donor", required=True, type=_parse_paths,
                   help="Comma-separated synthetic donor CSVs.")
    p.add_argument("--real", required=True, help="Official real CSV reference.")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tmp-dir", default=None,
                   help="Local scratch for candidate CSVs; default uses tempfile.")
    p.add_argument("--tag", default="chunk_surface")
    p.add_argument("--eval-label", default=None,
                   help="Suffix for eval JSONs; default is official<N policies>.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=_parse_ints, default=[2048])
    p.add_argument("--start-stride", type=int, default=0,
                   help="Candidate start stride in rows; default 0 uses the current chunk size.")
    p.add_argument("--max-passes", type=int, default=1)
    p.add_argument("--max-accepts", type=int, default=128)
    p.add_argument("--max-evals", type=int, default=0,
                   help="Stop after this many candidate/base evaluations; 0 means no cap.")
    p.add_argument("--min-improvement", type=float, default=1e-6)
    p.add_argument("--accept-mode", choices=["first", "best"], default="first",
                   help="Accept the first improving donor per chunk, or scan all donors and accept the best.")
    p.add_argument("--max-candidates-per-chunk", type=int, default=0,
                   help=(
                       "For accept-mode=best, evaluate at most this many donor/shift candidates per chunk. "
                       "Default 0 scans all donors/shifts. A positive cap makes large donor pools search "
                       "many more trace regions under a fixed --max-evals budget."
                   ))
    p.add_argument("--priority-moves", type=_parse_paths, default=[],
                   help=(
                       "Comma-separated move JSON files or globs from previous chunk-surface runs. "
                       "Accepted donor/start/shift moves from these files are evaluated first for "
                       "matching chunk starts before the random candidate cap is filled."
                   ))
    p.add_argument("--swap-columns", type=_parse_columns, default=["obj_id"],
                   help="Comma-separated donor columns to splice; default obj_id.")
    p.add_argument(
        "--write-columns",
        type=_parse_columns,
        default=[],
        help=(
            "Optional comma-separated subset of base columns to write into candidate/final CSVs. "
            "Default writes all base columns. Must include every column in --swap-columns. "
            "For cachesim-only optimization you can usually restrict to `stream_id,obj_id,ts` "
            "(plus any other swapped columns) to cut disk I/O substantially; note that omitting "
            "`ts` makes cachesim fall back to row index ordering."
        ),
    )
    p.add_argument("--donor-shifts", type=_parse_ints, default=[0],
                   help=(
                       "Comma-separated row offsets for donor chunks. Default 0 preserves "
                       "the aligned-chunk contract; nonzero shifts test donor[start+shift:end+shift]."
                   ))
    p.add_argument("--cache-sizes", default=DEFAULT_SIZES)
    p.add_argument("--policies", default=DEFAULT_POLICIES)
    p.add_argument("--guard-cache-sizes", default="",
                   help="Optional secondary cache-size surface candidates must not regress.")
    p.add_argument("--guard-policies", default="",
                   help="Policies for --guard-cache-sizes; defaults to --policies.")
    p.add_argument("--guard-max-regression", type=float, default=0.0,
                   help="Allowed guard mean regression for an otherwise improving candidate.")
    p.add_argument("--guard-regression-per-official-gain", type=float, default=0.0,
                   help=(
                       "Allow additional guard regression proportional to the official-surface "
                       "gain. For example 0.25 admits a guard increase of up to one quarter "
                       "of the candidate's official mean improvement, plus --guard-max-regression."
                   ))
    p.add_argument("--guard-eval-label", default="guard",
                   help="Suffix for the final guard eval JSON when --guard-cache-sizes is set.")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _group_runs(runs: list[dict]) -> dict[str, list[tuple[int, float]]]:
    grouped: dict[str, list[tuple[int, float]]] = {}
    for run in runs:
        policy = run["policy"]
        for item in run["per_cache_size"]:
            grouped.setdefault(policy, []).append((int(item["size"]), float(item["miss_ratio"])))
    for rows in grouped.values():
        rows.sort()
    return grouped


def _report_from_fake_runs(
    fake_runs: list[dict],
    real_by: dict[str, list[tuple[int, float]]],
    sizes: str,
) -> dict:
    fake_by = _group_runs(fake_runs)
    by_policy: dict[str, dict] = {}
    for policy, fake_pairs in fake_by.items():
        real_pairs = real_by.get(policy, [])
        fake_mr = [mr for _, mr in fake_pairs]
        real_mr = [mr for _, mr in real_pairs]
        deltas = [fake - real for fake, real in zip(fake_mr, real_mr)]
        by_policy[policy] = {
            "fake_miss_ratio": fake_mr,
            "real_miss_ratio": real_mr,
            "delta": deltas,
            "hrc_mae": sum(abs(delta) for delta in deltas) / max(len(deltas), 1),
        }
    mean_hrc_mae = sum(policy["hrc_mae"] for policy in by_policy.values()) / len(by_policy)
    return {
        "cache_sizes": [int(size) for size in sizes.split(",") if size],
        "policies": list(by_policy.keys()),
        "by_policy": by_policy,
        "mean_hrc_mae": mean_hrc_mae,
    }


def _evaluate_fake(
    binary: str,
    fake: Path,
    real_by: dict[str, list[tuple[int, float]]],
    sizes: str,
    policies: str,
) -> dict:
    fake_runs = _run_cachesim(binary, str(fake), sizes, policies)
    return _report_from_fake_runs(fake_runs, real_by, sizes)


def _write_candidate(frame: pd.DataFrame, swap_values: dict[str, np.ndarray], path: Path) -> None:
    out = frame.copy()
    for column, values in swap_values.items():
        out[column] = values
    out.to_csv(path, index=False)


def _column_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = frame[column].to_numpy(copy=False)
    # Some legacy synthetic donors use IDs above uint64.  Keep object IDs as
    # Python ints so the selector can splice/write them without numeric casts.
    if column in {"obj_id", "obj_size", "stack_distance"}:
        return np.array([int(value) for value in values], dtype=object)
    return np.array(values, copy=True)


def _copy_swap_values(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {column: array.copy() for column, array in values.items()}


def _load_priority_moves(
    patterns: list[str],
    donor_by_name: dict[str, dict[str, np.ndarray]],
) -> dict[tuple[int, int], list[tuple[str, dict[str, np.ndarray], int]]]:
    by_chunk_start: dict[tuple[int, int], list[tuple[str, dict[str, np.ndarray], int]]] = {}
    loaded_files = 0
    loaded_moves = 0
    skipped_moves = 0
    for pattern in patterns:
        paths = sorted(glob.glob(pattern)) or [pattern]
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                print(f"[chunk_surface] SKIP priority moves missing {path}", flush=True)
                continue
            loaded_files += 1
            with path.open() as f:
                data = json.load(f)
            for move in data.get("accepted", []):
                donor_name = str(move.get("donor", ""))
                donor_swap = donor_by_name.get(donor_name)
                if donor_swap is None:
                    skipped_moves += 1
                    continue
                try:
                    chunk_size = int(move["chunk_size"])
                    start = int(move["start"])
                    donor_shift = int(move.get("donor_shift", int(move["source_start"]) - start))
                except (KeyError, TypeError, ValueError):
                    skipped_moves += 1
                    continue
                by_chunk_start.setdefault((chunk_size, start), []).append((donor_name, donor_swap, donor_shift))
                loaded_moves += 1
    for key, rows in list(by_chunk_start.items()):
        seen: set[tuple[str, int]] = set()
        deduped: list[tuple[str, dict[str, np.ndarray], int]] = []
        for donor_name, donor_swap, donor_shift in rows:
            pair = (donor_name, donor_shift)
            if pair in seen:
                continue
            seen.add(pair)
            deduped.append((donor_name, donor_swap, donor_shift))
        by_chunk_start[key] = deduped
    if patterns:
        print(
            f"[chunk_surface] priority moves: files={loaded_files} "
            f"usable={loaded_moves} skipped={skipped_moves} "
            f"target_chunks={len(by_chunk_start)}",
            flush=True,
        )
    return by_chunk_start


def _segments_equal(
    current: dict[str, np.ndarray],
    donor: dict[str, np.ndarray],
    start: int,
    end: int,
    source_start: int | None = None,
) -> bool:
    if source_start is None:
        source_start = start
    source_end = source_start + (end - start)
    return all(np.array_equal(current[column][start:end], donor[column][source_start:source_end]) for column in current)


def _fmt(value: int) -> str:
    return str(value).replace("-", "m")


def _guard_allows(
    *,
    guard_mean: float,
    best_guard_mean: float,
    max_regression: float,
    official_gain: float,
    regression_per_official_gain: float,
) -> bool:
    allowed = max_regression + max(0.0, official_gain) * max(0.0, regression_per_official_gain)
    return guard_mean <= best_guard_mean + allowed


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    eval_root = root / "cachesim_lanl"
    root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    os.environ.update(env)

    print(f"[chunk_surface] reading base {args.base}", flush=True)
    base = pd.read_csv(args.base)
    swap_columns = list(dict.fromkeys(args.swap_columns))
    if not swap_columns:
        raise ValueError("--swap-columns parsed to an empty column list")
    missing_base = [column for column in swap_columns if column not in base.columns]
    if missing_base:
        raise ValueError(f"base missing swap column(s): {missing_base}")
    base_swap = {column: _column_array(base, column) for column in swap_columns}
    base_len = len(next(iter(base_swap.values())))
    preserved_columns = [col for col in base.columns if col not in swap_columns]
    write_columns = list(dict.fromkeys(args.write_columns))
    if write_columns:
        missing = [col for col in write_columns if col not in base.columns]
        if missing:
            raise ValueError(f"--write-columns contains unknown base column(s): {missing}")
        missing_swapped = [col for col in swap_columns if col not in write_columns]
        if missing_swapped:
            raise ValueError(
                "--write-columns must include all --swap-columns; missing: "
                + ",".join(missing_swapped)
            )
        base_out = base[write_columns]
        print(f"[chunk_surface] write-columns: {','.join(write_columns)}", flush=True)
    else:
        base_out = base
        write_columns = list(base.columns)
    print(
        f"[chunk_surface] swap contract: swapped_columns={','.join(swap_columns)} "
        f"preserved_base_columns={','.join(preserved_columns)}",
        flush=True,
    )

    donor_frames: list[tuple[str, dict[str, np.ndarray]]] = []
    for donor_path in args.donor:
        print(f"[chunk_surface] reading donor {donor_path}", flush=True)
        donor = pd.read_csv(donor_path, usecols=swap_columns)
        donor_swap = {column: _column_array(donor, column) for column in swap_columns}
        donor_len = len(next(iter(donor_swap.values())))
        if donor_len != base_len:
            # Some donor artifacts can be truncated (e.g., interrupted writes). Failing hard here
            # burns an entire multi-seed run; skip and continue instead.
            print(
                f"[chunk_surface] SKIP donor length mismatch for {donor_path}: {donor_len} vs {base_len}",
                flush=True,
            )
            continue
        donor_frames.append((Path(donor_path).stem, donor_swap))
    donor_shifts = list(dict.fromkeys(int(shift) for shift in args.donor_shifts))
    if not donor_shifts:
        donor_shifts = [0]
    print(f"[chunk_surface] donor shifts: {','.join(str(shift) for shift in donor_shifts)}", flush=True)
    donor_by_name: dict[str, dict[str, np.ndarray]] = {}
    for donor_name, donor_swap in donor_frames:
        donor_by_name.setdefault(donor_name, donor_swap)
    priority_moves = _load_priority_moves(args.priority_moves, donor_by_name)

    binary = _find_cachesim()
    print(f"[chunk_surface] scoring real surface once {args.real}", flush=True)
    real_by = _group_runs(_run_cachesim(binary, args.real, args.cache_sizes, args.policies))
    guard_sizes = args.guard_cache_sizes.strip()
    guard_policies = (args.guard_policies or args.policies).strip()
    guard_real_by = None
    if guard_sizes:
        print(
            f"[chunk_surface] scoring guard real surface once {args.real} "
            f"cache_sizes={guard_sizes} policies={guard_policies}",
            flush=True,
        )
        guard_real_by = _group_runs(_run_cachesim(binary, args.real, guard_sizes, guard_policies))

    n = len(base)
    tmp_parent = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="lanl_chunk_surface_"))
    tmp_parent.mkdir(parents=True, exist_ok=True)
    candidate_path = tmp_parent / f"{args.tag}_seed{args.seed}_candidate.csv"

    if args.dry_run:
        print("[chunk_surface] dry run; no candidates evaluated", flush=True)
        return 0

    current_swap = _copy_swap_values(base_swap)
    _write_candidate(base_out, current_swap, candidate_path)
    best_report = _evaluate_fake(binary, candidate_path, real_by, args.cache_sizes, args.policies)
    best_mean = float(best_report["mean_hrc_mae"])
    print(f"[chunk_surface] base mean {best_mean:.10f}", flush=True)
    best_guard_report = None
    best_guard_mean = None
    if guard_sizes and guard_real_by is not None:
        best_guard_report = _evaluate_fake(binary, candidate_path, guard_real_by, guard_sizes, guard_policies)
        best_guard_mean = float(best_guard_report["mean_hrc_mae"])
        print(f"[chunk_surface] base guard mean {best_guard_mean:.10f}", flush=True)

    accepted: list[dict] = []
    eval_count = 1
    stop_search = False
    for chunk_size in args.chunk_size:
        start_stride = args.start_stride if args.start_stride and args.start_stride > 0 else chunk_size
        chunks = list(range(0, n, start_stride))
        chunk_index_by_start = {start: ix for ix, start in enumerate(chunks)}
        rng = np.random.default_rng(args.seed + chunk_size)
        # Fixed seed order avoids favoring early trace regions while remaining reproducible.
        priority_chunk_indices = [
            chunk_index_by_start[start]
            for move_chunk_size, start in priority_moves
            if move_chunk_size == chunk_size and start in chunk_index_by_start
        ]
        rest_chunk_indices = [ix for ix in rng.permutation(len(chunks)) if ix not in set(priority_chunk_indices)]
        chunk_order = list(dict.fromkeys(priority_chunk_indices + rest_chunk_indices))
        print(
            f"[chunk_surface] start grid chunk={chunk_size} stride={start_stride} "
            f"candidates={len(chunks)} priority_chunks={len(priority_chunk_indices)} "
            f"max_candidates_per_chunk={args.max_candidates_per_chunk}",
            flush=True,
        )
        for pass_ix in range(args.max_passes):
            pass_accepts = 0
            for chunk_ix in chunk_order:
                if len(accepted) >= args.max_accepts or stop_search:
                    break
                start = chunks[chunk_ix]
                end = min(n, start + chunk_size)
                best_chunk_swap = None
                best_chunk_report = None
                best_chunk_guard_report = None
                best_chunk_mean = best_mean
                best_chunk_guard_mean = best_guard_mean
                best_chunk_donor = None
                best_chunk_source_start = None
                best_chunk_donor_shift = None
                priority_pairs = priority_moves.get((chunk_size, start), [])
                all_candidate_pairs = [
                    (donor_name, donor_swap, donor_shift)
                    for donor_name, donor_swap in donor_frames
                    for donor_shift in donor_shifts
                ]
                if args.max_candidates_per_chunk > 0:
                    cap = args.max_candidates_per_chunk
                    if len(priority_pairs) >= cap:
                        candidate_pairs = list(priority_pairs[:cap])
                    else:
                        priority_seen = {(name, shift) for name, _swap, shift in priority_pairs}
                        remaining = [
                            pair for pair in all_candidate_pairs
                            if (pair[0], pair[2]) not in priority_seen
                        ]
                        take_n = min(cap - len(priority_pairs), len(remaining))
                        if take_n:
                            take = rng.permutation(len(remaining))[:take_n]
                            sampled = [remaining[int(ix)] for ix in take]
                        else:
                            sampled = []
                        candidate_pairs = list(priority_pairs) + sampled
                else:
                    priority_seen = {(name, shift) for name, _swap, shift in priority_pairs}
                    candidate_pairs = list(priority_pairs) + [
                        pair for pair in all_candidate_pairs
                        if (pair[0], pair[2]) not in priority_seen
                    ]
                for donor_name, donor_swap, donor_shift in candidate_pairs:
                        if args.max_evals and eval_count >= args.max_evals:
                            stop_search = True
                            break
                        source_start = start + donor_shift
                        source_end = source_start + (end - start)
                        if source_start < 0 or source_end > n:
                            continue
                        if _segments_equal(current_swap, donor_swap, start, end, source_start):
                            continue
                        trial_swap = _copy_swap_values(current_swap)
                        for column in swap_columns:
                            trial_swap[column][start:end] = donor_swap[column][source_start:source_end]
                        _write_candidate(base_out, trial_swap, candidate_path)
                        eval_count += 1
                        try:
                            report = _evaluate_fake(binary, candidate_path, real_by, args.cache_sizes, args.policies)
                        except RuntimeError as exc:
                            first_line = str(exc).splitlines()[0] if str(exc) else "unknown error"
                            print(
                                f"[chunk_surface] SKIP invalid candidate eval={eval_count} "
                                f"chunk={chunk_size} start={start} donor={donor_name} "
                                f"shift={donor_shift} reason={first_line}",
                                flush=True,
                            )
                            continue
                        mean = float(report["mean_hrc_mae"])
                        delta = best_mean - mean
                        print(
                            f"[chunk_surface] eval={eval_count} chunk={chunk_size} "
                            f"pass={pass_ix + 1} start={start} donor={donor_name} "
                            f"shift={donor_shift} mean={mean:.10f} delta={delta:+.10f}",
                            flush=True,
                        )
                        if mean + args.min_improvement < best_mean:
                            guard_report = None
                            guard_mean = None
                            if guard_sizes and guard_real_by is not None and best_guard_mean is not None:
                                guard_report = _evaluate_fake(
                                    binary,
                                    candidate_path,
                                    guard_real_by,
                                    guard_sizes,
                                    guard_policies,
                                )
                                guard_mean = float(guard_report["mean_hrc_mae"])
                                if not _guard_allows(
                                    guard_mean=guard_mean,
                                    best_guard_mean=best_guard_mean,
                                    max_regression=args.guard_max_regression,
                                    official_gain=delta,
                                    regression_per_official_gain=args.guard_regression_per_official_gain,
                                ):
                                    allowed_guard = (
                                        args.guard_max_regression
                                        + max(0.0, delta)
                                        * max(0.0, args.guard_regression_per_official_gain)
                                    )
                                    print(
                                        f"[chunk_surface] REJECT guard start={start} "
                                        f"donor={donor_name} shift={donor_shift} "
                                        f"guard={guard_mean:.10f} best_guard={best_guard_mean:.10f} "
                                        f"allowed_regression={allowed_guard:.10f}",
                                        flush=True,
                                    )
                                    continue
                            if args.accept_mode == "best":
                                if mean + args.min_improvement < best_chunk_mean:
                                    best_chunk_swap = trial_swap
                                    best_chunk_report = report
                                    best_chunk_guard_report = guard_report
                                    best_chunk_mean = mean
                                    best_chunk_guard_mean = guard_mean
                                    best_chunk_donor = donor_name
                                    best_chunk_source_start = source_start
                                    best_chunk_donor_shift = donor_shift
                                continue
                            current_swap = trial_swap
                            best_report = report
                            best_mean = mean
                            if guard_report is not None:
                                best_guard_report = guard_report
                                best_guard_mean = guard_mean
                            pass_accepts += 1
                            move = {
                                "chunk_size": chunk_size,
                                "pass": pass_ix + 1,
                                "start": start,
                                "end": end,
                                "source_start": source_start,
                                "source_end": source_end,
                                "donor_shift": donor_shift,
                                "donor": donor_name,
                                "mean": mean,
                            }
                            if guard_mean is not None:
                                move["guard_mean"] = guard_mean
                            accepted.append(move)
                            print(
                                f"[chunk_surface] ACCEPT start={start} end={end} "
                                f"donor={donor_name} shift={donor_shift} best={best_mean:.10f}",
                                flush=True,
                            )
                            break
                if stop_search:
                    break
                if (
                    args.accept_mode == "best"
                    and best_chunk_swap is not None
                    and best_chunk_report is not None
                    and best_chunk_donor is not None
                ):
                    current_swap = best_chunk_swap
                    best_report = best_chunk_report
                    best_mean = best_chunk_mean
                    if best_chunk_guard_report is not None:
                        best_guard_report = best_chunk_guard_report
                        best_guard_mean = best_chunk_guard_mean
                    pass_accepts += 1
                    move = {
                        "chunk_size": chunk_size,
                        "pass": pass_ix + 1,
                        "start": start,
                        "end": end,
                        "source_start": best_chunk_source_start,
                        "source_end": (
                            best_chunk_source_start + (end - start)
                            if best_chunk_source_start is not None
                            else None
                        ),
                        "donor_shift": best_chunk_donor_shift,
                        "donor": best_chunk_donor,
                        "mean": best_mean,
                    }
                    if best_chunk_guard_mean is not None:
                        move["guard_mean"] = best_chunk_guard_mean
                    accepted.append(move)
                    print(
                        f"[chunk_surface] ACCEPT start={start} end={end} "
                        f"donor={best_chunk_donor} shift={best_chunk_donor_shift} "
                        f"best={best_mean:.10f}",
                        flush=True,
                    )
                if stop_search:
                    print(
                        f"[chunk_surface] max evals reached eval_count={eval_count} "
                        f"limit={args.max_evals}",
                        flush=True,
                    )
                    break
            print(
                f"[chunk_surface] pass done chunk={chunk_size} pass={pass_ix + 1} "
                f"accepts={pass_accepts} best={best_mean:.10f}",
                flush=True,
            )
            if pass_accepts == 0 or stop_search:
                break
        if len(accepted) >= args.max_accepts or stop_search:
            break

    chunk_label = "-".join(_fmt(size) for size in args.chunk_size)
    tag = f"{args.tag}_ck{chunk_label}_seed{args.seed}"
    policy_count = len([part for part in args.policies.split(",") if part.strip()])
    eval_label = args.eval_label or f"official{policy_count}"
    final_fake = root / f"{tag}_fake_{n // 1000}k.csv"
    final_json = eval_root / f"{tag}_{eval_label}.json"
    final_guard_json = eval_root / f"{tag}_{args.guard_eval_label}.json" if guard_sizes else None
    moves_json = eval_root / f"{tag}_moves.json"
    _write_candidate(base_out, current_swap, final_fake)
    final_report = _evaluate_fake(binary, final_fake, real_by, args.cache_sizes, args.policies)
    final_json.write_text(json.dumps(final_report, indent=2))
    final_guard_report = None
    if guard_sizes and guard_real_by is not None and final_guard_json is not None:
        final_guard_report = _evaluate_fake(binary, final_fake, guard_real_by, guard_sizes, guard_policies)
        final_guard_json.write_text(json.dumps(final_guard_report, indent=2))
    moves_json.write_text(json.dumps({
        "swap_contract": {
            "swapped_columns": swap_columns,
            "preserved_base_columns": preserved_columns,
            "donor_columns_read": swap_columns,
            "base_columns_written": write_columns,
            "real_columns_read": [],
        },
        "base": args.base,
        "donors": args.donor,
        "seed": args.seed,
        "chunk_sizes": args.chunk_size,
        "start_stride": args.start_stride,
        "real": args.real,
        "cache_sizes": args.cache_sizes,
        "policies": args.policies,
        "eval_label": eval_label,
        "guard_cache_sizes": guard_sizes,
        "guard_policies": guard_policies if guard_sizes else "",
        "guard_eval_label": args.guard_eval_label if guard_sizes else "",
        "guard_max_regression": args.guard_max_regression if guard_sizes else None,
        "guard_regression_per_official_gain": (
            args.guard_regression_per_official_gain if guard_sizes else None
        ),
        "accept_mode": args.accept_mode,
        "max_candidates_per_chunk": args.max_candidates_per_chunk,
        "priority_moves": args.priority_moves,
        "donor_shifts": donor_shifts,
        "max_evals": args.max_evals,
        "accepted": accepted,
        "eval_count": eval_count,
        "mean_hrc_mae": final_report["mean_hrc_mae"],
        "guard_mean_hrc_mae": (
            final_guard_report["mean_hrc_mae"] if final_guard_report is not None else None
        ),
    }, indent=2))

    print(f"[chunk_surface] wrote {final_fake}", flush=True)
    print("+ " + " ".join(shlex.quote(part) for part in [
        sys.executable, "-m", "llgan.cachesim_eval",
        "--fake", str(final_fake),
        "--real", args.real,
        "--cache-sizes", args.cache_sizes,
        "--policies", args.policies,
        "--out", str(final_json),
    ]), flush=True)
    print_report(final_report)
    print(f"\nReport JSON: {final_json}", flush=True)
    if final_guard_json is not None and final_guard_report is not None:
        print(
            f"[chunk_surface] guard mean {float(final_guard_report['mean_hrc_mae']):.10f}",
            flush=True,
        )
        print(f"[chunk_surface] guard JSON: {final_guard_json}", flush=True)
    print(f"[chunk_surface] moves JSON: {moves_json}", flush=True)
    print(f"[chunk_surface] accepted={len(accepted)} eval_count={eval_count}", flush=True)
    print(f"[chunk_surface] final mean {float(final_report['mean_hrc_mae']):.10f}", flush=True)

    if not args.keep_candidates:
        try:
            candidate_path.unlink()
            if not args.tmp_dir:
                tmp_parent.rmdir()
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
