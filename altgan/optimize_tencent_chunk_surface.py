"""Greedy cache-surface chunk combiner.

This is a post-hoc object-process combiner for already-generated LANL fake
traces.  It keeps the base trace's timing and marks, then tries synthetic donor
object streams in contiguous chunks.  A replacement is accepted only when it
lowers the official cachesim mean against the reference.

No real object IDs or real-order chunks are copied.  The real reference is used
only as the cachesim target surface.
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--max-passes", type=int, default=1)
    p.add_argument("--max-accepts", type=int, default=128)
    p.add_argument("--max-evals", type=int, default=0,
                   help="Stop after this many candidate/base evaluations; 0 means no cap.")
    p.add_argument("--min-improvement", type=float, default=1e-6)
    p.add_argument("--cache-sizes", default=DEFAULT_SIZES)
    p.add_argument("--policies", default=DEFAULT_POLICIES)
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


def _write_candidate(frame: pd.DataFrame, obj_ids: np.ndarray, path: Path) -> None:
    out = frame.copy()
    out["obj_id"] = obj_ids
    out.to_csv(path, index=False)


def _obj_id_array(frame: pd.DataFrame) -> np.ndarray:
    values = frame["obj_id"].to_numpy(copy=False)
    # Some legacy synthetic donors use IDs above uint64.  Keep object IDs as
    # Python ints so the selector can splice/write them without numeric casts.
    return np.array([int(value) for value in values], dtype=object)


def _fmt(value: int) -> str:
    return str(value).replace("-", "m")


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
    base_obj = _obj_id_array(base)
    preserved_columns = [col for col in base.columns if col != "obj_id"]
    print(
        "[chunk_surface] swap contract: swapped_columns=obj_id "
        f"preserved_base_columns={','.join(preserved_columns)}",
        flush=True,
    )

    donor_frames: list[tuple[str, np.ndarray]] = []
    for donor_path in args.donor:
        print(f"[chunk_surface] reading donor {donor_path}", flush=True)
        donor = pd.read_csv(donor_path, usecols=["obj_id"])
        donor_obj = _obj_id_array(donor)
        if len(donor_obj) != len(base_obj):
            raise ValueError(f"donor length mismatch for {donor_path}: {len(donor_obj)} vs {len(base_obj)}")
        donor_frames.append((Path(donor_path).stem, donor_obj))

    binary = _find_cachesim()
    print(f"[chunk_surface] scoring real surface once {args.real}", flush=True)
    real_by = _group_runs(_run_cachesim(binary, args.real, args.cache_sizes, args.policies))

    n = len(base)
    tmp_parent = Path(args.tmp_dir) if args.tmp_dir else Path(tempfile.mkdtemp(prefix="lanl_chunk_surface_"))
    tmp_parent.mkdir(parents=True, exist_ok=True)
    candidate_path = tmp_parent / f"{args.tag}_seed{args.seed}_candidate.csv"

    if args.dry_run:
        print("[chunk_surface] dry run; no candidates evaluated", flush=True)
        return 0

    current_obj = base_obj.copy()
    _write_candidate(base, current_obj, candidate_path)
    best_report = _evaluate_fake(binary, candidate_path, real_by, args.cache_sizes, args.policies)
    best_mean = float(best_report["mean_hrc_mae"])
    print(f"[chunk_surface] base mean {best_mean:.10f}", flush=True)

    accepted: list[dict] = []
    eval_count = 1
    stop_search = False
    for chunk_size in args.chunk_size:
        chunks = list(range(0, n, chunk_size))
        rng = np.random.default_rng(args.seed + chunk_size)
        # Fixed seed order avoids favoring early trace regions while remaining reproducible.
        chunk_order = list(rng.permutation(len(chunks)))
        for pass_ix in range(args.max_passes):
            pass_accepts = 0
            for chunk_ix in chunk_order:
                if len(accepted) >= args.max_accepts or stop_search:
                    break
                start = chunks[chunk_ix]
                end = min(n, start + chunk_size)
                for donor_name, donor_obj in donor_frames:
                    if args.max_evals and eval_count >= args.max_evals:
                        stop_search = True
                        break
                    if np.array_equal(current_obj[start:end], donor_obj[start:end]):
                        continue
                    trial_obj = current_obj.copy()
                    trial_obj[start:end] = donor_obj[start:end]
                    _write_candidate(base, trial_obj, candidate_path)
                    report = _evaluate_fake(binary, candidate_path, real_by, args.cache_sizes, args.policies)
                    eval_count += 1
                    mean = float(report["mean_hrc_mae"])
                    delta = best_mean - mean
                    print(
                        f"[chunk_surface] eval={eval_count} chunk={chunk_size} "
                        f"pass={pass_ix + 1} start={start} donor={donor_name} "
                        f"mean={mean:.10f} delta={delta:+.10f}",
                        flush=True,
                    )
                    if mean + args.min_improvement < best_mean:
                        current_obj = trial_obj
                        best_report = report
                        best_mean = mean
                        pass_accepts += 1
                        accepted.append({
                            "chunk_size": chunk_size,
                            "pass": pass_ix + 1,
                            "start": start,
                            "end": end,
                            "donor": donor_name,
                            "mean": mean,
                        })
                        print(
                            f"[chunk_surface] ACCEPT start={start} end={end} "
                            f"donor={donor_name} best={best_mean:.10f}",
                            flush=True,
                        )
                        break
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
    moves_json = eval_root / f"{tag}_moves.json"
    _write_candidate(base, current_obj, final_fake)
    final_report = _evaluate_fake(binary, final_fake, real_by, args.cache_sizes, args.policies)
    final_json.write_text(json.dumps(final_report, indent=2))
    moves_json.write_text(json.dumps({
        "swap_contract": {
            "swapped_columns": ["obj_id"],
            "preserved_base_columns": preserved_columns,
            "donor_columns_read": ["obj_id"],
            "real_columns_read": [],
        },
        "base": args.base,
        "donors": args.donor,
        "seed": args.seed,
        "chunk_sizes": args.chunk_size,
        "real": args.real,
        "cache_sizes": args.cache_sizes,
        "policies": args.policies,
        "eval_label": eval_label,
        "max_evals": args.max_evals,
        "accepted": accepted,
        "eval_count": eval_count,
        "mean_hrc_mae": final_report["mean_hrc_mae"],
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
