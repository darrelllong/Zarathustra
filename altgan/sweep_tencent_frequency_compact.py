"""Tencent frequency-compaction sweep.

Tencent's official cache surface has a sharp frequency head plus long scan
regions: ARC/CAR can protect the head, while LRU/FIFO/SLRU are punished by the
scans.  LANL's current PhaseAtlas fake matches the total reuse fraction but has
an overly flat object-count law.  This script keeps the generated timing/marks
and rewrites only synthetic object labels to test count-law architectures.

No real object IDs or real order are copied.  The real reference contributes
only the sorted object-count histogram used as a target frequency law.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_csv_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="LANL Tencent fake CSV to rewrite.")
    p.add_argument("--real", required=True, help="Official Tencent real CSV reference.")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tag", default="tencent_fc")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count-alpha", type=_parse_csv_floats, default=[1.0],
                   help="Comma-separated exponents for the target count law.")
    p.add_argument("--footprint-scale", type=_parse_csv_floats, default=[1.0],
                   help="Comma-separated multipliers for the real unique count.")
    p.add_argument("--mode", type=_parse_csv_strings,
                   default=["source_freq", "source_roundrobin", "shuffle"],
                   help="Comma-separated position assignment modes.")
    p.add_argument("--synthetic-base-id", type=int, default=8_000_000_000)
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _target_counts(
    real_counts: np.ndarray,
    n_records: int,
    footprint_scale: float,
    alpha: float,
) -> np.ndarray:
    footprint = max(1, min(n_records, int(round(len(real_counts) * footprint_scale))))
    if footprint <= len(real_counts):
        source = real_counts[:footprint].astype(np.float64, copy=True)
    else:
        tail = np.ones(footprint - len(real_counts), dtype=np.float64)
        source = np.concatenate([real_counts.astype(np.float64), tail])

    if (
        footprint == len(real_counts)
        and abs(alpha - 1.0) < 1e-12
        and int(real_counts.sum()) == n_records
    ):
        return real_counts.astype(np.int64, copy=True)

    weights = np.power(np.maximum(source, 1.0), alpha)
    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        weights = np.ones_like(weights)
        weights_sum = float(weights.sum())

    # Preserve the requested footprint exactly: every rank gets at least one
    # reference, and the remaining mass follows the exponentiated count law.
    remaining = max(0, n_records - footprint)
    raw_extra = weights / weights_sum * remaining
    extra = np.floor(raw_extra).astype(np.int64)
    diff = int(remaining - extra.sum())
    if diff > 0:
        order = np.argsort(-(raw_extra - extra), kind="stable")
        extra[order[:diff]] += 1
    counts = extra + 1
    if int(counts.sum()) != n_records:
        counts[0] += int(n_records - counts.sum())
    return counts.astype(np.int64, copy=False)


def _source_positions(base_ids: pd.Series, mode: str, rng: np.random.Generator) -> np.ndarray:
    n = len(base_ids)
    if mode == "shuffle":
        return rng.permutation(n).astype(np.int64, copy=False)

    grouped = base_ids.groupby(base_ids, sort=False).indices
    items = []
    for obj, positions in grouped.items():
        pos = np.asarray(positions, dtype=np.int64)
        items.append((obj, pos, int(pos[0]), int(len(pos))))

    if mode == "source_first":
        items.sort(key=lambda item: item[2])
        return np.concatenate([item[1] for item in items]).astype(np.int64, copy=False)

    if mode == "source_hash":
        # Deterministic per-seed merge order independent of pandas hash salt.
        items.sort(key=lambda item: (hash((int(item[0]), int(rng.bit_generator.random_raw()))) & 0xFFFFFFFFFFFFFFFF))
        return np.concatenate([item[1] for item in items]).astype(np.int64, copy=False)

    items.sort(key=lambda item: (-item[3], item[2]))
    if mode == "source_freq":
        return np.concatenate([item[1] for item in items]).astype(np.int64, copy=False)

    if mode == "source_roundrobin":
        out: list[int] = []
        max_count = max(item[3] for item in items)
        arrays = [item[1] for item in items]
        for ix in range(max_count):
            for pos in arrays:
                if ix < len(pos):
                    out.append(int(pos[ix]))
        return np.asarray(out, dtype=np.int64)

    raise ValueError(f"unknown assignment mode: {mode}")


def _rewrite(
    base: pd.DataFrame,
    real_counts: np.ndarray,
    *,
    seed: int,
    alpha: float,
    footprint_scale: float,
    mode: str,
    synthetic_base_id: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = _target_counts(real_counts, len(base), footprint_scale, alpha)
    positions = _source_positions(base["obj_id"], mode, rng)
    if len(positions) != len(base):
        raise RuntimeError(f"assignment produced {len(positions)} positions, expected {len(base)}")

    obj_out = np.empty(len(base), dtype=np.uint64)
    cursor = 0
    for rank, count in enumerate(counts):
        next_cursor = cursor + int(count)
        obj_out[positions[cursor:next_cursor]] = np.uint64(synthetic_base_id + rank)
        cursor = next_cursor
    if cursor != len(base):
        raise RuntimeError(f"assigned {cursor} records, expected {len(base)}")

    out = base.copy()
    out["obj_id"] = obj_out
    return out


def _run(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _eval_cmd(args: argparse.Namespace, fake: Path, out_json: Path) -> list[str]:
    return [
        sys.executable, "-u", "-m", "llgan.cachesim_eval",
        "--fake", str(fake),
        "--real", args.real,
        "--cache-sizes", args.cache_sizes,
        "--policies", args.policies,
        "--out", str(out_json),
    ]


def _mean(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return float(data.get("mean_hrc_mae", data.get("mean")))


def _fmt_float(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    eval_root = root / "cachesim_lanl"
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[tencent_fc] reading base {args.base}", flush=True)
    base = pd.read_csv(args.base)
    print(f"[tencent_fc] reading real histogram {args.real}", flush=True)
    real = pd.read_csv(args.real, usecols=["obj_id"])
    real_counts = real["obj_id"].value_counts(sort=True).to_numpy(dtype=np.int64)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"

    summary: list[tuple[str, float | None]] = []
    for mode in args.mode:
        for fp_scale in args.footprint_scale:
            for alpha in args.count_alpha:
                tag = (
                    f"{args.tag}_{mode}_a{_fmt_float(alpha)}_fp{_fmt_float(fp_scale)}"
                    f"_seed{args.seed}"
                )
                fake = root / f"{tag}_fake_{len(base) // 1000}k.csv"
                out_json = eval_root / f"{tag}_official6.json"
                if args.skip_existing and out_json.exists():
                    print(f"[tencent_fc] skip existing {out_json}", flush=True)
                else:
                    print(
                        f"[tencent_fc] generating mode={mode} alpha={alpha:g} "
                        f"footprint_scale={fp_scale:g}",
                        flush=True,
                    )
                    if not args.dry_run:
                        out = _rewrite(
                            base,
                            real_counts,
                            seed=args.seed,
                            alpha=alpha,
                            footprint_scale=fp_scale,
                            mode=mode,
                            synthetic_base_id=args.synthetic_base_id,
                        )
                        out.to_csv(fake, index=False)
                        print(f"[tencent_fc] wrote {fake}", flush=True)
                    _run(_eval_cmd(args, fake, out_json), args.dry_run, env)
                mean = _mean(out_json)
                summary.append((tag, mean))
                if mean is not None:
                    print(f"[tencent_fc] {tag}: {mean:.10f}", flush=True)

    print("\n=== TENCENT FREQUENCY-COMPACTION RESULTS ===", flush=True)
    for tag, mean in sorted(summary, key=lambda item: float("inf") if item[1] is None else item[1]):
        text = "pending" if mean is None else f"{mean:.10f}"
        print(f"{tag:<72} {text}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
