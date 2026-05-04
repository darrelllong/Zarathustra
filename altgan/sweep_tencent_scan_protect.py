"""Tencent scan/protected-hot generator sweep.

The standing Tencent fake is too smooth across cache policies: LRU/FIFO and
ARC/CAR miss curves sit close together, while the real reference has scan runs
that punish recency-only policies and a protected frequency head that adaptive
policies can keep.  This generator emits a two-state object process:

  scan phase: fresh cold IDs, usually one reference each
  hot phase: repeated synthetic hot IDs sampled from a Zipf-like head

Marks/timestamps come from an existing LANL Tencent fake, but object labels are
fresh synthetic IDs.  The real reference is used only for row count and target
footprint.
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


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="CSV providing marks/timestamps.")
    p.add_argument("--real", required=True, help="Official Tencent real CSV reference.")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tag", default="tencent_scanprotect")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hot-size", type=_parse_ints, default=[64, 128, 256])
    p.add_argument("--scan-len", type=_parse_ints, default=[128, 256, 512])
    p.add_argument("--hot-mult", type=_parse_floats, default=[1.5])
    p.add_argument("--zipf", type=_parse_floats, default=[0.5, 1.0])
    p.add_argument("--footprint-scale", type=_parse_floats, default=[1.0])
    p.add_argument("--cold-reuse-prob", type=_parse_floats, default=[0.0],
                   help="Probability a scan reference reuses a prior cold ID.")
    p.add_argument("--synthetic-base-id", type=int, default=9_000_000_000)
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _hot_probs(hot_size: int, zipf: float) -> np.ndarray:
    ranks = np.arange(1, hot_size + 1, dtype=np.float64)
    weights = np.power(ranks, -zipf) if zipf > 0 else np.ones(hot_size, dtype=np.float64)
    return weights / weights.sum()


def _generate_ids(
    n_records: int,
    target_footprint: int,
    *,
    hot_size: int,
    scan_len: int,
    hot_mult: float,
    zipf: float,
    cold_reuse_prob: float,
    synthetic_base_id: int,
    seed: int,
) -> np.ndarray:
    if hot_size >= target_footprint:
        raise ValueError("hot_size must be smaller than target footprint")
    rng = np.random.default_rng(seed)
    hot_ids = np.arange(synthetic_base_id, synthetic_base_id + hot_size, dtype=np.uint64)
    cold_base = synthetic_base_id + 10_000_000
    cold_target = max(1, target_footprint - hot_size)
    hot_len = max(1, int(round(scan_len * hot_mult)))
    probs = _hot_probs(hot_size, zipf)

    out = np.empty(n_records, dtype=np.uint64)
    cold_seen: list[np.uint64] = []
    cold_cursor = 0
    pos = 0
    prev_hot = np.uint64(0)
    have_prev_hot = False

    def emit_cold() -> np.uint64:
        nonlocal cold_cursor
        if cold_seen and rng.random() < cold_reuse_prob:
            return cold_seen[int(rng.integers(0, len(cold_seen)))]
        if cold_cursor < cold_target:
            obj = np.uint64(cold_base + cold_cursor)
            cold_cursor += 1
            cold_seen.append(obj)
            return obj
        # After the target footprint is exhausted, recycle cold IDs sparsely
        # instead of minting more footprint.
        return cold_seen[int(rng.integers(0, len(cold_seen)))]

    def emit_hot() -> np.uint64:
        nonlocal prev_hot, have_prev_hot
        for _ in range(8):
            idx = int(rng.choice(hot_size, p=probs))
            obj = hot_ids[idx]
            if not have_prev_hot or obj != prev_hot:
                prev_hot = obj
                have_prev_hot = True
                return obj
        idx = int(rng.integers(0, hot_size))
        obj = hot_ids[idx]
        prev_hot = obj
        have_prev_hot = True
        return obj

    while pos < n_records:
        for _ in range(scan_len):
            if pos >= n_records:
                break
            out[pos] = emit_cold()
            pos += 1
        for _ in range(hot_len):
            if pos >= n_records:
                break
            out[pos] = emit_hot()
            pos += 1
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


def _fmt(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    eval_root = root / "cachesim_lanl"
    root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base)
    real = pd.read_csv(args.real, usecols=["obj_id"])
    n_records = len(base)
    real_footprint = int(real["obj_id"].nunique())

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"

    summary: list[tuple[str, float | None]] = []
    for fp_scale in args.footprint_scale:
        target_footprint = max(1, min(n_records, int(round(real_footprint * fp_scale))))
        for hot_size in args.hot_size:
            for scan_len in args.scan_len:
                for hot_mult in args.hot_mult:
                    for zipf in args.zipf:
                        for cold_reuse in args.cold_reuse_prob:
                            tag = (
                                f"{args.tag}_h{hot_size}_s{scan_len}_hm{_fmt(hot_mult)}"
                                f"_z{_fmt(zipf)}_fp{_fmt(fp_scale)}_cr{_fmt(cold_reuse)}"
                                f"_seed{args.seed}"
                            )
                            fake = root / f"{tag}_fake_{n_records // 1000}k.csv"
                            out_json = eval_root / f"{tag}_official6.json"
                            if args.skip_existing and out_json.exists():
                                print(f"[scanprotect] skip existing {out_json}", flush=True)
                            else:
                                print(
                                    f"[scanprotect] generating hot={hot_size} scan={scan_len} "
                                    f"hot_mult={hot_mult:g} zipf={zipf:g} "
                                    f"footprint={target_footprint} cold_reuse={cold_reuse:g}",
                                    flush=True,
                                )
                                if not args.dry_run:
                                    out = base.copy()
                                    out["obj_id"] = _generate_ids(
                                        n_records,
                                        target_footprint,
                                        hot_size=hot_size,
                                        scan_len=scan_len,
                                        hot_mult=hot_mult,
                                        zipf=zipf,
                                        cold_reuse_prob=cold_reuse,
                                        synthetic_base_id=args.synthetic_base_id,
                                        seed=args.seed,
                                    )
                                    out.to_csv(fake, index=False)
                                    print(f"[scanprotect] wrote {fake}", flush=True)
                                _run(_eval_cmd(args, fake, out_json), args.dry_run, env)
                            mean = _mean(out_json)
                            summary.append((tag, mean))
                            if mean is not None:
                                print(f"[scanprotect] {tag}: {mean:.10f}", flush=True)

    print("\n=== TENCENT SCAN-PROTECT RESULTS ===", flush=True)
    for tag, mean in sorted(summary, key=lambda item: float("inf") if item[1] is None else item[1]):
        text = "pending" if mean is None else f"{mean:.10f}"
        print(f"{tag:<84} {text}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
