"""IDEA #67 burst injection sweep for Phase-PMF Atlas.

Sweeps --burst-prob values over the nophase atlas and reports HRC-MAE vs real
at each setting.  Finds the optimal temporal-clustering fix without training.

Root cause addressed (IDEA #67): marginal PMF sampling has no temporal burst
structure.  At cs=18: real HRC@18=0.056, LLNL atlas (no burst)=0.0007 (80×
worse).  Burst injection samples from the top-K LRU items with probability
burst_prob (within reuse events), adding working-set autocorrelation.

Usage on vinge.local:
    python -m llgan.burst_probe \\
        --atlas /home/darrell/llnl_phase_pmf_atlas_nophase.pkl.gz \\
        --real-json /home/darrell/llnl_phase_eval_nophase.json \\
        --corpus alibaba \\
        [--burst-probs 0.0 0.05 0.1 0.2 0.3 0.5] \\
        [--burst-pool-sizes 10 20] \\
        [--seeds 42 43 44] \\
        [--n-records 100000] [--n-streams 4] \\
        [--output /home/darrell/burst_probe_alibaba.json]

    # Tencent (use LANL-setup real JSON):
    python -m llgan.burst_probe \\
        --atlas /home/darrell/llnl_phase_pmf_atlas_tencent_lanl.pkl.gz \\
        --real-json /tiamat/zarathustra/checkpoints/tencent_v165/long_rollout_lanl_setup_real.json \\
        --corpus tencent \\
        --burst-probs 0.0 0.1 0.2 0.3 \\
        --n-records 100000 --n-streams 4 \\
        --output /home/darrell/burst_probe_tencent.json

Expected findings (theoretical estimate for alibaba):
  burst_prob=0.0  → HRC-MAE ≈ 0.021  (LANL-methodology baseline, no burst)
  burst_prob=0.1  → HRC-MAE ≈ 0.008  (approaching LANL's 0.00183)
  burst_prob=0.2  → HRC-MAE ≈ 0.003  (competitive with LANL)
  burst_prob=0.3  → HRC-MAE ≈ 0.002  (may beat LANL)
  (optimal value empirically determined by this script)

The reuse_access_rate column confirms the burst is not changing the coarse
reuse rate — only the working-set locality of reuse events.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase_pmf_atlas import PhasePMFAtlas


# ---------------------------------------------------------------------------
# HRC computation (inline, does not depend on LRU-diagnostic imports)
# ---------------------------------------------------------------------------

def _hrc_lru(obj_ids: np.ndarray, cache_sizes: np.ndarray) -> np.ndarray:
    """LRU hit-ratio curve for a single stream using an OrderedDict cache."""
    hrcs = np.zeros(len(cache_sizes), dtype=np.float64)
    if len(obj_ids) == 0:
        return hrcs
    for j, cs in enumerate(cache_sizes):
        cs = int(cs)
        cache: "OrderedDict[int, bool]" = OrderedDict()
        hits = 0
        for oid in obj_ids:
            key = int(oid)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = True
                if len(cache) > cs:
                    cache.popitem(last=False)
        hrcs[j] = hits / len(obj_ids)
    return hrcs


def _metrics(rows: list, cache_sizes: np.ndarray) -> dict:
    """Compute per-stream-averaged HRC-MAE-ready metrics from generate output."""
    df = pd.DataFrame(rows)
    stream_ids = sorted(df["stream_id"].unique())
    hrcs = []
    reuses = []
    footprints = []
    for sid in stream_ids:
        sdf = df[df["stream_id"] == sid].sort_values("ts")
        oids = sdf["obj_id"].to_numpy(dtype=np.int64)
        hrcs.append(_hrc_lru(oids, cache_sizes))
        n = len(oids)
        n_reuse = int((pd.Series(oids).duplicated()).sum())
        reuses.append(n_reuse / n if n > 0 else 0.0)
        footprints.append(int(len(np.unique(oids))))
    hrc_avg = np.mean(hrcs, axis=0) if hrcs else np.zeros(len(cache_sizes))
    return {
        "hrc": hrc_avg.tolist(),
        "reuse_access_rate": float(np.mean(reuses)),
        "footprint_mean": float(np.mean(footprints)),
        "n_streams": len(stream_ids),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(
    atlas_path: str,
    real_json_path: str,
    burst_probs: List[float],
    burst_pool_sizes: List[int],
    seeds: List[int],
    n_records: int,
    n_streams: int,
) -> list:
    print(f"Loading atlas: {atlas_path}")
    atlas = PhasePMFAtlas.load(atlas_path)

    print(f"Loading real eval JSON: {real_json_path}")
    with open(real_json_path) as f:
        real_d = json.load(f)
    real_node = real_d.get("real", real_d)
    real_hrc = np.array(real_node["hrc"], dtype=np.float64)
    cache_sizes = np.array(real_node.get("cache_sizes", []), dtype=np.float64)
    if len(cache_sizes) == 0:
        raise ValueError("real_json has no cache_sizes field")
    real_reuse = float(real_node.get("reuse_access_rate", -1.0))

    print(f"  Real reuse_access_rate: {real_reuse:.4f}")
    print(f"  Cache sizes: [{cache_sizes.min():.0f}..{cache_sizes.max():.0f}] ({len(cache_sizes)} pts)")
    print()

    results = []
    for bpool in burst_pool_sizes:
        for bprob in burst_probs:
            seed_results = []
            for seed in seeds:
                t0 = time.time()
                rows = atlas.generate(
                    n_records=n_records,
                    n_streams=n_streams,
                    seed=seed,
                    burst_prob=bprob,
                    burst_pool_size=bpool,
                )
                fake_m = _metrics(rows, cache_sizes)
                fake_hrc = np.array(fake_m["hrc"], dtype=np.float64)
                hrc_mae = float(np.mean(np.abs(fake_hrc - real_hrc)))
                elapsed = time.time() - t0
                seed_results.append({
                    "seed": seed,
                    "hrc_mae": hrc_mae,
                    "reuse_access_rate": fake_m["reuse_access_rate"],
                    "footprint_mean": fake_m["footprint_mean"],
                    "elapsed_s": round(elapsed, 1),
                })
                print(
                    f"  burst_prob={bprob:.3f}  pool={bpool:2d}  seed={seed}"
                    f"  HRC-MAE={hrc_mae:.6f}  reuse={fake_m['reuse_access_rate']:.4f}"
                    f"  ({elapsed:.1f}s)"
                )

            mean_mae = float(np.mean([r["hrc_mae"] for r in seed_results]))
            results.append({
                "burst_prob": bprob,
                "burst_pool_size": bpool,
                "mean_hrc_mae": mean_mae,
                "real_reuse": real_reuse,
                "seed_results": seed_results,
            })

    return results


def _print_summary(results: list, real_json_path: str) -> None:
    print("\n" + "=" * 70)
    print("BURST INJECTION SWEEP SUMMARY")
    print(f"Real JSON: {real_json_path}")
    print("=" * 70)
    print(f"{'burst_prob':>12}  {'pool':>6}  {'mean HRC-MAE':>14}  {'seeds':>6}")
    print("-" * 50)
    best = min(results, key=lambda r: r["mean_hrc_mae"])
    for r in results:
        flag = " ← BEST" if r is best else ""
        print(
            f"  {r['burst_prob']:>10.3f}  {r['burst_pool_size']:>6d}"
            f"  {r['mean_hrc_mae']:>14.6f}"
            f"  {len(r['seed_results']):>6d}{flag}"
        )
    print("=" * 70)
    print(f"Best: burst_prob={best['burst_prob']:.3f}  pool={best['burst_pool_size']}"
          f"  HRC-MAE={best['mean_hrc_mae']:.6f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atlas", required=True,
                   help="Path to nophase Phase-PMF Atlas (.pkl.gz)")
    p.add_argument("--real-json", required=True,
                   help="Path to long_rollout eval JSON with real metrics")
    p.add_argument("--corpus", default="",
                   help="Label for output (e.g. alibaba, tencent)")
    p.add_argument("--burst-probs", nargs="+", type=float,
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
                   help="burst_prob values to sweep")
    p.add_argument("--burst-pool-sizes", nargs="+", type=int,
                   default=[20],
                   help="burst_pool_size values to sweep")
    p.add_argument("--seeds", nargs="+", type=int, default=[42],
                   help="Seeds for fake generation")
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--output", default="",
                   help="Save results JSON to this path")
    args = p.parse_args()

    print(f"IDEA #67 burst injection sweep — corpus={args.corpus or 'unknown'}")
    print(f"  Atlas:     {args.atlas}")
    print(f"  Real JSON: {args.real_json}")
    print(f"  burst_probs: {args.burst_probs}")
    print(f"  burst_pool_sizes: {args.burst_pool_sizes}")
    print(f"  seeds: {args.seeds}")
    print(f"  n_records={args.n_records}  n_streams={args.n_streams}")
    print()

    results = sweep(
        atlas_path=args.atlas,
        real_json_path=args.real_json,
        burst_probs=args.burst_probs,
        burst_pool_sizes=args.burst_pool_sizes,
        seeds=args.seeds,
        n_records=args.n_records,
        n_streams=args.n_streams,
    )
    _print_summary(results, args.real_json)

    if args.output:
        out = {
            "corpus": args.corpus,
            "atlas": args.atlas,
            "real_json": args.real_json,
            "n_records": args.n_records,
            "n_streams": args.n_streams,
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved → {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
