"""MSR Exchange hot-pool compression sweep for LLNL atlas retake.

LANL retook MSR at 0.00484 (Round 70) using hp=0.25, rank=1.0, min_age=16.
LLNL's current best is 0.00921 (R282.F, hp=0.45, rank=1.3, min_age=0).

This script sweeps LLNL's R270 MSR atlas across the hot-pool compression
axis to find the equivalent sweet spot.  Run as:

  python3 -m altgan.sweep_msr_hotpool --seeds 42,80,81,82
  python3 -m altgan.sweep_msr_hotpool --seeds 42 --dry-run   # preview commands

Strategy:
  Phase 1 (seed=42 scout): run all 18 points, identify top 6 by HRC-MAE.
  Phase 2 (4-seed confirm): run top 6 points across all seeds to get multi-seed mean.

The 18 points cover rank_scale × hot_pool_prob grid anchored around LANL's winner:
  rank_scale in {0.9, 1.0, 1.1, 1.3}  ×  hp in {0.20, 0.25, 0.30, 0.35, 0.45}
  plus baseline (LLNL R282.F: rank=1.3 hp=0.45 min_age=0) and three LANL-style rows.
All non-baseline rows use min_age=16 (LANL's proven improvement over 0 or 8).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


LLNL_MSR_ATLAS = (
    "/tiamat/zarathustra/llgan-output/atlases/"
    "llnl_neural_atlas_msr_exchange_96f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz"
)
LLNL_MSR_TRACE_DIR = "/tiamat/zarathustra/traces/msr_exchange"
LLNL_MSR_CHAR = "/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl"
LLNL_MSR_MANIFEST = "/tiamat/zarathustra/llgan-output/manifests/msr_exchange_stackatlas.json"
LLNL_MSR_OFFICIAL_REF = "/tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv"
OUTPUT_ROOT = "/tiamat/zarathustra/altgan-output"

# Sweep grid (name, rank_scale, hot_pool_prob, min_age)
# All non-baseline rows use adj=0.40 tail=0.10 mf=0.5 rp=0.15 win=16 (LLNL R282.F values)
GRID = [
    # LLNL R282.F baseline (should reproduce 0.00921)
    ("llnl_r282f",    1.3, 0.45, 0),
    # Add min_age=16 only → how much does it help without rank/hp change?
    ("minage16",      1.3, 0.45, 16),
    # LANL rank=1.0 direction, keep LLNL hp
    ("r100_hp45",     1.0, 0.45, 16),
    # Move both rank and hp toward LANL's winner
    ("r100_hp35",     1.0, 0.35, 16),
    ("r100_hp30",     1.0, 0.30, 16),
    ("r100_hp25",     1.0, 0.25, 16),  # ← LANL's exact winner on their atlas
    ("r100_hp20",     1.0, 0.20, 16),  # push lower
    # Check whether rank matters much around hp=0.25
    ("r090_hp25",     0.9, 0.25, 16),
    ("r110_hp25",     1.1, 0.25, 16),
    ("r120_hp25",     1.2, 0.25, 16),
    ("r130_hp25",     1.3, 0.25, 16),
    # Grid at rank=1.0
    ("r100_hp28",     1.0, 0.28, 16),
    ("r100_hp22",     1.0, 0.22, 16),
    # Grid at rank=0.9
    ("r090_hp30",     0.9, 0.30, 16),
    ("r090_hp20",     0.9, 0.20, 16),
    # Grid at rank=1.1
    ("r110_hp30",     1.1, 0.30, 16),
    ("r110_hp20",     1.1, 0.20, 16),
    # Check if min_age=8 (R276 value) is worse than 16
    ("r100_hp25_ma8", 1.0, 0.25, 8),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="42", help="Comma-separated seeds (default: 42 for scout).")
    p.add_argument("--phase2-names", default="",
                   help="Comma-separated spec names for phase-2 multi-seed; empty = run all.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--atlas", default=LLNL_MSR_ATLAS)
    p.add_argument("--output-root", default=OUTPUT_ROOT)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    phase2 = {n.strip() for n in args.phase2_names.split(",") if n.strip()}
    grid = [(n, r, hp, ma) for n, r, hp, ma in GRID if not phase2 or n in phase2]

    bracket_args = [
        sys.executable, "-u", "-m", "altgan.launch_msr_cachesim_bracket",
        "--model", args.atlas,
        "--trace-dir", LLNL_MSR_TRACE_DIR,
        "--char-file", LLNL_MSR_CHAR,
        "--real-manifest", LLNL_MSR_MANIFEST,
        "--official-ref", LLNL_MSR_OFFICIAL_REF,
        "--output-root", args.output_root,
        "--cache-sizes", "32,128,512,2048,8192",
        "--policies", "lru,arc,fifo,sieve,slru,car",
    ]
    if args.skip_existing:
        bracket_args.append("--skip-existing")

    for seed in seeds:
        for name, rank, hp, minage in grid:
            spec = (
                f"llnl_hp_sweep_{name}_s{seed}:"
                f"seed={seed},"
                f"rank={rank},"
                f"hp={hp},"
                f"minage={minage},"
                "adj=0.40,"
                "tail=0.10,"
                "mf=0.5,"
                "rp=0.15,"
                "win=16,"
                "k=75,"
                "tb=1.0,"
                "lp=0.9"
            )
            cmd = bracket_args + ["--spec", spec]
            if args.dry_run:
                import shlex
                print("+ " + " ".join(shlex.quote(c) for c in cmd))
            else:
                print(f"[sweep_msr_hotpool] {name} seed={seed}", flush=True)
                subprocess.run(cmd, check=True)

    if args.dry_run:
        print(f"\n[sweep_msr_hotpool] dry-run: {len(seeds) * len(grid)} runs planned "
              f"({len(seeds)} seeds × {len(grid)} specs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
