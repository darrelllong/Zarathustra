"""Twitter recent-pool window sweep for LLNL atlas retake.

LANL leads Twitter at 0.0272 (4-seed, win=48). LLNL's current best is 0.1532
(R281.B, win=16 with R244lock atlas). LANL's audit showed:
  win=8  → 0.0298, win=16 → baseline, win=32 → 0.0277, win=48 → 0.0272 (best),
  win=64 → 0.0279 (slight regression).

The gap is almost entirely in the recent-pool window axis.  This script sweeps
win from 16 to 80 using LLNL's R270 Twitter atlas on the official reference.

Run as:
  python3 -m altgan.sweep_twitter_window --seeds 42
  python3 -m altgan.sweep_twitter_window --seeds 42,80,81,82  # multi-seed
"""

from __future__ import annotations

import argparse
import subprocess
import sys

LLNL_TWITTER_ATLAS = (
    "/tiamat/zarathustra/llgan-output/atlases/"
    "llnl_neural_atlas_twitter_237f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz"
)
LLNL_TWITTER_TRACE_DIR = "/tiamat/zarathustra/traces/twitter_cluster"
LLNL_TWITTER_CHAR = "/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl"
LLNL_TWITTER_MANIFEST = "/tiamat/zarathustra/llgan-output/manifests/twitter_cluster_stackatlas.json"
LLNL_TWITTER_OFFICIAL_REF = "/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv"
OUTPUT_ROOT = "/tiamat/zarathustra/altgan-output"

# Window sweep: LANL published win=8/32/48/64; also sweep win=16 (LLNL baseline)
# hp=0.65 and adj=0.40 are from LANL's promoted Twitter recipe
GRID = [
    # (name, recent_pool_window, hot_pool_prob, adj_dup_prob)
    ("win16_hp45",  16, 0.45, 0.40),   # LLNL R281.B baseline
    ("win24_hp65",  24, 0.65, 0.40),
    ("win32_hp65",  32, 0.65, 0.40),
    ("win40_hp65",  40, 0.65, 0.40),
    ("win48_hp65",  48, 0.65, 0.40),   # LANL's winner on their atlas
    ("win56_hp65",  56, 0.65, 0.40),
    ("win64_hp65",  64, 0.65, 0.40),
    ("win80_hp65",  80, 0.65, 0.40),
    # Cross: vary hp at win=48
    ("win48_hp45",  48, 0.45, 0.40),
    ("win48_hp55",  48, 0.55, 0.40),
    ("win48_hp75",  48, 0.75, 0.40),
    # Cross: vary adj at win=48 hp=0.65
    ("win48_adj35", 48, 0.65, 0.35),
    ("win48_adj45", 48, 0.65, 0.45),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", default="42")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--atlas", default=LLNL_TWITTER_ATLAS)
    p.add_argument("--output-root", default=OUTPUT_ROOT)
    p.add_argument("--phase2-names", default="",
                   help="Comma-separated names; empty = run all.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    phase2 = {n.strip() for n in args.phase2_names.split(",") if n.strip()}
    grid = [(n, w, hp, adj) for n, w, hp, adj in GRID if not phase2 or n in phase2]

    bracket_args = [
        sys.executable, "-u", "-m", "altgan.launch_msr_cachesim_bracket",
        "--model", args.atlas,
        "--trace-dir", LLNL_TWITTER_TRACE_DIR,
        "--char-file", LLNL_TWITTER_CHAR,
        "--real-manifest", LLNL_TWITTER_MANIFEST,
        "--official-ref", LLNL_TWITTER_OFFICIAL_REF,
        "--output-root", args.output_root,
        "--cache-sizes", "32,128,512,2048,8192",
        "--policies", "lru,arc,fifo,sieve,slru,car",
    ]
    if args.skip_existing:
        bracket_args.append("--skip-existing")

    for seed in seeds:
        for name, win, hp, adj in grid:
            spec = (
                f"llnl_tw_sweep_{name}_s{seed}:"
                f"seed={seed},"
                f"win={win},"
                f"hp={hp},"
                f"adj={adj},"
                "rank=1.0,"
                "tail=0.10,"
                "mf=0.5,"
                "rp=0.25,"
                "k=75,"
                "minage=16,"
                "tb=1.0,"
                "lp=0.9"
            )
            cmd = bracket_args + ["--spec", spec]
            if args.dry_run:
                import shlex
                print("+ " + " ".join(shlex.quote(c) for c in cmd))
            else:
                print(f"[sweep_twitter_window] {name} seed={seed}", flush=True)
                subprocess.run(cmd, check=True)

    if args.dry_run:
        print(f"\n[sweep_twitter_window] dry-run: {len(seeds) * len(grid)} runs planned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
