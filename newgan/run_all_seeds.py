"""Utility to run a training experiment for a set of seeds.

The script is a thin wrapper around ``newgan/run.py`` (the existing entry
point for training). It launches the training program three times – for
seeds 42, 11 and 7 – capturing the final HRC‑MAE reported by the
trainer.  The results are printed in a compact markdown table and
appended to ``RESPONSE‑Sandia.md`` so that a single round entry can be
created automatically.

Usage:
    python newgan/run_all_seeds.py --exp-name <name> --trace-dir <dir> [other flags]

The script forwards all flags that are understood by ``newgan/run.py``
except for ``--seed`` which it overrides per iteration.  Any
additional arguments are passed unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

SEEDS = [42, 11, 7]


def run_single(seed: int, args: argparse.Namespace) -> float:
    """Run ``newgan/run.py`` with a specific seed and return the final HRC‑MAE.

    The trainer prints the final combined score (our proxy for HRC‑MAE)
    to stdout.  We capture that line using ``subprocess.run`` and parse
    the numeric value.
    """

    # ``newgan/train.py`` is the training entry point for the Sandia
    # pipeline.  We invoke it with the supplied arguments and the
    # current seed.
    cmd: List[str] = [sys.executable, "newgan/train.py", "--seed", str(seed)]
    # Forward all arguments except ``seed`` and ``exp-name`` (handled separately)
    for k, v in vars(args).items():
        if k in {"seed", "exp_name"}:
            continue
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.extend([f"--{k}", str(v)])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    score = None
    for line in result.stdout.splitlines():
        if "Combined" in line and "score" in line:
            try:
                score = float(line.split()[-1])
                break
            except ValueError:
                continue
    if score is None:
        raise RuntimeError("Could not parse combined score from trainer output")
    return score


def main():
    parser = argparse.ArgumentParser(description="Run training for multiple seeds")
    parser.add_argument("--exp-name", required=True, help="Experiment name")
    parser.add_argument("--trace-dir", required=True, help="Trace directory path")
    parser.add_argument("--fmt", default="oracle_general", help="Trace format")
    parser.add_argument("--char-file", help="Characterization file path")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=8e-5, help="Generator LR (tuned)")
    parser.add_argument("--lr-d", type=float, default=8e-5, help="Critic LR (tuned)")
    parser.add_argument("--noise-dim", type=int, default=10, help="Noise dimension")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--latent-dim", type=int, default=24, help="Latent dimension")
    parser.add_argument("--timestep", type=int, default=12, help="Timestep")
    parser.add_argument("--pretrain-ae-epochs", type=int, default=50, help="AE pretrain epochs")
    parser.add_argument("--pretrain-sup-epochs", type=int, default=50, help="Supervisor epochs")
    parser.add_argument("--pretrain-g-epochs", type=int, default=100, help="G warmup epochs")
    parser.add_argument("--n-critic", type=int, default=7, help="Critic updates per G update (tuned)")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint frequency")
    parser.add_argument("--early-stop-patience", type=int, default=30, help="Early stop patience")
    parser.add_argument("--seed", type=int, default=42, help="Override for single run (ignored)")
    # parser.add_argument("--exp-name", help="Experiment name (ignored, provided separately)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    args = parser.parse_args()

    scores: dict[int, float] = {}
    for s in SEEDS:
        print(f"Running seed {s}…")
        scores[s] = run_single(s, args)

    table = "| Seed | Combined Score |\n|------|----------------|\n"
    for s in SEEDS:
        table += f"| {s} | {scores[s]:.5f} |\n"
    print("\nResults:\n")
    print(table)

    response_path = Path("RESPONSE-Sandia.md")
    round_line = f"## Round X (2026-05-01) — {args.exp_name}\n"
    content = f"{round_line}\n{table}\n"
    response_path.write_text(content, encoding="utf-8")
    print(f"Wrote round report to {response_path}")


if __name__ == "__main__":
    main()
