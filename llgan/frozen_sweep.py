"""
Frozen-bundle checkpoint sweep.

Runs eval.py under the deterministic frozen-bundle protocol
(--eval-real-seed 42) against every saved epoch_*.pt and best.pt in a
checkpoint directory, ranks them by the combined ATB score

    ★ = MMD² + 0.2 · (1 − β-recall)

and promotes a frozen_best.pt symlink so downstream tooling can refer to
the checkpoint that actually wins under the published benchmark protocol.

This is the post-train companion to training-time best.pt selection, which
uses a different (training-EMA combined) score and has been observed to
mis-rank relative to the frozen evaluation.

Usage
-----
    python -m llgan.frozen_sweep \\
        --checkpoint-dir /home/darrell/checkpoints/alibaba_v159 \\
        --trace-dir /tiamat/zarathustra/traces/alibaba \\
        --fmt oracle_general

The sweep writes:
    <checkpoint-dir>/frozen_sweep.json   machine-readable per-checkpoint scores
    <checkpoint-dir>/frozen_sweep.log    full eval.py stdout concatenated
    <checkpoint-dir>/frozen_best.pt      symlink to the winning checkpoint
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


_MMD_RE = re.compile(r"^\s{2}MMD²\s*:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
_RECALL_RE = re.compile(r"^\s{2}β-recall\s*:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
_PRECISION_RE = re.compile(r"^\s{2}α-precision\s*:\s*([0-9]*\.?[0-9]+)", re.MULTILINE)
_EPOCH_RE = re.compile(r"^Epoch\s*:\s*([0-9]+)", re.MULTILINE)


def _combined_score(mmd2: float, recall: float) -> float:
    return mmd2 + 0.2 * (1.0 - recall)


def _parse_eval_stdout(text: str) -> dict | None:
    m = _MMD_RE.search(text)
    r = _RECALL_RE.search(text)
    p = _PRECISION_RE.search(text)
    e = _EPOCH_RE.search(text)
    if not (m and r):
        return None
    mmd2 = float(m.group(1))
    recall = float(r.group(1))
    return {
        "mmd2": mmd2,
        "recall": recall,
        "precision": float(p.group(1)) if p else None,
        "epoch": int(e.group(1)) if e else None,
        "combined": _combined_score(mmd2, recall),
    }


def _discover_checkpoints(ckpt_dir: Path) -> list[Path]:
    # epoch_*.pt in numeric order, then best.pt and final.pt.
    # Skip pretrain_complete.pt (pre-GAN) and any existing frozen_best.pt symlink.
    epochs = sorted(
        ckpt_dir.glob("epoch_*.pt"),
        key=lambda p: int(re.search(r"epoch_(\d+)", p.name).group(1)),
    )
    ordered: list[Path] = list(epochs)
    for name in ("best.pt", "final.pt"):
        path = ckpt_dir / name
        if path.exists() and not path.is_symlink():
            ordered.append(path)
    return ordered


def _run_eval(
    ckpt: Path,
    trace_dir: str,
    fmt: str,
    real_seed: int,
    fake_seed: int,
    n_samples: int,
    eval_script: Path,
    python_exe: str,
) -> tuple[str, dict | None]:
    cmd = [
        python_exe,
        "-u",
        str(eval_script),
        "--checkpoint", str(ckpt),
        "--trace-dir", trace_dir,
        "--fmt", fmt,
        "--n-samples", str(n_samples),
        "--eval-real-seed", str(real_seed),
        "--eval-fake-seed", str(fake_seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + ("\n--- stderr ---\n" + stderr if stderr.strip() else "")
    parsed = _parse_eval_stdout(stdout) if proc.returncode == 0 else None
    return combined, parsed


def _promote_frozen_best(ckpt_dir: Path, winner: Path) -> Path:
    target = ckpt_dir / "frozen_best.pt"
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(winner.name, target)
    return target


def run_sweep(args) -> int:
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_dir():
        print(f"ERROR: checkpoint dir not found: {ckpt_dir}", file=sys.stderr)
        return 2

    checkpoints = _discover_checkpoints(ckpt_dir)
    if not checkpoints:
        print(f"ERROR: no epoch_*.pt or best.pt under {ckpt_dir}", file=sys.stderr)
        return 2

    eval_script = Path(__file__).parent / "eval.py"
    python_exe = sys.executable

    print(f"Frozen sweep: {ckpt_dir}")
    print(f"Trace dir   : {args.trace_dir}")
    print(f"Seed        : {args.eval_real_seed}")
    print(f"Checkpoints : {len(checkpoints)}")
    print()

    log_path = ckpt_dir / "frozen_sweep.log"
    json_path = ckpt_dir / "frozen_sweep.json"
    results: list[dict] = []

    with log_path.open("w") as logf:
        for ckpt in checkpoints:
            print(f"  evaluating {ckpt.name} …", flush=True)
            stdout, parsed = _run_eval(
                ckpt,
                args.trace_dir,
                args.fmt,
                args.eval_real_seed,
                args.eval_fake_seed,
                args.n_samples,
                eval_script,
                python_exe,
            )
            logf.write(f"\n{'='*72}\n{ckpt}\n{'='*72}\n{stdout}\n")
            logf.flush()
            if parsed is None:
                results.append({"checkpoint": ckpt.name, "error": True})
                print(f"    FAILED to parse eval output (see {log_path.name})")
                continue
            row = {"checkpoint": ckpt.name, **parsed}
            results.append(row)
            print(
                f"    ★={parsed['combined']:.5f}  "
                f"MMD²={parsed['mmd2']:.5f}  "
                f"β-rec={parsed['recall']:.4f}"
            )

    scored = [r for r in results if not r.get("error")]
    if not scored:
        print("\nERROR: every eval failed — see frozen_sweep.log", file=sys.stderr)
        json_path.write_text(json.dumps({"results": results}, indent=2))
        return 1

    scored.sort(key=lambda r: r["combined"])
    winner = scored[0]
    winner_path = ckpt_dir / winner["checkpoint"]

    summary = {
        "checkpoint_dir": str(ckpt_dir),
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "eval_real_seed": args.eval_real_seed,
        "eval_fake_seed": args.eval_fake_seed,
        "n_samples": args.n_samples,
        "frozen_best": winner["checkpoint"],
        "frozen_best_combined": winner["combined"],
        "results": results,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    promoted = _promote_frozen_best(ckpt_dir, winner_path)

    print(f"\n{'─'*60}")
    print(f"{'checkpoint':22s}  {'★':>8s}  {'MMD²':>9s}  {'β-recall':>9s}")
    print(f"{'─'*60}")
    for r in sorted(results, key=lambda x: x.get("combined", float("inf"))):
        if r.get("error"):
            print(f"  {r['checkpoint']:20s}  (failed)")
            continue
        marker = " ★" if r["checkpoint"] == winner["checkpoint"] else "  "
        print(
            f"{marker}{r['checkpoint']:20s}  "
            f"{r['combined']:8.5f}  {r['mmd2']:9.5f}  {r['recall']:9.4f}"
        )
    print(f"{'─'*60}")
    print(f"frozen-best: {winner['checkpoint']}  (★={winner['combined']:.5f})")
    print(f"promoted  : {promoted}  →  {winner_path.name}")
    print(f"log       : {log_path}")
    print(f"json      : {json_path}")

    best_pt = ckpt_dir / "best.pt"
    if best_pt.exists() and winner["checkpoint"] != "best.pt":
        best_row = next((r for r in scored if r["checkpoint"] == "best.pt"), None)
        if best_row:
            delta = best_row["combined"] - winner["combined"]
            pct = 100.0 * delta / winner["combined"] if winner["combined"] > 0 else 0.0
            print(
                f"\nNOTE: best.pt (★={best_row['combined']:.5f}) is "
                f"{delta:+.5f} ({pct:+.1f}%) worse than frozen-best."
            )
            print("      Training-time selector mis-ranked against the frozen bundle.")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", required=True, help="Directory containing epoch_*.pt / best.pt")
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", default="oracle_general")
    p.add_argument("--eval-real-seed", type=int, default=42,
                   help="Frozen-bundle seed. Default 42 matches the published protocol.")
    p.add_argument("--eval-fake-seed", type=int, default=42,
                   help="Fake-sampling seed. Default 42. Required for deterministic "
                        "cross-checkpoint comparison — see llgan.eval.")
    p.add_argument("--n-samples", type=int, default=2000)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run_sweep(parse_args()))
