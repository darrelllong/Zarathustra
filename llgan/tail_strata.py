"""
IDEA #34 tail-stratified eval bundle — partition trace files by tail-heaviness.

The R higher-moment audit (2026-04-18, R-ANALYSIS.md §"Generator-Relevant
Tail Results") shows block traces concentrate extreme 5th/6th moments on
iat_*, abs_stride_*, and reuse_ratio surfaces. Smooth mean/variance losses
cannot see those tails. IDEA #34's MVE is to *evaluate* candidate checkpoints
on a tail-heavy bundle separately from an ordinary bundle, requiring a win
to hold ordinary score while improving tail-stratum metrics.

This script does the per-file scoring step:
  1. Read per-trace characterizations (the same JSONL used for conditioning).
  2. Compute a tail-heaviness score per file from fields already present:
       iat_q99/iat_q50            (how heavy is the IAT tail vs its median)
       abs_stride_q99/q50         (how heavy is the stride tail)
       iat_std/iat_mean           (CV — breadth)
     Score = normalized geometric mean of the three ratios, so a file scoring
     in the top decile on any *one* surface doesn't singlehandedly dominate.
  3. Restrict to a requested trace-dir (match by resolved path or basename)
     and emit two manifests: top-decile tail-heavy, plus ordinary middle-80%.

Usage
-----
    python -m llgan.tail_strata \
        --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
        --trace-dir /tiamat/zarathustra/traces/alibaba \
        --out-dir /home/darrell/strata/alibaba

Writes:
    <out-dir>/tail_heavy.txt    top decile by tail-heaviness (one path per line)
    <out-dir>/ordinary.txt      middle 80% (bottom decile dropped as noise)
    <out-dir>/scores.csv        per-file debug: path, iat_ratio, stride_ratio,
                                iat_cv, tail_score, stratum

The manifests plug into eval.py's --eval-file-manifest (and frozen_sweep's).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path


def _score(profile: dict) -> tuple[float, float, float, float] | None:
    iat = profile.get("iat_stats") or {}
    stride = profile.get("abs_stride_stats") or {}
    try:
        iat_q50 = float(iat["q50"])
        iat_q99 = float(iat["q99"])
        iat_mean = float(iat["mean"])
        iat_std = float(iat["std"])
        st_q50 = float(stride["q50"])
        st_q99 = float(stride["q99"])
    except (KeyError, TypeError, ValueError):
        return None

    eps = 1e-9
    iat_ratio = iat_q99 / max(iat_q50, eps)
    stride_ratio = st_q99 / max(st_q50, eps)
    iat_cv = iat_std / max(iat_mean, eps)

    if any(not math.isfinite(x) for x in (iat_ratio, stride_ratio, iat_cv)):
        return None
    if any(x <= 0 for x in (iat_ratio, stride_ratio, iat_cv)):
        return None

    # geometric mean: equally-weighted on the log scale, dominated by the
    # smallest contributing factor rather than spiking on a single outlier
    tail_score = math.exp((math.log(iat_ratio) + math.log(stride_ratio)
                           + math.log(iat_cv)) / 3.0)
    return iat_ratio, stride_ratio, iat_cv, tail_score


def _collect_trace_files(trace_dir: str) -> set[str]:
    """Gather discoverable files so the manifest only lists paths that the
    eval harness can actually load. We match both full resolved path AND
    basename so manifests survive remote path rewrites."""
    found = set()
    root = Path(trace_dir)
    if not root.is_dir():
        raise SystemExit(f"trace-dir not a directory: {trace_dir}")
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            found.add(str(Path(dirpath) / fn))
    return found


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--char-file", required=True,
                   help="trace_characterizations.jsonl")
    p.add_argument("--trace-dir", required=True,
                   help="Restrict to characterizations whose path (or basename) "
                        "matches a file under this directory.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for tail_heavy.txt, ordinary.txt, scores.csv")
    p.add_argument("--top-quantile", type=float, default=0.10,
                   help="Fraction to treat as tail-heavy (default: top 10%%).")
    p.add_argument("--bottom-drop", type=float, default=0.10,
                   help="Fraction of lowest-tail files to drop from 'ordinary' "
                        "(default: 0.10 — ordinary = middle 80%%).")
    args = p.parse_args()

    trace_files = _collect_trace_files(args.trace_dir)
    basenames_to_path = {Path(f).name: f for f in trace_files}
    print(f"[tail_strata] trace-dir {args.trace_dir}: {len(trace_files)} files")

    # Characterizations JSONL can contain multiple entries that resolve to
    # the same on-disk file (e.g., duplicate basenames across source dirs).
    # Dedup by resolved path; first-wins.
    by_path: dict[str, tuple[float, float, float, float]] = {}
    skipped = 0
    with open(args.char_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = rec.get("path", "")
            if not path:
                continue
            resolved = None
            if path in trace_files:
                resolved = path
            else:
                resolved = basenames_to_path.get(Path(path).name)
            if resolved is None:
                continue
            if resolved in by_path:
                continue
            score_result = _score(rec.get("profile") or {})
            if score_result is None:
                skipped += 1
                continue
            by_path[resolved] = score_result
    scored = [(p, ir, sr, cv, ts) for p, (ir, sr, cv, ts) in by_path.items()]

    if not scored:
        raise SystemExit(
            f"No scored files. Checked {args.char_file} against "
            f"{len(trace_files)} files in {args.trace_dir}. "
            f"Dropped {skipped} missing-field characterizations."
        )

    scored.sort(key=lambda r: r[4], reverse=True)
    n = len(scored)
    top_n = max(1, int(round(args.top_quantile * n)))
    drop_n = int(round(args.bottom_drop * n))

    tail_heavy = scored[:top_n]
    ordinary = scored[top_n:n - drop_n] if drop_n > 0 else scored[top_n:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "tail_heavy.txt").write_text(
        "\n".join(r[0] for r in tail_heavy) + "\n"
    )
    (out_dir / "ordinary.txt").write_text(
        "\n".join(r[0] for r in ordinary) + "\n"
    )
    with (out_dir / "scores.csv").open("w", newline="") as cfh:
        w = csv.writer(cfh)
        w.writerow(["path", "iat_q99_over_q50", "stride_q99_over_q50",
                    "iat_cv", "tail_score", "stratum"])
        for rank, (p_, ir, sr, cv, ts) in enumerate(scored):
            if rank < top_n:
                stratum = "tail_heavy"
            elif rank < n - drop_n:
                stratum = "ordinary"
            else:
                stratum = "dropped_low"
            w.writerow([p_, f"{ir:.4f}", f"{sr:.4f}",
                        f"{cv:.4f}", f"{ts:.4f}", stratum])

    print(f"[tail_strata] scored {n} files; skipped {skipped} (missing fields)")
    print(f"[tail_strata] tail_heavy: {len(tail_heavy)} "
          f"(top {args.top_quantile*100:.0f}%); tail_score range "
          f"[{tail_heavy[-1][4]:.2f}, {tail_heavy[0][4]:.2f}]")
    print(f"[tail_strata] ordinary:   {len(ordinary)} "
          f"(middle, dropped bottom {args.bottom_drop*100:.0f}%); "
          f"tail_score range [{ordinary[-1][4]:.2f}, {ordinary[0][4]:.2f}]")
    print(f"[tail_strata] wrote {out_dir}/tail_heavy.txt, ordinary.txt, scores.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
