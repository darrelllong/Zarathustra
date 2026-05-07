"""One-shot launcher: Twitter tightening from the current r307 16K chunk-surface fakes.

This is a thin convenience wrapper around `altgan.launch_chunk_surface_multiseed`
using the official Twitter cachesim surface (6 policies) and LANL's current
Twitter donor-bank conventions.

Intended to run on `baase` / `vinge` where `/tiamat/zarathustra/...` exists.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tag-prefix",
        required=True,
        help="Tag prefix used for output file naming (should include the new round id).",
    )
    p.add_argument("--seeds", default="42,80,81,82")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--pipeline", default="8192", help="Comma-separated chunk sizes.")
    p.add_argument("--max-accepts", type=int, default=8)
    p.add_argument("--max-evals", type=int, default=250)
    p.add_argument("--min-improvement", type=float, default=1e-6)
    p.add_argument("--emit-markdown", action="store_true")
    p.add_argument(
        "--append-markdown",
        default=None,
        help="Comma-separated list of destination files to append the markdown snippet.",
    )
    p.add_argument("--markdown-title", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.rstrip("/")

    # Donor globs intentionally include (a) all prior chunk-surface variants for
    # the same seed, (b) per-seed base fakes, and (c) the shared seed-42 donor
    # bank LANL uses for cross-seed transfers. The launcher caps work via
    # --max-evals / --max-accepts.
    donor_globs = ",".join(
        [
            f"{output_root}/twitter_chunksurf_*_seed{{seed}}_fake_1000k.csv",
            f"{output_root}/twitter_cluster*_seed{{seed}}_fake_*.csv",
            f"{output_root}/twitter*_seed42_fake_*.csv",
        ]
    )

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "altgan.launch_chunk_surface_multiseed",
        "--real",
        "/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv",
        "--output-root",
        output_root,
        "--base-template",
        f"{output_root}/twitter_chunksurf_r307_refine16_ck16384_seed{{seed}}_fake_1000k.csv",
        "--donor-globs",
        donor_globs,
        "--tag-prefix",
        args.tag_prefix,
        "--pipeline",
        args.pipeline,
        "--cache-sizes",
        "32,128,512,2048,8192",
        "--policies",
        "lru,arc,fifo,sieve,slru,car",
        "--max-accepts",
        str(args.max_accepts),
        "--max-evals",
        str(args.max_evals),
        "--min-improvement",
        str(args.min_improvement),
        "--seeds",
        args.seeds,
    ]

    if args.emit_markdown:
        cmd.append("--emit-markdown")
    if args.append_markdown:
        cmd += ["--append-markdown", args.append_markdown]
    if args.markdown_title:
        cmd += ["--markdown-title", args.markdown_title]
    if args.dry_run:
        cmd.append("--dry-run")

    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

