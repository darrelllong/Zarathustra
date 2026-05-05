"""One-shot launcher: CloudPhysics 8K tightening from the current r306 16K fakes.

This is a thin convenience wrapper around `altgan.launch_chunk_surface_multiseed`
with the correct official CloudPhysics cachesim surface (8 policies + the extra
32768 cache size) and the current donor-bank conventions.

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
    p.add_argument("--max-accepts", type=int, default=4)
    p.add_argument("--max-evals", type=int, default=120)
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

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "altgan.launch_chunk_surface_multiseed",
        "--real",
        "/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv",
        "--base-template",
        "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed{seed}_fake_1000k.csv",
        "--donor-globs",
        ",".join(
            [
                "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306*_seed{seed}_fake_1000k.csv",
                "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305*_seed{seed}_fake_1000k.csv",
                "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304*_seed{seed}_fake_1000k.csv",
                "/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292*_seed{seed}_fake_1000k.csv",
            ]
        ),
        "--tag-prefix",
        args.tag_prefix,
        "--pipeline",
        "8192",
        "--cache-sizes",
        "32,128,512,2048,8192,32768",
        "--policies",
        "lru,arc,fifo,sieve,slru,car,lfu,lirs",
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

