"""Multi-seed launcher preset for Alibaba r431 (r426-base priority continuation).

This is a wrapper around `altgan.launch_chunk_surface_multiseed` that pins:
- Alibaba official reference path
- official cachesim surface (cache sizes + policies)
- Alibaba splice contract (`stream_id,obj_id,obj_size`)
- r426 banked bases + donor bank
- r426 accepted move JSONs as `--priority-moves` (passed as globs so the shell
  does not expand them into an argv explosion)

Intended usage is via SSH on a `/tiamat`-capable host:

  python3 -m altgan.ssh_chunk_surface_multiseed \
    --host baase \
    --tmux-session ali_r431_r426prio \
    --remote-module altgan.launch_alibaba_r431_chunk_surface_multiseed

This wrapper intentionally optimizes only against the official
`python3 -m llgan.cachesim_eval` surface (via `altgan.launch_chunk_surface_multiseed`).
"""

from __future__ import annotations

import sys

from altgan.launch_chunk_surface_multiseed import main as _main


def main() -> int:
    argv = sys.argv[1:]
    output_root = "/tiamat/zarathustra/altgan-output"
    eval_root = f"{output_root}/cachesim_lanl"
    defaults = [
        "--real",
        "/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv",
        "--cache-sizes",
        "32,128,512,2048,8192",
        "--policies",
        "lru,arc,fifo,sieve,slru,car",
        "--swap-columns",
        "stream_id,obj_id,obj_size",
        "--tag-prefix",
        "alibaba_chunksurf_r431_r426base4_cap16",
        "--pipeline",
        "4",
        "--accept-mode",
        "best",
        "--max-accepts",
        "8",
        "--max-evals",
        "700",
        "--max-candidates-per-chunk",
        "16",
        "--cross-seed-donors",
        "--priority-move-sort",
        "mean",
        "--priority-moves",
        ",".join(
            [
                f"{eval_root}/alibaba_chunksurf_r426_r413base4_cap16_ck4_seed*_moves.json",
                f"{eval_root}/alibaba_chunksurf_r425b_r413base4_ck4_seed*_moves.json",
            ]
        ),
        "--guard-cache-sizes",
        "128,512,2048,8192",
        "--guard-max-regression",
        "0.0",
        "--guard-regression-per-official-gain",
        "0.25",
        "--emit-markdown",
        "--append-markdown",
        "RESPONSE-LANL.md,altgan/RESULTS.md",
        "--base-template",
        f"{output_root}/alibaba_chunksurf_r426_r413base4_cap16_ck4_seed{{seed}}_fake_1000k.csv",
        "--donor-globs",
        ",".join(
            [
                # Always include base family as donors.
                f"{output_root}/alibaba_chunksurf_r426_r413base4_cap16_ck4_seed{{seed}}_fake_1000k.csv",
                # Recent Alibaba chunk-surface families (broad but scoped).
                f"{output_root}/alibaba_chunksurf_r42*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r41*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r40*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r39*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r38*_seed{{seed}}_fake_1000k.csv",
                # Fallback: any other Alibaba chunk-surface artifacts.
                f"{output_root}/alibaba_chunksurf_*_seed{{seed}}_fake_1000k.csv",
                # Last-resort fallback: any Alibaba artifacts under the standard naming contract.
                f"{output_root}/alibaba*_seed{{seed}}_fake_*.csv",
            ]
        ),
    ]
    return _main(defaults + argv)


if __name__ == "__main__":
    raise SystemExit(main())

