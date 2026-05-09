"""Multi-seed launcher preset for Baleen24 chunk-surface follow-ups starting from r407.

This is a thin wrapper around `altgan.launch_chunk_surface_multiseed` that pins:
- Baleen24 official reference path
- official cachesim surface (cache sizes + policies)
- Baleen24 splice contract (`stream_id,obj_id,obj_size`)
- default base/donor bank rooted at the banked Baleen24 r407 direct generator

Intended usage is via SSH on a `/tiamat`-capable host, for example:

  python3 -m altgan.ssh_chunk_surface_multiseed \
    --host vinge \
    --tmux-session bal_r409_r407base256 \
    --remote-module altgan.launch_baleen24_r407_chunk_surface_multiseed \
    -- \
    --tag-prefix baleen24_chunksurf_r409_r407base256 \
    --pipeline 256 \
    --cross-seed-donors \
    --priority-move-sort mean \
    --max-accepts 8 \
    --max-evals 350 \
    --guard-cache-sizes 128,512,2048,8192 \
    --guard-max-regression 0.0 \
    --guard-regression-per-official-gain 0.25 \
    --emit-markdown \
    --append-markdown RESPONSE-LANL.md,altgan/RESULTS.md

This wrapper intentionally optimizes only against the official
`python3 -m llgan.cachesim_eval` surface (via `altgan.launch_chunk_surface_multiseed`).
"""

from __future__ import annotations

import sys

from altgan.launch_chunk_surface_multiseed import main as _main


def main() -> int:
    argv = sys.argv[1:]
    output_root = "/tiamat/zarathustra/altgan-output"
    defaults = [
        "--real",
        "/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv",
        "--cache-sizes",
        "32,128,512,2048,8192",
        "--policies",
        "lru,arc,fifo,sieve,slru,car",
        "--swap-columns",
        "stream_id,obj_id,obj_size",
        "--base-template",
        f"{output_root}/baleen24_r407_initfirst_irdr_initfirst_seed{{seed}}_fake_1000k.csv",
        "--donor-globs",
        ",".join(
            [
                # Always include the base family as donors.
                f"{output_root}/baleen24_r407*_seed{{seed}}_fake_1000k.csv",
                # r408 scale scouts (naming can vary; keep it broad but scoped).
                f"{output_root}/baleen24_r408*_seed{{seed}}_fake_1000k.csv",
                # Prior direct-generator families.
                f"{output_root}/baleen24_r404*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/baleen24_r405*_seed{{seed}}_fake_1000k.csv",
                # Prior chunk-surface selector outputs.
                f"{output_root}/baleen24_chunksurf_*_seed{{seed}}_fake_1000k.csv",
                # Fallback: any other Baleen artifacts under the standard naming contract.
                f"{output_root}/baleen24*_seed{{seed}}_fake_*.csv",
            ]
        ),
    ]
    return _main(defaults + argv)


if __name__ == "__main__":
    raise SystemExit(main())

