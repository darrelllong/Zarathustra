"""Multi-seed launcher preset for Alibaba chunk-surface follow-ups starting from r413.

This is a thin wrapper around `altgan.launch_chunk_surface_multiseed` that pins:
- Alibaba official 1M reference path
- official cachesim surface (cache sizes + policies)
- Alibaba splice contract (`stream_id,obj_id,obj_size`)
- default base/donor bank rooted at the banked Alibaba r413 continuation

Intended usage is via SSH on a `/tiamat`-capable host, for example:

  python3 -m altgan.ssh_chunk_surface_multiseed \
    --host baase \
    --tmux-session ali_r425_r413base4 \
    --remote-module altgan.launch_alibaba_r413_chunk_surface_multiseed \
    -- \
    --tag-prefix alibaba_chunksurf_r425_r413base4 \
    --pipeline 4 \
    --cross-seed-donors \
    --accept-mode best \
    --max-accepts 8 \
    --max-evals 350 \
    --donor-shifts -4096,-2048,-1024,-512,-256,-128,-64,-32,-16,-8,0,8,16,32,64,128,256,512,1024,2048,4096 \
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
        "/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv",
        "--cache-sizes",
        "32,128,512,2048,8192",
        "--policies",
        "lru,arc,fifo,sieve,slru,car",
        "--swap-columns",
        "stream_id,obj_id,obj_size",
        "--base-template",
        f"{output_root}/alibaba_chunksurf_r413_r411base8_ck8_seed{{seed}}_fake_1000k.csv",
        "--donor-globs",
        ",".join(
            [
                # Always include the base family as donors.
                f"{output_root}/alibaba_chunksurf_r413_*_seed{{seed}}_fake_1000k.csv",
                # Banked continuation parents.
                f"{output_root}/alibaba_chunksurf_r411_*_seed{{seed}}_fake_1000k.csv",
                # Prior stabilized pool.
                f"{output_root}/alibaba_chunksurf_r386_*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r384_*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r368_*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r364_*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r360_*_seed{{seed}}_fake_1000k.csv",
                f"{output_root}/alibaba_chunksurf_r340_*_seed{{seed}}_fake_1000k.csv",
                # Any other Alibaba chunk-surface artifacts under the standard naming contract.
                f"{output_root}/alibaba_chunksurf_*_seed{{seed}}_fake_1000k.csv",
                # Fallback: any other Alibaba artifacts (scoped to this corpus prefix).
                f"{output_root}/alibaba*_seed{{seed}}_fake_*.csv",
            ]
        ),
    ]
    return _main(defaults + argv)


if __name__ == "__main__":
    raise SystemExit(main())

