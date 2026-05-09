"""Multi-seed launcher preset for Alibaba cache-surface chunk selector runs.

This is a thin wrapper around `altgan.launch_chunk_surface_multiseed` that pins
Alibaba's official 1M reference path and defaults to the LANL Alibaba splice
contract (`stream_id,obj_id,obj_size`).

Intended usage is via SSH on a /tiamat-capable host:

  python3 -m altgan.ssh_chunk_surface_multiseed \
    --host baase \
    --tmux-session ali_r411_selfshift16 \
    --remote-module altgan.launch_alibaba_r411_chunk_surface_multiseed \
    -- \
    --base-template "/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed{seed}_fake_1000k.csv" \
    --donor-globs "/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_*_seed{seed}_fake_1000k.csv" \
    --cross-seed-donors \
    --tag-prefix alibaba_chunksurf_r411_selfshift16 \
    --pipeline 16 \
    --accept-mode best \
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
    defaults = [
        "--real",
        "/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv",
        "--cache-sizes",
        "32,128,512,2048,8192",
        "--policies",
        "lru,arc,fifo,sieve,slru,car",
        "--swap-columns",
        "stream_id,obj_id,obj_size",
    ]
    return _main(defaults + argv)


if __name__ == "__main__":
    raise SystemExit(main())

