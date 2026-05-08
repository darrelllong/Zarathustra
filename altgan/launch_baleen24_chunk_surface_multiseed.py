"""Multi-seed launcher preset for Baleen24 cache-surface chunk selector runs.

This is a thin wrapper around `altgan.launch_chunk_surface_multiseed` that pins
Baleen24's official reference path and defaults to the LANL Baleen24 splice
contract (`stream_id,obj_id,obj_size`).

Intended usage is via SSH on a /tiamat-capable host:

  python3 -m altgan.ssh_chunk_surface_multiseed \
    --host vinge.local \
    --tmux-session bal_rXXX_refine \
    -- \
    --base-template "/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r382_shift1024_ck1024_seed{seed}_fake_1000k.csv" \
    --donor-globs "/tiamat/zarathustra/altgan-output/baleen24_chunksurf_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/baleen24*_seed{seed}_fake_*.csv" \
    --cross-seed-donors \
    --tag-prefix baleen24_chunksurf_rXXX_refine512 \
    --pipeline 512,256,128 \
    --accept-mode best \
    --max-accepts 8 \
    --max-evals 350 \
    --swap-columns stream_id,obj_id,obj_size \
    --guard-cache-sizes 128,512,2048,8192 \
    --guard-max-regression 0.0 \
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
        "/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv",
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
