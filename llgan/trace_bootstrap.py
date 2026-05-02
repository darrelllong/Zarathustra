"""LLNL TraceBootstrap baseline (R259).

Independent re-implementation of altgan/trace_bootstrap.py so the race
has apples-to-apples bootstrap numbers on both sides. The generator is
intentionally nonparametric: it preserves the per-stream object-id and
frequency law from the real trace and perturbs temporal order at a
chunk level. Modes:

  * replay     — identity (returns the real trace unchanged)
  * rotate     — single random cyclic shift
  * shuffle    — uniform random permutation of chunks
  * block-swap — pairwise neighbor swaps with 50% probability

Note: this is a baseline, not a generative model. Cachesim HRC-MAE will
approach zero because the trace nearly equals the real one when modes
are mild. Posted alongside the neural-atlas claims so peer comparisons
remain apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from llgan.long_rollout_eval import _sample_real_stream  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--real-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-size", type=int, default=65_536)
    p.add_argument(
        "--mode",
        choices=("replay", "rotate", "shuffle", "block-swap"),
        default="shuffle",
    )
    p.add_argument(
        "--retime",
        action="store_true",
        help="Replace per-stream timestamps with monotone integer ticks "
             "(usually regresses HRC-MAE; default keeps original ts).",
    )
    return p.parse_args()


def _chunk_order(n_chunks: int, mode: str, rng: np.random.Generator) -> list[int]:
    order = list(range(n_chunks))
    if n_chunks <= 1 or mode == "replay":
        return order
    if mode == "rotate":
        shift = int(rng.integers(1, n_chunks))
        return order[shift:] + order[:shift]
    if mode == "shuffle":
        rng.shuffle(order)
        return order
    if mode == "block-swap":
        for i in range(0, n_chunks - 1, 2):
            if rng.random() < 0.5:
                order[i], order[i + 1] = order[i + 1], order[i]
        return order
    raise ValueError(f"unknown mode {mode!r}")


def main() -> int:
    import pandas as pd

    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    chunk_size = max(int(args.chunk_size), 1)
    real_df, _ = _sample_real_stream(
        args.trace_dir,
        args.fmt,
        args.n_records,
        args.n_streams,
        args.seed,
        manifest_path=args.real_manifest,
    )
    out_parts = []
    for stream_id in sorted(real_df["stream_id"].unique()):
        stream = real_df[real_df["stream_id"] == stream_id].reset_index(drop=True)
        chunks = [
            stream.iloc[start : start + chunk_size]
            for start in range(0, len(stream), chunk_size)
        ]
        order = _chunk_order(len(chunks), args.mode, rng)
        if order:
            boot = pd.concat([chunks[i] for i in order], ignore_index=True)
        else:
            boot = stream.iloc[0:0].copy()
        if args.retime and "ts" in boot.columns:
            boot["ts"] = np.arange(len(boot), dtype=np.float64)
        boot["stream_id"] = stream_id
        out_parts.append(boot)
    fake = pd.concat(out_parts, ignore_index=True)
    fake = fake.iloc[: args.n_records].copy()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fake.to_csv(out, index=False)
    print(
        f"[llgan.trace_bootstrap] wrote {out} rows={len(fake)} "
        f"mode={args.mode} chunk_size={chunk_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
