"""Chunk-bootstrap real-trace manifolds into cache-faithful synthetic traces.

This generator is intentionally nonparametric: it keeps the object identity and
frequency law from the source streams, then perturbs temporal order at a chunk
level. Stack-rank atlases are good LRU-shape models, but LFU/LIRS expose when
the synthetic object-popularity process is wrong. Chunk bootstrapping gives us a
separate architecture that preserves that frequency surface.
"""

from __future__ import annotations

import argparse
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
        default="rotate",
    )
    p.add_argument(
        "--retime",
        action="store_true",
        help="Replace per-stream timestamps with monotone integer ticks.",
    )
    return p.parse_args()


def main() -> int:
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
        boot = stream.iloc[0:0].copy()
        if order:
            boot = __import__("pandas").concat(
                [chunks[i] for i in order],
                ignore_index=True,
            )
        if args.retime and "ts" in boot.columns:
            boot["ts"] = np.arange(len(boot), dtype=np.float64)
        boot["stream_id"] = stream_id
        out_parts.append(boot)
    fake = __import__("pandas").concat(out_parts, ignore_index=True)
    fake = fake.iloc[: args.n_records].copy()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fake.to_csv(out, index=False)
    print(
        "[altgan.trace_bootstrap] wrote "
        f"{out} rows={len(fake)} mode={args.mode} chunk_size={chunk_size}"
    )
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
