"""Baleen24 hot-head / singleton-tail object repair.

This post-hoc generator repairs the failure mode seen in LANL Baleen24 fakes:
too many adjacent repeats, too little singleton access mass, and not enough
long-range heavy-head reuse for frequency-aware policies.  It preserves the
base trace timing and marks, then rewrites only synthetic object identity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Input synthetic CSV.")
    parser.add_argument("--output", required=True, help="Output repaired CSV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-adjacent-frac",
        type=float,
        default=0.28,
        help="Target fraction of rows whose cache key equals the previous row.",
    )
    parser.add_argument(
        "--break-mode",
        choices=("unique", "hothead"),
        default="unique",
        help="Rewrite adjacent-repeat breaks as fresh one-shots or into the synthetic hot head.",
    )
    parser.add_argument(
        "--hothead-frac",
        type=float,
        default=0.08,
        help="Fraction of rows redirected to the synthetic long-range hot head.",
    )
    parser.add_argument("--hothead-ids", type=int, default=8)
    parser.add_argument("--hothead-alpha", type=float, default=1.35)
    parser.add_argument(
        "--hothead-min-gap",
        type=int,
        default=64,
        help="Minimum row gap between selected hot-head rewrite positions.",
    )
    parser.add_argument(
        "--hothead-eligible",
        choices=("all", "nonadjacent", "nonsingleton", "nonadjacent_nonsingleton"),
        default="nonadjacent_nonsingleton",
    )
    parser.add_argument(
        "--stream-mode",
        choices=("preserve", "zero"),
        default="preserve",
        help="Keep base stream IDs or collapse repaired trace to stream_id=0.",
    )
    parser.add_argument("--synthetic-base-id", type=int, default=900_000_000_000)
    return parser.parse_args()


def _codes(streams: np.ndarray, obj_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.Series(streams.astype(str)) + "\x1f" + pd.Series(obj_ids.astype(str))
    codes, uniques = pd.factorize(keys, sort=False)
    return codes.astype(np.int64, copy=False), uniques.to_numpy()


def _adjacent_mask(streams: np.ndarray, obj_ids: np.ndarray) -> np.ndarray:
    out = np.zeros(len(obj_ids), dtype=np.bool_)
    if len(obj_ids) > 1:
        out[1:] = (streams[1:] == streams[:-1]) & (obj_ids[1:] == obj_ids[:-1])
    return out


def _spaced_positions(candidates: np.ndarray, target: int, min_gap: int, rng: np.random.Generator) -> np.ndarray:
    if target <= 0 or len(candidates) == 0:
        return np.empty(0, dtype=np.int64)
    candidates = np.asarray(np.unique(candidates), dtype=np.int64)
    if min_gap <= 1:
        if len(candidates) <= target:
            return candidates
        return np.sort(rng.choice(candidates, size=target, replace=False))

    block_ids = (candidates - int(candidates[0])) // min_gap
    blocks = np.unique(block_ids)
    rng.shuffle(blocks)
    selected: list[int] = []
    for block in blocks[:target]:
        lo = int(np.searchsorted(block_ids, block, side="left"))
        hi = int(np.searchsorted(block_ids, block, side="right"))
        if hi <= lo:
            continue
        selected.append(int(candidates[int(rng.integers(lo, hi))]))
    return np.asarray(sorted(selected), dtype=np.int64)


def _stats(label: str, streams: np.ndarray, obj_ids: np.ndarray) -> str:
    codes, _ = _codes(streams, obj_ids)
    counts = np.bincount(codes)
    adj = float(_adjacent_mask(streams, obj_ids).sum()) / max(len(obj_ids), 1)
    singleton_access = float(counts[counts == 1].sum()) / max(len(obj_ids), 1)
    return (
        f"{label}: rows={len(obj_ids)} footprint={len(counts)} "
        f"adjacent_frac={adj:.6f} singleton_access_frac={singleton_access:.6f} "
        f"max_count={int(counts.max()) if len(counts) else 0}"
    )


def main() -> int:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    frame = pd.read_csv(args.base)
    for column in ("stream_id", "obj_id"):
        if column not in frame.columns:
            raise ValueError(f"base trace missing required column {column!r}")

    streams = frame["stream_id"].to_numpy(copy=True)
    if args.stream_mode == "zero":
        streams = np.zeros(len(frame), dtype=np.int64)
    obj_ids = frame["obj_id"].to_numpy(dtype=object, copy=True)

    print(_stats("[hothead] before", streams, obj_ids), flush=True)

    adj = _adjacent_mask(streams, obj_ids)
    current_adj = int(adj.sum())
    target_adj = max(0, min(len(frame), int(round(args.target_adjacent_frac * len(frame)))))
    break_count = max(0, current_adj - target_adj)
    break_candidates = np.flatnonzero(adj)
    if break_count > 0 and len(break_candidates) > 0:
        chosen = rng.choice(break_candidates, size=min(break_count, len(break_candidates)), replace=False)
        for ix, pos in enumerate(chosen):
            if args.break_mode == "hothead":
                stream_part = 0 if args.stream_mode == "zero" else int(streams[int(pos)])
                obj_ids[int(pos)] = int(args.synthetic_base_id + stream_part * 1_000_000 + ix % max(1, args.hothead_ids))
            else:
                obj_ids[int(pos)] = int(args.synthetic_base_id + 100_000_000 + args.seed * 10_000_000 + ix)
        print(f"[hothead] broke_adjacent={len(chosen)} target_adjacent_rows={target_adj}", flush=True)
    else:
        chosen = np.empty(0, dtype=np.int64)
        print("[hothead] broke_adjacent=0", flush=True)

    base_codes, _ = _codes(streams, frame["obj_id"].to_numpy(dtype=object, copy=False))
    base_counts = np.bincount(base_codes)
    base_singleton_pos = base_counts[base_codes] == 1
    base_adj = _adjacent_mask(streams, frame["obj_id"].to_numpy(dtype=object, copy=False))

    eligible = np.ones(len(frame), dtype=np.bool_)
    eligible[chosen] = False
    if args.hothead_eligible in {"nonadjacent", "nonadjacent_nonsingleton"}:
        eligible &= ~base_adj
    if args.hothead_eligible in {"nonsingleton", "nonadjacent_nonsingleton"}:
        eligible &= ~base_singleton_pos

    hot_target = max(0, int(round(args.hothead_frac * len(frame))))
    hot_positions = _spaced_positions(np.flatnonzero(eligible), hot_target, args.hothead_min_gap, rng)
    head_ids = max(1, int(args.hothead_ids))
    ranks = np.arange(1, head_ids + 1, dtype=np.float64)
    weights = np.power(ranks, -max(args.hothead_alpha, 0.0))
    cdf = np.cumsum(weights / weights.sum())
    hot_ranks = np.searchsorted(cdf, rng.random(len(hot_positions)), side="right")
    for pos, rank in zip(hot_positions, hot_ranks, strict=False):
        stream_part = 0 if args.stream_mode == "zero" else int(streams[int(pos)])
        obj_ids[int(pos)] = int(args.synthetic_base_id + stream_part * 1_000_000 + int(rank))
    print(
        f"[hothead] hothead_rewrites={len(hot_positions)} "
        f"hothead_ids={head_ids} eligible={args.hothead_eligible}",
        flush=True,
    )

    frame["stream_id"] = streams
    frame["obj_id"] = obj_ids
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(_stats("[hothead] after", frame["stream_id"].to_numpy(), frame["obj_id"].to_numpy(dtype=object)), flush=True)
    print(f"[hothead] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
