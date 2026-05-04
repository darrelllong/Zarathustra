"""LLNL ird_renewal.py — empirical IRD + IRM renewal generator.

Reverse-engineered from LANL R285 description in RESPONSE-LLNL.md / LEADER-BOARD.md.
Algorithm:
  1. Read real trace; compute per-position stack-distance (IRD).
  2. Build empirical IRD distribution (quantized into ird_scale buckets).
  3. Optional: condition the IRD distribution on rank (rank_ird_buckets > 0).
  4. Generation loop on a virtual LRU stack:
       - With prob independent_prob: emit a fresh new key (IRM miss).
       - Otherwise: sample an IRD from empirical distribution; emit the key
         currently at stack-position IRD.
       - After emission, move emitted key to top of stack (LRU update).

This is NOT an atlas; no neural net. Reproduces real-trace IRD / rank
shape directly. Designed for KV / non-storage corpora where neural-atlas
scale-sharpening over-concentrates rank=0.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def read_real_csv(path: str, max_rows: int | None = None) -> list[tuple]:
    """Read real trace CSV: (stream_id, ts, obj_id, obj_size, opcode, tenant)."""
    rows: list[tuple] = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)  # stream_id,ts,obj_id,obj_size,opcode,tenant
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            rows.append(tuple(row))
    return rows


def stack_distance_per_record(obj_ids) -> np.ndarray:
    """Compute LRU stack-distance for each record. -1 = first access (cold)."""
    n = len(obj_ids)
    sd = np.empty(n, dtype=np.int64)
    last_pos: dict[int, int] = {}
    # We do not need a true LRU stack to compute stack distance by 'unique-since'
    # approximation: distance = number of unique keys between this access and
    # the previous access of the same key. Use ordered dict-style tracking.
    # For speed, use a dict that maps obj -> last_index, plus a counter of
    # unique keys seen since that index. But that requires range-uniques which
    # is O(n^2). Simpler exact computation: maintain ordered list (LRU stack).
    stack: list[int] = []
    pos_of: dict[int, int] = {}  # key -> position in stack (top=last)
    for i, obj in enumerate(obj_ids.tolist()):
        if obj in pos_of:
            p = pos_of[obj]
            # stack-distance = (len(stack) - 1 - p)  (top has dist 0)
            sd[i] = len(stack) - 1 - p
            # remove from old position, push to top
            del stack[p]
            for k in stack[p:]:
                pos_of[k] -= 1
            stack.append(obj)
            pos_of[obj] = len(stack) - 1
        else:
            sd[i] = -1
            stack.append(obj)
            pos_of[obj] = len(stack) - 1
    return sd


def build_ird_distribution(sd: np.ndarray, ird_buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucket the stack-distances into ird_buckets quantile bins.
    Returns (bin_edges, bin_probs)."""
    valid = sd[sd >= 0]
    if len(valid) == 0:
        return np.array([0]), np.array([1.0])
    # Use log-spaced bins to capture heavy tail.
    log_max = float(np.log1p(valid.max()))
    edges = np.expm1(np.linspace(0, log_max, ird_buckets + 1)).astype(np.int64)
    edges[0] = 0
    edges[-1] = max(edges[-1], int(valid.max()) + 1)
    counts, _ = np.histogram(valid, bins=edges)
    probs = counts.astype(np.float64) / max(counts.sum(), 1)
    return edges, probs


def sample_ird(rng: np.random.Generator, edges: np.ndarray, probs: np.ndarray) -> int:
    """Sample an integer IRD from the bucketed distribution."""
    bin_idx = rng.choice(len(probs), p=probs)
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    if hi <= lo:
        return int(lo)
    return int(rng.integers(lo, hi))


def generate_ird_renewal(
    real_obj_ids: np.ndarray,
    real_sizes: np.ndarray,
    n_records: int,
    seed: int = 42,
    ird_buckets: int = 32,
    independent_prob: float = 0.10,
    rank_ird_buckets: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic trace using empirical IRD + IRM renewal.

    Returns (synthetic_obj_ids, synthetic_obj_sizes).
    """
    rng = np.random.default_rng(seed)

    # Build empirical IRD distribution from real trace.
    sd = stack_distance_per_record(real_obj_ids)
    edges, probs = build_ird_distribution(sd, ird_buckets)

    # Empirical size distribution (independent of stack position).
    sizes_real = real_sizes[real_sizes > 0]
    if len(sizes_real) == 0:
        sizes_real = np.array([4096])

    # Rank-conditioned IRD distributions (optional).
    rank_dists: list[tuple[np.ndarray, np.ndarray]] | None = None
    if rank_ird_buckets > 0:
        rank_dists = []
        valid = sd[sd >= 0]
        rank_edges = np.expm1(np.linspace(0, float(np.log1p(valid.max() if len(valid) > 0 else 1)),
                                          rank_ird_buckets + 1)).astype(np.int64)
        for r in range(rank_ird_buckets):
            lo, hi = rank_edges[r], rank_edges[r + 1]
            mask = (sd >= lo) & (sd < hi)
            sub = sd[mask]
            if len(sub) == 0:
                rank_dists.append((edges, probs))
            else:
                rank_dists.append(build_ird_distribution(sub, ird_buckets))

    # Generation: maintain virtual LRU stack.
    out_ids = np.empty(n_records, dtype=np.uint64)
    out_sizes = np.empty(n_records, dtype=np.int64)
    stack: list[int] = []
    pos_of: dict[int, int] = {}
    next_new_id = np.uint64(10_000_000)  # synthetic IDs

    for i in range(n_records):
        # Decide between independent (new) miss or IRD-driven reuse.
        if len(stack) == 0 or rng.random() < independent_prob:
            obj = next_new_id
            next_new_id += 1
            out_ids[i] = obj
            out_sizes[i] = sizes_real[rng.integers(0, len(sizes_real))]
            stack.append(obj)
            pos_of[obj] = len(stack) - 1
        else:
            # Sample IRD; pick key at that stack position from top.
            if rank_dists is not None:
                # Choose rank bucket uniformly (or could be weighted by rank PMF).
                rank_idx = rng.integers(0, len(rank_dists))
                e2, p2 = rank_dists[rank_idx]
                ird = sample_ird(rng, e2, p2)
            else:
                ird = sample_ird(rng, edges, probs)
            ird = min(ird, len(stack) - 1)
            p = len(stack) - 1 - ird
            obj = stack[p]
            out_ids[i] = obj
            out_sizes[i] = sizes_real[rng.integers(0, len(sizes_real))]
            # LRU update: move obj to top.
            del stack[p]
            for k in stack[p:]:
                pos_of[k] -= 1
            stack.append(obj)
            pos_of[obj] = len(stack) - 1

    return out_ids, out_sizes


def main():
    ap = argparse.ArgumentParser(description="LLNL IRD-renewal generator")
    ap.add_argument("--real", required=True, help="real reference CSV")
    ap.add_argument("--output", required=True, help="output synthetic CSV")
    ap.add_argument("--n", type=int, default=1_000_000, help="records to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ird-scale", type=int, default=32,
                    help="number of bins for empirical IRD distribution")
    ap.add_argument("--independent-prob", type=float, default=0.10,
                    help="probability of an independent (fresh) miss")
    ap.add_argument("--rank-ird-buckets", type=int, default=0,
                    help="rank-conditioned IRD buckets (0 = single global IRD)")
    ap.add_argument("--max-real-rows", type=int, default=1_000_000,
                    help="max real trace rows to read for IRD estimation")
    args = ap.parse_args()

    print(f"Loading real trace from {args.real} (max_rows={args.max_real_rows})")
    rows = read_real_csv(args.real, max_rows=args.max_real_rows)
    obj_ids = np.array([int(r[2]) for r in rows], dtype=np.uint64)
    obj_sizes = np.array([int(r[3]) for r in rows], dtype=np.int64)
    print(f"Read {len(obj_ids):,} real records; building IRD distribution "
          f"(buckets={args.ird_scale}, rank_buckets={args.rank_ird_buckets})")

    out_ids, out_sizes = generate_ird_renewal(
        obj_ids, obj_sizes, n_records=args.n, seed=args.seed,
        ird_buckets=args.ird_scale, independent_prob=args.independent_prob,
        rank_ird_buckets=args.rank_ird_buckets,
    )

    print(f"Generated {len(out_ids):,} synthetic records → {args.output}")
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(out_ids)):
            w.writerow([0, float(i + 1), int(out_ids[i]), int(out_sizes[i]), 0, 0])
    print("done")


if __name__ == "__main__":
    main()
