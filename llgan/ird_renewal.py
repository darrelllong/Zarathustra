"""LLNL ird_renewal.py — empirical IRD + IRM renewal generator.

Reverse-engineered from LANL R285 description in RESPONSE-LLNL.md / LEADER-BOARD.md.
Algorithm:
  1. Read real trace; compute LRU stack-distance per record (O(n log n) via
     sortedcontainers SortedList).
  2. Build empirical IRD distribution (log-spaced bucketing into ird_scale bins).
  3. Optional: rank-conditioned IRD distributions (rank_ird_buckets > 0).
  4. Generation loop on virtual LRU stack maintained as SortedList:
       - With prob independent_prob: emit a fresh new key (IRM miss).
       - Otherwise: sample IRD from empirical distribution; emit the key
         currently at stack-position IRD from top; move-to-top.

This is NOT a neural atlas; reproduces real-trace IRD shape directly.
Designed for KV / non-storage corpora where neural-atlas scale-sharpening
over-concentrates the rank=0 mass.
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
from sortedcontainers import SortedList


def read_real_csv(path: str, max_rows: int | None = None) -> list[tuple]:
    rows: list[tuple] = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            rows.append(tuple(row))
    return rows


def stack_distance_per_record(obj_ids) -> np.ndarray:
    """O(n log n) LRU stack-distance via SortedList of timestamps.
    Returns int64; -1 = first access (cold)."""
    n = len(obj_ids)
    sd = np.full(n, -1, dtype=np.int64)
    ts_of: dict = {}            # obj -> last timestamp
    timestamps = SortedList()    # all current per-key latest timestamps
    for t, obj in enumerate(obj_ids.tolist()):
        if obj in ts_of:
            old_ts = ts_of[obj]
            # Stack-distance = number of distinct keys with ts > old_ts.
            # SortedList.bisect_right(old_ts) gives position of old_ts; entries
            # to the right have larger timestamps and are distinct keys.
            depth = len(timestamps) - timestamps.bisect_right(old_ts)
            sd[t] = depth - 1  # depth=1 means top; sd=0 means top
            timestamps.remove(old_ts)
        timestamps.add(t)
        ts_of[obj] = t
    return sd


def build_ird_distribution(sd: np.ndarray, ird_buckets: int):
    valid = sd[sd >= 0]
    if len(valid) == 0:
        return np.array([0]), np.array([1.0])
    log_max = float(np.log1p(valid.max()))
    edges = np.expm1(np.linspace(0, log_max, ird_buckets + 1)).astype(np.int64)
    edges[0] = 0
    edges[-1] = max(edges[-1], int(valid.max()) + 1)
    counts, _ = np.histogram(valid, bins=edges)
    probs = counts.astype(np.float64) / max(counts.sum(), 1)
    return edges, probs


def sample_ird(rng: np.random.Generator, edges: np.ndarray, probs: np.ndarray) -> int:
    bin_idx = rng.choice(len(probs), p=probs)
    lo = edges[bin_idx]
    hi = edges[bin_idx + 1]
    if hi <= lo:
        return int(lo)
    return int(rng.integers(lo, hi))


def generate_ird_renewal(real_obj_ids, real_sizes, n_records: int, seed: int = 42,
                         ird_buckets: int = 32, independent_prob: float = 0.10,
                         rank_ird_buckets: int = 0, stack_cap: int = 0,
                         irm_mode: str = "fresh"):
    """Generate synthetic trace using empirical IRD + IRM renewal.

    Implementation maintains a virtual LRU stack as SortedList of timestamps.
    'Top' = largest timestamp. To pick at depth k: SortedList[len-1-k] (O(log n)).

    stack_cap: if > 0, evict LRU (oldest timestamp) when stack exceeds cap.
    irm_mode: how to choose object on an "independent miss":
        - "fresh": emit a brand-new synthetic ID (default; old behavior)
        - "real_pool_freq": pick key from real trace, weighted by access frequency
          (Zipf-like; matches real-trace IRM)
        - "real_pool_uniform": pick key uniformly from real-trace key pool
    """
    rng = np.random.default_rng(seed)

    sd = stack_distance_per_record(real_obj_ids)
    edges, probs = build_ird_distribution(sd, ird_buckets)

    sizes_real = real_sizes[real_sizes > 0]
    if len(sizes_real) == 0:
        sizes_real = np.array([4096])

    # Build IRM pool from real trace.
    real_pool: np.ndarray | None = None
    real_pool_probs: np.ndarray | None = None
    if irm_mode in ("real_pool_freq", "real_pool_uniform"):
        unique_keys, counts = np.unique(real_obj_ids, return_counts=True)
        real_pool = unique_keys
        if irm_mode == "real_pool_freq":
            real_pool_probs = counts.astype(np.float64) / counts.sum()
        else:
            real_pool_probs = np.full(len(unique_keys), 1.0 / len(unique_keys))

    rank_dists = None
    if rank_ird_buckets > 0:
        rank_dists = []
        valid = sd[sd >= 0]
        rmax = int(valid.max()) if len(valid) > 0 else 1
        rank_edges = np.expm1(np.linspace(0, float(np.log1p(rmax)),
                                          rank_ird_buckets + 1)).astype(np.int64)
        for r in range(rank_ird_buckets):
            lo, hi = rank_edges[r], rank_edges[r + 1]
            mask = (sd >= lo) & (sd < hi)
            sub = sd[mask]
            if len(sub) == 0:
                rank_dists.append((edges, probs))
            else:
                rank_dists.append(build_ird_distribution(sub, ird_buckets))

    # Generation: virtual LRU stack via SortedList of timestamps.
    # Each timestamp is uniquely associated with one obj_id (current latest access).
    out_ids = np.empty(n_records, dtype=np.uint64)
    out_sizes = np.empty(n_records, dtype=np.int64)

    timestamps = SortedList()
    obj_at_ts: dict = {}    # ts -> obj_id (only the latest ts for each obj)
    ts_of_obj: dict = {}    # obj_id -> latest ts
    next_new_id = np.uint64(10_000_000)
    counter = 0

    for i in range(n_records):
        is_new = (len(timestamps) == 0) or (rng.random() < independent_prob)
        if is_new:
            if real_pool is not None and real_pool_probs is not None:
                # Sample from IRM pool. If sampled key is already in stack,
                # treat as "warm hit" via stack lookup; otherwise inject fresh.
                idx = rng.choice(len(real_pool), p=real_pool_probs)
                obj = np.uint64(real_pool[idx])
                # If obj is already on the stack, remove old timestamp first
                # (this becomes a reuse, not a true insert).
                if obj in ts_of_obj:
                    old_ts = ts_of_obj[obj]
                    if old_ts in obj_at_ts:
                        timestamps.remove(old_ts)
                        del obj_at_ts[old_ts]
            else:
                obj = next_new_id
                next_new_id += np.uint64(1)
        else:
            if rank_dists is not None:
                rank_idx = rng.integers(0, len(rank_dists))
                e2, p2 = rank_dists[rank_idx]
                ird = sample_ird(rng, e2, p2)
            else:
                ird = sample_ird(rng, edges, probs)
            ird = min(ird, len(timestamps) - 1)
            top_idx = len(timestamps) - 1 - ird
            old_ts = timestamps[top_idx]
            obj = obj_at_ts[old_ts]
            timestamps.remove(old_ts)
            del obj_at_ts[old_ts]

        # Add at top with new timestamp.
        ts = counter
        counter += 1
        timestamps.add(ts)
        obj_at_ts[ts] = obj
        ts_of_obj[obj] = ts

        # Evict LRU if over cap.
        if stack_cap > 0 and len(timestamps) > stack_cap:
            old_ts = timestamps[0]
            old_obj = obj_at_ts[old_ts]
            timestamps.remove(old_ts)
            del obj_at_ts[old_ts]
            if ts_of_obj.get(old_obj) == old_ts:
                del ts_of_obj[old_obj]

        out_ids[i] = obj
        out_sizes[i] = sizes_real[rng.integers(0, len(sizes_real))]

    return out_ids, out_sizes


def main():
    ap = argparse.ArgumentParser(description="LLNL IRD-renewal generator")
    ap.add_argument("--real", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ird-scale", type=int, default=32)
    ap.add_argument("--independent-prob", type=float, default=0.10)
    ap.add_argument("--rank-ird-buckets", type=int, default=0)
    ap.add_argument("--max-real-rows", type=int, default=200_000)
    ap.add_argument("--stack-cap", type=int, default=0,
                    help="cap virtual LRU stack size (0 = unbounded). "
                         "Use real working-set size for bounded-WS workloads.")
    ap.add_argument("--irm-mode", default="fresh",
                    choices=["fresh", "real_pool_freq", "real_pool_uniform"],
                    help="how to pick obj on independent miss")
    args = ap.parse_args()

    print(f"Loading real trace from {args.real} (max_rows={args.max_real_rows})", flush=True)
    rows = read_real_csv(args.real, max_rows=args.max_real_rows)
    obj_ids = np.array([int(r[2]) for r in rows], dtype=np.uint64)
    obj_sizes = np.array([int(r[3]) for r in rows], dtype=np.int64)
    print(f"Read {len(obj_ids):,} records; computing IRD distribution "
          f"(ird_scale={args.ird_scale}, rank_ird_buckets={args.rank_ird_buckets})", flush=True)

    out_ids, out_sizes = generate_ird_renewal(
        obj_ids, obj_sizes, n_records=args.n, seed=args.seed,
        ird_buckets=args.ird_scale, independent_prob=args.independent_prob,
        rank_ird_buckets=args.rank_ird_buckets, stack_cap=args.stack_cap,
        irm_mode=args.irm_mode,
    )

    print(f"Writing {len(out_ids):,} records → {args.output}", flush=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(out_ids)):
            w.writerow([0, float(i + 1), int(out_ids[i]), int(out_sizes[i]), 0, 0])
    print("done", flush=True)


if __name__ == "__main__":
    main()
