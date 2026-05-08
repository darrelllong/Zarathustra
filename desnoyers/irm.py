"""Independent Reference Model (IRM) trace generator.

Per Coffman & Denning §6.6:
- Each page has a fixed reference probability β_i.
- Successive references are i.i.d. samples from {β_i}.
- The interreference distance for page i is geometric with parameter β_i.

Implementation:
- Fit: extract obj_id frequency distribution from real trace.
- Generate: sample obj_id i.i.d. from frequency distribution. Sample obj_size
  i.i.d. from size distribution conditional on obj_id rank bucket. Time
  stamps are sampled i.i.d. from the gap distribution (also IRM-style).

This is the classic Wang/Khor/Desnoyers baseline for cache-trace synthesis.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np


def fit(real_csv: str, max_rows: int = 0) -> dict:
    """Fit IRM parameters to a real CSV trace.

    Returns dict with:
      obj_freq       : np.ndarray of obj_id frequencies (sorted descending)
      obj_id_pool    : np.ndarray of unique obj_ids
      sizes_by_rank  : dict[rank_bucket -> list of obj_size samples]
      gap_dist       : np.ndarray of inter-event time gaps
      tenants        : Counter of tenant ids
      n_streams      : int (number of distinct stream_ids)
    """
    print(f"[desnoyers.irm fit] Loading {real_csv}", flush=True)
    obj_ids, sizes, tses, streams, tenants = [], [], [], [], []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                streams.append(int(row[0]))
                tses.append(float(row[1]))
                obj_ids.append(int(row[2]))
                sizes.append(int(row[3]))
                tenants.append(int(row[5]) if len(row) > 5 else 0)
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)
    obj_arr = np.array(obj_ids, dtype=np.uint64)
    size_arr = np.array(sizes, dtype=np.int64)
    ts_arr = np.array(tses, dtype=np.float64)

    # Frequency distribution over obj_ids.
    counter = Counter(obj_ids)
    most_common = counter.most_common()
    obj_id_pool = np.array([oid for oid, _ in most_common], dtype=np.uint64)
    obj_freq = np.array([cnt for _, cnt in most_common], dtype=np.float64)
    obj_freq /= obj_freq.sum()

    # Size distribution per rank-bucket (8 log-spaced rank buckets).
    n_unique = len(obj_id_pool)
    rank_to_size_samples = defaultdict(list)
    obj_id_to_rank = {int(oid): r for r, oid in enumerate(obj_id_pool)}
    for oid, sz in zip(obj_ids, sizes):
        rank = obj_id_to_rank[oid]
        bucket = int(np.log2(max(rank, 1) + 1))
        rank_to_size_samples[bucket].append(sz)
    sizes_by_rank = {b: np.array(v, dtype=np.int64) for b, v in rank_to_size_samples.items()}

    # Inter-event time gap distribution.
    if len(ts_arr) > 1:
        gaps = np.diff(ts_arr)
        gaps = gaps[gaps >= 0]  # filter non-monotonic anomalies
    else:
        gaps = np.array([1.0])

    n_streams = len(set(streams))
    tenant_counter = Counter(tenants)

    print(f"[desnoyers.irm fit] rows={n:,} unique_objs={n_unique:,} "
          f"streams={n_streams} top1_freq={obj_freq[0]:.5f}", flush=True)

    return {
        "obj_id_pool": obj_id_pool,
        "obj_freq": obj_freq,
        "sizes_by_rank": sizes_by_rank,
        "gap_dist": gaps.astype(np.float64),
        "tenants_pool": np.array(list(tenant_counter.keys()), dtype=np.int64),
        "tenants_freq": np.array(list(tenant_counter.values()), dtype=np.float64) / max(sum(tenant_counter.values()), 1),
        "n_streams": n_streams,
        "n_records": n,
    }


def generate(model: dict, n: int, seed: int) -> tuple:
    """Generate n synthetic records from the fitted IRM model.

    Returns (stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants) as np.ndarray.
    """
    rng = np.random.default_rng(seed)
    pool = model["obj_id_pool"]
    freq = model["obj_freq"]
    sizes_by_rank = model["sizes_by_rank"]
    gaps = model["gap_dist"]
    n_streams = model["n_streams"]
    tenants_pool = model["tenants_pool"]
    tenants_freq = model["tenants_freq"]

    # Sample obj_id ranks i.i.d. from freq distribution.
    ranks = rng.choice(len(pool), size=n, replace=True, p=freq)
    obj_ids = pool[ranks]

    # Sample sizes from per-rank-bucket distribution.
    obj_sizes = np.empty(n, dtype=np.int64)
    rank_buckets = np.maximum(np.log2(ranks.astype(np.float64) + 1).astype(int), 0)
    # Group by bucket for efficient sampling.
    for bucket in np.unique(rank_buckets):
        mask = rank_buckets == bucket
        n_in = int(mask.sum())
        sample_pool = sizes_by_rank.get(int(bucket), None)
        if sample_pool is None or len(sample_pool) == 0:
            # fall back to global size dist (use bucket 0)
            sample_pool = next(iter(sizes_by_rank.values()))
        idx = rng.integers(0, len(sample_pool), size=n_in)
        obj_sizes[mask] = sample_pool[idx]

    # Sample timestamps from gap distribution (cumulative).
    if len(gaps) >= n - 1:
        ts_gaps = rng.choice(gaps, size=n - 1, replace=False)
    else:
        ts_gaps = rng.choice(gaps, size=n - 1, replace=True)
    ts = np.empty(n, dtype=np.float64)
    ts[0] = 0.0
    ts[1:] = np.cumsum(ts_gaps)

    # Stream IDs uniformly sampled (IRM is stream-agnostic).
    stream_ids = rng.integers(0, max(n_streams, 1), size=n).astype(np.int64)

    # Tenants sampled i.i.d. from tenant frequency dist.
    if len(tenants_pool) > 0:
        tenant_idx = rng.choice(len(tenants_pool), size=n, replace=True, p=tenants_freq)
        tenants = tenants_pool[tenant_idx]
    else:
        tenants = np.zeros(n, dtype=np.int64)

    # Opcodes: 0 (read) — IRM doesn't model opcode structure.
    opcodes = np.zeros(n, dtype=np.int64)

    return stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants


def write_csv(out_path: str, header: list, stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(obj_ids)):
            w.writerow([int(stream_ids[i]), float(ts[i]), int(obj_ids[i]),
                        int(obj_sizes[i]), int(opcodes[i]), int(tenants[i])])


def cmd_fit(args):
    model = fit(args.real, max_rows=args.max_rows)
    with gzip.open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"[desnoyers.irm fit] saved {args.output}", flush=True)


def cmd_generate(args):
    with gzip.open(args.model, "rb") as f:
        model = pickle.load(f)
    print(f"[desnoyers.irm generate] loaded {args.model}", flush=True)
    stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants = generate(model, args.n, args.seed)
    header = ["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"]
    write_csv(args.output, header, stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants)
    print(f"[desnoyers.irm generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="LLNL replication of Wang/Khor/Desnoyers IRM baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Fit IRM parameters to a real trace")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.set_defaults(fn=cmd_fit)

    pg = sub.add_parser("generate", help="Generate a synthetic trace from a fitted IRM model")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seed", type=int, default=42)
    pg.set_defaults(fn=cmd_generate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
