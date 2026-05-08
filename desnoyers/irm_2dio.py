"""Two-dimensional IRM (2DIO) — rank PMF conditioned on time-of-trace bucket.

Per Wang/Khor/Desnoyers, the 2DIO extension to vanilla IRM partitions the
trace into time buckets and fits a separate rank PMF per bucket. This
captures coarse phase variation (e.g., diurnal cycles in CDN traffic) that
single-PMF IRM misses, while remaining within the IRM family
(memoryless / asymptotically uncorrelated within a bucket).

Implementation:
- Fit: bucket the trace into B equal-sized chunks by reference position.
       Within each bucket, build a frequency PMF over obj_id ranks (using
       the GLOBAL rank ordering so buckets share an obj_id universe).
- Generate: emit records by bucket. Within bucket b, sample ranks i.i.d.
            from PMF_b.

This remains an IRM model: no autocorrelation or recurrence structure
beyond what bucket-level PMF variation captures.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

# Reuse fit infrastructure.
from desnoyers.irm import write_csv


N_TIME_BUCKETS = 4


def fit(real_csv: str, max_rows: int = 0, n_buckets: int = N_TIME_BUCKETS) -> dict:
    """Fit 2DIO model with B time buckets."""
    print(f"[desnoyers.irm_2dio fit] Loading {real_csv}, n_buckets={n_buckets}", flush=True)
    obj_ids, sizes, tses, streams, tenants = [], [], [], [], []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
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

    # Global rank ordering: by total frequency.
    counter = Counter(obj_ids)
    most_common = counter.most_common()
    obj_id_pool = np.array([oid for oid, _ in most_common], dtype=np.uint64)
    obj_id_to_rank = {int(oid): r for r, oid in enumerate(obj_id_pool)}
    n_unique = len(obj_id_pool)

    # Per-bucket rank PMF.
    bucket_size = (n + n_buckets - 1) // n_buckets
    pmf_per_bucket = []
    for b in range(n_buckets):
        lo = b * bucket_size
        hi = min((b + 1) * bucket_size, n)
        bucket_obj_ids = obj_ids[lo:hi]
        bucket_counter = Counter(bucket_obj_ids)
        # Build PMF over global ranks (sparse — use dict).
        rank_pmf = np.zeros(n_unique, dtype=np.float64)
        for oid, cnt in bucket_counter.items():
            rank_pmf[obj_id_to_rank[oid]] = cnt
        rank_pmf /= max(rank_pmf.sum(), 1)
        pmf_per_bucket.append(rank_pmf)

    # Sizes per global rank bucket (log-spaced).
    rank_to_size_samples = defaultdict(list)
    for oid, sz in zip(obj_ids, sizes):
        rank = obj_id_to_rank[oid]
        bucket = int(np.log2(max(rank, 1) + 1))
        rank_to_size_samples[bucket].append(sz)
    sizes_by_rank = {b: np.array(v, dtype=np.int64) for b, v in rank_to_size_samples.items()}

    if len(ts_arr) > 1:
        gaps = np.diff(ts_arr)
        gaps = gaps[gaps >= 0]
    else:
        gaps = np.array([1.0])

    n_streams = len(set(streams))
    tenant_counter = Counter(tenants)

    print(f"[desnoyers.irm_2dio fit] rows={n:,} unique_objs={n_unique:,} "
          f"buckets={n_buckets} bucket_size={bucket_size:,}", flush=True)

    return {
        "obj_id_pool": obj_id_pool,
        "pmf_per_bucket": pmf_per_bucket,
        "n_buckets": n_buckets,
        "bucket_size": bucket_size,
        "sizes_by_rank": sizes_by_rank,
        "gap_dist": gaps.astype(np.float64),
        "tenants_pool": np.array(list(tenant_counter.keys()), dtype=np.int64),
        "tenants_freq": np.array(list(tenant_counter.values()), dtype=np.float64) / max(sum(tenant_counter.values()), 1),
        "n_streams": n_streams,
        "n_records": n,
    }


def generate(model: dict, n: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    pool = model["obj_id_pool"]
    pmfs = model["pmf_per_bucket"]
    n_buckets = model["n_buckets"]
    sizes_by_rank = model["sizes_by_rank"]
    gaps = model["gap_dist"]
    n_streams = model["n_streams"]
    tenants_pool = model["tenants_pool"]
    tenants_freq = model["tenants_freq"]
    n_unique = len(pool)

    bucket_size = (n + n_buckets - 1) // n_buckets

    # Sample ranks per bucket from per-bucket PMF.
    ranks = np.empty(n, dtype=np.int64)
    for b in range(n_buckets):
        lo = b * bucket_size
        hi = min((b + 1) * bucket_size, n)
        n_in = hi - lo
        if n_in <= 0:
            break
        # Some buckets may have all-zero PMF if real bucket had no records;
        # fall back to bucket 0 PMF.
        pmf = pmfs[b]
        if pmf.sum() == 0:
            pmf = pmfs[0]
        ranks[lo:hi] = rng.choice(n_unique, size=n_in, replace=True, p=pmf)

    obj_ids = pool[ranks]

    # Sizes from per-rank-bucket distribution.
    obj_sizes = np.empty(n, dtype=np.int64)
    rank_buckets = np.maximum(np.log2(ranks.astype(np.float64) + 1).astype(int), 0)
    for bucket in np.unique(rank_buckets):
        mask = rank_buckets == bucket
        n_in = int(mask.sum())
        sample_pool = sizes_by_rank.get(int(bucket), None)
        if sample_pool is None or len(sample_pool) == 0:
            sample_pool = next(iter(sizes_by_rank.values()))
        idx = rng.integers(0, len(sample_pool), size=n_in)
        obj_sizes[mask] = sample_pool[idx]

    # Time gaps and stream/tenant as in vanilla IRM.
    if len(gaps) >= n - 1:
        ts_gaps = rng.choice(gaps, size=n - 1, replace=False)
    else:
        ts_gaps = rng.choice(gaps, size=n - 1, replace=True)
    ts = np.empty(n, dtype=np.float64)
    ts[0] = 0.0
    ts[1:] = np.cumsum(ts_gaps)

    stream_ids = rng.integers(0, max(n_streams, 1), size=n).astype(np.int64)
    if len(tenants_pool) > 0:
        tenant_idx = rng.choice(len(tenants_pool), size=n, replace=True, p=tenants_freq)
        tenants = tenants_pool[tenant_idx]
    else:
        tenants = np.zeros(n, dtype=np.int64)
    opcodes = np.zeros(n, dtype=np.int64)
    return stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants


def cmd_fit(args):
    model = fit(args.real, max_rows=args.max_rows, n_buckets=args.n_buckets)
    with gzip.open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"[desnoyers.irm_2dio fit] saved {args.output}", flush=True)


def cmd_generate(args):
    with gzip.open(args.model, "rb") as f:
        model = pickle.load(f)
    print(f"[desnoyers.irm_2dio generate] loaded {args.model}", flush=True)
    stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants = generate(model, args.n, args.seed)
    header = ["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"]
    write_csv(args.output, header, stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants)
    print(f"[desnoyers.irm_2dio generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="LLNL replication of Wang/Khor/Desnoyers 2DIO baseline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.add_argument("--n-buckets", type=int, default=N_TIME_BUCKETS)
    pf.set_defaults(fn=cmd_fit)

    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seed", type=int, default=42)
    pg.set_defaults(fn=cmd_generate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
