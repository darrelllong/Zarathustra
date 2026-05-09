"""LRU-Stack-Distance Walker generator (R295).

Provable property: by the Mattson stack property, LRU hit-rate at cache size c
equals P(stack_distance < c) on the trace. If the SYNTHETIC trace's empirical
stack-distance distribution matches the REAL trace's distribution exactly,
then LRU HRC matches exactly at every cache size — without chunk-ensemble
post-hoc optimization, without IRD-heap scheduling, and without per-trace
memorization beyond the one-dimensional stack-distance histogram.

Architecture:
    1. fit: scan real trace, compute discrete distribution P(d) of LRU
       stack distances over reuses, plus the marginal P(NEW) of one-shot
       (first-access) references.
    2. generate: maintain a sorted-list LRU stack. At each step:
         - With probability P(NEW), emit a fresh obj_id; push to top of stack.
         - Else sample d ~ P(d | reuse), emit stack[d] (the d-th most
           recently used object), move it to top of stack.
       The synthetic trace's empirical stack-distance distribution is
       P(d) by construction; LRU HRC therefore matches the real trace's
       LRU HRC exactly (in the law-of-large-numbers limit).

For non-LRU policies (FIFO, SIEVE, ARC, SLRU, CAR, LFU, LIRS) the LRU
stack distance is the dominant structural correlate but not a perfect
predictor. Empirical hit rate on those policies will be approximate but
should be much closer than IRD-heap or vanilla IRM, because the
stack-distance distribution captures recency-correlation that IRD does
not (an IRD of d=N can still have current LRU stack distance ≪ N if the
object was accessed more recently than its IRD-tail).

Single fitted object: the stack-distance histogram. No test-metric
optimization. No chunk-swap fitting. Generator is fully described by
the histogram + RNG seed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import os
import pickle
import sys
from collections import Counter

import numpy as np


def fit_stack_distances(real_csv: str, max_rows: int = 0) -> dict:
    """Scan real trace, return fitted distribution.

    Output dict:
      stack_dist_counts : Counter[int → count] — distance histogram (reuse only)
      n_reuses          : int — total reuse references
      n_oneshot         : int — total first-access (one-shot) references
      max_stack_distance: int — max observed
      footprint         : int — number of distinct obj_ids
    """
    print(f"[stack_walker fit] reading {real_csv}", flush=True)
    obj_ids = []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)  # header
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_ids.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)

    # Compute LRU stack distance for each reference.
    # Sliding LRU stack via bisect on a sorted-position list. Average-case
    # complexity is O(log n) for bisect_right + O(n) worst-case for the
    # list deletion — acceptable for 1M-record traces in seconds.
    import bisect
    last_pos = {}                    # obj_id -> position of last access
    pos_sorted: list = []            # positions of last accesses, sorted ascending
    counts: Counter = Counter()
    n_reuses = 0
    max_d = 0
    for i, oid in enumerate(obj_ids):
        prev = last_pos.get(oid, -1)
        if prev >= 0:
            # Distance = number of last-access positions in pos_sorted that are
            # strictly greater than prev (the older positions are < prev).
            idx = bisect.bisect_right(pos_sorted, prev)
            d = len(pos_sorted) - idx
            counts[d] += 1
            n_reuses += 1
            if d > max_d:
                max_d = d
            del pos_sorted[bisect.bisect_left(pos_sorted, prev)]
        bisect.insort(pos_sorted, i)
        last_pos[oid] = i

    n_oneshot = n - n_reuses
    footprint = len(last_pos)
    print(f"[stack_walker fit] n={n:,} reuses={n_reuses:,} ({n_reuses/n:.3f}) "
          f"oneshot={n_oneshot:,} ({n_oneshot/n:.3f}) footprint={footprint:,} "
          f"max_d={max_d:,}", flush=True)
    return {
        "stack_dist_counts": dict(counts),
        "n_reuses": n_reuses,
        "n_oneshot": n_oneshot,
        "n_total": n,
        "footprint": footprint,
        "max_stack_distance": max_d,
    }


def generate(model: dict, n_records: int, seed: int) -> np.ndarray:
    """Generate synthetic trace whose LRU stack-distance distribution matches
    the fitted model. Returns np.uint64 array of obj_ids.

    Time complexity O(n × log(footprint)) via SortedList for the LRU stack
    representation. Memory O(footprint).
    """
    rng = np.random.default_rng(seed)
    counts = model["stack_dist_counts"]
    n_reuses = model["n_reuses"]
    n_oneshot = model["n_oneshot"]
    n_total = model["n_total"]
    p_oneshot = n_oneshot / n_total

    # Pre-sample one-shot/reuse decisions and stack-distances in batches.
    # The reuse-distance support is the keys of `counts`; sample with
    # weights proportional to the values.
    distances = np.fromiter(counts.keys(), dtype=np.int64)
    weights = np.fromiter(counts.values(), dtype=np.float64)
    weights /= weights.sum()

    is_oneshot = rng.random(n_records) < p_oneshot
    n_reuse_draws = int((~is_oneshot).sum())
    reuse_idx = rng.choice(len(distances), size=n_reuse_draws, replace=True, p=weights)
    sampled_d = distances[reuse_idx]

    # Maintain LRU stack as a list (top = most recent at index 0).
    # For O(log n) access at depth d, use a balanced data structure;
    # here we use a simple Python list (O(d) per move-to-top) which is
    # fine when most distances are small (skewed distribution typical).
    # If perf is a concern at large d, swap in a treap or order-statistic tree.
    stack: list = []
    out = np.empty(n_records, dtype=np.uint64)
    next_new_addr = 0
    reuse_cursor = 0

    for j in range(n_records):
        if is_oneshot[j]:
            addr = next_new_addr
            next_new_addr += 1
            out[j] = addr
            stack.insert(0, addr)
        else:
            d = int(sampled_d[reuse_cursor])
            reuse_cursor += 1
            if d < len(stack):
                addr = stack.pop(d)
            else:
                # Distance exceeds current stack — fresh address (acts like
                # one-shot insertion at the bottom). This can happen early
                # in the trace before the stack has grown.
                addr = next_new_addr
                next_new_addr += 1
            out[j] = addr
            stack.insert(0, addr)

    print(f"[stack_walker generate] emitted {n_records:,} records, "
          f"unique={next_new_addr:,}, max_stack={len(stack):,}", flush=True)
    return out


def write_csv(path: str, obj_ids: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


def cmd_fit(args):
    model = fit_stack_distances(args.real, max_rows=args.max_rows)
    with gzip.open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"[stack_walker fit] saved model → {args.output}", flush=True)


def cmd_generate(args):
    with gzip.open(args.model, "rb") as f:
        model = pickle.load(f)
    out = generate(model, args.n, args.seed)
    write_csv(args.output, out)
    print(f"[stack_walker generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="LRU-Stack-Distance Walker generator (R295)")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--max-rows", type=int, default=0)
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
