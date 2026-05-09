"""Locality-Aware Stack Walker (R296) — Denning §7.3 architecture.

Per Coffman & Denning, "Operating Systems Theory" (1973), §7.3.1: a real
reference string is a sequence of locality phases L_0, L_1, ..., L_t, ...,
each quasi-stationary. The principle of locality (L1-L3) asserts:

  L1. During any time interval, references concentrate nonuniformly.
  L2. Correlation between immediate past and future is high; falls to
      zero between non-overlapping reference substrings as their
      separation grows.
  L3. Frequency with which a page is referenced is quasi-stationary
      within a locality (changes slowly).

R295 (stack_walker.py) fitted a single global stack-distance distribution.
That kills L1 and L2 — the synthetic has correct global marginal but
loses the burst-and-quiet structure that recency-aware policies (SIEVE,
SLRU, LIRS) exploit.

R296 fits per-locality working sets and per-locality stack-distance
distributions, plus a locality-transition model. By the LRU-WS duality
(eq. 7.4.1: F_p(LRU) ≅ m(T_p) where s(T_p)=p), matching the per-locality
stack-distance distribution preserves LRU HRC; matching the locality
sequence preserves recency-aware policy behavior because the
within-locality bursts and across-locality cool-downs are reproduced.

Locality segmentation: simple equal-length windows of W references each.
This is the simplest L_t-sequence model in Denning §7.3.1; more
sophisticated changepoint detection is a follow-up.

Algorithm:
  fit:
    1. Read real trace.
    2. Split into n_locs = ceil(N / W) windows of W references.
    3. For each window:
       a. Compute working set W_i = set of distinct obj_ids in window.
       b. Compute LRU stack-distance distribution P_i(d) within window
          (relative to the running global LRU stack).
       c. Compute one-shot rate p_oneshot_i within window.
    4. Compute locality-transition statistics: for each consecutive pair
       (i, i+1), measure the Jaccard distance |W_i ∆ W_{i+1}| / |W_i ∪
       W_{i+1}|. Histogram these to get a locality-shift distribution.

  generate:
    1. For each output locality slot j:
       a. Sample a "shift fraction" s ~ shift_dist; new locality has
          fraction s of objects newly drawn from the global pool, and
          (1-s) inherited from the previous locality's working set.
       b. Within the locality, do a W-record stack-walk using P_i(d)
          sampled from the per-locality fit (we sample which fitted
          locality to mimic uniformly per slot).

This is the simplest faithful Denning-locality-sequence implementation
in the spirit of §7.3.1 — each locality is a stationary stack-walk;
locality transitions reset / shift the working set.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import os
import pickle
import sys
from collections import Counter

import numpy as np


def fit_localities(real_csv: str, window: int = 50_000, max_rows: int = 0) -> dict:
    """Read real trace, segment into windows, fit per-locality stats."""
    print(f"[locality_walker fit] reading {real_csv} window={window:,}", flush=True)
    obj_ids = []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_ids.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)

    # Compute global LRU stack distance for each reference.
    last_pos = {}
    pos_sorted: list = []
    distances = np.empty(n, dtype=np.int64)
    distances.fill(-1)
    for i, oid in enumerate(obj_ids):
        prev = last_pos.get(oid, -1)
        if prev >= 0:
            idx = bisect.bisect_right(pos_sorted, prev)
            distances[i] = len(pos_sorted) - idx
            del pos_sorted[bisect.bisect_left(pos_sorted, prev)]
        bisect.insort(pos_sorted, i)
        last_pos[oid] = i

    # Per-locality stats.
    n_locs = (n + window - 1) // window
    localities = []
    for li in range(n_locs):
        lo, hi = li * window, min((li + 1) * window, n)
        window_obj = obj_ids[lo:hi]
        window_dist = distances[lo:hi]
        W_i = set(window_obj)
        n_oneshot = int((window_dist == -1).sum())
        n_total_w = len(window_obj)
        # Stack-distance distribution within locality (reuses only).
        reuse_dist = window_dist[window_dist >= 0]
        d_counter = Counter(reuse_dist.tolist())
        localities.append({
            "obj_set": W_i,
            "n": n_total_w,
            "n_oneshot": n_oneshot,
            "n_reuse": n_total_w - n_oneshot,
            "stack_dist_counts": dict(d_counter),
        })
    print(f"[locality_walker fit] n={n:,} localities={n_locs} "
          f"avg_locality_size={n/n_locs:.0f} window={window:,}", flush=True)

    # Locality-transition (Jaccard distance) histogram.
    shifts = []
    for li in range(1, n_locs):
        prev_W = localities[li - 1]["obj_set"]
        cur_W = localities[li]["obj_set"]
        if not (prev_W or cur_W):
            shifts.append(0.0)
        else:
            jac = len(prev_W ^ cur_W) / max(len(prev_W | cur_W), 1)
            shifts.append(jac)
    shift_arr = np.array(shifts, dtype=np.float64)
    print(f"[locality_walker fit] shift mean={shift_arr.mean():.3f} "
          f"std={shift_arr.std():.3f} min={shift_arr.min():.3f} "
          f"max={shift_arr.max():.3f}", flush=True)

    return {
        "window": window,
        "n_total": n,
        "footprint": len(last_pos),
        # Strip obj_set down to size + member list to keep pickle smaller.
        "localities": [
            {"n": l["n"], "n_oneshot": l["n_oneshot"], "n_reuse": l["n_reuse"],
             "stack_dist_counts": l["stack_dist_counts"],
             "obj_count": len(l["obj_set"])}
            for l in localities
        ],
        "shift_dist": shift_arr,
    }


def generate(model: dict, n_records: int, seed: int) -> np.ndarray:
    """Generate synthetic trace by walking per-locality stack distributions
    with locality-transitions sampled from the fitted shift distribution."""
    rng = np.random.default_rng(seed)
    window = model["window"]
    localities = model["localities"]
    shift_dist = model["shift_dist"]
    n_locs_fit = len(localities)

    out = np.empty(n_records, dtype=np.uint64)
    next_new_addr = 0
    stack: list = []  # LRU stack: index 0 = most recent

    n_out_locs = (n_records + window - 1) // window
    for locj in range(n_out_locs):
        lo = locj * window
        hi = min((locj + 1) * window, n_records)
        L_size = hi - lo

        # Sample a fitted-locality template uniformly.
        template = localities[rng.integers(0, n_locs_fit)]
        p_oneshot = template["n_oneshot"] / max(template["n"], 1)
        d_keys = list(template["stack_dist_counts"].keys())
        d_weights = np.array(list(template["stack_dist_counts"].values()),
                             dtype=np.float64)
        if d_weights.sum() > 0:
            d_weights /= d_weights.sum()
        else:
            d_keys = [0]
            d_weights = np.array([1.0])
        d_keys_arr = np.array(d_keys, dtype=np.int64)

        # Apply a locality shift: with probability sampled from shift_dist,
        # turnover that fraction of the stack to fresh items at locality
        # boundary. This implements Denning §7.3.1 L_t = L OR L_{t-1}: a
        # locality shift moves a fraction of the working set.
        if locj > 0 and len(shift_dist) > 0:
            shift_frac = float(rng.choice(shift_dist))
            n_evict = int(round(shift_frac * len(stack) * 0.5))
            # Evict from the bottom of the stack (oldest).
            if n_evict > 0 and n_evict < len(stack):
                stack = stack[:-n_evict]

        # Pre-sample one-shot vs reuse decisions for the locality.
        is_oneshot = rng.random(L_size) < p_oneshot
        n_reuses = int((~is_oneshot).sum())
        sampled_d_idx = rng.choice(len(d_keys_arr), size=n_reuses,
                                    replace=True, p=d_weights)
        sampled_d = d_keys_arr[sampled_d_idx]

        reuse_cur = 0
        for j in range(L_size):
            if is_oneshot[j]:
                addr = next_new_addr
                next_new_addr += 1
                out[lo + j] = addr
                stack.insert(0, addr)
            else:
                d = int(sampled_d[reuse_cur])
                reuse_cur += 1
                if d < len(stack):
                    addr = stack.pop(d)
                else:
                    addr = next_new_addr
                    next_new_addr += 1
                out[lo + j] = addr
                stack.insert(0, addr)

    print(f"[locality_walker generate] emitted {n_records:,} records, "
          f"unique={next_new_addr:,}, max_stack={len(stack):,}", flush=True)
    return out


def write_csv(path: str, obj_ids: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


def cmd_fit(args):
    model = fit_localities(args.real, window=args.window, max_rows=args.max_rows)
    with gzip.open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"[locality_walker fit] saved → {args.output}", flush=True)


def cmd_generate(args):
    with gzip.open(args.model, "rb") as f:
        model = pickle.load(f)
    out = generate(model, args.n, args.seed)
    write_csv(args.output, out)
    print(f"[locality_walker generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="R296 Locality-Aware Stack Walker")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--window", type=int, default=50_000)
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
