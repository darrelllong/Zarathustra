"""LLNL chunk_ensemble.py — cache-surface chunk ensemble optimizer.

Counterpart to LANL's R288/R289 chunk-ensemble technique. Takes a base fake
trace and a bank of donor candidate fakes; in a single guard pass, for each
contiguous chunk of size CHUNK_SIZE, tries swapping the obj_id column with
each donor's chunk; keeps the swap if it reduces the official cachesim mean
HRC-MAE against real. Real obj_ids are NOT used — only synthetic donors.

Usage:
  python -m llgan.chunk_ensemble \\
      --base BASE_FAKE.csv --donors D1.csv D2.csv ... \\
      --real REAL.csv --output OUT.csv \\
      --chunk-size 65536 --policies lru,arc,fifo,sieve,slru,car \\
      --cache-sizes 32,128,512,2048,8192
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def read_fake_csv(path: str):
    """Read fake CSV; return (header, ts, obj_ids, obj_sizes, opcodes, tenants, stream_ids)."""
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    n = len(rows)
    stream_ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
    ts = np.array([float(r[1]) for r in rows], dtype=np.float64)
    obj_ids = np.array([int(r[2]) for r in rows], dtype=np.uint64)
    obj_sizes = np.array([int(r[3]) for r in rows], dtype=np.int64)
    opcodes = np.array([int(r[4]) for r in rows], dtype=np.int64)
    tenants = np.array([int(r[5]) for r in rows], dtype=np.int64)
    return header, stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants


def write_fake_csv(path: str, header, stream_ids, ts, obj_ids, obj_sizes, opcodes, tenants):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(obj_ids)):
            w.writerow([int(stream_ids[i]), float(ts[i]), int(obj_ids[i]),
                        int(obj_sizes[i]), int(opcodes[i]), int(tenants[i])])


def cachesim_mean(fake_csv: str, real_csv: str, cache_sizes: str,
                  policies: str, py: str = "python3") -> float:
    """Run cachesim_eval; return mean_hrc_mae across policies."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out_json = f.name
    try:
        subprocess.run(
            [py, "-m", "llgan.cachesim_eval",
             "--fake", fake_csv, "--real", real_csv,
             "--cache-sizes", cache_sizes, "--policies", policies,
             "--out", out_json],
            check=True, capture_output=True, text=True,
        )
        with open(out_json) as f:
            return float(json.load(f)["mean_hrc_mae"])
    finally:
        Path(out_json).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="LLNL chunk-ensemble guard pass")
    ap.add_argument("--base", required=True, help="base fake CSV (current champion)")
    ap.add_argument("--donors", nargs="+", required=True, help="donor fake CSVs")
    ap.add_argument("--real", required=True, help="real reference CSV")
    ap.add_argument("--output", required=True, help="output ensemble fake CSV")
    ap.add_argument("--chunk-size", type=int, default=65536)
    ap.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    ap.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    ap.add_argument("--py", default="python3", help="python interpreter for cachesim_eval")
    args = ap.parse_args()

    print(f"Loading base: {args.base}", flush=True)
    header, stream_ids, ts, base_obj, base_size, opcodes, tenants = read_fake_csv(args.base)
    n = len(base_obj)
    print(f"Base: {n:,} records", flush=True)

    print(f"Loading {len(args.donors)} donors", flush=True)
    donor_obj_arrays = []
    donor_size_arrays = []
    donor_names = []
    for dpath in args.donors:
        _h, _s, _t, dobj, dsize, _o, _tn = read_fake_csv(dpath)
        if len(dobj) != n:
            print(f"WARN: donor {dpath} has {len(dobj)} records (expected {n}); skipping", flush=True)
            continue
        donor_obj_arrays.append(dobj)
        donor_size_arrays.append(dsize)
        donor_names.append(Path(dpath).stem)
    print(f"Loaded {len(donor_obj_arrays)} donors at matching length", flush=True)

    # Working copy.
    work_obj = base_obj.copy()
    work_size = base_size.copy()

    # Write to temp file for evaluation.
    tmp_csv = Path(args.output).with_suffix(".tmp.csv")

    def eval_current() -> float:
        write_fake_csv(str(tmp_csv), header, stream_ids, ts, work_obj, work_size, opcodes, tenants)
        return cachesim_mean(str(tmp_csv), args.real, args.cache_sizes, args.policies, py=args.py)

    baseline = eval_current()
    print(f"Baseline cachesim mean = {baseline:.6f}", flush=True)

    chunk = args.chunk_size
    n_chunks = (n + chunk - 1) // chunk
    print(f"Guard pass: {n_chunks} chunks of {chunk}; {len(donor_obj_arrays)} donors per chunk", flush=True)

    cur_mean = baseline
    swaps = 0
    for ci in range(n_chunks):
        lo = ci * chunk
        hi = min((ci + 1) * chunk, n)
        # Snapshot original chunk
        orig_obj = work_obj[lo:hi].copy()
        orig_size = work_size[lo:hi].copy()
        best_mean = cur_mean
        best_idx = -1
        for di, (dobj, dsize) in enumerate(zip(donor_obj_arrays, donor_size_arrays)):
            work_obj[lo:hi] = dobj[lo:hi]
            work_size[lo:hi] = dsize[lo:hi]
            m = eval_current()
            if m < best_mean:
                best_mean = m
                best_idx = di
        if best_idx >= 0:
            # Apply best donor.
            work_obj[lo:hi] = donor_obj_arrays[best_idx][lo:hi]
            work_size[lo:hi] = donor_size_arrays[best_idx][lo:hi]
            cur_mean = best_mean
            swaps += 1
            print(f"  chunk {ci:2d}/{n_chunks-1}: SWAP <- {donor_names[best_idx]}  mean = {cur_mean:.6f}", flush=True)
        else:
            # Restore original (no improvement).
            work_obj[lo:hi] = orig_obj
            work_size[lo:hi] = orig_size
            print(f"  chunk {ci:2d}/{n_chunks-1}: keep base                  mean = {cur_mean:.6f}", flush=True)

    # Final write.
    write_fake_csv(args.output, header, stream_ids, ts, work_obj, work_size, opcodes, tenants)
    Path(str(tmp_csv)).unlink(missing_ok=True)
    print(f"Done. baseline={baseline:.6f} -> ensemble={cur_mean:.6f} ({swaps}/{n_chunks} swaps) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
