"""
IDEA #63: Compute conditional stack-distance PMF P(bucket | dt_bin) from real traces.

For each reuse event, we pair the true LRU stack distance (BIT-based) with the
interarrival time dt = time since the previous access to the same object.
Binning dt into N quantile bins gives a (N, 8) conditional PMF matrix.

Usage
-----
    python compute_cond_pmf.py \
        --trace-dir /tiamat/zarathustra/traces/alibaba \
        --output cond_pmf_alibaba.npy \
        [--n-bins 4] \
        [--max-events-per-file 200000]

Output
------
    cond_pmf.npy     : (N, 8) float64, rows normalised to sum 1
    cond_pmf_edges.npy : (N-1,) float64, dt_bin edges in log(dt+1) space
"""

import argparse
import glob
import os
import struct
import subprocess

import numpy as np

from lru_stack_decoder import N_BUCKETS, _EDGES


def _read_oracle_general(path: str, max_records: int):
    """Read (ts, obj_id) columns from oracle_general .zst file."""
    n_bytes = max_records * 24
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    else:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)

    dt = np.dtype([
        ("ts",       "<u4"),
        ("obj_id",   "<u8"),
        ("obj_size", "<u4"),
        ("vtime",    "<i4"),
        ("op",       "<i2"),
        ("tenant",   "<i2"),
    ])
    n = len(raw) // 24
    if n == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.int64)
    arr = np.frombuffer(raw[:n * 24], dtype=dt)
    return arr["ts"].astype(np.float64), arr["obj_id"].astype(np.int64)


def _stack_distances_with_dt(ts: np.ndarray, obj_ids: np.ndarray):
    """
    Compute (lru_stack_distance, interarrival_dt) pairs for all reuse events.
    Uses BIT-based exact stack distances.

    Returns: two arrays of equal length — stack_dists and dt_vals.
    """
    N = len(obj_ids)
    if N == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)

    # BIT for exact stack distances
    bit: list[int] = [0] * (N + 2)

    def update(idx, delta):
        idx += 1
        while idx <= N:
            bit[idx] += delta
            idx += idx & -idx

    def prefix(idx):
        idx += 1
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    last_pos: dict[int, int] = {}
    last_ts: dict[int, float] = {}
    active = 0
    stack_dists = []
    dt_vals = []

    for i in range(N):
        oid = int(obj_ids[i])
        t = float(ts[i])
        if oid in last_pos:
            prev = last_pos[oid]
            sd = active - prefix(prev)
            stack_dists.append(sd)
            dt_vals.append(t - last_ts[oid])
            update(prev, -1)
            active -= 1
        last_pos[oid] = i
        last_ts[oid] = t
        update(i, 1)
        active += 1

    return (np.asarray(stack_dists, dtype=np.int64),
            np.asarray(dt_vals, dtype=np.float64))


def compute_cond_pmf(
    trace_dir: str,
    n_bins: int = 4,
    max_files: int = 0,
    max_events_per_file: int = 200_000,
    verbose: bool = True,
):
    """
    Returns:
        cond_pmf   : (n_bins, N_BUCKETS) array, rows normalise to 1
        dt_edges   : (n_bins-1,) bin edges in log(dt+1) space
    """
    # Phase 1: collect all (log_dt, bucket) pairs to determine quantile edges
    log_dts_all = []
    file_data = []  # list of (ts_arr, obj_ids_arr)

    files = sorted(glob.glob(os.path.join(trace_dir, "*.oracleGeneral.zst")))
    if not files:
        files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
    if max_files > 0:
        files = files[:max_files]
    print(f"Found {len(files)} trace files")

    for fi, fpath in enumerate(files):
        try:
            ts_arr, obj_ids = _read_oracle_general(fpath, max_events_per_file)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue
        if len(ts_arr) < 2:
            continue
        file_data.append((ts_arr, obj_ids))
        # Quick IRD pass to get log_dt samples (faster than BIT for edge fitting)
        last_t: dict[int, float] = {}
        for i, oid in enumerate(obj_ids.tolist()):
            t = float(ts_arr[i])
            if oid in last_t:
                log_dts_all.append(np.log1p(abs(t - last_t[oid])))
            last_t[oid] = t
        if verbose and fi % 20 == 0:
            print(f"  pass1 [{fi+1}/{len(files)}]", flush=True)

    if not log_dts_all:
        raise ValueError("No reuse events found — check trace files")

    # Compute quantile edges from all log_dt values
    log_dts_arr = np.array(log_dts_all)
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    dt_edges = np.percentile(log_dts_arr, quantiles)
    print(f"dt_bin edges (log(dt+1) space): {np.round(dt_edges, 3)}")
    print(f"  → dt edges in raw units: {np.round(np.expm1(dt_edges), 1)}")

    # Phase 2: accumulate conditional PMF counts using exact BIT stack distances
    counts = np.zeros((n_bins, N_BUCKETS), dtype=np.float64)
    total = 0

    for fi, (ts_arr, obj_ids) in enumerate(file_data):
        sd, dt = _stack_distances_with_dt(ts_arr, obj_ids)
        if len(sd) == 0:
            continue
        log_dt = np.log1p(np.abs(dt))
        bins = np.searchsorted(dt_edges, log_dt, side="right")  # 0..n_bins-1
        buckets = np.searchsorted(_EDGES[1:], sd, side="right")
        buckets = np.clip(buckets, 0, N_BUCKETS - 1)
        for b_dt, b_sd in zip(bins.tolist(), buckets.tolist()):
            counts[b_dt][b_sd] += 1
            total += 1
        if verbose and fi % 20 == 0:
            print(f"  pass2 [{fi+1}/{len(file_data)}]  {total:,} pairs", flush=True)

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cond_pmf = counts / row_sums

    print(f"\nTotal (dt_bin, stack_bucket) pairs: {int(total):,}")
    from lru_stack_decoder import _EDGES as EDGES
    labels = [f"[{EDGES[k]},{EDGES[k+1]})" for k in range(N_BUCKETS)]
    labels[-1] = f"[{EDGES[-2]},+inf)"
    print(f"\nConditional PMF P(bucket | dt_bin):")
    header = f"  {'dt_bin':>8}  " + "  ".join(f"{l:>10s}" for l in labels)
    print(header)
    for i, row in enumerate(cond_pmf):
        dt_range = f"bin{i}"
        vals = "  ".join(f"{v:.4f}" for v in row)
        print(f"  {dt_range:>8s}  {vals}")

    return cond_pmf, dt_edges


def main():
    p = argparse.ArgumentParser(
        description="Compute P(stack_bucket | dt_bin) from real traces (IDEA #63)"
    )
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--output", default="cond_pmf.npy",
                   help="Output path for (n_bins, 8) conditional PMF array")
    p.add_argument("--edges-output", default="cond_pmf_edges.npy",
                   help="Output path for (n_bins-1,) dt bin edges")
    p.add_argument("--n-bins", type=int, default=4)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--max-events-per-file", type=int, default=200_000)
    args = p.parse_args()

    cond_pmf, dt_edges = compute_cond_pmf(
        args.trace_dir,
        n_bins=args.n_bins,
        max_files=args.max_files,
        max_events_per_file=args.max_events_per_file,
    )
    np.save(args.output, cond_pmf)
    np.save(args.edges_output, dt_edges)
    print(f"\nSaved: {args.output} shape={cond_pmf.shape}")
    print(f"Saved: {args.edges_output} shape={dt_edges.shape}")


if __name__ == "__main__":
    main()
