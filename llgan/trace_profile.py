#!/usr/bin/env python3
"""
Trace profiling tool for LLGAN.

Reads any supported trace format and writes a rigorous statistical profile
as a JSON file next to the trace. Useful for conditioning, corpus
characterization, and cross-corpus comparison.

Usage:
    python profile.py /path/to/trace.oracleGeneral.zst --fmt oracle_general
    python profile.py /path/to/trace.csv --fmt csv
    python profile.py /path/to/directory/ --fmt oracle_general
"""

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# Import readers from dataset.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import _READERS

MAX_RECORDS = 1_000_000
SAMPLE_EVERY_STACK = 100  # sample rate for stack distance


# ---------------------------------------------------------------------------
# Column resolution helpers
# ---------------------------------------------------------------------------

def _col(df, candidates, default=None):
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def _get_ts(df):
    c = _col(df, ["ts", "timestamp"])
    return df[c].values.astype(np.float64) if c else None


def _get_obj_id(df):
    c = _col(df, ["obj_id", "lba", "offset", "io_offset"])
    return df[c].values if c else None


def _get_obj_size(df):
    c = _col(df, ["obj_size", "size", "io_size"])
    return df[c].values.astype(np.float64) if c else None


def _get_opcode(df):
    c = _col(df, ["opcode", "type", "rw", "op"])
    if c is None:
        return None
    vals = df[c].values
    # Normalize to 0=read, 1=write
    if vals.dtype == object or vals.dtype.kind == 'U':
        return np.array([0 if str(v).lower().startswith('r') or v == '0' else 1
                         for v in vals], dtype=np.int8)
    return np.where(vals == 0, 0, 1).astype(np.int8)


def _get_tenant(df):
    c = _col(df, ["tenant", "tenant_id", "pipeline"])
    return df[c].values if c else None


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------

def _percentiles(arr, name_prefix=""):
    ps = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(arr, ps)
    return {f"p{p}": round(float(v), 6) for p, v in zip(ps, vals)}


def _compute_temporal(ts):
    """Temporal / inter-arrival time statistics."""
    if ts is None or len(ts) < 2:
        return {}
    ts_sorted = np.sort(ts)
    span = float(ts_sorted[-1] - ts_sorted[0])

    iat = np.diff(ts_sorted)
    # Convert to microseconds — ts units vary by format but most are seconds
    # Heuristic: if max ts > 1e12, assume microseconds already; if > 1e9, nanoseconds; else seconds
    if ts_sorted[-1] > 1e12:
        iat_us = iat  # already microseconds
        span_s = span / 1e6
    elif ts_sorted[-1] > 1e9:
        iat_us = iat / 1e3  # nanoseconds -> microseconds
        span_s = span / 1e9
    else:
        iat_us = iat * 1e6  # seconds -> microseconds
        span_s = span

    iops_per_sec = len(ts) / span_s if span_s > 0 else 0.0
    iat_mean = float(np.mean(iat_us))
    iat_std = float(np.std(iat_us))

    result = {
        "iops_mean": round(iops_per_sec, 6),
        "iops_std": 0.0,  # replaced by _compute_iops_std later
        "iat_mean": round(iat_mean, 6),
        "iat_std": round(iat_std, 6),
        "iat_min": round(float(np.min(iat_us)), 6),
        "iat_max": round(float(np.max(iat_us)), 6),
        "iat_percentiles": _percentiles(iat_us),
        "iat_cv": round(iat_std / iat_mean, 6) if iat_mean > 0 else 0.0,
    }

    # Autocorrelation
    if len(iat_us) > 10:
        iat_centered = iat_us - iat_mean
        var = np.var(iat_us)
        if var > 0:
            def _autocorr(lag):
                n = len(iat_centered)
                if lag >= n:
                    return None
                return round(float(np.mean(iat_centered[:n - lag] * iat_centered[lag:]) / var), 6)
            result["iat_autocorr_lag1"] = _autocorr(1)
            result["iat_autocorr_lag5"] = _autocorr(5)
        else:
            result["iat_autocorr_lag1"] = 0.0
            result["iat_autocorr_lag5"] = 0.0
    else:
        result["iat_autocorr_lag1"] = None
        result["iat_autocorr_lag5"] = None

    return result


def _compute_iops_std(ts, span_s):
    """Compute standard deviation of per-second IOPS. Handles various ts scales."""
    if span_s <= 0 or len(ts) < 2:
        return 0.0
    ts_sorted = np.sort(ts)
    # Normalize to seconds from start
    if ts_sorted[-1] > 1e12:
        ts_sec = (ts_sorted - ts_sorted[0]) / 1e6
    elif ts_sorted[-1] > 1e9:
        ts_sec = (ts_sorted - ts_sorted[0]) / 1e9
    else:
        ts_sec = ts_sorted - ts_sorted[0]
    n_buckets = min(int(span_s) + 1, 100000)
    if n_buckets < 2:
        return 0.0
    counts, _ = np.histogram(ts_sec, bins=n_buckets, range=(0, span_s))
    scale = span_s / n_buckets  # bucket width in seconds
    iops = counts / scale if scale > 0 else counts.astype(float)
    return round(float(np.std(iops)), 6)


def _compute_size(obj_size):
    """Object size statistics."""
    if obj_size is None or len(obj_size) == 0:
        return {}
    s = obj_size[obj_size > 0] if np.any(obj_size > 0) else obj_size

    result = {
        "obj_size_mean": round(float(np.mean(s)), 6),
        "obj_size_std": round(float(np.std(s)), 6),
        "obj_size_min": round(float(np.min(s)), 6),
        "obj_size_max": round(float(np.max(s)), 6),
        "obj_size_percentiles": _percentiles(s),
    }

    # Shannon entropy binned to powers of 2
    if len(s) > 0:
        log_bins = np.floor(np.log2(np.maximum(s, 1))).astype(int)
        _, counts = np.unique(log_bins, return_counts=True)
        probs = counts / counts.sum()
        result["obj_size_entropy"] = round(float(-np.sum(probs * np.log2(probs + 1e-30))), 6)

    # Common sizes (within ±10%)
    n = len(s)
    targets = {"fraction_512B": 512, "fraction_4KB": 4096, "fraction_8KB": 8192,
               "fraction_64KB": 65536, "fraction_1MB": 1048576}
    common = {}
    for name, target in targets.items():
        lo, hi = target * 0.9, target * 1.1
        common[name] = round(float(np.sum((s >= lo) & (s <= hi)) / n), 6)
    result["common_sizes"] = common

    return result


def _compute_access_pattern(obj_id, opcode, n):
    """Access pattern statistics."""
    result = {}
    if opcode is not None:
        reads = np.sum(opcode == 0)
        result["read_ratio"] = round(float(reads / n), 6)
        result["write_ratio"] = round(float(1 - reads / n), 6)
    if obj_id is not None:
        unique = len(np.unique(obj_id))
        result["unique_objects"] = int(unique)
        result["total_requests"] = int(n)
        # Reuse rate: fraction of requests to previously-seen obj_ids
        seen = set()
        reuse_count = 0
        for oid in obj_id:
            if oid in seen:
                reuse_count += 1
            else:
                seen.add(oid)
        result["reuse_rate"] = round(reuse_count / n, 6)
        result["working_set_fraction"] = round(unique / n, 6)
    return result


def _compute_locality(obj_id, obj_size):
    """Locality statistics: sequential fraction, strides, stack distance, Zipf."""
    if obj_id is None or len(obj_id) < 2:
        return {}

    deltas = np.diff(obj_id.astype(np.int64))
    abs_deltas = np.abs(deltas)
    n = len(deltas)

    # Sequential fraction: delta == 1, or delta == prev_size / 512
    if obj_size is not None and len(obj_size) >= 2:
        expected_stride = (obj_size[:-1] / 512).astype(np.int64)
        seq_mask = (deltas == 1) | (deltas == expected_stride)
    else:
        seq_mask = deltas == 1
    seq_frac = float(np.sum(seq_mask)) / n

    result = {
        "sequential_fraction": round(seq_frac, 6),
        "stride_mean": round(float(np.mean(abs_deltas)), 6),
        "stride_std": round(float(np.std(abs_deltas)), 6),
        "stride_zero_fraction": round(float(np.sum(deltas == 0)) / n, 6),
    }

    # Stack distance (reuse distance) — sampled for performance
    last_seen = {}  # obj_id -> position in unique-object stack
    stack = OrderedDict()
    distances = []
    for i, oid in enumerate(obj_id):
        oid_key = int(oid)
        if oid_key in stack:
            # Only sample every SAMPLE_EVERY_STACK'th repeat
            if len(distances) % SAMPLE_EVERY_STACK == 0 or len(obj_id) < 100000:
                # Count unique objects accessed since last access
                dist = 0
                for k in reversed(stack):
                    if k == oid_key:
                        break
                    dist += 1
                distances.append(dist)
            # Move to end (most recent)
            stack.move_to_end(oid_key)
        else:
            stack[oid_key] = True

        # Keep stack bounded for very large traces
        if len(stack) > 500000:
            stack.popitem(last=False)

    if distances:
        d = np.array(distances)
        result["reuse_distance_mean"] = round(float(np.mean(d)), 6)
        result["reuse_distance_median"] = round(float(np.median(d)), 6)
        result["reuse_distance_p95"] = round(float(np.percentile(d, 95)), 6)
    else:
        result["reuse_distance_mean"] = None
        result["reuse_distance_median"] = None
        result["reuse_distance_p95"] = None

    # Zipf alpha estimate: log-log linear regression on object frequency
    _, counts = np.unique(obj_id, return_counts=True)
    if len(counts) >= 100:
        freq_sorted = np.sort(counts)[::-1]
        ranks = np.arange(1, len(freq_sorted) + 1, dtype=np.float64)
        log_rank = np.log(ranks)
        log_freq = np.log(freq_sorted.astype(np.float64))
        # Linear regression: log_freq = -alpha * log_rank + c
        mask = np.isfinite(log_freq) & np.isfinite(log_rank)
        if mask.sum() > 10:
            slope, _ = np.polyfit(log_rank[mask], log_freq[mask], 1)
            result["zipf_alpha_estimate"] = round(float(-slope), 6)
        else:
            result["zipf_alpha_estimate"] = None
    else:
        result["zipf_alpha_estimate"] = None

    return result


def _compute_multi_tenant(tenant):
    """Multi-tenant statistics."""
    if tenant is None:
        return {"num_tenants": 1, "tenant_request_fractions": {}}
    unique, counts = np.unique(tenant, return_counts=True)
    total = counts.sum()
    # Top 10 by frequency
    top_idx = np.argsort(counts)[::-1][:10]
    fracs = {str(unique[i]): round(float(counts[i] / total), 6) for i in top_idx}
    return {
        "num_tenants": int(len(unique)),
        "tenant_request_fractions": fracs,
    }


def _classify_workload(temporal, size, access, locality):
    """Simple heuristic workload classification."""
    tags = []
    read_r = access.get("read_ratio", 0.5)
    if read_r > 0.8:
        tags.append("read_heavy")
    elif read_r < 0.2:
        tags.append("write_heavy")
    else:
        tags.append("mixed")

    seq = locality.get("sequential_fraction", 0)
    if seq > 0.5:
        tags.append("sequential")
    else:
        tags.append("random")

    cv = temporal.get("iat_cv", 0)
    if cv > 2.0:
        tags.append("bursty")

    # Primary label
    primary = tags[0] if tags else "mixed"

    # One-line description
    parts = []
    if read_r > 0.8:
        parts.append(f"Read-heavy ({read_r*100:.0f}% reads)")
    elif read_r < 0.2:
        parts.append(f"Write-heavy ({(1-read_r)*100:.0f}% writes)")
    else:
        parts.append(f"Mixed ({read_r*100:.0f}% reads)")

    # Dominant size
    cs = size.get("common_sizes", {})
    dominant = max(cs, key=cs.get) if cs else None
    if dominant and cs[dominant] > 0.3:
        parts.append(dominant.replace("fraction_", "").replace("B", "B"))

    reuse = access.get("reuse_rate", 0)
    parts.append(f"reuse={reuse*100:.0f}%")

    if cv > 0:
        parts.append(f"IAT CV={cv:.1f}")

    return {
        "workload_type": primary,
        "one_line": ", ".join(parts),
    }


# ---------------------------------------------------------------------------
# Main profiling function
# ---------------------------------------------------------------------------

def profile_trace(path, fmt, max_records=MAX_RECORDS):
    """Profile a single trace file. Returns a dict of statistics."""
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown format: {fmt}. Supported: {list(_READERS.keys())}")

    t0 = time.time()
    df = reader(path, max_records)
    n = len(df)
    if n == 0:
        return {"metadata": {"filename": os.path.basename(path), "error": "empty trace"}}

    sampled = n >= max_records

    # Extract arrays
    ts = _get_ts(df)
    obj_id = _get_obj_id(df)
    obj_size = _get_obj_size(df)
    opcode = _get_opcode(df)
    tenant = _get_tenant(df)

    # Compute time span
    if ts is not None and len(ts) > 0:
        ts_sorted = np.sort(ts)
        span = float(ts_sorted[-1] - ts_sorted[0])
        if ts_sorted[-1] > 1e12:
            span_s = span / 1e6
        elif ts_sorted[-1] > 1e9:
            span_s = span / 1e9
        else:
            span_s = span
    else:
        span_s = 0.0

    # File size
    try:
        file_size = os.path.getsize(path.split("::")[0])
    except OSError:
        file_size = None

    metadata = {
        "filename": os.path.basename(path),
        "format": fmt,
        "num_records": int(n),
        "sampled": sampled,
        "time_span_seconds": round(span_s, 6),
        "file_size_bytes": file_size,
        "profile_time_seconds": None,  # filled at end
    }

    temporal = _compute_temporal(ts)
    # Fix iops_std with dedicated function
    if ts is not None and span_s > 0:
        temporal["iops_std"] = _compute_iops_std(ts, span_s)

    size_stats = _compute_size(obj_size)
    access = _compute_access_pattern(obj_id, opcode, n)
    locality = _compute_locality(obj_id, obj_size)
    multi_tenant = _compute_multi_tenant(tenant)
    summary = _classify_workload(temporal, size_stats, access, locality)

    metadata["profile_time_seconds"] = round(time.time() - t0, 3)

    return {
        "metadata": metadata,
        "temporal": temporal,
        "size": size_stats,
        "access_pattern": access,
        "locality": locality,
        "multi_tenant": multi_tenant,
        "summary": summary,
    }


def _profile_worker(args):
    """Worker for multiprocessing directory mode."""
    path, fmt = args
    try:
        result = profile_trace(path, fmt)
        out_path = path + ".profile.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        return path, result, None
    except Exception as e:
        return path, None, str(e)


# ---------------------------------------------------------------------------
# Corpus summary aggregation
# ---------------------------------------------------------------------------

def _aggregate_corpus(results):
    """Aggregate per-file profiles into a corpus summary."""
    # Collect scalar stats across files
    scalar_keys = {
        "temporal": ["iops_mean", "iops_std", "iat_mean", "iat_std", "iat_cv",
                     "iat_autocorr_lag1", "iat_autocorr_lag5"],
        "size": ["obj_size_mean", "obj_size_std", "obj_size_entropy"],
        "access_pattern": ["read_ratio", "write_ratio", "reuse_rate",
                           "working_set_fraction", "unique_objects"],
        "locality": ["sequential_fraction", "stride_mean", "stride_zero_fraction",
                     "reuse_distance_mean", "zipf_alpha_estimate"],
    }

    aggregated = {}
    for group, keys in scalar_keys.items():
        aggregated[group] = {}
        for key in keys:
            vals = []
            for r in results:
                if r and group in r and key in r[group]:
                    v = r[group][key]
                    if v is not None and not (isinstance(v, float) and math.isnan(v)):
                        vals.append(float(v))
            if vals:
                arr = np.array(vals)
                aggregated[group][key] = {
                    "mean": round(float(np.mean(arr)), 6),
                    "std": round(float(np.std(arr)), 6),
                    "min": round(float(np.min(arr)), 6),
                    "max": round(float(np.max(arr)), 6),
                }

    total_records = sum(r["metadata"]["num_records"] for r in results if r and "metadata" in r)

    # Workload diversity: count distinct types
    types = [r["summary"]["workload_type"] for r in results
             if r and "summary" in r and "workload_type" in r["summary"]]
    type_counts = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "total_files": len(results),
        "total_records": total_records,
        "workload_type_distribution": type_counts,
        "per_stat_aggregation": aggregated,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Profile block I/O traces for LLGAN.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Supported formats: " + ", ".join(sorted(_READERS.keys())),
    )
    parser.add_argument("path", help="Trace file or directory to profile")
    parser.add_argument("--fmt", required=True, choices=list(_READERS.keys()),
                        help="Trace format")
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS,
                        help=f"Max records to read per file (default: {MAX_RECORDS})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers for directory mode (default: cpu_count)")
    args = parser.parse_args()

    target = args.path.rstrip("/")

    if os.path.isdir(target):
        # Directory mode
        files = sorted([
            os.path.join(target, f)
            for f in os.listdir(target)
            if os.path.isfile(os.path.join(target, f))
            and not f.endswith(".profile.json")
            and not f.startswith("_")
            and not f.startswith(".")
        ])
        if not files:
            print(f"No trace files found in {target}")
            sys.exit(1)

        print(f"Profiling {len(files)} files in {target} with format={args.fmt}")
        n_workers = args.workers or min(cpu_count(), len(files), 16)
        work = [(f, args.fmt) for f in files]

        results = []
        errors = 0
        with Pool(n_workers) as pool:
            for i, (fpath, result, err) in enumerate(pool.imap_unordered(_profile_worker, work)):
                if err:
                    print(f"  ERROR {os.path.basename(fpath)}: {err}")
                    errors += 1
                else:
                    results.append(result)
                if (i + 1) % 100 == 0 or i + 1 == len(files):
                    print(f"  {i+1}/{len(files)} done ({errors} errors)")

        # Write corpus summary
        summary = _aggregate_corpus(results)
        summary_path = os.path.join(target, "_corpus_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Corpus summary: {summary_path}")
        print(f"  {summary['total_files']} files, {summary['total_records']} total records")

    else:
        # Single file mode
        result = profile_trace(target, args.fmt, args.max_records)
        out_path = target + ".profile.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Profile written to {out_path}")
        print(f"  {result['metadata']['num_records']} records, "
              f"{result['metadata']['time_span_seconds']:.1f}s span, "
              f"profiled in {result['metadata']['profile_time_seconds']:.1f}s")
        if "summary" in result:
            print(f"  {result['summary']['one_line']}")


if __name__ == "__main__":
    main()
