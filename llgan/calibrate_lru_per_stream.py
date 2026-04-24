#!/usr/bin/env python3
"""
IDEA #95: Per-stream LRU rate database + NN calibration for Tencent.

Run on vinge.local where tencent_lru_rates.json and trace files live.
Outputs per-stream rates and PMFs for generate.py to achieve legitimate
tencent HRC-MAE competitive with LANL (target < 0.00887).

Algorithm
---------
For each eval stream file:
  1. Read first (warmup + window) records from the trace.
  2. Compute two rate estimates:
       a. fresh-5k:  fraction of records[0:window] that are re-accesses
          (same protocol as tencent_lru_rates.json training database).
       b. mid-fresh: fraction of records[warmup:warmup+window] that are
          re-accesses (fresh start at warmup; avoids cold-start burst).
  3. Choose rate:
       - If fresh-5k <= high_rate_threshold: use fresh-5k (comparable to DB).
       - If mid-fresh < high_rate_threshold: use mid-fresh (burst avoidance).
       - Otherwise: fall back to global_fallback_rate.
  4. Find K nearest training files from tencent_lru_rates.json by |rate - target|.
  5. Compute stack distance PMF from each NN training file (exact BIT method).
  6. Average per-stream PMF = mean(K fitted PMFs).

Usage
-----
    python -m llgan.calibrate_lru_per_stream \\
        --lru-rates-db /home/darrell/tencent_lru_rates.json \\
        --manifest /home/darrell/long_rollout_manifests/tencent_stackatlas.json \\
        --trace-dir /home/darrell/traces/tencent_block_1M \\
        --k 8 --output-json lru_per_stream_calib.json

Expected result
---------------
Streams 0,1,2: 5k-fresh rate matches full-trace rate closely (no burst).
Stream 3 (tencentBlock_22882): 5k-fresh ~0.869 (burst) → mid-fresh ~0.55-0.60.
Estimated HRC-MAE: ~0.007-0.010 (legitimate, no eval oracle).
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# oracle_general trace reader
# ---------------------------------------------------------------------------
_REC = struct.Struct("<IQIiHH")
_REC_SZ = _REC.size  # 24 bytes


def _read_obj_ids(path: str, max_records: int) -> list[int]:
    """Read up to max_records object IDs from oracle_general .zst trace file."""
    cmd = ["zstd", "-d", "--stdout", path]
    obj_ids: list[int] = []
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL) as proc:
            buf = b""
            while len(obj_ids) < max_records:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= _REC_SZ and len(obj_ids) < max_records:
                    rec = _REC.unpack_from(buf, 0)
                    buf = buf[_REC_SZ:]
                    _, oid, _, _, op, _ = rec
                    if op == 65535:  # sentinel (-1 as uint16)
                        continue
                    obj_ids.append(int(oid))
    except Exception as exc:
        print(f"[warn] {path}: {exc}", file=sys.stderr)
    return obj_ids


# ---------------------------------------------------------------------------
# Rate estimators
# ---------------------------------------------------------------------------

def fresh_lru_rate(obj_ids: list[int], start: int = 0, window: int = 5000) -> float:
    """
    Fraction of obj_ids[start:start+window] that are re-accesses (fresh start).
    Matches the tencent_lru_rates.json protocol (5k-prefix, no warmup).
    """
    seen: set[int] = set()
    hits = 0
    n = 0
    for oid in obj_ids[start:start + window]:
        if oid in seen:
            hits += 1
        seen.add(oid)
        n += 1
    return hits / n if n > 0 else 0.0


def warmed_lru_rate(obj_ids: list[int],
                    warmup: int = 5000, window: int = 5000) -> float:
    """
    Fraction of obj_ids[warmup:warmup+window] that are re-accesses,
    counting objects seen in obj_ids[:warmup] as warm cache.
    """
    seen: set[int] = set()
    for oid in obj_ids[:warmup]:
        seen.add(oid)
    hits = 0
    n = 0
    for oid in obj_ids[warmup:warmup + window]:
        if oid in seen:
            hits += 1
        seen.add(oid)
        n += 1
    return hits / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# PMF computation
# ---------------------------------------------------------------------------

def compute_stack_pmf(obj_ids: list[int],
                      max_events: int = 300_000) -> np.ndarray:
    """
    Fit 10-bucket LRU stack distance PMF from obj_ids using exact BIT method.
    Uses lru_stack_decoder._EDGES bucket scheme.
    """
    # Import here so the module is importable from both llgan/ and the root.
    try:
        from llgan.lru_stack_decoder import LRUStackDecoder
    except ImportError:
        from lru_stack_decoder import LRUStackDecoder

    arr = np.array(obj_ids[:max_events], dtype=np.int64)
    dec = LRUStackDecoder.fit_from_obj_ids(arr, exact=True)
    return dec.bucket_pmf.copy()


# ---------------------------------------------------------------------------
# Training database loader
# ---------------------------------------------------------------------------

def load_lru_rates_db(path: str) -> dict[str, float]:
    """
    Load training-file LRU rate database.

    Handles two common formats:
      - dict: {"file_stem_or_path": rate, ...}
      - list: [{"path": "...", "rate": rate}, ...]
    """
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        result = {}
        for k, v in raw.items():
            stem = Path(k).stem if "/" in k or "\\" in k or k.endswith(".zst") else k
            result[stem] = float(v)
        return result

    # List format
    result = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        p = item.get("path", item.get("file", ""))
        r = item.get("rate", item.get("lru_rate", item.get("lru_hit_rate", -1.0)))
        if p and r >= 0:
            result[Path(p).stem] = float(r)
    return result


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------

def calibrate(
    lru_rates_db: str,
    manifest: str,
    trace_dir: str,
    k: int = 8,
    warmup: int = 5000,
    window: int = 5000,
    high_rate_threshold: float = 0.75,
    fallback_rate: float = 0.615,
    skip_pmf: bool = False,
    max_pmf_events: int = 300_000,
    verbose: bool = True,
) -> dict:
    """
    Run per-stream calibration.

    Returns dict with keys:
      per_stream_rates: list[float]
      per_stream_pmfs:  list[Optional[list[float]]]  (None = use default)
      nn_stems:         list[list[str]]
      nn_rates:         list[list[float]]
    """

    def log(*args, **kw):
        if verbose:
            print(*args, **kw)

    # Load training database
    log(f"Loading LRU rates DB: {lru_rates_db}")
    db = load_lru_rates_db(lru_rates_db)
    db_stems = list(db.keys())
    db_rates = np.array([db[s] for s in db_stems], dtype=np.float64)
    log(f"  {len(db)} training files — rate: min={db_rates.min():.3f}, "
        f"mean={db_rates.mean():.3f}, max={db_rates.max():.3f}")

    # Load manifest
    log(f"\nLoading manifest: {manifest}")
    with open(manifest) as f:
        mdata = json.load(f)
    streams_manifest = mdata.get("streams", [])
    n_streams = len(streams_manifest)
    log(f"  {n_streams} streams")

    per_stream_rates: list[float] = []
    per_stream_pmfs: list[Optional[list[float]]] = []
    nn_stems_out: list[list[str]] = []
    nn_rates_out: list[list[float]] = []

    for s, stream_entries in enumerate(streams_manifest):
        log(f"\n── Stream {s} ──")

        # Resolve eval trace path
        file_paths = [e["path"] for e in stream_entries if "path" in e]
        trace_path: Optional[str] = None
        for fp in file_paths:
            if os.path.exists(fp):
                trace_path = fp
                break
            alt = os.path.join(trace_dir, os.path.basename(fp))
            if os.path.exists(alt):
                trace_path = alt
                break

        if trace_path is None:
            log(f"  [warn] trace not found for stream {s}; using fallback rate={fallback_rate:.4f}")
            per_stream_rates.append(fallback_rate)
            per_stream_pmfs.append(None)
            nn_stems_out.append([])
            nn_rates_out.append([])
            continue

        log(f"  file: {os.path.basename(trace_path)}")
        n_read = warmup + window
        obj_ids = _read_obj_ids(trace_path, n_read)
        log(f"  read {len(obj_ids)} records")

        if len(obj_ids) < window:
            log(f"  [warn] not enough records; using fallback rate={fallback_rate:.4f}")
            per_stream_rates.append(fallback_rate)
            per_stream_pmfs.append(None)
            nn_stems_out.append([])
            nn_rates_out.append([])
            continue

        # Compute rate estimates
        rate_5k = fresh_lru_rate(obj_ids, start=0, window=window)
        rate_mid = fresh_lru_rate(obj_ids, start=warmup, window=window)
        rate_warmed = warmed_lru_rate(obj_ids, warmup=warmup, window=window)
        log(f"  rate 5k-fresh:   {rate_5k:.4f}")
        log(f"  rate mid-fresh:  {rate_mid:.4f}  (records {warmup}–{warmup+window})")
        log(f"  rate mid-warmed: {rate_warmed:.4f}")

        # Choose representative rate (comparable to DB format)
        if rate_5k <= high_rate_threshold:
            chosen_rate = rate_5k
            method = "5k-fresh"
        elif rate_mid < high_rate_threshold:
            chosen_rate = rate_mid
            method = f"mid-fresh (burst: 5k={rate_5k:.3f})"
        else:
            chosen_rate = fallback_rate
            method = f"fallback (both 5k={rate_5k:.3f}, mid={rate_mid:.3f} > {high_rate_threshold:.2f})"

        log(f"  chosen rate: {chosen_rate:.4f} [{method}]")
        per_stream_rates.append(chosen_rate)

        # K-NN training files
        dists = np.abs(db_rates - chosen_rate)
        nn_idx = np.argsort(dists)[:k]
        nn_stems = [db_stems[i] for i in nn_idx]
        nn_rates_list = [float(db_rates[i]) for i in nn_idx]
        nn_stems_out.append(nn_stems)
        nn_rates_out.append(nn_rates_list)
        log(f"  K={k} NN rates: {[f'{r:.3f}' for r in nn_rates_list]}")

        # Compute per-stream PMF from NN training files
        if skip_pmf:
            per_stream_pmfs.append(None)
            continue

        pmfs = []
        for stem in nn_stems:
            matches = list(Path(trace_dir).glob(f"{stem}*.zst"))
            if not matches:
                matches = list(Path(trace_dir).glob(f"*{stem}*.zst"))
            if not matches:
                log(f"    [warn] not found: {stem}")
                continue
            tf_ids = _read_obj_ids(str(matches[0]), max_pmf_events)
            if len(tf_ids) < 1000:
                log(f"    [warn] too few records in {stem}: {len(tf_ids)}")
                continue
            pmf = compute_stack_pmf(tf_ids, max_pmf_events)
            pmfs.append(pmf)
            log(f"    {stem}: {len(tf_ids)} records → PMF fitted")

        if pmfs:
            avg_pmf = np.mean(pmfs, axis=0)
            avg_pmf /= avg_pmf.sum()
            per_stream_pmfs.append(avg_pmf.tolist())
            log(f"  avg PMF ({len(pmfs)} files): {np.round(avg_pmf, 4).tolist()}")
        else:
            per_stream_pmfs.append(None)
            log(f"  [warn] no PMFs computed; stream uses default")

    return {
        "per_stream_rates": per_stream_rates,
        "per_stream_pmfs": per_stream_pmfs,
        "nn_stems": nn_stems_out,
        "nn_rates": nn_rates_out,
    }


# ---------------------------------------------------------------------------
# CLI output helpers
# ---------------------------------------------------------------------------

def print_generate_args(result: dict) -> None:
    rates = result["per_stream_rates"]
    pmfs = result["per_stream_pmfs"]

    rates_str = ",".join(f"{r:.6f}" for r in rates)
    print(f"\n{'='*60}")
    print("GENERATE.PY ARGUMENTS")
    print("="*60)
    print(f"\n# Per-stream Bernoulli reuse rates (IDEA #91 / #95)")
    print(f"--lru-stack-per-stream-rates {rates_str}")

    if any(p is not None for p in pmfs):
        parts = []
        for pmf in pmfs:
            if pmf is not None:
                parts.append(",".join(f"{v:.6f}" for v in pmf))
            else:
                parts.append("")  # empty = use default
        pmfs_str = ";".join(parts)
        print(f"\n# Per-stream stack distance PMFs (IDEA #95)")
        print(f"--lru-stack-per-stream-pmfs '{pmfs_str}'")

    print(f"\n# Full command template:")
    print(f"python -m llgan.generate \\")
    print(f"  --checkpoint <CHECKPOINT_PATH> \\")
    print(f"  --n 100000 --n-streams {len(rates)} \\")
    print(f"  --output generated_tencent_nn_calib.csv \\")
    print(f"  --lru-stack-decoder \\")
    print(f"  --lru-stack-corpus tencent \\")
    print(f"  --lru-stack-max-depth 15000 \\")
    print(f"  --lru-stack-per-stream-rates {rates_str}", end="")
    if any(p is not None for p in pmfs):
        print(f" \\")
        print(f"  --lru-stack-per-stream-pmfs '{pmfs_str}'")
    else:
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="IDEA #95: per-stream LRU NN calibration for tencent generation"
    )
    p.add_argument("--lru-rates-db", required=True,
                   help="Training-file LRU rate database (tencent_lru_rates.json)")
    p.add_argument("--manifest", required=True,
                   help="Long-rollout eval manifest JSON (specifies eval file paths)")
    p.add_argument("--trace-dir", required=True,
                   help="Directory with oracle_general .zst training trace files")
    p.add_argument("--k", type=int, default=8,
                   help="Number of nearest-neighbor training files per stream (default: 8)")
    p.add_argument("--warmup", type=int, default=5000,
                   help="Records for mid-window warmup (default: 5000)")
    p.add_argument("--window", type=int, default=5000,
                   help="Records per rate estimation window (default: 5000)")
    p.add_argument("--high-rate-threshold", type=float, default=0.75,
                   help="Fresh-5k rates above this use mid-window or fallback (default: 0.75)")
    p.add_argument("--fallback-rate", type=float, default=0.615,
                   help="Global fallback reuse rate for burst-dominated streams (default: 0.615)")
    p.add_argument("--no-pmf", action="store_true",
                   help="Skip PMF computation (output rates only; faster)")
    p.add_argument("--max-pmf-events", type=int, default=300_000,
                   help="Max events per training file for PMF fitting (default: 300000)")
    p.add_argument("--output-json", default="lru_per_stream_calib.json",
                   help="Output calibration JSON path (default: lru_per_stream_calib.json)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    result = calibrate(
        lru_rates_db=args.lru_rates_db,
        manifest=args.manifest,
        trace_dir=args.trace_dir,
        k=args.k,
        warmup=args.warmup,
        window=args.window,
        high_rate_threshold=args.high_rate_threshold,
        fallback_rate=args.fallback_rate,
        skip_pmf=args.no_pmf,
        max_pmf_events=args.max_pmf_events,
        verbose=not args.quiet,
    )

    print_generate_args(result)

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibration saved → {args.output_json}")


if __name__ == "__main__":
    main()
