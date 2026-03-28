"""
LCS Trace Format Survey Tool
=============================

Documents the libCacheSim LCS binary format, shows the parsing logic,
and computes per-file statistics across an LCS corpus directory.

Format reference
----------------
Source: libCacheSim/traceReader/customizedReader/lcs.h
        https://github.com/1a1a11a/libCacheSim

File layout
-----------
  Header : 8192 bytes (= 1024 * 8)
    [0:8]     start_magic = 0x123456789abcdef0  (little-endian uint64)
    [8:16]    version     (uint64: 1, 2, 3, …)
    [16:8008] lcs_trace_stat struct (1000 × 8 bytes):
                int64 stat_version, n_req, n_obj, n_req_byte, n_obj_byte
                int64 start_timestamp, end_timestamp (seconds since epoch)
                int64 n_read, n_write, n_delete
                int64 smallest_obj_size, largest_obj_size
                int64 most_common_obj_sizes[16]
                float most_common_obj_size_ratio[16]
                int64 highest_freq[16]   int32 most_common_freq[16]
                float most_common_freq_ratio[16]   double skewness
                int32 n_tenant   int32 most_common_tenants[16]
                float most_common_tenant_ratio[16]
                … (TTL fields, 897 × int64 padding)
    [8008:8184] uint64 unused[21]
    [8184:8192] end_magic = 0x123456789abcdef0

  Records : immediately after the header, one struct per I/O request.

Record structs (all little-endian, packed)
------------------------------------------
  v1  24 bytes:
      uint32  clock_time          (seconds; absolute Unix time)
      uint64  obj_id              (LBA or object hash)
      uint32  obj_size            (bytes)
      int64   next_access_vtime   (index of next request to same obj; -1=last)
      → No op or tenant fields.

  v2  28 bytes:
      uint32  clock_time
      uint64  obj_id
      uint32  obj_size
      uint32  packed              (bits 0-7 = op code, bits 8-31 = tenant_id)
      int64   next_access_vtime

  v3  36 bytes: like v2 but obj_size is int64, adds uint32 ttl.
  v4-v8: v3 base + 1/2/4/8/16 extra uint32 feature fields.
  → This tool handles v1 and v2 only.

Op codes (from libCacheSim/include/libCacheSim/enum.h)
------------------------------------------------------
  OP_NOP=0  OP_GET=1  OP_GETS=2   OP_SET=3    OP_ADD=4    OP_CAS=5
  OP_REPLACE=6  OP_APPEND=7  OP_PREPEND=8  OP_DELETE=9
  OP_INCR=10  OP_DECR=11
  OP_READ=12  OP_WRITE=13  OP_UPDATE=14
  OP_INVALID=255

  For block-storage LCS files (tencentBlock, alibabaBlock):
    OP_READ=12 → 'r'
    OP_WRITE=13 → 'w'
  Cache-protocol ops (1-11) are treated as reads.

Parsing in Python (struct.unpack / numpy)
-----------------------------------------
  import struct, numpy as np

  MAGIC  = 0x123456789abcdef0
  HEADER = 8192  # bytes

  with open(path, 'rb') as f:
      header = f.read(HEADER)
      magic, version = struct.unpack_from('<QQ', header, 0)
      assert magic == MAGIC

      if version == 1:
          dtype = np.dtype([('ts','<u4'),('obj_id','<u8'),
                            ('obj_size','<u4'),('vtime','<i8')])
      elif version == 2:
          dtype = np.dtype([('ts','<u4'),('obj_id','<u8'),
                            ('obj_size','<u4'),('packed','<u4'),('vtime','<i8')])
      else:
          raise ValueError(f'unsupported LCS version {version}')

      raw = f.read()

  rec_size  = dtype.itemsize
  n_records = len(raw) // rec_size
  arr       = np.frombuffer(raw[:n_records * rec_size], dtype=dtype)

  if version == 2:
      op_raw  = arr['packed'] & 0xFF
      tenant  = arr['packed'] >> 8
      opcode  = np.where(op_raw == 13, 'w', 'r')

Usage
-----
  # Survey a single directory:
  python lcs_survey.py /path/to/cache_dataset_lcs/tencentBlock/

  # Survey multiple directories, write CSV:
  python lcs_survey.py /path/to/cache_dataset_lcs/tencentBlock/ \\
                       /path/to/cache_dataset_lcs/alibaba/ \\
                       --output lcs_stats.csv

  # Limit per-file sampling (faster, less accurate):
  python lcs_survey.py /path/to/lcs/ --max-records 50000
"""

import argparse
import glob
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LCS_MAGIC  = 0x123456789abcdef0
LCS_HEADER = 8192  # bytes

# Offset within the 8192-byte header where the stat struct begins
_STAT_OFFSET = 16


def _read_header(raw: bytes) -> dict:
    """Parse magic, version, and a subset of the stat fields from a raw header."""
    if len(raw) < 16:
        raise ValueError("header too short")
    magic, version = struct.unpack_from("<QQ", raw, 0)
    if magic != LCS_MAGIC:
        raise ValueError(f"bad magic: 0x{magic:016x}")

    stat = {}
    # Stat fields start at byte 16.  Only the first ~10 int64 fields are reliable;
    # later fields were added in newer traces and may be zero in older files.
    # Field order: stat_version, n_req, n_obj, n_req_byte, n_obj_byte,
    #              start_timestamp, end_timestamp, n_read, n_write, n_delete.
    if len(raw) >= _STAT_OFFSET + 10 * 8:
        fields = struct.unpack_from("<10q", raw, _STAT_OFFSET)
        stat["stat_version"]     = fields[0]
        stat["n_req_header"]     = fields[1]   # total records per header
        stat["n_obj"]            = fields[2]
        stat["n_req_byte"]       = fields[3]
        stat["n_obj_byte"]       = fields[4]
        stat["start_timestamp"]  = fields[5]
        stat["end_timestamp"]    = fields[6]
        stat["n_read_header"]    = fields[7]
        stat["n_write_header"]   = fields[8]
        stat["n_delete_header"]  = fields[9]

    return {"version": int(version), **stat}


def _dtype_for_version(version: int) -> np.dtype:
    if version == 1:
        return np.dtype([
            ("ts",       "<u4"),
            ("obj_id",   "<u8"),
            ("obj_size", "<u4"),
            ("vtime",    "<i8"),
        ])
    elif version == 2:
        return np.dtype([
            ("ts",       "<u4"),
            ("obj_id",   "<u8"),
            ("obj_size", "<u4"),
            ("packed",   "<u4"),   # op=bits0-7, tenant=bits8-31
            ("vtime",    "<i8"),
        ])
    else:
        raise ValueError(f"LCS version {version} not supported (only v1, v2)")


def survey_file(path: str, max_records: int = 0) -> dict:
    """
    Read an LCS file (.lcs or .lcs.zst) and return a dict of statistics.

    max_records=0 means read the whole file (can be slow for large files).
    """
    row = {"file": os.path.basename(path), "path": path, "error": None}

    try:
        if path.endswith(".zst"):
            proc = subprocess.Popen(
                ["zstd", "-d", "-c", path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            header_raw = proc.stdout.read(LCS_HEADER)
        elif path.endswith(".gz"):
            import gzip
            f = gzip.open(path, "rb")
            header_raw = f.read(LCS_HEADER)
        else:
            f = open(path, "rb")
            header_raw = f.read(LCS_HEADER)

        info = _read_header(header_raw)
        row.update(info)
        version = info["version"]

        try:
            dt = _dtype_for_version(version)
        except ValueError as e:
            row["error"] = str(e)
            if path.endswith(".zst"):
                proc.stdout.close(); proc.wait()
            else:
                f.close()
            return row

        rec_size = dt.itemsize
        n_bytes  = max_records * rec_size if max_records > 0 else -1

        if path.endswith(".zst"):
            raw = proc.stdout.read(n_bytes if n_bytes > 0 else None)
            proc.stdout.close()
            proc.wait()
        else:
            raw = f.read(n_bytes if n_bytes > 0 else None)
            f.close()

        n = len(raw) // rec_size
        if n == 0:
            row["error"] = "no records"
            return row

        arr = np.frombuffer(raw[:n * rec_size], dtype=dt)
        row["n_records_sampled"] = n
        row["n_records_total"]   = info.get("n_req_header", 0) or n

        # Timestamps
        ts = arr["ts"].astype(np.int64)
        row["ts_start"]   = int(ts.min())
        row["ts_end"]     = int(ts.max())
        row["duration_s"] = int(ts.max() - ts.min()) if len(ts) > 1 else 0

        # Inter-arrival times (clamp negative deltas to 0)
        deltas = np.diff(ts)
        deltas = deltas[deltas >= 0]
        row["iat_mean_ms"]  = round(float(deltas.mean()) * 1000, 3) if len(deltas) else 0
        row["iat_p50_ms"]   = round(float(np.median(deltas)) * 1000, 3) if len(deltas) else 0
        row["iat_p99_ms"]   = round(float(np.percentile(deltas, 99)) * 1000, 3) if len(deltas) else 0

        # Object sizes
        sz = arr["obj_size"]
        row["obj_size_min"]    = int(sz.min())
        row["obj_size_max"]    = int(sz.max())
        row["obj_size_mean"]   = round(float(sz.mean()), 1)
        row["obj_size_p50"]    = int(np.median(sz))
        row["obj_size_p99"]    = int(np.percentile(sz, 99))

        # Object locality
        row["n_unique_obj"] = int(np.unique(arr["obj_id"]).size)

        # Op codes and tenant (v2 only)
        if version == 2:
            op_raw = arr["packed"] & 0xFF
            row["read_pct"]  = round(float((op_raw != 13).mean()) * 100, 1)
            row["write_pct"] = round(float((op_raw == 13).mean()) * 100, 1)
            tenants = arr["packed"] >> 8
            row["n_tenants"] = int(np.unique(tenants).size)
            row["tenant_ids"] = ",".join(str(x) for x in sorted(set(tenants.tolist()))[:5])
        else:
            row["read_pct"]  = None
            row["write_pct"] = None
            row["n_tenants"] = None
            row["tenant_ids"] = None

    except Exception as e:
        row["error"] = str(e)

    return row


def main():
    p = argparse.ArgumentParser(
        description="Survey a directory of .lcs.zst files and print per-file statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dirs", nargs="+", help="Directory (or directories) of .lcs.zst files")
    p.add_argument("--output", "-o", default="", help="Write CSV to this path (default: print table)")
    p.add_argument("--max-records", "-n", type=int, default=50_000,
                   help="Records to sample per file (0 = all; default: 50000)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    files = []
    for d in args.dirs:
        found = sorted(glob.glob(os.path.join(d, "**", "*.lcs.zst"), recursive=True))
        if not found:
            found = sorted(glob.glob(os.path.join(d, "**", "*.lcs"), recursive=True))
        files.extend(found)

    if not files:
        print("No .lcs.zst files found in:", args.dirs, file=sys.stderr)
        sys.exit(1)

    print(f"Surveying {len(files)} files (max_records={args.max_records or 'all'}) …",
          file=sys.stderr)

    rows = []
    for i, path in enumerate(files, 1):
        if args.verbose or i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] {os.path.basename(path)}", file=sys.stderr)
        row = survey_file(path, max_records=args.max_records)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Print compact summary
    display_cols = [
        "file", "version", "n_records_sampled", "n_records_total",
        "duration_s", "read_pct", "write_pct",
        "iat_mean_ms", "iat_p99_ms",
        "obj_size_mean", "obj_size_p99",
        "n_unique_obj", "n_tenants", "error",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    with pd.option_context("display.max_rows", 200, "display.width", 200,
                           "display.float_format", "{:.1f}".format):
        print(df[display_cols].to_string(index=False))

    # Aggregate summary
    good = df[df["error"].isna()]
    if len(good):
        print(f"\n--- Aggregate ({len(good)} files, {args.max_records or 'all'} recs/file) ---")
        print(f"  n_records_total : {good['n_records_total'].sum():,}")
        if "read_pct" in good.columns and good["read_pct"].notna().any():
            print(f"  read%           : {good['read_pct'].mean():.1f}%  "
                  f"(min {good['read_pct'].min():.1f}%, max {good['read_pct'].max():.1f}%)")
        if "iat_mean_ms" in good.columns:
            print(f"  IAT mean (ms)   : {good['iat_mean_ms'].mean():.3f}  "
                  f"(p99 across files: {good['iat_p99_ms'].mean():.3f})")
        if "obj_size_mean" in good.columns:
            print(f"  obj_size mean(B): {good['obj_size_mean'].mean():.0f}  "
                  f"(p99 across files: {good['obj_size_p99'].mean():.0f})")
        if "n_unique_obj" in good.columns:
            print(f"  unique obj/file : {good['n_unique_obj'].mean():.0f}  "
                  f"(range {good['n_unique_obj'].min()}–{good['n_unique_obj'].max()})")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWrote {len(df)} rows → {args.output}")


if __name__ == "__main__":
    main()
