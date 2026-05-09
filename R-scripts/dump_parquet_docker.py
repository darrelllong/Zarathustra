#!/usr/bin/env python3
"""Dump 2017_docker parquet trace as request_sequence CSV.

Columns emitted (matching dump_trace_records.py request_sequence shape):
  ts, obj_id, obj_size, opcode, tenant, response_time
"""
from __future__ import annotations

import argparse
import csv
import sys
import math
from pathlib import Path

import pyarrow.parquet as pq

OP_MAP = {b"GET": 1, b"HEAD": 1, b"OPTIONS": 1,
          b"POST": -1, b"PUT": -1, b"DELETE": -1, b"PATCH": -1}


def stream(path: Path, out, max_records: int) -> int:
    pf = pq.ParquetFile(str(path))
    cols = ["timestamp", "http.request.method", "http.request.uri",
            "http.request.remoteaddr", "http.response.written",
            "http.request.duration"]
    w = csv.writer(out)
    w.writerow(["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"])
    n = 0
    for batch in pf.iter_batches(batch_size=65536, columns=cols):
        ts = batch.column("timestamp").to_pylist()
        method = batch.column("http.request.method").to_pylist()
        uri = batch.column("http.request.uri").to_pylist()
        remote = batch.column("http.request.remoteaddr").to_pylist()
        size = batch.column("http.response.written").to_pylist()
        dur = batch.column("http.request.duration").to_pylist()
        for i in range(len(ts)):
            if n >= max_records:
                return n
            t = ts[i]
            m = method[i] or ""
            mb = m.encode() if isinstance(m, str) else m
            op = OP_MAP.get(mb, 1)
            obj_id = abs(hash(uri[i] or "")) & 0x7FFFFFFFFFFFFFFF
            tenant = abs(hash(remote[i] or "")) & 0xFFFFFFFF
            sz = size[i] if size[i] is not None else math.nan
            rt = dur[i] if dur[i] is not None else math.nan
            w.writerow([t, obj_id, sz, op, tenant, rt])
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="-")
    ap.add_argument("--max-records", type=int, default=1_000_000)
    args = ap.parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"missing {p}", file=sys.stderr); return 2
    if args.out == "-":
        n = stream(p, sys.stdout, args.max_records)
    else:
        with open(args.out, "w", newline="") as fh:
            n = stream(p, fh, args.max_records)
    print(f"# emitted={n} format=parquet_docker", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
