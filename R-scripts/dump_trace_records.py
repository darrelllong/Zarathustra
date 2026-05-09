#!/usr/bin/env python3
"""Stream a Zarathustra trace to a CSV of numeric records.

Reuses parsers/core.py for format detection and zstd handling. Emits one
of three CSV shapes depending on the trace kind:

  request_sequence     ts,obj_id,obj_size,opcode,tenant,response_time
  aggregate_time_series ts,read_iops,read_bw,write_iops,write_bw,disk_usage
  structured_table     ts,v1,v2,v3,v4,v5    (first 6 numeric columns)

The R analysis driver reads the resulting CSV with data.table::fread.
"""
from __future__ import annotations

# Re-exec into venv python if it exists and our python lacks pyarrow,
# so parquet traces work without the caller setting PATH.
import os as _os, sys as _sys
_venv = "/tmp/parquet-env/bin/python3"
if (_os.environ.get("_PARQUET_REEXEC") != "1"
        and _os.path.exists(_venv)
        and _sys.executable != _venv):
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        _os.environ["_PARQUET_REEXEC"] = "1"
        _os.execv(_venv, [_venv, *_sys.argv])

import argparse
import csv
import gzip
import importlib.util
import math
import re
import struct
import sys
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def _load_core(parsers_path: Path):
    spec = importlib.util.spec_from_file_location("core", str(parsers_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load parser module {parsers_path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def _open_zstd_bytes(path: Path):
    """Stream zstd-compressed binary content via the `zstd` CLI."""
    import subprocess
    if str(path).endswith(".zst"):
        return subprocess.Popen(
            ["zstd", "-d", "-c", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
    return open(path, "rb")


def _open_zstd_text(path: Path):
    import subprocess
    if str(path).endswith(".zst"):
        return subprocess.Popen(
            ["zstd", "-d", "-c", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
    return open(path, "rt", errors="replace")


# --------------- Streamers per format ---------------------------------------

def stream_oracle_general(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    record_size = 24
    dt = struct.Struct("<IQIihh")
    n = 0
    fh = _open_zstd_bytes(path)
    try:
        while n < max_records:
            chunk = fh.read(record_size)
            if len(chunk) < record_size:
                break
            ts_v, obj_v, size_v, _vtime, op_v, tenant_v = dt.unpack(chunk)
            opcode = -1.0 if int(op_v) == 1 else 1.0
            yield (float(ts_v), float(obj_v), float(size_v), opcode,
                   float(tenant_v), math.nan)
            n += 1
    finally:
        try:
            fh.close()
        except Exception:
            pass


_LCS_EXTRA_FIELDS = {3: 0, 4: 1, 5: 2, 6: 4, 7: 8, 8: 16}


def stream_lcs(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    header_size = 8192
    magic = 0x123456789ABCDEF0
    fh = _open_zstd_bytes(path)
    try:
        header = fh.read(header_size)
        if len(header) < 16:
            return
        magic_v, version = struct.unpack_from("<QQ", header, 0)
        if magic_v != magic:
            return
        if version == 1:
            rec = struct.Struct("<IQIq")
            n = 0
            while n < max_records:
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                ts_v, obj_v, size_v, _vtime = rec.unpack(chunk)
                yield (float(ts_v), float(obj_v), float(size_v), 1.0,
                       math.nan, math.nan)
                n += 1
        elif version == 2:
            rec = struct.Struct("<IQIIq")
            n = 0
            while n < max_records:
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                ts_v, obj_v, size_v, packed, _vtime = rec.unpack(chunk)
                op_raw = packed & 0xFF
                tenant_v = packed >> 8
                opcode = -1.0 if int(op_raw) == 13 else 1.0
                yield (float(ts_v), float(obj_v), float(size_v), opcode,
                       float(tenant_v), math.nan)
                n += 1
        elif version in _LCS_EXTRA_FIELDS:
            extra_words = _LCS_EXTRA_FIELDS[version]
            base_fmt = "<IQqII" + ("I" * extra_words) + "q"
            rec = struct.Struct(base_fmt)
            n = 0
            while n < max_records:
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                unpacked = rec.unpack(chunk)
                ts_v = unpacked[0]; obj_v = unpacked[1]; size_v = unpacked[2]
                packed = unpacked[3]
                op_raw = packed & 0xFF
                tenant_v = packed >> 8
                opcode = -1.0 if int(op_raw) == 13 else 1.0
                yield (float(ts_v), float(obj_v), float(size_v), opcode,
                       float(tenant_v), math.nan)
                n += 1
    finally:
        try:
            fh.close()
        except Exception:
            pass


def stream_exchange_etw(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    pat = re.compile(
        r"^\s*(DiskRead|DiskWrite),\s+(\d+),\s+[^,]+,\s+\d+,\s+\S+,\s+"
        r"(0x[\da-fA-F]+),\s+(0x[\da-fA-F]+),\s+(\d+),\s+(\d+),"
    )
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", errors="replace")
    else:
        fh = open(path, "rt", errors="replace")
    n = 0
    try:
        for line in fh:
            m = pat.match(line)
            if not m:
                continue
            event_type, ts_v, offset_hex, size_hex, elapsed, disk = m.groups()
            opcode = 1.0 if event_type == "DiskRead" else -1.0
            yield (float(ts_v), float(int(offset_hex, 16)),
                   float(int(size_hex, 16)), opcode, float(disk),
                   float(elapsed))
            n += 1
            if n >= max_records:
                break
    finally:
        fh.close()


def stream_baleen24(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    fh = open(path, "rt", errors="replace")
    n = 0
    try:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                io_offset = int(parts[1])
                io_size = int(parts[2])
                op_time = float(parts[3])
                op_name = int(parts[4])
                pipeline = int(parts[5])
            except ValueError:
                continue
            opcode = -1.0 if op_name == 2 else 1.0
            yield (op_time, float(io_offset), float(io_size), opcode,
                   float(pipeline), math.nan)
            n += 1
            if n >= max_records:
                break
    finally:
        fh.close()


def stream_tencent_cloud_disk(path: Path, max_records: int) -> Iterator[Tuple[float, ...]]:
    fh = open(path, "rt", errors="replace")
    n = 0
    try:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                ts_v, ri, rb, wi, wb, du = row[:6]
                yield (float(ts_v), float(ri), float(rb), float(wi),
                       float(wb), float(du))
                n += 1
            except ValueError:
                continue
            if n >= max_records:
                break
    finally:
        fh.close()


def stream_systor_text(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    fh = _open_zstd_text(path)
    n = 0
    try:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0] == "Timestamp":
                continue
            if len(row) < 6:
                continue
            try:
                ts_v = float(row[0])
                opcode = -1.0 if row[2].strip().lower().startswith("w") else 1.0
                tenant_v = float(row[3])
                obj_id = float(row[4])
                obj_size = float(row[5])
                response = float(row[1]) if len(row) > 1 else math.nan
                yield (ts_v, obj_id, obj_size, opcode, tenant_v, response)
                n += 1
            except ValueError:
                continue
            if n >= max_records:
                break
    finally:
        fh.close()


def stream_wiki_hash_text(path: Path, max_records: int) -> Iterator[Tuple[float, float, float, float, float, float]]:
    fh = _open_zstd_text(path)
    n = 0
    try:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts_v = float(row[0]); obj_id = float(row[1])
            except ValueError:
                continue
            yield (ts_v, obj_id, math.nan, math.nan, math.nan, math.nan)
            n += 1
            if n >= max_records:
                break
    finally:
        fh.close()


def stream_docker_parquet(path: Path, max_records: int) -> Iterator[Tuple[float, ...]]:
    """2017_docker HTTP request log parquet → request_sequence rows."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return
    OP_MAP = {"GET": 1, "HEAD": 1, "OPTIONS": 1,
              "POST": -1, "PUT": -1, "DELETE": -1, "PATCH": -1}
    pf = pq.ParquetFile(str(path))
    cols = ["timestamp", "http.request.method", "http.request.uri",
            "http.request.remoteaddr", "http.response.written",
            "http.request.duration"]
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
                return
            op = OP_MAP.get(method[i] or "", 1)
            obj_id = abs(hash(uri[i] or "")) & 0x7FFFFFFFFFFFFFFF
            tenant = abs(hash(remote[i] or "")) & 0xFFFFFFFF
            sz = float(size[i]) if size[i] is not None else math.nan
            rt = float(dur[i]) if dur[i] is not None else math.nan
            yield (float(ts[i]), float(obj_id), sz, float(op),
                   float(tenant), rt)
            n += 1


def stream_generic_text(path: Path, max_records: int) -> Iterator[Tuple[float, ...]]:
    """First-numeric-column-as-time generic CSV/TSV streamer; emits up to
    six floats per row (ts + first 5 numeric columns). For
    structured_table kinds. Pads/truncates to 6 columns."""
    fh = _open_zstd_text(path)
    n = 0
    delim = None
    try:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if delim is None:
                counts = {",": line.count(","),
                          "\t": line.count("\t"),
                          " ": line.count(" ")}
                delim, _ = max(counts.items(), key=lambda kv: kv[1])
                if counts[delim] == 0:
                    delim = ","
            if delim == " ":
                fields = line.split()
            else:
                fields = next(csv.reader([line], delimiter=delim))
            nums: List[float] = []
            for f in fields:
                try:
                    nums.append(float(f))
                except (ValueError, TypeError):
                    continue
                if len(nums) >= 6:
                    break
            if not nums:
                continue
            while len(nums) < 6:
                nums.append(math.nan)
            yield tuple(nums)
            n += 1
            if n >= max_records:
                break
    finally:
        fh.close()


# --------------- Dispatcher -------------------------------------------------

def dispatch(path: Path, kind: Optional[str], fmt: Optional[str],
             family: Optional[str], max_records: int):
    name = path.name
    s = str(path)
    if fmt is None:
        # Mirror infer_format from core.py without importing on this branch.
        if "Cloud_Disk_dataset/disk_load_data/" in s:
            fmt = "tencent_cloud_disk"
        elif "Exchange-Server-Traces" in s and name.endswith(".csv.gz"):
            fmt = "exchange_etw"
        elif "/Baleen24/extracted/" in s and name.endswith(".trace"):
            fmt = "baleen24"
        elif ".oracleGeneral" in name:
            fmt = "oracle_general"
        elif ".lcs" in name:
            fmt = "lcs"
        elif name.endswith(".csv.gz"):
            fmt = "csv_gz"
        elif name.endswith(".csv.zst") or name.endswith(".zst"):
            fmt = "text_zst"
        else:
            fmt = "text"

    columns: List[str]
    iterator: Iterator[Tuple[float, ...]]

    if fmt == "oracle_general":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_oracle_general(path, max_records)
    elif fmt == "lcs":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_lcs(path, max_records)
    elif fmt == "exchange_etw":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_exchange_etw(path, max_records)
    elif fmt == "baleen24":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_baleen24(path, max_records)
    elif fmt == "tencent_cloud_disk":
        columns = ["ts", "read_iops", "read_bw", "write_iops", "write_bw",
                   "disk_usage"]
        iterator = stream_tencent_cloud_disk(path, max_records)
    elif fmt == "parquet_docker" or (fmt == "parquet" and family == "2017_docker"):
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_docker_parquet(path, max_records)
    elif family == "2017_systor":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_systor_text(path, max_records)
    elif family == "2007_wiki":
        columns = ["ts", "obj_id", "obj_size", "opcode", "tenant", "response_time"]
        iterator = stream_wiki_hash_text(path, max_records)
    else:
        columns = ["v0", "v1", "v2", "v3", "v4", "v5"]
        iterator = stream_generic_text(path, max_records)

    return columns, iterator, fmt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="-", help="output CSV path or '-' for stdout")
    ap.add_argument("--max-records", type=int, default=5_000_000)
    ap.add_argument("--family", default=None)
    ap.add_argument("--format", default=None)
    ap.add_argument("--kind", default=None)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"path not found: {path}", file=sys.stderr)
        return 2

    columns, iterator, fmt = dispatch(path, args.kind, args.format,
                                      args.family, args.max_records)

    if args.out == "-":
        out = sys.stdout
        close = False
    else:
        out = open(args.out, "w", newline="")
        close = True
    try:
        w = csv.writer(out)
        w.writerow(columns)
        emitted = 0
        for row in iterator:
            w.writerow(row)
            emitted += 1
        print(f"# emitted={emitted} format={fmt}", file=sys.stderr)
    finally:
        if close:
            out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
