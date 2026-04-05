"""
Trace loading and preprocessing for LLGAN.

All fields are normalized to [-1, 1].  Opcode (read/write) is binarized to
+1 / -1.  String/categorical fields are label-encoded then scaled.

Timestamp columns are delta-encoded (inter-arrival times) rather than stored
as absolute values.  This makes the distribution stationary and far easier for
an LSTM to model.  Inverse-transform reconstructs absolute times via cumsum.

Supported formats
-----------------
spc          : ASU, LBA, Size(512B), Opcode(r/w), Timestamp(s)
msr          : Timestamp(100ns), Hostname, DiskNumber, Type, Offset, Size, ResponseTime
k5cloud      : StorageVolumeId, Timestamp, Opcode(R/W), Offset, Size(1024B)
systor       : Timestamp, ResponseTime, Opcode(R/W), LUN, Offset, Size(512B)
oracle_general: libCacheSim oracleGeneral binary format (24 bytes/record):
                uint32 ts, uint64 obj_id, uint32 obj_size,
                int32 next_access_vtime, int16 op, int16 tenant_id
                obj_id is split into obj_id_reuse (±1) + obj_id_stride (signed-log delta)
                — see _OBJ_LOCALITY_COLS.
lcs          : libCacheSim native binary format (.lcs.zst), versioned header.
                Header: 8192 bytes (magic + version + trace stats).
                v1 (24 bytes/record): uint32 ts, uint64 obj_id, uint32 obj_size,
                                      int64 next_access_vtime  [no op/tenant]
                v2 (28 bytes/record): same + packed uint32 (op:8 bits | tenant:24 bits)
                op: OP_READ=12, OP_WRITE=13 (libCacheSim enum.h; remapped to r/w internally)
                Covers: tencentBlock, alibabaBlock LCS corpuses (4,900+ files, ~1 TB)
exchange_etw : Windows ETW .csv.gz (MSR Exchange Server traces, SNIA IOTTA):
               DiskRead/DiskWrite completion events only;
               ts(µs), opcode, offset(bytes), size(bytes), response_time(µs), disk
baleen24     : Microsoft Baleen 2024 storage traces (ASPLOS '24).
               Space-separated text, 8 columns, no header:
               block_id io_offset io_size op_time op_name pipeline user_namespace user_name
               op_name: 1=read, 2=write, 3=flush, 4=discard, 5=other (3-5 mapped to 'r').
               op_time is Unix epoch float (seconds).
               Path may be a plain text file or a member of a .tar.gz tarball.
               Canonical mapping: ts=op_time, obj_id=io_offset, obj_size=io_size,
               opcode=op_name, tenant=pipeline.
               Files: /Volumes/Archive/Traces/Baleen24/storage_*.tar.gz on wigner.
tencent_cloud_disk: Tencent Cloud Disk aggregate statistics (ASPLOS '25, TELA paper).
               CSV, 6 columns, no header, 300-second sampling interval:
               timestamp, read_iops, read_bw_kbps, write_iops, write_bw_kbps, disk_usage_mb
               16,297 disks; UUID-named files inside a zip archive.
               NOTE: This is AGGREGATE per-disk stats (not request-level traces) — the
               data model is fundamentally different from oracle_general/lcs.  Use for
               workload characterization and utilization modeling, not for generating
               synthetic request sequences.  Model must be trained separately from
               oracle_general models.
               Path: plain CSV file extracted from the zip, OR the zip path (reads first
               matching UUID entry via 'unzip -p').
csv          : generic — infer numeric cols; opcode col named 'opcode'/'type'/'rw'
"""

import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Precharacterized file-level conditioning (z_global)
# ---------------------------------------------------------------------------
#
# These functions load per-file workload statistics from the trace
# characterization JSONL (traces/characterization/trace_characterizations.jsonl)
# and convert them to a normalized cond_dim-vector for the Generator.
#
# This replaces the window-level compute_window_descriptors() approach: instead
# of estimating workload character from a noisy 12-step window, we look up
# stable ground-truth statistics measured from the full trace file.
#
# Normalization clip ranges derived empirically from 13,732 request_sequence
# traces across all characterized families (see traces/analysis/trace_rollup.py).
#
# 10-dim vector layout (matches current cond_dim=10 default):
#   0: write_ratio             linear [0, 1]
#   1: log(reuse_ratio+0.001)  log-offset, clipped [-7, 0]
#   2: log(burstiness_cv)      log, clipped [0, 4.5]
#   3: log1p(iat_q50)          log1p, clipped [0, 12.5]
#   4: log(obj_size_q50)       log, clipped [6, 15]
#   5: opcode_switch_ratio     linear [0, 0.5]
#   6: iat_lag1_autocorr       identity [-1, 1]
#   7: log(tenant_unique)      log, clipped [0, 6.5]
#   8: forward_seek_ratio      linear [0, 1]
#   9: backward_seek_ratio     linear [0, 1]
# All output values are in [-1, 1].


def _cn(val: float, lo: float, hi: float) -> float:
    """Clip-normalize val from [lo, hi] to [-1, 1]."""
    span = hi - lo
    if span <= 0:
        return 0.0
    return max(-1.0, min(1.0, (val - lo) / span * 2.0 - 1.0))


def profile_to_cond_vector(profile: dict, cond_dim: int = 10) -> List[float]:
    """Convert a trace characterization profile dict to a normalized cond vector.

    Args:
        profile: the 'profile' sub-dict from a trace_characterizations.jsonl row.
        cond_dim: length of output vector (default 10; padded/truncated as needed).

    Returns:
        List of cond_dim floats in [-1, 1]. Missing fields default to 0.
    """
    iat = profile.get("iat_stats") or {}
    obj = profile.get("obj_size_stats") or {}
    ten = profile.get("tenant_summary") or {}

    write_ratio         = float(profile.get("write_ratio") or 0.0)
    reuse_ratio         = float(profile.get("reuse_ratio") or 0.0)
    burstiness_cv       = float(profile.get("burstiness_cv") or 1.0)
    iat_q50             = float(iat.get("q50") or 0.0)
    obj_size_q50        = float(obj.get("q50") or 4096.0)
    opcode_switch_ratio = float(profile.get("opcode_switch_ratio") or 0.0)
    iat_lag1_autocorr   = float(profile.get("iat_lag1_autocorr") or 0.0)
    tenant_unique       = float(ten.get("unique") or 1.0)
    forward_seek_ratio  = float(profile.get("forward_seek_ratio") or 0.5)
    backward_seek_ratio = float(profile.get("backward_seek_ratio") or 0.0)

    vec = [
        _cn(write_ratio,                                    0.0,   1.0),   # 0
        _cn(math.log(max(reuse_ratio, 0.0) + 0.001),       -7.0,  0.0),   # 1
        _cn(math.log(max(burstiness_cv, 1e-6)),             0.0,  4.5),   # 2
        _cn(math.log1p(max(iat_q50, 0.0)),                  0.0, 12.5),   # 3
        _cn(math.log(max(obj_size_q50, 1.0)),               6.0, 15.0),   # 4
        _cn(opcode_switch_ratio,                            0.0,  0.5),   # 5
        _cn(max(-1.0, min(1.0, iat_lag1_autocorr)),        -1.0,  1.0),   # 6
        _cn(math.log(max(tenant_unique, 1.0)),              0.0,  6.5),   # 7
        _cn(forward_seek_ratio,                             0.0,  1.0),   # 8
        _cn(backward_seek_ratio,                            0.0,  1.0),   # 9
    ]

    if len(vec) < cond_dim:
        vec.extend([0.0] * (cond_dim - len(vec)))
    return vec[:cond_dim]


def load_file_characterizations(
    jsonl_path: str,
    cond_dim: int = 10,
) -> Dict[str, torch.Tensor]:
    """Load trace characterizations and return a filename→cond_vector lookup.

    Keyed by basename (e.g. 'tencentBlock_10007.oracleGeneral.zst') and also
    by the de-suffixed name (e.g. 'tencentBlock_10007.oracleGeneral') for
    flexible matching against compressed/uncompressed variants.

    Args:
        jsonl_path: path to trace_characterizations.jsonl
        cond_dim:   dimension of each conditioning vector (default 10)

    Returns:
        dict mapping filename strings to (cond_dim,) float32 tensors.
    """
    import json
    from pathlib import Path

    lookup: Dict[str, torch.Tensor] = {}
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("profile"):
                continue
            rel = row.get("rel_path", "")
            if not rel:
                continue
            basename = Path(rel).name
            vec = profile_to_cond_vector(row["profile"], cond_dim)
            t = torch.tensor(vec, dtype=torch.float32)
            lookup[basename] = t
            # Also index without .zst or .gz so we match either variant
            for ext in (".zst", ".gz"):
                if basename.endswith(ext):
                    lookup[basename[: -len(ext)]] = t
    return lookup


# ---------------------------------------------------------------------------
# Per-window workload descriptors (z_global conditioning, fallback)
# ---------------------------------------------------------------------------

def compute_window_descriptors(windows: torch.Tensor) -> torch.Tensor:
    """Compute per-window workload descriptors from preprocessed windows.

    Args:
        windows: (B, T, 6) tensor with columns [ts, obj_size, opcode,
                 tenant, obj_id_reuse, obj_id_stride], already normalized
                 to [-1, 1].

    Returns:
        (B, 10) tensor of descriptors (same device as input):
          ts_mean, ts_std, obj_size_mean, obj_size_std, opcode_mean,
          tenant_mean, obj_id_reuse_mean, obj_id_stride_mean, obj_id_stride_std
        ... wait, that's 9.  We add a 10th: ts inter-arrival std-of-diff
        (second-order burstiness).

    All computed from normalized values; no inversion needed.
    """
    # Ensure float32 for stability under AMP
    w = windows.float()
    ts       = w[:, :, 0]  # (B, T)
    obj_size = w[:, :, 1]
    opcode   = w[:, :, 2]
    tenant   = w[:, :, 3]
    reuse    = w[:, :, 4]
    stride   = w[:, :, 5]

    descs = torch.stack([
        ts.mean(dim=1),                          # 0: ts mean
        ts.std(dim=1).clamp(min=1e-6),           # 1: ts std
        obj_size.mean(dim=1),                    # 2: obj_size mean
        obj_size.std(dim=1).clamp(min=1e-6),     # 3: obj_size std
        opcode.mean(dim=1),                      # 4: opcode mean (read ratio proxy)
        tenant.mean(dim=1),                      # 5: tenant mean
        reuse.mean(dim=1),                       # 6: obj_id_reuse mean
        stride.mean(dim=1),                      # 7: obj_id_stride mean
        stride.std(dim=1).clamp(min=1e-6),       # 8: obj_id_stride std
        ts.diff(dim=1).std(dim=1).clamp(min=1e-6),  # 9: ts diff-of-diff std (burstiness)
    ], dim=1)  # (B, 10)
    return descs


# ---------------------------------------------------------------------------
# Format readers  →  returns a raw DataFrame with canonical column names
# ---------------------------------------------------------------------------

def _read_spc(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, sep=r"[,\s]+", engine="python",
                     names=["asu", "lba", "size", "opcode", "ts"],
                     nrows=max_records)
    return df


def _read_msr(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ts", "hostname", "disk", "opcode",
                             "offset", "size", "response_time"],
                     nrows=max_records)
    df["ts"] = df["ts"] / 1e7          # 100ns → seconds
    df = df.drop(columns=["hostname"])
    return df


def _read_k5cloud(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["volume_id", "ts", "opcode", "offset", "size"],
                     nrows=max_records)
    return df


def _read_systor(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, header=None,
                     names=["ts", "response_time", "opcode", "lun",
                             "offset", "size"],
                     nrows=max_records)
    return df


def _read_oracle_general(path: str, max_records: int) -> pd.DataFrame:
    """
    libCacheSim oracleGeneral binary format.
    24 bytes per record: uint32 ts | uint64 obj_id | uint32 obj_size |
                         int32 next_access_vtime | int16 op | int16 tenant_id
    op: 0=read, 1=write; next_access_vtime is a forward pointer (record index
    of next access to same object) — derived annotation, not a workload field.
    """
    import subprocess

    n_bytes = max_records * 24  # each record is exactly 24 bytes

    if path.endswith(".zst"):
        # Stream-decompress only the bytes we need — much faster than reading
        # the full file when max_records << total records.
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    elif path.endswith(".gz"):
        import gzip
        with gzip.open(path, "rb") as f:
            raw = f.read(n_bytes)
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
    n = min(len(raw) // 24, max_records)
    arr = np.frombuffer(raw[:n * 24], dtype=dt)
    df = pd.DataFrame(arr)
    df.drop(columns=["vtime"], inplace=True)
    df.rename(columns={"op": "opcode"}, inplace=True)
    return df


def _read_lcs(path: str, max_records: int) -> pd.DataFrame:
    """
    libCacheSim native LCS binary format (.lcs or .lcs.zst).

    File layout (from libCacheSim/traceReader/customizedReader/lcs.h):
      Header : 8192 bytes
        [0:8]   start_magic (0x123456789abcdef0 LE)
        [8:16]  version     (uint64: 1, 2, 3, …)
        [16:]   trace stats (n_req, n_obj, size histograms, …) — not used here
      Records : after the 8192-byte header, one struct per I/O request

    Record structs:
      v1 (24 bytes): uint32 ts | uint64 obj_id | uint32 obj_size | int64 vtime
      v2 (28 bytes): same + packed uint32 (bits 0-7 = op, bits 8-31 = tenant)
      v3+ : larger records (int64 obj_size, ttl, feature fields) — not yet supported

    op encoding (libCacheSim enum.h): OP_READ=12, OP_WRITE=13.
    Ops 1-11 are cache-protocol ops (GET/SET/CAS/etc.) treated as reads.
    We remap 13→'w', else→'r' before returning so the standard opcode encoder applies.

    The next_access_vtime field is a forward pointer (index of next request to the
    same object, -1 if last) — a derived annotation used by cache simulators,
    not a workload field.  It is dropped before returning.
    """
    import subprocess

    LCS_MAGIC   = 0x123456789abcdef0
    LCS_HEADER  = 8192

    # --- decompress or open ---
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        header_raw = proc.stdout.read(LCS_HEADER)
    elif path.endswith(".gz"):
        import gzip
        f = gzip.open(path, "rb")
        header_raw = f.read(LCS_HEADER)
    else:
        f = open(path, "rb")
        header_raw = f.read(LCS_HEADER)

    # Validate magic and read version
    import struct
    if len(header_raw) < 16:
        raise ValueError(f"LCS file too short for header: {path}")
    magic, version = struct.unpack_from("<QQ", header_raw, 0)
    if magic != LCS_MAGIC:
        raise ValueError(
            f"LCS magic mismatch in {path}: got 0x{magic:016x}, "
            f"expected 0x{LCS_MAGIC:016x}"
        )

    if version == 1:
        rec_size = 24
        dt = np.dtype([
            ("ts",       "<u4"),
            ("obj_id",   "<u8"),
            ("obj_size", "<u4"),
            ("vtime",    "<i8"),   # int64 next_access_vtime (dropped below)
        ])
    elif version == 2:
        rec_size = 28
        dt = np.dtype([
            ("ts",       "<u4"),
            ("obj_id",   "<u8"),
            ("obj_size", "<u4"),
            ("packed",   "<u4"),   # bits 0-7 = op, bits 8-31 = tenant
            ("vtime",    "<i8"),
        ])
    else:
        raise ValueError(
            f"LCS version {version} not yet supported (only v1 and v2). "
            f"File: {path}"
        )

    # Read only as many record bytes as needed
    n_bytes = max_records * rec_size
    if path.endswith(".zst"):
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    else:
        raw = f.read(n_bytes)
        f.close()

    n = min(len(raw) // rec_size, max_records)
    arr = np.frombuffer(raw[:n * rec_size], dtype=dt)
    df = pd.DataFrame(arr)
    df.drop(columns=["vtime"], inplace=True)

    if version == 1:
        # No op/tenant in v1; default to read (most block traces are read-dominant)
        df["opcode"] = "r"
    else:
        # Unpack op (low 8 bits) and tenant (high 24 bits) from packed field.
        # libCacheSim op_e enum (enum.h): OP_READ=12, OP_WRITE=13.
        # Keys 1-11 are cache ops (GET/SET/etc.) which we treat as reads since
        # they are cache lookups, not block writes.
        op_raw = arr["packed"] & 0xFF
        df["opcode"] = np.where(op_raw == 13, "w", "r")
        df["tenant"] = (arr["packed"] >> 8).astype(np.int32)
        df.drop(columns=["packed"], inplace=True)

    return df


def _read_baleen24(path: str, max_records: int) -> pd.DataFrame:
    """
    Microsoft Baleen 2024 storage traces (ASPLOS '24).

    Format: space-separated text, no header, 8 columns per line:
      block_id  io_offset  io_size  op_time  op_name  pipeline  user_namespace  user_name
    where:
      block_id      – integer disk/volume identifier (not stored as a feature)
      io_offset     – byte offset within the block device → used as obj_id
      io_size       – bytes transferred → obj_size
      op_time       – Unix epoch timestamp (float seconds) → ts
      op_name       – 1=read, 2=write, 3=flush, 4=discard, 5=other (3-5 → 'r')
      pipeline      – Baleen pipeline tag → tenant
      user_namespace – not used
      user_name     – not used

    `path` may be:
      (a) a plain text file (extracted member),
      (b) a .tar.gz archive — reads the first non-directory member, or
      (c) a .tar.gz path with a member suffix "archive.tar.gz::member/name"
          (separate the member name with '::').
    """
    import io
    import tarfile

    lines: list[bytes] = []

    if ".tar.gz" in path or ".tgz" in path:
        # Support "archive.tar.gz::member" syntax for targeting a specific member.
        if "::" in path:
            archive_path, member_name = path.split("::", 1)
        else:
            archive_path, member_name = path, None

        with tarfile.open(archive_path, "r:gz") as tar:
            if member_name:
                f = tar.extractfile(member_name)
                if f is None:
                    raise ValueError(f"Member '{member_name}' is not a regular file in {archive_path}")
            else:
                # Prefer .trace files (Baleen24 tarball layout); fall back to any
                # regular file large enough to contain at least a few records.
                f = None
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".trace"):
                        f = tar.extractfile(member)
                        break
                if f is None:
                    for member in tar.getmembers():
                        if member.isfile() and member.size > 1024:
                            f = tar.extractfile(member)
                            break
                if f is None:
                    raise ValueError(f"No data files found in tarball: {archive_path}")
            raw = f.read()
        lines = raw.splitlines()
    else:
        with open(path, "rb") as f:
            lines = f.read().splitlines()

    rows = []
    for line in lines:
        if len(rows) >= max_records:
            break
        line_s = line.decode("utf-8", errors="replace").strip()
        if not line_s or line_s.startswith("#"):
            continue
        parts = line_s.split()
        if len(parts) < 6:
            continue
        try:
            # block_id = int(parts[0])  — disk identifier; not used as a feature
            io_offset   = int(parts[1])
            io_size     = int(parts[2])
            op_time     = float(parts[3])
            op_name_int = int(parts[4])
            pipeline    = int(parts[5])
        except (ValueError, IndexError):
            continue
        opcode = 'w' if op_name_int == 2 else 'r'
        rows.append({
            'ts':       op_time,
            'obj_id':   io_offset,
            'obj_size': io_size,
            'opcode':   opcode,
            'tenant':   pipeline,
        })

    if not rows:
        raise ValueError(f"No valid Baleen24 records found in {path}")

    return pd.DataFrame(rows)


def _read_tencent_cloud_disk(path: str, max_records: int) -> pd.DataFrame:
    """
    Tencent Cloud Disk aggregate performance statistics (ASPLOS '25, TELA paper).

    Format: CSV, no header, 6 columns, 300-second sampling interval:
      timestamp      – Unix epoch seconds (integer)
      read_iops      – read operations per second (aggregate over 300s window)
      read_bw_kbps   – read bandwidth in KB/s
      write_iops     – write operations per second
      write_bw_kbps  – write bandwidth in KB/s
      disk_usage_mb  – disk utilization in MB (snapshot at end of window)

    Each file (UUID-named) is one cloud disk's full time series.

    IMPORTANT: This is NOT a request-level trace.  Each row is a 5-minute aggregate
    for one disk.  The model trained on this data produces synthetic utilization time
    series, not synthetic I/O request sequences.  Do not mix with oracle_general models.

    `path` may be:
      (a) a plain CSV file (extracted from the zip),
      (b) "zipfile.zip::member/name" — extracts that member via 'unzip -p'.
    """
    import subprocess
    import io

    if "::" in path:
        zip_path, member_name = path.split("::", 1)
        proc = subprocess.run(
            ["unzip", "-p", zip_path, member_name],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout
    else:
        with open(path, "rb") as f:
            raw = f.read()

    col_names = ["ts", "read_iops", "read_bw_kbps", "write_iops", "write_bw_kbps", "disk_usage_mb"]
    df = pd.read_csv(
        io.BytesIO(raw) if isinstance(raw, bytes) else io.StringIO(raw.decode()),
        header=None,
        names=col_names,
        nrows=max_records,
    )
    return df


def _read_csv(path: str, max_records: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=max_records)
    return df


def _read_exchange_etw(path: str, max_records: int) -> pd.DataFrame:
    """
    Microsoft Exchange Server ETW disk trace (MSR SNIA dataset).
    Format: Windows ETW .csv.gz with mixed event types; we extract only
    DiskRead/DiskWrite completion events (not DiskReadInit/DiskWriteInit).

    Fields extracted:
      ts            – TimeStamp in microseconds from trace start (→ delta-encoded IAT)
      opcode        – 'r' (DiskRead) or 'w' (DiskWrite)
      offset        – ByteOffset in bytes (hex → int)
      size          – IOSize in bytes (hex → int, log-transformed)
      response_time – ElapsedTime in microseconds (log-transformed)
      disk          – DiskNum (integer LUN id)
    """
    import gzip
    import re

    # Matches DiskRead/DiskWrite completion lines (but NOT DiskReadInit/DiskWriteInit).
    # Columns: EventType, TimeStamp, ProcessName(PID), ThreadID, IrpPtr,
    #          ByteOffset, IOSize, ElapsedTime, DiskNum, IrpFlags, ...
    pattern = re.compile(
        r'^\s*(DiskRead|DiskWrite),\s+(\d+),\s+[^,]+,\s+\d+,\s+\S+,\s+'
        r'(0x[\da-fA-F]+),\s+(0x[\da-fA-F]+),\s+(\d+),\s+(\d+),'
    )

    rows = []
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', errors='replace') as f:
        for line in f:
            if len(rows) >= max_records:
                break
            m = pattern.match(line)
            if m:
                event_type, ts, offset_hex, size_hex, elapsed, disk = m.groups()
                rows.append({
                    'ts':            int(ts),
                    'opcode':        'r' if event_type == 'DiskRead' else 'w',
                    'offset':        int(offset_hex, 16),
                    'size':          int(size_hex, 16),
                    'response_time': int(elapsed),
                    'disk':          int(disk),
                })

    if not rows:
        raise ValueError(f"No DiskRead/DiskWrite events found in {path}")

    return pd.DataFrame(rows)


_READERS = {
    "spc": _read_spc,
    "msr": _read_msr,
    "k5cloud": _read_k5cloud,
    "systor": _read_systor,
    "oracle_general": _read_oracle_general,
    "lcs": _read_lcs,
    "exchange_etw": _read_exchange_etw,
    "baleen24": _read_baleen24,
    "tencent_cloud_disk": _read_tencent_cloud_disk,
    "csv": _read_csv,
}

# Timestamp column names — will be delta-encoded (deltas clipped to ≥ 0)
_TS_COLS = {"ts", "timestamp"}

# Object-ID column names — locality-aware two-feature split (v15+).
# Each obj_id column is replaced with two derived columns:
#   {col}_reuse : ±1 binary — +1 if same object as previous (delta==0), -1 otherwise.
#                 The generator can learn this as a discrete binary feature; outputting
#                 exactly 0 in continuous delta-space to produce reuse is impossible
#                 in practice (v14g root cause: reuse_rate=0).
#   {col}_stride: signed-log delta for non-reuse accesses: sign(Δ)·log1p(|Δ|).
#                 Set to 0 when reuse=+1 (stride is undefined for same-object access).
#                 Preserves sequential (small Δ), strided (regular Δ), random (large Δ)
#                 and backward-seek (negative Δ) patterns.
# Raw obj_id values are meaningless as continuous numbers — the delta still captures
# the access structure. The split just makes reuse a learnable binary classification
# target rather than a zero-measure point in continuous space.
_OBJ_LOCALITY_COLS = {"obj_id"}

# Legacy: kept empty so no columns fall through the old signed-log delta path.
_OBJ_DELTA_COLS: set = set()

# Columns whose raw values are read/write opcodes
_OPCODE_COLS = {"opcode", "type", "rw", "op"}

# Columns that are log-transformed before min-max normalization.
# These have heavy-tailed / power-law distributions that confound tanh output.
# ts_delta (inter-arrival times) and obj_size are the canonical cases.
# Note: obj_id is NOT listed here — it gets a signed-log transform instead.
_LOG_COLS = {"ts", "timestamp", "obj_size", "size", "response_time"}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TracePreprocessor:
    """
    Fit on training data; transform train + val to [-1, 1].

    The three non-trivial preprocessing choices are explained here because
    they directly affect what the model can learn:

    (1) Timestamps → inter-arrival times (IAT / delta encoding)
        Absolute timestamps are non-stationary: trace A runs from 0–3600s,
        trace B from 0–7200s, so the same value means different things.
        Delta-encoding the timestamps produces inter-arrival times (IATs),
        which *are* stationary: the distribution of gaps between I/Os is
        roughly the same whether we're at second 100 or second 3000 of the
        trace. This is the right quantity for an LSTM to model because it
        can then be trained across files without the model memorising
        absolute clock values.  Clipping negative deltas to 0 handles the
        rare out-of-order timestamps in some trace formats.

    (2) Object/LBA IDs → locality-aware two-feature split (v15+)
        Raw object IDs are meaningless as continuous numbers.  The delta
        captures access structure, but the old approach of passing delta=0
        through signed-log gave reuse a single point (0.0) in continuous
        space — a zero-measure set the generator could never reliably hit
        (v14g root cause: reuse_rate=0.000 throughout 400 epochs).
        v15 splits obj_id into two features:
          obj_id_reuse  ±1 binary: +1 same object, -1 different object.
          obj_id_stride signed-log(Δ) for seeks; 0 for reuse.
        Now reuse is a learnable binary classification target instead of an
        exact zero in continuous regression space.

    (3) Sizes and IATs → log1p before min-max normalisation
        I/O sizes and inter-arrival times follow approximate power-law
        (Pareto) distributions: most accesses are small and frequent,
        but there are rare large / slow outliers.  Without log-transform,
        the bulk of the distribution (say, 4 KB accesses arriving every
        1 ms) is squashed into a tiny sliver of the normalised range and
        the model spends capacity learning the rare outliers.  Log1p
        approximately Gaussianises these distributions.
    """

    def __init__(self, obj_size_granularity: int = 0):
        """
        obj_size_granularity: snap obj_size to this multiple before encoding
            (0 = off, 4096 = page-aligned, 512 = sector-aligned).
            Real block traces have sizes drawn from a small discrete set
            (powers of 2 × sector size); quantizing before log-transform
            concentrates the distribution onto the real support and helps
            the generator learn the correct size distribution.
        """
        self.col_names: List[str] = []
        self.num_cols: int = 0
        self._stats: Dict[str, dict] = {}
        self._cat_maps: Dict[str, dict] = {}
        self._delta_cols: List[str] = []           # timestamp cols: delta, clip ≥ 0
        self._first_vals: Dict[str, float] = {}    # for inverse cumsum (ts)
        self._log_cols: List[str] = []             # log1p-transformed cols (non-negative)
        self._obj_delta_cols: List[str] = []       # legacy (now empty)
        self._obj_locality_cols: List[str] = []    # obj_id cols → reuse + stride split
        self._obj_locality_first: Dict[str, float] = {}  # first abs value for cumsum
        self._obj_size_granularity: int = obj_size_granularity
        self._dropped_cols: List[str] = []         # zero-variance cols auto-dropped

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "TracePreprocessor":
        df = df.copy()
        self._delta_cols = [c for c in df.columns if c.lower() in _TS_COLS]
        self._obj_delta_cols = []  # unused in v15+
        self._obj_locality_cols = [c for c in df.columns if c.lower() in _OBJ_LOCALITY_COLS]
        df = self._quantize_obj_size(df)
        df = self._apply_deltas(df, fit=True)
        df = self._apply_obj_locality(df, fit=True)
        df = self._encode_categoricals(df, fit=True)
        # Identify log-transform columns: heavy-tailed continuous columns.
        # Applied after delta-encoding so ts_delta (inter-arrival times) is
        # also log-transformed, which is the right thing — IAT is power-law.
        # obj_id_reuse and obj_id_stride are excluded (handled by _apply_obj_locality).
        self._log_cols = [
            c for c in df.columns
            if c.lower() in _LOG_COLS and c not in self._cat_maps
        ]
        df = self._apply_log(df)
        # Auto-drop zero-variance columns: columns where min == max carry no
        # information and waste model capacity.  Common cases: opcode in
        # pure-read corpora (alibaba, tencent_block), obj_id_reuse when reuse
        # ratio ≈ 0.  The dropped columns are recorded so inverse_transform
        # can re-insert them with their constant value.
        self._dropped_cols = []
        self._dropped_const: Dict[str, float] = {}  # col → constant value
        all_cols = list(df.columns)
        keep_cols = []
        for col in all_cols:
            lo = float(df[col].min())
            hi = float(df[col].max())
            self._stats[col] = {"lo": lo, "hi": hi}
            if hi == lo:
                self._dropped_cols.append(col)
                self._dropped_const[col] = lo
            else:
                keep_cols.append(col)
        if self._dropped_cols:
            df = df[keep_cols]
        self.col_names = list(df.columns)
        self.num_cols = len(self.col_names)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df = self._quantize_obj_size(df)
        df = self._apply_deltas(df, fit=False)
        df = self._apply_obj_locality(df, fit=False)
        df = self._encode_categoricals(df, fit=False)
        df = self._apply_log(df)
        out = np.zeros((len(df), self.num_cols), dtype=np.float32)
        for i, col in enumerate(self.col_names):
            lo = self._stats[col]["lo"]
            hi = self._stats[col]["hi"]
            span = hi - lo  # guaranteed > 0 for kept columns
            out[:, i] = np.clip(
                (df[col].values.astype(np.float32) - lo) / span * 2 - 1,
                -1.0, 1.0,
            )
        return out

    def inverse_transform(self, arr: np.ndarray) -> pd.DataFrame:
        data = {}
        for i, col in enumerate(self.col_names):
            lo = self._stats[col]["lo"]
            hi = self._stats[col]["hi"]
            span = hi - lo  # > 0 for kept columns
            vals = (arr[:, i].astype(np.float64) + 1) / 2 * span + lo
            # Undo log1p transform before any other post-processing
            if col in self._log_cols:
                vals = np.expm1(np.clip(vals, 0, None))
            if col in self._cat_maps:
                inv_map = {v: k for k, v in self._cat_maps[col].items()}
                vals = np.array([inv_map.get(int(round(v)), v) for v in vals])
            data[col] = vals

        df = pd.DataFrame(data)

        # Re-insert dropped zero-variance columns with their constant values.
        # This ensures generated output has the same schema as the original data.
        for col, const in getattr(self, "_dropped_const", {}).items():
            # Undo log1p if this was a log-transformed column
            val = const
            if col in self._log_cols:
                val = np.expm1(max(val, 0.0))
            if col in self._cat_maps:
                inv_map = {v: k for k, v in self._cat_maps[col].items()}
                val = inv_map.get(int(round(val)), val)
            df[col] = val

        # Undo timestamp delta encoding: cumsum of IATs + start restores absolute times.
        # The first delta is 0 by construction (_apply_deltas uses prepend=vals[0]),
        # so cumsum gives [0, d1, d1+d2, ...] and adding start gives the correct series.
        for col in self._delta_cols:
            if col in df.columns:
                start = self._first_vals.get(col, 0.0)
                df[col] = start + np.cumsum(df[col].values)

        # Undo obj_id locality split: reconstruct absolute obj_id from reuse + stride.
        # obj_id_reuse > 0 means same object (delta=0); otherwise use signed-log inverse.
        for col in self._obj_locality_cols:
            reuse_col  = f"{col}_reuse"
            stride_col = f"{col}_stride"
            if reuse_col in df.columns and stride_col in df.columns:
                s = df[stride_col].values
                stride_raw = np.sign(s) * np.expm1(np.abs(s))  # undo signed-log
                r = df[reuse_col].values
                delta = np.where(r > 0, 0.0, stride_raw)
                start = self._obj_locality_first.get(col, 0.0)
                df[col] = start + np.cumsum(delta)
                df.drop(columns=[reuse_col, stride_col], inplace=True)

        return df

    # ------------------------------------------------------------------
    def _quantize_obj_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Snap obj_size to the nearest multiple of obj_size_granularity.

        Block I/O sizes are drawn from a small discrete set (multiples of the
        storage sector or page size). Training on the raw continuous values
        causes the generator to produce non-quantized sizes; training on the
        quantized values concentrates the distribution onto the real support.
        Applied before log-transform so the log distribution has the same
        discrete peaks as the real data.
        """
        if self._obj_size_granularity > 0 and "obj_size" in df.columns:
            df = df.copy()
            g = self._obj_size_granularity
            raw = df["obj_size"].values.astype(np.float64)
            # Floor to multiple of g, with a floor of g itself (no zero-size records).
            df["obj_size"] = np.maximum(raw // g, 1.0) * g
        return df

    def _apply_obj_locality(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Replace each obj_id column with a (reuse, stride) pair.

        obj_id_reuse  : ±1 binary — +1 if delta==0 (same object), -1 otherwise.
        obj_id_stride : signed-log delta for seeks; 0 when reuse=+1.

        This replaces the old signed-log delta single-feature approach.  The
        old approach required the generator to output exactly 0 in continuous
        space to produce a reuse — a zero-measure event it never learned.
        With a binary reuse feature the generator can classify reuse vs seek,
        which is a learnable sigmoid output in [-1, 1].
        """
        df = df.copy()
        for col in self._obj_locality_cols:
            if col not in df.columns:
                continue
            vals = df[col].values.astype(np.float64)
            if fit:
                self._obj_locality_first[col] = float(vals[0])
            deltas = np.diff(vals, prepend=vals[0])   # first delta = 0
            reuse  = np.where(deltas == 0, 1.0, -1.0)
            stride = np.sign(deltas) * np.log1p(np.abs(deltas))
            stride[deltas == 0] = 0.0  # reuse: stride undefined → 0
            df.drop(columns=[col], inplace=True)
            df[f"{col}_reuse"]  = reuse
            df[f"{col}_stride"] = stride
        return df

    def _apply_deltas(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in self._delta_cols:
            if col not in df.columns:
                continue
            vals = df[col].values.astype(np.float64)
            if fit:
                self._first_vals[col] = float(vals[0])
            deltas = np.diff(vals, prepend=vals[0])   # first delta = 0
            # Clip negative deltas (out-of-order timestamps) to 0
            deltas = np.clip(deltas, 0, None)
            df = df.copy()
            df[col] = deltas
        return df

    def _apply_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """log1p-transform heavy-tailed columns to approximately Gaussianize them."""
        df = df.copy()
        for col in self._log_cols:
            if col in df.columns:
                df[col] = np.log1p(np.clip(df[col].values.astype(np.float64), 0, None))
        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if col.lower() in _OPCODE_COLS:
                df[col] = self._encode_opcode(df[col])
            elif df[col].dtype == object:
                df[col] = self._encode_labels(col, df[col], fit)
        return df.apply(pd.to_numeric, errors="coerce").fillna(0)

    @staticmethod
    def _encode_opcode(series: pd.Series) -> pd.Series:
        """Encode opcode to +1.0 (read) / -1.0 (write).

        Handles format conventions:
          0 / "r" / "read"  → +1.0 (read)
          1 / "w" / "write" → -1.0 (write)
          -1 / negative     → +1.0 (read/unknown sentinel; common in
                              libCacheSim oracleGeneral & LCS when opcode
                              is not recorded — NOT a write indicator)
        """
        s = series.astype(str).str.lower().str.strip()
        def _map(x):
            if x in {"r", "read", "0"}:
                return 1.0
            elif x in {"w", "write", "1"}:
                return -1.0
            else:
                try:
                    v = int(float(x))
                    # Negative values (e.g. -1) are missing/sentinel, not writes.
                    # Only positive 1 means write; 0 and negatives → read.
                    return -1.0 if v == 1 else 1.0
                except ValueError:
                    return 0.0
        return s.map(_map)

    def _encode_labels(self, col: str, series: pd.Series, fit: bool) -> pd.Series:
        if fit:
            unique = sorted(series.dropna().unique())
            self._cat_maps[col] = {v: i for i, v in enumerate(unique)}
        m = self._cat_maps.get(col, {})
        return series.map(lambda x: m.get(x, 0))


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TraceDataset(Dataset):
    """
    Sliding-window dataset over a normalized trace array.

    When file_cond is provided (a precharacterized per-file conditioning
    vector), __getitem__ returns (window, file_cond) tuples so the DataLoader
    can batch both together.  When file_cond is None, returns plain windows
    for backward compatibility.
    """

    def __init__(
        self,
        data: np.ndarray,
        timestep: int,
        file_cond: Optional[torch.Tensor] = None,
    ):
        self.data = torch.from_numpy(data)
        self.timestep = timestep
        self.n_windows = max(0, len(data) - timestep)
        # file_cond: (cond_dim,) float32 tensor, same for every window in this
        # file.  None = fall back to window-level compute_window_descriptors().
        self.file_cond = file_cond

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        window = self.data[idx: idx + self.timestep]
        if self.file_cond is not None:
            return window, self.file_cond
        return window


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------

def load_trace(
    path: str,
    fmt: str,
    max_records: int,
    timestep: int,
    train_split: float = 0.8,
    obj_size_granularity: int = 0,
    file_cond: Optional[torch.Tensor] = None,
) -> Tuple[TraceDataset, TraceDataset, TracePreprocessor]:
    """Load, preprocess, and split a trace file."""
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown trace format '{fmt}'. "
                         f"Choose from: {list(_READERS)}")

    # oracle_general and lcs handle their own decompression (binary format with
    # fixed-width records; streaming avoids loading the full file into memory).
    # baleen24 and tencent_cloud_disk handle their own archive formats (.tar.gz / .zip).
    # All other formats go through pandas after a full zstd decompress.
    path_s = str(path)
    self_decompressing = fmt in ("oracle_general", "lcs", "baleen24", "tencent_cloud_disk")
    if not self_decompressing and path_s.endswith(".zst"):
        import subprocess, tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        subprocess.run(["zstd", "-d", path_s, "-o", tmp.name, "-f"], check=True)
        df = reader(tmp.name, max_records)
        os.unlink(tmp.name)
    else:
        df = reader(path_s, max_records)

    n_train = int(len(df) * train_split)
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val   = df.iloc[n_train:].reset_index(drop=True)

    prep = TracePreprocessor(obj_size_granularity=obj_size_granularity)
    prep.fit(df_train)

    train_arr = prep.transform(df_train)
    val_arr   = prep.transform(df_val)

    train_ds = TraceDataset(train_arr, timestep, file_cond=file_cond)
    val_ds   = TraceDataset(val_arr,   timestep)  # val never needs cond

    return train_ds, val_ds, prep
