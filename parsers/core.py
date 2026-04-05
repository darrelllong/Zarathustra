from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import re
import struct
import subprocess
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


TRACE_ROOT = Path("/tiamat/zarathustra/traces")
SKIP_NAMES = {
    "README",
    "README.md",
    "Disclaimer.txt",
    "checksums.sha1",
    "download.log",
    "wget.log",
    ".gitattributes",
    ".gitignore",
    "s3_filelist.tsv",
}
SKIP_SUFFIXES = {".log", ".sha1", ".md", ".json"}
TEMP_MARKERS = (".tm", ".tmp", ".part")

_LCS_EXTRA_FIELDS = {
    3: 0,
    4: 1,
    5: 2,
    6: 4,
    7: 8,
    8: 16,
}


FAMILY_REGISTRY: Dict[Tuple[str, str], Dict[str, object]] = {
    ("Baleen24", "Baleen24"): {"kind": "request_sequence", "formats": ("baleen24",)},
    ("MSR", "exchange"): {"kind": "request_sequence", "formats": ("exchange_etw",)},
    ("alibaba", "alibaba"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2007_msr"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2007_wiki"): {"kind": "structured_table", "formats": ("text_zst",)},
    ("s3-cache-datasets", "2015_cloudphysics"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2016_wiki"): {"kind": "structured_table", "formats": ("text", "text_zst")},
    ("s3-cache-datasets", "2017_docker"): {"kind": "structured_table", "formats": ("parquet",)},
    ("s3-cache-datasets", "2017_systor"): {"kind": "request_sequence", "formats": ("text_zst",)},
    ("s3-cache-datasets", "2018_tencentPhoto"): {"kind": "request_sequence", "formats": ("oracle_general", "text_zst")},
    ("s3-cache-datasets", "2019_wiki"): {"kind": "request_sequence", "formats": ("oracle_general", "text", "text_zst")},
    ("s3-cache-datasets", "2020_alibabaBlock"): {"kind": "request_sequence", "formats": ("oracle_general", "text_zst")},
    ("s3-cache-datasets", "2020_tencentBlock"): {"kind": "request_sequence", "formats": ("oracle_general", "text_zst")},
    ("s3-cache-datasets", "2020_twitter"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2020_twr_cdn"): {"kind": "structured_table", "formats": ("parquet", "text_zst")},
    ("s3-cache-datasets", "2022_metaCDN"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2022_metaKV"): {"kind": "request_sequence", "formats": ("oracle_general", "text_zst")},
    ("s3-cache-datasets", "2022_metaStorage"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "2023_metaCDN"): {"kind": "structured_table", "formats": ("text_zst",)},
    ("s3-cache-datasets", "2023_metaStorage"): {"kind": "structured_table", "formats": ("text", "text_zst")},
    ("s3-cache-datasets", "2024_google"): {"kind": "structured_table", "formats": ("parquet", "text_zst")},
    ("s3-cache-datasets", "2026-alibaba-thinkahead"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "alibaba"): {"kind": "request_sequence", "formats": ("lcs",)},
    ("s3-cache-datasets", "cache_trace_twitter_memcache"): {"kind": "request_sequence", "formats": ("oracle_general", "parquet", "text_zst")},
    ("s3-cache-datasets", "cloudphysics"): {"kind": "request_sequence", "formats": ("lcs", "text")},
    ("s3-cache-datasets", "metaKV"): {"kind": "request_sequence", "formats": ("lcs",)},
    ("s3-cache-datasets", "msr"): {"kind": "request_sequence", "formats": ("lcs",)},
    ("s3-cache-datasets", "other"): {"kind": "request_sequence", "formats": ("oracle_general",)},
    ("s3-cache-datasets", "tencentBlock"): {"kind": "request_sequence", "formats": ("lcs",)},
    ("tencent", "cloud_disk"): {"kind": "aggregate_time_series", "formats": ("tencent_cloud_disk",)},
}

_FAMILY_ALIASES = {
    ("s3-cache-datasets", "2020_twr_cdn.zst"): ("s3-cache-datasets", "2020_twr_cdn"),
}


@dataclass(frozen=True)
class TraceIdentity:
    dataset: str
    family: str
    format: str
    logical_family_id: str
    rel_path: str
    path: str


@dataclass
class ParsedTrace:
    identity: TraceIdentity
    parser: str
    kind: str
    profile: Dict[str, object]


def logical_family_id(dataset: str, family: str) -> str:
    return f"{dataset}__{family}".replace("/", "__")


def normalize_rel(path: Path, root: Path = TRACE_ROOT) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def safe_stat(path: Path) -> os.stat_result:
    return path.stat()


def likely_temp_name(name: str) -> bool:
    if name.startswith(".") and any(marker in name for marker in TEMP_MARKERS):
        return True
    return name.endswith(".partial") or name.endswith(".incomplete")


def infer_dataset(path: Path) -> str:
    rel = normalize_rel(path)
    return rel.split("/", 1)[0]


def infer_family(path: Path) -> str:
    rel = normalize_rel(path)
    parts = rel.split("/")
    if not parts:
        return "unknown"
    dataset = parts[0]
    if dataset != "s3-cache-datasets":
        if dataset == "MSR":
            return "exchange"
        if dataset == "tencent":
            return "cloud_disk"
        return dataset
    if len(parts) >= 2:
        family = parts[1]
        if family in {"cache_dataset_txt", "cache_dataset_lcs", "cache_dataset_oracleGeneral", "cache_dataset_parquet"} and len(parts) >= 3:
            family = parts[2]
        family = _FAMILY_ALIASES.get((dataset, family), (dataset, family))[1]
        return family
    return dataset


def infer_format(path: Path) -> str:
    s = str(path)
    name = path.name
    if likely_temp_name(name):
        return "temp"
    if name in SKIP_NAMES:
        return "metadata"
    if "Cloud_Disk_dataset/disk_load_data/" in s:
        return "tencent_cloud_disk"
    if "Exchange-Server-Traces" in s and name.endswith(".csv.gz"):
        return "exchange_etw"
    if "/Baleen24/extracted/" in s and name.endswith(".trace"):
        return "baleen24"
    if ".oracleGeneral" in name or ".oracleGeneral." in name:
        return "oracle_general"
    if ".lcs" in name or ".lcs." in name:
        return "lcs"
    if name.endswith(".parquet"):
        return "parquet"
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return "tar_gz"
    if name.endswith(".tar"):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(".csv.gz"):
        return "csv_gz"
    if name.endswith(".csv.zst") or name.endswith(".zst"):
        return "text_zst"
    if name.endswith(".csv") or name.endswith(".trace") or name.endswith(".short") or name.endswith(".json"):
        return "text"
    return "other"


def canonical_identity_for_path(path: str | Path) -> TraceIdentity:
    path_obj = Path(path)
    dataset = infer_dataset(path_obj)
    family = infer_family(path_obj)
    dataset, family = _FAMILY_ALIASES.get((dataset, family), (dataset, family))
    fmt = infer_format(path_obj)
    rel = normalize_rel(path_obj)
    return TraceIdentity(
        dataset=dataset,
        family=family,
        format=fmt,
        logical_family_id=logical_family_id(dataset, family),
        rel_path=rel,
        path=str(path_obj),
    )


def classify_role(path: Path, fmt: str) -> Tuple[str, bool, Optional[str]]:
    name = path.name
    if fmt == "temp":
        return "temporary", False, "temporary file"
    if fmt == "metadata":
        return "metadata", False, "metadata/helper file"
    if name.startswith("."):
        return "hidden", False, "hidden/helper file"
    if fmt in {"tar_gz", "tar", "zip"}:
        return "archive", False, "container/archive"
    if fmt in {
        "parquet",
        "oracle_general",
        "lcs",
        "exchange_etw",
        "baleen24",
        "tencent_cloud_disk",
        "csv_gz",
        "text_zst",
        "text",
    }:
        return "trace", True, None
    if path.suffix in SKIP_SUFFIXES:
        return "metadata", False, "metadata suffix"
    return "other", False, "unclassified"


@contextmanager
def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        fh = gzip.open(path, "rt", errors="replace")
    else:
        fh = open(path, "rt", errors="replace")
    try:
        yield fh
    finally:
        fh.close()


@contextmanager
def open_maybe_zstd_text(path: Path):
    if str(path).endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            assert proc.stdout is not None
            yield proc.stdout
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    else:
        with open(path, "rt", errors="replace") as fh:
            yield fh


@contextmanager
def open_maybe_zstd_bytes(path: Path):
    if str(path).endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert proc.stdout is not None
            yield proc.stdout
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    else:
        fh = open(path, "rb")
        try:
            yield fh
        finally:
            fh.close()


def quantile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


def mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def stddev(values: Sequence[float]) -> Optional[float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if len(clean) < 2:
        return 0.0 if clean else None
    n = 0
    mean_v = 0.0
    m2 = 0.0
    for value in clean:
        n += 1
        delta = value - mean_v
        mean_v += delta / n
        delta2 = value - mean_v
        m2 += delta * delta2
    return math.sqrt(m2 / n) if n else None


def summarize_numeric(values: Sequence[float]) -> Optional[Dict[str, float]]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean:
        return None
    return {
        "min": min(clean),
        "max": max(clean),
        "mean": mean(clean),
        "std": stddev(clean),
        "q50": quantile(clean, 0.50),
        "q90": quantile(clean, 0.90),
        "q99": quantile(clean, 0.99),
    }


def lag_autocorr(values: Sequence[float], lag: int = 1) -> Optional[float]:
    if len(values) <= lag + 1:
        return None
    x = values[:-lag]
    y = values[lag:]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    if den == 0:
        return None
    return num / den


def summarize_counter(counter: Counter, total: int, topk: int = 10) -> Dict[str, object]:
    most = counter.most_common(topk)
    return {
        "unique": len(counter),
        "top1_share": (most[0][1] / total) if total and most else None,
        f"top{topk}_share": (sum(v for _, v in most) / total) if total and most else None,
        "top_values": most,
    }


def sniff_delimiter(line: str) -> str:
    counts = {
        ",": line.count(","),
        "\t": line.count("\t"),
        " ": line.count(" "),
    }
    delim, n = max(counts.items(), key=lambda kv: kv[1])
    return delim if n > 0 else ","


def looks_like_header(fields: Sequence[str]) -> bool:
    alpha = sum(any(ch.isalpha() for ch in f) for f in fields)
    numeric = sum(is_floatish(f) for f in fields)
    return alpha > 0 and numeric < len(fields)


def is_floatish(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def _to_sample_line(fields: Sequence[object]) -> str:
    return ",".join("" if v is None else str(v) for v in fields)


def summarize_request_fields(
    ts: Sequence[float],
    obj_id: Sequence[int],
    obj_size: Sequence[float],
    opcode: Sequence[int],
    tenant: Sequence[int],
) -> Dict[str, object]:
    n = max(len(ts), len(obj_id), len(obj_size), len(opcode), len(tenant))
    out: Dict[str, object] = {"sample_records": n}
    if ts:
        iats = [max(0.0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]
        duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        out["ts_span"] = {"start": ts[0], "end": ts[-1], "duration": duration}
        out["iat_stats"] = summarize_numeric(iats)
        out["iat_zero_ratio"] = (sum(1 for x in iats if x == 0) / len(iats)) if iats else None
        out["iat_lag1_autocorr"] = lag_autocorr(iats, 1)
        out["burstiness_cv"] = ((stddev(iats) / mean(iats)) if iats and mean(iats) not in (None, 0) else None)
        out["sample_record_rate"] = (len(ts) / duration) if duration > 0 else None
    if obj_size:
        out["obj_size_stats"] = summarize_numeric(obj_size)
    if opcode:
        writes = sum(1 for x in opcode if x < 0)
        switches = sum(1 for i in range(1, len(opcode)) if opcode[i] != opcode[i - 1])
        out["write_ratio"] = writes / len(opcode)
        out["opcode_switch_ratio"] = (switches / (len(opcode) - 1)) if len(opcode) > 1 else None
    if obj_id:
        deltas = [obj_id[i] - obj_id[i - 1] for i in range(1, len(obj_id))]
        reuse = [1 for d in deltas if d == 0]
        out["obj_id_summary"] = summarize_counter(Counter(obj_id), len(obj_id), topk=10)
        out["reuse_ratio"] = (len(reuse) / len(deltas)) if deltas else None
        out["forward_seek_ratio"] = (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else None
        out["backward_seek_ratio"] = (sum(1 for d in deltas if d < 0) / len(deltas)) if deltas else None
        out["abs_stride_stats"] = summarize_numeric([abs(d) for d in deltas]) if deltas else None
        out["signed_stride_lag1_autocorr"] = lag_autocorr([float(d) for d in deltas], 1) if deltas else None
    if tenant:
        out["tenant_summary"] = summarize_counter(Counter(tenant), len(tenant), topk=10)
    return out


def summarize_time_series(ts: Sequence[float], values: Dict[str, Sequence[float]]) -> Dict[str, object]:
    out: Dict[str, object] = {"sample_records": len(ts)}
    if ts:
        iats = [max(0.0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]
        out["ts_span"] = {
            "start": ts[0],
            "end": ts[-1],
            "duration": ts[-1] - ts[0] if len(ts) > 1 else 0.0,
        }
        out["sampling_interval_stats"] = summarize_numeric(iats)
        if iats:
            out["sampling_interval_seconds"] = quantile(iats, 0.50)
    for key, seq in values.items():
        out[f"{key}_stats"] = summarize_numeric(seq)
        if seq:
            out[f"{key}_lag1_autocorr"] = lag_autocorr([float(v) for v in seq], 1)
    read_iops = values.get("read_iops", [])
    write_iops = values.get("write_iops", [])
    if read_iops and write_iops and len(read_iops) == len(write_iops):
        total = [read_iops[i] + write_iops[i] for i in range(len(read_iops))]
        out["total_iops_stats"] = summarize_numeric(total)
        out["idle_ratio"] = sum(1 for v in total if v == 0) / len(total)
    return out


def summarize_structured_rows(
    headers: Sequence[Optional[str]],
    rows: Sequence[Sequence[object]],
    parser_name: str,
    sample_lines: Sequence[str],
) -> Dict[str, object]:
    numeric_values: Dict[int, List[float]] = {}
    token_counters: Dict[int, Counter] = {}
    col_counts: List[int] = []
    row_count = 0
    for row in rows:
        col_counts.append(len(row))
        for idx, field in enumerate(row[:16]):
            text = "" if field is None else str(field)
            token_counters.setdefault(idx, Counter())[text] += 1
            if is_floatish(text):
                numeric_values.setdefault(idx, []).append(float(text))
        row_count += 1
    columns = []
    max_cols = max(col_counts) if col_counts else 0
    for idx in range(min(max_cols, 16)):
        num = numeric_values.get(idx, [])
        tok = token_counters.get(idx, Counter())
        columns.append(
            {
                "index": idx,
                "header": headers[idx] if idx < len(headers) else None,
                "numeric_ratio": (len(num) / row_count) if row_count else None,
                "numeric_stats": summarize_numeric(num),
                "token_summary": summarize_counter(tok, row_count, topk=5) if tok else None,
            }
        )
    first_numeric = numeric_values.get(0, [])
    time_like = None
    if len(first_numeric) > 1:
        diffs = [first_numeric[i] - first_numeric[i - 1] for i in range(1, len(first_numeric))]
        nonneg = sum(1 for d in diffs if d >= 0)
        time_like = {
            "monotone_nonnegative_ratio": nonneg / len(diffs) if diffs else None,
            "diff_stats": summarize_numeric(diffs),
        }
    return {
        "parser": parser_name,
        "sample_records": row_count,
        "column_count_stats": summarize_numeric([float(x) for x in col_counts]) if col_counts else None,
        "columns": columns,
        "first_numeric_column_profile": time_like,
        "sample_lines": list(sample_lines[:3]),
    }


def _profile_for_rows(
    headers: Sequence[Optional[str]],
    rows: Sequence[Sequence[object]],
    parser_name: str,
) -> Dict[str, object]:
    sample_lines = [_to_sample_line(row) for row in rows[:3]]
    return summarize_structured_rows(headers=headers, rows=rows, parser_name=parser_name, sample_lines=sample_lines)


def read_oracle_general(path: Path, max_records: int) -> Dict[str, object]:
    record_size = 24
    dt = struct.Struct("<IQIihh")
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    with open_maybe_zstd_bytes(path) as fh:
        for _ in range(max_records):
            chunk = fh.read(record_size)
            if len(chunk) < record_size:
                break
            ts_v, obj_v, size_v, _vtime, op_v, tenant_v = dt.unpack(chunk)
            ts.append(float(ts_v))
            obj_id.append(int(obj_v))
            obj_size.append(float(size_v))
            opcode.append(-1 if int(op_v) == 1 else 1)
            tenant.append(int(tenant_v))
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out["parser"] = "oracle_general"
    return out


def read_lcs(path: Path, max_records: int) -> Dict[str, object]:
    header_size = 8192
    magic = 0x123456789ABCDEF0
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    with open_maybe_zstd_bytes(path) as fh:
        header = fh.read(header_size)
        if len(header) < 16:
            return {"parser": "lcs_error", "sample_records": 0, "error": "short header"}
        magic_v, version = struct.unpack_from("<QQ", header, 0)
        if magic_v != magic:
            return {"parser": "lcs_error", "sample_records": 0, "error": f"magic mismatch: {magic_v}"}
        if version == 1:
            rec = struct.Struct("<IQIq")
            for _ in range(max_records):
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                ts_v, obj_v, size_v, _vtime = rec.unpack(chunk)
                ts.append(float(ts_v))
                obj_id.append(int(obj_v))
                obj_size.append(float(size_v))
                opcode.append(1)
        elif version == 2:
            rec = struct.Struct("<IQIIq")
            for _ in range(max_records):
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                ts_v, obj_v, size_v, packed, _vtime = rec.unpack(chunk)
                ts.append(float(ts_v))
                obj_id.append(int(obj_v))
                obj_size.append(float(size_v))
                op_raw = packed & 0xFF
                tenant_v = packed >> 8
                opcode.append(-1 if int(op_raw) == 13 else 1)
                tenant.append(int(tenant_v))
        elif version in _LCS_EXTRA_FIELDS:
            extra_words = _LCS_EXTRA_FIELDS[version]
            base_fmt = "<IQqII" + ("I" * extra_words) + "q"
            rec = struct.Struct(base_fmt)
            for _ in range(max_records):
                chunk = fh.read(rec.size)
                if len(chunk) < rec.size:
                    break
                unpacked = rec.unpack(chunk)
                ts_v = unpacked[0]
                obj_v = unpacked[1]
                size_v = unpacked[2]
                packed = unpacked[3]
                ttl_v = unpacked[4]
                _vtime = unpacked[-1]
                ts.append(float(ts_v))
                obj_id.append(int(obj_v))
                obj_size.append(float(size_v))
                op_raw = packed & 0xFF
                tenant_v = packed >> 8
                opcode.append(-1 if int(op_raw) == 13 else 1)
                tenant.append(int(tenant_v))
            out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
            out["parser"] = "lcs"
            out["lcs_version"] = version
            out["ttl_present"] = True
            out["feature_field_count"] = extra_words
            return out
        else:
            return {
                "parser": "lcs_error",
                "sample_records": 0,
                "lcs_version": version,
                "error": f"unsupported lcs version {version}",
            }
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out["parser"] = "lcs"
    out["lcs_version"] = version
    return out


def read_exchange_etw(path: Path, max_records: int) -> Dict[str, object]:
    pattern = re.compile(
        r"^\s*(DiskRead|DiskWrite),\s+(\d+),\s+[^,]+,\s+\d+,\s+\S+,\s+"
        r"(0x[\da-fA-F]+),\s+(0x[\da-fA-F]+),\s+(\d+),\s+(\d+),"
    )
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    response: List[float] = []
    with open_maybe_gzip(path) as fh:
        for line in fh:
            match = pattern.match(line)
            if not match:
                continue
            event_type, ts_v, offset_hex, size_hex, elapsed, disk = match.groups()
            ts.append(float(ts_v))
            obj_id.append(int(offset_hex, 16))
            obj_size.append(float(int(size_hex, 16)))
            opcode.append(1 if event_type == "DiskRead" else -1)
            tenant.append(int(disk))
            response.append(float(elapsed))
            if len(ts) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out["parser"] = "exchange_etw"
    out["response_time_stats"] = summarize_numeric(response)
    return out


def read_baleen24(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    with open(path, "rt", errors="replace") as fh:
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
            ts.append(op_time)
            obj_id.append(io_offset)
            obj_size.append(float(io_size))
            opcode.append(-1 if op_name == 2 else 1)
            tenant.append(pipeline)
            if len(ts) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out["parser"] = "baleen24"
    return out


def read_tencent_cloud_disk(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    read_iops: List[float] = []
    read_bw: List[float] = []
    write_iops: List[float] = []
    write_bw: List[float] = []
    disk_usage: List[float] = []
    with open(path, "rt", errors="replace") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                ts_v, ri, rb, wi, wb, du = row[:6]
                ts.append(float(ts_v))
                read_iops.append(float(ri))
                read_bw.append(float(rb))
                write_iops.append(float(wi))
                write_bw.append(float(wb))
                disk_usage.append(float(du))
            except ValueError:
                continue
            if len(ts) >= max_records:
                break
    out = summarize_time_series(
        ts,
        {
            "read_iops": read_iops,
            "read_bw_kbps": read_bw,
            "write_iops": write_iops,
            "write_bw_kbps": write_bw,
            "disk_usage_mb": disk_usage,
        },
    )
    out["parser"] = "tencent_cloud_disk"
    return out


def read_wiki_hash_text(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    obj_id: List[int] = []
    rows: List[Tuple[object, ...]] = []
    with open_maybe_zstd_text(path) as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 2:
                continue
            rows.append(tuple(row[:2]))
            try:
                ts.append(float(row[0]))
                obj_id.append(int(row[1]))
            except ValueError:
                pass
            if len(rows) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, [], [], [])
    out["parser"] = "wiki_hash_text"
    out["sample_lines"] = [_to_sample_line(row) for row in rows[:3]]
    out["columns"] = _profile_for_rows([None, None], rows, "wiki_hash_text")["columns"]
    return out


def read_systor_text(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    rows: List[Tuple[object, ...]] = []
    with open_maybe_zstd_text(path) as fh:
        reader = csv.reader(fh)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0] == "Timestamp":
                header = row
                continue
            if len(row) < 6:
                continue
            rows.append(tuple(row[:6]))
            try:
                ts.append(float(row[0]))
                opcode.append(-1 if row[2].strip().lower().startswith("w") else 1)
                tenant.append(int(float(row[3])))
                obj_id.append(int(float(row[4])))
                obj_size.append(float(row[5]))
            except ValueError:
                pass
            if len(rows) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out["parser"] = "systor_text"
    out["sample_lines"] = [_to_sample_line(row) for row in rows[:3]]
    out["columns"] = _profile_for_rows(header or [None] * 6, rows, "systor_text")["columns"]
    return out


def _read_text_rows(path: Path, max_records: int, delimiter: Optional[str] = None) -> Tuple[List[Optional[str]], List[Tuple[object, ...]]]:
    headers: List[Optional[str]] = []
    rows: List[Tuple[object, ...]] = []
    with open_maybe_zstd_text(path) as fh:
        first_data_row = True
        sniffed = delimiter
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if sniffed is None:
                sniffed = sniff_delimiter(line)
            if sniffed == " ":
                fields = line.split()
            else:
                fields = next(csv.reader([line], delimiter=sniffed))
            if first_data_row and looks_like_header(fields):
                headers = list(fields)
                first_data_row = False
                continue
            rows.append(tuple(fields))
            first_data_row = False
            if len(rows) >= max_records:
                break
    return headers, rows


def read_structured_text(path: Path, max_records: int, parser_name: str, delimiter: Optional[str] = None) -> Dict[str, object]:
    headers, rows = _read_text_rows(path, max_records=max_records, delimiter=delimiter)
    return _profile_for_rows(headers=headers, rows=rows, parser_name=parser_name)


def read_generic_text(path: Path, max_records: int) -> Dict[str, object]:
    return read_structured_text(path, max_records=max_records, parser_name="generic_text")


def parquet_probe_path(path: Path) -> Path:
    if not path.name.endswith(".parquet"):
        return path
    if ".sample10." in path.name or ".sample100." in path.name:
        return path

    stem = path.name[: -len(".parquet")]
    candidates = [
        path.with_name(f"{stem}.sample100.parquet"),
        path.parent / "sample100" / f"{stem}.sample100.parquet",
        path.parent / "sample10" / f"{stem}.sample10.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def read_parquet_duckdb(path: Path, max_records: int) -> Dict[str, object]:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        return {
            "parser": "parquet_unavailable",
            "sample_records": 0,
            "error": f"duckdb not available: {exc}",
        }
    conn = duckdb.connect(database=":memory:")
    try:
        duckdb_threads = os.environ.get("TRACE_PARSER_DUCKDB_THREADS", "2")
        try:
            conn.execute(f"PRAGMA threads={max(1, int(duckdb_threads))}")
        except Exception:
            pass
        probe_path = parquet_probe_path(path)
        rel = str(probe_path)
        rows = conn.execute("SELECT * FROM read_parquet(?) LIMIT ?", [rel, max_records]).fetchall()
        description = conn.description or []
        headers = [str(col[0]) for col in description]
        profile = _profile_for_rows(headers=headers, rows=rows, parser_name="parquet_duckdb")
        profile["duckdb_column_types"] = [str(col[1]) for col in description]
        profile["duckdb_threads"] = max(1, int(duckdb_threads))
        if probe_path != path:
            profile["parquet_probe_path"] = str(probe_path)
        return profile
    finally:
        conn.close()


def parse_trace(
    path: str | Path,
    max_records: int = 4096,
    *,
    dataset: Optional[str] = None,
    family: Optional[str] = None,
    fmt: Optional[str] = None,
) -> ParsedTrace:
    identity = canonical_identity_for_path(path)
    if dataset or family or fmt:
        identity = TraceIdentity(
            dataset=dataset or identity.dataset,
            family=family or identity.family,
            format=fmt or identity.format,
            logical_family_id=logical_family_id(dataset or identity.dataset, family or identity.family),
            rel_path=identity.rel_path,
            path=identity.path,
        )
    path_obj = Path(identity.path)

    if identity.format == "oracle_general":
        profile = read_oracle_general(path_obj, max_records)
    elif identity.format == "lcs":
        profile = read_lcs(path_obj, max_records)
    elif identity.format == "exchange_etw":
        profile = read_exchange_etw(path_obj, max_records)
    elif identity.format == "baleen24":
        profile = read_baleen24(path_obj, max_records)
    elif identity.format == "tencent_cloud_disk":
        profile = read_tencent_cloud_disk(path_obj, max_records)
    elif identity.format == "parquet":
        profile = read_parquet_duckdb(path_obj, max_records)
    elif identity.family == "2007_wiki":
        profile = read_wiki_hash_text(path_obj, max_records)
    elif identity.family == "2017_systor":
        profile = read_systor_text(path_obj, max_records)
    elif identity.family in {"2023_metaCDN", "2023_metaStorage", "2020_twr_cdn", "2016_wiki", "2024_google", "cache_trace_twitter_memcache"}:
        profile = read_structured_text(path_obj, max_records, parser_name=f"{identity.family}_text")
    elif identity.format in {"text", "text_zst", "csv_gz"}:
        profile = read_generic_text(path_obj, max_records)
    else:
        profile = {"parser": "unsupported", "sample_records": 0, "error": f"unsupported format {identity.format}"}

    parser_name = str(profile.get("parser", "unknown"))
    kind = ml_use_case_for_profile(identity, profile)
    return ParsedTrace(identity=identity, parser=parser_name, kind=kind, profile=profile)


def characterize_path(path: str | Path, max_records: int = 4096) -> Dict[str, object]:
    return parse_trace(path, max_records=max_records).profile


def ml_use_case_for_profile(identity: TraceIdentity, profile: Dict[str, object]) -> str:
    parser = str(profile.get("parser", ""))
    if identity.format == "tencent_cloud_disk" or parser == "tencent_cloud_disk":
        return "aggregate_time_series"
    if parser in {"oracle_general", "lcs", "exchange_etw", "baleen24", "systor_text", "wiki_hash_text"}:
        return "request_sequence"
    if identity.format == "parquet":
        return "structured_table"
    if identity.format in {"text", "text_zst", "csv_gz"}:
        return "structured_table"
    return FAMILY_REGISTRY.get((identity.dataset, identity.family), {}).get("kind", "other")  # type: ignore[return-value]


def feature_hints_for_profile(identity: TraceIdentity, profile: Dict[str, object]) -> List[str]:
    parser = str(profile.get("parser", ""))
    hints: List[str] = []
    if parser in {"oracle_general", "lcs", "exchange_etw", "baleen24", "systor_text", "wiki_hash_text"}:
        hints.extend(
            [
                "ts_duration",
                "iat_quantiles",
                "burstiness_cv",
                "obj_size_quantiles",
                "write_ratio",
                "opcode_switch_ratio",
                "reuse_ratio",
                "stride_stats",
            ]
        )
        if profile.get("tenant_summary"):
            hints.append("tenant_mix")
        if profile.get("response_time_stats"):
            hints.append("response_time_quantiles")
        if profile.get("ttl_present"):
            hints.append("ttl_fields")
    elif parser == "tencent_cloud_disk":
        hints.extend(
            [
                "sampling_interval",
                "read_iops_quantiles",
                "write_iops_quantiles",
                "bandwidth_quantiles",
                "disk_usage_quantiles",
                "idle_ratio",
                "lag1_autocorr",
            ]
        )
    elif "columns" in profile:
        hints.extend(["schema_profile", "numeric_column_stats", "time_like_column_check"])
    return hints
