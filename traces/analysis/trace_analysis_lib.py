from __future__ import annotations

import csv
import gzip
import io
import json
import math
import os
import struct
import subprocess
import tarfile
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from parsers.core import (
    TRACE_ROOT as SHARED_TRACE_ROOT,
    canonical_identity_for_path as shared_canonical_identity_for_path,
    classify_role as shared_classify_role,
    infer_dataset as shared_infer_dataset,
    infer_family as shared_infer_family,
    infer_format as shared_infer_format,
    parse_trace as shared_parse_trace,
)

TRACE_ROOT = SHARED_TRACE_ROOT
SKIP_NAMES = {
    'README', 'README.md', 'Disclaimer.txt', 'checksums.sha1', 'download.log',
    'wget.log', '.gitattributes', '.gitignore', 's3_filelist.tsv',
}
SKIP_SUFFIXES = {'.log', '.sha1', '.md', '.json'}
TEMP_MARKERS = ('.tm', '.tmp', '.part')


def normalize_rel(path: Path, root: Path = TRACE_ROOT) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def safe_stat(path: Path) -> os.stat_result:
    return path.stat()


def likely_temp_name(name: str) -> bool:
    if name.startswith('.') and any(marker in name for marker in TEMP_MARKERS):
        return True
    return name.endswith('.partial') or name.endswith('.incomplete')


def path_tags(path: Path) -> List[str]:
    return [part for part in path.parts if part not in ('/', '')]


def infer_dataset(path: Path) -> str:
    return shared_infer_dataset(path)


def infer_family(path: Path) -> str:
    return shared_infer_family(path)


def infer_format(path: Path) -> str:
    return shared_infer_format(path)


def classify_role(path: Path, fmt: str) -> Tuple[str, bool, Optional[str]]:
    return shared_classify_role(path, fmt)


def inventory_record(path: Path, root: Path = TRACE_ROOT) -> Dict[str, object]:
    st = safe_stat(path)
    identity = shared_canonical_identity_for_path(path)
    fmt = identity.format
    role, is_trace, skip_reason = classify_role(path, fmt)
    rel = normalize_rel(path, root)
    return {
        'path': str(path),
        'rel_path': rel,
        'dataset': identity.dataset,
        'family': identity.family,
        'format': fmt,
        'role': role,
        'is_trace': is_trace,
        'skip_reason': skip_reason,
        'size_bytes': st.st_size,
        'mtime_epoch': int(st.st_mtime),
    }


def walk_inventory(root: Path = TRACE_ROOT) -> Iterator[Dict[str, object]]:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            yield inventory_record(path, root)


@contextmanager
def open_maybe_gzip(path: Path):
    if str(path).endswith('.gz'):
        fh = gzip.open(path, 'rt', errors='replace')
    else:
        fh = open(path, 'rt', errors='replace')
    try:
        yield fh
    finally:
        fh.close()


@contextmanager
def open_maybe_zstd_text(path: Path):
    if str(path).endswith('.zst'):
        proc = subprocess.Popen(
            ['zstd', '-d', '-c', str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        try:
            assert proc.stdout is not None
            yield proc.stdout
        finally:
            if proc.stdout:
                proc.stdout.close()
            proc.wait()
    else:
        with open(path, 'rt', errors='replace') as fh:
            yield fh


@contextmanager
def open_maybe_zstd_bytes(path: Path):
    if str(path).endswith('.zst'):
        proc = subprocess.Popen(
            ['zstd', '-d', '-c', str(path)],
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
        fh = open(path, 'rb')
        try:
            yield fh
        finally:
            fh.close()


def iter_text_lines(path: Path, max_lines: Optional[int] = None) -> Iterator[str]:
    with open_maybe_zstd_text(path) as fh:
        count = 0
        for line in fh:
            yield line.rstrip('\n')
            count += 1
            if max_lines is not None and count >= max_lines:
                break


def quantile(values: List[float], p: float) -> Optional[float]:
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


def mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def stddev(values: List[float]) -> Optional[float]:
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


def summarize_numeric(values: List[float]) -> Optional[Dict[str, float]]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean:
        return None
    return {
        'min': min(clean),
        'max': max(clean),
        'mean': mean(clean),
        'std': stddev(clean),
        'q50': quantile(clean, 0.50),
        'q90': quantile(clean, 0.90),
        'q99': quantile(clean, 0.99),
    }


def lag_autocorr(values: List[float], lag: int = 1) -> Optional[float]:
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
        'unique': len(counter),
        'top1_share': (most[0][1] / total) if total and most else None,
        'top{}_share'.format(topk): (sum(v for _, v in most) / total) if total and most else None,
        'top_values': most,
    }


def sniff_delimiter(line: str) -> str:
    counts = {
        ',': line.count(','),
        '\t': line.count('\t'),
        ' ': line.count(' '),
    }
    delim, n = max(counts.items(), key=lambda kv: kv[1])
    return delim if n > 0 else ','


def looks_like_header(fields: List[str]) -> bool:
    alpha = sum(any(ch.isalpha() for ch in f) for f in fields)
    numeric = sum(is_floatish(f) for f in fields)
    return alpha > 0 and numeric < len(fields)


def is_floatish(text: str) -> bool:
    try:
        float(text)
        return True
    except Exception:
        return False


def parse_intish(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    try:
        if text.startswith(('0x', '0X')):
            return int(text, 16)
        return int(float(text))
    except Exception:
        return None


def summarize_request_fields(ts: List[float], obj_id: List[int], obj_size: List[float], opcode: List[int], tenant: List[int]) -> Dict[str, object]:
    n = max(len(ts), len(obj_id), len(obj_size), len(opcode), len(tenant))
    out: Dict[str, object] = {'sample_records': n}
    if ts:
        iats = [max(0.0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]
        duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
        out['ts_span'] = {'start': ts[0], 'end': ts[-1], 'duration': duration}
        out['iat_stats'] = summarize_numeric(iats)
        out['iat_zero_ratio'] = (sum(1 for x in iats if x == 0) / len(iats)) if iats else None
        out['iat_lag1_autocorr'] = lag_autocorr(iats, 1)
        out['burstiness_cv'] = ((stddev(iats) / mean(iats)) if iats and mean(iats) not in (None, 0) else None)
        out['sample_record_rate'] = (len(ts) / duration) if duration > 0 else None
    if obj_size:
        out['obj_size_stats'] = summarize_numeric(obj_size)
    if opcode:
        writes = sum(1 for x in opcode if x < 0)
        switches = sum(1 for i in range(1, len(opcode)) if opcode[i] != opcode[i - 1])
        out['write_ratio'] = writes / len(opcode)
        out['opcode_switch_ratio'] = (switches / (len(opcode) - 1)) if len(opcode) > 1 else None
    if obj_id:
        deltas = [obj_id[i] - obj_id[i - 1] for i in range(1, len(obj_id))]
        reuse = [1 for d in deltas if d == 0]
        out['obj_id_summary'] = summarize_counter(Counter(obj_id), len(obj_id), topk=10)
        out['reuse_ratio'] = (len(reuse) / len(deltas)) if deltas else None
        out['forward_seek_ratio'] = (sum(1 for d in deltas if d > 0) / len(deltas)) if deltas else None
        out['backward_seek_ratio'] = (sum(1 for d in deltas if d < 0) / len(deltas)) if deltas else None
        out['abs_stride_stats'] = summarize_numeric([abs(d) for d in deltas]) if deltas else None
        out['signed_stride_lag1_autocorr'] = lag_autocorr([float(d) for d in deltas], 1) if deltas else None
    if tenant:
        out['tenant_summary'] = summarize_counter(Counter(tenant), len(tenant), topk=10)
    return out


def read_oracle_general(path: Path, max_records: int) -> Dict[str, object]:
    record_size = 24
    dt = struct.Struct('<IQIihh')
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
    out['parser'] = 'oracle_general'
    return out


def read_lcs(path: Path, max_records: int) -> Dict[str, object]:
    header_size = 8192
    magic = 0x123456789abcdef0
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    with open_maybe_zstd_bytes(path) as fh:
        header = fh.read(header_size)
        if len(header) < 16:
            return {'parser': 'lcs', 'error': 'short header'}
        magic_v, version = struct.unpack_from('<QQ', header, 0)
        if magic_v != magic:
            return {'parser': 'lcs', 'error': f'magic mismatch: {magic_v}'}
        if version == 1:
            rec = struct.Struct('<IQIq')
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
            rec = struct.Struct('<IQIIq')
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
        else:
            return {
                'parser': 'lcs_stub',
                'sample_records': 0,
                'lcs_version': version,
                'note': f'LCS version {version} not decoded yet; kept as inventory-only characterization',
            }
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out['parser'] = 'lcs'
    out['lcs_version'] = version
    return out


def read_exchange_etw(path: Path, max_records: int) -> Dict[str, object]:
    import re
    pattern = re.compile(
        r'^\s*(DiskRead|DiskWrite),\s+(\d+),\s+[^,]+,\s+\d+,\s+\S+,\s+'
        r'(0x[\da-fA-F]+),\s+(0x[\da-fA-F]+),\s+(\d+),\s+(\d+),'
    )
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    response: List[float] = []
    with open_maybe_gzip(path) as fh:
        for line in fh:
            m = pattern.match(line)
            if not m:
                continue
            event_type, ts_v, offset_hex, size_hex, elapsed, disk = m.groups()
            ts.append(float(ts_v))
            obj_id.append(int(offset_hex, 16))
            obj_size.append(float(int(size_hex, 16)))
            opcode.append(1 if event_type == 'DiskRead' else -1)
            tenant.append(int(disk))
            response.append(float(elapsed))
            if len(ts) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out['parser'] = 'exchange_etw'
    out['response_time_stats'] = summarize_numeric(response)
    return out


def read_baleen24(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    obj_id: List[int] = []
    obj_size: List[float] = []
    opcode: List[int] = []
    tenant: List[int] = []
    with open(path, 'rt', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                obj_id.append(int(parts[1]))
                obj_size.append(float(parts[2]))
                ts.append(float(parts[3]))
                opcode.append(-1 if int(parts[4]) == 2 else 1)
                tenant.append(int(parts[5]))
            except Exception:
                continue
            if len(ts) >= max_records:
                break
    out = summarize_request_fields(ts, obj_id, obj_size, opcode, tenant)
    out['parser'] = 'baleen24'
    return out


def read_tencent_cloud_disk(path: Path, max_records: int) -> Dict[str, object]:
    ts: List[float] = []
    read_iops: List[float] = []
    read_bw: List[float] = []
    write_iops: List[float] = []
    write_bw: List[float] = []
    usage: List[float] = []
    with open(path, 'rt', errors='replace') as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                ts.append(float(row[0]))
                read_iops.append(float(row[1]))
                read_bw.append(float(row[2]))
                write_iops.append(float(row[3]))
                write_bw.append(float(row[4]))
                usage.append(float(row[5]))
            except Exception:
                continue
            if len(ts) >= max_records:
                break
    total_iops = [r + w for r, w in zip(read_iops, write_iops)]
    total_bw = [r + w for r, w in zip(read_bw, write_bw)]
    iats = [max(0.0, ts[i] - ts[i - 1]) for i in range(1, len(ts))]
    idle = [1 for x in total_iops if x == 0]
    return {
        'parser': 'tencent_cloud_disk',
        'sample_records': len(ts),
        'ts_span': {'start': ts[0], 'end': ts[-1], 'duration': (ts[-1] - ts[0]) if len(ts) > 1 else 0.0} if ts else None,
        'sampling_interval_stats': summarize_numeric(iats),
        'read_iops_stats': summarize_numeric(read_iops),
        'write_iops_stats': summarize_numeric(write_iops),
        'total_iops_stats': summarize_numeric(total_iops),
        'read_bw_kbps_stats': summarize_numeric(read_bw),
        'write_bw_kbps_stats': summarize_numeric(write_bw),
        'total_bw_kbps_stats': summarize_numeric(total_bw),
        'disk_usage_mb_stats': summarize_numeric(usage),
        'idle_ratio': (len(idle) / len(total_iops)) if total_iops else None,
        'write_share_iops_mean': (sum(write_iops) / sum(total_iops)) if total_iops and sum(total_iops) > 0 else None,
        'total_iops_lag1_autocorr': lag_autocorr(total_iops, 1),
        'disk_usage_lag1_autocorr': lag_autocorr(usage, 1),
    }


def read_generic_text(path: Path, max_records: int) -> Dict[str, object]:
    sample_lines: List[str] = []
    row_count = 0
    delimiter = ','
    header = None
    col_counts: List[int] = []
    numeric_values: Dict[int, List[float]] = {}
    token_counters: Dict[int, Counter] = {}
    with open_maybe_zstd_text(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            sample_lines.append(line)
            delimiter = sniff_delimiter(line)
            fields = [f for f in (line.split() if delimiter == ' ' else next(csv.reader([line], delimiter=delimiter)))]
            if header is None and looks_like_header(fields):
                header = fields
                continue
            col_counts.append(len(fields))
            for idx, field in enumerate(fields[:16]):
                token_counters.setdefault(idx, Counter())[field] += 1
                try:
                    numeric_values.setdefault(idx, []).append(float(field))
                except Exception:
                    pass
            row_count += 1
            if row_count >= max_records:
                break
    columns = []
    max_cols = max(col_counts) if col_counts else 0
    for idx in range(min(max_cols, 16)):
        num = numeric_values.get(idx, [])
        tok = token_counters.get(idx, Counter())
        columns.append({
            'index': idx,
            'header': header[idx] if header and idx < len(header) else None,
            'numeric_ratio': (len(num) / row_count) if row_count else None,
            'numeric_stats': summarize_numeric(num),
            'token_summary': summarize_counter(tok, row_count, topk=5) if tok else None,
        })
    first_numeric_col = numeric_values.get(0, [])
    time_like = None
    if len(first_numeric_col) > 1:
        diffs = [first_numeric_col[i] - first_numeric_col[i - 1] for i in range(1, len(first_numeric_col))]
        nonneg = sum(1 for d in diffs if d >= 0)
        time_like = {
            'monotone_nonnegative_ratio': nonneg / len(diffs) if diffs else None,
            'diff_stats': summarize_numeric(diffs),
        }
    return {
        'parser': 'generic_text',
        'sample_records': row_count,
        'delimiter': '\\t' if delimiter == '\t' else delimiter,
        'has_header': header is not None,
        'column_count_stats': summarize_numeric([float(x) for x in col_counts]) if col_counts else None,
        'columns': columns,
        'first_numeric_column_profile': time_like,
        'sample_lines': sample_lines[:3],
    }


def read_parquet_stub(path: Path, max_records: int) -> Dict[str, object]:
    return {
        'parser': 'parquet_stub',
        'sample_records': 0,
        'note': 'pyarrow/pandas unavailable on remote host; inventory-only characterization',
    }


def characterize_trace(path: Path, fmt: str, max_records: int) -> Dict[str, object]:
    parsed = shared_parse_trace(path, max_records=max_records, fmt=fmt)
    return parsed.profile


def dump_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + '\n')


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
