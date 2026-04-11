#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_analysis_lib import characterize_trace, load_jsonl


def ml_use_case(fmt: str, parser: str) -> str:
    if fmt == 'tencent_cloud_disk' or parser == 'tencent_cloud_disk':
        return 'aggregate_time_series'
    if fmt in {'oracle_general', 'lcs', 'exchange_etw', 'baleen24'}:
        return 'request_sequence'
    if fmt in {'text', 'text_zst', 'csv_gz'}:
        return 'generic_sequence_or_log'
    if fmt == 'parquet':
        return 'inventory_only'
    return 'other'


def feature_hints(fmt: str, parser: str, profile: dict) -> list[str]:
    hints: list[str] = []
    if parser in {'oracle_general', 'lcs', 'exchange_etw', 'baleen24'}:
        hints.extend([
            'ts_duration', 'iat_quantiles', 'burstiness_cv', 'obj_size_quantiles',
            'write_ratio', 'opcode_switch_ratio', 'reuse_ratio', 'stride_stats',
        ])
        if profile.get('tenant_summary'):
            hints.append('tenant_mix')
        if profile.get('response_time_stats'):
            hints.append('response_time_quantiles')
    elif parser == 'tencent_cloud_disk':
        hints.extend([
            'sampling_interval', 'read_iops_quantiles', 'write_iops_quantiles',
            'bandwidth_quantiles', 'disk_usage_quantiles', 'idle_ratio',
            'lag1_autocorr',
        ])
    elif parser == 'generic_text':
        hints.extend(['schema_profile', 'numeric_column_stats', 'time_like_column_check'])
    elif parser in {'parquet_stub', 'lcs_stub'}:
        hints.append('inventory_only')
    return hints


def main() -> int:
    ap = argparse.ArgumentParser(description='Repair failed characterization rows in place')
    ap.add_argument('--characterizations', default='/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl')
    ap.add_argument('--sample-records', type=int, default=4096)
    args = ap.parse_args()

    path = Path(args.characterizations)
    rows = load_jsonl(path)
    repaired = 0
    for row in rows:
        prof = row.get('profile', {})
        needs_repair = bool(prof.get('error')) or prof.get('parser') == 'error'
        if not needs_repair:
            continue
        new_prof = characterize_trace(Path(row['path']), row['format'], args.sample_records)
        row['profile'] = new_prof
        row['ml_use_case'] = ml_use_case(row['format'], new_prof.get('parser', ''))
        row['feature_hints'] = feature_hints(row['format'], new_prof.get('parser', ''), new_prof)
        repaired += 1

    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + '\n')
    tmp.replace(path)
    print(json.dumps({'repaired': repaired, 'rows': len(rows)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
