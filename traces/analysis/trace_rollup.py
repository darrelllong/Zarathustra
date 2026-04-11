#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from trace_analysis_lib import load_jsonl


def get_nested(dct, *keys):
    cur = dct
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def agg_numeric(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    vals = sorted(vals)
    return {
        'count': len(vals),
        'min': vals[0],
        'median': vals[len(vals) // 2],
        'max': vals[-1],
        'mean': sum(vals) / len(vals),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Roll up trace characterizations')
    ap.add_argument('--inventory', default='/tiamat/zarathustra/analysis/out/trace_inventory.jsonl')
    ap.add_argument('--characterizations', default='/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl')
    ap.add_argument('--out-dir', default='/tiamat/zarathustra/analysis/out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    inv = load_jsonl(Path(args.inventory))
    chars = load_jsonl(Path(args.characterizations))

    by_group = defaultdict(list)
    parser_counts = Counter()
    use_case_counts = Counter()
    for row in chars:
        key = (row['dataset'], row['family'], row['format'])
        by_group[key].append(row)
        parser_counts[row['profile'].get('parser', 'unknown')] += 1
        use_case_counts[row.get('ml_use_case', 'unknown')] += 1

    groups = {}
    for key, rows in sorted(by_group.items()):
        dataset, family, fmt = key
        groups['/'.join(key)] = {
            'dataset': dataset,
            'family': family,
            'format': fmt,
            'files': len(rows),
            'bytes': sum(r['size_bytes'] for r in rows),
            'parser_counts': dict(Counter(r['profile'].get('parser', 'unknown') for r in rows)),
            'sample_records': agg_numeric([get_nested(r, 'profile', 'sample_records') for r in rows]),
            'write_ratio': agg_numeric([get_nested(r, 'profile', 'write_ratio') for r in rows]),
            'reuse_ratio': agg_numeric([get_nested(r, 'profile', 'reuse_ratio') for r in rows]),
            'burstiness_cv': agg_numeric([get_nested(r, 'profile', 'burstiness_cv') for r in rows]),
            'iat_q50': agg_numeric([get_nested(r, 'profile', 'iat_stats', 'q50') for r in rows]),
            'size_q50': agg_numeric([get_nested(r, 'profile', 'obj_size_stats', 'q50') for r in rows]),
            'tenant_unique': agg_numeric([get_nested(r, 'profile', 'tenant_summary', 'unique') for r in rows]),
            'idle_ratio': agg_numeric([get_nested(r, 'profile', 'idle_ratio') for r in rows]),
            'total_iops_q50': agg_numeric([get_nested(r, 'profile', 'total_iops_stats', 'q50') for r in rows]),
        }

    summary = {
        'inventory_files': len(inv),
        'characterized_files': len(chars),
        'trace_files': sum(1 for row in inv if row.get('is_trace')),
        'parser_counts': dict(parser_counts),
        'use_case_counts': dict(use_case_counts),
        'groups': groups,
        'recommended_conditioning_features': {
            'request_sequence': [
                'iat_q50', 'iat_q90', 'burstiness_cv', 'obj_size_q50', 'obj_size_q90',
                'write_ratio', 'opcode_switch_ratio', 'reuse_ratio', 'top1_object_share',
                'tenant_count', 'forward_seek_ratio', 'backward_seek_ratio',
            ],
            'aggregate_time_series': [
                'sampling_interval_q50', 'read_iops_q50', 'read_iops_q90', 'write_iops_q50',
                'write_iops_q90', 'total_bw_q50', 'disk_usage_q50', 'idle_ratio',
                'lag1_autocorr',
            ],
            'generic_sequence_or_log': [
                'column_count', 'numeric_column_ratio', 'time_like_first_column', 'schema_tokens',
            ],
        },
    }

    with open(out_dir / 'trace_rollup.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    lines = []
    lines.append('# Trace Characterization Summary')
    lines.append('')
    lines.append(f"- Inventory files: {summary['inventory_files']}")
    lines.append(f"- Trace candidates: {summary['trace_files']}")
    lines.append(f"- Characterized files: {summary['characterized_files']}")
    lines.append(f"- ML use cases: {json.dumps(summary['use_case_counts'], sort_keys=True)}")
    lines.append(f"- Parsers: {json.dumps(summary['parser_counts'], sort_keys=True)}")
    lines.append('')
    lines.append('## Group Highlights')
    lines.append('')
    for group_key, group in groups.items():
        lines.append(f"### {group_key}")
        lines.append(f"- files: {group['files']}")
        lines.append(f"- bytes: {group['bytes']}")
        if group['write_ratio']:
            lines.append(f"- write_ratio median: {group['write_ratio']['median']}")
        if group['reuse_ratio']:
            lines.append(f"- reuse_ratio median: {group['reuse_ratio']['median']}")
        if group['burstiness_cv']:
            lines.append(f"- burstiness_cv median: {group['burstiness_cv']['median']}")
        if group['idle_ratio']:
            lines.append(f"- idle_ratio median: {group['idle_ratio']['median']}")
        if group['total_iops_q50']:
            lines.append(f"- total_iops q50 median: {group['total_iops_q50']['median']}")
        lines.append('')

    with open(out_dir / 'ML_PRIMING_SUMMARY.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines).rstrip() + '\n')

    print(json.dumps({'groups': len(groups), 'characterized_files': len(chars)}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
