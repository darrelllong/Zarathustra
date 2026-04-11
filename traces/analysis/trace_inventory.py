#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from trace_analysis_lib import TRACE_ROOT, dump_jsonl, walk_inventory


def build_summary(rows):
    by_dataset = defaultdict(lambda: {'files': 0, 'trace_files': 0, 'bytes': 0})
    by_format = Counter()
    by_role = Counter()
    total_bytes = 0
    for row in rows:
        total_bytes += row['size_bytes']
        by_format[row['format']] += 1
        by_role[row['role']] += 1
        ds = by_dataset[row['dataset']]
        ds['files'] += 1
        ds['bytes'] += row['size_bytes']
        if row['is_trace']:
            ds['trace_files'] += 1
    return {
        'root': str(TRACE_ROOT),
        'total_files': len(rows),
        'total_bytes': total_bytes,
        'by_role': dict(by_role),
        'by_format': dict(by_format),
        'by_dataset': dict(sorted(by_dataset.items())),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Inventory /tiamat/zarathustra/traces')
    ap.add_argument('--out-dir', default='/tiamat/zarathustra/analysis/out')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(walk_inventory(TRACE_ROOT))
    dump_jsonl(out_dir / 'trace_inventory.jsonl', rows)
    summary = build_summary(rows)
    with open(out_dir / 'trace_inventory_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
