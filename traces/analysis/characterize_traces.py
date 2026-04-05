#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

from parsers.core import feature_hints_for_profile, ml_use_case_for_profile, parse_trace
from trace_analysis_lib import load_jsonl


class FileTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise FileTimeout('per-file timeout')


def main() -> int:
    ap = argparse.ArgumentParser(description='Characterize trace files for ML priming')
    ap.add_argument('--inventory', default='/tiamat/zarathustra/analysis/out/trace_inventory.jsonl')
    ap.add_argument('--out', default='/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl')
    ap.add_argument('--sample-records', type=int, default=8192)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--per-file-timeout', type=int, default=60)
    args = ap.parse_args()

    inventory_path = Path(args.inventory)
    out_path = Path(args.out)
    rows = load_jsonl(inventory_path)
    trace_rows = [row for row in rows if row.get('is_trace')]
    if args.limit > 0:
        trace_rows = trace_rows[:args.limit]

    completed = set()
    if args.resume and out_path.exists():
        for row in load_jsonl(out_path):
            rel = row.get('rel_path')
            if rel:
                completed.add(rel)
        trace_rows = [row for row in trace_rows if row.get('rel_path') not in completed]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mode = 'a' if args.resume and out_path.exists() else 'w'
    with open(out_path, mode, encoding='utf-8') as out_fh:
        for idx, row in enumerate(trace_rows, start=1):
            path = Path(row['path'])
            record = dict(row)
            record['analysis_sample_records'] = args.sample_records
            try:
                if args.per_file_timeout > 0:
                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(args.per_file_timeout)
                try:
                    parsed = parse_trace(
                        path,
                        max_records=args.sample_records,
                        dataset=row.get('dataset'),
                        family=row.get('family'),
                        fmt=row.get('format'),
                    )
                    profile = parsed.profile
                finally:
                    if args.per_file_timeout > 0:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
            except Exception as exc:
                parsed = None
                profile = {
                    'parser': 'error',
                    'sample_records': 0,
                    'error': f'{type(exc).__name__}: {exc}',
                }
            record['profile'] = profile
            if parsed is not None:
                record['ml_use_case'] = ml_use_case_for_profile(parsed.identity, profile)
                record['feature_hints'] = feature_hints_for_profile(parsed.identity, profile)
            else:
                record['ml_use_case'] = 'other'
                record['feature_hints'] = []
            out_fh.write(json.dumps(record, sort_keys=True) + '\n')
            if idx % 100 == 0 or idx == len(trace_rows):
                elapsed = time.time() - started
                rate = idx / elapsed if elapsed > 0 else 0.0
                print(
                    f'[{idx}/{len(trace_rows)}] {row["rel_path"]} parser={profile.get("parser")} rate={rate:.2f}/s resume={args.resume}',
                    file=sys.stderr,
                    flush=True,
                )
    print(f'wrote {len(trace_rows)} records to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
