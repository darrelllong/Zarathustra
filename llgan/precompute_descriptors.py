"""
Precompute cache descriptors per real trace file → JSONL.

Implements Phase A of IDEAS.md #18 (cache-descriptor distillation): scan a
trace directory, compute the 8-dim cache descriptor (cache_descriptor.py)
for every window in every file, aggregate to one descriptor per file via
median, and write a JSONL file mirroring the trace_characterizations.jsonl
layout. The training loop loads this JSONL via load_descriptor_jsonl.

Output line shape:
    {"file": "<basename>", "descriptor": [d0, ..., d7]}

Usage
-----
    python precompute_descriptors.py \
        --trace-dir /home/darrell/traces/tencent_block_1M \
        --fmt oracle_general \
        --output /home/darrell/traces/characterization/tencent_descriptors.jsonl \
        --window-len 12 \
        --max-records 200000

The descriptor is in *raw* (pre-normalisation) units; load_descriptor_jsonl
applies descriptor_normalise on read so checkpointed JSONLs survive
descriptor_normalise tweaks.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from cache_descriptor import (
    aggregate_file_descriptor,
    compute_descriptor_for_window,
)
from dataset import _READERS


def iter_trace_files(trace_dir: Path) -> Iterable[Path]:
    """Yield trace files in a directory, sorted, common compression handled."""
    exts = {".oracleGeneral", ".bin", ".csv", ".lcs"}
    candidates = []
    for p in sorted(trace_dir.rglob("*")):
        if not p.is_file():
            continue
        # Strip .zst/.gz to test inner extension.
        inner = p.name
        for ext in (".zst", ".gz"):
            if inner.endswith(ext):
                inner = inner[: -len(ext)]
        if Path(inner).suffix in exts or inner.endswith("oracleGeneral"):
            candidates.append(p)
        elif inner == p.name:  # no compression suffix and no recognised suffix
            # Tencent oracle_general files often have no extension at all
            candidates.append(p)
    return candidates


def file_descriptor(
    path: Path,
    fmt: str,
    max_records: int,
    window_len: int,
) -> np.ndarray:
    """Compute one file-level descriptor by aggregating per-window descriptors."""
    reader = _READERS[fmt]
    df = reader(str(path), max_records)
    if len(df) < window_len:
        return np.zeros(8, dtype=np.float32)

    obj_ids = df["obj_id"].to_numpy()
    ts = df["ts"].to_numpy() if "ts" in df.columns else None
    iats = None
    if ts is not None and len(ts) > 1:
        # Tencent ts is microseconds; alibaba similar. The descriptor
        # only uses the CV (std/mean), which is unit-invariant, so raw
        # diffs are fine.
        iats_full = np.diff(ts.astype(np.float64))
        iats_full = np.clip(iats_full, 0.0, None)
        # Pad to length T so windowing aligns; first event has no IAT.
        iats = np.concatenate([[0.0], iats_full])

    n_windows = len(obj_ids) // window_len
    descs = []
    for w in range(n_windows):
        s = w * window_len
        e = s + window_len
        descs.append(
            compute_descriptor_for_window(
                obj_ids[s:e],
                iats[s:e] if iats is not None else None,
            )
        )
    if not descs:
        return np.zeros(8, dtype=np.float32)
    return aggregate_file_descriptor(np.stack(descs, axis=0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", default="oracle_general", choices=list(_READERS))
    p.add_argument("--output", required=True)
    p.add_argument("--window-len", type=int, default=12)
    p.add_argument("--max-records", type=int, default=200_000,
                   help="Records per file used for descriptor estimate")
    p.add_argument("--limit-files", type=int, default=0,
                   help="Cap on files (0 = all). Use for quick smoke runs.")
    args = p.parse_args()

    trace_dir = Path(args.trace_dir)
    files = list(iter_trace_files(trace_dir))
    if args.limit_files:
        files = files[: args.limit_files]
    if not files:
        print(f"[precompute_descriptors] no trace files found under {trace_dir}",
              file=sys.stderr)
        sys.exit(1)
    print(f"[precompute_descriptors] {len(files)} files under {trace_dir}",
          flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as out:
        for i, f in enumerate(files):
            try:
                d = file_descriptor(f, args.fmt, args.max_records, args.window_len)
            except Exception as exc:  # surface but keep going
                print(f"[precompute_descriptors] {f.name}: {exc}", file=sys.stderr)
                continue
            row = {"file": f.name, "descriptor": d.tolist()}
            out.write(json.dumps(row) + "\n")
            written += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(files)} written={written}", flush=True)

    print(f"[precompute_descriptors] wrote {written} descriptors → {out_path}")


if __name__ == "__main__":
    main()
