"""Compare two existing trace CSVs with tools/cachesim.

This is the peer-trace companion to ``evaluate_neural_atlas --cachesim-bin``:
use it when a synthetic trace already exists and needs the same policy/capacity
MAE gate against a fixed real-manifest CSV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate_neural_atlas import (
    _compare_cachesim_rows,
    _run_cachesim,
    _write_namespaced_csv,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fake", required=True, help="Synthetic CSV trace.")
    p.add_argument("--real", required=True, help="Real-manifest CSV trace.")
    p.add_argument("--cachesim-bin", required=True, help="Path to tools/cachesim binary.")
    p.add_argument("--output", required=True, help="Output comparison JSON.")
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--no-namespace", action="store_true",
                   help="Do not rewrite obj_id with stream_id before cachesim.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    fake_path = Path(args.fake)
    real_path = Path(args.real)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fake_sim = fake_path
    real_sim = real_path
    if not args.no_namespace:
        stem = out_path.with_suffix("")
        fake_sim = stem.parent / f"{stem.name}.fake_namespaced.csv"
        real_sim = stem.parent / f"{stem.name}.real_namespaced.csv"
        _write_namespaced_csv(pd.read_csv(fake_path), fake_sim)
        _write_namespaced_csv(pd.read_csv(real_path), real_sim)

    fake_rows = _run_cachesim(Path(args.cachesim_bin), fake_sim, args.cache_sizes, args.policies)
    real_rows = _run_cachesim(Path(args.cachesim_bin), real_sim, args.cache_sizes, args.policies)
    report = _compare_cachesim_rows(fake_rows, real_rows, args.cache_sizes, args.policies)
    report.update({
        "fake_trace": str(fake_path),
        "real_trace": str(real_path),
        "fake_cachesim_trace": str(fake_sim),
        "real_cachesim_trace": str(real_sim),
        "binary": args.cachesim_bin,
    })
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(out_path),
        "mean_hrc_mae": report["mean_hrc_mae"],
        "policies": report["policies"],
        "cache_sizes": report["cache_sizes"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
