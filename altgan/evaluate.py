"""Evaluate StackAtlas with the same long-rollout cache metrics as llgan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.long_rollout_eval import (  # noqa: E402
    _gap,
    _metrics_for_stream,
    _per_stream_obj_ids,
    _sample_real_stream,
)

from .mark_quality import mark_quality  # noqa: E402
from .model import StackAtlasModel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real-manifest", default="")
    p.add_argument("--cache-sizes", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = StackAtlasModel.load(args.model)
    fake_df = model.generate(args.n_records, n_streams=args.n_streams, seed=args.seed)
    real_df, real_manifest = _sample_real_stream(
        args.trace_dir,
        args.fmt,
        args.n_records,
        args.n_streams,
        args.seed,
        manifest_path=args.real_manifest,
    )

    if args.cache_sizes:
        cache_sizes = np.array([int(x) for x in args.cache_sizes.split(",") if x.strip()],
                               dtype=np.int64)
    else:
        real_streams = _per_stream_obj_ids(real_df)
        footprint = int(np.mean([np.unique(s).size for s in real_streams])) if real_streams else 2
        cache_sizes = np.unique(np.geomspace(max(1, footprint // 1000), max(footprint, 2), 20)
                                .astype(np.int64))

    fake_m = _metrics_for_stream(fake_df, cache_sizes)
    real_m = _metrics_for_stream(real_df, cache_sizes)
    gap_m = _gap(fake_m, real_m)
    mark_m = mark_quality(fake_df, real_df)
    result = {
        "model": args.model,
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "seed": args.seed,
        "n_records": args.n_records,
        "n_streams": args.n_streams,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
        "mark_quality": mark_m,
        "real_manifest": real_manifest,
    }

    out_path = Path(args.output) if args.output else Path(args.model).with_suffix("").with_suffix(".eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[altgan.evaluate] wrote {out_path}")
    print(json.dumps({
        "hrc_mae": gap_m["hrc_mae"],
        "fake_reuse_access": fake_m["reuse_access_rate"],
        "real_reuse_access": real_m["reuse_access_rate"],
        "fake_stack_median": fake_m["stack_distance_median"],
        "real_stack_median": real_m["stack_distance_median"],
        "mark_score": mark_m["mark_score"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
