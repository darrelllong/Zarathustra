"""Mark-quality metrics for altgan traces.

The long-rollout panel measures cache behavior. These helpers measure the
non-object marks around that object process: timing, size, opcode, and tenant.
Lower scores are better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def mark_quality(fake_df: Any, real_df: Any) -> dict:
    """Compare generated and real mark distributions.

    The returned summary intentionally ignores ``obj_id`` and cache metrics.
    It focuses on marks that a neural sequence model might improve over a
    reservoir sampler.
    """

    fake_dt = _log_deltas(fake_df)
    real_dt = _log_deltas(real_df)
    fake_size = _log_sizes(fake_df)
    real_size = _log_sizes(real_df)

    dt_w1, dt_scale, dt_norm = _quantile_w1(fake_dt, real_dt)
    size_w1, size_scale, size_norm = _quantile_w1(fake_size, real_size)
    opcode_tv = _categorical_tv(fake_df, real_df, "opcode")
    tenant_tv = _categorical_tv(fake_df, real_df, "tenant")

    # A compact single number for ranking mark realism. Keep components visible
    # because different corpora may care about timing versus size/opcode.
    components = [dt_norm, size_norm, opcode_tv, tenant_tv]
    mark_score = float(np.mean([x for x in components if np.isfinite(x)]))
    return {
        "mark_score": mark_score,
        "ts_delta_log_w1": float(dt_w1),
        "ts_delta_log_scale": float(dt_scale),
        "ts_delta_log_w1_norm": float(dt_norm),
        "obj_size_log_w1": float(size_w1),
        "obj_size_log_scale": float(size_scale),
        "obj_size_log_w1_norm": float(size_norm),
        "opcode_tv": float(opcode_tv),
        "tenant_tv": float(tenant_tv),
    }


def _log_deltas(df: Any) -> np.ndarray:
    if "ts" not in df.columns:
        return np.zeros(0, dtype=np.float64)
    values = []
    group_key = "stream_id" if "stream_id" in df.columns else None
    groups = df.groupby(group_key, sort=True) if group_key else [(0, df)]
    for _, g in groups:
        ts = g["ts"].to_numpy(dtype=np.float64)
        if len(ts) == 0:
            continue
        dt = np.diff(ts, prepend=ts[0])
        values.append(np.log1p(np.maximum(dt, 0.0)))
    if not values:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(values)


def _log_sizes(df: Any) -> np.ndarray:
    if "obj_size" not in df.columns:
        return np.zeros(0, dtype=np.float64)
    sizes = df["obj_size"].to_numpy(dtype=np.float64)
    return np.log(np.maximum(sizes, 1.0))


def _quantile_w1(fake: np.ndarray, real: np.ndarray, n: int = 257) -> tuple[float, float, float]:
    if len(fake) == 0 or len(real) == 0:
        return 0.0, 1.0, 0.0
    q = np.linspace(0.0, 1.0, n)
    fq = np.quantile(fake, q)
    rq = np.quantile(real, q)
    w1 = float(np.mean(np.abs(fq - rq)))
    scale = float(np.subtract(*np.quantile(real, [0.95, 0.05])))
    # Some traces have nearly constant log inter-arrival times. In that case a
    # tiny absolute mismatch should remain tiny instead of exploding under an
    # almost-zero denominator.
    scale = max(scale, float(np.std(real)), 1.0)
    return w1, scale, w1 / scale


def _categorical_tv(fake_df: Any, real_df: Any, column: str) -> float:
    if column not in fake_df.columns or column not in real_df.columns:
        return 0.0
    fake = fake_df[column].astype(str).value_counts(normalize=True)
    real = real_df[column].astype(str).value_counts(normalize=True)
    keys = fake.index.union(real.index)
    diff = fake.reindex(keys, fill_value=0.0) - real.reindex(keys, fill_value=0.0)
    return float(0.5 * diff.abs().sum())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fake-csv", required=True)
    p.add_argument("--real-csv", required=True)
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> int:
    import pandas as pd

    args = _parse_args()
    fake_df = pd.read_csv(args.fake_csv)
    real_df = pd.read_csv(args.real_csv)
    result = mark_quality(fake_df, real_df)
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
