"""
Evaluate long-rollout metrics on a pre-generated CSV trace (IDEA #48 experiment).

Usage
-----
    python eval_pregenerated.py \
        --fake-csv generated.csv \
        --real-json v195_long_rollout_epoch_0110.json \
        --output eval_out.json

Accepts CSVs with columns: stream_id, obj_id (integer).
Compares against real metrics from an existing long_rollout JSON baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from long_rollout_eval import (
    _per_stream_obj_ids,
    _hrc_single,
    _stack_distances,
    _ird_positional,
    _reuse_rate_deciles_cumulative,
    _reuse_rate_deciles_local,
    _metrics_for_stream,
)


def evaluate(fake_csv: str, real_json: str, output: str) -> dict:
    fake_df = pd.read_csv(fake_csv)
    if "obj_id" not in fake_df.columns:
        raise ValueError(f"--fake-csv {fake_csv} has no obj_id column")
    if "stream_id" not in fake_df.columns:
        fake_df["stream_id"] = 0
    fake_df["obj_id"] = fake_df["obj_id"].astype(np.int64)

    # Load real metrics from existing long_rollout JSON (saved by long_rollout_eval.py)
    with open(real_json) as f:
        baseline = json.load(f)
    real_m = baseline["real"]
    cache_sizes = np.array(baseline.get("fake", {}).get("cache_sizes",
                           real_m.get("cache_sizes", [])), dtype=np.float64)
    if len(cache_sizes) == 0:
        raise ValueError("Could not find cache_sizes in --real-json")

    # Compute fake metrics
    fake_m = _metrics_for_stream(fake_df, cache_sizes)

    real_hrc = np.array(real_m["hrc"])
    fake_hrc = np.array(fake_m["hrc"])
    hrc_mae = float(np.mean(np.abs(fake_hrc - real_hrc)))

    result = {
        "fake_csv": fake_csv,
        "fake": {
            "hrc_mae_vs_real":          round(hrc_mae, 6),
            "reuse_access_rate":        round(fake_m["reuse_access_rate"], 5),
            "stack_distance_median":    int(fake_m["stack_distance_median"]),
            "stack_distance_p90":       int(fake_m["stack_distance_p90"]),
            "footprint_mean_per_stream":round(fake_m["footprint_mean_per_stream"], 1),
        },
        "real": {
            "reuse_access_rate":        round(real_m["reuse_access_rate"], 5),
            "stack_distance_median":    int(real_m["stack_distance_median"]),
            "stack_distance_p90":       int(real_m["stack_distance_p90"]),
            "footprint_mean_per_stream":round(real_m["footprint_mean_per_stream"], 1),
        },
    }

    print(f"\n{'Metric':<35} {'Fake':>12} {'Real':>12} {'Ratio':>8}")
    print("-" * 70)
    print(f"{'HRC-MAE':<35} {hrc_mae:>12.6f} {'—':>12}")
    print(f"{'reuse_access_rate':<35} {fake_m['reuse_access_rate']:>12.5f} "
          f"{real_m['reuse_access_rate']:>12.5f} "
          f"{fake_m['reuse_access_rate']/max(real_m['reuse_access_rate'],1e-9):>8.3f}x")
    print(f"{'stack_distance_median':<35} {fake_m['stack_distance_median']:>12} "
          f"{real_m['stack_distance_median']:>12} "
          f"{fake_m['stack_distance_median']/max(real_m['stack_distance_median'],1):>8.3f}x")
    print(f"{'stack_distance_p90':<35} {fake_m['stack_distance_p90']:>12} "
          f"{real_m['stack_distance_p90']:>12}")
    print(f"{'footprint_mean_per_stream':<35} {fake_m['footprint_mean_per_stream']:>12.0f} "
          f"{real_m['footprint_mean_per_stream']:>12.0f}")

    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n→ {output}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fake-csv", required=True)
    p.add_argument("--real-json", required=True,
                   help="Path to existing long_rollout_eval JSON with 'real' metrics section")
    p.add_argument("--output", default="eval_pregenerated_out.json")
    args = p.parse_args()
    evaluate(args.fake_csv, args.real_json, args.output)
