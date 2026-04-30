"""Per-cache-size HRC diagnostic: show where synthetic diverges from real.

Reads a fake CSV and a real manifest, computes miss-ratio at each cache size,
prints aligned HRC curves + adjacent-duplicate stats + hot-pool overlap.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/darrell/Zarathustra")
from llgan.long_rollout_eval import (
    _metrics_for_stream, _per_stream_obj_ids, _sample_real_stream,
)

if len(sys.argv) < 3:
    print("usage: diag_hrc.py FAKE.csv MANIFEST.json")
    sys.exit(1)

fake_csv = sys.argv[1]
manifest_path = sys.argv[2]

with open(manifest_path) as f:
    manifest = json.load(f)
real_df, _ = _sample_real_stream(
    trace_dir=manifest["trace_dir"], fmt=manifest["fmt"],
    n_records=manifest["n_records"], n_streams=manifest["n_streams"],
    seed=manifest["seed"], manifest_path=manifest_path,
)
gen_df = pd.read_csv(fake_csv)
if "stream_id" not in gen_df.columns:
    gen_df["stream_id"] = 0

per_real = _per_stream_obj_ids(real_df)
fp_mean = float(np.mean([len(np.unique(s)) for s in per_real]))

cache_sizes = np.unique(np.array([
    max(1, int(fp_mean * r)) for r in
    [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3,
     0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0]
]))

real_m = _metrics_for_stream(real_df, cache_sizes)
gen_m = _metrics_for_stream(gen_df, cache_sizes)

print(f"footprint mean per stream: real={real_m['footprint_mean_per_stream']:.0f} fake={gen_m['footprint_mean_per_stream']:.0f}")
print(f"reuse_access_rate:         real={real_m['reuse_access_rate']:.4f} fake={gen_m['reuse_access_rate']:.4f}")
print(f"stack p50:                 real={real_m['stack_distance_median']} fake={gen_m['stack_distance_median']}")
print(f"stack p90:                 real={real_m['stack_distance_p90']} fake={gen_m['stack_distance_p90']}")
print()
print(f"{'cache_size':>11} {'real_MR':>9} {'fake_MR':>9} {'delta':>9} {'rel%':>7}")
for cs, rmr, fmr in zip(cache_sizes, real_m["hrc"], gen_m["hrc"]):
    delta = fmr - rmr
    rel = (delta / max(rmr, 1e-9)) * 100
    print(f"{cs:>11} {rmr:>9.4f} {fmr:>9.4f} {delta:>+9.4f} {rel:>+6.1f}%")

# Adjacent-duplicate stats: how often fake emits the SAME obj_id back-to-back
def adj_dup_rate(df):
    out = []
    for sid, g in df.groupby("stream_id"):
        ids = g["obj_id"].values
        if len(ids) < 2:
            continue
        same = (ids[1:] == ids[:-1]).mean()
        out.append(same)
    return float(np.mean(out)) if out else 0.0

print()
print(f"adjacent-duplicate rate: real={adj_dup_rate(real_df):.4f} fake={adj_dup_rate(gen_df):.4f}")

# Top-K hot-object overlap: real top-K vs fake top-K of obj_id frequencies
from collections import Counter
def topk_overlap(real_df, gen_df, K=100):
    rc = Counter(int(x) for x in real_df["obj_id"].values)
    fc = Counter(int(x) for x in gen_df["obj_id"].values)
    rt = set(o for o, _ in rc.most_common(K))
    ft = set(o for o, _ in fc.most_common(K))
    if not rt:
        return 0.0
    return len(rt & ft) / len(rt)

# Note: fake obj_ids are synthetic (10000000+); real are tencent/alibaba IDs.
# Overlap will be near 0 unless fake mimics real-id space — but the
# REUSE-PATTERN footprint can still match. So skip the overlap and just
# report top-K residency CONCENTRATION (Gini-like).
def topk_share(df, K=100):
    c = Counter(int(x) for x in df["obj_id"].values)
    if not c:
        return 0.0
    top = sum(v for _, v in c.most_common(K))
    return top / sum(c.values())

print(f"top-100 access share:   real={topk_share(real_df):.4f} fake={topk_share(gen_df):.4f}")
print(f"top-1000 access share:  real={topk_share(real_df, 1000):.4f} fake={topk_share(gen_df, 1000):.4f}")
