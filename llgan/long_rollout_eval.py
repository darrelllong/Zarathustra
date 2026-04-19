"""
Long-rollout diagnostic sidecar for LLGAN checkpoints.

Round 20 peer-review P1 #2: the frozen-bundle evaluator is a short-window
selector. Persistent-memory / IRD / chained-window ideas (#28/#31/#32) need a
long-rollout diagnostic so a cache-fidelity win does not hide as a combined-
score wash AND a combined-score win does not hide over-copying.

This tool runs a deterministic long-rollout against a trained checkpoint and
computes:

  * HRC (LRU hit-ratio curve) computed per-stream and averaged (not stitched,
    since streams are independent after the AD-round-1 offset removal).
  * Overall reuse rate.
  * First-decile vs. last-decile reuse rate (drift indicator).
  * Inter-reference-distance (IRD) histogram.
  * First-half vs. second-half Wasserstein-1 drift on obj_size and ts_delta.

It emits both the fake metrics and a real-trace baseline, plus gap summaries.
Output is written as JSON next to the checkpoint and printed as a terminal
summary.

CLI
---
    python -m llgan.long_rollout_eval \\
        --checkpoint /home/darrell/checkpoints/alibaba_v162/final.pt \\
        --trace-dir /tiamat/zarathustra/traces/alibaba \\
        --fmt oracle_general

Default rollout: 100,000 records, 4 streams, seed=42 (torch + numpy + random +
CUDA). All results are strictly reproducible across runs for fixed inputs.

Integration
-----------
Promotion gate for #28/#31/#32 checkpoints:
  1. frozen_sweep pass (short-window ★ acceptable).
  2. long_rollout_eval pass:
       a. HRC-MAE vs. real ≤ threshold (track per corpus),
       b. reuse-rate drift (first-decile vs. last-decile) ≤ 0.15 abs,
       c. half-to-half Wasserstein on ts_delta / obj_size ≤ real-trace baseline
          × 1.25.

Notes
-----
- Does NOT modify training; strictly evaluation-side.
- Does NOT replace eval.py; complements it for long-horizon claims.
- The "real baseline" is sampled deterministically from --trace-dir with the
  same seed so cross-run comparisons are apples-to-apples.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Checkpoints pickled by train.py reference bare module names ("dataset",
# "model", "train") because train.py runs from the llgan/ directory. Add
# llgan/ AND its parent to sys.path so torch.load finds both "dataset" and
# "llgan.dataset". Must happen before any torch.load call.
_LLGAN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_LLGAN_DIR.parent))
sys.path.insert(0, str(_LLGAN_DIR))


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # cuDNN autotune and non-deterministic kernels would violate the
        # "strictly reproducible across runs" promise in the docstring.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Long-rollout generator (mirrors generate.generate internals but returns the
# stitched record array directly without CSV round-tripping).
# ---------------------------------------------------------------------------

def _rollout(ckpt, n_records: int, n_streams: int, device, *,
             cond_sample: int = 0, char_file: str = "") -> "pandas.DataFrame":
    """Run a deterministic long-rollout and return a pandas DataFrame with a
    stream_id column prepended. Mirrors generate.generate but avoids CSV I/O
    so metrics can be computed directly.
    """
    import pandas as pd
    from .model import Generator, Recovery

    cfg = ckpt["config"]
    prep = ckpt["prep"]

    latent_ae = "R" in ckpt
    latent_dim = getattr(cfg, "latent_dim", 0) if latent_ae else None
    cond_dim = getattr(cfg, "cond_dim", 0)

    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=latent_dim, cond_dim=cond_dim,
                  film_cond=getattr(cfg, "film_cond", False),
                  gmm_components=getattr(cfg, "gmm_components", 0),
                  var_cond=getattr(cfg, "var_cond", False),
                  n_regimes=getattr(cfg, "n_regimes", 0),
                  num_lstm_layers=getattr(cfg, "num_lstm_layers", 1),
                  gp_prior=getattr(cfg, "gp_prior", False),
                  timestep=cfg.timestep,
                  retrieval_memory=getattr(cfg, "retrieval_memory", False),
                  retrieval_mem_size=getattr(cfg, "retrieval_mem_size", 32),
                  retrieval_key_dim=getattr(cfg, "retrieval_key_dim", 32),
                  retrieval_val_dim=getattr(cfg, "retrieval_val_dim", 32),
                  retrieval_decay=getattr(cfg, "retrieval_decay", 0.85),
                  retrieval_tau_write=getattr(cfg, "retrieval_tau_write", 0.5),
                  retrieval_n_warmup=getattr(cfg, "retrieval_n_warmup", 4),
                  ssm_backbone=getattr(cfg, "ssm_backbone", False),
                  ssm_state_dim=getattr(cfg, "ssm_state_dim", 16),
                  mtpp_timing=getattr(cfg, "mtpp_timing", False),
                  mtpp_sigma_min=getattr(cfg, "mtpp_sigma_min", 0.05),
                  ).to(device)
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        r_keys = ckpt["R"].keys()
        binary_cols = None
        if any(k.startswith("fc_cont") for k in r_keys):
            binary_cols = [i for i, col in enumerate(prep.col_names)
                           if col.lower() in {"opcode", "type", "rw", "op"}
                           or col.endswith("_reuse")]
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols,
                     binary_cols=binary_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    opcode_col = -1
    for i, col in enumerate(prep.col_names):
        if col.lower() in {"opcode", "type", "rw", "op"}:
            opcode_col = i
            break

    timestep = cfg.timestep
    records_per_stream = math.ceil(n_records / n_streams)
    windows_per_stream = math.ceil(records_per_stream / timestep)

    stream_windows: list[list[np.ndarray]] = [[] for _ in range(n_streams)]

    with torch.no_grad():
        if cond_dim > 0:
            # Random conditioning; deterministic via seeded torch.randn.
            cond = torch.randn(n_streams, cond_dim, device=device) * 0.5
            if getattr(G, 'cond_encoder', None) is not None:
                cond, _ = G.cond_encoder(cond, training=False)
            if getattr(G, 'regime_sampler', None) is not None:
                cond = G.regime_sampler(cond)
            noise = G.sample_noise(n_streams, device, cond=cond)
            z_global = torch.cat([cond, noise], dim=1)
        else:
            z_global = torch.randn(n_streams, cfg.noise_dim, device=device)
        hidden = None

        for _ in range(windows_per_stream):
            z_local = torch.randn(n_streams, timestep, cfg.noise_dim, device=device)
            latent, hidden = G(z_global, z_local, hidden=hidden, return_hidden=True)
            # SSM backbone returns (state, None); LSTM returns (h, c). Guard the
            # second element so either backbone carries cleanly across windows.
            h0 = hidden[0].detach() if hidden[0] is not None else None
            h1 = hidden[1].detach() if hidden[1] is not None else None
            hidden = (h0, h1)
            out = R(latent) if R is not None else latent
            if opcode_col >= 0:
                out[:, :, opcode_col] = (out[:, :, opcode_col] >= 0).float() * 2 - 1
            out_np = out.cpu().numpy()
            for s in range(n_streams):
                stream_windows[s].append(out_np[s])

    per_stream_dfs = []
    for s in range(n_streams):
        arr_s = np.concatenate(stream_windows[s], axis=0)[:records_per_stream]
        df_s = prep.inverse_transform(arr_s)
        df_s.insert(0, "stream_id", s)
        per_stream_dfs.append(df_s)

    df = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)
    return df


# ---------------------------------------------------------------------------
# Real-trace sampler — deterministic, matches rollout scale.
# ---------------------------------------------------------------------------

def _sample_real_stream(trace_dir: str, fmt: str, n_records: int, n_streams: int,
                        seed: int) -> "pandas.DataFrame":
    """Sample n_streams independent real trace streams, deterministic under
    (seed, trace_dir, fmt, n_streams). Each stream draws from a disjoint
    slice of the shuffled file pool so the fake vs real comparison is
    apples-to-apples: both fake and real have N independent streams of
    records_per_stream records each, tagged stream_id 0..N-1."""
    import pandas as pd
    from llgan.train import _collect_files
    from llgan.dataset import _READERS

    reader = _READERS.get(fmt)
    if reader is None:
        raise RuntimeError(f"Unknown trace format '{fmt}'")

    all_files = sorted(_collect_files(trace_dir, fmt))
    if not all_files:
        raise RuntimeError(f"No files found in {trace_dir}")

    records_per_stream = math.ceil(n_records / n_streams)
    rng = random.Random(seed)
    pool = all_files[:]
    rng.shuffle(pool)

    per_stream_dfs: list = []
    errors: list[str] = []
    file_idx = 0
    for s in range(n_streams):
        acc, have = [], 0
        while have < records_per_stream and file_idx < len(pool):
            path = pool[file_idx]
            file_idx += 1
            try:
                df = reader(str(path), records_per_stream)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{Path(path).name}: {e}")
                continue
            if df is None or len(df) == 0:
                continue
            acc.append(df)
            have += len(df)
        if not acc:
            continue
        df_s = pd.concat(acc, ignore_index=True).head(records_per_stream)
        df_s.insert(0, "stream_id", s)
        per_stream_dfs.append(df_s)

    if not per_stream_dfs:
        raise RuntimeError(
            f"No real records from {trace_dir}; first errors: {errors[:3]}")
    df_all = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)
    return df_all


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _per_stream_obj_ids(df) -> list:
    """Return a list of per-stream obj_id numpy arrays in stream_id order.
    NO cross-stream offset — metrics are computed per-stream and aggregated
    by the caller. Cross-stream id collisions (e.g. two streams emitting
    the same id) are then correctly NOT counted as reuse, because each
    stream is measured independently."""
    if 'obj_id' not in df.columns:
        raise RuntimeError("DataFrame missing 'obj_id' column — long-rollout eval needs it.")
    out = []
    for _s, g in df.groupby('stream_id', sort=True):
        out.append(g['obj_id'].astype(np.int64).values)
    return out


def _hrc_single(obj_ids: np.ndarray, cache_sizes: np.ndarray) -> np.ndarray:
    """LRU HRC on a single stream at the given cache sizes."""
    hrcs = np.zeros(len(cache_sizes), dtype=np.float64)
    if len(obj_ids) == 0:
        return hrcs
    for j, cs in enumerate(cache_sizes):
        cache: "OrderedDict[int, bool]" = OrderedDict()
        hits = 0
        for oid in obj_ids:
            key = int(oid)
            if key in cache:
                hits += 1
                cache.move_to_end(key)
            else:
                cache[key] = True
                if len(cache) > cs:
                    cache.popitem(last=False)
        hrcs[j] = hits / len(obj_ids)
    return hrcs


def _ird_positional(obj_ids: np.ndarray) -> np.ndarray:
    """POSITIONAL inter-reference distance: for each position i where the
    object was seen before, the number of accesses between it and its
    previous occurrence (i.e., i - last_i). NOT stack distance / distinct-
    intervening IRD (which would require a SortedList of O(N log N) cost
    and is not implemented here). Caller should read the output as
    "positional IRD", not classical reuse-distance IRD."""
    last = {}
    dists = []
    for i, oid in enumerate(obj_ids):
        key = int(oid)
        if key in last:
            dists.append(i - last[key])
        last[key] = i
    if not dists:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(dists, dtype=np.int64)


def _reuse_rate_deciles_cumulative(obj_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Cumulative reuse rate per decile: access is reuse if the obj_id has
    been seen anywhere earlier in the stream. This is the classic coupon-
    collector / warmup curve — useful as a sanity check that both fake and
    real exhibit the same shape, NOT as a drift indicator (which AD
    correctly flagged: the curve is monotone by construction)."""
    N = len(obj_ids)
    if N == 0:
        return np.zeros(n_bins, dtype=np.float64)
    seen: set = set()
    is_reuse = np.zeros(N, dtype=np.int8)
    for i, oid in enumerate(obj_ids):
        key = int(oid)
        if key in seen:
            is_reuse[i] = 1
        else:
            seen.add(key)
    edges = np.linspace(0, N, n_bins + 1, dtype=np.int64)
    return np.array([is_reuse[edges[k]:edges[k+1]].mean() if edges[k+1] > edges[k] else 0.0
                     for k in range(n_bins)], dtype=np.float64)


def _reuse_rate_deciles_local(obj_ids: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """LOCAL reuse rate per decile: within each decile, fraction of accesses
    whose obj_id also appears elsewhere *within the same decile*. Isolates
    'is the generator's local locality stable across the stream' from
    cold-start warmup — this is the drift indicator the Round 20 reviewer
    actually cares about."""
    N = len(obj_ids)
    if N == 0:
        return np.zeros(n_bins, dtype=np.float64)
    edges = np.linspace(0, N, n_bins + 1, dtype=np.int64)
    out = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        chunk = obj_ids[edges[k]:edges[k+1]]
        if len(chunk) == 0:
            continue
        _uniq, counts = np.unique(chunk, return_counts=True)
        reused_ids = _uniq[counts > 1]
        # fraction of accesses landing on a locally-reused id
        hits = np.isin(chunk, reused_ids).sum()
        out[k] = hits / len(chunk)
    return out


def _half_drift(values: np.ndarray) -> tuple:
    """Wasserstein-1 between the first and second half of a 1-D sample,
    plus a scale constant (median absolute value) for unit-free comparison.

    Returns (w1, scale) where w1/scale is dimensionless and roughly
    comparable across corpora with very different value magnitudes. Caller
    divides: fake_w1 / max(scale, eps) vs real_w1 / max(scale, eps)."""
    if len(values) < 2:
        return 0.0, 1.0
    try:
        from scipy.stats import wasserstein_distance
    except Exception:
        mid = len(values) // 2
        a = np.sort(values[:mid])
        b = np.sort(values[mid:mid + len(a)])
        w1 = float(np.abs(a - b).mean())
    else:
        mid = len(values) // 2
        w1 = float(wasserstein_distance(values[:mid], values[mid:]))
    scale = float(np.median(np.abs(values))) if len(values) else 1.0
    # Pure div-by-zero guard (1e-12). A larger unit-bearing floor (e.g. 1.0)
    # would silently denormalize small-magnitude corpora; with this guard,
    # genuinely degenerate data produces an honest blow-up ratio instead.
    return w1, max(scale, 1e-12)


def _metrics_for_stream(df, cache_sizes: np.ndarray, n_ird_bins: int = 32) -> dict:
    """Compute metrics PER-STREAM and aggregate across streams. HRC is
    averaged across streams, IRD is pooled, reuse rates are averaged,
    footprint is averaged (so the 'per-stream working-set size' is
    comparable between fake and real regardless of n_streams)."""
    per_stream = _per_stream_obj_ids(df)
    if not per_stream:
        raise RuntimeError("empty DataFrame passed to _metrics_for_stream")

    footprints = []
    reuse_access_rates = []
    reuse_object_rates = []
    deciles_cumulative = []
    deciles_local = []
    ird_pooled: list = []
    hrcs = []

    for obj_ids in per_stream:
        if len(obj_ids) == 0:
            continue
        unique_ids, id_counts = np.unique(obj_ids, return_counts=True)
        footprints.append(int(unique_ids.size))
        reuse_object_rates.append(float((id_counts > 1).sum()) / max(unique_ids.size, 1))
        # reuse_access_rate: total_reuses / total_accesses (computed from counts
        # without a second pass: accesses on reused ids minus one "first" access
        # per reused id = total reuses).
        total_acc = int(id_counts.sum())
        total_first_accesses = int(unique_ids.size)
        reuse_access_rates.append((total_acc - total_first_accesses) / total_acc)

        deciles_cumulative.append(_reuse_rate_deciles_cumulative(obj_ids, n_bins=10))
        deciles_local.append(_reuse_rate_deciles_local(obj_ids, n_bins=10))
        ird_pooled.append(_ird_positional(obj_ids))
        hrcs.append(_hrc_single(obj_ids, cache_sizes))

    ird = np.concatenate(ird_pooled) if ird_pooled else np.zeros(0, dtype=np.int64)
    if len(ird) > 0:
        max_ird = int(ird.max())
        bin_edges = np.unique(np.geomspace(1, max(max_ird, 2), n_ird_bins + 1).astype(np.int64))
        hist, _ = np.histogram(ird, bins=bin_edges)
        hist = hist.astype(np.float64) / max(hist.sum(), 1)
        ird_hist = hist.tolist()
        ird_bin_edges = bin_edges.tolist()
        ird_positional_median = int(np.median(ird))
        ird_positional_p90 = int(np.percentile(ird, 90))
    else:
        ird_hist, ird_bin_edges, ird_positional_median, ird_positional_p90 = [], [], 0, 0

    hrc_avg = np.mean(hrcs, axis=0) if hrcs else np.zeros(len(cache_sizes))

    # Half-to-half W1 on PER-STREAM concatenated values — first pass at each
    # stream's values is stationary, so stream concatenation is fine for drift.
    # Scale-normalize by pooled median to make the ratio dimensionless.
    ts_deltas, sizes = [], []
    if 'ts' in df.columns:
        for _s, g in df.groupby('stream_id', sort=True):
            d = np.diff(g['ts'].astype(np.float64).values)
            if len(d): ts_deltas.append(d)
    if 'obj_size' in df.columns:
        sizes = df['obj_size'].astype(np.float64).values
    ts_delta_arr = np.concatenate(ts_deltas) if ts_deltas else np.zeros(0)
    drift_ts, scale_ts = _half_drift(ts_delta_arr)
    drift_size, scale_size = _half_drift(np.asarray(sizes) if len(sizes) else np.zeros(0))

    total_records = int(sum(len(x) for x in per_stream))
    return {
        "n_records": total_records,
        "n_streams": len(per_stream),
        "footprint_mean_per_stream": float(np.mean(footprints)) if footprints else 0.0,
        "footprints_per_stream": footprints,
        "reuse_access_rate": float(np.mean(reuse_access_rates)) if reuse_access_rates else 0.0,
        "reuse_object_rate": float(np.mean(reuse_object_rates)) if reuse_object_rates else 0.0,
        "reuse_decile_cumulative_first": float(np.mean([d[0] for d in deciles_cumulative])) if deciles_cumulative else 0.0,
        "reuse_decile_cumulative_last": float(np.mean([d[-1] for d in deciles_cumulative])) if deciles_cumulative else 0.0,
        "reuse_decile_local_first": float(np.mean([d[0] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_decile_local_last": float(np.mean([d[-1] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_decile_local_drift": float(np.mean([d[-1] - d[0] for d in deciles_local])) if deciles_local else 0.0,
        "reuse_deciles_cumulative": [float(x) for x in np.mean(deciles_cumulative, axis=0)] if deciles_cumulative else [],
        "reuse_deciles_local": [float(x) for x in np.mean(deciles_local, axis=0)] if deciles_local else [],
        "ird_positional_median": ird_positional_median,
        "ird_positional_p90": ird_positional_p90,
        "ird_positional_histogram": ird_hist,
        "ird_positional_bin_edges": ird_bin_edges,
        "hrc": hrc_avg.tolist(),
        "cache_sizes": cache_sizes.tolist(),
        "drift_ts_delta_w1": drift_ts,
        "drift_ts_delta_w1_normalized": drift_ts / scale_ts,
        "drift_obj_size_w1": drift_size,
        "drift_obj_size_w1_normalized": drift_size / scale_size,
    }


def _gap(fake: dict, real: dict) -> dict:
    """Compute fake-vs-real gaps for a small set of headline metrics.
    Drift ratios use the scale-normalized Wasserstein-1 so they are
    dimensionless and comparable across corpora with different value
    magnitudes."""
    hrc_fake = np.asarray(fake.get("hrc", []))
    hrc_real = np.asarray(real.get("hrc", []))
    n = min(len(hrc_fake), len(hrc_real))
    hrc_mae = float(np.abs(hrc_fake[:n] - hrc_real[:n]).mean()) if n else None

    def _ratio(f, r):
        return (f / r) if (r is not None and r > 0) else None

    return {
        "hrc_mae": hrc_mae,
        "reuse_access_rate_delta": fake["reuse_access_rate"] - real["reuse_access_rate"],
        "reuse_object_rate_delta": fake["reuse_object_rate"] - real["reuse_object_rate"],
        "reuse_decile_local_drift_fake_minus_real":
            fake["reuse_decile_local_drift"] - real["reuse_decile_local_drift"],
        "drift_ts_delta_w1_ratio":
            _ratio(fake["drift_ts_delta_w1_normalized"], real["drift_ts_delta_w1_normalized"]),
        "drift_obj_size_w1_ratio":
            _ratio(fake["drift_obj_size_w1_normalized"], real["drift_obj_size_w1_normalized"]),
        "ird_positional_median_ratio":
            _ratio(fake["ird_positional_median"], real["ird_positional_median"]),
        "ird_positional_p90_ratio":
            _ratio(fake["ird_positional_p90"], real["ird_positional_p90"]),
        "footprint_ratio":
            _ratio(fake["footprint_mean_per_stream"], real["footprint_mean_per_stream"]),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--trace-dir", required=True, help="Real trace directory.")
    p.add_argument("--fmt", required=True, help="Trace format (e.g. oracle_general).")
    p.add_argument("--n-records", type=int, default=100_000,
                   help="Number of records in the long rollout (default 100000).")
    p.add_argument("--n-streams", type=int, default=4,
                   help="Number of parallel streams (default 4).")
    p.add_argument("--seed", type=int, default=42,
                   help="Deterministic seed for torch/numpy/random (default 42).")
    p.add_argument("--cache-sizes", type=str, default="",
                   help="Comma-separated override cache sizes. Default = log-spaced "
                        "across 1..footprint, 20 points.")
    p.add_argument("--output", default="",
                   help="Override output JSON path. Default: <ckpt-parent>/"
                        "long_rollout_<ckpt-stem>.json.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[error] checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    print(f"[long_rollout_eval] device={device}  seed={args.seed}")
    print(f"[long_rollout_eval] checkpoint={ckpt_path}")

    t0 = time.time()
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    print(f"[long_rollout_eval] generating {args.n_records:,} records "
          f"across {args.n_streams} stream(s) …")
    fake_df = _rollout(ckpt, args.n_records, args.n_streams, device)
    t_fake = time.time() - t0

    print(f"[long_rollout_eval] sampling real trace baseline …")
    real_df = _sample_real_stream(args.trace_dir, args.fmt, args.n_records,
                                  args.n_streams, args.seed)
    t_real = time.time() - t0 - t_fake

    # Determine cache sizes from the mean per-stream real footprint (so cache
    # scales make sense for an N-stream comparison rather than N× the mean).
    real_per_stream = _per_stream_obj_ids(real_df)
    real_footprint = int(np.mean([np.unique(s).size for s in real_per_stream])) \
        if real_per_stream else 0
    if args.cache_sizes:
        cache_sizes = np.array([int(s.strip()) for s in args.cache_sizes.split(",") if s.strip()],
                               dtype=np.int64)
    else:
        cache_sizes = np.unique(np.geomspace(
            max(1, real_footprint // 1000), max(real_footprint, 2), 20
        ).astype(np.int64))
    print(f"[long_rollout_eval] footprint (real)={real_footprint}  "
          f"cache_sizes=[{cache_sizes.min()}…{cache_sizes.max()}] "
          f"({len(cache_sizes)} points)")

    fake_m = _metrics_for_stream(fake_df, cache_sizes)
    real_m = _metrics_for_stream(real_df, cache_sizes)
    gap_m = _gap(fake_m, real_m)

    result = {
        "checkpoint": str(ckpt_path),
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "seed": args.seed,
        "n_records": args.n_records,
        "n_streams": args.n_streams,
        "rollout_time_sec": t_fake,
        "real_sample_time_sec": t_real,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = ckpt_path.parent / f"long_rollout_{ckpt_path.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    # Terminal summary.
    print()
    print("─" * 68)
    print(f"{'metric':<32}{'fake':>14}{'real':>14}{'gap':>8}")
    print("─" * 68)
    def _row(name, f_val, r_val, fmt="{:>14.4f}"):
        gap_txt = ""
        if isinstance(f_val, (int, float)) and isinstance(r_val, (int, float)) and r_val:
            gap_txt = f"{(f_val - r_val) / r_val * 100:+7.1f}%"
        f_txt = fmt.format(f_val) if isinstance(f_val, (int, float)) else f"{f_val:>14}"
        r_txt = fmt.format(r_val) if isinstance(r_val, (int, float)) else f"{r_val:>14}"
        print(f"{name:<32}{f_txt}{r_txt}  {gap_txt}")
    _row("reuse_access_rate", fake_m["reuse_access_rate"], real_m["reuse_access_rate"])
    _row("reuse_object_rate", fake_m["reuse_object_rate"], real_m["reuse_object_rate"])
    _row("reuse_decile_local_first", fake_m["reuse_decile_local_first"], real_m["reuse_decile_local_first"])
    _row("reuse_decile_local_last", fake_m["reuse_decile_local_last"], real_m["reuse_decile_local_last"])
    _row("reuse_decile_local_drift", fake_m["reuse_decile_local_drift"], real_m["reuse_decile_local_drift"])
    _row("ird_positional_median", fake_m["ird_positional_median"], real_m["ird_positional_median"], fmt="{:>14.0f}")
    _row("ird_positional_p90", fake_m["ird_positional_p90"], real_m["ird_positional_p90"], fmt="{:>14.0f}")
    _row("drift_ts_delta_w1_norm", fake_m["drift_ts_delta_w1_normalized"], real_m["drift_ts_delta_w1_normalized"])
    _row("drift_obj_size_w1_norm", fake_m["drift_obj_size_w1_normalized"], real_m["drift_obj_size_w1_normalized"])
    _row("footprint_per_stream", fake_m["footprint_mean_per_stream"], real_m["footprint_mean_per_stream"], fmt="{:>14.0f}")
    print("─" * 68)
    print(f"HRC-MAE (mean across streams) : {gap_m['hrc_mae']:.4f}" if gap_m['hrc_mae'] else
          "HRC-MAE: unavailable")
    print(f"json     → {out_path}")
    print("─" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
