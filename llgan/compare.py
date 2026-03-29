"""
compare.py — file-to-file fidelity: real trace vs. synthetic CSV.

No checkpoint required. Loads both files, computes per-feature statistical
distances, temporal structure metrics, and operational I/O metrics that
directly parallel the NAS 2024 paper evaluation.

Metrics
-------
Statistical (delta space for ts/obj_id, raw for others):
  Per-feature: mean/std/median, KS-statistic, Wasserstein-1 distance
  Joint:       MMD² (multi-scale RBF), PRDC (precision/recall/density/coverage)
Temporal:
  AutoCorr     lag-1..5 ACF difference per feature (TSGBench)
  DMD-GEN      Grassmannian distance of dominant temporal modes (NeurIPS 2025)
Operational (no btreplay needed):
  R/W ratio    read fraction
  IAT          inter-arrival time: mean, p50, p95, p99, CV (burstiness)
  Size         obj_size/size: mean, p50, p95, p99
  Locality     obj_id delta distribution; reuse rate; unique-object fraction
  Seq. access  fraction of accesses where |delta_obj_id| < threshold

Usage
-----
    python compare.py \\
        --real  /Volumes/Archive/Traces/.../tw-storage-1.oracleGeneral.zst \\
        --fmt   oracle_general \\
        --synth generated.csv \\
        --n     50000 \\
        --timestep 12

    # Compare two synthetic files against the same real trace:
    python compare.py --real trace.zst --fmt oracle_general \\
        --synth a.csv --synth2 b.csv --n 50000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent))
from eval import mmd2_numpy, compute_prdc_metrics, autocorr_score, dmdgen
from dataset import _READERS


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_real(path: str, fmt: str, n: int) -> pd.DataFrame:
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown format '{fmt}'. Choose from: {list(_READERS)}")
    df = reader(path, n)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No records loaded from {path}")
    return df.head(n).reset_index(drop=True)


def _load_synthetic(path: str, n: int) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=n)
    return df.reset_index(drop=True)


def _find_numeric_cols(df: pd.DataFrame, exclude=()) -> list[str]:
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Column identification
# ---------------------------------------------------------------------------

_TS_COLS      = {"ts", "timestamp"}
_OBJ_COLS     = {"obj_id", "lba", "offset"}
_SIZE_COLS    = {"obj_size", "size"}
_OPCODE_COLS  = {"opcode", "type", "rw", "op"}
_STREAM_COLS  = {"stream_id"}


def _find_col(df: pd.DataFrame, names: set) -> str | None:
    return next((c for c in df.columns if c.lower() in names), None)


# ---------------------------------------------------------------------------
# Opcode normalisation → returns float array of 0.0 (read) or 1.0 (write)
# ---------------------------------------------------------------------------

def _opcode_to_binary(series: pd.Series) -> np.ndarray:
    """
    Convert opcode column to binary float: 0.0 = read, 1.0 = write.
    Handles: int 0/1, float 0.0/1.0, strings 'r'/'w'/'read'/'write',
    and the ±1 encoding used internally by TracePreprocessor.

    The preprocessor encodes read=+1.0, write=-1.0 (opposite of the natural
    convention) because it maps "0" (read) → 1.0 and "1" (write) → -1.0 via
    _encode_opcode.  Synthetic CSVs from generate.py + inverse_transform
    therefore have opcode ∈ {+1.0, -1.0} where +1.0 means read, -1.0 write.
    """
    if pd.api.types.is_numeric_dtype(series):
        v = series.to_numpy(dtype=float)
        if v.min() < -0.5:
            # ±1 encoding (TracePreprocessor convention): +1=read, -1=write
            # → write = (v < 0)
            return (v < 0).astype(float)
        # 0/1 encoding (raw trace files): 0=read, 1=write
        return (v > 0.5).astype(float)
    # String encoding
    s = series.astype(str).str.lower()
    return (~s.isin({"r", "read", "0"})).astype(float)


# ---------------------------------------------------------------------------
# Delta encoding: ts and obj_id are cumsum'd in synthetic; compare deltas
# ---------------------------------------------------------------------------

def _ts_deltas(df: pd.DataFrame, stream_col: str | None = None) -> np.ndarray:
    """Return non-negative IAT (inter-arrival time) values."""
    col = _find_col(df, _TS_COLS)
    if col is None:
        return np.array([])
    if stream_col and stream_col in df.columns:
        # Per-stream deltas to avoid negative jumps between streams
        parts = []
        for _, g in df.groupby(stream_col):
            dt = np.diff(g[col].to_numpy(dtype=float))
            parts.append(dt[dt >= 0])
        return np.concatenate(parts) if parts else np.array([])
    dt = np.diff(df[col].to_numpy(dtype=float))
    return dt[dt >= 0]


def _obj_deltas(df: pd.DataFrame, stream_col: str | None = None) -> np.ndarray:
    """Return absolute obj_id deltas (access stride)."""
    col = _find_col(df, _OBJ_COLS)
    if col is None:
        return np.array([])
    if stream_col and stream_col in df.columns:
        parts = []
        for _, g in df.groupby(stream_col):
            d = np.abs(np.diff(g[col].to_numpy(dtype=float)))
            parts.append(d)
        return np.concatenate(parts) if parts else np.array([])
    return np.abs(np.diff(df[col].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Operational metrics
# ---------------------------------------------------------------------------

def _rw_ratio(df: pd.DataFrame) -> float | None:
    col = _find_col(df, _OPCODE_COLS)
    if col is None:
        return None
    binary = _opcode_to_binary(df[col])
    return float(1.0 - binary.mean())   # fraction of reads


def _iat_stats(deltas: np.ndarray) -> dict | None:
    if len(deltas) == 0:
        return None
    return {
        "mean": float(np.mean(deltas)),
        "p50":  float(np.percentile(deltas, 50)),
        "p95":  float(np.percentile(deltas, 95)),
        "p99":  float(np.percentile(deltas, 99)),
        "cv":   float(np.std(deltas) / (np.mean(deltas) + 1e-12)),
    }


def _size_stats(df: pd.DataFrame) -> dict | None:
    col = _find_col(df, _SIZE_COLS)
    if col is None:
        return None
    s = df[col].to_numpy(dtype=float)
    s = s[s > 0]
    if len(s) == 0:
        return None
    return {
        "mean": float(np.mean(s)),
        "p50":  float(np.percentile(s, 50)),
        "p95":  float(np.percentile(s, 95)),
        "p99":  float(np.percentile(s, 99)),
    }


def _locality_stats(df: pd.DataFrame, window: int = 10_000,
                    stream_col: str | None = None) -> dict | None:
    col = _find_col(df, _OBJ_COLS)
    if col is None:
        return None

    # Obj deltas
    obj_d = _obj_deltas(df, stream_col)
    seq_frac = float((obj_d <= 1).mean()) if len(obj_d) > 0 else 0.0

    # Reuse rate: rolling window (whole file, not per-stream for this metric)
    ids = df[col].to_numpy(dtype=float)
    n = len(ids)
    reuse = 0
    seen: dict[int, int] = {}   # key → last-seen index
    for i, oid in enumerate(ids):
        key = int(round(oid))
        if key in seen and (i - seen[key]) <= window:
            reuse += 1
        seen[key] = i
    reuse_rate = float(reuse / max(n, 1))

    sample = ids[:min(n, window)]
    unique_frac = float(len(np.unique(sample.round(0))) / len(sample))

    return {
        "delta_mean":  float(np.mean(obj_d)) if len(obj_d) else 0.0,
        "delta_p99":   float(np.percentile(obj_d, 99)) if len(obj_d) else 0.0,
        "seq_frac":    seq_frac,
        "reuse_rate":  reuse_rate,
        "unique_frac": unique_frac,
    }


# ---------------------------------------------------------------------------
# Build a normalised feature matrix for distributional metrics
#
# We use DELTA-ENCODED versions of ts and obj_id so that both real and
# synthetic live in the same space (absolute values differ by starting epoch /
# starting obj_id and are not comparable).
# ---------------------------------------------------------------------------

def _build_feature_matrix(
    df: pd.DataFrame,
    stream_col: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Returns (arr, col_names) where arr is (N, d).
    ts  → IAT (delta, non-negative)
    obj_id → signed log delta: sign(Δ)*log1p(|Δ|)
    opcode → 0.0 / 1.0 binary
    All others → raw numeric values.

    Rows with NaN/Inf are dropped.
    """
    cols_out: list[str] = []
    arrays: list[np.ndarray] = []
    n = len(df)

    for col in df.columns:
        clow = col.lower()
        if clow in _STREAM_COLS:
            continue
        if clow in _TS_COLS:
            # IAT delta (per-stream)
            deltas = _ts_deltas(df, stream_col)
            if len(deltas) < n // 2:
                continue
            # Pad/truncate to match row count (lose 1 per stream/group — acceptable)
            arr = np.full(n, np.nan)
            arr[1:len(deltas)+1] = deltas[:n-1]
            cols_out.append(f"{col}_delta")
            arrays.append(arr)
        elif clow in _OBJ_COLS:
            # Signed-log delta (per-stream)
            if stream_col and stream_col in df.columns:
                arr = np.full(n, 0.0)
                for _, g in df.groupby(stream_col):
                    idx = g.index
                    raw = g[col].to_numpy(dtype=float)
                    d = np.diff(raw, prepend=raw[0])
                    arr[idx] = np.sign(d) * np.log1p(np.abs(d))
            else:
                raw = df[col].to_numpy(dtype=float)
                d = np.diff(raw, prepend=raw[0])
                arr = np.sign(d) * np.log1p(np.abs(d))
            cols_out.append(f"{col}_logdelta")
            arrays.append(arr)
        elif clow in _OPCODE_COLS:
            arr = _opcode_to_binary(df[col])
            cols_out.append(col)
            arrays.append(arr)
        elif pd.api.types.is_numeric_dtype(df[col]):
            arr = df[col].to_numpy(dtype=float)
            cols_out.append(col)
            arrays.append(arr)

    if not arrays:
        raise RuntimeError("No usable numeric columns found.")

    mat = np.column_stack(arrays)
    # Drop rows with NaN/Inf
    mask = np.all(np.isfinite(mat), axis=1)
    return mat[mask], cols_out


def _normalise(real_arr: np.ndarray, synth_arr: np.ndarray):
    lo   = np.nanmin(real_arr, axis=0, keepdims=True)
    hi   = np.nanmax(real_arr, axis=0, keepdims=True)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    r_n  = (real_arr  - lo) / span * 2 - 1
    s_n  = (synth_arr - lo) / span * 2 - 1
    return r_n, s_n


# ---------------------------------------------------------------------------
# Per-feature statistical comparison (on the transformed feature matrix)
# ---------------------------------------------------------------------------

def _per_feature_stats(r: np.ndarray, s: np.ndarray, col_names: list) -> list[dict]:
    results = []
    for i, col in enumerate(col_names):
        rv = r[:, i];  rv = rv[np.isfinite(rv)]
        sv = s[:, i];  sv = sv[np.isfinite(sv)]
        if len(rv) == 0 or len(sv) == 0:
            continue
        ks_stat, ks_p = sp_stats.ks_2samp(rv, sv)
        w1 = float(sp_stats.wasserstein_distance(rv, sv))
        results.append({
            "col":       col,
            "real_mean": float(np.mean(rv)),
            "real_std":  float(np.std(rv)),
            "synth_mean":float(np.mean(sv)),
            "synth_std": float(np.std(sv)),
            "ks_stat":   float(ks_stat),
            "ks_p":      float(ks_p),
            "wasserstein1": w1,
        })
    return results


# ---------------------------------------------------------------------------
# Windowing for sequence-aware metrics
# ---------------------------------------------------------------------------

def _to_windows(arr: np.ndarray, timestep: int) -> np.ndarray:
    n, d = arr.shape
    n_win = n // timestep
    return arr[:n_win * timestep].reshape(n_win, timestep, d)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

SEP_WIDE  = "=" * 64
SEP_MID   = "─" * 48
SEP_THIN  = "·" * 40


def _fmt(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def _print_metric_row(name, r_val, s_val):
    if isinstance(r_val, float) and isinstance(s_val, float):
        delta = s_val - r_val
        print(f"  {name:<24}  real: {_fmt(r_val):>12}   synth: {_fmt(s_val):>12}   Δ={delta:+.4f}")
    else:
        print(f"  {name:<24}  real: {r_val!s:>12}   synth: {s_val!s:>12}")


def _print_operational(label, r_stats, s_stats):
    if r_stats is None or s_stats is None:
        return
    print(f"\n  [{label}]")
    for k in r_stats:
        if k in s_stats:
            _print_metric_row(k, r_stats[k], s_stats[k])


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare(
    real_path: str,
    fmt: str,
    synth_path: str,
    n: int = 50_000,
    timestep: int = 12,
    prdc_k: int = 5,
    n_samples: int = 2000,
    synth2_path: str | None = None,
) -> dict:
    print(f"\n{SEP_WIDE}")
    print(f"  REAL      : {real_path}")
    print(f"  SYNTHETIC : {synth_path}")
    if synth2_path:
        print(f"  SYNTHETIC2: {synth2_path}")
    print(SEP_WIDE)

    # 1. Load ----------------------------------------------------------------
    print("\nLoading files…")
    df_r = _load_real(real_path, fmt, n)
    df_s = _load_synthetic(synth_path, n)
    stream_col_s = "stream_id" if "stream_id" in df_s.columns else None
    print(f"  Real      : {len(df_r):,} records, cols={list(df_r.columns)}")
    print(f"  Synthetic : {len(df_s):,} records, cols={list(df_s.columns)}")

    # 2. Operational metrics (raw feature space) -----------------------------
    print(f"\n{SEP_MID}")
    print("  OPERATIONAL METRICS (raw feature space)")
    print(SEP_MID)

    # IAT deltas — per-stream for synthetic to avoid boundary artifacts
    r_dt  = _ts_deltas(df_r)
    s_dt  = _ts_deltas(df_s, stream_col_s)

    r_rw  = _rw_ratio(df_r);         s_rw  = _rw_ratio(df_s)
    r_iat = _iat_stats(r_dt);        s_iat = _iat_stats(s_dt)
    r_sz  = _size_stats(df_r);       s_sz  = _size_stats(df_s)
    r_loc = _locality_stats(df_r);   s_loc = _locality_stats(df_s, stream_col=stream_col_s)

    if r_rw is not None and s_rw is not None:
        _print_metric_row("R/W ratio (reads)", r_rw, s_rw)
    _print_operational("IAT", r_iat, s_iat)
    _print_operational("Object/Block size", r_sz, s_sz)
    _print_operational("Object locality", r_loc, s_loc)

    # 3. Build normalised feature matrices -----------------------------------
    # Use delta-encoded ts / obj_id so real and synthetic are in the same space.
    print(f"\n{SEP_MID}")
    print("  PER-FEATURE STATISTICAL DISTANCE (delta space for ts/obj_id)")
    print(SEP_MID)

    r_mat, r_cols = _build_feature_matrix(df_r, stream_col=None)
    s_mat, s_cols = _build_feature_matrix(df_s, stream_col=stream_col_s)

    # Use columns present in both
    shared = [c for c in r_cols if c in s_cols]
    # Initialise to None; set below if shared columns exist
    mmd = None; prdc = None; ac = None; dmd = None
    r_flat = s_flat = r_seqs = s_seqs = None
    feat_stats = []; n_win = 0; batch_sz = 64; rng = np.random.default_rng(42)
    r_feat = s_feat = r_n = s_n = None

    if not shared:
        print(f"  WARNING: no shared feature columns between {r_cols} and {s_cols}")
    else:
        r_feat = r_mat[:, [r_cols.index(c) for c in shared]]
        s_feat = s_mat[:, [s_cols.index(c) for c in shared]]

        feat_stats = _per_feature_stats(r_feat, s_feat, shared)
        print(f"  {'Col':<22}  {'KS':>6}  {'W1':>10}  {'Δmean':>10}  {'Δstd':>10}")
        print(f"  {SEP_THIN}")
        for fs in feat_stats:
            dm = fs['synth_mean'] - fs['real_mean']
            ds = fs['synth_std']  - fs['real_std']
            print(f"  {fs['col']:<22}  {fs['ks_stat']:6.3f}  "
                  f"{fs['wasserstein1']:10.4f}  {dm:+10.4f}  {ds:+10.4f}")

        # 4. Joint distribution metrics
        print(f"\n{SEP_MID}")
        print("  JOINT DISTRIBUTION METRICS (normalised feature space)")
        print(SEP_MID)

        r_n, s_n = _normalise(r_feat, s_feat)
        r_idx2 = rng.choice(len(r_n), min(n_samples, len(r_n)), replace=False)
        s_idx2 = rng.choice(len(s_n), min(n_samples, len(s_n)), replace=False)
        r_flat = r_n[r_idx2]
        s_flat = s_n[s_idx2]

        mmd  = mmd2_numpy(r_flat, s_flat)
        prdc = compute_prdc_metrics(r_flat, s_flat, k=prdc_k)

        print(f"  MMD²          : {mmd:.5f}  (lower = better)")
        print(f"  α-precision   : {prdc['precision']:.4f}  (fake plausibility; >0.7 good)")
        rstr = "⚠ mode collapse" if prdc['recall'] < 0.3 else ">0.3 ok, >0.7 good"
        print(f"  β-recall      : {prdc['recall']:.4f}  (real coverage; {rstr})")
        print(f"  density       : {prdc['density']:.4f}")
        print(f"  coverage      : {prdc['coverage']:.4f}")

        # 5. Temporal metrics (windowed)
        print(f"\n{SEP_MID}")
        print("  TEMPORAL STRUCTURE METRICS (windowed sequences)")
        print(SEP_MID)

        r_wins = _to_windows(r_n, timestep)
        s_wins = _to_windows(s_n, timestep)
        n_win  = min(n_samples, min(len(r_wins), len(s_wins)))
        batch_sz = min(64, max(n_win // 2, 1))

        if n_win < 10:
            print("  Too few windows for temporal metrics — increase --n.")
        else:
            rw_idx = rng.choice(len(r_wins), n_win, replace=False)
            sw_idx = rng.choice(len(s_wins), n_win, replace=False)
            r_seqs = r_wins[rw_idx]
            s_seqs = s_wins[sw_idx]
            ac  = autocorr_score(r_seqs, s_seqs, max_lag=5)
            dmd = dmdgen(r_seqs, s_seqs, r=4, n_batches=20, batch_size=batch_sz)
            print(f"  AutoCorr      : {ac:.4f}  (lag-1..5 ACF diff; 0=perfect)")
            print(f"  DMD-GEN       : {dmd:.4f}  (Grassmannian; 0=perfect, >0.3=poor)")

    results = {
        "mmd2":           mmd,
        "precision":      prdc["precision"] if prdc else None,
        "recall":         prdc["recall"]    if prdc else None,
        "autocorr":       ac,
        "dmdgen":         dmd,
        "rw_ratio_real":  r_rw,
        "rw_ratio_synth": s_rw,
        "iat_real":       r_iat,
        "iat_synth":      s_iat,
        "size_real":      r_sz,
        "size_synth":     s_sz,
        "locality_real":  r_loc,
        "locality_synth": s_loc,
        "per_feature":    feat_stats if 'feat_stats' in dir() else [],
    }

    # 6. Optional second synthetic -------------------------------------------
    if synth2_path and shared:
        print(f"\n{SEP_WIDE}")
        print(f"  COMPARISON: synth1 vs synth2 (both vs same real)")
        print(f"  synth1: {synth_path}")
        print(f"  synth2: {synth2_path}")
        print(SEP_WIDE)

        df_s2 = _load_synthetic(synth2_path, n)
        sc2   = "stream_id" if "stream_id" in df_s2.columns else None
        s2_mat, s2_cols = _build_feature_matrix(df_s2, stream_col=sc2)
        shared2 = [c for c in r_cols if c in s2_cols]
        if shared2:
            s2_idx3 = [s2_cols.index(c) for c in shared2]
            s2_feat  = s2_mat[:, s2_idx3]
            _, s2_n  = _normalise(r_feat[:, [r_cols.index(c) for c in shared2]], s2_feat)
            s2_idx4  = rng.choice(len(s2_n), min(n_samples, len(s2_n)), replace=False)
            s2_flat  = s2_n[s2_idx4]
            s2_wins  = _to_windows(s2_n, timestep)
            s2_widx  = rng.choice(len(s2_wins), min(n_win, len(s2_wins)), replace=False)
            s2_seqs  = s2_wins[s2_widx]

            mmd2v   = mmd2_numpy(r_flat, s2_flat)
            prdc2   = compute_prdc_metrics(r_flat, s2_flat, k=prdc_k)
            ac2     = autocorr_score(r_seqs, s2_seqs, max_lag=5) if r_seqs is not None else None
            dmd2v   = dmdgen(r_seqs, s2_seqs, r=4, n_batches=20, batch_size=batch_sz) if r_seqs is not None else None

            print(f"\n  {'Metric':<18}  {'synth1':>10}  {'synth2':>10}  {'Δ(2−1)':>10}")
            print(f"  {SEP_THIN}")
            rows = [
                ("MMD²",       mmd,               mmd2v,              False),
                ("α-precision", prdc["precision"], prdc2["precision"], True),
                ("β-recall",   prdc["recall"],    prdc2["recall"],    True),
                ("density",    prdc["density"],   prdc2["density"],   True),
                ("coverage",   prdc["coverage"],  prdc2["coverage"],  True),
            ]
            if ac is not None and ac2 is not None:
                rows += [("AutoCorr", ac, ac2, False), ("DMD-GEN", dmd, dmd2v, False)]
            for label, v1, v2, higher_better in rows:
                delta = v2 - v1
                arrow = ("↑" if delta > 0 else "↓") if higher_better else (
                        "↓" if delta < 0 else "↑")
                print(f"  {label:<18}  {v1:10.4f}  {v2:10.4f}  {delta:+10.4f} {arrow}")

            results["synth2"] = {"mmd2": mmd2v, "precision": prdc2["precision"],
                                  "recall": prdc2["recall"],
                                  "autocorr": ac2, "dmdgen": dmd2v}

    print(f"\n{SEP_WIDE}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(
        description="File-to-file fidelity: real trace vs. synthetic CSV."
    )
    p.add_argument("--real",   required=True)
    p.add_argument("--fmt",    default="oracle_general",
                   help=f"Trace format: {list(_READERS)}")
    p.add_argument("--synth",  required=True,
                   help="Synthetic CSV (output of generate.py)")
    p.add_argument("--synth2", default=None,
                   help="Second synthetic CSV for side-by-side comparison")
    p.add_argument("--n",       type=int, default=50_000)
    p.add_argument("--timestep", type=int, default=12)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--k",       type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    compare(
        real_path=args.real,
        fmt=args.fmt,
        synth_path=args.synth,
        n=args.n,
        timestep=args.timestep,
        prdc_k=args.k,
        n_samples=args.n_samples,
        synth2_path=args.synth2,
    )
