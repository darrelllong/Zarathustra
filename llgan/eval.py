"""
Evaluation metrics for LLGAN: MMD², PRDC, DMD-GEN, Context-FID, AutoCorr, HRC.

DMD-GEN (Abba Haddou et al., NeurIPS 2025, arXiv 2412.11292) is the first
principled metric for temporal mode collapse in time series generative models.

Algorithm:
  1. Run truncated Dynamic Mode Decomposition (DMD) on batches of real and
     generated sequences to extract their dominant temporal eigenvectors.
  2. Represent the r-dimensional DMD subspace as a point on the Grassmann
     manifold Gr(r, d) — the space of r-dim subspaces of R^d.
  3. Compute the principal angles between the real and generated subspaces
     (via SVD of U_real^T @ U_fake).
  4. Compute the 1-Wasserstein distance between the sorted principal-angle
     distributions over multiple mini-batches.

Interpretation:
  DMD-GEN ≈ 0  : generated sequences have the same temporal dynamics as real.
  DMD-GEN > 0.3: dynamical modes are meaningfully different (burst structure
                  wrong, wrong autocorrelation, missing periodic regimes).

Unlike MMD², DMD-GEN catches cases where the marginal distribution looks right
but the temporal autocorrelation / regime structure is wrong.


MMD² measures distributional closeness but cannot distinguish mode collapse from
good coverage — a generator that nails one narrow mode scores well on MMD².
PRDC separates these concerns:

  α-precision  : fraction of fake samples that are "plausible" (near real data)
  β-recall     : fraction of real data modes covered by fake samples
  density      : local precision averaged over nearest-neighbour balls
  coverage     : fraction of real samples with at least one fake neighbour

Rule of thumb (TransFusion / prdc paper):
  β-recall < 0.3  → mode collapse; generator covers only a slice of real distribution
  α-precision < 0.5 → generator hallucinates (outputs unrealistic sequences)
  Both > 0.7 is excellent.

Usage
-----
    python eval.py --checkpoint checkpoints/tencent_v7/best.pt \\
                   --trace-dir /Volumes/Archive/Traces/... \\
                   --fmt oracle_general \\
                   --n-samples 2000

    # Compare two checkpoints
    python eval.py --checkpoint checkpoints/tencent_v7/best.pt \\
                   --baseline   checkpoints/tencent_v4/best.pt \\
                   --trace-dir  /Volumes/Archive/Traces/... \\
                   --fmt oracle_general
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

from mmd import dmdgen, _dmd_subspace  # DMD-GEN lives in mmd.py; imported here to avoid duplication

# ---------------------------------------------------------------------------
# Context-FID  (TSGBench, VLDB 2024, arXiv 2309.03755)
# ---------------------------------------------------------------------------

def context_fid(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    E,
    device,
    batch_size: int = 256,
) -> float:
    """
    Fréchet distance in the trained Encoder's latent space.

    Uses the checkpoint's Encoder (trained on real data) as a fixed feature
    extractor — no separate model to train.  More sensitive to temporal
    structure than MMD² because the LSTM Encoder captures sequential context
    before projecting to a flat embedding.

    real_seqs / fake_seqs : (N, T, d) numpy arrays.
    E : trained Encoder module (from checkpoint), frozen.

    Returns a scalar ≥ 0; lower = closer temporal structure.
    Typical ranges (from TSGBench): < 10 good, < 50 acceptable, > 100 poor.
    """
    E.eval()

    def _encode(seqs: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(seqs), batch_size):
            batch = torch.tensor(seqs[i:i+batch_size], dtype=torch.float32,
                                 device=device)
            with torch.no_grad():
                h = E(batch)               # (B, T, latent_dim)
            out.append(h.mean(dim=1).cpu().numpy())  # mean-pool over time
        return np.concatenate(out, axis=0)   # (N, latent_dim)

    r_enc = _encode(real_seqs)
    f_enc = _encode(fake_seqs)

    mu_r, mu_f = r_enc.mean(0), f_enc.mean(0)
    # Regularise covariance matrices to avoid numerical issues with small samples
    sigma_r = np.cov(r_enc, rowvar=False) + np.eye(r_enc.shape[1]) * 1e-6
    sigma_f = np.cov(f_enc, rowvar=False) + np.eye(f_enc.shape[1]) * 1e-6

    diff  = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2.0 * covmean))
    return fid


# ---------------------------------------------------------------------------
# Autocorrelation score  (TSGBench, VLDB 2024)
# ---------------------------------------------------------------------------

def autocorr_score(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    max_lag: int = 5,
) -> float:
    """
    Average absolute difference in per-feature autocorrelation at lags 1..max_lag.

    Directly measures whether the generated sequences reproduce the temporal
    dependencies (burst periodicity, diurnal patterns) of real data.

    real_seqs / fake_seqs : (N, T, d) numpy arrays.
    Returns a scalar ≥ 0; lower = better temporal correlation matching.
    """
    def _acf(X: np.ndarray, lag: int) -> np.ndarray:
        # X: (N, T, d)
        X_c = X - X.mean(axis=1, keepdims=True)         # centre over time
        c0 = (X_c ** 2).mean(axis=1)                    # (N, d) variance
        cl = (X_c[:, :X.shape[1]-lag, :] *
              X_c[:, lag:, :]).mean(axis=1)              # (N, d) lag-cov
        return (cl / (c0 + 1e-8)).mean(axis=0)          # (d,) avg over samples

    total = 0.0
    for lag in range(1, max_lag + 1):
        total += np.abs(_acf(real_seqs, lag) - _acf(fake_seqs, lag)).mean()
    return float(total / max_lag)


# ---------------------------------------------------------------------------
# HRC (Hit Ratio Curve) — cache fidelity metric
# ---------------------------------------------------------------------------
# 2DIO (Wang et al., EUROSYS '26) and DiffGen (Liu et al., ICA3PP '25) both
# show that distributional metrics (MMD², recall) do NOT imply cache fidelity.
# A trace with perfect marginal distributions can produce completely wrong
# cache behaviour if the inter-reference distance (IRD) structure is wrong.
#
# We simulate LRU caches at multiple sizes and compare the hit ratio curves
# between real and generated traces.  The metric is MAE across cache sizes.

from collections import OrderedDict


def _lru_hit_ratio(obj_ids: np.ndarray, cache_size: int) -> float:
    """Simulate an LRU cache and return the hit ratio.

    obj_ids: 1-D array of integer object identifiers (quantised).
    cache_size: number of distinct objects the cache can hold.
    """
    if cache_size <= 0 or len(obj_ids) == 0:
        return 0.0
    cache = OrderedDict()
    hits = 0
    for oid in obj_ids:
        if oid in cache:
            hits += 1
            cache.move_to_end(oid)
        else:
            cache[oid] = True
            if len(cache) > cache_size:
                cache.popitem(last=False)
    return hits / len(obj_ids)


def _compute_hrc(obj_ids: np.ndarray, footprint: int,
                 n_points: int = 20) -> np.ndarray:
    """Compute LRU hit ratio curve at n_points cache sizes from 1% to 100% of footprint.

    Returns: (n_points,) array of hit ratios.
    """
    if footprint <= 0:
        return np.zeros(n_points)
    sizes = np.unique(np.linspace(
        max(1, footprint // 100), footprint, n_points
    ).astype(int))
    hrc = np.array([_lru_hit_ratio(obj_ids, int(s)) for s in sizes])
    # Pad to n_points if fewer unique sizes
    if len(hrc) < n_points:
        hrc = np.pad(hrc, (0, n_points - len(hrc)), constant_values=hrc[-1])
    return hrc[:n_points]


def _seqs_to_obj_ids(seqs: np.ndarray) -> np.ndarray:
    """Reconstruct integer object IDs from normalised (N, T, d) windows.

    The generator outputs normalised features in [-1, 1].  For HRC evaluation
    we only need *relative* object identity (which positions revisit the same
    object).  We reconstruct from the locality split:
      - Column index for obj_id_reuse: 3 (after ts, obj_size, tenant)
      - Column index for obj_id_stride: 4
    Reuse > 0 → same object (delta=0); otherwise delta = sign(stride)*expm1(|stride|).

    The resulting cumulative IDs are quantised to integers.
    """
    # seqs: (N, T, d) normalised
    N, T, d = seqs.shape
    # Flatten windows into one long stream per reconstruction
    flat = seqs.reshape(N * T, d)
    # Use obj_id_reuse (col 3) and obj_id_stride (col 4) if available
    if d >= 5:
        reuse = flat[:, 3]    # >0 means reuse
        stride = flat[:, 4]   # signed-log delta
        delta = np.where(reuse > 0, 0.0, np.sign(stride) * np.expm1(np.abs(stride)))
    elif d >= 2:
        # Fallback: use column 1 as raw obj_id delta
        delta = flat[:, 1]
    else:
        return np.arange(N * T)  # degenerate: every access is unique

    obj_ids = np.cumsum(delta).astype(np.int64)
    return obj_ids


def hrc_mae(real_seqs: np.ndarray, fake_seqs: np.ndarray,
            n_points: int = 20) -> tuple:
    """Compute HRC MAE between real and generated traces.

    Args:
        real_seqs, fake_seqs: (N, T, d) normalised windows.
        n_points: number of cache-size sample points on the HRC.

    Returns:
        (mae, real_hrc, fake_hrc): scalar MAE and the two HRC arrays.
    """
    real_ids = _seqs_to_obj_ids(real_seqs)
    fake_ids = _seqs_to_obj_ids(fake_seqs)

    # Use real footprint as reference for both curves
    footprint = len(np.unique(real_ids))
    if footprint == 0:
        return 0.0, np.zeros(n_points), np.zeros(n_points)

    real_hrc = _compute_hrc(real_ids, footprint, n_points)
    fake_hrc = _compute_hrc(fake_ids, footprint, n_points)

    mae = float(np.abs(real_hrc - fake_hrc).mean())
    return mae, real_hrc, fake_hrc


# prdc ships compute_prdc; fall back to a manual nearest-neighbour impl if absent
try:
    from prdc import compute_prdc
    _HAS_PRDC = True
except ImportError:
    _HAS_PRDC = False


# ---------------------------------------------------------------------------
# Nearest-neighbour PRDC fallback (pure numpy, O(n²) — fine for n ≤ 5000)
# ---------------------------------------------------------------------------

def _pairwise_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,d),(m,d) → (n,m) squared Euclidean distances."""
    a2 = (a ** 2).sum(1, keepdims=True)   # (n,1)
    b2 = (b ** 2).sum(1, keepdims=True)   # (m,1)
    return np.maximum(a2 + b2.T - 2 * a @ b.T, 0.0)


def _compute_prdc_numpy(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    """Compute PRDC via k-NN manifold estimation (no external library)."""
    rr = _pairwise_sq_dist(real, real)
    ff = _pairwise_sq_dist(fake, fake)
    rf = _pairwise_sq_dist(real, fake)   # (n_real, n_fake)

    # k-th nearest neighbour radius for each real / fake point (exclude self → k+1)
    r_real = np.sqrt(np.partition(rr, k + 1, axis=1)[:, k + 1])  # (n_real,)
    r_fake = np.sqrt(np.partition(ff, k + 1, axis=1)[:, k + 1])  # (n_fake,)

    # precision: fake[j] is "plausible" if it falls inside any real ball
    # (min dist from fake[j] to any real ≤ r_real of that real)
    # shape: (n_real, n_fake) — is fake[j] inside real[i]'s ball?
    in_real_ball = (np.sqrt(rf) <= r_real[:, None])  # (n_real, n_fake)
    precision = in_real_ball.any(axis=0).mean()       # fraction of fake that is plausible

    # recall: real[i] is "covered" if any fake falls inside real[i]'s ball
    recall = in_real_ball.any(axis=1).mean()          # fraction of real covered

    # density: average number of fake points per real ball / k
    density = in_real_ball.sum(axis=1).mean() / k

    # coverage: fraction of real balls that contain at least one fake point
    coverage = in_real_ball.any(axis=1).mean()

    return {"precision": float(precision), "recall": float(recall),
            "density": float(density),    "coverage": float(coverage)}


def compute_prdc_metrics(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict:
    if _HAS_PRDC:
        return compute_prdc(real_features=real, fake_features=fake, nearest_k=k)
    return _compute_prdc_numpy(real, fake, k)


# ---------------------------------------------------------------------------
# MMD² (multi-scale RBF, same as mmd.py but operates on numpy arrays)
# ---------------------------------------------------------------------------

def _rbf(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    diff = x[:, None, :] - y[None, :, :]      # (n, m, d)
    return np.exp(-(diff ** 2).sum(-1) / (2 * sigma ** 2))


def mmd2_numpy(real: np.ndarray, fake: np.ndarray,
               sigmas=(0.1, 0.5, 1.0, 2.0, 5.0)) -> float:
    total = 0.0
    for s in sigmas:
        total += (_rbf(real, real, s).mean()
                  + _rbf(fake, fake, s).mean()
                  - 2 * _rbf(real, fake, s).mean())
    return float(total / len(sigmas))


# ---------------------------------------------------------------------------
# Sample generation helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str, device):
    import os, sys
    sys.path.insert(0, str(Path(path).parent.parent))   # add llgan/ to path
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def _sample_fake(ckpt, n_samples: int, device,
                 batch_size: int = 256,
                 real_windows: np.ndarray = None) -> np.ndarray:
    """Generate fake samples.

    When cond_dim > 0, conditioning descriptors are computed from randomly
    sampled real_windows (instead of random noise), so the generated
    distribution matches the real workload descriptor distribution.

    Args:
        real_windows: (N, T, d) numpy array of real windows.  Required when
            cond_dim > 0; ignored otherwise.
    """
    from model import Generator, Recovery
    from dataset import compute_window_descriptors
    cfg  = ckpt["config"]
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
                  ).to(device)
    # Prefer EMA weights: smoother, less oscillated, consistently produces
    # better samples than the instantaneous live weights at any given epoch.
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    if cond_dim > 0 and real_windows is None:
        raise ValueError("real_windows required when cond_dim > 0")

    chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            noise = torch.randn(n, cfg.noise_dim, device=device)
            if cond_dim > 0:
                # Sample real windows and compute their descriptors
                idx = np.random.choice(len(real_windows), n, replace=True)
                rw = torch.tensor(real_windows[idx], dtype=torch.float32,
                                  device=device)
                cond = compute_window_descriptors(rw)  # (n, cond_dim)
                z_g = torch.cat([cond, noise], dim=1)
            else:
                z_g = noise
            z_l = torch.randn(n, cfg.timestep, cfg.noise_dim, device=device)
            out = G(z_g, z_l)
            if R is not None:
                out = R(out)
            chunks.append(out.cpu().numpy())
    fake = np.concatenate(chunks, axis=0)   # (n_samples, T, d)
    return fake.reshape(n_samples, -1)


def _load_encoder(ckpt, device):
    """Load the trained Encoder from a latent-AE checkpoint, or None."""
    if "E" not in ckpt:
        return None
    from model import Encoder
    cfg = ckpt["config"]
    prep = ckpt["prep"]
    latent_dim = getattr(cfg, "latent_dim", 0)
    E = Encoder(prep.num_cols, cfg.hidden_size, latent_dim).to(device)
    E.load_state_dict(ckpt["E"])
    E.eval()
    return E


def _sample_real(ckpt, trace_dir: str, fmt: str, n_samples: int) -> np.ndarray:
    import random
    sys.path.insert(0, ".")
    from train import _collect_files, _load_epoch_dataset
    prep = ckpt["prep"]
    cfg  = ckpt["config"]

    all_files = _collect_files(trace_dir, fmt)
    if not all_files:
        raise RuntimeError(f"No files found in {trace_dir}")
    files = random.sample(all_files, min(4, len(all_files)))
    ds, _ = _load_epoch_dataset(files, fmt, 15000, prep, cfg.timestep)
    if ds is None or len(ds) == 0:
        raise RuntimeError("Could not load real data for evaluation.")
    idx = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    windows = np.stack([ds[i].numpy() for i in idx])
    return windows.reshape(len(idx), -1)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, trace_dir: str, fmt: str,
             n_samples: int = 2000, k: int = 5,
             baseline_path: str = None) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")

    ckpt = _load_checkpoint(checkpoint_path, device)
    print(f"Epoch      : {ckpt['epoch']+1}")
    if "mmd" in ckpt:
        print(f"Saved MMD² : {ckpt['mmd']:.5f}")

    print(f"\nSampling {n_samples} real and fake windows …")
    real_flat = _sample_real(ckpt, trace_dir, fmt, n_samples)

    cfg  = ckpt["config"]
    timestep  = cfg.timestep
    num_cols  = ckpt["prep"].num_cols
    # Reshape flat (N, T*d) → (N, T, d) for sequence-aware metrics
    real_seqs = real_flat.reshape(-1, timestep, num_cols)

    # Pass real windows to _sample_fake so conditioning uses real descriptors
    cond_dim = getattr(cfg, "cond_dim", 0)
    fake_flat = _sample_fake(ckpt, n_samples, device,
                             real_windows=real_seqs if cond_dim > 0 else None)
    fake_seqs = fake_flat.reshape(-1, timestep, num_cols)

    mmd  = mmd2_numpy(real_flat, fake_flat)
    prdc = compute_prdc_metrics(real_flat, fake_flat, k=k)

    # Temporal structure metrics
    dmd_score = dmdgen(real_seqs, fake_seqs, r=4, n_batches=20,
                       batch_size=min(64, len(real_seqs) // 2))
    ac_score = autocorr_score(real_seqs, fake_seqs, max_lag=5)

    # Context-FID: Fréchet distance in the Encoder's latent space
    E = _load_encoder(ckpt, device)
    cfid = context_fid(real_seqs, fake_seqs, E, device) if E is not None else None

    # Reuse rate: fraction of positions (t>0) whose obj_id exactly matches
    # a prior position within the same window.  Real block I/O: ~40-50%.
    def _reuse_rate(seqs: np.ndarray) -> float:
        # seqs: (N, T, d); obj_id is column 1
        obj = seqs[:, :, 1]          # (N, T)
        hits = 0
        for t in range(1, obj.shape[1]):
            hits += (np.abs(obj[:, :t] - obj[:, t:t+1]) < 1e-6).any(axis=1).sum()
        return hits / (obj.shape[0] * (obj.shape[1] - 1))

    real_reuse = _reuse_rate(real_seqs)
    fake_reuse = _reuse_rate(fake_seqs)

    # HRC: cache fidelity via LRU hit ratio curve comparison
    hrc_score, real_hrc, fake_hrc = hrc_mae(real_seqs, fake_seqs, n_points=20)

    print(f"\n{'─'*40}")
    print(f"  MMD²         : {mmd:.5f}")
    print(f"  α-precision  : {prdc['precision']:.4f}  (fake plausibility)")
    print(f"  β-recall     : {prdc['recall']:.4f}  (real coverage)  {'⚠ mode collapse' if prdc['recall'] < 0.3 else ''}")
    print(f"  density      : {prdc['density']:.4f}")
    print(f"  coverage     : {prdc['coverage']:.4f}")
    print(f"  DMD-GEN      : {dmd_score:.4f}  (temporal dynamics; 0=perfect, >0.3=poor)")
    print(f"  AutoCorr     : {ac_score:.4f}  (lag-1..5 ACF diff; 0=perfect)")
    if cfid is not None:
        print(f"  Context-FID  : {cfid:.2f}  (Fréchet in encoder latent space; <10 good)")
    print(f"  reuse rate   : real={real_reuse:.3f}  fake={fake_reuse:.3f}"
          f"  {'⚠ locality gap' if abs(real_reuse - fake_reuse) > 0.1 else ''}")
    print(f"  HRC-MAE      : {hrc_score:.4f}  (LRU hit ratio curve MAE; 0=perfect)")
    print(f"    real HRC   : [{', '.join(f'{v:.2f}' for v in real_hrc[::5])}]  (1%..100% footprint)")
    print(f"    fake HRC   : [{', '.join(f'{v:.2f}' for v in fake_hrc[::5])}]")
    print(f"{'─'*40}")

    if baseline_path:
        print(f"\nBaseline   : {baseline_path}")
        b_ckpt = _load_checkpoint(baseline_path, device)
        b_real_flat = _sample_real(b_ckpt, trace_dir, fmt, n_samples)
        b_fake_flat = _sample_fake(b_ckpt, n_samples, device)
        b_cfg  = b_ckpt["config"]
        b_seqs_real = b_real_flat.reshape(-1, b_cfg.timestep, b_ckpt["prep"].num_cols)
        b_seqs_fake = b_fake_flat.reshape(-1, b_cfg.timestep, b_ckpt["prep"].num_cols)
        b_mmd  = mmd2_numpy(b_real_flat, b_fake_flat)
        b_prdc = compute_prdc_metrics(b_real_flat, b_fake_flat, k=k)
        b_dmd  = dmdgen(b_seqs_real, b_seqs_fake, r=4, n_batches=20,
                        batch_size=min(64, len(b_seqs_real) // 2))
        b_ac   = autocorr_score(b_seqs_real, b_seqs_fake, max_lag=5)
        b_E    = _load_encoder(b_ckpt, device)
        b_cfid = context_fid(b_seqs_real, b_seqs_fake, b_E, device) if b_E else None
        print(f"  MMD²         : {b_mmd:.5f}  (Δ {mmd - b_mmd:+.5f})")
        print(f"  β-recall     : {b_prdc['recall']:.4f}  (Δ {prdc['recall'] - b_prdc['recall']:+.4f})")
        print(f"  α-precision  : {b_prdc['precision']:.4f}  (Δ {prdc['precision'] - b_prdc['precision']:+.4f})")
        print(f"  DMD-GEN      : {b_dmd:.4f}  (Δ {dmd_score - b_dmd:+.4f})")
        print(f"  AutoCorr     : {b_ac:.4f}  (Δ {ac_score - b_ac:+.4f})")
        if cfid is not None and b_cfid is not None:
            print(f"  Context-FID  : {b_cfid:.2f}  (Δ {cfid - b_cfid:+.2f})")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate LLGAN checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--baseline",   default=None, help="Optional baseline checkpoint to compare")
    p.add_argument("--trace-dir",  required=True)
    p.add_argument("--fmt",        default="oracle_general")
    p.add_argument("--n-samples",  type=int, default=2000)
    p.add_argument("--k",          type=int, default=5,
                   help="Nearest-neighbour k for PRDC")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.trace_dir, args.fmt,
             args.n_samples, args.k, args.baseline)
