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

from mmd import dmdgen, _dmd_subspace, spectral_divergence  # DMD-GEN + Fourier live in mmd.py

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


def _seqs_to_obj_ids(seqs: np.ndarray, prep=None) -> np.ndarray:
    """Reconstruct integer object IDs from (N, T, d) windows.

    Uses the preprocessor's inverse transform to get real-scale object IDs.
    Each window is inverse-transformed independently (obj_id base offset doesn't
    matter for within-window LRU simulation; across windows we add large offsets
    to prevent cross-window collisions).

    Args:
        seqs: (N, T, d) numpy array of windows.
        prep: TracePreprocessor instance with scaler params (from checkpoint).
    """
    N, T, d = seqs.shape

    if prep is not None and hasattr(prep, 'inverse_transform'):
        all_ids = []
        for i in range(N):
            window = seqs[i]  # (T, d)
            try:
                df_inv = prep.inverse_transform(window)
                if 'obj_id' in df_inv.columns:
                    ids = df_inv['obj_id'].values.astype(np.int64)
                    # Offset each window by a large amount to prevent cross-window
                    # ID collisions (simulating separate streams)
                    all_ids.append(ids + i * 10_000_000)
                    continue
            except Exception:
                pass
            all_ids.append(np.arange(T, dtype=np.int64) + i * 10_000_000)
        return np.concatenate(all_ids)

    # Fallback without prep: each access is unique (worst case)
    return np.arange(N * T, dtype=np.int64)


def hrc_mae(real_seqs: np.ndarray, fake_seqs: np.ndarray,
            prep=None, n_points: int = 20) -> tuple:
    """Compute HRC MAE between real and generated traces.

    Simulates LRU caches per-window (T=12 timesteps per window) and averages
    hit ratios across all windows at each cache size.  This avoids the need to
    stitch windows into long streams while still measuring cache locality.

    Args:
        real_seqs, fake_seqs: (N, T, d) windows (normalised).
        prep: TracePreprocessor for inverse transform (from checkpoint).
        n_points: number of cache-size sample points on the HRC.

    Returns:
        (mae, real_hrc, fake_hrc): scalar MAE and the two HRC arrays.
    """
    def _per_window_hrc(seqs, prep, n_pts):
        """Compute average LRU hit ratio curve across windows."""
        N, T, d = seqs.shape
        # Determine cache sizes from 1 to T (window length)
        sizes = np.unique(np.linspace(1, T, n_pts).astype(int))
        n_pts_actual = len(sizes)

        hit_ratios = np.zeros(n_pts_actual)
        for i in range(N):
            window = seqs[i]  # (T, d)
            # Get real-scale obj_ids for this window
            if prep is not None and hasattr(prep, 'inverse_transform'):
                try:
                    df_inv = prep.inverse_transform(window)
                    if 'obj_id' in df_inv.columns:
                        ids = df_inv['obj_id'].values.astype(np.int64)
                    else:
                        ids = np.arange(T)
                except Exception:
                    ids = np.arange(T)
            else:
                ids = np.arange(T)

            for j, cs in enumerate(sizes):
                hit_ratios[j] += _lru_hit_ratio(ids, int(cs))
        hit_ratios /= N

        return hit_ratios  # return actual points only, no padding

    real_hrc = _per_window_hrc(real_seqs, prep, n_points)
    fake_hrc = _per_window_hrc(fake_seqs, prep, n_points)

    # MAE over actual cache sizes only (no artificial padding)
    n_actual = min(len(real_hrc), len(fake_hrc))
    mae = float(np.abs(real_hrc[:n_actual] - fake_hrc[:n_actual]).mean())
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
    """Compute PRDC via k-NN manifold estimation (no external library).

    Matches the official definitions from Naeem et al. 2020 as implemented in
    clovaai/generative-evaluation-prdc (Round 15 peer review P1 fix):
      - precision: fraction of fake inside any real k-NN ball (uses r_real)
      - recall:    fraction of real inside any fake k-NN ball (uses r_fake)
      - density:   mean number of real balls containing each fake, scaled by 1/k
      - coverage:  fraction of real whose nearest fake lies inside real's own ball
    """
    rr = _pairwise_sq_dist(real, real)
    ff = _pairwise_sq_dist(fake, fake)
    rf_sq = _pairwise_sq_dist(real, fake)   # (n_real, n_fake)
    rf = np.sqrt(rf_sq)

    # k-th nearest neighbour radius for each real / fake point (exclude self → k+1)
    r_real = np.sqrt(np.partition(rr, k + 1, axis=1)[:, k + 1])  # (n_real,)
    r_fake = np.sqrt(np.partition(ff, k + 1, axis=1)[:, k + 1])  # (n_fake,)

    # precision: fake[j] inside real[i]'s ball for some i
    in_real_ball = (rf < r_real[:, None])             # (n_real, n_fake)
    precision = in_real_ball.any(axis=0).mean()

    # recall: real[i] inside fake[j]'s ball for some j  (uses r_fake, not r_real)
    in_fake_ball = (rf < r_fake[None, :])             # (n_real, n_fake)
    recall = in_fake_ball.any(axis=1).mean()

    # density: average over fakes of #real-balls containing it, divided by k
    density = (1.0 / float(k)) * in_real_ball.sum(axis=0).mean()

    # coverage: fraction of real whose nearest fake lies within real's own ball
    coverage = (rf.min(axis=1) < r_real).mean()

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
                 real_windows: np.ndarray = None,
                 cond_noise_scale: float = 0.0) -> np.ndarray:
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
    # Prefer EMA weights: smoother, less oscillated, consistently produces
    # better samples than the instantaneous live weights at any given epoch.
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        # Detect mixed-type-recovery from checkpoint keys
        r_keys = ckpt["R"].keys()
        binary_cols = None
        if any(k.startswith("fc_cont") for k in r_keys):
            # Reconstruct binary_cols from saved column names
            binary_cols = [i for i, col in enumerate(prep.col_names)
                           if col.lower() in {"opcode", "type", "rw", "op"}
                           or col.endswith("_reuse")]
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols,
                     binary_cols=binary_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    # Build conditioning pool: prefer char-file (stable per-file stats) over
    # noisy window-level descriptors (which may fail on auto-dropped columns).
    cond_pool = None
    if cond_dim > 0:
        char_file = getattr(cfg, "char_file", "")
        if char_file:
            from dataset import load_file_characterizations
            char_lookup = load_file_characterizations(char_file, cond_dim)
            if char_lookup:
                cond_pool = torch.stack(list(char_lookup.values())).to(device)

        if cond_pool is None and real_windows is None:
            raise ValueError("real_windows required when cond_dim > 0 and no char_file")

    chunks = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            if cond_dim > 0:
                if cond_pool is not None:
                    idx = torch.randint(0, len(cond_pool), (n,))
                    cond = cond_pool[idx]
                else:
                    idx = np.random.choice(len(real_windows), n, replace=True)
                    rw = torch.tensor(real_windows[idx], dtype=torch.float32,
                                      device=device)
                    cond = compute_window_descriptors(
                        rw, col_names=getattr(prep, "col_names", None),
                        cond_dim=cond_dim)
                # Round 5 fix: optionally add scaled stochastic noise to
                # conditioning at eval to match training distribution.
                if getattr(G, 'cond_encoder', None) is not None:
                    if cond_noise_scale > 0:
                        mu = G.cond_encoder.mu_net(cond)
                        logvar = G.cond_encoder.logvar_net(cond).clamp(-10, 2)
                        std = (0.5 * logvar).exp()
                        cond = mu + cond_noise_scale * std * torch.randn_like(mu)
                    else:
                        cond, _ = G.cond_encoder(cond, training=False)
                if getattr(G, 'regime_sampler', None) is not None:
                    cond = G.regime_sampler(cond)
                noise = G.sample_noise(n, device, cond=cond)
                z_g = torch.cat([cond, noise], dim=1)
            else:
                z_g = torch.randn(n, cfg.noise_dim, device=device)
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


def _sample_real(ckpt, trace_dir: str, fmt: str, n_samples: int,
                 real_seed: int = None) -> np.ndarray:
    """Sample real windows for evaluation.

    When real_seed is provided, file selection and window subsampling are
    deterministic, so the "real bundle" is frozen across runs. This isolates
    fake-sample variance from benchmark variance (Round 15 peer review P1).
    Use the same seed across evals intended to be directly compared.
    """
    import random
    sys.path.insert(0, ".")
    from train import _collect_files, _load_epoch_dataset
    prep = ckpt["prep"]
    cfg  = ckpt["config"]

    all_files = _collect_files(trace_dir, fmt)
    if not all_files:
        raise RuntimeError(f"No files found in {trace_dir}")

    if real_seed is not None:
        file_rng = random.Random(real_seed)
        win_rng  = np.random.RandomState(real_seed)
        files = file_rng.sample(sorted(all_files), min(4, len(all_files)))
    else:
        files = random.sample(all_files, min(4, len(all_files)))
        win_rng = np.random

    ds, _ = _load_epoch_dataset(files, fmt, 15000, prep, cfg.timestep)
    if ds is None or len(ds) == 0:
        raise RuntimeError("Could not load real data for evaluation.")
    idx = win_rng.choice(len(ds), min(n_samples, len(ds)), replace=False)
    windows = np.stack([ds[i].numpy() for i in idx])
    return windows.reshape(len(idx), -1)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(checkpoint_path: str, trace_dir: str, fmt: str,
             n_samples: int = 2000, k: int = 5,
             baseline_path: str = None, args=None) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Checkpoint : {checkpoint_path}")

    ckpt = _load_checkpoint(checkpoint_path, device)
    print(f"Epoch      : {ckpt['epoch']+1}")
    if "mmd" in ckpt:
        print(f"Saved MMD² : {ckpt['mmd']:.5f}")

    real_seed = getattr(args, 'eval_real_seed', None) if args else None
    fake_seed = getattr(args, 'eval_fake_seed', None) if args else None
    bundle_note = f" (frozen bundle seed={real_seed})" if real_seed is not None else ""
    if fake_seed is not None:
        bundle_note += f" (fake seed={fake_seed})"
    print(f"\nSampling {n_samples} real and fake windows{bundle_note} …")
    real_flat = _sample_real(ckpt, trace_dir, fmt, n_samples, real_seed=real_seed)

    if fake_seed is not None:
        import random as _random
        _random.seed(fake_seed)
        np.random.seed(fake_seed)
        torch.manual_seed(fake_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fake_seed)

    cfg  = ckpt["config"]
    timestep  = cfg.timestep
    num_cols  = ckpt["prep"].num_cols
    # Reshape flat (N, T*d) → (N, T, d) for sequence-aware metrics
    real_seqs = real_flat.reshape(-1, timestep, num_cols)

    # Pass real windows to _sample_fake so conditioning uses real descriptors
    cond_dim = getattr(cfg, "cond_dim", 0)
    fake_flat = _sample_fake(ckpt, n_samples, device,
                             real_windows=real_seqs if cond_dim > 0 else None,
                             cond_noise_scale=getattr(args, 'cond_noise_scale', 0.0) if args else 0.0)
    fake_seqs = fake_flat.reshape(-1, timestep, num_cols)

    mmd  = mmd2_numpy(real_flat, fake_flat)
    prdc = compute_prdc_metrics(real_flat, fake_flat, k=k)

    # Temporal structure metrics
    dmd_score = dmdgen(real_seqs, fake_seqs, r=4, n_batches=20,
                       batch_size=min(64, len(real_seqs) // 2))
    ac_score = autocorr_score(real_seqs, fake_seqs, max_lag=5)

    # Fourier spectral divergence: per-feature PSD comparison
    spectral_score, real_psd, fake_psd = spectral_divergence(real_seqs, fake_seqs)

    # Context-FID: Fréchet distance in the Encoder's latent space
    E = _load_encoder(ckpt, device)
    cfid = context_fid(real_seqs, fake_seqs, E, device) if E is not None else None

    # Reuse rate: fraction of accesses where obj_id_reuse > 0 (same object as prior).
    # Dynamically resolves the reuse column index from the preprocessor's col_names
    # to handle column dropout (e.g. tenant column auto-dropped for single-tenant traces).
    prep = ckpt["prep"]
    def _reuse_rate(seqs: np.ndarray) -> float:
        d = seqs.shape[2]
        # Resolve reuse column dynamically
        reuse_col = None
        if hasattr(prep, 'col_names'):
            if 'obj_id_reuse' in prep.col_names:
                reuse_col = prep.col_names.index('obj_id_reuse')
        if reuse_col is None and d >= 4:
            reuse_col = 3  # fallback for legacy layout
        if reuse_col is not None and reuse_col < d:
            reuse = seqs[:, 1:, reuse_col]   # skip first timestep (always miss)
            return float((reuse > 0).mean())
        # Fallback for legacy layout: check exact column-1 matches
        obj = seqs[:, :, 1]
        hits = 0
        for t in range(1, obj.shape[1]):
            hits += (np.abs(obj[:, :t] - obj[:, t:t+1]) < 1e-6).any(axis=1).sum()
        return hits / (obj.shape[0] * (obj.shape[1] - 1))

    real_reuse = _reuse_rate(real_seqs)
    fake_reuse = _reuse_rate(fake_seqs)

    # HRC: cache fidelity via LRU hit ratio curve comparison.
    # NOTE: With T=12 windows, per-window HRC is limited.  For full cache
    # fidelity evaluation, use generate.py → compare.py on long traces.
    hrc_score, real_hrc, fake_hrc = hrc_mae(real_seqs, fake_seqs,
                                             prep=ckpt.get("prep"),
                                             n_points=20)

    print(f"\n{'─'*40}")
    print(f"  MMD²         : {mmd:.5f}")
    print(f"  α-precision  : {prdc['precision']:.4f}  (fake plausibility)")
    print(f"  β-recall     : {prdc['recall']:.4f}  (real coverage)  {'⚠ mode collapse' if prdc['recall'] < 0.3 else ''}")
    print(f"  density      : {prdc['density']:.4f}")
    print(f"  coverage     : {prdc['coverage']:.4f}")
    print(f"  DMD-GEN      : {dmd_score:.4f}  (temporal dynamics; 0=perfect, >0.3=poor)")
    print(f"  AutoCorr     : {ac_score:.4f}  (lag-1..5 ACF diff; 0=perfect)")
    print(f"  Spectral     : {spectral_score:.4f}  (Fourier PSD divergence; 0=perfect)")
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
        b_cond_dim = getattr(b_ckpt["config"], "cond_dim", 0)
        b_real_seqs_for_cond = (b_real_flat.reshape(-1, b_ckpt["config"].timestep,
                                b_ckpt["prep"].num_cols) if b_cond_dim > 0 else None)
        b_fake_flat = _sample_fake(b_ckpt, n_samples, device,
                                   real_windows=b_real_seqs_for_cond)
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
    p.add_argument("--cond-noise-scale", type=float, default=0.0,
                   help="Scale for stochastic conditioning noise at eval (0=deterministic)")
    p.add_argument("--eval-real-seed", type=int, default=None,
                   help="Seed for deterministic real-bundle selection (file list + "
                        "window subsample). If unset, sampling is random (legacy "
                        "behaviour). Use a fixed seed across runs to isolate "
                        "fake-sample variance from benchmark variance. Evals using "
                        "the same seed on the same trace_dir are directly comparable.")
    p.add_argument("--eval-fake-seed", type=int, default=None,
                   help="Seed for deterministic fake-sample generation (generator "
                        "noise, cond pool indices, recovery RNG). Required alongside "
                        "--eval-real-seed for fully deterministic comparison across "
                        "checkpoints — without it, ★ varies ~±0.01 across reruns of "
                        "the same weights purely from fake-sample draw noise.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, args.trace_dir, args.fmt,
             args.n_samples, args.k, args.baseline, args=args)
