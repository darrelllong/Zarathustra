"""
Evaluation metrics for LLGAN: MMD², PRDC, and DMD-GEN.

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
from scipy.linalg import subspace_angles

# ---------------------------------------------------------------------------
# DMD-GEN: Grassmannian Geometry + Dynamic Mode Decomposition metric
# (Abba Haddou et al., NeurIPS 2025, arXiv 2412.11292)
# Code not yet released — implemented from paper description.
# ---------------------------------------------------------------------------

def _dmd_subspace(X: np.ndarray, r: int) -> np.ndarray:
    """
    Truncated DMD on a batch of sequences.

    X: (T, N) data matrix where T = timesteps, N = batch_size * num_features
       (sequences are stacked column-wise).
    r: number of DMD modes to retain.

    Returns U: (N, r) orthonormal basis for the r-dimensional DMD subspace,
    a point on the Grassmann manifold Gr(r, N).
    """
    X1 = X[:-1, :]   # (T-1, N)
    X2 = X[1:, :]    # (T-1, N)

    # Thin SVD of X1 → truncate to r modes
    r_eff = min(r, min(X1.shape) - 1)
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    U_r  = U[:, :r_eff]           # (T-1, r)
    S_r  = s[:r_eff]
    Vh_r = Vh[:r_eff, :]          # (r, N)

    # Reduced DMD operator Ã = U_r^T X2 Vh_r^T S_r^{-1}
    A_tilde = U_r.T @ X2 @ Vh_r.T @ np.diag(1.0 / S_r)   # (r, r)

    # Eigendecomposition of Ã
    eigvals, W = np.linalg.eig(A_tilde)   # W: (r, r) right eigenvectors

    # Exact DMD modes Φ = X2 Vh_r^T S_r^{-1} W  (shape: N, r)
    Phi = (X2 @ Vh_r.T @ np.diag(1.0 / S_r) @ W).real   # (N, r)

    # Orthonormalise columns via QR to get a Grassmann point
    Q, _ = np.linalg.qr(Phi)
    return Q[:, :r_eff]   # (N, r_eff)


def dmdgen(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    r: int = 4,
    n_batches: int = 10,
    batch_size: int = 64,
    rng: np.random.Generator = None,
) -> float:
    """
    DMD-GEN metric: 1-Wasserstein distance between principal-angle distributions
    of real and generated sequence batches on the Grassmann manifold.

    real_seqs / fake_seqs: (N, T, d) arrays of sequences.
    r      : DMD rank (number of modes to retain).
    n_batches: number of mini-batches; average over them for stability.

    Returns a scalar ≥ 0; lower = generated dynamics closer to real dynamics.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = min(len(real_seqs), len(fake_seqs))
    T, d = real_seqs.shape[1], real_seqs.shape[2]

    angle_distances = []
    for _ in range(n_batches):
        ri = rng.choice(len(real_seqs), batch_size, replace=False)
        fi = rng.choice(len(fake_seqs), batch_size, replace=False)

        # Stack batch into (T, batch*d) matrix for DMD
        Xr = real_seqs[ri].transpose(1, 0, 2).reshape(T, -1)   # (T, B*d)
        Xf = fake_seqs[fi].transpose(1, 0, 2).reshape(T, -1)   # (T, B*d)

        try:
            Ur = _dmd_subspace(Xr, r)   # (B*d, r_eff)
            Uf = _dmd_subspace(Xf, r)

            # Principal angles between subspaces (in [0, π/2])
            r_eff = min(Ur.shape[1], Uf.shape[1])
            angles = subspace_angles(Ur[:, :r_eff], Uf[:, :r_eff])
            angle_distances.append(float(np.mean(angles)))
        except np.linalg.LinAlgError:
            pass   # degenerate batch — skip

    if not angle_distances:
        return float("nan")

    # Mean Grassmannian distance across batches
    return float(np.mean(angle_distances))


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


def _sample_fake(ckpt, n_samples: int, device) -> np.ndarray:
    from model import Generator, Recovery
    cfg  = ckpt["config"]
    prep = ckpt["prep"]
    latent_ae = "R" in ckpt
    latent_dim = getattr(cfg, "latent_dim", 0) if latent_ae else None

    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=latent_dim).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    R = None
    if latent_ae:
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    with torch.no_grad():
        z_g = torch.randn(n_samples, cfg.noise_dim, device=device)
        z_l = torch.randn(n_samples, cfg.timestep, cfg.noise_dim, device=device)
        fake = G(z_g, z_l)
        if R is not None:
            fake = R(fake)
    return fake.cpu().numpy().reshape(n_samples, -1)


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
    fake_flat = _sample_fake(ckpt, n_samples, device)

    cfg  = ckpt["config"]
    timestep  = cfg.timestep
    num_cols  = ckpt["prep"].num_cols
    # Reshape flat (N, T*d) → (N, T, d) for sequence-aware metrics
    real_seqs = real_flat.reshape(-1, timestep, num_cols)
    fake_seqs = fake_flat.reshape(-1, timestep, num_cols)

    mmd  = mmd2_numpy(real_flat, fake_flat)
    prdc = compute_prdc_metrics(real_flat, fake_flat, k=k)

    # DMD-GEN: temporal dynamics distance on Grassmann manifold
    dmd_score = dmdgen(real_seqs, fake_seqs, r=4, n_batches=20,
                       batch_size=min(64, len(real_seqs) // 2))

    print(f"\n{'─'*40}")
    print(f"  MMD²         : {mmd:.5f}")
    print(f"  α-precision  : {prdc['precision']:.4f}  (fake plausibility)")
    print(f"  β-recall     : {prdc['recall']:.4f}  (real coverage)  {'⚠ mode collapse' if prdc['recall'] < 0.3 else ''}")
    print(f"  density      : {prdc['density']:.4f}")
    print(f"  coverage     : {prdc['coverage']:.4f}")
    print(f"  DMD-GEN      : {dmd_score:.4f}  (temporal dynamics; 0=perfect, >0.3=poor)")
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
        print(f"  MMD²         : {b_mmd:.5f}  (Δ {mmd - b_mmd:+.5f})")
        print(f"  β-recall     : {b_prdc['recall']:.4f}  (Δ {prdc['recall'] - b_prdc['recall']:+.4f})")
        print(f"  α-precision  : {b_prdc['precision']:.4f}  (Δ {prdc['precision'] - b_prdc['precision']:+.4f})")
        print(f"  DMD-GEN      : {b_dmd:.4f}  (Δ {dmd_score - b_dmd:+.4f})")


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
