"""
Evaluation metrics for LLGAN: MMD² and PRDC (Precision / Recall / Density / Coverage).

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
    real = _sample_real(ckpt, trace_dir, fmt, n_samples)
    fake = _sample_fake(ckpt, n_samples, device)

    mmd = mmd2_numpy(real, fake)
    prdc = compute_prdc_metrics(real, fake, k=k)

    print(f"\n{'─'*40}")
    print(f"  MMD²         : {mmd:.5f}")
    print(f"  α-precision  : {prdc['precision']:.4f}  (fake plausibility)")
    print(f"  β-recall     : {prdc['recall']:.4f}  (real coverage)  {'⚠ mode collapse' if prdc['recall'] < 0.3 else ''}")
    print(f"  density      : {prdc['density']:.4f}")
    print(f"  coverage     : {prdc['coverage']:.4f}")
    print(f"{'─'*40}")

    if baseline_path:
        print(f"\nBaseline   : {baseline_path}")
        b_ckpt = _load_checkpoint(baseline_path, device)
        b_real = _sample_real(b_ckpt, trace_dir, fmt, n_samples)
        b_fake = _sample_fake(b_ckpt, n_samples, device)
        b_mmd  = mmd2_numpy(b_real, b_fake)
        b_prdc = compute_prdc_metrics(b_real, b_fake, k=k)
        print(f"  MMD²         : {b_mmd:.5f}  (Δ {mmd - b_mmd:+.5f})")
        print(f"  β-recall     : {b_prdc['recall']:.4f}  (Δ {prdc['recall'] - b_prdc['recall']:+.4f})")
        print(f"  α-precision  : {b_prdc['precision']:.4f}  (Δ {prdc['precision'] - b_prdc['precision']:+.4f})")


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
