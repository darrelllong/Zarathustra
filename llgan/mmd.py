"""
Maximum Mean Discrepancy (MMD) with RBF kernel, plus multi-metric evaluation.

Why MMD²?
---------
MMD² is the squared maximum mean discrepancy: it measures how different two
distributions are by comparing their kernel-mean embeddings. A value of 0
means the generator is perfect; higher = worse.

We use a multi-scale RBF kernel (σ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}) rather than
a single bandwidth. A single σ is sensitive to the choice: too small → the
kernel saturates at 1 for close points and underestimates distance between
distributions; too large → distant points look similar. The multi-scale
average is more robust across the mixed-scale features of I/O traces (IATs
vary from microseconds to minutes; object sizes vary from 512B to 128MB).

Why evaluate_metrics() instead of evaluate_mmd()?
--------------------------------------------------
MMD² alone cannot distinguish two failure modes with similar MMD² scores:

  (a) Low MMD², high recall: the generator covers most of the real distribution
      — this is the desired outcome.
  (b) Low MMD², low recall: the generator has collapsed onto the most common
      mode (e.g., "small read at typical IAT") and achieves low distributional
      distance by nailing the mode, while missing burst events, writes, and
      large sequential reads entirely.

β-recall (from PRDC) measures the fraction of real samples that fall within
the k-NN ball of at least one generated sample — directly quantifying mode
coverage. A model with recall < 0.3 is typically mode-collapsed.

The combined score weights them:
    combined = MMD² + 0.2 × (1 − recall) + dmd_weight × DMD-GEN

At recall=0 (total collapse) the penalty is +0.2, which is larger than the
MMD² improvement we'd expect from memorising the mode. At recall=1 the penalty
vanishes. The weight 0.2 was chosen so that the recall term is a secondary
tiebreaker (doesn't override a large MMD² improvement) but blocks saving a
mode-collapsed checkpoint over a well-covered one.

DMD-GEN (Abba Haddou et al., NeurIPS 2025, arXiv 2412.11292) measures the
Grassmannian distance between dominant temporal modes of real and generated
sequences. It is stuck at ~0.71 across all versions and is not improved by
MMD/recall-only checkpoint selection. Including it in combined with a small
weight (e.g. 0.05) makes the selector prefer checkpoints with better temporal
dynamics as a tiebreaker, without overriding large MMD² improvements.
"""

from typing import Optional

import numpy as np
import torch
from scipy.linalg import subspace_angles


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian (RBF) kernel K(x, y) = exp(-||x-y||² / (2σ²))."""
    diff = x.unsqueeze(1) - y.unsqueeze(0)          # (n, m, d)
    sq_dist = (diff ** 2).sum(-1)                    # (n, m)
    return torch.exp(-sq_dist / (2 * sigma ** 2))


def mmd(
    real: torch.Tensor,
    fake: torch.Tensor,
    sigmas: tuple = (0.1, 0.5, 1.0, 2.0, 5.0),
) -> float:
    """
    Unbiased multi-scale RBF MMD².

    Args:
        real : (n, d) tensor  — samples from the real distribution
        fake : (m, d) tensor  — samples from the generated distribution
        sigmas: bandwidth values; using multiple gives a better estimate

    Returns scalar MMD² value (float).
    """
    real = real.float()
    fake = fake.float()

    mmd2 = 0.0
    for sigma in sigmas:
        kxx = _rbf_kernel(real, real, sigma).mean()
        kyy = _rbf_kernel(fake, fake, sigma).mean()
        kxy = _rbf_kernel(real, fake, sigma).mean()
        mmd2 += (kxx + kyy - 2 * kxy).item()

    return mmd2 / len(sigmas)


@torch.no_grad()
def evaluate_mmd(
    generator,
    val_data: torch.Tensor,
    n_samples: int,
    timestep: int,
    device: torch.device,
    recovery=None,
) -> float:
    """
    Draw n_samples windows from the generator and the validation set,
    flatten each window to a vector, compute MMD.

    recovery: optional Recovery module (latent AE mode). When provided,
              generated latents are decoded to feature space before MMD
              so the metric is comparable across architectures.
    """
    generator.eval()
    if recovery is not None:
        recovery.eval()

    # Real samples (always in feature space)
    idx = torch.randperm(len(val_data))[:n_samples]
    real = val_data[idx].to(device)                  # (n, timestep, num_cols)
    real_flat = real.view(n_samples, -1)

    # Fake samples — decode latents to feature space when recovery is present
    z_global = torch.randn(n_samples, generator.noise_dim, device=device)
    z_local  = torch.randn(n_samples, timestep, generator.noise_dim, device=device)
    fake = generator(z_global, z_local)
    if recovery is not None:
        fake = recovery(fake)
    fake_flat = fake.view(n_samples, -1)

    return mmd(real_flat.cpu(), fake_flat.cpu())


def _pairwise_sq_dist_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,d),(m,d) → (n,m) squared L2 distances."""
    a2 = (a ** 2).sum(1, keepdims=True)
    b2 = (b ** 2).sum(1, keepdims=True)
    return np.maximum(a2 + b2.T - 2 * a @ b.T, 0.0)


def _recall_np(real: np.ndarray, fake: np.ndarray, k: int = 5) -> float:
    """
    β-recall: fraction of real samples whose k-NN ball contains at least one
    fake sample.  Pure numpy — no external prdc package required.

    Lower recall = mode collapse (generator not covering real data modes).
    """
    rr = _pairwise_sq_dist_np(real, real)
    # k-th NN radius (exclude self → k+1 th smallest)
    r_real = np.sqrt(np.partition(rr, min(k + 1, rr.shape[1] - 1), axis=1)[:, min(k + 1, rr.shape[1] - 1)])
    rf = _pairwise_sq_dist_np(real, fake)   # (n_real, n_fake)
    # real[i] is "covered" if any fake falls within real[i]'s k-NN ball
    covered = (np.sqrt(rf) <= r_real[:, None]).any(axis=1)
    return float(covered.mean())


# ---------------------------------------------------------------------------
# DMD-GEN: Grassmannian distance between dominant temporal modes
# (Abba Haddou et al., NeurIPS 2025, arXiv 2412.11292)
# Implemented here (not in eval.py) so training-time checkpoint selection
# can include it without importing eval.py (which imports train.py artifacts).
# eval.py imports dmdgen from here to avoid duplication.
# ---------------------------------------------------------------------------

def _dmd_subspace(X: np.ndarray, r: int) -> np.ndarray:
    """
    Truncated DMD on a data matrix.

    X: (T, N) where T = timesteps, N = batch_size * num_features
    r: number of DMD modes to retain

    Returns Q: (N, r_eff) orthonormal basis for the DMD subspace (Grassmann point).
    """
    X1 = X[:-1, :]
    X2 = X[1:, :]
    r_eff = min(r, min(X1.shape) - 1)
    U, s, Vh = np.linalg.svd(X1, full_matrices=False)
    U_r, S_r, Vh_r = U[:, :r_eff], s[:r_eff], Vh[:r_eff, :]
    # pinv of diag(S_r) — zeroes out singular values below numpy's default tol,
    # avoiding 1/0 blow-up when X1 is rank-deficient (Gemini R2 #3).
    S_r_inv = np.linalg.pinv(np.diag(S_r))
    A_tilde = U_r.T @ X2 @ Vh_r.T @ S_r_inv
    _, W = np.linalg.eig(A_tilde)
    Phi = (X2 @ Vh_r.T @ S_r_inv @ W).real
    Q, _ = np.linalg.qr(Phi)
    return Q[:, :r_eff]


def dmdgen(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    r: int = 4,
    n_batches: int = 10,
    batch_size: int = 64,
    rng: np.random.Generator = None,
) -> float:
    """
    DMD-GEN metric: mean Grassmannian distance between real and generated
    DMD subspaces over n_batches random mini-batches.

    real_seqs / fake_seqs: (N, T, d) arrays.
    Returns scalar ≥ 0; lower = dynamics closer to real. nan if degenerate.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    T, d = real_seqs.shape[1], real_seqs.shape[2]
    angle_distances = []
    for _ in range(n_batches):
        ri = rng.choice(len(real_seqs), min(batch_size, len(real_seqs)), replace=False)
        fi = rng.choice(len(fake_seqs), min(batch_size, len(fake_seqs)), replace=False)
        Xr = real_seqs[ri].transpose(1, 0, 2).reshape(T, -1)
        Xf = fake_seqs[fi].transpose(1, 0, 2).reshape(T, -1)
        try:
            Ur = _dmd_subspace(Xr, r)
            Uf = _dmd_subspace(Xf, r)
            r_eff = min(Ur.shape[1], Uf.shape[1])
            angles = subspace_angles(Ur[:, :r_eff], Uf[:, :r_eff])
            angle_distances.append(float(np.mean(angles)))
        except np.linalg.LinAlgError:
            pass
    return float(np.mean(angle_distances)) if angle_distances else float("nan")


def spectral_divergence(
    real_seqs: np.ndarray,
    fake_seqs: np.ndarray,
    n_pad: int = 32,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Fourier spectral divergence between real and generated sequences.

    Computes the per-feature power spectral density (PSD) via zero-padded FFT,
    then measures the mean L1 distance between the averaged real and fake PSDs.

    real_seqs / fake_seqs: (N, T, d) arrays, normalized to [-1, 1].
    n_pad: FFT length (zero-pad T to this for better frequency resolution).

    Returns:
      - score: mean absolute PSD difference across features (0 = perfect)
      - real_psd: (n_freq, d) mean PSD per feature for real data
      - fake_psd: (n_freq, d) mean PSD per feature for fake data
    """
    T, d = real_seqs.shape[1], real_seqs.shape[2]
    n_fft = max(n_pad, T)
    n_freq = n_fft // 2 + 1

    # Hann window to reduce spectral leakage on short (T=12) windows
    window = np.hanning(T).astype(np.float32)

    def _psd(seqs):
        # seqs: (N, T, d) → apply window, FFT, power spectrum
        windowed = seqs * window[np.newaxis, :, np.newaxis]
        # rfft along time axis with zero-padding
        F = np.fft.rfft(windowed, n=n_fft, axis=1)  # (N, n_freq, d)
        power = np.abs(F) ** 2  # (N, n_freq, d)
        # Average across windows, normalize by max to get relative PSD
        mean_psd = power.mean(axis=0)  # (n_freq, d)
        # Normalize each feature's PSD to sum to 1 (distribution comparison)
        totals = mean_psd.sum(axis=0, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        return mean_psd / totals

    real_psd = _psd(real_seqs)
    fake_psd = _psd(fake_seqs)

    # Mean absolute difference across frequency bins and features
    score = float(np.abs(real_psd - fake_psd).mean())

    return score, real_psd, fake_psd


@torch.no_grad()
def evaluate_metrics(
    generator,
    val_data: torch.Tensor,
    n_samples: int,
    timestep: int,
    device: torch.device,
    recovery=None,
    k: int = 5,
    dmd_weight: float = 0.0,
    cond_pool: "Optional[torch.Tensor]" = None,
) -> tuple[float, float, float]:
    """
    Compute MMD², β-recall, and combined score for checkpoint selection.

    combined = MMD² + 0.2*(1−recall) + dmd_weight*DMD-GEN

    dmd_weight=0.0 (default) skips DMD-GEN computation (backward compatible).
    Use 0.05 to add a temporal-dynamics tiebreaker without overriding MMD²
    improvements (reviewer rec: "do not let best.pt be chosen without a
    temporal law penalty").

    cond_pool: optional (N, cond_dim) tensor of pre-computed conditioning
        vectors to sample from for generation (e.g., char-file file-level
        stats).  When provided, overrides compute_window_descriptors so that
        EMA eval uses the same conditioning distribution as training.  Each
        generated sample draws a random row from cond_pool.

    Returns (mmd2, recall, combined).
    """
    generator.eval()
    if recovery is not None:
        recovery.eval()

    try:
        idx = torch.randperm(len(val_data))[:n_samples]
        real = val_data[idx].to(device)
        real_flat = real.view(n_samples, -1).cpu().numpy().astype(np.float32)

        if getattr(generator, 'cond_dim', 0) > 0:
            if cond_pool is not None:
                cidx = torch.randint(len(cond_pool), (n_samples,))
                cond = cond_pool[cidx].to(device)
            else:
                from dataset import compute_window_descriptors
                cond = compute_window_descriptors(real)
            # Unify z_global path with training (Round 5 TODO):
            # Apply cond_encoder (deterministic μ), regime_sampler, and
            # gmm_prior — the same stack _make_z_global uses at train time.
            # Previously this was raw torch.cat([cond, noise]), skipping all
            # three transformations and causing 30-75% train→eval gaps.
            if getattr(generator, 'cond_encoder', None) is not None:
                cond, _ = generator.cond_encoder(cond, training=False)
            if getattr(generator, 'regime_sampler', None) is not None:
                cond = generator.regime_sampler(cond)
            noise = generator.sample_noise(n_samples, device, cond=cond)
            z_global = torch.cat([cond, noise], dim=1)
        else:
            z_global = torch.randn(n_samples, generator.noise_dim, device=device)
        z_local  = torch.randn(n_samples, timestep, generator.noise_dim, device=device)
        fake = generator(z_global, z_local)
        if recovery is not None:
            fake = recovery(fake)
        fake_flat = fake.view(n_samples, -1).cpu().numpy().astype(np.float32)

        mmd2_val  = mmd(torch.tensor(real_flat), torch.tensor(fake_flat))
        recall    = _recall_np(real_flat, fake_flat, k=k)
        combined  = mmd2_val + 0.2 * (1.0 - recall)

        if dmd_weight > 0.0:
            num_cols = real_flat.shape[1] // timestep
            real_seqs = real_flat.reshape(-1, timestep, num_cols)
            fake_seqs = fake_flat.reshape(-1, timestep, num_cols)
            # Use 5 batches (vs eval.py's 20) for speed during training.
            dmd_val = dmdgen(real_seqs, fake_seqs, r=4, n_batches=5,
                             batch_size=min(64, n_samples // 4))
            if not np.isnan(dmd_val):
                combined += dmd_weight * dmd_val

        return float(mmd2_val), float(recall), float(combined)
    finally:
        generator.train()
        if recovery is not None:
            recovery.train()
