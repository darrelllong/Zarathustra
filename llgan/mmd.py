"""
Maximum Mean Discrepancy (MMD) with RBF kernel, plus multi-metric evaluation.

Lower MMD → generated distribution is closer to real.
LLGAN targets 0.015–0.05 per the paper.

evaluate_metrics() returns both MMD² and β-recall (PRDC) so that best.pt
is selected on a combined score that penalises mode collapse (low recall)
as well as distributional mismatch (high MMD²):

    combined = MMD² + 0.2 * (1 − recall)

Both metrics are computed in flat feature space (flattened windows) so they
are directly comparable across checkpoints regardless of architecture.
"""

import numpy as np
import torch


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


@torch.no_grad()
def evaluate_metrics(
    generator,
    val_data: torch.Tensor,
    n_samples: int,
    timestep: int,
    device: torch.device,
    recovery=None,
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Compute MMD², β-recall, and combined score for checkpoint selection.

    combined = MMD² + 0.2 * (1 − recall)

    Returns (mmd2, recall, combined).
    """
    generator.eval()
    if recovery is not None:
        recovery.eval()

    idx = torch.randperm(len(val_data))[:n_samples]
    real = val_data[idx].to(device)
    real_flat = real.view(n_samples, -1).cpu().numpy().astype(np.float32)

    z_global = torch.randn(n_samples, generator.noise_dim, device=device)
    z_local  = torch.randn(n_samples, timestep, generator.noise_dim, device=device)
    fake = generator(z_global, z_local)
    if recovery is not None:
        fake = recovery(fake)
    fake_flat = fake.view(n_samples, -1).cpu().numpy().astype(np.float32)

    mmd2_val  = mmd(torch.tensor(real_flat), torch.tensor(fake_flat))
    recall    = _recall_np(real_flat, fake_flat, k=k)
    combined  = mmd2_val + 0.2 * (1.0 - recall)
    return float(mmd2_val), float(recall), float(combined)
