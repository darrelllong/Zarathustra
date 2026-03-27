"""
Maximum Mean Discrepancy (MMD) with RBF kernel.

Lower MMD → generated distribution is closer to real.
LLGAN targets 0.015–0.05 per the paper.
"""

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
) -> float:
    """
    Draw n_samples windows from the generator and the validation set,
    flatten each window to a vector, compute MMD.
    """
    generator.eval()

    # Real samples
    idx = torch.randperm(len(val_data))[:n_samples]
    real = val_data[idx].to(device)                  # (n, timestep, num_cols)
    real_flat = real.view(n_samples, -1)

    # Fake samples
    z_global = torch.randn(n_samples, generator.noise_dim, device=device)
    z_local  = torch.randn(n_samples, timestep, generator.noise_dim, device=device)
    fake = generator(z_global, z_local)
    fake_flat = fake.view(n_samples, -1)

    return mmd(real_flat.cpu(), fake_flat.cpu())
