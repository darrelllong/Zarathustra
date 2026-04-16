"""
Latent diffusion stage for hybrid pivot (IDEAS.md idea #22 — design + module).

Status
------
DESIGN+IMPL — Stage-1 (latent denoising) only. The full three-stage
hybrid (diffusion → supervisor → critic) is documented in this
module's docstring but Stage 2/3 reuse the existing Supervisor and
Critic from model.py, no new module needed. NOT YET WIRED into
train.py — separate commit will add a new training mode
``--training-mode hybrid`` that swaps the joint-GAN curriculum for
the three-stage pipeline. Backward compatible by construction:
default training mode is the existing GAN.

Motivation
----------
IDEAS.md #22: "the high-ceiling pivot if the team decides it has
learned most of what it can from pure GAN structural tweaks." The
current four-phase TimeGAN/SeriesGAN curriculum couples generation
and discrimination tightly. Hybrid time-series work (AVATAR, TIMED,
DiTTO) suggests decoupling: let a denoising model handle global
shape, an autoregressive supervisor handle local dependencies, and a
critic judge realism.

Three-stage hybrid pipeline
---------------------------
**Stage 1 — Latent diffusion (this module)**
  Train a 1D U-Net-lite to denoise corrupted latent sequences in the
  smooth latent space already learned by E (Encoder). Sampling
  produces a coarse, globally-correct latent trajectory.

**Stage 2 — Autoregressive refinement** (reuses existing Supervisor)
  Given Stage-1's coarse latent, run S as a teacher-forced refiner:
  S(z_t) → z_{t+1}_corrected. Trained with MSE against the original
  real latent. This fixes local burst structure that diffusion may
  smooth out.

**Stage 3 — Adversarial polish** (reuses existing Critic)
  Pass refined latent through R (Recovery) and through C (Critic) for
  one epoch of WGAN polish. This is the smallest dose of the existing
  GAN training, applied at the end rather than throughout.

Why this ordering?
  Diffusion gives globally consistent structure; AR supervision fixes
  locality; adversarial gives the final realism push. Each tool used
  for what it's best at — the explicit decoupling means failure
  modes are diagnosable per-stage rather than entangled.

MVE — what this commit ships
-----------------------------
Only Stage 1's denoiser. Architecture: 4-block 1D U-Net-lite operating
on (B, T, latent_dim) with sinusoidal step embedding and FiLM
conditioning on the workload cond vector.

Why a *latent* diffusion (not feature-space)?
  Feature space has 5-13 mixed-type columns spanning 10 decades of
  dynamic range. Latent space is smooth, 8-32 dim, normalised — a
  much friendlier diffusion target. Same decoupling rationale that
  motivates AVATAR's choice.

Cosine noise schedule:
  Standard improved-DDPM cosine schedule (Nichol & Dhariwal 2021),
  500 steps default. Faster than linear schedule on short sequences.

CLI flags (NOT YET WIRED):
  --training-mode {gan, hybrid}     default gan
  --hybrid-diffusion-steps INT      default 500
  --hybrid-stage1-epochs INT        default 100  (Stage 1 budget)
  --hybrid-stage2-epochs INT        default 50   (Stage 2 budget)
  --hybrid-stage3-epochs INT        default 20   (Stage 3 budget)
  --hybrid-channels INT             default 64

Failure modes monitored:
  - Diffusion training loss not converging → channels too small or
    schedule wrong. Check val NLL.
  - Stage 2 supervisor erasing diffusion structure → MSE schedule
    needs warmup. Diagnostic: residual norm per Stage-2 epoch.
  - Stage 3 critic destabilising → reduce WGAN learning rate or use
    fewer epochs.

Cost estimate
-------------
Stage 1 (100 epochs, 12 files/epoch): ~5h on GB10 at H=64.
Stage 2 (50 epochs): ~2h (existing supervisor cost).
Stage 3 (20 epochs): ~1h.
Total: ~8h vs current ~6h GAN training. ~30% more compute, much
clearer per-stage diagnostics.

References
----------
  Kim et al., DiTTO (arXiv 2025) — diffusion for storage traces.
  EskandariNasab et al., AVATAR (SDM 2025) — adversarial AE + AR.
  EskandariNasab et al., TIMED (ICDM 2025) — full diffusion + AR + adv.
  Nichol & Dhariwal, Improved DDPM (ICML 2021) — cosine schedule.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cosine noise schedule (Improved-DDPM)
# ---------------------------------------------------------------------------

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine β schedule. Returns (n_steps,) tensor."""
    t = torch.arange(n_steps + 1, dtype=torch.float64) / n_steps
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(0.0, 0.999).to(torch.float32)


# ---------------------------------------------------------------------------
# Sinusoidal step embedding (standard DDPM)
# ---------------------------------------------------------------------------

class StepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.size(-1) < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.size(-1)))
        return emb


# ---------------------------------------------------------------------------
# Latent denoiser — 1D U-Net-lite
# ---------------------------------------------------------------------------

class LatentDenoiser(nn.Module):
    """
    Predict noise added to a latent sequence at step t.

    Input  : (B, T, latent_dim) noisy latent + step + optional cond vector
    Output : (B, T, latent_dim) predicted noise
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int = 0,
        channels: int = 64,
        step_emb_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.channels = channels

        self.step_emb = StepEmbedding(step_emb_dim)
        film_in_dim = step_emb_dim + cond_dim
        self.film = nn.Linear(film_in_dim, 2 * channels)

        # 1D conv stack over the time axis. Treat latent_dim as channel.
        self.in_conv = nn.Conv1d(latent_dim, channels, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.out_conv = nn.Conv1d(channels, latent_dim, kernel_size=3, padding=1)

    def forward(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_noisy : (B, T, latent_dim)
        t       : (B,) integer step indices
        cond    : (B, cond_dim) or None
        """
        emb = self.step_emb(t)                               # (B, step_emb)
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.zeros(x_noisy.size(0), self.cond_dim,
                                   device=x_noisy.device, dtype=x_noisy.dtype)
            emb = torch.cat([emb, cond], dim=-1)             # (B, step+cond)
        gamma_beta = self.film(emb)                          # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)            # each (B, C)

        # (B, T, latent) → (B, latent, T) for Conv1d
        h = x_noisy.transpose(1, 2)
        h = self.in_conv(h)                                  # (B, C, T)
        # FiLM modulate.
        h = (1 + gamma.unsqueeze(-1)) * h + beta.unsqueeze(-1)

        h = h + self.block1(h)
        h = h + self.block2(h)

        out = self.out_conv(h)                               # (B, latent, T)
        return out.transpose(1, 2)                            # (B, T, latent)


# ---------------------------------------------------------------------------
# Diffusion training utilities
# ---------------------------------------------------------------------------

class LatentDiffusion:
    """
    Lightweight DDPM wrapper. Holds α/β schedule and exposes
    training and sampling primitives. No nn.Module — the LatentDenoiser
    holds the trainable parameters; this class is purely arithmetic.
    """

    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps
        self.betas = cosine_beta_schedule(n_steps)               # (n_steps,)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - self.alphas_cumprod)

    def _to(self, device, dtype):
        # Lazy-move buffers to the model's device on first call.
        if self.betas.device != device:
            self.betas = self.betas.to(device, dtype)
            self.alphas = self.alphas.to(device, dtype)
            self.alphas_cumprod = self.alphas_cumprod.to(device, dtype)
            self.sqrt_acp = self.sqrt_acp.to(device, dtype)
            self.sqrt_one_minus_acp = self.sqrt_one_minus_acp.to(device, dtype)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward noising: x_t = √ᾱ x0 + √(1-ᾱ) ε. Returns (x_t, ε)."""
        self._to(x0.device, x0.dtype)
        eps = torch.randn_like(x0)
        sa = self.sqrt_acp[t].view(-1, 1, 1)
        sm = self.sqrt_one_minus_acp[t].view(-1, 1, 1)
        return sa * x0 + sm * eps, eps

    def training_loss(
        self,
        denoiser: LatentDenoiser,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard noise-prediction MSE."""
        self._to(x0.device, x0.dtype)
        B = x0.size(0)
        t = torch.randint(0, self.n_steps, (B,), device=x0.device)
        x_t, eps = self.q_sample(x0, t)
        eps_pred = denoiser(x_t, t, cond)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(
        self,
        denoiser: LatentDenoiser,
        shape: Tuple[int, int, int],
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Ancestral sampling. shape = (B, T, latent_dim)."""
        self._to(device, torch.float32)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.n_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps_pred = denoiser(x, t, cond)
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            mean = (1.0 / torch.sqrt(alpha)) * (
                x - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * eps_pred
            )
            if i > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[i])
                x = mean + sigma * noise
            else:
                x = mean
        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, L, C = 4, 12, 16, 0
    denoiser = LatentDenoiser(latent_dim=L, cond_dim=C, channels=32)
    diff = LatentDiffusion(n_steps=50)   # short for smoke test

    x0 = torch.randn(B, T, L, requires_grad=True)
    loss = diff.training_loss(denoiser, x0)
    loss.backward()
    print(f"training MSE: {loss.item():.4f}")
    n_params = sum(p.numel() for p in denoiser.parameters())
    print(f"denoiser params: {n_params:,}")

    with torch.no_grad():
        sample = diff.sample(denoiser, (2, T, L), torch.device("cpu"))
    print(f"sample shape: {sample.shape}")
    print(f"sample range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
    print("hybrid_diffusion smoke test OK")
