"""
Path Characteristic Function (PCF) loss for sequence-level distribution matching.

Replaces handcrafted auxiliary losses (ACF, FFT, moment, quantile, cross-cov,
locality) with a single learned functional that compares real and fake sequence
distributions via their characteristic functions evaluated on path increments.

The key idea (Horvath et al., PCF-GAN, NeurIPS 2023): two stochastic processes
have the same law iff their characteristic functions agree on all frequencies.
We learn frequency vectors that MAXIMIZE the discrepancy between real and fake
(adversarial), while G minimizes it. This is a minimax game on frequency space.

CRITICAL: The frequency parameters must be optimized by a SEPARATE optimizer
that MAXIMIZES the PCF loss (or equivalently, by the critic optimizer with
negated loss). If frequencies are co-optimized with G (which minimizes all
losses), they collapse to zero — providing no gradient signal.

v71 bug: frequencies were in G optimizer → collapsed to 0.0000 by epoch 10.
v72 fix: frequencies in C optimizer with MAXIMIZATION (gradient ascent).

Reference:
  Horvath et al., "PCF-GAN: Generating Sequential Data via the Characteristic
  Function of Measures on the Path Space", NeurIPS 2023.
"""

import torch
import torch.nn as nn


class PCFLoss(nn.Module):
    """
    Adversarial path characteristic function loss.

    Frequency vectors are trained to MAXIMIZE the characteristic function
    distance (find frequencies where real and fake differ most), while the
    generator is trained to MINIMIZE it (make fake match real at those
    frequencies). This is a minimax game analogous to the GAN critic.

    Usage:
        pcf = PCFLoss(num_cols=5, timestep=12)

        # Frequency params go in C optimizer (or separate optimizer):
        opt_C = Adam(list(C.parameters()) + list(pcf.parameters()), lr=lr_d)

        # Critic step: MAXIMIZE PCF distance (train frequencies)
        loss_pcf_c = pcf(real, fake.detach())
        (-loss_pcf_c).backward()  # negate for gradient ascent

        # Generator step: MINIMIZE PCF distance
        loss_pcf_g = pcf(real.detach(), fake)
        (pcf_weight * loss_pcf_g).backward()
    """

    def __init__(
        self,
        num_cols: int,
        timestep: int,
        n_freqs: int = 32,
        freq_scale: float = 1.0,
    ):
        super().__init__()
        self.num_cols = num_cols
        self.timestep = timestep
        self.n_freqs = n_freqs

        # Frequency vectors: (n_freqs, (T-1) * d)
        # Initialized at scale=1.0 (not 0.1) so they start sensitive.
        inc_dim = (timestep - 1) * num_cols
        self.freqs = nn.Parameter(
            torch.randn(n_freqs, inc_dim) * freq_scale
        )

    def _char_fn(self, increments: torch.Tensor) -> torch.Tensor:
        """
        Compute empirical characteristic function at learned frequencies.

        increments: (B, T-1, d) path increments
        Returns: (n_freqs, 2) — [Re(φ), Im(φ)] for each frequency
        """
        B = increments.shape[0]
        inc_flat = increments.reshape(B, -1)  # (B, (T-1)*d)

        # Inner product: (B, n_freqs)
        inner = inc_flat @ self.freqs.t()

        # Characteristic function: E[exp(i * inner)]
        phi_re = inner.cos().mean(dim=0)  # (n_freqs,)
        phi_im = inner.sin().mean(dim=0)  # (n_freqs,)

        return torch.stack([phi_re, phi_im], dim=1)  # (n_freqs, 2)

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute PCF distance between real and fake sequences.

        Returns: scalar distance (0 = distributions match perfectly).
                 Caller must negate for frequency maximization step.
        """
        # Path increments
        real_inc = real[:, 1:, :] - real[:, :-1, :]
        fake_inc = fake[:, 1:, :] - fake[:, :-1, :]

        phi_real = self._char_fn(real_inc)
        phi_fake = self._char_fn(fake_inc)

        # L2 distance between characteristic functions
        return ((phi_real - phi_fake) ** 2).sum()

    def extra_repr(self) -> str:
        return (f"n_freqs={self.n_freqs}, "
                f"inc_dim={self.freqs.shape[1]}")
