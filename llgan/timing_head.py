"""
Marked Temporal Point Process timing head (IDEAS.md idea #20).

Status
------
DESIGN+IMPL — standalone log-Normal timing head module with NLL loss.
NOT YET WIRED into Generator/train.py; separate commit will gate via
``--mtpp-timing`` CLI flag. Backward compatible by construction.

Motivation
----------
Currently, `ts_delta` (inter-arrival time after delta-encoding) is one
continuous channel among many: a tanh head emits a single point
estimate, the preprocessor inverse-transforms it back to an IAT. This
treats event timing as a regression target rather than a *distribution*,
conflating three different sources of error:
  - the *expected* IAT given the workload regime
  - the *spread* of IATs (burstiness)
  - mark/timing dependence (do reuses happen at the right times?)

The MTPP framing decouples these: at each step, predict a *distribution*
over the next-event delay, then sample. NLL of the predicted
distribution against the real next IAT becomes a clean timing loss
that trains G's hidden state to encode burst structure.

MVE: hybrid log-Normal head
---------------------------
Smallest viable departure from current architecture:
  - Keep current IAT regression column (don't break old eval / Recovery
    pipeline).
  - Add a parallel TimingHead that emits (μ_t, σ_t) for log(IAT_{t+1})
    given the LSTM hidden state h_t.
  - At training time, add ``mtpp_nll(μ, σ, real_log_iat)`` as a loss
    term. G now has *two* timing signals: regression (existing) and
    distributional NLL (new).
  - At generation time, optionally sample from the log-Normal instead
    of using the regression output. Behind a flag — initial experiments
    use the NLL only as a regulariser.

Why log-Normal?
  - IATs are positive and heavy-tailed — log-Normal fits naturally and
    is stable to train (closed-form NLL, exp() never blows up if σ
    is reasonable).
  - Two parameters per step: μ (location), σ (scale). Cheap.
  - Recovers the existing point-estimate behaviour as μ → real, σ → 0,
    so the regularisation is "soft" — it nudges G toward better timing
    distributions without forcing a switch.

Future extensions (NOT in this MVE):
  - Replace log-Normal with mixture-of-log-Normals (multimodal IATs).
  - Hawkes-like history-conditional intensity.
  - Joint mark|time conditioning (e.g., reuse probability conditioned
    on the sampled IAT).

CLI flags (NOT YET WIRED):
  --mtpp-timing                bool, default False
  --mtpp-timing-weight FLOAT   default 0.5 (NLL term weight in G loss)
  --mtpp-sigma-min FLOAT       default 0.05 (numerical floor on σ)
  --mtpp-use-at-generation     bool, default False (sample at gen time)

Failure modes monitored:
  - σ collapsing to floor → distribution becomes Dirac → no calibration
    benefit. Diagnostic: log mean(σ) per epoch.
  - μ predicting log(IAT) but unable to adjust σ to fit burstiness
    → NLL stays high. Diagnostic: log NLL trend per epoch.

References
----------
  Yujee Song et al., Decoupled MTPP using Neural ODEs (ICLR 2024)
  Hui Chen et al., Marked Temporal Bayesian Flow Point Processes
    (arXiv 2024)
"""

from __future__ import annotations

from typing import Tuple

import math

import torch
import torch.nn as nn


class LogNormalTimingHead(nn.Module):
    """
    Per-timestep log-Normal head: produces (μ, σ) for log(IAT) from h_t.

    Parameters
    ----------
    hidden_size : int
        Width of the upstream LSTM hidden state h_t.
    sigma_min : float
        Numerical floor on σ to prevent NLL from blowing up.
    """

    def __init__(self, hidden_size: int, sigma_min: float = 0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.sigma_min = float(sigma_min)
        # Two outputs per step: μ and raw σ-pre-softplus.
        self.fc = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.fc.weight, 0.0, 0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h : (B, T, H)

        Returns
        -------
        mu    : (B, T)  log-Normal location
        sigma : (B, T)  log-Normal scale (>= sigma_min)
        """
        out = self.fc(h)                                  # (B, T, 2)
        mu = out[..., 0]                                  # (B, T)
        sigma_raw = out[..., 1]                           # (B, T)
        sigma = torch.nn.functional.softplus(sigma_raw) + self.sigma_min
        return mu, sigma

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Sample IAT > 0 from log-Normal(μ, σ)."""
        eps = torch.randn_like(mu)
        log_iat = mu + sigma * eps
        return torch.exp(log_iat)


def log_normal_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    real_iat: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Mean negative log-likelihood of real_iat under log-Normal(μ, σ).

    NLL = log(iat * σ * sqrt(2π)) + (log(iat) - μ)^2 / (2 σ^2)

    Parameters
    ----------
    mu, sigma : (B, T)
    real_iat  : (B, T) — must be > 0 (we clamp at eps).

    Returns
    -------
    scalar
    """
    iat = torch.clamp(real_iat, min=eps)
    log_iat = torch.log(iat)
    log_norm_const = torch.log(iat) + torch.log(sigma) + 0.5 * math.log(2.0 * math.pi)
    quad = (log_iat - mu) ** 2 / (2.0 * sigma ** 2)
    return (log_norm_const + quad).mean()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H = 4, 12, 64
    head = LogNormalTimingHead(hidden_size=H)
    h = torch.randn(B, T, H, requires_grad=True)
    mu, sigma = head(h)
    print(f"mu range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
    print(f"sigma range: [{sigma.min().item():.3f}, {sigma.max().item():.3f}]")

    # Real IAT: uniform [1e-4, 1e-1] simulates millisecond-scale events.
    real_iat = torch.empty(B, T).uniform_(1e-4, 1e-1)
    nll = log_normal_nll(mu, sigma, real_iat)
    print(f"NLL: {nll.item():.4f}")
    nll.backward()
    print(f"head fc grad norm: {head.fc.weight.grad.norm().item():.4f}")

    # Sampling:
    with torch.no_grad():
        samples = head.sample(mu, sigma)
        print(f"sample range: [{samples.min().item():.4e}, {samples.max().item():.4e}]")
        print(f"sample positivity: {(samples > 0).all().item()}")
    print("timing_head smoke test OK")
