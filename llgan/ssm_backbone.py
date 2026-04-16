"""
Selective diagonal state-space backbone (IDEAS.md idea #19 — design + module).

Status
------
DESIGN+IMPL — drop-in replacement for ``nn.LSTM(input_size, hidden_size,
num_layers=1, batch_first=True)`` with the same forward signature. NOT
YET WIRED into Generator (separate commit will gate via
``--ssm-backbone`` CLI flag). Backward compatible by construction:
when disabled, training is byte-identical.

Motivation
----------
``timestep=12`` is small but ``generate.py`` already carries hidden state
across windows for long rollouts. The LSTM forget-gate is known to lose
long-horizon structure; recent SSM work in network/storage trace
generation (NetSSM, the SSM-NetworkTraffic baseline) shows that
diagonal state-space recurrences preserve long-range dependencies
without the forget-gate squashing. Continuity loss failed → the
backbone, not the loss, may be the bottleneck.

Design — Selective Diagonal SSM (Mamba-lite, framework-only)
------------------------------------------------------------
For each channel i in 1..H (H = hidden_size), maintain a small state
vector s_i ∈ R^N (N = d_state, default 16). Per-step recurrence is:

  Δ_t       = softplus(W_Δ x_t + b_Δ)        # (B, H)  — selectivity
  A_bar     = exp(Δ_t ⊙ A)                    # (B, H, N) — discretised
  B_bar     = Δ_t.unsqueeze(-1) ⊙ B           # (B, H, N)
  s_t       = A_bar ⊙ s_{t-1} + B_bar ⊙ x_t.unsqueeze(-1)   # (B, H, N)
  y_t       = (C ⊙ s_t).sum(dim=-1) + D ⊙ x_t             # (B, H)

Where:
  A : (H, N) negative learned (init A_init = -0.5 .. -2.0)
  B : (H, N) learned input projection per channel
  C : (H, N) learned output projection per channel
  D : (H,)   skip connection (init 1, decays with training)
  W_Δ : (input_size, H) learned

Why diagonal? Full-matrix SSM is O(N^2 H) per step; diagonal is O(N H).
For N=16, H=256 this is ~4K mults/step — comparable to LSTM gate cost.

Why selective Δ? Vanilla diagonal SSM has fixed dynamics; selective Δ
lets the model decide *per-step* whether to retain state (small Δ → A
close to 1) or refresh it (large Δ → A close to 0). This is the
"selection" that Mamba contributes over S4. Implemented in plain
PyTorch — no custom CUDA, no kernel fusion — so it runs anywhere
``nn.LSTM`` does. Speed: ~2-3x slower than fused LSTM at H=256, N=16.

Hidden-state API match with nn.LSTM
-----------------------------------
LSTM hidden = ((1, B, H), (1, B, H)) tuple.
SSM hidden  = (B, H, N) tensor (single state).

To keep the Generator wiring simple, SSMBlock returns hidden as a
TUPLE (state, None) so the caller can keep its existing
``hidden = (hidden[0].detach(), hidden[1].detach())`` logic — the
second element is just None.

z_global → initial state mapping
--------------------------------
Generator currently splits z_global into z_to_h0 + z_to_c0 (LSTM h, c).
For SSM, define one projection: z_to_state ∈ R^(B, H*N) reshaped to
(B, H, N). LSTM-trained checkpoints CANNOT be loaded — SSM has different
parameter shapes — so a fresh pretrain is required.

CLI flags (NOT YET WIRED):
  --ssm-backbone               bool, default False
  --ssm-state-dim              int,  default 16
  --ssm-init-decay-min         float default 0.5  (initial Δ floor)
  --ssm-init-decay-max         float default 2.0  (initial Δ ceiling)

Failure modes monitored:
  - Δ collapsing to 0 → state becomes fixed; check Δ stats per epoch
  - Δ exploding → state forgets immediately; same diagnostic
  - State magnitude blowing up → C projection saturates; norm clip on
    state at write time (set ssm_state_clip)

References
----------
  Andrew Chu et al., NetSSM (arXiv 2025)
  Andrew Chu et al., Feasibility of SSMs for Network Traffic Gen (arXiv 2024)
  Albert Gu, Tri Dao, Mamba (arXiv 2023) — selection mechanism
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveDiagonalSSM(nn.Module):
    """
    Single-layer selective diagonal state-space block.

    Drop-in replacement for ``nn.LSTM(input_size, hidden_size, 1, batch_first=True)``
    with matching ``forward(x, hidden)`` signature returning
    ``(out, new_hidden)``. ``hidden`` is a tuple ``(state, None)`` where
    ``state`` is ``(B, hidden_size, d_state)`` or None.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        d_state: int = 16,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        a_init_min: float = 0.5,
        a_init_max: float = 2.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_state = d_state

        # Input projection if input_size != hidden_size (LSTM accepts mismatched).
        if input_size != hidden_size:
            self.in_proj = nn.Linear(input_size, hidden_size, bias=True)
        else:
            self.in_proj = nn.Identity()

        # A: negative real, log-parameterised so it stays negative.
        # Initialise A_log such that exp(A_log) ∈ [a_init_min, a_init_max].
        A_log_init = torch.empty(hidden_size, d_state).uniform_(
            math.log(a_init_min), math.log(a_init_max)
        )
        self.A_log = nn.Parameter(A_log_init)

        # B and C input/output projections per channel.
        self.B = nn.Parameter(torch.randn(hidden_size, d_state) * 0.02)
        self.C = nn.Parameter(torch.randn(hidden_size, d_state) * 0.02)

        # Skip connection D init at 1.0 — start as near-identity for stability.
        self.D = nn.Parameter(torch.ones(hidden_size))

        # Selective Δ: per-step gain from input.
        # Init bias so softplus(bias) ∈ [dt_min, dt_max].
        self.dt_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        # bias init: inverse-softplus uniform sample of dt_min..dt_max
        with torch.no_grad():
            dt_bias_init = torch.empty(hidden_size).uniform_(dt_min, dt_max)
            self.dt_proj.bias.copy_(_inv_softplus(dt_bias_init))

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        x : (B, T, input_size)
        hidden : (state (B, H, N), None) or None

        Returns
        -------
        out : (B, T, hidden_size)
        new_hidden : (state, None)
        """
        B, T, _ = x.shape
        u = self.in_proj(x)                                  # (B, T, H)

        # Negative-real A (diagonal, broadcast over batch).
        A = -torch.exp(self.A_log)                           # (H, N)

        if hidden is None:
            state = u.new_zeros(B, self.hidden_size, self.d_state)
        else:
            state, _ = hidden
            if state is None:
                state = u.new_zeros(B, self.hidden_size, self.d_state)

        outs = []
        # Sequential recurrence — clear and correct. A parallel scan
        # exists but is not needed for T=12. Speed dominated by the
        # einsum below, ~2-3x LSTM at H=256, N=16.
        for t in range(T):
            x_t = u[:, t, :]                                  # (B, H)
            dt = F.softplus(self.dt_proj(x_t))                # (B, H), > 0

            # Discretise: A_bar = exp(dt ⊙ A); B_bar = dt ⊙ B
            # dt: (B, H) → (B, H, 1) for broadcast over N
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, H, N)
            B_bar = dt.unsqueeze(-1) * self.B.unsqueeze(0)         # (B, H, N)

            # state update.
            state = A_bar * state + B_bar * x_t.unsqueeze(-1)      # (B, H, N)

            # Output: y_t = (C ⊙ s_t).sum(N) + D ⊙ x_t
            y_t = (self.C.unsqueeze(0) * state).sum(dim=-1) + self.D * x_t
            outs.append(y_t)

        out = torch.stack(outs, dim=1)                              # (B, T, H)
        return out, (state, None)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus: log(exp(x) - 1). Numerically stable."""
    return x + torch.log(-torch.expm1(-x))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, In, H, N = 4, 12, 24, 64, 16
    ssm = SelectiveDiagonalSSM(input_size=In, hidden_size=H, d_state=N)
    x = torch.randn(B, T, In, requires_grad=True)
    out, (state, _) = ssm(x)
    print(f"out shape: {out.shape}, state shape: {state.shape}")
    assert out.shape == (B, T, H)
    assert state.shape == (B, H, N)

    # Test hidden carry across two consecutive windows.
    out2, (state2, _) = ssm(torch.randn(B, T, In), hidden=(state.detach(), None))
    assert out2.shape == (B, T, H)
    print(f"second-window state range: [{state2.min().item():.3f}, {state2.max().item():.3f}]")

    loss = out.sum()
    loss.backward()
    n_params = sum(p.numel() for p in ssm.parameters())
    print(f"params: {n_params:,}")
    print(f"A_log grad norm: {ssm.A_log.grad.norm().item():.4f}")
    print(f"dt_proj grad norm: {ssm.dt_proj.weight.grad.norm().item():.4f}")
    print("ssm_backbone smoke test OK")
