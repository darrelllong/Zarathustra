"""LLGAN model: LSTM generator + LSTM critic (WGAN-SN) + latent AE components."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Latent autoencoder components (TimeGAN / SeriesGAN architecture)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    GRU encoder: maps feature sequences to a smooth latent space.

    (batch, timestep, num_cols) → (batch, timestep, latent_dim) ∈ [0, 1]

    Sigmoid output keeps latents bounded, matching the generator's output
    range and stabilising training of the recovery and supervisor.
    """

    def __init__(self, num_cols: int, hidden_size: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(num_cols, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, latent_dim)
        self.act = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.act(self.fc(h))


class Recovery(nn.Module):
    """
    GRU recovery: decodes latent sequences back to feature space.

    (batch, timestep, latent_dim) → (batch, timestep, num_cols) ∈ [-1, 1]

    Tanh output matches the [-1, 1] range of the normalised input features.
    """

    def __init__(self, latent_dim: int, hidden_size: int, num_cols: int):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, num_cols)
        self.act = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        return self.act(self.fc(out))


class Supervisor(nn.Module):
    """
    GRU supervisor: predicts H_{t+1} from H_t.

    Trained on real latent sequences to capture temporal dynamics.
    During joint training it acts as a consistency regulariser for the
    generator: output[t] ≈ input[t+1], so S(G(z))[:, :-1, :] ≈ G(z)[:, 1:, :].

    (batch, timestep, latent_dim) → (batch, timestep, latent_dim) ∈ [0, 1]
    """

    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(latent_dim, hidden_size, num_layers=1, batch_first=True)
        self.fc  = nn.Linear(hidden_size, latent_dim)
        self.act = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        return self.act(self.fc(out))


# ---------------------------------------------------------------------------
# Generator and Critic
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    """
    LSTM generator with split-latent noise design.

    Noise is split into two independent components:
      z_global (batch, noise_dim)           — encodes workload identity;
                                              projected to LSTM initial state.
      z_local  (batch, timestep, noise_dim) — per-step innovation noise;
                                              fed as LSTM input at each step.

    When latent_dim is set (latent AE mode):
      Output is (batch, timestep, latent_dim) ∈ [0, 1] via Sigmoid —
      compatible with the Encoder's output range. The Recovery module
      decodes to feature space.

    When latent_dim is None (legacy direct mode):
      Output is (batch, timestep, num_cols) ∈ (-1, 1) via Tanh.
    """

    def __init__(
        self,
        noise_dim: int,
        num_cols: int,
        hidden_size: int,
        latent_dim: Optional[int] = None,
    ):
        super().__init__()
        self.noise_dim   = noise_dim
        self.num_cols    = num_cols
        self.hidden_size = hidden_size
        self.latent_dim  = latent_dim  # None → legacy feature-space output

        out_dim = latent_dim if latent_dim is not None else num_cols

        # Project global code → initial LSTM hidden and cell states
        self.z_to_h0 = nn.Linear(noise_dim, hidden_size)
        self.z_to_c0 = nn.Linear(noise_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc      = nn.Linear(hidden_size, out_dim)
        # Sigmoid matches Encoder range [0,1]; Tanh for legacy feature output
        self.out_act = nn.Sigmoid() if latent_dim is not None else nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        hidden=None,
        return_hidden: bool = False,
    ):
        """
        z_global: (batch, noise_dim)           — workload identity code;
                  used to initialise hidden state when `hidden` is None.
        z_local:  (batch, timestep, noise_dim) — per-step innovation noise.
        hidden:   optional (h, c) tuple to continue from a previous window;
                  if None, h0/c0 are derived from z_global.
        return_hidden: if True, return (output, hidden) instead of output.

        Returns: (batch, timestep, out_dim),
                 or ((batch, timestep, out_dim), (h_n, c_n)) when return_hidden.
        out_dim is latent_dim (Sigmoid) in latent AE mode, num_cols (Tanh) otherwise.
        """
        if hidden is None:
            h0 = self.z_to_h0(z_global).unsqueeze(0)   # (1, batch, hidden)
            c0 = self.z_to_c0(z_global).unsqueeze(0)   # (1, batch, hidden)
            hidden = (h0, c0)
        h, hidden_out = self.lstm(z_local, hidden)
        out = self.out_act(self.fc(h))
        if return_hidden:
            return out, hidden_out
        return out

    @torch.no_grad()
    def generate(
        self,
        n_windows: int,
        timestep: int,
        device: torch.device,
        opcode_col: int = -1,
    ) -> torch.Tensor:
        self.eval()
        z_global = torch.randn(n_windows, self.noise_dim, device=device)
        z_local  = torch.randn(n_windows, timestep, self.noise_dim, device=device)
        out = self(z_global, z_local)
        if opcode_col >= 0:
            out[:, :, opcode_col] = (out[:, :, opcode_col] >= 0).float() * 2 - 1
        return out


class Critic(nn.Module):
    """
    LSTM critic for Wasserstein training (WGAN-SN variant).

    Outputs an unbounded real score (no sigmoid) — higher = more "real".
    Lipschitz constraint is partially enforced via spectral normalisation on
    the output linear layer only. The LSTM weights are unconstrained — a known
    weakness. Full enforcement (WGAN-GP gradient penalty or SN on all weights)
    requires CUDA and will be added once the NVIDIA box is available.

    Temporal pooling uses learned attention instead of mean-pooling.
    Mean-pooling washes out short bursts, sudden regime changes, and rare
    write-heavy segments; attention lets the critic weight the most
    discriminative timesteps.
    """

    def __init__(self, num_cols: int, hidden_size: int):
        super().__init__()
        from torch.nn.utils import spectral_norm

        # Spectral norm on the output projection only — applying it to LSTM
        # weight matrices is prohibitively slow due to per-step power iteration.
        self.lstm = nn.LSTM(
            input_size=num_cols,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        # Learned attention scorer: maps each hidden state → scalar weight
        self.attn = nn.Linear(hidden_size, 1)
        self.fc = spectral_norm(nn.Linear(hidden_size, 1))
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_orig" in name or ("weight" in name and "norm" not in name):
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, timestep, num_cols) → (batch, 1) unbounded score."""
        h, _ = self.lstm(x)                                    # (B, T, H)
        attn_w = torch.softmax(self.attn(h), dim=1)            # (B, T, 1)
        pooled = (attn_w * h).sum(dim=1)                       # (B, H)
        return self.fc(pooled)                                  # (B, 1)


# Keep old name as alias so existing code that imports Discriminator still works
Discriminator = Critic
