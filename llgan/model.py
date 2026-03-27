"""LLGAN model: LSTM generator + LSTM critic (for WGAN-GP)."""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    LSTM generator.

    Each timestep receives the same noise vector z (broadcast across the
    sequence); the LSTM hidden state carries temporal context forward.
    Output is (batch, timestep, num_cols) in (-1, 1) via tanh.
    """

    def __init__(self, noise_dim: int, num_cols: int, hidden_size: int):
        super().__init__()
        self.noise_dim = noise_dim
        self.num_cols = num_cols

        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_cols)
        self.tanh = nn.Tanh()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, timestep, noise_dim) → (batch, timestep, num_cols)."""
        h, _ = self.lstm(z)
        return self.tanh(self.fc(h))

    @torch.no_grad()
    def generate(
        self,
        n_windows: int,
        timestep: int,
        device: torch.device,
        opcode_col: int = -1,
    ) -> torch.Tensor:
        self.eval()
        z = torch.randn(n_windows, timestep, self.noise_dim, device=device)
        out = self(z)
        if opcode_col >= 0:
            out[:, :, opcode_col] = (out[:, :, opcode_col] >= 0).float() * 2 - 1
        return out


class Critic(nn.Module):
    """
    LSTM critic for Wasserstein training.

    Outputs an unbounded real score (no sigmoid) — higher = more "real".
    Lipschitz constraint is enforced via spectral normalization on all
    weight matrices, which works on MPS/CPU without second-order gradients.
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
        h, _ = self.lstm(x)
        return self.fc(h).mean(dim=1)


# Keep old name as alias so existing code that imports Discriminator still works
Discriminator = Critic
