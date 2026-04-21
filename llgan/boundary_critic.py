"""
Learned boundary critic for IDEA #36.

Replaces hand-written boundary-smoothness scalars with a small adversarial
critic that learns what realistic window-boundary transitions look like from
data.  The critic sees the last K decoded-feature steps of window A and the
first K decoded-feature steps of window B concatenated into a (2K*F) vector,
and is trained WGAN-SN style to score real adjacent joins higher than
generated ones.

Training signal:
  Real pairs:  consecutive non-overlapping windows drawn from real trace files
               (data[i+T-K : i+T] + data[i+T : i+T+K] = a 2K-step slice
               centered on an in-file adjacent-window boundary at stride T).
               Note: these are within-file window joins sampled at every T-record
               stride, not cross-file boundaries.
  Fake pairs:  G generates chunk A, carries hidden state, generates chunk B;
               decoded features of A's tail + B's head form the fake boundary.

Generator objective:  G tries to fool the boundary critic on its adjacent-
window joins, giving it direct gradient feedback on cross-window continuity
without prescribing position/velocity equality.

Spectral normalisation is applied to every linear layer to satisfy the
Lipschitz constraint required by WGAN.

CLI flags (all in train.py):
  --boundary-critic-weight FLOAT   adversarial weight on G loss (default 0.0 = off)
  --boundary-critic-k INT          context steps K on each side of boundary (default 4)
  --boundary-critic-hidden INT     hidden size of the MLP (default 128)

References
----------
  IDEAS.md #36: Learned boundary prior instead of deterministic BS/OC penalties.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BoundaryCritic(nn.Module):
    """
    MLP WGAN-SN critic for window-boundary realism.

    Input: last K decoded-feature steps of chunk A concatenated with first K
    decoded-feature steps of chunk B → (B, 2*K*feat_dim).
    Output: unbounded scalar score (higher = more realistic boundary join).
    """

    def __init__(self, feat_dim: int, k: int = 4, hidden: int = 128):
        super().__init__()
        self.k = k
        from torch.nn.utils import spectral_norm
        input_dim = 2 * k * feat_dim
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden, hidden // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(hidden // 2, 1)),
        )

    def forward(self, tail: Tensor, head: Tensor) -> Tensor:
        """
        Parameters
        ----------
        tail : (B, K, F)  last K decoded-feature steps of window A
        head : (B, K, F)  first K decoded-feature steps of window B

        Returns
        -------
        (B, 1) unbounded WGAN score
        """
        B = tail.size(0)
        x = torch.cat([tail.reshape(B, -1), head.reshape(B, -1)], dim=1)
        return self.net(x)


def sample_real_boundaries(
    per_file_arrays: List[Tensor],
    n: int,
    k: int,
    T: int,
    device: torch.device,
    full_window: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Sample n real boundary pairs from a list of per-file tensors.

    Each boundary is defined by a boundary index b (measured in records) such
    that the records [b-K : b] form the tail of one window and [b : b+K] form
    the head of the next non-overlapping window.  We enforce b = i*T for some
    integer i so that the boundary falls exactly at a window edge (i.e., not
    in the middle of an arbitrary window).

    Parameters
    ----------
    per_file_arrays : list of (N_i, F) float32 tensors — one per file
    n               : number of boundary pairs to return
    k               : context steps on each side (K)
    T               : window length (so boundaries are at multiples of T)
    device          : target device
    full_window     : if True, return full T-step windows (B, T, F) suitable
                      for encoding with E; if False (default), return K-step
                      boundary slices (B, K, F).

    Returns
    -------
    tail : (n, T or K, F)  window A (full) or last K steps before boundary
    head : (n, T or K, F)  window B (full) or first K steps after boundary
    """
    # Collect all valid boundary positions across all files.
    # full_window=True needs T steps on both sides; False only needs K steps.
    _min_after = T if full_window else k
    valid: List[Tuple[Tensor, int]] = []
    for arr in per_file_arrays:
        N = arr.size(0)
        for b in range(T, N - _min_after + 1, T):
            valid.append((arr, b))

    if not valid:
        F = per_file_arrays[0].size(1)
        z = torch.zeros(n, _min_after, F, device=device)
        return z, z

    chosen = random.choices(valid, k=n)
    tails, heads = [], []
    for arr, b in chosen:
        if full_window:
            tails.append(arr[b - T : b])      # (T, F)
            heads.append(arr[b : b + T])      # (T, F)
        else:
            tails.append(arr[b - k : b])      # (K, F)
            heads.append(arr[b : b + k])      # (K, F)

    tail = torch.stack(tails, dim=0).to(device)
    head = torch.stack(heads, dim=0).to(device)
    return tail, head
