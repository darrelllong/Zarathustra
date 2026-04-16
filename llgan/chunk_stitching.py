"""
Chunk stitching / boundary-consistency loss (IDEAS.md idea #21).

Status
------
DESIGN+IMPL — standalone loss callable; integration into the joint-GAN
training step is deferred to a separate commit. Backward compatible:
when disabled, training is byte-identical to the current recipe.

Motivation
----------
The previous scalar continuity loss (--continuity-loss-weight) failed on
tencent and was within-noise on alibaba. That failure does NOT mean
boundary stitching is the wrong idea; it means the previous loss was
too weak — a single-scalar penalty on adjacent-step similarity within
ONE window. The real "12-step train / long-trace generate" mismatch
lives at the WINDOW boundary, where generate.py carries hidden state
across windows but training never sees the boundary.

This module trains the generator on pairs of adjacent chunks with the
hidden-state carry that generate.py uses, supervising:
  (a) latent-space smoothness across the boundary, and
  (b) feature-space overlap consistency (post-Recovery decode).

When training succeeds at the boundary, the LSTM hidden state at chunk
B's first step is forced to be a meaningful continuation of chunk A's
final hidden state — exactly what generate.py needs at long rollout.

Two complementary losses
------------------------
1. ``boundary_latent_smoothness(latent_A, latent_B)``:
   MSE between the last K=2 steps of chunk A's latent and the first K=2
   steps of chunk B's latent, computed with an exponential-decay weight
   so step boundary > step boundary+1. Cheap (O(B*K*D_lat)) and
   directly trainable through the LSTM.

2. ``overlap_consistency(feat_A, feat_B, k_overlap)``:
   For settings where chunks deliberately overlap by k_overlap real
   steps (chunk B starts at chunk A's t = T-k_overlap), the overlap
   region must be self-consistent. MSE on the overlapping feature
   columns (post-Recovery). Cheap, but requires generation in
   "overlap mode" (re-using the same hidden state offset).

Phase A (this commit) ships only the latent smoothness loss. Phase B
adds overlap-mode generation if Phase A shows signal.

Integration sketch (NOT YET WIRED)
----------------------------------
In the joint-GAN training step (train.py, around the G update):

    z_global, z_local_A = sample(...)              # current code
    latent_A, hidden_A = G(z_global, z_local_A, return_hidden=True)
    if cfg.boundary_smoothness_weight > 0:
        z_local_B = sample_z_local(B, T, device)
        latent_B, _ = G(z_global, z_local_B, hidden=hidden_A,
                         return_hidden=True)
        bs_loss = boundary_latent_smoothness(latent_A, latent_B)
        g_loss = g_loss + cfg.boundary_smoothness_weight * bs_loss

    # rest of G update unchanged

Cost: ~1.5x per G-step (one extra forward through G; no extra critic
or supervisor work). Memory: roughly +1x batch in activations.

CLI flag (to add):
  --boundary-smoothness-weight FLOAT   default 0.0 (off)
  --boundary-smoothness-k INT          default 2 (steps to compare)
  --boundary-smoothness-decay FLOAT    default 0.5

Why this is on-target
---------------------
generate.py carries hidden state across windows; eval.py never does.
Training never does either. So G learns to produce nice 12-step windows
but has zero supervision that the LSTM's terminal state is a useful
input for the NEXT 12 steps. Long-rollout drift is the expected
consequence. This loss is the smallest fix that touches that exact gap.

References
----------
  Aditya Shankar et al., WaveStitch (arXiv 2025) — overlap-and-stitch
  diffusion baseline.
  Hou et al., Stage-Diff (arXiv 2025) — stage-wise long TS gen.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def boundary_latent_smoothness(
    latent_A: torch.Tensor,
    latent_B: torch.Tensor,
    k: int = 2,
    decay: float = 0.5,
) -> torch.Tensor:
    """
    MSE between trailing-k steps of latent_A and leading-k steps of latent_B.

    Parameters
    ----------
    latent_A : (B, T, D)  generator output for the first chunk
    latent_B : (B, T, D)  generator output for the second chunk, generated
                          with hidden = final hidden of chunk A
    k : number of boundary steps on each side to compare
    decay : weight on step i is decay**i, so step closest to boundary
            counts most (i=0)

    Returns
    -------
    scalar loss
    """
    if latent_A.dim() != 3 or latent_B.dim() != 3:
        raise ValueError("expected (B, T, D) tensors for both chunks")
    if latent_A.size(0) != latent_B.size(0) or latent_A.size(2) != latent_B.size(2):
        raise ValueError("chunks must share batch and feature dims")

    T = latent_A.size(1)
    k_eff = min(k, T)

    # Last k of A, first k of B — order so that index 0 in each = boundary step.
    a_tail = latent_A[:, T - k_eff:, :].flip(dims=[1])    # (B, k, D)
    b_head = latent_B[:, :k_eff, :]                       # (B, k, D)

    weights = torch.tensor(
        [decay ** i for i in range(k_eff)],
        device=latent_A.device,
        dtype=latent_A.dtype,
    ).view(1, k_eff, 1)

    sq = (a_tail - b_head) ** 2                           # (B, k, D)
    weighted = sq * weights
    return weighted.mean()


def overlap_consistency(
    feat_A: torch.Tensor,
    feat_B: torch.Tensor,
    k_overlap: int,
) -> torch.Tensor:
    """
    Phase-B helper: MSE on the overlap region between two chunks generated
    in deliberate-overlap mode (chunk B starts at chunk A's index
    T - k_overlap, sharing those steps).

    Parameters
    ----------
    feat_A : (B, T, F)  decoded feature output of chunk A
    feat_B : (B, T, F)  decoded feature output of chunk B
    k_overlap : number of overlapping steps (B's first k = A's last k)
    """
    if k_overlap <= 0:
        return feat_A.new_zeros(())
    a_tail = feat_A[:, -k_overlap:, :]
    b_head = feat_B[:, :k_overlap, :]
    return F.mse_loss(a_tail, b_head)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 4, 12, 8
    latent_A = torch.randn(B, T, D, requires_grad=True)
    latent_B = torch.randn(B, T, D, requires_grad=True)
    loss = boundary_latent_smoothness(latent_A, latent_B, k=2, decay=0.5)
    loss.backward()
    print(f"boundary_latent_smoothness loss: {loss.item():.4f}")
    print(f"  grad norm A: {latent_A.grad.norm().item():.4f}")
    print(f"  grad norm B: {latent_B.grad.norm().item():.4f}")

    feat_A = torch.randn(B, T, 5)
    feat_B = torch.randn(B, T, 5)
    oc = overlap_consistency(feat_A, feat_B, k_overlap=3)
    print(f"overlap_consistency loss: {oc.item():.4f}")
    print("chunk_stitching smoke test OK")
