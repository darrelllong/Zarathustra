"""
Chunk stitching / boundary-consistency loss (IDEAS.md idea #21).

Status
------
WIRED. Sub-loss (a) boundary_latent_smoothness and sub-loss (b)
feature-space overlap-consistency are both integrated into the joint-GAN
training step (llgan/train.py). BS and OC are computed on INDEPENDENT
forward pairs so BS always carries adjacent-window semantics regardless
of OC mode. Sub-loss (b) is driven by --overlap-consistency-weight and
controlled by --overlap-consistency-mode:
  * mode=overlap (default, 2026-04-18): TRUE WaveStitch-style overlap.
    Chunk A is split at step T-k to capture h_mid. A's suffix and B's
    prefix both start from h_mid with INDEPENDENT local noise — so A's
    last k and B's first k decoded features refer to the same absolute
    timesteps. loss = overlap_consistency(feat_A, feat_B, k). Drives
    noise-invariance in the overlap region.
  * mode=boundary (legacy): adjacent-window pair, decay-weighted MSE
    on decoded features. Kept for provenance of pre-2026-04-18 runs
    (v140).
Backward compatible: when both weights are 0, training is byte-identical
to the prior recipe.

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

Wiring (see llgan/train.py, G-update block)
-------------------------------------------
BS and OC are computed on SEPARATE forward pairs when both are enabled,
so BS's adjacent-window semantics are stable regardless of OC mode:

  BS pair (always adjacent): A full forward → h_b_carry; B starts from
    h_b_carry. loss = boundary_latent_smoothness(H_A, H_B).
  OC pair (mode=overlap): A split at T-k producing h_mid; A's suffix
    and B's prefix both start from h_mid with INDEPENDENT local noise.
    loss = overlap_consistency(R(H_A), R(H_B), k).
  OC pair (mode=boundary, legacy): adjacent pair, decay-weighted MSE on
    decoded features — retained for v140-era reproducibility.

Cost: ~2x per G-step when both losses enabled (two extra G forwards);
~1.5x when only one is enabled. Memory: +1-2x batch in activations.

CLI flags:
  --boundary-smoothness-weight FLOAT   default 0.0 (off)
  --boundary-smoothness-k INT          default 2
  --boundary-smoothness-decay FLOAT    default 0.5
  --overlap-consistency-weight FLOAT   default 0.0 (off)
  --overlap-consistency-mode {overlap,boundary}  default overlap
  --overlap-consistency-k INT          default 2
  --overlap-consistency-decay FLOAT    default 0.5  (boundary mode only)

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
    Derivative-matching continuity loss at the boundary between two latent chunks.

    For order i in 0..k-1, compute the i-th forward finite difference at A's
    trailing edge and at B's leading edge, and MSE-match them. Order 0 is
    position continuity (A[T-1] ↔ B[0]); order 1 is velocity continuity
    ((A[T-1]-A[T-2]) ↔ (B[1]-B[0])); order 2 is acceleration continuity; etc.
    decay**i weights higher-order (farther-from-boundary) terms less, so the
    boundary position dominates.

    Parameters
    ----------
    latent_A : (B, T, D)  generator output for the first chunk
    latent_B : (B, T, D)  generator output for the second chunk, generated
                          with hidden = final hidden of chunk A
    k : how many orders of continuity to enforce (i = 0..k-1)
    decay : weight on order i is decay**i, so position (i=0) counts most

    Returns
    -------
    scalar loss

    Note
    ----
    An earlier version reversed A's trailing window with ``.flip(dims=[1])``
    and compared positions A[T-1-i] against B[i]. For i≥1 that forced
    A[T-1-i] = B[i] — a palindrome constraint around the boundary — which
    penalizes directional trends rather than smoothing them (peer review
    Round 27 P1 #4 / Gemini Round 3 P1 #1, 2026-04-19). Replaced here with
    the derivative-matching formulation above.
    """
    if latent_A.dim() != 3 or latent_B.dim() != 3:
        raise ValueError("expected (B, T, D) tensors for both chunks")
    if latent_A.size(0) != latent_B.size(0) or latent_A.size(2) != latent_B.size(2):
        raise ValueError("chunks must share batch and feature dims")

    T_A = latent_A.size(1)
    T_B = latent_B.size(1)
    k_eff = min(k, T_A, T_B)
    if k_eff <= 0:
        return latent_A.new_zeros(())

    import math
    loss = latent_A.new_zeros(())
    total_w = 0.0
    for i in range(k_eff):
        # i-th forward finite difference: Δ^i x[n] = Σ_j C(i,j)·(-1)^(i-j)·x[n+j]
        # Evaluated at n = T-1-i on A (uses A[T-1-i..T-1], i+1 points)
        # and at n = 0 on B (uses B[0..i], i+1 points).
        if i == 0:
            d_a = latent_A[:, T_A - 1 : T_A, :]
            d_b = latent_B[:, 0:1, :]
        else:
            coeffs = torch.tensor(
                [((-1) ** (i - j)) * math.comb(i, j) for j in range(i + 1)],
                device=latent_A.device,
                dtype=latent_A.dtype,
            ).view(1, i + 1, 1)
            a_pts = latent_A[:, T_A - 1 - i : T_A, :]     # (B, i+1, D)
            b_pts = latent_B[:, : i + 1, :]               # (B, i+1, D)
            d_a = (a_pts * coeffs).sum(dim=1, keepdim=True)
            d_b = (b_pts * coeffs).sum(dim=1, keepdim=True)
        w = decay ** i
        loss = loss + w * ((d_a - d_b) ** 2).mean()
        total_w += w
    return loss / max(total_w, 1e-9)


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
