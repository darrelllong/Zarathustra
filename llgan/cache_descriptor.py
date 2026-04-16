"""
Cache-descriptor distillation (IDEAS.md idea #18 — design + module).

Status
------
DESIGN+IMPL — module with descriptor computation, encoder, and a non-
differentiable reconstruction monitor is implemented. CLI wiring and the
offline precomputation script are deferred to a separate commit.

Motivation (from IDEAS.md #18)
------------------------------
HRC is in eval but not in training. Competitor systems (TRAGEN, JEDI,
2DIO) keep winning by building compact workload descriptors that preserve
recency/frequency/footprint structure. Zarathustra's existing
``cond_dim=10`` vector encodes general workload character (write_ratio,
IAT q50, tenant_unique, …) but contains no cache-native shape
descriptors — no HRC slices, no reuse-distance distribution, no working-
set growth curve. This module adds those.

Descriptor selection (8-dim default)
------------------------------------
All cheap to compute, all bounded, all defined on a single trace window
or file aggregate:

  0. log(unique_obj_count_in_window)      working-set size proxy
  1. top1_obj_share                       popularity tail (0..1)
  2. top3_obj_share                       popularity tail (0..1)
  3. reuse_distance_q25 / window_len      reuse-distance lower quartile
  4. reuse_distance_q50 / window_len      reuse-distance median
  5. reuse_distance_q75 / window_len      reuse-distance upper quartile
  6. reuse_share                          fraction of accesses that hit
                                          a previously seen obj in window
  7. burstiness_index                     std(IAT) / mean(IAT), clipped

These are *cache-native*: they directly govern HRC shape via the
Che approximation (HRC ≈ 1 - F_reuse(C / S)), which depends on the
reuse-distance CDF and working-set size — both captured here.

Two-phase design
----------------
Phase A (this module, MVE):
  - Offline precompute descriptors per real file → JSONL (mirrors the
    trace_characterizations.jsonl pattern).
  - At training time, compute descriptors on generated windows post-
    Recovery (in feature space), MSE against the file-level targets.
  - NON-differentiable in this phase — used only as monitor metric and
    checkpoint tiebreaker. Logged each ★ epoch as ``desc_mse``.
  - Optionally feed the 8-dim descriptor vector into G's conditioning
    path via DescriptorEncoder, alongside the existing 10-dim cond.

Phase B (deferred — only if Phase A shows signal):
  - Replace descriptors with soft, differentiable variants (Gaussian-RBF
    soft unique count, soft histograms over reuse distance) so the loss
    can backprop into G.
  - Promote desc_mse to a real training-loss term with weight.

Why start with Phase A?
  Differentiable cache descriptors are non-trivial (the Hawkes-process
  baselines explicitly punt on this). If file-level descriptor matching
  alone moves the needle on HRC-MAE / reuse-rate, Phase B is justified.
  If not, we've ruled out the channel cheaply.

CLI flags (to add in train.py — NOT YET WIRED):
  --cache-descriptor-file PATH       offline JSONL of per-file targets
  --cache-descriptor-condition       feed targets into G's z_global path
  --cache-descriptor-monitor         log desc_mse each ★ epoch
  --cache-descriptor-dim INT         default 8

Offline precompute (helper in this module):
  ``compute_descriptor_for_window(events, window_len)`` returns the 8-dim
  vector for a single window. Aggregate over all windows in a file to
  produce the file-level target.

References
----------
  Wang, Khor, Desnoyers — 2DIO, EuroSys 2026
  Sabnis & Sitaraman — TRAGEN (IMC 2021), JEDI (IMC 2022)
  Li et al. — TraceGen, HPCC 2024
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Numpy descriptor computation (offline target precompute + online monitor)
# ---------------------------------------------------------------------------

def compute_descriptor_for_window(
    obj_ids: np.ndarray,
    iats: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the 8-dim cache descriptor for one window.

    Parameters
    ----------
    obj_ids : (T,) integer object IDs (in real-trace space).
    iats    : (T,) inter-arrival times (positive). Optional; when None,
              the burstiness slot is set to 0.

    Returns
    -------
    (8,) float32 vector — see module docstring for slot layout.
    """
    T = len(obj_ids)
    if T == 0:
        return np.zeros(8, dtype=np.float32)

    # Slot 0: working-set size (log scale).
    unique, counts = np.unique(obj_ids, return_counts=True)
    n_unique = unique.size
    s0 = math.log(max(n_unique, 1))

    # Slots 1, 2: top-k popularity share.
    sorted_counts = np.sort(counts)[::-1]
    s1 = float(sorted_counts[0]) / T
    s2 = float(sorted_counts[:3].sum()) / T

    # Slots 3, 4, 5: reuse-distance quantiles (in absolute steps).
    last_seen: Dict[int, int] = {}
    reuse_dists: List[int] = []
    for t, oid in enumerate(obj_ids):
        oid_int = int(oid)
        if oid_int in last_seen:
            reuse_dists.append(t - last_seen[oid_int])
        last_seen[oid_int] = t
    if reuse_dists:
        rd = np.asarray(reuse_dists, dtype=np.float32) / max(T, 1)
        s3 = float(np.quantile(rd, 0.25))
        s4 = float(np.quantile(rd, 0.50))
        s5 = float(np.quantile(rd, 0.75))
    else:
        s3 = s4 = s5 = 0.0

    # Slot 6: reuse share (fraction of events that are reuses).
    s6 = float(len(reuse_dists)) / T

    # Slot 7: burstiness CV, clipped.
    if iats is not None and len(iats) > 1:
        mean = float(np.mean(iats))
        std = float(np.std(iats))
        s7 = std / mean if mean > 1e-9 else 0.0
        s7 = min(s7, 5.0) / 5.0     # normalise to [0, 1]
    else:
        s7 = 0.0

    return np.asarray([s0, s1, s2, s3, s4, s5, s6, s7], dtype=np.float32)


def aggregate_file_descriptor(
    window_descriptors: np.ndarray,
) -> np.ndarray:
    """
    Aggregate per-window descriptors into one file-level descriptor.

    Uses median per slot — robust to outlier windows. Future variants
    could keep multiple quantiles per slot for richer conditioning.
    """
    if window_descriptors.size == 0:
        return np.zeros(8, dtype=np.float32)
    return np.median(window_descriptors, axis=0).astype(np.float32)


def descriptor_normalise(d: np.ndarray, dim: int = 8) -> np.ndarray:
    """
    Map raw descriptor to [-1, 1]. Slot 0 is log-space already; slots 1-7
    are in [0, 1] or normalised. Empirical clip ranges:
      slot 0: [0, 12]   (log of unique count up to ~163K)
    """
    out = np.zeros(dim, dtype=np.float32)
    out[0] = max(-1.0, min(1.0, d[0] / 6.0 - 1.0))   # log(uniq) ∈ [0,12] → [-1,1]
    for i in range(1, min(dim, 8)):
        out[i] = max(-1.0, min(1.0, d[i] * 2.0 - 1.0))
    return out


# ---------------------------------------------------------------------------
# JSONL I/O — mirrors load_file_characterizations
# ---------------------------------------------------------------------------

def load_descriptor_jsonl(
    path: str,
    dim: int = 8,
) -> Dict[str, torch.Tensor]:
    """
    Load per-file descriptor targets from JSONL.

    Each line is ``{"file": "<basename>", "descriptor": [d0, ..., dK-1]}``.
    """
    out: Dict[str, torch.Tensor] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open() as f:
        for line in f:
            row = json.loads(line)
            v = np.asarray(row["descriptor"], dtype=np.float32)
            v = descriptor_normalise(v, dim)
            out[row["file"]] = torch.from_numpy(v)
    return out


# ---------------------------------------------------------------------------
# Encoder module — embeds descriptor into a small dense vector for G
# ---------------------------------------------------------------------------

class DescriptorEncoder(nn.Module):
    """
    Small MLP: descriptor (D_desc,) → embedding (emb_dim,).

    The embedding is concatenated to the existing cond vector in G's
    z_global path.
    """

    def __init__(self, descriptor_dim: int = 8, emb_dim: int = 16,
                 hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, emb_dim),
            nn.Tanh(),     # bounded output keeps it on the same scale as cond
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return self.net(d)


# ---------------------------------------------------------------------------
# Online monitor — non-differentiable MSE between generated and target descriptors
# ---------------------------------------------------------------------------

def descriptor_monitor_mse(
    fake_obj_ids: np.ndarray,
    fake_iats: Optional[np.ndarray],
    target: np.ndarray,
    n_windows_max: int = 256,
) -> float:
    """
    Compute median descriptor for a batch of fake windows and return MSE
    against the (already-normalised) target.

    Parameters
    ----------
    fake_obj_ids : (B, T) integer obj IDs (post-Recovery, post-inverse-transform).
    fake_iats    : (B, T) IATs in same units as targets, or None.
    target       : (D,) normalised file-level descriptor (from load_descriptor_jsonl).
    n_windows_max: cap to keep online overhead bounded.
    """
    B = min(fake_obj_ids.shape[0], n_windows_max)
    descs = []
    for b in range(B):
        ids = fake_obj_ids[b]
        iat = fake_iats[b] if fake_iats is not None else None
        descs.append(compute_descriptor_for_window(ids, iat))
    file_d = aggregate_file_descriptor(np.stack(descs, axis=0))
    file_d_norm = descriptor_normalise(file_d, target.shape[0])
    return float(np.mean((file_d_norm - target) ** 2))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    obj_ids = rng.integers(0, 100, size=120)
    iats = rng.exponential(0.01, size=120)
    d = compute_descriptor_for_window(obj_ids, iats)
    print("descriptor:", d)
    print("normalised:", descriptor_normalise(d))

    enc = DescriptorEncoder()
    out = enc(torch.from_numpy(descriptor_normalise(d)).unsqueeze(0))
    print("encoded shape:", out.shape, "first 4:", out[0, :4].tolist())

    # Monitor MSE: random fake batch vs target = real descriptor.
    fake_obj_ids = rng.integers(0, 100, size=(16, 120))
    target = descriptor_normalise(d)
    mse = descriptor_monitor_mse(fake_obj_ids, None, target)
    print(f"monitor MSE: {mse:.4f}")
    print("cache_descriptor smoke test OK")
