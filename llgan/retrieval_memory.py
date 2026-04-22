"""
Retrieval memory module (IDEAS.md idea #17 — design + implementation).

Status
------
IMPL — wired into Generator/Recovery, gated by `--retrieval-memory` CLI flag.
Confirmed HARMFUL on alibaba (4x failures); do not enable without a specific
experiment plan. Backward-compatible: when disabled, checkpoint loading and
training are unaffected.

Motivation (from IDEAS.md #17)
------------------------------
Locality should be a structural decision, not a scalar penalty. Copy-path
losses showed per-timestep reuse supervision matters, but the generator has
no explicit mechanism for "reuse a recent object" vs "create a fresh
object." Reuse-rate remains the most stubborn realism gap on alibaba and
tencent. The current `obj_id_reuse` and `obj_id_stride` split already
defines the right latent decision; we now give the generator the
mechanism to act on it.

Design overview
---------------
Per-window memory bank (kept inside the module, replayed across timesteps
within a window; carry-across-windows is future work):

  K (B, M, D_key)   – object key embeddings (latest M unique objects)
  V (B, M, D_val)   – per-object value summary (size/opcode/timestamp echo)
  T (B, M)          – soft "freshness" scalar in [0, 1] (1 = just written,
                       decays geometrically each step). Used for both LRU
                       eviction and as a feature in the retrieval head.
  mask (B, M)       – which slots are filled (avoid attending to junk slots)

Per-timestep flow (called by the generator wrapper after the LSTM):

  Inputs:  h_t  (B, H)        – LSTM hidden state at step t
           gt_obj_emb (B, D_val)  – optional teacher signal during training
                                    (e.g., last-observed reuse target's
                                    feature row); used for write/teacher
                                    forcing, not for the reuse decision.

  Module computes (parameterless choices marked *):
    p_reuse_t = sigmoid(W_p · h_t)                         (B,)
    q_t       = W_q · h_t                                  (B, D_key)
    fresh_t   = W_f · h_t                                  (B, D_val)
    *attn_t   = softmax((K · q_t) / sqrt(D_key) + mask_log) (B, M)
    retrv_t   = sum_m attn_t[:,m] · V[:,m,:]               (B, D_val)
    e_t       = p_reuse_t · retrv_t + (1-p_reuse_t) · fresh_t  (B, D_val)

  Memory update (per item in batch, vectorised):
    if p_reuse_t < tau_write:   # treat as "new object"
        write fresh_t into the LRU slot (lowest T[b, m])
        T[b, m_evict] = 1.0
        mask[b, m_evict] = 1
    else:                        # reuse → bump freshness on attended slot
        T[b, argmax(attn_t)] = 1.0
    T = T * decay_rate          # geometric decay each step

  Returns:
    e_t        (B, D_val)        – fused per-step "object state"
    p_reuse_t  (B,)              – exposed for BCE aux loss
    aux        dict for diagnostics (entropy of attn, # writes, etc.)

Integration with Generator (NOT YET WIRED — separate commit):
  After LSTM produces h (B, T, H), iterate over T steps:
    e_t, p_reuse_t = self.retrieval(h[:, t, :], ...)
    h_aug[:, t, :] = projection([h[:, t], e_t])
  Then existing fc → out_act → latent → Recovery decode.

  Aux loss: BCE(p_reuse_t, gt_reuse_t) where gt_reuse_t is read from the
  preprocessed real window's `obj_id_reuse` column (rescaled from [-1,1]
  to [0,1]). Weight controlled by `--retrieval-reuse-bce-weight`.

Why M small (default 32–64)?
  Real I/O windows are short (timestep=12). A larger memory just stores
  noise. M=32 covers the typical reuse horizon on both tencent (median
  reuse distance ~10-20 within bursts) and alibaba (~5-15).

Why this stays bounded / does not fall back to "nearest-neighbour copier":
  - Memory is per-window and reinitialised each new window, so global
    overfitting to a few recent objects can't happen.
  - The reuse gate p_reuse is explicit; G must commit to a binary-ish
    choice each step, and BCE supervision keeps that calibrated.
  - Fresh-object branch is always written, so even at p_reuse=1 the next
    step still has room to introduce novelty.

Failure modes monitored:
  - p_reuse collapsing to 0 or 1 (both diagnosed via aux entropy).
  - Attention concentrating on a single slot → poor diversity (entropy
    over attn rows is a quick check).
  - Memory mask staying empty (cold-start failure) — first n_warmup steps
    write fresh_t unconditionally regardless of gate.

CLI flags (to add in train.py, NOT YET WIRED):
  --retrieval-memory                bool, default False
  --retrieval-mem-size              int,  default 32
  --retrieval-key-dim               int,  default 32
  --retrieval-val-dim               int,  default 32
  --retrieval-decay                 float, default 0.85
  --retrieval-tau-write             float, default 0.5
  --retrieval-reuse-bce-weight      float, default 1.0

References
----------
  Yuhuai Wu et al., Memorizing Transformers (ICLR 2022)
  Ali Safaya & Deniz Yuret, Neurocache (NAACL 2024)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalMemory(nn.Module):
    """
    Per-window object retrieval memory with learned reuse gate.

    Parameters
    ----------
    hidden_size : int
        Width of the upstream LSTM hidden state ``h_t``.
    mem_size : int
        Number of memory slots (M). Default 32.
    key_dim : int
        Dimensionality of attention keys/queries (D_key). Default 32.
    val_dim : int
        Dimensionality of stored value vectors (D_val). Default 32.
    decay : float
        Per-step geometric decay applied to slot freshness scalars
        (lower = faster eviction). Default 0.85.
    tau_write : float
        ``p_reuse`` threshold below which the step is treated as
        "new object" and a memory write is performed. Default 0.5.
    n_warmup : int
        Number of initial steps where writes are forced regardless of
        the reuse gate, to seed the memory. Default 4.
    """

    def __init__(
        self,
        hidden_size: int,
        mem_size: int = 32,
        key_dim: int = 32,
        val_dim: int = 32,
        decay: float = 0.85,
        tau_write: float = 0.5,
        n_warmup: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mem_size = mem_size
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.decay = decay
        self.tau_write = tau_write
        self.n_warmup = n_warmup

        # Heads: read p_reuse, query, fresh-object proposal from h_t.
        self.gate = nn.Linear(hidden_size, 1)
        self.query = nn.Linear(hidden_size, key_dim)
        self.fresh_key = nn.Linear(hidden_size, key_dim)
        self.fresh_val = nn.Linear(hidden_size, val_dim)
        # Output projection — module returns a (B, val_dim) per step that
        # callers concat with h_t before projecting to the LSTM hidden width.

        # Initial state buffers are allocated lazily on first call.
        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.gate, self.query, self.fresh_key, self.fresh_val):
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.zeros_(m.bias)

    def init_state(self, batch_size: int, device: torch.device,
                   dtype: torch.dtype = torch.float32):
        """Return a fresh empty memory state for one window."""
        K = torch.zeros(batch_size, self.mem_size, self.key_dim,
                        device=device, dtype=dtype)
        V = torch.zeros(batch_size, self.mem_size, self.val_dim,
                        device=device, dtype=dtype)
        T = torch.zeros(batch_size, self.mem_size,
                        device=device, dtype=dtype)
        mask = torch.zeros(batch_size, self.mem_size,
                           device=device, dtype=dtype)
        return {"K": K, "V": V, "T": T, "mask": mask, "step": 0}

    def forward(
        self,
        h_t: torch.Tensor,
        state: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        One retrieval/write step.

        Parameters
        ----------
        h_t : (B, hidden_size)
        state : dict produced by ``init_state`` and threaded across steps.

        Returns
        -------
        e_t : (B, val_dim)
            Fused per-step object state (retrieved + fresh blend).
        p_reuse_t : (B,)
            Exposed for BCE aux supervision.
        state : dict
            Updated memory state.
        """
        B = h_t.size(0)
        K, V, T, mask = state["K"], state["V"], state["T"], state["mask"]
        step = state["step"]

        # Heads.
        p_reuse_t = torch.sigmoid(self.gate(h_t).squeeze(-1))     # (B,)
        q_t = self.query(h_t)                                     # (B, D_key)
        fresh_k = self.fresh_key(h_t)                             # (B, D_key)
        fresh_v = self.fresh_val(h_t)                             # (B, D_val)

        # Attention over K. Mask empty slots with -inf so they get 0 weight.
        # Use a numerically safe additive mask (-1e9 instead of -inf for fp16).
        attn_logits = torch.bmm(K, q_t.unsqueeze(-1)).squeeze(-1) \
                      / (self.key_dim ** 0.5)                     # (B, M)
        mask_log = (1.0 - mask) * (-1.0e9)
        # If memory is entirely empty (B-row sum=0) the softmax of all -1e9
        # is uniform 1/M — but we ignore retrv_t in that case via gate clamping.
        attn = F.softmax(attn_logits + mask_log, dim=-1)          # (B, M)
        retrv_t = torch.bmm(attn.unsqueeze(1), V).squeeze(1)      # (B, D_val)

        # Force fresh-only behaviour during warmup or when memory empty.
        empty_row = (mask.sum(dim=-1) == 0).float()                # (B,)
        warm = 1.0 if step < self.n_warmup else 0.0
        gate = p_reuse_t * (1.0 - empty_row) * (1.0 - warm)
        e_t = gate.unsqueeze(-1) * retrv_t + (1.0 - gate).unsqueeze(-1) * fresh_v

        # ---- Memory update (vectorised, no python loop over B) ----
        # Decision per row: write iff gate < tau_write OR forced fresh.
        write = ((gate < self.tau_write) | (warm > 0)).float()    # (B,)

        # Choose eviction slot: argmin of T (oldest). Break ties by mask=0 first.
        # Score = T + mask*1e-3 keeps empty slots preferred.
        evict_score = T + mask * 1e-3
        evict_idx = torch.argmin(evict_score, dim=-1)              # (B,)

        # Build write update tensors via scatter.
        idx_oh = F.one_hot(evict_idx, self.mem_size).to(K.dtype)   # (B, M)
        write_w = write.unsqueeze(-1) * idx_oh                     # (B, M)

        # New K, V: replace evicted slot with fresh_k/fresh_v on write rows.
        K_new = K * (1.0 - write_w.unsqueeze(-1)) \
                + write_w.unsqueeze(-1) * fresh_k.unsqueeze(1)
        V_new = V * (1.0 - write_w.unsqueeze(-1)) \
                + write_w.unsqueeze(-1) * fresh_v.unsqueeze(1)

        # On reuse rows (write=0), bump T at argmax(attn). On write rows, bump
        # T at evict_idx.
        reuse_idx = torch.argmax(attn, dim=-1)                     # (B,)
        reuse_oh = F.one_hot(reuse_idx, self.mem_size).to(T.dtype)
        bump_w = write.unsqueeze(-1) * idx_oh \
                 + (1.0 - write).unsqueeze(-1) * reuse_oh          # (B, M)
        T_new = T * self.decay + bump_w * (1.0 - T * self.decay)
        # mask: any slot ever written stays masked-in.
        mask_new = torch.clamp(mask + write_w, max=1.0)

        new_state = {
            "K": K_new, "V": V_new, "T": T_new, "mask": mask_new,
            "step": step + 1,
        }
        return e_t, p_reuse_t, new_state


# ---------------------------------------------------------------------------
# Smoke test (run as `python -m llgan.retrieval_memory`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T_steps, H = 4, 12, 64
    rm = RetrievalMemory(hidden_size=H, mem_size=8, key_dim=16, val_dim=16)
    state = rm.init_state(B, torch.device("cpu"))
    h_seq = torch.randn(B, T_steps, H)
    p_reuse_log = []
    for t in range(T_steps):
        e_t, p_reuse_t, state = rm(h_seq[:, t, :], state)
        p_reuse_log.append(p_reuse_t)
        assert e_t.shape == (B, 16)
        assert p_reuse_t.shape == (B,)
    p_reuse_seq = torch.stack(p_reuse_log, dim=1)  # (B, T)
    print("retrieval_memory smoke test OK")
    print(f"  p_reuse mean: {p_reuse_seq.mean().item():.4f}")
    print(f"  p_reuse std:  {p_reuse_seq.std().item():.4f}")
    print(f"  final mask sum: {state['mask'].sum().item():.0f} / {B*8}")
    print(f"  final T range: [{state['T'].min().item():.3f}, {state['T'].max().item():.3f}]")
