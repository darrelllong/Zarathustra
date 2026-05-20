"""R303 — LSTM + WS context + stateful generation + WS-KL + aux WS head + dropout.

Extends R302 (birth-KL, 2D birth-KL, FiLM, WS-rank sampler) with five
structural improvements that close the gap to LANL r449–r463:

1. Stateful single-step generation (replaces history-window).
   R302 ran a full forward pass over `history=seq_len` tokens for every
   generated token — O(n * seq_len) cost and no long-range LSTM memory.
   R303 warms up the LSTM hidden state on the first `warmup_steps` tokens,
   then drives generation one token at a time, passing h between steps.
   Cost: O(n) — 256× speedup at seq_len=256.  Long-range context is
   carried in h rather than an explicit sliding window.  Analogue to
   LANL r446+ (stateful generation throughout).

2. Auxiliary WS-prediction head (--aux-ws-loss-weight, default 0.0).
   A second linear head `ws_head: Linear(hidden, ws_vocab)` predicts the
   next WS0 bin alongside the rank token.  The CE loss on the next WS0
   token forces the LSTM hidden state to encode WS dynamics — the key
   driver of HRC-MAE.  Weight: try 0.1–0.3.  Analogue to LANL's ws_heads
   (one per window in LANL; LLNL uses WS0 only as primary WS signal).

3. WS-conditional rank distribution loss (--ws-kl-loss-weight, default 0.0).
   During tokenization, the empirical distribution P(rank_token | ws0_bin)
   is computed as `rank_token_freq_table[ws0_bin, :]`.  During training a
   KL divergence term penalises the LSTM whenever its predicted rank
   distribution diverges from the empirically observed distribution for the
   current WS0 bin:

       KL( rank_token_freq[ws0_bin] || softmax(logits) )

   This aligns rank predictions with the observed WS-conditional cache
   behaviour and is the strongest structural driver of HRC-MAE improvement
   in LANL's r449+ sweep.  Weight: try 0.25.

4. Dropout (--dropout, default 0.0).
   Applied in two places: (a) between LSTM layers via nn.LSTM's `dropout`
   kwarg (only active when lstm_layers > 1), and (b) on the LSTM output
   before the rank head and WS head.  Reduces overfitting on small
   training sets (<100k rows).  LANL uses 0.1.

5. Short-reuse class weighting (--short-reuse-loss-weight, default 0.0).
   The CE loss weights rank-bin tokens whose midpoint falls below
   ws_windows[0] (=32) by a factor of (1 + gain).  Rank bins between
   ws_windows[0] and ws_windows[1] (=128) receive a linearly decaying
   partial bonus.  Higher-rank bins receive weight 1.  This amplifies the
   training signal for the short-reuse regime that dominates HRC-MAE on
   small-cache sizes.  Analogue to LANL r449's short_reuse_loss_weight.

All R302 features are preserved.  Old R300/R301/R302 checkpoints load fine
— new features default to off when the checkpoint key is absent.

Theory — WS-KL loss
--------------------
The main CE loss minimises token-prediction entropy but treats all WS states
equally.  In practice, HRC-MAE is almost entirely determined by the joint
distribution (rank_token, ws0_bin): caches of size s capture a fixed
fraction of objects whose reuse rank is < s.  If the LSTM predicts a rank
distribution that matches P(rank | ws0) — the empirical conditional — it
will reproduce the correct HRC up to sampling noise.  The WS-KL loss
directly penalises the difference between the predicted rank distribution
and the empirical conditional, in proportion to ws_kl_loss_weight.

Theory — stateful generation
-----------------------------
During training, each seq_len-length window starts with h=0 (LSTM state is
not carried between windows by default in the permuted-batch approach used
here).  During generation, h is initialised by running a warmup sequence
through the LSTM (so the initial h is not identically zero), then carried
across all n_records steps.  The LSTM's early positions are trained on
h=0 inputs and behave correctly in the warmup phase; the later positions
may receive an h with more accumulated context than training — in practice
this generalises well (LANL does the same), because the LSTM learns
position-invariant features and the extra h context only helps.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import os
import sys
import time
from collections import Counter, deque

import numpy as np


NEW_TOKEN = 0
DEFAULT_WS_WINDOWS = (32, 128, 512, 2048, 8192)


def _try_torch():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        return True
    except ImportError:
        return False


def make_log_edges(max_value: int, n_bins: int) -> np.ndarray:
    if max_value <= 1:
        return np.array([0, 1], dtype=np.int64)
    edges = np.unique(np.round(
        np.geomspace(1, max_value + 1, n_bins + 1)
    ).astype(np.int64))
    if edges[0] > 0:
        edges = np.concatenate([[0], edges])
    return edges


def value_to_bin(value: int, edges: np.ndarray) -> int:
    return max(0, min(int(np.searchsorted(edges, value, side='right') - 1),
                       len(edges) - 2))


def make_rank_bins(footprint: int, n_bins: int) -> np.ndarray:
    if footprint <= 1:
        return np.array([0, 2], dtype=np.int64)
    edges = np.unique(np.round(
        np.geomspace(1, max(footprint, 2), n_bins + 1)
    ).astype(np.int64))
    if edges[0] > 0:
        edges = np.concatenate([[0], edges])
    return edges


def rank_to_token(rank: int, edges: np.ndarray) -> int:
    bin_idx = bisect.bisect_right(edges, rank) - 1
    bin_idx = max(0, min(bin_idx, len(edges) - 2))
    return bin_idx + 1


def _compute_short_reuse_class_weights(rank_edges: np.ndarray,
                                        windows: list,
                                        gain: float,
                                        vocab: int) -> np.ndarray:
    """Per-token class weights for CE loss.  NEW_TOKEN gets weight 1.
    Rank-bin tokens (indices 1..K) get weight 1 + gain * bonus(mid_rank).
    bonus = 1.0 for mid < primary window, tapering to 0 at secondary window.
    """
    weights = np.ones(vocab, dtype=np.float32)
    if gain <= 0.0 or not windows:
        return weights
    primary = max(int(windows[0]), 1)
    secondary = max(int(windows[min(1, len(windows) - 1)]), primary)
    span = max(float(secondary - primary), 1.0)
    n_rank_bins = len(rank_edges) - 1
    for cls in range(n_rank_bins):
        lo = int(rank_edges[cls])
        hi = int(rank_edges[min(cls + 1, len(rank_edges) - 1)])
        mid = 0.5 * (lo + max(lo + 1, hi) - 1)
        if mid < primary:
            bonus = 1.0
        elif mid < secondary:
            bonus = 0.5 * (1.0 - (mid - primary) / span)
        else:
            bonus = 0.0
        tok_idx = cls + 1  # rank bin cls maps to vocab token cls+1
        if tok_idx < vocab:
            weights[tok_idx] = 1.0 + float(gain) * max(0.0, bonus)
    return weights


def tokenize(real_csv: str, max_rows: int, n_rank_bins: int,
             n_ws_bins: int, windows=DEFAULT_WS_WINDOWS):
    """Tokenize trace into rank_tokens + ws_tokens[t, window].

    Returns:
        rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
        bin_ranks_arr, windows, birth_rate_by_ws0, birth_count_by_ws0,
        birth_rate_by_ws01, rank_samples_by_token_ws0,
        rank_token_freq_table   (shape [n_ws0_bins, vocab])
    """
    obj_ids = []
    with open(real_csv) as f:
        r = csv.reader(f); next(r)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_ids.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)

    # LRU stack distance per access.
    last_pos = {}
    pos_sorted: list = []
    dists = np.full(n, -1, dtype=np.int64)
    for i, oid in enumerate(obj_ids):
        prev = last_pos.get(oid, -1)
        if prev >= 0:
            idx = bisect.bisect_right(pos_sorted, prev)
            dists[i] = len(pos_sorted) - idx
            del pos_sorted[bisect.bisect_left(pos_sorted, prev)]
        bisect.insort(pos_sorted, i)
        last_pos[oid] = i
    footprint = len(last_pos)
    rank_edges = make_rank_bins(footprint, n_rank_bins)

    # Working-set unique counts per window, BEFORE access at position t.
    ws_max = max(windows)
    ws_edges = make_log_edges(ws_max, n_ws_bins)
    queues = [deque() for _ in windows]
    counts = [Counter() for _ in windows]
    ws_tokens = np.empty((n, len(windows)), dtype=np.int64)
    for t in range(n):
        for wi, w in enumerate(windows):
            ws_tokens[t, wi] = value_to_bin(len(counts[wi]), ws_edges)
            queues[wi].append(obj_ids[t])
            counts[wi][obj_ids[t]] += 1
            while len(queues[wi]) > w:
                old = queues[wi].popleft()
                counts[wi][old] -= 1
                if counts[wi][old] <= 0:
                    del counts[wi][old]

    # Tokenize rank.
    K = len(rank_edges) - 1
    vocab = K + 1  # NEW_TOKEN=0, rank bins 1..K
    rank_tokens = np.empty(n, dtype=np.int64)
    bin_ranks = {k: [] for k in range(K)}
    for i, d in enumerate(dists):
        if d < 0:
            rank_tokens[i] = NEW_TOKEN
        else:
            t = rank_to_token(int(d), rank_edges)
            rank_tokens[i] = t
            bin_ranks[t - 1].append(int(d))
    bin_ranks_arr = {k: np.asarray(v, dtype=np.int64) for k, v in bin_ranks.items() if v}

    n_ws0_bins = len(ws_edges) - 1
    ws0_col = ws_tokens[:, 0]

    # 1D empirical birth rate conditioned on WS0 bin.
    birth_count_by_ws0 = np.zeros(n_ws0_bins, dtype=np.int64)
    birth_rate_by_ws0 = np.zeros(n_ws0_bins, dtype=np.float64)
    for t in range(n):
        w0 = int(ws0_col[t])
        birth_count_by_ws0[w0] += 1
        if rank_tokens[t] == NEW_TOKEN:
            birth_rate_by_ws0[w0] += 1.0
    global_birth_rate = float(np.sum(rank_tokens == NEW_TOKEN)) / n
    for w in range(n_ws0_bins):
        if birth_count_by_ws0[w] > 0:
            birth_rate_by_ws0[w] /= birth_count_by_ws0[w]
        else:
            birth_rate_by_ws0[w] = global_birth_rate

    # 2D empirical birth rate conditioned on (WS0 bin, WS1 bin).
    birth_rate_by_ws01: np.ndarray | None = None
    if len(windows) >= 2:
        ws1_col = ws_tokens[:, 1]
        birth_count_2d = np.zeros((n_ws0_bins, n_ws0_bins), dtype=np.int64)
        birth_rate_2d = np.zeros((n_ws0_bins, n_ws0_bins), dtype=np.float64)
        for t in range(n):
            w0 = int(ws0_col[t])
            w1 = int(ws1_col[t])
            birth_count_2d[w0, w1] += 1
            if rank_tokens[t] == NEW_TOKEN:
                birth_rate_2d[w0, w1] += 1.0
        for w0 in range(n_ws0_bins):
            for w1 in range(n_ws0_bins):
                if birth_count_2d[w0, w1] > 0:
                    birth_rate_2d[w0, w1] /= birth_count_2d[w0, w1]
                else:
                    birth_rate_2d[w0, w1] = birth_rate_by_ws0[w0]
        birth_rate_by_ws01 = birth_rate_2d

    # WS-conditioned rank sampler: (bin_idx, ws0_bin) → observed ranks.
    rank_samples_by_token_ws0: dict[tuple[int, int], np.ndarray] = {}
    cell_lists: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dists):
        if d >= 0:
            bin_idx = int(rank_tokens[i]) - 1  # 0-indexed rank bin
            ws0_bin = int(ws0_col[i])
            key = (bin_idx, ws0_bin)
            if key not in cell_lists:
                cell_lists[key] = []
            cell_lists[key].append(int(d))
    for key, lst in cell_lists.items():
        rank_samples_by_token_ws0[key] = np.array(lst, dtype=np.int64)

    # Empirical rank-token frequency table P(rank_token | ws0_bin).
    # Shape [n_ws0_bins, vocab].  Used for WS-KL training loss.
    rank_token_freq_counts = np.zeros((n_ws0_bins, vocab), dtype=np.float64)
    for t in range(n):
        w0 = int(ws0_col[t])
        tok = int(rank_tokens[t])
        if 0 <= tok < vocab:
            rank_token_freq_counts[w0, tok] += 1.0
    row_sums = rank_token_freq_counts.sum(axis=1, keepdims=True)
    rank_token_freq_table = rank_token_freq_counts / np.maximum(row_sums, 1.0)

    n_ws_bins_eff = len(ws_edges) - 1
    print(f"[lstm_ws tokenize] n={n:,} footprint={footprint:,} "
          f"rank_K={K} vocab={vocab} ws_K={n_ws_bins_eff} windows={list(windows)} "
          f"bin_ranks_filled={len(bin_ranks_arr)} "
          f"global_birth={global_birth_rate:.3f} "
          f"ws01_cells={len(rank_samples_by_token_ws0):,}", flush=True)
    return (rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
            bin_ranks_arr, list(windows), birth_rate_by_ws0, birth_count_by_ws0,
            birth_rate_by_ws01, rank_samples_by_token_ws0, rank_token_freq_table)


def build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed, hidden,
                lstm_layers: int = 2, film_cond: bool = False,
                dropout: float = 0.0, ws_pred_head: bool = False):
    import torch
    import torch.nn as nn

    class LstmWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.film_cond = film_cond
            self.ws_pred_head = ws_pred_head
            self.rank_emb = nn.Embedding(vocab, rank_embed)
            self.ws_emb = nn.ModuleList(
                [nn.Embedding(ws_vocab, ws_embed) for _ in range(n_windows)]
            )
            ws_total = n_windows * ws_embed
            input_dim = rank_embed + ws_total
            # inter-layer dropout only applies when lstm_layers > 1
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=lstm_layers,
                                batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, vocab)
            # FiLM post-LSTM
            if film_cond:
                self.film_gamma = nn.Linear(ws_total, hidden, bias=False)
                self.film_beta = nn.Linear(ws_total, hidden, bias=False)
            else:
                self.film_gamma = None
                self.film_beta = None
            # Auxiliary WS prediction head
            if ws_pred_head:
                self.ws_head = nn.Linear(hidden, ws_vocab)
            else:
                self.ws_head = None

        def forward(self, rank_tok, ws_tok, h=None):
            rank_x = self.rank_emb(rank_tok)
            ws_parts = [emb(ws_tok[:, :, wi]) for wi, emb in enumerate(self.ws_emb)]
            ws_flat = torch.cat(ws_parts, dim=-1)
            x = torch.cat([rank_x, ws_flat], dim=-1)
            out, h = self.lstm(x, h)
            if self.film_gamma is not None:
                out = out * (1.0 + self.film_gamma(ws_flat)) + self.film_beta(ws_flat)
            out = self.drop(out)
            logits = self.head(out)
            ws_logits = self.ws_head(out) if self.ws_head is not None else None
            return logits, ws_logits, h

    return LstmWS()


def train_model(rank_tokens, ws_tokens, vocab, ws_vocab, n_windows,
                rank_embed, ws_embed, hidden, seq_len, batch, epochs, lr,
                seed, label_smoothing: float = 0.0,
                grad_clip: float = 0.0, lr_schedule: str = "constant",
                lstm_layers: int = 2, film_cond: bool = False,
                dropout: float = 0.0,
                birth_rate_by_ws0: np.ndarray | None = None,
                birth_kl_loss_weight: float = 0.0,
                birth_rate_by_ws01: np.ndarray | None = None,
                birth_kl_loss_weight_2d: float = 0.0,
                rank_token_freq_table: np.ndarray | None = None,
                ws_kl_loss_weight: float = 0.0,
                aux_ws_loss_weight: float = 0.0,
                rank_edges: np.ndarray | None = None,
                windows: list | None = None,
                short_reuse_loss_weight: float = 0.0):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_pred = aux_ws_loss_weight > 0.0
    model = build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed,
                        hidden, lstm_layers=lstm_layers,
                        film_cond=film_cond, dropout=dropout,
                        ws_pred_head=ws_pred).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rank_t = torch.from_numpy(rank_tokens).long()
    ws_t = torch.from_numpy(ws_tokens).long()
    n_total = len(rank_tokens)
    n_train = n_total - seq_len - 1
    total_steps = max(1, (n_train // batch) * epochs)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=lr * 0.05)

    # Birth-KL tensors.
    birth_rate_t: torch.Tensor | None = None
    if birth_kl_loss_weight > 0.0 and birth_rate_by_ws0 is not None:
        birth_rate_t = torch.from_numpy(
            np.asarray(birth_rate_by_ws0, dtype=np.float32)).to(device)
    birth_rate_t_2d: torch.Tensor | None = None
    if birth_kl_loss_weight_2d > 0.0 and birth_rate_by_ws01 is not None:
        birth_rate_t_2d = torch.from_numpy(
            np.asarray(birth_rate_by_ws01, dtype=np.float32)).to(device)

    # WS-KL tensor: P(rank_token | ws0_bin), shape [n_ws0_bins, vocab].
    ws_kl_freq_t: torch.Tensor | None = None
    if ws_kl_loss_weight > 0.0 and rank_token_freq_table is not None:
        ws_kl_freq_t = torch.from_numpy(
            np.asarray(rank_token_freq_table, dtype=np.float32)).to(device)

    # Short-reuse class weights for CE loss, shape [vocab].
    class_weights_t: torch.Tensor | None = None
    if short_reuse_loss_weight > 0.0 and rank_edges is not None and windows:
        cw = _compute_short_reuse_class_weights(rank_edges, list(windows),
                                                 short_reuse_loss_weight, vocab)
        class_weights_t = torch.from_numpy(cw).to(device)

    print(f"[lstm_ws train] V={vocab} ws_V={ws_vocab} E_rank={rank_embed} "
          f"E_ws={ws_embed} H={hidden} layers={lstm_layers} film={film_cond} "
          f"dropout={dropout} seq={seq_len} batch={batch} epochs={epochs} lr={lr} "
          f"schedule={lr_schedule} label_smooth={label_smoothing} "
          f"grad_clip={grad_clip} birth_kl={birth_kl_loss_weight} "
          f"birth_kl_2d={birth_kl_loss_weight_2d} ws_kl={ws_kl_loss_weight} "
          f"aux_ws={aux_ws_loss_weight} sr_weight={short_reuse_loss_weight} "
          f"ws_pred_head={ws_pred} on {device}", flush=True)
    print(f"[lstm_ws train] params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        n_batches = n_train // batch
        running_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch:(bi + 1) * batch]
            x_rank = torch.stack([rank_t[i:i + seq_len] for i in idx]).to(device)
            x_ws = torch.stack([ws_t[i:i + seq_len] for i in idx]).to(device)
            y = torch.stack([rank_t[i + 1:i + 1 + seq_len] for i in idx]).to(device)
            y_ws = torch.stack([ws_t[i + 1:i + 1 + seq_len] for i in idx]).to(device)
            logits, ws_logits, _ = model(x_rank, x_ws)

            # Primary CE loss (rank tokens).
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                                   weight=class_weights_t,
                                   label_smoothing=label_smoothing)

            # Birth-KL loss: teach NEW-token logit to match empirical P(NEW|ws0).
            if birth_rate_t is not None or birth_rate_t_2d is not None:
                birth_logits_flat = logits.reshape(-1, vocab)[:, NEW_TOKEN]

                if birth_rate_t is not None:
                    ws0_idx = y_ws[:, :, 0].reshape(-1).clamp(
                        0, birth_rate_t.shape[0] - 1)
                    target_soft = birth_rate_t[ws0_idx]
                    birth_kl = F.binary_cross_entropy_with_logits(
                        birth_logits_flat, target_soft)
                    loss = loss + birth_kl_loss_weight * birth_kl

                if birth_rate_t_2d is not None and ws_tokens.shape[1] >= 2:
                    ws0_idx2 = y_ws[:, :, 0].reshape(-1).clamp(
                        0, birth_rate_t_2d.shape[0] - 1)
                    ws1_idx2 = y_ws[:, :, 1].reshape(-1).clamp(
                        0, birth_rate_t_2d.shape[1] - 1)
                    target_soft_2d = birth_rate_t_2d[ws0_idx2, ws1_idx2]
                    birth_kl_2d = F.binary_cross_entropy_with_logits(
                        birth_logits_flat, target_soft_2d)
                    loss = loss + birth_kl_loss_weight_2d * birth_kl_2d

            # WS-KL loss: align rank distribution to empirical P(rank|ws0).
            if ws_kl_freq_t is not None:
                ws0_idx_kl = y_ws[:, :, 0].reshape(-1).clamp(
                    0, ws_kl_freq_t.shape[0] - 1)
                target_dist = ws_kl_freq_t[ws0_idx_kl]  # [B*T, vocab]
                pred_log = F.log_softmax(logits.reshape(-1, vocab), dim=-1)
                kl = (target_dist * (torch.log(target_dist + 1e-10) - pred_log)).sum(dim=-1)
                loss = loss + ws_kl_loss_weight * kl.mean()

            # Auxiliary WS-prediction loss: predict next WS0 bin.
            if ws_logits is not None and aux_ws_loss_weight > 0.0:
                y_ws0 = y_ws[:, :, 0].reshape(-1)
                ws_loss = F.cross_entropy(ws_logits.reshape(-1, ws_vocab), y_ws0)
                loss = loss + aux_ws_loss_weight * ws_loss

            opt.zero_grad(); loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
            running_loss += float(loss.item())
            if bi % max(1, n_batches // 10) == 0:
                print(f"[lstm_ws train] ep {ep+1}/{epochs} batch {bi}/{n_batches} "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"[lstm_ws train] ep {ep+1}: avg loss = {running_loss / n_batches:.4f}",
              flush=True)
    return model


def update_ws_state(queues, counts, obj_id, windows, ws_edges):
    """After emitting obj_id, update each window's state and return ws tokens."""
    ws_now = np.empty(len(windows), dtype=np.int64)
    for wi, w in enumerate(windows):
        ws_now[wi] = value_to_bin(len(counts[wi]), ws_edges)
        queues[wi].append(obj_id)
        counts[wi][obj_id] += 1
        while len(queues[wi]) > w:
            old = queues[wi].popleft()
            counts[wi][old] -= 1
            if counts[wi][old] <= 0:
                del counts[wi][old]
    return ws_now


def generate(model, rank_edges, ws_edges, bin_ranks_arr, windows, n_records,
             seed, vocab, ws_vocab, warmup_steps: int,
             birth_rate_by_ws0: np.ndarray | None = None,
             birth_rate_blend: float = 0.0,
             birth_rate_by_ws01: np.ndarray | None = None,
             birth_rate_blend_2d: float = 0.0,
             rank_samples_by_token_ws0: dict | None = None,
             rank_sampler: str = "uniform"):
    """Stateful single-step autoregressive generation.

    Warm up the LSTM hidden state on `warmup_steps` NEW_TOKEN inputs with
    zero WS, then drive generation one token at a time, passing h between
    steps.  O(n_records) cost vs O(n_records * seq_len) in R302.
    """
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    stack: list = []
    next_new = 0
    out = np.empty(n_records, dtype=np.uint64)

    queues = [deque() for _ in windows]
    counts = [Counter() for _ in windows]

    n_ws0_bins_max = len(ws_edges) - 2  # max valid bin index

    with torch.no_grad():
        # Warm-up: feed warmup_steps NEW_TOKEN inputs to initialise h.
        ws_zero = torch.zeros(1, warmup_steps, len(windows), dtype=torch.long,
                              device=device)
        init_tok = torch.zeros(1, warmup_steps, dtype=torch.long, device=device)
        _, _, h = model(init_tok, ws_zero)

        # Current WS state (starts at zero for all windows).
        ws_now = np.zeros(len(windows), dtype=np.int64)

        prev_tok = NEW_TOKEN
        fresh = recycle = stack_n = 0

        for j in range(n_records):
            step_tok = torch.tensor(prev_tok, dtype=torch.long, device=device).reshape(1, 1)
            step_ws = torch.from_numpy(ws_now).to(device=device, dtype=torch.long).reshape(1, 1, -1)
            logits, _, h = model(step_tok, step_ws, h)
            ll = logits[0, -1].cpu().numpy()
            ll -= ll.max()
            probs = np.exp(ll); probs /= probs.sum()

            # 1D birth-rate blend: anchor P(NEW) to empirical P(NEW | ws0).
            p_now = float(probs[NEW_TOKEN])
            p_blended = p_now
            if birth_rate_blend > 0.0 and birth_rate_by_ws0 is not None:
                ws0_bin = min(int(ws_now[0]), n_ws0_bins_max)
                emp = float(birth_rate_by_ws0[ws0_bin])
                p_blended = birth_rate_blend * emp + (1.0 - birth_rate_blend) * p_now

            # 2D birth-rate blend: refine using P(NEW | ws0, ws1).
            if (birth_rate_blend_2d > 0.0 and birth_rate_by_ws01 is not None
                    and len(windows) >= 2):
                ws0_bin = min(int(ws_now[0]), birth_rate_by_ws01.shape[0] - 1)
                ws1_bin = min(int(ws_now[1]), birth_rate_by_ws01.shape[1] - 1)
                emp_2d = float(birth_rate_by_ws01[ws0_bin, ws1_bin])
                p_blended = birth_rate_blend_2d * emp_2d + (1.0 - birth_rate_blend_2d) * p_blended

            if p_blended != p_now and p_now < 1.0 - 1e-9:
                p_blended = max(0.0, min(1.0, p_blended))
                scale = (1.0 - p_blended) / (1.0 - p_now)
                probs = probs.copy()
                probs[NEW_TOKEN] = p_blended
                probs[1:] *= scale
                total = probs.sum()
                if total > 0:
                    probs /= total

            tok = int(rng.choice(vocab, p=probs))
            prev_tok = tok

            if tok == NEW_TOKEN or not stack:
                addr = next_new; next_new += 1
                stack.insert(0, addr); fresh += 1
            else:
                bin_idx = tok - 1

                # WS-conditioned rank sampling.
                rank = None
                if rank_sampler == "empirical" and rank_samples_by_token_ws0 is not None:
                    ws0_bin = min(int(ws_now[0]), n_ws0_bins_max)
                    key = (bin_idx, ws0_bin)
                    samples = rank_samples_by_token_ws0.get(key)
                    if samples is not None and len(samples) >= 5:
                        rank = int(rng.choice(samples))

                if rank is None:
                    if bin_idx in bin_ranks_arr:
                        rank = int(rng.choice(bin_ranks_arr[bin_idx]))
                    else:
                        lo = int(rank_edges[bin_idx])
                        hi = int(rank_edges[min(bin_idx + 1, len(rank_edges) - 1)])
                        rank = int(rng.integers(lo, max(hi, lo + 1)))

                if rank < 0 or rank >= len(stack):
                    addr = next_new; next_new += 1
                    stack.insert(0, addr); recycle += 1
                else:
                    addr = stack.pop(rank)
                    stack.insert(0, addr); stack_n += 1
            out[j] = addr

            ws_now = update_ws_state(queues, counts, addr, windows, ws_edges)

            if (j + 1) % 25000 == 0:
                print(f"[lstm_ws gen] {j+1:,}/{n_records:,}", flush=True)

    print(f"[lstm_ws gen] FRESH={fresh:,} ({100*fresh/n_records:.1f}%) "
          f"RECYCLE={recycle:,} IN_STACK={stack_n:,}", flush=True)
    return out


def write_csv(path, obj_ids):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


def cmd_fit(args):
    import torch
    if not _try_torch():
        print("ERROR: pytorch missing", file=sys.stderr); sys.exit(1)
    (rank_tok, ws_tok, rank_edges, ws_edges, footprint,
     bin_ranks, windows, birth_rate_by_ws0, birth_count_by_ws0,
     birth_rate_by_ws01, rank_samples_by_token_ws0,
     rank_token_freq_table) = tokenize(
        args.real, args.max_rows, args.n_bins, args.ws_bins,
        windows=tuple(int(x) for x in args.ws_windows.split(",")))
    vocab = len(rank_edges)
    ws_vocab = len(ws_edges) - 1
    model = train_model(rank_tok, ws_tok, vocab=vocab, ws_vocab=ws_vocab,
                        n_windows=len(windows),
                        rank_embed=args.rank_embed, ws_embed=args.ws_embed,
                        hidden=args.hidden, seq_len=args.seq_len,
                        batch=args.batch, epochs=args.epochs, lr=args.lr,
                        seed=args.seed,
                        label_smoothing=args.label_smoothing,
                        grad_clip=args.grad_clip,
                        lr_schedule=args.lr_schedule,
                        lstm_layers=args.lstm_layers,
                        film_cond=args.film_cond,
                        dropout=args.dropout,
                        birth_rate_by_ws0=birth_rate_by_ws0,
                        birth_kl_loss_weight=args.birth_kl_loss_weight,
                        birth_rate_by_ws01=birth_rate_by_ws01,
                        birth_kl_loss_weight_2d=args.birth_kl_loss_weight_2d,
                        rank_token_freq_table=rank_token_freq_table,
                        ws_kl_loss_weight=args.ws_kl_loss_weight,
                        aux_ws_loss_weight=args.aux_ws_loss_weight,
                        rank_edges=rank_edges,
                        windows=windows,
                        short_reuse_loss_weight=args.short_reuse_loss_weight)

    # Serialise rank_samples_by_token_ws0.
    rswt_serialised = [
        (int(k[0]), int(k[1]), v.tolist())
        for k, v in rank_samples_by_token_ws0.items()
    ]

    state = {
        "rank_edges": rank_edges, "ws_edges": ws_edges, "windows": windows,
        "vocab": vocab, "ws_vocab": ws_vocab,
        "footprint": footprint, "n_bins": args.n_bins, "ws_bins": args.ws_bins,
        "history": args.seq_len,
        "birth_rate_by_ws0": birth_rate_by_ws0.tolist(),
        "birth_count_by_ws0": birth_count_by_ws0.tolist(),
        "birth_rate_by_ws01": birth_rate_by_ws01.tolist() if birth_rate_by_ws01 is not None else None,
        "rank_samples_by_token_ws0": rswt_serialised,
        "rank_sampler": args.rank_sampler,
        "rank_token_freq_table": rank_token_freq_table.tolist(),
        "model_state": model.state_dict(),
        "model_config": {
            "rank_embed": args.rank_embed, "ws_embed": args.ws_embed,
            "hidden": args.hidden, "vocab": vocab, "ws_vocab": ws_vocab,
            "n_windows": len(windows), "lstm_layers": args.lstm_layers,
            "film_cond": args.film_cond, "dropout": args.dropout,
            "ws_pred_head": args.aux_ws_loss_weight > 0.0,
        },
        "bin_ranks_arr": bin_ranks,
    }
    torch.save(state, args.output)
    print(f"[lstm_ws fit] saved → {args.output}", flush=True)


def cmd_generate(args):
    import torch
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    cfg = state["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["vocab"], cfg["n_windows"], cfg["ws_vocab"],
                        cfg["rank_embed"], cfg["ws_embed"], cfg["hidden"],
                        lstm_layers=cfg.get("lstm_layers", 2),
                        film_cond=cfg.get("film_cond", False),
                        dropout=0.0,  # no dropout at inference
                        ws_pred_head=cfg.get("ws_pred_head", False)).to(device)
    model.load_state_dict(state["model_state"])

    birth_rate_by_ws0 = None
    raw = state.get("birth_rate_by_ws0")
    if raw:
        birth_rate_by_ws0 = np.asarray(raw, dtype=np.float64)

    birth_rate_by_ws01 = None
    raw2d = state.get("birth_rate_by_ws01")
    if raw2d is not None:
        birth_rate_by_ws01 = np.asarray(raw2d, dtype=np.float64)

    rank_samples_by_token_ws0 = None
    rswt_raw = state.get("rank_samples_by_token_ws0")
    if rswt_raw:
        rank_samples_by_token_ws0 = {
            (int(bin_idx), int(ws0_bin)): np.array(ranks, dtype=np.int64)
            for bin_idx, ws0_bin, ranks in rswt_raw
        }

    rank_sampler = state.get("rank_sampler", "uniform")
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(state.get("history", 64))

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        if len(seeds) > 1:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_s{seed}{ext}"
        else:
            out_path = args.output
        print(f"[lstm_ws gen] seed={seed} warmup={warmup_steps} → {out_path}", flush=True)
        out = generate(model,
                       state["rank_edges"], state["ws_edges"],
                       state["bin_ranks_arr"],
                       state["windows"], args.n, seed,
                       cfg["vocab"], cfg["ws_vocab"],
                       warmup_steps=warmup_steps,
                       birth_rate_by_ws0=birth_rate_by_ws0,
                       birth_rate_blend=args.birth_rate_blend,
                       birth_rate_by_ws01=birth_rate_by_ws01,
                       birth_rate_blend_2d=args.birth_rate_blend_2d,
                       rank_samples_by_token_ws0=rank_samples_by_token_ws0,
                       rank_sampler=rank_sampler)
        write_csv(out_path, out)
        print(f"[lstm_ws gen] wrote {args.n:,} → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="R303 LSTM + WS-KL + aux WS head + stateful gen + dropout")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.add_argument("--n-bins", type=int, default=200)
    pf.add_argument("--ws-bins", type=int, default=32)
    pf.add_argument("--ws-windows", default="32,128,512,2048,8192")
    pf.add_argument("--rank-embed", type=int, default=64)
    pf.add_argument("--ws-embed", type=int, default=16)
    pf.add_argument("--hidden", type=int, default=256)
    pf.add_argument("--lstm-layers", type=int, default=2)
    pf.add_argument("--seq-len", type=int, default=256)
    pf.add_argument("--batch", type=int, default=128)
    pf.add_argument("--epochs", type=int, default=25)
    pf.add_argument("--lr", type=float, default=1e-3)
    pf.add_argument("--label-smoothing", type=float, default=0.05)
    pf.add_argument("--grad-clip", type=float, default=1.0)
    pf.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--film-cond", action="store_true", default=False,
                    help="FiLM post-LSTM conditioning (residual WS modulation)")
    pf.add_argument("--dropout", type=float, default=0.0,
                    help="Dropout rate applied between LSTM layers and after output (try 0.1)")
    pf.add_argument("--birth-kl-loss-weight", type=float, default=0.0,
                    help="soft-target BCE weight on birth logit (0=off, try 0.10–0.25)")
    pf.add_argument("--birth-kl-loss-weight-2d", type=float, default=0.0,
                    help="2D soft-target BCE weight on birth logit (0=off, try 0.05–0.10)")
    pf.add_argument("--ws-kl-loss-weight", type=float, default=0.0,
                    help="KL weight aligning predicted rank dist to empirical P(rank|ws0) (try 0.25)")
    pf.add_argument("--aux-ws-loss-weight", type=float, default=0.0,
                    help="CE weight for auxiliary next-WS0 prediction head (try 0.1–0.3)")
    pf.add_argument("--short-reuse-loss-weight", type=float, default=0.0,
                    help="CE class-weight gain for short-reuse rank bins (try 1.0–3.0)")
    pf.add_argument("--rank-sampler", choices=["uniform", "empirical"],
                    default="uniform",
                    help="rank sampling strategy at generation time")
    pf.set_defaults(fn=cmd_fit)

    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seeds", default="42",
                    help="comma-separated seeds, e.g. 42,80,81,82")
    pg.add_argument("--birth-rate-blend", type=float, default=0.5)
    pg.add_argument("--birth-rate-blend-2d", type=float, default=0.25)
    pg.add_argument("--warmup-steps", type=int, default=0,
                    help="LSTM warmup steps before generation (0=use seq_len from checkpoint)")
    pg.set_defaults(fn=cmd_generate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
