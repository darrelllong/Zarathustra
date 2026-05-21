"""R305 — delta-WS conditioned empirical token blend (ahead of LANL #48).

Extends R304 with:

11. Delta-WS conditioned empirical token blend (--ws-token-blend-delta).
    Conditions the empirical rank-token blend on WS trajectory (rising / stable /
    falling) rather than just the current WS level.  Two identical ws0=W states
    with opposite trajectories have materially different rank distributions: rising
    WS concentrates fresh tokens and deep reuse; falling WS concentrates shallow
    reuse.  3D table: rank_token_freq_delta[ws0_bin][delta_sign][rank_token] where
    delta_sign ∈ {0=falling, 1=stable, 2=rising}.  Zero-refit on any R304+
    checkpoint; falls back to 1D table for sparse bins (< 8 samples).
    LLNL implements ahead of LANL idea #48 (queued, code not yet written).

R304 — cache-ladder + WS-token-blend + stack-depth conditioning + gen-time pressure.

Extends R303 (WS-KL, aux WS head, stateful gen, dropout, short-reuse weights) with:

6. Cache-ladder rank vocabulary alignment (--cache-ladder, --ladder-sizes).
   Injects mandatory rank-edge boundaries at [32,128,512,2048,8192] so no bin
   straddles a cachesim evaluation boundary.  Eliminates the structural HRC-MAE
   floor that persists regardless of model quality.  Analogue to LANL r448.

7. WS-cache-ladder WS vocabulary alignment (--ws-cache-ladder).
   Same mandatory boundaries applied to the WS bin edges so the empirical
   WS-conditioned tables use cache-aligned conditioning axes.  Analogue to LANL r451.

8. Generation-time WS-conditioned empirical rank blend (--ws-token-blend,
   --ws-token-blend-2d, --ws-blend-confidence-tau).
   At each generation step blends the LSTM's predicted rank distribution with the
   empirical P(rank_token | ws0_bin) stored in the checkpoint.  Zero-refit: works
   on any R303+ checkpoint.  Analogue to LANL r449/r450/r452.
   Combined with the WS-KL training loss this closes the calibration gap between
   MDLSTM and PhaseAtlas.

9. Generation-time WS-feedback short-reuse pressure (--short-reuse-pressure GAIN).
   Dynamic controller that biases rank-token probabilities toward short-reuse bins
   (rank < primary=32) whenever the running WS0 exceeds the empirically expected
   rate, preventing WS runaway.  Zero-refit.  Analogue to LANL's
   _apply_short_reuse_pressure().  Addresses the root cause of R301's seed-80 failure.

10. Stack-depth conditioning (--stack-depth-bins N).
    Bins the running LRU footprint (unique-object count so far) on a log-scale array
    and feeds it as an additional LSTM input embedding.  Lets the model distinguish
    early-trace (small stack) from steady-state (large stack) contexts.  Requires
    refit.  Analogue to LANL r446.

All R303 features are preserved and all old checkpoints load fine (new features
default to off when the checkpoint key is absent).

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


def make_rank_bins(footprint: int, n_bins: int,
                   cache_sizes: list | None = None) -> np.ndarray:
    if footprint <= 1:
        return np.array([0, 2], dtype=np.int64)
    edges = np.unique(np.round(
        np.geomspace(1, max(footprint, 2), n_bins + 1)
    ).astype(np.int64))
    if edges[0] > 0:
        edges = np.concatenate([[0], edges])
    if cache_sizes:
        mandatory = np.array([s for s in cache_sizes if 0 < s <= footprint],
                             dtype=np.int64)
        if len(mandatory):
            edges = np.unique(np.concatenate([edges, mandatory]))
    return edges


def make_log_edges_with_ladder(max_value: int, n_bins: int,
                               cache_sizes: list | None = None) -> np.ndarray:
    """make_log_edges with optional mandatory boundaries for WS-cache-ladder."""
    edges = make_log_edges(max_value, n_bins)
    if cache_sizes:
        mandatory = np.array([s for s in cache_sizes if 0 < s <= max_value],
                             dtype=np.int64)
        if len(mandatory):
            edges = np.unique(np.concatenate([edges, mandatory]))
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
             n_ws_bins: int, windows=DEFAULT_WS_WINDOWS,
             cache_sizes: list | None = None,
             ws_cache_sizes: list | None = None,
             stack_depth_bins: int = 0):
    """Tokenize trace into rank_tokens + ws_tokens[t, window].

    cache_sizes: if set, inject mandatory rank-edge boundaries at these values
                 (cache-ladder alignment, eliminates structural HRC-MAE floor).
    ws_cache_sizes: same for WS bin edges.
    stack_depth_bins: if > 0, also compute per-step footprint tokens.

    Returns:
        rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
        bin_ranks_arr, windows, birth_rate_by_ws0, birth_count_by_ws0,
        birth_rate_by_ws01, rank_samples_by_token_ws0,
        rank_token_freq_table        (shape [n_ws0_bins, vocab]),
        rank_token_freq_table_2d     (shape [n_ws0_bins, n_ws0_bins, vocab] or None),
        rank_token_freq_counts       (shape [n_ws0_bins]),
        fp_tokens                    (shape [n] int64 or None),
        fp_edges                     (1-D int64 array or None),
        rank_token_freq_table_delta  (shape [n_ws0_bins, 3, vocab]),
        rank_token_freq_delta_counts (shape [n_ws0_bins, 3]),
        birth_rate_by_ws0_delta      (shape [n_ws0_bins, 3]),
        birth_count_by_ws0_delta     (shape [n_ws0_bins, 3])
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
    rank_edges = make_rank_bins(footprint, n_rank_bins, cache_sizes=cache_sizes)

    # Working-set unique counts per window, BEFORE access at position t.
    ws_max = max(windows)
    ws_edges = make_log_edges_with_ladder(ws_max, n_ws_bins,
                                          cache_sizes=ws_cache_sizes)
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

    # Delta-WS birth rate P(NEW | ws0_bin, delta_sign) — R307.
    # Same trajectory encoding as rank delta table: 0=falling, 1=stable, 2=rising.
    birth_rate_by_ws0_delta = np.zeros((n_ws0_bins, 3), dtype=np.float64)
    birth_count_by_ws0_delta = np.zeros((n_ws0_bins, 3), dtype=np.int64)
    _ws0_prev_b = int(ws0_col[0])
    for t in range(n):
        w0 = int(ws0_col[t])
        d_sign = 0 if w0 < _ws0_prev_b else (2 if w0 > _ws0_prev_b else 1)
        birth_count_by_ws0_delta[w0, d_sign] += 1
        if rank_tokens[t] == NEW_TOKEN:
            birth_rate_by_ws0_delta[w0, d_sign] += 1.0
        _ws0_prev_b = w0
    for w in range(n_ws0_bins):
        for d in range(3):
            if birth_count_by_ws0_delta[w, d] > 0:
                birth_rate_by_ws0_delta[w, d] /= birth_count_by_ws0_delta[w, d]
            else:
                birth_rate_by_ws0_delta[w, d] = birth_rate_by_ws0[w]

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
    # Shape [n_ws0_bins, vocab].  Used for WS-KL training loss and gen-time blend.
    rank_token_freq_raw = np.zeros((n_ws0_bins, vocab), dtype=np.float64)
    for t in range(n):
        w0 = int(ws0_col[t])
        tok = int(rank_tokens[t])
        if 0 <= tok < vocab:
            rank_token_freq_raw[w0, tok] += 1.0
    # Per-ws0-bin total count (for confidence weighting).
    rank_token_freq_counts = rank_token_freq_raw.sum(axis=1)  # shape [n_ws0_bins]
    row_sums = rank_token_freq_raw.sum(axis=1, keepdims=True)
    rank_token_freq_table = rank_token_freq_raw / np.maximum(row_sums, 1.0)

    # Delta-WS table P(rank_token | ws0_bin, delta_sign) for gen-time blend.
    # delta_sign: 0=falling, 1=stable, 2=rising (ws0_bin vs previous step).
    N_DELTA = 3
    delta_raw = np.zeros((n_ws0_bins, N_DELTA, vocab), dtype=np.float64)
    ws0_bin_prev = int(ws0_col[0])
    for t in range(n):
        w0 = int(ws0_col[t])
        tok = int(rank_tokens[t])
        if 0 <= tok < vocab:
            d = 0 if w0 < ws0_bin_prev else (2 if w0 > ws0_bin_prev else 1)
            delta_raw[w0, d, tok] += 1.0
        ws0_bin_prev = w0
    delta_sums = delta_raw.sum(axis=2, keepdims=True)
    rank_token_freq_table_delta = delta_raw / np.maximum(delta_sums, 1.0)
    # Fall back to 1D marginal for empty delta bins.
    for d in range(N_DELTA):
        empty_d = (delta_sums[:, d, 0] == 0)
        rank_token_freq_table_delta[empty_d, d, :] = rank_token_freq_table[empty_d]
    rank_token_freq_delta_counts = delta_sums[:, :, 0]  # shape [n_ws0_bins, 3]

    # 2D empirical table P(rank_token | ws0_bin, ws1_bin) for gen-time blend.
    rank_token_freq_table_2d: np.ndarray | None = None
    if len(windows) >= 2:
        ws1_col = ws_tokens[:, 1]
        n_ws1_bins = n_ws0_bins
        freq_2d = np.zeros((n_ws0_bins, n_ws1_bins, vocab), dtype=np.float64)
        for t in range(n):
            w0 = int(ws0_col[t])
            w1 = int(ws1_col[t])
            tok = int(rank_tokens[t])
            if 0 <= tok < vocab:
                freq_2d[w0, w1, tok] += 1.0
        sums_2d = freq_2d.sum(axis=2, keepdims=True)
        rank_token_freq_table_2d = freq_2d / np.maximum(sums_2d, 1.0)
        # Fall back to 1D marginal for empty 2D bins.
        empty_mask = (sums_2d[..., 0] == 0)
        rank_token_freq_table_2d[empty_mask] = rank_token_freq_table[
            np.where(empty_mask)[0]]

    # Footprint tokens for stack-depth conditioning (idea #35 / LANL r446).
    fp_tokens: np.ndarray | None = None
    fp_edges: np.ndarray | None = None
    if stack_depth_bins > 0:
        fp_edges = make_rank_bins(footprint, stack_depth_bins)
        fp_tok_list = []
        running_fp = set()
        for oid in obj_ids:
            running_fp.add(oid)
            fp_tok_list.append(value_to_bin(len(running_fp), fp_edges))
        fp_tokens = np.array(fp_tok_list, dtype=np.int64)

    n_ws_bins_eff = len(ws_edges) - 1
    print(f"[lstm_ws tokenize] n={n:,} footprint={footprint:,} "
          f"rank_K={K} vocab={vocab} ws_K={n_ws_bins_eff} windows={list(windows)} "
          f"bin_ranks_filled={len(bin_ranks_arr)} "
          f"global_birth={global_birth_rate:.3f} "
          f"ws01_cells={len(rank_samples_by_token_ws0):,} "
          f"fp_bins={stack_depth_bins}", flush=True)
    return (rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
            bin_ranks_arr, list(windows), birth_rate_by_ws0, birth_count_by_ws0,
            birth_rate_by_ws01, rank_samples_by_token_ws0, rank_token_freq_table,
            rank_token_freq_table_2d, rank_token_freq_counts,
            fp_tokens, fp_edges,
            rank_token_freq_table_delta, rank_token_freq_delta_counts,
            birth_rate_by_ws0_delta, birth_count_by_ws0_delta)


def build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed, hidden,
                lstm_layers: int = 2, film_cond: bool = False,
                dropout: float = 0.0, ws_pred_head: bool = False,
                n_fp_bins: int = 0):
    import torch
    import torch.nn as nn

    class LstmWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.film_cond = film_cond
            self.ws_pred_head = ws_pred_head
            self.n_fp_bins = n_fp_bins
            self.rank_emb = nn.Embedding(vocab, rank_embed)
            self.ws_emb = nn.ModuleList(
                [nn.Embedding(ws_vocab, ws_embed) for _ in range(n_windows)]
            )
            ws_total = n_windows * ws_embed
            input_dim = rank_embed + ws_total
            # Stack-depth (footprint) conditioning (idea #35 / LANL r446)
            if n_fp_bins > 0:
                self.fp_emb = nn.Embedding(n_fp_bins, ws_embed)
                input_dim += ws_embed
                self.film_ws_dim = ws_total + ws_embed  # FiLM uses full ctx
            else:
                self.fp_emb = None
                self.film_ws_dim = ws_total
            # inter-layer dropout only applies when lstm_layers > 1
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=lstm_layers,
                                batch_first=True,
                                dropout=dropout if lstm_layers > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, vocab)
            # FiLM post-LSTM
            if film_cond:
                self.film_gamma = nn.Linear(self.film_ws_dim, hidden, bias=False)
                self.film_beta = nn.Linear(self.film_ws_dim, hidden, bias=False)
            else:
                self.film_gamma = None
                self.film_beta = None
            # Auxiliary WS prediction head
            if ws_pred_head:
                self.ws_head = nn.Linear(hidden, ws_vocab)
            else:
                self.ws_head = None

        def forward(self, rank_tok, ws_tok, h=None, fp_tok=None):
            rank_x = self.rank_emb(rank_tok)
            ws_parts = [emb(ws_tok[:, :, wi]) for wi, emb in enumerate(self.ws_emb)]
            ws_flat = torch.cat(ws_parts, dim=-1)
            ctx = ws_flat
            if self.fp_emb is not None and fp_tok is not None:
                fp_x = self.fp_emb(fp_tok)
                ctx = torch.cat([ws_flat, fp_x], dim=-1)
                x = torch.cat([rank_x, ctx], dim=-1)
            else:
                x = torch.cat([rank_x, ws_flat], dim=-1)
            out, h = self.lstm(x, h)
            if self.film_gamma is not None:
                out = out * (1.0 + self.film_gamma(ctx)) + self.film_beta(ctx)
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
                short_reuse_loss_weight: float = 0.0,
                fp_tokens: np.ndarray | None = None,
                n_fp_bins: int = 0,
                rank_token_freq_table_delta: np.ndarray | None = None,
                ws_delta_kl_loss_weight: float = 0.0):
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
                        ws_pred_head=ws_pred,
                        n_fp_bins=n_fp_bins).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rank_t = torch.from_numpy(rank_tokens).long()
    ws_t = torch.from_numpy(ws_tokens).long()
    fp_t = torch.from_numpy(fp_tokens).long() if fp_tokens is not None and n_fp_bins > 0 else None
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

    # Delta-WS KL tensor: P(rank|ws0,delta_sign), shape [n_ws0_bins, 3, vocab] — R308.
    ws_delta_kl_freq_t: torch.Tensor | None = None
    if ws_delta_kl_loss_weight > 0.0 and rank_token_freq_table_delta is not None:
        ws_delta_kl_freq_t = torch.from_numpy(
            np.asarray(rank_token_freq_table_delta, dtype=np.float32)).to(device)

    # Short-reuse class weights for CE loss, shape [vocab].
    class_weights_t: torch.Tensor | None = None
    if short_reuse_loss_weight > 0.0 and rank_edges is not None and windows:
        cw = _compute_short_reuse_class_weights(rank_edges, list(windows),
                                                 short_reuse_loss_weight, vocab)
        class_weights_t = torch.from_numpy(cw).to(device)

    print(f"[lstm_ws train] V={vocab} ws_V={ws_vocab} E_rank={rank_embed} "
          f"E_ws={ws_embed} H={hidden} layers={lstm_layers} film={film_cond} "
          f"dropout={dropout} fp_bins={n_fp_bins} seq={seq_len} batch={batch} "
          f"epochs={epochs} lr={lr} schedule={lr_schedule} "
          f"label_smooth={label_smoothing} grad_clip={grad_clip} "
          f"birth_kl={birth_kl_loss_weight} birth_kl_2d={birth_kl_loss_weight_2d} "
          f"ws_kl={ws_kl_loss_weight} ws_delta_kl={ws_delta_kl_loss_weight} "
          f"aux_ws={aux_ws_loss_weight} "
          f"sr_weight={short_reuse_loss_weight} ws_pred_head={ws_pred} on {device}",
          flush=True)
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
            x_fp = (torch.stack([fp_t[i:i + seq_len] for i in idx]).to(device)
                    if fp_t is not None else None)
            logits, ws_logits, _ = model(x_rank, x_ws, fp_tok=x_fp)

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

            # Delta-WS KL loss: align rank dist to P(rank|ws0, trajectory) — R308.
            if ws_delta_kl_freq_t is not None:
                # delta_sign at position t = sign(ws0[t] - ws0[t-1]).
                # For the first element of each sequence, treat as stable (delta=1).
                ws0_next = y_ws[:, :, 0]              # [B, T]
                ws0_cur  = x_ws[:, :, 0]              # [B, T] (ws0 at input step)
                delta_sign = torch.where(
                    ws0_next < ws0_cur,
                    torch.zeros_like(ws0_next),        # falling=0
                    torch.where(
                        ws0_next > ws0_cur,
                        torch.full_like(ws0_next, 2),  # rising=2
                        torch.ones_like(ws0_next),     # stable=1
                    )
                )  # [B, T], dtype long
                ws0_flat = ws0_next.reshape(-1).clamp(0, ws_delta_kl_freq_t.shape[0] - 1)
                d_flat   = delta_sign.reshape(-1).clamp(0, 2)
                target_delta = ws_delta_kl_freq_t[ws0_flat, d_flat]  # [B*T, vocab]
                pred_log_d = F.log_softmax(logits.reshape(-1, vocab), dim=-1)
                kl_d = (target_delta * (torch.log(target_delta + 1e-10) - pred_log_d)).sum(dim=-1)
                loss = loss + ws_delta_kl_loss_weight * kl_d.mean()

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
             rank_sampler: str = "uniform",
             rank_token_freq_table: np.ndarray | None = None,
             ws_token_blend: float = 0.0,
             rank_token_freq_table_2d: np.ndarray | None = None,
             ws_token_blend_2d: float = 0.0,
             rank_token_freq_counts: np.ndarray | None = None,
             ws_blend_confidence_tau: float = 0.0,
             short_reuse_pressure: float = 0.0,
             fp_edges: np.ndarray | None = None,
             rank_token_freq_table_delta: np.ndarray | None = None,
             ws_token_blend_delta: float = 0.0,
             rank_token_freq_delta_counts: np.ndarray | None = None,
             birth_rate_by_ws0_delta: np.ndarray | None = None,
             birth_rate_blend_delta: float = 0.0,
             temperature: float = 1.0,
             top_p: float = 1.0):
    """Stateful single-step autoregressive generation.

    Warm up the LSTM hidden state on `warmup_steps` NEW_TOKEN inputs with
    zero WS, then drive generation one token at a time, passing h between
    steps.  O(n_records) cost vs O(n_records * seq_len) in R302.

    New in R304:
    - ws_token_blend: blend LSTM rank-token probs with empirical P(rank|ws0).
    - ws_token_blend_2d: blend with 2D empirical P(rank|ws0,ws1).
    - ws_blend_confidence_tau: scale blend by sqrt(bucket_count/tau) for sparse bins.
    - short_reuse_pressure: dynamic WS-feedback controller biasing toward short-reuse.
    - fp_edges: footprint bin edges for stack-depth conditioning (LANL r446 analogue).
    - rank_token_freq_table_delta: P(rank|ws0,delta_sign) shape [ws0,3,vocab] (R305).
    - ws_token_blend_delta: blend weight for delta-conditioned table (try 0.3).
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
    use_fp = (fp_edges is not None and getattr(model, 'fp_emb', None) is not None)
    n_fp_max = (len(fp_edges) - 2) if use_fp else 0
    running_fp_set: set = set()

    # Pre-compute short-reuse bin mask for generation-time pressure.
    sr_boost_mask: np.ndarray | None = None
    if short_reuse_pressure > 0.0:
        primary = int(windows[0]) if windows else 32
        sr_boost_mask = np.zeros(vocab, dtype=np.float32)
        for i in range(1, vocab):
            lo = int(rank_edges[i - 1])
            hi = int(rank_edges[min(i, len(rank_edges) - 1)])
            mid = 0.5 * (lo + max(lo + 1, hi) - 1)
            if mid < primary:
                sr_boost_mask[i] = 1.0

    with torch.no_grad():
        # Warm-up: feed warmup_steps NEW_TOKEN inputs to initialise h.
        ws_zero = torch.zeros(1, warmup_steps, len(windows), dtype=torch.long,
                              device=device)
        init_tok = torch.zeros(1, warmup_steps, dtype=torch.long, device=device)
        fp_zero = (torch.zeros(1, warmup_steps, dtype=torch.long, device=device)
                   if use_fp else None)
        _, _, h = model(init_tok, ws_zero, fp_tok=fp_zero)

        # Current WS state (starts at zero for all windows).
        ws_now = np.zeros(len(windows), dtype=np.int64)
        ws0_bin_prev: int = 0  # for delta-WS conditioning

        prev_tok = NEW_TOKEN
        fresh = recycle = stack_n = 0

        for j in range(n_records):
            fp_bin = value_to_bin(len(running_fp_set), fp_edges) if use_fp else 0
            step_tok = torch.tensor(prev_tok, dtype=torch.long, device=device).reshape(1, 1)
            step_ws = torch.from_numpy(ws_now).to(device=device, dtype=torch.long).reshape(1, 1, -1)
            step_fp = (torch.tensor(fp_bin, dtype=torch.long, device=device).reshape(1, 1)
                       if use_fp else None)
            logits, _, h = model(step_tok, step_ws, h, fp_tok=step_fp)
            ll = logits[0, -1].cpu().numpy()
            if temperature != 1.0:
                ll = ll / temperature
            ll -= ll.max()
            probs = np.exp(ll); probs /= probs.sum()
            # Top-p (nucleus) sampling: zero out tokens beyond cumulative p.
            if top_p < 1.0:
                sorted_idx = np.argsort(probs)[::-1]
                cumsum = np.cumsum(probs[sorted_idx])
                cutoff = np.searchsorted(cumsum, top_p) + 1
                mask = np.zeros_like(probs)
                mask[sorted_idx[:cutoff]] = 1.0
                probs = probs * mask
                s = probs.sum()
                if s > 0:
                    probs /= s

            # Delta-sign once per step (shared by rank-delta blend and birth-delta blend).
            ws0_bin_cur = int(ws_now[0])
            ws0_delta_sign = (0 if ws0_bin_cur < ws0_bin_prev
                              else (2 if ws0_bin_cur > ws0_bin_prev else 1))

            # --- WS-token blend: anchor rank distribution to P(rank|ws0) ---
            if ws_token_blend > 0.0 and rank_token_freq_table is not None:
                ws0_bin = min(int(ws_now[0]), rank_token_freq_table.shape[0] - 1)
                alpha = ws_token_blend
                if ws_blend_confidence_tau > 0.0 and rank_token_freq_counts is not None:
                    cnt = float(rank_token_freq_counts[ws0_bin])
                    alpha *= min(1.0, np.sqrt(cnt / ws_blend_confidence_tau))
                emp1d = rank_token_freq_table[ws0_bin]
                probs = (1.0 - alpha) * probs + alpha * emp1d
                probs = np.maximum(probs, 0.0)
                probs /= probs.sum()

            # --- 2D WS-token blend: further anchor using P(rank|ws0,ws1) ---
            if (ws_token_blend_2d > 0.0 and rank_token_freq_table_2d is not None
                    and len(windows) >= 2):
                ws0_bin = min(int(ws_now[0]), rank_token_freq_table_2d.shape[0] - 1)
                ws1_bin = min(int(ws_now[1]), rank_token_freq_table_2d.shape[1] - 1)
                alpha2 = ws_token_blend_2d
                if ws_blend_confidence_tau > 0.0 and rank_token_freq_counts is not None:
                    cnt2 = float(rank_token_freq_counts[ws0_bin])
                    alpha2 *= min(1.0, np.sqrt(cnt2 / ws_blend_confidence_tau))
                emp2d = rank_token_freq_table_2d[ws0_bin, ws1_bin]
                probs = (1.0 - alpha2) * probs + alpha2 * emp2d
                probs = np.maximum(probs, 0.0)
                probs /= probs.sum()

            # --- Delta-WS blend: condition on WS trajectory (rising/stable/falling) ---
            if ws_token_blend_delta > 0.0 and rank_token_freq_table_delta is not None:
                ws0_bin = min(ws0_bin_cur, rank_token_freq_table_delta.shape[0] - 1)
                alpha_d = ws_token_blend_delta
                if ws_blend_confidence_tau > 0.0 and rank_token_freq_delta_counts is not None:
                    cnt_d = float(rank_token_freq_delta_counts[ws0_bin, ws0_delta_sign])
                    alpha_d *= min(1.0, np.sqrt(cnt_d / ws_blend_confidence_tau))
                emp_d = rank_token_freq_table_delta[ws0_bin, ws0_delta_sign]
                probs = (1.0 - alpha_d) * probs + alpha_d * emp_d
                probs = np.maximum(probs, 0.0)
                probs /= probs.sum()

            # --- Short-reuse pressure: WS-feedback controller ---
            if short_reuse_pressure > 0.0 and sr_boost_mask is not None:
                ws0_bin = min(int(ws_now[0]), n_ws0_bins_max)
                if birth_rate_by_ws0 is not None:
                    expected_fresh = float(birth_rate_by_ws0[ws0_bin])
                    actual_fresh_est = float(probs[NEW_TOKEN])
                    surplus = actual_fresh_est - expected_fresh
                    if surplus > 0.0:
                        # Boost short-reuse bins, dampen fresh
                        log_boost = short_reuse_pressure * surplus * sr_boost_mask
                        log_probs = np.log(np.maximum(probs, 1e-12))
                        log_probs += log_boost
                        log_probs -= log_probs.max()
                        probs = np.exp(log_probs)
                        probs /= probs.sum()

            # --- 1D birth-rate blend: anchor P(NEW) to empirical P(NEW | ws0) ---
            p_now = float(probs[NEW_TOKEN])
            p_blended = p_now
            if birth_rate_blend > 0.0 and birth_rate_by_ws0 is not None:
                ws0_bin = min(int(ws_now[0]), n_ws0_bins_max)
                emp = float(birth_rate_by_ws0[ws0_bin])
                p_blended = birth_rate_blend * emp + (1.0 - birth_rate_blend) * p_now

            # --- 2D birth-rate blend: refine using P(NEW | ws0, ws1) ---
            if (birth_rate_blend_2d > 0.0 and birth_rate_by_ws01 is not None
                    and len(windows) >= 2):
                ws0_bin = min(int(ws_now[0]), birth_rate_by_ws01.shape[0] - 1)
                ws1_bin = min(int(ws_now[1]), birth_rate_by_ws01.shape[1] - 1)
                emp_2d = float(birth_rate_by_ws01[ws0_bin, ws1_bin])
                p_blended = birth_rate_blend_2d * emp_2d + (1.0 - birth_rate_blend_2d) * p_blended

            # --- Delta-WS birth-rate blend: P(NEW | ws0, trajectory) — R307 ---
            if birth_rate_blend_delta > 0.0 and birth_rate_by_ws0_delta is not None:
                ws0_bin = min(int(ws_now[0]), birth_rate_by_ws0_delta.shape[0] - 1)
                emp_delta = float(birth_rate_by_ws0_delta[ws0_bin, ws0_delta_sign])
                p_blended = birth_rate_blend_delta * emp_delta + (1.0 - birth_rate_blend_delta) * p_blended

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
            if use_fp:
                running_fp_set.add(addr)

            ws0_bin_prev = int(ws_now[0])
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
    cache_sizes = ([int(x) for x in args.ladder_sizes.split(",")]
                   if args.cache_ladder else None)
    ws_cache_sizes = ([int(x) for x in args.ladder_sizes.split(",")]
                      if args.ws_cache_ladder else None)
    (rank_tok, ws_tok, rank_edges, ws_edges, footprint,
     bin_ranks, windows, birth_rate_by_ws0, birth_count_by_ws0,
     birth_rate_by_ws01, rank_samples_by_token_ws0,
     rank_token_freq_table, rank_token_freq_table_2d,
     rank_token_freq_counts, fp_tokens, fp_edges,
     rank_token_freq_table_delta, rank_token_freq_delta_counts,
     birth_rate_by_ws0_delta, _birth_count_by_ws0_delta) = tokenize(
        args.real, args.max_rows, args.n_bins, args.ws_bins,
        windows=tuple(int(x) for x in args.ws_windows.split(",")),
        cache_sizes=cache_sizes,
        ws_cache_sizes=ws_cache_sizes,
        stack_depth_bins=args.stack_depth_bins)
    vocab = len(rank_edges)
    ws_vocab = len(ws_edges) - 1
    n_fp_bins = (len(fp_edges) - 1) if fp_edges is not None else 0
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
                        short_reuse_loss_weight=args.short_reuse_loss_weight,
                        fp_tokens=fp_tokens,
                        n_fp_bins=n_fp_bins,
                        rank_token_freq_table_delta=rank_token_freq_table_delta,
                        ws_delta_kl_loss_weight=args.ws_delta_kl_loss_weight)

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
        "rank_token_freq_table_2d": (rank_token_freq_table_2d.tolist()
                                     if rank_token_freq_table_2d is not None else None),
        "rank_token_freq_counts": rank_token_freq_counts.tolist(),
        "rank_token_freq_table_delta": rank_token_freq_table_delta.tolist(),
        "rank_token_freq_delta_counts": rank_token_freq_delta_counts.tolist(),
        "birth_rate_by_ws0_delta": birth_rate_by_ws0_delta.tolist(),
        "fp_edges": fp_edges.tolist() if fp_edges is not None else None,
        "model_state": model.state_dict(),
        "model_config": {
            "rank_embed": args.rank_embed, "ws_embed": args.ws_embed,
            "hidden": args.hidden, "vocab": vocab, "ws_vocab": ws_vocab,
            "n_windows": len(windows), "lstm_layers": args.lstm_layers,
            "film_cond": args.film_cond, "dropout": args.dropout,
            "ws_pred_head": args.aux_ws_loss_weight > 0.0,
            "n_fp_bins": n_fp_bins,
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
                        ws_pred_head=cfg.get("ws_pred_head", False),
                        n_fp_bins=cfg.get("n_fp_bins", 0)).to(device)
    model.load_state_dict(state["model_state"], strict=False)

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

    rank_token_freq_table = None
    raw_freq = state.get("rank_token_freq_table")
    if raw_freq is not None:
        rank_token_freq_table = np.asarray(raw_freq, dtype=np.float64)

    rank_token_freq_table_2d = None
    raw_freq_2d = state.get("rank_token_freq_table_2d")
    if raw_freq_2d is not None:
        rank_token_freq_table_2d = np.asarray(raw_freq_2d, dtype=np.float64)

    rank_token_freq_counts = None
    raw_counts = state.get("rank_token_freq_counts")
    if raw_counts is not None:
        rank_token_freq_counts = np.asarray(raw_counts, dtype=np.float64)

    rank_token_freq_table_delta = None
    raw_delta = state.get("rank_token_freq_table_delta")
    if raw_delta is not None:
        rank_token_freq_table_delta = np.asarray(raw_delta, dtype=np.float64)

    rank_token_freq_delta_counts = None
    raw_delta_counts = state.get("rank_token_freq_delta_counts")
    if raw_delta_counts is not None:
        rank_token_freq_delta_counts = np.asarray(raw_delta_counts, dtype=np.float64)

    birth_rate_by_ws0_delta = None
    raw_birth_delta = state.get("birth_rate_by_ws0_delta")
    if raw_birth_delta is not None:
        birth_rate_by_ws0_delta = np.asarray(raw_birth_delta, dtype=np.float64)

    fp_edges = None
    raw_fp = state.get("fp_edges")
    if raw_fp is not None:
        fp_edges = np.asarray(raw_fp, dtype=np.int64)

    rank_sampler = state.get("rank_sampler", "uniform")
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else int(state.get("history", 64))

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        if len(seeds) > 1:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_s{seed}{ext}"
        else:
            out_path = args.output
        print(f"[lstm_ws gen] seed={seed} warmup={warmup_steps} "
              f"ws_blend={args.ws_token_blend} ws_blend_2d={args.ws_token_blend_2d} "
              f"ws_blend_delta={args.ws_token_blend_delta} "
              f"sr_pressure={args.short_reuse_pressure} → {out_path}", flush=True)
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
                       rank_sampler=rank_sampler,
                       rank_token_freq_table=rank_token_freq_table,
                       ws_token_blend=args.ws_token_blend,
                       rank_token_freq_table_2d=rank_token_freq_table_2d,
                       ws_token_blend_2d=args.ws_token_blend_2d,
                       rank_token_freq_counts=rank_token_freq_counts,
                       ws_blend_confidence_tau=args.ws_blend_confidence_tau,
                       short_reuse_pressure=args.short_reuse_pressure,
                       fp_edges=fp_edges,
                       rank_token_freq_table_delta=rank_token_freq_table_delta,
                       ws_token_blend_delta=args.ws_token_blend_delta,
                       rank_token_freq_delta_counts=rank_token_freq_delta_counts,
                       birth_rate_by_ws0_delta=birth_rate_by_ws0_delta,
                       birth_rate_blend_delta=args.birth_rate_blend_delta,
                       temperature=args.temperature,
                       top_p=args.top_p)
        write_csv(out_path, out)
        print(f"[lstm_ws gen] wrote {args.n:,} → {out_path}", flush=True)


def cmd_multiseed(args):
    """Fit (optionally) + generate N seeds + cachesim eval → Constitution claim panel."""
    import json
    import datetime
    import torch
    from llgan import cachesim_eval

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Fit or locate existing model
    model_path = args.model
    if args.fit:
        model_path = os.path.join(args.outdir, f"{args.tag}.pt")
        args.output = model_path
        print(f"[multiseed] fitting → {model_path}", flush=True)
        cmd_fit(args)
    if not model_path or not os.path.exists(model_path):
        raise SystemExit(
            f"multiseed: model not found: {model_path!r}. "
            "Use --fit to train or --model for an existing checkpoint."
        )

    # 2. Load model + state
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = state["model_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["vocab"], cfg["n_windows"], cfg["ws_vocab"],
                        cfg["rank_embed"], cfg["ws_embed"], cfg["hidden"],
                        lstm_layers=cfg.get("lstm_layers", 2),
                        film_cond=cfg.get("film_cond", False),
                        dropout=0.0,
                        ws_pred_head=cfg.get("ws_pred_head", False),
                        n_fp_bins=cfg.get("n_fp_bins", 0)).to(device)
    model.load_state_dict(state["model_state"], strict=False)

    def _arr(key, dtype=np.float64):
        v = state.get(key)
        return np.asarray(v, dtype=dtype) if v is not None else None

    birth_rate_by_ws0        = _arr("birth_rate_by_ws0")
    birth_rate_by_ws01       = _arr("birth_rate_by_ws01")
    birth_rate_by_ws0_delta  = _arr("birth_rate_by_ws0_delta")
    rank_token_freq_table    = _arr("rank_token_freq_table")
    rank_token_freq_table_2d = _arr("rank_token_freq_table_2d")
    rank_token_freq_counts   = _arr("rank_token_freq_counts")
    rank_token_freq_table_delta  = _arr("rank_token_freq_table_delta")
    rank_token_freq_delta_counts = _arr("rank_token_freq_delta_counts")
    fp_edges = _arr("fp_edges", dtype=np.int64)

    rswt_raw = state.get("rank_samples_by_token_ws0")
    rank_samples_by_token_ws0 = None
    if rswt_raw:
        rank_samples_by_token_ws0 = {
            (int(bi), int(wb)): np.array(ranks, dtype=np.int64)
            for bi, wb, ranks in rswt_raw
        }

    rank_sampler = state.get("rank_sampler", "uniform")
    warmup_steps = (args.warmup_steps if args.warmup_steps > 0
                    else int(state.get("history", 64)))

    # 3. Generate one CSV per seed
    seeds = [int(s) for s in args.seeds.split(",")]
    fake_csvs: list[tuple[int, str]] = []
    for seed in seeds:
        out_path = os.path.join(args.outdir, f"{args.tag}_s{seed}.csv")
        print(f"[multiseed] generate seed={seed} → {out_path}", flush=True)
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
                       rank_sampler=rank_sampler,
                       rank_token_freq_table=rank_token_freq_table,
                       ws_token_blend=args.ws_token_blend,
                       rank_token_freq_table_2d=rank_token_freq_table_2d,
                       ws_token_blend_2d=args.ws_token_blend_2d,
                       rank_token_freq_counts=rank_token_freq_counts,
                       ws_blend_confidence_tau=args.ws_blend_confidence_tau,
                       short_reuse_pressure=args.short_reuse_pressure,
                       fp_edges=fp_edges,
                       rank_token_freq_table_delta=rank_token_freq_table_delta,
                       ws_token_blend_delta=args.ws_token_blend_delta,
                       rank_token_freq_delta_counts=rank_token_freq_delta_counts,
                       birth_rate_by_ws0_delta=birth_rate_by_ws0_delta,
                       birth_rate_blend_delta=args.birth_rate_blend_delta,
                       temperature=args.temperature,
                       top_p=args.top_p)
        write_csv(out_path, out)
        print(f"[multiseed] wrote {args.n:,} rows → {out_path}", flush=True)
        fake_csvs.append((seed, out_path))

    # 4. Cachesim eval each seed
    per_seed_results: list[dict] = []
    for seed, fake_csv in fake_csvs:
        print(f"[multiseed] cachesim eval seed={seed} …", flush=True)
        report = cachesim_eval.evaluate(
            fake_csv, args.real, args.cache_sizes, args.policies)
        cachesim_eval.print_report(report)
        per_seed_results.append({
            "seed": seed,
            "fake_csv": fake_csv,
            "mean_hrc_mae": report["mean_hrc_mae"],
            "by_policy": report["by_policy"],
        })

    # 5. Aggregate: mean + range (Constitution Article VI)
    per_seed_mae = [r["mean_hrc_mae"] for r in per_seed_results]
    mean_mae  = sum(per_seed_mae) / len(per_seed_mae)
    range_mae = max(per_seed_mae) - min(per_seed_mae)
    seeds_str     = "/".join(str(s) for s in seeds)
    per_seed_str  = " / ".join(f"{v:.10f}" for v in per_seed_mae)

    print(f"\n{'='*72}", flush=True)
    print(f"TAG: {args.tag}", flush=True)
    print(f"Per-seed ({seeds_str}): {per_seed_str}", flush=True)
    print(f"4-seed mean HRC-MAE : {mean_mae:.10f}  (range {range_mae:.10f})", flush=True)
    print(f"{'='*72}", flush=True)

    # 6. Constitution-compliant markdown panel
    today = datetime.date.today().isoformat()
    md_lines = [
        f"\n## {today} — {args.tag}\n",
        f"Per-seed ({seeds_str}): {per_seed_str}  ",
        f"4-seed mean: **{mean_mae:.10f}** (range {range_mae:.10f})  ",
        "",
        "### Per-policy breakdown",
        "",
    ]
    policies = list(per_seed_results[0]["by_policy"].keys())
    for pol in policies:
        pol_mae_list = [r["by_policy"][pol]["hrc_mae"] for r in per_seed_results]
        pol_mean = sum(pol_mae_list) / len(pol_mae_list)
        pol_str  = " / ".join(f"{v:.6f}" for v in pol_mae_list)
        md_lines.append(f"- **{pol}**: {pol_str}  →  mean {pol_mean:.6f}")
    md_lines += [
        "",
        f"> cachesim sizes: `{args.cache_sizes}`  policies: `{args.policies}`",
        f"> model: `{model_path}`",
        "",
    ]
    md_panel = "\n".join(md_lines) + "\n"

    if args.append_markdown:
        with open(args.append_markdown, "a") as f:
            f.write(md_panel)
        print(f"[multiseed] appended claim panel → {args.append_markdown}", flush=True)

    # 7. JSON report
    json_payload = {
        "tag": args.tag,
        "date": today,
        "model": model_path,
        "seeds": seeds,
        "per_seed_mae": per_seed_mae,
        "mean_hrc_mae": mean_mae,
        "range_hrc_mae": range_mae,
        "cache_sizes": args.cache_sizes,
        "policies": args.policies,
        "per_seed_results": per_seed_results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(json_payload, indent=2))
        print(f"[multiseed] JSON report → {args.json_out}", flush=True)

    return json_payload


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
    pf.add_argument("--ws-delta-kl-loss-weight", type=float, default=0.0,
                    help="KL loss aligning rank dist to P(rank|ws0,trajectory) (R308; try 0.1–0.25)")
    pf.add_argument("--rank-sampler", choices=["uniform", "empirical"],
                    default="uniform",
                    help="rank sampling strategy at generation time")
    pf.add_argument("--cache-ladder", action="store_true", default=False,
                    help="inject mandatory rank-edge boundaries at cache eval sizes (eliminates HRC-MAE floor)")
    pf.add_argument("--ws-cache-ladder", action="store_true", default=False,
                    help="inject mandatory WS-edge boundaries at cache eval sizes")
    pf.add_argument("--ladder-sizes", default="32,128,512,2048,8192",
                    help="comma-separated mandatory boundary sizes for --cache-ladder/--ws-cache-ladder")
    pf.add_argument("--stack-depth-bins", type=int, default=0,
                    help="footprint (LRU stack depth) conditioning bins; 0=disabled (try 32)")
    pf.set_defaults(fn=cmd_fit)

    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seeds", default="42",
                    help="comma-separated seeds, e.g. 42,80,81,82")
    pg.add_argument("--birth-rate-blend", type=float, default=0.5)
    pg.add_argument("--birth-rate-blend-2d", type=float, default=0.25)
    pg.add_argument("--birth-rate-blend-delta", type=float, default=0.0,
                    help="delta-WS conditioned birth-rate blend; 0=off, try 0.3 (R307)")
    pg.add_argument("--ws-token-blend", type=float, default=0.0,
                    help="blend LSTM rank-dist with empirical P(rank|ws0); 0=LSTM-only, 1=empirical-only (try 0.5)")
    pg.add_argument("--ws-token-blend-2d", type=float, default=0.0,
                    help="blend with 2D empirical P(rank|ws0,ws1) (try 0.25)")
    pg.add_argument("--ws-blend-confidence-tau", type=float, default=0.0,
                    help="scale blend alpha by sqrt(bucket_count/tau); 0=disabled (try 50)")
    pg.add_argument("--ws-token-blend-delta", type=float, default=0.0,
                    help="delta-WS conditioned blend weight P(rank|ws0,trajectory); "
                         "0=off, try 0.3 (R305, ahead of LANL #48)")
    pg.add_argument("--short-reuse-pressure", type=float, default=0.0,
                    help="generation-time WS-feedback pressure toward short-reuse bins (try 1.0-3.0)")
    pg.add_argument("--warmup-steps", type=int, default=0,
                    help="LSTM warmup steps before generation (0=use seq_len from checkpoint)")
    pg.add_argument("--temperature", type=float, default=1.0,
                    help="softmax temperature at generation; <1=sharper (try 0.8–0.95), >1=flatter")
    pg.add_argument("--top-p", type=float, default=1.0,
                    help="nucleus sampling: zero-out tokens beyond cumulative prob p (try 0.95)")
    pg.set_defaults(fn=cmd_generate)

    # ------------------------------------------------------------------
    # multiseed: fit + generate(4 seeds) + cachesim eval → claim panel
    # Constitution-compliant (Article VI): 4-seed mean + range + per-
    # seed literal cachesim lines written to markdown / JSON.
    # ------------------------------------------------------------------
    pm = sub.add_parser("multiseed",
                        help="fit + generate 4 seeds + cachesim eval → claim panel")
    # Fit flags (same as 'fit' subparser)
    pm.add_argument("--real", required=True, help="reference CSV for tokenize + cachesim eval")
    pm.add_argument("--max-rows", type=int, default=0)
    pm.add_argument("--n-bins", type=int, default=200)
    pm.add_argument("--ws-bins", type=int, default=32)
    pm.add_argument("--ws-windows", default="32,128,512,2048,8192")
    pm.add_argument("--rank-embed", type=int, default=64)
    pm.add_argument("--ws-embed", type=int, default=16)
    pm.add_argument("--hidden", type=int, default=256)
    pm.add_argument("--lstm-layers", type=int, default=2)
    pm.add_argument("--seq-len", type=int, default=256)
    pm.add_argument("--batch", type=int, default=128)
    pm.add_argument("--epochs", type=int, default=25)
    pm.add_argument("--lr", type=float, default=1e-3)
    pm.add_argument("--label-smoothing", type=float, default=0.05)
    pm.add_argument("--grad-clip", type=float, default=1.0)
    pm.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine")
    pm.add_argument("--seed", type=int, default=42, help="training seed")
    pm.add_argument("--film-cond", action="store_true", default=False)
    pm.add_argument("--dropout", type=float, default=0.0)
    pm.add_argument("--birth-kl-loss-weight", type=float, default=0.0)
    pm.add_argument("--birth-kl-loss-weight-2d", type=float, default=0.0)
    pm.add_argument("--ws-kl-loss-weight", type=float, default=0.0)
    pm.add_argument("--ws-delta-kl-loss-weight", type=float, default=0.0)
    pm.add_argument("--aux-ws-loss-weight", type=float, default=0.0)
    pm.add_argument("--short-reuse-loss-weight", type=float, default=0.0)
    pm.add_argument("--rank-sampler", choices=["uniform", "empirical"], default="uniform")
    pm.add_argument("--cache-ladder", action="store_true", default=False)
    pm.add_argument("--ws-cache-ladder", action="store_true", default=False)
    pm.add_argument("--ladder-sizes", default="32,128,512,2048,8192")
    pm.add_argument("--stack-depth-bins", type=int, default=0)
    # Fit / load
    pm.add_argument("--fit", action="store_true", default=False,
                    help="train the model before generating (omit to use --model)")
    pm.add_argument("--model", default=None,
                    help="path to existing checkpoint (used if --fit is not set)")
    # Generation flags
    pm.add_argument("--n", type=int, default=1_000_000, help="records per generated trace")
    pm.add_argument("--seeds", default="42,80,81,82",
                    help="comma-separated generation seeds for multi-seed eval")
    pm.add_argument("--birth-rate-blend", type=float, default=0.5)
    pm.add_argument("--birth-rate-blend-2d", type=float, default=0.25)
    pm.add_argument("--birth-rate-blend-delta", type=float, default=0.0)
    pm.add_argument("--ws-token-blend", type=float, default=0.0)
    pm.add_argument("--ws-token-blend-2d", type=float, default=0.0)
    pm.add_argument("--ws-blend-confidence-tau", type=float, default=0.0)
    pm.add_argument("--ws-token-blend-delta", type=float, default=0.0)
    pm.add_argument("--short-reuse-pressure", type=float, default=0.0)
    pm.add_argument("--warmup-steps", type=int, default=0)
    pm.add_argument("--temperature", type=float, default=1.0)
    pm.add_argument("--top-p", type=float, default=1.0)
    # Cachesim eval flags
    pm.add_argument("--cache-sizes", default="32,128,512,2048,8192",
                    help="cache sizes for cachesim evaluation")
    pm.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car",
                    help="policies for cachesim evaluation")
    # Output flags
    pm.add_argument("--tag", required=True,
                    help="tag for this run (used in output filenames and markdown)")
    pm.add_argument("--outdir", default=".",
                    help="directory for temporary fake CSVs and JSON report")
    pm.add_argument("--append-markdown", default=None,
                    help="markdown file to append claim panel to")
    pm.add_argument("--json-out", default=None,
                    help="write full JSON report to this path")
    pm.set_defaults(fn=cmd_multiseed)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
