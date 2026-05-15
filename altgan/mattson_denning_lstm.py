"""Learned Mattson-depth / Denning-working-set sequence generator.

This module is LANL-owned ML code. It trains an autoregressive LSTM on a real
trace sequence where each event is represented by:

* a Mattson LRU stack-depth token (NEW or log-binned reuse depth), and
* Denning working-set tokens (log-binned unique counts in trailing windows).

The training loss is next-token cross entropy. Cachesim is used only after
generation, never as a training loss or an accept/reject oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NEW_TOKEN = 0
FRESH_TOKEN = 0
RECYCLE_TOKEN = 1
LEGACY_REUSE_TOKEN_OFFSET = 1
SPLIT_REUSE_TOKEN_OFFSET = 2
DEFAULT_WINDOWS = "32,128,512,2048,8192"
DEFAULT_SIZES = "32,128,512,2048,8192"
DEFAULT_POLICIES = "lru,arc,fifo,sieve,slru,car"


class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int32)

    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += int(self.tree[i])
            i -= i & -i
        return s

    def range_sum(self, lo: int, hi: int) -> int:
        if hi < lo:
            return 0
        return self.sum(hi) - self.sum(lo - 1)


@dataclass
class RealTrace:
    stream_ids: np.ndarray
    obj_ids: np.ndarray
    obj_sizes: np.ndarray
    opcodes: np.ndarray
    tenants: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.obj_ids))


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def read_real_csv(path: str | Path, max_rows: int = 0) -> RealTrace:
    stream_ids: list[int] = []
    obj_ids: list[int] = []
    obj_sizes: list[int] = []
    opcodes: list[int] = []
    tenants: list[int] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            if len(row) < 3:
                continue
            stream_ids.append(_safe_int(row, 0, 0))
            obj_ids.append(_safe_int(row, 2, 0))
            obj_sizes.append(_safe_int(row, 3, 4096))
            opcodes.append(_safe_int(row, 4, 0))
            tenants.append(_safe_int(row, 5, 0))
    return RealTrace(
        stream_ids=np.asarray(stream_ids, dtype=np.int64),
        obj_ids=np.asarray(obj_ids, dtype=np.int64),
        obj_sizes=np.asarray(obj_sizes, dtype=np.int64),
        opcodes=np.asarray(opcodes, dtype=np.int64),
        tenants=np.asarray(tenants, dtype=np.int64),
    )


def _safe_int(row: list[str], ix: int, default: int) -> int:
    try:
        return int(float(row[ix]))
    except (IndexError, TypeError, ValueError):
        return default


def make_log_edges(max_value: int, n_bins: int) -> np.ndarray:
    max_value = max(int(max_value), 1)
    if n_bins <= 1:
        return np.asarray([0, max_value + 1], dtype=np.int64)
    raw = np.unique(
        np.round(np.logspace(0, math.log10(max_value + 1), n_bins)).astype(np.int64)
    )
    edges = np.concatenate(([0], raw))
    edges = np.unique(np.clip(edges, 0, max_value + 1))
    if edges[-1] <= max_value:
        edges = np.concatenate((edges, [max_value + 1]))
    return edges.astype(np.int64)


def _coerce_ws_edges(raw):
    if isinstance(raw, np.ndarray) and raw.ndim == 1:
        return raw.astype(np.int64)
    if isinstance(raw, np.ndarray) and raw.dtype != object and raw.ndim == 2:
        return [raw[i].astype(np.int64) for i in range(raw.shape[0])]
    return [np.asarray(edge, dtype=np.int64) for edge in list(raw)]


def _ws_edges_for_window(edges, wi: int) -> np.ndarray:
    if isinstance(edges, np.ndarray) and edges.ndim == 1:
        return edges
    return np.asarray(edges[wi], dtype=np.int64)


def _ws_bins_by_window(edges, n_windows: int) -> list[int]:
    if isinstance(edges, np.ndarray) and edges.ndim == 1:
        return [max(1, len(edges) - 1)] * n_windows
    return [max(1, len(_ws_edges_for_window(edges, wi)) - 1) for wi in range(n_windows)]


def _normalize_ws_bins(ws_bins, n_windows: int) -> list[int]:
    if isinstance(ws_bins, (int, np.integer)):
        return [int(ws_bins)] * n_windows
    return [int(v) for v in list(ws_bins)]


def position_tokens(n: int, pos_bins: int) -> np.ndarray:
    """Coarse absolute trace phase tokens, disabled when pos_bins <= 0."""
    n = int(n)
    pos_bins = int(pos_bins)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    if pos_bins <= 0:
        return np.zeros(n, dtype=np.int64)
    scaled = (np.arange(n, dtype=np.int64) * pos_bins) // max(n, 1)
    return np.clip(scaled, 0, pos_bins - 1).astype(np.int64)


def position_token_at(i: int, n: int, pos_bins: int) -> int:
    pos_bins = int(pos_bins)
    if pos_bins <= 0:
        return 0
    return int(min(pos_bins - 1, max(0, (int(i) * pos_bins) // max(int(n), 1))))


def make_rank_edges(
    max_value: int,
    n_bins: int,
    exact_rank_cutoff: int = 0,
    mandatory_edges: list[int] | None = None,
) -> np.ndarray:
    """Log rank bins with optional exact-rank prefix and mandatory boundary points.

    mandatory_edges forces exact bin boundaries at specified values (e.g. cache
    ladder sizes) so rank samples never straddle a cachesim evaluation boundary.
    """
    max_value = max(int(max_value), 1)
    cutoff = max(0, min(int(exact_rank_cutoff), max_value + 1))
    if cutoff <= 1:
        base = make_log_edges(max_value, n_bins)
    else:
        exact_edges = np.arange(0, cutoff + 1, dtype=np.int64)
        if cutoff > max_value:
            base = np.arange(0, max_value + 2, dtype=np.int64)
        else:
            tail_span = max_value - cutoff + 1
            raw = np.unique(
                np.round(np.logspace(0, math.log10(tail_span + 1), n_bins)).astype(np.int64)
            )
            tail_edges = cutoff + raw
            base = np.unique(np.concatenate((exact_edges, tail_edges)))
            base = np.unique(np.clip(base, 0, max_value + 1))
            if base[-1] <= max_value:
                base = np.concatenate((base, [max_value + 1]))
    if mandatory_edges:
        extra = np.asarray(
            [int(e) for e in mandatory_edges if 0 < int(e) <= max_value],
            dtype=np.int64,
        )
        if len(extra):
            base = np.unique(np.concatenate((base, extra))).astype(np.int64)
    return base.astype(np.int64)


def running_footprint_tokens(depths: np.ndarray, fp_edges: np.ndarray) -> np.ndarray:
    """Bin the running LRU footprint (unique-object count before each event)."""
    n = len(depths)
    out = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(n):
        out[i] = value_to_bin(count, fp_edges)
        if int(depths[i]) < 0:
            count += 1
    return out


def _running_footprint_from_tokens(
    tokens: list[int],
    fp_edges: np.ndarray,
    start_count: int = 0,
) -> list[int]:
    """Bin running footprint from a token sequence (FRESH_TOKEN marks new objects)."""
    count = start_count
    out: list[int] = []
    for t in tokens:
        out.append(value_to_bin(count, fp_edges))
        if t == FRESH_TOKEN:
            count += 1
    return out


def _rank_band_bounds(windows: list[int]) -> np.ndarray:
    bounds = sorted({int(w) for w in windows if int(w) > 0})
    return np.asarray(bounds, dtype=np.int64)


def _rank_band_count(mode: str, windows: list[int]) -> int:
    if mode != "window":
        return 0
    return int(len(_rank_band_bounds(windows)) + 1)


def _rank_band_for_span(lo: int, hi: int, bounds: np.ndarray) -> int:
    if len(bounds) == 0:
        return 0
    hi = max(int(hi), int(lo) + 1)
    mid = 0.5 * (int(lo) + hi - 1)
    return int(np.searchsorted(bounds, mid, side="right"))


def _rank_token_band_map(
    rank_edges: np.ndarray,
    windows: list[int],
    vocab: int,
    reuse_token_offset: int,
    mode: str,
) -> np.ndarray:
    mapping = np.full(int(vocab), -100, dtype=np.int64)
    if mode != "window":
        return mapping
    bounds = _rank_band_bounds(windows)
    for tok in range(int(reuse_token_offset), int(vocab)):
        ix = tok - int(reuse_token_offset)
        if ix < 0 or ix >= len(rank_edges) - 1:
            continue
        lo = int(rank_edges[ix])
        hi = int(rank_edges[min(ix + 1, len(rank_edges) - 1)])
        mapping[tok] = _rank_band_for_span(lo, hi, bounds)
    return mapping


def value_to_bin(value: int, edges: np.ndarray) -> int:
    ix = int(np.searchsorted(edges, int(value), side="right") - 1)
    return max(0, min(ix, len(edges) - 2))


def rank_to_token(
    rank: int,
    edges: np.ndarray,
    reuse_token_offset: int = LEGACY_REUSE_TOKEN_OFFSET,
) -> int:
    return value_to_bin(rank, edges) + int(reuse_token_offset)


def token_to_rank(
    token: int,
    edges: np.ndarray,
    rng: np.random.Generator,
    max_rank: int | None = None,
    reuse_token_offset: int = LEGACY_REUSE_TOKEN_OFFSET,
) -> int:
    if token < reuse_token_offset:
        return -1
    ix = token - int(reuse_token_offset)
    lo = int(edges[ix])
    hi = int(edges[min(ix + 1, len(edges) - 1)])
    if max_rank is not None:
        hi = min(hi, int(max_rank) + 1)
    if hi <= lo:
        return -1
    return int(rng.integers(lo, hi))


def mattson_depths(trace: RealTrace) -> tuple[np.ndarray, int]:
    """Exact LRU stack depths using a Fenwick tree over last-touch positions."""
    keys = list(zip(trace.stream_ids.tolist(), trace.obj_ids.tolist()))
    n = len(keys)
    bit = Fenwick(n)
    last_pos: dict[tuple[int, int], int] = {}
    depths = np.full(n, -1, dtype=np.int64)
    for i, key in enumerate(keys):
        prev = last_pos.get(key)
        if prev is not None:
            depths[i] = bit.range_sum(prev + 2, n)
            bit.add(prev + 1, -1)
        bit.add(i + 1, 1)
        last_pos[key] = i
    return depths, len(last_pos)


def denning_working_sets(trace: RealTrace, windows: list[int], edges) -> np.ndarray:
    """Binned trailing-window working-set sizes before each event."""
    from collections import Counter, deque

    keys = list(zip(trace.stream_ids.tolist(), trace.obj_ids.tolist()))
    queues = [deque() for _ in windows]
    counts = [Counter() for _ in windows]
    edge_by_window = [_ws_edges_for_window(edges, wi) for wi in range(len(windows))]
    features = np.empty((len(keys), len(windows)), dtype=np.int64)

    for i, key in enumerate(keys):
        for wi, window in enumerate(windows):
            features[i, wi] = value_to_bin(len(counts[wi]), edge_by_window[wi])
            queues[wi].append(key)
            counts[wi][key] += 1
            while len(queues[wi]) > window:
                old = queues[wi].popleft()
                counts[wi][old] -= 1
                if counts[wi][old] <= 0:
                    del counts[wi][old]
    return features


def tokenize(
    trace: RealTrace,
    n_rank_bins: int,
    n_ws_bins: int,
    windows: list[int],
    recycle_rank_cap: int = 0,
    exact_rank_cutoff: int = 0,
    ws_edge_mode: str = "footprint",
    n_stack_depth_bins: int = 0,
    cache_sizes: list[int] | None = None,
    ws_cache_sizes: list[int] | None = None,
):
    depths, footprint = mattson_depths(trace)
    reuse_token_offset = SPLIT_REUSE_TOKEN_OFFSET if recycle_rank_cap > 0 else LEGACY_REUSE_TOKEN_OFFSET
    rank_max = min(footprint, max(1, int(recycle_rank_cap))) if recycle_rank_cap > 0 else footprint
    rank_edges = make_rank_edges(rank_max, n_rank_bins, exact_rank_cutoff, mandatory_edges=cache_sizes)
    if ws_edge_mode == "per-window":
        ws_edges = [make_log_edges(window, n_ws_bins) for window in windows]
        ws_edge_max = max(windows, default=footprint)
    else:
        ws_edge_max = max(windows, default=footprint) if ws_edge_mode == "max-window" else footprint
        ws_edges = make_log_edges(ws_edge_max, n_ws_bins)
        if ws_cache_sizes:
            extra_ws = np.asarray(
                [int(e) for e in ws_cache_sizes if 0 < int(e) <= int(ws_edge_max)],
                dtype=np.int64,
            )
            if len(extra_ws):
                ws_edges = np.unique(np.concatenate((ws_edges, extra_ws))).astype(np.int64)
    tokens = tokens_from_depths(depths, rank_edges, trace.n, recycle_rank_cap)
    ws_tokens = denning_working_sets(trace, windows, ws_edges)
    fresh = int((tokens == FRESH_TOKEN).sum())
    recycle = int((tokens == RECYCLE_TOKEN).sum()) if recycle_rank_cap > 0 else 0
    reuse = trace.n - fresh - recycle
    rank_samples_by_token = rank_samples_from_depths(
        depths,
        tokens,
        len(rank_edges) + (reuse_token_offset - 1),
        reuse_token_offset,
        recycle_rank_cap,
    )
    fp_bins = max(0, int(n_stack_depth_bins))
    if fp_bins > 0:
        fp_edges = make_log_edges(footprint, fp_bins)
        fp_tokens = running_footprint_tokens(depths, fp_edges)
    else:
        fp_edges = np.asarray([], dtype=np.int64)
        fp_tokens = np.zeros(trace.n, dtype=np.int64)
    vocab_size = len(rank_edges) + (reuse_token_offset - 1)
    rank_samples_by_token_fp = (
        rank_samples_from_depths_fp(
            depths, tokens, fp_tokens, vocab_size, fp_bins, reuse_token_offset, recycle_rank_cap
        )
        if fp_bins > 0
        else []
    )
    ws_bins_list = _ws_bins_by_window(ws_edges, len(windows))
    ws0_bins = ws_bins_list[0] if ws_bins_list else 1
    ws1_bins = ws_bins_list[1] if len(ws_bins_list) > 1 else 1
    rank_token_freq_table, rank_token_freq_counts = rank_token_freqs_by_ws0(tokens, ws_tokens, vocab_size, reuse_token_offset, ws0_bins)
    rank_token_freq_table_2d, rank_token_freq_counts_2d = rank_token_freqs_by_ws01(tokens, ws_tokens, vocab_size, reuse_token_offset, ws0_bins, ws1_bins)
    empirical_birth_rates, birth_rate_counts = birth_rate_by_ws0(tokens, ws_tokens, ws0_bins)
    empirical_birth_rates_2d, birth_rate_counts_2d = birth_rate_by_ws01(tokens, ws_tokens, ws0_bins, ws1_bins)
    rank_samples_by_token_ws0 = rank_samples_from_depths_ws0(
        depths, tokens, ws_tokens, vocab_size, ws0_bins, reuse_token_offset, recycle_rank_cap,
    )
    cache_ladder_str = ",".join(str(c) for c in sorted(cache_sizes)) if cache_sizes else "none"
    print(
        "[mattson_denning tokenize] "
        f"n={trace.n:,} footprint={footprint:,} rank_vocab={len(rank_edges)} "
        f"reuse_offset={reuse_token_offset} recycle_rank_cap={int(recycle_rank_cap)} "
        f"exact_rank_cutoff={int(exact_rank_cutoff)} cache_ladder={cache_ladder_str} "
        f"fresh={fresh:,} recycle={recycle:,} reuse={reuse:,} "
        f"ws_bins={ws_bins_list} ws_edge_mode={ws_edge_mode} "
        f"ws_edge_max={int(ws_edge_max)} windows={windows} fp_bins={fp_bins}",
        flush=True,
    )
    return tokens, ws_tokens, rank_edges, ws_edges, footprint, rank_samples_by_token, fp_tokens, fp_edges, rank_samples_by_token_fp, rank_token_freq_table, rank_token_freq_table_2d, rank_token_freq_counts, rank_token_freq_counts_2d, empirical_birth_rates, birth_rate_counts, rank_samples_by_token_ws0, empirical_birth_rates_2d, birth_rate_counts_2d


def tokens_from_depths(
    depths: np.ndarray,
    rank_edges: np.ndarray,
    n: int,
    recycle_rank_cap: int = 0,
) -> np.ndarray:
    tokens = np.empty(n, dtype=np.int64)
    reuse_token_offset = SPLIT_REUSE_TOKEN_OFFSET if recycle_rank_cap > 0 else LEGACY_REUSE_TOKEN_OFFSET
    for i, depth in enumerate(depths):
        if depth < 0:
            tokens[i] = FRESH_TOKEN
        elif recycle_rank_cap > 0 and int(depth) >= int(recycle_rank_cap):
            tokens[i] = RECYCLE_TOKEN
        else:
            tokens[i] = rank_to_token(int(depth), rank_edges, reuse_token_offset)
    return tokens


def rank_samples_from_depths(
    depths: np.ndarray,
    tokens: np.ndarray,
    vocab: int,
    reuse_token_offset: int,
    recycle_rank_cap: int = 0,
) -> list[list[int]]:
    samples: list[list[int]] = [[] for _ in range(vocab)]
    for depth, token in zip(depths.tolist(), tokens.tolist()):
        depth_i = int(depth)
        token_i = int(token)
        if token_i < reuse_token_offset or depth_i < 0:
            continue
        if recycle_rank_cap > 0 and depth_i >= recycle_rank_cap:
            continue
        if 0 <= token_i < vocab:
            samples[token_i].append(depth_i)
    return samples


def rank_samples_from_depths_fp(
    depths: np.ndarray,
    tokens: np.ndarray,
    fp_tokens: np.ndarray,
    vocab: int,
    fp_bins: int,
    reuse_token_offset: int,
    recycle_rank_cap: int = 0,
) -> list[list[list[int]]]:
    """Footprint-conditioned rank samples: out[token][fp_bin] = [rank, ...]."""
    samples: list[list[list[int]]] = [[[] for _ in range(fp_bins)] for _ in range(vocab)]
    for depth, token, fp_tok in zip(depths.tolist(), tokens.tolist(), fp_tokens.tolist()):
        depth_i = int(depth)
        token_i = int(token)
        fp_i = int(fp_tok)
        if token_i < reuse_token_offset or depth_i < 0:
            continue
        if recycle_rank_cap > 0 and depth_i >= recycle_rank_cap:
            continue
        if 0 <= token_i < vocab and 0 <= fp_i < fp_bins:
            samples[token_i][fp_i].append(depth_i)
    return samples


def birth_rate_by_ws01(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    ws0_bins: int,
    ws1_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical fresh-token probability conditioned on (ws0_bin, ws1_bin) jointly.

    Returns (rates, counts) where rates[w0, w1] = P(fresh | ws0_bin=w0, ws1_bin=w1)
    and counts[w0, w1] = total tokens in that (ws0, ws1) bin pair.  Empty pairs
    fall back to the ws0 marginal, then to the global rate.
    """
    fresh = np.zeros((ws0_bins, ws1_bins), dtype=np.float64)
    total = np.zeros((ws0_bins, ws1_bins), dtype=np.float64)
    for tok, ws_row in zip(tokens.tolist(), ws_tokens.tolist()):
        w0 = min(max(int(ws_row[0]), 0), ws0_bins - 1)
        w1 = min(max(int(ws_row[1]), 0), ws1_bins - 1) if len(ws_row) > 1 else 0
        total[w0, w1] += 1.0
        if int(tok) == FRESH_TOKEN:
            fresh[w0, w1] += 1.0
    global_rate = float(fresh.sum()) / max(1.0, float(total.sum()))
    # Marginal over ws1 for fallback
    marginal_fresh = fresh.sum(axis=1)
    marginal_total = total.sum(axis=1)
    marginal_rates = np.where(marginal_total > 0, marginal_fresh / marginal_total, global_rate)
    rates = np.where(total > 0, fresh / np.maximum(total, 1.0), marginal_rates[:, np.newaxis])
    return rates.astype(np.float32), total.astype(np.float32)


def rank_samples_from_depths_ws0(
    depths: np.ndarray,
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    vocab: int,
    ws0_bins: int,
    reuse_token_offset: int,
    recycle_rank_cap: int = 0,
) -> list[list[list[int]]]:
    """WS0-conditioned rank samples: out[token][ws0_bin] = [rank, ...].

    Conditions within-bin rank draws on the primary Denning WS bin so the
    rank sampler respects the same WS state that the token distribution blend
    uses.  This closes the remaining gap: token blend calibrates P(token|ws0),
    WS-rank sampler calibrates P(rank|token,ws0).
    """
    samples: list[list[list[int]]] = [[[] for _ in range(ws0_bins)] for _ in range(vocab)]
    for depth, token, ws_row in zip(depths.tolist(), tokens.tolist(), ws_tokens.tolist()):
        depth_i = int(depth)
        token_i = int(token)
        if token_i < reuse_token_offset or depth_i < 0:
            continue
        if recycle_rank_cap > 0 and depth_i >= recycle_rank_cap:
            continue
        if 0 <= token_i < vocab:
            w0 = min(max(int(ws_row[0]), 0), ws0_bins - 1)
            samples[token_i][w0].append(depth_i)
    return samples


def rank_token_freqs_by_ws0(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    vocab: int,
    reuse_token_offset: int,
    ws0_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical rank-token distribution conditioned on primary WS window bin.

    Returns (table, counts) where table has shape (ws0_bins, vocab) and counts
    has shape (ws0_bins,).  Each table row sums to 1 over [reuse_token_offset:]
    and is zero elsewhere.  Rows with no observations fall back to uniform over
    the reuse range so generation always has a valid fallback.
    """
    table = np.zeros((ws0_bins, vocab), dtype=np.float32)
    for tok, ws_row in zip(tokens.tolist(), ws_tokens.tolist()):
        tok_i = int(tok)
        if tok_i < reuse_token_offset or tok_i >= vocab:
            continue
        w0 = min(max(int(ws_row[0]), 0), ws0_bins - 1)
        table[w0, tok_i] += 1.0
    counts = table[:, reuse_token_offset:].sum(axis=1)
    n_reuse = max(0, vocab - reuse_token_offset)
    uniform_val = 1.0 / n_reuse if n_reuse > 0 else 0.0
    for w0 in range(ws0_bins):
        total = float(counts[w0])
        if total > 0.0:
            table[w0] /= total
        elif n_reuse > 0:
            table[w0, reuse_token_offset:] = uniform_val
    return table, counts.astype(np.float32)


def rank_token_freqs_by_ws01(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    vocab: int,
    reuse_token_offset: int,
    ws0_bins: int,
    ws1_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical rank-token distribution conditioned on (ws0, ws1) joint WS bins.

    Returns (table, counts) where table has shape (ws0_bins, ws1_bins, vocab)
    and counts has shape (ws0_bins, ws1_bins).  Each [w0, w1] row sums to 1
    over [reuse_token_offset:].  Empty buckets fall back to marginalising over
    ws1 (row sum for that ws0), then to uniform if still empty.
    """
    table = np.zeros((ws0_bins, ws1_bins, vocab), dtype=np.float32)
    for tok, ws_row in zip(tokens.tolist(), ws_tokens.tolist()):
        tok_i = int(tok)
        if tok_i < reuse_token_offset or tok_i >= vocab:
            continue
        w0 = min(max(int(ws_row[0]), 0), ws0_bins - 1)
        w1 = min(max(int(ws_row[1]), 0), ws1_bins - 1) if len(ws_row) > 1 else 0
        table[w0, w1, tok_i] += 1.0
    counts = table[:, :, reuse_token_offset:].sum(axis=2)  # (ws0_bins, ws1_bins)
    n_reuse = max(0, vocab - reuse_token_offset)
    uniform_val = 1.0 / n_reuse if n_reuse > 0 else 0.0
    # Marginalise over ws1 for fallback
    marginal = table.sum(axis=1)  # (ws0_bins, vocab)
    for w0 in range(ws0_bins):
        m_total = float(marginal[w0, reuse_token_offset:].sum())
        if m_total > 0.0:
            marginal[w0] /= m_total
        elif n_reuse > 0:
            marginal[w0, reuse_token_offset:] = uniform_val
    for w0 in range(ws0_bins):
        for w1 in range(ws1_bins):
            total = float(counts[w0, w1])
            if total > 0.0:
                table[w0, w1] /= total
            else:
                table[w0, w1] = marginal[w0]
    return table, counts.astype(np.float32)


def birth_rate_by_ws0(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    ws0_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical fresh-token probability conditioned on primary WS window bin.

    Returns (rates, counts) where rates[w0] = P(fresh | ws0_bin=w0) and
    counts[w0] = total tokens observed in that WS bin.  Empty bins fall back
    to the global empirical birth rate.
    """
    fresh_counts = np.zeros(ws0_bins, dtype=np.float64)
    total_counts = np.zeros(ws0_bins, dtype=np.float64)
    for tok, ws_row in zip(tokens.tolist(), ws_tokens.tolist()):
        w0 = min(max(int(ws_row[0]), 0), ws0_bins - 1)
        total_counts[w0] += 1.0
        if int(tok) == FRESH_TOKEN:
            fresh_counts[w0] += 1.0
    global_rate = float(fresh_counts.sum()) / max(1.0, float(total_counts.sum()))
    rates = np.where(total_counts > 0, fresh_counts / total_counts, global_rate)
    return rates.astype(np.float32), total_counts.astype(np.float32)


def _try_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError("PyTorch is required for training and generation") from e


def build_model(
    vocab: int,
    n_windows: int,
    ws_bins,
    token_embed: int,
    ws_embed: int,
    hidden: int,
    pos_bins: int = 0,
    pos_embed: int = 0,
    rank_bands: int = 0,
    fp_bins: int = 0,
    dropout: float = 0.0,
    film_cond: bool = False,
    lstm_layers: int = 2,
):
    torch, nn = _try_torch()
    ws_bins_by_window = _normalize_ws_bins(ws_bins, n_windows)
    use_pos = int(pos_bins) > 0 and int(pos_embed) > 0
    rank_bands = max(0, int(rank_bands))
    fp_bins = max(0, int(fp_bins))
    use_fp = fp_bins > 0
    lstm_layers = max(1, int(lstm_layers))

    class MattsonDenningLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab, token_embed)
            self.ws_emb = nn.ModuleList([nn.Embedding(n_bins, ws_embed) for n_bins in ws_bins_by_window])
            self.pos_emb = nn.Embedding(int(pos_bins), int(pos_embed)) if use_pos else None
            self.fp_emb = nn.Embedding(fp_bins, ws_embed) if use_fp else None
            input_dim = (
                token_embed
                + n_windows * ws_embed
                + (int(pos_embed) if use_pos else 0)
                + (ws_embed if use_fp else 0)
            )
            lstm_dropout = float(dropout) if float(dropout) > 0.0 and lstm_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_dim,
                hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.head = nn.Linear(hidden, vocab)
            self.birth_head = nn.Linear(hidden, 1)
            self.ws_heads = nn.ModuleList([nn.Linear(hidden, n_bins) for n_bins in ws_bins_by_window])
            self.rank_band_head = nn.Linear(hidden, rank_bands) if rank_bands > 0 else None
            ws_context_dim = n_windows * ws_embed + (ws_embed if use_fp else 0)
            self.film_gamma = nn.Linear(ws_context_dim, hidden, bias=False) if film_cond else None
            self.film_beta = nn.Linear(ws_context_dim, hidden, bias=False) if film_cond else None

        def forward(self, tok, ws, pos=None, fp=None, h=None):
            ws_embs = [emb(ws[:, :, i]) for i, emb in enumerate(self.ws_emb)]
            parts = [self.token_emb(tok)] + ws_embs
            if self.pos_emb is not None:
                if pos is None:
                    pos = tok.new_zeros(tok.shape)
                parts.append(self.pos_emb(pos))
            fp_emb_val = None
            if self.fp_emb is not None:
                if fp is None:
                    fp = tok.new_zeros(tok.shape)
                fp_emb_val = self.fp_emb(fp)
                parts.append(fp_emb_val)
            x = torch.cat(parts, dim=-1)
            out, h = self.lstm(x, h)
            if self.film_gamma is not None:
                film_parts = ws_embs + ([fp_emb_val] if fp_emb_val is not None else [])
                ws_ctx = torch.cat(film_parts, dim=-1)
                out = out * (1.0 + self.film_gamma(ws_ctx)) + self.film_beta(ws_ctx)
            rank_band_logits = self.rank_band_head(out) if self.rank_band_head is not None else None
            return (
                self.head(out),
                self.birth_head(out).squeeze(-1),
                [head(out) for head in self.ws_heads],
                rank_band_logits,
                h,
            )

    return MattsonDenningLSTM()


def train_model(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    *,
    rank_edges: np.ndarray,
    windows: list[int],
    vocab: int,
    ws_bins,
    hidden: int,
    token_embed: int,
    ws_embed: int,
    pos_bins: int,
    pos_embed: int,
    rank_band_mode: str,
    rank_band_loss_weight: float,
    seq_len: int,
    batch: int,
    epochs: int,
    lr: float,
    seed: int,
    aux_ws_loss_weight: float,
    short_reuse_loss_weight: float,
    fp_tokens: np.ndarray | None = None,
    fp_bins: int = 0,
    dropout: float = 0.0,
    lr_schedule: str = "none",
    ws_freq_table: np.ndarray | None = None,
    ws_kl_loss_weight: float = 0.0,
    ws_freq_table_2d: np.ndarray | None = None,
    ws_kl_loss_weight_2d: float = 0.0,
    film_cond: bool = False,
    lstm_layers: int = 2,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    birth_rate_table: np.ndarray | None = None,
    birth_kl_loss_weight: float = 0.0,
    birth_rate_table_2d: np.ndarray | None = None,
    birth_kl_loss_weight_2d: float = 0.0,
):
    torch, _nn = _try_torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ws_bins_by_window = _normalize_ws_bins(ws_bins, ws_tokens.shape[1])
    model = build_model(
        vocab,
        ws_tokens.shape[1],
        ws_bins_by_window,
        token_embed,
        ws_embed,
        hidden,
        pos_bins,
        pos_embed,
        _rank_band_count(rank_band_mode, windows),
        int(fp_bins),
        float(dropout),
        bool(film_cond),
        int(lstm_layers),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, epochs), eta_min=lr * 0.01
        )
    else:
        scheduler = None
    reuse_token_offset = max(LEGACY_REUSE_TOKEN_OFFSET, int(vocab) - len(rank_edges) + 1)
    reuse_class_weights = None
    if short_reuse_loss_weight > 0.0:
        weights = _short_reuse_class_weights(rank_edges, windows, short_reuse_loss_weight)
        if reuse_token_offset == SPLIT_REUSE_TOKEN_OFFSET:
            weights = np.concatenate((np.ones(1, dtype=np.float32), weights))
        reuse_class_weights = torch.from_numpy(weights).to(device)
    rank_band_map = torch.from_numpy(
        _rank_token_band_map(rank_edges, windows, vocab, reuse_token_offset, rank_band_mode)
    ).to(device)

    tok_t = torch.from_numpy(tokens.astype(np.int64))
    ws_t = torch.from_numpy(ws_tokens.astype(np.int64))
    pos_t = torch.from_numpy(position_tokens(len(tokens), pos_bins))
    fp_t = torch.from_numpy(fp_tokens.astype(np.int64)) if fp_tokens is not None else None
    ws_freq_t: torch.Tensor | None = None
    if ws_freq_table is not None and float(ws_kl_loss_weight) > 0.0:
        ws_freq_t = torch.from_numpy(
            np.asarray(ws_freq_table, dtype=np.float32)
        ).to(device)
    ws_freq_t_2d: torch.Tensor | None = None
    if ws_freq_table_2d is not None and float(ws_kl_loss_weight_2d) > 0.0:
        ws_freq_t_2d = torch.from_numpy(
            np.asarray(ws_freq_table_2d, dtype=np.float32)
        ).to(device)
    birth_rate_t: torch.Tensor | None = None
    if birth_rate_table is not None and float(birth_kl_loss_weight) > 0.0:
        birth_rate_t = torch.from_numpy(
            np.asarray(birth_rate_table, dtype=np.float32)
        ).to(device)
    birth_rate_t_2d: torch.Tensor | None = None
    if birth_rate_table_2d is not None and float(birth_kl_loss_weight_2d) > 0.0:
        birth_rate_t_2d = torch.from_numpy(
            np.asarray(birth_rate_table_2d, dtype=np.float32)
        ).to(device)
    n_train = len(tokens) - seq_len - 1
    if n_train <= 0:
        raise ValueError("not enough tokens for requested sequence length")
    n_batches = max(1, n_train // batch)
    rng = np.random.default_rng(seed)
    print(
        "[mattson_denning train] "
        f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
        f"seq={seq_len} batch={batch} epochs={epochs} n_batches={n_batches} "
        f"reuse_offset={reuse_token_offset} pos_bins={int(pos_bins)} pos_embed={int(pos_embed)} "
        f"short_reuse_loss_weight={short_reuse_loss_weight} "
        f"rank_band_mode={rank_band_mode} rank_band_loss_weight={rank_band_loss_weight} "
        f"fp_bins={int(fp_bins)} dropout={float(dropout)} lr_schedule={lr_schedule} "
        f"ws_kl_loss_weight={float(ws_kl_loss_weight)} ws_kl_loss_weight_2d={float(ws_kl_loss_weight_2d)} "
        f"film_cond={bool(film_cond)} lstm_layers={int(lstm_layers)} "
        f"label_smoothing={float(label_smoothing)} grad_clip={float(grad_clip)} "
        f"birth_kl_loss_weight={float(birth_kl_loss_weight)} birth_kl_loss_weight_2d={float(birth_kl_loss_weight_2d)}",
        flush=True,
    )

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        running = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch:(bi + 1) * batch]
            if len(idx) == 0:
                continue
            x_tok = torch.stack([tok_t[i:i + seq_len] for i in idx]).to(device)
            x_ws = torch.stack([ws_t[i:i + seq_len] for i in idx]).to(device)
            x_pos = torch.stack([pos_t[i:i + seq_len] for i in idx]).to(device)
            x_fp = torch.stack([fp_t[i:i + seq_len] for i in idx]).to(device) if fp_t is not None else None
            y = torch.stack([tok_t[i + 1:i + seq_len + 1] for i in idx]).to(device)
            y_ws = torch.stack([ws_t[i + 1:i + seq_len + 1] for i in idx]).to(device)
            logits, birth_logits, ws_logits, rank_band_logits, _ = model(x_tok, x_ws, x_pos, x_fp)
            y_birth = (y == FRESH_TOKEN).float()
            birth_loss = F.binary_cross_entropy_with_logits(
                birth_logits.reshape(-1),
                y_birth.reshape(-1),
            )
            birth_kl_loss = logits.sum() * 0.0
            if birth_rate_t is not None and float(birth_kl_loss_weight) > 0.0:
                # Soft-target BCE: teach birth head to output P(fresh|ws0) from empirical table
                ws0_idx_b = y_ws[:, :, 0].clamp(0, birth_rate_t.shape[0] - 1)
                target_birth_soft = birth_rate_t[ws0_idx_b]  # (B, T)
                birth_kl_loss = F.binary_cross_entropy_with_logits(
                    birth_logits.reshape(-1),
                    target_birth_soft.reshape(-1),
                )
            birth_kl_loss_2d = logits.sum() * 0.0
            if birth_rate_t_2d is not None and float(birth_kl_loss_weight_2d) > 0.0 and y_ws.shape[2] > 1:
                # Soft-target BCE: teach birth head to output P(fresh|ws0,ws1) from 2D empirical table
                ws0_idx_b2 = y_ws[:, :, 0].clamp(0, birth_rate_t_2d.shape[0] - 1)
                ws1_idx_b2 = y_ws[:, :, 1].clamp(0, birth_rate_t_2d.shape[1] - 1)
                target_birth_soft_2d = birth_rate_t_2d[ws0_idx_b2, ws1_idx_b2]  # (B, T)
                birth_kl_loss_2d = F.binary_cross_entropy_with_logits(
                    birth_logits.reshape(-1),
                    target_birth_soft_2d.reshape(-1),
                )
            reuse_mask = y != FRESH_TOKEN
            if bool(reuse_mask.any()):
                reuse_loss = F.cross_entropy(
                    logits[:, :, 1:][reuse_mask],
                    y[reuse_mask] - 1,
                    weight=reuse_class_weights,
                    label_smoothing=float(label_smoothing),
                )
            else:
                reuse_loss = logits.sum() * 0.0
            ws_loss = torch.stack([
                F.cross_entropy(ws_logits[wi].reshape(-1, ws_bins_by_window[wi]), y_ws[:, :, wi].reshape(-1))
                for wi in range(ws_tokens.shape[1])
            ]).mean()
            if (
                rank_band_logits is not None
                and float(rank_band_loss_weight) > 0.0
                and int(rank_band_map.max().detach().cpu()) >= 0
            ):
                y_rank_band = rank_band_map[y]
                rank_band_mask = y_rank_band >= 0
                if bool(rank_band_mask.any()):
                    rank_band_loss = F.cross_entropy(
                        rank_band_logits[rank_band_mask],
                        y_rank_band[rank_band_mask],
                    )
                else:
                    rank_band_loss = logits.sum() * 0.0
            else:
                rank_band_loss = logits.sum() * 0.0
            token_loss = birth_loss + reuse_loss
            ws_kl_loss = logits.sum() * 0.0
            if ws_freq_t is not None and float(ws_kl_loss_weight) > 0.0 and bool(reuse_mask.any()):
                # y_ws[:,: ,0] = WS before event t+1 (the event being predicted)
                ws0_idx = y_ws[:, :, 0].clamp(0, ws_freq_t.shape[0] - 1)
                target_dist = ws_freq_t[ws0_idx][:, :, reuse_token_offset:]  # (B, T, V-off)
                target_dist = target_dist / (target_dist.sum(dim=-1, keepdim=True) + 1e-10)
                pred_log = torch.log_softmax(logits[:, :, reuse_token_offset:], dim=-1)
                kl = (target_dist * (torch.log(target_dist + 1e-10) - pred_log)).sum(dim=-1)
                ws_kl_loss = kl[reuse_mask].mean()
            ws_kl_loss_2d = logits.sum() * 0.0
            if ws_freq_t_2d is not None and float(ws_kl_loss_weight_2d) > 0.0 and bool(reuse_mask.any()) and y_ws.shape[2] > 1:
                ws0_idx_2 = y_ws[:, :, 0].clamp(0, ws_freq_t_2d.shape[0] - 1)
                ws1_idx_2 = y_ws[:, :, 1].clamp(0, ws_freq_t_2d.shape[1] - 1)
                target_dist_2d = ws_freq_t_2d[ws0_idx_2, ws1_idx_2][:, :, reuse_token_offset:]
                target_dist_2d = target_dist_2d / (target_dist_2d.sum(dim=-1, keepdim=True) + 1e-10)
                pred_log_2 = torch.log_softmax(logits[:, :, reuse_token_offset:], dim=-1)
                kl_2d = (target_dist_2d * (torch.log(target_dist_2d + 1e-10) - pred_log_2)).sum(dim=-1)
                ws_kl_loss_2d = kl_2d[reuse_mask].mean()
            loss = (
                token_loss
                + float(aux_ws_loss_weight) * ws_loss
                + float(rank_band_loss_weight) * rank_band_loss
                + float(ws_kl_loss_weight) * ws_kl_loss
                + float(ws_kl_loss_weight_2d) * ws_kl_loss_2d
                + float(birth_kl_loss_weight) * birth_kl_loss
                + float(birth_kl_loss_weight_2d) * birth_kl_loss_2d
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            running += float(loss.detach().cpu())

            if bi == 0 or (bi + 1) % max(1, n_batches // 5) == 0:
                print(
                    f"[mattson_denning train] ep={ep + 1}/{epochs} "
                    f"batch={bi + 1}/{n_batches} loss={float(loss.detach().cpu()):.5f} "
                    f"tok={float(token_loss.detach().cpu()):.5f} "
                    f"birth={float(birth_loss.detach().cpu()):.5f} "
                    f"reuse={float(reuse_loss.detach().cpu()):.5f} "
                    f"ws={float(ws_loss.detach().cpu()):.5f} "
                    f"rank_band={float(rank_band_loss.detach().cpu()):.5f} "
                    f"ws_kl={float(ws_kl_loss.detach().cpu()):.5f} "
                    f"ws_kl_2d={float(ws_kl_loss_2d.detach().cpu()):.5f} "
                    f"birth_kl={float(birth_kl_loss.detach().cpu()):.5f} "
                    f"birth_kl_2d={float(birth_kl_loss_2d.detach().cpu()):.5f}",
                    flush=True,
                )
        print(
            f"[mattson_denning train] ep={ep + 1}/{epochs} avg_loss={running / n_batches:.5f}",
            flush=True,
        )
        if scheduler is not None:
            scheduler.step()
    return model


def _short_reuse_class_weights(rank_edges: np.ndarray, windows: list[int], gain: float) -> np.ndarray:
    weights = np.ones(max(0, len(rank_edges) - 1), dtype=np.float32)
    if gain <= 0.0 or len(windows) == 0:
        return weights
    primary = max(int(windows[0]), 1)
    secondary = max(int(windows[min(1, len(windows) - 1)]), primary)
    span = max(float(secondary - primary), 1.0)
    for cls in range(len(weights)):
        lo = int(rank_edges[cls])
        hi = int(rank_edges[min(cls + 1, len(rank_edges) - 1)])
        mid = 0.5 * (lo + max(lo + 1, hi) - 1)
        if mid < primary:
            bonus = 1.0
        elif mid < secondary:
            bonus = 0.5 * (1.0 - (mid - primary) / span)
        else:
            bonus = 0.0
        weights[cls] = 1.0 + float(gain) * max(0.0, bonus)
    return weights


def save_checkpoint(path: str | Path, model, meta: dict) -> None:
    torch, _nn = _try_torch()
    state = dict(meta)
    state["model_state"] = model.state_dict()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"[mattson_denning fit] wrote {path}", flush=True)


def load_checkpoint(path: str | Path):
    torch, _nn = _try_torch()
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(
        int(state["vocab"]),
        len(state["windows"]),
        state.get("ws_bins_by_window", int(state["ws_bins"])),
        int(state["token_embed"]),
        int(state["ws_embed"]),
        int(state["hidden"]),
        int(state.get("pos_bins") or 0),
        int(state.get("pos_embed") or 0),
        int(state.get("rank_bands") or 0),
        int(state.get("fp_bins") or 0),
        film_cond=bool(state.get("film_cond", False)),
        lstm_layers=int(state.get("lstm_layers") or 2),
    )
    incompatible = model.load_state_dict(state["model_state"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            "[mattson_denning load] "
            f"missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
    return state, model


def fit(args) -> None:
    windows = _parse_ints(args.ws_windows)
    trace = read_real_csv(args.real, args.max_rows)
    stack_depth_bins = int(getattr(args, "stack_depth_bins", 0))
    cache_sizes = _parse_ints(args.ladder_sizes) if getattr(args, "cache_ladder", False) else None
    ws_cache_sizes = _parse_ints(args.ladder_sizes) if getattr(args, "ws_cache_ladder", False) else None
    tokens, ws_tokens, rank_edges, ws_edges, footprint, rank_samples_by_token, fp_tokens, fp_edges, rank_samples_by_token_fp, rank_token_freq_table, rank_token_freq_table_2d, rank_token_freq_counts, rank_token_freq_counts_2d, empirical_birth_rates, birth_rate_counts, rank_samples_by_token_ws0, empirical_birth_rates_2d, birth_rate_counts_2d = tokenize(
        trace,
        args.rank_bins,
        args.ws_bins,
        windows,
        args.recycle_rank_cap,
        args.exact_rank_cutoff,
        args.ws_edge_mode,
        stack_depth_bins,
        cache_sizes,
        ws_cache_sizes,
    )
    reuse_token_offset = SPLIT_REUSE_TOKEN_OFFSET if args.recycle_rank_cap > 0 else LEGACY_REUSE_TOKEN_OFFSET
    vocab = len(rank_edges) + (reuse_token_offset - 1)
    ws_bins_by_window = _ws_bins_by_window(ws_edges, len(windows))
    rank_bands = _rank_band_count(args.rank_band_mode, windows)
    fp_bins = max(1, len(fp_edges) - 1) if len(fp_edges) > 1 else 0
    model = train_model(
        tokens,
        ws_tokens,
        rank_edges=rank_edges,
        windows=windows,
        vocab=vocab,
        ws_bins=ws_bins_by_window,
        hidden=args.hidden,
        token_embed=args.token_embed,
        ws_embed=args.ws_embed,
        pos_bins=args.pos_bins,
        pos_embed=args.pos_embed,
        rank_band_mode=args.rank_band_mode,
        rank_band_loss_weight=args.rank_band_loss_weight,
        seq_len=args.seq_len,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        aux_ws_loss_weight=args.aux_ws_loss_weight,
        short_reuse_loss_weight=args.short_reuse_loss_weight,
        fp_tokens=fp_tokens if fp_bins > 0 else None,
        fp_bins=fp_bins,
        dropout=float(getattr(args, "dropout", 0.0)),
        lr_schedule=getattr(args, "lr_schedule", "none"),
        ws_freq_table=rank_token_freq_table if float(getattr(args, "ws_kl_loss_weight", 0.0)) > 0.0 else None,
        ws_kl_loss_weight=float(getattr(args, "ws_kl_loss_weight", 0.0)),
        ws_freq_table_2d=rank_token_freq_table_2d if float(getattr(args, "ws_kl_loss_weight_2d", 0.0)) > 0.0 else None,
        ws_kl_loss_weight_2d=float(getattr(args, "ws_kl_loss_weight_2d", 0.0)),
        film_cond=bool(getattr(args, "film_cond", False)),
        lstm_layers=int(getattr(args, "lstm_layers", 2)),
        label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
        grad_clip=float(getattr(args, "grad_clip", 1.0)),
        birth_rate_table=empirical_birth_rates if float(getattr(args, "birth_kl_loss_weight", 0.0)) > 0.0 else None,
        birth_kl_loss_weight=float(getattr(args, "birth_kl_loss_weight", 0.0)),
        birth_rate_table_2d=empirical_birth_rates_2d if float(getattr(args, "birth_kl_loss_weight_2d", 0.0)) > 0.0 else None,
        birth_kl_loss_weight_2d=float(getattr(args, "birth_kl_loss_weight_2d", 0.0)),
    )
    mark_sample = {
        "obj_sizes": _sample_pool(trace.obj_sizes),
        "opcodes": _sample_pool(trace.opcodes),
        "tenants": _sample_pool(trace.tenants),
    }
    save_checkpoint(
        args.output,
        model,
        {
            "version": 7,
            "real": str(args.real),
            "max_rows": args.max_rows,
            "n_train_rows": trace.n,
            "footprint": footprint,
            "rank_edges": rank_edges,
            "ws_edges": ws_edges,
            "ws_edge_mode": args.ws_edge_mode,
            "windows": windows,
            "vocab": vocab,
            "fresh_token": FRESH_TOKEN,
            "recycle_token": RECYCLE_TOKEN if args.recycle_rank_cap > 0 else None,
            "reuse_token_offset": reuse_token_offset,
            "recycle_rank_cap": int(args.recycle_rank_cap),
            "ws_bins": max(ws_bins_by_window),
            "ws_bins_by_window": ws_bins_by_window,
            "hidden": args.hidden,
            "token_embed": args.token_embed,
            "ws_embed": args.ws_embed,
            "pos_bins": int(args.pos_bins),
            "pos_embed": int(args.pos_embed),
            "rank_band_mode": args.rank_band_mode,
            "rank_band_bounds": _rank_band_bounds(windows).tolist(),
            "rank_bands": rank_bands,
            "seq_len": args.seq_len,
            "rank_bins": args.rank_bins,
            "exact_rank_cutoff": int(args.exact_rank_cutoff),
            "mark_sample": mark_sample,
            "aux_ws_loss_weight": args.aux_ws_loss_weight,
            "short_reuse_loss_weight": args.short_reuse_loss_weight,
            "rank_band_loss_weight": args.rank_band_loss_weight,
            "rank_sampler": args.rank_sampler,
            "rank_samples_by_token": rank_samples_by_token if args.rank_sampler == "empirical" else [],
            "rank_samples_by_token_fp": rank_samples_by_token_fp if args.rank_sampler == "empirical" and fp_bins > 0 else [],
            "rank_samples_by_token_ws0": rank_samples_by_token_ws0 if args.rank_sampler == "empirical" else [],
            "fp_bins": fp_bins,
            "fp_edges": fp_edges.tolist() if len(fp_edges) > 0 else [],
            "cache_ladder_sizes": cache_sizes if cache_sizes else [],
            "ws_cache_ladder_sizes": ws_cache_sizes if ws_cache_sizes else [],
            "rank_token_freq_by_ws0": rank_token_freq_table.tolist(),
            "rank_token_freq_by_ws0_counts": rank_token_freq_counts.tolist(),
            "rank_token_freq_by_ws01": rank_token_freq_table_2d.tolist(),
            "rank_token_freq_by_ws01_counts": rank_token_freq_counts_2d.tolist(),
            "empirical_birth_rates_by_ws0": empirical_birth_rates.tolist(),
            "birth_rate_counts_by_ws0": birth_rate_counts.tolist(),
            "empirical_birth_rates_by_ws01": empirical_birth_rates_2d.tolist(),
            "birth_rate_counts_by_ws01": birth_rate_counts_2d.tolist(),
            "film_cond": bool(getattr(args, "film_cond", False)),
            "lstm_layers": int(getattr(args, "lstm_layers", 2)),
            "training_loss": "binary fresh-access cross entropy plus non-fresh action cross entropy over optional RECYCLE and Mattson-depth reuse tokens plus auxiliary next-Denning-working-set cross entropy; optional empirical within-bin Mattson-rank sampler is fit from exact real depths; optional absolute phase embeddings condition the recurrent state; optional window-band Mattson-depth auxiliary head learns cache-ladder reuse depth bands; optional running-LRU-stack-depth embedding conditions the LSTM on the current footprint growth trajectory; optional footprint-conditioned empirical rank sampler conditions within-bin rank draws on the current LRU stack depth bin; optional cache-ladder-aligned rank boundaries ensure rank bins never straddle a cachesim evaluation cache-size boundary; optional WS-conditioned empirical rank token blend calibrates the LSTM predicted token distribution against training-time empirical rank-token frequencies conditioned on the primary Denning WS bin",
        },
    )


def _sample_pool(values: np.ndarray, limit: int = 4096) -> list[int]:
    if len(values) <= limit:
        return [int(v) for v in values]
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(values), size=limit, replace=False)
    return [int(v) for v in values[idx]]


def _ws_feature_from_counts(counts, edges) -> list[int]:
    return [
        value_to_bin(len(counter), _ws_edges_for_window(edges, wi))
        for wi, (_queue, counter) in enumerate(counts)
    ]


def _slice_trace(trace: RealTrace, start: int, end: int) -> RealTrace:
    return RealTrace(
        stream_ids=trace.stream_ids[start:end],
        obj_ids=trace.obj_ids[start:end],
        obj_sizes=trace.obj_sizes[start:end],
        opcodes=trace.opcodes[start:end],
        tenants=trace.tenants[start:end],
    )


def _warmstart_context(
    state: dict,
    seed: int,
    rank_edges: np.ndarray,
    ws_edges,
    windows: list[int],
    history: int,
):
    from collections import Counter, deque

    queues = [deque() for _ in windows]
    counters = [Counter() for _ in windows]
    stack: list[int] = []
    next_new = 0
    recycle_rank_cap = int(state.get("recycle_rank_cap") or 0)
    stale_pool = deque(maxlen=max(10_000, recycle_rank_cap * 16 if recycle_rank_cap > 0 else 10_000))
    default_tokens = [NEW_TOKEN] * history
    default_ws = [[0] * len(windows) for _ in range(history)]

    real_path = state.get("real")
    if not real_path:
        return default_tokens, default_ws, queues, counters, stack, next_new, stale_pool

    try:
        trace = read_real_csv(real_path, int(state.get("max_rows") or 0))
    except OSError:
        return default_tokens, default_ws, queues, counters, stack, next_new, stale_pool

    min_warm = min(trace.n, max(history, max(windows, default=history)))
    if min_warm <= 0:
        return default_tokens, default_ws, queues, counters, stack, next_new, stale_pool

    rng = np.random.default_rng(seed + 7919)
    end = int(rng.integers(min_warm, trace.n + 1)) if trace.n > min_warm else trace.n
    warm = _slice_trace(trace, 0, end)
    depths, _footprint = mattson_depths(warm)
    warm_tokens = tokens_from_depths(depths, rank_edges, warm.n, recycle_rank_cap).tolist()
    warm_ws = denning_working_sets(warm, windows, ws_edges).tolist()

    id_map: dict[tuple[int, int], int] = {}
    for sid, oid in zip(warm.stream_ids.tolist(), warm.obj_ids.tolist()):
        real_key = (int(sid), int(oid))
        mapped = id_map.get(real_key)
        if mapped is None:
            mapped = next_new
            id_map[real_key] = mapped
            next_new += 1
        else:
            try:
                stack.remove(mapped)
            except ValueError:
                pass
        stack.insert(0, mapped)
        if recycle_rank_cap > 0 and len(stack) > recycle_rank_cap:
            stale_pool.append(int(stack.pop()))
        key = (0, mapped)
        for wi, (queue, counter) in enumerate(zip(queues, counters)):
            queue.append(key)
            counter[key] += 1
            while len(queue) > windows[wi]:
                old = queue.popleft()
                counter[old] -= 1
                if counter[old] <= 0:
                    del counter[old]

    if len(warm_tokens) < history:
        pad = history - len(warm_tokens)
        warm_tokens = [NEW_TOKEN] * pad + warm_tokens
        warm_ws = [[0] * len(windows) for _ in range(pad)] + warm_ws
    else:
        warm_tokens = warm_tokens[-history:]
        warm_ws = warm_ws[-history:]

    print(
        "[mattson_denning generate] "
        f"warmstart_rows={warm.n:,} prefix_end={end:,} unique={next_new:,} "
        f"stack={len(stack):,} stale_pool={len(stale_pool):,}",
        flush=True,
    )
    return warm_tokens, warm_ws, queues, counters, stack, next_new, stale_pool


def _birth_targets_from_real(state: dict, n_records: int) -> np.ndarray | None:
    real_path = state.get("real")
    if not real_path:
        return None
    try:
        trace = read_real_csv(real_path, n_records)
    except OSError:
        return None
    seen: set[tuple[int, int]] = set()
    targets = np.empty(n_records, dtype=np.int64)
    last = 0
    for i in range(n_records):
        if i < trace.n:
            seen.add((int(trace.stream_ids[i]), int(trace.obj_ids[i])))
            last = len(seen)
        targets[i] = last
    return targets


def _ws_targets_from_real(state: dict, n_records: int, windows: list[int]) -> np.ndarray | None:
    real_path = state.get("real")
    if not real_path:
        return None
    try:
        trace = read_real_csv(real_path, n_records)
    except OSError:
        return None
    from collections import Counter, deque

    keys = list(zip(trace.stream_ids.tolist(), trace.obj_ids.tolist()))
    queues = [deque() for _ in windows]
    counts = [Counter() for _ in windows]
    targets = np.empty((n_records, len(windows)), dtype=np.int64)
    last = np.zeros(len(windows), dtype=np.int64)
    for i in range(n_records):
        if i < len(keys):
            for wi, window in enumerate(windows):
                last[wi] = len(counts[wi])
                queues[wi].append(keys[i])
                counts[wi][keys[i]] += 1
                while len(queues[wi]) > window:
                    old = queues[wi].popleft()
                    counts[wi][old] -= 1
                    if counts[wi][old] <= 0:
                        del counts[wi][old]
        targets[i] = last
    return targets


def _ws_target_from_logits(
    ws_logits,
    ws_edges,
    windows: list[int],
    mode: str = "clamp",
) -> np.ndarray:
    """Convert learned next-working-set logits into feasible per-window counts."""
    torch, _nn = _try_torch()
    targets: list[float] = []
    for wi, head_logits in enumerate(ws_logits):
        edges = _ws_edges_for_window(ws_edges, wi)
        mids = 0.5 * (edges[:-1].astype(np.float64) + edges[1:].astype(np.float64) - 1.0)
        mids = np.maximum(mids, 0.0)
        window_cap = float(windows[wi]) if wi < len(windows) else float(np.max(mids))
        local_mids = mids[: head_logits.shape[-1]]
        probs = torch.softmax(head_logits[0, -1], dim=-1).detach().cpu().numpy()
        if mode == "masked":
            feasible = local_mids <= window_cap
            if np.any(feasible):
                local_probs = probs.copy()
                local_probs[~feasible[: len(local_probs)]] = 0.0
                total = float(local_probs.sum())
                if total > 0.0:
                    local_probs /= total
                    targets.append(float(np.dot(local_probs, local_mids[: len(local_probs)])))
                    continue
        feasible_mids = np.minimum(local_mids, window_cap)
        targets.append(float(np.dot(probs, feasible_mids[: len(probs)])))
    return np.asarray(targets, dtype=np.float64)


def _force_emitted_reuse(
    stack: list[int],
    emitted_seen: set[int],
    emitted_order: list[int],
    rank_edges: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
    eligible: set[int] | None = None,
    reuse_token_offset: int = LEGACY_REUSE_TOKEN_OFFSET,
) -> tuple[int, int] | None:
    if not stack or not emitted_order:
        return None
    reuse_probs = probs.copy()
    reuse_probs[:reuse_token_offset] = 0.0
    for tok in range(reuse_token_offset, len(reuse_probs)):
        if int(rank_edges[tok - reuse_token_offset]) >= len(stack):
            reuse_probs[tok] = 0.0
    total = float(reuse_probs.sum())
    if total > 0.0:
        reuse_probs /= total
        for _ in range(16):
            tok = int(rng.choice(len(reuse_probs), p=reuse_probs))
            if tok < reuse_token_offset:
                continue
            rank = _sample_rank_for_token(
                tok,
                rank_edges,
                len(stack),
                rng,
                reuse_token_offset,
                [],
                ws0_bin=-1,
                rank_samples_by_token_ws0=None,
            )
            if rank < 0 or rank >= len(stack):
                continue
            oid = stack[rank]
            if oid not in emitted_seen:
                continue
            if eligible is not None and oid not in eligible:
                continue
            stack.pop(rank)
            stack.insert(0, oid)
            return oid, rank_to_token(rank, rank_edges, reuse_token_offset)
    for _ in range(8):
        oid = int(emitted_order[int(rng.integers(0, len(emitted_order)))])
        if eligible is not None and oid not in eligible:
            continue
        try:
            rank = stack.index(oid)
        except ValueError:
            continue
        stack.pop(rank)
        stack.insert(0, oid)
        return oid, rank_to_token(rank, rank_edges, reuse_token_offset)
    for rank, oid in enumerate(stack):
        if oid in emitted_seen:
            if eligible is not None and oid not in eligible:
                continue
            stack.pop(rank)
            stack.insert(0, oid)
            return int(oid), rank_to_token(rank, rank_edges, reuse_token_offset)
    return None


def _apply_short_reuse_pressure(
    probs: np.ndarray,
    rank_edges: np.ndarray,
    stack_len: int,
    windows: list[int],
    current_ws: np.ndarray,
    target_ws: np.ndarray,
    gain: float,
    reuse_token_offset: int = LEGACY_REUSE_TOKEN_OFFSET,
) -> np.ndarray:
    if gain <= 0.0 or stack_len <= 0 or len(windows) == 0:
        return probs
    surplus = 0.0
    for wi, weight in enumerate((1.0, 0.5)):
        if wi >= len(windows):
            break
        surplus += weight * max(
            0.0,
            (float(current_ws[wi]) - float(target_ws[wi])) / max(float(windows[wi]), 1.0),
        )
    if surplus <= 0.0:
        return probs

    primary = max(int(windows[0]), 1)
    cutoff = max(int(windows[min(1, len(windows) - 1)]), primary)
    biased = probs.copy()
    for tok in range(reuse_token_offset, len(biased)):
        if biased[tok] <= 0.0:
            continue
        ix = tok - reuse_token_offset
        lo = int(rank_edges[ix])
        hi = min(int(rank_edges[min(ix + 1, len(rank_edges) - 1)]), stack_len)
        if hi <= lo:
            biased[tok] = 0.0
            continue
        mid = 0.5 * (lo + hi - 1)
        if mid < primary:
            log_weight = 1.5 * gain * surplus
        elif mid < cutoff:
            log_weight = gain * surplus * (1.0 - mid / max(float(cutoff), 1.0))
        else:
            log_weight = -gain * surplus * min(1.0, math.log1p(mid / float(cutoff)))
        biased[tok] *= math.exp(max(-8.0, min(8.0, log_weight)))

    total = float(biased.sum())
    if total > 0.0:
        biased /= total
        return biased
    return probs


def _apply_rank_band_bias(
    probs: np.ndarray,
    rank_edges: np.ndarray,
    windows: list[int],
    rank_band_probs: np.ndarray | None,
    gain: float,
    reuse_token_offset: int,
) -> np.ndarray:
    if rank_band_probs is None or gain <= 0.0 or len(rank_band_probs) == 0:
        return probs
    biased = probs.copy()
    bounds = _rank_band_bounds(windows)
    n_bands = len(rank_band_probs)
    for tok in range(int(reuse_token_offset), len(biased)):
        if biased[tok] <= 0.0:
            continue
        ix = tok - int(reuse_token_offset)
        if ix < 0 or ix >= len(rank_edges) - 1:
            continue
        lo = int(rank_edges[ix])
        hi = int(rank_edges[min(ix + 1, len(rank_edges) - 1)])
        band = _rank_band_for_span(lo, hi, bounds)
        if band < 0 or band >= n_bands:
            continue
        scale = max(1e-9, float(rank_band_probs[band]) * float(n_bands))
        biased[tok] *= math.exp(max(-8.0, min(8.0, float(gain) * math.log(scale))))
    total = float(biased.sum())
    if total > 0.0:
        biased /= total
        return biased
    return probs


def _sample_stale_recycle(stale_pool, stack: list[int], rng: np.random.Generator) -> int | None:
    if not stale_pool:
        return None
    for _ in range(32):
        oid = int(stale_pool[int(rng.integers(0, len(stale_pool)))])
        try:
            stack.remove(oid)
        except ValueError:
            pass
        return oid
    return None


def _sample_rank_for_token(
    token: int,
    rank_edges: np.ndarray,
    stack_len: int,
    rng: np.random.Generator,
    reuse_token_offset: int,
    rank_samples_by_token,
    fp_bin: int = -1,
    rank_samples_by_token_fp=None,
    ws0_bin: int = -1,
    rank_samples_by_token_ws0=None,
    min_fp_bucket: int = 8,
) -> int:
    if stack_len <= 0:
        return -1
    # WS0-conditioned sampler: prefers the bucket matching current WS0 bin
    if (
        rank_samples_by_token_ws0 is not None
        and ws0_bin >= 0
        and 0 <= token < len(rank_samples_by_token_ws0)
        and ws0_bin < len(rank_samples_by_token_ws0[token])
    ):
        ws0_samples = rank_samples_by_token_ws0[token][ws0_bin]
        if len(ws0_samples) >= min_fp_bucket:
            for _ in range(16):
                rank = int(ws0_samples[int(rng.integers(0, len(ws0_samples)))])
                if 0 <= rank < stack_len:
                    return rank
            eligible = [int(r) for r in ws0_samples if 0 <= int(r) < stack_len]
            if eligible:
                return int(eligible[int(rng.integers(0, len(eligible)))])
    # Footprint-conditioned sampler: prefers the bucket matching current stack depth
    if (
        rank_samples_by_token_fp is not None
        and fp_bin >= 0
        and 0 <= token < len(rank_samples_by_token_fp)
        and fp_bin < len(rank_samples_by_token_fp[token])
    ):
        fp_samples = rank_samples_by_token_fp[token][fp_bin]
        if len(fp_samples) >= min_fp_bucket:
            for _ in range(16):
                rank = int(fp_samples[int(rng.integers(0, len(fp_samples)))])
                if 0 <= rank < stack_len:
                    return rank
            eligible = [int(r) for r in fp_samples if 0 <= int(r) < stack_len]
            if eligible:
                return int(eligible[int(rng.integers(0, len(eligible)))])
    # Global sampler fallback
    if rank_samples_by_token and 0 <= token < len(rank_samples_by_token):
        samples = rank_samples_by_token[token]
        if samples:
            for _ in range(16):
                rank = int(samples[int(rng.integers(0, len(samples)))])
                if 0 <= rank < stack_len:
                    return rank
            eligible = [int(rank) for rank in samples if 0 <= int(rank) < stack_len]
            if eligible:
                return int(eligible[int(rng.integers(0, len(eligible)))])
    return token_to_rank(
        token,
        rank_edges,
        rng,
        max_rank=stack_len - 1,
        reuse_token_offset=reuse_token_offset,
    )


def generate_ids(
    state: dict,
    model,
    n_records: int,
    seed: int,
    temperature: float,
    birth_control: bool = True,
    birth_control_mode: str = "footprint",
    short_reuse_pressure: float = 0.0,
    rank_band_bias: float = 0.0,
    ws_token_blend: float = 0.0,
    ws_token_blend_2d: float = 0.0,
    ws_blend_confidence_tau: float = 0.0,
    birth_rate_blend: float = 0.0,
    birth_rate_blend_2d: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch, _nn = _try_torch()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rank_edges = np.asarray(state["rank_edges"], dtype=np.int64)
    ws_edges = _coerce_ws_edges(state["ws_edges"])
    windows = [int(w) for w in state["windows"]]
    history = int(state["seq_len"])
    vocab = int(state["vocab"])
    reuse_token_offset = int(state.get("reuse_token_offset") or LEGACY_REUSE_TOKEN_OFFSET)
    recycle_token_raw = state.get("recycle_token")
    recycle_token = int(recycle_token_raw) if recycle_token_raw is not None else -1
    recycle_rank_cap = int(state.get("recycle_rank_cap") or 0)
    rank_samples_by_token = state.get("rank_samples_by_token") or []
    rank_samples_by_token_fp = state.get("rank_samples_by_token_fp") or []
    pos_bins = int(state.get("pos_bins") or 0)
    fp_bins = int(state.get("fp_bins") or 0)
    fp_edges_raw = state.get("fp_edges") or []
    fp_edges = np.asarray(fp_edges_raw, dtype=np.int64) if fp_bins > 0 and len(fp_edges_raw) > 0 else None
    ws_token_freq_raw = state.get("rank_token_freq_by_ws0") or []
    ws_token_freq_table: np.ndarray | None = (
        np.asarray(ws_token_freq_raw, dtype=np.float64) if ws_token_freq_raw else None
    )
    ws_token_freq_counts_raw = state.get("rank_token_freq_by_ws0_counts") or []
    ws_token_freq_counts: np.ndarray | None = (
        np.asarray(ws_token_freq_counts_raw, dtype=np.float64) if ws_token_freq_counts_raw else None
    )
    ws_token_freq_raw_2d = state.get("rank_token_freq_by_ws01") or []
    ws_token_freq_table_2d: np.ndarray | None = (
        np.asarray(ws_token_freq_raw_2d, dtype=np.float64) if ws_token_freq_raw_2d else None
    )
    ws_token_freq_counts_raw_2d = state.get("rank_token_freq_by_ws01_counts") or []
    ws_token_freq_counts_2d: np.ndarray | None = (
        np.asarray(ws_token_freq_counts_raw_2d, dtype=np.float64) if ws_token_freq_counts_raw_2d else None
    )
    rank_samples_by_token_ws0 = state.get("rank_samples_by_token_ws0") or []
    birth_rates_raw = state.get("empirical_birth_rates_by_ws0") or []
    empirical_birth_rates: np.ndarray | None = (
        np.asarray(birth_rates_raw, dtype=np.float64) if birth_rates_raw else None
    )
    birth_rate_counts_raw = state.get("birth_rate_counts_by_ws0") or []
    empirical_birth_rate_counts: np.ndarray | None = (
        np.asarray(birth_rate_counts_raw, dtype=np.float64) if birth_rate_counts_raw else None
    )
    birth_rates_2d_raw = state.get("empirical_birth_rates_by_ws01") or []
    empirical_birth_rates_2d: np.ndarray | None = (
        np.asarray(birth_rates_2d_raw, dtype=np.float64) if birth_rates_2d_raw else None
    )
    birth_rate_counts_2d_raw = state.get("birth_rate_counts_by_ws01") or []
    empirical_birth_rate_counts_2d: np.ndarray | None = (
        np.asarray(birth_rate_counts_2d_raw, dtype=np.float64) if birth_rate_counts_2d_raw else None
    )
    init_tokens, init_ws_tokens, queues, counters, stack, next_new, stale_pool = _warmstart_context(
        state,
        seed,
        rank_edges,
        ws_edges,
        windows,
        history,
    )
    ws_counts = list(zip(queues, counters))
    if not birth_control:
        birth_control_mode = "none"
    birth_targets = _birth_targets_from_real(state, n_records) if birth_control_mode == "footprint" else None
    ws_targets = _ws_targets_from_real(state, n_records, windows) if birth_control_mode == "ws" else None
    use_learned_ws = birth_control_mode in {"learned-ws", "learned-ws-masked"}
    learned_ws_decode = "masked" if birth_control_mode == "learned-ws-masked" else "clamp"
    window_scale = np.asarray(windows, dtype=np.float64)
    emitted_seen: set[int] = set()
    emitted_order: list[int] = []
    out = np.empty(n_records, dtype=np.int64)

    mark_sample = state.get("mark_sample", {})
    size_pool = np.asarray(mark_sample.get("obj_sizes") or [4096], dtype=np.int64)
    opcode_pool = np.asarray(mark_sample.get("opcodes") or [0], dtype=np.int64)
    tenant_pool = np.asarray(mark_sample.get("tenants") or [0], dtype=np.int64)
    sizes = np.empty(n_records, dtype=np.int64)
    opcodes = np.empty(n_records, dtype=np.int64)
    tenants = np.empty(n_records, dtype=np.int64)

    with torch.inference_mode():
        init_tok = torch.tensor([init_tokens], dtype=torch.long, device=device)
        init_ws = torch.tensor([init_ws_tokens], dtype=torch.long, device=device)
        init_pos = torch.zeros_like(init_tok)
        if fp_edges is not None:
            n_fresh_in_init = sum(1 for t in init_tokens if t == FRESH_TOKEN)
            init_fp_start = max(0, next_new - n_fresh_in_init)
            init_fp_list = _running_footprint_from_tokens(init_tokens, fp_edges, start_count=init_fp_start)
            init_fp = torch.tensor([init_fp_list], dtype=torch.long, device=device)
        else:
            init_fp = None
        logits, birth_logits, ws_logits, rank_band_logits, h = model(init_tok, init_ws, init_pos, init_fp)
        progress_every = max(25_000, min(100_000, max(1, n_records // 4)))
        for i in range(n_records):
            ws_pre = _ws_feature_from_counts(ws_counts, ws_edges)
            fp_pre = value_to_bin(len(stack), fp_edges) if fp_edges is not None else 0
            birth_target = int(birth_targets[i]) if birth_targets is not None else 0
            force_new = birth_targets is not None and len(emitted_seen) < birth_target
            force_reuse = birth_targets is not None and len(emitted_seen) >= birth_target
            birth_logit = float(birth_logits[0, -1].detach().cpu())
            rank_band_probs = (
                torch.softmax(rank_band_logits[0, -1], dim=-1).detach().cpu().numpy()
                if rank_band_logits is not None and float(rank_band_bias) > 0.0
                else None
            )
            reuse_eligible: set[int] | None = None
            current_ws: np.ndarray | None = None
            target_ws: np.ndarray | None = None
            if ws_targets is not None or use_learned_ws:
                current_ws = np.asarray([len(counter) for _queue, counter in ws_counts], dtype=np.float64)
                target_ws = (
                    _ws_target_from_logits(ws_logits, ws_edges, windows, learned_ws_decode)
                    if use_learned_ws
                    else ws_targets[i].astype(np.float64)
                )
                pressure_by_window = (target_ws - current_ws) / np.maximum(window_scale, 1.0)
                pressure_weights = np.asarray([0.35, 0.25, 0.20, 0.12, 0.08])[: len(pressure_by_window)]
                pressure = float(np.dot(pressure_by_window[: len(pressure_weights)], pressure_weights))
                birth_logit += 5.0 * max(-0.75, min(0.75, pressure))
                if float(np.max(pressure_by_window)) > 0.20:
                    force_new = True
                if float(np.min(pressure_by_window)) < -0.20:
                    force_reuse = True
                    force_new = False
                    reuse_eligible = {int(oid) for _sid, oid in ws_counts[0][1].keys()}
            birth_prob = float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, birth_logit)))))
            if birth_rate_blend > 0.0 and empirical_birth_rates is not None and len(ws_pre) > 0:
                w0_br = min(max(int(ws_pre[0]), 0), len(empirical_birth_rates) - 1)
                alpha_br = float(birth_rate_blend)
                if ws_blend_confidence_tau > 0.0 and empirical_birth_rate_counts is not None:
                    cnt_br = float(empirical_birth_rate_counts[w0_br]) if w0_br < len(empirical_birth_rate_counts) else 0.0
                    alpha_br *= min(1.0, math.sqrt(cnt_br / float(ws_blend_confidence_tau)))
                birth_prob = (1.0 - alpha_br) * birth_prob + alpha_br * float(empirical_birth_rates[w0_br])
            if birth_rate_blend_2d > 0.0 and empirical_birth_rates_2d is not None and len(ws_pre) > 1:
                w0_b2 = min(max(int(ws_pre[0]), 0), empirical_birth_rates_2d.shape[0] - 1)
                w1_b2 = min(max(int(ws_pre[1]), 0), empirical_birth_rates_2d.shape[1] - 1)
                alpha_b2 = float(birth_rate_blend_2d)
                if ws_blend_confidence_tau > 0.0 and empirical_birth_rate_counts_2d is not None:
                    cnt_b2 = float(empirical_birth_rate_counts_2d[w0_b2, w1_b2]) if (w0_b2 < empirical_birth_rate_counts_2d.shape[0] and w1_b2 < empirical_birth_rate_counts_2d.shape[1]) else 0.0
                    alpha_b2 *= min(1.0, math.sqrt(cnt_b2 / float(ws_blend_confidence_tau)))
                birth_prob = (1.0 - alpha_b2) * birth_prob + alpha_b2 * float(empirical_birth_rates_2d[w0_b2, w1_b2])
            z = logits[0, -1, 1:] / max(float(temperature), 1e-6)
            reuse_probs = torch.softmax(z, dim=-1).detach().cpu().numpy()
            probs = np.zeros(vocab, dtype=np.float64)
            probs[1:] = reuse_probs
            probs = probs.copy()
            if recycle_token >= 0 and not stale_pool:
                probs[recycle_token] = 0.0
            for tok in range(reuse_token_offset, vocab):
                if int(rank_edges[tok - reuse_token_offset]) >= len(stack):
                    probs[tok] = 0.0
            total = float(probs.sum())
            if total > 0.0:
                probs /= total
            else:
                probs = np.zeros(vocab, dtype=np.float64)
            if ws_token_blend > 0.0 and ws_token_freq_table is not None and len(ws_pre) > 0:
                w0 = min(max(int(ws_pre[0]), 0), len(ws_token_freq_table) - 1)
                alpha1 = float(ws_token_blend)
                if ws_blend_confidence_tau > 0.0 and ws_token_freq_counts is not None:
                    cnt = float(ws_token_freq_counts[w0]) if w0 < len(ws_token_freq_counts) else 0.0
                    alpha1 *= min(1.0, math.sqrt(cnt / float(ws_blend_confidence_tau)))
                empirical = ws_token_freq_table[w0].copy()
                empirical[:reuse_token_offset] = 0.0
                if recycle_token >= 0 and not stale_pool:
                    empirical[recycle_token] = 0.0
                for tok in range(reuse_token_offset, vocab):
                    if int(rank_edges[tok - reuse_token_offset]) >= len(stack):
                        empirical[tok] = 0.0
                emp_total = float(empirical.sum())
                if emp_total > 0.0 and alpha1 > 0.0:
                    empirical /= emp_total
                    probs = (1.0 - alpha1) * probs + alpha1 * empirical
                    blend_total = float(probs.sum())
                    if blend_total > 0.0:
                        probs /= blend_total
            if ws_token_blend_2d > 0.0 and ws_token_freq_table_2d is not None and len(ws_pre) > 1:
                w0 = min(max(int(ws_pre[0]), 0), len(ws_token_freq_table_2d) - 1)
                w1 = min(max(int(ws_pre[1]), 0), len(ws_token_freq_table_2d[w0]) - 1)
                alpha2 = float(ws_token_blend_2d)
                if ws_blend_confidence_tau > 0.0 and ws_token_freq_counts_2d is not None:
                    cnt2 = float(ws_token_freq_counts_2d[w0, w1]) if (w0 < ws_token_freq_counts_2d.shape[0] and w1 < ws_token_freq_counts_2d.shape[1]) else 0.0
                    alpha2 *= min(1.0, math.sqrt(cnt2 / float(ws_blend_confidence_tau)))
                empirical_2d = ws_token_freq_table_2d[w0][w1].copy()
                empirical_2d[:reuse_token_offset] = 0.0
                if recycle_token >= 0 and not stale_pool:
                    empirical_2d[recycle_token] = 0.0
                for tok in range(reuse_token_offset, vocab):
                    if int(rank_edges[tok - reuse_token_offset]) >= len(stack):
                        empirical_2d[tok] = 0.0
                emp2_total = float(empirical_2d.sum())
                if emp2_total > 0.0 and alpha2 > 0.0:
                    empirical_2d /= emp2_total
                    probs = (1.0 - alpha2) * probs + alpha2 * empirical_2d
                    blend2_total = float(probs.sum())
                    if blend2_total > 0.0:
                        probs /= blend2_total
            if current_ws is not None and target_ws is not None:
                probs = _apply_short_reuse_pressure(
                    probs,
                    rank_edges,
                    len(stack),
                    windows,
                    current_ws,
                    target_ws,
                    float(short_reuse_pressure),
                    reuse_token_offset,
                )
            probs = _apply_rank_band_bias(
                probs,
                rank_edges,
                windows,
                rank_band_probs,
                float(rank_band_bias),
                reuse_token_offset,
            )
            if force_new or (not force_reuse and (not stack or rng.random() < birth_prob)):
                token = FRESH_TOKEN
            elif float(probs.sum()) > 0.0:
                token = int(rng.choice(vocab, p=probs))
            else:
                token = FRESH_TOKEN
            if token == FRESH_TOKEN:
                oid = next_new
                next_new += 1
                stack.insert(0, oid)
            elif token == recycle_token:
                oid = _sample_stale_recycle(stale_pool, stack, rng)
                if oid is None:
                    oid = next_new
                    next_new += 1
                    token = FRESH_TOKEN
                stack.insert(0, int(oid))
            else:
                rank = _sample_rank_for_token(
                    token,
                    rank_edges,
                    len(stack),
                    rng,
                    reuse_token_offset,
                    rank_samples_by_token,
                    fp_bin=fp_pre if fp_edges is not None else -1,
                    rank_samples_by_token_fp=rank_samples_by_token_fp if rank_samples_by_token_fp else None,
                    ws0_bin=int(ws_pre[0]) if ws_pre else -1,
                    rank_samples_by_token_ws0=rank_samples_by_token_ws0 if rank_samples_by_token_ws0 else None,
                )
                if rank < 0 or rank >= len(stack):
                    oid = next_new
                    next_new += 1
                    stack.insert(0, oid)
                    token = FRESH_TOKEN
                else:
                    oid = stack.pop(rank)
                    stack.insert(0, oid)
            if recycle_rank_cap > 0:
                while len(stack) > recycle_rank_cap:
                    stale_pool.append(int(stack.pop()))
            should_force_reuse = (
                birth_targets is not None
                and not force_new
                and oid not in emitted_seen
                and len(emitted_seen) >= birth_target
            ) or (
                ws_targets is not None
                and force_reuse
                and oid not in emitted_seen
            )
            if should_force_reuse:
                forced = _force_emitted_reuse(
                    stack,
                    emitted_seen,
                    emitted_order,
                    rank_edges,
                    probs,
                    rng,
                    eligible=reuse_eligible,
                    reuse_token_offset=reuse_token_offset,
                )
                if forced is not None:
                    oid, token = forced
            out[i] = oid
            if oid not in emitted_seen:
                emitted_seen.add(int(oid))
                emitted_order.append(int(oid))
            sizes[i] = int(size_pool[rng.integers(0, len(size_pool))])
            opcodes[i] = int(opcode_pool[rng.integers(0, len(opcode_pool))])
            tenants[i] = int(tenant_pool[rng.integers(0, len(tenant_pool))])
            key = (0, int(oid))
            for wi, (queue, counter) in enumerate(ws_counts):
                queue.append(key)
                counter[key] += 1
                while len(queue) > windows[wi]:
                    old = queue.popleft()
                    counter[old] -= 1
                    if counter[old] <= 0:
                        del counter[old]
            step_tok = torch.tensor([[token]], dtype=torch.long, device=device)
            step_ws = torch.tensor([[ws_pre]], dtype=torch.long, device=device)
            step_pos = torch.tensor([[position_token_at(i, n_records, pos_bins)]], dtype=torch.long, device=device)
            step_fp = torch.tensor([[fp_pre]], dtype=torch.long, device=device) if fp_edges is not None else None
            logits, birth_logits, ws_logits, rank_band_logits, h = model(step_tok, step_ws, step_pos, step_fp, h)
            if i + 1 < n_records and (i + 1) % progress_every == 0:
                print(
                    f"[mattson_denning generate] emitted={i + 1:,} "
                    f"internal_unique={next_new:,} output_unique={len(emitted_seen):,} "
                    f"stack={len(stack):,} stale_pool={len(stale_pool):,}",
                    flush=True,
                )
    print(
        f"[mattson_denning generate] emitted={n_records:,} "
        f"internal_unique={next_new:,} output_unique={len(emitted_seen):,} "
        f"stack={len(stack):,} stale_pool={len(stale_pool):,}",
        flush=True,
    )
    return out, sizes, opcodes, tenants


def write_csv(path: str | Path, obj_ids: np.ndarray, sizes: np.ndarray, opcodes: np.ndarray, tenants: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i, oid in enumerate(obj_ids):
            writer.writerow([0, i, int(oid), int(sizes[i]), int(opcodes[i]), int(tenants[i])])
    print(f"[mattson_denning generate] wrote {path}", flush=True)


def generate(args) -> None:
    state, model = load_checkpoint(args.model)
    obj_ids, sizes, opcodes, tenants = generate_ids(
        state,
        model,
        args.n_records,
        args.seed,
        args.temperature,
        not args.no_birth_control,
        args.birth_control_mode,
        args.short_reuse_pressure,
        args.rank_band_bias,
        ws_token_blend=float(getattr(args, "ws_token_blend", 0.0)),
        ws_token_blend_2d=float(getattr(args, "ws_token_blend_2d", 0.0)),
        ws_blend_confidence_tau=float(getattr(args, "ws_blend_confidence_tau", 0.0)),
        birth_rate_blend=float(getattr(args, "birth_rate_blend", 0.0)),
        birth_rate_blend_2d=float(getattr(args, "birth_rate_blend_2d", 0.0)),
    )
    write_csv(args.output, obj_ids, sizes, opcodes, tenants)


def _literal(mean: float) -> str:
    return f"mean HRC-MAE across policies: {mean:.4f}"


def multiseed(args) -> None:
    model_path = Path(args.model)
    if args.fit or not model_path.exists():
        fit_args = argparse.Namespace(**vars(args))
        fit_args.output = str(model_path)
        fit(fit_args)
    state, model = load_checkpoint(model_path)

    from llgan.cachesim_eval import evaluate, print_report

    out_root = Path(args.output_root)
    eval_root = out_root / "cachesim_lanl"
    eval_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in _parse_ints(args.seeds):
        fake = out_root / f"{args.tag}_seed{seed}_fake_{args.n_records // 1000}k.csv"
        obj_ids, sizes, opcodes, tenants = generate_ids(
            state,
            model,
            args.n_records,
            seed,
            args.temperature,
            not args.no_birth_control,
            args.birth_control_mode,
            args.short_reuse_pressure,
            args.rank_band_bias,
            ws_token_blend=float(getattr(args, "ws_token_blend", 0.0)),
            ws_token_blend_2d=float(getattr(args, "ws_token_blend_2d", 0.0)),
            ws_blend_confidence_tau=float(getattr(args, "ws_blend_confidence_tau", 0.0)),
            birth_rate_blend=float(getattr(args, "birth_rate_blend", 0.0)),
            birth_rate_blend_2d=float(getattr(args, "birth_rate_blend_2d", 0.0)),
        )
        write_csv(fake, obj_ids, sizes, opcodes, tenants)
        report = evaluate(str(fake), args.real, args.cache_sizes, args.policies)
        print_report(report)
        report_json = eval_root / f"{args.tag}_seed{seed}_official.json"
        report_json.write_text(json.dumps(report, indent=2))
        mean = float(report["mean_hrc_mae"])
        print(f"Report JSON: {report_json}", flush=True)
        rows.append((seed, fake, report_json, mean, _literal(mean)))

    means = [row[3] for row in rows]
    mean_all = float(np.mean(means))
    range_all = float(np.max(means) - np.min(means))
    title = args.markdown_title or f"## {args.tag} Mattson-Denning LSTM multi-seed summary"
    md = _markdown(title, rows, mean_all, range_all)
    print(md, flush=True)
    if args.append_markdown:
        for target in [p.strip() for p in args.append_markdown.split(",") if p.strip()]:
            with open(target, "a") as fh:
                fh.write("\n\n")
                fh.write(md)
                fh.write("\n")
            print(f"[mattson_denning multiseed] appended {target}", flush=True)


def _markdown(title: str, rows, mean_all: float, range_all: float) -> str:
    lines = [
        title,
        "",
        "| seed | fake CSV | literal cachesim mean line | JSON mean |",
        "|---:|---|---|---:|",
    ]
    for seed, fake, _report, mean, literal in rows:
        lines.append(f"| {seed} | `{fake}` | `{literal}` | {mean:.10f} |")
    seeds = "{" + ",".join(str(row[0]) for row in rows) + "}"
    lines.extend([
        "",
        f"Mean across seeds `{seeds}`: `{mean_all:.10f}` (race display `{mean_all:.4f}`; range `{range_all:.10f}`).",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_train_flags(q):
        q.add_argument("--real", required=True)
        q.add_argument("--max-rows", type=int, default=0)
        q.add_argument("--rank-bins", type=int, default=64)
        q.add_argument("--ws-bins", type=int, default=32)
        q.add_argument("--ws-windows", default=DEFAULT_WINDOWS)
        q.add_argument(
            "--ws-edge-mode",
            choices=["footprint", "max-window", "per-window"],
            default="footprint",
            help="bin Denning working-set counts against the full footprint, largest WS control window, or each window independently",
        )
        q.add_argument("--hidden", type=int, default=128)
        q.add_argument("--token-embed", type=int, default=64)
        q.add_argument("--ws-embed", type=int, default=16)
        q.add_argument(
            "--pos-bins",
            type=int,
            default=0,
            help="if >0, add learned absolute trace-phase embeddings with this many bins",
        )
        q.add_argument("--pos-embed", type=int, default=8)
        q.add_argument("--seq-len", type=int, default=256)
        q.add_argument("--batch", type=int, default=256)
        q.add_argument("--epochs", type=int, default=10)
        q.add_argument("--lr", type=float, default=1e-3)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--aux-ws-loss-weight", type=float, default=0.15)
        q.add_argument("--short-reuse-loss-weight", type=float, default=0.0)
        q.add_argument(
            "--rank-sampler",
            choices=["uniform", "empirical"],
            default="uniform",
            help="within predicted Mattson-depth bin, sample uniformly or from exact fitted real ranks",
        )
        q.add_argument(
            "--exact-rank-cutoff",
            type=int,
            default=0,
            help="use one token per exact Mattson rank below this cutoff, then log-bin the tail",
        )
        q.add_argument(
            "--recycle-rank-cap",
            type=int,
            default=0,
            help="if >0, label Mattson depths at or beyond this cap as RECYCLE instead of ordinary rank-bin reuse",
        )
        q.add_argument(
            "--rank-band-mode",
            choices=["none", "window"],
            default="none",
            help="add an auxiliary Mattson-depth band head keyed to the cache-window ladder",
        )
        q.add_argument("--rank-band-loss-weight", type=float, default=0.0)
        q.add_argument(
            "--stack-depth-bins",
            type=int,
            default=0,
            help="if >0, add running LRU stack depth (footprint) as a log-binned LSTM input with this many bins",
        )
        q.add_argument(
            "--cache-ladder",
            action="store_true",
            default=False,
            help="add exact rank-edge boundaries at each cache-ladder size so no rank bin straddles a cachesim evaluation point",
        )
        q.add_argument(
            "--ws-cache-ladder",
            action="store_true",
            default=False,
            help="add exact WS-edge boundaries at each cache-ladder size so WS-conditioned empirical bins never straddle a cachesim evaluation point",
        )
        q.add_argument(
            "--ladder-sizes",
            default=DEFAULT_SIZES,
            help="comma-separated cache-ladder sizes used for --cache-ladder / --ws-cache-ladder boundaries (default: 32,128,512,2048,8192)",
        )
        q.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="LSTM inter-layer dropout probability (0.0 = disabled; only affects num_layers>1)",
        )
        q.add_argument(
            "--lr-schedule",
            choices=["none", "cosine"],
            default="none",
            help="learning-rate schedule: none = constant, cosine = CosineAnnealingLR over epochs",
        )
        q.add_argument(
            "--ws-kl-loss-weight",
            type=float,
            default=0.0,
            help="if >0, add KL(empirical_ws0_dist || LSTM_reuse_dist) auxiliary loss during training",
        )
        q.add_argument(
            "--ws-kl-loss-weight-2d",
            type=float,
            default=0.0,
            help="if >0, add KL(empirical_ws01_dist || LSTM_reuse_dist) 2D auxiliary loss during training",
        )
        q.add_argument(
            "--film-cond",
            action="store_true",
            default=False,
            help="apply FiLM (feature-wise linear modulation) to LSTM output using WS context before prediction heads",
        )
        q.add_argument(
            "--lstm-layers",
            type=int,
            default=2,
            help="number of LSTM stacked layers (default: 2; try 3 for deeper model with same hidden size)",
        )
        q.add_argument(
            "--label-smoothing",
            type=float,
            default=0.0,
            help="label smoothing epsilon for reuse token cross-entropy loss (0.0 = disabled, try 0.05–0.10)",
        )
        q.add_argument(
            "--grad-clip",
            type=float,
            default=1.0,
            help="gradient norm clip threshold (0.0 = disabled, default 1.0)",
        )
        q.add_argument(
            "--birth-kl-loss-weight",
            type=float,
            default=0.0,
            help="if >0, add soft-target BCE on birth head using empirical P(fresh|ws0) as targets during training",
        )
        q.add_argument(
            "--birth-kl-loss-weight-2d",
            type=float,
            default=0.0,
            help="if >0, add soft-target BCE on birth head using empirical P(fresh|ws0,ws1) 2D joint table as targets during training",
        )

    fit_p = sub.add_parser("fit")
    add_train_flags(fit_p)
    fit_p.add_argument("--output", "--model", dest="output", required=True)
    fit_p.set_defaults(fn=fit)

    gen_p = sub.add_parser("generate")
    gen_p.add_argument("--model", required=True)
    gen_p.add_argument("--output", required=True)
    gen_p.add_argument("--n-records", type=int, default=100_000)
    gen_p.add_argument("--seed", type=int, default=42)
    gen_p.add_argument("--temperature", type=float, default=1.0)
    gen_p.add_argument("--no-birth-control", action="store_true")
    gen_p.add_argument(
        "--birth-control-mode",
        choices=["footprint", "ws", "learned-ws", "learned-ws-masked"],
        default="footprint",
    )
    gen_p.add_argument("--short-reuse-pressure", type=float, default=0.0)
    gen_p.add_argument("--rank-band-bias", type=float, default=0.0)
    gen_p.add_argument(
        "--ws-token-blend",
        type=float,
        default=0.0,
        help="blend LSTM token distribution with WS-conditioned empirical rank-token table (0=LSTM only, 1=empirical only)",
    )
    gen_p.add_argument(
        "--ws-token-blend-2d",
        type=float,
        default=0.0,
        help="blend LSTM token distribution with (ws0,ws1)-joint empirical rank-token table (0=LSTM only, 1=empirical only)",
    )
    gen_p.add_argument(
        "--ws-blend-confidence-tau",
        type=float,
        default=0.0,
        help="if >0, scale empirical blend weight by sqrt(bucket_count/tau); sparse bins fall back to LSTM",
    )
    gen_p.add_argument(
        "--birth-rate-blend",
        type=float,
        default=0.0,
        help="blend LSTM fresh-token probability with WS-conditioned empirical birth rate (0=LSTM only, 1=empirical only)",
    )
    gen_p.add_argument(
        "--birth-rate-blend-2d",
        type=float,
        default=0.0,
        help="blend LSTM fresh-token probability with (ws0,ws1)-joint empirical birth rate (0=LSTM only, 1=empirical only)",
    )
    gen_p.set_defaults(fn=generate)

    multi = sub.add_parser("multiseed")
    add_train_flags(multi)
    multi.add_argument("--model", required=True)
    multi.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    multi.add_argument("--tag", required=True)
    multi.add_argument("--n-records", type=int, default=100_000)
    multi.add_argument("--seeds", default="42,80,81,82")
    multi.add_argument("--temperature", type=float, default=1.0)
    multi.add_argument("--cache-sizes", default=DEFAULT_SIZES)
    multi.add_argument("--policies", default=DEFAULT_POLICIES)
    multi.add_argument("--fit", action="store_true")
    multi.add_argument("--no-birth-control", action="store_true")
    multi.add_argument(
        "--birth-control-mode",
        choices=["footprint", "ws", "learned-ws", "learned-ws-masked"],
        default="footprint",
    )
    multi.add_argument("--short-reuse-pressure", type=float, default=0.0)
    multi.add_argument("--rank-band-bias", type=float, default=0.0)
    multi.add_argument(
        "--ws-token-blend",
        type=float,
        default=0.0,
        help="blend LSTM token distribution with WS-conditioned empirical rank-token table (0=LSTM only, 1=empirical only)",
    )
    multi.add_argument(
        "--ws-token-blend-2d",
        type=float,
        default=0.0,
        help="blend LSTM token distribution with (ws0,ws1)-joint empirical rank-token table (0=LSTM only, 1=empirical only)",
    )
    multi.add_argument(
        "--ws-blend-confidence-tau",
        type=float,
        default=0.0,
        help="if >0, scale empirical blend weight by sqrt(bucket_count/tau); sparse bins fall back to LSTM",
    )
    multi.add_argument(
        "--birth-rate-blend",
        type=float,
        default=0.0,
        help="blend LSTM fresh-token probability with WS-conditioned empirical birth rate (0=LSTM only, 1=empirical only)",
    )
    multi.add_argument(
        "--birth-rate-blend-2d",
        type=float,
        default=0.0,
        help="blend LSTM fresh-token probability with (ws0,ws1)-joint empirical birth rate (0=LSTM only, 1=empirical only)",
    )
    multi.add_argument("--append-markdown", default="")
    multi.add_argument("--markdown-title", default="")
    multi.set_defaults(fn=multiseed)
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
