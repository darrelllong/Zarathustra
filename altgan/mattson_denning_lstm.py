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


def make_rank_edges(max_value: int, n_bins: int, exact_rank_cutoff: int = 0) -> np.ndarray:
    """Log rank bins with an optional exact-rank prefix for short-cache depths."""
    max_value = max(int(max_value), 1)
    cutoff = max(0, min(int(exact_rank_cutoff), max_value + 1))
    if cutoff <= 1:
        return make_log_edges(max_value, n_bins)
    exact_edges = np.arange(0, cutoff + 1, dtype=np.int64)
    if cutoff > max_value:
        return np.arange(0, max_value + 2, dtype=np.int64)
    tail_span = max_value - cutoff + 1
    raw = np.unique(
        np.round(np.logspace(0, math.log10(tail_span + 1), n_bins)).astype(np.int64)
    )
    tail_edges = cutoff + raw
    edges = np.unique(np.concatenate((exact_edges, tail_edges)))
    edges = np.unique(np.clip(edges, 0, max_value + 1))
    if edges[-1] <= max_value:
        edges = np.concatenate((edges, [max_value + 1]))
    return edges.astype(np.int64)


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


def denning_working_sets(trace: RealTrace, windows: list[int], edges: np.ndarray) -> np.ndarray:
    """Binned trailing-window working-set sizes before each event."""
    from collections import Counter, deque

    keys = list(zip(trace.stream_ids.tolist(), trace.obj_ids.tolist()))
    queues = [deque() for _ in windows]
    counts = [Counter() for _ in windows]
    features = np.empty((len(keys), len(windows)), dtype=np.int64)

    for i, key in enumerate(keys):
        for wi, window in enumerate(windows):
            features[i, wi] = value_to_bin(len(counts[wi]), edges)
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
):
    depths, footprint = mattson_depths(trace)
    reuse_token_offset = SPLIT_REUSE_TOKEN_OFFSET if recycle_rank_cap > 0 else LEGACY_REUSE_TOKEN_OFFSET
    rank_max = min(footprint, max(1, int(recycle_rank_cap))) if recycle_rank_cap > 0 else footprint
    rank_edges = make_rank_edges(rank_max, n_rank_bins, exact_rank_cutoff)
    ws_edges = make_log_edges(footprint, n_ws_bins)
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
    print(
        "[mattson_denning tokenize] "
        f"n={trace.n:,} footprint={footprint:,} rank_vocab={len(rank_edges)} "
        f"reuse_offset={reuse_token_offset} recycle_rank_cap={int(recycle_rank_cap)} "
        f"exact_rank_cutoff={int(exact_rank_cutoff)} "
        f"fresh={fresh:,} recycle={recycle:,} reuse={reuse:,} "
        f"ws_bins={len(ws_edges) - 1} windows={windows}",
        flush=True,
    )
    return tokens, ws_tokens, rank_edges, ws_edges, footprint, rank_samples_by_token


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


def _try_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for mattson_denning_lstm") from exc


def build_model(vocab: int, n_windows: int, ws_bins: int, token_embed: int, ws_embed: int, hidden: int):
    torch, nn = _try_torch()

    class MattsonDenningLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab, token_embed)
            self.ws_emb = nn.ModuleList([nn.Embedding(ws_bins, ws_embed) for _ in range(n_windows)])
            self.lstm = nn.LSTM(
                token_embed + n_windows * ws_embed,
                hidden,
                num_layers=2,
                batch_first=True,
            )
            self.head = nn.Linear(hidden, vocab)
            self.birth_head = nn.Linear(hidden, 1)
            self.ws_heads = nn.ModuleList([nn.Linear(hidden, ws_bins) for _ in range(n_windows)])

        def forward(self, tok, ws, h=None):
            parts = [self.token_emb(tok)]
            for i, emb in enumerate(self.ws_emb):
                parts.append(emb(ws[:, :, i]))
            x = torch.cat(parts, dim=-1)
            out, h = self.lstm(x, h)
            return self.head(out), self.birth_head(out).squeeze(-1), [head(out) for head in self.ws_heads], h

    return MattsonDenningLSTM()


def train_model(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    *,
    rank_edges: np.ndarray,
    windows: list[int],
    vocab: int,
    ws_bins: int,
    hidden: int,
    token_embed: int,
    ws_embed: int,
    seq_len: int,
    batch: int,
    epochs: int,
    lr: float,
    seed: int,
    aux_ws_loss_weight: float,
    short_reuse_loss_weight: float,
):
    torch, _nn = _try_torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab, ws_tokens.shape[1], ws_bins, token_embed, ws_embed, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    reuse_token_offset = max(LEGACY_REUSE_TOKEN_OFFSET, int(vocab) - len(rank_edges) + 1)
    reuse_class_weights = None
    if short_reuse_loss_weight > 0.0:
        weights = _short_reuse_class_weights(rank_edges, windows, short_reuse_loss_weight)
        if reuse_token_offset == SPLIT_REUSE_TOKEN_OFFSET:
            weights = np.concatenate((np.ones(1, dtype=np.float32), weights))
        reuse_class_weights = torch.from_numpy(weights).to(device)

    tok_t = torch.from_numpy(tokens.astype(np.int64))
    ws_t = torch.from_numpy(ws_tokens.astype(np.int64))
    n_train = len(tokens) - seq_len - 1
    if n_train <= 0:
        raise ValueError("not enough tokens for requested sequence length")
    n_batches = max(1, n_train // batch)
    rng = np.random.default_rng(seed)
    print(
        "[mattson_denning train] "
        f"device={device} params={sum(p.numel() for p in model.parameters()):,} "
        f"seq={seq_len} batch={batch} epochs={epochs} n_batches={n_batches} "
        f"reuse_offset={reuse_token_offset} short_reuse_loss_weight={short_reuse_loss_weight}",
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
            y = torch.stack([tok_t[i + 1:i + seq_len + 1] for i in idx]).to(device)
            y_ws = torch.stack([ws_t[i + 1:i + seq_len + 1] for i in idx]).to(device)
            logits, birth_logits, ws_logits, _ = model(x_tok, x_ws)
            y_birth = (y == FRESH_TOKEN).float()
            birth_loss = F.binary_cross_entropy_with_logits(
                birth_logits.reshape(-1),
                y_birth.reshape(-1),
            )
            reuse_mask = y != FRESH_TOKEN
            if bool(reuse_mask.any()):
                reuse_loss = F.cross_entropy(
                    logits[:, :, 1:][reuse_mask],
                    y[reuse_mask] - 1,
                    weight=reuse_class_weights,
                )
            else:
                reuse_loss = logits.sum() * 0.0
            ws_loss = torch.stack([
                F.cross_entropy(ws_logits[wi].reshape(-1, ws_bins), y_ws[:, :, wi].reshape(-1))
                for wi in range(ws_tokens.shape[1])
            ]).mean()
            token_loss = birth_loss + reuse_loss
            loss = token_loss + float(aux_ws_loss_weight) * ws_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += float(loss.detach().cpu())
            if bi == 0 or (bi + 1) % max(1, n_batches // 5) == 0:
                print(
                    f"[mattson_denning train] ep={ep + 1}/{epochs} "
                    f"batch={bi + 1}/{n_batches} loss={float(loss.detach().cpu()):.5f} "
                    f"tok={float(token_loss.detach().cpu()):.5f} "
                    f"birth={float(birth_loss.detach().cpu()):.5f} "
                    f"reuse={float(reuse_loss.detach().cpu()):.5f} "
                    f"ws={float(ws_loss.detach().cpu()):.5f}",
                    flush=True,
                )
        print(
            f"[mattson_denning train] ep={ep + 1}/{epochs} avg_loss={running / n_batches:.5f}",
            flush=True,
        )
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
        int(state["ws_bins"]),
        int(state["token_embed"]),
        int(state["ws_embed"]),
        int(state["hidden"]),
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
    tokens, ws_tokens, rank_edges, ws_edges, footprint, rank_samples_by_token = tokenize(
        trace,
        args.rank_bins,
        args.ws_bins,
        windows,
        args.recycle_rank_cap,
        args.exact_rank_cutoff,
    )
    reuse_token_offset = SPLIT_REUSE_TOKEN_OFFSET if args.recycle_rank_cap > 0 else LEGACY_REUSE_TOKEN_OFFSET
    vocab = len(rank_edges) + (reuse_token_offset - 1)
    model = train_model(
        tokens,
        ws_tokens,
        rank_edges=rank_edges,
        windows=windows,
        vocab=vocab,
        ws_bins=len(ws_edges) - 1,
        hidden=args.hidden,
        token_embed=args.token_embed,
        ws_embed=args.ws_embed,
        seq_len=args.seq_len,
        batch=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        aux_ws_loss_weight=args.aux_ws_loss_weight,
        short_reuse_loss_weight=args.short_reuse_loss_weight,
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
            "version": 5,
            "real": str(args.real),
            "max_rows": args.max_rows,
            "n_train_rows": trace.n,
            "footprint": footprint,
            "rank_edges": rank_edges,
            "ws_edges": ws_edges,
            "windows": windows,
            "vocab": vocab,
            "fresh_token": FRESH_TOKEN,
            "recycle_token": RECYCLE_TOKEN if args.recycle_rank_cap > 0 else None,
            "reuse_token_offset": reuse_token_offset,
            "recycle_rank_cap": int(args.recycle_rank_cap),
            "ws_bins": len(ws_edges) - 1,
            "hidden": args.hidden,
            "token_embed": args.token_embed,
            "ws_embed": args.ws_embed,
            "seq_len": args.seq_len,
            "rank_bins": args.rank_bins,
            "exact_rank_cutoff": int(args.exact_rank_cutoff),
            "mark_sample": mark_sample,
            "aux_ws_loss_weight": args.aux_ws_loss_weight,
            "short_reuse_loss_weight": args.short_reuse_loss_weight,
            "rank_sampler": args.rank_sampler,
            "rank_samples_by_token": rank_samples_by_token if args.rank_sampler == "empirical" else [],
            "training_loss": "binary fresh-access cross entropy plus non-fresh action cross entropy over optional RECYCLE and Mattson-depth reuse tokens plus auxiliary next-Denning-working-set cross entropy; optional empirical within-bin Mattson-rank sampler is fit from exact real depths",
        },
    )


def _sample_pool(values: np.ndarray, limit: int = 4096) -> list[int]:
    if len(values) <= limit:
        return [int(v) for v in values]
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(values), size=limit, replace=False)
    return [int(v) for v in values[idx]]


def _ws_feature_from_counts(counts, edges: np.ndarray) -> list[int]:
    return [value_to_bin(len(counter), edges) for _queue, counter in counts]


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
    ws_edges: np.ndarray,
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
    ws_edges: np.ndarray,
    windows: list[int],
    mode: str = "clamp",
) -> np.ndarray:
    """Convert learned next-working-set logits into feasible per-window counts."""
    torch, _nn = _try_torch()
    mids = 0.5 * (ws_edges[:-1].astype(np.float64) + ws_edges[1:].astype(np.float64) - 1.0)
    mids = np.maximum(mids, 0.0)
    targets: list[float] = []
    for wi, head_logits in enumerate(ws_logits):
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
            token = int(rng.choice(len(reuse_probs), p=reuse_probs))
            lo = int(rank_edges[token - reuse_token_offset])
            hi = min(int(rank_edges[min(token - reuse_token_offset + 1, len(rank_edges) - 1)]), len(stack))
            if hi <= lo:
                continue
            for _attempt in range(4):
                rank = int(rng.integers(lo, hi))
                oid = int(stack[rank])
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
) -> int:
    if stack_len <= 0:
        return -1
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch, _nn = _try_torch()

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    rank_edges = np.asarray(state["rank_edges"], dtype=np.int64)
    ws_edges = np.asarray(state["ws_edges"], dtype=np.int64)
    windows = [int(w) for w in state["windows"]]
    history = int(state["seq_len"])
    vocab = int(state["vocab"])
    reuse_token_offset = int(state.get("reuse_token_offset") or LEGACY_REUSE_TOKEN_OFFSET)
    recycle_token_raw = state.get("recycle_token")
    recycle_token = int(recycle_token_raw) if recycle_token_raw is not None else -1
    recycle_rank_cap = int(state.get("recycle_rank_cap") or 0)
    rank_samples_by_token = state.get("rank_samples_by_token") or []
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
        logits, birth_logits, ws_logits, h = model(init_tok, init_ws)
        progress_every = max(25_000, min(100_000, max(1, n_records // 4)))
        for i in range(n_records):
            ws_pre = _ws_feature_from_counts(ws_counts, ws_edges)
            birth_target = int(birth_targets[i]) if birth_targets is not None else 0
            force_new = birth_targets is not None and len(emitted_seen) < birth_target
            force_reuse = birth_targets is not None and len(emitted_seen) >= birth_target
            birth_logit = float(birth_logits[0, -1].detach().cpu())
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
            logits, birth_logits, ws_logits, h = model(step_tok, step_ws, h)
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
        q.add_argument("--hidden", type=int, default=128)
        q.add_argument("--token-embed", type=int, default=64)
        q.add_argument("--ws-embed", type=int, default=16)
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
