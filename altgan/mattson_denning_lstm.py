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


def value_to_bin(value: int, edges: np.ndarray) -> int:
    ix = int(np.searchsorted(edges, int(value), side="right") - 1)
    return max(0, min(ix, len(edges) - 2))


def rank_to_token(rank: int, edges: np.ndarray) -> int:
    return value_to_bin(rank, edges) + 1


def token_to_rank(
    token: int,
    edges: np.ndarray,
    rng: np.random.Generator,
    max_rank: int | None = None,
) -> int:
    if token == NEW_TOKEN:
        return -1
    ix = token - 1
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


def tokenize(trace: RealTrace, n_rank_bins: int, n_ws_bins: int, windows: list[int]):
    depths, footprint = mattson_depths(trace)
    rank_edges = make_log_edges(footprint, n_rank_bins)
    ws_edges = make_log_edges(footprint, n_ws_bins)
    tokens = tokens_from_depths(depths, rank_edges, trace.n)
    ws_tokens = denning_working_sets(trace, windows, ws_edges)
    print(
        "[mattson_denning tokenize] "
        f"n={trace.n:,} footprint={footprint:,} rank_vocab={len(rank_edges)} "
        f"ws_bins={len(ws_edges) - 1} windows={windows}",
        flush=True,
    )
    return tokens, ws_tokens, rank_edges, ws_edges, footprint


def tokens_from_depths(depths: np.ndarray, rank_edges: np.ndarray, n: int) -> np.ndarray:
    tokens = np.empty(n, dtype=np.int64)
    for i, depth in enumerate(depths):
        tokens[i] = NEW_TOKEN if depth < 0 else rank_to_token(int(depth), rank_edges)
    return tokens


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
            self.ws_heads = nn.ModuleList([nn.Linear(hidden, ws_bins) for _ in range(n_windows)])

        def forward(self, tok, ws, h=None):
            parts = [self.token_emb(tok)]
            for i, emb in enumerate(self.ws_emb):
                parts.append(emb(ws[:, :, i]))
            x = torch.cat(parts, dim=-1)
            out, h = self.lstm(x, h)
            return self.head(out), [head(out) for head in self.ws_heads], h

    return MattsonDenningLSTM()


def train_model(
    tokens: np.ndarray,
    ws_tokens: np.ndarray,
    *,
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
):
    torch, _nn = _try_torch()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab, ws_tokens.shape[1], ws_bins, token_embed, ws_embed, hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
        f"seq={seq_len} batch={batch} epochs={epochs} n_batches={n_batches}",
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
            logits, ws_logits, _ = model(x_tok, x_ws)
            token_loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            ws_loss = torch.stack([
                F.cross_entropy(ws_logits[wi].reshape(-1, ws_bins), y_ws[:, :, wi].reshape(-1))
                for wi in range(ws_tokens.shape[1])
            ]).mean()
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
                    f"ws={float(ws_loss.detach().cpu()):.5f}",
                    flush=True,
                )
        print(
            f"[mattson_denning train] ep={ep + 1}/{epochs} avg_loss={running / n_batches:.5f}",
            flush=True,
        )
    return model


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
    tokens, ws_tokens, rank_edges, ws_edges, footprint = tokenize(
        trace,
        args.rank_bins,
        args.ws_bins,
        windows,
    )
    model = train_model(
        tokens,
        ws_tokens,
        vocab=len(rank_edges),
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
            "version": 2,
            "real": str(args.real),
            "max_rows": args.max_rows,
            "n_train_rows": trace.n,
            "footprint": footprint,
            "rank_edges": rank_edges,
            "ws_edges": ws_edges,
            "windows": windows,
            "vocab": len(rank_edges),
            "ws_bins": len(ws_edges) - 1,
            "hidden": args.hidden,
            "token_embed": args.token_embed,
            "ws_embed": args.ws_embed,
            "seq_len": args.seq_len,
            "rank_bins": args.rank_bins,
            "mark_sample": mark_sample,
            "aux_ws_loss_weight": args.aux_ws_loss_weight,
            "training_loss": "next-token cross entropy on Mattson depth tokens plus auxiliary next-Denning-working-set cross entropy",
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
    default_tokens = [NEW_TOKEN] * history
    default_ws = [[0] * len(windows) for _ in range(history)]

    real_path = state.get("real")
    if not real_path:
        return default_tokens, default_ws, queues, counters, stack, next_new

    try:
        trace = read_real_csv(real_path, int(state.get("max_rows") or 0))
    except OSError:
        return default_tokens, default_ws, queues, counters, stack, next_new

    min_warm = min(trace.n, max(history, max(windows, default=history)))
    if min_warm <= 0:
        return default_tokens, default_ws, queues, counters, stack, next_new

    rng = np.random.default_rng(seed + 7919)
    end = int(rng.integers(min_warm, trace.n + 1)) if trace.n > min_warm else trace.n
    warm = _slice_trace(trace, 0, end)
    depths, _footprint = mattson_depths(warm)
    warm_tokens = tokens_from_depths(depths, rank_edges, warm.n).tolist()
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
        f"warmstart_rows={warm.n:,} prefix_end={end:,} unique={next_new:,} stack={len(stack):,}",
        flush=True,
    )
    return warm_tokens, warm_ws, queues, counters, stack, next_new


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


def _force_emitted_reuse(
    stack: list[int],
    emitted_seen: set[int],
    emitted_order: list[int],
    rank_edges: np.ndarray,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if not stack or not emitted_order:
        return None
    for _ in range(8):
        oid = int(emitted_order[int(rng.integers(0, len(emitted_order)))])
        try:
            rank = stack.index(oid)
        except ValueError:
            continue
        stack.pop(rank)
        stack.insert(0, oid)
        return oid, rank_to_token(rank, rank_edges)
    for rank, oid in enumerate(stack):
        if oid in emitted_seen:
            stack.pop(rank)
            stack.insert(0, oid)
            return int(oid), rank_to_token(rank, rank_edges)
    return None


def generate_ids(
    state: dict,
    model,
    n_records: int,
    seed: int,
    temperature: float,
    birth_control: bool = True,
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
    init_tokens, init_ws_tokens, queues, counters, stack, next_new = _warmstart_context(
        state,
        seed,
        rank_edges,
        ws_edges,
        windows,
        history,
    )
    ws_counts = list(zip(queues, counters))
    birth_targets = _birth_targets_from_real(state, n_records) if birth_control else None
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
        logits, _ws_logits, h = model(init_tok, init_ws)
        progress_every = max(25_000, min(100_000, max(1, n_records // 4)))
        for i in range(n_records):
            ws_pre = _ws_feature_from_counts(ws_counts, ws_edges)
            z = logits[0, -1] / max(float(temperature), 1e-6)
            probs = torch.softmax(z, dim=-1).detach().cpu().numpy()
            if stack:
                probs = probs.copy()
                for tok in range(1, vocab):
                    if int(rank_edges[tok - 1]) >= len(stack):
                        probs[tok] = 0.0
                total = float(probs.sum())
                if total > 0.0:
                    probs /= total
                else:
                    probs[:] = 0.0
                    probs[NEW_TOKEN] = 1.0
            else:
                probs = np.zeros(vocab, dtype=np.float64)
                probs[NEW_TOKEN] = 1.0
            token = int(rng.choice(vocab, p=probs))
            if token == NEW_TOKEN or not stack:
                oid = next_new
                next_new += 1
                stack.insert(0, oid)
            else:
                rank = token_to_rank(token, rank_edges, rng, max_rank=len(stack) - 1)
                if rank < 0 or rank >= len(stack):
                    oid = next_new
                    next_new += 1
                    stack.insert(0, oid)
                    token = NEW_TOKEN
                else:
                    oid = stack.pop(rank)
                    stack.insert(0, oid)
            if (
                birth_targets is not None
                and oid not in emitted_seen
                and len(emitted_seen) >= int(birth_targets[i])
            ):
                forced = _force_emitted_reuse(stack, emitted_seen, emitted_order, rank_edges, rng)
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
            logits, _ws_logits, h = model(step_tok, step_ws, h)
            if i + 1 < n_records and (i + 1) % progress_every == 0:
                print(
                    f"[mattson_denning generate] emitted={i + 1:,} "
                    f"internal_unique={next_new:,} output_unique={len(emitted_seen):,} "
                    f"stack={len(stack):,}",
                    flush=True,
                )
    print(
        f"[mattson_denning generate] emitted={n_records:,} "
        f"internal_unique={next_new:,} output_unique={len(emitted_seen):,} "
        f"stack={len(stack):,}",
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
