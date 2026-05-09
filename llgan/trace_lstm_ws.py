"""R300 — Autoregressive LSTM with Denning working-set count context tokens.

Builds on R298 LSTM (rank-bin tokens, NEW special) and R298e empirical
within-bin sampler. Adds: at every position t, log-binned counts of unique
obj_ids in trailing windows W in {32, 128, 512, 2048, 8192}. These five
auxiliary tokens are embedded and concatenated with the rank-bin token
embedding before going into the LSTM.

Architecture
============
  Inputs at position t:
    rank_token_t  ∈ {0..K}                               (NEW or rank bin)
    ws_token_t,w  ∈ {0..n_ws_bins-1}  for each window w  (log-binned uniques)
  Embeddings:
    rank_emb (E_rank dim) + 5 × ws_emb (E_ws dim each) → input dim
    E_rank + 5*E_ws → 2-layer LSTM (hidden=H) → linear(vocab) head
  Loss:
    cross-entropy on next rank_token

Generation
==========
At each step, online-maintain trailing-window working-set counters via
deque + dict per window. Compute current ws tokens from observed history.
Forward pass yields next-token distribution; sample full softmax;
materialize via empirical-bin rank sampling.

The hypothesis (per Coffman & Denning §7.4): conditioning on multi-scale
working-set sizes gives the model a stable locality-state proxy that
makes its next-token distribution low-entropy at decision points,
reducing seed-trajectory divergence.
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


def tokenize(real_csv: str, max_rows: int, n_rank_bins: int,
             n_ws_bins: int, windows=DEFAULT_WS_WINDOWS):
    """Tokenize trace into rank_tokens + ws_tokens[t, window].

    Returns (rank_tokens, ws_tokens, rank_edges, ws_edges, footprint, bin_ranks).
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

    n_ws_bins_eff = len(ws_edges) - 1
    print(f"[lstm_ws tokenize] n={n:,} footprint={footprint:,} "
          f"rank_K={K} ws_K={n_ws_bins_eff} windows={list(windows)} "
          f"bin_ranks_filled={len(bin_ranks_arr)}", flush=True)
    return rank_tokens, ws_tokens, rank_edges, ws_edges, footprint, bin_ranks_arr, list(windows)


def build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed, hidden):
    import torch
    import torch.nn as nn

    class LstmWS(nn.Module):
        def __init__(self):
            super().__init__()
            self.rank_emb = nn.Embedding(vocab, rank_embed)
            self.ws_emb = nn.ModuleList(
                [nn.Embedding(ws_vocab, ws_embed) for _ in range(n_windows)]
            )
            input_dim = rank_embed + n_windows * ws_embed
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, vocab)

        def forward(self, rank_tok, ws_tok, h=None):
            parts = [self.rank_emb(rank_tok)]
            for wi, emb in enumerate(self.ws_emb):
                parts.append(emb(ws_tok[:, :, wi]))
            x = torch.cat(parts, dim=-1)
            o, h = self.lstm(x, h)
            return self.head(o), h

    return LstmWS()


def train_model(rank_tokens, ws_tokens, vocab, ws_vocab, n_windows,
                 rank_embed, ws_embed, hidden, seq_len, batch, epochs, lr, seed):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rank_t = torch.from_numpy(rank_tokens).long()
    ws_t = torch.from_numpy(ws_tokens).long()
    n_total = len(rank_tokens)
    n_train = n_total - seq_len - 1
    print(f"[lstm_ws train] V={vocab} ws_V={ws_vocab} E_rank={rank_embed} "
          f"E_ws={ws_embed} H={hidden} seq={seq_len} batch={batch} epochs={epochs} "
          f"on {device}", flush=True)
    print(f"[lstm_ws train] params={sum(p.numel() for p in model.parameters()):,}", flush=True)

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
            logits, _ = model(x_rank, x_ws)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += float(loss.item())
            if bi % max(1, n_batches // 10) == 0:
                print(f"[lstm_ws train] ep {ep+1}/{epochs} batch {bi}/{n_batches} "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"[lstm_ws train] ep {ep+1}: avg loss = {running_loss / n_batches:.4f}", flush=True)
    return model


def update_ws_state(queues, counts, obj_id, windows, ws_edges):
    """After emitting obj_id, update each window's state and return current ws tokens (BEFORE next access)."""
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
              seed, vocab, ws_vocab, history):
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

    rank_hist = [NEW_TOKEN] * history
    ws_hist = np.zeros((history, len(windows)), dtype=np.int64)

    fresh = recycle = stack_n = 0
    with torch.no_grad():
        for j in range(n_records):
            x_rank = torch.tensor([rank_hist], dtype=torch.long, device=device)
            x_ws = torch.tensor(ws_hist[None, :, :], dtype=torch.long, device=device)
            logits, _ = model(x_rank, x_ws)
            ll = logits[0, -1].cpu().numpy()
            ll -= ll.max()
            probs = np.exp(ll); probs /= probs.sum()
            tok = int(rng.choice(vocab, p=probs))

            if tok == NEW_TOKEN or not stack:
                addr = next_new; next_new += 1
                stack.insert(0, addr); fresh += 1
            else:
                bin_idx = tok - 1
                if bin_idx in bin_ranks_arr:
                    rank = int(rng.choice(bin_ranks_arr[bin_idx]))
                else:
                    lo, hi = int(rank_edges[bin_idx]), int(rank_edges[min(bin_idx + 1, len(rank_edges) - 1)])
                    rank = int(rng.integers(lo, max(hi, lo + 1)))
                if rank < 0 or rank >= len(stack):
                    addr = next_new; next_new += 1
                    stack.insert(0, addr); recycle += 1
                else:
                    addr = stack.pop(rank)
                    stack.insert(0, addr); stack_n += 1
            out[j] = addr

            ws_now = update_ws_state(queues, counts, addr, windows, ws_edges)

            rank_hist.pop(0); rank_hist.append(tok)
            ws_hist = np.roll(ws_hist, -1, axis=0)
            ws_hist[-1] = ws_now

            if (j + 1) % 25000 == 0:
                print(f"[lstm_ws gen] {j+1:,}/{n_records:,}", flush=True)

    print(f"[lstm_ws gen] FRESH={fresh:,} RECYCLE={recycle:,} IN_STACK={stack_n:,}", flush=True)
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
    rank_tok, ws_tok, rank_edges, ws_edges, footprint, bin_ranks, windows = tokenize(
        args.real, args.max_rows, args.n_bins, args.ws_bins,
        windows=tuple(int(x) for x in args.ws_windows.split(",")))
    vocab = len(rank_edges)
    ws_vocab = len(ws_edges) - 1
    model = train_model(rank_tok, ws_tok, vocab=vocab, ws_vocab=ws_vocab,
                         n_windows=len(windows),
                         rank_embed=args.rank_embed, ws_embed=args.ws_embed,
                         hidden=args.hidden, seq_len=args.seq_len,
                         batch=args.batch, epochs=args.epochs, lr=args.lr,
                         seed=args.seed)
    state = {
        "rank_edges": rank_edges, "ws_edges": ws_edges, "windows": windows,
        "vocab": vocab, "ws_vocab": ws_vocab,
        "footprint": footprint, "n_bins": args.n_bins, "ws_bins": args.ws_bins,
        "history": args.seq_len,
        "model_state": model.state_dict(),
        "model_config": {
            "rank_embed": args.rank_embed, "ws_embed": args.ws_embed,
            "hidden": args.hidden, "vocab": vocab, "ws_vocab": ws_vocab,
            "n_windows": len(windows),
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
                         cfg["rank_embed"], cfg["ws_embed"], cfg["hidden"]).to(device)
    model.load_state_dict(state["model_state"])
    out = generate(model,
                   state["rank_edges"], state["ws_edges"], state["bin_ranks_arr"],
                   state["windows"], args.n, args.seed, cfg["vocab"], cfg["ws_vocab"],
                   state.get("history", 64))
    write_csv(args.output, out)
    print(f"[lstm_ws gen] wrote {args.n:,} → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="R300 LSTM + WS context")
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
    pf.add_argument("--seq-len", type=int, default=256)
    pf.add_argument("--batch", type=int, default=128)
    pf.add_argument("--epochs", type=int, default=25)
    pf.add_argument("--lr", type=float, default=1e-3)
    pf.add_argument("--seed", type=int, default=42)
    pf.set_defaults(fn=cmd_fit)
    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seed", type=int, default=42)
    pg.set_defaults(fn=cmd_generate)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
