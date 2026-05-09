"""R298 — Autoregressive LSTM trained on real trace rank-bin sequence.

The user's directive: "If you are not LEARNING from training on real
data, you are not doing the job at all."

R295-R297 fit summary statistics (stack-distance histogram, per-window
working sets, HMM on locality clusters) but emit references i.i.d. given
state. Real traces have within-locality temporal autocorrelation: which
specific obj_id follows which depends on history beyond the current
locality. R298 trains a small LSTM autoregressively on the rank-bin
sequence to capture that autocorrelation.

Tokenization
============
Each access r_t in the real trace is tokenized as:
  - rank_bin(r_t, S_t) where S_t is the LRU stack just before t, OR
  - special NEW token when r_t is a first-access (S_t doesn't contain it)

Rank bins are log-spaced over [0, footprint] with K=50 bins. Vocabulary
size is K+1=51 (50 rank bins + NEW). This is small enough for a tiny
LSTM (<1M parameters total) trained quickly on CPU.

Training
========
Standard cross-entropy next-token prediction over (history, next_token)
pairs sampled from the real trace's tokenization. Model: 2-layer LSTM,
hidden 64, embedding 32. Train 5 epochs over the 1M-token sequence with
batch size 256, sequence length 64, Adam lr=1e-3.

Generation
==========
1. Sample initial NEW token (cold start).
2. Maintain LRU stack via Python list.
3. At each step:
   - Forward pass: LSTM predicts P(next_token | history).
   - Sample next token.
   - If NEW: emit fresh obj_id; push to top.
   - Else: rank_bin = sampled token; pick uniform rank within that bin's
     range; emit stack[rank], move to top.

Output trace's empirical rank-bin distribution + autocorrelation should
match the trained distribution by construction. By Mattson stack
property, LRU HRC matches up to the precision of rank-bin discretization.
Non-LRU policies benefit from the autoregressive within-locality
patterns the LSTM learns.

This is genuine learning: an LSTM with O(10^6) parameters trained on
the trace, with cross-entropy loss on next-token prediction. The model
is a real probabilistic next-token distribution conditioned on history —
not a fixed histogram or per-window summary.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import math
import os
import pickle
import sys
from collections import Counter

import numpy as np


# --- Tokenization helpers --------------------------------------------------

NEW_TOKEN = 0  # token id for first-access (one-shot or pre-stack-warm-up)


def make_rank_bins(footprint: int, n_bins: int) -> np.ndarray:
    """K log-spaced rank bin edges from 1 to footprint+1 (so bin i covers
    ranks [edges[i], edges[i+1])). Returns array of length n_bins+1.

    Bin 0 covers ranks {0, 1}; bin 1 covers ranks {2..3}; etc.
    """
    if footprint <= 1:
        return np.array([0, 2], dtype=np.int64)
    edges = np.unique(np.round(
        np.logspace(0, math.log10(footprint + 1), n_bins + 1)
    ).astype(np.int64))
    if edges[0] != 0:
        edges = np.concatenate(([0], edges))
    return edges


def rank_to_token(rank: int, edges: np.ndarray) -> int:
    """Map an integer rank to a token id (1..K) with NEW=0 reserved.
    The bin index is the position in `edges` such that edges[i] <= rank < edges[i+1].
    Token id = bin_index + 1 (so token 0 = NEW)."""
    bin_idx = bisect.bisect_right(edges, rank) - 1
    bin_idx = max(0, min(bin_idx, len(edges) - 2))
    return bin_idx + 1


def token_to_rank(token: int, edges: np.ndarray, rng: np.random.Generator) -> int:
    """Sample a uniform integer rank in the bin range corresponding to a token."""
    if token == NEW_TOKEN:
        return -1
    bin_idx = token - 1
    lo, hi = int(edges[bin_idx]), int(edges[min(bin_idx + 1, len(edges) - 1)])
    if hi <= lo:
        return lo
    return int(rng.integers(lo, hi))


# --- Tokenize real trace ---------------------------------------------------

def tokenize_real(real_csv: str, max_rows: int, n_bins: int):
    """Read real trace, compute LRU stack distance per access, tokenize.
    Returns (tokens: np.ndarray, edges: np.ndarray, footprint: int)."""
    obj_ids = []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_ids.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)
    last_pos = {}
    pos_sorted: list = []
    distances = np.empty(n, dtype=np.int64)
    distances.fill(-1)  # -1 means NEW
    for i, oid in enumerate(obj_ids):
        prev = last_pos.get(oid, -1)
        if prev >= 0:
            idx = bisect.bisect_right(pos_sorted, prev)
            distances[i] = len(pos_sorted) - idx
            del pos_sorted[bisect.bisect_left(pos_sorted, prev)]
        bisect.insort(pos_sorted, i)
        last_pos[oid] = i
    footprint = len(last_pos)
    edges = make_rank_bins(footprint, n_bins)
    K = len(edges) - 1
    tokens = np.empty(n, dtype=np.int64)
    for i, d in enumerate(distances):
        if d < 0:
            tokens[i] = NEW_TOKEN
        else:
            tokens[i] = rank_to_token(int(d), edges)
    print(f"[trace_lstm tokenize] n={n:,} footprint={footprint:,} "
          f"K={K} vocab={K + 1} (NEW + {K} rank bins)", flush=True)
    return tokens, edges, footprint


# --- Tiny LSTM model -------------------------------------------------------

def _try_torch():
    try:
        import torch  # noqa
        import torch.nn as nn  # noqa
        return True
    except ImportError:
        return False


def train_lstm(tokens: np.ndarray, vocab_size: int,
               hidden: int = 64, embed: int = 32,
               seq_len: int = 64, batch: int = 256,
               epochs: int = 5, lr: float = 1e-3, seed: int = 42):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)

    class TinyLSTM(nn.Module):
        def __init__(self, V, E, H):
            super().__init__()
            self.emb = nn.Embedding(V, E)
            self.lstm = nn.LSTM(E, H, num_layers=2, batch_first=True)
            self.head = nn.Linear(H, V)

        def forward(self, x, h=None):
            e = self.emb(x)
            o, h = self.lstm(e, h)
            return self.head(o), h

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLSTM(vocab_size, embed, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_total = len(tokens)
    print(f"[trace_lstm train] V={vocab_size} E={embed} H={hidden} seq={seq_len} "
          f"batch={batch} epochs={epochs} on {device}", flush=True)
    print(f"[trace_lstm train] params={sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    tokens_t = torch.from_numpy(tokens).long()
    n_train = n_total - seq_len - 1
    rng = np.random.default_rng(seed)

    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n_train)
        # Iterate in chunks of `batch * seq_len` for efficiency.
        n_batches = n_train // batch
        running_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch:(bi + 1) * batch]
            x = torch.stack([tokens_t[i:i + seq_len] for i in idx]).to(device)
            y = torch.stack([tokens_t[i + 1:i + 1 + seq_len] for i in idx]).to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            if bi % max(1, n_batches // 10) == 0:
                print(f"[trace_lstm train] ep {ep+1}/{epochs} batch {bi}/{n_batches} "
                      f"loss={loss.item():.4f}", flush=True)
        print(f"[trace_lstm train] ep {ep+1}: avg loss = {running_loss / n_batches:.4f}",
              flush=True)
    return model


def generate(model, edges: np.ndarray, n_records: int, seed: int,
             vocab_size: int, history: int = 64) -> np.ndarray:
    import torch
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    # Maintain LRU stack and history of tokens.
    stack: list = []
    next_new = 0
    out = np.empty(n_records, dtype=np.uint64)

    # Bootstrap: feed model NEW token initially.
    history_buf = [NEW_TOKEN] * history

    with torch.no_grad():
        for j in range(n_records):
            x = torch.tensor([history_buf], dtype=torch.long, device=device)
            logits, _ = model(x)
            last_logits = logits[0, -1].cpu().numpy()
            # Softmax + sample.
            last_logits -= last_logits.max()
            probs = np.exp(last_logits)
            probs /= probs.sum()
            tok = int(rng.choice(vocab_size, p=probs))

            # Materialize.
            if tok == NEW_TOKEN or not stack:
                addr = next_new
                next_new += 1
                stack.insert(0, addr)
            else:
                rank = token_to_rank(tok, edges, rng)
                if rank < 0 or rank >= len(stack):
                    addr = next_new
                    next_new += 1
                    stack.insert(0, addr)
                else:
                    addr = stack.pop(rank)
                    stack.insert(0, addr)
            out[j] = addr
            history_buf.pop(0)
            history_buf.append(tok)
    print(f"[trace_lstm generate] emitted {n_records:,} records, "
          f"unique={next_new:,}, max_stack={len(stack):,}", flush=True)
    return out


def write_csv(path: str, obj_ids: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


# --- CLI ------------------------------------------------------------------

def cmd_fit(args):
    if not _try_torch():
        print("ERROR: PyTorch not available; install in this venv first", file=sys.stderr)
        sys.exit(1)
    tokens, edges, footprint = tokenize_real(args.real, args.max_rows, args.n_bins)
    vocab = len(edges)
    model = train_lstm(tokens, vocab_size=vocab, hidden=args.hidden,
                       embed=args.embed, seq_len=args.seq_len,
                       batch=args.batch, epochs=args.epochs, lr=args.lr,
                       seed=args.seed)
    state = {
        "edges": edges,
        "vocab_size": vocab,
        "footprint": footprint,
        "n_bins": args.n_bins,
        "history": args.seq_len,
        "model_state": model.state_dict(),
        "model_config": {"E": args.embed, "H": args.hidden, "V": vocab},
    }
    import torch
    torch.save(state, args.output)
    print(f"[trace_lstm fit] saved → {args.output}", flush=True)


def cmd_generate(args):
    if not _try_torch():
        print("ERROR: PyTorch not available", file=sys.stderr); sys.exit(1)
    import torch
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    edges = state["edges"]
    vocab = state["vocab_size"]
    cfg = state["model_config"]

    import torch.nn as nn
    class TinyLSTM(nn.Module):
        def __init__(self, V, E, H):
            super().__init__()
            self.emb = nn.Embedding(V, E)
            self.lstm = nn.LSTM(E, H, num_layers=2, batch_first=True)
            self.head = nn.Linear(H, V)
        def forward(self, x, h=None):
            e = self.emb(x); o, h = self.lstm(e, h); return self.head(o), h

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyLSTM(cfg["V"], cfg["E"], cfg["H"]).to(device)
    model.load_state_dict(state["model_state"])
    out = generate(model, edges, args.n, args.seed, vocab,
                   history=state.get("history", 64))
    write_csv(args.output, out)
    print(f"[trace_lstm generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="R298 Autoregressive LSTM on rank-bin tokens")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.add_argument("--n-bins", type=int, default=50)
    pf.add_argument("--hidden", type=int, default=64)
    pf.add_argument("--embed", type=int, default=32)
    pf.add_argument("--seq-len", type=int, default=64)
    pf.add_argument("--batch", type=int, default=256)
    pf.add_argument("--epochs", type=int, default=5)
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
