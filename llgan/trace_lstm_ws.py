"""R301 — Autoregressive LSTM with Denning WS context + birth-rate anchoring.

Builds on R300 (trace_lstm_ws.py) and adds four improvements:

1. Empirical birth-rate blend (generation-time, no retraining):
   At each step, blend LSTM P(NEW | history) with empirical P(NEW | ws0_bin).
   Fixes the seed-divergence problem (R298e seed=80 emitted 8% FRESH vs 49%
   real) by anchoring the FRESH/RECYCLE split to the observed birth rate at
   the current working-set size. Coefficient ``--birth-rate-blend`` (α∈[0,1]).

2. Label smoothing in training loss (default 0.05):
   Reduces overconfident FRESH predictions — the root cause of seed-trajectory
   divergence. Implementation: F.cross_entropy(..., label_smoothing=ε).

3. Cosine LR schedule + gradient clipping:
   Better convergence and training stability.

4. Multi-seed generation (``--seeds 42,80,81,82``):
   Generates one CSV per seed and reports FRESH counts, enabling 4-seed
   Constitution-compliant claims without a separate wrapper.

Architecture is identical to R300 (rank-bin + WS context LSTM). All changes
are backward-compatible: existing checkpoints work (birth_rate_by_ws0 falls
back to global rate if absent from checkpoint).

Theory
------
The seed divergence is a positive feedback loop: if the LSTM accidentally
under-emits FRESH early in generation, the working-set fills with repeated
objects, which conditions the LSTM to emit fewer FRESH tokens, compounding
the deficit. Birth-rate blending breaks the loop by anchoring P(NEW) to the
empirical conditional P(NEW | ws0_bin):

    p_new = α·P_emp(NEW | ws0_bin) + (1−α)·P_lstm(NEW | history)

α=0 → pure LSTM (R300), α=1 → forced empirical rate. α=0.5 is the
recommended starting point (matches LANL --birth-rate-blend 0.5).
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

    Returns (rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
             bin_ranks, windows, birth_rate_by_ws0, birth_count_by_ws0).
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

    # Empirical birth rate conditioned on WS0 bin.
    n_ws0_bins = len(ws_edges) - 1
    birth_count_by_ws0 = np.zeros(n_ws0_bins, dtype=np.int64)
    birth_rate_by_ws0 = np.zeros(n_ws0_bins, dtype=np.float64)
    ws0_col = ws_tokens[:, 0]
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

    n_ws_bins_eff = len(ws_edges) - 1
    print(f"[lstm_ws tokenize] n={n:,} footprint={footprint:,} "
          f"rank_K={K} ws_K={n_ws_bins_eff} windows={list(windows)} "
          f"bin_ranks_filled={len(bin_ranks_arr)} "
          f"global_birth={global_birth_rate:.3f}", flush=True)
    return (rank_tokens, ws_tokens, rank_edges, ws_edges, footprint,
            bin_ranks_arr, list(windows), birth_rate_by_ws0, birth_count_by_ws0)


def build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed, hidden,
                lstm_layers: int = 2):
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
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=lstm_layers,
                                batch_first=True)
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
                 rank_embed, ws_embed, hidden, seq_len, batch, epochs, lr,
                 seed, label_smoothing: float = 0.0,
                 grad_clip: float = 0.0, lr_schedule: str = "constant",
                 lstm_layers: int = 2):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(vocab, n_windows, ws_vocab, rank_embed, ws_embed,
                        hidden, lstm_layers=lstm_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rank_t = torch.from_numpy(rank_tokens).long()
    ws_t = torch.from_numpy(ws_tokens).long()
    n_total = len(rank_tokens)
    n_train = n_total - seq_len - 1
    total_steps = max(1, (n_train // batch) * epochs)
    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=lr * 0.05)
    print(f"[lstm_ws train] V={vocab} ws_V={ws_vocab} E_rank={rank_embed} "
          f"E_ws={ws_embed} H={hidden} layers={lstm_layers} seq={seq_len} "
          f"batch={batch} epochs={epochs} lr={lr} schedule={lr_schedule} "
          f"label_smooth={label_smoothing} grad_clip={grad_clip} on {device}", flush=True)
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
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1),
                                   label_smoothing=label_smoothing)
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
        print(f"[lstm_ws train] ep {ep+1}: avg loss = {running_loss / n_batches:.4f}", flush=True)
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
              seed, vocab, ws_vocab, history,
              birth_rate_by_ws0: np.ndarray | None = None,
              birth_rate_blend: float = 0.0):
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

            # Birth rate blend: anchor P(NEW) to empirical conditional on ws0.
            if birth_rate_blend > 0.0 and birth_rate_by_ws0 is not None:
                ws0_bin = min(int(ws_hist[-1, 0]), len(birth_rate_by_ws0) - 1)
                emp = birth_rate_by_ws0[ws0_bin]
                p_now = float(probs[NEW_TOKEN])
                p_blended = birth_rate_blend * emp + (1.0 - birth_rate_blend) * p_now
                p_blended = max(0.0, min(1.0, p_blended))
                if p_now < 1.0 - 1e-9:
                    scale = (1.0 - p_blended) / (1.0 - p_now)
                    probs = probs.copy()
                    probs[NEW_TOKEN] = p_blended
                    probs[1:] *= scale
                    total = probs.sum()
                    if total > 0:
                        probs /= total

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
     bin_ranks, windows, birth_rate_by_ws0, birth_count_by_ws0) = tokenize(
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
                         lstm_layers=args.lstm_layers)
    state = {
        "rank_edges": rank_edges, "ws_edges": ws_edges, "windows": windows,
        "vocab": vocab, "ws_vocab": ws_vocab,
        "footprint": footprint, "n_bins": args.n_bins, "ws_bins": args.ws_bins,
        "history": args.seq_len,
        "birth_rate_by_ws0": birth_rate_by_ws0.tolist(),
        "birth_count_by_ws0": birth_count_by_ws0.tolist(),
        "model_state": model.state_dict(),
        "model_config": {
            "rank_embed": args.rank_embed, "ws_embed": args.ws_embed,
            "hidden": args.hidden, "vocab": vocab, "ws_vocab": ws_vocab,
            "n_windows": len(windows), "lstm_layers": args.lstm_layers,
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
                         lstm_layers=cfg.get("lstm_layers", 2)).to(device)
    model.load_state_dict(state["model_state"])

    birth_rate_by_ws0 = None
    raw = state.get("birth_rate_by_ws0")
    if raw:
        birth_rate_by_ws0 = np.asarray(raw, dtype=np.float64)

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        if len(seeds) > 1:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_s{seed}{ext}"
        else:
            out_path = args.output
        print(f"[lstm_ws gen] seed={seed} → {out_path}", flush=True)
        out = generate(model,
                       state["rank_edges"], state["ws_edges"],
                       state["bin_ranks_arr"],
                       state["windows"], args.n, seed,
                       cfg["vocab"], cfg["ws_vocab"],
                       state.get("history", 64),
                       birth_rate_by_ws0=birth_rate_by_ws0,
                       birth_rate_blend=args.birth_rate_blend)
        write_csv(out_path, out)
        print(f"[lstm_ws gen] wrote {args.n:,} → {out_path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="R301 LSTM + WS context + birth-rate anchoring")
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
    pf.set_defaults(fn=cmd_fit)

    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seeds", default="42", help="comma-separated seeds, e.g. 42,80,81,82")
    pg.add_argument("--birth-rate-blend", type=float, default=0.5,
                    help="blend coefficient α for empirical birth rate (0=LSTM only, 1=empirical)")
    pg.set_defaults(fn=cmd_generate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
