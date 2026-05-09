"""R299 — Cache-State-Aware TraceTransformer.

The user's directive: "Ad hoc histogram fits are bullshit. They are YOU
doing what the model should be doing: learning how to fit the data."

R298's LSTM bins ranks into 50 log-spaced buckets — that is hand-encoded
representation. R299 lets the model learn its own representation
end-to-end. Three architectural choices honor the directive:

1. **Hash-factored obj_id embedding** instead of binning. Each obj_id
   maps to 16 hash codes (each in [0, 4095]); its embedding is the sum
   of 16 small embedding tables. Bounded memory regardless of footprint.
   The model decides what features of the obj_id matter; we don't bin.

2. **Cache-state-aware decoding**. At each step the model gets the
   current LRU stack snapshot (top-K entries) as cross-attention
   context. The discrete output is over actions {NEW, stack[0],
   stack[1], ..., stack[K-1]} — exactly the cache-action vocabulary,
   not a rank-bin proxy. The model knows the cache configuration when
   deciding what to emit.

3. **Loss = cross-entropy on real-trace cache-actions**. Per real
   trace position, the "ground truth" is NEW (if first access) or
   REUSE@k (where k is the actual LRU stack position of the accessed
   obj_id at that moment). No binning; the action is the discrete
   cache event itself.

Generation: maintain LRU stack; at each step run model; sample action;
emit fresh obj_id if NEW or take stack[k] if REUSE@k. Output trace's
cache-action sequence is exactly what the model was trained to predict.

This is the right learning problem for HRC-MAE. By Mattson, LRU HRC at
cache c = P(REUSE @ k for k < c). If the model's action distribution
matches real, LRU HRC matches by construction.

Trade-off: K (max stack depth modeled) caps the deepest reuse the
model can predict. Reuses at depth > K become NEW (fresh allocations).
With K=128 we model the cache-relevant range (race surface tops out at
cache=8192 but most weight is at ≤128).
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import os
import pickle
import sys

import numpy as np


# ----- Tokenization: real trace → action sequence -------------------------

ACTION_FRESH = -1   # true one-shot: never seen before
ACTION_RECYCLE = -2  # deep reuse: stack distance > K (was seen but evicted)


def trace_to_actions(real_csv: str, max_rows: int, K: int):
    """Read real trace, compute cache-action sequence + stack snapshots.

    Action vocabulary (R299b — fixes the conflation bug):
      - ACTION_FRESH (-1)   if d < 0 (first access — true one-shot)
      - ACTION_RECYCLE (-2) if d >= K (deep reuse beyond modeled stack)
      - REUSE@k (0..K-1)    otherwise (in-stack reuse at position k = d)

    Splitting the previous "NEW" class into FRESH vs RECYCLE preserves the
    distinction between (a) genuine first accesses and (b) reuses too deep
    for the K-position attention. At decode, RECYCLE samples from the stale
    pool of recently-evicted obj_ids; FRESH allocates a new address.

    Returns:
      obj_ids: np.ndarray of shape [N] (uint64) — the trace itself
      actions: np.ndarray of shape [N] (int64) — -1, -2, or 0..K-1
      footprint: int — distinct obj_ids in trace
    """
    print(f"[trace_transformer] reading {real_csv} K={K}", flush=True)
    obj_list = []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_list.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_list)
    obj_arr = np.array(obj_list, dtype=np.uint64)

    # Compute LRU stack distance for each access.
    last_pos = {}
    pos_sorted: list = []
    actions = np.empty(n, dtype=np.int64)
    actions.fill(ACTION_FRESH)
    for i, oid in enumerate(obj_list):
        prev = last_pos.get(oid, -1)
        if prev >= 0:
            idx = bisect.bisect_right(pos_sorted, prev)
            d = len(pos_sorted) - idx
            actions[i] = d if d < K else ACTION_RECYCLE
            del pos_sorted[bisect.bisect_left(pos_sorted, prev)]
        bisect.insort(pos_sorted, i)
        last_pos[oid] = i
    footprint = len(last_pos)

    n_fresh = int((actions == ACTION_FRESH).sum())
    n_recycle = int((actions == ACTION_RECYCLE).sum())
    n_in_stack = n - n_fresh - n_recycle
    print(f"[trace_transformer] n={n:,} footprint={footprint:,} "
          f"fresh={n_fresh:,} ({n_fresh/n:.3f}) "
          f"recycle={n_recycle:,} ({n_recycle/n:.3f}) "
          f"in_stack={n_in_stack:,} ({n_in_stack/n:.3f})",
          flush=True)
    return obj_arr, actions, footprint


# ----- Hash-factored obj_id embedding -------------------------------------

def obj_id_to_hashes(obj_ids: np.ndarray, n_hashes: int = 16,
                     bucket_size: int = 4096) -> np.ndarray:
    """Map obj_ids to n_hashes integer buckets via different hash seeds.
    Uses simple multiplicative hashes with prime constants.
    Returns int64 array of shape [N, n_hashes]."""
    primes = [
        2654435761, 40503, 2246822519, 3266489917, 374761393, 1376312589,
        2073600871, 2150448427, 2807114953, 3132291347, 691013249,
        1779033703, 3144134277, 1013904242, 2773480762, 1359893119,
    ][:n_hashes]
    out = np.empty((len(obj_ids), n_hashes), dtype=np.int64)
    obj64 = obj_ids.astype(np.uint64)
    for i, p in enumerate(primes):
        # Simple multiplicative hash, mod bucket_size.
        out[:, i] = ((obj64 * np.uint64(p)) % np.uint64(bucket_size)).astype(np.int64)
    return out


# ----- Model definition (PyTorch) -----------------------------------------

def _build_model(K: int, n_hashes: int, bucket_size: int,
                 emb_dim: int, n_heads: int, n_layers: int, history: int):
    """Build TraceTransformer with K+2 output classes:
    {REUSE@0, ..., REUSE@K-1, RECYCLE, FRESH}."""
    import torch
    import torch.nn as nn

    class TraceTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.history = history
            self.emb_dim = emb_dim
            self.n_hashes = n_hashes
            self.bucket_size = bucket_size
            # Hash-factored obj_id embedding.
            self.hash_embed = nn.Embedding(n_hashes * bucket_size, emb_dim)
            # Special tokens: RECYCLE (deep reuse), FRESH (one-shot).
            self.recycle_token = nn.Parameter(torch.randn(emb_dim))
            self.fresh_token = nn.Parameter(torch.randn(emb_dim))
            # Position encodings for history and stack.
            self.history_pos = nn.Embedding(history, emb_dim)
            self.stack_pos = nn.Embedding(K, emb_dim)
            # Transformer encoder over (history ∪ stack).
            enc_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim, nhead=n_heads,
                dim_feedforward=4 * emb_dim, batch_first=True,
                norm_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.query_proj = nn.Linear(emb_dim, emb_dim)
            self.key_proj = nn.Linear(emb_dim, emb_dim)

        def embed_objs(self, hash_codes: "torch.Tensor") -> "torch.Tensor":
            offsets = (torch.arange(self.n_hashes,
                                    device=hash_codes.device) * self.bucket_size)
            offset_codes = hash_codes + offsets
            emb = self.hash_embed(offset_codes)
            return emb.sum(dim=-2)

        def forward(self, hist_hash: "torch.Tensor", stack_hash: "torch.Tensor"):
            """Returns: logits [B, K+2] over (REUSE@0..K-1, RECYCLE, FRESH)."""
            B = hist_hash.shape[0]
            M = hist_hash.shape[1]
            K = stack_hash.shape[1]
            hist_emb = self.embed_objs(hist_hash)
            stack_emb = self.embed_objs(stack_hash)
            hist_pos = self.history_pos(torch.arange(M, device=hist_hash.device))
            stack_pos = self.stack_pos(torch.arange(K, device=hist_hash.device))
            hist_emb = hist_emb + hist_pos.unsqueeze(0)
            stack_emb = stack_emb + stack_pos.unsqueeze(0)
            seq = torch.cat([hist_emb, stack_emb], dim=1)
            enc = self.encoder(seq)
            query = self.query_proj(enc[:, :M].mean(dim=1, keepdim=True))
            recycle_key = self.recycle_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            fresh_key = self.fresh_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
            keys = self.key_proj(
                torch.cat([enc[:, M:], recycle_key, fresh_key], dim=1)
            )  # [B, K+2, D]
            logits = (query * keys).sum(dim=-1) / (self.emb_dim ** 0.5)  # [B, K+2]
            return logits

    return TraceTransformer()


# ----- Training data prep -------------------------------------------------

def build_training_examples(obj_ids: np.ndarray, actions: np.ndarray,
                             K: int, history: int, n_hashes: int,
                             bucket_size: int):
    """For each position t in the trace (after warm-up), build:
      - hist_hash: hash codes of last `history` obj_ids
      - stack_hash: hash codes of current top-K LRU stack
      - target: action[t] (REUSE@k or ACTION_NEW)
    Returns lists of arrays usable for batched training.
    """
    n = len(obj_ids)
    # Replay the trace, building stack and snapshots.
    hashes_full = obj_id_to_hashes(obj_ids, n_hashes=n_hashes,
                                    bucket_size=bucket_size)  # [N, H]
    stack_idx_seq = np.empty((n, K), dtype=np.int64)  # indices into obj_ids of top-K
    stack_idx_seq.fill(-1)  # -1 = empty slot
    stack: list = []  # list of indices in obj_ids
    obj_to_stack_pos = {}  # obj_id -> position in stack
    for t in range(n):
        # Snapshot top-K BEFORE incorporating obj at t.
        for k in range(K):
            stack_idx_seq[t, k] = stack[k] if k < len(stack) else -1
        # Now process obj at t.
        oid = int(obj_ids[t])
        if oid in obj_to_stack_pos:
            old_pos = obj_to_stack_pos[oid]
            stack.pop(old_pos)
        stack.insert(0, t)
        obj_to_stack_pos[oid] = 0
        # Update positions for shifted-down items (within K).
        if len(stack) > K * 2:
            # Truncate stack to K for memory; we lose info on deep reuses.
            removed = stack[K * 2:]
            stack = stack[:K * 2]
            for r_t in removed:
                obj_to_stack_pos.pop(int(obj_ids[r_t]), None)
        # Refresh obj_to_stack_pos mappings.
        for pos, idx in enumerate(stack[:K * 2]):
            obj_to_stack_pos[int(obj_ids[idx])] = pos
    return hashes_full, stack_idx_seq


# ----- Training -----------------------------------------------------------

def train_transformer(real_csv: str, K: int = 128, history: int = 64,
                       n_hashes: int = 16, bucket_size: int = 4096,
                       emb_dim: int = 96, n_heads: int = 4, n_layers: int = 3,
                       batch: int = 128, epochs: int = 10, lr: float = 1e-3,
                       seed: int = 42, max_rows: int = 0,
                       class_balance: bool = False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obj_ids, actions, footprint = trace_to_actions(real_csv, max_rows, K)
    n = len(obj_ids)
    print(f"[trace_transformer train] building training examples...", flush=True)
    hashes_full, stack_idx_seq = build_training_examples(
        obj_ids, actions, K, history, n_hashes, bucket_size)
    print(f"[trace_transformer train] examples ready", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(K, n_hashes, bucket_size, emb_dim, n_heads, n_layers,
                          history).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[trace_transformer train] device={device} params={n_params:,}", flush=True)

    # Pre-compute zero-vector embedding row for empty slots.
    # We'll use bucket_size as a sentinel hash that points outside normal range;
    # adjust by adding a single special row at offset n_hashes*bucket_size.
    # Simpler: fill empty slots with hash codes of a "padding" obj_id (0).
    # Use zeros — note this aliases to the same row for all empty slots, fine.

    hashes_t = torch.from_numpy(hashes_full).long()
    actions_t = torch.from_numpy(actions).long()

    # Output classes: {REUSE@0..K-1, RECYCLE = K, FRESH = K+1}.
    target_class = actions_t.clone()
    target_class[actions_t == ACTION_RECYCLE] = K
    target_class[actions_t == ACTION_FRESH] = K + 1

    # Precompute stack hash sequence as needed during training.
    stack_idx_t = torch.from_numpy(stack_idx_seq).long()

    n_train = max(history, K)
    valid_t = list(range(n_train, n))
    rng.shuffle(valid_t)
    valid_t = np.array(valid_t)
    print(f"[trace_transformer train] training positions: {len(valid_t):,}", flush=True)

    # Class-balanced weights: inverse-sqrt of class frequency, normalized to mean 1.
    # Counters class imbalance (49% FRESH dominates). Square-root softens the
    # extreme weighting that pure inverse-frequency would impose on rare in-stack
    # classes (which can have <0.1% frequency individually).
    weights = None
    if class_balance:
        cls_count = np.bincount(target_class.numpy(), minlength=K + 2)
        cls_freq = cls_count / max(cls_count.sum(), 1)
        # inverse-sqrt with smoothing
        w = 1.0 / np.sqrt(cls_freq + 1e-6)
        # Normalize so mean weight = 1.
        w = w * (cls_freq * w).sum()
        w[cls_freq == 0] = 0.0
        weights = torch.from_numpy(w.astype(np.float32)).to(device)
        print(f"[trace_transformer train] class-balanced weights: "
              f"FRESH={w[K+1]:.2f} RECYCLE={w[K]:.2f} "
              f"IN_STACK[0]={w[0]:.2f} IN_STACK[127]={w[127]:.2f}", flush=True)

    for ep in range(epochs):
        model.train()
        rng.shuffle(valid_t)
        n_batches = len(valid_t) // batch
        running_loss = 0.0
        running_acc = 0.0
        for bi in range(n_batches):
            ts = valid_t[bi * batch:(bi + 1) * batch]
            # Build batch.
            hist_idx = np.stack([np.arange(t - history, t) for t in ts])  # [B, M]
            hist_h = hashes_t[hist_idx]  # [B, M, H]
            stack_idx_batch = stack_idx_t[ts]  # [B, K]
            # Replace -1 with 0 (use obj_id 0 as padding; hash table same row each).
            sb = stack_idx_batch.clamp(min=0)
            stack_h = hashes_t[sb]  # [B, K, H]
            # Mask the slots that were -1 (empty); use a dedicated "empty" hash row
            # by setting those hash codes to 0 (same as padding).
            mask = (stack_idx_batch == -1).unsqueeze(-1).expand(-1, -1, n_hashes)
            stack_h = stack_h.masked_fill(mask, 0)

            tgt = target_class[ts]

            hist_h = hist_h.to(device)
            stack_h = stack_h.to(device)
            tgt = tgt.to(device)

            logits = model(hist_h, stack_h)
            loss = F.cross_entropy(logits, tgt, weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            running_acc += float((logits.argmax(-1) == tgt).float().mean().item())
            if bi % max(1, n_batches // 20) == 0:
                print(f"[trace_transformer train] ep {ep+1}/{epochs} bi {bi}/{n_batches} "
                      f"loss={loss.item():.4f} acc={(logits.argmax(-1)==tgt).float().mean().item():.3f}",
                      flush=True)
        print(f"[trace_transformer train] ep {ep+1}: avg loss = {running_loss/n_batches:.4f} "
              f"avg acc = {running_acc/n_batches:.3f}", flush=True)
    return model, K, history, n_hashes, bucket_size


# ----- Generation ---------------------------------------------------------

def generate(model, K: int, history: int, n_hashes: int, bucket_size: int,
             n_records: int, seed: int, stale_pool_size: int = 100_000,
             temperature: float = 1.0) -> np.ndarray:
    """R299b generation with split FRESH / RECYCLE classes.

    When model predicts FRESH (class K+1): allocate brand-new obj_id.
    When model predicts RECYCLE (class K): sample uniformly from stale pool
      (recently-evicted obj_ids — items that fell off the bottom of the K-stack).
    Else REUSE@k: take stack[k].
    """
    import torch
    from collections import deque
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()

    stack: list = []
    next_new = 0
    history_buf: list = [0] * history
    stale_pool: deque = deque(maxlen=stale_pool_size)  # FIFO of evicted obj_ids
    out = np.empty(n_records, dtype=np.uint64)

    primes = np.array([
        2654435761, 40503, 2246822519, 3266489917, 374761393, 1376312589,
        2073600871, 2150448427, 2807114953, 3132291347, 691013249,
        1779033703, 3144134277, 1013904242, 2773480762, 1359893119,
    ][:n_hashes], dtype=np.uint64)

    def hash_one(oid: int) -> np.ndarray:
        x = np.uint64(oid)
        return ((x * primes) % np.uint64(bucket_size)).astype(np.int64)

    with torch.no_grad():
        for j in range(n_records):
            hist_h = np.stack([hash_one(o) for o in history_buf])
            sk = np.zeros((K, n_hashes), dtype=np.int64)
            for k in range(min(len(stack), K)):
                sk[k] = hash_one(stack[k])
            hist_t = torch.from_numpy(hist_h).long().unsqueeze(0).to(device)
            stack_t = torch.from_numpy(sk).long().unsqueeze(0).to(device)
            logits = model(hist_t, stack_t)[0].cpu().numpy()  # [K+2]

            # Mask in-stack positions that are empty.
            if len(stack) < K:
                logits[len(stack):K] = -1e9
            # Mask RECYCLE if stale pool is empty (rare warm-up case).
            if not stale_pool:
                logits[K] = -1e9

            if temperature <= 0:
                # Greedy / argmax decoding.
                action = int(np.argmax(logits))
            else:
                scaled = logits / max(temperature, 1e-6)
                scaled -= scaled.max()
                p = np.exp(scaled)
                p /= p.sum()
                action = int(rng.choice(K + 2, p=p))

            if action == K + 1 or (action == K and not stale_pool):
                # FRESH
                addr = next_new
                next_new += 1
            elif action == K:
                # RECYCLE from stale pool (sample uniformly).
                addr = int(stale_pool[rng.integers(0, len(stale_pool))])
            else:
                # REUSE @ position k
                addr = stack.pop(action)

            # Update stack: push to top; if stack overflows K, evict bottom to stale pool.
            stack.insert(0, addr)
            if len(stack) > K:
                evicted = stack.pop()
                stale_pool.append(evicted)
            out[j] = addr
            history_buf.pop(0)
            history_buf.append(int(addr))

    print(f"[trace_transformer generate] emitted {n_records:,} records, "
          f"unique={next_new:,}, max_stack={len(stack):,}, "
          f"stale_pool={len(stale_pool):,}", flush=True)
    return out


def write_csv(path: str, obj_ids: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


# ----- CLI -----------------------------------------------------------------

def cmd_fit(args):
    import torch
    model, K, history, n_hashes, bucket_size = train_transformer(
        args.real, K=args.K, history=args.history, n_hashes=args.n_hashes,
        bucket_size=args.bucket_size, emb_dim=args.emb_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, batch=args.batch, epochs=args.epochs,
        lr=args.lr, seed=args.seed, max_rows=args.max_rows,
        class_balance=args.class_balance,
    )
    state = {
        "K": K, "history": history, "n_hashes": n_hashes,
        "bucket_size": bucket_size, "emb_dim": args.emb_dim,
        "n_heads": args.n_heads, "n_layers": args.n_layers,
        "model_state": model.state_dict(),
    }
    torch.save(state, args.output)
    print(f"[trace_transformer fit] saved → {args.output}", flush=True)


def cmd_generate(args):
    import torch
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    K = state["K"]
    history = state["history"]
    model = _build_model(K, state["n_hashes"], state["bucket_size"],
                          state["emb_dim"], state["n_heads"], state["n_layers"],
                          history)
    model.load_state_dict(state["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out = generate(model, K, history, state["n_hashes"], state["bucket_size"],
                   args.n, args.seed, temperature=args.temperature)
    write_csv(args.output, out)
    print(f"[trace_transformer generate] wrote {args.n:,} records → {args.output}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description="R299 cache-state-aware TraceTransformer")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--K", type=int, default=128)
    pf.add_argument("--history", type=int, default=64)
    pf.add_argument("--n-hashes", type=int, default=16)
    pf.add_argument("--bucket-size", type=int, default=4096)
    pf.add_argument("--emb-dim", type=int, default=96)
    pf.add_argument("--n-heads", type=int, default=4)
    pf.add_argument("--n-layers", type=int, default=3)
    pf.add_argument("--batch", type=int, default=128)
    pf.add_argument("--epochs", type=int, default=10)
    pf.add_argument("--lr", type=float, default=1e-3)
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.add_argument("--class-balance", action="store_true",
                    help="Use inverse-sqrt-frequency class weights in the cross-entropy "
                         "loss to counter FRESH-class dominance.")
    pf.set_defaults(fn=cmd_fit)
    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--temperature", type=float, default=1.0,
                    help="Softmax temperature for sampling. 1.0 = raw model dist; "
                         "<1.0 sharpens toward argmax; 0 = greedy. Lower T may "
                         "fix under-emit of dominant action class when softmax mass "
                         "is spread thin across K+2 outcomes.")
    pg.set_defaults(fn=cmd_generate)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
