"""LLNL b2: conditional transition net over a compact stack-distance state space.

Round 170. Closes the per-file conditioning gap exposed in R163-R168: PhaseAtlas
samples ranks i.i.d. from a global PMF, so the architectural ceiling on alibaba
HRC-MAE is 0.019 (R168 manifest-oracle). LANL's NeuralAtlas (altgan) closes this
to 0.0018 by learning P(next_state | cond, prev_state). This file ports that idea
in compact form: 6-state space (1 NEW + 5 stack-distance bucket REUSE classes)
conditioned on per-file characterization features.

Pipeline:
  fit:    read traces → bin events into states → train CondTransitionNet
  generate: per stream, compute cond vector from manifest-file char features,
            roll out state sequence, decode to (action, rank), use existing
            PhaseAtlas rank PMF + obj_id stack.

CLI:
  python -m llgan.neural_atlas fit --trace-dir DIR --char-file CHAR --output PKL
  python -m llgan.neural_atlas generate --model PKL --manifest M.json --output CSV
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# State space: NEW + 5 stack-distance bucket REUSE classes.
STATE_BUCKET_EDGES = np.array([0, 8, 32, 128, 512, 1 << 30], dtype=np.int64)
N_STATES = len(STATE_BUCKET_EDGES)  # 1 NEW + 5 buckets = 6
STATE_NEW = 0
STATE_REUSE_OFFSET = 1


def state_from_sd(sd: int) -> int:
    """Encode stack distance to state. sd<0 = NEW; otherwise bucket index + offset."""
    if sd < 0:
        return STATE_NEW
    bucket = int(np.searchsorted(STATE_BUCKET_EDGES[1:], sd, side="right"))
    return STATE_REUSE_OFFSET + min(bucket, len(STATE_BUCKET_EDGES) - 2)


# Conditioning features pulled from trace_characterizations.jsonl. Skips
# `reuse_ratio` because its 4096-sample value is unreliable for alibaba (R169).
COND_FEATURES = [
    "burstiness_cv",
    "iat_q50",
    "iat_q90",
    "obj_size_q50",
    "obj_size_q90",
    "write_ratio",
    "opcode_switch_ratio",
    "forward_seek_ratio",
    "backward_seek_ratio",
    "ts_duration",
]
COND_DIM = len(COND_FEATURES)


def _safe_log1p(x):
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.log1p(max(float(x), 0.0)))


def cond_from_profile(profile: dict) -> np.ndarray:
    """Build a fixed-shape cond vector from a trace_characterization profile."""
    out = np.zeros(COND_DIM, dtype=np.float32)
    for i, k in enumerate(COND_FEATURES):
        v = profile.get(k)
        if v is None:
            out[i] = 0.0
        elif k.endswith("_cv") or k.endswith("_ratio"):
            out[i] = float(v) if np.isfinite(float(v)) else 0.0
        elif k.startswith("iat_") or k.startswith("obj_size_") or k == "ts_duration":
            out[i] = _safe_log1p(v)
        else:
            out[i] = float(v) if np.isfinite(float(v)) else 0.0
    return out


# Lazy torch import — keep module importable without torch when only generation
# from a precomputed transition table is needed.
def _torch():
    import torch
    return torch


def make_net(cond_dim: int, hidden: int, n_states: int):
    torch = _torch()
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.state_emb = nn.Embedding(n_states, hidden)
            self.init_head = nn.Linear(hidden, n_states)
            self.trans_head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.SiLU(),
                nn.Linear(hidden, n_states),
            )

        def forward_init(self, cond):
            return self.init_head(self.cond_mlp(cond))

        def forward_trans(self, cond, prev_state):
            h = self.cond_mlp(cond)
            s = self.state_emb(prev_state)
            return self.trans_head(torch.cat([h, s], dim=-1))

    return Net()


@dataclass
class NeuralAtlas:
    n_states: int
    cond_dim: int
    hidden: int
    state_dict: dict          # torch state_dict (CPU tensors)
    file_index: Dict[str, np.ndarray]  # filename → cond vector (cached)
    rank_pmf_per_state: Dict[int, np.ndarray]  # state → fine-bin rank PMF (29 bins)
    rank_edges: np.ndarray
    opcode_pmf: Optional[Dict[int, float]] = None
    metadata: dict = field(default_factory=dict)
    # Round 171: cond normalization stats (mean, std per feature). Applied at
    # both fit and generate time so the net sees zero-mean unit-variance cond.
    cond_mean: Optional[np.ndarray] = None
    cond_std: Optional[np.ndarray] = None

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "NeuralAtlas":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def _read_chars(char_file: str) -> Dict[str, dict]:
    """Read trace_characterizations.jsonl into {basename: profile}."""
    out: Dict[str, dict] = {}
    with open(char_file) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            p = d.get("path", "")
            nm = os.path.basename(p)
            prof = d.get("profile", {})
            if nm and prof:
                out[nm] = prof
    return out


def _stack_distance(obj_ids: np.ndarray) -> np.ndarray:
    """Reuse BIT-based exact LRU stack distance from phase_pmf_atlas (O(N log N))."""
    sys.path.insert(0, "/home/darrell/Zarathustra")
    from llgan.phase_pmf_atlas import _stack_distance as _sd
    return _sd(obj_ids)


def _read_oracle_general_obj_ids(path: str, max_records: int) -> np.ndarray:
    """Read obj_ids from oracleGeneral .zst (skip sentinels)."""
    sys.path.insert(0, "/home/darrell/Zarathustra")
    from llgan.phase_pmf_atlas import _read_trace
    out = []
    for ev in _read_trace(path):
        out.append(ev[1])
        if len(out) >= max_records:
            break
    return np.array(out, dtype=np.int64)


def fit(
    trace_dir: str,
    char_file: str,
    output: str,
    *,
    max_files: int = 64,
    records_per_file: int = 25_000,
    hidden: int = 64,
    epochs: int = 200,
    lr: float = 2e-3,
    seed: int = 7,
) -> None:
    """Train a CondTransitionNet on (cond, prev_state, next_state) tuples."""
    torch = _torch()
    rng = np.random.default_rng(seed)

    print(f"Reading characterizations from {char_file} ...")
    chars = _read_chars(char_file)
    print(f"  loaded {len(chars):,} entries")

    import glob
    files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
    if max_files > 0 and len(files) > max_files:
        idx = rng.choice(len(files), max_files, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"Selected {len(files)} training files")

    # Per-file: read records, compute stack distances, encode states, build cond vector.
    file_conds: Dict[str, np.ndarray] = {}
    transitions: List[Tuple[np.ndarray, int, int]] = []  # (cond, prev_state, next_state)
    initial_states: List[Tuple[np.ndarray, int]] = []     # (cond, init_state)
    rank_observations: Dict[int, List[int]] = {s: [] for s in range(N_STATES)}

    for fi, fpath in enumerate(files):
        nm = os.path.basename(fpath)
        if nm not in chars:
            continue
        cond = cond_from_profile(chars[nm])
        file_conds[nm] = cond

        obj_ids = _read_oracle_general_obj_ids(fpath, records_per_file)
        if len(obj_ids) < 2:
            continue
        sd = _stack_distance(obj_ids)
        states = np.array([state_from_sd(int(x)) for x in sd], dtype=np.int64)

        initial_states.append((cond, int(states[0])))
        for t in range(len(states) - 1):
            transitions.append((cond, int(states[t]), int(states[t + 1])))

        # Per-state rank observations (for decoding REUSE state → rank)
        for i, s in enumerate(states):
            if int(s) >= STATE_REUSE_OFFSET and sd[i] >= 0:
                rank_observations[int(s)].append(int(sd[i]))

        if fi % 8 == 0:
            print(f"  pass {fi+1}/{len(files)}: {len(transitions):,} transitions accumulated")

    print(f"Total: {len(transitions):,} transitions, {len(initial_states):,} initial states")
    print(f"Per-state rank counts: {[len(v) for v in rank_observations.values()]}")

    # Round 171: normalize cond features (zero-mean, unit-variance per feature)
    # so each dim contributes equally and the MLP doesn't have to learn rescaling.
    cond_arr = np.stack(list(file_conds.values())).astype(np.float64)
    cond_mean = cond_arr.mean(axis=0).astype(np.float32)
    cond_std = cond_arr.std(axis=0).astype(np.float32)
    cond_std[cond_std < 1e-6] = 1.0
    print(f"Cond mean: {cond_mean.round(3).tolist()}")
    print(f"Cond std:  {cond_std.round(3).tolist()}")
    # Normalize all stored conds in-place
    for nm in list(file_conds.keys()):
        file_conds[nm] = ((file_conds[nm] - cond_mean) / cond_std).astype(np.float32)
    transitions = [(((c - cond_mean) / cond_std).astype(np.float32), p, n) for c, p, n in transitions]
    initial_states = [(((c - cond_mean) / cond_std).astype(np.float32), s) for c, s in initial_states]

    # Build per-state fine-bin rank PMF (29 bins, matches existing PhaseAtlas eval edges)
    fine_edges = np.array([
        1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 32, 42, 56, 74, 97, 127, 167, 219, 288,
        377, 495, 649, 851, 1116, 1463, 1919, 2516, 3299, 4323, 5669,
    ], dtype=np.int64)
    rank_pmf_per_state: Dict[int, np.ndarray] = {}
    for s, ranks in rank_observations.items():
        if not ranks:
            rank_pmf_per_state[s] = np.ones(len(fine_edges) - 1) / (len(fine_edges) - 1)
            continue
        ranks_arr = np.array(ranks, dtype=np.int64)
        counts, _ = np.histogram(ranks_arr, bins=fine_edges)
        if counts.sum() == 0:
            rank_pmf_per_state[s] = np.ones(len(fine_edges) - 1) / (len(fine_edges) - 1)
        else:
            rank_pmf_per_state[s] = counts.astype(np.float64) / counts.sum()

    # Train net
    print(f"Training CondTransitionNet (hidden={hidden}, epochs={epochs}, lr={lr})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_net(COND_DIM, hidden, N_STATES).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    init_conds = torch.tensor(np.stack([c for c, _ in initial_states]), dtype=torch.float32, device=device)
    init_targets = torch.tensor([s for _, s in initial_states], dtype=torch.long, device=device)
    trans_conds = torch.tensor(np.stack([c for c, _, _ in transitions]), dtype=torch.float32, device=device)
    trans_prev = torch.tensor([p for _, p, _ in transitions], dtype=torch.long, device=device)
    trans_next = torch.tensor([n for _, _, n in transitions], dtype=torch.long, device=device)

    import torch.nn.functional as F
    for ep in range(epochs):
        opt.zero_grad()
        init_logits = net.forward_init(init_conds)
        init_loss = F.cross_entropy(init_logits, init_targets)
        trans_logits = net.forward_trans(trans_conds, trans_prev)
        trans_loss = F.cross_entropy(trans_logits, trans_next)
        loss = init_loss + trans_loss
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:>4d}  init_loss={init_loss.item():.4f}  trans_loss={trans_loss.item():.4f}")

    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    model = NeuralAtlas(
        n_states=N_STATES,
        cond_dim=COND_DIM,
        hidden=hidden,
        state_dict=state_dict,
        file_index=file_conds,
        rank_pmf_per_state=rank_pmf_per_state,
        rank_edges=fine_edges,
        cond_mean=cond_mean,
        cond_std=cond_std,
        metadata={
            "trace_dir": trace_dir,
            "char_file": char_file,
            "max_files": max_files,
            "records_per_file": records_per_file,
            "epochs": epochs,
            "lr": lr,
            "n_transitions": len(transitions),
        },
    )
    model.save(output)
    print(f"Saved → {output}")


def generate(
    model_path: str,
    manifest_path: str,
    output_path: str,
    *,
    n_records: int = 100_000,
    seed: int = 42,
    max_stack_depth: int = 8192,
) -> None:
    """Roll out per-stream state sequences via the trained net + decode."""
    torch = _torch()
    import pandas as pd

    print(f"Loading model from {model_path} ...")
    m = NeuralAtlas.load(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_net(m.cond_dim, m.hidden, m.n_states).to(device)
    net.load_state_dict(m.state_dict)
    net.eval()

    with open(manifest_path) as f:
        manifest = json.load(f)
    streams = manifest["streams"]
    n_streams = len(streams)
    per_stream = int(np.ceil(n_records / n_streams))

    # Read characterizations to look up cond per manifest file
    chars = _read_chars(m.metadata.get("char_file", "/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl"))

    rng = np.random.default_rng(seed)
    rows = []

    with torch.no_grad():
        for sid, segs in enumerate(streams):
            nm = os.path.basename(segs[0]["path"])
            cond = chars.get(nm, {})
            cond_vec = cond_from_profile(cond)
            # Round 171: apply training-time normalization at generate
            if m.cond_mean is not None and m.cond_std is not None:
                cond_vec = (cond_vec - m.cond_mean) / m.cond_std
            cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # Initial state
            init_logits = net.forward_init(cond_t)
            init_p = torch.softmax(init_logits, dim=-1).cpu().numpy()[0]
            state = int(rng.choice(m.n_states, p=init_p))

            stack: List[int] = []
            next_new_id = 10_000_000 + sid * (per_stream + 1_000_003)
            ts = 0.0

            for step in range(per_stream):
                # Decode current state to (action, rank)
                if state == STATE_NEW:
                    obj_id = next_new_id
                    next_new_id += 1
                    stack.insert(0, obj_id)
                else:
                    pmf = m.rank_pmf_per_state.get(state, np.ones(len(m.rank_edges) - 1) / (len(m.rank_edges) - 1))
                    if pmf.sum() <= 0:
                        pmf = np.ones_like(pmf) / len(pmf)
                    pmf = pmf / pmf.sum()
                    fine_i = int(rng.choice(len(pmf), p=pmf))
                    lo = int(m.rank_edges[fine_i])
                    hi = int(m.rank_edges[fine_i + 1]) - 1 if fine_i + 1 < len(m.rank_edges) else lo
                    stack_sz = len(stack)
                    if stack_sz == 0:
                        # fallback to NEW
                        obj_id = next_new_id
                        next_new_id += 1
                        stack.insert(0, obj_id)
                    else:
                        lo_eff = min(lo, stack_sz - 1)
                        hi_eff = min(hi, stack_sz - 1)
                        rank = int(rng.integers(lo_eff, hi_eff + 1))
                        obj_id = stack[rank]
                        del stack[rank]
                        stack.insert(0, obj_id)
                if len(stack) > max_stack_depth:
                    stack.pop()

                # Sample opcode + size + ts (placeholder; trivial since cache eval ignores)
                op_sampled = 0
                if m.opcode_pmf:
                    keys = sorted(m.opcode_pmf.keys())
                    probs = [m.opcode_pmf[k] for k in keys]
                    op_sampled = int(np.random.choice(keys, p=probs))

                ts += 1.0
                rows.append({
                    "stream_id": sid,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": 4096,
                    "opcode": op_sampled,
                    "tenant": 0,
                })

                # Roll the state forward
                state_t = torch.tensor([state], dtype=torch.long, device=device)
                trans_logits = net.forward_trans(cond_t, state_t)
                trans_p = torch.softmax(trans_logits, dim=-1).cpu().numpy()[0]
                state = int(rng.choice(m.n_states, p=trans_p))

    df = pd.DataFrame(rows[:n_records])
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records ({n_streams} streams) → {output_path}")


def main():
    p = argparse.ArgumentParser(description="LLNL b2 — conditional transition net over compact stack-distance state space")
    sub = p.add_subparsers(dest="cmd", required=True)

    pfit = sub.add_parser("fit")
    pfit.add_argument("--trace-dir", required=True)
    pfit.add_argument("--char-file", required=True)
    pfit.add_argument("--output", required=True)
    pfit.add_argument("--max-files", type=int, default=64)
    pfit.add_argument("--records-per-file", type=int, default=25_000)
    pfit.add_argument("--hidden", type=int, default=64)
    pfit.add_argument("--epochs", type=int, default=200)
    pfit.add_argument("--lr", type=float, default=2e-3)
    pfit.add_argument("--seed", type=int, default=7)

    pgen = sub.add_parser("generate")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--manifest", required=True)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "fit":
        fit(
            args.trace_dir, args.char_file, args.output,
            max_files=args.max_files, records_per_file=args.records_per_file,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed,
        )
    elif args.cmd == "generate":
        generate(
            args.model, args.manifest, args.output,
            n_records=args.n, seed=args.seed,
        )


if __name__ == "__main__":
    main()
