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
N_DIST_STATES = len(STATE_BUCKET_EDGES)  # 1 NEW + 5 REUSE buckets = 6
STATE_NEW = 0
STATE_REUSE_OFFSET = 1
PHASE_WINDOW = 200  # records per running-unique-rate window


def n_states_for(n_phase_bins: int) -> int:
    return N_DIST_STATES * max(int(n_phase_bins), 1)


def _dist_state_from_sd(sd: int) -> int:
    """Encode stack distance to {NEW, REUSE_b0, ..., REUSE_b4} = 6 dist states."""
    if sd < 0:
        return STATE_NEW
    bucket = int(np.searchsorted(STATE_BUCKET_EDGES[1:], sd, side="right"))
    return STATE_REUSE_OFFSET + min(bucket, len(STATE_BUCKET_EDGES) - 2)


def state_from_sd(sd: int, phase_bin: int = 0) -> int:
    """Encode (sd, phase_bin) to state index = phase_bin*N_DIST_STATES + dist_state."""
    return int(phase_bin) * N_DIST_STATES + _dist_state_from_sd(sd)


# Backwards-compat: legacy 6-state tencent / alibaba models pickled in R170-173.
N_STATES = N_DIST_STATES


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


def cond_from_trace(path: str, max_records: int = 25_000) -> np.ndarray:
    """Round 172: compute the 10-feature cond vector directly from the trace
    file. Closes the data-quality gap from R171 where 6 of 10 features were
    zero-variance in trace_characterizations.jsonl (iat_q50/90, obj_size_q50/90,
    write_ratio, opcode_switch_ratio, ts_duration not populated).
    """
    sys.path.insert(0, "/home/darrell/Zarathustra")
    from llgan.phase_pmf_atlas import _read_trace
    ts_arr, sz_arr, op_arr = [], [], []
    for ev in _read_trace(path, keep_sentinel=True):
        ts_arr.append(ev[0])
        sz_arr.append(ev[2])
        op_arr.append(ev[3])
        if len(ts_arr) >= max_records:
            break
    if len(ts_arr) < 2:
        return np.zeros(COND_DIM, dtype=np.float32)
    ts = np.asarray(ts_arr, dtype=np.float64)
    sz = np.asarray(sz_arr, dtype=np.float64)
    op = np.asarray(op_arr, dtype=np.int64)

    iat = np.diff(ts)
    iat = iat[iat >= 0]
    if len(iat) < 1:
        iat = np.array([0.0])
    iat_q50 = float(np.quantile(iat, 0.5))
    iat_q90 = float(np.quantile(iat, 0.9))
    if iat.std() > 1e-9 and iat.mean() > 1e-9:
        burst_cv = float(iat.std() / iat.mean())
    else:
        burst_cv = 0.0

    sz_q50 = float(np.quantile(sz, 0.5))
    sz_q90 = float(np.quantile(sz, 0.9))

    # write_ratio: fraction with op==1 (write); read is 0, sentinel is -1
    write_ratio = float(np.mean(op == 1)) if len(op) else 0.0

    # opcode_switch_ratio: fraction of consecutive opcode-changes
    if len(op) > 1:
        opcode_switch_ratio = float(np.mean(op[1:] != op[:-1]))
    else:
        opcode_switch_ratio = 0.0

    # forward / backward seek ratio: from obj_id deltas (need obj_id which we
    # don't track here — rough proxy via size delta sign)
    # Use a minimal stand-in: never zero. forward = sz[i+1] >= sz[i]
    if len(sz) > 1:
        diffs = np.sign(np.diff(sz))
        forward_seek_ratio = float(np.mean(diffs >= 0))
        backward_seek_ratio = float(np.mean(diffs < 0))
    else:
        forward_seek_ratio = 0.5
        backward_seek_ratio = 0.5

    ts_duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0

    profile = {
        "burstiness_cv": burst_cv,
        "iat_q50": iat_q50,
        "iat_q90": iat_q90,
        "obj_size_q50": sz_q50,
        "obj_size_q90": sz_q90,
        "write_ratio": write_ratio,
        "opcode_switch_ratio": opcode_switch_ratio,
        "forward_seek_ratio": forward_seek_ratio,
        "backward_seek_ratio": backward_seek_ratio,
        "ts_duration": ts_duration,
    }
    return cond_from_profile(profile)


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
    # Round 174: state-space expansion. n_phase_bins=1 reproduces R170-173's
    # 6-state encoding; n_phase_bins=4 expands to 24 states with per-window
    # unique-rate phase. phase_edges are corpus quantiles fitted at training.
    n_phase_bins: int = 1
    phase_edges: Optional[np.ndarray] = None

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
    inline_cond: bool = False,
    n_phase_bins: int = 1,
) -> None:
    """Train a CondTransitionNet on (cond, prev_state, next_state) tuples.

    When `inline_cond=True` (R172), compute cond features directly from the
    trace file at fit time. When `n_phase_bins>1` (R174), expand state space
    to phase × dist bins (24 with n_phase_bins=4).
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    n_phase_bins = max(int(n_phase_bins), 1)
    n_states = n_states_for(n_phase_bins)
    print(f"State space: n_phase_bins={n_phase_bins} × {N_DIST_STATES} dist = {n_states} states")

    if inline_cond:
        print("Inline cond mode: features computed from trace files directly")
        chars = {}
    else:
        print(f"Reading characterizations from {char_file} ...")
        chars = _read_chars(char_file)
        print(f"  loaded {len(chars):,} entries")

    import glob
    files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
    if max_files > 0 and len(files) > max_files:
        idx = rng.choice(len(files), max_files, replace=False)
        files = [files[i] for i in sorted(idx)]
    print(f"Selected {len(files)} training files")

    # Pass 1: collect unique-rate samples to fit phase quantile edges
    phase_edges = np.array([], dtype=np.float64)
    if n_phase_bins > 1:
        ur_samples: List[float] = []
        for fpath in files:
            obj_ids = _read_oracle_general_obj_ids(fpath, records_per_file)
            if len(obj_ids) < PHASE_WINDOW * 2:
                continue
            seen: set = set()
            for i, x in enumerate(obj_ids):
                seen.add(int(x))
                if (i + 1) % PHASE_WINDOW == 0:
                    ur_samples.append(len(seen) / PHASE_WINDOW)
                    seen = set()
        if ur_samples:
            ur = np.array(ur_samples, dtype=np.float64)
            q = np.linspace(0, 100, n_phase_bins + 1)[1:-1]
            phase_edges = np.unique(np.percentile(ur, q))
            print(f"Phase edges (n={len(ur)} windows): {phase_edges.round(3).tolist()}")

    # Per-file: read records, compute stack distances, encode states, build cond.
    file_conds: Dict[str, np.ndarray] = {}
    transitions: List[Tuple[np.ndarray, int, int]] = []
    initial_states: List[Tuple[np.ndarray, int]] = []
    rank_observations: Dict[int, List[int]] = {s: [] for s in range(n_states)}

    for fi, fpath in enumerate(files):
        nm = os.path.basename(fpath)
        if inline_cond:
            cond = cond_from_trace(fpath, max_records=records_per_file)
        else:
            if nm not in chars:
                continue
            cond = cond_from_profile(chars[nm])
        file_conds[nm] = cond

        obj_ids = _read_oracle_general_obj_ids(fpath, records_per_file)
        if len(obj_ids) < 2:
            continue
        sd = _stack_distance(obj_ids)

        # Compute per-event phase bin (running unique-rate window)
        if n_phase_bins > 1:
            phase_per_event = np.zeros(len(obj_ids), dtype=np.int64)
            current_rate = 0.0
            seen: set = set()
            for i in range(len(obj_ids)):
                pb = int(np.searchsorted(phase_edges, current_rate, side="right"))
                phase_per_event[i] = min(pb, n_phase_bins - 1)
                seen.add(int(obj_ids[i]))
                if (i + 1) % PHASE_WINDOW == 0:
                    current_rate = len(seen) / PHASE_WINDOW
                    seen = set()
        else:
            phase_per_event = np.zeros(len(obj_ids), dtype=np.int64)

        states = np.array([
            state_from_sd(int(sd[i]), int(phase_per_event[i]))
            for i in range(len(sd))
        ], dtype=np.int64)

        initial_states.append((cond, int(states[0])))
        for t in range(len(states) - 1):
            transitions.append((cond, int(states[t]), int(states[t + 1])))

        # Per-state rank observations
        for i, s in enumerate(states):
            dist_state = int(s) % N_DIST_STATES
            if dist_state >= STATE_REUSE_OFFSET and sd[i] >= 0:
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
    n_uniform = len(fine_edges) - 1
    for s in range(n_states):
        ranks = rank_observations.get(s, [])
        if not ranks:
            rank_pmf_per_state[s] = np.ones(n_uniform) / n_uniform
            continue
        ranks_arr = np.array(ranks, dtype=np.int64)
        counts, _ = np.histogram(ranks_arr, bins=fine_edges)
        if counts.sum() == 0:
            rank_pmf_per_state[s] = np.ones(n_uniform) / n_uniform
        else:
            rank_pmf_per_state[s] = counts.astype(np.float64) / counts.sum()

    # Train net
    print(f"Training CondTransitionNet (hidden={hidden}, epochs={epochs}, lr={lr})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = make_net(COND_DIM, hidden, n_states).to(device)
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
        n_states=n_states,
        cond_dim=COND_DIM,
        hidden=hidden,
        state_dict=state_dict,
        file_index=file_conds,
        rank_pmf_per_state=rank_pmf_per_state,
        rank_edges=fine_edges,
        cond_mean=cond_mean,
        cond_std=cond_std,
        n_phase_bins=n_phase_bins,
        phase_edges=phase_edges if len(phase_edges) > 0 else None,
        metadata={
            "trace_dir": trace_dir,
            "char_file": char_file,
            "max_files": max_files,
            "records_per_file": records_per_file,
            "epochs": epochs,
            "lr": lr,
            "n_transitions": len(transitions),
            "inline_cond": inline_cond,
            "n_phase_bins": n_phase_bins,
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
    temperature: float = 1.0,
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

    # Read characterizations OR compute cond inline depending on training mode
    use_inline_cond = bool(m.metadata.get("inline_cond", False))
    if use_inline_cond:
        chars = {}
        print("Inline cond mode: computing per-stream cond from manifest files")
    else:
        chars = _read_chars(m.metadata.get("char_file", "/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl"))

    rng = np.random.default_rng(seed)
    rows = []

    with torch.no_grad():
        for sid, segs in enumerate(streams):
            nm = os.path.basename(segs[0]["path"])
            if use_inline_cond:
                cond_vec = cond_from_trace(segs[0]["path"],
                                           max_records=segs[0].get("records_taken", 25_000))
            else:
                cond = chars.get(nm, {})
                cond_vec = cond_from_profile(cond)
            # Round 171: apply training-time normalization at generate
            if m.cond_mean is not None and m.cond_std is not None:
                cond_vec = (cond_vec - m.cond_mean) / m.cond_std
            cond_t = torch.tensor(cond_vec, dtype=torch.float32, device=device).unsqueeze(0)

            # Initial state (temperature-scaled softmax)
            init_logits = net.forward_init(cond_t)
            init_p = torch.softmax(init_logits / max(temperature, 1e-3), dim=-1).cpu().numpy()[0]
            state = int(rng.choice(m.n_states, p=init_p))

            stack: List[int] = []
            next_new_id = 10_000_000 + sid * (per_stream + 1_000_003)
            ts = 0.0
            # R174: running unique-rate phase tracking (mirrors fit-time phase)
            n_phase_bins = max(int(getattr(m, "n_phase_bins", 1)), 1)
            phase_edges = getattr(m, "phase_edges", None)
            current_rate = 0.0
            phase_seen: set = set()
            phase_count = 0

            for step in range(per_stream):
                # Decode current state to (phase, dist_state) → action
                dist_state = state % N_DIST_STATES
                if dist_state == STATE_NEW:
                    obj_id = next_new_id
                    next_new_id += 1
                    stack.insert(0, obj_id)
                else:
                    # R174: use phase-marginalized rank PMF (sum over phase bins
                    # of states with same dist_state). Decouples phase-conditioned
                    # transitions from rank decoding so the dist_state's true
                    # rank distribution drives the rank pick instead of a
                    # phase-restricted slice that may be heavily skewed.
                    if n_phase_bins > 1:
                        pmf_acc = np.zeros(len(m.rank_edges) - 1, dtype=np.float64)
                        for pb in range(n_phase_bins):
                            s_pb = pb * N_DIST_STATES + dist_state
                            p = m.rank_pmf_per_state.get(s_pb, None)
                            if p is not None and p.sum() > 0:
                                pmf_acc += p
                        if pmf_acc.sum() > 0:
                            pmf = pmf_acc / pmf_acc.sum()
                        else:
                            pmf = np.ones(len(m.rank_edges) - 1) / (len(m.rank_edges) - 1)
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

                # Update running unique-rate phase
                if n_phase_bins > 1 and phase_edges is not None:
                    phase_seen.add(int(obj_id))
                    phase_count += 1
                    if phase_count >= PHASE_WINDOW:
                        current_rate = len(phase_seen) / PHASE_WINDOW
                        phase_seen = set()
                        phase_count = 0

                # Roll the state forward (sampled state already encodes phase)
                state_t = torch.tensor([state], dtype=torch.long, device=device)
                trans_logits = net.forward_trans(cond_t, state_t)
                trans_p = torch.softmax(trans_logits / max(temperature, 1e-3), dim=-1).cpu().numpy()[0]
                state = int(rng.choice(m.n_states, p=trans_p))
                # Force the phase component to track the running unique-rate so
                # the sampled state's phase doesn't drift from the empirical phase.
                if n_phase_bins > 1 and phase_edges is not None:
                    expected_pb = int(np.searchsorted(phase_edges, current_rate, side="right"))
                    expected_pb = min(expected_pb, n_phase_bins - 1)
                    state = expected_pb * N_DIST_STATES + (state % N_DIST_STATES)

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
    pfit.add_argument("--inline-cond", action="store_true",
                      help="R172: compute cond features directly from trace files")
    pfit.add_argument("--n-phase-bins", type=int, default=1,
                      help="R174: state-space expansion. 1 (default) = 6-state R170-173 encoding; "
                           "4 = 24 states with running-unique-rate phase quartiles.")

    pgen = sub.add_parser("generate")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--manifest", required=True)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--seed", type=int, default=42)
    pgen.add_argument("--temperature", type=float, default=1.0,
                      help="Softmax temperature for state-transition sampling. "
                           "Lower = sharper / closer to argmax. R175 experiment.")

    args = p.parse_args()
    if args.cmd == "fit":
        fit(
            args.trace_dir, args.char_file, args.output,
            max_files=args.max_files, records_per_file=args.records_per_file,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed,
            inline_cond=args.inline_cond, n_phase_bins=args.n_phase_bins,
        )
    elif args.cmd == "generate":
        generate(
            args.model, args.manifest, args.output,
            n_records=args.n, seed=args.seed, temperature=args.temperature,
        )


if __name__ == "__main__":
    main()
