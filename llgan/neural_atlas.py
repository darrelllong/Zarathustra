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

# R180: 29 fine bins (matches existing PhaseAtlas eval edges) for AR-rank
# binning. R220 (IRD-diagnostic-driven): extend the upper tail from 5669 to
# 251236 (14 added bins), because the IRD diagnostic showed real alibaba has
# 15% of IRDs > 31k and the fit silently drops ranks >= 5669 from the PMF.
# This is the actual architectural ceiling that R211/R213/R217 phase-bin
# expansion couldn't address. Total bins now 43.
FINE_EDGES_R180 = np.array([
    1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 32, 42, 56, 74, 97, 127, 167, 219, 288,
    377, 495, 649, 851, 1116, 1463, 1919, 2516, 3299, 4323, 5669,
    7437, 9750, 12780, 16752, 21959, 28785, 37736, 49467, 64853, 85036,
    111472, 146148, 191628, 251236,
], dtype=np.int64)
N_RANK_BINS = len(FINE_EDGES_R180) - 1
STATE_NEW = 0
STATE_REUSE_OFFSET = 1
PHASE_WINDOW = 200  # records per running-unique-rate window


def n_states_for(n_phase_bins: int, n_time_bins: int = 1, n_size_bins: int = 1) -> int:
    """R270: state = phase × n_time × n_size × N_DIST_STATES.
    Backwards-compat: n_time=n_size=1 reproduces pre-R270 encoding."""
    return (max(int(n_phase_bins), 1)
            * max(int(n_time_bins), 1)
            * max(int(n_size_bins), 1)
            * N_DIST_STATES)


def _dist_state_from_sd(sd: int) -> int:
    """Encode stack distance to {NEW, REUSE_b0, ..., REUSE_b4} = 6 dist states."""
    if sd < 0:
        return STATE_NEW
    bucket = int(np.searchsorted(STATE_BUCKET_EDGES[1:], sd, side="right"))
    return STATE_REUSE_OFFSET + min(bucket, len(STATE_BUCKET_EDGES) - 2)


def state_from_sd(sd: int, phase_bin: int = 0,
                  time_bin: int = 0, size_bin: int = 0,
                  n_time_bins: int = 1, n_size_bins: int = 1) -> int:
    """R270: encode (sd, phase, time, size) to state index.
       state = ((phase * n_time + time) * n_size + size) * N_DIST_STATES + dist
    Backwards-compat: time_bin=0, size_bin=0, n_time=1, n_size=1 reproduces
    pre-R270 encoding (phase_bin*N_DIST_STATES + dist_state)."""
    nt = max(int(n_time_bins), 1)
    nz = max(int(n_size_bins), 1)
    return (((int(phase_bin) * nt + int(time_bin)) * nz + int(size_bin))
            * N_DIST_STATES + _dist_state_from_sd(sd))


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """R270: corpus quantile edges for time/size binning. Mirrors altgan."""
    if n_bins <= 1:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, qs)).astype(np.float64)


# Backwards-compat: legacy 6-state tencent / alibaba models pickled in R170-173.
N_STATES = N_DIST_STATES


# Conditioning features pulled from trace_characterizations.jsonl. Skips
# `reuse_ratio` from JSONL because its 4096-sample value is unreliable for
# alibaba (R169). Round 178 adds 3 inline-computable features that were
# missing: inline reuse_rate (the most-informative per-file signal),
# hot10_residency (working-set concentration), iat_lag1_autocorr (timing
# predictability). With these, COND_DIM=13.
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
    "reuse_rate_inline",
    "hot10_residency",
    "iat_lag1_autocorr",
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
        elif (k.endswith("_cv") or k.endswith("_ratio") or k == "hot10_residency"
              or k == "reuse_rate_inline" or k == "iat_lag1_autocorr"):
            out[i] = float(v) if np.isfinite(float(v)) else 0.0
        elif k.startswith("iat_") or k.startswith("obj_size_") or k == "ts_duration":
            out[i] = _safe_log1p(v)
        else:
            out[i] = float(v) if np.isfinite(float(v)) else 0.0
    return out


def cond_from_trace(path: str, max_records: int = 25_000) -> np.ndarray:
    """Round 172/178: compute the cond vector directly from the trace file.
    R178: add 3 inline-computable features (reuse_rate, hot10_residency,
    iat_lag1_autocorr) that the JSONL pipeline doesn't populate or computes
    on too-short samples.
    """
    sys.path.insert(0, "/home/darrell/Zarathustra")
    from llgan.phase_pmf_atlas import _read_trace
    ts_arr, oid_arr, sz_arr, op_arr = [], [], [], []
    for ev in _read_trace(path, keep_sentinel=True):
        ts_arr.append(ev[0])
        oid_arr.append(ev[1])
        sz_arr.append(ev[2])
        op_arr.append(ev[3])
        if len(ts_arr) >= max_records:
            break
    if len(ts_arr) < 2:
        return np.zeros(COND_DIM, dtype=np.float32)
    ts = np.asarray(ts_arr, dtype=np.float64)
    oid = np.asarray(oid_arr, dtype=np.int64)
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

    # R178: inline reuse_rate (the most-informative per-file feature for
    # alibaba's bimodal distribution; trace_characterization.jsonl's
    # 4096-sample reuse_ratio is unreliable per R169)
    seen: set = set()
    n_reuses = 0
    for x in oid:
        v = int(x)
        if v in seen:
            n_reuses += 1
        seen.add(v)
    reuse_rate_inline = n_reuses / len(oid)

    # R178: hot10_residency — share of accesses to the top-10 most-frequent ids
    from collections import Counter
    counts = Counter(int(x) for x in oid)
    top_n = min(10, len(counts))
    top_share = sum(c for _, c in counts.most_common(top_n)) / len(oid)
    hot10_residency = float(top_share)

    # R178: iat_lag1_autocorr — Pearson correlation of consecutive IATs
    if len(iat) >= 3 and iat.std() > 1e-9:
        lag = np.corrcoef(iat[:-1], iat[1:])[0, 1]
        iat_lag1_autocorr = float(lag) if np.isfinite(lag) else 0.0
    else:
        iat_lag1_autocorr = 0.0

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
        "reuse_rate_inline": reuse_rate_inline,
        "hot10_residency": hot10_residency,
        "iat_lag1_autocorr": iat_lag1_autocorr,
    }
    return cond_from_profile(profile)


# Lazy torch import — keep module importable without torch when only generation
# from a precomputed transition table is needed.
def _torch():
    import torch
    return torch


def make_net(cond_dim: int, hidden: int, n_states: int, dropout: float = 0.0):
    torch = _torch()
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout_p = dropout
            cond_layers = [nn.Linear(cond_dim, hidden), nn.SiLU()]
            if dropout > 0:
                cond_layers.append(nn.Dropout(dropout))
            cond_layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            if dropout > 0:
                cond_layers.append(nn.Dropout(dropout))
            self.cond_mlp = nn.Sequential(*cond_layers)
            self.state_emb = nn.Embedding(n_states, hidden)
            self.init_head = nn.Linear(hidden, n_states)
            trans_layers = [nn.Linear(hidden * 2, hidden), nn.SiLU()]
            if dropout > 0:
                trans_layers.append(nn.Dropout(dropout))
            trans_layers.append(nn.Linear(hidden, n_states))
            self.trans_head = nn.Sequential(*trans_layers)

        def forward_init(self, cond):
            return self.init_head(self.cond_mlp(cond))

        def forward_trans(self, cond, prev_state):
            h = self.cond_mlp(cond)
            s = self.state_emb(prev_state)
            return self.trans_head(torch.cat([h, s], dim=-1))

    return Net()


def make_rank_ar_net(cond_dim: int, hidden: int, n_states: int, n_rank_bins: int, dropout: float = 0.0):
    """Round 180: AR-rank net.

    Inputs: (cond, dist_state, prev_rank_bin) → rank_bin_logits.
    Replaces the empirical per-state rank PMF with a learned conditional
    P(rank_t | dist_state_t, prev_rank_t-1, cond). The history-dependent rank
    prediction is what b2-light's i.i.d. PMF lookup is missing — closes the
    overfitting-vs-HRC-MAE tradeoff R175-179 hit.
    """
    torch = _torch()
    import torch.nn as nn

    class RankNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            # NEW_FROM_NEW = sentinel for "no prev rank yet" (first step or after NEW)
            self.prev_rank_emb = nn.Embedding(n_rank_bins + 1, hidden // 2)
            self.state_emb = nn.Embedding(n_states, hidden // 2)
            self.head = nn.Sequential(
                nn.Linear(hidden + hidden // 2 + hidden // 2, hidden),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, n_rank_bins),
            )

        def forward(self, cond, dist_state, prev_rank_bin):
            h_cond = self.cond_mlp(cond)
            h_state = self.state_emb(dist_state)
            h_prev = self.prev_rank_emb(prev_rank_bin)
            return self.head(torch.cat([h_cond, h_state, h_prev], dim=-1))

    return RankNet()


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
    # Round 180: optional AR-rank net. If non-None, generate samples rank
    # bins from learned P(rank_t | dist_state_t, prev_rank_t-1, cond) instead
    # of the empirical rank_pmf_per_state lookup. n_rank_bins matches
    # rank_edges minus 1.
    rank_ar_state_dict: Optional[dict] = None
    rank_ar_hidden: int = 96
    # R270: time × size × phase state binning (port of altgan's architecture).
    # When n_time_bins>1 or n_size_bins>1, fit() expands the state space to
    # n_phase × n_time × n_size × N_DIST_STATES. time_edges from log1p(dt)
    # quantiles, size_edges from log(obj_size) quantiles. Per-state dt and
    # size PMFs allow generate-time emission of realistic dt/size that match
    # the chosen state's bin. Backwards-compat: defaults of 1 keep pre-R270
    # encoding identical.
    n_time_bins: int = 1
    n_size_bins: int = 1
    time_edges: Optional[np.ndarray] = None
    size_edges: Optional[np.ndarray] = None
    dt_pmf_per_state: Optional[Dict[int, np.ndarray]] = None
    size_pmf_per_state: Optional[Dict[int, np.ndarray]] = None
    dt_edges: Optional[np.ndarray] = None
    size_edges_pmf: Optional[np.ndarray] = None  # PMF bin edges (separate from quantile state edges)

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


def _read_oracle_general_full(path: str, max_records: int):
    """R270: read (ts, obj_id, obj_size) tuples from oracleGeneral .zst.
    Returns three numpy arrays. Skips sentinel records (op=-1)."""
    sys.path.insert(0, "/home/darrell/Zarathustra")
    from llgan.phase_pmf_atlas import _read_trace
    ts_l, oid_l, sz_l = [], [], []
    for ev in _read_trace(path):
        # ev = (ts, obj_id, obj_size, op_signed)
        ts_l.append(ev[0]); oid_l.append(ev[1]); sz_l.append(ev[2])
        if len(oid_l) >= max_records:
            break
    return (np.asarray(ts_l, dtype=np.float64),
            np.asarray(oid_l, dtype=np.int64),
            np.asarray(sz_l, dtype=np.int64))


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
    n_time_bins: int = 1,
    n_size_bins: int = 1,
    dropout: float = 0.0,
    cond_noise_std: float = 0.0,
    rank_ar: bool = False,
    rank_ar_hidden: int = 96,
    rank_ar_epochs: int = 600,
) -> None:
    """Train a CondTransitionNet on (cond, prev_state, next_state) tuples.

    When `inline_cond=True` (R172), compute cond features directly from the
    trace file at fit time. When `n_phase_bins>1` (R174), expand state space
    to phase × dist bins. When n_time_bins>1 or n_size_bins>1 (R270), further
    expand to phase × time × size × dist (port of altgan architecture).
    """
    torch = _torch()
    rng = np.random.default_rng(seed)
    n_phase_bins = max(int(n_phase_bins), 1)
    n_time_bins = max(int(n_time_bins), 1)
    n_size_bins = max(int(n_size_bins), 1)
    n_states = n_states_for(n_phase_bins, n_time_bins, n_size_bins)
    print(f"State space: n_phase={n_phase_bins} × n_time={n_time_bins} × n_size={n_size_bins} × {N_DIST_STATES} dist = {n_states} states")

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

    # R270 Pass 0a: collect dt/size samples for time/size quantile edges.
    # Mirrors altgan _quantile_edges over log1p(dt) and log(obj_size).
    time_edges = np.array([], dtype=np.float64)
    size_edges = np.array([], dtype=np.float64)
    if n_time_bins > 1 or n_size_bins > 1:
        all_dt_log: List[np.ndarray] = []
        all_sz_log: List[np.ndarray] = []
        for fpath in files:
            ts, oid, sz = _read_oracle_general_full(fpath, records_per_file)
            if len(ts) < 2:
                continue
            dt = np.diff(ts, prepend=ts[0])  # interarrival
            dt = np.maximum(dt, 0.0)
            all_dt_log.append(np.log1p(dt))
            sz_clamped = np.maximum(sz.astype(np.float64), 1.0)
            all_sz_log.append(np.log(sz_clamped))
        if all_dt_log:
            dt_log = np.concatenate(all_dt_log)
            sz_log = np.concatenate(all_sz_log)
            time_edges = _quantile_edges(dt_log, n_time_bins)
            size_edges = _quantile_edges(sz_log, n_size_bins)
            print(f"Time edges (n_time={n_time_bins}, log1p): {time_edges.round(3).tolist()}")
            print(f"Size edges (n_size={n_size_bins}, log):   {size_edges.round(3).tolist()}")

    # Per-file: read records, compute stack distances, encode states, build cond.
    file_conds: Dict[str, np.ndarray] = {}
    transitions: List[Tuple[np.ndarray, int, int]] = []
    initial_states: List[Tuple[np.ndarray, int]] = []
    rank_observations: Dict[int, List[int]] = {s: [] for s in range(n_states)}
    # R270: per-state dt and obj_size observations for generate-time emission.
    dt_observations: Dict[int, List[float]] = {s: [] for s in range(n_states)}
    size_observations: Dict[int, List[int]] = {s: [] for s in range(n_states)}
    # Round 180: rank-AR observations: (cond, dist_state, prev_rank_bin, next_rank_bin)
    rank_ar_obs: List[Tuple[np.ndarray, int, int, int]] = []

    use_full_state = n_time_bins > 1 or n_size_bins > 1

    for fi, fpath in enumerate(files):
        nm = os.path.basename(fpath)
        if inline_cond:
            cond = cond_from_trace(fpath, max_records=records_per_file)
        else:
            if nm not in chars:
                continue
            cond = cond_from_profile(chars[nm])
        file_conds[nm] = cond

        if use_full_state:
            ts, obj_ids, obj_sizes = _read_oracle_general_full(fpath, records_per_file)
        else:
            obj_ids = _read_oracle_general_obj_ids(fpath, records_per_file)
            ts = obj_sizes = None
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

        # R270: time/size bin per event from quantile edges
        if use_full_state:
            dt = np.maximum(np.diff(ts, prepend=ts[0]), 0.0)
            time_per_event = np.searchsorted(time_edges, np.log1p(dt), side="right")
            time_per_event = np.clip(time_per_event, 0, n_time_bins - 1)
            sz_clamped = np.maximum(obj_sizes.astype(np.float64), 1.0)
            size_per_event = np.searchsorted(size_edges, np.log(sz_clamped), side="right")
            size_per_event = np.clip(size_per_event, 0, n_size_bins - 1)
        else:
            time_per_event = np.zeros(len(obj_ids), dtype=np.int64)
            size_per_event = np.zeros(len(obj_ids), dtype=np.int64)
            dt = None

        states = np.array([
            state_from_sd(int(sd[i]), int(phase_per_event[i]),
                          int(time_per_event[i]), int(size_per_event[i]),
                          n_time_bins, n_size_bins)
            for i in range(len(sd))
        ], dtype=np.int64)

        initial_states.append((cond, int(states[0])))
        for t in range(len(states) - 1):
            transitions.append((cond, int(states[t]), int(states[t + 1])))

        # Per-state rank observations + R180 AR-rank pair observations
        prev_rank_bin = -1  # sentinel "no prev rank" => sentinel idx N_RANK_BINS
        for i, s in enumerate(states):
            dist_state = int(s) % N_DIST_STATES
            if dist_state >= STATE_REUSE_OFFSET and sd[i] >= 0:
                this_bin = int(np.searchsorted(FINE_EDGES_R180[1:], int(sd[i]), side="right"))
                this_bin = min(this_bin, N_RANK_BINS - 1)
                rank_observations[int(s)].append(int(sd[i]))
                if rank_ar:
                    pr_idx = prev_rank_bin if prev_rank_bin >= 0 else N_RANK_BINS
                    rank_ar_obs.append((cond, dist_state, pr_idx, this_bin))
                prev_rank_bin = this_bin
            else:
                prev_rank_bin = -1  # NEW or non-reuse resets

        # R270: per-state dt and size observations
        if use_full_state and ts is not None:
            for i, s in enumerate(states):
                dt_observations[int(s)].append(float(dt[i]))
                size_observations[int(s)].append(int(obj_sizes[i]))

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

    # Build per-state fine-bin rank PMF. R220: extended to 43 bins (was 29)
    # to cover the deep-IRD tail real alibaba exhibits (up to ~250k). The old
    # cap at 5669 silently dropped 15% of real ranks from the PMF, making
    # deep IRDs architecturally unreachable at generate time.
    fine_edges = FINE_EDGES_R180.astype(np.int64)
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

    # R270: build per-state dt and obj_size PMFs from observed events.
    # Used at generate time to emit realistic dt/size matching the chosen
    # state's bin (so that the next state computed from the emitted event
    # is consistent with the model's transition target). Uses log-spaced
    # bins. When n_time=n_size=1, these PMFs are unused; skip building.
    dt_pmf_per_state: Optional[Dict[int, np.ndarray]] = None
    size_pmf_per_state: Optional[Dict[int, np.ndarray]] = None
    dt_edges_pmf: Optional[np.ndarray] = None
    size_edges_pmf_arr: Optional[np.ndarray] = None
    if use_full_state:
        # Use 32 log-spaced bins for dt and size PMFs.
        all_dt_obs = np.array(
            [v for vs in dt_observations.values() for v in vs],
            dtype=np.float64,
        )
        all_sz_obs = np.array(
            [v for vs in size_observations.values() for v in vs],
            dtype=np.float64,
        )
        if len(all_dt_obs) > 0:
            dt_min = max(float(np.min(all_dt_obs)), 0.0)
            dt_max = max(float(np.max(all_dt_obs)), dt_min + 1e-6)
            dt_edges_pmf = np.geomspace(
                max(dt_min, 1e-6), dt_max + 1e-6, num=33, dtype=np.float64
            )
            dt_edges_pmf = np.concatenate(([0.0], dt_edges_pmf[1:]))
        else:
            dt_edges_pmf = np.linspace(0.0, 1.0, 33, dtype=np.float64)
        if len(all_sz_obs) > 0:
            sz_min = max(float(np.min(all_sz_obs)), 1.0)
            sz_max = max(float(np.max(all_sz_obs)), sz_min + 1.0)
            size_edges_pmf_arr = np.geomspace(sz_min, sz_max, num=33, dtype=np.float64)
        else:
            size_edges_pmf_arr = np.linspace(1.0, 1024.0, 33, dtype=np.float64)
        dt_pmf_per_state = {}
        size_pmf_per_state = {}
        for s in range(n_states):
            dts = dt_observations.get(s, [])
            szs = size_observations.get(s, [])
            if dts:
                arr = np.asarray(dts, dtype=np.float64)
                counts, _ = np.histogram(arr, bins=dt_edges_pmf)
                if counts.sum() > 0:
                    dt_pmf_per_state[s] = counts.astype(np.float64) / counts.sum()
                else:
                    dt_pmf_per_state[s] = np.ones(len(dt_edges_pmf) - 1) / (len(dt_edges_pmf) - 1)
            else:
                dt_pmf_per_state[s] = np.ones(len(dt_edges_pmf) - 1) / (len(dt_edges_pmf) - 1)
            if szs:
                arr = np.asarray(szs, dtype=np.float64)
                counts, _ = np.histogram(arr, bins=size_edges_pmf_arr)
                if counts.sum() > 0:
                    size_pmf_per_state[s] = counts.astype(np.float64) / counts.sum()
                else:
                    size_pmf_per_state[s] = np.ones(len(size_edges_pmf_arr) - 1) / (len(size_edges_pmf_arr) - 1)
            else:
                size_pmf_per_state[s] = np.ones(len(size_edges_pmf_arr) - 1) / (len(size_edges_pmf_arr) - 1)
        print(f"R270: dt PMF over {len(dt_edges_pmf)-1} bins, "
              f"size PMF over {len(size_edges_pmf_arr)-1} bins, "
              f"per-state observations: dt={sum(len(v) for v in dt_observations.values()):,} "
              f"size={sum(len(v) for v in size_observations.values()):,}")

    # Train net
    print(f"Training CondTransitionNet (hidden={hidden}, epochs={epochs}, lr={lr}, "
          f"dropout={dropout}, cond_noise_std={cond_noise_std})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # R235: ALWAYS seed torch RNGs (regardless of cond_noise_std), so
    # (a) model init weights, (b) dropout patterns, and (c) cond-noise
    # injection are all jointly reproducible from `seed`. This removes
    # the on/off A/B confound AD flagged: when only flag-on seeded torch,
    # a cond_noise > 0 vs cond_noise = 0 comparison conflated init-weight
    # change with the noise objective. Now the only difference between
    # the two paths is the noise term itself.
    # Caveat: pre-R235 atlas pkls (R206, R221, R224) were fit with torch
    # RNG in the unseeded process-default state, so re-fitting their
    # exact recipes with this code will produce a DIFFERENT (but
    # deterministic) atlas. The existing pkl files on /tiamat remain
    # the source of truth for those standing claims.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    net = make_net(COND_DIM, hidden, n_states, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    init_conds = torch.tensor(np.stack([c for c, _ in initial_states]), dtype=torch.float32, device=device)
    init_targets = torch.tensor([s for _, s in initial_states], dtype=torch.long, device=device)
    trans_conds = torch.tensor(np.stack([c for c, _, _ in transitions]), dtype=torch.float32, device=device)
    trans_prev = torch.tensor([p for _, p, _ in transitions], dtype=torch.long, device=device)
    trans_next = torch.tensor([n for _, _, n in transitions], dtype=torch.long, device=device)

    import torch.nn.functional as F
    for ep in range(epochs):
        opt.zero_grad()
        # R235: train-time cond-vector augmentation. Noise is in normalized
        # cond space (std 1 = one full cond-feature stdev). Targets the
        # diagnosed cond-mlp generalization failure mode (R225-R234 saddle).
        if cond_noise_std > 0.0:
            init_in = init_conds + cond_noise_std * torch.randn_like(init_conds)
            trans_in = trans_conds + cond_noise_std * torch.randn_like(trans_conds)
        else:
            init_in = init_conds
            trans_in = trans_conds
        init_logits = net.forward_init(init_in)
        init_loss = F.cross_entropy(init_logits, init_targets)
        trans_logits = net.forward_trans(trans_in, trans_prev)
        trans_loss = F.cross_entropy(trans_logits, trans_next)
        loss = init_loss + trans_loss
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  ep{ep:>4d}  init_loss={init_loss.item():.4f}  trans_loss={trans_loss.item():.4f}")

    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}

    # Round 180: optionally train AR-rank net
    rank_ar_state_dict_out = None
    if rank_ar and rank_ar_obs:
        # Apply same cond normalization to rank-AR observations
        norm_obs = [(((c - cond_mean) / cond_std).astype(np.float32), ds, pr, nb)
                    for c, ds, pr, nb in rank_ar_obs]
        ar_conds = torch.tensor(np.stack([o[0] for o in norm_obs]),
                                dtype=torch.float32, device=device)
        ar_dist = torch.tensor([o[1] for o in norm_obs], dtype=torch.long, device=device)
        ar_prev = torch.tensor([o[2] for o in norm_obs], dtype=torch.long, device=device)
        ar_next = torch.tensor([o[3] for o in norm_obs], dtype=torch.long, device=device)
        print(f"Training Rank-AR net on {len(norm_obs):,} observations "
              f"(hidden={rank_ar_hidden}, epochs={rank_ar_epochs})")
        rank_net = make_rank_ar_net(
            COND_DIM, rank_ar_hidden, N_DIST_STATES, N_RANK_BINS, dropout=dropout
        ).to(device)
        rank_opt = torch.optim.Adam(rank_net.parameters(), lr=lr)
        for ep in range(rank_ar_epochs):
            rank_opt.zero_grad()
            logits = rank_net(ar_conds, ar_dist, ar_prev)
            r_loss = F.cross_entropy(logits, ar_next)
            r_loss.backward()
            rank_opt.step()
            if ep % 50 == 0 or ep == rank_ar_epochs - 1:
                print(f"  rank-ep{ep:>4d}  rank_loss={r_loss.item():.4f}")
        rank_ar_state_dict_out = {k: v.cpu() for k, v in rank_net.state_dict().items()}

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
        rank_ar_state_dict=rank_ar_state_dict_out,
        rank_ar_hidden=rank_ar_hidden,
        n_time_bins=n_time_bins,
        n_size_bins=n_size_bins,
        time_edges=time_edges if len(time_edges) > 0 else None,
        size_edges=size_edges if len(size_edges) > 0 else None,
        dt_pmf_per_state=dt_pmf_per_state,
        size_pmf_per_state=size_pmf_per_state,
        dt_edges=dt_edges_pmf,
        size_edges_pmf=size_edges_pmf_arr,
        metadata={
            "trace_dir": trace_dir,
            "char_file": char_file,
            "max_files": max_files,
            "records_per_file": records_per_file,
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "dropout": dropout,
            "cond_noise_std": cond_noise_std,
            "n_transitions": len(transitions),
            "inline_cond": inline_cond,
            "n_phase_bins": n_phase_bins,
            "rank_ar": rank_ar,
            "rank_ar_obs": len(rank_ar_obs),
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
    hot_pool_prob: float = 0.0,
    hot_pool_k: int = 100,
    hot_pool_window: int = 5_000,
    adj_dup_prob: float = 0.0,
    tail_reuse_prob: float = 0.0,
    tail_reuse_min_frac: float = 0.5,
    recent_pool_prob: float = 0.0,
    recent_pool_window: int = 200,
    tail_reuse_rank_power: float = 1.0,
    hot_pool_refresh_jitter: bool = False,
    reuse_boost_prob: float = 0.0,
    reuse_boost_min_rank: int = 0,
    reuse_boost_rank_power: float = 1.0,
    stack_rank_scale: float = 1.0,
    stack_rank_max: int = -1,
    stack_rank_tail_pivot: int = -1,
    stack_rank_tail_scale: float = 1.0,
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

    # R180: optional AR-rank net
    rank_net = None
    if getattr(m, "rank_ar_state_dict", None) is not None:
        rank_net = make_rank_ar_net(
            m.cond_dim, m.rank_ar_hidden, N_DIST_STATES, N_RANK_BINS
        ).to(device)
        rank_net.load_state_dict(m.rank_ar_state_dict)
        rank_net.eval()
        print("Rank-AR net loaded; using P(rank|state,prev_rank,cond) for rank sampling")

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
                # Match the training-time cond scope so cond features at
                # generate match the distribution the net learned.
                cond_records = int(m.metadata.get(
                    "records_per_file",
                    segs[0].get("records_taken", 25_000),
                ))
                cond_vec = cond_from_trace(segs[0]["path"], max_records=cond_records)
                # Backwards-compat: if model was trained with fewer cond dims
                # (R170-177 used COND_DIM=10), trim to that size.
                if m.cond_dim < len(cond_vec):
                    cond_vec = cond_vec[: m.cond_dim]
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
            # R180: prev_rank_bin tracking for AR-rank sampling
            prev_rank_bin = N_RANK_BINS  # sentinel "no prev" = idx N_RANK_BINS
            # R181: per-stream hot-pool tracking (object-id frequency over a
            # sliding window). Closes the "fake is too uniform" failure mode
            # diagnosed in llgan/diag_hrc.py: real top-100 share = 35% on
            # tencent vs fake 1.7% before this fix.
            #
            # R184: scale window with per_stream and use exponential decay so
            # 1M-record runs don't exhaust + refresh the pool every 25 steps
            # (which was killing SIEVE at 1M, R182 corrective).
            from collections import Counter, deque
            hot_counts: Counter = Counter()
            # Decay-weighted: floats not ints
            hot_freq: Dict[int, float] = {}
            effective_window = max(int(hot_pool_window), per_stream // 4)
            hot_window: deque = deque(maxlen=effective_window)
            hot_pool_list: List[int] = []
            hot_pool_refresh_every = 200
            # R231 (WaveStitch lesson): optionally jitter the refresh interval
            # via Poisson(hot_pool_refresh_every) so the periodic hot-pool
            # composition cycle becomes irregular. Mean interval unchanged.
            # last_refresh_step starts at -refresh_every so the first refresh
            # at step=0 sees period=refresh_every (matches the prior fixed-
            # interval `decay**refresh_every` semantic exactly when flag-off).
            next_refresh_step = 0
            last_refresh_step = -hot_pool_refresh_every
            # R184 decay: each new access multiplies all counts by decay,
            # then adds 1.0 to the new id. Half-life ≈ ln(0.5)/ln(decay) steps.
            hot_decay = 0.9999  # half-life ~6900 steps
            # R194: short-window recency pool — addresses LIRS/SIEVE which need
            # recency-clustered concentration (IRR-aware) that the long-term
            # frequency hot pool washes out. Sample uniformly from last N emitted
            # obj_ids when recent_pool_prob > 0.
            recent_window: deque = deque(maxlen=max(int(recent_pool_window), 1))

            for step in range(per_stream):
                # Decode current state to (phase, dist_state) → action
                dist_state = state % N_DIST_STATES
                if dist_state == STATE_NEW:
                    if (reuse_boost_prob > 0.0 and len(stack) > 0
                            and rng.random() < reuse_boost_prob):
                        # R258: LANL-style reuse-boost — flip a NEW into a reuse.
                        # Port of altgan's boosted_reuse semantics: when boost fires,
                        # route through the existing reuse-handling chain
                        # (tail-reuse → adj-dup → recent-pool) with reuse-boost-rank
                        # as the FALLBACK. Hot-pool is SKIPPED on boosted reuses
                        # (matches altgan/neural_atlas.py:271 `not boosted_reuse` gate).
                        # Default off (prob=0.0) → bit-identical to pre-R258.
                        stack_sz = len(stack)
                        if (tail_reuse_prob > 0.0
                                and rng.random() < tail_reuse_prob):
                            lo = max(int(stack_sz * tail_reuse_min_frac), stack_sz // 2)
                            lo = min(lo, stack_sz - 1)
                            if tail_reuse_rank_power == 1.0:
                                rank = int(rng.integers(lo, stack_sz))
                            else:
                                u = float(rng.random())
                                biased = u ** tail_reuse_rank_power
                                rank = int(lo + (stack_sz - 1 - lo) * biased)
                                rank = max(lo, min(rank, stack_sz - 1))
                            obj_id = stack[rank]
                            del stack[rank]
                            stack.insert(0, obj_id)
                        elif (adj_dup_prob > 0.0
                              and rng.random() < adj_dup_prob):
                            obj_id = stack[0]
                            # already at rank 0; do not re-insert
                        elif (recent_pool_prob > 0.0
                              and len(recent_window) > 0
                              and rng.random() < recent_pool_prob):
                            rec_idx = int(rng.integers(0, len(recent_window)))
                            obj_id = recent_window[rec_idx]
                            if obj_id in stack:
                                rank = stack.index(obj_id)
                                del stack[rank]
                            stack.insert(0, obj_id)
                        else:
                            # Reuse-boost-rank fallback: sample a rank near-uniformly
                            # across the stack. Math mirrors altgan _boosted_reuse_rank:
                            #   span   = stack_len - lo
                            #   offset = floor(u^(1/p) * span)
                            #   rank   = min(lo + offset, stack_len - 1)
                            if stack_sz <= 1:
                                rank = 0
                            else:
                                lo = min(max(int(reuse_boost_min_rank), 0),
                                         stack_sz - 1)
                                span = stack_sz - lo
                                if span <= 1:
                                    rank = lo
                                else:
                                    u = float(rng.random())
                                    inv_p = 1.0 / max(
                                        float(reuse_boost_rank_power), 1e-6)
                                    offset = int(np.floor((u ** inv_p) * span))
                                    rank = min(lo + offset, stack_sz - 1)
                            obj_id = stack[rank]
                            del stack[rank]
                            stack.insert(0, obj_id)
                        prev_rank_bin = N_RANK_BINS  # sentinel: not from PMF
                    else:
                        obj_id = next_new_id
                        next_new_id += 1
                        stack.insert(0, obj_id)
                        prev_rank_bin = N_RANK_BINS  # sentinel: no prev rank after NEW
                elif (tail_reuse_prob > 0.0 and len(stack) > 0
                      and rng.random() < tail_reuse_prob):
                    # R187: deep-rank reuse injection (LANL-style min-rank boost).
                    # Pick a rank from [stack*min_frac, stack_size). Default uniform
                    # (rank_power=1.0); R211 adds power-law weighting to bias toward
                    # min_frac end (rank_power>1) or very-deep end (rank_power<1).
                    # Port of LANL's `_boosted_reuse_rank` lever (their alibaba win).
                    stack_sz = len(stack)
                    lo = max(int(stack_sz * tail_reuse_min_frac), stack_sz // 2)
                    lo = min(lo, stack_sz - 1)
                    if tail_reuse_rank_power == 1.0:
                        rank = int(rng.integers(lo, stack_sz))
                    else:
                        # u^p: p>1 biases toward lo (shallow deep-tail);
                        #       p<1 biases toward stack_sz-1 (very deep tail).
                        u = float(rng.random())
                        biased = u ** tail_reuse_rank_power
                        rank = int(lo + (stack_sz - 1 - lo) * biased)
                        rank = max(lo, min(rank, stack_sz - 1))
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                    if rank_net is not None:
                        prev_rank_bin = N_RANK_BINS - 1
                elif (adj_dup_prob > 0.0 and len(stack) > 0
                      and rng.random() < adj_dup_prob):
                    # R182: back-to-back duplicate of the most-recently-emitted
                    # obj_id (rank=0). Real adj-dup rate is ~0.3% on tencent;
                    # b2-light's i.i.d. PMF emits 0% which costs SIEVE/CLOCK
                    # policies catastrophically (see R181 corrective).
                    obj_id = stack[0]
                    # already at rank 0; do not re-insert
                    if rank_net is not None:
                        prev_rank_bin = 0
                elif (recent_pool_prob > 0.0
                      and len(recent_window) > 0
                      and rng.random() < recent_pool_prob):
                    # R194: recency-clustered redirect — pick uniformly from
                    # last N emitted obj_ids. LIRS uses IRR (Inter-Reference
                    # Recency) to classify low-IRR vs high-IRR objects; SIEVE
                    # uses referenced-bit dynamics that depend on temporal
                    # proximity. The long-term freq hot pool (R181/R184)
                    # smooths over both. recent_window preserves the burst
                    # structure that policies-with-recency need.
                    rec_idx = int(rng.integers(0, len(recent_window)))
                    obj_id = recent_window[rec_idx]
                    if obj_id in stack:
                        rank = stack.index(obj_id)
                        del stack[rank]
                    stack.insert(0, obj_id)
                    if rank_net is not None:
                        prev_rank_bin = N_RANK_BINS  # sentinel: recent-pool isn't a sampled rank
                elif (hot_pool_prob > 0.0
                      and len(hot_pool_list) > 0
                      and rng.random() < hot_pool_prob):
                    # R181: redirect to hot-pool object — closes the top-K
                    # access-share gap (real concentrates 35% of tencent
                    # accesses on top-100 objects; fake spreads uniformly).
                    hot_idx = int(rng.integers(0, len(hot_pool_list)))
                    obj_id = hot_pool_list[hot_idx]
                    if obj_id in stack:
                        rank = stack.index(obj_id)
                        del stack[rank]
                    stack.insert(0, obj_id)
                    if rank_net is not None:
                        prev_rank_bin = N_RANK_BINS  # sentinel: hot-pool isn't a sampled rank
                else:
                    # R180: use AR-rank net if available, else fall back to
                    # empirical PMF (R172/R174 logic).
                    if rank_net is not None:
                        ds_t = torch.tensor([dist_state], dtype=torch.long, device=device)
                        pr_t = torch.tensor([prev_rank_bin], dtype=torch.long, device=device)
                        with torch.no_grad():
                            r_logits = rank_net(cond_t, ds_t, pr_t)
                            pmf = torch.softmax(r_logits / max(temperature, 1e-3), dim=-1).cpu().numpy()[0]
                    elif n_phase_bins > 1:
                        # R174: phase-marginalized rank PMF.
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
                        # fallback to NEW; AR history resets to sentinel (LANL R20 fix)
                        obj_id = next_new_id
                        next_new_id += 1
                        stack.insert(0, obj_id)
                        if rank_net is not None:
                            prev_rank_bin = N_RANK_BINS
                    else:
                        lo_eff = min(lo, stack_sz - 1)
                        hi_eff = min(hi, stack_sz - 1)
                        rank = int(rng.integers(lo_eff, hi_eff + 1))
                        # R263: scout-rank calibration (port from altgan
                        # _calibrated_stack_rank). Defaults (scale=1.0,
                        # max=-1, tail_pivot=-1) → bit-identical no-op.
                        # LANL's MSR/Baleen24 wins use stack_rank_scale=5.0
                        # to push raw ranks deeper.
                        if (stack_rank_scale != 1.0 or stack_rank_max >= 0
                                or stack_rank_tail_pivot >= 0):
                            rank = int(round(max(rank, 0) * stack_rank_scale))
                            if stack_rank_tail_pivot >= 0 and rank > stack_rank_tail_pivot:
                                tail = rank - int(stack_rank_tail_pivot)
                                rank = int(stack_rank_tail_pivot) + int(round(tail * stack_rank_tail_scale))
                            if stack_rank_max >= 0:
                                rank = min(rank, int(stack_rank_max))
                            rank = min(max(rank, 0), stack_sz - 1)
                        obj_id = stack[rank]
                        del stack[rank]
                        stack.insert(0, obj_id)
                        # R180/LANL R20 fix: record EMITTED rank's bin, not the
                        # pre-clamp sampled bin. Otherwise AR history points at
                        # ranks that were never actually emitted.
                        if rank_net is not None:
                            emitted_bin = int(np.searchsorted(
                                FINE_EDGES_R180[1:], rank, side="right"
                            ))
                            prev_rank_bin = min(emitted_bin, N_RANK_BINS - 1)
                if len(stack) > max_stack_depth:
                    stack.pop()

                # Sample opcode + size + ts (placeholder; trivial since cache eval ignores)
                op_sampled = 0
                if m.opcode_pmf:
                    keys = sorted(m.opcode_pmf.keys())
                    probs = [m.opcode_pmf[k] for k in keys]
                    op_sampled = int(np.random.choice(keys, p=probs))

                # R270: emit realistic dt and obj_size from per-state PMFs.
                # When dt_pmf_per_state is None (n_time_bins=n_size_bins=1
                # atlases pre-R270), fall back to placeholder ts+=1.0 and
                # obj_size=4096 — bit-identical to pre-R270 generate.
                if (getattr(m, "dt_pmf_per_state", None) is not None
                        and getattr(m, "size_pmf_per_state", None) is not None
                        and getattr(m, "dt_edges", None) is not None
                        and getattr(m, "size_edges_pmf", None) is not None):
                    dt_pmf = m.dt_pmf_per_state.get(int(state))
                    if dt_pmf is None:
                        dt_emit = 1.0
                    else:
                        bin_idx = int(rng.choice(len(dt_pmf), p=dt_pmf))
                        lo = float(m.dt_edges[bin_idx])
                        hi = float(m.dt_edges[bin_idx + 1])
                        dt_emit = float(rng.uniform(lo, max(hi, lo + 1e-9)))
                    size_pmf = m.size_pmf_per_state.get(int(state))
                    if size_pmf is None:
                        size_emit = 4096
                    else:
                        bin_idx = int(rng.choice(len(size_pmf), p=size_pmf))
                        lo = float(m.size_edges_pmf[bin_idx])
                        hi = float(m.size_edges_pmf[bin_idx + 1])
                        size_emit = int(rng.uniform(lo, max(hi, lo + 1.0)))
                    ts += max(dt_emit, 0.0)
                else:
                    ts += 1.0
                    size_emit = 4096
                rows.append({
                    "stream_id": sid,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": int(size_emit),
                    "opcode": op_sampled,
                    "tenant": 0,
                })

                # R194: append every emitted obj_id to the short-window
                # recency deque. Cheap (deque maxlen auto-evicts oldest).
                if recent_pool_prob > 0.0:
                    recent_window.append(int(obj_id))

                # R184: decay-weighted hot pool — scales to 1M-record streams
                # without the R181 sliding-window exhaustion issue. Counts
                # are multiplied by decay each step then incremented by 1.0
                # for the new access. Top-K is computed from current weights.
                if hot_pool_prob > 0.0:
                    oid_int = int(obj_id)
                    # Lazy decay: scan dict only on refresh to bound cost
                    hot_freq[oid_int] = hot_freq.get(oid_int, 0.0) + 1.0
                    if step == next_refresh_step:
                        # Apply decay (compound for actual elapsed period) and
                        # rebuild top-K. Drop entries below 0.01 to bound size.
                        period = max(step - last_refresh_step, 0)
                        decay_factor = hot_decay ** period
                        hot_freq = {k: v * decay_factor for k, v in hot_freq.items()
                                    if v * decay_factor >= 0.01}
                        hot_pool_list = [
                            k for k, _ in
                            sorted(hot_freq.items(), key=lambda kv: -kv[1])[:hot_pool_k]
                        ]
                        last_refresh_step = step
                        if hot_pool_refresh_jitter:
                            # WaveStitch jitter: Poisson(mean=hot_pool_refresh_every)
                            interval = max(int(rng.poisson(hot_pool_refresh_every)), 1)
                            next_refresh_step = step + interval
                        else:
                            next_refresh_step = step + hot_pool_refresh_every

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
                # R270: with time/size axes, decode the inner state (time*size*dist)
                # and only override the phase component.
                if n_phase_bins > 1 and phase_edges is not None:
                    expected_pb = int(np.searchsorted(phase_edges, current_rate, side="right"))
                    expected_pb = min(expected_pb, n_phase_bins - 1)
                    nt = max(int(getattr(m, "n_time_bins", 1)), 1)
                    nz = max(int(getattr(m, "n_size_bins", 1)), 1)
                    n_inner = nt * nz * N_DIST_STATES
                    inner = state % n_inner
                    state = expected_pb * n_inner + inner

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
    pfit.add_argument("--n-time-bins", type=int, default=1,
                      help="R270: time-axis state expansion. 1 (default) = no time bins. "
                           "altgan default = 4 (log1p(dt) quartiles). State space grows by this factor.")
    pfit.add_argument("--n-size-bins", type=int, default=1,
                      help="R270: object-size axis state expansion. 1 (default) = no size bins. "
                           "altgan default = 4 (log(obj_size) quartiles). State space grows by this factor.")
    pfit.add_argument("--dropout", type=float, default=0.0,
                      help="R179: dropout in cond_mlp + trans_head; regularizes overfitting.")
    pfit.add_argument("--cond-noise-std", type=float, default=0.0,
                      help="R235: train-time Gaussian noise (std in normalized "
                           "cond space) added to cond vectors each epoch. "
                           "Targets cond-mlp generalization failure to off-"
                           "manifold generation-time conds. 0.0 = no noise "
                           "(no randn draws). Note: torch RNG is always seeded "
                           "from --seed at fit start, so re-fitting an existing "
                           "recipe under this code produces a deterministic but "
                           "DIFFERENT atlas than pre-R235 pkls — the on-disk "
                           "atlases remain authoritative for prior race claims.")
    pfit.add_argument("--rank-ar", action="store_true",
                      help="R180: also train an AR-rank net P(rank|dist_state,prev_rank,cond) "
                           "that replaces the empirical rank PMF lookup at generate time. "
                           "Targets the b2-light i.i.d. PMF ceiling.")
    pfit.add_argument("--rank-ar-hidden", type=int, default=96)
    pfit.add_argument("--rank-ar-epochs", type=int, default=600)

    pgen = sub.add_parser("generate")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--manifest", required=True)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--seed", type=int, default=42)
    pgen.add_argument("--temperature", type=float, default=1.0,
                      help="Softmax temperature for state-transition sampling. "
                           "Lower = sharper / closer to argmax. R175 experiment.")
    pgen.add_argument("--hot-pool-prob", type=float, default=0.0,
                      help="R181: probability of redirecting a reuse to a hot-pool "
                           "object (top-K by sliding-window frequency) instead of "
                           "the i.i.d. PMF rank. Closes the 20x top-100 access "
                           "concentration gap diagnosed in llgan/diag_hrc.py.")
    pgen.add_argument("--hot-pool-k", type=int, default=100,
                      help="R181: hot-pool size (top-K by sliding-window count).")
    pgen.add_argument("--hot-pool-window", type=int, default=5_000,
                      help="R181: sliding window size for hot-pool counting.")
    pgen.add_argument("--adj-dup-prob", type=float, default=0.0,
                      help="R182: prob of emitting back-to-back duplicate "
                           "(rank=0) instead of sampling. Calibrate to corpus "
                           "real adj-dup rate (~0.003 tencent). Fixes the "
                           "SIEVE catastrophic gap diagnosed in R181 corrective.")
    pgen.add_argument("--tail-reuse-prob", type=float, default=0.0,
                      help="R187: prob of redirecting a reuse to a uniform "
                           "deep-rank pick (rank in [stack*tail_min_frac, stack)). "
                           "Closes the FIFO/LIRS small-cap over-miss without "
                           "architectural change.")
    pgen.add_argument("--tail-reuse-min-frac", type=float, default=0.5,
                      help="R187: lower bound for tail-reuse rank as a "
                           "fraction of stack size (default 0.5 = deep half).")
    pgen.add_argument("--recent-pool-prob", type=float, default=0.0,
                      help="R194: prob of redirecting a reuse to a uniform "
                           "pick from the last N emitted obj_ids. Targets "
                           "LIRS/SIEVE which need recency-clustered (IRR-aware) "
                           "concentration that the long-term freq hot pool "
                           "washes out.")
    pgen.add_argument("--recent-pool-window", type=int, default=200,
                      help="R194: deque size for recent-pool sampling.")
    pgen.add_argument("--tail-reuse-rank-power", type=float, default=1.0,
                      help="R211: power-law weight for tail-reuse rank sampling. "
                           "1.0 = uniform (R187 default); >1.0 biases toward "
                           "min_frac (shallower deep-tail); <1.0 biases toward "
                           "stack_size-1 (very deep tail). Port of LANL's "
                           "rank_power lever (their alibaba win).")
    pgen.add_argument("--max-stack-depth", type=int, default=8192,
                      help="R220 (IRD diag): cap on per-stream stack depth. "
                           "Default 8192 was chosen pre-IRD-diagnostic. Real "
                           "alibaba has 15% of IRDs >31k — cap=8192 makes those "
                           "architecturally unreachable. Larger cap allows deep "
                           "reuses; cost is generation time (linear in stack scan).")
    pgen.add_argument("--hot-pool-refresh-jitter", action="store_true",
                      help="R231 (WaveStitch lesson): jitter hot-pool refresh "
                           "interval as Poisson(200) instead of fixed 200 steps. "
                           "Mean unchanged; breaks the periodic refresh artifact "
                           "in generated traces.")
    pgen.add_argument("--reuse-boost-prob", type=float, default=0.0,
                      help="R258: prob of converting a STATE_NEW emission into "
                           "a reuse, with rank sampled near-uniformly across "
                           "the stack. Port of LANL altgan stack_reuse_boost_prob "
                           "(their Baleen24 win at prob=0.60 min_rank=0 power=0.1).")
    pgen.add_argument("--reuse-boost-min-rank", type=int, default=0,
                      help="R258: minimum rank for reuse-boost sampling. "
                           "0 (default) allows boosting from rank 0 (front of stack).")
    pgen.add_argument("--reuse-boost-rank-power", type=float, default=1.0,
                      help="R258: power-law weight for reuse-boost rank sampling. "
                           "Matches altgan _boosted_reuse_rank: offset = "
                           "floor(u^(1/p) * span). 1.0 = uniform; <1.0 → exponent "
                           ">1 → u^(1/p) biased toward 0 → rank biased toward lo "
                           "(shallow). LANL's 0.1 = strong shallow-bias.")
    pgen.add_argument("--stack-rank-scale", type=float, default=1.0,
                      help="R263 (port of altgan _calibrated_stack_rank): "
                           "multiply the PMF-sampled raw rank by this scale. "
                           "scale > 1 pushes ranks deeper; scale < 1 compresses "
                           "shallower. LANL's MSR scout uses scale=5.0.")
    pgen.add_argument("--stack-rank-max", type=int, default=-1,
                      help="R263: cap calibrated rank at this max (-1 = no cap). "
                           "Prevents rank from exceeding stack diversity.")
    pgen.add_argument("--stack-rank-tail-pivot", type=int, default=-1,
                      help="R263: above this rank, apply tail-scale instead of "
                           "global scale. -1 = no separate tail handling.")
    pgen.add_argument("--stack-rank-tail-scale", type=float, default=1.0,
                      help="R263: scale applied to ranks above tail-pivot. "
                           "Compresses (<1) or stretches (>1) the deep tail "
                           "independently of the bulk distribution.")

    args = p.parse_args()
    if args.cmd == "fit":
        fit(
            args.trace_dir, args.char_file, args.output,
            max_files=args.max_files, records_per_file=args.records_per_file,
            hidden=args.hidden, epochs=args.epochs, lr=args.lr, seed=args.seed,
            inline_cond=args.inline_cond, n_phase_bins=args.n_phase_bins,
            n_time_bins=args.n_time_bins, n_size_bins=args.n_size_bins,
            dropout=args.dropout,
            cond_noise_std=args.cond_noise_std,
            rank_ar=args.rank_ar, rank_ar_hidden=args.rank_ar_hidden,
            rank_ar_epochs=args.rank_ar_epochs,
        )
    elif args.cmd == "generate":
        generate(
            args.model, args.manifest, args.output,
            n_records=args.n, seed=args.seed, temperature=args.temperature,
            hot_pool_prob=args.hot_pool_prob, hot_pool_k=args.hot_pool_k,
            hot_pool_window=args.hot_pool_window,
            adj_dup_prob=args.adj_dup_prob,
            tail_reuse_prob=args.tail_reuse_prob,
            tail_reuse_min_frac=args.tail_reuse_min_frac,
            recent_pool_prob=args.recent_pool_prob,
            recent_pool_window=args.recent_pool_window,
            tail_reuse_rank_power=args.tail_reuse_rank_power,
            max_stack_depth=args.max_stack_depth,
            hot_pool_refresh_jitter=args.hot_pool_refresh_jitter,
            reuse_boost_prob=args.reuse_boost_prob,
            reuse_boost_min_rank=args.reuse_boost_min_rank,
            reuse_boost_rank_power=args.reuse_boost_rank_power,
            stack_rank_scale=args.stack_rank_scale,
            stack_rank_max=args.stack_rank_max,
            stack_rank_tail_pivot=args.stack_rank_tail_pivot,
            stack_rank_tail_scale=args.stack_rank_tail_scale,
        )


if __name__ == "__main__":
    main()
