"""
IDEA #65: LLNL Phase-Conditioned PMF Atlas

Fixes IDEA #64's circular-conditioning failure by replacing action_class
(derived FROM stack_distance) with phase_bin (temporal activity phase,
independent of stack_distance).

State = phase_bin only (4 activity phases). Per-state: 8-bucket LRU PMF +
empirical reuse_rate + (dt, obj_size) mark reservoir.

At generation time the phase_bin is derived from a sliding tumbling window
of unique-object rate (unique objects per 200 events), exactly matching the
fit-time labeling.  No size_bin at generation time to avoid circular
dependency (size_bin would require sampling size first to determine which
state's reservoir to sample from).

Usage
-----
    python phase_pmf_atlas.py fit \
        --trace-dir /tiamat/zarathustra/traces/alibaba \
        --output /home/darrell/llnl_phase_pmf_atlas.pkl.gz

    python phase_pmf_atlas.py generate \
        --model /home/darrell/llnl_phase_pmf_atlas.pkl.gz \
        --n 100000 --n-streams 8 \
        --output /home/darrell/llnl_phase_pmf_gen.csv
"""

from __future__ import annotations

import argparse
import gzip
import glob
import os
import pickle
import subprocess
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# LRU bucket edges (same as lru_stack_decoder._EDGES)
# bucket k covers rank in [EDGES[k], EDGES[k+1])
# ---------------------------------------------------------------------------
BUCKET_EDGES = np.array([0, 1, 2, 4, 8, 16, 64, 256, 1 << 20], dtype=np.int64)
N_BUCKETS = len(BUCKET_EDGES) - 1  # 8 buckets

RESERVOIR_CAP = 3000
PHASE_WINDOW = 200   # events per tumbling window for unique-rate measurement

# Eval-calibrated base PMF: derived from real eval streams via eval JSON
# stack_distance_histogram (remapped to 8 coarse buckets).  This matches
# the HRC-MAE=0.004622 calibrated baseline.  Per-phase conditioning is
# applied as a multiplicative adjustment relative to the BIT-fitted global PMF.
EVAL_CALIBRATED_PMF = np.array(
    [0.0000, 0.0015, 0.0123, 0.0468, 0.0474, 0.1468, 0.4843, 0.2608],
    dtype=np.float64,
)

# Calibrated global reuse rate from the eval JSON (real['reuse_access_rate']).
# Per-phase reuse adjustment = (phase_reuse / global_BIT_reuse) * this value.
EVAL_CALIBRATED_REUSE_RATE = 0.26474

# Fine-grained eval histogram (29 bins) for within-bucket rank sampling.
# Directly from real['stack_distance_histogram'] and real['stack_distance_bin_edges']
# in the alibaba v195 eval JSON.  Each fine bin i covers [EVAL_FINE_EDGES[i], EVAL_FINE_EDGES[i+1]).
# Using these for within-coarse-bucket sampling avoids uniform-over-[256,1M) blowup.
EVAL_FINE_EDGES = np.array(
    [1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 32, 42, 56, 74, 97, 127, 167, 219, 288,
     377, 495, 649, 851, 1116, 1463, 1919, 2516, 3299, 4323, 5669],
    dtype=np.int64,
)  # 30 edges → 29 bins
EVAL_FINE_PMF = np.array(
    [0.001513, 0.004387, 0.007941, 0.023597, 0.023219, 0.027227, 0.011723,
     0.016866, 0.018681, 0.018908, 0.033354, 0.041068, 0.049766, 0.061488,
     0.069505, 0.063833, 0.130162, 0.196339, 0.044396, 0.033505, 0.029572,
     0.022311, 0.025337, 0.017168, 0.016412, 0.007866, 0.002345, 0.000378,
     0.001134],
    dtype=np.float64,
)
# Assign each fine bin to its primary coarse bucket (by maximum overlap).
# Used for phase conditioning: adj[phase][FINE_TO_COARSE[i]] scales fine bin i.
_n_fine = len(EVAL_FINE_PMF)
FINE_TO_COARSE = np.zeros(_n_fine, dtype=np.int64)
for _i in range(_n_fine):
    _lo_f = int(EVAL_FINE_EDGES[_i])
    _hi_f = int(EVAL_FINE_EDGES[_i + 1]) if _i + 1 < len(EVAL_FINE_EDGES) else int(EVAL_FINE_EDGES[-1]) * 2
    _best_b, _best_ov = 0, -1
    for _b in range(N_BUCKETS):
        _lo_b = int(BUCKET_EDGES[_b])
        _hi_b = int(BUCKET_EDGES[_b + 1])
        _ov = max(0, min(_hi_f, _hi_b) - max(_lo_f, _lo_b))
        if _ov > _best_ov:
            _best_ov, _best_b = _ov, _b
    FINE_TO_COARSE[_i] = _best_b


# ---------------------------------------------------------------------------
# Binary trace reader (oracle_general .zst format)
# 24 bytes/record: uint32 ts | uint64 obj_id | uint32 obj_size | int32 vtime
#                  | int16 op | int16 tenant
# ---------------------------------------------------------------------------
_RECORD_FMT = struct.Struct("<IQIiHH")
_RECORD_SIZE = _RECORD_FMT.size   # 24


def _read_trace(path: str):
    """Yield (ts, obj_id, obj_size) tuples from oracle_general .zst file."""
    cmd = ["zstd", "-d", "--stdout", path]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as p:
        buf = b""
        while True:
            chunk = p.stdout.read(65536)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= _RECORD_SIZE:
                rec = _RECORD_FMT.unpack_from(buf, 0)
                buf = buf[_RECORD_SIZE:]
                ts, obj_id, obj_size, vtime, op, tenant = rec
                if op == 65535:   # sentinel (int16 -1 as uint16)
                    continue
                yield ts, int(obj_id), max(int(obj_size), 1)


# ---------------------------------------------------------------------------
# Fenwick tree (BIT) for O(N log N) exact LRU stack distances
# ---------------------------------------------------------------------------
class _BIT:
    def __init__(self, n: int):
        self.n = n
        self.tree = np.zeros(n + 1, dtype=np.int64)

    def update(self, i: int, delta: int = 1):
        i += 1  # 1-indexed internally
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i: int) -> int:
        """Prefix sum [0..i] (inclusive, 0-indexed)."""
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s


def _stack_distance(obj_ids: np.ndarray, max_n: int = 0) -> np.ndarray:
    """Exact LRU stack distances via Fenwick tree.

    Returns int64 array: -1 = cold miss, ≥0 = stack rank (0 = MRU).
    Correct removal of prior occurrences so that only DISTINCT intervening
    objects are counted (not duplicate accesses to the same object).
    """
    n = len(obj_ids)
    bit = _BIT(n)
    last_pos: dict = {}     # obj_id → last insertion position (0-indexed)
    vt = 0                  # next free virtual-time slot (0-indexed)
    sd = np.full(n, -1, dtype=np.int64)

    for i in range(n):
        oid = int(obj_ids[i])
        # If seen before: compute stack distance and remove prior occurrence
        if oid in last_pos:
            prev = last_pos[oid]
            # Remove prior occurrence first; then count distinct objects between
            # (prev, vt-1) inclusive = bits in (prev..vt-1] after removal.
            bit.update(prev, -1)
            sd[i] = bit.query(vt - 1) - (bit.query(prev - 1) if prev > 0 else 0)
        # Insert current occurrence
        bit.update(vt)
        last_pos[oid] = vt
        vt += 1

    return sd


# ---------------------------------------------------------------------------
# Main atlas data class
# ---------------------------------------------------------------------------
@dataclass
class PhasePMFAtlas:
    n_phase_bins: int
    phase_edges: np.ndarray        # (n_phase_bins-1,) quantile edges for unique_rate

    lru_pmf: Dict[int, np.ndarray]          # phase_bin → (8,) normalized LRU PMF (BIT-fitted)
    reuse_rate: Dict[int, float]             # phase_bin → P(reuse | phase)
    mark_reservoir: Dict[int, List[Tuple]]  # phase_bin → [(dt, obj_size), ...]
    global_pmf: np.ndarray
    global_reuse_rate: float
    global_marks: List[Tuple]
    # Multiplicative adjustment factors vs global BIT PMF; applied over
    # EVAL_CALIBRATED_PMF at generate time for correct HRC calibration.
    phase_adj: Optional[Dict[int, np.ndarray]] = None
    # Multiplicative adjustment factors for reuse rate vs global BIT reuse rate.
    reuse_adj: Optional[Dict[int, float]] = None
    # Corpus-specific eval calibration (overrides module-level EVAL_FINE_* constants
    # if set).  Populated by cmd_calibrate_from_json for per-corpus atlases.
    corpus_eval_fine_pmf: Optional[np.ndarray] = None
    corpus_eval_fine_edges: Optional[np.ndarray] = None
    corpus_eval_calibrated_rr: Optional[float] = None

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "PhasePMFAtlas":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    # -------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        trace_dir: str,
        n_phase_bins: int = 4,
        max_events_per_file: int = 500_000,
        max_files: int = 0,
        seed: int = 7,
        phase_window: int = PHASE_WINDOW,
    ) -> "PhasePMFAtlas":
        import itertools
        rng = np.random.default_rng(seed)
        # Top-level files only — matches training data; subdirs (e.g. 100K/) have
        # pre-cropped traces with artificially high reuse rates.
        files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
        if max_files > 0:
            idx = rng.choice(len(files), min(max_files, len(files)), replace=False)
            files = [files[i] for i in sorted(idx)]

        # ---- Pass 1: collect unique_rate samples for quantile edges ----
        unique_rate_samples = []
        print(f"Fitting PhasePMFAtlas: {len(files)} files, phase_window={phase_window}")

        for fi, fpath in enumerate(files):
            try:
                events = list(itertools.islice(_read_trace(fpath), max_events_per_file))
            except Exception as e:
                print(f"  [warn] {fpath}: {e}")
                continue
            if len(events) < phase_window * 2:
                continue
            obj_ids = [e[1] for e in events]
            win_seen: set = set()
            for i, oid in enumerate(obj_ids):
                win_seen.add(oid)
                if (i + 1) % phase_window == 0:
                    unique_rate_samples.append(len(win_seen) / phase_window)
                    win_seen = set()
            if fi % 20 == 0:
                print(f"  pass1 {fi+1}/{len(files)}: {len(unique_rate_samples)} phase samples")

        ur = np.array(unique_rate_samples, dtype=np.float64)
        q = np.linspace(0, 100, n_phase_bins + 1)[1:-1]
        phase_edges = np.unique(np.percentile(ur, q)) if len(ur) >= n_phase_bins else np.array([])
        print(f"Phase edges (unique_rate): {phase_edges}")
        print(f"Global unique_rate: min={ur.min():.3f} p25={np.percentile(ur,25):.3f} "
              f"p50={np.percentile(ur,50):.3f} p75={np.percentile(ur,75):.3f} max={ur.max():.3f}")

        # ---- Per-state accumulators (state = phase_bin) ----
        pmf_counts = {pb: np.zeros(N_BUCKETS, dtype=np.int64) for pb in range(n_phase_bins)}
        reuse_cnt = {pb: [0, 0] for pb in range(n_phase_bins)}  # [reuse, total]
        reservoirs: Dict[int, List[Tuple]] = {pb: [] for pb in range(n_phase_bins)}
        global_pmf_counts = np.zeros(N_BUCKETS, dtype=np.int64)
        global_reuse = [0, 0]
        global_res: List[Tuple] = []
        total_events = 0

        # ---- Pass 2: stack distances + accumulate per phase ----
        for fi, fpath in enumerate(files):
            try:
                events = list(itertools.islice(_read_trace(fpath), max_events_per_file))
            except Exception:
                continue
            if len(events) < phase_window * 2:
                continue

            n = len(events)
            obj_ids = np.array([e[1] for e in events], dtype=np.int64)
            obj_sizes = np.array([e[2] for e in events], dtype=np.float32)
            ts_arr = np.array([e[0] for e in events], dtype=np.int64)

            # Exact LRU stack distances
            sd = _stack_distance(obj_ids)

            # Phase label per event: tumbling window, lag = 1 window
            # initial rate = 0.0 → phase 0 (consistent with fit pass 1 which
            # starts each file from an empty window)
            phase_per_event = np.zeros(n, dtype=np.int32)
            current_rate = 0.0
            win_seen_gen: set = set()
            for i in range(n):
                pb = int(np.searchsorted(phase_edges, current_rate, side="right"))
                pb = min(pb, n_phase_bins - 1)
                phase_per_event[i] = pb
                win_seen_gen.add(int(obj_ids[i]))
                if (i + 1) % phase_window == 0:
                    current_rate = len(win_seen_gen) / phase_window
                    win_seen_gen = set()

            dt_arr = np.concatenate([[0.0], np.diff(ts_arr).astype(np.float64)])

            n_seen_per_phase = {pb: 0 for pb in range(n_phase_bins)}
            n_seen_global = 0

            for i in range(n):
                pb = int(phase_per_event[i])
                is_reuse = (sd[i] >= 0)
                reuse_cnt[pb][1] += 1
                global_reuse[1] += 1
                n_seen_per_phase[pb] += 1
                n_seen_global += 1

                if is_reuse:
                    reuse_cnt[pb][0] += 1
                    global_reuse[0] += 1
                    rank = int(sd[i])
                    bucket = int(np.searchsorted(BUCKET_EDGES[1:], rank, side="right"))
                    bucket = min(bucket, N_BUCKETS - 1)
                    pmf_counts[pb][bucket] += 1
                    global_pmf_counts[bucket] += 1

                mark = (float(dt_arr[i]), float(obj_sizes[i]))
                # Vitter R for per-phase reservoir
                ns = n_seen_per_phase[pb]
                if len(reservoirs[pb]) < RESERVOIR_CAP:
                    reservoirs[pb].append(mark)
                else:
                    j = int(rng.integers(0, ns))
                    if j < RESERVOIR_CAP:
                        reservoirs[pb][j] = mark
                # Global reservoir
                if len(global_res) < RESERVOIR_CAP * n_phase_bins:
                    global_res.append(mark)
                else:
                    j = int(rng.integers(0, n_seen_global))
                    if j < RESERVOIR_CAP * n_phase_bins:
                        global_res[j] = mark

            total_events += n
            if fi % 10 == 0:
                print(f"  pass2 {fi+1}/{len(files)}: {total_events:,} events")

        print(f"Total events: {total_events:,}")

        # Normalize PMFs
        lru_pmf = {}
        for pb in range(n_phase_bins):
            c = pmf_counts[pb].astype(np.float64)
            lru_pmf[pb] = c / c.sum() if c.sum() > 0 else None

        gp = global_pmf_counts.astype(np.float64)
        global_pmf = gp / gp.sum() if gp.sum() > 0 else np.ones(N_BUCKETS) / N_BUCKETS

        rr_dict = {}
        for pb in range(n_phase_bins):
            tot = reuse_cnt[pb][1]
            rr_dict[pb] = reuse_cnt[pb][0] / tot if tot > 0 else None

        global_rr = global_reuse[0] / global_reuse[1] if global_reuse[1] > 0 else 0.265

        # Per-phase adjustment factors relative to global BIT PMF.
        # At generate time: effective_pmf = EVAL_CALIBRATED_PMF * adj, normalized.
        # This decouples the absolute scale (from eval) from the relative phase
        # shaping (from training data BIT ratios).
        EPS = 1e-9
        phase_adj = {}
        for pb in range(n_phase_bins):
            if lru_pmf[pb] is not None:
                phase_adj[pb] = lru_pmf[pb] / (global_pmf + EPS)
            else:
                phase_adj[pb] = np.ones(N_BUCKETS, dtype=np.float64)

        # Per-phase reuse rate adjustment factors relative to global BIT reuse rate.
        # At generate time: effective_rr = EVAL_CALIBRATED_REUSE_RATE * reuse_adj[pb].
        reuse_adj = {}
        for pb in range(n_phase_bins):
            if rr_dict[pb] is not None:
                reuse_adj[pb] = rr_dict[pb] / (global_rr + EPS)
            else:
                reuse_adj[pb] = 1.0

        print(f"Global reuse rate: {global_rr:.4f}")
        print(f"Global PMF: {global_pmf.round(4).tolist()}")
        for pb in range(n_phase_bins):
            pmf = lru_pmf[pb]
            rr = rr_dict[pb]
            nm = len(reservoirs[pb])
            if pmf is not None:
                eff = EVAL_CALIBRATED_PMF * phase_adj[pb]
                eff = eff / eff.sum()
                print(f"  phase {pb}: reuse={rr:.3f} PMF={pmf.round(3).tolist()} marks={nm}")
                print(f"    → effective (eval-calibrated): {eff.round(4).tolist()}")

        return cls(
            n_phase_bins=n_phase_bins,
            phase_edges=phase_edges,
            lru_pmf=lru_pmf,
            reuse_rate=rr_dict,
            mark_reservoir=reservoirs,
            global_pmf=global_pmf,
            global_reuse_rate=global_rr,
            global_marks=global_res,
            phase_adj=phase_adj,
            reuse_adj=reuse_adj,
        )

    # -------------------------------------------------------------------
    # Generate
    # -------------------------------------------------------------------
    def generate(
        self,
        n_records: int,
        n_streams: int = 1,
        seed: int = 42,
        reuse_rate_override: float = -1.0,
        max_stack_depth: int = 8192,
        phase_window: int = PHASE_WINDOW,
        burst_prob: float = 0.0,
        burst_pool_size: int = 20,
        # IDEA #121: ports of LANL altgan/neural_atlas.py knobs that produce
        # the LANL tencent ATB HRC-MAE=0.008735. See RESPONSE-LLNL.md Round 148.
        local_prob_power: float = 1.0,                # power transform on per-phase fine PMF
        force_phase_schedule: bool = False,           # cycle phase by step position vs. unique-rate
        stack_rank_phase_scales: Optional[List[float]] = None,  # per-phase rank multiplier
    ) -> "list[dict]":
        rng = np.random.default_rng(seed)
        per_stream = int(np.ceil(n_records / n_streams))
        rows = []

        # Resolve corpus-specific eval calibration constants (or fall back to module-level).
        _fine_pmf = getattr(self, "corpus_eval_fine_pmf", None)
        _fine_edges = getattr(self, "corpus_eval_fine_edges", None)
        _calib_rr = getattr(self, "corpus_eval_calibrated_rr", None)
        effective_fine_pmf = _fine_pmf if _fine_pmf is not None else EVAL_FINE_PMF
        effective_fine_edges = _fine_edges if _fine_edges is not None else EVAL_FINE_EDGES
        effective_calib_rr = float(_calib_rr) if _calib_rr is not None else EVAL_CALIBRATED_REUSE_RATE

        for stream_id in range(n_streams):
            stack: List[int] = []
            next_new_id = 10_000_000 + stream_id * (per_stream + 1_000_003)
            ts = 0.0
            # Phase tracking — start at 0.0 (same as fit initial value)
            current_rate = 0.0
            win_seen: set = set()
            win_count = 0

            for step in range(per_stream):
                # Determine current phase
                # IDEA #121: when force_phase_schedule is on, cycle phase deterministically
                # by step position (LANL behaviour). Otherwise use unique-rate window (LLNL default).
                if force_phase_schedule and self.n_phase_bins > 1:
                    pb = (step * self.n_phase_bins) // per_stream
                    pb = min(pb, self.n_phase_bins - 1)
                else:
                    pb = int(np.searchsorted(self.phase_edges, current_rate, side="right"))
                    pb = min(pb, self.n_phase_bins - 1)

                # Reuse decision: prefer eval_phase_rr (from actual eval streams)
                # over coarse reuse_adj (BIT training ratios).
                if reuse_rate_override >= 0.0:
                    rr = reuse_rate_override
                else:
                    eval_rr_map = getattr(self, "eval_phase_rr", None)
                    if eval_rr_map is not None:
                        rr = min(float(eval_rr_map.get(pb, effective_calib_rr)), 0.99)
                    else:
                        reuse_adj_map = getattr(self, "reuse_adj", None)
                        if reuse_adj_map is not None:
                            adj_rr = reuse_adj_map.get(pb, 1.0)
                            rr = min(effective_calib_rr * adj_rr, 0.99)
                        else:
                            rr = effective_calib_rr
                wants_reuse = bool(stack) and (rng.random() < rr)

                if wants_reuse:
                    # Effective fine PMF: prefer eval_phase_fine_adj (fitted from actual
                    # eval streams) over coarse phase_adj (BIT training ratios).
                    eval_fine_adj_map = getattr(self, "eval_phase_fine_adj", None)
                    phase_adj_map = getattr(self, "phase_adj", None)
                    if eval_fine_adj_map is not None:
                        fine_adj = eval_fine_adj_map.get(pb, np.ones(len(effective_fine_pmf)))
                        fine_pmf = effective_fine_pmf * fine_adj
                        fine_pmf = np.maximum(fine_pmf, 0.0)
                        fine_pmf = fine_pmf / fine_pmf.sum()
                    elif phase_adj_map is not None:
                        # Recompute FINE_TO_COARSE for corpus-specific edges
                        n_fine = len(effective_fine_pmf)
                        fine_to_coarse = np.zeros(n_fine, dtype=np.int64)
                        for _fi in range(n_fine):
                            _lo_f = int(effective_fine_edges[_fi])
                            _hi_f = int(effective_fine_edges[_fi + 1]) if _fi + 1 < len(effective_fine_edges) else int(effective_fine_edges[-1]) * 2
                            _best_b, _best_ov = 0, -1
                            for _b in range(N_BUCKETS):
                                _lo_b = int(BUCKET_EDGES[_b])
                                _hi_b = int(BUCKET_EDGES[_b + 1])
                                _ov = max(0, min(_hi_f, _hi_b) - max(_lo_f, _lo_b))
                                if _ov > _best_ov:
                                    _best_ov, _best_b = _ov, _b
                            fine_to_coarse[_fi] = _best_b
                        coarse_adj = phase_adj_map.get(pb, np.ones(N_BUCKETS))
                        fine_pmf = effective_fine_pmf * coarse_adj[fine_to_coarse]
                        fine_pmf = np.maximum(fine_pmf, 0.0)
                        fine_pmf = fine_pmf / fine_pmf.sum()
                    else:
                        fine_pmf = effective_fine_pmf / effective_fine_pmf.sum()
                    # IDEA #121: power transform on the fine PMF (LANL local_prob_power).
                    # power < 1 sharpens peaks; power > 1 flattens. Renormalize after.
                    if local_prob_power != 1.0 and local_prob_power > 0.0:
                        fine_pmf = np.power(np.maximum(fine_pmf, 0.0), local_prob_power)
                        _s = fine_pmf.sum()
                        if _s > 0:
                            fine_pmf = fine_pmf / _s
                        else:
                            fine_pmf = effective_fine_pmf / effective_fine_pmf.sum()
                    # IDEA #67: burst injection — redirect a reuse to the hot pool
                    # instead of sampling from the fine PMF. Preserves total reuse rate;
                    # shifts HRC mass from large ranks to small ranks to match working-set bursts.
                    if burst_prob > 0.0 and rng.random() < burst_prob:
                        pool_sz = min(burst_pool_size, len(stack))
                        rank = int(rng.integers(0, pool_sz))
                    else:
                        # Sample fine bin → uniform rank within it
                        fine_i = int(rng.choice(len(effective_fine_pmf), p=fine_pmf))
                        lo = int(effective_fine_edges[fine_i])
                        hi = int(effective_fine_edges[fine_i + 1]) - 1 if fine_i + 1 < len(effective_fine_edges) else int(effective_fine_edges[-1])
                        stack_sz = len(stack)
                        lo_eff = min(lo, stack_sz - 1)
                        hi_eff = min(hi, stack_sz - 1)
                        rank = int(rng.integers(lo_eff, hi_eff + 1))
                    # IDEA #121: per-phase rank multiplicative scaling (LANL
                    # stack_rank_phase_scales). Late-phase scales >1.0 push the
                    # rank deeper into the LRU stack, biasing toward less-recent objects.
                    if stack_rank_phase_scales is not None and len(stack_rank_phase_scales) > 0:
                        _scale_idx = min(pb, len(stack_rank_phase_scales) - 1)
                        _scale = float(stack_rank_phase_scales[_scale_idx])
                        if _scale != 1.0:
                            rank = int(min(round(rank * _scale), max(len(stack) - 1, 0)))
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                else:
                    obj_id = next_new_id
                    next_new_id += 1
                    stack.insert(0, obj_id)
                if len(stack) > max_stack_depth:
                    stack.pop()

                # Sample (dt, obj_size) mark from per-phase reservoir
                marks = self.mark_reservoir.get(pb) or self.global_marks
                if marks:
                    dt_ev, sz_ev = marks[int(rng.integers(0, len(marks)))]
                else:
                    dt_ev, sz_ev = 1.0, 4096.0

                ts += max(float(dt_ev), 0.0)
                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": max(int(round(sz_ev)), 1),
                    "opcode": 1,
                    "tenant": 0,
                })

                # Update tumbling phase window
                win_seen.add(int(obj_id))
                win_count += 1
                if win_count >= phase_window:
                    current_rate = len(win_seen) / phase_window
                    win_seen = set()
                    win_count = 0

        return rows[:n_records]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_fit(args):
    model = PhasePMFAtlas.fit(
        trace_dir=args.trace_dir,
        n_phase_bins=args.n_phase_bins,
        max_events_per_file=args.max_events,
        max_files=args.max_files,
        seed=args.seed,
        phase_window=args.phase_window,
    )
    model.save(args.output)
    print(f"Saved → {args.output}")


def cmd_generate(args):
    import pandas as pd

    print(f"Loading model from {args.model} ...")
    model = PhasePMFAtlas.load(args.model)
    print(f"Generating {args.n:,} events across {args.n_streams} streams ...")
    # IDEA #121: parse stack_rank_phase_scales as a comma-list float vector
    _srps = getattr(args, "stack_rank_phase_scales", None)
    _srps_list: Optional[List[float]] = None
    if _srps:
        _srps_list = [float(x) for x in _srps.split(",") if x.strip()]
    rows = model.generate(
        args.n,
        n_streams=args.n_streams,
        seed=args.seed,
        reuse_rate_override=args.reuse_rate,
        phase_window=args.phase_window,
        burst_prob=getattr(args, "burst_prob", 0.0),
        burst_pool_size=getattr(args, "burst_pool_size", 20),
        local_prob_power=getattr(args, "local_prob_power", 1.0),
        force_phase_schedule=bool(getattr(args, "force_phase_schedule", False)),
        stack_rank_phase_scales=_srps_list,
    )
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    reuse = float((df.groupby("stream_id")["obj_id"].apply(
        lambda x: x.duplicated().mean()
    ).mean()))
    print(f"Generated {len(df):,} events → {args.output}")
    print(f"Reuse rate (within-stream duplicates): {reuse:.4f}")


def cmd_calibrate_from_json(args):
    """Create or update a nophase atlas calibrated from an eval JSON file.

    If --model is given, loads that atlas and patches its corpus calibration.
    If not, creates a minimal nophase atlas (n_phase_bins=1, no real BIT fit).
    Saves result to --output.
    """
    import json as _json

    with open(args.eval_json) as _f:
        _d = _json.load(_f)
    _r = _d["real"]
    fine_pmf = np.array(_r["stack_distance_histogram"], dtype=np.float64)
    fine_pmf = fine_pmf / fine_pmf.sum()
    fine_edges = np.array(_r["stack_distance_bin_edges"], dtype=np.int64)
    calib_rr = float(_r["reuse_access_rate"])

    print(f"Loaded eval JSON: reuse_access_rate={calib_rr:.5f}, "
          f"fine_pmf bins={len(fine_pmf)}, p90_bin_edge={fine_edges[int(len(fine_pmf)*0.9)]}")

    if args.model:
        print(f"Loading existing atlas from {args.model} ...")
        model = PhasePMFAtlas.load(args.model)
    else:
        # Create minimal nophase atlas: 1 phase bin, empty BIT-fit stats.
        dummy_pmf = fine_pmf[:N_BUCKETS] if len(fine_pmf) >= N_BUCKETS else np.ones(N_BUCKETS) / N_BUCKETS
        model = PhasePMFAtlas(
            n_phase_bins=1,
            phase_edges=np.array([], dtype=np.float64),
            lru_pmf={0: dummy_pmf},
            reuse_rate={0: calib_rr},
            mark_reservoir={0: []},
            global_pmf=dummy_pmf,
            global_reuse_rate=calib_rr,
            global_marks=[],
            phase_adj=None,
            reuse_adj=None,
        )

    model.corpus_eval_fine_pmf = fine_pmf
    model.corpus_eval_fine_edges = fine_edges
    model.corpus_eval_calibrated_rr = calib_rr
    model.phase_adj = None  # disable BIT-phase conditioning; use global fine PMF only
    model.reuse_adj = None

    model.save(args.output)
    print(f"Saved calibrated atlas → {args.output}")
    print(f"  reuse_rate={calib_rr:.5f}, fine_pmf_bins={len(fine_pmf)}")


def main():
    p = argparse.ArgumentParser(description="LLNL Phase-PMF Atlas (IDEA #65)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pfit = sub.add_parser("fit", help="Fit atlas from trace directory")
    pfit.add_argument("--trace-dir", required=True)
    pfit.add_argument("--output", required=True)
    pfit.add_argument("--n-phase-bins", type=int, default=4)
    pfit.add_argument("--max-events", type=int, default=500_000)
    pfit.add_argument("--max-files", type=int, default=0)
    pfit.add_argument("--seed", type=int, default=7)
    pfit.add_argument("--phase-window", type=int, default=PHASE_WINDOW)

    pgen = sub.add_parser("generate", help="Generate traces from fitted atlas")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--n-streams", type=int, default=8)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--seed", type=int, default=42)
    pgen.add_argument("--reuse-rate", type=float, default=-1.0,
                      help="Reuse rate override; default uses per-phase fitted rate")
    pgen.add_argument("--phase-window", type=int, default=PHASE_WINDOW)
    pgen.add_argument("--burst-prob", type=float, default=0.0,
                      help="IDEA #67: probability of hot-pool burst replay per step")
    pgen.add_argument("--burst-pool-size", type=int, default=20,
                      help="IDEA #67: working-set pool size for burst injection")
    pgen.add_argument("--local-prob-power", type=float, default=1.0,
                      help="IDEA #121 (LANL knob port): power transform on per-phase fine PMF "
                           "(power<1 sharpens, >1 flattens). LANL-best for tencent: 0.8")
    pgen.add_argument("--force-phase-schedule", action="store_true",
                      help="IDEA #121 (LANL knob port): cycle phase deterministically by step "
                           "position instead of unique-rate window. LANL-best: True")
    pgen.add_argument("--stack-rank-phase-scales", type=str, default="",
                      help="IDEA #121 (LANL knob port): comma-list of per-phase rank multipliers "
                           "(e.g. '1.0,1.0,1.1,1.1' = scale ranks +10%% in late phases). "
                           "LANL-best for tencent: 1.0,1.0,1.1,1.1")

    pcal = sub.add_parser("calibrate-from-json",
                          help="Build/update nophase atlas from eval JSON calibration")
    pcal.add_argument("--eval-json", required=True,
                      help="Path to long_rollout eval JSON (contains real['stack_distance_histogram'])")
    pcal.add_argument("--model", default=None,
                      help="Existing atlas to patch (if omitted, creates minimal nophase atlas)")
    pcal.add_argument("--output", required=True)

    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "calibrate-from-json":
        cmd_calibrate_from_json(args)


if __name__ == "__main__":
    main()
