"""
LLNL Markov compound-state atlas (Round 155 / IDEA #122 / option b).

Empirical-Markov port of LANL altgan/neural_atlas.py architecture without
the learned conditional transition net. BIT-counted (state, next_state)
transitions on training files = maximum-likelihood transition matrix.

State encoding: (time_bin × size_bin × action × phase_bin) → 4×4×3×4 = 192.

Strict-holdout fit + generate + eval pipeline:
    python -m llgan.markov_atlas fit \\
        --trace-dir /home/darrell/traces/tencent_block_1M \\
        --output /home/darrell/llnl_markov_atlas_tencent.pkl.gz \\
        --max-files 128 \\
        --exclude tencentBlock_19784,tencentBlock_2893,tencentBlock_20249,tencentBlock_22882

    python -m llgan.markov_atlas generate \\
        --model /home/darrell/llnl_markov_atlas_tencent.pkl.gz \\
        --n 100000 --n-streams 4 --seed 42 \\
        --output /home/darrell/markov_gen.csv

    python /home/darrell/eval_csv_hrc.py /home/darrell/markov_gen.csv \\
        /home/darrell/long_rollout_manifests/tencent_stackatlas.json
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from llgan.phase_pmf_atlas import _read_trace, BUCKET_EDGES, N_BUCKETS
except ModuleNotFoundError:
    from phase_pmf_atlas import _read_trace, BUCKET_EDGES, N_BUCKETS


# ---------------------------------------------------------------------------
# State-space constants
# ---------------------------------------------------------------------------
N_TIME_BINS = 1   # tencent ts is integer ticks; dt mostly clamps to floor — degenerate
N_SIZE_BINS = 1   # tencent block sizes cluster at 4-8KB — degenerate
N_ACTIONS = 3              # 0=NEW, 1=REUSE_NEAR, 2=REUSE_FAR
N_PHASE_BINS = 4
N_STATES = N_TIME_BINS * N_SIZE_BINS * N_ACTIONS * N_PHASE_BINS  # 1*1*3*4 = 12

REUSE_NEAR_RANK_CUTOFF = 64
PHASE_WINDOW = 200
TRANSITION_SMOOTHING = 1e-6


def _encode_state(time_bin: int, size_bin: int, action: int, phase_bin: int) -> int:
    return (
        ((time_bin * N_SIZE_BINS + size_bin) * N_ACTIONS + action) * N_PHASE_BINS
        + phase_bin
    )


def _bin_value(v: float, edges: np.ndarray, n_bins: int) -> int:
    """Map a value into [0, n_bins) using edge boundaries (right-open).
    `edges` must have length n_bins+1 with edges[0]=-inf-equivalent, edges[-1]=+inf."""
    b = int(np.searchsorted(edges, v, side="right")) - 1
    return max(0, min(b, n_bins - 1))


def _label_streams(
    obj_ids: np.ndarray,
    dts: np.ndarray,
    sizes: np.ndarray,
    dt_edges: np.ndarray,
    size_edges: np.ndarray,
    phase_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Walk a stream, label each step's (state, rank_bucket) using LRU stack.

    Returns:
        states: (T,) int array of compound state at each step
        rank_buckets: (T,) int array of LRU rank bucket if action != NEW,
                      else -1 for NEW steps.
    """
    T = len(obj_ids)
    states = np.empty(T, dtype=np.int64)
    rank_buckets = np.full(T, -1, dtype=np.int64)

    # LRU stack: most-recent at index 0 (deque-like semantics with list)
    stack: List[int] = []
    in_stack: Dict[int, int] = {}  # obj_id → current rank in stack

    # Phase tracking via tumbling unique-rate window
    win_unique: set = set()
    win_count = 0
    cur_phase_rate = 0.0

    for t in range(T):
        oid = int(obj_ids[t])
        dt = float(dts[t])
        sz = float(sizes[t])

        time_bin = _bin_value(dt, dt_edges, N_TIME_BINS)
        size_bin = _bin_value(sz, size_edges, N_SIZE_BINS)
        phase_bin = _bin_value(cur_phase_rate, phase_edges, N_PHASE_BINS)

        if oid in in_stack:
            rank = in_stack[oid]
            if rank < REUSE_NEAR_RANK_CUTOFF:
                action = 1  # REUSE_NEAR
            else:
                action = 2  # REUSE_FAR
            rb = int(np.searchsorted(BUCKET_EDGES, rank, side="right")) - 1
            rb = max(0, min(rb, N_BUCKETS - 1))
            rank_buckets[t] = rb
            # Move obj_id to position 0
            stack.pop(rank)
            stack.insert(0, oid)
            # Update positions for everything that shifted
            for i in range(rank + 1):
                in_stack[stack[i]] = i
        else:
            action = 0  # NEW
            stack.insert(0, oid)
            for i in range(len(stack)):
                in_stack[stack[i]] = i

        states[t] = _encode_state(time_bin, size_bin, action, phase_bin)

        # Update phase window
        if oid not in win_unique:
            win_unique.add(oid)
        win_count += 1
        if win_count >= PHASE_WINDOW:
            cur_phase_rate = len(win_unique) / float(win_count)
            win_unique.clear()
            win_count = 0

    return states, rank_buckets


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------
@dataclass
class MarkovAtlas:
    init_probs: np.ndarray = field(default_factory=lambda: np.zeros(N_STATES))
    transition_p: np.ndarray = field(default_factory=lambda: np.zeros((N_STATES, N_STATES)))
    rank_pmf: np.ndarray = field(default_factory=lambda: np.zeros((N_STATES, N_BUCKETS)))
    dt_pmf: np.ndarray = field(default_factory=lambda: np.zeros(N_TIME_BINS))
    dt_edges: np.ndarray = field(default_factory=lambda: np.zeros(N_TIME_BINS + 1))
    size_edges: np.ndarray = field(default_factory=lambda: np.zeros(N_SIZE_BINS + 1))
    phase_edges: np.ndarray = field(default_factory=lambda: np.zeros(N_PHASE_BINS + 1))
    n_train_files: int = 0
    n_train_records: int = 0
    seed: int = 7

    @classmethod
    def fit(
        cls,
        trace_dir: str,
        max_files: int = 128,
        max_records_per_file: int = 25000,
        seed: int = 7,
        exclude_files: Optional[List[str]] = None,
    ) -> "MarkovAtlas":
        rng = np.random.default_rng(seed)
        all_files = sorted([
            f for f in os.listdir(trace_dir) if f.startswith("tencent")
        ])
        exclude_set = set(exclude_files or [])
        available = [f for f in all_files if f not in exclude_set]
        n_pick = min(max_files, len(available))
        idx = rng.choice(len(available), n_pick, replace=False)
        sampled = sorted([available[i] for i in idx])
        print(f"[markov_atlas.fit] {n_pick} files (excluding {len(exclude_set)} eval files)")

        # Pass 1: collect dt and size samples to fit quartile edges
        dt_samples: List[float] = []
        size_samples: List[float] = []
        for fname in sampled[:8]:  # first 8 files for edges; sufficient
            path = os.path.join(trace_dir, fname)
            ts_prev = None
            count = 0
            for (ts, _oid, sz) in _read_trace(path):
                if ts_prev is not None:
                    dt_samples.append(max(ts - ts_prev, 1e-9))
                size_samples.append(sz)
                ts_prev = ts
                count += 1
                if count >= max_records_per_file:
                    break

        dt_arr = np.array(dt_samples, dtype=np.float64)
        size_arr = np.array(size_samples, dtype=np.float64)
        # Quartile edges: include -inf/+inf sentinels
        dt_qs = np.quantile(dt_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        size_qs = np.quantile(size_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        dt_edges = np.array([-np.inf, dt_qs[1], dt_qs[2], dt_qs[3], np.inf], dtype=np.float64)
        size_edges = np.array([-np.inf, size_qs[1], size_qs[2], size_qs[3], np.inf], dtype=np.float64)

        # Pre-compute phase_rate samples to fit quartile edges (otherwise phase is degenerate).
        phase_rate_samples: List[float] = []
        for fname in sampled[:8]:
            path = os.path.join(trace_dir, fname)
            win_unique: set = set()
            win_count = 0
            count = 0
            for (_ts, oid, _sz) in _read_trace(path):
                win_unique.add(oid)
                win_count += 1
                if win_count >= PHASE_WINDOW:
                    phase_rate_samples.append(len(win_unique) / float(win_count))
                    win_unique = set()
                    win_count = 0
                count += 1
                if count >= max_records_per_file:
                    break
        phase_rate_arr = np.array(phase_rate_samples, dtype=np.float64) if phase_rate_samples else np.array([0.5])
        ph_qs = np.quantile(phase_rate_arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        phase_edges = np.array([-np.inf, ph_qs[1], ph_qs[2], ph_qs[3], np.inf], dtype=np.float64)
        print(f"  dt edges: {dt_edges[1:-1]}  size edges: {size_edges[1:-1]}")

        # Pass 2: compute states, transitions, rank PMFs
        init_counts = np.zeros(N_STATES, dtype=np.int64)
        trans_counts = np.zeros((N_STATES, N_STATES), dtype=np.int64)
        rank_counts = np.zeros((N_STATES, N_BUCKETS), dtype=np.int64)
        dt_per_bin = [[] for _ in range(N_TIME_BINS)]
        size_per_bin = [[] for _ in range(N_SIZE_BINS)]
        n_total_records = 0

        for fi, fname in enumerate(sampled):
            path = os.path.join(trace_dir, fname)
            obj_ids, ts_arr, sz_arr = [], [], []
            for (ts, oid, sz) in _read_trace(path):
                obj_ids.append(oid)
                ts_arr.append(ts)
                sz_arr.append(sz)
                if len(obj_ids) >= max_records_per_file:
                    break
            obj_ids = np.array(obj_ids, dtype=np.int64)
            ts_arr = np.array(ts_arr, dtype=np.float64)
            sz_arr = np.array(sz_arr, dtype=np.float64)
            dts = np.empty_like(ts_arr)
            dts[0] = 1e-9
            dts[1:] = np.maximum(np.diff(ts_arr), 1e-9)
            for tb in range(N_TIME_BINS):
                lo = dt_edges[tb]
                hi = dt_edges[tb + 1]
                mask = (dts > lo) & (dts <= hi)
                dt_per_bin[tb].extend(dts[mask].tolist())
            for sb in range(N_SIZE_BINS):
                lo = size_edges[sb]
                hi = size_edges[sb + 1]
                mask = (sz_arr > lo) & (sz_arr <= hi)
                size_per_bin[sb].extend(sz_arr[mask].tolist())

            states, rank_buckets = _label_streams(
                obj_ids, dts, sz_arr, dt_edges, size_edges, phase_edges
            )
            n_total_records += len(states)
            init_counts[states[0]] += 1
            for t in range(len(states) - 1):
                trans_counts[states[t], states[t + 1]] += 1
            mask = rank_buckets >= 0
            for t in np.where(mask)[0]:
                rank_counts[states[t], rank_buckets[t]] += 1
            if (fi + 1) % 16 == 0:
                print(f"  fitted {fi + 1}/{len(sampled)} files")

        # Normalize
        init_probs = init_counts.astype(np.float64) + TRANSITION_SMOOTHING
        init_probs /= init_probs.sum()

        trans_p = trans_counts.astype(np.float64) + TRANSITION_SMOOTHING
        row_sums = trans_p.sum(axis=1, keepdims=True)
        trans_p /= np.maximum(row_sums, TRANSITION_SMOOTHING)

        rank_p = rank_counts.astype(np.float64) + TRANSITION_SMOOTHING
        rank_row_sums = rank_p.sum(axis=1, keepdims=True)
        rank_p /= np.maximum(rank_row_sums, TRANSITION_SMOOTHING)
        # For NEW-action states, no rank — zero out and normalize ones row
        for s in range(N_STATES):
            _, _, action, _ = (
                (s // N_PHASE_BINS // N_ACTIONS // N_SIZE_BINS),
                (s // N_PHASE_BINS // N_ACTIONS) % N_SIZE_BINS,
                (s // N_PHASE_BINS) % N_ACTIONS,
                s % N_PHASE_BINS,
            )
            if action == 0:
                rank_p[s] = 0.0  # NEW state has no rank distribution

        # dt_pmf marginal — used for sampling dt at gen time given time_bin
        dt_pmf = np.array([len(d) for d in dt_per_bin], dtype=np.float64)
        dt_pmf /= max(dt_pmf.sum(), 1.0)

        atlas = cls(
            init_probs=init_probs,
            transition_p=trans_p,
            rank_pmf=rank_p,
            dt_pmf=dt_pmf,
            dt_edges=dt_edges,
            size_edges=size_edges,
            phase_edges=phase_edges,
            n_train_files=len(sampled),
            n_train_records=n_total_records,
            seed=seed,
        )
        # Per-bin sample reservoirs for dt and size emission at gen time
        atlas._dt_reservoir = [
            np.array(d[:5000], dtype=np.float64) if d else np.array([1e-3])
            for d in dt_per_bin
        ]
        atlas._size_reservoir = [
            np.array(s[:5000], dtype=np.float64) if s else np.array([4096.0])
            for s in size_per_bin
        ]
        print(f"  atlas: n_records={n_total_records:,} states-with-mass="
              f"{int((init_probs > TRANSITION_SMOOTHING * 10).sum())}/{N_STATES}")
        return atlas

    def generate(
        self,
        n_records: int,
        n_streams: int = 4,
        seed: int = 42,
    ) -> "list[dict]":
        rng = np.random.default_rng(seed)
        per_stream = int(np.ceil(n_records / n_streams))
        rows = []

        # Reservoirs for dt/size sampling per bin
        dt_reservoir = getattr(self, "_dt_reservoir", None)
        size_reservoir = getattr(self, "_size_reservoir", None)
        if dt_reservoir is None or size_reservoir is None:
            # Persisted atlas without reservoirs → fall back to bin midpoints
            dt_reservoir = [
                np.array([(self.dt_edges[i] + self.dt_edges[i + 1]) / 2 if np.isfinite(self.dt_edges[i] + self.dt_edges[i + 1]) else 1e-3])
                for i in range(N_TIME_BINS)
            ]
            size_reservoir = [
                np.array([(self.size_edges[i] + self.size_edges[i + 1]) / 2 if np.isfinite(self.size_edges[i] + self.size_edges[i + 1]) else 4096.0])
                for i in range(N_SIZE_BINS)
            ]

        for stream_id in range(n_streams):
            stack: List[int] = []
            in_stack: Dict[int, int] = {}
            next_new_id = 10_000_000 + stream_id * (per_stream + 1_000_003)
            ts = 0.0
            state = int(rng.choice(N_STATES, p=self.init_probs))
            for step in range(per_stream):
                _t, _s, action, _p = (
                    (state // N_PHASE_BINS // N_ACTIONS) // N_SIZE_BINS,
                    (state // N_PHASE_BINS // N_ACTIONS) % N_SIZE_BINS,
                    (state // N_PHASE_BINS) % N_ACTIONS,
                    state % N_PHASE_BINS,
                )
                # Sample dt and size from per-bin reservoir
                dt = float(rng.choice(dt_reservoir[_t]))
                size_v = float(rng.choice(size_reservoir[_s]))
                ts += dt

                if action == 0 or not stack:
                    obj_id = next_new_id
                    next_new_id += 1
                    stack.insert(0, obj_id)
                    in_stack[obj_id] = 0
                    for i in range(min(len(stack), 64)):
                        in_stack[stack[i]] = i
                else:
                    bucket_p = self.rank_pmf[state]
                    if bucket_p.sum() <= 0:
                        rb = 0
                    else:
                        rb = int(rng.choice(N_BUCKETS, p=bucket_p))
                    lo = int(BUCKET_EDGES[rb])
                    hi = int(BUCKET_EDGES[rb + 1]) - 1
                    stack_sz = len(stack)
                    lo_eff = min(lo, stack_sz - 1)
                    hi_eff = min(hi, stack_sz - 1)
                    rank = int(rng.integers(lo_eff, hi_eff + 1))
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                    for i in range(rank + 1):
                        in_stack[stack[i]] = i

                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": obj_id,
                    "obj_size": size_v,
                })
                # Sample next state from transition matrix
                state = int(rng.choice(N_STATES, p=self.transition_p[state]))

        return rows

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "MarkovAtlas":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_fit(args: argparse.Namespace) -> None:
    exclude = [s.strip() for s in (args.exclude or "").split(",") if s.strip()]
    # Append .oracleGeneral.zst if user gave bare names
    exclude = [e if "." in e else f"{e}.oracleGeneral.zst" for e in exclude]
    atlas = MarkovAtlas.fit(
        trace_dir=args.trace_dir,
        max_files=args.max_files,
        max_records_per_file=args.max_records_per_file,
        seed=args.seed,
        exclude_files=exclude or None,
    )
    atlas.save(args.output)
    print(f"[markov_atlas.fit] saved → {args.output}")


def cmd_generate(args: argparse.Namespace) -> None:
    import pandas as pd
    atlas = MarkovAtlas.load(args.model)
    rows = atlas.generate(args.n, n_streams=args.n_streams, seed=args.seed)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    reuse = float((df.groupby("stream_id")["obj_id"].apply(
        lambda x: x.duplicated().mean()
    ).mean()))
    print(f"[markov_atlas.generate] {len(df):,} events → {args.output}  reuse={reuse:.4f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pfit = sub.add_parser("fit")
    pfit.add_argument("--trace-dir", required=True)
    pfit.add_argument("--output", required=True)
    pfit.add_argument("--max-files", type=int, default=128)
    pfit.add_argument("--max-records-per-file", type=int, default=25000)
    pfit.add_argument("--seed", type=int, default=7)
    pfit.add_argument("--exclude", default="")

    pgen = sub.add_parser("generate")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--n-streams", type=int, default=4)
    pgen.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    elif args.cmd == "generate":
        cmd_generate(args)
    else:
        p.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
