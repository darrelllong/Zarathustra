"""
IDEA #64: LLNL StackAtlas — per-object Markov state generator for HRC parity.

Independently implements the same algorithm as LANL's altgan/model.py
StackAtlasModel, reading oracle_general .zst trace files directly.

State = (time_bin, size_bin, action_class) where:
  time_bin    = log(dt+1) quantile bucket (dt = interarrival of any event)
  size_bin    = log(obj_size) quantile bucket
  action_class: NEW=0 (cold miss), NEAR=1 (rank≤4), MID=2 (rank≤64), FAR=3

Markov chain over states; reservoir of EventSamples per state for mark sampling.
At generation time, steps through chain, samples event from reservoir[state],
uses ev.stack_distance to position obj_id in LRU stack.

Usage
-----
    # Fit model from training traces (saves model.pkl.gz)
    python stack_atlas.py fit \
        --trace-dir /tiamat/zarathustra/traces/alibaba \
        --output /home/darrell/llnl_stack_atlas.pkl.gz

    # Generate and evaluate
    python stack_atlas.py generate \
        --model /home/darrell/llnl_stack_atlas.pkl.gz \
        --n 100000 --n-streams 8 \
        --output /home/darrell/llnl_atlas_gen.csv
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
# Action classes (matching LANL's StackAtlasModel)
# ---------------------------------------------------------------------------
ACTION_NEW  = 0
ACTION_NEAR = 1   # stack_distance <= 4
ACTION_MID  = 2   # stack_distance <= 64
ACTION_FAR  = 3   # stack_distance > 64
N_ACTIONS   = 4


def _action_class(stack_distance: int) -> int:
    if stack_distance < 0:
        return ACTION_NEW
    if stack_distance <= 4:
        return ACTION_NEAR
    if stack_distance <= 64:
        return ACTION_MID
    return ACTION_FAR


def _state_id(time_bin: int, size_bin: int, action_class: int,
              n_size_bins: int) -> int:
    return (time_bin * n_size_bins + size_bin) * N_ACTIONS + action_class


# ---------------------------------------------------------------------------
# Event sample (what we store in the reservoir)
# ---------------------------------------------------------------------------
@dataclass
class EventSample:
    dt: float
    obj_size: float
    opcode: int
    tenant: int
    stack_distance: int
    action_class: int


# ---------------------------------------------------------------------------
# Reservoir sampling (Vitter Algorithm R)
# ---------------------------------------------------------------------------
def _reservoir_add(
    bucket: List[EventSample],
    ev: EventSample,
    seen: int,
    max_size: int,
    rng: np.random.Generator,
) -> None:
    if len(bucket) < max_size:
        bucket.append(ev)
    else:
        j = int(rng.integers(0, seen + 1))
        if j < max_size:
            bucket[j] = ev


# ---------------------------------------------------------------------------
# Trace reading
# ---------------------------------------------------------------------------
def _read_oracle_general(path: str, max_records: int = 500_000):
    """Read (ts, obj_id, obj_size, opcode, tenant) from oracle_general .zst."""
    n_bytes = max_records * 24
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    else:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)

    dt = np.dtype([
        ("ts",       "<u4"),
        ("obj_id",   "<u8"),
        ("obj_size", "<u4"),
        ("vtime",    "<i4"),
        ("op",       "<i2"),
        ("tenant",   "<i2"),
    ])
    n = len(raw) // 24
    if n == 0:
        return None
    arr = np.frombuffer(raw[:n * 24], dtype=dt)
    return arr


# ---------------------------------------------------------------------------
# Stack distances (BIT-based, exact)
# ---------------------------------------------------------------------------
def _stack_distances_bit(obj_ids: np.ndarray) -> np.ndarray:
    """Return per-event stack distance, -1 for cold misses. O(N log N)."""
    N = len(obj_ids)
    out = np.full(N, -1, dtype=np.int64)
    bit = [0] * (N + 2)

    def update(i, d):
        i += 1
        while i <= N:
            bit[i] += d
            i += i & -i

    def prefix(i):
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    last: Dict[int, int] = {}
    active = 0
    for i, oid in enumerate(obj_ids.tolist()):
        oid = int(oid)
        if oid in last:
            prev = last[oid]
            out[i] = active - prefix(prev)
            update(prev, -1)
            active -= 1
        last[oid] = i
        update(i, 1)
        active += 1
    return out


# ---------------------------------------------------------------------------
# StackAtlas model
# ---------------------------------------------------------------------------
@dataclass
class LLNLStackAtlas:
    n_time_bins: int
    n_size_bins: int
    time_edges: np.ndarray
    size_edges: np.ndarray
    initial_states: np.ndarray
    initial_probs: np.ndarray
    transitions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    samples_by_state: Dict[int, List[EventSample]]
    global_samples: List[EventSample]
    max_obj_id: int
    metadata: dict = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        trace_dir: str,
        n_time_bins: int = 4,
        n_size_bins: int = 4,
        max_events_per_file: int = 500_000,
        max_samples_per_state: int = 4096,
        max_files: int = 0,
        seed: int = 7,
        verbose: bool = True,
    ) -> "LLNLStackAtlas":
        rng = np.random.default_rng(seed)

        files = sorted(glob.glob(os.path.join(trace_dir, "*.oracleGeneral.zst")))
        if not files:
            files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
        if max_files > 0:
            files = files[:max_files]

        print(f"Fitting LLNL StackAtlas from {len(files)} files...")

        # Pass 1: collect all dt and obj_size samples to compute quantile edges
        all_log_dt = []
        all_log_size = []
        for fi, fpath in enumerate(files):
            arr = _read_oracle_general(fpath, max_events_per_file)
            if arr is None or len(arr) < 2:
                continue
            ts = arr["ts"].astype(np.float64)
            sizes = np.maximum(arr["obj_size"].astype(np.float64), 1.0)
            dt_arr = np.diff(ts, prepend=ts[0])
            dt_arr = np.clip(dt_arr, 0, None)
            all_log_dt.append(np.log1p(dt_arr))
            all_log_size.append(np.log(sizes))
            if verbose and fi % 40 == 0:
                print(f"  pass1 [{fi+1}/{len(files)}]", flush=True)

        all_log_dt_flat = np.concatenate(all_log_dt)
        all_log_size_flat = np.concatenate(all_log_size)

        def quantile_edges(arr, n_bins):
            if n_bins <= 1:
                return np.array([], dtype=np.float64)
            qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
            return np.unique(np.quantile(arr, qs)).astype(np.float64)

        time_edges = quantile_edges(all_log_dt_flat, n_time_bins)
        size_edges = quantile_edges(all_log_size_flat, n_size_bins)
        print(f"time_edges (log(dt+1)): {np.round(time_edges, 3)}")
        print(f"size_edges (log(size)): {np.round(size_edges, 3)}")

        # Pass 2: accumulate state transitions and reservoir samples
        initial_counts: Dict[int, int] = {}
        transition_counts: Dict[int, Dict[int, int]] = {}
        samples_by_state: Dict[int, List[EventSample]] = {}
        seen_by_state: Dict[int, int] = {}
        global_samples: List[EventSample] = []
        global_seen = 0
        max_obj_id = 0
        total_events = 0

        for fi, fpath in enumerate(files):
            arr = _read_oracle_general(fpath, max_events_per_file)
            if arr is None or len(arr) < 2:
                continue

            ts_arr = arr["ts"].astype(np.float64)
            obj_ids = arr["obj_id"].astype(np.int64)
            sizes = np.maximum(arr["obj_size"].astype(np.float64), 1.0)
            opcodes = arr["op"].astype(np.int32)
            tenants = arr["tenant"].astype(np.int32)

            dt_arr = np.diff(ts_arr, prepend=ts_arr[0])
            dt_arr = np.clip(dt_arr, 0, None)
            log_dt = np.log1p(dt_arr)
            log_size = np.log(sizes)

            stack_d = _stack_distances_bit(obj_ids)
            time_bins = np.searchsorted(time_edges, log_dt, side="right")
            size_bins = np.searchsorted(size_edges, log_size, side="right")
            action_classes = np.array([_action_class(int(x)) for x in stack_d],
                                      dtype=np.int64)

            states = np.array([
                _state_id(int(t), int(s), int(a), n_size_bins)
                for t, s, a in zip(time_bins, size_bins, action_classes)
            ], dtype=np.int64)

            if len(states):
                initial_counts[int(states[0])] = (
                    initial_counts.get(int(states[0]), 0) + 1
                )

            for prev_s, nxt_s in zip(states[:-1], states[1:]):
                d = transition_counts.setdefault(int(prev_s), {})
                d[int(nxt_s)] = d.get(int(nxt_s), 0) + 1

            for i in range(len(states)):
                ev = EventSample(
                    dt=float(dt_arr[i]),
                    obj_size=float(sizes[i]),
                    opcode=int(opcodes[i]),
                    tenant=int(tenants[i]),
                    stack_distance=int(stack_d[i]),
                    action_class=int(action_classes[i]),
                )
                sid = int(states[i])
                bucket = samples_by_state.setdefault(sid, [])
                seen = seen_by_state.get(sid, 0)
                _reservoir_add(bucket, ev, seen, max_samples_per_state, rng)
                seen_by_state[sid] = seen + 1
                _reservoir_add(global_samples, ev, global_seen,
                               max_samples_per_state * 4, rng)
                global_seen += 1

            max_obj_id = max(max_obj_id, int(np.max(obj_ids)))
            total_events += len(arr)

            if verbose and fi % 40 == 0:
                print(f"  pass2 [{fi+1}/{len(files)}]  "
                      f"states={len(samples_by_state)}  "
                      f"events={total_events:,}", flush=True)

        def normalize_counts(counts):
            if not counts:
                return np.array([0], np.int64), np.array([1.0], np.float64)
            states_arr = np.array(sorted(counts), dtype=np.int64)
            weights = np.array([float(counts[int(s)]) for s in states_arr])
            probs = weights / weights.sum()
            return states_arr, probs

        init_states, init_probs = normalize_counts(initial_counts)
        transitions = {
            s: normalize_counts(c)
            for s, c in transition_counts.items()
        }

        metadata = {
            "n_files": len(files),
            "n_events": total_events,
            "n_states": len(samples_by_state),
            "seed": seed,
        }
        print(f"Atlas fitted: {len(samples_by_state)} states, "
              f"{total_events:,} events, {len(transitions)} transitions")
        return cls(
            n_time_bins=n_time_bins,
            n_size_bins=n_size_bins,
            time_edges=time_edges,
            size_edges=size_edges,
            initial_states=init_states,
            initial_probs=init_probs,
            transitions=transitions,
            samples_by_state=samples_by_state,
            global_samples=global_samples,
            max_obj_id=max_obj_id,
            metadata=metadata,
        )

    def generate(
        self,
        n_records: int,
        n_streams: int = 1,
        seed: int = 42,
    ) -> "list[dict]":
        rng = np.random.default_rng(seed)
        per_stream = int(np.ceil(n_records / n_streams))
        rows = []

        for stream_id in range(n_streams):
            state = int(rng.choice(self.initial_states, p=self.initial_probs))
            stack: List[int] = []
            in_stack: set[int] = set()
            next_new_id = self.max_obj_id + 1 + stream_id * (per_stream + 1_000_003)
            ts = 0.0

            for _ in range(per_stream):
                ev = self._sample_event(state, rng)
                wants_reuse = ev.action_class != ACTION_NEW
                if wants_reuse and stack:
                    rank = min(max(int(ev.stack_distance), 0), len(stack) - 1)
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                else:
                    obj_id = next_new_id
                    next_new_id += 1
                    stack.insert(0, obj_id)
                    in_stack.add(obj_id)
                if len(stack) > 4096:
                    stack.pop()

                ts += max(float(ev.dt), 0.0)
                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": max(int(round(ev.obj_size)), 1),
                    "opcode": ev.opcode,
                    "tenant": ev.tenant,
                })
                state = self._next_state(state, rng)

        return rows[:n_records]

    def _sample_event(self, state: int, rng: np.random.Generator) -> EventSample:
        bucket = self.samples_by_state.get(state) or self.global_samples
        return bucket[int(rng.integers(0, len(bucket)))]

    def _next_state(self, state: int, rng: np.random.Generator) -> int:
        trans = self.transitions.get(state)
        if trans is None:
            return int(rng.choice(self.initial_states, p=self.initial_probs))
        states, probs = trans
        return int(rng.choice(states, p=probs))

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "LLNLStackAtlas":
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cmd_fit(args):
    model = LLNLStackAtlas.fit(
        trace_dir=args.trace_dir,
        n_time_bins=args.n_time_bins,
        n_size_bins=args.n_size_bins,
        max_events_per_file=args.max_events,
        max_files=args.max_files,
        seed=args.seed,
    )
    model.save(args.output)
    print(f"Saved → {args.output}")


def cmd_generate(args):
    import pandas as pd

    print(f"Loading model from {args.model} ...")
    model = LLNLStackAtlas.load(args.model)
    print(f"Generating {args.n:,} events across {args.n_streams} streams ...")
    rows = model.generate(args.n, n_streams=args.n_streams, seed=args.seed)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    reuse = float((df.groupby("stream_id")["obj_id"].apply(
        lambda x: x.duplicated().mean()
    ).mean()))
    print(f"Generated {len(df):,} events → {args.output}")
    print(f"Reuse rate (within-stream duplicates): {reuse:.4f}")


def main():
    p = argparse.ArgumentParser(description="LLNL StackAtlas (IDEA #64)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pfit = sub.add_parser("fit", help="Fit atlas from trace directory")
    pfit.add_argument("--trace-dir", required=True)
    pfit.add_argument("--output", required=True)
    pfit.add_argument("--n-time-bins", type=int, default=4)
    pfit.add_argument("--n-size-bins", type=int, default=4)
    pfit.add_argument("--max-events", type=int, default=500_000)
    pfit.add_argument("--max-files", type=int, default=0)
    pfit.add_argument("--seed", type=int, default=7)

    pgen = sub.add_parser("generate", help="Generate traces from fitted atlas")
    pgen.add_argument("--model", required=True)
    pgen.add_argument("--n", type=int, default=100_000)
    pgen.add_argument("--n-streams", type=int, default=8)
    pgen.add_argument("--output", required=True)
    pgen.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    if args.cmd == "fit":
        cmd_fit(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
