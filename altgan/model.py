"""StackAtlas: a cache-native alternative generator.

This is intentionally not a small variation on llgan.  It models the output
object process directly: new-object vs reuse, stack-distance bucket, and mark
sampling conditioned on a coarse regime Markov chain.  The goal is to beat the
current long-rollout failure mode where short-window scores look good while
generated object streams have positional IRD=1 and stack-distance=0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Tuple
import gzip
import pickle

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class EventSample:
    dt: float
    obj_size: float
    opcode: object
    tenant: object
    stride: int
    stack_distance: int
    action_class: int


@dataclass
class StackAtlasModel:
    """Stateful stack-distance generator with a regime transition atlas."""

    version: int
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

    ACTION_NEW = 0
    ACTION_NEAR = 1
    ACTION_MID = 2
    ACTION_FAR = 3
    N_ACTIONS = 4

    @classmethod
    def fit(
        cls,
        frames: Sequence["pd.DataFrame"],
        *,
        n_time_bins: int = 4,
        n_size_bins: int = 4,
        max_samples_per_state: int = 4096,
        seed: int = 7,
    ) -> "StackAtlasModel":
        if not frames:
            raise ValueError("StackAtlasModel.fit requires at least one frame")

        rng = np.random.default_rng(seed)
        clean = [_canonical_frame(df) for df in frames if len(df) > 1]
        if not clean:
            raise ValueError("no usable frames after canonicalization")

        all_dt = np.concatenate([_interarrival(df["ts"].to_numpy()) for df in clean])
        all_size = np.concatenate([
            np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
            for df in clean
        ])
        time_edges = _quantile_edges(np.log1p(all_dt), n_time_bins)
        size_edges = _quantile_edges(np.log(all_size), n_size_bins)

        initial_counts: Dict[int, int] = {}
        transition_counts: Dict[int, Dict[int, int]] = {}
        samples_by_state: Dict[int, List[EventSample]] = {}
        seen_by_state: Dict[int, int] = {}
        global_samples: List[EventSample] = []
        global_seen = 0
        max_obj_id = 0

        for df in clean:
            ts = df["ts"].to_numpy(dtype=np.float64)
            obj_ids = df["obj_id"].to_numpy(dtype=np.int64)
            sizes = np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
            dt = _interarrival(ts)
            stack_d = stack_distances(obj_ids)
            strides = np.diff(obj_ids, prepend=obj_ids[0]).astype(np.int64)

            time_bins = np.searchsorted(time_edges, np.log1p(dt), side="right")
            size_bins = np.searchsorted(size_edges, np.log(sizes), side="right")
            action_classes = np.array([_action_class(int(x)) for x in stack_d], dtype=np.int64)
            states = _state_id(time_bins, size_bins, action_classes, n_size_bins)

            if len(states):
                initial_counts[int(states[0])] = initial_counts.get(int(states[0]), 0) + 1
            for prev, nxt in zip(states[:-1], states[1:]):
                d = transition_counts.setdefault(int(prev), {})
                d[int(nxt)] = d.get(int(nxt), 0) + 1

            opcodes = df["opcode"].to_numpy(dtype=object)
            tenants = df["tenant"].to_numpy(dtype=object)
            for i, state in enumerate(states):
                ev = EventSample(
                    dt=float(dt[i]),
                    obj_size=float(sizes[i]),
                    opcode=opcodes[i],
                    tenant=tenants[i],
                    stride=int(strides[i]) if stack_d[i] < 0 else 0,
                    stack_distance=int(stack_d[i]),
                    action_class=int(action_classes[i]),
                )
                state_i = int(state)
                bucket = samples_by_state.setdefault(state_i, [])
                seen = seen_by_state.get(state_i, 0)
                _reservoir_add(bucket, ev, seen, max_samples_per_state, rng)
                seen_by_state[state_i] = seen + 1

                _reservoir_add(global_samples, ev, global_seen,
                               max_samples_per_state * 4, rng)
                global_seen += 1

            max_obj_id = max(max_obj_id, int(np.max(obj_ids)))

        initial_states, initial_probs = _normalize_counts(initial_counts)
        transitions = {
            state: _normalize_counts(counts)
            for state, counts in transition_counts.items()
        }
        metadata = {
            "n_frames": len(clean),
            "n_records": int(sum(len(df) for df in clean)),
            "n_states_with_samples": len(samples_by_state),
            "max_samples_per_state": max_samples_per_state,
            "seed": seed,
        }
        return cls(
            version=1,
            n_time_bins=n_time_bins,
            n_size_bins=n_size_bins,
            time_edges=time_edges,
            size_edges=size_edges,
            initial_states=initial_states,
            initial_probs=initial_probs,
            transitions=transitions,
            samples_by_state=samples_by_state,
            global_samples=global_samples,
            max_obj_id=max_obj_id,
            metadata=metadata,
        )

    def generate(self, n_records: int, *, n_streams: int = 1, seed: int = 42) -> "pd.DataFrame":
        if n_records <= 0:
            raise ValueError("n_records must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if not self.global_samples:
            raise ValueError("model has no samples")

        rng = np.random.default_rng(seed)
        per_stream = int(np.ceil(n_records / n_streams))
        rows = []

        for stream_id in range(n_streams):
            state = int(rng.choice(self.initial_states, p=self.initial_probs))
            stack: List[int] = []
            in_stack: set[int] = set()
            next_new_id = self.max_obj_id + 1 + stream_id * (per_stream + 1_000_003)
            prev_obj = next_new_id
            ts = 0.0

            for _ in range(per_stream):
                ev = self._sample_event(state, rng)
                wants_reuse = ev.action_class != self.ACTION_NEW
                if wants_reuse and stack:
                    rank = min(max(int(ev.stack_distance), 0), len(stack) - 1)
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                else:
                    obj_id, next_new_id = self._new_object_id(
                        prev_obj, next_new_id, int(ev.stride), in_stack
                    )
                    stack.insert(0, obj_id)
                    in_stack.add(obj_id)

                ts += max(float(ev.dt), 0.0)
                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": max(int(round(ev.obj_size)), 1),
                    "opcode": ev.opcode,
                    "tenant": ev.tenant,
                })
                prev_obj = obj_id
                state = self._next_state(state, rng)

        import pandas as pd
        return pd.DataFrame(rows[:n_records])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "StackAtlasModel":
        with gzip.open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a StackAtlasModel")
        return obj

    def _sample_event(self, state: int, rng: np.random.Generator) -> EventSample:
        samples = self.samples_by_state.get(int(state)) or self.global_samples
        return samples[int(rng.integers(0, len(samples)))]

    def _next_state(self, state: int, rng: np.random.Generator) -> int:
        states_probs = self.transitions.get(int(state))
        if states_probs is None:
            return int(rng.choice(self.initial_states, p=self.initial_probs))
        states, probs = states_probs
        return int(rng.choice(states, p=probs))

    @staticmethod
    def _new_object_id(prev_obj: int, next_new_id: int, stride: int,
                       in_stack: set[int]) -> Tuple[int, int]:
        candidate = prev_obj + stride if stride else next_new_id
        if candidate <= 0 or candidate in in_stack:
            candidate = next_new_id
        next_new_id = max(next_new_id + 1, candidate + 1)
        return int(candidate), int(next_new_id)


def stack_distances(obj_ids: np.ndarray) -> np.ndarray:
    """Return per-access stack distance, with -1 for cold misses."""
    n = len(obj_ids)
    out = np.full(n, -1, dtype=np.int64)
    bit = np.zeros(n + 2, dtype=np.int64)

    def update(idx: int, delta: int) -> None:
        idx += 1
        while idx <= n:
            bit[idx] += delta
            idx += idx & -idx

    def prefix(idx: int) -> int:
        idx += 1
        total = 0
        while idx > 0:
            total += int(bit[idx])
            idx -= idx & -idx
        return total

    last: Dict[int, int] = {}
    active = 0
    for i, raw in enumerate(obj_ids):
        key = int(raw)
        if key in last:
            prev = last[key]
            out[i] = active - prefix(prev)
            update(prev, -1)
            active -= 1
        last[key] = i
        update(i, 1)
        active += 1
    return out


def _canonical_frame(df: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    df = df.copy()
    rename = {}
    if "obj_size" not in df.columns and "size" in df.columns:
        rename["size"] = "obj_size"
    if "obj_id" not in df.columns and "offset" in df.columns:
        rename["offset"] = "obj_id"
    if rename:
        df.rename(columns=rename, inplace=True)

    for required in ("ts", "obj_id"):
        if required not in df.columns:
            raise ValueError(f"frame missing required column {required!r}")
    if "obj_size" not in df.columns:
        df["obj_size"] = 1
    if "opcode" not in df.columns:
        df["opcode"] = 0
    if "tenant" not in df.columns:
        df["tenant"] = 0

    keep = ["ts", "obj_id", "obj_size", "opcode", "tenant"]
    df = df[keep].copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0)
    df["obj_id"] = pd.to_numeric(df["obj_id"], errors="coerce").fillna(0).astype(np.int64)
    df["obj_size"] = pd.to_numeric(df["obj_size"], errors="coerce").fillna(1)
    df = df[df["obj_id"] > 0].reset_index(drop=True)
    return df


def _interarrival(ts: np.ndarray) -> np.ndarray:
    ts = ts.astype(np.float64)
    return np.clip(np.diff(ts, prepend=ts[0]), 0.0, None)


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    if n_bins <= 1:
        return np.array([], dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, qs)).astype(np.float64)


def _action_class(stack_distance: int) -> int:
    if stack_distance < 0:
        return StackAtlasModel.ACTION_NEW
    if stack_distance <= 4:
        return StackAtlasModel.ACTION_NEAR
    if stack_distance <= 64:
        return StackAtlasModel.ACTION_MID
    return StackAtlasModel.ACTION_FAR


def _state_id(time_bins: np.ndarray, size_bins: np.ndarray,
              action_classes: np.ndarray, n_size_bins: int) -> np.ndarray:
    return ((time_bins * n_size_bins + size_bins) * StackAtlasModel.N_ACTIONS
            + action_classes).astype(np.int64)


def _normalize_counts(counts: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if not counts:
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)
    states = np.array(sorted(counts), dtype=np.int64)
    weights = np.array([counts[int(s)] for s in states], dtype=np.float64)
    probs = weights / weights.sum()
    return states, probs


def _reservoir_add(bucket: List[EventSample], value: EventSample, seen: int,
                   cap: int, rng: np.random.Generator) -> None:
    if len(bucket) < cap:
        bucket.append(value)
        return
    j = int(rng.integers(0, seen + 1))
    if j < cap:
        bucket[j] = value


def frames_from_paths(paths: Iterable[Path], reader, records_per_file: int) -> List["pd.DataFrame"]:
    frames = []
    for path in paths:
        frames.append(reader(str(path), records_per_file))
    return frames
