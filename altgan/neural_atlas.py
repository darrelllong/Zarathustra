"""Profile-conditioned transition atlas with explicit LRU object decoding.

NeuralAtlas is the second altgan challenger.  It keeps StackAtlas's cache-native
decoder, but trains the stream-profile -> state-transition law instead of
sampling one global transition atlas.  That matters because the first
NeuralStack model learned only per-file action/rank marginals; it could match
some reuse totals, but it lost the temporal transition structure that makes
Tencent's HRC shape work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import gzip
import pickle

import numpy as np

from .model import (
    EventSample,
    StackAtlasModel,
    _action_class,
    _canonical_frame,
    _interarrival,
    _quantile_edges,
    _reservoir_add,
    _state_id,
    stack_distances,
)


@dataclass
class AtlasReservoir:
    name: str
    cond: np.ndarray
    initial_states: np.ndarray
    initial_probs: np.ndarray
    transitions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    samples_by_state: Dict[int, List[EventSample]]
    max_obj_id: int


@dataclass
class NeuralAtlasModel:
    version: int
    cond_dim: int
    hidden_dim: int
    n_time_bins: int
    n_size_bins: int
    n_phase_bins: int
    n_states: int
    time_edges: np.ndarray
    size_edges: np.ndarray
    state_dict: dict
    file_names: List[str]
    file_conds: np.ndarray
    reservoirs: List[AtlasReservoir]
    global_initial_states: np.ndarray
    global_initial_probs: np.ndarray
    global_transitions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    global_samples_by_state: Dict[int, List[EventSample]]
    global_samples: List[EventSample]
    max_obj_id: int
    mark_model: object | None = None
    metadata: dict = field(default_factory=dict)

    def generate(
        self,
        n_records: int,
        *,
        n_streams: int = 1,
        seed: int = 42,
        conds: np.ndarray | None = None,
        temperature: float = 1.0,
        transition_blend: float = 0.75,
        local_prob_power: float = 1.0,
        force_phase_schedule: bool = False,
        stack_rank_scale: float = 1.0,
        stack_rank_max: int | None = None,
        stack_rank_phase_scales: Sequence[float] | None = None,
        stack_rank_phase_maxes: Sequence[int] | None = None,
        mark_temperature: float | None = None,
        mark_numeric_noise: float = 0.05,
        mark_numeric_blend: float = 1.0,
        mark_numeric_blend_space: str = "raw",
        mark_numeric_fields: str = "both",
        mark_categorical_source: str = "neural",
    ):
        import pandas as pd
        import torch

        if n_records <= 0:
            raise ValueError("n_records must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        if not self.global_samples:
            raise ValueError("model has no samples")

        rng = np.random.default_rng(seed)
        conds = self._resolve_conds(conds, n_streams, rng)
        transition_blend = float(np.clip(transition_blend, 0.0, 1.0))
        local_prob_power = max(float(local_prob_power), 0.0)
        stack_rank_scale = max(float(stack_rank_scale), 0.0)
        mark_numeric_blend = float(np.clip(mark_numeric_blend, 0.0, 1.0))
        if mark_numeric_blend_space not in {"raw", "log"}:
            raise ValueError("mark_numeric_blend_space must be 'raw' or 'log'")
        if mark_numeric_fields not in {"both", "dt", "size"}:
            raise ValueError("mark_numeric_fields must be 'both', 'dt', or 'size'")
        if mark_categorical_source not in {"neural", "reservoir"}:
            raise ValueError("mark_categorical_source must be 'neural' or 'reservoir'")
        mark_runtime = None
        mark_model = getattr(self, "mark_model", None)
        if mark_model is not None:
            mark_runtime = mark_model.runtime(
                n_streams=n_streams,
                seed=seed + 17,
                temperature=temperature if mark_temperature is None else mark_temperature,
                numeric_noise=mark_numeric_noise,
            )

        net = _CondTransitionNet(self.cond_dim, self.hidden_dim, self.n_states)
        net.load_state_dict(self.state_dict)
        net.eval()
        with torch.no_grad():
            cond_t = torch.tensor(conds, dtype=torch.float32)
            init_logits = net.forward_init(cond_t)
            init_probs = torch.softmax(init_logits / max(temperature, 1e-6), dim=-1).cpu().numpy()

            cond_rep = np.repeat(conds, self.n_states, axis=0)
            state_rep = np.tile(np.arange(self.n_states, dtype=np.int64), n_streams)
            trans_logits = net.forward_trans(
                torch.tensor(cond_rep, dtype=torch.float32),
                torch.tensor(state_rep, dtype=torch.long),
            )
            trans_probs = torch.softmax(trans_logits / max(temperature, 1e-6), dim=-1)
            trans_probs = trans_probs.cpu().numpy().reshape(n_streams, self.n_states, self.n_states)

        per_stream = int(np.ceil(n_records / n_streams))
        rows = []
        base_span = self.n_time_bins * self.n_size_bins * StackAtlasModel.N_ACTIONS
        for stream_id in range(n_streams):
            reservoir = self._nearest_reservoir(conds[stream_id])
            init_p = self._blend_initial(
                init_probs[stream_id],
                reservoir,
                transition_blend,
                local_prob_power,
            )
            state = int(rng.choice(self.n_states, p=init_p))
            stack: List[int] = []
            in_stack: set[int] = set()
            next_new_id = self.max_obj_id + 1 + stream_id * (per_stream + 1_000_003)
            prev_obj = next_new_id
            ts = 0.0

            for pos in range(per_stream):
                phase = min((pos * self.n_phase_bins) // per_stream, self.n_phase_bins - 1)
                if force_phase_schedule and self.n_phase_bins > 1:
                    state = _state_with_phase(state, phase, base_span)
                ev = self._sample_event(reservoir, state, rng)
                wants_reuse = ev.action_class != StackAtlasModel.ACTION_NEW
                if wants_reuse and stack:
                    phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, stack_rank_scale)
                    phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                    if phase_rank_max is not None and phase_rank_max < 0:
                        phase_rank_max = None
                    rank = _calibrated_stack_rank(
                        ev.stack_distance,
                        stack_rank_scale=phase_rank_scale,
                        stack_rank_max=phase_rank_max,
                        stack_len=len(stack),
                    )
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                else:
                    obj_id, next_new_id = StackAtlasModel._new_object_id(
                        prev_obj, next_new_id, int(ev.stride), in_stack
                    )
                    stack.insert(0, obj_id)
                    in_stack.add(obj_id)

                mark = ev
                if mark_runtime is not None:
                    neural_mark = mark_runtime.sample(
                        stream_id=stream_id,
                        cond=conds[stream_id],
                        state=state,
                        action_class=ev.action_class,
                        stack_distance=ev.stack_distance,
                        stride=ev.stride,
                    )
                    if (
                        mark_numeric_blend >= 1.0
                        and mark_numeric_fields == "both"
                        and mark_categorical_source == "neural"
                    ):
                        mark = neural_mark
                    else:
                        dt, obj_size = _blend_numeric_marks(
                            ev,
                            neural_mark,
                            mark_numeric_blend,
                            mark_numeric_blend_space,
                            mark_numeric_fields,
                        )
                        mark = EventSample(
                            dt=float(dt),
                            obj_size=float(obj_size),
                            opcode=(
                                neural_mark.opcode
                                if mark_categorical_source == "neural"
                                else ev.opcode
                            ),
                            tenant=(
                                neural_mark.tenant
                                if mark_categorical_source == "neural"
                                else ev.tenant
                            ),
                            stride=int(ev.stride),
                            stack_distance=int(ev.stack_distance),
                            action_class=int(ev.action_class),
                        )

                ts += max(float(mark.dt), 0.0)
                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": max(int(round(mark.obj_size)), 1),
                    "opcode": mark.opcode,
                    "tenant": mark.tenant,
                })
                prev_obj = int(obj_id)
                state = self._next_state(
                    reservoir,
                    state,
                    trans_probs[stream_id, state],
                    transition_blend,
                    local_prob_power,
                    rng,
                )

        return pd.DataFrame(rows[:n_records])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralAtlasModel":
        with gzip.open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a NeuralAtlasModel")
        return obj

    def _resolve_conds(self, conds: np.ndarray | None, n_streams: int,
                       rng: np.random.Generator) -> np.ndarray:
        if conds is None:
            idx = rng.integers(0, len(self.file_conds), size=n_streams)
            return self.file_conds[idx]
        arr = np.asarray(conds, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.repeat(arr[None, :], n_streams, axis=0)
        if len(arr) < n_streams:
            reps = int(np.ceil(n_streams / len(arr)))
            arr = np.tile(arr, (reps, 1))[:n_streams]
        return arr[:n_streams]

    def _nearest_reservoir(self, cond: np.ndarray) -> AtlasReservoir:
        d = ((self.file_conds - cond[None, :]) ** 2).sum(axis=1)
        return self.reservoirs[int(np.argmin(d))]

    def _sample_event(self, reservoir: AtlasReservoir, state: int,
                      rng: np.random.Generator) -> EventSample:
        samples = (
            reservoir.samples_by_state.get(int(state))
            or self.global_samples_by_state.get(int(state))
            or self.global_samples
        )
        return samples[int(rng.integers(0, len(samples)))]

    def _blend_initial(self, neural_p: np.ndarray, reservoir: AtlasReservoir,
                       transition_blend: float, local_prob_power: float) -> np.ndarray:
        local_p = np.zeros(self.n_states, dtype=np.float64)
        local_p[reservoir.initial_states] = reservoir.initial_probs
        if local_p.sum() <= 0:
            local_p[self.global_initial_states] = self.global_initial_probs
        local_p = _power_probs(local_p, local_prob_power)
        p = transition_blend * neural_p.astype(np.float64) + (1.0 - transition_blend) * local_p
        return _renorm(p)

    def _next_state(self, reservoir: AtlasReservoir, state: int, neural_p: np.ndarray,
                    transition_blend: float, local_prob_power: float,
                    rng: np.random.Generator) -> int:
        local_p = np.zeros(self.n_states, dtype=np.float64)
        states_probs = reservoir.transitions.get(int(state)) or self.global_transitions.get(int(state))
        if states_probs is not None:
            states, probs = states_probs
            local_p[states] = probs
        else:
            local_p[self.global_initial_states] = self.global_initial_probs
        local_p = _power_probs(local_p, local_prob_power)
        p = transition_blend * neural_p.astype(np.float64) + (1.0 - transition_blend) * local_p
        return int(rng.choice(self.n_states, p=_renorm(p)))


class _CondTransitionNet:
    """Tiny conditional transition model; torch is imported lazily."""

    def __new__(cls, cond_dim: int, hidden_dim: int, n_states: int):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cond = nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.state = nn.Embedding(n_states, hidden_dim)
                self.init = nn.Linear(hidden_dim, n_states)
                self.trans = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, n_states),
                )

            def forward_init(self, cond):
                return self.init(self.cond(cond))

            def forward_trans(self, cond, state):
                h = self.cond(cond)
                s = self.state(state)
                return self.trans(torch.cat([h, s], dim=-1))

        return Net()


def fit_neural_atlas(
    frames: Sequence["pd.DataFrame"],
    conds: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    n_time_bins: int = 4,
    n_size_bins: int = 4,
    n_phase_bins: int = 1,
    hidden_dim: int = 96,
    epochs: int = 800,
    lr: float = 2e-3,
    max_samples_per_state: int = 1024,
    seed: int = 7,
) -> NeuralAtlasModel:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    clean = [_canonical_frame(df) for df in frames if len(df) > 1]
    if len(clean) != len(conds):
        raise ValueError("frames and conds must have the same length after filtering")
    if not clean:
        raise ValueError("no usable frames")

    all_dt = np.concatenate([_interarrival(df["ts"].to_numpy()) for df in clean])
    all_size = np.concatenate([
        np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
        for df in clean
    ])
    time_edges = _quantile_edges(np.log1p(all_dt), n_time_bins)
    size_edges = _quantile_edges(np.log(all_size), n_size_bins)
    n_phase_bins = max(int(n_phase_bins), 1)
    n_states = n_phase_bins * n_time_bins * n_size_bins * StackAtlasModel.N_ACTIONS

    reservoirs: List[AtlasReservoir] = []
    file_conds = []
    file_names = []
    init_targets = []
    trans_conds = []
    trans_states = []
    trans_targets = []
    trans_weights = []
    global_initial_counts: Dict[int, int] = {}
    global_transition_counts: Dict[int, Dict[int, int]] = {}
    global_samples_by_state: Dict[int, List[EventSample]] = {}
    global_seen_by_state: Dict[int, int] = {}
    global_samples: List[EventSample] = []
    global_seen = 0
    max_obj_id = 0

    for df, cond, name in zip(clean, conds, names):
        reservoir, init_counts, transition_counts = _summarize_file(
            df,
            np.asarray(cond, dtype=np.float32),
            name,
            time_edges=time_edges,
            size_edges=size_edges,
            n_time_bins=n_time_bins,
            n_size_bins=n_size_bins,
            n_phase_bins=n_phase_bins,
            n_states=n_states,
            max_samples_per_state=max_samples_per_state,
            rng=rng,
        )
        reservoirs.append(reservoir)
        file_conds.append(reservoir.cond)
        file_names.append(name)
        max_obj_id = max(max_obj_id, reservoir.max_obj_id)
        init_targets.append(_dense_counts(init_counts, n_states))

        for state, counts in transition_counts.items():
            total = sum(counts.values())
            if total <= 0:
                continue
            trans_conds.append(reservoir.cond)
            trans_states.append(int(state))
            trans_targets.append(_dense_counts(counts, n_states))
            trans_weights.append(np.log1p(float(total)))
            g = global_transition_counts.setdefault(int(state), {})
            for nxt, count in counts.items():
                g[int(nxt)] = g.get(int(nxt), 0) + int(count)

        for state, count in init_counts.items():
            global_initial_counts[int(state)] = global_initial_counts.get(int(state), 0) + int(count)
        for state, samples in reservoir.samples_by_state.items():
            bucket = global_samples_by_state.setdefault(int(state), [])
            seen = global_seen_by_state.get(int(state), 0)
            for ev in samples:
                _reservoir_add(bucket, ev, seen, max_samples_per_state * 8, rng)
                seen += 1
                _reservoir_add(global_samples, ev, global_seen, max_samples_per_state * 16, rng)
                global_seen += 1
            global_seen_by_state[int(state)] = seen

    cond_arr = np.vstack(file_conds).astype(np.float32)
    init_arr = np.vstack(init_targets).astype(np.float32)
    trans_cond_arr = np.vstack(trans_conds).astype(np.float32)
    trans_state_arr = np.array(trans_states, dtype=np.int64)
    trans_arr = np.vstack(trans_targets).astype(np.float32)
    weight_arr = np.asarray(trans_weights, dtype=np.float32)
    weight_arr = weight_arr / max(float(weight_arr.mean()), 1e-6)

    torch.manual_seed(seed)
    net = _CondTransitionNet(cond_arr.shape[1], hidden_dim, n_states)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    x_init = torch.tensor(cond_arr)
    y_init = torch.tensor(init_arr)
    x_trans = torch.tensor(trans_cond_arr)
    state_trans = torch.tensor(trans_state_arr, dtype=torch.long)
    y_trans = torch.tensor(trans_arr)
    w_trans = torch.tensor(weight_arr)

    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        init_logits = net.forward_init(x_init)
        trans_logits = net.forward_trans(x_trans, state_trans)
        loss_init = F.kl_div(
            F.log_softmax(init_logits, dim=-1), y_init, reduction="batchmean"
        )
        per_row = F.kl_div(
            F.log_softmax(trans_logits, dim=-1), y_trans, reduction="none"
        ).sum(dim=1)
        loss_trans = (per_row * w_trans).mean()
        loss = loss_init + loss_trans
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            losses.append(float(loss.detach().cpu()))

    global_initial_states, global_initial_probs = _normalize_counts(global_initial_counts)
    global_transitions = {
        state: _normalize_counts(counts)
        for state, counts in global_transition_counts.items()
    }
    metadata = {
        "n_files": len(file_names),
        "n_records": int(sum(len(df) for df in clean)),
        "epochs": epochs,
        "lr": lr,
        "loss_trace": losses,
        "seed": seed,
        "model": "NeuralAtlasModel",
        "n_transition_rows": int(len(trans_states)),
        "n_phase_bins": n_phase_bins,
    }
    return NeuralAtlasModel(
        version=1,
        cond_dim=cond_arr.shape[1],
        hidden_dim=hidden_dim,
        n_time_bins=n_time_bins,
        n_size_bins=n_size_bins,
        n_phase_bins=n_phase_bins,
        n_states=n_states,
        time_edges=time_edges,
        size_edges=size_edges,
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        file_names=file_names,
        file_conds=cond_arr,
        reservoirs=reservoirs,
        global_initial_states=global_initial_states,
        global_initial_probs=global_initial_probs,
        global_transitions=global_transitions,
        global_samples_by_state=global_samples_by_state,
        global_samples=global_samples,
        max_obj_id=max_obj_id,
        metadata=metadata,
    )


def _summarize_file(df, cond: np.ndarray, name: str, *, time_edges: np.ndarray,
                    size_edges: np.ndarray, n_time_bins: int, n_size_bins: int,
                    n_phase_bins: int, n_states: int,
                    max_samples_per_state: int, rng: np.random.Generator):
    ts = df["ts"].to_numpy(dtype=np.float64)
    obj_ids = df["obj_id"].to_numpy(dtype=np.int64)
    sizes = np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
    dt = _interarrival(ts)
    stack_d = stack_distances(obj_ids)
    strides = np.diff(obj_ids, prepend=obj_ids[0]).astype(np.int64)
    time_bins = np.searchsorted(time_edges, np.log1p(dt), side="right")
    size_bins = np.searchsorted(size_edges, np.log(sizes), side="right")
    phase_bins = _phase_bins(len(df), n_phase_bins)
    actions = np.array([_action_class(int(x)) for x in stack_d], dtype=np.int64)
    base_states = _state_id(time_bins, size_bins, actions, n_size_bins)
    states = (phase_bins * (n_time_bins * n_size_bins * StackAtlasModel.N_ACTIONS)
              + base_states).astype(np.int64)
    opcodes = df["opcode"].to_numpy(dtype=object)
    tenants = df["tenant"].to_numpy(dtype=object)

    initial_counts: Dict[int, int] = {}
    transition_counts: Dict[int, Dict[int, int]] = {}
    samples_by_state: Dict[int, List[EventSample]] = {}
    seen_by_state: Dict[int, int] = {}

    if len(states):
        initial_counts[int(states[0])] = 1
    for prev, nxt in zip(states[:-1], states[1:]):
        d = transition_counts.setdefault(int(prev), {})
        d[int(nxt)] = d.get(int(nxt), 0) + 1

    for i, state in enumerate(states):
        ev = EventSample(
            dt=float(dt[i]),
            obj_size=float(sizes[i]),
            opcode=opcodes[i],
            tenant=tenants[i],
            stride=int(strides[i]) if stack_d[i] < 0 else 0,
            stack_distance=int(stack_d[i]),
            action_class=int(actions[i]),
        )
        state_i = int(state)
        if state_i < 0 or state_i >= n_states:
            continue
        bucket = samples_by_state.setdefault(state_i, [])
        seen = seen_by_state.get(state_i, 0)
        _reservoir_add(bucket, ev, seen, max_samples_per_state, rng)
        seen_by_state[state_i] = seen + 1

    initial_states, initial_probs = _normalize_counts(initial_counts)
    transitions = {
        state: _normalize_counts(counts)
        for state, counts in transition_counts.items()
    }
    reservoir = AtlasReservoir(
        name=name,
        cond=cond,
        initial_states=initial_states,
        initial_probs=initial_probs,
        transitions=transitions,
        samples_by_state=samples_by_state,
        max_obj_id=int(np.max(obj_ids)),
    )
    return reservoir, initial_counts, transition_counts


def _dense_counts(counts: Dict[int, int], n_states: int, alpha: float = 1e-4) -> np.ndarray:
    out = np.full(n_states, alpha, dtype=np.float64)
    for state, count in counts.items():
        if 0 <= int(state) < n_states:
            out[int(state)] += float(count)
    return out / out.sum()


def _phase_bins(n_rows: int, n_phase_bins: int) -> np.ndarray:
    if n_phase_bins <= 1 or n_rows <= 0:
        return np.zeros(max(n_rows, 0), dtype=np.int64)
    raw = np.floor(np.arange(n_rows, dtype=np.float64) * n_phase_bins / n_rows).astype(np.int64)
    return np.clip(raw, 0, n_phase_bins - 1)


def _state_with_phase(state: int, phase: int, base_span: int) -> int:
    return int(phase) * int(base_span) + (int(state) % int(base_span))


def _calibrated_stack_rank(
    raw_rank: int,
    *,
    stack_rank_scale: float,
    stack_rank_max: int | None,
    stack_len: int,
) -> int:
    rank = int(round(max(int(raw_rank), 0) * stack_rank_scale))
    if stack_rank_max is not None and stack_rank_max >= 0:
        rank = min(rank, int(stack_rank_max))
    return min(max(rank, 0), int(stack_len) - 1)


def _power_probs(probs: np.ndarray, power: float) -> np.ndarray:
    arr = probs.astype(np.float64, copy=True)
    if power == 1.0:
        return arr
    support = arr > 0
    if not np.any(support):
        return arr
    arr[support] = np.power(arr[support], power)
    arr[~support] = 0.0
    return _renorm(arr)


def _phase_value(values: Sequence, phase: int, default):
    if not values:
        return default
    idx = min(max(int(phase), 0), len(values) - 1)
    return values[idx]


def _blend_numeric_marks(
    reservoir_mark: EventSample,
    neural_mark: EventSample,
    blend: float,
    space: str,
    fields: str,
) -> tuple[float, float]:
    r_dt = max(float(reservoir_mark.dt), 0.0)
    n_dt = max(float(neural_mark.dt), 0.0)
    r_size = max(float(reservoir_mark.obj_size), 1.0)
    n_size = max(float(neural_mark.obj_size), 1.0)
    blend_dt = blend if fields in {"both", "dt"} else 0.0
    blend_size = blend if fields in {"both", "size"} else 0.0
    if space == "log":
        dt_log = (1.0 - blend_dt) * np.log1p(r_dt) + blend_dt * np.log1p(n_dt)
        size_log = (1.0 - blend_size) * np.log(r_size) + blend_size * np.log(n_size)
        return max(float(np.expm1(dt_log)), 0.0), max(float(np.exp(size_log)), 1.0)
    return (
        (1.0 - blend_dt) * r_dt + blend_dt * n_dt,
        (1.0 - blend_size) * r_size + blend_size * n_size,
    )


def _normalize_counts(counts: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if not counts:
        return np.array([0], dtype=np.int64), np.array([1.0], dtype=np.float64)
    states = np.array(sorted(counts), dtype=np.int64)
    weights = np.array([counts[int(s)] for s in states], dtype=np.float64)
    return states, weights / weights.sum()


def _renorm(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=np.float64)
    total = float(p.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(len(p), 1.0 / len(p), dtype=np.float64)
    return p / total
