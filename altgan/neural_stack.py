"""Trainable profile-conditioned stack-distance generator.

NeuralStack keeps StackAtlas's explicit LRU object process, but replaces the
single global action/rank reservoir with a learned mapping:

    file/workload descriptor -> P(new/near/mid/far), P(stack-rank bucket)

The mark side remains empirical for now.  For each generated stream we select a
nearest training-file reservoir from the conditioning vector, then a small MLP
chooses the object action and stack-rank bucket.  This is deliberately modest:
the first goal is to prove that profile-conditioned object-state generation can
beat the peer's long-rollout locality, not to solve every mark dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import gzip
import pickle

import numpy as np

from .model import EventSample, StackAtlasModel, _action_class, _canonical_frame, _interarrival, stack_distances


RANK_EDGES = np.array(
    [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    dtype=np.int64,
)


@dataclass
class FileReservoir:
    name: str
    cond: np.ndarray
    samples_by_action: Dict[int, List[EventSample]]
    samples_by_rank_bin: Dict[int, List[EventSample]]
    max_obj_id: int


@dataclass
class NeuralStackModel:
    version: int
    cond_dim: int
    hidden_dim: int
    rank_edges: np.ndarray
    state_dict: dict
    file_names: List[str]
    file_conds: np.ndarray
    reservoirs: List[FileReservoir]
    global_by_action: Dict[int, List[EventSample]]
    global_by_rank_bin: Dict[int, List[EventSample]]
    max_obj_id: int
    metadata: dict = field(default_factory=dict)

    def generate(
        self,
        n_records: int,
        *,
        n_streams: int = 1,
        seed: int = 42,
        conds: np.ndarray | None = None,
        temperature: float = 1.0,
    ):
        import pandas as pd
        import torch

        if n_records <= 0:
            raise ValueError("n_records must be positive")
        if n_streams <= 0:
            raise ValueError("n_streams must be positive")
        rng = np.random.default_rng(seed)

        if conds is None:
            idx = rng.integers(0, len(self.file_conds), size=n_streams)
            conds = self.file_conds[idx]
        else:
            conds = np.asarray(conds, dtype=np.float32)
            if conds.ndim == 1:
                conds = np.repeat(conds[None, :], n_streams, axis=0)
            if len(conds) < n_streams:
                reps = int(np.ceil(n_streams / len(conds)))
                conds = np.tile(conds, (reps, 1))[:n_streams]
            conds = conds[:n_streams]

        net = _CondStackNet(self.cond_dim, self.hidden_dim, len(self.rank_edges))
        net.load_state_dict(self.state_dict)
        net.eval()
        with torch.no_grad():
            x = torch.tensor(conds, dtype=torch.float32)
            action_logits, rank_logits = net(x)
            action_probs = torch.softmax(action_logits / max(temperature, 1e-6), dim=-1).cpu().numpy()
            rank_probs = torch.softmax(rank_logits / max(temperature, 1e-6), dim=-1).cpu().numpy()

        per_stream = int(np.ceil(n_records / n_streams))
        rows = []
        for stream_id in range(n_streams):
            reservoir = self._nearest_reservoir(conds[stream_id])
            stack: List[int] = []
            in_stack: set[int] = set()
            next_new_id = self.max_obj_id + 1 + stream_id * (per_stream + 1_000_003)
            prev_obj = next_new_id
            ts = 0.0

            for _ in range(per_stream):
                action = int(rng.choice(StackAtlasModel.N_ACTIONS, p=action_probs[stream_id]))
                if action != StackAtlasModel.ACTION_NEW and stack:
                    rank_bin = int(rng.choice(len(self.rank_edges), p=rank_probs[stream_id]))
                    ev = self._sample_rank_event(reservoir, rank_bin, rng)
                    rank = min(max(int(ev.stack_distance), 0), len(stack) - 1)
                    obj_id = stack[rank]
                    del stack[rank]
                    stack.insert(0, obj_id)
                else:
                    ev = self._sample_action_event(reservoir, StackAtlasModel.ACTION_NEW, rng)
                    obj_id, next_new_id = _new_object_id(prev_obj, next_new_id, int(ev.stride), in_stack)
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
                prev_obj = int(obj_id)

        return pd.DataFrame(rows[:n_records])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "NeuralStackModel":
        with gzip.open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a NeuralStackModel")
        return obj

    def _nearest_reservoir(self, cond: np.ndarray) -> FileReservoir:
        d = ((self.file_conds - cond[None, :]) ** 2).sum(axis=1)
        return self.reservoirs[int(np.argmin(d))]

    def _sample_action_event(self, reservoir: FileReservoir, action: int,
                             rng: np.random.Generator) -> EventSample:
        samples = reservoir.samples_by_action.get(action) or self.global_by_action.get(action)
        if not samples:
            samples = self.global_by_action[StackAtlasModel.ACTION_NEW]
        return samples[int(rng.integers(0, len(samples)))]

    def _sample_rank_event(self, reservoir: FileReservoir, rank_bin: int,
                           rng: np.random.Generator) -> EventSample:
        samples = reservoir.samples_by_rank_bin.get(rank_bin) or self.global_by_rank_bin.get(rank_bin)
        if not samples:
            samples = self.global_by_action.get(StackAtlasModel.ACTION_MID) or self.global_by_action[StackAtlasModel.ACTION_NEW]
        return samples[int(rng.integers(0, len(samples)))]


class _CondStackNet:
    """Tiny MLP; defined without inheriting until torch is imported."""

    def __new__(cls, cond_dim: int, hidden_dim: int, n_rank_bins: int):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.trunk = nn.Sequential(
                    nn.Linear(cond_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.action = nn.Linear(hidden_dim, StackAtlasModel.N_ACTIONS)
                self.rank = nn.Linear(hidden_dim, n_rank_bins)

            def forward(self, x):
                h = self.trunk(x)
                return self.action(h), self.rank(h)

        return Net()


def fit_neural_stack(
    frames: Sequence["pd.DataFrame"],
    conds: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    hidden_dim: int = 64,
    epochs: int = 500,
    lr: float = 2e-3,
    max_samples_per_file: int = 4096,
    seed: int = 7,
) -> NeuralStackModel:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    clean = [_canonical_frame(df) for df in frames if len(df) > 1]
    if len(clean) != len(conds):
        raise ValueError("frames and conds must have the same length after filtering")

    reservoirs: List[FileReservoir] = []
    action_targets = []
    rank_targets = []
    file_conds = []
    file_names = []
    global_by_action: Dict[int, List[EventSample]] = {i: [] for i in range(StackAtlasModel.N_ACTIONS)}
    global_by_rank_bin: Dict[int, List[EventSample]] = {i: [] for i in range(len(RANK_EDGES))}
    max_obj_id = 0

    for df, cond, name in zip(clean, conds, names):
        reservoir, action_p, rank_p = _summarize_file(
            df, np.asarray(cond, dtype=np.float32), name,
            max_samples_per_file=max_samples_per_file,
            rng=rng,
        )
        reservoirs.append(reservoir)
        action_targets.append(action_p)
        rank_targets.append(rank_p)
        file_conds.append(reservoir.cond)
        file_names.append(name)
        max_obj_id = max(max_obj_id, reservoir.max_obj_id)
        for action, samples in reservoir.samples_by_action.items():
            global_by_action.setdefault(action, []).extend(samples)
        for rank_bin, samples in reservoir.samples_by_rank_bin.items():
            global_by_rank_bin.setdefault(rank_bin, []).extend(samples)

    cond_arr = np.vstack(file_conds).astype(np.float32)
    action_arr = np.vstack(action_targets).astype(np.float32)
    rank_arr = np.vstack(rank_targets).astype(np.float32)

    torch.manual_seed(seed)
    net = _CondStackNet(cond_arr.shape[1], hidden_dim, len(RANK_EDGES))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.tensor(cond_arr)
    y_action = torch.tensor(action_arr)
    y_rank = torch.tensor(rank_arr)

    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        action_logits, rank_logits = net(x)
        loss_action = F.kl_div(
            F.log_softmax(action_logits, dim=-1), y_action, reduction="batchmean"
        )
        loss_rank = F.kl_div(
            F.log_softmax(rank_logits, dim=-1), y_rank, reduction="batchmean"
        )
        loss = loss_action + loss_rank
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            losses.append(float(loss.detach().cpu()))

    metadata = {
        "n_files": len(file_names),
        "n_records": int(sum(len(df) for df in clean)),
        "epochs": epochs,
        "lr": lr,
        "loss_trace": losses,
        "seed": seed,
        "model": "NeuralStackModel",
    }
    return NeuralStackModel(
        version=1,
        cond_dim=cond_arr.shape[1],
        hidden_dim=hidden_dim,
        rank_edges=RANK_EDGES.copy(),
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        file_names=file_names,
        file_conds=cond_arr,
        reservoirs=reservoirs,
        global_by_action={k: _cap(v, max_samples_per_file * 8, rng) for k, v in global_by_action.items() if v},
        global_by_rank_bin={k: _cap(v, max_samples_per_file * 8, rng) for k, v in global_by_rank_bin.items() if v},
        max_obj_id=max_obj_id,
        metadata=metadata,
    )


def _summarize_file(df, cond: np.ndarray, name: str, *,
                    max_samples_per_file: int, rng: np.random.Generator):
    ts = df["ts"].to_numpy(dtype=np.float64)
    obj_ids = df["obj_id"].to_numpy(dtype=np.int64)
    sizes = np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
    dt = _interarrival(ts)
    stack_d = stack_distances(obj_ids)
    strides = np.diff(obj_ids, prepend=obj_ids[0]).astype(np.int64)
    actions = np.array([_action_class(int(x)) for x in stack_d], dtype=np.int64)
    rank_bins = np.array([_rank_bin(int(x)) if x >= 0 else 0 for x in stack_d], dtype=np.int64)
    opcodes = df["opcode"].to_numpy(dtype=object)
    tenants = df["tenant"].to_numpy(dtype=object)

    samples_by_action: Dict[int, List[EventSample]] = {i: [] for i in range(StackAtlasModel.N_ACTIONS)}
    samples_by_rank_bin: Dict[int, List[EventSample]] = {i: [] for i in range(len(RANK_EDGES))}
    seen_action = {i: 0 for i in range(StackAtlasModel.N_ACTIONS)}
    seen_rank = {i: 0 for i in range(len(RANK_EDGES))}

    for i, action in enumerate(actions):
        ev = EventSample(
            dt=float(dt[i]),
            obj_size=float(sizes[i]),
            opcode=opcodes[i],
            tenant=tenants[i],
            stride=int(strides[i]) if stack_d[i] < 0 else 0,
            stack_distance=int(stack_d[i]),
            action_class=int(action),
        )
        action_i = int(action)
        _reservoir_add(samples_by_action[action_i], ev, seen_action[action_i],
                       max_samples_per_file, rng)
        seen_action[action_i] += 1
        if stack_d[i] >= 0:
            rb = int(rank_bins[i])
            _reservoir_add(samples_by_rank_bin[rb], ev, seen_rank[rb],
                           max_samples_per_file, rng)
            seen_rank[rb] += 1

    action_counts = np.bincount(actions, minlength=StackAtlasModel.N_ACTIONS).astype(np.float64)
    action_probs = _smooth(action_counts)
    reuse_bins = rank_bins[stack_d >= 0]
    rank_counts = np.bincount(reuse_bins, minlength=len(RANK_EDGES)).astype(np.float64)
    rank_probs = _smooth(rank_counts)
    reservoir = FileReservoir(
        name=name,
        cond=cond,
        samples_by_action={k: v for k, v in samples_by_action.items() if v},
        samples_by_rank_bin={k: v for k, v in samples_by_rank_bin.items() if v},
        max_obj_id=int(np.max(obj_ids)),
    )
    return reservoir, action_probs, rank_probs


def _rank_bin(stack_distance: int) -> int:
    return int(np.searchsorted(RANK_EDGES, max(stack_distance, 0), side="right") - 1)


def _smooth(counts: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    out = counts.astype(np.float64) + alpha
    return out / out.sum()


def _reservoir_add(bucket: List[EventSample], value: EventSample, seen: int,
                   cap: int, rng: np.random.Generator) -> None:
    if len(bucket) < cap:
        bucket.append(value)
        return
    j = int(rng.integers(0, seen + 1))
    if j < cap:
        bucket[j] = value


def _cap(values: List[EventSample], cap: int, rng: np.random.Generator) -> List[EventSample]:
    if len(values) <= cap:
        return values
    idx = rng.choice(len(values), size=cap, replace=False)
    return [values[int(i)] for i in idx]


def _new_object_id(prev_obj: int, next_new_id: int, stride: int,
                   in_stack: set[int]) -> Tuple[int, int]:
    candidate = prev_obj + stride if stride else next_new_id
    if candidate <= 0 or candidate in in_stack:
        candidate = next_new_id
    next_new_id = max(next_new_id + 1, candidate + 1)
    return int(candidate), int(next_new_id)
