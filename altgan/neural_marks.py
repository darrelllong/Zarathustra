"""Autoregressive mark head for NeuralAtlas/PhaseAtlas object processes.

The atlas family already generates the cache-relevant object stream explicitly.
This sidecar learns the non-object marks around that stream: inter-arrival time,
size, opcode, and tenant.  It is intentionally optional so old atlas checkpoints
remain valid and the object-law champion can be frozen while marks improve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .model import EventSample, StackAtlasModel, _action_class, _canonical_frame, _interarrival
from .model import _state_id, stack_distances
from .neural_atlas import NeuralAtlasModel, _phase_bins


@dataclass
class NeuralMarkHead:
    version: int
    cond_dim: int
    hidden_dim: int
    n_states: int
    n_rank_buckets: int
    state_dict: dict
    dt_mean: float
    dt_std: float
    size_mean: float
    size_std: float
    opcodes: list[Any]
    tenants: list[Any]
    default_opcode: int
    default_tenant: int
    metadata: dict = field(default_factory=dict)

    def runtime(
        self,
        *,
        n_streams: int,
        seed: int = 42,
        temperature: float = 1.0,
        numeric_noise: float = 0.05,
    ) -> "NeuralMarkRuntime":
        return NeuralMarkRuntime(
            self,
            n_streams=n_streams,
            seed=seed,
            temperature=temperature,
            numeric_noise=numeric_noise,
        )


class NeuralMarkRuntime:
    def __init__(
        self,
        model: NeuralMarkHead,
        *,
        n_streams: int,
        seed: int,
        temperature: float,
        numeric_noise: float,
    ):
        import torch

        self.model = model
        self.rng = np.random.default_rng(seed)
        self.temperature = max(float(temperature), 1e-6)
        self.numeric_noise = max(float(numeric_noise), 0.0)
        self.net = _MarkHeadNet(
            model.cond_dim,
            model.hidden_dim,
            model.n_states,
            model.n_rank_buckets,
            len(model.opcodes),
            len(model.tenants),
        )
        self.net.load_state_dict(model.state_dict)
        self.net.eval()
        self.torch = torch
        self.hidden = [None for _ in range(n_streams)]
        self.prev_num = np.zeros((n_streams, 2), dtype=np.float32)
        self.prev_opcode = np.full(n_streams, model.default_opcode, dtype=np.int64)
        self.prev_tenant = np.full(n_streams, model.default_tenant, dtype=np.int64)

    def sample(
        self,
        *,
        stream_id: int,
        cond: np.ndarray,
        state: int,
        action_class: int,
        stack_distance: int,
        stride: int,
    ) -> EventSample:
        torch = self.torch
        sid = int(stream_id)
        cond_arr = np.asarray(cond, dtype=np.float32)[: self.model.cond_dim]
        if len(cond_arr) < self.model.cond_dim:
            cond_arr = np.pad(cond_arr, (0, self.model.cond_dim - len(cond_arr)))

        with torch.no_grad():
            out, hidden = self.net(
                torch.tensor(cond_arr[None, None, :], dtype=torch.float32),
                torch.tensor([[np.clip(int(state), 0, self.model.n_states - 1)]], dtype=torch.long),
                torch.tensor([[np.clip(int(action_class), 0, StackAtlasModel.N_ACTIONS - 1)]],
                             dtype=torch.long),
                torch.tensor([[rank_bucket(stack_distance)]], dtype=torch.long),
                torch.tensor(self.prev_num[sid][None, None, :], dtype=torch.float32),
                torch.tensor([[int(self.prev_opcode[sid])]], dtype=torch.long),
                torch.tensor([[int(self.prev_tenant[sid])]], dtype=torch.long),
                self.hidden[sid],
            )
        self.hidden[sid] = hidden
        dt_z = float(out["dt_z"][0, 0, 0])
        size_z = float(out["size_z"][0, 0, 0])
        if self.numeric_noise:
            dt_z += float(self.rng.normal(0.0, self.numeric_noise))
            size_z += float(self.rng.normal(0.0, self.numeric_noise))

        op_idx = _sample_logits(out["opcode_logits"][0, 0].detach().cpu().numpy(),
                                self.temperature, self.rng)
        tenant_idx = _sample_logits(out["tenant_logits"][0, 0].detach().cpu().numpy(),
                                    self.temperature, self.rng)

        self.prev_num[sid] = np.array([dt_z, size_z], dtype=np.float32)
        self.prev_opcode[sid] = op_idx
        self.prev_tenant[sid] = tenant_idx

        dt_log = dt_z * self.model.dt_std + self.model.dt_mean
        size_log = size_z * self.model.size_std + self.model.size_mean
        return EventSample(
            dt=max(float(np.expm1(dt_log)), 0.0),
            obj_size=max(float(np.exp(size_log)), 1.0),
            opcode=self.model.opcodes[op_idx],
            tenant=self.model.tenants[tenant_idx],
            stride=int(stride),
            stack_distance=int(stack_distance),
            action_class=int(action_class),
        )


def fit_mark_head(
    base: NeuralAtlasModel,
    frames: Sequence[Any],
    conds: Sequence[np.ndarray],
    *,
    hidden_dim: int = 128,
    epochs: int = 20,
    steps_per_epoch: int = 400,
    batch_size: int = 64,
    window_len: int = 128,
    lr: float = 1e-3,
    seed: int = 7,
    device: str = "auto",
    progress_every: int = 1,
) -> NeuralMarkHead:
    import torch
    import torch.nn.functional as F

    if device == "auto":
        run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        run_device = torch.device(device)
    rng = np.random.default_rng(seed)
    raw = [_mark_sequence(base, df, cond) for df, cond in zip(frames, conds) if len(df) > 1]
    raw = [seq for seq in raw if len(seq["states"]) > 1]
    if not raw:
        raise ValueError("no usable mark sequences")

    dt_values = np.concatenate([seq["dt_log"] for seq in raw])
    size_values = np.concatenate([seq["size_log"] for seq in raw])
    dt_mean, dt_std = _mean_std(dt_values)
    size_mean, size_std = _mean_std(size_values)
    opcodes = _ordered_values(np.concatenate([seq["opcode"] for seq in raw]))
    tenants = _ordered_values(np.concatenate([seq["tenant"] for seq in raw]))
    opcode_to_idx = {str(v): i for i, v in enumerate(opcodes)}
    tenant_to_idx = {str(v): i for i, v in enumerate(tenants)}
    seqs = []
    for seq in raw:
        seqs.append({
            "cond": seq["cond"],
            "states": seq["states"],
            "actions": seq["actions"],
            "ranks": seq["ranks"],
            "dt_z": ((seq["dt_log"] - dt_mean) / dt_std).astype(np.float32),
            "size_z": ((seq["size_log"] - size_mean) / size_std).astype(np.float32),
            "opcode": np.array([opcode_to_idx[str(x)] for x in seq["opcode"]], dtype=np.int64),
            "tenant": np.array([tenant_to_idx[str(x)] for x in seq["tenant"]], dtype=np.int64),
        })

    actual_window = min(int(window_len), max(len(seq["states"]) for seq in seqs))
    actual_window = max(actual_window, 2)
    eligible = [seq for seq in seqs if len(seq["states"]) >= actual_window]
    if not eligible:
        raise ValueError("no mark sequence is long enough for the requested window")

    torch.manual_seed(seed)
    net = _MarkHeadNet(
        base.cond_dim,
        hidden_dim,
        base.n_states,
        _N_RANK_BUCKETS,
        len(opcodes),
        len(tenants),
    )
    net.to(run_device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    for epoch in range(int(epochs)):
        net.train()
        epoch_loss = 0.0
        for _ in range(int(steps_per_epoch)):
            batch = _to_device(_sample_batch(eligible, actual_window, batch_size, rng), run_device)
            opt.zero_grad()
            out, _ = net(
                batch["cond"],
                batch["state"],
                batch["action"],
                batch["rank"],
                batch["prev_num"],
                batch["prev_opcode"],
                batch["prev_tenant"],
                None,
            )
            loss_num = F.mse_loss(out["dt_z"].squeeze(-1), batch["dt_z"])
            loss_num = loss_num + F.mse_loss(out["size_z"].squeeze(-1), batch["size_z"])
            loss_cat = F.cross_entropy(
                out["opcode_logits"].reshape(-1, len(opcodes)),
                batch["opcode"].reshape(-1),
            )
            loss_cat = loss_cat + F.cross_entropy(
                out["tenant_logits"].reshape(-1, len(tenants)),
                batch["tenant"].reshape(-1),
            )
            loss = loss_num + 0.5 * loss_cat
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach().cpu())
        avg_loss = epoch_loss / max(int(steps_per_epoch), 1)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            losses.append(avg_loss)
        if progress_every and ((epoch + 1) % max(int(progress_every), 1) == 0):
            print(
                "[altgan.neural_marks] "
                f"epoch={epoch + 1}/{int(epochs)} loss={avg_loss:.6f} device={run_device}",
                flush=True,
            )

    return NeuralMarkHead(
        version=1,
        cond_dim=base.cond_dim,
        hidden_dim=hidden_dim,
        n_states=base.n_states,
        n_rank_buckets=_N_RANK_BUCKETS,
        state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
        dt_mean=dt_mean,
        dt_std=dt_std,
        size_mean=size_mean,
        size_std=size_std,
        opcodes=opcodes,
        tenants=tenants,
        default_opcode=0,
        default_tenant=0,
        metadata={
            "model": "NeuralMarkHead",
            "n_sequences": len(seqs),
            "n_records": int(sum(len(seq["states"]) for seq in seqs)),
            "epochs": int(epochs),
            "steps_per_epoch": int(steps_per_epoch),
            "batch_size": int(batch_size),
            "window_len": int(actual_window),
            "lr": float(lr),
            "seed": int(seed),
            "device": str(run_device),
            "loss_trace": losses,
        },
    )


class _MarkHeadNet:
    def __new__(
        cls,
        cond_dim: int,
        hidden_dim: int,
        n_states: int,
        n_rank_buckets: int,
        n_opcodes: int,
        n_tenants: int,
    ):
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
                self.action = nn.Embedding(StackAtlasModel.N_ACTIONS, hidden_dim)
                self.rank = nn.Embedding(n_rank_buckets, hidden_dim)
                self.opcode = nn.Embedding(n_opcodes, hidden_dim)
                self.tenant = nn.Embedding(n_tenants, hidden_dim)
                self.gru = nn.GRU(hidden_dim * 6 + 2, hidden_dim, batch_first=True)
                self.dt = nn.Linear(hidden_dim, 1)
                self.size = nn.Linear(hidden_dim, 1)
                self.opcode_out = nn.Linear(hidden_dim, n_opcodes)
                self.tenant_out = nn.Linear(hidden_dim, n_tenants)

            def forward(self, cond, state, action, rank, prev_num, prev_opcode, prev_tenant, hidden):
                cond_h = self.cond(cond)
                x = torch.cat([
                    cond_h,
                    self.state(state),
                    self.action(action),
                    self.rank(rank),
                    prev_num,
                    self.opcode(prev_opcode),
                    self.tenant(prev_tenant),
                ], dim=-1)
                y, next_hidden = self.gru(x, hidden)
                return {
                    "dt_z": self.dt(y),
                    "size_z": self.size(y),
                    "opcode_logits": self.opcode_out(y),
                    "tenant_logits": self.tenant_out(y),
                }, next_hidden

        return Net()


def rank_bucket(stack_distance: int) -> int:
    d = int(stack_distance)
    if d < 0:
        return 0
    if d <= 4:
        return 1
    if d <= 16:
        return 2
    if d <= 64:
        return 3
    if d <= 256:
        return 4
    if d <= 1024:
        return 5
    return 6


def _mark_sequence(base: NeuralAtlasModel, df: Any, cond: np.ndarray) -> dict[str, np.ndarray]:
    clean = _canonical_frame(df)
    ts = clean["ts"].to_numpy(dtype=np.float64)
    obj_ids = clean["obj_id"].to_numpy(dtype=np.int64)
    sizes = np.maximum(clean["obj_size"].to_numpy(dtype=np.float64), 1.0)
    dt = _interarrival(ts)
    stack_d = stack_distances(obj_ids)
    time_bins = np.searchsorted(base.time_edges, np.log1p(dt), side="right")
    size_bins = np.searchsorted(base.size_edges, np.log(sizes), side="right")
    phase_bins = _phase_bins(len(clean), getattr(base, "n_phase_bins", 1))
    actions = np.array([_action_class(int(x)) for x in stack_d], dtype=np.int64)
    base_states = _state_id(time_bins, size_bins, actions, base.n_size_bins)
    base_span = base.n_time_bins * base.n_size_bins * StackAtlasModel.N_ACTIONS
    states = (phase_bins * base_span + base_states).astype(np.int64)
    return {
        "cond": _fixed_cond(cond, base.cond_dim),
        "states": states,
        "actions": actions,
        "ranks": np.array([rank_bucket(int(x)) for x in stack_d], dtype=np.int64),
        "dt_log": np.log1p(np.maximum(dt, 0.0)).astype(np.float32),
        "size_log": np.log(sizes).astype(np.float32),
        "opcode": clean["opcode"].to_numpy(dtype=object),
        "tenant": clean["tenant"].to_numpy(dtype=object),
    }


def _sample_batch(seqs: Sequence[dict[str, np.ndarray]], length: int, batch_size: int,
                  rng: np.random.Generator) -> dict[str, Any]:
    import torch

    batch = {
        "cond": [],
        "state": [],
        "action": [],
        "rank": [],
        "prev_num": [],
        "prev_opcode": [],
        "prev_tenant": [],
        "dt_z": [],
        "size_z": [],
        "opcode": [],
        "tenant": [],
    }
    for _ in range(int(batch_size)):
        seq = seqs[int(rng.integers(0, len(seqs)))]
        start = int(rng.integers(0, len(seq["states"]) - length + 1))
        end = start + length
        prev_slice = slice(max(start - 1, 0), max(end - 1, 0))
        prev_dt = seq["dt_z"][prev_slice].copy()
        prev_size = seq["size_z"][prev_slice].copy()
        prev_opcode = seq["opcode"][prev_slice].copy()
        prev_tenant = seq["tenant"][prev_slice].copy()
        if start == 0:
            prev_dt = np.concatenate([np.array([0.0], dtype=np.float32), prev_dt])
            prev_size = np.concatenate([np.array([0.0], dtype=np.float32), prev_size])
            prev_opcode = np.concatenate([np.array([0], dtype=np.int64), prev_opcode])
            prev_tenant = np.concatenate([np.array([0], dtype=np.int64), prev_tenant])
        batch["cond"].append(np.repeat(seq["cond"][None, :], length, axis=0))
        batch["state"].append(seq["states"][start:end])
        batch["action"].append(seq["actions"][start:end])
        batch["rank"].append(seq["ranks"][start:end])
        batch["prev_num"].append(np.stack([prev_dt, prev_size], axis=-1))
        batch["prev_opcode"].append(prev_opcode)
        batch["prev_tenant"].append(prev_tenant)
        batch["dt_z"].append(seq["dt_z"][start:end])
        batch["size_z"].append(seq["size_z"][start:end])
        batch["opcode"].append(seq["opcode"][start:end])
        batch["tenant"].append(seq["tenant"][start:end])

    return {
        "cond": torch.tensor(np.stack(batch["cond"]), dtype=torch.float32),
        "state": torch.tensor(np.stack(batch["state"]), dtype=torch.long),
        "action": torch.tensor(np.stack(batch["action"]), dtype=torch.long),
        "rank": torch.tensor(np.stack(batch["rank"]), dtype=torch.long),
        "prev_num": torch.tensor(np.stack(batch["prev_num"]), dtype=torch.float32),
        "prev_opcode": torch.tensor(np.stack(batch["prev_opcode"]), dtype=torch.long),
        "prev_tenant": torch.tensor(np.stack(batch["prev_tenant"]), dtype=torch.long),
        "dt_z": torch.tensor(np.stack(batch["dt_z"]), dtype=torch.float32),
        "size_z": torch.tensor(np.stack(batch["size_z"]), dtype=torch.float32),
        "opcode": torch.tensor(np.stack(batch["opcode"]), dtype=torch.long),
        "tenant": torch.tensor(np.stack(batch["tenant"]), dtype=torch.long),
    }


def _to_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _sample_logits(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    z = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - np.max(z)
    p = np.exp(z)
    p = p / max(float(p.sum()), 1e-12)
    return int(rng.choice(len(p), p=p))


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    return mean, max(std, 1e-3)


def _ordered_values(values: np.ndarray) -> list[Any]:
    seen = {}
    for value in values:
        seen.setdefault(str(value), value)
    return [seen[key] for key in sorted(seen)]


def _fixed_cond(cond: np.ndarray, cond_dim: int) -> np.ndarray:
    arr = np.asarray(cond, dtype=np.float32)
    if len(arr) < cond_dim:
        arr = np.pad(arr, (0, cond_dim - len(arr)))
    return arr[:cond_dim]


_N_RANK_BUCKETS = 7
