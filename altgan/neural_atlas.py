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
from collections import Counter, deque
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
    stack_distances,
)

_INT64_MAX = np.iinfo(np.int64).max
_MAX_GENERATED_ID_BASE = 1_000_000_000_000_000
_PHASE_WINDOW = 200
RANK_PMF_EDGES = np.array([
    0, 1, 2, 3, 4, 6, 8, 11, 14, 18, 24, 32, 42, 56, 74, 97, 127, 167,
    219, 288, 377, 495, 649, 851, 1116, 1463, 1919, 2516, 3299, 4323,
    5669, 7437, 9750, 12780, 16752, 21959, 28785, 37736, 49467, 64853,
    85036, 111472, 146148, 191628, 251236, 1 << 30,
], dtype=np.int64)


@dataclass
class AtlasReservoir:
    name: str
    cond: np.ndarray
    initial_states: np.ndarray
    initial_probs: np.ndarray
    transitions: Dict[int, Tuple[np.ndarray, np.ndarray]]
    samples_by_state: Dict[int, List[EventSample]]
    max_obj_id: int
    rank_pmf_by_state: Dict[int, np.ndarray] = field(default_factory=dict)


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
    rank_pmf_edges: np.ndarray | None = None
    rank_pmf_by_state: Dict[int, np.ndarray] = field(default_factory=dict)
    rank_state_edges: np.ndarray | None = None
    n_dist_states: int = StackAtlasModel.N_ACTIONS
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
        stack_rank_tail_pivot: int | None = None,
        stack_rank_tail_scale: float = 1.0,
        stack_rank_position_scales: Sequence[float] | None = None,
        stack_rank_stream_scales: Sequence[float] | None = None,
        stack_rank_pmf_prob: float = 0.0,
        stack_rank_pmf_scale: float = 1.0,
        stack_rank_pmf_bin_power: float = 1.0,
        stack_rank_pmf_tail_bin_power: float | None = None,
        stack_rank_pmf_tail_power_pivot: int | None = None,
        stack_rank_pmf_local_prob: float = 0.0,
        stack_rank_pmf_feedback_strength: float = 0.0,
        stack_rank_pmf_feedback_alpha: float = 32.0,
        stack_rank_pmf_guard_prob: float = 0.0,
        stack_rank_pmf_guard_strength: float = 1.0,
        stack_footprint_target_curves: Sequence[Sequence[int]] | None = None,
        stack_footprint_feedback_strength: float = 0.0,
        stack_footprint_feedback_deadband: float = 0.02,
        stack_reuse_boost_prob: float = 0.0,
        stack_reuse_boost_min_rank: int = 0,
        stack_reuse_boost_rank_power: float = 1.0,
        stack_reuse_drop_prob: float = 0.0,
        stack_reuse_drop_position_probs: Sequence[float] | None = None,
        stack_adj_dup_prob: float = 0.0,
        stack_adj_dup_position_probs: Sequence[float] | None = None,
        stack_adj_dup_min_rank: int = 0,
        stack_adj_dup_max_rank: int = 0,
        stack_adj_dup_band_prob: float = 1.0,
        stack_rank_band_reuse_prob: float = 0.0,
        stack_rank_band_reuse_position_probs: Sequence[float] | None = None,
        stack_rank_band_reuse_min_rank: int = 0,
        stack_rank_band_reuse_max_rank: int | None = None,
        stack_rank_band_reuse_power: float = 1.0,
        stack_hot_pool_prob: float = 0.0,
        stack_hot_pool_position_probs: Sequence[float] | None = None,
        stack_hot_pool_k: int = 100,
        stack_hot_pool_window: int = 5000,
        stack_hot_pool_weight_power: float = 1.0,
        stack_hot_pool_max_search: int = 0,
        stack_hot_pool_min_age: int = 0,
        stack_frequency_pool_prob: float = 0.0,
        stack_frequency_pool_k: int = 100,
        stack_frequency_pool_max_candidates: int = 1000,
        stack_frequency_pool_refresh_interval: int = 512,
        stack_frequency_pool_min_count_rank: int = 0,
        stack_frequency_pool_max_count_rank: int | None = None,
        stack_frequency_pool_weight_power: float = 1.0,
        stack_frequency_pool_min_age: int = 0,
        stack_frequency_pool_min_rank: int = 0,
        stack_frequency_pool_max_rank: int | None = None,
        stack_frequency_pool_max_search: int = 0,
        stack_frequency_pool_sample_attempts: int = 8,
        stack_anchor_pool_prob: float = 0.0,
        stack_anchor_pool_position_probs: Sequence[float] | None = None,
        stack_anchor_pool_k: int = 256,
        stack_anchor_pool_promote_prob: float = 0.0,
        stack_anchor_pool_weight_power: float = 1.0,
        stack_anchor_pool_min_age: int = 0,
        stack_anchor_pool_min_rank: int = 0,
        stack_anchor_pool_max_rank: int | None = None,
        stack_anchor_pool_sample_attempts: int = 8,
        stack_delayed_reuse_prob: float = 0.0,
        stack_delayed_reuse_position_probs: Sequence[float] | None = None,
        stack_delayed_reuse_schedule_prob: float = 0.0,
        stack_delayed_reuse_schedule_reuses: bool = False,
        stack_delayed_reuse_min_delay: int = 8192,
        stack_delayed_reuse_max_delay: int = 65536,
        stack_delayed_reuse_min_rank: int = 0,
        stack_delayed_reuse_max_rank: int | None = None,
        stack_delayed_reuse_max_pending: int = 4096,
        stack_delayed_reuse_sample_attempts: int = 8,
        stack_tail_reuse_prob: float = 0.0,
        stack_tail_reuse_position_probs: Sequence[float] | None = None,
        stack_tail_reuse_min_frac: float = 0.5,
        stack_tail_reuse_rank_power: float = 1.0,
        stack_recent_pool_prob: float = 0.0,
        stack_recent_pool_position_probs: Sequence[float] | None = None,
        stack_recent_pool_window: int = 200,
        stack_rank_phase_scales: Sequence[float] | None = None,
        stack_rank_phase_maxes: Sequence[int] | None = None,
        mark_temperature: float | None = None,
        mark_numeric_noise: float = 0.05,
        mark_numeric_blend: float = 1.0,
        mark_numeric_blend_space: str = "raw",
        mark_numeric_fields: str = "both",
        mark_categorical_source: str = "neural",
        mark_feedback_numeric_blend: float | None = None,
        mark_feedback_numeric_blend_space: str = "log",
        mark_feedback_numeric_fields: str = "both",
        progress_interval: int = 0,
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
        stack_rank_tail_scale = max(float(stack_rank_tail_scale), 0.0)
        stack_rank_position_scales = _nonnegative_float_list(stack_rank_position_scales)
        stack_rank_stream_scales = _nonnegative_float_list(stack_rank_stream_scales)
        stack_rank_pmf_prob = float(np.clip(stack_rank_pmf_prob, 0.0, 1.0))
        stack_rank_pmf_scale = max(float(stack_rank_pmf_scale), 0.0)
        stack_rank_pmf_bin_power = max(float(stack_rank_pmf_bin_power), 1e-6)
        if stack_rank_pmf_tail_bin_power is not None:
            stack_rank_pmf_tail_bin_power = max(float(stack_rank_pmf_tail_bin_power), 1e-6)
        if stack_rank_pmf_tail_power_pivot is not None and stack_rank_pmf_tail_power_pivot < 0:
            stack_rank_pmf_tail_power_pivot = None
        stack_rank_pmf_local_prob = float(np.clip(stack_rank_pmf_local_prob, 0.0, 1.0))
        stack_rank_pmf_feedback_strength = max(float(stack_rank_pmf_feedback_strength), 0.0)
        stack_rank_pmf_feedback_alpha = max(float(stack_rank_pmf_feedback_alpha), 0.0)
        stack_rank_pmf_guard_prob = float(np.clip(stack_rank_pmf_guard_prob, 0.0, 1.0))
        stack_rank_pmf_guard_strength = max(float(stack_rank_pmf_guard_strength), 0.0)
        footprint_target_curves = _normalize_footprint_curves(
            stack_footprint_target_curves,
            n_streams,
        )
        stack_footprint_feedback_strength = max(float(stack_footprint_feedback_strength), 0.0)
        stack_footprint_feedback_deadband = max(float(stack_footprint_feedback_deadband), 0.0)
        if stack_rank_tail_pivot is not None and stack_rank_tail_pivot < 0:
            stack_rank_tail_pivot = None
        stack_reuse_boost_prob = float(np.clip(stack_reuse_boost_prob, 0.0, 1.0))
        stack_reuse_boost_min_rank = max(int(stack_reuse_boost_min_rank), 0)
        stack_reuse_boost_rank_power = max(float(stack_reuse_boost_rank_power), 1e-6)
        stack_reuse_drop_prob = float(np.clip(stack_reuse_drop_prob, 0.0, 1.0))
        stack_reuse_drop_position_probs = _clip_prob_list(stack_reuse_drop_position_probs)
        stack_adj_dup_prob = float(np.clip(stack_adj_dup_prob, 0.0, 1.0))
        stack_adj_dup_position_probs = _clip_prob_list(stack_adj_dup_position_probs)
        stack_adj_dup_min_rank = max(int(stack_adj_dup_min_rank), 0)
        stack_adj_dup_max_rank = max(int(stack_adj_dup_max_rank), stack_adj_dup_min_rank)
        stack_adj_dup_band_prob = float(np.clip(stack_adj_dup_band_prob, 0.0, 1.0))
        stack_rank_band_reuse_prob = float(np.clip(stack_rank_band_reuse_prob, 0.0, 1.0))
        stack_rank_band_reuse_position_probs = _clip_prob_list(
            stack_rank_band_reuse_position_probs
        )
        stack_rank_band_reuse_min_rank = max(int(stack_rank_band_reuse_min_rank), 0)
        if stack_rank_band_reuse_max_rank is not None and stack_rank_band_reuse_max_rank < 0:
            stack_rank_band_reuse_max_rank = None
        stack_rank_band_reuse_power = max(float(stack_rank_band_reuse_power), 1e-6)
        stack_hot_pool_prob = float(np.clip(stack_hot_pool_prob, 0.0, 1.0))
        stack_hot_pool_position_probs = _clip_prob_list(stack_hot_pool_position_probs)
        stack_hot_pool_k = max(int(stack_hot_pool_k), 1)
        stack_hot_pool_window = max(int(stack_hot_pool_window), 1)
        stack_hot_pool_weight_power = max(float(stack_hot_pool_weight_power), 1e-6)
        stack_hot_pool_max_search = max(int(stack_hot_pool_max_search), 0)
        stack_hot_pool_min_age = max(int(stack_hot_pool_min_age), 0)
        stack_frequency_pool_prob = float(np.clip(stack_frequency_pool_prob, 0.0, 1.0))
        stack_frequency_pool_k = max(int(stack_frequency_pool_k), 1)
        stack_frequency_pool_max_candidates = max(
            int(stack_frequency_pool_max_candidates),
            stack_frequency_pool_k,
        )
        stack_frequency_pool_refresh_interval = max(
            int(stack_frequency_pool_refresh_interval),
            1,
        )
        stack_frequency_pool_min_count_rank = max(
            int(stack_frequency_pool_min_count_rank),
            0,
        )
        if (
            stack_frequency_pool_max_count_rank is not None
            and stack_frequency_pool_max_count_rank < 0
        ):
            stack_frequency_pool_max_count_rank = None
        if stack_frequency_pool_max_count_rank is not None:
            stack_frequency_pool_max_count_rank = max(
                int(stack_frequency_pool_max_count_rank),
                stack_frequency_pool_min_count_rank,
            )
            stack_frequency_pool_max_candidates = max(
                stack_frequency_pool_max_candidates,
                stack_frequency_pool_max_count_rank,
            )
        elif stack_frequency_pool_min_count_rank > 0:
            stack_frequency_pool_max_candidates = max(
                stack_frequency_pool_max_candidates,
                stack_frequency_pool_min_count_rank + stack_frequency_pool_k,
            )
        stack_frequency_pool_weight_power = max(float(stack_frequency_pool_weight_power), 1e-6)
        stack_frequency_pool_min_age = max(int(stack_frequency_pool_min_age), 0)
        stack_frequency_pool_min_rank = max(int(stack_frequency_pool_min_rank), 0)
        if stack_frequency_pool_max_rank is not None and stack_frequency_pool_max_rank < 0:
            stack_frequency_pool_max_rank = None
        stack_frequency_pool_max_search = max(int(stack_frequency_pool_max_search), 0)
        stack_frequency_pool_sample_attempts = max(int(stack_frequency_pool_sample_attempts), 1)
        stack_tail_reuse_prob = float(np.clip(stack_tail_reuse_prob, 0.0, 1.0))
        stack_tail_reuse_position_probs = _clip_prob_list(stack_tail_reuse_position_probs)
        stack_tail_reuse_min_frac = float(np.clip(stack_tail_reuse_min_frac, 0.0, 1.0))
        stack_tail_reuse_rank_power = max(float(stack_tail_reuse_rank_power), 1e-6)
        stack_recent_pool_prob = float(np.clip(stack_recent_pool_prob, 0.0, 1.0))
        stack_recent_pool_position_probs = _clip_prob_list(stack_recent_pool_position_probs)
        stack_recent_pool_window = max(int(stack_recent_pool_window), 1)
        mark_numeric_blend = float(np.clip(mark_numeric_blend, 0.0, 1.0))
        if mark_feedback_numeric_blend is not None:
            mark_feedback_numeric_blend = float(np.clip(mark_feedback_numeric_blend, 0.0, 1.0))
        if mark_numeric_blend_space not in {"raw", "log"}:
            raise ValueError("mark_numeric_blend_space must be 'raw' or 'log'")
        if mark_feedback_numeric_blend_space not in {"raw", "log"}:
            raise ValueError("mark_feedback_numeric_blend_space must be 'raw' or 'log'")
        if mark_numeric_fields not in {"both", "dt", "size"}:
            raise ValueError("mark_numeric_fields must be 'both', 'dt', or 'size'")
        if mark_feedback_numeric_fields not in {"both", "dt", "size"}:
            raise ValueError("mark_feedback_numeric_fields must be 'both', 'dt', or 'size'")
        if mark_categorical_source not in {"neural", "reservoir"}:
            raise ValueError("mark_categorical_source must be 'neural' or 'reservoir'")
        progress_interval = max(int(progress_interval), 0)
        mark_runtime = None
        mark_model = getattr(self, "mark_model", None)
        if mark_model is not None:
            mark_runtime = mark_model.runtime(
                n_streams=n_streams,
                seed=seed + 17,
                temperature=temperature if mark_temperature is None else mark_temperature,
                numeric_noise=mark_numeric_noise,
            )
            mark_runtime.set_conditions(conds)

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
        rank_state_edges = getattr(self, "rank_state_edges", None)
        rank_pmf_edges = getattr(self, "rank_pmf_edges", None)
        rank_pmf_by_state = getattr(self, "rank_pmf_by_state", {}) or {}
        n_dist_states = int(getattr(self, "n_dist_states", StackAtlasModel.N_ACTIONS))
        if rank_state_edges is None:
            n_dist_states = StackAtlasModel.N_ACTIONS
        base_span = self.n_time_bins * self.n_size_bins * n_dist_states
        for stream_id in range(n_streams):
            reservoir = self._nearest_reservoir(conds[stream_id])
            init_p = self._blend_initial(
                init_probs[stream_id],
                reservoir,
                transition_blend,
                local_prob_power,
            )
            state = int(rng.choice(self.n_states, p=init_p))
            stack = _RankedLRUStack()
            in_stack: set[int] = set()
            id_base = int(self.max_obj_id)
            if id_base > _MAX_GENERATED_ID_BASE:
                id_base = 0
            next_new_id = id_base + 1 + stream_id * (per_stream + 1_000_003)
            prev_obj = next_new_id
            ts = 0.0
            hot_counts: Counter[int] = Counter()
            hot_last_pos: dict[int, int] = {}
            hot_window: deque[int] = deque()
            hot_pool: list[tuple[int, int]] = []
            freq_counts: Counter[int] = Counter()
            freq_last_pos: dict[int, int] = {}
            freq_pool: list[tuple[int, int]] = []
            anchor_counts: Counter[int] = Counter()
            anchor_last_pos: dict[int, int] = {}
            rank_pmf_feedback_counts: Dict[int, np.ndarray] = {}
            delayed_due: dict[int, deque[int]] = {}
            delayed_ready: deque[int] = deque()
            delayed_pending = 0
            recent_window: deque[int] = deque(maxlen=stack_recent_pool_window)
            footprint_curve = footprint_target_curves[stream_id]

            for pos in range(per_stream):
                due_now = delayed_due.pop(pos, None)
                if due_now:
                    delayed_ready.extend(due_now)
                if stack_hot_pool_prob > 0.0 and pos % 512 == 0 and hot_counts:
                    hot_pool = hot_counts.most_common(stack_hot_pool_k)
                if (
                    stack_frequency_pool_prob > 0.0
                    and pos % stack_frequency_pool_refresh_interval == 0
                    and freq_counts
                ):
                    freq_pool = _frequency_rank_pool(
                        freq_counts,
                        pool_size=stack_frequency_pool_k,
                        min_count_rank=stack_frequency_pool_min_count_rank,
                        max_count_rank=stack_frequency_pool_max_count_rank,
                    )
                phase = min((pos * self.n_phase_bins) // per_stream, self.n_phase_bins - 1)
                if force_phase_schedule and self.n_phase_bins > 1:
                    state = _state_with_phase(state, phase, base_span)
                ev = self._sample_event(reservoir, state, rng)
                wants_reuse = ev.action_class != StackAtlasModel.ACTION_NEW
                position_rank_scale = _position_value(
                    stack_rank_position_scales,
                    pos,
                    per_stream,
                    stack_rank_scale,
                )
                if stack_rank_stream_scales:
                    stream_idx = min(stream_id, len(stack_rank_stream_scales) - 1)
                    position_rank_scale = float(stack_rank_stream_scales[stream_idx])
                adj_dup_prob = _position_value(
                    stack_adj_dup_position_probs,
                    pos,
                    per_stream,
                    stack_adj_dup_prob,
                )
                hot_pool_prob = _position_value(
                    stack_hot_pool_position_probs,
                    pos,
                    per_stream,
                    stack_hot_pool_prob,
                )
                recent_pool_prob = _position_value(
                    stack_recent_pool_position_probs,
                    pos,
                    per_stream,
                    stack_recent_pool_prob,
                )
                tail_reuse_prob = _position_value(
                    stack_tail_reuse_position_probs,
                    pos,
                    per_stream,
                    stack_tail_reuse_prob,
                )
                rank_band_reuse_prob = _position_value(
                    stack_rank_band_reuse_position_probs,
                    pos,
                    per_stream,
                    stack_rank_band_reuse_prob,
                )
                anchor_pool_prob = _position_value(
                    stack_anchor_pool_position_probs,
                    pos,
                    per_stream,
                    stack_anchor_pool_prob,
                )
                delayed_reuse_prob = _position_value(
                    stack_delayed_reuse_position_probs,
                    pos,
                    per_stream,
                    stack_delayed_reuse_prob,
                )
                reuse_drop_prob = _position_value(
                    stack_reuse_drop_position_probs,
                    pos,
                    per_stream,
                    stack_reuse_drop_prob,
                )
                dropped_reuse = (
                    wants_reuse
                    and reuse_drop_prob > 0.0
                    and rng.random() < reuse_drop_prob
                )
                if dropped_reuse:
                    wants_reuse = False
                boosted_reuse = (
                    not wants_reuse
                    and not dropped_reuse
                    and bool(stack)
                    and stack_reuse_boost_prob > 0.0
                    and rng.random() < stack_reuse_boost_prob
                )
                if boosted_reuse:
                    wants_reuse = True
                if stack_footprint_feedback_strength > 0.0 and footprint_curve:
                    target_footprint = _position_curve_value(footprint_curve, pos, per_stream)
                    if target_footprint is not None and target_footprint > 0:
                        pressure = (len(stack) - target_footprint) / max(target_footprint, 1)
                        deadband = stack_footprint_feedback_deadband
                        if wants_reuse and pressure < -deadband:
                            prob = min(
                                stack_footprint_feedback_strength * (-pressure - deadband),
                                1.0,
                            )
                            if rng.random() < prob:
                                wants_reuse = False
                                boosted_reuse = False
                                dropped_reuse = True
                        elif (
                            not wants_reuse
                            and not dropped_reuse
                            and bool(stack)
                            and pressure > deadband
                        ):
                            prob = min(
                                stack_footprint_feedback_strength * (pressure - deadband),
                                1.0,
                            )
                            if rng.random() < prob:
                                wants_reuse = True
                                boosted_reuse = True
                if wants_reuse and stack:
                    if tail_reuse_prob > 0.0 and rng.random() < tail_reuse_prob:
                        lo = max(int(len(stack) * stack_tail_reuse_min_frac), len(stack) // 2)
                        lo = min(max(lo, 0), len(stack) - 1)
                        if stack_tail_reuse_rank_power == 1.0:
                            rank = int(rng.integers(lo, len(stack)))
                        else:
                            u = float(rng.random())
                            biased = u ** stack_tail_reuse_rank_power
                            rank = int(lo + (len(stack) - 1 - lo) * biased)
                            rank = max(lo, min(rank, len(stack) - 1))
                    elif rank_band_reuse_prob > 0.0 and rng.random() < rank_band_reuse_prob:
                        rank = _sample_rank_band(
                            stack_len=len(stack),
                            min_rank=stack_rank_band_reuse_min_rank,
                            max_rank=stack_rank_band_reuse_max_rank,
                            rank_power=stack_rank_band_reuse_power,
                            rng=rng,
                        )
                    elif (
                        not boosted_reuse
                        and delayed_reuse_prob > 0.0
                        and delayed_ready
                        and rng.random() < delayed_reuse_prob
                    ):
                        rank_from_delayed, delayed_popped = _rank_from_delayed_pool(
                            stack,
                            delayed_ready,
                            min_rank=stack_delayed_reuse_min_rank,
                            max_rank=stack_delayed_reuse_max_rank,
                            sample_attempts=stack_delayed_reuse_sample_attempts,
                        )
                        delayed_pending = max(delayed_pending - delayed_popped, 0)
                        if rank_from_delayed is not None:
                            rank = rank_from_delayed
                        else:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    elif (
                        not boosted_reuse
                        and anchor_pool_prob > 0.0
                        and anchor_counts
                        and rng.random() < anchor_pool_prob
                    ):
                        rank_from_pool = _rank_from_object_pool(
                            stack,
                            list(anchor_counts.items()),
                            last_pos=anchor_last_pos,
                            current_pos=pos,
                            min_age=stack_anchor_pool_min_age,
                            min_rank=stack_anchor_pool_min_rank,
                            max_rank=stack_anchor_pool_max_rank,
                            weight_power=stack_anchor_pool_weight_power,
                            max_search=0,
                            sample_attempts=stack_anchor_pool_sample_attempts,
                            rng=rng,
                        )
                        if rank_from_pool is not None:
                            rank = rank_from_pool
                        else:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    elif adj_dup_prob > 0.0 and rng.random() < adj_dup_prob:
                        use_band = (
                            stack_adj_dup_max_rank > 0
                            and rng.random() < stack_adj_dup_band_prob
                        )
                        if use_band:
                            adj_lo = min(stack_adj_dup_min_rank, len(stack) - 1)
                            adj_hi = min(stack_adj_dup_max_rank, len(stack) - 1)
                            if adj_hi <= adj_lo:
                                rank = adj_lo
                            else:
                                rank = int(rng.integers(adj_lo, adj_hi + 1))
                        else:
                            rank = 0
                    elif (
                        recent_pool_prob > 0.0
                        and recent_window
                        and rng.random() < recent_pool_prob
                    ):
                        recent_obj = int(recent_window[int(rng.integers(0, len(recent_window)))])
                        try:
                            rank = stack.index(recent_obj)
                        except ValueError:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    elif (
                        not boosted_reuse
                        and stack_frequency_pool_prob > 0.0
                        and freq_pool
                        and rng.random() < stack_frequency_pool_prob
                    ):
                        rank_from_pool = _rank_from_object_pool(
                            stack,
                            freq_pool,
                            last_pos=freq_last_pos,
                            current_pos=pos,
                            min_age=stack_frequency_pool_min_age,
                            min_rank=stack_frequency_pool_min_rank,
                            max_rank=stack_frequency_pool_max_rank,
                            weight_power=stack_frequency_pool_weight_power,
                            max_search=stack_frequency_pool_max_search,
                            sample_attempts=stack_frequency_pool_sample_attempts,
                            rng=rng,
                        )
                        if rank_from_pool is not None:
                            rank = rank_from_pool
                        else:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    elif (
                        not boosted_reuse
                        and hot_pool_prob > 0.0
                        and hot_pool
                        and rng.random() < hot_pool_prob
                    ):
                        eligible_hot_pool = _eligible_hot_pool(
                            hot_pool,
                            last_pos=hot_last_pos,
                            current_pos=pos,
                            min_age=stack_hot_pool_min_age,
                        )
                        if eligible_hot_pool:
                            hot_obj = _sample_hot_pool_obj(
                                eligible_hot_pool,
                                weight_power=stack_hot_pool_weight_power,
                                rng=rng,
                            )
                            try:
                                if stack_hot_pool_max_search > 0:
                                    rank = stack.index(
                                        hot_obj,
                                        0,
                                        min(len(stack), stack_hot_pool_max_search),
                                    )
                                else:
                                    rank = stack.index(hot_obj)
                            except ValueError:
                                phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                                phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                                if phase_rank_max is not None and phase_rank_max < 0:
                                    phase_rank_max = None
                                rank = _calibrated_stack_rank(
                                    ev.stack_distance,
                                    stack_rank_scale=phase_rank_scale,
                                    stack_rank_max=phase_rank_max,
                                    stack_rank_tail_pivot=stack_rank_tail_pivot,
                                    stack_rank_tail_scale=stack_rank_tail_scale,
                                    stack_len=len(stack),
                                )
                        else:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    elif boosted_reuse:
                        rank = _boosted_reuse_rank(
                            stack_len=len(stack),
                            min_rank=stack_reuse_boost_min_rank,
                            rank_power=stack_reuse_boost_rank_power,
                            rng=rng,
                        )
                    else:
                        rank = None
                        if stack_rank_pmf_prob > 0.0 and rng.random() < stack_rank_pmf_prob:
                            pmf = rank_pmf_by_state.get(int(state))
                            if stack_rank_pmf_local_prob > 0.0 and rng.random() < stack_rank_pmf_local_prob:
                                local_pmfs = getattr(reservoir, "rank_pmf_by_state", {}) or {}
                                local_pmf = local_pmfs.get(int(state))
                                if local_pmf is not None:
                                    pmf = local_pmf
                            rank = _sample_rank_pmf(
                                rank_pmf_edges,
                                pmf,
                                stack_len=len(stack),
                                bin_power=stack_rank_pmf_bin_power,
                                tail_bin_power=stack_rank_pmf_tail_bin_power,
                                tail_power_pivot=stack_rank_pmf_tail_power_pivot,
                                feedback_counts=rank_pmf_feedback_counts.get(int(state)),
                                feedback_strength=stack_rank_pmf_feedback_strength,
                                feedback_alpha=stack_rank_pmf_feedback_alpha,
                                rng=rng,
                            )
                            if rank is not None:
                                rank = min(
                                    int(round(int(rank) * stack_rank_pmf_scale)),
                                    len(stack) - 1,
                                )
                        if rank is None:
                            phase_rank_scale = _phase_value(stack_rank_phase_scales, phase, position_rank_scale)
                            phase_rank_max = _phase_value(stack_rank_phase_maxes, phase, stack_rank_max)
                            if phase_rank_max is not None and phase_rank_max < 0:
                                phase_rank_max = None
                            rank = _calibrated_stack_rank(
                                ev.stack_distance,
                                stack_rank_scale=phase_rank_scale,
                                stack_rank_max=phase_rank_max,
                                stack_rank_tail_pivot=stack_rank_tail_pivot,
                                stack_rank_tail_scale=stack_rank_tail_scale,
                                stack_len=len(stack),
                            )
                    if (
                        stack_rank_pmf_guard_prob > 0.0
                        and rank_pmf_edges is not None
                        and rng.random() < stack_rank_pmf_guard_prob
                    ):
                        guard_pmf = rank_pmf_by_state.get(int(state))
                        if stack_rank_pmf_local_prob > 0.0 and rng.random() < stack_rank_pmf_local_prob:
                            local_pmfs = getattr(reservoir, "rank_pmf_by_state", {}) or {}
                            local_pmf = local_pmfs.get(int(state))
                            if local_pmf is not None:
                                guard_pmf = local_pmf
                        feedback_counts = rank_pmf_feedback_counts.get(int(state))
                        if _rank_pmf_bin_over_target(
                            int(rank),
                            rank_pmf_edges,
                            guard_pmf,
                            feedback_counts,
                            feedback_alpha=stack_rank_pmf_feedback_alpha,
                        ):
                            guarded_rank = _sample_rank_pmf(
                                rank_pmf_edges,
                                guard_pmf,
                                stack_len=len(stack),
                                bin_power=stack_rank_pmf_bin_power,
                                tail_bin_power=stack_rank_pmf_tail_bin_power,
                                tail_power_pivot=stack_rank_pmf_tail_power_pivot,
                                feedback_counts=feedback_counts,
                                feedback_strength=stack_rank_pmf_guard_strength,
                                feedback_alpha=stack_rank_pmf_feedback_alpha,
                                rng=rng,
                            )
                            if guarded_rank is not None:
                                rank = min(
                                    int(round(int(guarded_rank) * stack_rank_pmf_scale)),
                                    len(stack) - 1,
                                )
                    effective_stack_distance = int(rank)
                    effective_action_class = _action_class(effective_stack_distance)
                    obj_id = stack.move_to_front(rank)
                    emitted_new = False
                    if (
                        stack_rank_pmf_feedback_strength > 0.0
                        and rank_pmf_edges is not None
                    ):
                        counts = rank_pmf_feedback_counts.setdefault(
                            int(state),
                            np.zeros(len(rank_pmf_edges) - 1, dtype=np.float64),
                        )
                        counts[_rank_bin_index(effective_stack_distance, rank_pmf_edges)] += 1.0
                else:
                    effective_stack_distance = int(ev.stack_distance)
                    effective_action_class = int(ev.action_class)
                    obj_id, next_new_id = StackAtlasModel._new_object_id(
                        prev_obj,
                        next_new_id,
                        _bounded_new_stride(ev.stride),
                        in_stack,
                    )
                    stack.insert_front(obj_id)
                    in_stack.add(obj_id)
                    emitted_new = True
                mark_dist_state = _dist_state_from_stack_distance(
                    effective_stack_distance,
                    rank_state_edges,
                )
                mark_state = _state_with_dist_state(
                    state,
                    mark_dist_state,
                    n_dist_states,
                )

                mark = ev
                if mark_runtime is not None:
                    neural_mark = mark_runtime.sample(
                        stream_id=stream_id,
                        cond=conds[stream_id],
                        state=mark_state,
                        action_class=effective_action_class,
                        stack_distance=effective_stack_distance,
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
                            stack_distance=effective_stack_distance,
                            action_class=effective_action_class,
                        )
                    feedback_mark = mark
                    if mark_feedback_numeric_blend is not None:
                        feedback_dt, feedback_size = _blend_numeric_marks(
                            ev,
                            neural_mark,
                            mark_feedback_numeric_blend,
                            mark_feedback_numeric_blend_space,
                            mark_feedback_numeric_fields,
                        )
                        feedback_mark = EventSample(
                            dt=float(feedback_dt),
                            obj_size=float(feedback_size),
                            opcode=mark.opcode,
                            tenant=mark.tenant,
                            stride=int(ev.stride),
                            stack_distance=effective_stack_distance,
                            action_class=effective_action_class,
                        )
                    mark_runtime.observe(stream_id, feedback_mark)

                ts += max(float(mark.dt), 0.0)
                rows.append({
                    "stream_id": stream_id,
                    "ts": ts,
                    "obj_id": int(obj_id),
                    "obj_size": max(int(round(mark.obj_size)), 1),
                    "opcode": mark.opcode,
                    "tenant": mark.tenant,
                    "stack_distance": effective_stack_distance,
                    "action_class": effective_action_class,
                })
                if (
                    stack_delayed_reuse_schedule_prob > 0.0
                    and delayed_pending < max(int(stack_delayed_reuse_max_pending), 1)
                    and (emitted_new or stack_delayed_reuse_schedule_reuses)
                    and rng.random() < stack_delayed_reuse_schedule_prob
                ):
                    lo_delay = max(int(stack_delayed_reuse_min_delay), 1)
                    hi_delay = max(int(stack_delayed_reuse_max_delay), lo_delay)
                    due_pos = pos + int(rng.integers(lo_delay, hi_delay + 1))
                    if due_pos < per_stream:
                        delayed_due.setdefault(due_pos, deque()).append(int(obj_id))
                        delayed_pending += 1
                if stack_hot_pool_prob > 0.0 or stack_hot_pool_position_probs:
                    if len(hot_window) >= stack_hot_pool_window:
                        old_obj = hot_window.popleft()
                        hot_counts[old_obj] -= 1
                        if hot_counts[old_obj] <= 0:
                            del hot_counts[old_obj]
                            hot_last_pos.pop(old_obj, None)
                    hot_window.append(int(obj_id))
                    hot_counts[int(obj_id)] += 1
                    hot_last_pos[int(obj_id)] = pos
                if stack_frequency_pool_prob > 0.0:
                    freq_obj = int(obj_id)
                    freq_counts[freq_obj] += 1
                    freq_last_pos[freq_obj] = pos
                    if len(freq_counts) > stack_frequency_pool_max_candidates:
                        keep_n = _frequency_rank_keep_count(
                            stack_frequency_pool_k,
                            stack_frequency_pool_min_count_rank,
                            stack_frequency_pool_max_count_rank,
                        )
                        freq_counts = Counter(dict(freq_counts.most_common(keep_n)))
                        freq_last_pos = {
                            obj: freq_last_pos[obj]
                            for obj in freq_counts
                            if obj in freq_last_pos
                        }
                        freq_pool = _frequency_rank_pool(
                            freq_counts,
                            pool_size=stack_frequency_pool_k,
                            min_count_rank=stack_frequency_pool_min_count_rank,
                            max_count_rank=stack_frequency_pool_max_count_rank,
                        )
                if stack_anchor_pool_prob > 0.0 or stack_anchor_pool_position_probs:
                    anchor_obj = int(obj_id)
                    if anchor_obj in anchor_counts:
                        anchor_counts[anchor_obj] += 1
                        anchor_last_pos[anchor_obj] = pos
                    elif (
                        len(anchor_counts) < max(int(stack_anchor_pool_k), 1)
                        and stack_anchor_pool_promote_prob > 0.0
                        and rng.random() < stack_anchor_pool_promote_prob
                    ):
                        anchor_counts[anchor_obj] = 1
                        anchor_last_pos[anchor_obj] = pos
                if stack_recent_pool_prob > 0.0 or stack_recent_pool_position_probs:
                    recent_window.append(int(obj_id))
                if progress_interval > 0 and (pos + 1) % progress_interval == 0:
                    print(
                        "[altgan.neural_atlas] "
                        f"stream={stream_id + 1}/{n_streams} "
                        f"pos={pos + 1}/{per_stream} "
                        f"rows={len(rows)} stack={len(stack)}",
                        flush=True,
                    )
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
    phase_mode: str = "position",
    hidden_dim: int = 96,
    epochs: int = 800,
    lr: float = 2e-3,
    cond_noise_std: float = 0.0,
    max_samples_per_state: int = 1024,
    rank_state_edges: Sequence[int] | None = None,
    transition_weight_mode: str = "log",
    seed: int = 7,
) -> NeuralAtlasModel:
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    if not (len(frames) == len(conds) == len(names)):
        raise ValueError("frames, conds, and names must have the same length")
    clean_triples = [
        (_canonical_frame(df), cond, name)
        for df, cond, name in zip(frames, conds, names)
        if len(df) > 1
    ]
    if not clean_triples:
        raise ValueError("no usable frames")
    clean = [item[0] for item in clean_triples]
    clean_conds = [item[1] for item in clean_triples]
    clean_names = [item[2] for item in clean_triples]

    all_dt = np.concatenate([_interarrival(df["ts"].to_numpy()) for df in clean])
    all_size = np.concatenate([
        np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
        for df in clean
    ])
    time_edges = _quantile_edges(np.log1p(all_dt), n_time_bins)
    size_edges = _quantile_edges(np.log(all_size), n_size_bins)
    n_phase_bins = max(int(n_phase_bins), 1)
    phase_mode = str(phase_mode or "position").strip().lower()
    if phase_mode not in {"position", "unique_rate"}:
        raise ValueError("phase_mode must be one of: position, unique_rate")
    phase_edges = (
        _unique_rate_phase_edges(clean, n_phase_bins)
        if phase_mode == "unique_rate"
        else np.array([], dtype=np.float64)
    )
    rank_state_edges_arr = _rank_state_edges_array(rank_state_edges)
    n_dist_states = _n_dist_states(rank_state_edges_arr)
    n_states = n_phase_bins * n_time_bins * n_size_bins * n_dist_states

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
    global_rank_counts_by_state: Dict[int, np.ndarray] = {}
    global_samples_by_state: Dict[int, List[EventSample]] = {}
    global_seen_by_state: Dict[int, int] = {}
    global_samples: List[EventSample] = []
    global_seen = 0
    max_obj_id = 0

    for df, cond, name in zip(clean, clean_conds, clean_names):
        reservoir, init_counts, transition_counts, rank_counts_by_state = _summarize_file(
            df,
            np.asarray(cond, dtype=np.float32),
            name,
            time_edges=time_edges,
            size_edges=size_edges,
            n_time_bins=n_time_bins,
            n_size_bins=n_size_bins,
            n_phase_bins=n_phase_bins,
            phase_mode=phase_mode,
            phase_edges=phase_edges,
            n_states=n_states,
            rank_state_edges=rank_state_edges_arr,
            n_dist_states=n_dist_states,
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
            trans_weights.append(_transition_weight(total, transition_weight_mode))
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
        for state, counts in rank_counts_by_state.items():
            bucket = global_rank_counts_by_state.setdefault(
                int(state),
                np.zeros(len(RANK_PMF_EDGES) - 1, dtype=np.float64),
            )
            bucket += counts

    cond_arr = np.vstack(file_conds).astype(np.float32)
    init_arr = np.vstack(init_targets).astype(np.float32)
    trans_cond_arr = np.vstack(trans_conds).astype(np.float32)
    trans_state_arr = np.array(trans_states, dtype=np.int64)
    trans_arr = np.vstack(trans_targets).astype(np.float32)
    weight_arr = np.asarray(trans_weights, dtype=np.float32)
    weight_arr = weight_arr / max(float(weight_arr.mean()), 1e-6)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _CondTransitionNet(cond_arr.shape[1], hidden_dim, n_states)
    net.to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    x_init = torch.tensor(cond_arr, device=device)
    y_init = torch.tensor(init_arr, device=device)
    x_trans = torch.tensor(trans_cond_arr, device=device)
    state_trans = torch.tensor(trans_state_arr, dtype=torch.long, device=device)
    y_trans = torch.tensor(trans_arr, device=device)
    w_trans = torch.tensor(weight_arr, device=device)
    cond_noise_std = max(float(cond_noise_std), 0.0)

    losses = []
    for epoch in range(epochs):
        opt.zero_grad()
        if cond_noise_std > 0.0:
            init_in = x_init + cond_noise_std * torch.randn_like(x_init)
            trans_in = x_trans + cond_noise_std * torch.randn_like(x_trans)
        else:
            init_in = x_init
            trans_in = x_trans
        init_logits = net.forward_init(init_in)
        trans_logits = net.forward_trans(trans_in, state_trans)
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
    rank_pmf_by_state = _rank_pmfs(global_rank_counts_by_state)
    metadata = {
        "n_files": len(file_names),
        "n_records": int(sum(len(df) for df in clean)),
        "epochs": epochs,
        "lr": lr,
        "cond_noise_std": cond_noise_std,
        "device": str(device),
        "loss_trace": losses,
        "seed": seed,
        "model": "NeuralAtlasModel",
        "n_transition_rows": int(len(trans_states)),
        "n_phase_bins": n_phase_bins,
        "phase_mode": phase_mode,
        "phase_edges": phase_edges.tolist(),
        "rank_state_edges": (
            None if rank_state_edges_arr is None else rank_state_edges_arr.tolist()
        ),
        "rank_pmf_edges": RANK_PMF_EDGES.tolist(),
        "n_dist_states": int(n_dist_states),
        "transition_weight_mode": transition_weight_mode,
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
        rank_pmf_edges=RANK_PMF_EDGES.copy(),
        rank_pmf_by_state=rank_pmf_by_state,
        rank_state_edges=rank_state_edges_arr,
        n_dist_states=n_dist_states,
        metadata=metadata,
    )


def _summarize_file(df, cond: np.ndarray, name: str, *, time_edges: np.ndarray,
                    size_edges: np.ndarray, n_time_bins: int, n_size_bins: int,
                    n_phase_bins: int, phase_mode: str, phase_edges: np.ndarray,
                    n_states: int,
                    rank_state_edges: np.ndarray | None, n_dist_states: int,
                    max_samples_per_state: int, rng: np.random.Generator):
    ts = df["ts"].to_numpy(dtype=np.float64)
    obj_ids = _object_ids_for_stack(df["obj_id"])
    sizes = np.maximum(df["obj_size"].to_numpy(dtype=np.float64), 1.0)
    dt = _interarrival(ts)
    stack_d = stack_distances(obj_ids)
    strides = np.diff(obj_ids, prepend=obj_ids[0]).astype(np.int64)
    time_bins = np.searchsorted(time_edges, np.log1p(dt), side="right")
    size_bins = np.searchsorted(size_edges, np.log(sizes), side="right")
    if phase_mode == "unique_rate":
        phase_bins = _unique_rate_phase_bins(obj_ids, n_phase_bins, phase_edges)
    else:
        phase_bins = _phase_bins(len(df), n_phase_bins)
    actions = np.array([_action_class(int(x)) for x in stack_d], dtype=np.int64)
    dist_states = np.array([
        _dist_state_from_stack_distance(int(x), rank_state_edges)
        for x in stack_d
    ], dtype=np.int64)
    base_states = _state_id_with_dist(
        time_bins,
        size_bins,
        dist_states,
        n_size_bins,
        n_dist_states,
    )
    states = (phase_bins * (n_time_bins * n_size_bins * n_dist_states)
              + base_states).astype(np.int64)
    opcodes = df["opcode"].to_numpy(dtype=object)
    tenants = df["tenant"].to_numpy(dtype=object)

    initial_counts: Dict[int, int] = {}
    transition_counts: Dict[int, Dict[int, int]] = {}
    samples_by_state: Dict[int, List[EventSample]] = {}
    seen_by_state: Dict[int, int] = {}
    rank_counts_by_state: Dict[int, np.ndarray] = {}

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
        if int(stack_d[i]) >= 0:
            counts = rank_counts_by_state.setdefault(
                state_i,
                np.zeros(len(RANK_PMF_EDGES) - 1, dtype=np.float64),
            )
            counts[_rank_bin_index(int(stack_d[i]), RANK_PMF_EDGES)] += 1.0

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
        rank_pmf_by_state=_rank_pmfs(rank_counts_by_state),
    )
    return reservoir, initial_counts, transition_counts, rank_counts_by_state


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


def _unique_rate_phase_edges(frames: Sequence, n_phase_bins: int) -> np.ndarray:
    if n_phase_bins <= 1:
        return np.array([], dtype=np.float64)
    samples: list[float] = []
    for df in frames:
        obj_ids = _object_ids_for_stack(df["obj_id"])
        if len(obj_ids) < _PHASE_WINDOW * 2:
            continue
        seen: set[int] = set()
        for idx, obj_id in enumerate(obj_ids, start=1):
            seen.add(int(obj_id))
            if idx % _PHASE_WINDOW == 0:
                samples.append(len(seen) / float(_PHASE_WINDOW))
                seen = set()
    if not samples:
        return np.array([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_phase_bins + 1)[1:-1]
    return np.unique(np.quantile(np.asarray(samples, dtype=np.float64), quantiles))


def _unique_rate_phase_bins(
    obj_ids: np.ndarray,
    n_phase_bins: int,
    phase_edges: np.ndarray,
) -> np.ndarray:
    if n_phase_bins <= 1 or len(obj_ids) == 0:
        return np.zeros(len(obj_ids), dtype=np.int64)
    out = np.zeros(len(obj_ids), dtype=np.int64)
    seen: set[int] = set()
    current_rate = 0.0
    for idx, obj_id in enumerate(obj_ids):
        phase = int(np.searchsorted(phase_edges, current_rate, side="right"))
        out[idx] = min(max(phase, 0), n_phase_bins - 1)
        seen.add(int(obj_id))
        if (idx + 1) % _PHASE_WINDOW == 0:
            current_rate = len(seen) / float(_PHASE_WINDOW)
            seen = set()
    return out


def _state_with_phase(state: int, phase: int, base_span: int) -> int:
    return int(phase) * int(base_span) + (int(state) % int(base_span))


def _state_with_action(state: int, action_class: int) -> int:
    action = min(max(int(action_class), 0), StackAtlasModel.N_ACTIONS - 1)
    return int(state) - (int(state) % StackAtlasModel.N_ACTIONS) + action


def _state_id_with_dist(time_bins: np.ndarray, size_bins: np.ndarray,
                        dist_states: np.ndarray, n_size_bins: int,
                        n_dist_states: int) -> np.ndarray:
    return ((time_bins * n_size_bins + size_bins) * int(n_dist_states)
            + dist_states).astype(np.int64)


def _state_with_dist_state(state: int, dist_state: int, n_dist_states: int) -> int:
    n = max(int(n_dist_states), 1)
    dist = min(max(int(dist_state), 0), n - 1)
    return int(state) - (int(state) % n) + dist


def _rank_state_edges_array(values: Sequence[int] | None) -> np.ndarray | None:
    if not values:
        return None
    arr = np.unique(np.asarray([int(v) for v in values], dtype=np.int64))
    arr = arr[arr >= 0]
    if len(arr) < 2:
        raise ValueError("rank_state_edges must contain at least two non-negative edges")
    if int(arr[0]) != 0:
        arr = np.concatenate([np.array([0], dtype=np.int64), arr])
    return arr.astype(np.int64)


def _n_dist_states(rank_state_edges: np.ndarray | None) -> int:
    if rank_state_edges is None:
        return StackAtlasModel.N_ACTIONS
    return int(len(rank_state_edges))


def _transition_weight(total: int, mode: str) -> float:
    total_f = max(float(total), 1.0)
    norm_mode = str(mode or "log").strip().lower()
    if norm_mode == "log":
        return float(np.log1p(total_f))
    if norm_mode == "sqrt":
        return float(np.sqrt(total_f))
    if norm_mode == "total":
        return total_f
    if norm_mode == "uniform":
        return 1.0
    raise ValueError(
        "transition_weight_mode must be one of: log, sqrt, total, uniform"
    )


def _dist_state_from_stack_distance(
    stack_distance: int,
    rank_state_edges: np.ndarray | None,
) -> int:
    if rank_state_edges is None:
        return _action_class(int(stack_distance))
    if int(stack_distance) < 0:
        return 0
    bucket = int(np.searchsorted(rank_state_edges[1:], int(stack_distance), side="right"))
    return 1 + min(bucket, len(rank_state_edges) - 2)


def _calibrated_stack_rank(
    raw_rank: int,
    *,
    stack_rank_scale: float,
    stack_rank_max: int | None,
    stack_rank_tail_pivot: int | None,
    stack_rank_tail_scale: float,
    stack_len: int,
) -> int:
    rank = int(round(max(int(raw_rank), 0) * stack_rank_scale))
    if stack_rank_tail_pivot is not None and rank > stack_rank_tail_pivot:
        tail = rank - int(stack_rank_tail_pivot)
        rank = int(stack_rank_tail_pivot) + int(round(tail * stack_rank_tail_scale))
    if stack_rank_max is not None and stack_rank_max >= 0:
        rank = min(rank, int(stack_rank_max))
    return min(max(rank, 0), int(stack_len) - 1)


def _rank_bin_index(rank: int, edges: np.ndarray) -> int:
    idx = int(np.searchsorted(edges[1:], max(int(rank), 0), side="right"))
    return min(max(idx, 0), len(edges) - 2)


def _rank_pmf_bin_over_target(
    rank: int,
    edges: np.ndarray | None,
    pmf: np.ndarray | None,
    feedback_counts: np.ndarray | None,
    *,
    feedback_alpha: float,
) -> bool:
    if edges is None or pmf is None or feedback_counts is None:
        return False
    probs = np.asarray(pmf, dtype=np.float64)
    counts = np.asarray(feedback_counts, dtype=np.float64)
    if len(edges) != len(probs) + 1 or len(counts) != len(probs):
        return False
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return False
    if abs(total - 1.0) > 1e-6:
        probs = probs / total
    raw_total = float(counts.sum())
    if not np.isfinite(raw_total) or raw_total <= 0.0:
        return False
    smooth = max(float(feedback_alpha), 0.0)
    observed = counts + smooth * probs
    observed_total = float(observed.sum())
    if not np.isfinite(observed_total) or observed_total <= 0.0:
        return False
    idx = _rank_bin_index(int(rank), edges)
    target = max(float(probs[idx]), 1e-12)
    observed_share = float(observed[idx]) / observed_total
    return observed_share > target


def _rank_pmfs(
    counts_by_state: Dict[int, np.ndarray],
    alpha: float = 1e-4,
) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for state, counts in counts_by_state.items():
        arr = np.asarray(counts, dtype=np.float64) + float(alpha)
        total = float(arr.sum())
        if total > 0.0 and np.isfinite(total):
            out[int(state)] = arr / total
    return out


def _sample_rank_pmf(
    edges: np.ndarray | None,
    pmf: np.ndarray | None,
    *,
    stack_len: int,
    bin_power: float,
    tail_bin_power: float | None,
    tail_power_pivot: int | None,
    rng: np.random.Generator,
    feedback_counts: np.ndarray | None = None,
    feedback_strength: float = 0.0,
    feedback_alpha: float = 32.0,
) -> int | None:
    if edges is None or pmf is None or stack_len <= 0:
        return None
    probs = np.asarray(pmf, dtype=np.float64)
    if len(edges) != len(probs) + 1:
        return None
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    if abs(total - 1.0) > 1e-6:
        probs = probs / total
    if (
        feedback_counts is not None
        and feedback_strength > 0.0
        and len(feedback_counts) == len(probs)
    ):
        counts = np.asarray(feedback_counts, dtype=np.float64)
        smooth = max(float(feedback_alpha), 0.0)
        observed = counts + smooth * probs
        observed_total = float(observed.sum())
        if np.isfinite(observed_total) and observed_total > 0.0:
            observed = observed / observed_total
            correction = np.power(
                probs / np.maximum(observed, 1e-12),
                max(float(feedback_strength), 0.0),
            )
            shaped = probs * correction
            shaped_total = float(shaped.sum())
            if np.isfinite(shaped_total) and shaped_total > 0.0:
                probs = shaped / shaped_total
    idx = int(rng.choice(len(probs), p=probs))
    lo = int(edges[idx])
    hi = int(edges[idx + 1]) - 1
    if lo >= int(stack_len):
        return int(stack_len) - 1
    hi = min(max(hi, lo), int(stack_len) - 1)
    if hi <= lo:
        return max(lo, 0)
    power = max(float(bin_power), 1e-6)
    if (
        tail_bin_power is not None
        and tail_power_pivot is not None
        and hi >= int(tail_power_pivot)
    ):
        power = max(float(tail_bin_power), 1e-6)
    span = hi - max(lo, 0) + 1
    offset = int(np.floor((float(rng.random()) ** (1.0 / power)) * span))
    return min(max(lo, 0) + offset, hi)


def _boosted_reuse_rank(
    *,
    stack_len: int,
    min_rank: int,
    rank_power: float,
    rng: np.random.Generator,
) -> int:
    if stack_len <= 1:
        return 0
    lo = min(max(int(min_rank), 0), int(stack_len) - 1)
    span = int(stack_len) - lo
    if span <= 1:
        return lo
    u = float(rng.random())
    offset = int(np.floor((u ** (1.0 / max(float(rank_power), 1e-6))) * span))
    return min(lo + offset, int(stack_len) - 1)


def _sample_rank_band(
    *,
    stack_len: int,
    min_rank: int,
    max_rank: int | None,
    rank_power: float,
    rng: np.random.Generator,
) -> int:
    if stack_len <= 1:
        return 0
    lo = min(max(int(min_rank), 0), int(stack_len) - 1)
    if max_rank is None:
        hi = int(stack_len) - 1
    else:
        hi = min(max(int(max_rank), lo), int(stack_len) - 1)
    span = hi - lo + 1
    if span <= 1:
        return lo
    u = float(rng.random())
    offset = int(np.floor((u ** (1.0 / max(float(rank_power), 1e-6))) * span))
    return min(lo + offset, hi)


def _sample_hot_pool_obj(
    hot_pool: Sequence[tuple[int, int]],
    *,
    weight_power: float,
    rng: np.random.Generator,
) -> int:
    if len(hot_pool) > 512 and weight_power <= 0.25:
        idx = int(rng.integers(0, len(hot_pool)))
        return int(hot_pool[idx][0])
    weights = np.array([max(int(count), 1) for _, count in hot_pool], dtype=np.float64)
    weights = np.power(weights, max(float(weight_power), 1e-6))
    weights = weights / max(float(weights.sum()), 1e-12)
    idx = int(rng.choice(len(hot_pool), p=weights))
    return int(hot_pool[idx][0])


def _object_ids_for_stack(column) -> np.ndarray:
    raw = column.to_numpy()
    if raw.size:
        dtype = raw.dtype
        if np.issubdtype(dtype, np.unsignedinteger) and int(raw.max()) > _INT64_MAX:
            return _dense_object_ids(column)
    try:
        return column.to_numpy(dtype=np.int64)
    except (OverflowError, ValueError):
        return _dense_object_ids(column)


def _dense_object_ids(column) -> np.ndarray:
    import pandas as pd

    codes, _ = pd.factorize(column, sort=False)
    return np.maximum(codes.astype(np.int64) + 1, 0)


def _bounded_new_stride(stride: int) -> int:
    stride_i = int(stride)
    if abs(stride_i) > _MAX_GENERATED_ID_BASE:
        return 0
    return stride_i


def _eligible_hot_pool(
    hot_pool: Sequence[tuple[int, int]],
    *,
    last_pos: dict[int, int],
    current_pos: int,
    min_age: int,
) -> Sequence[tuple[int, int]]:
    if min_age <= 0:
        return hot_pool
    current = int(current_pos)
    age = int(min_age)
    return [
        (int(obj_id), int(count))
        for obj_id, count in hot_pool
        if current - int(last_pos.get(int(obj_id), -age)) >= age
    ]


def _frequency_rank_keep_count(
    pool_size: int,
    min_count_rank: int,
    max_count_rank: int | None,
) -> int:
    start = max(int(min_count_rank), 0)
    size = max(int(pool_size), 1)
    if max_count_rank is None:
        return start + size
    return max(int(max_count_rank), start + 1)


def _frequency_rank_pool(
    counts: Counter[int],
    *,
    pool_size: int,
    min_count_rank: int,
    max_count_rank: int | None,
) -> list[tuple[int, int]]:
    start = max(int(min_count_rank), 0)
    stop = _frequency_rank_keep_count(pool_size, start, max_count_rank)
    ranked = counts.most_common(stop)
    pool = ranked[start:stop]
    size = max(int(pool_size), 1)
    if len(pool) > size:
        return pool[:size]
    return pool


def _rank_from_object_pool(
    stack,
    pool: Sequence[tuple[int, int]],
    *,
    last_pos: dict[int, int],
    current_pos: int,
    min_age: int,
    min_rank: int,
    max_rank: int | None,
    weight_power: float,
    max_search: int,
    sample_attempts: int,
    rng: np.random.Generator,
) -> int | None:
    if not pool:
        return None
    lo = max(int(min_rank), 0)
    hi = None if max_rank is None else max(int(max_rank), lo)
    sample_first = len(pool) > 512
    eligible = pool
    if not sample_first:
        eligible = _eligible_hot_pool(
            pool,
            last_pos=last_pos,
            current_pos=current_pos,
            min_age=min_age,
        )
        if not eligible:
            return None
    for _ in range(max(int(sample_attempts), 1)):
        obj_id = _sample_hot_pool_obj(
            eligible,
            weight_power=weight_power,
            rng=rng,
        )
        if (
            sample_first
            and min_age > 0
            and int(current_pos) - int(last_pos.get(int(obj_id), -int(min_age))) < int(min_age)
        ):
            continue
        try:
            if max_search > 0:
                rank = int(stack.index(obj_id, 0, min(len(stack), int(max_search))))
            else:
                rank = int(stack.index(obj_id))
        except ValueError:
            continue
        if rank < lo:
            continue
        if hi is not None and rank > hi:
            continue
        return rank
    return None


def _rank_from_delayed_pool(
    stack,
    ready: deque[int],
    *,
    min_rank: int,
    max_rank: int | None,
    sample_attempts: int,
) -> tuple[int | None, int]:
    if not ready:
        return None, 0
    lo = max(int(min_rank), 0)
    hi = None if max_rank is None else max(int(max_rank), lo)
    popped = 0
    for _ in range(min(max(int(sample_attempts), 1), len(ready))):
        obj_id = int(ready.popleft())
        popped += 1
        try:
            rank = int(stack.index(obj_id))
        except ValueError:
            continue
        if rank < lo:
            continue
        if hi is not None and rank > hi:
            continue
        return rank, popped
    return None, popped


class _RankedLRUNode:
    __slots__ = ("obj_id", "priority", "size", "left", "right", "parent")

    def __init__(self, obj_id: int):
        self.obj_id = int(obj_id)
        self.priority = _stable_priority(self.obj_id)
        self.size = 1
        self.left = None
        self.right = None
        self.parent = None


class _RankedLRUStack:
    """LRU stack with O(log n) rank lookup and move-to-front operations."""

    def __init__(self):
        self._root: _RankedLRUNode | None = None
        self._nodes: dict[int, _RankedLRUNode] = {}

    def __len__(self) -> int:
        return _node_size(self._root)

    def insert_front(self, obj_id: int) -> None:
        obj_id = int(obj_id)
        if obj_id in self._nodes:
            self.move_obj_to_front(obj_id)
            return
        node = _RankedLRUNode(obj_id)
        self._nodes[obj_id] = node
        self._root = _treap_merge(node, self._root)
        if self._root is not None:
            self._root.parent = None

    def index(self, obj_id: int, start: int = 0, stop: int | None = None) -> int:
        if start != 0:
            raise ValueError("_RankedLRUStack.index only supports start=0")
        node = self._nodes.get(int(obj_id))
        if node is None:
            raise ValueError(obj_id)
        rank = _node_rank(node)
        if stop is not None and rank >= int(stop):
            raise ValueError(obj_id)
        return rank

    def move_to_front(self, rank: int) -> int:
        rank = int(rank)
        size = len(self)
        if rank < 0 or rank >= size:
            raise IndexError(rank)
        if rank == 0:
            node = _node_at(self._root, 0)
            if node is None:
                raise IndexError(rank)
            return int(node.obj_id)
        left, rest = _treap_split(self._root, rank)
        node, right = _treap_split(rest, 1)
        if node is None:
            raise IndexError(rank)
        self._root = _treap_merge(node, _treap_merge(left, right))
        if self._root is not None:
            self._root.parent = None
        return int(node.obj_id)

    def move_obj_to_front(self, obj_id: int) -> None:
        self.move_to_front(self.index(obj_id))


def _node_size(node: _RankedLRUNode | None) -> int:
    return 0 if node is None else int(node.size)


def _refresh_node(node: _RankedLRUNode | None) -> None:
    if node is None:
        return
    node.size = 1 + _node_size(node.left) + _node_size(node.right)
    if node.left is not None:
        node.left.parent = node
    if node.right is not None:
        node.right.parent = node


def _treap_merge(
    left: _RankedLRUNode | None,
    right: _RankedLRUNode | None,
) -> _RankedLRUNode | None:
    if left is None:
        if right is not None:
            right.parent = None
        return right
    if right is None:
        left.parent = None
        return left
    if left.priority < right.priority:
        left.right = _treap_merge(left.right, right)
        _refresh_node(left)
        left.parent = None
        return left
    right.left = _treap_merge(left, right.left)
    _refresh_node(right)
    right.parent = None
    return right


def _treap_split(
    root: _RankedLRUNode | None,
    left_size: int,
) -> tuple[_RankedLRUNode | None, _RankedLRUNode | None]:
    if root is None:
        return None, None
    if _node_size(root.left) >= left_size:
        left, new_left = _treap_split(root.left, left_size)
        root.left = new_left
        _refresh_node(root)
        if left is not None:
            left.parent = None
        root.parent = None
        return left, root
    new_right, right = _treap_split(
        root.right,
        left_size - _node_size(root.left) - 1,
    )
    root.right = new_right
    _refresh_node(root)
    root.parent = None
    if right is not None:
        right.parent = None
    return root, right


def _node_rank(node: _RankedLRUNode) -> int:
    rank = _node_size(node.left)
    cur = node
    while cur.parent is not None:
        parent = cur.parent
        if cur is parent.right:
            rank += 1 + _node_size(parent.left)
        cur = parent
    return int(rank)


def _node_at(root: _RankedLRUNode | None, rank: int) -> _RankedLRUNode | None:
    cur = root
    rank = int(rank)
    while cur is not None:
        left_size = _node_size(cur.left)
        if rank < left_size:
            cur = cur.left
        elif rank == left_size:
            return cur
        else:
            rank -= left_size + 1
            cur = cur.right
    return None


def _stable_priority(obj_id: int) -> int:
    value = (int(obj_id) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return int(value ^ (value >> 31))


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


def _position_value(values: Sequence[float] | None, pos: int, per_stream: int, default: float) -> float:
    if not values:
        return float(default)
    idx = int(int(pos) * len(values) / max(int(per_stream), 1))
    idx = min(max(idx, 0), len(values) - 1)
    return float(values[idx])


def _position_curve_value(values: Sequence[int], pos: int, per_stream: int) -> int | None:
    if not values:
        return None
    if len(values) == 1:
        return max(int(values[0]), 0)
    progress = min(max((int(pos) + 1) / max(int(per_stream), 1), 0.0), 1.0)
    scaled = progress * (len(values) - 1)
    lo = int(np.floor(scaled))
    hi = min(lo + 1, len(values) - 1)
    frac = float(scaled - lo)
    target = (1.0 - frac) * float(values[lo]) + frac * float(values[hi])
    return max(int(round(target)), 0)


def _normalize_footprint_curves(
    curves: Sequence[Sequence[int]] | None,
    n_streams: int,
) -> list[list[int]]:
    out: list[list[int]] = []
    for idx in range(max(int(n_streams), 0)):
        if curves is None or idx >= len(curves):
            out.append([])
            continue
        curve = [max(int(x), 0) for x in curves[idx]]
        out.append(curve)
    return out


def _clip_prob_list(values: Sequence[float] | None) -> list[float]:
    if not values:
        return []
    return [float(np.clip(float(v), 0.0, 1.0)) for v in values]


def _nonnegative_float_list(values: Sequence[float] | None) -> list[float]:
    if not values:
        return []
    return [max(float(v), 0.0) for v in values]


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
