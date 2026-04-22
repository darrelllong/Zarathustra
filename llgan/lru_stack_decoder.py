"""
Post-hoc LRU stack decoder for long-rollout generation (IDEA #48).

Status
------
IMPL — applied as a post-processing step in generate.py via ``--lru-stack-decoder``.
Replaces the generated obj_id stream with one produced by an explicit LRU stack,
conditioned on:
  (a) obj_id_reuse signal from generator output (> 0 → reuse, ≤ 0 → new object)
  (b) a stack-distance bucket PMF, either fitted from a real trace or default

Motivation
----------
Short-window ★ (MMD² + β-recall) can be excellent while HRC-MAE is catastrophically
wrong (v191 HRC-MAE=0.103, LANL NeuralAtlas=0.00183). The root cause: the neural
decoder emits obj_id_reuse as a scalar signal but has no mechanism that enforces
the LRU stack law. A post-hoc decoder gives the output stream an explicit LRU
structure without retraining, letting us measure the HRC improvement directly.

Algorithm
---------
Per event (in sequence order within each stream):

  1. is_reuse = (obj_id_reuse > 0)
  2. If is_reuse and stack non-empty:
       sample stack-distance bucket from PMF
       rank = uniform draw within [bucket_lo, min(bucket_hi, stack_depth-1)]
       obj = stack[rank]
       delete stack[rank]; insert obj at front
  3. Else (new object):
       obj = next_id++
       insert obj at front
  4. Trim stack to max_stack_depth if needed.

Stack distance buckets (8 buckets):
  [0,1), [1,2), [2,4), [4,8), [8,16), [16,64), [64,256), [256,+∞)

Fitting (approximate, O(N)):
  Uses inter-reference distance (IRD) as a fast proxy for stack distance.
  For bucket-level statistics the approximation is sufficient.
  Pass ``exact=True`` to fit_from_obj_ids for BIT-based exact computation.

Performance
-----------
Stack maintained as a Python list, capped at max_stack_depth.
At max_stack_depth=2048, cost is ~O(2048) per event.
For 1M events: ~2s on modern hardware. Acceptable for generation post-processing.

References
----------
  IDEA #48 (IDEAS.md) — stateful stack-distance object decoder
  long_rollout_eval._stack_distances — exact BIT implementation (O(N log N))
  altgan/model.py — LANL StackAtlas explicit stack (del+insert, per event)
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# Stack-distance bucket edges: [EDGES[k], EDGES[k+1]) for bucket k.
_EDGES = np.array([0, 1, 2, 4, 8, 16, 64, 256, 1 << 20], dtype=np.int64)
N_BUCKETS = len(_EDGES) - 1  # 8 buckets

# Default PMFs fitted from real corpus long-rollout eval
# (aggregated from stack_distance_histogram/bin_edges in long_rollout JSON).
#
# Alibaba (source: alibaba_v195/long_rollout_epoch_0110.json real section,
# stack_distance_median=174, p90=577): distribution concentrated in
# [64,256) and [256+) buckets — matches heavy-tailed LRU stack law.
#
# Tencent (source: v165 long-rollout real section, stack_distance_median~60):
# slightly shorter-range, more mass in mid-range buckets.
_DEFAULT_PMFS = {
    "alibaba": np.array([0.0000, 0.0015, 0.0123, 0.0468, 0.0474, 0.1470, 0.4842, 0.2608]),
    "tencent": np.array([0.0000, 0.0050, 0.0200, 0.0600, 0.0850, 0.2200, 0.4100, 0.2000]),
}


class LRUStackDecoder:
    """
    Stateful LRU stack decoder for one trace stream.

    Parameters
    ----------
    bucket_pmf        : (8,) probability vector over stack-distance buckets
    max_stack_depth   : cap on LRU stack size (objects beyond depth are evicted)
    seed              : RNG seed for reproducibility
    transition_matrix : optional (8, 8) Markov transition matrix (IDEA #62, unused)
    cond_pmf          : optional (N, 8) conditional PMF P(bucket | dt_bin)
                        (IDEA #63 StackAtlas Lite). When supplied with dt_edges,
                        bucket is sampled from cond_pmf[dt_bin] instead of the
                        global PMF.
    dt_edges          : optional (N-1,) log1p-dt quantile bin edges for cond_pmf.
    """

    def __init__(
        self,
        bucket_pmf: np.ndarray,
        max_stack_depth: int = 2048,
        seed: Optional[int] = None,
        transition_matrix: Optional[np.ndarray] = None,
        cond_pmf: Optional[np.ndarray] = None,
        dt_edges: Optional[np.ndarray] = None,
    ):
        self.bucket_pmf = np.asarray(bucket_pmf, dtype=np.float64)
        self.bucket_pmf /= self.bucket_pmf.sum()
        self.max_stack_depth = max_stack_depth
        self.rng = np.random.default_rng(seed)
        self._stack: list[int] = []
        self._next_id: int = 0
        self._prev_bucket: int = -1  # -1 = no prior reuse event yet (cold start)
        self.markov_blend: float = 1.0  # 0=pure pmf, 1=pure markov; adjustable
        if transition_matrix is not None:
            T = np.asarray(transition_matrix, dtype=np.float64)
            # Normalise each row; add tiny uniform prior to avoid zero rows.
            T += 1e-8
            self.transition_matrix: Optional[np.ndarray] = T / T.sum(axis=1, keepdims=True)
        else:
            self.transition_matrix = None
        # IDEA #63: time-conditioned PMF
        if cond_pmf is not None and dt_edges is not None:
            C = np.asarray(cond_pmf, dtype=np.float64)
            C += 1e-10  # small prior to avoid zero rows
            self.cond_pmf: Optional[np.ndarray] = C / C.sum(axis=1, keepdims=True)
            self.dt_edges: Optional[np.ndarray] = np.asarray(dt_edges, dtype=np.float64)
        else:
            self.cond_pmf = None
            self.dt_edges = None

    def reset(self) -> None:
        self._stack = []
        self._next_id = 0
        self._prev_bucket = -1

    def step(self, is_reuse: bool) -> int:
        """Process one event. Returns assigned obj_id."""
        if is_reuse and self._stack:
            # IDEA #62: Markov atlas — condition on previous bucket when available.
            if self.transition_matrix is not None and self._prev_bucket >= 0:
                # Blend: (1-markov_blend)*pmf + markov_blend*T[prev]
                pmf = ((1.0 - self.markov_blend) * self.bucket_pmf
                       + self.markov_blend * self.transition_matrix[self._prev_bucket])
            else:
                pmf = self.bucket_pmf
            bucket = int(self.rng.choice(N_BUCKETS, p=pmf))
            self._prev_bucket = bucket
            lo = int(_EDGES[bucket])
            hi = int(_EDGES[bucket + 1]) - 1
            max_rank = len(self._stack) - 1
            # Clamp lo so we never request a rank beyond stack depth.
            lo = min(lo, max_rank)
            hi = min(hi, max_rank)
            rank = int(self.rng.integers(lo, hi + 1))
            obj = self._stack[rank]
            del self._stack[rank]
        else:
            self._prev_bucket = -1  # cold miss resets Markov state
            obj = self._next_id
            self._next_id += 1
        self._stack.insert(0, obj)
        if len(self._stack) > self.max_stack_depth:
            self._stack.pop()
        return obj

    def step_dt(self, is_reuse: bool, dt_log1p: float) -> int:
        """Like step() but uses conditional PMF P(bucket | dt_bin) if set."""
        if is_reuse and self._stack:
            if self.cond_pmf is not None and self.dt_edges is not None:
                dt_bin = int(np.searchsorted(self.dt_edges, dt_log1p, side="right"))
                dt_bin = max(0, min(dt_bin, len(self.cond_pmf) - 1))
                pmf = self.cond_pmf[dt_bin]
            elif self.transition_matrix is not None and self._prev_bucket >= 0:
                pmf = ((1.0 - self.markov_blend) * self.bucket_pmf
                       + self.markov_blend * self.transition_matrix[self._prev_bucket])
            else:
                pmf = self.bucket_pmf
            bucket = int(self.rng.choice(N_BUCKETS, p=pmf))
            self._prev_bucket = bucket
            lo = int(_EDGES[bucket])
            hi = int(_EDGES[bucket + 1]) - 1
            max_rank = len(self._stack) - 1
            lo = min(lo, max_rank)
            hi = min(hi, max_rank)
            rank = int(self.rng.integers(lo, hi + 1))
            obj = self._stack[rank]
            del self._stack[rank]
        else:
            self._prev_bucket = -1
            obj = self._next_id
            self._next_id += 1
        self._stack.insert(0, obj)
        if len(self._stack) > self.max_stack_depth:
            self._stack.pop()
        return obj

    def decode_stream(self, reuse_signal: np.ndarray,
                      dt_log1p: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Parameters
        ----------
        reuse_signal : (N,) float — > 0 → reuse, ≤ 0 → new object
        dt_log1p     : (N,) float — log1p(interarrival_time) for conditional PMF;
                       if None, uses global PMF or Markov chain.

        Returns
        -------
        (N,) int64 array of assigned obj_ids
        """
        out = np.empty(len(reuse_signal), dtype=np.int64)
        if dt_log1p is not None and len(dt_log1p) == len(reuse_signal):
            for i in range(len(reuse_signal)):
                out[i] = self.step_dt(float(reuse_signal[i]) > 0.0,
                                      float(dt_log1p[i]))
        else:
            for i in range(len(reuse_signal)):
                out[i] = self.step(float(reuse_signal[i]) > 0.0)
        return out

    # ------------------------------------------------------------------
    # Factory / fitting
    # ------------------------------------------------------------------

    @classmethod
    def from_default(
        cls, corpus: str = "alibaba", **kwargs
    ) -> "LRUStackDecoder":
        """Construct from pre-calibrated defaults when no real trace available."""
        key = corpus.lower()
        pmf = _DEFAULT_PMFS.get(key, _DEFAULT_PMFS["alibaba"]).copy()
        return cls(pmf, **kwargs)

    @classmethod
    def fit_from_obj_ids(
        cls,
        obj_ids: np.ndarray,
        max_fit_events: int = 300_000,
        exact: bool = False,
        **kwargs,
    ) -> "LRUStackDecoder":
        """
        Fit bucket PMF from a real obj_id sequence.

        Parameters
        ----------
        obj_ids         : integer array of object IDs from a real trace
        max_fit_events  : subsample to this many events (speed vs accuracy)
        exact           : use BIT-based exact stack distances (slower, more accurate)
        """
        arr = np.asarray(obj_ids, dtype=np.int64)
        if len(arr) > max_fit_events:
            arr = arr[:max_fit_events]
        if exact:
            dists = _stack_distances_bit(arr)
            counts = np.zeros(N_BUCKETS, dtype=np.int64)
            if len(dists):
                buckets = np.searchsorted(_EDGES[1:], dists, side="right")
                buckets = np.clip(buckets, 0, N_BUCKETS - 1)
                np.add.at(counts, buckets, 1)
        else:
            counts = _fit_ird_counts(arr)
        pmf = (counts + 1.0) / (counts.sum() + N_BUCKETS)
        return cls(pmf, **kwargs)

    @classmethod
    def fit_from_df(
        cls,
        df,
        obj_col: str = "obj_id",
        stream_col: Optional[str] = "stream_id",
        max_fit_events: int = 300_000,
        exact: bool = False,
        **kwargs,
    ) -> "LRUStackDecoder":
        """
        Fit from a pandas DataFrame. If stream_col exists, concatenates
        obj_ids across all streams (up to max_fit_events total).
        """
        if stream_col is not None and stream_col in df.columns:
            arr = df.sort_values([stream_col, df.index.name or "index"]
                                  if df.index.name else stream_col)[obj_col].values
        else:
            arr = df[obj_col].values
        return cls.fit_from_obj_ids(arr, max_fit_events=max_fit_events,
                                    exact=exact, **kwargs)

    @classmethod
    def from_cond_pmf(
        cls,
        cond_pmf: np.ndarray,
        dt_edges: np.ndarray,
        corpus: str = "alibaba",
        **kwargs,
    ) -> "LRUStackDecoder":
        """Construct with IDEA #63 time-conditioned PMF; marginal from corpus default."""
        pmf = _DEFAULT_PMFS.get(corpus.lower(), _DEFAULT_PMFS["alibaba"]).copy()
        return cls(pmf, cond_pmf=cond_pmf, dt_edges=dt_edges, **kwargs)

    @classmethod
    def from_markov_matrix(
        cls,
        transition_matrix: np.ndarray,
        corpus: str = "alibaba",
        **kwargs,
    ) -> "LRUStackDecoder":
        """Construct with Markov atlas; marginal PMF from corpus default."""
        pmf = _DEFAULT_PMFS.get(corpus.lower(), _DEFAULT_PMFS["alibaba"]).copy()
        return cls(pmf, transition_matrix=transition_matrix, **kwargs)

    @classmethod
    def fit_transition_matrix_from_obj_ids(
        cls,
        obj_ids: np.ndarray,
        max_fit_events: int = 1_000_000,
        exact: bool = False,
    ) -> np.ndarray:
        """
        Compute 8×8 Markov transition matrix from a real obj_id sequence.
        Returns T[i][j] = P(next_bucket=j | prev_bucket=i), not normalized.
        Caller normalizes or passes to from_markov_matrix().
        """
        arr = np.asarray(obj_ids, dtype=np.int64)
        if len(arr) > max_fit_events:
            arr = arr[:max_fit_events]

        T = np.zeros((N_BUCKETS, N_BUCKETS), dtype=np.float64)
        last_pos: dict[int, int] = {}
        prev_bucket: int = -1
        for i, oid in enumerate(arr.tolist()):
            oid = int(oid)
            if oid in last_pos:
                dist = i - last_pos[oid]
                b = int(np.searchsorted(_EDGES[1:], dist, side="right"))
                b = min(b, N_BUCKETS - 1)
                if prev_bucket >= 0:
                    T[prev_bucket][b] += 1
                prev_bucket = b
            else:
                prev_bucket = -1  # cold miss resets chain
            last_pos[oid] = i
        return T

    def print_pmf(self) -> None:
        """Display fitted bucket distribution."""
        labels = [f"[{_EDGES[k]},{_EDGES[k+1]})" for k in range(N_BUCKETS)]
        labels[-1] = f"[{_EDGES[-2]},+∞)"
        for label, p in zip(labels, self.bucket_pmf):
            bar = "█" * int(p * 40)
            print(f"  {label:>12s}  {p:.3f}  {bar}")

    def print_transition_matrix(self) -> None:
        """Display the Markov transition matrix if set."""
        if self.transition_matrix is None:
            print("  [no Markov atlas — using i.i.d. PMF]")
            return
        labels = [f"[{_EDGES[k]},{_EDGES[k+1]})" for k in range(N_BUCKETS)]
        labels[-1] = f"[{_EDGES[-2]},+∞)"
        print(f"  {'':>12s}  " + "  ".join(f"{l:>12s}" for l in labels))
        for i, row in enumerate(self.transition_matrix):
            vals = "  ".join(f"{v:.3f}" for v in row)
            print(f"  {labels[i]:>12s}  {vals}")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _fit_ird_counts(obj_ids: np.ndarray) -> np.ndarray:
    """
    Approximate stack-distance bucket counts via IRD (inter-reference distance).
    IRD = raw access count between two references to the same object (not distinct).
    This is an O(N) proxy; overestimates distances slightly vs true stack distance.
    """
    counts = np.zeros(N_BUCKETS, dtype=np.int64)
    last_pos: dict[int, int] = {}
    for i, oid in enumerate(obj_ids.tolist()):
        if oid in last_pos:
            ird = i - last_pos[oid]
            k = int(np.searchsorted(_EDGES[1:], ird, side="right"))
            k = min(k, N_BUCKETS - 1)
            counts[k] += 1
        last_pos[oid] = i
    return counts


def _stack_distances_bit(obj_ids: np.ndarray) -> np.ndarray:
    """
    Exact O(N log N) stack distances via Fenwick tree.
    Mirrors long_rollout_eval._stack_distances — kept here to avoid circular import.
    Returns distances only for reuse events (cold misses excluded).
    """
    N = len(obj_ids)
    if N == 0:
        return np.zeros(0, dtype=np.int64)

    bit: list[int] = [0] * (N + 2)

    def update(idx: int, delta: int) -> None:
        idx += 1
        while idx <= N:
            bit[idx] += delta
            idx += idx & -idx

    def prefix(idx: int) -> int:
        idx += 1
        s = 0
        while idx > 0:
            s += bit[idx]
            idx -= idx & -idx
        return s

    last: dict[int, int] = {}
    out: list[int] = []
    active = 0
    for i, oid in enumerate(obj_ids.tolist()):
        key = int(oid)
        if key in last:
            prev = last[key]
            sd = active - prefix(prev)
            out.append(sd)
            update(prev, -1)
            active -= 1
        last[key] = i
        update(i, 1)
        active += 1

    return np.asarray(out, dtype=np.int64) if out else np.zeros(0, dtype=np.int64)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N = 50_000

    # Synthetic obj_id sequence: 80% reuse from a 500-object working set,
    # 20% new objects. Stack distances roughly geometric.
    working_set = list(range(500))
    obj_seq = []
    for _ in range(N):
        if rng.random() < 0.75 and working_set:
            obj_seq.append(int(rng.choice(working_set)))
        else:
            new = len(working_set)
            working_set.append(new)
            obj_seq.append(new)
    obj_ids = np.array(obj_seq, dtype=np.int64)

    # Fit from sequence
    dec = LRUStackDecoder.fit_from_obj_ids(obj_ids, exact=True)
    print("Fitted PMF (exact BIT):")
    dec.print_pmf()

    dec_ird = LRUStackDecoder.fit_from_obj_ids(obj_ids, exact=False)
    print("Fitted PMF (IRD approx):")
    dec_ird.print_pmf()

    # Decode a synthetic reuse signal
    reuse_signal = np.where(rng.random(1000) < 0.65, 1.0, -1.0)
    ids = dec.decode_stream(reuse_signal)
    print(f"\ndecoded 1000 events: {(reuse_signal > 0).sum()} reuses, "
          f"{len(np.unique(ids))} unique objects")

    # Test defaults
    a = LRUStackDecoder.from_default("alibaba")
    t = LRUStackDecoder.from_default("tencent")
    print(f"alibaba default PMF sum={a.bucket_pmf.sum():.4f}")
    print(f"tencent default PMF sum={t.bucket_pmf.sum():.4f}")
    print("lru_stack_decoder smoke test OK")
