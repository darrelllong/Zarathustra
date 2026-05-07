"""LLNL ird_renewal.py — empirical IRD + heap-based renewal generator (R288 rewrite).

R286 POSTMORTEM AND FIX
-----------------------
R286 used LRU *stack distances* — expensive O(n log n) per-position rank in the
current LRU stack.  That is the wrong metric: sampling a depth in today's stack
does not reproduce tomorrow's cache pressure at small sizes.  The 0.20 MAE floor
was intrinsic to that algorithm.

CORRECT ALGORITHM (matches LANL R285 disclosure; independently derived)
------------------------------------------------------------------------
1. Read real trace; compute **position-based IRD**: for each pair of consecutive
   accesses to the same object at positions p_i < p_j, IRD = p_j - p_i (raw gap,
   not stack depth).  High-frequency objects have small IRDs; rare objects large.
2. Optionally bucket IRDs by object rank (frequency rank 0 = most frequent) so
   popular objects get their own tighter distributions.  --rank-ird-smooth blends
   sparse tail buckets with neighbors, reducing cross-seed variance.
3. Generation uses a **min-heap renewal scheduler**: when object X is emitted at
   position p, push (p + sample_ird(X), version, rank) onto the heap.  At each
   step, pop due objects (honoured with prob dependent_admit_prob), introduce new
   objects at the correct pace (controlled by new_debt_priority), or draw from a
   frequency-weighted IRM pool.
4. Per-stream mode: fit one renewal profile per stream and interleave using the
   real stream schedule.

Expected cachesim outcome (vs R286 0.20 floor):
  Wikipedia (ip=0.10, ird_s=32)      → target ≤0.01146 (LANL baseline)
  CloudPhysics (ip=0.00, ird_s=16, rb=32) → target ≤0.0267; with rb_smooth aim ≤0.025
  Meta KV                            → target ~0.011 class (LANL atlas claim)
"""
from __future__ import annotations

import argparse
import csv
import copy
import heapq
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

class IrdProfile(NamedTuple):
    source_rows: int
    counts_by_rank: np.ndarray    # int64, sorted descending (rank 0 = most frequent)
    irds: np.ndarray              # int64, all observed position-gap IRDs (≥ 1)
    ird_ranks: np.ndarray         # int64, rank of emitting object for each IRD observation
    sizes: np.ndarray             # int64, obj_size column for size sampling
    streams: np.ndarray           # int64, stream_id column

    @property
    def footprint(self) -> int:
        return int(len(self.counts_by_rank))


def fit_profile(path: str, max_rows: int = 0, ird_quantile_max: float = 1.0,
                stream_col: str | None = None) -> IrdProfile:
    rows = _read_csv(path, max_rows)
    # obj_id may be hash-keyed (uint64, > int64 max) on Wiki/Twitter/Meta KV/CDN.
    obj_ids = np.array([int(r[2]) for r in rows], dtype=np.uint64)
    obj_sizes = np.array([int(r[3]) for r in rows], dtype=np.int64)
    streams_col = 0
    if stream_col:
        idx = None
        if rows:
            header_guess = rows[0]  # rows already have header stripped
        # fall back: parse header separately
    # Try to get stream column from CSV header
    stream_arr = _read_stream_col(path, max_rows)
    return _fit_from_arrays(obj_ids, obj_sizes, stream_arr, ird_quantile_max)


def _read_csv(path: str, max_rows: int) -> list:
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # skip header
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            rows.append(row)
    return rows


def _read_stream_col(path: str, max_rows: int) -> np.ndarray:
    """Read stream_id column (col 0) from real CSV."""
    vals = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        col = 0  # stream_id is always first column
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                vals.append(int(float(row[col])))
            except (ValueError, IndexError):
                vals.append(0)
    return np.array(vals, dtype=np.int64)


def _fit_from_arrays(obj_ids: np.ndarray, obj_sizes: np.ndarray,
                     streams: np.ndarray, ird_quantile_max: float) -> IrdProfile:
    n = len(obj_ids)
    unique_ids, codes = np.unique(obj_ids, return_inverse=True)
    codes = codes.astype(np.int64)
    counts_by_code = np.bincount(codes).astype(np.int64)
    rank_to_code = np.argsort(-counts_by_code, kind="stable")
    code_to_rank = np.empty(len(rank_to_code), dtype=np.int64)
    code_to_rank[rank_to_code] = np.arange(len(rank_to_code), dtype=np.int64)
    counts_by_rank = counts_by_code[rank_to_code]

    # Position-based IRD: gap between consecutive accesses to same object.
    last = np.full(len(unique_ids), -1, dtype=np.int64)
    gaps: list[int] = []
    gap_ranks: list[int] = []
    for pos, code in enumerate(codes.tolist()):
        prev = int(last[code])
        if prev >= 0:
            gaps.append(pos - prev)
            gap_ranks.append(int(code_to_rank[code]))
        last[code] = pos

    if not gaps:
        gaps = [max(1, n)]
        gap_ranks = [0]

    irds = np.maximum(np.asarray(gaps, dtype=np.int64), 1)
    ird_ranks = np.asarray(gap_ranks, dtype=np.int64)

    if 0.0 < ird_quantile_max < 1.0 and len(irds) > 1:
        cap = max(1, int(np.quantile(irds, ird_quantile_max)))
        irds = np.minimum(irds, cap)

    return IrdProfile(
        source_rows=n,
        counts_by_rank=counts_by_rank,
        irds=irds,
        ird_ranks=ird_ranks,
        sizes=obj_sizes,
        streams=streams if streams is not None else np.zeros(n, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Rank-conditioned IRD buckets
# ---------------------------------------------------------------------------

def _rank_bucket(rank: int, footprint: int, bucket_count: int) -> int:
    if bucket_count <= 1 or footprint <= 1:
        return 0
    scaled = np.log1p(max(0, rank)) / np.log1p(max(1, footprint - 1))
    return max(0, min(bucket_count - 1, int(scaled * bucket_count)))


def _build_rank_ird_buckets(profile: IrdProfile, bucket_count: int,
                             min_samples: int, smooth: bool) -> list[np.ndarray]:
    if bucket_count <= 0:
        return []
    footprint = profile.footprint
    raw: list[list[int]] = [[] for _ in range(bucket_count)]
    for delay, rank in zip(profile.irds.tolist(), profile.ird_ranks.tolist()):
        raw[_rank_bucket(int(rank), footprint, bucket_count)].append(int(delay))
    if smooth:
        return _smooth_buckets(raw, profile.irds, min_samples)
    return [
        np.asarray(b, dtype=np.int64) if len(b) >= min_samples else profile.irds
        for b in raw
    ]


def _smooth_buckets(raw: list[list[int]], global_irds: np.ndarray,
                    min_samples: int) -> list[np.ndarray]:
    n = len(raw)
    out: list[np.ndarray] = []
    for i, bucket in enumerate(raw):
        if len(bucket) >= min_samples:
            out.append(np.asarray(bucket, dtype=np.int64))
            continue
        combined = list(bucket)
        for radius in range(1, n):
            for j in (i - radius, i + radius):
                if 0 <= j < n:
                    combined.extend(raw[j])
            if len(combined) >= min_samples:
                break
        out.append(np.asarray(combined, dtype=np.int64) if combined else global_irds)
    return out


# ---------------------------------------------------------------------------
# Generation (heap-based renewal scheduler)
# ---------------------------------------------------------------------------

def generate_ird_renewal(
    profile: IrdProfile,
    n_records: int,
    seed: int = 42,
    ird_scale: float = 1.0,
    independent_prob: float = 0.10,
    rank_ird_buckets: int = 0,
    rank_ird_min_samples: int = 256,
    rank_ird_smooth: bool = False,
    new_debt_priority: float = 0.85,
    dependent_admit_prob: float = 1.0,
    synthetic_base_id: int = 10_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic trace using empirical IRD + heap renewal scheduler.

    Returns (out_ids, out_sizes) as uint64/int64 arrays of length n_records.
    """
    rng = np.random.default_rng(seed)
    footprint = profile.footprint

    # Scale real counts to target n_records.
    total_real = int(profile.counts_by_rank.sum())
    if total_real == n_records:
        counts = profile.counts_by_rank.astype(np.int64, copy=True)
    else:
        counts = np.maximum(
            1,
            np.rint(profile.counts_by_rank.astype(np.float64) * n_records / max(float(total_real), 1.0)),
        ).astype(np.int64)
        diff = int(n_records - counts.sum())
        while diff > 0:
            ix = rng.integers(0, footprint, size=min(diff, 8192))
            np.add.at(counts, ix, 1)
            diff = int(n_records - counts.sum())
        while diff < 0:
            candidates = np.flatnonzero(counts > 1)
            if len(candidates) == 0:
                break
            ix = rng.choice(candidates, size=min(-diff, len(candidates)), replace=False)
            counts[ix] -= 1
            diff = int(n_records - counts.sum())

    remaining = counts.copy()
    total_remaining = int(remaining.sum())
    seen = np.zeros(footprint, dtype=np.bool_)
    versions = np.zeros(footprint, dtype=np.int64)
    seen_list: list[int] = []
    unique_seen = 0
    new_cursor = 0
    due_heap: list[tuple[int, int, int]] = []

    # Rank-conditioned IRD source arrays.
    bucket_arrays = _build_rank_ird_buckets(profile, rank_ird_buckets, rank_ird_min_samples, rank_ird_smooth)

    # Frequency-weighted CDF for IRM sampling.
    weights = np.maximum(counts.astype(np.float64), 1.0)
    weight_cdf = np.cumsum(weights)

    sizes_real = profile.sizes[profile.sizes > 0]
    if len(sizes_real) == 0:
        sizes_real = np.array([4096], dtype=np.int64)

    def _sample_ird(rank: int) -> int:
        source = profile.irds
        if bucket_arrays:
            bi = _rank_bucket(rank, footprint, len(bucket_arrays))
            if len(bucket_arrays[bi]) >= rank_ird_min_samples:
                source = bucket_arrays[bi]
        delay = int(source[int(rng.integers(0, len(source)))])
        delay = max(1, int(round(delay * ird_scale)))
        return delay

    def _schedule(rank: int, pos: int) -> None:
        if remaining[rank] <= 0:
            versions[rank] += 1
            return
        versions[rank] += 1
        heapq.heappush(due_heap, (pos + _sample_ird(rank), int(versions[rank]), int(rank)))

    def _peek_due(pos: int) -> int | None:
        while due_heap:
            due, ver, rank = due_heap[0]
            if remaining[rank] <= 0 or int(versions[rank]) != ver:
                heapq.heappop(due_heap)
                continue
            if due > pos:
                return None
            return rank
        return None

    def _pop_due(pos: int) -> int | None:
        r = _peek_due(pos)
        if r is None:
            return None
        heapq.heappop(due_heap)
        return r

    def _next_new() -> int | None:
        nonlocal new_cursor
        while new_cursor < footprint:
            r = int(new_cursor)
            new_cursor += 1
            if not seen[r] and remaining[r] > 0:
                return r
        return None

    def _freq_rank(allow_unseen: bool) -> int | None:
        for _ in range(32):
            r = int(np.searchsorted(weight_cdf, rng.random() * weight_cdf[-1], side="right"))
            if r >= footprint or remaining[r] <= 0:
                continue
            if allow_unseen or seen[r]:
                return r
        if seen_list:
            for _ in range(32):
                r = int(seen_list[int(rng.integers(0, len(seen_list)))])
                if remaining[r] > 0:
                    return r
        mask = (remaining > 0) & (seen if not allow_unseen else np.ones(footprint, dtype=np.bool_))
        cands = np.flatnonzero(mask)
        if len(cands) == 0:
            cands = np.flatnonzero(remaining > 0)
        if len(cands) == 0:
            return None
        return int(cands[int(rng.integers(0, len(cands)))])

    out_ids = np.empty(n_records, dtype=np.uint64)
    out_sizes = np.empty(n_records, dtype=np.int64)

    for pos in range(n_records):
        if total_remaining <= 0:
            raise RuntimeError("frequency budget exhausted before n_records")

        target_unique = min(footprint, int(round((pos + 1) * footprint / n_records)))
        rank: int | None = None
        due_rank = _peek_due(pos)

        if unique_seen < target_unique and rng.random() < new_debt_priority:
            rank = _next_new()
        if rank is None and due_rank is not None and rng.random() < dependent_admit_prob:
            rank = _pop_due(pos)
        if rank is None and unique_seen < target_unique:
            rank = _next_new()
        if rank is None and rng.random() < independent_prob:
            rank = _freq_rank(allow_unseen=(unique_seen < target_unique))
        if rank is None:
            rank = _pop_due(pos)
        if rank is None:
            rank = _freq_rank(allow_unseen=(unique_seen < target_unique))
        if rank is None:
            rank = _next_new()
        if rank is None:
            cands = np.flatnonzero(remaining > 0)
            if len(cands) == 0:
                raise RuntimeError(f"no emit-capable object at position {pos}")
            rank = int(cands[int(rng.integers(0, len(cands)))])

        if not seen[rank]:
            seen[rank] = True
            seen_list.append(rank)
            unique_seen += 1
        remaining[rank] -= 1
        total_remaining -= 1
        out_ids[pos] = np.uint64(synthetic_base_id + rank)
        out_sizes[pos] = int(sizes_real[int(rng.integers(0, len(sizes_real)))])
        _schedule(rank, pos)

    return out_ids, out_sizes


# ---------------------------------------------------------------------------
# Per-stream generation
# ---------------------------------------------------------------------------

def generate_per_stream(path: str, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    """Fit one renewal profile per stream_id and interleave by real stream schedule."""
    rows = _read_csv(path, args.max_real_rows)
    obj_ids = np.array([int(r[2]) for r in rows], dtype=np.int64)
    obj_sizes = np.array([int(r[3]) for r in rows], dtype=np.int64)
    stream_arr = _read_stream_col(path, args.max_real_rows)

    stream_ids = np.unique(stream_arr)
    # Build per-stream profiles and target record counts.
    total_per_stream = {sid: int((stream_arr == sid).sum()) for sid in stream_ids}
    profiles = {}
    target_counts = {}
    for sid in stream_ids:
        mask = stream_arr == sid
        p = _fit_from_arrays(obj_ids[mask], obj_sizes[mask], stream_arr[mask], args.ird_quantile_max)
        if p.footprint == 0:
            continue
        profiles[int(sid)] = p
        n_s = int(round(args.n * total_per_stream[int(sid)] / max(len(rows), 1)))
        target_counts[int(sid)] = max(1, n_s)

    # Adjust totals to exactly args.n
    total_assigned = sum(target_counts.values())
    if total_assigned != args.n and target_counts:
        largest = max(target_counts, key=target_counts.__getitem__)
        target_counts[largest] += args.n - total_assigned

    # Generate per stream.
    stream_fakes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid, profile in profiles.items():
        n_s = target_counts.get(sid, 0)
        if n_s <= 0:
            continue
        ids, sizes = generate_ird_renewal(
            profile, n_s, seed=args.seed,
            ird_scale=args.ird_scale,
            independent_prob=args.independent_prob,
            rank_ird_buckets=args.rank_ird_buckets,
            rank_ird_min_samples=args.rank_ird_min_samples,
            rank_ird_smooth=args.rank_ird_smooth,
            new_debt_priority=args.new_debt_priority,
            dependent_admit_prob=args.dependent_admit_prob,
        )
        stream_fakes[sid] = (ids, sizes)

    # Interleave in real stream order (tile to n_records length).
    out_ids = np.empty(args.n, dtype=np.uint64)
    out_sizes = np.empty(args.n, dtype=np.int64)
    cursors = {sid: 0 for sid in stream_fakes}
    schedule = np.tile(stream_arr, int(np.ceil(args.n / max(len(stream_arr), 1))))[:args.n]
    out_pos = 0
    for sid_val in schedule.tolist():
        sid = int(sid_val)
        if sid not in stream_fakes:
            continue
        cursor = cursors[sid]
        ids_s, sizes_s = stream_fakes[sid]
        if cursor >= len(ids_s):
            continue
        out_ids[out_pos] = ids_s[cursor]
        out_sizes[out_pos] = sizes_s[cursor]
        cursors[sid] = cursor + 1
        out_pos += 1
        if out_pos >= args.n:
            break
    # Pad if needed (should be rare).
    while out_pos < args.n:
        out_ids[out_pos] = np.uint64(10_000_000)
        out_sizes[out_pos] = int(sizes_real[0]) if len(sizes_real := np.array([4096])) > 0 else 4096
        out_pos += 1
    return out_ids, out_sizes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="LLNL IRD-renewal generator (R288 rewrite — position-based IRD + heap scheduler)")
    ap.add_argument("--real", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ird-scale", type=float, default=1.0)
    ap.add_argument("--independent-prob", type=float, default=0.10)
    ap.add_argument("--rank-ird-buckets", type=int, default=0,
                    help="Number of rank-conditioned IRD distribution buckets (0 = global only).")
    ap.add_argument("--rank-ird-min-samples", type=int, default=256,
                    help="Min samples per rank bucket before falling back to global.")
    ap.add_argument("--rank-ird-smooth", action="store_true",
                    help="Blend sparse rank buckets with neighbors to reduce seed variance.")
    ap.add_argument("--ird-quantile-max", type=float, default=1.0,
                    help="Quantile cap applied to fitted IRDs before generation (e.g. 0.99).")
    ap.add_argument("--max-real-rows", type=int, default=0,
                    help="Cap rows read from real CSV (0 = all rows).")
    ap.add_argument("--new-debt-priority", type=float, default=0.85,
                    help="Probability of introducing a new object when behind footprint pace.")
    ap.add_argument("--dependent-admit-prob", type=float, default=1.0,
                    help="Probability of honouring an overdue renewal event.")
    ap.add_argument("--per-stream", action="store_true",
                    help="Fit one renewal profile per stream_id and interleave.")
    args = ap.parse_args()

    print(f"[llgan.ird_renewal R288] Loading {args.real} (max_rows={args.max_real_rows})", flush=True)

    if args.per_stream:
        print("[llgan.ird_renewal R288] per-stream mode", flush=True)
        out_ids, out_sizes = generate_per_stream(args.real, args)
    else:
        profile = fit_profile(args.real, args.max_real_rows, args.ird_quantile_max)
        print(
            f"[llgan.ird_renewal R288] fitted rows={profile.source_rows} "
            f"footprint={profile.footprint} irds={len(profile.irds)} "
            f"ird_scale={args.ird_scale} ip={args.independent_prob} "
            f"rank_ird_buckets={args.rank_ird_buckets} smooth={args.rank_ird_smooth}",
            flush=True,
        )
        out_ids, out_sizes = generate_ird_renewal(
            profile, args.n, seed=args.seed,
            ird_scale=args.ird_scale,
            independent_prob=args.independent_prob,
            rank_ird_buckets=args.rank_ird_buckets,
            rank_ird_min_samples=args.rank_ird_min_samples,
            rank_ird_smooth=args.rank_ird_smooth,
            new_debt_priority=args.new_debt_priority,
            dependent_admit_prob=args.dependent_admit_prob,
        )

    print(f"[llgan.ird_renewal R288] Writing {len(out_ids):,} records → {args.output}", flush=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(out_ids)):
            w.writerow([0, i + 1, int(out_ids[i]), int(out_sizes[i]), 0, 0])
    print("[llgan.ird_renewal R288] done", flush=True)


if __name__ == "__main__":
    main()
