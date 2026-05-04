"""Generate cache-shaped traces from empirical IRD + IRM renewal profiles.

This is LANL's 2DIO-inspired non-atlas path: learn recency from real
inter-reference distances, learn long-run frequency from object counts, then
emit a synthetic object stream through a heap of scheduled dependent arrivals
mixed with independent frequency arrivals. Object labels are synthetic ranks,
not copied real IDs.
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class RenewalProfile:
    source_rows: int
    counts_by_rank: np.ndarray
    new_order: np.ndarray
    irds: np.ndarray
    sizes: np.ndarray
    opcodes: np.ndarray
    tenants: np.ndarray
    streams: np.ndarray

    @property
    def footprint(self) -> int:
        return int(len(self.counts_by_rank))

    @property
    def unique_fraction(self) -> float:
        return self.footprint / max(float(self.source_rows), 1.0)

    @property
    def reuse_fraction(self) -> float:
        return 1.0 - self.unique_fraction


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real", required=True, help="Official real CSV reference.")
    p.add_argument("--output", required=True, help="Synthetic CSV output path.")
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fit-rows", type=int, default=0,
                   help="Rows to fit from the real CSV; 0 means all rows.")
    p.add_argument("--synthetic-base-id", type=int, default=10_000_000)
    p.add_argument("--independent-prob", type=float, default=0.10,
                   help="Probability of an IRM frequency arrival when no new object is due.")
    p.add_argument("--dependent-admit-prob", type=float, default=1.0,
                   help="Probability of honoring an overdue renewal event.")
    p.add_argument("--new-debt-priority", type=float, default=0.85,
                   help="Probability of emitting a new object when below target footprint pace.")
    p.add_argument("--frequency-alpha", type=float, default=1.0,
                   help="Exponent applied to empirical object-count weights for IRM sampling.")
    p.add_argument("--ird-scale", type=float, default=1.0)
    p.add_argument("--ird-tail-pivot", type=int, default=0,
                   help="If >0, only IRDs above this raw distance get tail scaling.")
    p.add_argument("--ird-tail-scale", type=float, default=1.0,
                   help="Multiplier for raw distance above --ird-tail-pivot.")
    p.add_argument("--ird-jitter", type=float, default=0.0,
                   help="Uniform multiplicative jitter half-width for sampled IRDs.")
    p.add_argument("--ird-min", type=int, default=1)
    p.add_argument("--ird-max", type=int, default=0,
                   help="Hard cap for sampled IRDs; 0 disables the cap.")
    p.add_argument("--ird-quantile-max", type=float, default=1.0,
                   help="Quantile cap for fitted IRDs before generation.")
    p.add_argument("--mark-mode", choices=("sequence", "sample"), default="sequence")
    p.add_argument("--preserve-streams", action="store_true",
                   help="Use fitted stream IDs. Default stream_id=0 keeps synthetic IDs global.")
    p.add_argument("--zero-ts", action="store_true",
                   help="Write ts=0 for every row instead of monotone synthetic ticks.")
    p.add_argument("--progress-interval", type=int, default=200_000)
    return p.parse_args()


def fit_profile(path: Path, fit_rows: int = 0, ird_quantile_max: float = 1.0) -> RenewalProfile:
    nrows = fit_rows if fit_rows and fit_rows > 0 else None
    df = pd.read_csv(path, nrows=nrows)
    if "obj_id" not in df.columns:
        raise ValueError(f"{path} is missing obj_id")
    codes, _ = pd.factorize(df["obj_id"], sort=False)
    codes = codes.astype(np.int64, copy=False)
    counts_by_code = np.bincount(codes)
    rank_to_code = np.argsort(-counts_by_code, kind="stable")
    code_to_rank = np.empty(len(rank_to_code), dtype=np.int64)
    code_to_rank[rank_to_code] = np.arange(len(rank_to_code), dtype=np.int64)
    counts_by_rank = counts_by_code[rank_to_code].astype(np.int64, copy=False)
    new_order = code_to_rank[np.arange(len(counts_by_code), dtype=np.int64)]

    last = np.full(len(counts_by_code), -1, dtype=np.int64)
    gaps: list[int] = []
    for pos, code in enumerate(codes):
        prev = int(last[code])
        if prev >= 0:
            gaps.append(pos - prev)
        last[code] = pos
    if not gaps:
        gaps = [max(1, len(codes))]
    irds = np.asarray(gaps, dtype=np.int64)
    if 0.0 < ird_quantile_max < 1.0 and len(irds) > 1:
        cap = max(1, int(np.quantile(irds, ird_quantile_max)))
        irds = np.minimum(irds, cap)

    return RenewalProfile(
        source_rows=int(len(df)),
        counts_by_rank=counts_by_rank,
        new_order=new_order.astype(np.int64, copy=False),
        irds=np.maximum(irds, 1),
        sizes=_column_or_default(df, "obj_size", 1).astype(np.int64, copy=False),
        opcodes=_column_or_default(df, "opcode", 0).astype(np.int64, copy=False),
        tenants=_column_or_default(df, "tenant", 0).astype(np.int64, copy=False),
        streams=_column_or_default(df, "stream_id", 0).astype(np.int64, copy=False),
    )


def generate(profile: RenewalProfile, args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    counts = _scaled_counts(profile.counts_by_rank, args.n_records, rng)
    footprint = int(len(counts))
    remaining = counts.copy()
    seen = np.zeros(footprint, dtype=np.bool_)
    versions = np.zeros(footprint, dtype=np.int64)
    new_order = _scaled_new_order(profile.new_order, footprint, rng)
    new_cursor = 0
    seen_list: list[int] = []
    due_heap: list[tuple[int, int, int]] = []

    weights = np.power(np.maximum(counts.astype(np.float64), 1.0), args.frequency_alpha)
    weight_cdf = np.cumsum(weights)
    if not np.isfinite(weight_cdf[-1]) or weight_cdf[-1] <= 0:
        weight_cdf = np.arange(1, footprint + 1, dtype=np.float64)

    obj_out = np.empty(args.n_records, dtype=np.uint64)
    total_remaining = int(remaining.sum())
    unique_seen = 0

    def sample_ird() -> int:
        delay = int(profile.irds[int(rng.integers(0, len(profile.irds)))])
        if args.ird_tail_pivot > 0 and delay > args.ird_tail_pivot:
            pivot = int(args.ird_tail_pivot)
            delay = pivot + int(round((delay - pivot) * args.ird_tail_scale))
        if args.ird_jitter > 0:
            lo = max(0.0, 1.0 - args.ird_jitter)
            hi = 1.0 + args.ird_jitter
            delay = int(round(delay * float(rng.uniform(lo, hi))))
        delay = int(round(delay * args.ird_scale))
        delay = max(int(args.ird_min), delay)
        if args.ird_max > 0:
            delay = min(delay, int(args.ird_max))
        return max(1, delay)

    def schedule(rank: int, pos: int) -> None:
        if remaining[rank] <= 0:
            versions[rank] += 1
            return
        versions[rank] += 1
        heapq.heappush(due_heap, (pos + sample_ird(), int(versions[rank]), rank))

    def peek_due(pos: int) -> int | None:
        while due_heap:
            due, version, rank = due_heap[0]
            if remaining[rank] <= 0 or int(versions[rank]) != version:
                heapq.heappop(due_heap)
                continue
            if due > pos:
                return None
            return rank
        return None

    def pop_due(pos: int) -> int | None:
        rank = peek_due(pos)
        if rank is None:
            return None
        heapq.heappop(due_heap)
        return rank

    def next_new_rank() -> int | None:
        nonlocal new_cursor
        while new_cursor < len(new_order):
            rank = int(new_order[new_cursor])
            new_cursor += 1
            if rank < footprint and (not seen[rank]) and remaining[rank] > 0:
                return rank
        return None

    def sample_frequency_rank(allow_unseen: bool) -> int | None:
        for _ in range(32):
            rank = int(np.searchsorted(weight_cdf, rng.random() * weight_cdf[-1], side="right"))
            if rank >= footprint or remaining[rank] <= 0:
                continue
            if allow_unseen or seen[rank]:
                return rank
        for _ in range(32):
            if not seen_list:
                break
            rank = int(seen_list[int(rng.integers(0, len(seen_list)))])
            if remaining[rank] > 0:
                return rank
        return fallback_remaining_rank(allow_unseen=allow_unseen)

    def fallback_remaining_rank(allow_unseen: bool) -> int | None:
        if allow_unseen:
            candidates = np.flatnonzero(remaining > 0)
        else:
            candidates = np.flatnonzero((remaining > 0) & seen)
            if len(candidates) == 0:
                candidates = np.flatnonzero(remaining > 0)
        if len(candidates) == 0:
            return None
        return int(candidates[int(rng.integers(0, len(candidates)))])

    for pos in range(args.n_records):
        if total_remaining <= 0:
            raise RuntimeError("frequency budget exhausted before n_records")
        target_unique = min(footprint, int(round((pos + 1) * footprint / args.n_records)))
        rank: int | None = None
        due_rank = peek_due(pos)

        if unique_seen < target_unique and rng.random() < args.new_debt_priority:
            rank = next_new_rank()
        if rank is None and due_rank is not None and rng.random() < args.dependent_admit_prob:
            rank = pop_due(pos)
        if rank is None and unique_seen < target_unique:
            rank = next_new_rank()
        if rank is None and rng.random() < args.independent_prob:
            rank = sample_frequency_rank(allow_unseen=unique_seen < target_unique)
        if rank is None:
            rank = pop_due(pos)
        if rank is None:
            rank = sample_frequency_rank(allow_unseen=unique_seen < target_unique)
        if rank is None:
            rank = next_new_rank()
        if rank is None:
            rank = fallback_remaining_rank(allow_unseen=True)
        if rank is None or remaining[rank] <= 0:
            raise RuntimeError(f"no emit-capable object at position {pos}")

        if not seen[rank]:
            seen[rank] = True
            seen_list.append(rank)
            unique_seen += 1
        remaining[rank] -= 1
        total_remaining -= 1
        obj_out[pos] = np.uint64(args.synthetic_base_id + rank)
        schedule(rank, pos)

        if args.progress_interval > 0 and (pos + 1) % args.progress_interval == 0:
            print(
                "[altgan.ird_renewal] "
                f"pos={pos + 1}/{args.n_records} unique={unique_seen}/{footprint} "
                f"pending={len(due_heap)} remaining={total_remaining}",
                flush=True,
            )

    marks = _make_marks(profile, args, rng)
    df = pd.DataFrame(
        {
            "stream_id": marks["stream_id"],
            "ts": np.zeros(args.n_records, dtype=np.int64)
            if args.zero_ts
            else np.arange(args.n_records, dtype=np.int64),
            "obj_id": obj_out,
            "obj_size": marks["obj_size"],
            "opcode": marks["opcode"],
            "tenant": marks["tenant"],
        }
    )
    return df


def _column_or_default(df: pd.DataFrame, name: str, default: int) -> np.ndarray:
    if name in df.columns:
        return df[name].fillna(default).to_numpy()
    return np.full(len(df), default)


def _scaled_counts(source_counts: np.ndarray, n_records: int, rng: np.random.Generator) -> np.ndarray:
    if int(source_counts.sum()) == n_records:
        return source_counts.astype(np.int64, copy=True)
    unique_fraction = len(source_counts) / max(float(source_counts.sum()), 1.0)
    footprint = max(1, min(n_records, int(round(unique_fraction * n_records))))
    if footprint <= len(source_counts):
        counts = source_counts[:footprint].astype(np.float64, copy=True)
    else:
        extra = rng.choice(source_counts, size=footprint - len(source_counts), replace=True)
        counts = np.concatenate([source_counts.astype(np.float64), extra.astype(np.float64)])
    counts = np.maximum(1, np.rint(counts * (n_records / max(float(counts.sum()), 1.0)))).astype(np.int64)
    diff = int(n_records - counts.sum())
    while diff > 0:
        ix = rng.integers(0, len(counts), size=min(diff, 8192))
        np.add.at(counts, ix, 1)
        diff = int(n_records - counts.sum())
    while diff < 0:
        candidates = np.flatnonzero(counts > 1)
        if len(candidates) == 0:
            break
        ix = rng.choice(candidates, size=min(-diff, len(candidates)), replace=False)
        counts[ix] -= 1
        diff = int(n_records - counts.sum())
    return counts.astype(np.int64, copy=False)


def _scaled_new_order(source_order: np.ndarray, footprint: int, rng: np.random.Generator) -> np.ndarray:
    if footprint <= len(source_order):
        order = source_order[source_order < footprint]
        if len(order) == footprint:
            return order.astype(np.int64, copy=True)
    order = np.arange(footprint, dtype=np.int64)
    rng.shuffle(order)
    return order


def _make_marks(profile: RenewalProfile, args: argparse.Namespace, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = int(args.n_records)
    if args.mark_mode == "sample":
        idx = rng.integers(0, profile.source_rows, size=n)
    else:
        idx = np.arange(n) % profile.source_rows
    streams = profile.streams[idx].astype(np.int64, copy=True) if args.preserve_streams else np.zeros(n, dtype=np.int64)
    return {
        "stream_id": streams,
        "obj_size": profile.sizes[idx].astype(np.int64, copy=True),
        "opcode": profile.opcodes[idx].astype(np.int64, copy=True),
        "tenant": profile.tenants[idx].astype(np.int64, copy=True),
    }


def main() -> int:
    args = _parse_args()
    profile = fit_profile(Path(args.real), args.fit_rows, args.ird_quantile_max)
    print(
        "[altgan.ird_renewal] fitted "
        f"rows={profile.source_rows} footprint={profile.footprint} "
        f"reuse_fraction={profile.reuse_fraction:.6f} irds={len(profile.irds)}",
        flush=True,
    )
    fake = generate(profile, args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fake.to_csv(out, index=False)
    print(f"[altgan.ird_renewal] wrote {out} rows={len(fake)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
