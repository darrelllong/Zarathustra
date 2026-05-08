"""Audit cache-size contributions in existing ``llgan.cachesim_eval`` JSONs.

The race surface is the official fixed cache grid, but very small capacities
can be low-signal on huge working sets.  This helper reads already-produced
cachesim JSON reports and derives per-cache MAE plus a "drop cache 32" mean
without rerunning the simulator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurfaceAudit:
    path: Path
    label: str
    cache_sizes: list[int]
    official_mean: float
    dropped_mean: float
    dropped_caches: tuple[int, ...]
    per_cache_mae: list[float]
    worst_cache: int
    worst_cache_mae: float

    @property
    def drop_delta(self) -> float:
        return self.official_mean - self.dropped_mean


def _parse_drop_caches(text: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _restricted_policy_mean(deltas: list[float], indices: list[int]) -> float:
    vals = [abs(float(deltas[i])) for i in indices if i < len(deltas)]
    return _mean(vals)


def audit_report(path: Path, *, label: str | None, drop_caches: tuple[int, ...]) -> SurfaceAudit:
    with path.open() as f:
        report = json.load(f)

    cache_sizes = [int(v) for v in report["cache_sizes"]]
    by_policy = report["by_policy"]
    official_mean = float(report.get("mean_hrc_mae", report.get("mean")))

    all_indices = list(range(len(cache_sizes)))
    kept_indices = [i for i, size in enumerate(cache_sizes) if size not in drop_caches]
    if not kept_indices:
        kept_indices = all_indices

    per_cache_mae: list[float] = []
    for i in all_indices:
        per_cache_mae.append(
            _mean([abs(float(policy["delta"][i])) for policy in by_policy.values() if i < len(policy["delta"])])
        )

    dropped_mean = _mean(
        [_restricted_policy_mean(policy["delta"], kept_indices) for policy in by_policy.values()]
    )
    worst_idx = max(range(len(per_cache_mae)), key=lambda i: per_cache_mae[i])
    return SurfaceAudit(
        path=path,
        label=label or path.stem,
        cache_sizes=cache_sizes,
        official_mean=official_mean,
        dropped_mean=dropped_mean,
        dropped_caches=drop_caches,
        per_cache_mae=per_cache_mae,
        worst_cache=cache_sizes[worst_idx],
        worst_cache_mae=per_cache_mae[worst_idx],
    )


def _print_text(rows: list[SurfaceAudit]) -> None:
    drop_label = ",".join(str(c) for c in rows[0].dropped_caches) if rows else ""
    print(
        f"{'label':<34} {'official':>10} {'drop[' + drop_label + ']':>12} "
        f"{'delta':>10} {'worst_cap':>9} {'worst_mae':>10}"
    )
    for row in rows:
        print(
            f"{row.label:<34} {row.official_mean:10.10f} {row.dropped_mean:12.10f} "
            f"{row.drop_delta:10.10f} {row.worst_cache:9d} {row.worst_cache_mae:10.10f}"
        )
    if len(rows) > 1:
        print()
        print(
            f"{'mean':<34} {_mean([r.official_mean for r in rows]):10.10f} "
            f"{_mean([r.dropped_mean for r in rows]):12.10f} "
            f"{_mean([r.drop_delta for r in rows]):10.10f}"
        )


def _print_markdown(rows: list[SurfaceAudit]) -> None:
    drop_label = ",".join(str(c) for c in rows[0].dropped_caches) if rows else ""
    print("| label | official mean | drop-cache mean | official-minus-drop | worst cache | worst-cache MAE |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.label} | {row.official_mean:.10f} | {row.dropped_mean:.10f} | "
            f"{row.drop_delta:.10f} | {row.worst_cache} | {row.worst_cache_mae:.10f} |"
        )
    if len(rows) > 1:
        print(
            f"| mean | {_mean([r.official_mean for r in rows]):.10f} | "
            f"{_mean([r.dropped_mean for r in rows]):.10f} | "
            f"{_mean([r.drop_delta for r in rows]):.10f} |  |  |"
        )
    print()
    print(f"Drop-cache mean excludes cache size(s) `{drop_label}` from each policy before averaging.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", help="cachesim_eval JSON path(s).")
    p.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated labels, aligned with report paths.",
    )
    p.add_argument("--drop-caches", default="32", help="Comma-separated cache sizes to exclude.")
    p.add_argument("--markdown", action="store_true", help="Emit a markdown table.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    labels = [part.strip() for part in args.labels.split(",") if part.strip()]
    drop_caches = _parse_drop_caches(args.drop_caches)
    rows = [
        audit_report(
            Path(report),
            label=labels[i] if i < len(labels) else None,
            drop_caches=drop_caches,
        )
        for i, report in enumerate(args.reports)
    ]
    if args.markdown:
        _print_markdown(rows)
    else:
        _print_text(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
