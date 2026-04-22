"""Render neural-mark hybrid sweep summaries as markdown.

``altgan.sweep_mark_hybrids`` writes a CSV for every evaluated cell and a JSON
file with candidate means when multiple seeds are used. This helper turns those
artifacts into a compact ``RESULTS.md`` section so finished remote sweeps can
be evaluated without hand-written tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--best-json", default="",
                   help="Optional sweep best JSON; enables seed-mean table.")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--mean-top-n", type=int, default=5)
    p.add_argument("--title", default="Neural Mark Hybrid Sweep")
    p.add_argument("--output-md", default="")
    p.add_argument("--append-results", default="",
                   help="Append the markdown section to this RESULTS.md path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rows = _read_rows(Path(args.summary_csv))
    _augment_rows_from_eval_json(rows)
    means = _read_means(Path(args.best_json)) if args.best_json else []
    _augment_means_from_rows(means, rows)
    markdown = _render_markdown(
        rows,
        means,
        top_n=args.top_n,
        mean_top_n=args.mean_top_n,
        title=args.title,
        source=args.summary_csv,
    )

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print(f"[altgan.report_mark_hybrids] wrote {out}", flush=True)
    else:
        print(markdown, end="")

    if args.append_results:
        path = Path(args.append_results)
        with path.open("a") as fh:
            fh.write("\n\n")
            fh.write(markdown.rstrip())
            fh.write("\n")
        print(f"[altgan.report_mark_hybrids] appended {path}", flush=True)
    return 0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return sorted(rows, key=lambda r: (_as_float(r["hrc_mae"]), _as_float(r["mark_score"])))


def _read_means(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    means = data.get("by_candidate_mean", [])
    return sorted(
        means,
        key=lambda r: (_as_float(r["mean_hrc_mae"]), _as_float(r["mean_mark_score"])),
    )


def _render_markdown(
    rows: list[dict[str, str]],
    means: list[dict[str, Any]],
    *,
    top_n: int,
    mean_top_n: int,
    title: str,
    source: str,
) -> str:
    top = rows[: max(int(top_n), 1)]
    lines = [
        f"## {title}",
        "",
        f"Recorded: {date.today().isoformat()}. Source: `{source}`.",
        "",
        "| rank | seed | transition blend | local power | source | numeric blend | space | temp | noise | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | reuse drift delta | timing drift ratio | size drift ratio | mark score | timing norm | size norm | opcode TV | tenant TV |",
        "|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            "| {rank} | {seed} | {tb} | {lp} | {source} | {blend} | {space} | {temp} | {noise} | "
            "**{hrc}** | {fake_reuse} | {real_reuse} | {fake_med} | {real_med} | "
            "{fake_p90} | {real_p90} | {reuse_drift} | {ts_drift} | {size_drift} | "
            "{mark} | {ts} | {size} | {opcode} | {tenant} |".format(
                rank=idx,
                seed=row["seed"],
                tb=_fmt(row["transition_blend"]),
                lp=_fmt(row["local_prob_power"]),
                source=row["categorical_source"],
                blend=_fmt(row["mark_numeric_blend"]),
                space=row["mark_numeric_blend_space"] or "-",
                temp=_fmt(row["mark_temperature"]),
                noise=_fmt(row["mark_numeric_noise"]),
                hrc=_fmt(row["hrc_mae"]),
                fake_reuse=_fmt(row["fake_reuse"]),
                real_reuse=_fmt(row["real_reuse"]),
                fake_med=_fmt(row["fake_stack_median"]),
                real_med=_fmt(row["real_stack_median"]),
                fake_p90=_fmt(row["fake_stack_p90"]),
                real_p90=_fmt(row["real_stack_p90"]),
                reuse_drift=_fmt(row.get("reuse_local_drift_delta", "")),
                ts_drift=_fmt(row.get("drift_ts_delta_ratio", "")),
                size_drift=_fmt(row.get("drift_obj_size_ratio", "")),
                mark=_fmt(row["mark_score"]),
                ts=_fmt(row["ts_delta_norm"]),
                size=_fmt(row["size_norm"]),
                opcode=_fmt(row["opcode_tv"]),
                tenant=_fmt(row["tenant_tv"]),
            )
        )

    if means:
        lines.extend([
            "",
            "Seed-mean ranking:",
            "",
            "| rank | seeds | transition blend | local power | source | numeric blend | space | temp | noise | mean HRC-MAE | mean mark score | mean timing drift ratio | mean size drift ratio |",
            "|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for idx, row in enumerate(means[: max(int(mean_top_n), 1)], start=1):
            lines.append(
                "| {rank} | {seeds} | {tb} | {lp} | {source} | {blend} | {space} | {temp} | {noise} | "
                "**{hrc}** | {mark} | {ts_drift} | {size_drift} |".format(
                    rank=idx,
                    seeds=row["n_seeds"],
                    tb=_fmt(row["transition_blend"]),
                    lp=_fmt(row["local_prob_power"]),
                    source=row["categorical_source"],
                    blend=_fmt(row["mark_numeric_blend"]),
                    space=row["mark_numeric_blend_space"] or "-",
                    temp=_fmt(row["mark_temperature"]),
                    noise=_fmt(row["mark_numeric_noise"]),
                    hrc=_fmt(row["mean_hrc_mae"]),
                    mark=_fmt(row["mean_mark_score"]),
                    ts_drift=_fmt(row.get("mean_drift_ts_delta_ratio", "")),
                    size_drift=_fmt(row.get("mean_drift_obj_size_ratio", "")),
                )
            )

    lines.append("")
    return "\n".join(lines)


def _fmt(value: object) -> str:
    text = str(value)
    if not text:
        return "-"
    try:
        num = float(text)
    except ValueError:
        return text
    if num.is_integer() and abs(num) >= 10:
        return str(int(num))
    return f"{num:.6g}"


def _augment_rows_from_eval_json(rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("drift_ts_delta_ratio") and row.get("drift_obj_size_ratio"):
            continue
        path = row.get("path", "")
        if not path:
            continue
        try:
            data = json.loads(Path(path).read_text())
        except OSError:
            continue
        gap = data.get("gap", {})
        if not row.get("reuse_local_drift_delta"):
            row["reuse_local_drift_delta"] = str(
                gap.get("reuse_decile_local_drift_fake_minus_real", "")
            )
        if not row.get("drift_ts_delta_ratio"):
            row["drift_ts_delta_ratio"] = str(gap.get("drift_ts_delta_w1_ratio", ""))
        if not row.get("drift_obj_size_ratio"):
            row["drift_obj_size_ratio"] = str(gap.get("drift_obj_size_w1_ratio", ""))


def _augment_means_from_rows(means: list[dict[str, Any]], rows: list[dict[str, str]]) -> None:
    grouped: dict[tuple[str, str, str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row["transition_blend"],
            row["local_prob_power"],
            row["categorical_source"],
            row["mark_numeric_blend"],
            row["mark_numeric_blend_space"],
            row["mark_temperature"],
            row["mark_numeric_noise"],
        )
        grouped.setdefault(key, []).append(row)

    for mean in means:
        if mean.get("mean_drift_ts_delta_ratio") and mean.get("mean_drift_obj_size_ratio"):
            continue
        key = (
            str(mean["transition_blend"]),
            str(mean["local_prob_power"]),
            str(mean["categorical_source"]),
            str(mean["mark_numeric_blend"]),
            str(mean["mark_numeric_blend_space"]),
            str(mean["mark_temperature"]),
            str(mean["mark_numeric_noise"]),
        )
        group = grouped.get(key)
        if not group:
            continue
        if not mean.get("mean_reuse_local_drift_delta"):
            mean["mean_reuse_local_drift_delta"] = _mean_present(
                g.get("reuse_local_drift_delta", "") for g in group
            )
        if not mean.get("mean_drift_ts_delta_ratio"):
            mean["mean_drift_ts_delta_ratio"] = _mean_present(
                g.get("drift_ts_delta_ratio", "") for g in group
            )
        if not mean.get("mean_drift_obj_size_ratio"):
            mean["mean_drift_obj_size_ratio"] = _mean_present(
                g.get("drift_obj_size_ratio", "") for g in group
            )


def _mean_present(values: Any) -> str:
    nums = []
    for value in values:
        if value == "":
            continue
        nums.append(float(value))
    if not nums:
        return ""
    return str(sum(nums) / len(nums))


def _as_float(value: object) -> float:
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
