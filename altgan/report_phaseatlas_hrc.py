"""Render PhaseAtlas HRC sweep summaries as markdown.

The sweep runner writes machine-readable CSV/JSON artifacts. This script turns
the CSV into a compact results table that can be appended to ``RESULTS.md`` as
soon as a remote run finishes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--title", default="PhaseAtlas HRC Calibration Sweep")
    p.add_argument("--output-md", default="")
    p.add_argument("--append-results", default="",
                   help="Append the markdown section to this RESULTS.md path.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rows = _read_rows(Path(args.summary_csv))
    markdown = _render_markdown(rows, top_n=args.top_n, title=args.title, source=args.summary_csv)

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown)
        print(f"[altgan.report_phaseatlas_hrc] wrote {out}", flush=True)
    else:
        print(markdown, end="")

    if args.append_results:
        path = Path(args.append_results)
        with path.open("a") as fh:
            fh.write("\n\n")
            fh.write(markdown.rstrip())
            fh.write("\n")
        print(f"[altgan.report_phaseatlas_hrc] appended {path}", flush=True)
    return 0


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return sorted(rows, key=lambda r: _as_float(r["hrc_mae"]))


def _render_markdown(rows: list[dict], *, top_n: int, title: str, source: str) -> str:
    top = rows[: max(int(top_n), 1)]
    lines = [
        f"## {title}",
        "",
        f"Recorded: {date.today().isoformat()}. Source: `{source}`.",
        "",
        "| rank | panel | seed | blend | phase | rank scale | rank max | phase scale schedule | phase max schedule | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |",
        "|---:|---|---:|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(top, start=1):
        lines.append(
            "| {rank} | {panel} | {seed} | {blend} | {phase} | {rank_scale} | {rank_max} | "
            "{phase_scales} | {phase_maxes} | **{hrc}** | {fake_reuse} | {real_reuse} | "
            "{fake_med} | {real_med} | {fake_p90} | {real_p90} | {mark_score} |".format(
                rank=idx,
                panel=f"{row['n_streams']}x{row['n_records']}",
                seed=row["seed"],
                blend=_fmt(row["transition_blend"]),
                phase="forced" if _truthy(row["force_phase_schedule"]) else "natural",
                rank_scale=_fmt(row["stack_rank_scale"]),
                rank_max=row["stack_rank_max"],
                phase_scales=row.get("stack_rank_phase_scales") or "-",
                phase_maxes=row.get("stack_rank_phase_maxes") or "-",
                hrc=_fmt(row["hrc_mae"]),
                fake_reuse=_fmt(row["fake_reuse"]),
                real_reuse=_fmt(row["real_reuse"]),
                fake_med=_fmt(row["fake_stack_median"]),
                real_med=_fmt(row["real_stack_median"]),
                fake_p90=_fmt(row["fake_stack_p90"]),
                real_p90=_fmt(row["real_stack_p90"]),
                mark_score=_fmt(row.get("mark_score", "")),
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


def _as_float(value: object) -> float:
    return float(value)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    raise SystemExit(main())
