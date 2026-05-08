"""Multi-seed Twitter recent-window sweep + cachesim aggregation.

This is an orchestration wrapper around:
  - `altgan.sweep_twitter_window` (grid of generation knobs)
  - `llgan.cachesim_eval` (official HRC-MAE surface via `launch_msr_cachesim_bracket`)

It exists because the sweep produces per-seed `*_official6.json` reports, but
LANL/LLNL claims must be multi-seed (mean + range). This wrapper runs the sweep
and emits a paste-ready Markdown snippet for the best (lowest-mean) row.

Intended to run on a /tiamat-capable host.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from altgan.sweep_twitter_window import GRID, LLNL_TWITTER_ATLAS  # noqa: TID252


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _mean_token_from_json_text(path: Path) -> tuple[str, float]:
    text = path.read_text()
    for field in ("mean_hrc_mae", "mean"):
        m = re.search(rf"\"{re.escape(field)}\"\s*:\s*({_NUMBER_RE})", text)
        if m:
            token = m.group(1)
            d = Decimal(token)
            return token, float(d)
    with path.open() as f:
        data = json.load(f)
    if "mean_hrc_mae" in data:
        value = float(data["mean_hrc_mae"])
    elif "mean" in data:
        value = float(data["mean"])
    else:
        raise KeyError(f"{path} missing mean_hrc_mae/mean")
    return f"{value:.10f}", value


def _literal_cachesim_mean_line(mean_hrc_mae: float) -> str:
    # Must match `llgan.cachesim_eval.print_report()`.
    return f"mean HRC-MAE across policies: {mean_hrc_mae:.4f}"


def _print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    _print_cmd(cmd)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


@dataclass(frozen=True)
class RowResult:
    row_name: str
    grid_name: str
    seed: int
    report_json: Path
    mean_token: str
    mean_value: float


def _grid_names() -> list[str]:
    return [name for name, _win, _hp, _adj in GRID]


def _expected_json_path(*, output_root: Path, row_name: str) -> Path:
    # `sweep_twitter_window` names rows like `llnl_tw_sweep_<grid>_s<seed>`.
    # `launch_msr_cachesim_bracket` writes:
    #   <output_root>/cachesim_lanl/twitter_cluster_lanl_<row_name>_official6.json
    cache_root = output_root / "cachesim_lanl"
    return cache_root / f"twitter_cluster_lanl_{row_name}_official6.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=_parse_ints, default=[42, 80, 81, 82])
    p.add_argument("--atlas", default=LLNL_TWITTER_ATLAS)
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--phase2-names", default="", help="Comma-separated GRID names; empty = run all.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-markdown", action="store_true")
    p.add_argument(
        "--emit-markdown-to",
        default=None,
        help="Write the best-row Markdown snippet to this file path (also implies --emit-markdown).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Print the best K multi-seed rows by mean.",
    )
    return p.parse_args()


def _markdown_for_best(*, grid_name: str, seeds: list[int], results: list[RowResult]) -> str:
    means = [r.mean_value for r in results]
    overall_mean = sum(means) / len(means) if means else 0.0
    overall_range = max(means) - min(means) if means else 0.0

    lines: list[str] = []
    lines.append(f"Twitter window sweep best row: `{grid_name}`.")
    lines.append("")
    lines.append("| seed | literal `llgan.cachesim_eval` mean line | JSON mean |")
    lines.append("|---:|---|---:|")
    for r in sorted(results, key=lambda rr: rr.seed):
        mean_line = _literal_cachesim_mean_line(r.mean_value)
        lines.append(f"| {r.seed} | `{mean_line}` | {r.mean_token} |")
    lines.append("")
    seed_label = "Four-seed" if len(seeds) == 4 else f"{len(seeds)}-seed"
    lines.append(f"{seed_label} mean: `{overall_mean:.10f}`, range `{overall_range:.10f}`.")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(key, "1")

    if args.emit_markdown_to:
        args.emit_markdown = True

    output_root = Path(args.output_root)
    selected = {n.strip() for n in args.phase2_names.split(",") if n.strip()}
    grid_names = [n for n in _grid_names() if not selected or n in selected]
    if selected:
        unknown = sorted(selected - set(_grid_names()))
        if unknown:
            raise SystemExit(f"Unknown --phase2-names entries: {unknown}")

    # Step 1: run the underlying sweep (which produces cachesim JSONs).
    sweep_cmd = [
        sys.executable,
        "-u",
        "-m",
        "altgan.sweep_twitter_window",
        "--seeds",
        ",".join(str(s) for s in args.seeds),
        "--atlas",
        args.atlas,
        "--output-root",
        str(output_root),
        "--phase2-names",
        ",".join(grid_names),
    ]
    if args.skip_existing:
        sweep_cmd.append("--skip-existing")
    if args.dry_run:
        sweep_cmd.append("--dry-run")
    if args.dry_run:
        _print_cmd(sweep_cmd)
        subprocess.run(sweep_cmd, check=True, env=env)
        return 0
    _run(sweep_cmd, env=env, dry_run=False)

    # Step 2: collect results and aggregate by GRID row across seeds.
    per_seed: list[RowResult] = []
    for seed in args.seeds:
        for grid_name in grid_names:
            row_name = f"llnl_tw_sweep_{grid_name}_s{seed}"
            report = _expected_json_path(output_root=output_root, row_name=row_name)
            if not report.exists():
                if args.skip_existing:
                    continue
                raise SystemExit(f"Missing expected cachesim report JSON: {report}")
            token, value = _mean_token_from_json_text(report)
            per_seed.append(
                RowResult(
                    row_name=row_name,
                    grid_name=grid_name,
                    seed=seed,
                    report_json=report,
                    mean_token=token,
                    mean_value=value,
                )
            )

    grouped: dict[str, list[RowResult]] = {}
    for r in per_seed:
        grouped.setdefault(r.grid_name, []).append(r)

    scored: list[tuple[str, float, float, list[RowResult]]] = []
    for grid_name, results in grouped.items():
        means = [r.mean_value for r in results]
        if not means:
            continue
        scored.append((grid_name, sum(means) / len(means), max(means) - min(means), results))

    scored.sort(key=lambda item: (item[1], item[2]))
    if not scored:
        print("[launch_twitter_window_pack] no scored rows found (did the sweep run?)", flush=True)
        return 2

    print("\n[launch_twitter_window_pack] multi-seed ranking (lower mean wins):", flush=True)
    for i, (grid_name, mean, spread, results) in enumerate(scored[: max(1, args.top_k)], start=1):
        seed_count = len({r.seed for r in results})
        print(
            f"  {i:2d}. {grid_name:12s}  mean={mean:.10f}  range={spread:.10f}  seeds={seed_count}",
            flush=True,
        )

    best_name, _best_mean, _best_range, best_results = scored[0]
    if args.emit_markdown:
        snippet = _markdown_for_best(grid_name=best_name, seeds=args.seeds, results=best_results)
        print("\n" + snippet, flush=True)
        if args.emit_markdown_to:
            path = Path(args.emit_markdown_to)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(snippet)
            print(f"[launch_twitter_window_pack] wrote {path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
