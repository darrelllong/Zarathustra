"""Generic multi-seed launcher for LANL's cache-surface chunk selector.

This wraps `altgan.optimize_tencent_chunk_surface` (despite the name, it is a
generic cache-surface chunk combiner) to run a reproducible, multi-seed
pipeline that optimizes only against the official `llgan.cachesim_eval` surface.

It prints:
  - literal `mean HRC-MAE across policies: ...` lines (matching cachesim_eval)
  - exact JSON means (`mean_hrc_mae`)
  - a ready-to-paste markdown snippet (optional)

This is intended for remote runs on `baase` / `vinge` where the official refs
live under `/tiamat/zarathustra/llgan-output/refs/...`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _render_template(template: str, *, seed: int) -> str:
    # Support both "{seed}" and "{seed:03d}" style formats.
    try:
        return template.format(seed=seed)
    except (KeyError, ValueError):
        return template.replace("{seed}", str(seed))


def _needs_seed_expansion(template: str) -> bool:
    # We support both "{seed}" and "{seed:03d}" style formats.
    return "{seed" in template


def _count_rows(csv_path: Path) -> int:
    # Fast-enough pure-Python line count; avoids pandas dependency in this launcher.
    with csv_path.open("rb") as f:
        lines = sum(1 for _ in f)
    # subtract header
    return max(lines - 1, 0)


def _mean_from_json(path: Path) -> float:
    with path.open() as f:
        data = json.load(f)
    if "mean_hrc_mae" in data:
        return float(data["mean_hrc_mae"])
    if "mean" in data:
        return float(data["mean"])
    raise KeyError(f"{path} missing mean_hrc_mae/mean")


def _literal_cachesim_mean_line(mean_hrc_mae: float) -> str:
    # Must match `llgan.cachesim_eval.print_report()`.
    return f"mean HRC-MAE across policies: {mean_hrc_mae:.4f}"


def _race_display(value: float) -> str:
    return f"{value:.4f}"


def _policy_count(policies: str) -> int:
    return len([part for part in policies.split(",") if part.strip()])


def _markdown_snippet(
    *,
    title: str,
    seeds: list[int],
    final_means: list[tuple[int, float, Path, Path, str]],
    guard_means: list[tuple[int, float, Path]] | None,
    guard_label: str,
    overall_mean: float,
    overall_range: float,
) -> str:
    seeds_text = "{" + ",".join(str(seed) for seed in seeds) + "}"
    lines: list[str] = [
        title,
        "",
        "| seed | fake CSV | literal cachesim mean line | JSON mean |",
        "|---:|---|---|---:|",
    ]
    for seed, mean, fake_csv, _report_json, literal_mean_line in final_means:
        lines.append(
            f"| {seed} | `{fake_csv}` | `{literal_mean_line}` | {mean:.10f} |"
        )
    lines += [
        "",
        f"Mean across seeds `{seeds_text}`: `{overall_mean:.10f}` (race display `{_race_display(overall_mean)}`; range `{overall_range:.10f}`).",
        "",
    ]
    if guard_means:
        guard_values = [mean for _seed, mean, _json_path in guard_means]
        guard_mean = sum(guard_values) / len(guard_values)
        guard_range = max(guard_values) - min(guard_values) if guard_values else 0.0
        label = guard_label or "guard"
        lines += [
            f"Guard surface `{label}`:",
            "",
            "| seed | guard JSON | guard mean |",
            "|---:|---|---:|",
        ]
        for seed, mean, json_path in guard_means:
            lines.append(f"| {seed} | `{json_path}` | {mean:.10f} |")
        lines += [
            "",
            f"Guard mean across seeds `{seeds_text}`: `{guard_mean:.10f}` (range `{guard_range:.10f}`).",
            "",
        ]
    return "\n".join(lines)


def _print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> str | None:
    _print_cmd(cmd)
    if dry_run:
        return None
    mean_line: str | None = None
    mean_re = re.compile(r"^mean HRC-MAE across policies:\s*[0-9.]+\s*$")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        print(line, flush=True)
        if mean_re.match(line.strip()):
            mean_line = line.strip()
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return mean_line


@dataclass(frozen=True)
class StageOutput:
    seed: int
    chunk_size: int
    fake_csv: Path
    report_json: Path
    mean_hrc_mae: float
    literal_mean_line: str
    guard_json: Path | None = None
    guard_mean_hrc_mae: float | None = None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=_parse_ints, default=[42, 80, 81, 82])
    p.add_argument("--real", required=True, help="Official real CSV reference.")
    p.add_argument(
        "--base-template",
        required=True,
        help="Template path for the per-seed base fake CSV (supports {seed}).",
    )
    p.add_argument(
        "--donor-templates",
        default="",
        help="Comma-separated template paths for donor fake CSVs (supports {seed}).",
    )
    p.add_argument(
        "--donor-globs",
        default="",
        help="Comma-separated glob patterns for donor fake CSVs (supports {seed}).",
    )
    p.add_argument(
        "--cross-seed-donors",
        action="store_true",
        help=(
            "Expand donor templates/globs across *all* seeds and use the combined pool for every target seed. "
            "This enables cross-seed stabilization (e.g., letting seed80 borrow chunks from seed42 donors)."
        ),
    )
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tag-prefix", default="chunksurf")
    p.add_argument(
        "--pipeline",
        type=_parse_ints,
        default=[65536],
        help="Comma-separated chunk sizes to run in sequence; output of each stage feeds the next.",
    )
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--guard-cache-sizes", default="")
    p.add_argument("--guard-policies", default="")
    p.add_argument("--guard-max-regression", type=float, default=0.0)
    p.add_argument("--guard-eval-label", default="guard")
    p.add_argument("--max-passes", type=int, default=1)
    p.add_argument("--max-accepts", type=int, default=128)
    p.add_argument("--max-evals", type=int, default=0)
    p.add_argument("--min-improvement", type=float, default=1e-6)
    p.add_argument(
        "--accept-mode",
        choices=["first", "best"],
        default="first",
        help="Per chunk, accept the first improving donor or scan all donors and accept the best.",
    )
    p.add_argument(
        "--swap-columns",
        default="obj_id",
        help="Comma-separated donor columns to splice in each chunk; default obj_id.",
    )
    p.add_argument(
        "--donor-shifts",
        default="0",
        help=(
            "Comma-separated row offsets for donor chunks. Default 0 preserves aligned "
            "chunking; nonzero values are forwarded to optimize_tencent_chunk_surface."
        ),
    )
    p.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Print a ready-to-paste markdown snippet (table + mean/range).",
    )
    p.add_argument(
        "--append-markdown",
        default=None,
        help=(
            "Append the markdown snippet to one or more files (comma-separated), e.g. "
            "`RESPONSE-LANL.md` or `altgan/RESULTS.md`."
        ),
    )
    p.add_argument(
        "--markdown-title",
        default=None,
        help="Optional markdown title line (defaults to an auto header with UTC timestamp + tag-prefix).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


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

    output_root = Path(args.output_root)
    eval_root = output_root / "cachesim_lanl"
    real = Path(args.real)

    if not args.dry_run:
        eval_root.mkdir(parents=True, exist_ok=True)

    donor_templates = _parse_list(args.donor_templates) if args.donor_templates else []
    donor_globs = _parse_list(args.donor_globs) if args.donor_globs else []
    policy_count = _policy_count(args.policies)
    eval_label = f"official{policy_count}"

    final_means: list[tuple[int, float, Path, Path, str]] = []
    final_guards: list[tuple[int, float, Path]] = []

    for seed in args.seeds:
        base = Path(_render_template(args.base_template, seed=seed))
        if not args.dry_run and not base.exists():
            raise FileNotFoundError(f"base not found: {base}")

        n_rows = 0 if args.dry_run else _count_rows(base)
        if not args.dry_run and n_rows == 0:
            raise ValueError(f"base appears empty (0 rows): {base}")
        n_k = max(n_rows // 1000, 1) if not args.dry_run else 0

        donors: list[Path] = []
        donor_seeds = list(args.seeds) if args.cross_seed_donors else [seed]
        for template in donor_templates:
            expand_seeds = donor_seeds if _needs_seed_expansion(template) else [seed]
            for donor_seed in expand_seeds:
                donors.append(Path(_render_template(template, seed=donor_seed)))
        for pattern in donor_globs:
            expand_seeds = donor_seeds if _needs_seed_expansion(pattern) else [seed]
            for donor_seed in expand_seeds:
                rendered = _render_template(pattern, seed=donor_seed)
                matches = sorted(
                    (Path(p) for p in glob.glob(rendered)),
                    key=lambda p: str(p),
                )
                if matches:
                    donors.extend(matches)
                elif args.dry_run:
                    # In dry-run mode we often don't have `/tiamat/...` mounted
                    # locally; keep the rendered glob so the user can sanity-check
                    # which donor patterns would be expanded on the remote host.
                    donors.append(Path(rendered))

        # Preserve order but de-dupe paths.
        unique_donors: list[Path] = []
        seen: set[Path] = set()
        for p in donors:
            rp = p
            if rp in seen:
                continue
            seen.add(rp)
            unique_donors.append(rp)
        donors = unique_donors

        if not donors:
            raise ValueError("no donors provided; set --donor-templates and/or --donor-globs")
        if not args.dry_run:
            missing = [p for p in donors if not p.exists()]
            if missing:
                raise FileNotFoundError("missing donor(s):\n" + "\n".join(str(p) for p in missing))

        current_base = base
        last_out: StageOutput | None = None
        for chunk_size in args.pipeline:
            # `altgan.optimize_tencent_chunk_surface` appends its own `_ck<chunk>` suffix,
            # so `--tag` should be a stable prefix (no `_ck...` here) to avoid
            # `..._ck8192_ck8192_seed42` style duplication.
            stage_tag = args.tag_prefix
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "altgan.optimize_tencent_chunk_surface",
                "--base",
                str(current_base),
                "--donor",
                ",".join(str(p) for p in donors),
                "--real",
                str(real),
                "--output-root",
                str(output_root),
                "--tag",
                stage_tag,
                "--seed",
                str(seed),
                "--chunk-size",
                str(chunk_size),
                "--max-passes",
                str(args.max_passes),
                "--max-accepts",
                str(args.max_accepts),
                "--max-evals",
                str(args.max_evals),
                "--min-improvement",
                str(args.min_improvement),
                "--accept-mode",
                args.accept_mode,
                "--swap-columns",
                args.swap_columns,
                "--donor-shifts",
                args.donor_shifts,
                "--cache-sizes",
                args.cache_sizes,
                "--policies",
                args.policies,
            ]
            if args.guard_cache_sizes:
                cmd += [
                    "--guard-cache-sizes",
                    args.guard_cache_sizes,
                    "--guard-policies",
                    args.guard_policies or args.policies,
                    "--guard-max-regression",
                    str(args.guard_max_regression),
                    "--guard-eval-label",
                    args.guard_eval_label,
                ]
            stage_literal_line = _run(cmd, env=env, dry_run=args.dry_run)

            # `optimize_tencent_chunk_surface` builds:
            #   tag = f"{args.tag}_ck{chunk_label}_seed{seed}"
            out_tag = f"{stage_tag}_ck{chunk_size}_seed{seed}"
            out_fake = output_root / f"{out_tag}_fake_{n_k}k.csv"
            out_json = eval_root / f"{out_tag}_{eval_label}.json"
            guard_json = (
                eval_root / f"{out_tag}_{args.guard_eval_label}.json"
                if args.guard_cache_sizes
                else None
            )

            if args.dry_run:
                current_base = out_fake
                continue

            mean = _mean_from_json(out_json)
            guard_mean = (
                _mean_from_json(guard_json)
                if guard_json is not None and guard_json.exists()
                else None
            )
            last_out = StageOutput(
                seed=seed,
                chunk_size=chunk_size,
                fake_csv=out_fake,
                report_json=out_json,
                mean_hrc_mae=mean,
                literal_mean_line=stage_literal_line or _literal_cachesim_mean_line(mean),
                guard_json=guard_json if guard_mean is not None else None,
                guard_mean_hrc_mae=guard_mean,
            )
            current_base = out_fake

        if args.dry_run:
            continue
        if last_out is None:
            raise RuntimeError("pipeline produced no stages")
        final_means.append(
            (seed, last_out.mean_hrc_mae, last_out.fake_csv, last_out.report_json, last_out.literal_mean_line)
        )
        if last_out.guard_mean_hrc_mae is not None and last_out.guard_json is not None:
            final_guards.append((seed, last_out.guard_mean_hrc_mae, last_out.guard_json))

    if args.dry_run:
        print("\n[dry-run] No stages executed; exiting.", flush=True)
        return 0

    print("\n=== CHUNK-SURFACE MULTI-SEED SUMMARY ===", flush=True)
    for seed, mean, fake_csv, report_json, literal_mean_line in final_means:
        print(f"\nseed {seed}", flush=True)
        print(f"fake CSV: {fake_csv}", flush=True)
        print(literal_mean_line, flush=True)
        print(f"JSON mean: {mean:.10f}", flush=True)
        print(f"Report JSON: {report_json}", flush=True)
        guard_row = next((row for row in final_guards if row[0] == seed), None)
        if guard_row is not None:
            _guard_seed, guard_mean, guard_json = guard_row
            print(f"Guard mean ({args.guard_eval_label}): {guard_mean:.10f}", flush=True)
            print(f"Guard JSON: {guard_json}", flush=True)

    means = [mean for _, mean, _, _, _ in final_means]
    overall_mean = sum(means) / len(means)
    overall_range = max(means) - min(means) if means else 0.0
    print(f"\nMean across seeds {args.seeds}: {overall_mean:.10f}", flush=True)
    print(f"Range: {overall_range:.10f}", flush=True)
    if final_guards:
        guard_values = [mean for _seed, mean, _json_path in final_guards]
        guard_mean = sum(guard_values) / len(guard_values)
        guard_range = max(guard_values) - min(guard_values) if guard_values else 0.0
        print(
            f"Guard mean across seeds {args.seeds} ({args.guard_eval_label}): "
            f"{guard_mean:.10f}",
            flush=True,
        )
        print(f"Guard range: {guard_range:.10f}", flush=True)

    if args.emit_markdown or args.append_markdown:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
        title = args.markdown_title or f"## {timestamp} -- {args.tag_prefix} multi-seed summary"
        snippet = _markdown_snippet(
            title=title,
            seeds=list(args.seeds),
            final_means=final_means,
            guard_means=final_guards,
            guard_label=args.guard_eval_label if args.guard_cache_sizes else "",
            overall_mean=overall_mean,
            overall_range=overall_range,
        )
        if args.emit_markdown:
            print("\n=== MARKDOWN SNIPPET (paste into RESPONSE-LANL.md / altgan/RESULTS.md) ===", flush=True)
            print(snippet, flush=True)
        if args.append_markdown:
            dests = [Path(p.strip()) for p in args.append_markdown.split(",") if p.strip()]
            if not dests:
                raise ValueError("--append-markdown was provided but no paths were parsed")
            for dest in dests:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open("a", encoding="utf-8") as f:
                    f.write("\n")
                    f.write(snippet)
                print(f"[markdown] appended snippet to {dest}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
