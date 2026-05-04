"""Multi-seed launcher for the Tencent cache-surface chunk selector.

This is a small orchestration layer around `altgan.optimize_tencent_chunk_surface`
so LANL can re-run (and extend) the R287-style Tencent cache-surface chunk
selector with a reproducible multi-seed pipeline.

It intentionally optimizes only against the official `llgan.cachesim_eval`
surface (via `altgan.optimize_tencent_chunk_surface`, which uses the same
`llgan.cachesim_eval._run_cachesim` backend).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_templates(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _render_template(template: str, *, seed: int) -> str:
    # Support both "{seed}" and "{seed:03d}" style formats.
    try:
        return template.format(seed=seed)
    except (KeyError, ValueError):
        # If braces are present but not valid `.format`, fall back to a simple replace.
        return template.replace("{seed}", str(seed))


@dataclass(frozen=True)
class StageResult:
    seed: int
    stage: str
    fake_csv: Path
    report_json: Path
    mean_hrc_mae: float


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


def _cmd_optimize(
    *,
    base: Path,
    donors: list[Path],
    real: Path,
    output_root: Path,
    seed: int,
    tag: str,
    chunk_size: int,
    cache_sizes: str,
    policies: str,
    max_passes: int,
    max_accepts: int,
    min_improvement: float,
) -> list[str]:
    donor_csv = ",".join(str(p) for p in donors)
    return [
        sys.executable,
        "-u",
        "-m",
        "altgan.optimize_tencent_chunk_surface",
        "--base",
        str(base),
        "--donor",
        donor_csv,
        "--real",
        str(real),
        "--output-root",
        str(output_root),
        "--tag",
        tag,
        "--seed",
        str(seed),
        "--chunk-size",
        str(chunk_size),
        "--max-passes",
        str(max_passes),
        "--max-accepts",
        str(max_accepts),
        "--min-improvement",
        str(min_improvement),
        "--cache-sizes",
        cache_sizes,
        "--policies",
        policies,
    ]


def _print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    _print_cmd(cmd)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=_parse_ints, default=[42, 80, 81, 82])
    p.add_argument("--real", required=True, help="Official Tencent real CSV reference.")
    p.add_argument(
        "--base-template",
        required=True,
        help="Template path for the per-seed base fake CSV (supports {seed}).",
    )
    p.add_argument(
        "--donor-templates",
        required=True,
        type=_parse_templates,
        help="Comma-separated template paths for per-seed donor fake CSVs (supports {seed}).",
    )
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tag-prefix", default="tencent_chunksurf")
    p.add_argument(
        "--pipeline",
        type=_parse_ints,
        default=[2048, 1024, 512, 256],
        help="Comma-separated chunk sizes to run in sequence; output of each stage feeds the next.",
    )
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--max-passes", type=int, default=1)
    p.add_argument("--max-accepts", type=int, default=128)
    p.add_argument("--min-improvement", type=float, default=1e-6)
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

    results: list[StageResult] = []
    final_means: list[tuple[int, float, Path, Path]] = []

    for seed in args.seeds:
        base = Path(_render_template(args.base_template, seed=seed))
        donors = [Path(_render_template(t, seed=seed)) for t in args.donor_templates]

        current_base = base
        for chunk_size in args.pipeline:
            stage = f"ck{chunk_size}"
            tag = f"{args.tag_prefix}_{stage}"
            cmd = _cmd_optimize(
                base=current_base,
                donors=donors,
                real=real,
                output_root=output_root,
                seed=seed,
                tag=tag,
                chunk_size=chunk_size,
                cache_sizes=args.cache_sizes,
                policies=args.policies,
                max_passes=args.max_passes,
                max_accepts=args.max_accepts,
                min_improvement=args.min_improvement,
            )
            _run(cmd, env=env, dry_run=args.dry_run)

            out_fake = output_root / f"{tag}_ck{chunk_size}_seed{seed}_fake_100k.csv"
            out_json = eval_root / f"{tag}_ck{chunk_size}_seed{seed}_official6.json"
            if args.dry_run:
                continue
            mean = _mean_from_json(out_json)
            results.append(
                StageResult(
                    seed=seed,
                    stage=stage,
                    fake_csv=out_fake,
                    report_json=out_json,
                    mean_hrc_mae=mean,
                )
            )
            current_base = out_fake

        if not args.dry_run:
            final_means.append((seed, results[-1].mean_hrc_mae, results[-1].fake_csv, results[-1].report_json))

    if args.dry_run:
        print("\n[dry-run] No stages executed; exiting.", flush=True)
        return 0

    print("\n=== TENCENT CHUNK-SURFACE MULTI-SEED SUMMARY ===", flush=True)
    for seed, mean, fake_csv, report_json in final_means:
        print(f"\nseed {seed}", flush=True)
        print(f"fake CSV: {fake_csv}", flush=True)
        print(_literal_cachesim_mean_line(mean), flush=True)
        print(f"JSON mean: {mean:.10f}", flush=True)
        print(f"Report JSON: {report_json}", flush=True)

    means = [m for _, m, _, _ in final_means]
    overall_mean = sum(means) / len(means)
    overall_range = max(means) - min(means) if means else 0.0
    print(f"\nMean across seeds {args.seeds}: {overall_mean:.10f}", flush=True)
    print(f"Range: {overall_range:.10f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
