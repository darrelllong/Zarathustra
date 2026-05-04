"""Multi-seed launcher for TraceBootstrap + official cachesim evaluation.

This is a small orchestration layer around:
  - `altgan.trace_bootstrap` (chunk bootstrapping real streams)
  - `llgan.cachesim_eval` (official HRC-MAE surface)

It is meant to close out "bootstrap ledger" rows (Twitter / Meta KV / Meta CDN,
etc.) with a reproducible multi-seed protocol and pasteable literal cachesim
mean lines + exact JSON means.
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


def _records_label(n_records: int) -> str:
    if n_records == 1_000_000:
        return "1M"
    if n_records == 100_000:
        return "100k"
    if n_records % 1000 == 0:
        return f"{n_records // 1000}k"
    return str(n_records)


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


def _print_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)


def _run(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    _print_cmd(cmd)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


@dataclass(frozen=True)
class SeedResult:
    seed: int
    fake_csv: Path
    report_json: Path
    mean_hrc_mae: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="Corpus label used for output naming.")
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument(
        "--real-manifest",
        required=True,
        help="Pinned long-rollout manifest JSON used as the real stream source.",
    )
    p.add_argument(
        "--real-ref",
        required=True,
        help="Official cachesim real CSV reference (llgan.cachesim_eval --real).",
    )
    p.add_argument("--seeds", type=_parse_ints, default=[42, 80, 81, 82])
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=65_536)
    p.add_argument(
        "--mode",
        choices=("replay", "rotate", "shuffle", "block-swap"),
        default="shuffle",
    )
    p.add_argument("--retime", action="store_true")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Print a paste-ready Markdown results table after running.",
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
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        eval_root.mkdir(parents=True, exist_ok=True)

    retime_tag = "ret" if args.retime else "nort"
    records_tag = _records_label(args.n_records)
    policy_count = len([p for p in args.policies.split(",") if p.strip()])

    results: list[SeedResult] = []
    for seed in args.seeds:
        fake_csv = output_root / (
            f"{args.corpus}_lanl_boot_{args.mode}{args.chunk_size}_{retime_tag}"
            f"_seed{seed}_fake_{records_tag}.csv"
        )
        report_json = eval_root / (
            f"{args.corpus}_lanl_boot_{args.mode}{args.chunk_size}_{retime_tag}"
            f"_seed{seed}_official{policy_count}.json"
        )

        cmd_boot = [
            sys.executable,
            "-u",
            "-m",
            "altgan.trace_bootstrap",
            "--trace-dir",
            args.trace_dir,
            "--fmt",
            args.fmt,
            "--real-manifest",
            args.real_manifest,
            "--output",
            str(fake_csv),
            "--n-records",
            str(args.n_records),
            "--n-streams",
            str(args.n_streams),
            "--seed",
            str(seed),
            "--chunk-size",
            str(args.chunk_size),
            "--mode",
            args.mode,
        ]
        if args.retime:
            cmd_boot.append("--retime")
        _run(cmd_boot, env=env, dry_run=args.dry_run)

        cmd_eval = [
            sys.executable,
            "-u",
            "-m",
            "llgan.cachesim_eval",
            "--fake",
            str(fake_csv),
            "--real",
            args.real_ref,
            "--cache-sizes",
            args.cache_sizes,
            "--policies",
            args.policies,
            "--out",
            str(report_json),
        ]
        _run(cmd_eval, env=env, dry_run=args.dry_run)

        if not args.dry_run:
            mean = _mean_from_json(report_json)
            results.append(
                SeedResult(
                    seed=seed,
                    fake_csv=fake_csv,
                    report_json=report_json,
                    mean_hrc_mae=mean,
                )
            )

    if args.dry_run:
        print("\n[dry-run] No seeds executed; exiting.", flush=True)
        return 0

    print(f"\n=== TRACEBOOTSTRAP MULTI-SEED SUMMARY ({args.corpus}) ===", flush=True)
    for r in results:
        print(f"\nseed {r.seed}", flush=True)
        print(f"fake CSV: {r.fake_csv}", flush=True)
        print(_literal_cachesim_mean_line(r.mean_hrc_mae), flush=True)
        print(f"JSON mean: {r.mean_hrc_mae:.10f}", flush=True)
        print(f"Report JSON: {r.report_json}", flush=True)

    means = [r.mean_hrc_mae for r in results]
    overall_mean = sum(means) / len(means) if means else 0.0
    overall_range = max(means) - min(means) if means else 0.0
    print(f"\nMean across seeds {args.seeds}: {overall_mean:.10f}", flush=True)
    print(f"Range: {overall_range:.10f}", flush=True)

    if args.emit_markdown:
        seeds_fmt = ",".join(str(s) for s in args.seeds)
        print("\n---", flush=True)
        print("\nPaste-ready Markdown:", flush=True)
        print("", flush=True)
        print("| seed | fake CSV | literal cachesim mean line | JSON mean |", flush=True)
        print("|---:|---|---|---:|", flush=True)
        for r in results:
            mean_line = _literal_cachesim_mean_line(r.mean_hrc_mae)
            print(
                f"| {r.seed} | `{r.fake_csv}` | `{mean_line}` | {r.mean_hrc_mae:.10f} |",
                flush=True,
            )
        print("", flush=True)
        print(
            f"Mean across seeds `{{{seeds_fmt}}}`: `{overall_mean:.10f}`",
            flush=True,
        )
        print(f"Range: `{overall_range:.10f}`", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
