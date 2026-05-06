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
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
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


_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _mean_token_from_json_text(path: Path) -> tuple[str, float]:
    """Return the *literal* numeric token used for the mean field in a JSON report.

    `llgan.cachesim_eval` writes JSON via Python's `json.dumps()` which may choose
    either fixed-point or scientific notation; LANL docs want the exact token
    string so it can be pasted verbatim.
    """

    text = path.read_text()
    for field in ("mean_hrc_mae", "mean"):
        m = re.search(rf"\"{re.escape(field)}\"\s*:\s*({_NUMBER_RE})", text)
        if m:
            token = m.group(1)
            d = Decimal(token)
            # Preserve the exact JSON numeric value but render without scientific notation
            # so the docs stay readable.
            as_fixed = format(d, "f")
            if "." in as_fixed:
                as_fixed = as_fixed.rstrip("0").rstrip(".") or "0"
            return as_fixed, float(d)

    # Fallback: parse as JSON (loses formatting), but still returns a value.
    return f"{_mean_from_json(path):.10f}", _mean_from_json(path)


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


def _mkdir_or_die(path: Path, *, label: str) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise SystemExit(
            f"Cannot create {label} directory: {path}\n"
            "This launcher is intended to run on a /tiamat-capable host.\n"
            "Tip: override `--output-root` to a writable directory on that host."
        ) from e


@dataclass(frozen=True)
class SeedResult:
    seed: int
    fake_csv: Path
    report_json: Path
    mean_hrc_mae: float
    mean_hrc_mae_token: str


def _markdown_snippet(
    *,
    corpus: str,
    mode: str,
    chunk_size: int,
    retime: bool,
    n_records: int,
    n_streams: int,
    cache_sizes: str,
    policies: str,
    real_manifest: str,
    real_ref: str,
    results: list[SeedResult],
    seeds: list[int],
) -> str:
    means = [r.mean_hrc_mae for r in results]
    overall_mean = sum(means) / len(means) if means else 0.0
    overall_range = max(means) - min(means) if means else 0.0
    seeds_fmt = ",".join(str(s) for s in seeds)

    lines: list[str] = []
    lines.append(f"**Corpus:** `{corpus}`")
    lines.append(
        "**Protocol:** "
        + f"`mode={mode}` "
        + f"`chunk_size={chunk_size}` "
        + f"`n_records={_records_label(n_records)}` "
        + f"`n_streams={n_streams}` "
        + f"`retime={'on' if retime else 'off'}`"
    )
    lines.append(f"**Eval surface:** `cache_sizes={cache_sizes}` `policies={policies}`")
    lines.append(f"**Refs:** `real_manifest={real_manifest}` `real_ref={real_ref}`")
    lines.append("")
    lines.append("| seed | fake CSV | literal cachesim mean line | JSON mean |")
    lines.append("|---:|---|---|---:|")
    for r in results:
        mean_line = _literal_cachesim_mean_line(r.mean_hrc_mae)
        lines.append(f"| {r.seed} | `{r.fake_csv}` | `{mean_line}` | {r.mean_hrc_mae_token} |")
    lines.append("")
    lines.append(f"Literal cachesim mean line (mean across seeds): `{_literal_cachesim_mean_line(overall_mean)}`")
    lines.append(f"Mean across seeds `{{{seeds_fmt}}}`: `{overall_mean:.10f}`")
    lines.append(f"Range: `{overall_range:.10f}`")
    return "\n".join(lines) + "\n"


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
        "--skip-existing",
        action="store_true",
        help="Skip seeds whose fake CSV + official report JSON already exist.",
    )
    p.add_argument(
        "--emit-markdown",
        action="store_true",
        help="Print a paste-ready Markdown results table after running.",
    )
    p.add_argument(
        "--emit-markdown-to",
        default=None,
        help="Write the paste-ready Markdown snippet to this file path (also implies --emit-markdown).",
    )
    p.add_argument(
        "--emit-summary-json-to",
        default=None,
        help="Write a machine-readable JSON summary (per-seed means + overall mean/range) to this file path.",
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
        missing: list[str] = []
        for label, raw_path in (("real-manifest", args.real_manifest), ("real-ref", args.real_ref)):
            p = Path(raw_path)
            if not p.exists():
                missing.append(f"{label}: {p}")
        if missing:
            raise SystemExit(
                "Missing required input files for official evaluation:\n"
                + "\n".join(f"- {m}" for m in missing)
                + "\nRun this on a host with `/tiamat/zarathustra` mounted."
            )

        _mkdir_or_die(output_root, label="output-root")
        _mkdir_or_die(eval_root, label="eval-root")

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

        if args.skip_existing and not args.dry_run:
            if fake_csv.exists() and report_json.exists():
                mean_token, mean = _mean_token_from_json_text(report_json)
                results.append(
                    SeedResult(
                        seed=seed,
                        fake_csv=fake_csv,
                        report_json=report_json,
                        mean_hrc_mae=mean,
                        mean_hrc_mae_token=mean_token,
                    )
                )
                print(
                    f"[skip-existing] seed={seed} already has fake+report; "
                    f"mean={mean_token}",
                    flush=True,
                )
                continue

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
        if not (args.skip_existing and not args.dry_run and fake_csv.exists()):
            _run(cmd_boot, env=env, dry_run=args.dry_run)
        else:
            print(f"[skip-existing] seed={seed} fake CSV exists: {fake_csv}", flush=True)

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
        if not (args.skip_existing and not args.dry_run and report_json.exists()):
            _run(cmd_eval, env=env, dry_run=args.dry_run)
        else:
            print(f"[skip-existing] seed={seed} report JSON exists: {report_json}", flush=True)

        if not args.dry_run:
            mean_token, mean = _mean_token_from_json_text(report_json)
            results.append(
                SeedResult(
                    seed=seed,
                    fake_csv=fake_csv,
                    report_json=report_json,
                    mean_hrc_mae=mean,
                    mean_hrc_mae_token=mean_token,
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
        print(f"JSON mean: {r.mean_hrc_mae_token}", flush=True)
        print(f"Report JSON: {r.report_json}", flush=True)

    means = [r.mean_hrc_mae for r in results]
    overall_mean = sum(means) / len(means) if means else 0.0
    overall_range = max(means) - min(means) if means else 0.0
    print(f"\nMean across seeds {args.seeds}: {overall_mean:.10f}", flush=True)
    print(f"Range: {overall_range:.10f}", flush=True)

    if args.emit_summary_json_to:
        out_path = Path(args.emit_summary_json_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "corpus": args.corpus,
            "mode": args.mode,
            "chunk_size": int(args.chunk_size),
            "retime": bool(args.retime),
            "n_records": int(args.n_records),
            "n_streams": int(args.n_streams),
            "cache_sizes": args.cache_sizes,
            "policies": args.policies,
            "real_manifest": args.real_manifest,
            "real_ref": args.real_ref,
            "seeds": list(args.seeds),
            "results": [
                {
                    "seed": r.seed,
                    "fake_csv": str(r.fake_csv),
                    "report_json": str(r.report_json),
                    "mean_hrc_mae": float(r.mean_hrc_mae),
                    "mean_hrc_mae_token": r.mean_hrc_mae_token,
                }
                for r in results
            ],
            "overall_mean": float(overall_mean),
            "overall_range": float(overall_range),
        }
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote summary JSON: {out_path}", flush=True)

    markdown_text = None
    if args.emit_markdown_to:
        markdown_text = _markdown_snippet(
            corpus=args.corpus,
            mode=args.mode,
            chunk_size=int(args.chunk_size),
            retime=bool(args.retime),
            n_records=int(args.n_records),
            n_streams=int(args.n_streams),
            cache_sizes=args.cache_sizes,
            policies=args.policies,
            real_manifest=args.real_manifest,
            real_ref=args.real_ref,
            results=results,
            seeds=list(args.seeds),
        )
        out_path = Path(args.emit_markdown_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown_text)
        print(f"\nWrote Markdown snippet: {out_path}", flush=True)

    if args.emit_markdown or args.emit_markdown_to:
        if markdown_text is None:
            markdown_text = _markdown_snippet(
                corpus=args.corpus,
                mode=args.mode,
                chunk_size=int(args.chunk_size),
                retime=bool(args.retime),
                n_records=int(args.n_records),
                n_streams=int(args.n_streams),
                cache_sizes=args.cache_sizes,
                policies=args.policies,
                real_manifest=args.real_manifest,
                real_ref=args.real_ref,
                results=results,
                seeds=list(args.seeds),
            )
        print("\n---", flush=True)
        print("\nPaste-ready Markdown:", flush=True)
        print("", flush=True)
        print(markdown_text, end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
