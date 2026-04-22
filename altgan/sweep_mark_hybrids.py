"""Run and summarize neural/reservoir mark hybrid sweeps for NeuralAtlas.

This is a small orchestration wrapper around ``altgan.evaluate_neural_atlas``.
It exists because the mark-head work needs many paired controls with the same
object process, and hand-written shell loops are too easy to misquote.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--char-file", required=True)
    p.add_argument("--real-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", default="mark_hybrid")
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42,
                   help="Single evaluation seed. Ignored when --seeds is set.")
    p.add_argument("--seeds", default="",
                   help="Comma-separated evaluation seeds for stability sweeps.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--transition-blend", type=float, default=0.0)
    p.add_argument("--mark-temperatures", default="1.0,0.5,0.25,0.05")
    p.add_argument("--mark-numeric-noises", default="0.0")
    p.add_argument("--mark-numeric-blends", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--categorical-sources", default="reservoir,neural")
    p.add_argument("--include-reservoir-control", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse existing eval JSONs instead of rerunning completed cells.")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--best-json", default="",
                   help="Optional path for a compact best-candidate summary.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    seeds = _split_int(args.seeds) if args.seeds else [args.seed]

    if args.include_reservoir_control:
        for seed in seeds:
            control = out_dir / _label(
                args.prefix,
                seed,
                "reservoir_control_eval_100k.json",
                include_seed=len(seeds) > 1,
            )
            _run_eval(args, control, seed=seed, extra=["--disable-neural-marks"])
            rows.append(_summarize(control, seed, "reservoir", "", "", ""))

    for source in _split_str(args.categorical_sources):
        for blend in _split_float(args.mark_numeric_blends):
            for temp in _split_float(args.mark_temperatures):
                for noise in _split_float(args.mark_numeric_noises):
                    for seed in seeds:
                        suffix = (
                            f"cat-{source}_blend-{_slug(blend)}"
                            f"_temp-{_slug(temp)}_noise-{_slug(noise)}_eval_100k.json"
                        )
                        path = out_dir / _label(args.prefix, seed, suffix, include_seed=len(seeds) > 1)
                        _run_eval(
                            args,
                            path,
                            seed=seed,
                            extra=[
                                "--mark-categorical-source", source,
                                "--mark-numeric-blend", str(blend),
                                "--mark-temperature", str(temp),
                                "--mark-numeric-noise", str(noise),
                            ],
                        )
                        rows.append(_summarize(path, seed, source, blend, temp, noise))

    summary_path = Path(args.summary_csv) if args.summary_csv else out_dir / f"{args.prefix}_summary.csv"
    _write_summary(summary_path, rows)
    print(f"[altgan.sweep_mark_hybrids] wrote {summary_path}", flush=True)
    best_path = Path(args.best_json) if args.best_json else out_dir / f"{args.prefix}_best.json"
    _write_best(best_path, rows)
    print(f"[altgan.sweep_mark_hybrids] wrote {best_path}", flush=True)
    return 0


def _run_eval(args: argparse.Namespace, output: Path, *, seed: int, extra: list[str]) -> None:
    if args.skip_existing and output.exists():
        print(f"[altgan.sweep_mark_hybrids] reusing {output}", flush=True)
        return
    cmd = [
        sys.executable, "-u", "-m", "altgan.evaluate_neural_atlas",
        "--model", args.model,
        "--trace-dir", args.trace_dir,
        "--fmt", args.fmt,
        "--char-file", args.char_file,
        "--cond-dim", str(args.cond_dim),
        "--condition-from-real-manifest",
        "--transition-blend", str(args.transition_blend),
        "--temperature", str(args.temperature),
        "--n-records", str(args.n_records),
        "--n-streams", str(args.n_streams),
        "--seed", str(seed),
        "--real-manifest", args.real_manifest,
        "--output", str(output),
        *extra,
    ]
    print(f"[altgan.sweep_mark_hybrids] running {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _summarize(path: Path, seed: int, source: object, blend: object, temp: object, noise: object) -> dict:
    data = json.loads(path.read_text())
    mark = data["mark_quality"]
    fake = data["fake"]
    real = data["real"]
    gap = data["gap"]
    return {
        "path": str(path),
        "seed": seed,
        "categorical_source": source,
        "mark_numeric_blend": blend,
        "mark_temperature": temp,
        "mark_numeric_noise": noise,
        "uses_neural_marks": data.get("uses_neural_marks"),
        "hrc_mae": gap["hrc_mae"],
        "fake_reuse": fake["reuse_access_rate"],
        "real_reuse": real["reuse_access_rate"],
        "fake_stack_median": fake["stack_distance_median"],
        "real_stack_median": real["stack_distance_median"],
        "fake_stack_p90": fake["stack_distance_p90"],
        "real_stack_p90": real["stack_distance_p90"],
        "mark_score": mark["mark_score"],
        "ts_delta_norm": mark["ts_delta_log_w1_norm"],
        "size_norm": mark["obj_size_log_w1_norm"],
        "opcode_tv": mark["opcode_tv"],
        "tenant_tv": mark["tenant_tv"],
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_best(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    by_hrc = sorted(rows, key=lambda r: (float(r["hrc_mae"]), float(r["mark_score"])))
    by_mark = sorted(rows, key=lambda r: (float(r["mark_score"]), float(r["hrc_mae"])))
    payload = {
        "best_hrc": by_hrc[0],
        "best_mark_score": by_mark[0],
        "by_candidate_mean": _candidate_means(rows),
        "n_rows": len(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _split_float(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _split_str(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _split_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _label(prefix: str, seed: int, suffix: str, *, include_seed: bool) -> str:
    if include_seed:
        return f"{prefix}_seed-{seed}_{suffix}"
    return f"{prefix}_{suffix}"


def _candidate_means(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[object, object, object, object], list[dict]] = {}
    for row in rows:
        key = (
            row["categorical_source"],
            row["mark_numeric_blend"],
            row["mark_temperature"],
            row["mark_numeric_noise"],
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (source, blend, temp, noise), group in grouped.items():
        summary.append({
            "categorical_source": source,
            "mark_numeric_blend": blend,
            "mark_temperature": temp,
            "mark_numeric_noise": noise,
            "n_seeds": len(group),
            "mean_hrc_mae": float(np_mean([g["hrc_mae"] for g in group])),
            "mean_mark_score": float(np_mean([g["mark_score"] for g in group])),
        })
    return sorted(summary, key=lambda r: (r["mean_hrc_mae"], r["mean_mark_score"]))


def np_mean(values: list[object]) -> float:
    return sum(float(v) for v in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
