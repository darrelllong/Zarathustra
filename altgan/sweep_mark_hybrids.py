"""Run and summarize neural/reservoir mark hybrid sweeps for NeuralAtlas.

This is a small orchestration wrapper around ``altgan.evaluate_neural_atlas``.
It exists because the mark-head work needs many paired controls with the same
object process, and hand-written shell loops are too easy to misquote.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument("--transition-blends", default="",
                   help="Comma-separated transition blends. Overrides --transition-blend when set.")
    p.add_argument("--local-prob-powers", default="1.0",
                   help="Comma-separated empirical local probability powers.")
    p.add_argument("--object-candidates", default="",
                   help=(
                       "Comma-separated transition:local-power pairs. When set, "
                       "uses these exact object cells instead of the "
                       "--transition-blends x --local-prob-powers grid."
                   ))
    p.add_argument("--force-phase-schedule", action="store_true",
                   help="Force phase from synthetic stream position during evaluation.")
    p.add_argument("--stack-rank-scale", type=float, default=1.0,
                   help="Global reuse stack-rank scale passed to altgan.evaluate_neural_atlas.")
    p.add_argument("--stack-rank-max", type=int, default=-1,
                   help="Optional global reuse stack-rank cap; negative disables.")
    p.add_argument("--stack-rank-tail-pivot", type=int, default=-1,
                   help="Leave scaled ranks at/below this pivot unchanged, then stretch the tail.")
    p.add_argument("--stack-rank-tail-scale", type=float, default=1.0,
                   help="Scale only the rank excess above --stack-rank-tail-pivot.")
    p.add_argument("--stack-reuse-boost-prob", type=float, default=0.0,
                   help="Probability of converting sampled NEW events into reuse events.")
    p.add_argument("--stack-reuse-boost-min-rank", type=int, default=0,
                   help="Minimum LRU rank for injected new-to-reuse events.")
    p.add_argument("--stack-reuse-boost-rank-power", type=float, default=1.0,
                   help="Injected reuse rank shaping; values >1 favor deeper ranks.")
    p.add_argument("--stack-rank-phase-scales", default="",
                   help="Comma-separated per-phase reuse stack-rank scales.")
    p.add_argument("--stack-rank-phase-maxes", default="",
                   help="Comma-separated per-phase reuse stack-rank caps; negative disables a phase cap.")
    p.add_argument("--mark-temperatures", default="1.0,0.5,0.25,0.05")
    p.add_argument("--mark-numeric-noises", default="0.0")
    p.add_argument("--mark-numeric-blends", default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--mark-numeric-blend-spaces", default="raw",
                   help="Comma-separated numeric blend spaces: raw, log.")
    p.add_argument("--mark-numeric-fields", default="both",
                   help="Comma-separated numeric field sets: both, dt, size.")
    p.add_argument("--mark-feedback-numeric-blends", default="",
                   help="Comma-separated feedback-only numeric blends. Empty disables feedback override.")
    p.add_argument("--mark-feedback-numeric-blend-spaces", default="log",
                   help="Comma-separated feedback-only numeric blend spaces: raw, log.")
    p.add_argument("--mark-feedback-numeric-fields", default="both",
                   help="Comma-separated feedback-only numeric field sets: both, dt, size.")
    p.add_argument("--categorical-sources", default="reservoir,neural")
    p.add_argument("--include-reservoir-control", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse existing eval JSONs instead of rerunning completed cells.")
    p.add_argument("--summary-csv", default="")
    p.add_argument("--best-json", default="",
                   help="Optional path for a compact best-candidate summary.")
    p.add_argument("--jobs", type=int, default=1,
                   help="Number of evaluate_neural_atlas subprocesses to run in parallel.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []

    seeds = _split_int(args.seeds) if args.seeds else [args.seed]
    object_candidates = _object_candidates(args)
    include_transition_label = len(object_candidates) > 1
    feedback_grid = _feedback_grid(args)

    if args.include_reservoir_control:
        for transition_blend, local_prob_power in object_candidates:
            for seed in seeds:
                control = out_dir / _label(
                    args.prefix,
                    seed,
                    _transition_suffix(
                        "reservoir_control_eval_100k.json",
                        transition_blend,
                        local_prob_power,
                        include=include_transition_label,
                    ),
                    include_seed=len(seeds) > 1,
                )
                jobs.append((
                    control,
                    seed,
                    transition_blend,
                    local_prob_power,
                    ["--disable-neural-marks"],
                    "reservoir",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ))

    for source in _split_str(args.categorical_sources):
        for transition_blend, local_prob_power in object_candidates:
            for blend_space in _split_blend_spaces(args.mark_numeric_blend_spaces):
                for fields in _split_numeric_fields(args.mark_numeric_fields):
                    for blend in _split_float(args.mark_numeric_blends):
                        for temp in _split_float(args.mark_temperatures):
                            for noise in _split_float(args.mark_numeric_noises):
                                for feedback_blend, feedback_space, feedback_fields in feedback_grid:
                                    for seed in seeds:
                                        feedback_suffix = ""
                                        feedback_extra: list[str] = []
                                        if feedback_blend is not None:
                                            feedback_suffix = (
                                                f"_fbblend-{_slug(feedback_blend)}"
                                                f"_fbspace-{feedback_space}"
                                                f"_fbfields-{feedback_fields}"
                                            )
                                            feedback_extra = [
                                                "--mark-feedback-numeric-blend", str(feedback_blend),
                                                "--mark-feedback-numeric-blend-space", feedback_space,
                                                "--mark-feedback-numeric-fields", feedback_fields,
                                            ]
                                        suffix = _transition_suffix(
                                            (
                                                f"cat-{source}_blend-{_slug(blend)}"
                                                f"_space-{blend_space}_fields-{fields}"
                                                f"_temp-{_slug(temp)}_noise-{_slug(noise)}"
                                                f"{feedback_suffix}_eval_100k.json"
                                            ),
                                            transition_blend,
                                            local_prob_power,
                                            include=include_transition_label,
                                        )
                                        path = out_dir / _label(
                                            args.prefix,
                                            seed,
                                            suffix,
                                            include_seed=len(seeds) > 1,
                                        )
                                        jobs.append((
                                            path,
                                            seed,
                                            transition_blend,
                                            local_prob_power,
                                            [
                                                "--mark-categorical-source", source,
                                                "--mark-numeric-blend", str(blend),
                                                "--mark-numeric-blend-space", blend_space,
                                                "--mark-numeric-fields", fields,
                                                "--mark-temperature", str(temp),
                                                "--mark-numeric-noise", str(noise),
                                                *feedback_extra,
                                            ],
                                            source,
                                            blend,
                                            blend_space,
                                            fields,
                                            temp,
                                            noise,
                                            "" if feedback_blend is None else feedback_blend,
                                            feedback_space,
                                            feedback_fields,
                                        ))

    rows = _run_jobs(args, jobs)
    summary_path = Path(args.summary_csv) if args.summary_csv else out_dir / f"{args.prefix}_summary.csv"
    _write_summary(summary_path, rows)
    print(f"[altgan.sweep_mark_hybrids] wrote {summary_path}", flush=True)
    best_path = Path(args.best_json) if args.best_json else out_dir / f"{args.prefix}_best.json"
    _write_best(best_path, rows)
    print(f"[altgan.sweep_mark_hybrids] wrote {best_path}", flush=True)
    return 0


def _run_jobs(args: argparse.Namespace, jobs: list[tuple]) -> list[dict]:
    if args.jobs <= 1 or len(jobs) <= 1:
        return [_run_one(args, job) for job in jobs]

    rows: list[dict | None] = [None] * len(jobs)
    max_workers = min(max(1, args.jobs), len(jobs))
    print(f"[altgan.sweep_mark_hybrids] running {len(jobs)} cells with jobs={max_workers}",
          flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {
            pool.submit(_run_one, args, job): idx
            for idx, job in enumerate(jobs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            rows[idx] = future.result()
    return [row for row in rows if row is not None]


def _run_one(args: argparse.Namespace, job: tuple) -> dict:
    (
        path,
        seed,
        transition_blend,
        local_prob_power,
        extra,
        source,
        blend,
        blend_space,
        fields,
        temp,
        noise,
        feedback_blend,
        feedback_space,
        feedback_fields,
    ) = job
    _run_eval(
        args,
        path,
        seed=seed,
        transition_blend=transition_blend,
        local_prob_power=local_prob_power,
        extra=extra,
    )
    return _summarize(
        path,
        seed,
        transition_blend,
        local_prob_power,
        source,
        blend,
        blend_space,
        fields,
        temp,
        noise,
        feedback_blend,
        feedback_space,
        feedback_fields,
    )


def _run_eval(
    args: argparse.Namespace,
    output: Path,
    *,
    seed: int,
    transition_blend: float,
    local_prob_power: float,
    extra: list[str],
) -> None:
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
        "--transition-blend", str(transition_blend),
        "--local-prob-power", str(local_prob_power),
        "--temperature", str(args.temperature),
        "--stack-rank-scale", str(args.stack_rank_scale),
        "--stack-rank-max", str(args.stack_rank_max),
        "--stack-rank-tail-pivot", str(args.stack_rank_tail_pivot),
        "--stack-rank-tail-scale", str(args.stack_rank_tail_scale),
        "--stack-reuse-boost-prob", str(args.stack_reuse_boost_prob),
        "--stack-reuse-boost-min-rank", str(args.stack_reuse_boost_min_rank),
        "--stack-reuse-boost-rank-power", str(args.stack_reuse_boost_rank_power),
        "--n-records", str(args.n_records),
        "--n-streams", str(args.n_streams),
        "--seed", str(seed),
        "--real-manifest", args.real_manifest,
        "--output", str(output),
        *extra,
    ]
    if args.force_phase_schedule:
        cmd.append("--force-phase-schedule")
    if args.stack_rank_phase_scales:
        cmd.extend(["--stack-rank-phase-scales", args.stack_rank_phase_scales])
    if args.stack_rank_phase_maxes:
        cmd.extend(["--stack-rank-phase-maxes", args.stack_rank_phase_maxes])
    print(f"[altgan.sweep_mark_hybrids] running {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        env.setdefault(key, "1")
    subprocess.run(cmd, check=True, env=env)


def _summarize(
    path: Path,
    seed: int,
    transition_blend: object,
    local_prob_power: object,
    source: object,
    blend: object,
    blend_space: object,
    fields: object,
    temp: object,
    noise: object,
    feedback_blend: object,
    feedback_space: object,
    feedback_fields: object,
) -> dict:
    data = json.loads(path.read_text())
    mark = data["mark_quality"]
    fake = data["fake"]
    real = data["real"]
    gap = data["gap"]
    return {
        "path": str(path),
        "seed": seed,
        "transition_blend": transition_blend,
        "local_prob_power": local_prob_power,
        "force_phase_schedule": data.get("force_phase_schedule"),
        "stack_rank_scale": data.get("stack_rank_scale"),
        "stack_rank_max": data.get("stack_rank_max"),
        "stack_rank_tail_pivot": data.get("stack_rank_tail_pivot"),
        "stack_rank_tail_scale": data.get("stack_rank_tail_scale"),
        "stack_reuse_boost_prob": data.get("stack_reuse_boost_prob"),
        "stack_reuse_boost_min_rank": data.get("stack_reuse_boost_min_rank"),
        "stack_reuse_boost_rank_power": data.get("stack_reuse_boost_rank_power"),
        "stack_rank_phase_scales": ",".join(str(x) for x in data.get("stack_rank_phase_scales", [])),
        "stack_rank_phase_maxes": ",".join(str(x) for x in data.get("stack_rank_phase_maxes", [])),
        "categorical_source": source,
        "mark_numeric_blend": blend,
        "mark_numeric_blend_space": blend_space,
        "mark_numeric_fields": fields,
        "mark_temperature": temp,
        "mark_numeric_noise": noise,
        "mark_feedback_numeric_blend": feedback_blend,
        "mark_feedback_numeric_blend_space": feedback_space,
        "mark_feedback_numeric_fields": feedback_fields,
        "uses_neural_marks": data.get("uses_neural_marks"),
        "hrc_mae": gap["hrc_mae"],
        "fake_reuse": fake["reuse_access_rate"],
        "real_reuse": real["reuse_access_rate"],
        "fake_stack_median": fake["stack_distance_median"],
        "real_stack_median": real["stack_distance_median"],
        "fake_stack_p90": fake["stack_distance_p90"],
        "real_stack_p90": real["stack_distance_p90"],
        "reuse_local_drift_delta": gap["reuse_decile_local_drift_fake_minus_real"],
        "drift_ts_delta_ratio": gap["drift_ts_delta_w1_ratio"],
        "drift_obj_size_ratio": gap["drift_obj_size_w1_ratio"],
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


def _object_candidates(args: argparse.Namespace) -> list[tuple[float, float]]:
    if not args.object_candidates:
        transition_blends = (
            _split_float(args.transition_blends)
            if args.transition_blends
            else [args.transition_blend]
        )
        return [
            (transition_blend, local_prob_power)
            for transition_blend in transition_blends
            for local_prob_power in _split_float(args.local_prob_powers)
        ]

    pairs: list[tuple[float, float]] = []
    for raw in _split_str(args.object_candidates):
        if ":" not in raw:
            raise ValueError(
                f"invalid object candidate {raw!r}; expected transition:local_power"
            )
        transition_text, local_text = raw.split(":", 1)
        pairs.append((float(transition_text), float(local_text)))
    return pairs


def _split_str(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _split_blend_spaces(text: str) -> list[str]:
    values = _split_str(text)
    invalid = [v for v in values if v not in {"raw", "log"}]
    if invalid:
        raise ValueError(f"invalid mark numeric blend spaces: {', '.join(invalid)}")
    return values


def _split_numeric_fields(text: str) -> list[str]:
    values = _split_str(text)
    invalid = [v for v in values if v not in {"both", "dt", "size"}]
    if invalid:
        raise ValueError(f"invalid mark numeric fields: {', '.join(invalid)}")
    return values


def _feedback_grid(args: argparse.Namespace) -> list[tuple[float | None, str, str]]:
    if not args.mark_feedback_numeric_blends:
        return [(None, "", "")]
    grid: list[tuple[float | None, str, str]] = []
    for blend in _split_float(args.mark_feedback_numeric_blends):
        for space in _split_blend_spaces(args.mark_feedback_numeric_blend_spaces):
            for fields in _split_numeric_fields(args.mark_feedback_numeric_fields):
                grid.append((blend, space, fields))
    return grid


def _split_int(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _slug(value: float) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _label(prefix: str, seed: int, suffix: str, *, include_seed: bool) -> str:
    if include_seed:
        return f"{prefix}_seed-{seed}_{suffix}"
    return f"{prefix}_{suffix}"


def _transition_suffix(suffix: str, transition_blend: float, local_prob_power: float, *, include: bool) -> str:
    if not include:
        return suffix
    return f"tb-{_slug(transition_blend)}_lp-{_slug(local_prob_power)}_{suffix}"


def _candidate_means(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[object, ...], list[dict]] = {}
    for row in rows:
        key = (
            row["transition_blend"],
            row["local_prob_power"],
            row["categorical_source"],
            row["mark_numeric_blend"],
            row["mark_numeric_blend_space"],
            row.get("mark_numeric_fields", "both"),
            row["mark_temperature"],
            row["mark_numeric_noise"],
            row.get("mark_feedback_numeric_blend", ""),
            row.get("mark_feedback_numeric_blend_space", ""),
            row.get("mark_feedback_numeric_fields", ""),
        )
        grouped.setdefault(key, []).append(row)

    summary = []
    for (
        transition_blend,
        local_prob_power,
        source,
        blend,
        blend_space,
        fields,
        temp,
        noise,
        feedback_blend,
        feedback_space,
        feedback_fields,
    ), group in grouped.items():
        summary.append({
            "transition_blend": transition_blend,
            "local_prob_power": local_prob_power,
            "categorical_source": source,
            "mark_numeric_blend": blend,
            "mark_numeric_blend_space": blend_space,
            "mark_numeric_fields": fields,
            "mark_temperature": temp,
            "mark_numeric_noise": noise,
            "mark_feedback_numeric_blend": feedback_blend,
            "mark_feedback_numeric_blend_space": feedback_space,
            "mark_feedback_numeric_fields": feedback_fields,
            "n_seeds": len(group),
            "mean_hrc_mae": float(np_mean([g["hrc_mae"] for g in group])),
            "mean_mark_score": float(np_mean([g["mark_score"] for g in group])),
            "mean_reuse_local_drift_delta": float(np_mean([
                g["reuse_local_drift_delta"] for g in group
            ])),
            "mean_drift_ts_delta_ratio": float(np_mean([
                g["drift_ts_delta_ratio"] for g in group
            ])),
            "mean_drift_obj_size_ratio": float(np_mean([
                g["drift_obj_size_ratio"] for g in group
            ])),
        })
    return sorted(summary, key=lambda r: (r["mean_hrc_mae"], r["mean_mark_score"]))


def np_mean(values: list[object]) -> float:
    return sum(float(v) for v in values) / len(values)


if __name__ == "__main__":
    raise SystemExit(main())
