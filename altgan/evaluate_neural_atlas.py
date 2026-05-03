"""Evaluate a profile-conditioned NeuralAtlas model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.long_rollout_eval import _gap, _metrics_for_stream, _per_stream_obj_ids, _sample_real_stream  # noqa: E402

from .mark_quality import mark_quality  # noqa: E402
from .neural_atlas import NeuralAtlasModel  # noqa: E402
from .train_neural_atlas import _cond_lookup_keys, _load_file_characterizations  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--char-file", required=True)
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--source-traces", default="",
                   help="Comma-separated source trace basenames for stream conditioning.")
    p.add_argument("--condition-from-real-manifest", action="store_true",
                   help="Use the real manifest's stream files as source conditioning.")
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--transition-blend", type=float, default=0.75,
                   help="1.0 means pure neural transitions; 0.0 means nearest-file atlas.")
    p.add_argument("--local-prob-power", type=float, default=1.0,
                   help="Power applied to nearest-file empirical initial/transition probabilities before blending.")
    p.add_argument("--force-phase-schedule", action="store_true",
                   help="For phase atlases, force phase from synthetic stream position.")
    p.add_argument("--stack-rank-scale", type=float, default=1.0,
                   help="Scale sampled reuse stack ranks before LRU lookup.")
    p.add_argument("--stack-rank-max", type=int, default=-1,
                   help="Optional maximum reuse stack rank after scaling; negative disables.")
    p.add_argument("--stack-rank-tail-pivot", type=int, default=-1,
                   help="Leave scaled ranks at or below this pivot unchanged before tail stretch; negative disables.")
    p.add_argument("--stack-rank-tail-scale", type=float, default=1.0,
                   help="Scale only the excess above --stack-rank-tail-pivot.")
    p.add_argument("--stack-reuse-boost-prob", type=float, default=0.0,
                   help="Probability of converting a sampled NEW event into a reuse when the stack is nonempty.")
    p.add_argument("--stack-reuse-boost-min-rank", type=int, default=0,
                   help="Minimum LRU rank for injected new-to-reuse events.")
    p.add_argument("--stack-reuse-boost-rank-power", type=float, default=1.0,
                   help="Power shaping for injected reuse ranks; values >1 favor deeper ranks.")
    p.add_argument("--stack-reuse-drop-prob", type=float, default=0.0,
                   help="Probability of converting a sampled reuse into a new object.")
    p.add_argument("--stack-adj-dup-prob", type=float, default=0.0,
                   help="Probability that a sampled reuse emits the current MRU object.")
    p.add_argument("--stack-hot-pool-prob", type=float, default=0.0,
                   help="Probability of redirecting an ordinary sampled reuse to the recent hot pool.")
    p.add_argument("--stack-hot-pool-k", type=int, default=100,
                   help="Number of recent hot objects eligible for hot-pool reuse.")
    p.add_argument("--stack-hot-pool-window", type=int, default=5000,
                   help="Sliding window length used to estimate hot-pool frequency.")
    p.add_argument("--stack-hot-pool-weight-power", type=float, default=1.0,
                   help="Power applied to hot-pool frequency weights.")
    p.add_argument("--stack-hot-pool-max-search", type=int, default=0,
                   help="Optional maximum stack prefix searched for hot-pool objects; 0 searches full stack.")
    p.add_argument("--stack-hot-pool-min-age", type=int, default=0,
                   help="Minimum stream positions since last emission before a hot-pool object can be reused.")
    p.add_argument("--stack-frequency-pool-prob", type=float, default=0.0,
                   help="Probability of redirecting a sampled reuse to a long-memory frequent object.")
    p.add_argument("--stack-frequency-pool-k", type=int, default=100,
                   help="Number of long-memory frequent objects eligible for frequency-pool reuse.")
    p.add_argument("--stack-frequency-pool-max-candidates", type=int, default=1000,
                   help="Maximum approximate frequency candidates retained before pruning to top-k.")
    p.add_argument("--stack-frequency-pool-weight-power", type=float, default=1.0,
                   help="Power applied to long-memory frequency-pool counts.")
    p.add_argument("--stack-frequency-pool-min-age", type=int, default=0,
                   help="Minimum stream positions since last emission before a frequency-pool object can be reused.")
    p.add_argument("--stack-frequency-pool-min-rank", type=int, default=0,
                   help="Minimum current LRU stack rank for frequency-pool redirects.")
    p.add_argument("--stack-frequency-pool-max-rank", type=int, default=-1,
                   help="Maximum current LRU stack rank for frequency-pool redirects; negative disables.")
    p.add_argument("--stack-frequency-pool-sample-attempts", type=int, default=8,
                   help="Number of frequency-pool samples tried to satisfy age/rank filters.")
    p.add_argument("--stack-tail-reuse-prob", type=float, default=0.0,
                   help="Probability of redirecting a sampled reuse to a deep stack-tail object.")
    p.add_argument("--stack-tail-reuse-min-frac", type=float, default=0.5,
                   help="Minimum stack fraction for --stack-tail-reuse-prob redirects.")
    p.add_argument("--stack-tail-reuse-rank-power", type=float, default=1.0,
                   help="Power shaping for tail-reuse rank redirects; values <1 favor deeper tail ranks.")
    p.add_argument("--stack-recent-pool-prob", type=float, default=0.0,
                   help="Probability of redirecting a sampled reuse to a recent emitted object.")
    p.add_argument("--stack-recent-pool-window", type=int, default=200,
                   help="Number of emitted objects kept for --stack-recent-pool-prob redirects.")
    p.add_argument("--stack-rank-phase-scales", default="",
                   help="Comma-separated per-phase stack-rank scales; overrides the global scale.")
    p.add_argument("--stack-rank-phase-maxes", default="",
                   help="Comma-separated per-phase stack-rank caps; negative disables a phase cap.")
    p.add_argument("--disable-neural-marks", action="store_true",
                   help="Ignore an attached neural mark head and use atlas reservoir marks.")
    p.add_argument("--mark-temperature", type=float, default=None,
                   help="Sampling temperature for an attached neural mark head.")
    p.add_argument("--mark-numeric-noise", type=float, default=0.05,
                   help="Gaussian noise applied to neural mark dt/size z-scores.")
    p.add_argument("--mark-numeric-blend", type=float, default=1.0,
                   help="Blend reservoir and neural dt/size: 0.0 reservoir, 1.0 neural.")
    p.add_argument("--mark-numeric-blend-space", choices=["raw", "log"], default="raw",
                   help="Blend neural/reservoir numeric marks in raw units or log-transformed units.")
    p.add_argument("--mark-numeric-fields", choices=["both", "dt", "size"], default="both",
                   help="Apply numeric blending to both marks, only timing dt, or only obj_size.")
    p.add_argument("--mark-categorical-source", choices=["neural", "reservoir"], default="neural",
                   help="Use neural or reservoir opcode/tenant when neural marks are attached.")
    p.add_argument("--mark-feedback-numeric-blend", type=float, default=None,
                   help="Optional numeric blend used only for autoregressive mark feedback.")
    p.add_argument("--mark-feedback-numeric-blend-space", choices=["raw", "log"], default="log",
                   help="Blend space for feedback-only numeric marks.")
    p.add_argument("--mark-feedback-numeric-fields", choices=["both", "dt", "size"], default="both",
                   help="Numeric fields affected by feedback-only blending.")
    p.add_argument("--real-manifest", default="")
    p.add_argument("--cache-sizes", default="")
    p.add_argument("--output", default="")
    p.add_argument("--fake-output", default="",
                   help="Optional CSV path for the generated fake trace.")
    p.add_argument("--real-output", default="",
                   help="Optional CSV path for the sampled real manifest trace.")
    p.add_argument("--cachesim-bin", default="",
                   help="Optional tools/cachesim binary; when set, run fake and real through cachesim.")
    p.add_argument("--cachesim-output", default="",
                   help="Optional JSON path for the cachesim comparison report.")
    p.add_argument("--cachesim-cache-sizes", default="32,128,512,2048,8192",
                   help="Comma-separated cache sizes for --cachesim-bin.")
    p.add_argument("--cachesim-policies", default="lru,arc,fifo,sieve,slru,car",
                   help="Comma-separated cachesim policies for --cachesim-bin.")
    p.add_argument("--progress-interval", type=int, default=0,
                   help="Print generation progress every N records per stream; 0 disables.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = NeuralAtlasModel.load(args.model)

    cond_lookup = _load_file_characterizations(args.char_file, cond_dim=args.cond_dim)
    source_names = [s.strip() for s in args.source_traces.split(",") if s.strip()]

    real_df, real_manifest = _sample_real_stream(
        args.trace_dir,
        args.fmt,
        args.n_records,
        args.n_streams,
        args.seed,
        manifest_path=args.real_manifest,
    )
    if args.condition_from_real_manifest:
        source_names = _source_names_from_manifest(real_manifest)

    conds = None
    if source_names:
        conds = np.vstack([_lookup_cond(cond_lookup, name, args.cond_dim) for name in source_names])

    saved_mark_model = getattr(model, "mark_model", None)
    if args.disable_neural_marks:
        model.mark_model = None
    fake_df = model.generate(
        args.n_records,
        n_streams=args.n_streams,
        seed=args.seed,
        conds=conds,
        temperature=args.temperature,
        transition_blend=args.transition_blend,
        local_prob_power=args.local_prob_power,
        force_phase_schedule=args.force_phase_schedule,
        stack_rank_scale=args.stack_rank_scale,
        stack_rank_max=None if args.stack_rank_max < 0 else args.stack_rank_max,
        stack_rank_tail_pivot=None if args.stack_rank_tail_pivot < 0 else args.stack_rank_tail_pivot,
        stack_rank_tail_scale=args.stack_rank_tail_scale,
        stack_reuse_boost_prob=args.stack_reuse_boost_prob,
        stack_reuse_boost_min_rank=args.stack_reuse_boost_min_rank,
        stack_reuse_boost_rank_power=args.stack_reuse_boost_rank_power,
        stack_reuse_drop_prob=args.stack_reuse_drop_prob,
        stack_adj_dup_prob=args.stack_adj_dup_prob,
        stack_hot_pool_prob=args.stack_hot_pool_prob,
        stack_hot_pool_k=args.stack_hot_pool_k,
        stack_hot_pool_window=args.stack_hot_pool_window,
        stack_hot_pool_weight_power=args.stack_hot_pool_weight_power,
        stack_hot_pool_max_search=args.stack_hot_pool_max_search,
        stack_hot_pool_min_age=args.stack_hot_pool_min_age,
        stack_frequency_pool_prob=args.stack_frequency_pool_prob,
        stack_frequency_pool_k=args.stack_frequency_pool_k,
        stack_frequency_pool_max_candidates=args.stack_frequency_pool_max_candidates,
        stack_frequency_pool_weight_power=args.stack_frequency_pool_weight_power,
        stack_frequency_pool_min_age=args.stack_frequency_pool_min_age,
        stack_frequency_pool_min_rank=args.stack_frequency_pool_min_rank,
        stack_frequency_pool_max_rank=None if args.stack_frequency_pool_max_rank < 0 else args.stack_frequency_pool_max_rank,
        stack_frequency_pool_sample_attempts=args.stack_frequency_pool_sample_attempts,
        stack_tail_reuse_prob=args.stack_tail_reuse_prob,
        stack_tail_reuse_min_frac=args.stack_tail_reuse_min_frac,
        stack_tail_reuse_rank_power=args.stack_tail_reuse_rank_power,
        stack_recent_pool_prob=args.stack_recent_pool_prob,
        stack_recent_pool_window=args.stack_recent_pool_window,
        stack_rank_phase_scales=_parse_float_list(args.stack_rank_phase_scales),
        stack_rank_phase_maxes=_parse_int_list(args.stack_rank_phase_maxes),
        mark_temperature=args.mark_temperature,
        mark_numeric_noise=args.mark_numeric_noise,
        mark_numeric_blend=args.mark_numeric_blend,
        mark_numeric_blend_space=args.mark_numeric_blend_space,
        mark_numeric_fields=args.mark_numeric_fields,
        mark_categorical_source=args.mark_categorical_source,
        mark_feedback_numeric_blend=args.mark_feedback_numeric_blend,
        mark_feedback_numeric_blend_space=args.mark_feedback_numeric_blend_space,
        mark_feedback_numeric_fields=args.mark_feedback_numeric_fields,
        progress_interval=args.progress_interval,
    )
    if args.disable_neural_marks:
        model.mark_model = saved_mark_model
    if args.fake_output:
        fake_out = Path(args.fake_output)
        fake_out.parent.mkdir(parents=True, exist_ok=True)
        fake_df.to_csv(fake_out, index=False)
        print(f"[altgan.evaluate_neural_atlas] wrote fake trace {fake_out}")
    if args.real_output:
        real_out = Path(args.real_output)
        real_out.parent.mkdir(parents=True, exist_ok=True)
        real_df.to_csv(real_out, index=False)
        print(f"[altgan.evaluate_neural_atlas] wrote real trace {real_out}")

    if args.cache_sizes:
        cache_sizes = np.array([int(x) for x in args.cache_sizes.split(",") if x.strip()],
                               dtype=np.int64)
    else:
        real_streams = _per_stream_obj_ids(real_df)
        footprint = int(np.mean([np.unique(s).size for s in real_streams])) if real_streams else 2
        cache_sizes = np.unique(np.geomspace(max(1, footprint // 1000), max(footprint, 2), 20)
                                .astype(np.int64))

    fake_m = _metrics_for_stream(fake_df, cache_sizes)
    real_m = _metrics_for_stream(real_df, cache_sizes)
    gap_m = _gap(fake_m, real_m)
    mark_m = mark_quality(fake_df, real_df)
    out_path = Path(args.output) if args.output else Path(args.model).with_suffix("").with_suffix(".neural_atlas_eval.json")
    result = {
        "model": args.model,
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "seed": args.seed,
        "n_records": args.n_records,
        "n_streams": args.n_streams,
        "temperature": args.temperature,
        "transition_blend": args.transition_blend,
        "local_prob_power": args.local_prob_power,
        "force_phase_schedule": args.force_phase_schedule,
        "stack_rank_scale": args.stack_rank_scale,
        "stack_rank_max": args.stack_rank_max,
        "stack_rank_tail_pivot": args.stack_rank_tail_pivot,
        "stack_rank_tail_scale": args.stack_rank_tail_scale,
        "stack_reuse_boost_prob": args.stack_reuse_boost_prob,
        "stack_reuse_boost_min_rank": args.stack_reuse_boost_min_rank,
        "stack_reuse_boost_rank_power": args.stack_reuse_boost_rank_power,
        "stack_reuse_drop_prob": args.stack_reuse_drop_prob,
        "stack_adj_dup_prob": args.stack_adj_dup_prob,
        "stack_hot_pool_prob": args.stack_hot_pool_prob,
        "stack_hot_pool_k": args.stack_hot_pool_k,
        "stack_hot_pool_window": args.stack_hot_pool_window,
        "stack_hot_pool_weight_power": args.stack_hot_pool_weight_power,
        "stack_hot_pool_max_search": args.stack_hot_pool_max_search,
        "stack_hot_pool_min_age": args.stack_hot_pool_min_age,
        "stack_frequency_pool_prob": args.stack_frequency_pool_prob,
        "stack_frequency_pool_k": args.stack_frequency_pool_k,
        "stack_frequency_pool_max_candidates": args.stack_frequency_pool_max_candidates,
        "stack_frequency_pool_weight_power": args.stack_frequency_pool_weight_power,
        "stack_frequency_pool_min_age": args.stack_frequency_pool_min_age,
        "stack_frequency_pool_min_rank": args.stack_frequency_pool_min_rank,
        "stack_frequency_pool_max_rank": args.stack_frequency_pool_max_rank,
        "stack_frequency_pool_sample_attempts": args.stack_frequency_pool_sample_attempts,
        "stack_tail_reuse_prob": args.stack_tail_reuse_prob,
        "stack_tail_reuse_min_frac": args.stack_tail_reuse_min_frac,
        "stack_tail_reuse_rank_power": args.stack_tail_reuse_rank_power,
        "stack_recent_pool_prob": args.stack_recent_pool_prob,
        "stack_recent_pool_window": args.stack_recent_pool_window,
        "stack_rank_phase_scales": _parse_float_list(args.stack_rank_phase_scales),
        "stack_rank_phase_maxes": _parse_int_list(args.stack_rank_phase_maxes),
        "mark_temperature": args.mark_temperature,
        "mark_numeric_noise": args.mark_numeric_noise,
        "mark_numeric_blend": args.mark_numeric_blend,
        "mark_numeric_blend_space": args.mark_numeric_blend_space,
        "mark_numeric_fields": args.mark_numeric_fields,
        "mark_categorical_source": args.mark_categorical_source,
        "mark_feedback_numeric_blend": args.mark_feedback_numeric_blend,
        "mark_feedback_numeric_blend_space": args.mark_feedback_numeric_blend_space,
        "mark_feedback_numeric_fields": args.mark_feedback_numeric_fields,
        "uses_neural_marks": saved_mark_model is not None and not args.disable_neural_marks,
        "progress_interval": args.progress_interval,
        "source_traces": source_names,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
        "mark_quality": mark_m,
        "real_manifest": real_manifest,
    }
    if args.cachesim_bin:
        cachesim_output = (
            Path(args.cachesim_output)
            if args.cachesim_output
            else out_path.with_suffix(".cachesim.json")
        )
        result["cachesim"] = _run_cachesim_comparison(
            fake_df,
            real_df,
            binary=Path(args.cachesim_bin),
            cache_sizes=args.cachesim_cache_sizes,
            policies=args.cachesim_policies,
            output=cachesim_output,
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[altgan.evaluate_neural_atlas] wrote {out_path}")
    summary = {
        "hrc_mae": gap_m["hrc_mae"],
        "fake_reuse_access": fake_m["reuse_access_rate"],
        "real_reuse_access": real_m["reuse_access_rate"],
        "fake_stack_median": fake_m["stack_distance_median"],
        "real_stack_median": real_m["stack_distance_median"],
        "fake_stack_p90": fake_m["stack_distance_p90"],
        "real_stack_p90": real_m["stack_distance_p90"],
        "mark_score": mark_m["mark_score"],
    }
    if "cachesim" in result:
        summary["cachesim_mean_hrc_mae"] = result["cachesim"]["mean_hrc_mae"]
    print(json.dumps(summary, indent=2))
    return 0


def _source_names_from_manifest(manifest: dict) -> list[str]:
    names = []
    for entries in manifest.get("streams", []):
        if not entries:
            continue
        names.append(Path(entries[0]["path"]).name)
    return names


def _lookup_cond(cond_lookup: dict, name_or_path: str, cond_dim: int) -> np.ndarray:
    keys = _cond_lookup_keys(name_or_path)
    for key in keys:
        val = cond_lookup.get(key)
        if val is not None:
            arr = val.detach().cpu().numpy().astype(np.float32)
            if len(arr) < cond_dim:
                arr = np.pad(arr, (0, cond_dim - len(arr)))
            return arr[:cond_dim]
    raise KeyError(f"no characterization vector for {name_or_path}")


def _run_cachesim_comparison(
    fake_df,
    real_df,
    *,
    binary: Path,
    cache_sizes: str,
    policies: str,
    output: Path,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    stem = output.with_suffix("")
    fake_csv = stem.parent / f"{stem.name}.fake_namespaced.csv"
    real_csv = stem.parent / f"{stem.name}.real_namespaced.csv"
    _write_namespaced_csv(fake_df, fake_csv)
    _write_namespaced_csv(real_df, real_csv)

    fake_rows = _run_cachesim(binary, fake_csv, cache_sizes, policies)
    real_rows = _run_cachesim(binary, real_csv, cache_sizes, policies)
    report = _compare_cachesim_rows(fake_rows, real_rows, cache_sizes, policies)
    report.update({
        "fake_trace": str(fake_csv),
        "real_trace": str(real_csv),
        "binary": str(binary),
    })
    output.write_text(json.dumps(report, indent=2))
    print(f"[altgan.evaluate_neural_atlas] wrote cachesim report {output}")
    return report


def _write_namespaced_csv(df, path: Path) -> None:
    out = df.copy()
    if "stream_id" in out.columns:
        out["obj_id"] = (
            out["stream_id"].astype("int64") * 10_000_000_000_000
            + out["obj_id"].astype("int64")
        )
    out.to_csv(path, index=False)


def _run_cachesim(binary: Path, trace: Path, cache_sizes: str, policies: str) -> list[dict]:
    proc = subprocess.run(
        [
            str(binary),
            "--trace", str(trace),
            "--format", "csv",
            "--policy", policies,
            "--cache-sizes", cache_sizes,
            "--out", "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cachesim failed for {trace}: {proc.stderr or proc.stdout}")
    match = re.search(r"(\[\s*\{.*\}\s*\])", proc.stdout, re.DOTALL)
    if not match:
        raise RuntimeError(f"cachesim produced no JSON for {trace}: {proc.stdout[:500]}")
    return json.loads(match.group(1))


def _compare_cachesim_rows(
    fake_rows: list[dict],
    real_rows: list[dict],
    cache_sizes: str,
    policies: str,
) -> dict:
    fake = _group_cachesim_rows(fake_rows)
    real = _group_cachesim_rows(real_rows)
    requested_policies = [p.strip() for p in policies.split(",") if p.strip()]
    requested_sizes = [int(s.strip()) for s in cache_sizes.split(",") if s.strip()]
    by_policy = {}
    maes = []
    for policy in requested_policies:
        fake_policy = fake.get(policy, {})
        real_policy = real.get(policy, {})
        deltas = []
        fake_miss = []
        real_miss = []
        for size in requested_sizes:
            f = fake_policy[size]
            r = real_policy[size]
            fake_miss.append(f)
            real_miss.append(r)
            deltas.append(f - r)
        mae = float(sum(abs(d) for d in deltas) / max(len(deltas), 1))
        maes.append(mae)
        by_policy[policy] = {
            "fake_miss_ratio": fake_miss,
            "real_miss_ratio": real_miss,
            "delta": deltas,
            "hrc_mae": mae,
        }
    return {
        "cache_sizes": requested_sizes,
        "policies": requested_policies,
        "by_policy": by_policy,
        "mean_hrc_mae": float(sum(maes) / max(len(maes), 1)),
    }


def _group_cachesim_rows(rows: list[dict]) -> dict[str, dict[int, float]]:
    grouped: dict[str, dict[int, float]] = {}
    for row in rows:
        policy = row["policy"]
        for cache_row in row.get("per_cache_size", []):
            grouped.setdefault(policy, {})[int(cache_row["size"])] = float(cache_row["miss_ratio"])
    return grouped


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
