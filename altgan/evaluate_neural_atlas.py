"""Evaluate a profile-conditioned NeuralAtlas model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import load_file_characterizations  # noqa: E402
from llgan.long_rollout_eval import _gap, _metrics_for_stream, _per_stream_obj_ids, _sample_real_stream  # noqa: E402

from .mark_quality import mark_quality  # noqa: E402
from .neural_atlas import NeuralAtlasModel  # noqa: E402


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
    p.add_argument("--real-manifest", default="")
    p.add_argument("--cache-sizes", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = NeuralAtlasModel.load(args.model)

    cond_lookup = load_file_characterizations(args.char_file, cond_dim=args.cond_dim)
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
        stack_rank_phase_scales=_parse_float_list(args.stack_rank_phase_scales),
        stack_rank_phase_maxes=_parse_int_list(args.stack_rank_phase_maxes),
        mark_temperature=args.mark_temperature,
        mark_numeric_noise=args.mark_numeric_noise,
        mark_numeric_blend=args.mark_numeric_blend,
        mark_numeric_blend_space=args.mark_numeric_blend_space,
        mark_numeric_fields=args.mark_numeric_fields,
        mark_categorical_source=args.mark_categorical_source,
    )
    if args.disable_neural_marks:
        model.mark_model = saved_mark_model

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
        "stack_rank_phase_scales": _parse_float_list(args.stack_rank_phase_scales),
        "stack_rank_phase_maxes": _parse_int_list(args.stack_rank_phase_maxes),
        "mark_temperature": args.mark_temperature,
        "mark_numeric_noise": args.mark_numeric_noise,
        "mark_numeric_blend": args.mark_numeric_blend,
        "mark_numeric_blend_space": args.mark_numeric_blend_space,
        "mark_numeric_fields": args.mark_numeric_fields,
        "mark_categorical_source": args.mark_categorical_source,
        "uses_neural_marks": saved_mark_model is not None and not args.disable_neural_marks,
        "source_traces": source_names,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
        "mark_quality": mark_m,
        "real_manifest": real_manifest,
    }
    out_path = Path(args.output) if args.output else Path(args.model).with_suffix("").with_suffix(".neural_atlas_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[altgan.evaluate_neural_atlas] wrote {out_path}")
    print(json.dumps({
        "hrc_mae": gap_m["hrc_mae"],
        "fake_reuse_access": fake_m["reuse_access_rate"],
        "real_reuse_access": real_m["reuse_access_rate"],
        "fake_stack_median": fake_m["stack_distance_median"],
        "real_stack_median": real_m["stack_distance_median"],
        "fake_stack_p90": fake_m["stack_distance_p90"],
        "real_stack_p90": real_m["stack_distance_p90"],
        "mark_score": mark_m["mark_score"],
    }, indent=2))
    return 0


def _source_names_from_manifest(manifest: dict) -> list[str]:
    names = []
    for entries in manifest.get("streams", []):
        if not entries:
            continue
        names.append(Path(entries[0]["path"]).name)
    return names


def _lookup_cond(cond_lookup: dict, name_or_path: str, cond_dim: int) -> np.ndarray:
    name = Path(name_or_path).name
    keys = [name]
    for suffix in (".zst", ".gz"):
        if name.endswith(suffix):
            keys.append(name[: -len(suffix)])
    for key in keys:
        val = cond_lookup.get(key)
        if val is not None:
            arr = val.detach().cpu().numpy().astype(np.float32)
            if len(arr) < cond_dim:
                arr = np.pad(arr, (0, cond_dim - len(arr)))
            return arr[:cond_dim]
    raise KeyError(f"no characterization vector for {name_or_path}")


def _parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
