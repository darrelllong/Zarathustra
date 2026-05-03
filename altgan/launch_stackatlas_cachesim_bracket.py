"""Launch generic NeuralAtlas official cache-sim brackets.

This wrapper generates fake traces with ``altgan.evaluate_neural_atlas`` and
then scores the literal race surface with ``llgan.cachesim_eval`` against an
official StackAtlas reference CSV.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    name: str
    seed: int = 42
    transition_blend: float = 0.2
    local_prob_power: float = 0.9
    rank_scale: float = 1.0
    rank_max: int = -1
    rank_tail_pivot: int = -1
    rank_tail_scale: float = 1.0
    rank_position_scales: str = ""
    adj_dup_prob: float = 0.0
    adj_dup_position_probs: str = ""
    adj_dup_min_rank: int = 0
    adj_dup_max_rank: int = 0
    adj_dup_band_prob: float = 1.0
    rank_band_reuse_prob: float = 0.0
    rank_band_reuse_position_probs: str = ""
    rank_band_reuse_min_rank: int = 0
    rank_band_reuse_max_rank: int = -1
    rank_band_reuse_power: float = 1.0
    hot_pool_prob: float = 0.45
    hot_pool_position_probs: str = ""
    hot_pool_k: int = 75
    hot_pool_window: int = 10000
    hot_pool_weight_power: float = 1.0
    hot_pool_min_age: int = 0
    frequency_pool_prob: float = 0.0
    frequency_pool_k: int = 100
    frequency_pool_max_candidates: int = 1000
    frequency_pool_refresh_interval: int = 512
    frequency_pool_min_count_rank: int = 0
    frequency_pool_max_count_rank: int = -1
    frequency_pool_weight_power: float = 1.0
    frequency_pool_min_age: int = 0
    frequency_pool_min_rank: int = 0
    frequency_pool_max_rank: int = -1
    frequency_pool_sample_attempts: int = 8
    recent_pool_prob: float = 0.15
    recent_pool_position_probs: str = ""
    recent_pool_window: int = 16
    tail_reuse_prob: float = 0.10
    tail_reuse_position_probs: str = ""
    tail_reuse_min_frac: float = 0.5
    tail_reuse_rank_power: float = 1.0
    reuse_boost_prob: float = 0.0
    reuse_boost_min_rank: int = 32768
    reuse_boost_rank_power: float = 2.0
    reuse_drop_prob: float = 0.0
    reuse_drop_position_probs: str = ""


def _parse_spec(text: str) -> Spec:
    if ":" in text:
        name, rest = text.split(":", 1)
    else:
        name, rest = "", text
    values: dict[str, str] = {}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"spec component {part!r} must be key=value"
            )
        key, value = part.split("=", 1)
        values[key.strip().replace("-", "_")] = value.strip()
    defaults = Spec(name=name)
    aliases = {
        "tb": "transition_blend",
        "lp": "local_prob_power",
        "rank": "rank_scale",
        "rankmax": "rank_max",
        "tailpivot": "rank_tail_pivot",
        "tailscale": "rank_tail_scale",
        "rank_pos": "rank_position_scales",
        "rankpos": "rank_position_scales",
        "adj": "adj_dup_prob",
        "adj_pos": "adj_dup_position_probs",
        "adjpos": "adj_dup_position_probs",
        "adj_min": "adj_dup_min_rank",
        "adjmin": "adj_dup_min_rank",
        "adj_max": "adj_dup_max_rank",
        "adjmax": "adj_dup_max_rank",
        "adj_band": "adj_dup_band_prob",
        "adjband": "adj_dup_band_prob",
        "rank_band": "rank_band_reuse_prob",
        "rankband": "rank_band_reuse_prob",
        "rb": "rank_band_reuse_prob",
        "rank_band_pos": "rank_band_reuse_position_probs",
        "rankband_pos": "rank_band_reuse_position_probs",
        "rb_pos": "rank_band_reuse_position_probs",
        "rbpos": "rank_band_reuse_position_probs",
        "rank_band_min": "rank_band_reuse_min_rank",
        "rankband_min": "rank_band_reuse_min_rank",
        "rb_min": "rank_band_reuse_min_rank",
        "rbmin": "rank_band_reuse_min_rank",
        "rank_band_max": "rank_band_reuse_max_rank",
        "rankband_max": "rank_band_reuse_max_rank",
        "rb_max": "rank_band_reuse_max_rank",
        "rbmax": "rank_band_reuse_max_rank",
        "rank_band_power": "rank_band_reuse_power",
        "rankband_power": "rank_band_reuse_power",
        "rb_power": "rank_band_reuse_power",
        "rbpow": "rank_band_reuse_power",
        "hp": "hot_pool_prob",
        "hp_pos": "hot_pool_position_probs",
        "hppos": "hot_pool_position_probs",
        "k": "hot_pool_k",
        "hpwin": "hot_pool_window",
        "hpwp": "hot_pool_weight_power",
        "minage": "hot_pool_min_age",
        "fp": "frequency_pool_prob",
        "freq": "frequency_pool_prob",
        "fpk": "frequency_pool_k",
        "fp_k": "frequency_pool_k",
        "fpmaxcand": "frequency_pool_max_candidates",
        "fp_max_candidates": "frequency_pool_max_candidates",
        "fprefresh": "frequency_pool_refresh_interval",
        "fp_refresh": "frequency_pool_refresh_interval",
        "fp_count_min": "frequency_pool_min_count_rank",
        "fpmincount": "frequency_pool_min_count_rank",
        "fpr_min": "frequency_pool_min_count_rank",
        "fp_count_max": "frequency_pool_max_count_rank",
        "fpmaxcount": "frequency_pool_max_count_rank",
        "fpr_max": "frequency_pool_max_count_rank",
        "fpwp": "frequency_pool_weight_power",
        "fp_weight": "frequency_pool_weight_power",
        "fpage": "frequency_pool_min_age",
        "fp_min_age": "frequency_pool_min_age",
        "fp_min_rank": "frequency_pool_min_rank",
        "fpminrank": "frequency_pool_min_rank",
        "fp_max_rank": "frequency_pool_max_rank",
        "fpmaxrank": "frequency_pool_max_rank",
        "fp_attempts": "frequency_pool_sample_attempts",
        "fpattempts": "frequency_pool_sample_attempts",
        "rp": "recent_pool_prob",
        "rp_pos": "recent_pool_position_probs",
        "rppos": "recent_pool_position_probs",
        "win": "recent_pool_window",
        "tail": "tail_reuse_prob",
        "tail_pos": "tail_reuse_position_probs",
        "tailpos": "tail_reuse_position_probs",
        "mf": "tail_reuse_min_frac",
        "tail_power": "tail_reuse_rank_power",
        "tpow": "tail_reuse_rank_power",
        "reuse": "reuse_boost_prob",
        "reuse_min": "reuse_boost_min_rank",
        "reuse_power": "reuse_boost_rank_power",
        "drop": "reuse_drop_prob",
        "reuse_drop": "reuse_drop_prob",
        "drop_pos": "reuse_drop_position_probs",
        "reuse_drop_pos": "reuse_drop_position_probs",
    }
    fields = {field.name: field.type for field in Spec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs = {"name": defaults.name}
    for key, value in values.items():
        field = aliases.get(key, key)
        if field not in fields:
            raise argparse.ArgumentTypeError(f"unknown spec key {key!r}")
        if field == "name":
            kwargs[field] = value
        else:
            current = getattr(defaults, field)
            if isinstance(current, str):
                kwargs[field] = value.replace(";", ",")
            else:
                kwargs[field] = type(current)(value)
    spec = Spec(**kwargs)
    if not spec.name:
        object.__setattr__(spec, "name", _auto_name(spec))
    return spec


def _auto_name(spec: Spec) -> str:
    return (
        f"seed{spec.seed}_tb{_tag(spec.transition_blend)}"
        f"_lp{_tag(spec.local_prob_power)}_rank{_tag(spec.rank_scale)}"
        f"max{spec.rank_max}_tailp{spec.rank_tail_pivot}"
        f"s{_tag(spec.rank_tail_scale)}_adj{_tag(spec.adj_dup_prob)}"
        f"_rb{_tag(spec.rank_band_reuse_prob)}"
        f"_hp{_tag(spec.hot_pool_prob)}k{spec.hot_pool_k}"
        f"w{spec.hot_pool_window}wp{_tag(spec.hot_pool_weight_power)}"
        f"_minage{spec.hot_pool_min_age}_fp{_tag(spec.frequency_pool_prob)}"
        f"k{spec.frequency_pool_k}cr{spec.frequency_pool_min_count_rank}"
        f"-{spec.frequency_pool_max_count_rank}"
        f"_rp{_tag(spec.recent_pool_prob)}"
        f"w{spec.recent_pool_window}_tail{_tag(spec.tail_reuse_prob)}"
        f"mf{_tag(spec.tail_reuse_min_frac)}pow{_tag(spec.tail_reuse_rank_power)}"
        f"_drop{_tag(spec.reuse_drop_prob)}"
    )


def _tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True)
    p.add_argument("--spec", action="append", type=_parse_spec, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", default="oracle_general")
    p.add_argument(
        "--char-file",
        default="/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl",
    )
    p.add_argument("--real-manifest", required=True)
    p.add_argument("--official-ref", required=True)
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--eval-tag", default="official")
    p.add_argument("--progress-interval", type=int, default=50_000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--no-force-phase-schedule",
        dest="force_phase",
        action="store_false",
    )
    p.set_defaults(force_phase=True)
    return p.parse_args()


def _eval_cmd(args: argparse.Namespace, spec: Spec, fake: Path, eval_json: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "altgan.evaluate_neural_atlas",
        "--model",
        args.model,
        "--trace-dir",
        args.trace_dir,
        "--fmt",
        args.fmt,
        "--char-file",
        args.char_file,
        "--cond-dim",
        str(args.cond_dim),
        "--condition-from-real-manifest",
        "--real-manifest",
        args.real_manifest,
        "--transition-blend",
        str(spec.transition_blend),
        "--local-prob-power",
        str(spec.local_prob_power),
        "--temperature",
        "1.0",
        "--stack-rank-scale",
        str(spec.rank_scale),
        "--stack-rank-max",
        str(spec.rank_max),
        "--stack-rank-tail-pivot",
        str(spec.rank_tail_pivot),
        "--stack-rank-tail-scale",
        str(spec.rank_tail_scale),
        "--stack-rank-position-scales",
        spec.rank_position_scales,
        "--stack-adj-dup-prob",
        str(spec.adj_dup_prob),
        "--stack-adj-dup-position-probs",
        spec.adj_dup_position_probs,
        "--stack-adj-dup-min-rank",
        str(spec.adj_dup_min_rank),
        "--stack-adj-dup-max-rank",
        str(spec.adj_dup_max_rank),
        "--stack-adj-dup-band-prob",
        str(spec.adj_dup_band_prob),
        "--stack-rank-band-reuse-prob",
        str(spec.rank_band_reuse_prob),
        "--stack-rank-band-reuse-position-probs",
        spec.rank_band_reuse_position_probs,
        "--stack-rank-band-reuse-min-rank",
        str(spec.rank_band_reuse_min_rank),
        "--stack-rank-band-reuse-max-rank",
        str(spec.rank_band_reuse_max_rank),
        "--stack-rank-band-reuse-power",
        str(spec.rank_band_reuse_power),
        "--stack-reuse-boost-prob",
        str(spec.reuse_boost_prob),
        "--stack-reuse-boost-min-rank",
        str(spec.reuse_boost_min_rank),
        "--stack-reuse-boost-rank-power",
        str(spec.reuse_boost_rank_power),
        "--stack-reuse-drop-prob",
        str(spec.reuse_drop_prob),
        "--stack-hot-pool-prob",
        str(spec.hot_pool_prob),
        "--stack-hot-pool-position-probs",
        spec.hot_pool_position_probs,
        "--stack-hot-pool-k",
        str(spec.hot_pool_k),
        "--stack-hot-pool-window",
        str(spec.hot_pool_window),
        "--stack-hot-pool-weight-power",
        str(spec.hot_pool_weight_power),
        "--stack-hot-pool-min-age",
        str(spec.hot_pool_min_age),
        "--stack-frequency-pool-prob",
        str(spec.frequency_pool_prob),
        "--stack-frequency-pool-k",
        str(spec.frequency_pool_k),
        "--stack-frequency-pool-max-candidates",
        str(spec.frequency_pool_max_candidates),
        "--stack-frequency-pool-refresh-interval",
        str(spec.frequency_pool_refresh_interval),
        "--stack-frequency-pool-min-count-rank",
        str(spec.frequency_pool_min_count_rank),
        "--stack-frequency-pool-max-count-rank",
        str(spec.frequency_pool_max_count_rank),
        "--stack-frequency-pool-weight-power",
        str(spec.frequency_pool_weight_power),
        "--stack-frequency-pool-min-age",
        str(spec.frequency_pool_min_age),
        "--stack-frequency-pool-min-rank",
        str(spec.frequency_pool_min_rank),
        "--stack-frequency-pool-max-rank",
        str(spec.frequency_pool_max_rank),
        "--stack-frequency-pool-sample-attempts",
        str(spec.frequency_pool_sample_attempts),
        "--stack-recent-pool-prob",
        str(spec.recent_pool_prob),
        "--stack-recent-pool-position-probs",
        spec.recent_pool_position_probs,
        "--stack-recent-pool-window",
        str(spec.recent_pool_window),
        "--stack-tail-reuse-prob",
        str(spec.tail_reuse_prob),
        "--stack-tail-reuse-position-probs",
        spec.tail_reuse_position_probs,
        "--stack-tail-reuse-min-frac",
        str(spec.tail_reuse_min_frac),
        "--stack-tail-reuse-rank-power",
        str(spec.tail_reuse_rank_power),
        "--n-records",
        str(args.n_records),
        "--n-streams",
        str(args.n_streams),
        "--seed",
        str(spec.seed),
        "--output",
        str(eval_json),
        "--fake-output",
        str(fake),
        "--progress-interval",
        str(args.progress_interval),
    ]
    if spec.reuse_drop_position_probs:
        cmd.extend([
            "--stack-reuse-drop-position-probs",
            spec.reuse_drop_position_probs,
        ])
    if args.force_phase:
        cmd.insert(cmd.index("--stack-adj-dup-prob"), "--force-phase-schedule")
    return cmd


def _cachesim_cmd(args: argparse.Namespace, fake: Path, out_json: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "llgan.cachesim_eval",
        "--fake",
        str(fake),
        "--real",
        args.official_ref,
        "--cache-sizes",
        args.cache_sizes,
        "--policies",
        args.policies,
        "--out",
        str(out_json),
    ]


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    cache_root = root / "cachesim_lanl"
    cache_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"

    for spec in args.spec:
        base = f"{args.corpus}_lanl_{spec.name}"
        fake = root / f"{base}_fake_1M.csv"
        eval_json = root / f"{base}_eval_1M.json"
        cachesim_json = cache_root / f"{base}_{args.eval_tag}.json"
        if args.skip_existing and cachesim_json.exists():
            print(f"[altgan.launch_stackatlas_cachesim_bracket] skip existing {cachesim_json}")
            continue
        print(f"[altgan.launch_stackatlas_cachesim_bracket] running {base}", flush=True)
        _run(_eval_cmd(args, spec, fake, eval_json), env)
        _run(_cachesim_cmd(args, fake, cachesim_json), env)
        print(f"[altgan.launch_stackatlas_cachesim_bracket] wrote {cachesim_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
