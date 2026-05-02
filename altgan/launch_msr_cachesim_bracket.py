"""Launch MSR Exchange NeuralAtlas official cache-sim brackets.

The race surface is the literal ``llgan.cachesim_eval`` panel against the
official MSR Exchange reference CSV. This wrapper first generates a fake trace
with ``altgan.evaluate_neural_atlas`` and then runs that official comparison.
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
    adj_dup_prob: float = 0.40
    hot_pool_prob: float = 0.45
    hot_pool_k: int = 75
    hot_pool_window: int = 10000
    hot_pool_weight_power: float = 1.0
    hot_pool_min_age: int = 0
    recent_pool_prob: float = 0.15
    recent_pool_window: int = 16
    tail_reuse_prob: float = 0.10
    tail_reuse_min_frac: float = 0.5
    reuse_boost_prob: float = 0.0
    reuse_boost_min_rank: int = 32768
    reuse_boost_rank_power: float = 2.0


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
        "adj": "adj_dup_prob",
        "hp": "hot_pool_prob",
        "k": "hot_pool_k",
        "hpwin": "hot_pool_window",
        "hpwp": "hot_pool_weight_power",
        "minage": "hot_pool_min_age",
        "rp": "recent_pool_prob",
        "win": "recent_pool_window",
        "tail": "tail_reuse_prob",
        "mf": "tail_reuse_min_frac",
        "reuse": "reuse_boost_prob",
        "reuse_min": "reuse_boost_min_rank",
        "reuse_power": "reuse_boost_rank_power",
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
        f"_hp{_tag(spec.hot_pool_prob)}k{spec.hot_pool_k}"
        f"w{spec.hot_pool_window}wp{_tag(spec.hot_pool_weight_power)}"
        f"_minage{spec.hot_pool_min_age}_rp{_tag(spec.recent_pool_prob)}"
        f"w{spec.recent_pool_window}_tail{_tag(spec.tail_reuse_prob)}"
        f"mf{_tag(spec.tail_reuse_min_frac)}"
    )


def _tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", action="append", type=_parse_spec, required=True)
    p.add_argument(
        "--model",
        default="/tiamat/zarathustra/checkpoints/altgan/msr_exchange_phaseatlas_92x50k_h128_phase8_e900_seed17.pkl.gz",
    )
    p.add_argument("--trace-dir", default="/tiamat/zarathustra/traces/msr_exchange")
    p.add_argument(
        "--char-file",
        default="/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl",
    )
    p.add_argument(
        "--real-manifest",
        default="/tiamat/zarathustra/llgan-output/manifests/msr_exchange_stackatlas.json",
    )
    p.add_argument(
        "--official-ref",
        default="/tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv",
    )
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--progress-interval", type=int, default=50_000)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument(
        "--no-force-phase-schedule",
        dest="force_phase",
        action="store_false",
    )
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
        "oracle_general",
        "--char-file",
        args.char_file,
        "--cond-dim",
        "13",
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
        "--stack-adj-dup-prob",
        str(spec.adj_dup_prob),
        "--stack-reuse-boost-prob",
        str(spec.reuse_boost_prob),
        "--stack-reuse-boost-min-rank",
        str(spec.reuse_boost_min_rank),
        "--stack-reuse-boost-rank-power",
        str(spec.reuse_boost_rank_power),
        "--stack-hot-pool-prob",
        str(spec.hot_pool_prob),
        "--stack-hot-pool-k",
        str(spec.hot_pool_k),
        "--stack-hot-pool-window",
        str(spec.hot_pool_window),
        "--stack-hot-pool-weight-power",
        str(spec.hot_pool_weight_power),
        "--stack-hot-pool-min-age",
        str(spec.hot_pool_min_age),
        "--stack-recent-pool-prob",
        str(spec.recent_pool_prob),
        "--stack-recent-pool-window",
        str(spec.recent_pool_window),
        "--stack-tail-reuse-prob",
        str(spec.tail_reuse_prob),
        "--stack-tail-reuse-min-frac",
        str(spec.tail_reuse_min_frac),
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
        base = f"msr_exchange_lanl_{spec.name}"
        fake = root / f"{base}_fake_1M.csv"
        eval_json = root / f"{base}_eval_1M.json"
        cachesim_json = cache_root / f"{base}_official6.json"
        if args.skip_existing and cachesim_json.exists():
            print(f"[altgan.launch_msr_cachesim_bracket] skip existing {cachesim_json}")
            continue
        print(f"[altgan.launch_msr_cachesim_bracket] running {base}", flush=True)
        _run(_eval_cmd(args, spec, fake, eval_json), env)
        _run(_cachesim_cmd(args, fake, cachesim_json), env)
        print(f"[altgan.launch_msr_cachesim_bracket] wrote {cachesim_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
