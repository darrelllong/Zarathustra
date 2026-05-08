"""Launch Alibaba NeuralAtlas brackets and score the official cachesim surface.

This wrapper exists to make the current Alibaba race recipe repeatable and
safe:

- Generates a fake trace via ``altgan.evaluate_neural_atlas``.
- Scores the literal race surface via ``python -m llgan.cachesim_eval`` against
  the official Alibaba reference CSV.
- Pins math-library threads to avoid oversubscription.
- Keeps ``--force-phase-schedule`` on by default (disable only for ablations).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


OFFICIAL_CACHE_SIZES = "32,128,512,2048,8192"
OFFICIAL_POLICIES = "lru,arc,fifo,sieve,slru,car"


class Spec(NamedTuple):
    reuse_prob: str
    hot_prob: str
    hot_k: str
    seed: str
    reuse_tag: str
    hot_tag: str


def _parse_spec(text: str) -> Spec:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--spec must be reuse_prob,hot_prob,hot_k,seed,reuse_tag,hot_tag"
        )
    spec = Spec(*parts)
    _validate_prob_tag(spec.reuse_prob, spec.reuse_tag, "reuse")
    _validate_prob_tag(spec.hot_prob, spec.hot_tag, "hot-pool")
    return spec


def _validate_prob_tag(prob_text: str, tag: str, label: str) -> None:
    prob = float(prob_text)
    tag_prob = _prob_from_tag(tag)
    if abs(prob - tag_prob) > 1e-9:
        raise argparse.ArgumentTypeError(
            f"{label} probability {prob_text!r} does not match tag {tag!r} "
            f"(tag decodes to {tag_prob:g}); use {tag_prob:g} or choose "
            "an unambiguous tag"
        )


def _prob_from_tag(tag: str) -> float:
    text = tag.strip().lower().replace("p", "")
    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"probability tag must be digits, got {tag!r}")
    # Historical LANL tags use 3 digits for percentages: 006 -> 0.06,
    # 044 -> 0.44. Four-digit tags cover sub-percent steps: 0075 -> 0.075.
    if len(text) <= 3:
        scale = 100.0
    else:
        scale = 1000.0
    return int(text) / scale


def _artifact_base(
    spec: Spec,
    *,
    hot_pool_weight_power: float = 1.0,
    hot_pool_min_age: int = 0,
) -> str:
    weight_tag = ""
    if abs(float(hot_pool_weight_power) - 1.0) > 1e-12:
        weight_tag = f"_hpwp{_decimal_tag(hot_pool_weight_power)}"
    age_tag = ""
    if int(hot_pool_min_age) > 0:
        age_tag = f"_hpminage{int(hot_pool_min_age)}"
    return (
        "alibaba_phaseatlas_marks_tb020_lp090_"
        f"reuseboost{spec.reuse_tag}_"
        f"hotpool{spec.hot_tag}k{spec.hot_k}w10000{weight_tag}{age_tag}_"
        f"p{spec.reuse_tag}hp{spec.hot_tag}k{spec.hot_k}_"
        f"seed{spec.seed}_realmanifest42"
    )


def _decimal_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _command(args: argparse.Namespace, spec: Spec, base: str) -> list[str]:
    outroot = Path(args.output_root)
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
        "0.2",
        "--local-prob-power",
        "0.9",
        "--temperature",
        "1.0",
        "--stack-rank-scale",
        "1.0",
        "--stack-rank-max",
        "-1",
        "--stack-reuse-boost-prob",
        spec.reuse_prob,
        "--stack-reuse-boost-min-rank",
        "32768",
        "--stack-reuse-boost-rank-power",
        "2.0",
        "--stack-hot-pool-prob",
        spec.hot_prob,
        "--stack-hot-pool-k",
        spec.hot_k,
        "--stack-hot-pool-window",
        "10000",
        "--stack-hot-pool-weight-power",
        str(args.hot_pool_weight_power),
        "--stack-hot-pool-max-search",
        "0",
        "--stack-hot-pool-min-age",
        str(args.hot_pool_min_age),
        "--n-records",
        str(args.n_records),
        "--n-streams",
        str(args.n_streams),
        "--seed",
        spec.seed,
        "--output",
        str(outroot / f"{base}_eval_1M.json"),
        "--fake-output",
        str(outroot / f"{base}_fake_1M.csv"),
        "--real-output",
        str(outroot / "alibaba_real_manifest_seed42_1M_eval_real.csv"),
        "--progress-interval",
        str(args.progress_interval),
    ]
    if args.internal_cachesim_bin:
        cmd.extend(
            [
                "--cachesim-bin",
                args.internal_cachesim_bin,
                "--cachesim-output",
                str(outroot / "cachesim_lanl" / f"{base}_internal.json"),
                "--cachesim-cache-sizes",
                args.internal_cache_sizes,
                "--cachesim-policies",
                args.internal_policies,
            ]
        )
    if args.force_phase:
        cmd.insert(cmd.index("--stack-rank-scale"), "--force-phase-schedule")
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


def _env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[key] = "1"
    return env


def _quote(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + _quote(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        action="append",
        type=_parse_spec,
        required=True,
        help="reuse_prob,hot_prob,hot_k,seed,reuse_tag,hot_tag",
    )
    parser.add_argument(
        "--model",
        default="/tiamat/zarathustra/checkpoints/altgan/alibaba_phaseatlas_marks_e20.pkl.gz",
    )
    parser.add_argument("--trace-dir", default="/tiamat/zarathustra/traces/alibaba")
    parser.add_argument(
        "--char-file",
        default="/tiamat/zarathustra/analysis/out/trace_characterizations.jsonl",
    )
    parser.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    parser.add_argument(
        "--real-manifest",
        default="/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_manifest.json",
        help=(
            "Real manifest JSON used for conditional generation. This should match the "
            "1M Alibaba stack-atlas manifest used for scoring."
        ),
    )
    parser.add_argument(
        "--official-ref",
        default="/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv",
        help="Official reference CSV for llgan.cachesim_eval.",
    )
    parser.add_argument("--n-records", type=int, default=1_000_000)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--cache-sizes", default=OFFICIAL_CACHE_SIZES)
    parser.add_argument("--policies", default=OFFICIAL_POLICIES)
    parser.add_argument(
        "--internal-cachesim-bin",
        default="",
        help=(
            "Optional tools/cachesim binary passed to altgan.evaluate_neural_atlas for "
            "extra diagnostics. Official scoring always uses llgan.cachesim_eval."
        ),
    )
    parser.add_argument("--internal-cache-sizes", default=OFFICIAL_CACHE_SIZES)
    parser.add_argument("--internal-policies", default=OFFICIAL_POLICIES)
    parser.add_argument(
        "--hot-pool-weight-power",
        type=float,
        default=1.0,
        help="Power applied to hot-pool frequency weights during generation.",
    )
    parser.add_argument(
        "--hot-pool-min-age",
        type=int,
        default=0,
        help="Minimum stream positions before a hot-pool object can be reused.",
    )
    parser.add_argument("--progress-interval", type=int, default=50000)
    parser.add_argument(
        "--no-force-phase-schedule",
        dest="force_phase",
        action="store_false",
        help="Only use for explicit ablations; current race rows require forced phase.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of running them.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--allow-nonofficial-surface",
        action="store_true",
        help=(
            "Allow non-official cache sizes/policies. Without this, the wrapper "
            "refuses to run if --cache-sizes or --policies differ from the official "
            "race surface."
        ),
    )
    args = parser.parse_args(argv)
    if (
        (args.cache_sizes != OFFICIAL_CACHE_SIZES or args.policies != OFFICIAL_POLICIES)
        and not args.allow_nonofficial_surface
    ):
        raise SystemExit(
            "Refusing to run a non-official cachesim surface. Either keep "
            f"--cache-sizes {OFFICIAL_CACHE_SIZES!r} and --policies {OFFICIAL_POLICIES!r} "
            "or pass --allow-nonofficial-surface for ablations."
        )

    outroot = Path(args.output_root)
    cachesim_root = outroot / "cachesim_lanl"
    if not args.dry_run:
        cachesim_root.mkdir(parents=True, exist_ok=True)
    env = _env()
    for spec in args.spec:
        base = _artifact_base(
            spec,
            hot_pool_weight_power=args.hot_pool_weight_power,
            hot_pool_min_age=args.hot_pool_min_age,
        )
        cmd = _command(args, spec, base)
        fake = outroot / f"{base}_fake_1M.csv"
        cachesim_json = cachesim_root / f"{base}_official6.json"
        if args.skip_existing and cachesim_json.exists():
            print(f"[altgan.launch_alibaba_cachesim_bracket] skip existing {cachesim_json}")
            continue
        if args.dry_run:
            print(_quote(cmd))
            print(_quote(_cachesim_cmd(args, fake, cachesim_json)))
            continue
        print(f"[altgan.launch_alibaba_cachesim_bracket] running {base}", flush=True)
        _run(cmd, env)
        _run(_cachesim_cmd(args, fake, cachesim_json), env)
        print(f"[altgan.launch_alibaba_cachesim_bracket] wrote {cachesim_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
