"""Launch forced-phase Alibaba NeuralAtlas cache-sim brackets.

This wrapper exists to make the current race recipe repeatable. It builds the
long ``evaluate_neural_atlas`` command, pins math-library threads, writes the
same fake/real/cachesim artifact names as the manual runs, and keeps
``--force-phase-schedule`` on by default.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, NamedTuple


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
    return Spec(*parts)


def _artifact_base(spec: Spec) -> str:
    return (
        "alibaba_phaseatlas_marks_tb020_lp090_"
        f"reuseboost{spec.reuse_tag}_"
        f"hotpool{spec.hot_tag}k{spec.hot_k}w10000_"
        f"p{spec.reuse_tag}hp{spec.hot_tag}k{spec.hot_k}_"
        f"seed{spec.seed}_realmanifest42"
    )


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
        str(outroot / "alibaba_real_manifest_seed42_1M_manifest.json"),
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
        "1.0",
        "--stack-hot-pool-max-search",
        "0",
        "--n-records",
        "1000000",
        "--n-streams",
        "4",
        "--seed",
        spec.seed,
        "--output",
        str(outroot / f"{base}_eval_1M.json"),
        "--fake-output",
        str(outroot / f"{base}_fake_1M.csv"),
        "--real-output",
        str(outroot / "alibaba_real_manifest_seed42_1M_eval_real.csv"),
        "--cachesim-bin",
        args.cachesim_bin,
        "--cachesim-output",
        str(outroot / "cachesim_lanl" / f"{base}_eight_policy_caps.json"),
        "--cachesim-policies",
        args.cachesim_policies,
        "--progress-interval",
        str(args.progress_interval),
    ]
    if args.force_phase:
        cmd.insert(cmd.index("--stack-rank-scale"), "--force-phase-schedule")
    return cmd


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
        "--cachesim-bin", default="tools/cachesim/target/release/cachesim"
    )
    parser.add_argument(
        "--cachesim-policies", default="lru,arc,fifo,sieve,slru,car,lfu,lirs"
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
        help="Print commands instead of launching them.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run:
        Path(args.output_root, "cachesim_lanl").mkdir(parents=True, exist_ok=True)
    env = _env()
    for spec in args.spec:
        base = _artifact_base(spec)
        cmd = _command(args, spec, base)
        log_path = Path(args.output_root) / f"{base}.log"
        if args.dry_run:
            print(f"{_quote(cmd)} > {shlex.quote(str(log_path))} 2>&1")
            continue
        with log_path.open("ab") as log_file:
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        print(f"{proc.pid} {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
