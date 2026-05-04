"""IRD-renewal sweep launcher for Wikipedia / CloudPhysics retake.

Sweeps the axes LANL has NOT published results for:
  - rank_ird_buckets > 0 (finer per-rank IRD conditioning)
  - --per-stream (fit one renewal profile per stream)
  - ird_quantile_max < 1.0 (cap extreme tail IRDs to reduce seed variance)
  - heap=priority (2DIO-style priority-sleep dependent arrivals)

Compared to LANL's published baselines:
  Wikipedia: ird_s=32 ip=0.10 global → mean 0.01146 range 0.000533
  CloudPhysics: rank_b=32 ird_s=16 ip=0.00 → mean 0.0267 range 0.0045

Usage examples (run from repo root):

  # Wikipedia sweep — rank_ird_buckets + per-stream
  python3 -m altgan.launch_ird_renewal_sweep \\
      --real /tiamat/zarathustra/llgan-output/refs/wiki_real.csv \\
      --output-root /tiamat/zarathustra/altgan-output \\
      --corpus wiki \\
      --cache-sizes 32,128,512,2048,8192 \\
      --policies lru,arc,fifo,sieve,slru,car \\
      --seeds 42,80,81,82 \\
      --spec "base:ird_s=32,ip=0.10" \\
      --spec "rb16:ird_s=32,ip=0.10,rb=16" \\
      --spec "rb16_sm:ird_s=32,ip=0.10,rb=16,smooth=1" \\
      --spec "rb32:ird_s=32,ip=0.10,rb=32" \\
      --spec "rb32_sm:ird_s=32,ip=0.10,rb=32,smooth=1" \\
      --spec "rb16_s28:ird_s=28,ip=0.10,rb=16" \\
      --spec "rb32_s28:ird_s=28,ip=0.10,rb=32" \\
      --spec "rb16_ps:ird_s=32,ip=0.10,rb=16,per_stream=1" \\
      --spec "rb32_ps:ird_s=32,ip=0.10,rb=32,per_stream=1" \\
      --spec "qmax99:ird_s=32,ip=0.10,qmax=0.99" \\
      --spec "prio:ird_s=32,ip=0.10,heap=priority"

  # CloudPhysics sweep — finer rank_ird_buckets + per-stream + smoothing + quantile cap
  # LANL's rank_b=32 had range=0.0045 (seed-80 outlier at 0.0295 vs seed-42 0.0250).
  # --rank-ird-smooth blends sparse tail buckets with neighbors to reduce this variance.
  python3 -m altgan.launch_ird_renewal_sweep \\
      --real /tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv \\
      --output-root /tiamat/zarathustra/altgan-output \\
      --corpus cloudphysics \\
      --cache-sizes 32,128,512,2048,8192,32768 \\
      --policies lru,arc,fifo,sieve,slru,car,lfu,lirs \\
      --seeds 42,80,81,82 \\
      --spec "lanl_ref:ird_s=16,ip=0.00,rb=32" \\
      --spec "rb32_sm:ird_s=16,ip=0.00,rb=32,smooth=1" \\
      --spec "rb48:ird_s=16,ip=0.00,rb=48" \\
      --spec "rb48_sm:ird_s=16,ip=0.00,rb=48,smooth=1" \\
      --spec "rb64:ird_s=16,ip=0.00,rb=64" \\
      --spec "rb64_sm:ird_s=16,ip=0.00,rb=64,smooth=1" \\
      --spec "rb96:ird_s=16,ip=0.00,rb=96" \\
      --spec "rb64_q99:ird_s=16,ip=0.00,rb=64,qmax=0.99" \\
      --spec "rb64_ps:ird_s=16,ip=0.00,rb=64,per_stream=1" \\
      --spec "rb32_ps:ird_s=16,ip=0.00,rb=32,per_stream=1"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Spec:
    name: str
    ird_scale: float = 32.0
    independent_prob: float = 0.10
    rank_ird_buckets: int = 0
    rank_ird_min_samples: int = 256
    per_stream: bool = False
    rank_ird_smooth: bool = False
    ird_quantile_max: float = 1.0
    ird_jitter: float = 0.0
    frequency_alpha: float = 1.0
    new_debt_priority: float = 0.85
    dependent_admit_prob: float = 1.0
    heap_mode: str = "due"


def _parse_spec(text: str) -> Spec:
    """Parse name:key=value,key=value spec strings."""
    if ":" in text:
        name, rest = text.split(":", 1)
    else:
        name, rest = "", text
    aliases = {
        "ird_s": "ird_scale",
        "ip": "independent_prob",
        "rb": "rank_ird_buckets",
        "min_s": "rank_ird_min_samples",
        "per_stream": "per_stream",
        "smooth": "rank_ird_smooth",
        "qmax": "ird_quantile_max",
        "jitter": "ird_jitter",
        "alpha": "frequency_alpha",
        "debt": "new_debt_priority",
        "admit": "dependent_admit_prob",
        "heap": "heap_mode",
    }
    fields = {f.name: f for f in Spec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    kwargs: dict[str, object] = {"name": name}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"spec component {part!r} must be key=value")
        key, value = part.split("=", 1)
        fname = aliases.get(key.strip(), key.strip())
        if fname not in fields:
            raise argparse.ArgumentTypeError(f"unknown spec key {key!r}")
        current_type = type(getattr(Spec(name="_"), fname))
        if current_type is bool:
            kwargs[fname] = value.strip() not in ("0", "false", "False", "no")
        elif current_type is str:
            kwargs[fname] = value.strip()
        else:
            kwargs[fname] = current_type(value.strip())
    if not kwargs.get("name"):
        kwargs["name"] = _auto_name(Spec(**kwargs))  # type: ignore[arg-type]
    return Spec(**kwargs)  # type: ignore[arg-type]


def _auto_name(spec: Spec) -> str:
    parts = [f"s{spec.ird_scale:g}", f"ip{spec.independent_prob:g}"]
    if spec.rank_ird_buckets > 0:
        parts.append(f"rb{spec.rank_ird_buckets}")
    if spec.per_stream:
        parts.append("ps")
    if spec.rank_ird_smooth:
        parts.append("sm")
    if spec.ird_quantile_max < 1.0:
        parts.append(f"qmax{spec.ird_quantile_max:g}")
    if spec.ird_jitter > 0:
        parts.append(f"jit{spec.ird_jitter:g}")
    if spec.frequency_alpha != 1.0:
        parts.append(f"fa{spec.frequency_alpha:g}")
    if spec.heap_mode != "due":
        parts.append(spec.heap_mode)
    return "_".join(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True, help="Official real CSV reference path.")
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--corpus", required=True, help="Corpus tag used in output filenames.")
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--seeds", default="42,80,81,82", help="Comma-separated generation seeds.")
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--spec", action="append", type=_parse_spec, required=True,
                   help="Spec string: [name:]key=value,... (repeatable)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print commands only, do not run.")
    p.add_argument("--progress-interval", type=int, default=200_000)
    return p.parse_args()


def _renewal_cmd(
    spec: Spec,
    args: argparse.Namespace,
    seed: int,
    fake: Path,
) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "altgan.ird_renewal",
        "--real", args.real,
        "--output", str(fake),
        "--n-records", str(args.n_records),
        "--seed", str(seed),
        "--ird-scale", str(spec.ird_scale),
        "--independent-prob", str(spec.independent_prob),
        "--rank-ird-min-samples", str(spec.rank_ird_min_samples),
        "--frequency-alpha", str(spec.frequency_alpha),
        "--new-debt-priority", str(spec.new_debt_priority),
        "--dependent-admit-prob", str(spec.dependent_admit_prob),
        "--heap-mode", spec.heap_mode,
        "--progress-interval", str(args.progress_interval),
    ]
    if spec.rank_ird_buckets > 0:
        cmd += ["--rank-ird-buckets", str(spec.rank_ird_buckets)]
    if spec.per_stream:
        cmd.append("--per-stream")
    if spec.rank_ird_smooth:
        cmd.append("--rank-ird-smooth")
    if 0.0 < spec.ird_quantile_max < 1.0:
        cmd += ["--ird-quantile-max", str(spec.ird_quantile_max)]
    if spec.ird_jitter > 0:
        cmd += ["--ird-jitter", str(spec.ird_jitter)]
    return cmd


def _cachesim_cmd(args: argparse.Namespace, fake: Path, out_json: Path) -> list[str]:
    return [
        sys.executable, "-u", "-m", "llgan.cachesim_eval",
        "--fake", str(fake),
        "--real", args.real,
        "--cache-sizes", args.cache_sizes,
        "--policies", args.policies,
        "--out", str(out_json),
    ]


def _run(cmd: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(p) for p in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _read_mean(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
        return float(data.get("mean_hrc_mae", data.get("mean", float("nan"))))
    except Exception:
        return None


def main() -> int:
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"

    summary: list[dict] = []

    for spec in args.spec:
        seed_means: list[float] = []
        for seed in seeds:
            tag = f"{args.corpus}_irdr_{spec.name}_seed{seed}"
            fake = root / f"{tag}_fake_{args.n_records // 1_000}k.csv"
            cs_json = root / "cachesim_lanl" / f"{tag}_official.json"
            cs_json.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and cs_json.exists():
                print(f"[irdr_sweep] skip existing {cs_json.name}", flush=True)
            else:
                print(f"[irdr_sweep] generating {fake.name}", flush=True)
                _run(_renewal_cmd(spec, args, seed, fake), env, args.dry_run)
                print(f"[irdr_sweep] eval {cs_json.name}", flush=True)
                _run(_cachesim_cmd(args, fake, cs_json), env, args.dry_run)

            mean = _read_mean(cs_json)
            if mean is not None:
                seed_means.append(mean)
            print(f"[irdr_sweep] {tag}: {mean:.7f}" if mean is not None else f"[irdr_sweep] {tag}: pending", flush=True)

        if seed_means:
            overall_mean = sum(seed_means) / len(seed_means)
            overall_range = max(seed_means) - min(seed_means)
            per = " / ".join(f"{m:.7f}" for m in seed_means)
            print(
                f"[irdr_sweep] SUMMARY {spec.name}: "
                f"mean={overall_mean:.7f} range={overall_range:.7f} per_seed=[{per}]",
                flush=True,
            )
            summary.append({"spec": spec.name, "mean": overall_mean, "range": overall_range, "seeds": seed_means})

    if summary:
        print("\n=== SWEEP RESULTS ===", flush=True)
        print(f"{'spec':<30} {'mean':>10} {'range':>10}", flush=True)
        for row in sorted(summary, key=lambda r: r["mean"]):
            print(f"{row['spec']:<30} {row['mean']:>10.7f} {row['range']:>10.7f}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
