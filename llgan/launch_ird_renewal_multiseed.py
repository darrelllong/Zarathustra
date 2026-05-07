"""LLNL IRD-renewal multi-seed sweep launcher — Wikipedia / CloudPhysics / Meta KV retakes.

This launcher runs the R288 position-based IRD-renewal implementation
(`llgan.ird_renewal`) over the axes LANL has NOT published, targeting three
corpora where LLNL is significantly behind:

  Wikipedia:    LLNL 0.01707 vs LANL 0.01146 (−32.9%)  ← IRD-renewal expected ~0.0114
  CloudPhysics: LLNL 0.03017 vs LANL 0.0267  (−13.0%)  ← rank_ird_smooth to cut variance
  Meta KV:      LLNL 0.04807 vs LANL 0.0109  (−77.3%)  ← IRD-renewal expected ~0.011 class

Sweep axes:
  --rank-ird-buckets {0, 8, 16, 32}   — rank-conditioned per-bucket IRD distributions
  --rank-ird-smooth                   — blend sparse tail buckets with neighbors
  --ird-scale {0.5, 1.0, 2.0, 4.0}   — compress/expand IRD tails
  --per-stream                        — per-stream fitting for heterogeneous corpora
  --ird-quantile-max {0.99, 1.0}      — cap extreme tail IRDs

Usage examples (from repo root):

  # Wikipedia retake sweep
  python3 -m llgan.launch_ird_renewal_multiseed \\
      --real /tiamat/zarathustra/llgan-output/refs/wiki_real.csv \\
      --output-root /tiamat/zarathustra/llgan-output/ird_renewal \\
      --corpus wiki \\
      --cache-sizes 32,128,512,2048,8192 \\
      --policies lru,arc,fifo,sieve,slru,car \\
      --seeds 42,80,81,82 \\
      --spec "base:ird_s=1.0,ip=0.10" \\
      --spec "s32:ird_s=32.0,ip=0.10" \\
      --spec "rb16_s32:ird_s=32.0,ip=0.10,rb=16" \\
      --spec "rb32_s32:ird_s=32.0,ip=0.10,rb=32" \\
      --spec "rb32_s32_sm:ird_s=32.0,ip=0.10,rb=32,smooth=1" \\
      --spec "rb16_ps:ird_s=32.0,ip=0.10,rb=16,ps=1" \\
      --spec "rb32_ps:ird_s=32.0,ip=0.10,rb=32,ps=1" \\
      --spec "qmax99:ird_s=32.0,ip=0.10,qmax=0.99" \\
      --emit-markdown

  # CloudPhysics variance-reduction sweep
  python3 -m llgan.launch_ird_renewal_multiseed \\
      --real /tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv \\
      --output-root /tiamat/zarathustra/llgan-output/ird_renewal \\
      --corpus cloudphysics \\
      --cache-sizes 32,128,512,2048,8192,32768 \\
      --policies lru,arc,fifo,sieve,slru,car,lfu,lirs \\
      --seeds 42,80,81,82 \\
      --spec "lanl_ref:ird_s=16.0,ip=0.00,rb=32" \\
      --spec "rb32_sm:ird_s=16.0,ip=0.00,rb=32,smooth=1" \\
      --spec "rb48:ird_s=16.0,ip=0.00,rb=48" \\
      --spec "rb48_sm:ird_s=16.0,ip=0.00,rb=48,smooth=1" \\
      --spec "rb64:ird_s=16.0,ip=0.00,rb=64" \\
      --spec "rb64_sm:ird_s=16.0,ip=0.00,rb=64,smooth=1" \\
      --spec "rb64_ps:ird_s=16.0,ip=0.00,rb=64,ps=1" \\
      --spec "rb64_qmax99:ird_s=16.0,ip=0.00,rb=64,qmax=0.99" \\
      --emit-markdown

  # Meta KV sweep (LANL used atlas; LLNL probes IRD-renewal as alternative path)
  python3 -m llgan.launch_ird_renewal_multiseed \\
      --real /tiamat/zarathustra/llgan-output/refs/meta_kv_real.csv \\
      --output-root /tiamat/zarathustra/llgan-output/ird_renewal \\
      --corpus meta_kv \\
      --cache-sizes 32,128,512,2048,8192 \\
      --policies lru,arc,fifo,sieve,slru,car \\
      --seeds 42,80,81,82 \\
      --spec "base:ird_s=1.0,ip=0.10" \\
      --spec "s16:ird_s=16.0,ip=0.10" \\
      --spec "s32:ird_s=32.0,ip=0.10" \\
      --spec "rb32_s32:ird_s=32.0,ip=0.10,rb=32" \\
      --spec "rb32_s32_sm:ird_s=32.0,ip=0.10,rb=32,smooth=1" \\
      --spec "rb32_s16:ird_s=16.0,ip=0.10,rb=32" \\
      --spec "rb32_s16_ps:ird_s=16.0,ip=0.10,rb=32,ps=1" \\
      --emit-markdown
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
    ird_scale: float = 1.0
    independent_prob: float = 0.10
    rank_ird_buckets: int = 0
    rank_ird_min_samples: int = 256
    rank_ird_smooth: bool = False
    ird_quantile_max: float = 1.0
    per_stream: bool = False
    new_debt_priority: float = 0.85
    dependent_admit_prob: float = 1.0


@dataclass(frozen=True)
class SeedResult:
    spec: str
    seed: int
    fake: Path
    cachesim_json: Path
    cachesim_line: str | None
    mean: float | None


def _parse_spec(text: str) -> Spec:
    if ":" in text:
        name, rest = text.split(":", 1)
    else:
        name, rest = "", text
    aliases = {
        "ird_s": "ird_scale",
        "ip": "independent_prob",
        "rb": "rank_ird_buckets",
        "min_s": "rank_ird_min_samples",
        "smooth": "rank_ird_smooth",
        "qmax": "ird_quantile_max",
        "ps": "per_stream",
        "debt": "new_debt_priority",
        "admit": "dependent_admit_prob",
    }
    fnames = {f: True for f in Spec.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs: dict = {"name": name}
    for part in rest.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"spec {part!r} must be key=value")
        key, value = part.split("=", 1)
        fname = aliases.get(key.strip(), key.strip())
        if fname not in fnames:
            raise argparse.ArgumentTypeError(f"unknown spec key {key!r}")
        default_val = getattr(Spec(name="_"), fname)
        t = type(default_val)
        if t is bool:
            kwargs[fname] = value.strip() not in ("0", "false", "False", "no")
        elif t is str:
            kwargs[fname] = value.strip()
        else:
            kwargs[fname] = t(value.strip())
    if not kwargs.get("name"):
        kwargs["name"] = _auto_name(Spec(**kwargs))  # type: ignore[arg-type]
    return Spec(**kwargs)  # type: ignore[arg-type]


def _auto_name(spec: Spec) -> str:
    parts = [f"s{spec.ird_scale:g}", f"ip{spec.independent_prob:g}"]
    if spec.rank_ird_buckets > 0:
        parts.append(f"rb{spec.rank_ird_buckets}")
    if spec.rank_ird_smooth:
        parts.append("sm")
    if spec.per_stream:
        parts.append("ps")
    if spec.ird_quantile_max < 1.0:
        parts.append(f"qmax{spec.ird_quantile_max:g}")
    return "_".join(parts)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--real", required=True, help="Official real CSV reference path.")
    p.add_argument("--output-root", default="/tiamat/zarathustra/llgan-output/ird_renewal")
    p.add_argument("--corpus", required=True, help="Corpus tag for output filenames.")
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--seeds", default="42,80,81,82")
    p.add_argument("--n-records", type=int, default=1_000_000)
    p.add_argument("--max-real-rows", type=int, default=0)
    p.add_argument("--spec", action="append", type=_parse_spec, required=True,
                   help="Spec string: [name:]key=value,... (repeatable)")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--emit-markdown", action="store_true")
    p.add_argument("--append-markdown", help="Append markdown summary to this file path.")
    p.add_argument("--progress-interval", type=int, default=200_000)
    return p.parse_args()


def _renewal_cmd(spec: Spec, args: argparse.Namespace, seed: int, fake: Path) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "llgan.ird_renewal",
        "--real", args.real,
        "--output", str(fake),
        "--n", str(args.n_records),
        "--seed", str(seed),
        "--ird-scale", str(spec.ird_scale),
        "--independent-prob", str(spec.independent_prob),
        "--rank-ird-min-samples", str(spec.rank_ird_min_samples),
        "--new-debt-priority", str(spec.new_debt_priority),
        "--dependent-admit-prob", str(spec.dependent_admit_prob),
    ]
    if args.max_real_rows > 0:
        cmd += ["--max-real-rows", str(args.max_real_rows)]
    if spec.rank_ird_buckets > 0:
        cmd += ["--rank-ird-buckets", str(spec.rank_ird_buckets)]
    if spec.rank_ird_smooth:
        cmd.append("--rank-ird-smooth")
    if spec.per_stream:
        cmd.append("--per-stream")
    if 0.0 < spec.ird_quantile_max < 1.0:
        cmd += ["--ird-quantile-max", str(spec.ird_quantile_max)]
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


def _run(cmd: list[str], env: dict, dry_run: bool) -> None:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _run_capture(cmd: list[str], env: dict, dry_run: bool) -> str:
    print("+ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry_run:
        return ""
    result = subprocess.run(cmd, check=True, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.stdout:
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    return result.stdout or ""


def _extract_mean_line(output: str) -> str | None:
    for line in output.splitlines():
        if "mean HRC-MAE across policies:" in line:
            return line.strip()
    return None


def _read_mean(json_path: Path) -> float | None:
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
        return float(data.get("mean_hrc_mae", data.get("mean", float("nan"))))
    except Exception:
        return None


def _format_markdown(*, corpus: str, real: str, cache_sizes: str, policies: str,
                     results: list[SeedResult]) -> str:
    lines = [
        f"## {corpus} IRD-renewal R288 sweep",
        "",
        f"Official reference: `{real}`.",
        "",
        "| spec | seed | mean |",
        "|---|---:|---:|",
    ]
    for r in results:
        mean_cell = f"`{r.mean:.7f}`" if r.mean is not None else ""
        lines.append(f"| `{r.spec}` | {r.seed} | {mean_cell} |")

    by_spec: dict[str, list[float]] = {}
    for r in results:
        if r.mean is None:
            continue
        by_spec.setdefault(r.spec, []).append(r.mean)

    if by_spec:
        lines += ["", "| spec | 4-seed mean | range |", "|---|---:|---:|"]
        for spec_name, means in sorted(by_spec.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            m = sum(means) / len(means)
            rng = max(means) - min(means)
            lines.append(f"| `{spec_name}` | `{m:.7f}` | `{rng:.7f}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"

    summary: list[dict] = []
    all_results: list[SeedResult] = []

    for spec in args.spec:
        seed_means: list[float] = []
        for seed in seeds:
            tag = f"{args.corpus}_llnl_irdr_{spec.name}_seed{seed}"
            fake = root / f"{tag}_fake_{args.n_records // 1000}k.csv"
            cs_json = root / "cachesim" / f"{tag}_official.json"
            cs_json.parent.mkdir(parents=True, exist_ok=True)
            cachesim_line = None

            if args.skip_existing and cs_json.exists():
                print(f"[irdr] skip existing {cs_json.name}", flush=True)
            else:
                print(f"[irdr] generating {fake.name}", flush=True)
                _run(_renewal_cmd(spec, args, seed, fake), env, args.dry_run)
                print(f"[irdr] eval {cs_json.name}", flush=True)
                output = _run_capture(_cachesim_cmd(args, fake, cs_json), env, args.dry_run)
                cachesim_line = _extract_mean_line(output)

            mean = _read_mean(cs_json)
            if mean is not None:
                seed_means.append(mean)
            print(
                f"[irdr] {tag}: {mean:.7f}" if mean is not None else f"[irdr] {tag}: pending",
                flush=True,
            )
            all_results.append(SeedResult(spec=spec.name, seed=seed, fake=fake,
                                          cachesim_json=cs_json, cachesim_line=cachesim_line,
                                          mean=mean))

        if seed_means:
            m = sum(seed_means) / len(seed_means)
            r = max(seed_means) - min(seed_means)
            per = " / ".join(f"{x:.7f}" for x in seed_means)
            print(f"[irdr] SUMMARY {spec.name}: mean={m:.7f} range={r:.7f} per_seed=[{per}]",
                  flush=True)
            summary.append({"spec": spec.name, "mean": m, "range": r, "seeds": seed_means})

    if summary:
        print("\n=== SWEEP RESULTS ===", flush=True)
        print(f"{'spec':<35} {'mean':>10} {'range':>10}", flush=True)
        for row in sorted(summary, key=lambda x: x["mean"]):
            print(f"{row['spec']:<35} {row['mean']:>10.7f} {row['range']:>10.7f}", flush=True)

    if args.emit_markdown or args.append_markdown:
        md = _format_markdown(corpus=args.corpus, real=args.real,
                              cache_sizes=args.cache_sizes, policies=args.policies,
                              results=all_results)
        if args.emit_markdown:
            print("\n" + md, flush=True)
        if args.append_markdown:
            p = Path(args.append_markdown)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.read_text() if p.exists() else ""
            p.write_text(existing + ("\n\n" if existing else "") + md)
            print(f"[irdr] appended markdown → {p}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
