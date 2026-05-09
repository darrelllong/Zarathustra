"""Launch Baleen24 micro-head repair scouts.

The direct hot-head repair was too blunt: large scattered rewrites damaged the
official cache surface.  This launcher keeps the experiment small and
reproducible by sweeping tiny head fractions from a fixed base trace, then
evaluating both the official Baleen24 race surface and the no-cache-32 guard.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _tag_float(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _mean(path: Path) -> float:
    with path.open() as f:
        return float(json.load(f)["mean_hrc_mae"])


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag-prefix", default="baleen24_r400b_microhead")
    p.add_argument("--target-adjacent-frac", type=float, default=0.535)
    p.add_argument("--hothead-fracs", type=_parse_floats, default=[0.005, 0.01, 0.02])
    p.add_argument("--hothead-ids", type=_parse_ints, default=[8, 32, 128])
    p.add_argument("--hothead-alpha", type=float, default=1.20)
    p.add_argument("--hothead-min-gap", type=int, default=32)
    p.add_argument("--hothead-eligible", default="nonadjacent_nonsingleton")
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--guard-cache-sizes", default="128,512,2048,8192")
    p.add_argument("--guard-label", default="no32guard")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.output_root)
    scout_root = root / "headgraft_scouts"
    eval_root = root / "cachesim_lanl"
    scout_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.setdefault(key, "1")

    rows: list[tuple[str, float, float, Path, Path]] = []
    for frac in args.hothead_fracs:
        for ids in args.hothead_ids:
            tag = (
                f"{args.tag_prefix}_t{_tag_float(args.target_adjacent_frac)}"
                f"_f{_tag_float(frac)}_ids{ids}_seed{args.seed}"
            )
            fake = scout_root / f"{tag}_fake_1000k.csv"
            official_json = eval_root / f"{tag}_official6.json"
            guard_json = eval_root / f"{tag}_{args.guard_label}.json"
            print(f"\n=== {tag} ===", flush=True)
            _run(
                [
                    sys.executable,
                    "-m",
                    "altgan.baleen_hothead_repair",
                    "--base",
                    args.base,
                    "--output",
                    str(fake),
                    "--seed",
                    str(args.seed),
                    "--target-adjacent-frac",
                    str(args.target_adjacent_frac),
                    "--break-mode",
                    "unique",
                    "--hothead-frac",
                    str(frac),
                    "--hothead-ids",
                    str(ids),
                    "--hothead-alpha",
                    str(args.hothead_alpha),
                    "--hothead-min-gap",
                    str(args.hothead_min_gap),
                    "--hothead-eligible",
                    args.hothead_eligible,
                    "--stream-mode",
                    "preserve",
                ],
                env=env,
            )
            _run(
                [
                    sys.executable,
                    "-m",
                    "llgan.cachesim_eval",
                    "--fake",
                    str(fake),
                    "--real",
                    args.real,
                    "--cache-sizes",
                    args.cache_sizes,
                    "--policies",
                    args.policies,
                    "--out",
                    str(official_json),
                ],
                env=env,
            )
            _run(
                [
                    sys.executable,
                    "-m",
                    "llgan.cachesim_eval",
                    "--fake",
                    str(fake),
                    "--real",
                    args.real,
                    "--cache-sizes",
                    args.guard_cache_sizes,
                    "--policies",
                    args.policies,
                    "--out",
                    str(guard_json),
                ],
                env=env,
            )
            official = _mean(official_json)
            guard = _mean(guard_json)
            rows.append((tag, official, guard, fake, official_json))
            print(
                f"SUMMARY {tag} official {official:.10f} "
                f"{args.guard_label} {guard:.10f}",
                flush=True,
            )

    print("\n=== MICROHEAD SUMMARY ===", flush=True)
    print("| tag | literal cachesim mean line | JSON mean | guard mean | fake CSV |")
    print("|---|---|---:|---:|---|")
    for tag, official, guard, fake, _official_json in rows:
        print(
            f"| {tag} | `mean HRC-MAE across policies: {official:.4f}` | "
            f"{official:.10f} | {guard:.10f} | `{fake}` |"
        )
    if rows:
        best = min(rows, key=lambda row: row[1])
        print(
            f"\nBest official: {best[0]} {best[1]:.10f} "
            f"({args.guard_label} {best[2]:.10f})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
