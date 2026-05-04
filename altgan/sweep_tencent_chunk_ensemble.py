"""Tencent chunk-level object-process ensemble sweep.

Combines the standing Tencent atlas fake with a policy-specialized donor fake by
replacing contiguous obj_id blocks.  This tests whether two object processes
with complementary cache-policy errors can be mixed without destroying the mark
schedule learned by the atlas.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _parse_floats(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--donor", required=True)
    p.add_argument("--real", required=True)
    p.add_argument("--output-root", default="/tiamat/zarathustra/altgan-output")
    p.add_argument("--tag", default="tencent_chunk_ensemble")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mix", type=_parse_floats, default=[0.25, 0.50, 0.75],
                   help="Comma-separated fraction of chunks to take from donor.")
    p.add_argument("--chunk-size", type=_parse_ints, default=[128, 512, 2048])
    p.add_argument("--cache-sizes", default="32,128,512,2048,8192")
    p.add_argument("--policies", default="lru,arc,fifo,sieve,slru,car")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _make_mix(base: pd.DataFrame, donor: pd.DataFrame, *, mix: float, chunk_size: int, seed: int) -> pd.DataFrame:
    if len(base) != len(donor):
        raise ValueError(f"base/donor length mismatch: {len(base)} vs {len(donor)}")
    rng = np.random.default_rng(seed)
    n = len(base)
    out = base.copy()
    obj = base["obj_id"].to_numpy(copy=True)
    donor_obj = donor["obj_id"].to_numpy(copy=False)
    chunks = list(range(0, n, chunk_size))
    take = rng.random(len(chunks)) < mix
    if not take.any() and mix > 0:
        take[int(rng.integers(0, len(take)))] = True
    for use_donor, start in zip(take, chunks, strict=False):
        if not use_donor:
            continue
        end = min(n, start + chunk_size)
        obj[start:end] = donor_obj[start:end]
    out["obj_id"] = obj
    return out


def _run(cmd: list[str], dry_run: bool, env: dict[str, str]) -> None:
    print("+ " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, env=env)


def _eval_cmd(args: argparse.Namespace, fake: Path, out_json: Path) -> list[str]:
    return [
        sys.executable, "-u", "-m", "llgan.cachesim_eval",
        "--fake", str(fake),
        "--real", args.real,
        "--cache-sizes", args.cache_sizes,
        "--policies", args.policies,
        "--out", str(out_json),
    ]


def _mean(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open() as f:
        data = json.load(f)
    return float(data.get("mean_hrc_mae", data.get("mean")))


def _fmt(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def main() -> int:
    args = _parse_args()
    root = Path(args.output_root)
    eval_root = root / "cachesim_lanl"
    root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    print(f"[chunk_ensemble] reading base {args.base}", flush=True)
    base = pd.read_csv(args.base)
    print(f"[chunk_ensemble] reading donor {args.donor}", flush=True)
    donor = pd.read_csv(args.donor)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"

    summary: list[tuple[str, float | None]] = []
    for chunk_size in args.chunk_size:
        for mix in args.mix:
            tag = f"{args.tag}_ck{chunk_size}_m{_fmt(mix)}_seed{args.seed}"
            fake = root / f"{tag}_fake_{len(base) // 1000}k.csv"
            out_json = eval_root / f"{tag}_official6.json"
            if args.skip_existing and out_json.exists():
                print(f"[chunk_ensemble] skip existing {out_json}", flush=True)
            else:
                print(f"[chunk_ensemble] generating chunk={chunk_size} mix={mix:g}", flush=True)
                if not args.dry_run:
                    out = _make_mix(base, donor, mix=mix, chunk_size=chunk_size, seed=args.seed)
                    out.to_csv(fake, index=False)
                    print(f"[chunk_ensemble] wrote {fake}", flush=True)
                _run(_eval_cmd(args, fake, out_json), args.dry_run, env)
            mean = _mean(out_json)
            summary.append((tag, mean))
            if mean is not None:
                print(f"[chunk_ensemble] {tag}: {mean:.10f}", flush=True)

    print("\n=== TENCENT CHUNK-ENSEMBLE RESULTS ===", flush=True)
    for tag, mean in sorted(summary, key=lambda item: float("inf") if item[1] is None else item[1]):
        text = "pending" if mean is None else f"{mean:.10f}"
        print(f"{tag:<72} {text}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
