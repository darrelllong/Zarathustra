"""Fit a StackAtlas alternative generator from raw traces."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import _READERS  # noqa: E402

from .model import StackAtlasModel  # noqa: E402


def _collect_files(trace_dir: str, fmt: str) -> list[Path]:
    d = Path(trace_dir)
    if not d.exists():
        raise FileNotFoundError(trace_dir)
    files = [
        p for p in d.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.name.upper() not in {"README", "README.TXT"}
    ]
    if fmt in {"oracle_general", "lcs"}:
        files = [
            p for p in files
            if p.suffix in {"", ".zst", ".gz", ".bin", ".lcs"}
            or ".oracleGeneral" in p.name
            or ".lcs" in p.name
        ]
    return sorted(files)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Single trace file.")
    src.add_argument("--trace-dir", help="Directory of trace files.")
    p.add_argument("--fmt", required=True, choices=sorted(_READERS),
                   help="Trace format understood by llgan.dataset.")
    p.add_argument("--output", default="altgan_stackatlas.pkl.gz",
                   help="Output model path.")
    p.add_argument("--max-files", type=int, default=16,
                   help="Maximum files sampled from --trace-dir; 0 means all.")
    p.add_argument("--records-per-file", type=int, default=50_000,
                   help="Records read from each file.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--time-bins", type=int, default=4)
    p.add_argument("--size-bins", type=int, default=4)
    p.add_argument("--max-samples-per-state", type=int, default=4096)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reader = _READERS[args.fmt]

    if args.trace:
        paths = [Path(args.trace)]
    else:
        paths = _collect_files(args.trace_dir, args.fmt)
        if args.max_files and len(paths) > args.max_files:
            rng = random.Random(args.seed)
            rng.shuffle(paths)
            paths = sorted(paths[:args.max_files])
    if not paths:
        raise RuntimeError("no trace files selected")

    frames = []
    for i, path in enumerate(paths, start=1):
        print(f"[altgan.train] reading {i}/{len(paths)} {path}")
        frames.append(reader(str(path), args.records_per_file))

    model = StackAtlasModel.fit(
        frames,
        n_time_bins=args.time_bins,
        n_size_bins=args.size_bins,
        max_samples_per_state=args.max_samples_per_state,
        seed=args.seed,
    )
    model.metadata.update({
        "fmt": args.fmt,
        "paths": [str(p) for p in paths],
        "records_per_file": args.records_per_file,
    })
    model.save(args.output)
    print(f"[altgan.train] wrote {args.output}")
    print(f"[altgan.train] metadata={model.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
