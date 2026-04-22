"""Train a profile-conditioned NeuralStack altgan model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import _READERS, load_file_characterizations  # noqa: E402

from .neural_stack import fit_neural_stack  # noqa: E402
from .train import _collect_files  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Single trace file.")
    src.add_argument("--trace-dir", help="Directory of trace files.")
    p.add_argument("--fmt", required=True, choices=sorted(_READERS))
    p.add_argument("--char-file", required=True,
                   help="trace_characterizations.jsonl used for workload conditioning.")
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--output", required=True)
    p.add_argument("--max-files", type=int, default=64,
                   help="Maximum files sampled from --trace-dir; 0 means all.")
    p.add_argument("--records-per-file", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-samples-per-file", type=int, default=2048)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reader = _READERS[args.fmt]
    cond_lookup = load_file_characterizations(args.char_file, cond_dim=args.cond_dim)

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
    conds = []
    names = []
    missing = []
    for i, path in enumerate(paths, start=1):
        cond = _lookup_cond(cond_lookup, path, args.cond_dim)
        if cond is None:
            missing.append(path.name)
            continue
        print(f"[altgan.train_neural] reading {i}/{len(paths)} {path}")
        frames.append(reader(str(path), args.records_per_file))
        conds.append(cond)
        names.append(path.name)
    if missing:
        print(f"[altgan.train_neural] skipped {len(missing)} files without char profiles")
    if not frames:
        raise RuntimeError("no files had usable conditioning profiles")

    model = fit_neural_stack(
        frames,
        conds,
        names,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        max_samples_per_file=args.max_samples_per_file,
        seed=args.seed,
    )
    model.metadata.update({
        "fmt": args.fmt,
        "char_file": args.char_file,
        "paths": [str(p) for p in paths],
        "records_per_file": args.records_per_file,
    })
    model.save(args.output)
    print(f"[altgan.train_neural] wrote {args.output}")
    print(f"[altgan.train_neural] metadata={model.metadata}")
    return 0


def _lookup_cond(cond_lookup: dict, path: Path, cond_dim: int) -> np.ndarray | None:
    keys = [path.name]
    for suffix in (".zst", ".gz"):
        if path.name.endswith(suffix):
            keys.append(path.name[: -len(suffix)])
    for key in keys:
        val = cond_lookup.get(key)
        if val is not None:
            arr = val.detach().cpu().numpy().astype(np.float32)
            if len(arr) < cond_dim:
                arr = np.pad(arr, (0, cond_dim - len(arr)))
            return arr[:cond_dim]
    return None


if __name__ == "__main__":
    raise SystemExit(main())
