"""Attach an autoregressive neural mark head to a NeuralAtlas/PhaseAtlas model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import _READERS, load_file_characterizations  # noqa: E402

from .neural_atlas import NeuralAtlasModel  # noqa: E402
from .neural_marks import fit_mark_head  # noqa: E402
from .train import _collect_files  # noqa: E402
from .train_neural_atlas import _lookup_cond, _manifest_source_names  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="Existing NeuralAtlas/PhaseAtlas .pkl.gz checkpoint.")
    p.add_argument("--output", required=True,
                   help="Output checkpoint with mark_model attached.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="Single trace file.")
    src.add_argument("--trace-dir", help="Directory of trace files.")
    p.add_argument("--fmt", required=True, choices=sorted(_READERS))
    p.add_argument("--char-file", required=True)
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--max-files", type=int, default=64,
                   help="Maximum files sampled from --trace-dir; 0 means all.")
    p.add_argument("--exclude-manifest", default="",
                   help="Long-rollout real manifest whose source files must be held out.")
    p.add_argument("--records-per-file", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--window-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = NeuralAtlasModel.load(args.model)
    reader = _READERS[args.fmt]
    cond_lookup = load_file_characterizations(args.char_file, cond_dim=args.cond_dim)
    paths = _select_paths(args)
    frames = []
    conds = []
    missing = []
    for i, path in enumerate(paths, start=1):
        cond = _lookup_cond(cond_lookup, path, args.cond_dim)
        if cond is None:
            missing.append(path.name)
            continue
        print(f"[altgan.train_neural_marks] reading {i}/{len(paths)} {path}")
        frames.append(reader(str(path), args.records_per_file))
        conds.append(cond)
    if missing:
        print(f"[altgan.train_neural_marks] skipped {len(missing)} files without char profiles")
    if not frames:
        raise RuntimeError("no files had usable conditioning profiles")

    mark_model = fit_mark_head(
        model,
        frames,
        conds,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        window_len=args.window_len,
        lr=args.lr,
        seed=args.seed,
    )
    model.mark_model = mark_model
    model.metadata.setdefault("attached_models", {})
    model.metadata["attached_models"]["mark_model"] = mark_model.metadata
    model.metadata["mark_training"] = {
        "fmt": args.fmt,
        "char_file": args.char_file,
        "paths": [str(p) for p in paths],
        "records_per_file": args.records_per_file,
    }
    model.save(args.output)
    print(f"[altgan.train_neural_marks] wrote {args.output}")
    print(json.dumps(mark_model.metadata, indent=2))
    return 0


def _select_paths(args: argparse.Namespace) -> list[Path]:
    if args.trace:
        return [Path(args.trace)]
    paths = _collect_files(args.trace_dir, args.fmt)
    excluded = _manifest_source_names(args.exclude_manifest)
    if excluded:
        before = len(paths)
        paths = [p for p in paths if p.name not in excluded]
        print(
            "[altgan.train_neural_marks] excluded "
            f"{before - len(paths)} manifest source files from training"
        )
    if args.max_files and len(paths) > args.max_files:
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        paths = sorted(paths[:args.max_files])
    if not paths:
        raise RuntimeError("no trace files selected")
    return paths


if __name__ == "__main__":
    raise SystemExit(main())
