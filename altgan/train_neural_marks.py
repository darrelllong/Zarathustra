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

from llgan.dataset import _READERS  # noqa: E402

from .neural_atlas import NeuralAtlasModel  # noqa: E402
from .neural_marks import fit_mark_head  # noqa: E402
from .train import _collect_files  # noqa: E402
from .train_neural_atlas import (  # noqa: E402
    _load_file_characterizations,
    _lookup_cond,
    _manifest_source_names,
)


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
    p.add_argument("--numeric-loss-weight", type=float, default=1.0,
                   help="Weight for dt/size regression loss in the mark head.")
    p.add_argument("--categorical-loss-weight", type=float, default=0.5,
                   help="Weight for opcode/tenant classification loss in the mark head.")
    p.add_argument("--device", default="auto",
                   help="Torch device for mark-head training: auto, cuda, or cpu.")
    p.add_argument("--progress-every", type=int, default=1,
                   help="Print flushed progress every N epochs; 0 disables progress prints.")
    p.add_argument("--snapshot-epochs", default="",
                   help="Comma-separated 1-based epochs to save during training.")
    p.add_argument("--snapshot-template", default="",
                   help="Optional output template for snapshots; may include {epoch}.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = NeuralAtlasModel.load(args.model)
    reader = _READERS[args.fmt]
    cond_lookup = _load_file_characterizations(args.char_file, cond_dim=args.cond_dim)
    paths = _select_paths(args)
    frames = []
    conds = []
    missing = []
    for i, path in enumerate(paths, start=1):
        cond = _lookup_cond(cond_lookup, path, args.cond_dim)
        if cond is None:
            missing.append(path.name)
            continue
        print(f"[altgan.train_neural_marks] reading {i}/{len(paths)} {path}", flush=True)
        frames.append(reader(str(path), args.records_per_file))
        conds.append(cond)
    if missing:
        print(
            f"[altgan.train_neural_marks] skipped {len(missing)} files without char profiles",
            flush=True,
        )
    if not frames:
        raise RuntimeError("no files had usable conditioning profiles")

    training_meta = {
        "fmt": args.fmt,
        "char_file": args.char_file,
        "paths": [str(p) for p in paths],
        "records_per_file": args.records_per_file,
    }
    snapshot_epochs = _parse_epoch_list(args.snapshot_epochs)

    def save_snapshot(epoch: int, mark_model) -> None:
        path = _snapshot_path(args.output, args.snapshot_template, epoch)
        path.parent.mkdir(parents=True, exist_ok=True)
        _attach_mark_model(model, mark_model, training_meta)
        model.save(path)
        print(f"[altgan.train_neural_marks] wrote snapshot epoch={epoch} {path}", flush=True)

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
        numeric_loss_weight=args.numeric_loss_weight,
        categorical_loss_weight=args.categorical_loss_weight,
        seed=args.seed,
        device=args.device,
        progress_every=args.progress_every,
        checkpoint_epochs=snapshot_epochs,
        checkpoint_callback=save_snapshot if snapshot_epochs else None,
    )
    _attach_mark_model(model, mark_model, training_meta)
    model.save(args.output)
    print(f"[altgan.train_neural_marks] wrote {args.output}", flush=True)
    print(json.dumps(mark_model.metadata, indent=2), flush=True)
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
            f"{before - len(paths)} manifest source files from training",
            flush=True,
        )
    if args.max_files and len(paths) > args.max_files:
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        paths = sorted(paths[:args.max_files])
    if not paths:
        raise RuntimeError("no trace files selected")
    return paths


def _attach_mark_model(model: NeuralAtlasModel, mark_model, training_meta: dict) -> None:
    model.mark_model = mark_model
    model.metadata.setdefault("attached_models", {})
    model.metadata["attached_models"]["mark_model"] = mark_model.metadata
    model.metadata["mark_training"] = dict(training_meta)


def _parse_epoch_list(value: str) -> set[int]:
    epochs: set[int] = set()
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        epoch = int(part)
        if epoch <= 0:
            raise ValueError("--snapshot-epochs values must be positive")
        epochs.add(epoch)
    return epochs


def _snapshot_path(output: str, template: str, epoch: int) -> Path:
    if template:
        return Path(template.format(epoch=epoch))
    path = Path(output)
    name = path.name
    if name.endswith(".pkl.gz"):
        snap = f"{name[:-7]}_epoch{epoch:04d}.pkl.gz"
    else:
        snap = f"{path.stem}_epoch{epoch:04d}{path.suffix}"
    return path.with_name(snap)


if __name__ == "__main__":
    raise SystemExit(main())
