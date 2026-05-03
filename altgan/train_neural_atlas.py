"""Train a profile-conditioned NeuralAtlas altgan model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import (  # noqa: E402
    _READERS,
    load_file_characterizations as _llgan_load_file_characterizations,
    profile_to_cond_vector,
)

from .neural_atlas import fit_neural_atlas  # noqa: E402
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
    p.add_argument("--exclude-manifest", default="",
                   help="Long-rollout real manifest whose source files must be held out.")
    p.add_argument("--records-per-file", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--cond-noise-std", type=float, default=0.0,
                   help="Train-time Gaussian noise added to conditioning vectors.")
    p.add_argument("--n-time-bins", type=int, default=4)
    p.add_argument("--n-size-bins", type=int, default=4)
    p.add_argument("--n-phase-bins", type=int, default=1,
                   help="Within-file position bins added to atlas state for nonstationary traces.")
    p.add_argument("--rank-state-edges", default="",
                   help="Comma-separated stack-distance edges for optional distance-state bins.")
    p.add_argument("--max-samples-per-state", type=int, default=1024)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    reader = _READERS[args.fmt]
    cond_lookup = _load_file_characterizations(args.char_file, cond_dim=args.cond_dim)

    if args.trace:
        paths = [Path(args.trace)]
    else:
        paths = _collect_files(args.trace_dir, args.fmt)
        excluded = _manifest_source_names(args.exclude_manifest)
        if excluded:
            before = len(paths)
            paths = [p for p in paths if p.name not in excluded]
            print(
                "[altgan.train_neural_atlas] excluded "
                f"{before - len(paths)} manifest source files from training"
            )
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
        print(f"[altgan.train_neural_atlas] reading {i}/{len(paths)} {path}")
        frames.append(reader(str(path), args.records_per_file))
        conds.append(cond)
        names.append(path.name)
    if missing:
        print(f"[altgan.train_neural_atlas] skipped {len(missing)} files without char profiles")
    if not frames:
        raise RuntimeError("no files had usable conditioning profiles")

    model = fit_neural_atlas(
        frames,
        conds,
        names,
        n_time_bins=args.n_time_bins,
        n_size_bins=args.n_size_bins,
        n_phase_bins=args.n_phase_bins,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        cond_noise_std=args.cond_noise_std,
        max_samples_per_state=args.max_samples_per_state,
        rank_state_edges=_parse_int_list(args.rank_state_edges),
        seed=args.seed,
    )
    model.metadata.update({
        "fmt": args.fmt,
        "char_file": args.char_file,
        "paths": [str(p) for p in paths],
        "records_per_file": args.records_per_file,
        "cond_noise_std": args.cond_noise_std,
        "rank_state_edges": _parse_int_list(args.rank_state_edges),
    })
    model.save(args.output)
    print(f"[altgan.train_neural_atlas] wrote {args.output}")
    print(f"[altgan.train_neural_atlas] metadata={model.metadata}")
    return 0


def _lookup_cond(cond_lookup: dict, path: Path, cond_dim: int) -> np.ndarray | None:
    keys = _cond_lookup_keys(path.name)
    for key in keys:
        val = cond_lookup.get(key)
        if val is not None:
            arr = val.detach().cpu().numpy().astype(np.float32)
            if len(arr) < cond_dim:
                arr = np.pad(arr, (0, cond_dim - len(arr)))
            return arr[:cond_dim]
    return None


def _cond_lookup_keys(name_or_path: str) -> list[str]:
    name = Path(name_or_path).name
    keys = [name]
    for suffix in (".oracleGeneral.bin.zst", ".oracleGeneral.zst", ".zst", ".gz"):
        if name.endswith(suffix):
            keys.append(name[: -len(suffix)])
    msr_raw = _msr_exchange_raw_name(name)
    if msr_raw:
        keys.append(msr_raw)
        if msr_raw.endswith(".gz"):
            keys.append(msr_raw[: -len(".gz")])
    baleen_raw = _baleen24_raw_rel_path(name)
    if baleen_raw:
        keys.extend([
            baleen_raw,
            f"/tiamat/zarathustra/traces/{baleen_raw}",
            Path(baleen_raw).name,
        ])
    cp_raws = _cloudphysics_raw_names(name)
    keys.extend(cp_raws)
    return list(dict.fromkeys(keys))


def _load_file_characterizations(jsonl_path: str, cond_dim: int = 10) -> dict:
    lookup = _llgan_load_file_characterizations(jsonl_path, cond_dim=cond_dim)
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            profile = row.get("profile")
            if not profile:
                continue
            rel = row.get("rel_path", "")
            path = row.get("path", "")
            if not rel and not path:
                continue
            vec = torch.tensor(profile_to_cond_vector(profile, cond_dim), dtype=torch.float32)
            for key in _char_row_keys(rel, path):
                lookup[key] = vec
    return lookup


def _parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _char_row_keys(rel_path: str, abs_path: str) -> list[str]:
    keys = []
    for raw in (rel_path, abs_path):
        if not raw:
            continue
        p = Path(raw)
        keys.extend([raw, p.name])
        for suffix in (".zst", ".gz"):
            if p.name.endswith(suffix):
                keys.append(p.name[: -len(suffix)])
    return list(dict.fromkeys(keys))


def _msr_exchange_raw_name(name: str) -> str | None:
    stem = name
    if stem.endswith(".zst"):
        stem = stem[: -len(".zst")]
    if not stem.endswith(".oracleGeneral"):
        return None
    stem = stem[: -len(".oracleGeneral")]
    if not stem.startswith("Exchange_"):
        return None
    rest = stem[len("Exchange_"):]
    if "_" not in rest:
        return None
    date, time = rest.split("_", 1)
    return f"Exchange.{date}.{time}.trace.csv.gz"


def _baleen24_raw_rel_path(name: str) -> str | None:
    stem = name
    if stem.endswith(".zst"):
        stem = stem[: -len(".zst")]
    for suffix in (".oracleGeneral.bin", ".oracleGeneral"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("__")
    if len(parts) < 4 or not parts[0].startswith("storage"):
        return None
    return "Baleen24/extracted/" + "/".join(parts[:-1]) + f"/{parts[-1]}.trace"


def _cloudphysics_raw_names(name: str) -> list[str]:
    stem = name
    if stem.endswith(".zst"):
        stem = stem[: -len(".zst")]
    for suffix in (".oracleGeneral.bin", ".oracleGeneral"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem.startswith("w") or not stem[1:].isdigit():
        return []
    keys = []
    for disk in ("vscsi1", "vscsi2"):
        raw = f"s3-cache-datasets/cache_dataset_lcs/cloudphysics/{stem}_{disk}.vscsitrace.lcs.zst"
        keys.extend([raw, Path(raw).name, Path(raw).name[: -len(".zst")]])
    return keys


def _manifest_source_names(manifest_path: str) -> set[str]:
    if not manifest_path:
        return set()
    manifest = json.loads(Path(manifest_path).read_text())
    names = set()
    for entries in manifest.get("streams", []):
        for entry in entries:
            path = entry.get("path")
            if path:
                source_name = Path(path).name
                names.add(source_name)
                for key in _cond_lookup_keys(source_name):
                    names.add(Path(key).name)
    return names


if __name__ == "__main__":
    raise SystemExit(main())
