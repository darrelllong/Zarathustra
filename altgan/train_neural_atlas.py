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
    profile_to_cond_vector as _llgan_profile_to_cond_vector,
)

from .neural_atlas import fit_neural_atlas  # noqa: E402
from .train import _collect_files  # noqa: E402


_HASH_KEY_NAME_TOKENS = (
    "twitter",
    "metakv",
    "meta_kv",
    "meta-kv",
    "metacdn",
    "meta_cdn",
    "meta-cdn",
    "wikipedia",
    "wiki",
)


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
    p.add_argument("--phase-mode", default="position",
                   choices=("position", "unique_rate"),
                   help="Phase encoding: fixed position bins or running unique-rate bins.")
    p.add_argument("--rank-state-edges", default="",
                   help="Comma-separated stack-distance edges for optional distance-state bins.")
    p.add_argument("--transition-weight-mode", default="log",
                   choices=("log", "sqrt", "total", "uniform"),
                   help="How strongly repeated transition rows are weighted during transition-net training.")
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
    fallback_conds = 0
    for i, path in enumerate(paths, start=1):
        cond = _lookup_cond(cond_lookup, path, args.cond_dim)
        print(f"[altgan.train_neural_atlas] reading {i}/{len(paths)} {path}")
        frame = reader(str(path), args.records_per_file)
        if cond is None:
            cond = _frame_to_cond_vector(frame, path, args.cond_dim)
            fallback_conds += 1
        if cond is None:
            missing.append(path.name)
            continue
        frames.append(frame)
        conds.append(cond)
        names.append(path.name)
    if missing:
        print(f"[altgan.train_neural_atlas] skipped {len(missing)} files without char profiles")
    if fallback_conds:
        print(
            "[altgan.train_neural_atlas] computed "
            f"{fallback_conds} conditioning vectors from parsed traces"
        )
    if not frames:
        raise RuntimeError("no files had usable conditioning profiles")

    model = fit_neural_atlas(
        frames,
        conds,
        names,
        n_time_bins=args.n_time_bins,
        n_size_bins=args.n_size_bins,
        n_phase_bins=args.n_phase_bins,
        phase_mode=args.phase_mode,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        cond_noise_std=args.cond_noise_std,
        max_samples_per_state=args.max_samples_per_state,
        rank_state_edges=_parse_int_list(args.rank_state_edges),
        transition_weight_mode=args.transition_weight_mode,
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
    lookup = _load_lanl_file_characterizations(jsonl_path, cond_dim=cond_dim)
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            profile = row.get("profile")
            if not profile:
                continue
            if not _has_request_conditioning(profile):
                continue
            rel = row.get("rel_path", "")
            path = row.get("path", "")
            if not rel and not path:
                continue
            vec = torch.tensor(
                _profile_to_cond_vector(profile, cond_dim, rel, path),
                dtype=torch.float32,
            )
            for key in _char_row_keys(rel, path):
                lookup[key] = vec
    return lookup


def _load_lanl_file_characterizations(jsonl_path: str, cond_dim: int = 10) -> dict:
    raw_lookup = {}
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            profile = row.get("profile")
            rel = row.get("rel_path", "")
            if not profile or not rel or not _has_request_conditioning(profile):
                continue
            basename = Path(rel).name
            vec = torch.tensor(
                _profile_to_cond_vector(profile, cond_dim, rel),
                dtype=torch.float32,
            )
            raw_lookup[basename] = vec
            for ext in (".zst", ".gz"):
                if basename.endswith(ext):
                    raw_lookup[basename[: -len(ext)]] = vec
    return raw_lookup


def _profile_to_cond_vector(
    profile: dict,
    cond_dim: int = 10,
    *source_names: str,
) -> list[float]:
    return _llgan_profile_to_cond_vector(
        _sanitize_hash_key_profile(profile, source_names),
        cond_dim,
    )


def _has_request_conditioning(profile: dict) -> bool:
    request_keys = {
        "reuse_ratio",
        "burstiness_cv",
        "write_ratio",
        "opcode_switch_ratio",
        "iat_stats",
        "obj_size_stats",
        "tenant_summary",
        "obj_id_summary",
    }
    return any(key in profile for key in request_keys)


def _frame_to_cond_vector(df, path: Path | str, cond_dim: int = 10) -> np.ndarray | None:
    if df is None or len(df) == 0:
        return None
    profile: dict = {}
    path = Path(path)

    if "opcode" in df.columns:
        ops = df["opcode"].astype(str).str.lower().to_numpy()
        is_write = np.array([op.startswith("w") or op in {"13", "write"} for op in ops], dtype=bool)
        profile["write_ratio"] = float(is_write.mean()) if len(is_write) else 0.0
        profile["opcode_switch_ratio"] = (
            float(np.mean(ops[1:] != ops[:-1])) if len(ops) > 1 else 0.0
        )
    else:
        profile["write_ratio"] = 0.0
        profile["opcode_switch_ratio"] = 0.0

    if "ts" in df.columns:
        ts = _numeric_array(df["ts"])
        if len(ts) > 1:
            iat = np.diff(ts)
            iat = iat[np.isfinite(iat)]
            iat = np.maximum(iat, 0.0)
        else:
            iat = np.array([], dtype=np.float64)
        profile["iat_stats"] = {"q50": _quantile(iat, 0.5)}
        profile["iat_lag1_autocorr"] = _lag1_autocorr(iat)
        mean_iat = float(np.mean(iat)) if len(iat) else 0.0
        profile["burstiness_cv"] = float(np.std(iat) / mean_iat) if mean_iat > 0 else 1.0
    else:
        profile["iat_stats"] = {"q50": 0.0}
        profile["iat_lag1_autocorr"] = 0.0
        profile["burstiness_cv"] = 1.0

    if "obj_size" in df.columns:
        sizes = _numeric_array(df["obj_size"])
        profile["obj_size_stats"] = {
            "q50": _quantile(sizes, 0.5, default=4096.0),
            "std": float(np.std(sizes)) if len(sizes) else 0.0,
        }
    else:
        profile["obj_size_stats"] = {"q50": 4096.0, "std": 0.0}

    if "tenant" in df.columns:
        profile["tenant_summary"] = {"unique": int(df["tenant"].nunique(dropna=True))}
    else:
        profile["tenant_summary"] = {"unique": 1}

    if "obj_id" in df.columns:
        obj_series = df["obj_id"]
        profile["reuse_ratio"] = float(obj_series.duplicated().mean()) if len(obj_series) else 0.0
        profile["obj_id_summary"] = {"unique": int(obj_series.nunique(dropna=True))}
        obj_id_kind = "hash" if _looks_hash_keyed(path, obj_series) else "address"
        profile["obj_id_kind"] = obj_id_kind
        if obj_id_kind == "hash":
            profile["forward_seek_ratio"] = 0.5
            profile["backward_seek_ratio"] = 0.5
            profile["signed_stride_lag1_autocorr"] = 0.0
            profile["abs_stride_stats"] = None
        else:
            obj = _numeric_array(obj_series)
            diffs = np.diff(obj) if len(obj) > 1 else np.array([], dtype=np.float64)
            diffs = diffs[np.isfinite(diffs)]
            profile["forward_seek_ratio"] = float(np.mean(diffs > 0)) if len(diffs) else 0.5
            profile["backward_seek_ratio"] = float(np.mean(diffs < 0)) if len(diffs) else 0.5
            signed = np.sign(diffs) * np.log1p(np.abs(diffs)) if len(diffs) else diffs
            profile["signed_stride_lag1_autocorr"] = _lag1_autocorr(signed)
    else:
        profile["reuse_ratio"] = 0.0
        profile["obj_id_summary"] = {"unique": 1}
        profile["forward_seek_ratio"] = 0.5
        profile["backward_seek_ratio"] = 0.5
        profile["signed_stride_lag1_autocorr"] = 0.0

    return np.asarray(_profile_to_cond_vector(profile, cond_dim), dtype=np.float32)


def _sanitize_hash_key_profile(
    profile: dict,
    source_names: tuple[str, ...] = (),
) -> dict:
    """Neutralize address-stride conditioning for hash-keyed object IDs."""
    is_hash_keyed = profile.get("obj_id_kind") == "hash" or any(
        _name_looks_hash_keyed(name) for name in source_names
    )
    if not is_hash_keyed:
        return profile
    clean = dict(profile)
    clean["obj_id_kind"] = "hash"
    clean["forward_seek_ratio"] = 0.5
    clean["backward_seek_ratio"] = 0.5
    clean["signed_stride_lag1_autocorr"] = 0.0
    clean["abs_stride_stats"] = None
    return clean


def _looks_hash_keyed(path: Path, obj_series) -> bool:
    lower = str(path).lower()
    if any(token in lower for token in _HASH_KEY_NAME_TOKENS):
        return True
    sample = _numeric_array(obj_series.head(min(len(obj_series), 4096)))
    if len(sample) < 2:
        return False
    finite = sample[np.isfinite(sample)]
    if len(finite) < 2:
        return False
    unique_ratio = len(np.unique(finite)) / len(finite)
    max_abs = float(np.max(np.abs(finite)))
    diffs = np.abs(np.diff(finite))
    med_diff = float(np.median(diffs[np.isfinite(diffs)])) if len(diffs) else 0.0
    return unique_ratio > 0.95 and max_abs > 1e12 and med_diff > 1e9


def _name_looks_hash_keyed(name: str) -> bool:
    return any(token in str(name).lower() for token in _HASH_KEY_NAME_TOKENS)


def _numeric_array(series) -> np.ndarray:
    vals = []
    for raw in series.to_numpy():
        try:
            vals.append(float(raw))
        except (TypeError, ValueError, OverflowError):
            continue
    return np.asarray(vals, dtype=np.float64)


def _quantile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if len(values) == 0:
        return float(default)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(default)
    return float(np.quantile(finite, q))


def _lag1_autocorr(values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return 0.0
    a = finite[:-1]
    b = finite[1:]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    return corr if np.isfinite(corr) else 0.0


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
