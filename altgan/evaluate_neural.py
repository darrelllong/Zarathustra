"""Evaluate a profile-conditioned NeuralStack model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_LLGAN = _ROOT / "llgan"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_LLGAN))

from llgan.dataset import load_file_characterizations  # noqa: E402
from llgan.long_rollout_eval import _gap, _metrics_for_stream, _per_stream_obj_ids, _sample_real_stream  # noqa: E402

from .neural_stack import NeuralStackModel  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--fmt", required=True)
    p.add_argument("--char-file", required=True)
    p.add_argument("--cond-dim", type=int, default=13)
    p.add_argument("--source-traces", default="",
                   help="Comma-separated source trace basenames for stream conditioning.")
    p.add_argument("--condition-from-real-manifest", action="store_true",
                   help="Use the real manifest's stream files as source conditioning.")
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--real-manifest", default="")
    p.add_argument("--cache-sizes", default="")
    p.add_argument("--output", default="")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = NeuralStackModel.load(args.model)

    cond_lookup = load_file_characterizations(args.char_file, cond_dim=args.cond_dim)
    source_names = [s.strip() for s in args.source_traces.split(",") if s.strip()]

    real_df, real_manifest = _sample_real_stream(
        args.trace_dir,
        args.fmt,
        args.n_records,
        args.n_streams,
        args.seed,
        manifest_path=args.real_manifest,
    )
    if args.condition_from_real_manifest:
        source_names = _source_names_from_manifest(real_manifest)

    conds = None
    if source_names:
        conds = np.vstack([_lookup_cond(cond_lookup, name, args.cond_dim) for name in source_names])

    fake_df = model.generate(
        args.n_records,
        n_streams=args.n_streams,
        seed=args.seed,
        conds=conds,
        temperature=args.temperature,
    )

    if args.cache_sizes:
        cache_sizes = np.array([int(x) for x in args.cache_sizes.split(",") if x.strip()],
                               dtype=np.int64)
    else:
        real_streams = _per_stream_obj_ids(real_df)
        footprint = int(np.mean([np.unique(s).size for s in real_streams])) if real_streams else 2
        cache_sizes = np.unique(np.geomspace(max(1, footprint // 1000), max(footprint, 2), 20)
                                .astype(np.int64))

    fake_m = _metrics_for_stream(fake_df, cache_sizes)
    real_m = _metrics_for_stream(real_df, cache_sizes)
    gap_m = _gap(fake_m, real_m)
    result = {
        "model": args.model,
        "trace_dir": args.trace_dir,
        "fmt": args.fmt,
        "seed": args.seed,
        "n_records": args.n_records,
        "n_streams": args.n_streams,
        "temperature": args.temperature,
        "source_traces": source_names,
        "fake": fake_m,
        "real": real_m,
        "gap": gap_m,
        "real_manifest": real_manifest,
    }
    out_path = Path(args.output) if args.output else Path(args.model).with_suffix("").with_suffix(".neural_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[altgan.evaluate_neural] wrote {out_path}")
    print(json.dumps({
        "hrc_mae": gap_m["hrc_mae"],
        "fake_reuse_access": fake_m["reuse_access_rate"],
        "real_reuse_access": real_m["reuse_access_rate"],
        "fake_stack_median": fake_m["stack_distance_median"],
        "real_stack_median": real_m["stack_distance_median"],
        "fake_stack_p90": fake_m["stack_distance_p90"],
        "real_stack_p90": real_m["stack_distance_p90"],
    }, indent=2))
    return 0


def _source_names_from_manifest(manifest: dict) -> list[str]:
    names = []
    for entries in manifest.get("streams", []):
        if not entries:
            continue
        names.append(Path(entries[0]["path"]).name)
    return names


def _lookup_cond(cond_lookup: dict, name_or_path: str, cond_dim: int) -> np.ndarray:
    name = Path(name_or_path).name
    keys = [name]
    for suffix in (".zst", ".gz"):
        if name.endswith(suffix):
            keys.append(name[: -len(suffix)])
    for key in keys:
        val = cond_lookup.get(key)
        if val is not None:
            arr = val.detach().cpu().numpy().astype(np.float32)
            if len(arr) < cond_dim:
                arr = np.pad(arr, (0, cond_dim - len(arr)))
            return arr[:cond_dim]
    raise KeyError(f"no characterization vector for {name_or_path}")


if __name__ == "__main__":
    raise SystemExit(main())
