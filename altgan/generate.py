"""Generate a synthetic trace from a fitted StackAtlas model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .model import StackAtlasModel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Path to .pkl.gz model.")
    p.add_argument("--n-records", type=int, default=100_000)
    p.add_argument("--n-streams", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True,
                   help="CSV output path. Contains stream_id, ts, obj_id, obj_size, opcode, tenant.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    model = StackAtlasModel.load(args.model)
    df = model.generate(args.n_records, n_streams=args.n_streams, seed=args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[altgan.generate] wrote {len(df):,} records to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
