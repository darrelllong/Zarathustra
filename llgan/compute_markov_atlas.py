"""
IDEA #62 Phase A: Compute empirical Markov transition matrix over LRU stack-rank
buckets from real training traces.

The matrix T[i][j] = P(next_bucket=j | prev_bucket=i) captures first-order
autocorrelation in LRU stack distances across consecutive reuse events.
Saving it as a .npy file for use with --lru-markov-atlas in generate.py.

Usage
-----
    python -m llgan.compute_markov_atlas \
        --trace-dir /tiamat/zarathustra/traces/alibaba \
        --output markov_atlas_alibaba.npy \
        [--max-files 239] \
        [--max-events-per-file 500000]

Output
------
  markov_atlas_alibaba.npy: (8, 8) float64 array, rows normalised to sum 1.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from lru_stack_decoder import LRUStackDecoder, N_BUCKETS, _EDGES


def compute_transition_matrix(
    trace_dir: str,
    fmt: str = "oracle_general",
    max_files: int = 0,
    max_events_per_file: int = 500_000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Walk all CSV files in trace_dir and accumulate the 8×8 transition matrix
    using IRD-proxy stack distances (same approximation as LRUStackDecoder.fit).

    Returns raw count matrix (unnormalised).
    """
    T = np.zeros((N_BUCKETS, N_BUCKETS), dtype=np.float64)
    total_transitions = 0

    files = sorted(glob.glob(os.path.join(trace_dir, "*.csv")))
    if max_files > 0:
        files = files[:max_files]

    for fi, fpath in enumerate(files):
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        if "obj_id" not in df.columns:
            continue

        obj_ids = df["obj_id"].values[:max_events_per_file].astype(np.int64)
        T_file = LRUStackDecoder.fit_transition_matrix_from_obj_ids(
            obj_ids, max_fit_events=max_events_per_file
        )
        n_file = int(T_file.sum())
        T += T_file
        total_transitions += n_file

        if verbose and (fi % 20 == 0 or fi == len(files) - 1):
            print(f"  [{fi+1:3d}/{len(files)}] {os.path.basename(fpath):40s} "
                  f"+{n_file:7d} transitions  (total={total_transitions:,})")

    return T


def main():
    p = argparse.ArgumentParser(
        description="Compute Markov atlas transition matrix from real traces (IDEA #62)"
    )
    p.add_argument("--trace-dir", required=True,
                   help="Directory containing real trace CSV files")
    p.add_argument("--output", default="markov_atlas.npy",
                   help="Output .npy path for the 8x8 transition matrix")
    p.add_argument("--max-files", type=int, default=0,
                   help="Limit number of files processed (0 = all)")
    p.add_argument("--max-events-per-file", type=int, default=500_000,
                   help="Max events read per trace file")
    p.add_argument("--corpus", default="alibaba",
                   choices=["alibaba", "tencent"],
                   help="Corpus label (used for output naming only)")
    args = p.parse_args()

    print(f"Computing Markov atlas from: {args.trace_dir}")
    T_raw = compute_transition_matrix(
        args.trace_dir,
        max_files=args.max_files,
        max_events_per_file=args.max_events_per_file,
    )

    # Normalise rows; rows with zero counts fall back to uniform.
    row_sums = T_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T_norm = T_raw / row_sums

    np.save(args.output, T_norm)
    print(f"\nSaved normalised transition matrix → {args.output}")
    print(f"Total cross-bucket transitions counted: {int(T_raw.sum()):,}")

    # Print the matrix for inspection
    labels = [f"[{_EDGES[k]},{_EDGES[k+1]})" for k in range(N_BUCKETS)]
    labels[-1] = f"[{_EDGES[-2]},+inf)"
    print(f"\nTransition matrix T[prev_bucket][next_bucket]:")
    header = f"  {'':>12s}  " + "  ".join(f"{l:>12s}" for l in labels)
    print(header)
    for i, row in enumerate(T_norm):
        vals = "  ".join(f"{v:.4f}" for v in row)
        print(f"  {labels[i]:>12s}  {vals}")

    # Compare marginal (row-sum) vs default PMF
    marginal = T_raw.sum(axis=0)
    marginal = marginal / marginal.sum()
    print(f"\nMarginal PMF from transition counts:")
    for label, p_val in zip(labels, marginal):
        bar = "█" * int(p_val * 40)
        print(f"  {label:>12s}  {p_val:.4f}  {bar}")


if __name__ == "__main__":
    main()
