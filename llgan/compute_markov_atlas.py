"""
IDEA #62 Phase A: Compute empirical Markov transition matrix over LRU stack-rank
buckets from real training traces.

The matrix T[i][j] = P(next_bucket=j | prev_bucket=i) captures first-order
autocorrelation in LRU stack distances across consecutive reuse events.
Saving it as a .npy file for use with --lru-markov-atlas in generate.py.

Usage
-----
    python compute_markov_atlas.py \
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
import subprocess

import numpy as np

from lru_stack_decoder import LRUStackDecoder, N_BUCKETS, _EDGES


def _read_oracle_general_obj_ids(path: str, max_records: int) -> np.ndarray:
    """Read obj_id column from libCacheSim oracleGeneral .zst binary file."""
    n_bytes = max_records * 24
    if path.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-d", "-c", path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        raw = proc.stdout.read(n_bytes)
        proc.stdout.close()
        proc.wait()
    else:
        with open(path, "rb") as f:
            raw = f.read(n_bytes)

    dt = np.dtype([
        ("ts",       "<u4"),
        ("obj_id",   "<u8"),
        ("obj_size", "<u4"),
        ("vtime",    "<i4"),
        ("op",       "<i2"),
        ("tenant",   "<i2"),
    ])
    n = len(raw) // 24
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    arr = np.frombuffer(raw[:n * 24], dtype=dt)
    return arr["obj_id"].astype(np.int64)


def compute_transition_matrix(
    trace_dir: str,
    max_files: int = 0,
    max_events_per_file: int = 500_000,
    verbose: bool = True,
) -> np.ndarray:
    """
    Walk all .zst files in trace_dir and accumulate the 8×8 transition matrix
    using IRD-proxy stack distances (same approximation as LRUStackDecoder.fit).

    Returns raw count matrix (unnormalised).
    """
    T = np.zeros((N_BUCKETS, N_BUCKETS), dtype=np.float64)
    total_transitions = 0

    files = sorted(glob.glob(os.path.join(trace_dir, "*.oracleGeneral.zst")))
    if not files:
        files = sorted(glob.glob(os.path.join(trace_dir, "*.zst")))
    if max_files > 0:
        files = files[:max_files]

    print(f"Found {len(files)} trace files in {trace_dir}")

    for fi, fpath in enumerate(files):
        try:
            obj_ids = _read_oracle_general_obj_ids(fpath, max_events_per_file)
        except Exception as e:
            print(f"  SKIP {os.path.basename(fpath)}: {e}")
            continue

        if len(obj_ids) == 0:
            continue

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
    total_counts = T_raw.sum()
    if total_counts > 0:
        marginal = T_raw.sum(axis=0) / total_counts
        print(f"\nMarginal PMF from transition counts:")
        for label, p_val in zip(labels, marginal):
            bar = "█" * int(p_val * 40)
            print(f"  {label:>12s}  {p_val:.4f}  {bar}")
    else:
        print("\nWARN: no transitions counted — check trace files and format")


if __name__ == "__main__":
    main()
