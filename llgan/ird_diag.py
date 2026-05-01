"""IRD distribution diagnostic — learning from 2DIO (EuroSys '26).

2DIO's claim: non-concave HRC cliffs come from the IRD distribution shape, not
from the frequency distribution. If LLNL's b2 fake has a different IRD shape
than real, that explains why post-hoc-frequency knobs (hot-pool, K, adj-dup)
plateau short of LANL's PhaseAtlas+marks on alibaba.

Computes per-stream IRD histograms (log-spaced bins, 24 bins from 1 to ~1M)
for:
  - REAL alibaba 1M
  - LLNL R208 fake (b2 + post-hoc knobs)
  - LANL fake (PhaseAtlas+marks)

Output: tab-separated bin midpoint vs PMF for each. Lets us see *where* the
LLNL synthetic's IRD shape diverges from real, then design a state-space
or sampling refinement specific to that region.

Usage:
  python -m llgan.ird_diag --csv real=/path/to/real.csv --csv llnl=/path/to/llnl_fake.csv --csv lanl=/path/to/lanl_fake.csv
"""
import argparse
import csv
import math
from collections import defaultdict


def compute_per_stream_irds(csv_path: str, n_bins: int = 24) -> dict:
    """Read a fake/real CSV (stream_id,ts,obj_id,...) and return per-stream IRD
    histograms + global histogram. IRD = gap-between-consecutive-same-key.
    """
    # Read all (stream_id, obj_id) pairs in order; track last-seen per (stream, key)
    last_seen: dict = {}  # (stream_id, obj_id) -> last position within stream
    stream_pos: dict = defaultdict(int)
    irds_per_stream: dict = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["stream_id"])
            oid = int(row["obj_id"])
            pos = stream_pos[sid]
            key = (sid, oid)
            if key in last_seen:
                ird = pos - last_seen[key]
                if ird > 0:
                    irds_per_stream[sid].append(ird)
            last_seen[key] = pos
            stream_pos[sid] += 1

    # Build histograms with log-spaced bins from 1 to max-stream-length
    max_pos = max(stream_pos.values()) if stream_pos else 1
    bin_edges = [10 ** (i * math.log10(max_pos) / n_bins) for i in range(n_bins + 1)]
    bin_edges[0] = 1.0  # floor

    def histogram(irds):
        h = [0] * n_bins
        for d in irds:
            # Linear search; n_bins is tiny
            for i in range(n_bins):
                if d <= bin_edges[i + 1]:
                    h[i] += 1
                    break
            else:
                h[-1] += 1
        total = sum(h)
        if total == 0:
            return [0.0] * n_bins
        return [c / total for c in h]

    # Per-stream histograms
    per_stream_hist = {sid: histogram(irds) for sid, irds in irds_per_stream.items()}
    # Global histogram (all streams combined)
    all_irds = []
    for irds in irds_per_stream.values():
        all_irds.extend(irds)
    global_hist = histogram(all_irds)

    return {
        "bin_edges": bin_edges,
        "per_stream": per_stream_hist,
        "global": global_hist,
        "n_streams": len(stream_pos),
        "stream_lens": dict(stream_pos),
        "total_irds": len(all_irds),
    }


def print_comparison(results: dict) -> None:
    """results: {label: histogram_dict from compute_per_stream_irds}.
    Print a per-bin side-by-side comparison and a divergence summary.
    """
    labels = list(results.keys())
    if not labels:
        return
    bin_edges = results[labels[0]]["bin_edges"]
    n_bins = len(bin_edges) - 1

    print(f"\nPer-bin global IRD PMF comparison (3 sources, {n_bins} log bins)")
    print(f"{'bin':>4}  {'low':>8}  {'high':>10}  " + "  ".join(f"{lbl:>10}" for lbl in labels) + "  divergence")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        vals = [results[lbl]["global"][i] for lbl in labels]
        # Divergence = max - min across labels for this bin
        div = max(vals) - min(vals) if vals else 0.0
        print(f"{i:>4}  {lo:>8.0f}  {hi:>10.0f}  " + "  ".join(f"{v:>10.4f}" for v in vals) + f"  {div:>+8.4f}")

    # Total |Δ| between LLNL and REAL, between LANL and REAL
    if "real" in results:
        real = results["real"]["global"]
        for lbl in labels:
            if lbl == "real":
                continue
            other = results[lbl]["global"]
            l1 = sum(abs(r - o) for r, o in zip(real, other))
            print(f"\nL1(real, {lbl}) = {l1:.4f}  (lower = closer to real IRD shape)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="label=path; pass multiple, e.g. --csv real=/path --csv llnl=/path")
    ap.add_argument("--n-bins", type=int, default=24)
    args = ap.parse_args()

    results = {}
    for spec in args.csv:
        if "=" not in spec:
            raise ValueError(f"--csv expects label=path, got {spec!r}")
        label, path = spec.split("=", 1)
        print(f"Reading {label}: {path}")
        results[label] = compute_per_stream_irds(path, n_bins=args.n_bins)
        print(f"  {results[label]['n_streams']} streams, {results[label]['total_irds']} IRDs total")

    print_comparison(results)


if __name__ == "__main__":
    main()
