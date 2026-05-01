"""LLNL cachesim-based eval — replaces eval-csv-hrc as the headline metric.

Reason (R181 corrective): the Python eval-csv-hrc surface uses
cache_sizes = footprint_mean × [0.005..3.0] which under-samples the
small-cache regime where shape error dominates. tools/cachesim runs the
6 production policies (LRU, ARC, FIFO, SLRU, CAR, SIEVE) at fixed
cache sizes, giving a policy-relevant HRC-MAE that matches what users
actually care about.

Usage:
    python -m llgan.cachesim_eval --fake FAKE.csv --real REAL.csv
                                  [--cache-sizes 32,128,512,2048,8192]
                                  [--policies lru,arc,fifo,sieve,slru,car]

Returns aligned per-policy / per-cache miss-ratio table + mean HRC-MAE.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_SIZES = "32,128,512,2048,8192"
DEFAULT_POLICIES = "lru,arc,fifo,sieve,slru,car"


def _find_cachesim() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "tools" / "cachesim" / "target" / "release" / "cachesim"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("cachesim")
    if found:
        return found
    raise FileNotFoundError(
        f"cachesim binary not found. Build it first:\n"
        f"  cd {repo_root}/tools/cachesim && cargo build --release"
    )


def _run_cachesim(binary: str, trace: str, sizes: str, policies: str) -> list[dict]:
    """Run cachesim once on a trace. Strips the [cachesim] log preamble that
    precedes the JSON output on stdout."""
    out = subprocess.run(
        [binary, "--trace", trace, "--policy", policies,
         "--cache-sizes", sizes, "--out", "-"],
        capture_output=True, text=True, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"cachesim failed (exit {out.returncode}):\n{out.stderr}"
        )
    txt = out.stdout
    # cachesim writes log lines to stdout BEFORE the JSON [...] block
    m = re.search(r"(\[\s*\{.*\}\s*\])", txt, re.DOTALL)
    if not m:
        raise RuntimeError(f"no JSON in cachesim output:\n{txt[:500]}")
    return json.loads(m.group(1))


def evaluate(fake: str, real: str, sizes: str = DEFAULT_SIZES,
             policies: str = DEFAULT_POLICIES) -> dict:
    """Run cachesim on (fake, real) at matched sizes; return aligned report."""
    binary = _find_cachesim()
    fake_runs = _run_cachesim(binary, fake, sizes, policies)
    real_runs = _run_cachesim(binary, real, sizes, policies)

    # cachesim flattens to one entry per (policy, cache_size). Group by policy.
    def group(runs: list[dict]) -> dict[str, list[tuple[int, float]]]:
        out: dict[str, list[tuple[int, float]]] = {}
        for r in runs:
            pol = r["policy"]
            for c in r["per_cache_size"]:
                out.setdefault(pol, []).append((c["size"], c["miss_ratio"]))
        for pol, rows in out.items():
            rows.sort()
        return out

    fake_by = group(fake_runs)
    real_by = group(real_runs)

    by_policy: dict[str, dict] = {}
    for pol in fake_by:
        f_pairs = fake_by[pol]
        r_pairs = real_by.get(pol, [])
        f_mr = [mr for _, mr in f_pairs]
        r_mr = [mr for _, mr in r_pairs]
        deltas = [fm - rm for fm, rm in zip(f_mr, r_mr)]
        hrc_mae = sum(abs(d) for d in deltas) / max(len(deltas), 1)
        by_policy[pol] = {
            "fake_miss_ratio": f_mr,
            "real_miss_ratio": r_mr,
            "delta": deltas,
            "hrc_mae": hrc_mae,
        }

    cap_list = [int(s) for s in sizes.split(",")]
    mean_hrc_mae = sum(p["hrc_mae"] for p in by_policy.values()) / len(by_policy)
    return {
        "cache_sizes": cap_list,
        "policies": list(by_policy.keys()),
        "by_policy": by_policy,
        "mean_hrc_mae": mean_hrc_mae,
    }


def print_report(report: dict) -> None:
    """Print REAL / FAKE / Δ columns per cap so the absolute miss-ratios
    (not just deltas) are visible. Matches Darrell's PEER-REVIEW.md format."""
    sizes = report["cache_sizes"]
    # Header: per cap show "REAL FAKE Δ" triplet, 22 chars wide each.
    cap_header = "  ".join(f"{s:^21}" for s in sizes)
    triplet_header = "  ".join(f"{'REAL':>6} {'FAKE':>6} {'Δ':>6}" for _ in sizes)
    print(f"{'cap:':>16}   {cap_header}")
    print(f"{'policy':<8} {'HRC-MAE':>7} | {triplet_header}")
    for pol, p in report["by_policy"].items():
        cells = []
        for r, f, d in zip(p["real_miss_ratio"], p["fake_miss_ratio"], p["delta"]):
            cells.append(f"{r:6.4f} {f:6.4f} {d:+6.4f}")
        print(f"{pol:<8} {p['hrc_mae']:>7.4f} | {'  '.join(cells)}")
    print()
    print(f"mean HRC-MAE across policies: {report['mean_hrc_mae']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="LLNL cachesim eval")
    ap.add_argument("--fake", required=True, help="Synthetic CSV/zst trace")
    ap.add_argument("--real", required=True, help="Real reference CSV/zst")
    ap.add_argument("--cache-sizes", default=DEFAULT_SIZES,
                    help=f"Comma-separated cache sizes (default {DEFAULT_SIZES})")
    ap.add_argument("--policies", default=DEFAULT_POLICIES,
                    help=f"Comma-separated policies (default {DEFAULT_POLICIES})")
    ap.add_argument("--out", default="-", help="JSON report output (- = stdout)")
    args = ap.parse_args()

    report = evaluate(args.fake, args.real, args.cache_sizes, args.policies)
    print_report(report)

    if args.out != "-":
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nReport JSON: {args.out}")


if __name__ == "__main__":
    main()
