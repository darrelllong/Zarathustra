"""3-way cachesim panel: LLNL_FAKE vs LANL_FAKE vs REAL across 8 policies.

Builds an aligned table where each policy / cap shows REAL miss-ratio,
LLNL synth miss-ratio (Δ vs REAL), LANL synth miss-ratio (Δ vs REAL),
and the WINNER (whichever synth has smaller |Δ|).
"""
import argparse, json, re, subprocess, sys
from pathlib import Path

CS = "/home/darrell/Zarathustra/tools/cachesim/target/release/cachesim"
SIZES = "32,128,512,2048,8192,32768"
POLS = "lru,arc,fifo,sieve,slru,car,lfu,lirs"


def run(path: str) -> dict:
    out = subprocess.run(
        [CS, "--trace", path, "--policy", POLS, "--cache-sizes", SIZES, "--out", "-"],
        capture_output=True, text=True, check=False,
    )
    txt = out.stdout
    m = re.search(r"(\[\s*\{.*\}\s*\])", txt, re.DOTALL)
    if not m:
        raise RuntimeError(f"no JSON in cachesim output for {path}")
    data = json.loads(m.group(1))
    by = {}
    for r in data:
        for c in r["per_cache_size"]:
            by.setdefault(r["policy"], []).append((c["size"], c["miss_ratio"]))
    for pol in by:
        by[pol].sort()
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llnl", required=True, help="LLNL synthetic CSV")
    ap.add_argument("--lanl", required=True, help="LANL synthetic CSV")
    ap.add_argument("--real", required=True, help="Real reference CSV")
    args = ap.parse_args()

    print(f"\n{'-' * 130}")
    print(f"3-WAY CACHESIM PANEL  caps={SIZES}")
    print(f"  REAL: {Path(args.real).name}")
    print(f"  LLNL: {Path(args.llnl).name}")
    print(f"  LANL: {Path(args.lanl).name}")
    print(f"{'-' * 130}\n")

    real = run(args.real)
    llnl = run(args.llnl)
    lanl = run(args.lanl)

    sizes = [int(s) for s in SIZES.split(",")]
    pols = list(real.keys())

    # Per-policy summary table
    print(f"{'policy':<6} | {'LLNL HRC-MAE':>13}  {'LANL HRC-MAE':>13}  {'winner':>9}")
    print(f"{'-' * 6}-+-{'-' * 13}--{'-' * 13}--{'-' * 9}")
    llnl_total, lanl_total = 0.0, 0.0
    llnl_wins, lanl_wins = 0, 0
    for pol in pols:
        r_mr = [m for _, m in real[pol]]
        l_mr = [m for _, m in llnl[pol]]
        a_mr = [m for _, m in lanl[pol]]
        l_mae = sum(abs(l - r) for l, r in zip(l_mr, r_mr)) / len(r_mr)
        a_mae = sum(abs(a - r) for a, r in zip(a_mr, r_mr)) / len(r_mr)
        llnl_total += l_mae
        lanl_total += a_mae
        if l_mae < a_mae:
            winner = "LLNL"
            llnl_wins += 1
        elif a_mae < l_mae:
            winner = "LANL"
            lanl_wins += 1
        else:
            winner = "tie"
        print(f"{pol:<6} | {l_mae:>13.4f}  {a_mae:>13.4f}  {winner:>9}")
    print(f"{'-' * 6}-+-{'-' * 13}--{'-' * 13}--{'-' * 9}")
    print(f"{'mean':<6} | {llnl_total/len(pols):>13.4f}  {lanl_total/len(pols):>13.4f}  "
          f"{'LLNL ' + str(llnl_wins) + '/' + str(len(pols)):>9}")
    print(f"{'wins':<6} | {llnl_wins:>13}  {lanl_wins:>13}")

    # Full per-cap detail (compact format)
    print(f"\n{'-' * 130}\nPER-CAP MISS RATIOS (REAL = ground truth; Δ = synth − REAL)\n{'-' * 130}")
    for pol in pols:
        print(f"\n{pol.upper()}")
        print(f"  {'cap':>9}  {'REAL':>7}  {'LLNL':>7}  {'Δ_LLNL':>8}  {'LANL':>7}  {'Δ_LANL':>8}  {'better':>7}")
        for i, cap in enumerate(sizes):
            r = real[pol][i][1]
            l = llnl[pol][i][1]
            a = lanl[pol][i][1]
            d_l = l - r
            d_a = a - r
            better = "LLNL" if abs(d_l) < abs(d_a) else ("LANL" if abs(d_a) < abs(d_l) else "tie")
            print(f"  {cap:>9}  {r:>7.4f}  {l:>7.4f}  {d_l:>+8.4f}  {a:>7.4f}  {d_a:>+8.4f}  {better:>7}")


if __name__ == "__main__":
    main()
