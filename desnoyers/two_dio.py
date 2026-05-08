"""Faithful re-implementation of Wang/Khor/Desnoyers 2DIO Algorithm 2
(Gen-from-2D), per the EUROSYS 2026 paper.

Algorithm 2 (paper, p.6):
    Input: P_IRM, g, f, M, N
    Output: π_s
    Heap ← ∅; a ← 0
    while Heap.size < M:
        t ~ f i.i.d.
        if t ≠ ∞:
            Heap.insert(<t, a>)
        a ← a + 1
    π_s ← []
    for j in 0..N-1:
        u ~ Uniform(0, 1)
        if u < P_IRM:
            addr ~ g i.i.d.; π_s.append(addr)
        else:
            t ~ f i.i.d.
            if t = ∞:
                π_s.append(a); a ← a + 1
            else:
                <t_0, a_0> ← Heap.pop()
                π_s.append(a_0)
                Heap.replace(<t_0 + t, a_0>)
    return π_s

f via fgen(k, I, ε)  — piecewise PMF on {1,...,k}:
    f(i) = (1-ε)/|I|  if i ∈ I    (spike)
    f(i) = ε/(k-|I|)  if i ∉ I    (hole)

IRD sample-space S = {1,...,T_max} where T_max auto-tuned so mean(IRD) = M:
    T_max = 2Mk / Σ_i (2i-1)·f(i)

Sampling t ~ f over S: pick bin index i ∈ {1..k} with prob f(i),
then sample uniformly from integer interval [(i-1)·T_max/k + 1, i·T_max/k].

g supports Table 2: Zipfian, Pareto, Normal, Uniform, Empirical.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import sys

import numpy as np


def fgen_pmf(k: int, spike_indices: list[int], eps: float) -> np.ndarray:
    """Eq. 3 of the paper: PMF over {0,1,...,k-1} (0-indexed for Python).
    spike_indices ⊂ {0..k-1}; spikes share (1-ε) mass equally; holes share ε.
    """
    pmf = np.zeros(k, dtype=np.float64)
    spike_set = set(int(i) for i in spike_indices)
    n_spikes = len(spike_set)
    n_holes = k - n_spikes
    if n_spikes > 0:
        spike_mass = (1.0 - eps) / n_spikes
        for i in spike_set:
            pmf[i] = spike_mass
    if n_holes > 0:
        hole_mass = eps / n_holes
        for i in range(k):
            if i not in spike_set:
                pmf[i] = hole_mass
    pmf /= pmf.sum()  # normalize defensively
    return pmf


def auto_tune_T_max(pmf: np.ndarray, M: int) -> int:
    """Solve T_max so mean of drawn IRDs = M, given bin midpoints
    b_i = (2i-1)/2 × T_max/k (Section 4.1)."""
    k = len(pmf)
    # M = sum_i b_i * f(i) = T_max/(2k) * sum_i (2i-1) * f(i)
    # T_max = 2*M*k / sum_i (2i-1) * f(i)
    weight = sum((2 * (i + 1) - 1) * pmf[i] for i in range(k))  # 1-indexed (2i-1)
    if weight <= 0:
        return max(M, 1)
    T_max = int(round(2 * M * k / weight))
    return max(T_max, k)


def sample_ird(pmf: np.ndarray, T_max: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n integer IRDs from the piecewise PMF over {1..T_max}.
    Pick bin i with prob pmf[i], then uniform in integer interval [lo, hi]."""
    k = len(pmf)
    bin_idx = rng.choice(k, size=n, replace=True, p=pmf)
    bin_width = T_max / k
    los = (bin_idx * bin_width).astype(np.int64) + 1
    his = ((bin_idx + 1) * bin_width).astype(np.int64)
    his = np.maximum(his, los)
    out = rng.integers(los, his + 1)
    return out.astype(np.int64)


def sample_g(g_spec: str, M: int, rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n addresses from item-frequency distribution g over {0..M-1}.
    g_spec format: 'zipf:alpha' | 'pareto:alpha,xm' | 'normal:mu,sigma'
                   | 'uniform' | 'none'."""
    if g_spec.lower() in ("none", ""):
        return np.zeros(n, dtype=np.int64)
    parts = g_spec.split(":")
    name = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    if name == "zipf":
        alpha = float(args) if args else 1.2
        # Zipfian PMF: g(i) = (1/i)^α / Z over i=1..M
        ranks = np.arange(1, M + 1, dtype=np.float64)
        pmf = ranks ** (-alpha)
        pmf /= pmf.sum()
        return rng.choice(M, size=n, replace=True, p=pmf)
    if name == "pareto":
        a, xm = (float(x) for x in args.split(","))
        # Pareto over discrete support {1..M}; PMF ∝ x_m^α / i^(α+1)
        ranks = np.arange(1, M + 1, dtype=np.float64)
        pmf = (xm ** a) / (ranks ** (a + 1))
        pmf /= pmf.sum()
        return rng.choice(M, size=n, replace=True, p=pmf)
    if name == "normal":
        mu, sigma = (float(x) for x in args.split(","))
        ranks = np.arange(M, dtype=np.float64)
        pmf = np.exp(-((ranks - mu) ** 2) / (2 * sigma ** 2))
        pmf /= pmf.sum()
        return rng.choice(M, size=n, replace=True, p=pmf)
    if name == "uniform":
        return rng.integers(0, M, size=n, dtype=np.int64)
    raise ValueError(f"unknown g spec: {g_spec}")


def gen_from_2d(p_irm: float, g_spec: str, fgen_args: tuple,
                M: int, N: int, seed: int) -> np.ndarray:
    """Run Algorithm 2 of the 2DIO paper.

    fgen_args: (k, spike_indices, eps) — as in paper Eq. 3
              OR None to disable IRD channel (set p_irm=1.0 in that case).
    """
    rng = np.random.default_rng(seed)

    # Build f and T_max.
    if fgen_args is not None:
        k, spike_indices, eps = fgen_args
        pmf = fgen_pmf(k, spike_indices, eps)
        T_max = auto_tune_T_max(pmf, M)
        print(f"[2dio] fgen k={k} spikes={spike_indices} eps={eps} → T_max={T_max:,}", flush=True)
    else:
        pmf, T_max = None, None

    # Heap initialization: M items each with sleep time t ~ f.
    heap: list = []
    a = 0
    if pmf is not None:
        ts_init = sample_ird(pmf, T_max, rng, M)
        for t in ts_init:
            heapq.heappush(heap, (int(t), a))
            a += 1
    else:
        # No IRD channel: still create M items (trivial, never popped except via IRM)
        for _ in range(M):
            a += 1

    # Trace generation.
    π_s = np.empty(N, dtype=np.uint64)
    if pmf is not None:
        ts_trace = sample_ird(pmf, T_max, rng, N)
    randoms = rng.random(N)
    g_addrs = sample_g(g_spec, M, rng, N) if p_irm > 0 else None

    cur_pos = 0
    for j in range(N):
        if randoms[j] < p_irm and g_addrs is not None:
            π_s[j] = int(g_addrs[j])
        else:
            if pmf is None:
                # Fallback to IRM if no f.
                π_s[j] = int(g_addrs[j]) if g_addrs is not None else 0
                continue
            t = int(ts_trace[j])
            # Pop minimum next-due item.
            t0, a0 = heapq.heappop(heap)
            π_s[j] = a0
            heapq.heappush(heap, (t0 + t, a0))
        cur_pos += 1

    return π_s


def write_csv(path: str, π_s: np.ndarray, n_streams: int = 1):
    """Write trace as standard 6-column CSV (stream_id, ts, obj_id, obj_size, opcode, tenant)."""
    n = len(π_s)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(n):
            # 2DIO emits abstract block addresses; assign uniform timestamps and a default size.
            w.writerow([0, float(i), int(π_s[i]), 4096, 0, 0])


def parse_fgen_arg(s: str) -> tuple:
    """Parse fgen argument: 'k:I:eps' where I is comma-separated 0-indexed bin indices.
    Example: '30:9,13,17,19:5e-3'  → fgen(k=30, I={9,13,17,19}, ε=5e-3)
             'none' or '' → None
    """
    if not s or s.lower() == "none":
        return None
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("fgen format is k:i1,i2,...:eps")
    k = int(parts[0])
    I = [int(x) for x in parts[1].split(",")] if parts[1] else []
    eps = float(parts[2])
    return (k, I, eps)


def main():
    p = argparse.ArgumentParser(description="Faithful 2DIO Algorithm 2 (Gen-from-2D)")
    p.add_argument("--p-irm", type=float, required=True,
                   help="P_IRM probability; if 1.0 use only IRM channel; if 0 use only IRD")
    p.add_argument("--g", default="none",
                   help="Item-freq distribution: zipf:1.2 | pareto:2.5,1 | normal:M/2,M/6 | uniform | none")
    p.add_argument("--f", required=True,
                   help="IRD distribution: 'k:I:eps' fgen-spec, or 'none' to disable IRD channel")
    p.add_argument("-m", "--M", type=int, required=True, help="trace footprint")
    p.add_argument("-n", "--N", type=int, required=True, help="trace length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    fgen_args = parse_fgen_arg(args.f)
    print(f"[2dio] generating M={args.M:,} N={args.N:,} P_IRM={args.p_irm} g={args.g} f={args.f}",
          flush=True)
    π_s = gen_from_2d(args.p_irm, args.g, fgen_args, args.M, args.N, args.seed)
    write_csv(args.output, π_s)
    print(f"[2dio] wrote {len(π_s):,} records → {args.output}", flush=True)


if __name__ == "__main__":
    main()
