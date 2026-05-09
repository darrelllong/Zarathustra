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


_INF_IRD = -1  # Sentinel returned by sample_ird to indicate t=∞.


def fgen_pmf(k: int, spike_indices_zero_idx: list[int], eps: float) -> np.ndarray:
    """Eq. 3 of the paper: PMF over k bins.

    INDEXING CONVENTION: this function takes spike_indices in 0-indexed Python
    convention (∈ {0..k-1}). The paper uses 1-indexed convention (∈ {1..k}).
    The CLI parser (parse_fgen_arg) accepts paper-1-indexed input from users
    matching Table 3, and subtracts 1 internally. Internal code must treat
    spike_indices as 0-indexed.

    Per Eq. 3: spikes share (1-ε) mass equally; holes share ε equally.
    Returned PMF sums to 1 over {0..k-1}.
    """
    pmf = np.zeros(k, dtype=np.float64)
    spike_set = set(int(i) for i in spike_indices_zero_idx)
    if any(i < 0 or i >= k for i in spike_set):
        raise ValueError(f"spike index out of range for k={k}: {spike_set}")
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
    """Solve T_max so mean of finite IRDs = M, given bin midpoints
    b_i = (2(i+1)-1)/2 × T_max/k for 0-indexed i (Section 4.1).

    Note: this is conditional on t < ∞. The ∞-atom does not contribute
    to T_max calibration since it doesn't sample from the finite support.
    """
    k = len(pmf)
    weight = sum((2 * (i + 1) - 1) * pmf[i] for i in range(k))
    if weight <= 0:
        return max(M, 1)
    T_max = int(round(2 * M * k / weight))
    return max(T_max, k)


def sample_ird(pmf: np.ndarray, T_max: int, p_inf: float,
               rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample n IRDs from the 2DIO distribution: with probability p_inf
    return _INF_IRD (sentinel for t=∞); else sample from piecewise fgen
    PMF over integer support {1..T_max}.

    Algorithm 2 (paper p.6) explicitly branches on t=∞ in both initialization
    and trace-generation loops; this corresponds to one-shot accesses (objects
    referenced once and never re-accessed). p_inf is the per-draw probability
    of that branch; 0 reproduces the all-finite default fgen.
    """
    k = len(pmf)
    out = np.empty(n, dtype=np.int64)
    if p_inf > 0:
        inf_mask = rng.random(n) < p_inf
        n_finite = int((~inf_mask).sum())
        out[inf_mask] = _INF_IRD
    else:
        inf_mask = np.zeros(n, dtype=bool)
        n_finite = n
    if n_finite > 0:
        bin_idx = rng.choice(k, size=n_finite, replace=True, p=pmf)
        bin_width = T_max / k
        los = (bin_idx * bin_width).astype(np.int64) + 1
        his = ((bin_idx + 1) * bin_width).astype(np.int64)
        his = np.maximum(his, los)
        finite_samples = rng.integers(los, his + 1).astype(np.int64)
        out[~inf_mask] = finite_samples
    return out


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
                p_inf: float, M: int, N: int, seed: int) -> np.ndarray:
    """Faithful implementation of Algorithm 2 (Gen-from-2D) of the
    Wang/Khor/Desnoyers EUROSYS 2026 2DIO paper.

    fgen_args: (k, spike_indices_zero_idx, eps) per Eq. 3.
               None to disable IRD channel; in that case set p_irm=1.
    p_inf:     per-draw probability that t=∞ from the IRD channel. The
               paper's Algorithm 2 has explicit if-t=∞ branches in both
               heap initialization (skip insert, increment a) and
               trace-generation loops (emit fresh a, increment a). p_inf
               controls the one-shot rate of the synthetic.
    """
    rng = np.random.default_rng(seed)

    # Build f and T_max.
    if fgen_args is not None:
        k, spike_indices, eps = fgen_args
        pmf = fgen_pmf(k, spike_indices, eps)
        T_max = auto_tune_T_max(pmf, M)
        print(f"[2dio] fgen k={k} spikes(0-idx)={spike_indices} eps={eps} "
              f"p_inf={p_inf} → T_max={T_max:,}", flush=True)
    else:
        pmf, T_max = None, None

    # Heap initialization: paper Algorithm 2 lines 4-7. While Heap.size < M:
    #   t ~ f; if t ≠ ∞: Heap.insert(<t, a>); a ← a + 1   (always increment)
    # We may need more than M draws if some are ∞.
    heap: list = []
    a = 0
    if pmf is not None:
        # Pre-sample with a buffer; reserve more if ∞ rate is high.
        # Expected draws needed = M / (1 - p_inf). Use 2× safety margin.
        margin = 2.0 if p_inf > 0 else 1.0
        n_buffer = int(M * margin / max(1 - p_inf, 1e-9)) + M
        ts_init = sample_ird(pmf, T_max, p_inf, rng, n_buffer)
        idx = 0
        while len(heap) < M:
            if idx >= len(ts_init):
                # Top up if buffer ran out (rare; only if p_inf is very high).
                ts_init = np.concatenate([ts_init,
                                          sample_ird(pmf, T_max, p_inf, rng, M)])
            t = int(ts_init[idx])
            idx += 1
            if t != _INF_IRD:
                heapq.heappush(heap, (t, a))
            a += 1
    else:
        # No IRD channel: still create M items (only consumed by IRM channel).
        a = M

    # Trace generation: paper Algorithm 2 lines 9-19.
    π_s = np.empty(N, dtype=np.uint64)
    if pmf is not None:
        ts_trace = sample_ird(pmf, T_max, p_inf, rng, N)
    randoms = rng.random(N)
    g_addrs = sample_g(g_spec, M, rng, N) if p_irm > 0 else None

    for j in range(N):
        if randoms[j] < p_irm and g_addrs is not None:
            # IRM-arrival channel: addr ~ g; π_s.append(addr)
            π_s[j] = int(g_addrs[j])
            continue
        if pmf is None:
            # No IRD channel configured; fall back to IRM if present.
            π_s[j] = int(g_addrs[j]) if g_addrs is not None else 0
            continue
        t = int(ts_trace[j])
        if t == _INF_IRD:
            # Paper line 14-15: t = ∞ → emit fresh a, a ← a + 1.
            π_s[j] = a
            a += 1
        elif heap:
            # Paper line 17-19: pop min-due item, emit, push back with t0+t.
            t0, a0 = heapq.heappop(heap)
            π_s[j] = a0
            heapq.heappush(heap, (t0 + t, a0))
        else:
            # Heap empty (only if every init draw was ∞); emit fresh.
            π_s[j] = a
            a += 1
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
    """Parse fgen argument: 'k:I:eps' where I is comma-separated bin indices.

    INDEXING CONVENTION: I-values match the PAPER's TABLE 3 convention, which
    is 0-indexed despite the formal §3.3.1 definition stating support
    "i ∈ {1,2,...,k}". The Table 3 entries v766 fgen(40,[0,5],5.7e-3) and
    v827 fgen(40,[0,13],5e-3) contain 0, which is invalid under 1-indexed
    {1..k}. Hence Table 3 is 0-indexed (probably aligned with the C++
    trace-gen tool at github.com/Effygal/trace-gen). We follow Table 3.

    Pass the I-values verbatim from Table 3 — no offset.

    Examples (all from paper Table 3):
        '100:2:2e-3'           → v521  spike at bin 2  (0-indexed)
        '40:0,5:5.7e-3'        → v766  spikes at bins 0 and 5
        '60:0,13:5e-3'         → v827
        '30:9,13,17,19:5e-3'   → w24   spikes at bins 9,13,17,19
        'none' or ''           → None (disable IRD channel; set p_irm=1)
    """
    if not s or s.lower() == "none":
        return None
    parts = s.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("fgen format is k:i1,i2,...:eps "
                                         "(I in TABLE-3 0-INDEXED convention)")
    k = int(parts[0])
    I = [int(x) for x in parts[1].split(",")] if parts[1] else []
    eps = float(parts[2])
    if any(i < 0 or i >= k for i in I):
        raise argparse.ArgumentTypeError(
            f"Table-3 0-indexed I values must be in {{0..{k-1}}}: got {I}")
    return (k, I, eps)


def main():
    p = argparse.ArgumentParser(description="Faithful 2DIO Algorithm 2 (Gen-from-2D)")
    p.add_argument("--p-irm", type=float, required=True,
                   help="P_IRM probability; if 1.0 use only IRM channel; if 0 use only IRD")
    p.add_argument("--g", default="none",
                   help="Item-freq distribution: zipf:1.2 | pareto:2.5,1 | normal:M/2,M/6 | uniform | none")
    p.add_argument("--f", required=True,
                   help="IRD distribution: 'k:I:eps' fgen-spec (paper-1-indexed I), "
                        "or 'none' to disable IRD channel")
    p.add_argument("--p-inf", type=float, default=0.0,
                   help="Per-draw probability t=∞ from IRD channel (one-shot rate). "
                        "Algorithm 2 has explicit ∞ branches; the paper's fgen has "
                        "finite support so the t=∞ mass enters via this parameter.")
    p.add_argument("-m", "--M", type=int, required=True, help="trace footprint")
    p.add_argument("-n", "--N", type=int, required=True, help="trace length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    fgen_args = parse_fgen_arg(args.f)
    print(f"[2dio] generating M={args.M:,} N={args.N:,} P_IRM={args.p_irm} "
          f"g={args.g} f={args.f} p_inf={args.p_inf}", flush=True)
    π_s = gen_from_2d(args.p_irm, args.g, fgen_args, args.p_inf,
                      args.M, args.N, args.seed)
    write_csv(args.output, π_s)
    print(f"[2dio] wrote {len(π_s):,} records → {args.output}", flush=True)


if __name__ == "__main__":
    main()
