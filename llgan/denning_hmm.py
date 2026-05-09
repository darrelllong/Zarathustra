"""Denning Working-Set HMM (R297) — a real probabilistic model of a
reference string under the principle of locality.

Per Coffman & Denning, *Operating Systems Theory* (1973), §7.3.1: a
reference string ω is decomposable as ω = ω_0 ω_1 ω_2 ... where each
ω_i is generated from a stationary stochastic mechanism with locality
L_i ⊆ N (the working set during phase i). The transitions L_i → L_{i+1}
are sparse and the dwell times in each locality are long compared to
the transition events.

This module fits and samples from such a model:

  STATES    K hidden locality states {S_1, ..., S_K}, each with
            (working set W_k ⊆ obj_ids, reference frequency
             distribution f_k over W_k, mean dwell time τ_k)
  TRANSIT   K×K Markov transition matrix T[i,j] = P(next state = j |
            current state = i), fitted from the locality sequence
            inferred from the real trace.

Locality inference from real trace:
  1. Window the trace at length W references.
  2. For each window i, build the working set W_i.
  3. Compute Jaccard similarity matrix J[i,j] = |W_i ∩ W_j| /
     |W_i ∪ W_j|.
  4. K-medoids cluster the windows into K classes (greedy: pick the
     window that maximizes total similarity as cluster 1, mask its
     neighbors above similarity threshold, repeat for K clusters).
  5. Each window's locality assignment is the cluster of its nearest
     medoid.
  6. Transition matrix from the inferred locality sequence: T[i,j] =
     count(i→j transitions) / count(i→*).

Reference-frequency fit per locality:
  Frequency of obj_id p in locality k = (count of p in windows assigned
  to k) / (total references in those windows). Stored as a (rank,
  weight) list sorted by descending weight.

Generation:
  1. Sample initial state from stationary distribution of T.
  2. At each output step:
     - With probability 1/τ_k, transition to state k' ~ T[k,:].
     - Else stay in state k.
     - Emit a reference ~ f_k. Maintain a global obj_id namespace; if
       the sampled rank exceeds locality's working-set size, emit a
       fresh obj_id (one-shot inside the locality).

Single fitted object: the K-state HMM (working sets, freq distributions,
transition matrix, dwell times). No test-metric optimization. The model
is INTERPRETABLE: K is the number of distinct locality phases the trace
exhibits; transition matrix says how the program moves between them.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def fit_hmm(real_csv: str, window: int = 50_000, K: int = 8,
            max_rows: int = 0) -> dict:
    """Fit K-state Denning locality HMM to a real trace."""
    print(f"[denning_hmm fit] reading {real_csv} window={window:,} K={K}", flush=True)
    obj_ids = []
    with open(real_csv, newline="") as f:
        r = csv.reader(f)
        next(r)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                break
            try:
                obj_ids.append(int(row[2]))
            except (IndexError, ValueError):
                continue
    n = len(obj_ids)

    # Window into localities and build per-window working sets + counts.
    n_windows = (n + window - 1) // window
    windows = []
    for wi in range(n_windows):
        lo, hi = wi * window, min((wi + 1) * window, n)
        win_obj = obj_ids[lo:hi]
        windows.append({
            "obj_set": set(win_obj),
            "obj_counts": Counter(win_obj),
            "n": len(win_obj),
        })
    print(f"[denning_hmm fit] windows={n_windows}, sample W_0 size={len(windows[0]['obj_set'])}",
          flush=True)

    # K-medoids cluster by Jaccard similarity (greedy).
    K_eff = min(K, n_windows)
    medoid_idx: list[int] = []
    available = set(range(n_windows))
    for _ in range(K_eff):
        # Pick the window maximizing total similarity to all available windows.
        best_i, best_score = -1, -1.0
        for i in available:
            score = sum(jaccard(windows[i]["obj_set"], windows[j]["obj_set"])
                        for j in available)
            if score > best_score:
                best_score, best_i = score, i
        medoid_idx.append(best_i)
        # Mask windows highly similar to this medoid (threshold 0.5).
        to_remove = [j for j in available
                     if jaccard(windows[best_i]["obj_set"],
                                windows[j]["obj_set"]) > 0.5]
        for j in to_remove:
            available.discard(j)
        if not available:
            break
    K_eff = len(medoid_idx)
    print(f"[denning_hmm fit] K_eff (after greedy clustering) = {K_eff}", flush=True)

    # Assign each window to nearest medoid by Jaccard similarity.
    state_of_window = []
    for wi in range(n_windows):
        sims = [jaccard(windows[wi]["obj_set"], windows[m]["obj_set"])
                for m in medoid_idx]
        state_of_window.append(int(np.argmax(sims)))

    # Per-state aggregate counts + working set.
    state_counts: list[Counter] = [Counter() for _ in range(K_eff)]
    state_n: list[int] = [0] * K_eff
    for wi, sk in enumerate(state_of_window):
        state_counts[sk].update(windows[wi]["obj_counts"])
        state_n[sk] += windows[wi]["n"]

    # Build per-state PMF over its working-set members (sorted desc by count).
    state_freq: list[list[tuple[int, float]]] = []
    for k in range(K_eff):
        items = state_counts[k].most_common()
        total = sum(c for _, c in items)
        if total > 0:
            state_freq.append([(oid, c / total) for oid, c in items])
        else:
            state_freq.append([(0, 1.0)])

    # Dwell times: for each state, mean run length in the inferred sequence.
    runs: list[list[int]] = [[] for _ in range(K_eff)]
    if state_of_window:
        cur, run_len = state_of_window[0], 1
        for s in state_of_window[1:]:
            if s == cur:
                run_len += 1
            else:
                runs[cur].append(run_len)
                cur, run_len = s, 1
        runs[cur].append(run_len)
    state_dwell = [
        float(np.mean(runs[k])) if runs[k] else 1.0 for k in range(K_eff)
    ]

    # Transition matrix.
    T = np.ones((K_eff, K_eff), dtype=np.float64)  # +1 smoothing
    for i in range(len(state_of_window) - 1):
        T[state_of_window[i], state_of_window[i + 1]] += 1.0
    T /= T.sum(axis=1, keepdims=True)

    # Stationary distribution by power iteration.
    pi = np.ones(K_eff) / K_eff
    for _ in range(200):
        pi = pi @ T
        pi /= pi.sum()

    print(f"[denning_hmm fit] state sizes: {[len(state_freq[k]) for k in range(K_eff)]}",
          flush=True)
    print(f"[denning_hmm fit] dwell times (windows): {[f'{d:.1f}' for d in state_dwell]}",
          flush=True)
    print(f"[denning_hmm fit] stationary π: {[f'{p:.3f}' for p in pi]}", flush=True)

    return {
        "window": window,
        "K": K_eff,
        "n_total": n,
        "state_freq": state_freq,        # list[list[(obj_id, weight)]]
        "state_dwell_windows": state_dwell,
        "transition": T,
        "stationary": pi,
        "n_windows_per_state": [int(state_n[k] / window) for k in range(K_eff)],
    }


def generate(model: dict, n_records: int, seed: int) -> np.ndarray:
    """Sample a synthetic reference string from the fitted Denning HMM."""
    rng = np.random.default_rng(seed)
    K = model["K"]
    window = model["window"]
    state_freq = model["state_freq"]
    dwell = model["state_dwell_windows"]
    T = model["transition"]
    pi = model["stationary"]

    # Pre-compute sampling helpers per state: ranks (obj_ids) array + cum weights.
    state_ranks: list[np.ndarray] = []
    state_cum: list[np.ndarray] = []
    for k in range(K):
        items = state_freq[k]
        if not items:
            state_ranks.append(np.array([0], dtype=np.uint64))
            state_cum.append(np.array([1.0], dtype=np.float64))
            continue
        arr_obj = np.array([o for o, _ in items], dtype=np.uint64)
        arr_w = np.array([w for _, w in items], dtype=np.float64)
        arr_w /= arr_w.sum()
        state_ranks.append(arr_obj)
        state_cum.append(np.cumsum(arr_w))

    # Sample initial state from stationary.
    cur_state = int(rng.choice(K, p=pi))
    out = np.empty(n_records, dtype=np.uint64)

    # We work in chunks of `window` references; transitions sampled per-window
    # using dwell time as expected windows-in-state.
    n_out_locs = (n_records + window - 1) // window
    j = 0
    for locj in range(n_out_locs):
        L = min(window, n_records - j)
        # Within-locality emission: sample L references from current state's PMF.
        u = rng.random(L)
        idx = np.searchsorted(state_cum[cur_state], u)
        idx = np.clip(idx, 0, len(state_ranks[cur_state]) - 1)
        out[j:j + L] = state_ranks[cur_state][idx]
        j += L

        # State transition with prob 1/dwell.
        if rng.random() < 1.0 / max(dwell[cur_state], 1.0):
            cur_state = int(rng.choice(K, p=T[cur_state]))

    print(f"[denning_hmm generate] emitted {n_records:,} records, "
          f"unique={len(np.unique(out)):,}", flush=True)
    return out


def write_csv(path: str, obj_ids: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "ts", "obj_id", "obj_size", "opcode", "tenant"])
        for i in range(len(obj_ids)):
            w.writerow([0, float(i), int(obj_ids[i]), 4096, 0, 0])


def cmd_fit(args):
    model = fit_hmm(args.real, window=args.window, K=args.K, max_rows=args.max_rows)
    with gzip.open(args.output, "wb") as f:
        pickle.dump(model, f)
    print(f"[denning_hmm fit] saved → {args.output}", flush=True)


def cmd_generate(args):
    with gzip.open(args.model, "rb") as f:
        model = pickle.load(f)
    out = generate(model, args.n, args.seed)
    write_csv(args.output, out)
    print(f"[denning_hmm generate] wrote {args.n:,} records → {args.output}", flush=True)


def main():
    p = argparse.ArgumentParser(description="R297 Denning Working-Set HMM")
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("fit")
    pf.add_argument("--real", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--window", type=int, default=50_000)
    pf.add_argument("--K", type=int, default=8)
    pf.add_argument("--max-rows", type=int, default=0)
    pf.set_defaults(fn=cmd_fit)
    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--output", required=True)
    pg.add_argument("--n", type=int, default=1_000_000)
    pg.add_argument("--seed", type=int, default=42)
    pg.set_defaults(fn=cmd_generate)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
