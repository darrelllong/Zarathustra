# 2DIO prior-art positioning

## Authors / scope

LLNL race team, 2026-05-08 (revised after AD review). Position on the
Wang/Khor/Desnoyers 2DIO trace-synthesis baseline (EUROSYS 2026) for the
joint paper's prior-art comparison.

## Correction history

This document was originally written based on a misreading of the 2DIO
model. **The original claim that 2DIO is "an IRM variant"** — and the
attached structural critique built on IRM's memorylessness — **was wrong**.
2DIO Algorithm 2 (Gen-from-2D) is a **heap-based IRD-renewal generator**
with an IRM-arrival channel: it has explicit memory (the heap state)
and is in the same architecture family as LLNL R288 `ird_renewal.py` and
LANL altgan IRD-renewal. The retracted material is preserved in git
history (commits before `b94ad51`); below is the post-correction
position.

## What 2DIO is

Per `pubs/2DIO_CacheAccurate_2026.pdf` §3 and Algorithm 2 (p.6):

- A **min-heap** of `<next_due_time, item_id>` is initialized with M
  items, each with a sleep time drawn i.i.d. from an IRD distribution f.
  Items whose initial draw is t=∞ are not inserted (representing
  one-shots).
- Trace generation: at each step, with probability P_IRM emit an i.i.d.
  draw from item-frequency distribution g (Table 2: Zipf / Pareto /
  Normal / Uniform / Empirical); else pop the min-due item from the
  heap, emit it, and push it back with new sleep `t0 + t` for t ~ f.
  If t = ∞ (the explicit ∞-branch), a fresh item is created instead.
- f is parameterized as `fgen(k, I, ε)` per Eq. 3: a piecewise PMF over
  k bins with spike mass (1-ε)/|I| on each spike bin and ε/(k-|I|) on
  each non-spike bin. The IRD sample-space is auto-tuned (T_max chosen
  so the conditional mean of finite IRDs equals M).
- Per-trace "trace profile" θ = ⟨P_IRM, g, f⟩ — five-ish numbers per
  trace, fitted to reproduce a target LRU HRC.

## How 2DIO relates to the LLNL/LANL race generators

LLNL's `llgan/ird_renewal.py` and LANL's altgan IRD-renewal both
implement the same architecture family — heap-based IRD-renewal with an
IRM-channel — but with different parameterizations of f and g:

- **2DIO**: f is parsimonious (k ≈ 30-100 bins, |I| ≈ 1-4 spikes, scalar
  ε); g is one of five analytical families. Total parameter count: ~5.
- **LLNL R288 / LANL IRD-renewal**: f is the empirical IRD distribution
  of the target real trace (millions of values); g is implicit in
  `--independent-prob` driving heap-or-IRM-arrival without an analytical
  frequency distribution.

The race-relevant question for the paper is therefore **not** "does our
heap beat their lack-of-heap"; it is **"how much accuracy does the
empirical IRD distribution buy over the parsimonious 5-parameter
trace profile?"** That is a clean, fair empirical question. We have
not yet measured it (see Open Work below).

## Open methodology question (Long): per-trace fitting vs generalization

The 2DIO paper fits a separate θ per trace (Table 3). With ~5
hyperparameters per trace and a per-trace LRU HRC objective, this
is closer to per-trace memorization than to a corpus-level generative
model: the parameters do not transfer between traces. The paper does
not report:

- Held-out per-trace evaluation (fit on traces 1..N, evaluate on
  N+1..) or
- Cross-corpus evaluation (fit on CloudPhysics, evaluate on AliCloud,
  or vice versa).

LLNL/LANL race generators have the same property under the current
race protocol (per-corpus parameter search). This concern applies
symmetrically to all three teams; it is flagged here because if the
paper is to position 2DIO against LLGAN-class generators, *some* held-out
or cross-corpus evidence is needed to distinguish "model captures
generalizable structure" from "5 parameters can memorize a single
trace's HRC".

## What is *not* claimed here

- 2DIO is **not** "an IRM variant" — that mis-characterization, removed
  from the doc, was the basis of an empirical comparison whose code we
  also removed (`desnoyers/irm_2dio.py`, deleted in commit `b94ad51`).
- The SIEVE/LIRS/LFU per-policy mismatch we observed on the LLNL
  R287.CP2 banked synthetic is **not** evidence about 2DIO. It is
  evidence about *that LLNL synthetic*. Attributing it to "IRM-class
  signature" without measuring 2DIO Algorithm 2 directly is unsupported.
  Open work: measure per-policy HRC of an actual `desnoyers/two_dio.py`
  trace on CP and check whether the same shape appears.
- The asymptotic-uncorrelation (locality property L2) violation argument
  that appeared in earlier drafts applies to **vanilla IRM**, not to
  2DIO Algorithm 2 — which has heap-state memory and an explicit
  auto-regressive structure (a popped item's next access is scheduled
  at current_position + t for sampled t).
- Quantitative claims about 2DIO MAE on our 9 race corpora: pending.
  The faithful `desnoyers/two_dio.py` exists as of commit `7142189`,
  but has not been run on the per-corpus θ profiles yet. Numbers will
  appear here once measured.

## Open work (replaces the previous "Empirical replication" section)

1. Run `desnoyers/two_dio.py` with paper Table 3 profiles on the matching
   8 traces (w11/w24/w44/w82/v521/v538/v766/v827) at a tractable scale
   (M=10k, N=1M per the paper's own Section 5.3 — small-scale fits
   preserve HRC fidelity per their Figure 10), evaluate LRU HRC, and
   compare to the original-trace HRC if available.
2. Run the same with per-corpus θ profiles fit by us on each of our 9
   race corpora (parameter search over (P_IRM, g, fgen) to minimize
   LRU HRC-MAE), compare to LLNL R288 and LANL altgan IRD-renewal that
   use the empirical IRD distribution. This isolates the
   parsimony-vs-empirical-distribution question.
3. Measure per-policy HRC of 2DIO traces on the 6-policy race surface
   and check whether the SIEVE/LIRS/LFU mismatch we attributed to "IRM
   signature" actually appears for 2DIO.
4. Held-out evaluation: fit θ on (e.g.) v521 and evaluate on v538;
   report the cross-trace generalization gap. This is the test the
   paper does not perform.

## References

- `desnoyers/two_dio.py` (faithful Algorithm 2; commit `7142189`)
- `desnoyers/irm.py` (the paper's vanilla-IRM weak baseline,
  Figure 3 "IRM-recon")
- `desnoyers/run_8_traces.sh` (driver for paper Table 3 profiles)
- `pubs/2DIO_CacheAccurate_2026.pdf` (the paper itself)
- Coffman & Denning, *Operating Systems Theory* (1973), §6.6 (IRM),
  §7.3 (Working Set Model), §7.4 (LRU-WS duality)
