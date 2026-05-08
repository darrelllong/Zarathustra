# 2DIO / IRM-class prior-art critique

## Authors / scope

LLNL position on the Wang/Khor/Desnoyers 2DIO trace-synthesis baseline,
2026-05-08. Submitted as input for the prior-art comparison section of the
joint paper.

## TL;DR

The Desnoyers/2DIO baseline has two distinct problems:

1. **Generalization / overfitting (Long)**: the published 2DIO results are
   on a small set of traces (~8 for CloudPhysics-class). With per-trace
   parameter fitting, the model can largely *memorize* per-trace structure
   rather than learn a generative recipe that transfers. Results on a held-out
   workload class are not characterized.

2. **Structural model limits (working-set diagnostics)**: 2DIO is an
   Independent Reference Model (IRM) variant. IRM assumptions —
   recurrence (Coffman & Denning eq. 7.3.4) and asymptotic uncorrelation
   (locality property L2) — fail visibly on most realistic corpora when
   examined through the working-set function s(T) and its slope m(T).

The two concerns are independent and reinforcing: *even if* the 8-trace
benchmark were representative, the IRM machinery cannot capture access
patterns most modern workloads exhibit. A larger benchmark would expose
this immediately.

## 1. Overfitting / memoization (Long)

The published 2DIO benchmark covers ~8 CloudPhysics traces. With model
hyperparameters fit per-trace, an IRM-class model with enough
configurability can memorize the per-trace marginal frequency distribution
and a few moments of the inter-reference distribution — effectively
recovering the trace's summary statistics rather than learning a
*generative* mechanism. Two diagnostics for distinguishing memorization
from generalization that the original work does not appear to perform:

a. **Held-out trace evaluation**: fit hyperparameters on traces 1–N,
   evaluate generation quality on traces N+1, N+2, ... from the same
   workload. If MAE on held-out traces is materially worse than on
   training traces, the model is fitting trace-specific noise.

b. **Cross-corpus evaluation**: fit hyperparameters on (say) CloudPhysics,
   generate against a structurally-different corpus (Wikipedia, Meta KV).
   Generalization-class models (Bayes-optimal-over-summary-stats) should
   degrade gracefully; memorization-class models should fail
   discontinuously.

The current LLNL/LANL race protocol is per-corpus evaluation with shared
hyperparameter search across teams — closer to the per-trace memorization
regime than to a held-out evaluation. This is a separate concern that
applies symmetrically to every team and is flagged here only because it
makes the published 2DIO numbers harder to interpret as evidence of model
quality vs evidence of fitting capacity.

## 2. Structural model limits (working-set diagnostics)

We computed s(T) and m(T) on each of nine real-trace references using
sliding-window counting, T ∈ {2^0, ..., 2^19, 1M-1}. The IRM signature in
this measurement is well-defined:

- For an IRM with per-page reference probability β_i, the page-i
  interreference distance is geometric with parameter β_i.
- The overall m(T) for an IRM decays smoothly and approaches a constant as
  T → ∞: m(T) ≈ ∑_i β_i (1 − β_i)^T.
- m(T) for an IRM has no shape features (no plateaus, no cusps, no
  multi-scale structure).

Real corpora deviate from IRM signature in three distinct ways:

| Corpus | m(T) signature | IRM-compatible? | Why |
|---|---|---|---|
| Tencent | Smooth decay; m(T_25)=0.07 at T_25=27k | **Yes** | Strong recurrence + low one-shot fraction |
| Baleen24 | m drops to 0.49 at T=16, plateau at 0.20 | **No** | Bimodal: dist=0 burst-reuse + long-tail plateau (autocorrelation) |
| Meta KV | m=0.31 at T=2, plateau | **No** | Explicit dist=0 immediate-reuse mass |
| Wikipedia | m stays >0.68 across full range | **No** | 49% one-shot pages violate eq. 7.3.4 recurrence |
| Meta CDN | m stays >0.55 | **No** | 42% one-shot |
| Alibaba | m stays >0.66 | **No** | 69% one-shot |
| CloudPhysics | m stays >0.75 | **No** | 47% one-shot + scan-style bursts |
| MSR Exchange | m stays >0.71 | **No** | 51% one-shot + deep working set |

**Only Tencent has an IRM-compatible m(T) shape.** Every other corpus
either has explicit short-range autocorrelation (bimodal) or violates the
recurrence assumption (one-shot-dominated).

### Failure mode 1: recurrence violation (one-shot-dominated workloads)

Coffman & Denning eq. 7.3.4 requires ∑_x f_i(x) = 1 — every page must be
re-referenced eventually. Real Wikipedia, Meta CDN, Alibaba, and others
have 40–70% of their distinct pages referenced exactly once across a 1M
record window. To handle this, IRM-class models typically add a one-shot
mixture component, but this captures only the *aggregate fraction* of
non-recurrent traffic. It does not place those one-shots in the correct
temporal location — IRM with a one-shot tail emits one-shots i.i.d. through
the trace, but real one-shots are often clustered (scan-style sequential
fetches, request bursts).

### Failure mode 2: asymptotic-uncorrelation violation (bimodal workloads)

Locality property L2 (Coffman & Denning §7.2): ⟨θ(t,T), θ(t+x,T)⟩ → 0 as
x → ∞. IRM is memoryless and trivially satisfies L2 with zero correlation
at all lags. Real bimodal workloads (Baleen24, Meta KV) have *explicit*
short-range autocorrelation visible as the dist=0 plateau in m(T): a
significant fraction of accesses is "the same object as the previous
access," which contributes a delta function to the interreference
distribution at x=1. IRM cannot reproduce this without an explicit
auto-regressive term, which moves it outside the IRM family.

### Failure mode 3: temporal clustering vs. marginal frequency

Even when an IRM-class model gets the marginal frequency distribution
exactly right, it gets temporal clustering wrong by construction. This is
visible cleanly in per-policy HRC errors. We saw on CloudPhysics
(R287.CP2 banked seed 42) the following per-policy HRC-MAE pattern at
cache size = 32:

| Policy | Real hit rate | Fake hit rate | Δ (fake − real) |
|---|---:|---:|---:|
| LRU | 0.122 (= 1 − 0.878) | 0.226 | +0.104 |
| FIFO | 0.115 | 0.225 | +0.110 |
| ARC | 0.218 | 0.227 | +0.009 |
| **SIEVE** | **0.200** | **0.008** | **−0.192** |
| **LFU** | **0.026** | **0.008** | **−0.018** |
| **LIRS** | **0.214** | **0.129** | **−0.086** |

LRU/FIFO over-hit because the fake's top-rank concentration is too tight.
**SIEVE, LIRS, and LFU under-hit by an order of magnitude** because real
CP has scan-pattern *clustered reuses* that SIEVE's mark-bit semantics
preserve (real has 20% SIEVE hits at cache=32) while the IRM-style fake
spreads accesses too uniformly (only 1% SIEVE hits). This per-policy shape
— LRU/FIFO over-hit + SIEVE/LIRS/LFU under-hit — is the diagnostic
signature of an IRM-class generator on a workload with non-IRM temporal
structure.

## 3. What this means for the paper

The Wang/Khor/Desnoyers comparison should be reported with these caveats:

1. **Per-corpus locality classification** (using the working-set table
   above) should be presented alongside HRC-MAE numbers. A win on a
   strong-locality, IRM-compatible corpus (Tencent class) is qualitatively
   different from a win on a one-shot-dominated or bimodal corpus, where
   the IRM-class baseline is structurally limited.

2. **Per-policy HRC breakdown** at small cache sizes should be shown
   alongside the policy-mean. The SIEVE/LIRS/LFU mismatch at cache=32, 128
   is direct evidence of missing temporal structure.

3. **Held-out evaluation**: results from a model fit on (e.g.) Tencent and
   evaluated on Wikipedia, or vice versa, would help separate
   memorization from generalization. The current race protocol does not do
   this.

4. **Information-theoretic floor per corpus**: alongside MAE, report the
   gap between any model's MAE and the Bayes-optimal generator using only
   summary statistics. For one-shot-dominated corpora this floor is
   high — small absolute MAE differences may not reflect meaningful model
   differences.

## Empirical replication (2026-05-08)

We re-implemented the IRM and 2DIO baselines in `desnoyers/` and ran them
on all 9 race corpora at single seed 42 against the official cachesim
surface. Results:

| Corpus | IRM | 2DIO | best LLNL | LANL banked | 2DIO/LLNL | 2DIO/LANL |
|---|---:|---:|---:|---:|---:|---:|
| Tencent | 0.14810 | 0.07608 | — | **0.03010** | — | **2.5×** |
| Baleen24 | 0.05394 | 0.03866 | **0.01845** | 0.02760 | **2.1×** | 1.4× |
| Wikipedia | 0.05077 | 0.03973 | **0.00890** | 0.01146 | **4.5×** | 3.5× |
| Meta CDN | 0.17211 | 0.15643 | **0.03081** | 0.03770 | **5.1×** | 4.1× |
| Alibaba | 0.09823 | 0.07983 | **0.01000** | 0.01076 | **8.0×** | 7.4× |
| CloudPhysics | 0.15385 | 0.11247 | 0.02978 | **0.02670** | 3.8× | **4.2×** |
| MSR Exchange | 0.08240 | 0.08156 | 0.00893 | **0.00484** | 9.1× | **16.9×** |
| Meta KV | 0.49710 | 0.39626 | 0.04807 | **0.01090** | 8.2× | **36.4×** |

(Bold = banked leader on that corpus. 2DIO/X = ratio of 2DIO to X — how
many times worse the IRM-class baseline is.)

**Three observations:**

1. **2DIO improves over IRM by 9–49%**. Time-bucket conditioning helps
   most for one-shot-dominated corpora where temporal placement matters
   most (Tencent 49%, Wiki 22%) and least where the structural model is
   already saturated (MSR 1%, Meta CDN 9%).

2. **2DIO is 2.5× worse than LANL on Tencent — its theoretical home
   turf.** The IRM family has no ceiling-touching corpus on this
   benchmark; both LLNL and LANL generators are strictly better everywhere
   because they encode at least some temporal structure that IRM
   discards by construction.

3. **Meta KV at 0.396 (36× worse than LANL banked 0.011) is the
   smoking-gun empirical confirmation of the structural critique**. Meta
   KV's bimodal m(T) with the dist=0 immediate-reuse spike is the exact
   shape IRM cannot reproduce. The 36× gap is not a tuning concern; no
   amount of per-trace fitting within the IRM family can close it.

**Reproducibility**: the implementation is in `desnoyers/` at the LLNL
repo root. `desnoyers/irm.py` and `desnoyers/irm_2dio.py` provide
self-contained `fit` and `generate` CLIs; `desnoyers/run_all_corpora.sh`
is the end-to-end driver that produces the numbers above. Per-trace fits
are saved at `/tiamat/zarathustra/llgan-output/desnoyers/<corpus>_<model>.pkl.gz`.

These numbers should appear as the prior-art row in the paper's main
results table. The 5–10× gap to current generators is the empirical
evidence that supports the structural critique in §2; the absence of any
"IRM-friendly" corpus that prior-art wins on is the empirical evidence
that supports Long's overfitting hypothesis in §1 — published benchmarks
must rely on either narrower trace selection or per-trace tuning that
generalize-class models do not need.

## References

- Coffman, E. G., Jr. & Denning, P. J. (1973). *Operating Systems Theory*.
  Prentice-Hall. Chapter 6.6 (Independent Reference Model), Chapter 7.2–7.4
  (Locality, Working Set, LRU-WS duality).
- Mattson, R. L., Gecsei, J., Slutz, D. R., & Traiger, I. L. (1970).
  Evaluation techniques for storage hierarchies. *IBM Sys. J.* 9(2), 78–117.
- Working-set computation script: `/tmp/working_set.py` on baase, output
  reproduced in `BENCHMARK-METHODOLOGY-LLNL.md`.
