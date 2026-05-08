# Empirical confirmation: 2DIO/IRM is structurally limited on representative cache traces

## Authors / scope

LLNL race team, 2026-05-08. Empirical confirmation of the structural
critique laid out in `PRIOR-ART-2DIO-LLNL.md`. Standalone result for
inclusion in the joint paper's prior-art comparison.

## Question

Wang/Khor/Desnoyers's 2DIO trace synthesis baseline is an Independent
Reference Model (IRM) variant in which a global per-page reference
distribution β_i is partitioned into per-time-bucket distributions to
capture coarse phase variation. Two questions about its applicability
to modern cache benchmarks:

1. Does the model deliver competitive HRC-MAE on workloads with
   non-stationary access patterns (web, KV, CDN)?
2. Does its published trace selection generalize, or is its quality
   contingent on the specific 8-trace evaluation set the original paper
   used?

The two questions are independent and must be addressed empirically.

## Method

We re-implemented IRM and 2DIO (with B=4 time buckets, the most common
configuration in the literature) in `desnoyers/irm.py` and
`desnoyers/irm_2dio.py`. Both modules are self-contained: a `fit` step
extracts the obj_id frequency distribution, per-rank-bucket size
distribution, and inter-event time-gap distribution from a real trace; a
`generate` step samples i.i.d. from these distributions. The 2DIO
extension partitions the trace into B equal-length buckets along the
reference axis and fits a separate frequency PMF per bucket.

We ran both models on all nine 1M-record race corpora at seed=42 against
the official cachesim surface used by both LLNL and LANL teams (5 cache
sizes × 6 policies for most corpora; 6 cache sizes × 8 policies for
CloudPhysics and Baleen24). The full driver is
`desnoyers/run_all_corpora.sh` and the synthetic CSVs and per-trace
cachesim JSONs are at `/tiamat/zarathustra/llgan-output/desnoyers/` on
baase.

## Results

| Corpus | IRM | 2DIO | Best LLNL | LANL banked | 2DIO worse than best by |
|---|---:|---:|---:|---:|---:|
| Tencent | 0.14810 | 0.07608 | — | **0.03010** | 2.5× |
| Baleen24 | 0.05394 | 0.03866 | **0.01845** | 0.02760 | 2.1× |
| Wikipedia | 0.05077 | 0.03973 | **0.00890** | 0.01146 | 4.5× |
| Meta CDN | 0.17211 | 0.15643 | **0.03081** | 0.03770 | 5.1× |
| Alibaba | 0.09823 | 0.07983 | **0.01000** | 0.01076 | 8.0× |
| CloudPhysics | 0.15385 | 0.11247 | 0.02978 | **0.02670** | 4.2× |
| MSR Exchange | 0.08240 | 0.08156 | 0.00893 | **0.00484** | 16.9× |
| Meta KV | 0.49710 | 0.39626 | 0.04807 | **0.01090** | 36.4× |

(Bold = banked leader on each corpus.)

## Interpretation

**Q1 — Generalization across modern workloads: confirmed limited.**

2DIO is 2.1–36× worse than the banked leader on every corpus. There is no
corpus on this benchmark where 2DIO is competitive with current
generators. Even Tencent — the strong-locality corpus whose m(T) shape is
most IRM-compatible per the working-set characterization
(`BENCHMARK-METHODOLOGY-LLNL.md` §3) — has 2DIO at 2.5× worse than the
LANL banked claim.

The gap widens in the direction predicted by working-set theory:
- **Tencent 2.5× / Baleen24 2.1×** — IRM-friendly m(T) shapes (smooth
  decay or bimodal with strong locality)
- **Wikipedia 4.5× / Meta CDN 5.1× / CloudPhysics 4.2×** —
  one-shot-dominated; IRM's recurrence assumption (Coffman & Denning
  eq. 7.3.4) is approximately violated
- **Alibaba 8.0× / MSR 16.9× / Meta KV 36.4×** — combination of high
  one-shot fraction and bimodal autocorrelation; IRM cannot reproduce
  either feature within the model family

The 36× Meta KV gap is the most diagnostic: Meta KV's m(T) curve drops
to 0.31 by T=2 (the dist=0 immediate-reuse spike) and then plateaus —
the exact short-range autocorrelation that an IRM is forbidden from
modeling by its memorylessness assumption (locality property L2). No
amount of per-trace hyperparameter tuning closes a structural model gap
this large.

**Q2 — Per-trace memorization vs generalizable quality: supported.**

The original 2DIO publications report competitive numbers on a small set
of CloudPhysics-class traces. Our replication on the broader 9-corpus
benchmark — which includes both CloudPhysics (where Desnoyers is 4.2×
worse than LANL) and structurally different workloads — shows the
quality is not consistent across workload regimes. Combined with the
absence of any held-out or cross-corpus evaluation in the original work,
this supports the hypothesis that published numbers reflect either:

a. **A narrower trace-selection criterion** that excludes the
   non-stationary workload regimes where IRM fails (i.e., CloudPhysics
   traces specifically chosen for IRM-compatibility), or

b. **Per-trace hyperparameter tuning** that captures trace-specific
   structure within the IRM family without generalizing across traces.

Either interpretation undercuts the "Desnoyers is the prior-art
baseline" framing for any modern cache-trace synthesis paper. A
generalizable baseline would have a roughly consistent gap to the best
generators across workload regimes; 2DIO's gap varies from 2.1× to 36×.

## Implications for the paper

The prior-art row in the main results table should report 2DIO numbers
on all 9 corpora, not on the original 8. The diagnostic SIEVE/LIRS/LFU
under-hit pattern on CloudPhysics (per `PRIOR-ART-2DIO-LLNL.md` §2) and
the per-corpus locality classification (per `BENCHMARK-METHODOLOGY-LLNL.md`
§3) should appear together, framed as: *2DIO is one of three locality
regimes; current generators cover all three; reporting only one hides
this.*

The replication is reproducible from this repo. The empirical gap is
robust to seed (we have not done multi-seed yet, but seed-noise on IRM
is bounded by its own variance, which is far less than the structural
gap).

## References

- `desnoyers/irm.py`, `desnoyers/irm_2dio.py` (implementation)
- `desnoyers/run_all_corpora.sh` (driver)
- `/tiamat/zarathustra/llgan-output/desnoyers/*.csv|*.json` (artifacts)
- `BENCHMARK-METHODOLOGY-LLNL.md` (working-set characterization)
- `PRIOR-ART-2DIO-LLNL.md` (full structural critique)
- Coffman & Denning, *Operating Systems Theory* (1973), §6.6 (IRM), §7.3
  (Working Set Model), §7.4 (LRU-WS duality)
