# Benchmark methodology — open issues with the cachesim HRC-MAE surface

## Authors / scope
LLNL race observations, 2026-05-08. Submitted as input for the joint
methodology section of the paper.

## TL;DR

The race protocol uses an arithmetic mean of **per-policy, per-cache-size
hit-rate-curve mean-absolute-error (HRC-MAE)** across a fixed cache-size ladder
`[32, 128, 512, 2048, 8192]` (most corpora) or `[32, 128, 512, 2048, 8192,
32768]` (CloudPhysics, Baleen24) for 6 or 8 policies. The protocol is
**internally consistent** — both teams measure on identical surfaces with
identical eval code (`llgan/cachesim_eval.py`) and identical real-trace
references — but it has **three structural weaknesses that should be
acknowledged in any paper reporting these numbers**.

## 1. Mattson inclusion → redundancy applies to LRU only on the race surface

The Mattson-Gecsei-Slutz-Traiger 1970 stack property guarantees that for
**stack-property policies**, the set of hits at cache size *n* is a strict
subset of the set of hits at cache size *n+1*; equivalently the
hit-rate-curve is monotonically non-decreasing with no Belady-style
inversions.

The race surface uses six policies (lru, arc, fifo, sieve, slru, car).
Of these, **only LRU strictly satisfies the Mattson stack property.** ARC's
adaptive partition `p` admits configurations where a smaller cache holds
items a larger cache evicted (Megiddo & Modha do not claim strict
inclusion). FIFO has been known since Belady (1969) to violate the stack
property (the Belady anomaly). SIEVE, SLRU, and CAR are all FIFO/CLOCK-
derived and do not have the stack property either — `tools/cachesim/src/
policy/sieve.rs` confirms SIEVE is "FIFO with visited bit", a
non-stack policy.

**Implication for cache-size redundancy**:

- For **LRU**, adjacent cache sizes in the ladder `[32, 128, 512, 2048,
  8192]` are highly correlated (each successive size adds the hits in the
  stack-distance interval, bounded above by `1 − HR(smaller)`), so the
  per-cache-size HRC-MAE values for LRU are largely redundant under
  arithmetic averaging.
- For **ARC, FIFO, SIEVE, SLRU, CAR** (5 of 6 policies), no such
  guarantee holds. A generator can over-fit at cache=32 for SIEVE
  without over-fitting at cache=128, and vice versa. The redundancy
  argument for these policies is empirical-only and weaker.

So the stronger version of the redundancy critique is: the **LRU column**
of the per-policy × per-cache HRC-MAE matrix is highly redundant by
construction; the other five policy columns are not (they may exhibit
genuine cache-size discrimination). A protocol revision motivated by
"Mattson inclusion makes adjacent cache sizes redundant" addresses 1/6
of the metric, not 5/6.

## 1b. The cache=32 measurement is a structural signal, not noise

Earlier drafts framed cache=32 as "noise-dominated". This was wrong.
Under the LRU stack property, hits at cache=32 are exactly the trace's
hottest-32-object accesses — a strict structural property of the
access-frequency head, not a random alignment. A generator that wins at
cache=32 wins by reproducing that concentration. The legitimate concern
is not that cache=32 measures nothing meaningful, but that *for LRU
specifically* it measures a quantity that is largely subsumed by larger
cache sizes (per §1 above), so arithmetic averaging weighs it
disproportionately.

## 2. The fixed power-of-2 ladder is mismatched to most corpora

The race surface `[32...8192]` (or `...32768`) is constant across corpora.
But corpora differ enormously in working-set size and stack-distance
distribution. We computed the LRU stack-distance percentiles per corpus from
1M-record real references:

| Corpus | Footprint | One-shot % | p10 | p50 | p90 | p99 | Sizes ∈ [p10,p90] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Alibaba | 693,535 | 69.4% | 17 | 276 | 44,829 | 190,491 | 5/5 |
| Tencent (100k) | 38,507 | 38.5% | 10 | 60 | 174 | 1,103 | **2/5** |
| CloudPhysics | 469,369 | 46.9% | 13 | 93 | 23,429 | 140,831 | 5/5 |
| Baleen24 | 147,783 | 14.8% | 0 | 2 | 1,838 | 21,563 | **0/5** below p25 |
| MSR Exchange | 507,523 | 50.7% | 2 | 1,590 | 177,400 | 364,018 | 5/5 |
| Meta KV | 190,110 | 19.0% | 0 | 0 | 2,117 | 24,401 | 0/5 below p75 |
| Meta CDN | 417,390 | 41.7% | 0 | 22 | 417,389 | 417,389 | 0/5 below p75 |
| Wikipedia | 492,052 | 49.2% | 177 | 14,285 | 492,051 | 492,051 | **3/5** below p25 |

**Two distinct mismatches:**

a. **Cache sizes too BIG relative to working set**: Baleen24, Meta KV, and
   Tencent have most accesses concentrated at stack-distance < 200. Cache
   sizes 512, 2048, 8192 sit far above the meaningful-discrimination range and
   measure essentially "have we seen this object before?" (saturated near
   full hit). The metric loses discriminative power — most generator
   variation is invisible at these cache sizes for these corpora.

b. **Cache sizes too SMALL relative to working set**: Wikipedia, Meta CDN,
   and to a lesser extent MSR have meaningful structure extending to 100k–500k
   stack distances. Cache=8192 only probes the lower quartile of meaningful
   working-set behavior. The metric never measures whether the generator
   reproduces the long-tail recurrence structure that real traces exhibit at
   production-realistic cache sizes (multi-GB ≈ 100k–10M objects for typical
   workloads).

## 3. One-shot-dominated corpora have an information-theoretic prediction floor

Corpora with high one-shot-access fractions (Wikipedia 49%, Meta CDN 42%,
Alibaba 69%) cannot be matched in cachesim better than:

> hit_rate(c) ≤ P(reuse) × P(stack_dist < c | reuse)

For Wikipedia at cache=8192, with P(reuse)=0.51 and P(stack_dist<8192 | reuse)
≈ 0.25 (from the percentile table above), the *theoretical maximum* hit rate
is bounded near 13% even for a Bayes-optimal generator over summary statistics
of the real trace. Observed real hit rate is 43%, indicating real has
recurrent structure beyond what summary IRD/rank distributions express.
Matching real hit rate at this cache size requires the generator to encode
that structure — but the metric does not separate "predictable structure not
yet captured" from "fundamentally unpredictable one-shot variance."

## What a better protocol would look like

### Proposed cache-size rule (Long, 2026-05-08)

> **Cache sizes shall be powers of two from 1 up to the smallest power of
> two strictly greater than the trace's working-set footprint** (i.e.,
> `2^⌈log₂(N+1)⌉` where N = number of distinct objects in the trace).

Rationale:
- **Self-scaling** to each corpus — Wiki (footprint 492k) ends at 2^19 =
  524,288; Baleen24 (footprint 148k) ends at 2^18 = 262,144. No fixed ladder
  is appropriate across corpora with 13× working-set range.
- **Coverage**: spans from the trivial extreme (cache=1, measures only
  adjacent-duplicate rate) to the saturation extreme (cache > footprint,
  every reuse hits). Captures the full HRC dynamic range per corpus.
- **Deterministic / unambiguous**: no policy choices, no percentile
  thresholds, just `footprint` and `log₂`. Both teams can compute it
  identically from the real CSV.
- **Tractable on 128 GB hosts**: cache = 524,288 needs ~32 MB working memory
  per cachesim instance; ~20 cache sizes per corpus increase per-trace eval
  time roughly 4× over the current 5-size surface (still seconds per
  multi-policy run on a Rust simulator).

Per-corpus ladders under this rule (from observed footprints):

| Corpus | Footprint | Largest cache | Ladder length |
|---|---:|---:|---:|
| Alibaba | 693,535 | 2^20 = 1,048,576 | 21 |
| Tencent | 38,507 (100k) / TBD (1M) | 2^16 (or larger if footprint exceeds) | ~17 |
| CloudPhysics | 469,369 | 2^19 = 524,288 | 20 |
| Baleen24 | 147,783 | 2^18 = 262,144 | 19 |
| MSR Exchange | 507,523 | 2^19 = 524,288 | 20 |
| Meta KV | 190,110 | 2^18 = 262,144 | 19 |
| Meta CDN | 417,390 | 2^19 = 524,288 | 20 |
| Wikipedia | 492,052 | 2^19 = 524,288 | 20 |

### Other corrections

1. **Per-corpus prediction-floor reporting**: alongside the absolute mean
   HRC-MAE, report the gap to the Bayes-optimal generator over summary
   statistics (IRD distribution + rank PMF + frequency histogram). This
   reframes "LANL beats LLNL by 0.001" as "LANL closes 18% of the remaining
   structure-prediction gap that LLNL doesn't" — which is what the number
   actually means.

2. **Drop arithmetic mean across correlated cache sizes** in favor of the
   integral of |HRC_real(c) − HRC_fake(c)| over log-cache-size, normalized
   by the local dynamic range. This treats each cache size as a measurement
   of *additional* prediction quality in its discriminative band, not a
   duplicate of its neighbor. (Equivalent to a Riemann sum on log scale, which
   is the natural measure for power-of-two ladders.)

## Working-set characterization per corpus (Coffman & Denning 1973)

The "right" cache-size ladder per corpus is given directly by Denning's
working-set theory. Per Coffman & Denning, *Operating Systems Theory*
(1973), Chapter 7.3:

- **s(T)** = mean working-set size at window T = lim_t E[|W(t,T)|]
- **m(T)** = missing-page probability at window T = lim P[r_{t+1} ∉ W(t,T)]
- **s(T+1) − s(T) = m(T)** (eq. 7.3.16; m(T) is the slope of s(T) vs T)
- **F_p(LRU) ≅ m(T_p)** where **s(T_p) = p** (eq. 7.4.1; LRU-WS duality)

The duality means: cache size *p* and window size *T_p* such that *s(T_p) = p*
are conjugate variables. The LRU page-fault rate at memory size *p* equals
the working-set miss rate at the corresponding window. So the "natural"
cache-size ladder per corpus is the inverse of the corpus's s(T) curve at a
chosen set of windows.

We computed s(T) and m(T) on the real trace for each corpus (full 1M-record
sliding-window scan, T ∈ {2^0...2^19, 999999}). Per-corpus summary:

| Corpus | Footprint *n* | T_25 (s=n/4) | m(T_25) | T_50 (s=n/2) | Locality class |
|---|---:|---:|---:|---:|---|
| Tencent | 38,507 | **26,676** | 0.07 | 53,683 | Strong locality |
| Baleen24 | 147,783 | 199,272 | 0.20 | 410,327 | Bimodal (dominant immediate-reuse) |
| Meta KV | 190,110 | 237,560 | 0.20 | 509,662 | Bimodal (dominant immediate-reuse) |
| Meta CDN | 417,390 | 174,337 | 0.55 | 342,524 | One-shot-dominated |
| Wikipedia | 492,052 | 165,817 | 0.68 | 335,218 | One-shot-dominated |
| Alibaba | 693,535 | 214,453 | 0.66 | 424,357 | One-shot-dominated |
| CloudPhysics | 469,369 | 250,166 | 0.75 | 533,617 | One-shot-dominated |
| MSR Exchange | 507,523 | 192,396 | 0.71 | 431,444 | One-shot-dominated |

**Single per-trace metric**: **T_25** (the window size at which the working
set covers one quarter of the footprint) is the most informative scalar
descriptor. It varies smoothly across corpora, is finite for all of them
(unlike T_m=0.5, which is undefined for traces whose miss rate never drops
below 0.5), and corresponds via 7.4.1 directly to the cache size *n/4* —
which is in the discriminative range for every corpus.

**Three locality classes** are visible:

1. **Strong-locality** (Tencent only): T_25 ≪ trace length; the working set
   converges fast. IRD-renewal and stack-frequency models work well on these.
2. **Bimodal** (Baleen24, Meta KV): m(T) drops sharply at very small T (high
   immediate-reuse rate) and then plateaus. Generators that match the dist=0
   reuse rate plus the plateau height match cachesim well.
3. **One-shot-dominated** (Wiki, CDN, Alibaba, CP, MSR): m(T) stays
   high (>0.45) across the entire window range. These traces have an
   information-theoretic floor below which no generator can predict miss
   rate from summary stats alone.

**Implication for cache-size ladder design**: per the LRU-WS duality, choose
windows T_i and use the corresponding cache sizes s(T_i). For paper
clarity, log-spaced windows like {T_25, T_50, T_75, T_90, T_95} would yield
five cache sizes that each measure a *different structural slice* of the
trace — eliminating the inclusion-redundancy of the current power-of-2
ladder. Or, equivalent and simpler: powers of 2 from 1 to 2^⌈log₂(n+1)⌉
(Long's rule) — which under the LRU-WS duality corresponds to powers of 2
in window size, and gives a clean enumeration.

## Empirical demonstration: Wikipedia seed-42 single-seed comparison

**Caveat (added in revision)**: the table below uses **single-seed
(seed=42)** values, not the multi-seed banked claims, on three Wikipedia
synthetic traces. We did not have time to redo this with multi-seed at
the time of writing. LANL r328 is the team's **banked** Wikipedia claim
(4-seed mean 0.0054596500, range 0.0008116333; per `LEADER-BOARD.md`
line 34 and `RESPONSE-LANL.md`), not "audit-pending" as an earlier
draft of this doc stated.

| Trace (seed 42 only) | Old 5-size surface | New 20-size ladder | Ratio |
|---|---:|---:|---:|
| LLNL R288.W IRD-renewal s32 | 0.008925 | 0.0232 | 2.6× |
| LANL irdr s3200 (older variant) | 0.011229 | 0.0282 | 2.5× |
| LANL r328 chunksurf (banked seed-42 value) | 0.005117 | 0.0234 | 4.6× |

**Two observations stand:**

1. **At cache=524288 (saturation),** every generator under-predicts real
   Wikipedia hit-rate by 0.13–0.18 absolute. The real trace at saturation
   hits 63% (47% miss); fakes hit only 49% (51% miss). This 14-point gap
   is invisible to the current surface (which terminates at cache=8192)
   and is the actual prediction-failure mode none of the current
   generators address.

2. **At cache=1, 2, 4, 8,** per-policy errors across all three seed-42
   traces are 0.0001–0.005, well below seed-noise. These sizes add
   ladder length but minimal discriminative power.

**Withdrawn**: an earlier draft of this section claimed that "the 43%
gap between LANL r328 and LLNL R288.W on the old surface collapses to
<1% (statistical tie) on the new ladder." That single-seed observation
cannot decide a tie claim against the LANL banked 4-seed range
(0.0008116333) — which is itself ~4× larger than the seed-42 ratio I
quoted. The new-ladder values for the three traces are within ~20% of
each other on seed 42; whether that translates into a multi-seed tie
requires re-running the new-ladder computation with all four seeds for
both teams' traces.

**Net implication for the paper**: ranking claims on the current
surface depend on the cache-size ladder choice; under a footprint-spanning
ladder the small-cache discrimination that drives the current surface
attenuates. To make this rigorous we need multi-seed measurements on
the proposed ladder. Open work.

(The next paragraph continues the original "Net implication" prose
with that caveat in mind.)
captures real-trace structure at production-realistic cache scales — where
**all of us currently fail by similar amounts**.

Synthetic CSVs and per-trace cachesim JSON for this comparison are at
`/tmp/protoladder_*.json` on baase (full per-policy per-cache breakdowns).

## Why this matters for the paper

The race results as posted in `LEADER-BOARD.md` are **internally fair** under
the agreed protocol — both teams measure on identical surfaces. But two
generators ranked 17% apart on this metric may be ranked very differently on
a metric weighted toward production-realistic cache sizes or normalized for
prediction-floor differences. Any paper using these numbers should:

- Acknowledge the inclusion-induced redundancy at small caches.
- Either justify the chosen cache ladder per-corpus (using percentile
  evidence) or re-measure on a percentile-derived ladder.
- Report per-corpus theoretical-floor estimates so margins are interpretable
  as fraction-of-remaining-gap, not absolute MAE.

These are not race-tactic concerns; they are protocol-validity concerns that
hold regardless of which team wins under the current protocol.
