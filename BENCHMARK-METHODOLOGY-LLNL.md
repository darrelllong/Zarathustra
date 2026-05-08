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

## 1. The Mattson inclusion property makes adjacent cache sizes redundant

For LRU and other stack-property policies (Mattson, Gecsei, Slutz, Traiger
1970), the set of hits at cache size *n* is a strict subset of the set of hits
at cache size *n+1*. Concretely:

- A reference is an LRU hit at cache *c* iff its current LRU stack distance
  is < *c*.
- Therefore: **hit rate is monotonically non-decreasing** in cache size, with
  no policy-dependent inversion possible for stack-property policies.

A consequence: **cache=32 measures the hottest-32-object hit rate, which is a
strict structural property of the trace** — not random noise. A generator
that nails real's top-32 concentration wins at cache=32 *as a result of*
matching the head of the access-frequency distribution, which is a meaningful
structural feature.

However, the same property makes the cache-size ladder **highly correlated**:
the marginal information added by including cache=*c* on top of cache=2*c*
(say) is the difference in the cumulative-distribution-function of stack
distances over the interval [*c*, 2*c*]. For policies that obey strict
inclusion, this is bounded above by 1 minus the smaller-cache hit rate, and
in practice is small (<5% of HRC range) for most corpora.

**Implication**: arithmetic mean across `[32, 128, 512, 2048, 8192]` weights
five highly-correlated measurements equally. A generator that does well at
cache=32 typically does well at cache=128 (because most of the cache=128 hits
are *the same* hits as at cache=32, plus some); arithmetic averaging
therefore over-weights the small-cache regime.

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

## Empirical demonstration: Wikipedia seed 42, three claims

We ran the proposed full power-of-2 ladder (1, 2, 4, ..., 524288; 20 sizes;
6 policies lru/arc/fifo/sieve/slru/car) on three Wikipedia seed-42 traces
and compared per-policy mean HRC-MAE under the old 5-size surface vs the
new 20-size ladder:

| Trace | Old 5-size surface | New 20-size ladder | Ratio |
|---|---:|---:|---:|
| LLNL R288.W (banked, IRD-renewal s32) | 0.008925 | 0.0232 | 2.6× |
| LANL irdr s3200 (banked-class IRD-renewal) | 0.011229 | 0.0282 | 2.5× |
| LANL r328 chunksurf (audit-pending) | **0.005117** | 0.0234 | **4.6×** |

**Three findings:**

1. **The 43% gap between LANL r328 and LLNL R288.W on the old surface
   collapses to <1% (statistical tie) on the new ladder.** LANL r328's
   chunksurf cascade was over-fitting the small-cache regime that the old
   surface measures.

2. **At cache=524288 (saturation, equal to footprint+1 sample),** every
   generator has |Δ| ∈ [0.13, 0.18] HRC against real. The real Wikipedia
   trace at saturation hits 63% (47% miss); fakes hit only 49% (51% miss).
   This 14-point gap is the actual prediction failure mode, and it is
   **completely invisible to the current surface** which terminates at
   cache=8192.

3. **At cache=1, 2, 4, 8, the per-policy errors are 0.0001–0.005 across all
   three traces.** These sizes contribute almost no information because
   even adjacent-duplicate rate is matched well; including them in the
   ladder is for completeness, not discriminative power.

**Net implication for the paper**: claims like "LLNL beats LANL by 22%" or
"LANL beats LLNL by 43%" on Wikipedia depend critically on the cache-size
ladder choice. Under a footprint-spanning ladder, the three generators
tested are within ~20% of each other. The "race" is then less about which
team wins on the small-cache regime and more about which team's generator
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
