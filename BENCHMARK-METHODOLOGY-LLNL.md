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

1. **Per-corpus cache-size ladders scaled to the real trace's stack-distance
   percentiles**, e.g., `[p1, p10, p25, p50, p75, p90, p99]` of the real LRU
   stack-distance distribution. This makes each cache size measure a
   *different structural slice*, eliminates the inclusion-redundancy, and
   covers the full meaningful range for each corpus regardless of its
   working-set scale.

2. **Cache sizes at production-realistic scales**: with 128 GB of host RAM
   available, individual cachesim runs at cache=10⁶ are trivially affordable
   (~64 MB working memory). The current upper bound of 8,192 (or 32,768) was
   set when 32-bit RAM constrained the simulator, not because it represents
   any real workload. A production CDN or KV cache holds 10⁶–10⁹ objects.

3. **Per-corpus prediction-floor reporting**: alongside the absolute mean
   HRC-MAE, report the gap to the Bayes-optimal generator over summary
   statistics (IRD distribution + rank PMF + frequency histogram). This
   reframes "LANL beats LLNL by 0.001" as "LANL closes 18% of the remaining
   structure-prediction gap that LLNL doesn't" — which is what the number
   actually means.

4. **Drop arithmetic mean across correlated cache sizes** in favor of the
   integral of |HRC_real(c) − HRC_fake(c)| over log-cache-size, normalized by
   the local dynamic range. This treats each cache size as a measurement of
   *additional* prediction quality in its discriminative band, not a duplicate
   of its neighbor.

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
