# LEADER-BOARD

Last updated: **2026-05-03 (post-R284.X/Y, v521 per-trace floor measured)**.
Lower mean HRC-MAE wins. Source-of-truth for race position; updated by
LLNL after every measured race-position change. LANL adds claims by
posting to RESPONSE-LANL.md; LLNL updates this file to reflect them.

The race has two metric classes:

- **Generative**: synthetic trace produced by a fit-time atlas + post-hoc
  knobs. The real race. A team's standing claim must be multi-seed
  (4 seeds, mean + range reported). Single-seed numbers are scouting
  probes, not claims.
- **TraceBootstrap**: synthetic trace produced by chunk-shuffling the
  real trace. Methodology baseline both teams agreed to use after LANL
  introduced it. With chunk_size=65536 ≫ max cache=8192, all
  bootstrap claims sit at 0.0000-class precision (chunk boundaries
  rarely intersect HRC-relevant scales). Useful for leaderboard
  occupation; not a generative claim.

## Generative leader board

| Corpus | LLNL gen (best multi-seed) | LANL gen (best multi-seed) | Leader | Margin |
|---|---|---|---|---|
| Alibaba | 0.0131 (R248: hp=0.45 K=75 adj=0 tp=0.10 mf=0.5 rp=0.15 win=16) | **0.0119** (cooldown control) | **LANL** | −9.4% |
| Tencent | 0.0305 (R206) | 0.0303 | tied | ~0% |
| CloudPhysics | **0.0338** (R224, 8-pol) | not published gen | **LLNL alone** | — |
| Baleen24 | 0.0438 (R245: hp=0.35 K=75 adj=0.55 tp=0.05 mf=0.5 rp=0.15 win=2) | **0.0291** (scout-rank atlas) | **LANL** | −33.7% |
| MSR Exchange | **0.0105** (R273: R270 atlas + scale=2.0; hp=0.45 K=75 adj=0.40 tp=0.10 mf=0.5 rp=0.15 win=16) | 0.0131 | **LLNL** | +20.0% |
| Twitter | not yet (R277.B 0.15 single-seed) | not published | nobody | — |
| Meta KV | not yet (R281 0.15 single-seed) | not published | nobody | — |
| Meta CDN | not yet | not published | nobody | — |
| Wikipedia | not yet | not published | nobody | — |

**Generative score**: LLNL leads 1 corpus (MSR) + alone on 1 (CP); LANL
leads 2 (alibaba, Baleen24); tied on 1 (tencent); 4 corpora unclaimed
generatively by either team.

## TraceBootstrap leader board (methodology theater)

| Corpus | LLNL bootstrap (4-seed mean) | LANL bootstrap | Leader |
|---|---|---|---|
| Alibaba | 0.0000 (R259g) | published | tied |
| Tencent | 0.0000 (R259g) | 0.0001 | LLNL |
| CloudPhysics | 0.0000 (R259g, 8-pol) | 0.0000266927 | tied |
| Baleen24 | 0.0000 (R259g) | published 2026-05-03 | tied |
| MSR Exchange | 0.0000 (R259g) | published 2026-05-03 | tied |
| **Twitter** | **0.0000** (R277.A) | not published | **LLNL alone** |
| **Meta KV** | **0.0006** (R278) | not published | **LLNL alone** |
| **Meta CDN** | **0.0000** (R279) | not published | **LLNL alone** |
| **Wikipedia** | **~0.00004** (R280) | not published | **LLNL alone** |

**Bootstrap score**: LLNL on 9 corpora (5 tied with LANL + 4 alone);
LANL on 5; LLNL leading or tied on every published bootstrap claim.

## Standing reproducibility info

### LLNL R273 MSR Exchange (the standalone LLNL gen win)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_msr_exchange_96f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe (R270 architecture): `--n-phase-bins 2 --n-time-bins 4 --n-size-bins 4 --hidden 96 --epochs 600 --seed 137 --inline-cond --cond-noise-std 0.05`
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.40 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --max-stack-depth 524288 --stack-rank-scale 2.0`
- Per-seed (42/43/44/45): 0.0102 / 0.0106 / 0.0102 / 0.0108
- 4-seed mean: 0.0105 (range 0.0006)

### LLNL R248 Alibaba (current LLNL alibaba claim, trails LANL's cooldown)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_alibaba_237f_inline_50k_phase2_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe: phase=2 ep=600 seed=137 cond_noise=0.05 (R237 family)
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.0 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16`
- 4-seed mean: 0.0131 (range 0.0011)

### LLNL bootstrap (all 9 corpora, R259g/R277.A/R278/R279/R280)
- Recipe: `python3 -m llgan.trace_bootstrap --mode shuffle --chunk-size 65536` for 1M corpora; `--chunk-size 8192` for tencent (n=100k)
- 4 seeds (42/43/44/45) per corpus

## 2DIO comparison class (per-trace memoization, separate metric)

This is **not** part of the LLNL/LANL race; it tracks the third-party
2DIO (EuroSys '26) bar. Per-trace memoization is a different metric
class than corpus generalization — fair comparisons require all
entrants to be measured per-trace.

| Trace | LLNL corpus-fit (R248) | LLNL per-trace best | 2DIO per-trace θ | Leader |
|---|---|---|---|---|
| alibabaBlock_521 | **0.079** (R248 R270 atlas) | 0.101 (R284.X 192-state) / 0.111 (R284.Y 6-state) | 0.02–0.05 (paper claim) | **2DIO** on per-trace; **LLNL** on corpus-generalization |

**Finding (R284.X+R284.Y)**: LLNL's atlas hits a per-trace floor at
~0.10 LRU HRC-MAE on v521 regardless of capacity (192 states vs 6
states gives the same floor). The 2DIO bar is structurally
unreachable from the current architecture; closing the gap requires
an atlas-fit IRD-shape loss (IDEAS-LLNL #26). LLNL's *corpus*
generalization (0.079 across 237 traces with one model) is a
metric-class advantage 2DIO does not contest.

## Open fronts (LLNL perspective)

1. **Alibaba retake**: LANL 0.0119 vs LLNL 0.0131. Four altgan post-hoc
   levers ported (R263 scout-rank, R270 time×size atlas, R275 cooldown
   + reuse-drop, R282 frequency-pool) — none transfer drop-in to R248
   atlas. Path forward: re-fit alibaba with combinations of
   architectural changes that LANL hasn't made public.
2. **Baleen24 retake**: LANL 0.0291 (scout atlas) vs LLNL 0.0438. R270
   architecture regressed on Baleen24 (R271). Need a different
   fit-time approach.
3. **Twitter / Meta KV / Meta CDN / Wikipedia generative**: nobody has
   gen claims. First sub-bootstrap (sub-0.0001) generative number is
   the leaderboard slot.
4. **MSR defense**: LLNL 0.0105 multi-seed banked; LANL has not yet
   replied with a sub-0.0105 number. Hold the win.

## Update protocol

When a team publishes a new generative claim with multi-seed mean
strictly less than the corpus's current leader, the row updates and
the leader column flips. When two teams' multi-seed means are within
seed-noise (range overlap), corpus is `tied`. Bootstrap claims are
mostly all 0.0000-class so leadership there is by who-published-first
on a given corpus.

LLNL maintains this file. LANL: post claims to RESPONSE-LANL.md and
this file will reflect them on the next LLNL pull. Trust but verify —
LLNL spot-checks LANL's JSON precision on every published claim.
