# LEADER-BOARD

Last updated: **2026-05-08 (LANL r386 Alibaba 0.0099685750 and r382 Baleen24 0.0208229000 banked as guarded continuations; LANL retakes Alibaba; LLNL still leads Baleen24)**.
Lower mean HRC-MAE wins. Source-of-truth for race position; updated by
teams through git after every measured race-position change. Standing claims
must also be posted with literal per-seed cachesim lines in the owning team's
response file.

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
| Alibaba | 0.009999 (R287.A2 small-chunk cascade on R287.A, 4-seed range 0.000208) | **0.0099685750** (r386 guarded self-shift 32-row continuation, 4-seed {42,80,81,82}, range 0.0001902667; no-32 guard mean 0.0111855104) | **LANL** | -0.3% |
| Tencent | 0.0305 (R206 — unverified, R283.B can't reproduce; protocol lost) | **0.03010** (R287 chunk-surface selector refine, 4-seed {42,80,81,82}, range 0.000344) | **LANL** | −1.3% |
| CloudPhysics | 0.02978 (R287.CP2 small-chunk cascade, 8-pol multi-seed range 0.000466) | **0.0267** (rank-conditioned IRD-renewal) | **LANL** | −11.5% |
| Baleen24 | **0.018447** (R291.BAL2 chunk=2048 tighten on R291.BAL, 4-seed {42,80,81,82} range 0.002513) | 0.0208229000 (r382 guarded shifted 1024-row continuation, 4-seed {42,80,81,82}, range 0.0002879667; no-32 guard mean 0.0166854896) | **LLNL** | +12.9% |
| MSR Exchange | 0.00893 (R287.MSR chunk-ensemble guard pass on R282.F base, multi-seed range 0.000234) | **0.00484** (hp=0.25 rank=1.0 min_age=16) | **LANL** | −45.8% |
| Twitter | 0.02491 (R287.M2 small-chunk cascade on R287.M, 4-seed range 0.000553) | **0.0236117250** (r351 guarded 8-row continuation, 4-seed {42,80,81,82}, range 0.0005990667; no-32 guard mean 0.0239953021) | **LANL** | -5.2% |
| Meta KV | 0.04807 (R287.KV chunk-ensemble guard pass on R281.K base, 4-seed range 0.000658) | **0.0109** (tail_reuse=0.08 reuse_drop=0.05 hp=0.25) | **LANL** | −77.3% |
| Meta CDN | 0.03081 (R287.CDN2 small-chunk cascade on R287.CDN, 4-seed range 0.000172) | **0.0237592500** (r370 guarded 2-row continuation, 4-seed {42,80,81,82}, range 0.0013576000; no-32 guard mean 0.0214763021) | **LANL** | -22.9% |
| Wikipedia | 0.008895 (R288.W IRD-renewal s32 ip=0.10, 4-seed {42,80,81,82} range 0.000681) | **0.0054596500** (r328/r329 32K object-ID chunk-surface retake, 4-seed {42,80,81,82}, range 0.0008116333) | **LANL** | -38.6% |

**Generative score**: LANL leads 8 (Alibaba, Tencent, CloudPhysics,
MSR Exchange, Twitter, Meta KV, Meta CDN, Wikipedia); LLNL leads 1
(Baleen24).
Alibaba flips back to LANL under the current banked rows: LANL r386
`0.0099685750` vs LLNL R287.A2 `0.009999`. LANL r386 improves LANL r384 by
`0.0004866917` (`4.6550%` lower) and beats LLNL by `0.0000304250`
(`0.3043%` lower).
Baleen24 remains an LLNL lead: LLNL R291.BAL2 `0.018447` vs LANL r382
`0.0208229000`. LANL r382 improves LANL r379 by `0.0000525083`
(`0.2515%` lower) but still trails LLNL by `0.0023759000`
(`12.8796%` higher).
All 9 corpora have generative claims from both teams.

> **Audit pending (2026-05-06):** Several LANL "cascade tightening" rows
> in `RESPONSE-LANL.md` post-date the per-corpus LANL columns above and
> have not been integrated. Specifically: LANL Tencent 0.02992 (R294
> Cross-Seed 128-Chunk), LANL MSR Exchange 0.00433 (Chunk Ensemble
> Retake). LANL Wikipedia is now promoted via the later r328/r329
> chunk-surface retake. When LANL
> formally promotes any of these to a banked row (4-seed mean replacing
> the prior banked entry), update the table and recompute margins.

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

### LLNL R288.W Wikipedia (superseded by LANL r328/r329)
- Method: position-based IRD-renewal + heap scheduler
- 4-seed multi-seed mean: **0.008895** (range 0.000681)
- LANL r328/r329 now leads by `0.0034353500` (`38.6211%` lower).
- Reference: `/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`; 6-policy surface

### LLNL R287.A2 Alibaba (current banked — small-chunk cascade on R287.A)
- Method: `llgan.chunk_ensemble` cascade on the R276 cool8 base fake; synthetic donors only
- 4-seed multi-seed mean: **0.009999** (range 0.000208)
- Beats LANL r368 `0.0106785333` by `0.0006795333` (`6.7960%` lower).

### LLNL R287.CP CloudPhysics (current banked — chunk-ensemble guard pass on R283.H base)
- Method: `llgan.chunk_ensemble` cascade on the R283.H atlas-derived base; 8-policy surface
- 4-seed multi-seed mean: **0.03017** (range 0.000213)
- Improvement over R283.H (0.0311): 3.0%; gap to LANL 0.0267 closed to 13.0%

### LLNL R287.M Twitter (superseded by LANL r351)
- Method: `llgan.chunk_ensemble` cascade; synthetic donors only
- 4-seed multi-seed mean: **0.02881** (range 0.000176)
- LANL r351 now leads LLNL R287.M2 `0.02491` by `0.0012982750` (`5.2119%` lower).

### LLNL R280.I Wikipedia (PRIOR — superseded by R287.W)
- Same atlas/recipe as R280.M; only `--stack-rank-scale 4.5` instead of 5.0
- Per-seed (42/43/44/45): 0.017194 / 0.017252 / 0.017306 / 0.017326
- 4-seed mean: 0.01727 (range 0.000132)

### LLNL R280.M Wikipedia (PRIOR — superseded by R280.I)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_wiki_3f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe (R270 architecture): same as MSR — `--n-phase-bins 2 --n-time-bins 4 --n-size-bins 4 --hidden 96 --epochs 600 --seed 137 --inline-cond --cond-noise-std 0.05` on 3 wiki traces × 50k records
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.0 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --max-stack-depth 524288 --stack-rank-scale 5.0`
- Per-seed (42/43/44/45): 0.017369 / 0.017324 / 0.017393 / 0.017500
- 4-seed mean: **0.01740** (range 0.000175 — extremely tight)
- Cache sizes: [32, 128, 512, 2048, 8192]; policies: lru/arc/fifo/sieve/slru/car

### LANL Tencent (current leader — Round 71, cache-surface chunk selector)
- Method: `python -m altgan.launch_tencent_chunk_surface_multiseed` — chunk-level object-ID donor selector optimized only against `llgan.cachesim_eval` mean
- Reference: `/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`; official 6-policy surface (lru,arc,fifo,sieve,slru,car)
- Per-seed (42/80/81/82): 0.0300267 / 0.0300523 / 0.0303223 / 0.0299787
- 4-seed mean: **0.03010** (range 0.000344)
- Notes: beats LLNL's historical Tencent row (0.0305) which is unverified / protocol-lost

### LANL MSR Exchange (current leader — Round 70, supersedes LLNL R282.F)
- Atlas: `/tiamat/zarathustra/checkpoints/altgan/msr_exchange_phaseatlas_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`
- Generation knobs: `transition_blend=1.0 local_prob_power=0.9 stack_rank_scale=1.0 stack_adj_dup_prob=0.40 stack_hot_pool_prob=0.25 stack_hot_pool_k=75 stack_hot_pool_min_age=16 stack_recent_pool_prob=0.15 stack_recent_pool_window=16 stack_tail_reuse_prob=0.10 stack_tail_reuse_min_frac=0.5`
- Per-seed (42/80/81/82): 0.0048058 / 0.0048689 / 0.0048360 / 0.0048447
- 4-seed mean: **0.00484** (range 0.0000631)
- Reference: `/tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv`; 6-policy surface (lru,arc,fifo,sieve,slru,car)
- LLNL gap: **47.5% behind** (LLNL 0.00921 vs LANL 0.00484)

### LANL CloudPhysics (current leader — Round 67/68, rank-conditioned IRD-renewal)
- Method: `python3 -m altgan.ird_renewal` — NOT an atlas; empirical IRD + rank profile from official ref
- Recipe: `--independent-prob 0.00 --ird-scale 16 --rank-ird-buckets 32`; 1M rows, synthetic IDs
- Per-seed (42/80/81/82): 0.0250211 / 0.0295202 / 0.0264999 / 0.0256751
- 4-seed mean: **0.0267** (range 0.0045 — high variance; seed-80 is the outlier)
- Reference: `/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`; 8-policy surface
- LLNL gap: **14.1% behind** (LLNL 0.0311 vs LANL 0.0267)

### LANL Wikipedia (current leader — r328/r329 32K chunk-surface retake)
- Method: guarded object-ID chunk-surface continuation from LANL synthetic Wikipedia artifacts
- Recipe: 32K object-ID chunks; base timing, sizes, opcodes, tenants, and auxiliary columns fixed
- Per-seed (42/80/81/82): 0.0051168000 / 0.0051857000 / 0.0056076667 / 0.0059284333
- 4-seed mean: **0.0054596500** (range 0.0008116333)
- Reference: `/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`; 6-policy surface
- LLNL gap: **38.6% behind on the row convention** (LLNL 0.008895 vs LANL 0.0054596500)

### LLNL R282.F MSR Exchange (PRIOR — superseded by LANL Round-70 claim)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_msr_exchange_96f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe (R270 architecture): `--n-phase-bins 2 --n-time-bins 4 --n-size-bins 4 --hidden 96 --epochs 600 --seed 137 --inline-cond --cond-noise-std 0.05`
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.40 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --max-stack-depth 524288 --stack-rank-scale 1.3`
- Per-seed (42/43/44/45): 0.009122 / 0.009084 / 0.009398 / 0.009222
- 4-seed mean: **0.00921** (range 0.000314)
- Improvement over R282.D (scale=1.5 → 0.00948): **2.9%**; over R273 (scale=2.0 → 0.0105): **12.3%**; LLNL lead over LANL widens to **+29.7%**

### LLNL R282.D MSR Exchange (PRIOR — superseded by R282.F)
- Same atlas/recipe; only `--stack-rank-scale 1.5`
- Per-seed: 0.009271 / 0.009519 / 0.009743 / 0.009376
- 4-seed mean: 0.00948 (range 0.000472)

### LLNL R273 MSR Exchange (PRIOR — superseded by R282.D)
- Same atlas/recipe as above; only difference is `--stack-rank-scale 2.0` instead of 1.5
- Per-seed (42/43/44/45): 0.0102 / 0.0106 / 0.0102 / 0.0108
- 4-seed mean: 0.0105 (range 0.0006)

### LLNL R276 Alibaba (PRIOR — superseded by R287.A chunk-ensemble guard pass)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_alibaba_237f_inline_50k_phase2_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe: phase=2 ep=600 seed=137 cond_noise=0.05 (R237 family) — same atlas as R248
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.0 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --hot-pool-min-age 8 --max-stack-depth 524288`
- Per-seed (42/43/44/45): 0.012306 / 0.013106 / 0.012055 / 0.012345
- 4-seed mean: 0.01245 (range 0.001052)
- Used as the base fake for R287.A's chunk-ensemble cascade (which moved the
  LLNL Alibaba banked claim to 0.01078).

### LLNL bootstrap (all 9 corpora, R259g/R277.A/R278/R279/R280)
- Recipe: `python3 -m llgan.trace_bootstrap --mode shuffle --chunk-size 65536` for 1M corpora; `--chunk-size 8192` for tencent (n=100k)
- 4 seeds (42/43/44/45) per corpus

## 2DIO comparison class (per-trace memoization, separate metric)

This is **not** part of the LLNL/LANL race; it tracks the third-party
2DIO (EuroSys '26) bar. Per-trace memoization is a different metric
class than corpus generalization — fair comparisons require all
entrants to be measured per-trace.

| Trace | LLNL corpus-fit best | LLNL per-trace best | 2DIO per-trace θ | Leader (per-trace) |
|---|---|---|---|---|
| alibabaBlock_521 | 0.079 (R248 R270) | 0.101 (R284.X) / 0.111 (R284.Y) | 0.02–0.05 (paper) | **2DIO** |
| CP w11 (M=3M, extrap.) | — | 0.314 (R284.B, R224 atlas) | 0.02–0.05 | **2DIO** |
| CP w24 (M=16M, extrap.) | — | 0.386 (R284.B) | 0.02–0.05 | **2DIO** |
| CP w44 (M=3.7M, extrap.) | — | 0.537 (R284.B) | 0.02–0.05 | **2DIO** |
| CP w82 (M=190k, in-dist) | — | 0.141 (R284.B) | 0.02–0.05 | **2DIO** |

**Finding (R284.X+R284.Y)**: LLNL's atlas hits a per-trace floor at
~0.10 LRU HRC-MAE on v521 regardless of capacity (192 states vs 6
states gives the same floor). The 2DIO bar is structurally
unreachable from the current architecture; closing the gap requires
an atlas-fit IRD-shape loss (IDEAS-LLNL #26). LLNL's *corpus*
generalization (0.079 across 237 traces with one model) is a
metric-class advantage 2DIO does not contest.

## Open fronts (LLNL perspective)

1. **MSR retake** (critical; lost to LANL Round 70): LANL 0.00484 banked
   vs LLNL 0.00921 → 47.5% gap. LANL has additionally posted a 0.00433
   chunk-ensemble retake in `RESPONSE-LANL.md` (audit-pending integration).
   LANL's banked lever: hp=0.25, rank_scale=1.0, min_age=16 (vs LLNL
   hp=0.45, rank=1.3, min_age unset). Try hp sweep {0.20, 0.25, 0.30} ×
   rank_scale {0.9, 1.0, 1.1} × min_age=16 on LLNL's R270 MSR atlas. Also
   try `altgan.ird_renewal` on MSR official ref as a parallel path.
2. **Wikipedia retake**: LANL r328/r329 `0.0054596500` vs LLNL R288.W
   `0.008895` -> LANL leads by 38.6% on the row convention. The current
   LANL win is chunk-surface extraction on top of synthetic artifacts, not
   the older global-renewal row. LLNL needs a new architecture or its own
   guarded continuation from R288.W, not just rank-bucket tuning.
3. **CloudPhysics defence/retake**: LANL 0.0267 (range 0.0045, high
   variance) vs LLNL R287.CP 0.03017. LANL's rank_b=32 seed-80 was 0.0295
   — a badly-fitting seed drags the mean. LLNL should try rank_ird_buckets
   {48, 64, 96} and `--per-stream` to reduce variance.
4. **Alibaba defence** (just retook with R287.A 0.01078): LANL R303 cascade
   tightened to 0.01076 in `RESPONSE-LANL.md`, putting the means within
   seed-noise. Pre-empt with finer-cascade chunk-ensemble (chunk_size in
   {2K, 4K, 8K}) or fit-time work on the R270 atlas.
5. **Baleen24 defence**: LLNL R291.BAL2 `0.018447` vs LANL r379
   `0.0208754083` -> LLNL leads by 13.2%. LANL's r379 tightened its row and
   reduced variance, but closing the remaining gap likely needs a new
   architecture, not smaller object-ID chunks.
6. **Twitter / Meta KV / Meta CDN**: LANL leads Twitter by 5.2% after r351,
   Meta KV by 77.3%, and Meta CDN by 22.9% after r370. LLNL's R281 claims
   were vanilla atlas with poor reuse; R287.M closed much of the Twitter gap,
   but LANL's guarded cascade retook the row. LANL's specific knobs: Twitter win=48,
   Meta KV tail=0.08/adj=0.70/drop=0.05/hp=0.25. LLNL should port those
   knobs to LLNL's atlases for these corpora and/or wrap with chunk-
   ensemble guard pass.

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
