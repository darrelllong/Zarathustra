# LEADER-BOARD

Last updated: **2026-05-04 (R281.K KV-class scale=0.001 banked + R285 LANL claims merged; LANL leads 8/9 corpora generatively)**.
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
| Alibaba | 0.01245 (R276 cool8: R244 lock + --hot-pool-min-age 8) | **0.01188** (cooldown control, 4-seed {42,80,81,82}) | **LANL** | −4.6% |
| Tencent | 0.0305 (R206 — unverified, R283.B can't reproduce; protocol lost) | 0.0336 (100k retarget) | tied (with caveat) | — |
| CloudPhysics | 0.0311 (R283.H: R237 atlas + scale=0.7; 8-pol multi-seed, range 0.000469) | **0.0267** (rank-conditioned IRD-renewal, 8-pol 4-seed {42,80,81,82}) | **LANL** | −14.1% |
| Baleen24 | 0.0438 (R245: hp=0.35 K=75 adj=0.55 tp=0.05 mf=0.5 rp=0.15 win=2) | **0.0276** (scout-rank atlas, 4-seed {42,80,81,82}) | **LANL** | −37.0% |
| MSR Exchange | 0.00921 (R282.F: R270 atlas + scale=1.3) | **0.00484** (hp=0.25 rank=1.0 min_age=16; 6-pol 4-seed {42,80,81,82}, range 0.0000631) | **LANL** | −47.5% |
| Twitter | **0.02936** (R281.K scale=0.001 multi-seed, range 0.000121) | **0.0272** (Twitter atlas win=48; 4-seed {42,80,81,82}, range 0.000340) | **LANL** | −7.4% |
| Meta KV | **0.05587** (R281.K scale=0.001 multi-seed, range 0.000417) | **0.0109** (tail_reuse=0.08 reuse_drop=0.05 hp=0.25; 4-seed) | **LANL** | −80.5% |
| Meta CDN | **0.04625** (R281.K scale=0.001 multi-seed, range 0.000378) | **0.0377** (low-tail update; 4-seed {42,80,81,82}) | **LANL** | −18.5% |
| Wikipedia | 0.01727 (R280.I scale=4.5 multi-seed, range 0.000132) | **0.01146** (IRD-renewal ird_s=32 ip=0.10; 4-seed {42,80,81,82}, range 0.000533) | **LANL** | −33.6% |

**Generative score**: LLNL leads 0 corpora; LANL leads 8 (Alibaba, CP, Baleen24, MSR, Twitter,
Meta KV, Meta CDN, Wikipedia); tied 1 (Tencent, with protocol caveat on LLNL's historical row).
All 9 corpora have generative claims from both teams.

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

### LLNL R280.I Wikipedia (current banked, supersedes R280.M)
- Same atlas/recipe as R280.M; only `--stack-rank-scale 4.5` instead of 5.0
- Per-seed (42/43/44/45): 0.017194 / 0.017252 / 0.017306 / 0.017326
- 4-seed mean: **0.01727** (range 0.000132 — among the tightest in the project)
- Improvement over R280.M (scale=5 → 0.01740): 0.7%
- Demonstrates the "every coarse-grid winner has a fine-fine improvement" pattern

### LLNL R280.M Wikipedia (PRIOR — superseded by R280.I)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_wiki_3f_inline_50k_phase2_t4s4_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe (R270 architecture): same as MSR — `--n-phase-bins 2 --n-time-bins 4 --n-size-bins 4 --hidden 96 --epochs 600 --seed 137 --inline-cond --cond-noise-std 0.05` on 3 wiki traces × 50k records
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.0 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --max-stack-depth 524288 --stack-rank-scale 5.0`
- Per-seed (42/43/44/45): 0.017369 / 0.017324 / 0.017393 / 0.017500
- 4-seed mean: **0.01740** (range 0.000175 — extremely tight)
- Cache sizes: [32, 128, 512, 2048, 8192]; policies: lru/arc/fifo/sieve/slru/car

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

### LANL Wikipedia (current leader — Round 68, IRD-renewal)
- Method: `python3 -m altgan.ird_renewal` — empirical IRD + IRM renewal from official ref
- Recipe: `--independent-prob 0.10 --ird-scale 32`; 1M rows, synthetic IDs, no rank buckets
- Per-seed (42/80/81/82): 0.0112286 / 0.0115764 / 0.0112679 / 0.0117615
- 4-seed mean: **0.01146** (range 0.000533)
- Reference: `/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`; 6-policy surface
- LLNL gap: **34.1% behind** (LLNL 0.01740 vs LANL 0.01146)

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

### LLNL R276 Alibaba (current LLNL alibaba claim — R248 atlas + cool8 lever)
- Atlas: `/tiamat/zarathustra/llgan-output/atlases/llnl_neural_atlas_alibaba_237f_inline_50k_phase2_ep600_extbins_seed137_noise0p05.pkl.gz`
- Recipe: phase=2 ep=600 seed=137 cond_noise=0.05 (R237 family) — same atlas as R248
- Generation knobs: `--hot-pool-prob 0.45 --hot-pool-k 75 --adj-dup-prob 0.0 --tail-reuse-prob 0.10 --tail-reuse-min-frac 0.5 --recent-pool-prob 0.15 --recent-pool-window 16 --hot-pool-min-age 8 --max-stack-depth 524288`
- Per-seed (42/43/44/45): 0.012306 / 0.013106 / 0.012055 / 0.012345
- 4-seed mean: **0.01245** (range 0.001052)
- Cache sizes: [32, 128, 512, 2048, 8192]; policies: lru/arc/fifo/sieve/slru/car
- Improvement over R248 baseline (0.0131): **5%**; gap to LANL 0.0119: **4.7% behind** (was 9.4%)

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

1. **MSR retake** (critical; lost to LANL Round 70): LANL 0.00484 vs LLNL
   0.00921 → 47.5% gap. LANL's lever: hp=0.25, rank_scale=1.0, min_age=16
   (vs LLNL hp=0.45, rank=1.3, min_age unset). Try hp sweep {0.20, 0.25,
   0.30} × rank_scale {0.9, 1.0, 1.1} × min_age=16 on LLNL's R270 MSR
   atlas. Also try altgan.ird_renewal on MSR official ref as a parallel path.
2. **Wikipedia retake**: LANL 0.01146 (IRD-renewal ird_s=32 ip=0.10) vs
   LLNL 0.01740 → 34.1% gap. LANL used global renewal only; they have NOT
   published rank_ird_buckets or --per-stream results for Wikipedia. LLNL
   should sweep rank_ird_buckets {8, 16, 32} × ird_scale {24, 28, 32, 36}
   and --per-stream. If Wikipedia ref has multi-stream rows, per-stream
   could be decisive.
3. **CloudPhysics defense/retake**: LANL 0.0267 (range 0.0045, high
   variance) vs LLNL 0.0311. LANL's rank_b=32 seed-80 was 0.0295 — a
   badly-fitting seed drags the mean. LLNL should try rank_ird_buckets
   {48, 64, 96} and --per-stream to reduce variance. If LLNL hits mean
   0.027 with range < 0.002, that's a clean retake via tighter uncertainty.
4. **Alibaba retake**: LANL 0.01188 vs LLNL 0.01245 → 4.7% gap. Finer
   cooldown sweep (age 6/7/9/10), or new atlas fit. Also worth trying
   altgan.ird_renewal on alibaba ref.
5. **Baleen24 retake**: LANL 0.0276 vs LLNL 0.0438 → 37% gap. R270
   architecture regressed on Baleen24 (R271). Need a different fit-time
   approach — Baleen24 is write-heavy; opcode-transition fidelity is key.
6. **Twitter / Meta KV / Meta CDN**: LANL leads by 62–96%. LLNL's R281
   claims were vanilla atlas with poor reuse. LANL's specific knobs:
   Twitter win=48, Meta KV tail=0.08/adj=0.70/drop=0.05/hp=0.25. LLNL
   should port those knobs to LLNL's atlases for these corpora.

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
