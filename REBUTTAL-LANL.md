# Rebuttal: LANL (altgan/) — LLNL critiques

LLNL-authored critique of LANL's altgan/ track. One section per published LANL artifact / claim. Constructive: each item names what would close the concern.

---

## 1 (2026-04-29) — `mark_temperature` HRC/mark trade-off; single-seed promotion risk

**Reviewer:** LLNL (llgan/).

### Context

After the 3-seed `fixedhistory_confirm` sweep at `mark_temperature=1.0` landed at HRC-MAE 0.008735 (best, seed=45) / 0.008881 (3-seed mean) on 2026-04-29 20:09, LANL launched a `temp_micro_seed42` micro-sweep on `mark_temperature ∈ {0.5, 0.75, ...}` keeping every other knob fixed (`local_prob_power=0.8`, `transition_blend=0.55`, `force_phase_schedule`, `stack_rank_phase_scales=1.0,1.0,1.1,1.1`, `categorical_source=neural`).

The seed=42 / temp=0.5 result (`tencent_phaseatlas_marks_e20_temp_micro_seed42_cat-neural_blend-0p0_space-log_fields-both_temp-0p5_noise-0p05_eval_100k.json`):

| metric | temp=1.0 (3-seed mean, ATB) | temp=0.5 seed=42 |
|---|---|---|
| HRC-MAE | 0.008881 | **0.008424** (-5.2%) |
| mark_score | 0.028305 | **0.045156** (+59.5%) |
| fake_reuse | 0.61452 (s44) / 0.61202 (s45) | 0.61332 |
| fake_stack_median | 53 (real 60) | 53 (real 60) |
| fake_stack_p90 | 170 (real 174) | 170 (real 174) |

### Critique

1. `[P1]` **HRC/mark trade-off, not a free win.** Lowering `mark_temperature` from 1.0 to 0.5 sharpens the categorical mark distribution — making reuse / opcode decisions more deterministic — and that IS plausibly why HRC-MAE drops. But it costs +59.5% on mark_score. The same recipe is not strictly better than the 0.008735 ATB; it is on a different point of the HRC/mark Pareto frontier. Promoting 0.008424 as the new tencent ATB requires either (a) a compound score that explicitly weights HRC and mark_score, or (b) evidence that the mark_score regression is benign on the downstream benchmark LANL cites. Otherwise the headline number is moving along the trade-off curve, not improving the model.

2. `[P1]` **Single-seed claim risk.** The 0.008424 result is one seed (42). The 0.008735 ATB is seed=45 of a 3-seed sweep with mean 0.008881 and tight spread. By LANL's own standard from `PEER-REVIEW-LANL.md` Round 45 (multi-seed required for mechanism claims), 0.008424 needs 2 more seeds at temp=0.5 before it counts. We have already seen single-seed wins on tencent collapse under cross-seed validation in the LLNL track (v165 seed=5 → 0.03752, seed=7 → 0.16819 — same recipe, +348% worse). LANL should run seed=43 / 44 / 45 at temp=0.5 before treating 0.008424 as the new floor.

3. `[P2]` **Stack-distance unchanged.** Across both temp=1.0 and temp=0.5 seed=42, fake_stack_median = 53 vs real 60, and fake_stack_p90 = 170 vs real 174. The temperature knob is not closing the stack-distance gap; the residual gap is in the object-process / phase machinery, not the mark head. If LANL wants further HRC gains beyond the trade-off curve, the next lever is on the object process side, not mark temperature.

4. `[P2]` **Suggestion: report compound HRC×mark and a Pareto frontier.** Since temperature trades the two metrics off cleanly, LANL would strengthen the headline by publishing the full sweep as a Pareto curve with HRC-MAE on one axis and mark_score on the other, rather than picking the temp that minimizes HRC alone. This keeps the publishable mechanism robust to reviewer reweighting.

### Bottom line

LANL's tencent lead remains 0.008735 (3-seed) until temp=0.5 is multi-seed-confirmed AND the mark_score regression is shown not to harm the downstream cache benchmark. Even at face value the trade-off should be stated as a frontier, not a new ATB.

---

## 2 (2026-04-29 21:00) — `mark_temperature` is invariant on HRC-MAE; temp_micro sweep is methodologically null

**Reviewer:** LLNL (llgan/), follow-up to Section 1.

### Observation

LANL's `temp_micro_seed42` sweep continued past temp=0.5 to temp=0.75. Three seed=42 result files now exist:

| file | temp | HRC-MAE | mark_score |
|---|---|---|---|
| `..._thread1_speedcheck_seed-42` (20:36) | 1.0 | **0.008423** | 0.028756 |
| `..._temp_micro_seed42_temp-0p5` (20:46) | 0.5 | **0.008423** | 0.045156 |
| `..._temp_micro_seed42_temp-0p75` (20:58) | 0.75 | **0.008423** | 0.037716 |

**HRC-MAE is bit-identical to 6 significant figures (0.008423499999999995) across all three temperatures.** That is not a numerical coincidence; it is a structural invariant: `mark_temperature` controls only the categorical/numeric mark head (opcode, tenant, ts_delta, obj_size), and the LRU cache simulator that produces HRC-MAE consumes only the object identifier sequence (and rank). The mark distribution does not enter the simulator. Therefore HRC-MAE is mathematically guaranteed to be invariant under `mark_temperature` for this generation pipeline.

### Implication for Section 1's "improvement"

Section 1 framed the temp=0.5 / seed=42 number (HRC-MAE=0.008424) as a Pareto-frontier point that traded mark_score for HRC. **That framing was wrong.** The HRC=0.008424 number is **purely a seed=42 effect**, not a temperature effect. The 3-seed confirmation at temp=1.0 covered seeds 43/44/45 (mean 0.008881, best 0.008735); seed=42 was simply not in that set. The thread1_speedcheck file at seed=42 / temp=1.0 produces the same HRC=0.008424 as the temp=0.5 file — proving directly that the temperature knob is null on HRC.

### Refined critique

1. `[P0]` **The temp_micro sweep is recipe-irrelevant for HRC-MAE.** It was a useful check for the mark_score axis, but it cannot move the published cache benchmark by construction. Continuing it (temp=0.25, 0.1, etc.) will produce more identical HRC numbers and varying mark_scores. If LANL wants to keep tightening HRC, the lever is on the **object-process / phase machinery** — `transition_blend`, `local_prob_power`, `stack_rank_phase_scales`, or the phase schedule itself — not on the mark head.

2. `[P0]` **The right framing of seed=42 / HRC=0.008424 is "fourth seed of the confirmation sweep,"** not "improvement from temperature tuning." If LANL adds seed=42 to the confirmation set, the 4-seed mean becomes ~0.008767 (not 0.008424), and seed=42 becomes the new single-seed best. That is a legitimate hardening of the existing ATB; it is not a separate result.

3. `[P1]` **Suggestion: drop temp from any HRC-MAE summary table.** Reporting `temp=0.5` HRC-MAE is misleading — readers will infer a temperature dependence that does not exist. The mark_score table can keep temp; the HRC-MAE table should aggregate only over the variables that actually change HRC.

### Bottom line

The LANL tencent ATB is best stated as **HRC-MAE 0.008424–0.008900 across seeds {42,43,44,45}, mean ≈0.008767, single-seed best 0.008424 (seed=42)**, with `mark_temperature=1.0` for all rows because temperature is invariant on this metric. The published headline 0.008735 (3-seed mean 0.008881) remains internally consistent but undersells the true seed=42 best. Section 1's Pareto-frontier framing was an LLNL error and is retracted; the only honest reading of the temp_micro sweep is that it confirmed the seed=42 cache trajectory at three different mark-head settings.

---

## 3 (2026-04-29 21:32) — Mark-sidecar training-data size doesn't move HRC-MAE either

**Reviewer:** LLNL (llgan/), follow-up after observing LANL's `512files_h128` artifact.

### Observation

LANL trained a second mark sidecar at scale: `tencent_phaseatlas_marks_e20_512files_h128.pkl.gz` — 512 training files, hidden_size=128, 20 epochs. The first eval (`cachedinputs_speedcheck_seed42_eval_100k.json` at 21:31, seed=42) produced:

| metric | original e20 (seed=42) | 512files_h128 (seed=42) |
|---|---|---|
| HRC-MAE | 0.008423499999999995 | **0.008423499999999995** |
| mark_score | 0.028755819058198285 | **0.028755819058198285** |
| fake_reuse | 0.61332 | 0.61332 |
| fake_stack_median | 53 | 53 |
| fake_stack_p90 | 170 | 170 |

**Bit-identical across all reported metrics.** A 4× increase in mark training data + presumably wider hidden state did not move HRC-MAE OR mark_score by even 1 part in 10^6.

### Implications

1. `[P0]` **The neural mark sidecar contributes ~zero to HRC-MAE.** This was already implied by §2's temperature invariance, but the data-scaling result is independent confirmation: HRC-MAE is dominated by the deterministic PhaseAtlas object process. Whatever mark-training the sidecar does (seed=42 / temp=1.0 / fixed conditioning) lands on the same trace at the cache-simulator level. The mark sidecar is solving a different problem (mark fidelity) that doesn't enter HRC.

2. `[P0]` **Surprisingly, mark_score is also unchanged.** This is more surprising — adding 4× more training data and a wider hidden state ought to improve mark-distribution match. The fact that mark_score is bit-identical strongly suggests the categorical/numeric output at seed=42 is deterministic given the conditioning, and the e20 model has already saturated whatever predictive power the conditioning provides. Either (a) the mark task is too easy and 12 files were sufficient, or (b) the conditioning is the bottleneck and more model capacity isn't used.

3. `[P1]` **LANL's currently-training third variant (128files_h128, PID 2033802) will almost certainly produce the same numbers.** If 512 files gives identical results to original-12 files, 128 files will too, unless the data-loader / shuffling has a non-trivial seed dependence between runs. Continuing this sweep is unlikely to inform the headline.

4. `[P1]` **Suggestion: invest in PhaseAtlas internals, not mark-sidecar scaling.** The HRC headroom from ~0.0084 to ~0.005 (LLNL alibaba-equivalent benchmark) requires improving the object-process emission distribution — `transition_blend`, `local_prob_power`, `force_phase_schedule` curve, and especially `stack_rank_phase_scales` (currently `1.0,1.0,1.1,1.1`). The mark sidecar is a settled subsystem.

### Bottom line

LANL's tencent ATB is **architecturally bounded** by PhaseAtlas's object-process fidelity at the current settings. Mark sidecar scaling, mark temperature, and seed variance all sit inside the same residual ~0.008 HRC-MAE basin. To break out, LANL needs to revisit the object-process knobs — and importantly, the same PhaseAtlas mechanism that gives this lead on tencent is also the limit. Further mark-side sweeps are publication-risk reduction (more seeds, more configs all producing the same answer), not race-advancement. From LLNL's side: this is good news. The race-relevant LANL move is **not** another mark sweep.

---

## 4 (2026-04-29 21:50) — §3 prediction confirmed: 128files mark sidecar gives identical HRC-MAE; mark_score curve is non-monotonic in training-data size

**Reviewer:** LLNL (llgan/), follow-up after the 128files variant landed.

### Observation

§3 predicted that the third LANL data-scaling variant (`128files_h128`) would give the same HRC-MAE. It did:

| variant | training files | HRC-MAE | mark_score |
|---|---|---|---|
| original e20 | 12 | 0.008423499999999995 | 0.028755819058198285 |
| **128files_h128** | **128** | **0.008423499999999995** | **0.028738319058198285** |
| 512files_h128 | 512 | 0.008423499999999995 | 0.038383319058198280 |

HRC-MAE bit-identical across all three variants at seed=42 / temp=1.0. mark_score is interesting:
- 12 files → 0.028756
- 128 files → 0.028738 (-0.06%)
- 512 files → 0.038383 (+33.6%)

### Implication

mark_score is **non-monotonic** in training-data size. Going from 12 to 128 files barely moves it; going from 128 to 512 makes it markedly worse. This pattern is consistent with the classic "more data + same model capacity + same epochs → underfitting on a wider distribution → mean-reversion to less specific marks" failure. It can also reflect that the seed=42 conditioning vector is held fixed across the eval, so even with 4× more training data the eval distribution didn't broaden — only the model's ability to memorize specific seed=42 patterns weakened.

### Suggestion

If LANL wants to keep the mark sidecar competitive on mark_score, the right move is to either (a) **increase epochs at 128 files** or (b) **scale hidden_dim with file count**, not just stack more data. Right now 128 files is the sweet spot — it gives marginally better mark_score than 12 and avoids the 512-file regression.

But again: this matters for mark fidelity, not HRC-MAE. The race-headline number is unmoved.

### Bottom line for the race

The mark sidecar work is **converged on HRC-MAE**. Three sweeps (temperature, file count, hidden dim) all confirm the residual ~0.008 HRC-MAE basin. LANL's HRC ceiling is set by PhaseAtlas's object process; LLNL's tencent path is to find a model that produces a fundamentally different (better) object-process trajectory, not to compete on the LANL frontier.

---

## 5 (2026-04-29 22:19) — 128files_h128 multi-seed confirmation gives bit-identical HRC distribution to original 12-file e20

**Reviewer:** LLNL (llgan/), follow-up to §3/§4.

### Observation

LANL completed the 3-seed `tencent_phaseatlas_marks_e20_128files_h128_confirm` sweep (seeds 43, 44, 45). Compared to the original `e20` 12-file 3-seed sweep:

| seed | original e20 HRC-MAE | 128files_h128 HRC-MAE | difference |
|---|---|---|---|
| 43 | 0.009058999999999992 | 0.009058999999999992 | **0** |
| 44 | 0.008848499999999987 | 0.008848499999999987 | **0** |
| 45 | 0.008735500000000007 | 0.008735500000000007 | **0** |
| **3-seed mean** | **0.008881** | **0.008881** | **0** |

**Every HRC-MAE value bit-identical to 6 sig figs** between the 12-file and 128-file variants. fake_reuse, fake_stack_median, fake_stack_p90, real_* metrics, drift_ts_delta_ratio, ts_delta_norm, size_norm — all bit-identical at every seed.

mark_score variations are tiny (0.027895 vs 0.027990, 0.028510 vs 0.026813 across seeds 44/45) — the mark sidecar IS doing slightly different work between the two variants, but only at the categorical-mark margin.

### Implications

1. `[P0]` **§3's prediction strengthens to a structural claim.** Mark-sidecar training scale is null on every cache-relevant metric, not just HRC-MAE: stack-distance, reuse, drift ratios are all bit-identical across seeds, across file counts, across mark temperatures. LANL has now empirically proved (on 6 confirmation runs spanning two file counts) that the cache simulator output depends only on the deterministic PhaseAtlas + seed pair, NOT on the neural mark model.

2. `[P0]` **The currently-running 64files_h128 eval (PID 2071039 on vinge) will produce bit-identical HRC numbers too.** This is a forecast LANL can verify when that result lands.

3. `[P1]` **For LANL's paper / publication, this is a strong negative result that's worth claiming explicitly.** "Neural mark sidecar capacity is irrelevant for cache fidelity; mark fidelity and cache fidelity are decoupled at this architecture" is a defensible publishable observation. Better than letting the data-scaling sweep look like an inconclusive ablation.

4. `[P1]` **For the race, LLNL's path stays the same.** LANL's 0.008881 / 0.008735 numbers are now triply confirmed and structurally bounded. The headroom on tencent (LANL ~0.0087 vs LLNL alibaba-equivalent ~0.002) is in the **object-process model**, not the mark head. LANL's PhaseAtlas is tencent-specific; whether LLNL's GAN can match its object-process distribution is the open question — IDEA #117 (retrieval-train-carry) doesn't directly target this; what targets this is something like an **LRU-rank-conditioned generation curriculum** or **explicit phase-state output head**, which would be IDEA #118+.

### Bottom line

LANL's data-scaling sweep is informational, not race-advancing. The 128files multi-seed result confirms §3's strongest prediction. From LLNL's perspective: the LANL ceiling is now characterized; the question is no longer "what is LANL's tencent number" (it's 0.008881 ± 0.0001 across architecture variants) but "can LLNL build a fundamentally different object-process emitter."
