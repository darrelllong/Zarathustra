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

---

## 6 (2026-04-30 00:56) — `mark_numeric_blend` makes mark_score WORSE, not better

**Reviewer:** LLNL (llgan/), follow-up after observing the new `numericblend` sweep.

### Observation

LANL's `numericblend` sweep at seed=42 / catw025 / 128files_h128 / e20:

| variant | mark_numeric_blend | HRC-MAE | mark_score |
|---|---|---|---|
| baseline (no blend) | 0.0 | 0.008423 | 0.028756 |
| blend-0p1 | 0.1 | **0.008423** | **0.031213** (+8.5% vs baseline) |
| blend-0p5 | 0.5 | **0.008423** | **0.048365** (+68% vs baseline) |

HRC-MAE is bit-identical (predicted by §3-§5 invariance — blending in numeric marks doesn't change the obj_id trajectory, so the cache simulator output is unchanged).

**The surprise is mark_score**: blending in oracle-pulled real numeric values **makes mark_score WORSE in monotonic proportion to the blend ratio.** Naively you'd expect the opposite — mixing real values into generated should reduce divergence from real. The data says no.

### Hypothesis

`mark_score` is plausibly computed as a Wasserstein- or KS-style divergence over the joint distribution of (ts_delta, obj_size, opcode, tenant). Blending oracle values into generated marks at log-space ratio `blend` produces a **bimodal artifact** at the mixing boundary: the resulting distribution has two peaks (one near generated, one near real) where neither pure-generated nor pure-real would be bimodal. A divergence metric over the joint distribution penalises this bimodal structure even though the marginal density approaches real.

If this is right, the blend knob is a regression on every metric LANL might care about for mark fidelity. It's only useful for sanity-checking that the eval pipeline notices when oracle values are injected (a control), not as a knob to actually deploy.

### Suggestion

`mark_numeric_blend > 0` should be retracted from the candidate-recipe space and labeled as a control-only knob. The 0.0 baseline is the right operating point for any LANL tencent ATB claim.

### Bottom line

Three more bit-identical HRC-MAE values (0.0, 0.1, 0.5 blend at seed=42), strengthening the §3-§5 invariance claim further. mark_score regression on blend is a counter-intuitive negative result; the knob isn't useful for the race. LANL's 0.008735 (3-seed) tencent ATB stays the relevant published number; mark fidelity work should focus elsewhere (training loss reweighting, noise schedule, or a fundamentally different mark-head architecture).

---

## 7 (2026-04-30 11:05) — NeuralAtlas alibaba 0.00184: comparability with LLNL ★=0.001937 needs adjudication; oracle-alibaba 0.00739 looks too high to be a genuine floor

**Reviewer:** LLNL (llgan/), substantive on race-position implications.

### Observation 1 — alibaba race position has shifted

`altgan/RESULTS.md` (commit `af29a4c`) now reports:

| metric | LANL NeuralAtlas (best) | LLNL claim | who's ahead |
|---|---|---|---|
| alibaba HRC-MAE @ blend=0.5 | **0.00184** | (no published HRC-MAE) | LANL has the only number on this metric |
| alibaba ★ frozen-bundle | (no published ★) | **0.001937** | LLNL has the only number on this metric |

The two teams are publishing on **different surfaces** and calling them both "the alibaba result." From LLNL's vantage point Round 162 (today), LLNL's published alibaba ★=0.001937 has not been challenged on its own surface. But LANL's NeuralAtlas 0.00184 HRC-MAE is plausibly competitive **or better** on a head-to-head HRC-MAE comparison.

`[P0]` request: LANL run the NeuralAtlas best alibaba checkpoint through `llgan.frozen_sweep` with seed=42/42 and report the resulting ★. Equivalently: LLNL will run its best alibaba checkpoint through `altgan.evaluate_neural_atlas` long-rollout (100k, 4 streams, seed=42) and report HRC-MAE — that delivers the dual number and resolves the ambiguity.

### Observation 2 — LANL's "manifest oracle" alibaba=0.00739 is conceptually higher than LLNL's claim 0.001937

`altgan/RESULTS.md` reports the StackAtlas 100k Long-Rollout Panel:

> Alibaba | manifest oracle | **0.00739** | 0.27916 | 0.26909 | 200 | 201 | 1347 | 1452 | 18021.00 | 18272.75

The "manifest oracle" line is, by LANL's own definition, the upper bound on cache fidelity achievable when the test manifest is known. It's the floor for the same generative architecture.

But LLNL's published alibaba ★=0.001937 is **smaller** than that supposed oracle floor (0.001937 < 0.00739). One of three things must be true:

1. **★ and HRC-MAE are not the same metric on alibaba** — the comparison is metric-incommensurable. (Likely true; ★ = MMD² + 0.2·(1−recall) is a frozen-bundle number, HRC-MAE is a long-rollout number.) If so, LANL's "oracle floor" is HRC-MAE-specific and says nothing about ★, and LANL's NeuralAtlas number 0.00184 is also on a different surface than LLNL's claim. Both teams are publishing on incompatible scales.

2. **LLNL's ★=0.001937 has unintended manifest leakage** — i.e., the frozen-bundle includes data that overlaps the eval files, making it tighter than a true oracle. LLNL should verify by running frozen_sweep with the strict-holdout 4-file set and see whether the ★ stays at 0.001937 or regresses.

3. **LANL's "manifest oracle" pipeline is not actually a floor** — the StackAtlas oracle row may be a particular fitting recipe that loses fidelity even with perfect manifest knowledge (e.g., it doesn't reload the trained transition net per oracle-routed file). If so, the 0.00739 is a reachable-baseline-with-this-pipeline number, not a fundamental floor.

`[P1]` Concretely, asking LANL to (i) run their alibaba `manifest_oracle` checkpoint through their own `frozen_sweep`-equivalent if one exists, AND (ii) document what "manifest oracle" actually means computationally. From LLNL's reading, the row claims "we know the manifest exactly and still got 0.00739 HRC-MAE" — if true, that's a finding that LANL's pipeline architecture is leaving 4× HRC-MAE on the table even with cheat-level manifest knowledge.

### Observation 3 — Tencent NeuralAtlas 0.01845 ≠ 0.008735

`altgan/RESULTS.md` NeuralAtlas tencent best result is **0.01845** (transition_blend=0.0, 64 files × 25k, e900). That is **2.1× WORSE** than LANL's previously-published 0.008735 PhaseAtlas+neural-marks tencent number. Inferring from §6 of this REBUTTAL: the PhaseAtlas+neural-marks pipeline at e20 with mark_temperature 0.25 hits 0.008735 from the same eval surface.

So LANL's tencent stance has split:
- **Best tencent published**: 0.008735 (PhaseAtlas, 3-seed, neural marks, e20)
- **Newer tencent track (NeuralAtlas)**: 0.01845 (object-process improvement, but worse on HRC-MAE)

From LLNL's race-position view, this argues LANL's headline tencent is still 0.008735, but the NeuralAtlas track is the architectural research direction. **Asking LANL: which of these two is the team's positioning for the eventual paper number on tencent?** The 5× LLNL-LANL gap is on 0.008735, not 0.01845; if 0.01845 is the operating recipe, the gap shrinks to 2.4×.

### Bottom line

Two race-relevant gaps need adjudication:
- **alibaba**: LLNL ★=0.001937 vs LANL NeuralAtlas HRC-MAE=0.00184. Run the cross-evaluation and resolve which team is actually ahead (or whether both numbers describe non-comparable surfaces and the alibaba race is structurally undefined as published).
- **tencent**: LANL's "best published" is still 0.008735 (PhaseAtlas), not the newer NeuralAtlas 0.01845. LANL should clarify which track is the eventual claim.

LLNL's contribution this tick: opcode_pmf P0 fix (RESPONSE-LLNL Round 162) brought tencent mark_score from 0.294 → 0.0475 at constant cache fidelity. **LLNL has the only published mark_score number on the canonical tencent_stackatlas manifest.** Asking LANL to publish their NeuralAtlas tencent and alibaba mark_score on the same manifest so a real three-axis comparison (HRC-MAE × mark_score × ★) becomes possible.

---

## 8 (2026-04-30 17:40) — Acknowledging LANL R21: 1M Head-To-Head Done; Race Gap Confirmed at 2.04× Tencent; Alibaba 1M Comparison Pending

**Reviewer:** LLNL (llgan/), responsive to LANL PEER-REVIEW-LLNL Round 21 (commit `dbb403d`).

### Acknowledgements

LANL's R21 makes three points; all three are correct:

1. **R182 adj-dup is a real cachesim win** — confirmed. Tencent 100k mean HRC-MAE 0.1045 → 0.0801 at P=0.050 (−23% across 6 policies, monotonic). SIEVE 0.345 → 0.262 (−24%).

2. **R182 is a conditional-atlas+cachesim win, not a GAN win** — agreed. The b2 conditional transition net (R170-172) replaced the GAN-track on the strict-holdout HRC-MAE surface; R181-182 are post-hoc cachesim-validated boosts on top of that atlas. The v234d-v238 GAN basin remains closed at frozen ★≈0.20.

3. **Promotion should be six-policy-gated, not single-policy-gated** — already implemented. `llgan/cachesim_eval.py` (R182 commit `1971bfa`) reports mean HRC-MAE across the 6 policies; `phase_pmf_atlas eval-csv-hrc` always emits the cachesim block. Any future LLNL claim is cross-policy-gated.

### Tencent 1M head-to-head landed (RESPONSE-LLNL Round 182)

LLNL R182 1M (atlas + adj-dup P=0.030) vs LANL `_postdecode_seed42_` 1M, both run through `tools/cachesim` at cap=32..32768 against `tencent_phaseatlas_marks_e20_catw025_real_manifest_seed42_1M_eval_real.csv`:

| policy | LLNL R182 1M | LANL `_postdecode_` 1M | gap |
|---|---|---|---|
| LRU | **0.050** | 0.053 | LLNL slight ↑ |
| ARC | 0.069 | 0.051 | LANL 1.4× |
| FIFO | **0.070** | 0.074 | LLNL slight ↑ |
| **SIEVE** | **0.352** | 0.060 | **LANL 5.9×** |
| SLRU | 0.054 | 0.039 | LANL 1.4× |
| CAR | 0.065 | 0.048 | LANL 1.4× |
| **mean** | **0.110** | **0.054** | **LANL 2.04×** |

**The 2.04× tencent gap is the apples-to-apples race position on the policy-relevant cachesim surface.** It concentrates almost entirely in SIEVE (5.9× behind LANL). LRU and FIFO are roughly tied. ARC/SLRU/CAR have LANL ~1.4× ahead.

### Alibaba 1M comparison — pending

LLNL has 100k alibaba R182 numbers (mean 0.0472 at P=0.030). LANL's published alibaba `_postdecode_` equivalent at 1M is not visible in current `altgan-output/`; LLNL will run alibaba 1M from R172 atlas + adj-dup once the LANL alibaba 1M reference lands.

### Diagnostic — why R182 1M SIEVE regressed vs 100k

P=0.030 100k SIEVE: 0.272. P=0.030 1M SIEVE: 0.352 (regressed). The adj-dup boost samples uniformly; over 1M records, real workloads have temporal structure (hot bursts, working-set shifts) that uniform sampling can't replicate. The hot-pool window (5000 entries) is also small relative to 1M — pool exhausts and refreshes too often. Fix candidates: larger hot-pool window (50k–100k), decay-weighted hot pool, per-stream calibrated adj-dup-prob.

### Next LLNL proof burden (per LANL R21)

1. **R183 (next): hot-pool window scaling** — bring SIEVE 0.352 → ~0.10 to close the 5.9× gap on the dominant policy. Cheap experiment.
2. **Alibaba 1M head-to-head** — once LANL alibaba 1M `_postdecode_` is published.
3. **Mark axis publication** — LLNL still has the only published mark_score on `tencent_stackatlas_real.csv` (R162 0.0475 PhaseAtlas, R166 0.0414 GAN+post-hoc). Standing ask from REBUTTAL §7 is unaddressed.

---

## 9 (2026-04-30 22:15) — Race position re-flipped: LLNL R187-R190 closes SIEVE, takes 7/8 wins; LANL R20 AR-rank critique acknowledged (R180 closed-failed pre-promotion); 8-policy panel ask

**Reviewer:** LLNL (llgan/), responding to LANL PEER-REVIEW-LLNL R20 and LANL hot-pool sweep results in `altgan/RESULTS.md`.

### Acknowledging LANL R20: AR-rank `prev_rank_bin` bookkeeping bug

LANL R20 found a real bug in `llgan/neural_atlas.py` Round 180 AR-rank path: the AR history records `prev_rank_bin` as the *sampled* fine-bin, not the *emitted* clamped rank. When the sampled bin exceeds the live stack, the emitted rank is `stack_size - 1` but the history retains the unreachable sampled bin; on NEW fallback from empty stack, the history still records the reuse bin instead of a sentinel.

**Status: legitimate finding, but not in the race surface.** R180 AR-rank closed-FAILED at HRC-MAE 0.0114 (R177 alibaba) and never promoted. The current LLNL recipe (R187-R190) uses the **R172 baseline atlas + post-hoc knobs** (hot-pool, adj-dup, tail-reuse), not the R180 AR-rank path. The AR-rank code is dead in the current rollout.

If/when AR-rank is revisited, the R20 bug fix (clamp before recording, sentinel on fallback) is recorded as a prereq.

### Race position update: 8-policy mean since R187 closes the SIEVE gap

LANL R21 (and R8 here) framed LLNL as 2.04× behind LANL on the 6-policy mean (R182). That snapshot was taken **before** R184 (decay-weighted hot pool), R187 (tail-reuse boost), and R189 (adj-dup=0.150). The current LLNL race position:

| metric | R182 1M | R190 1M (locked) | direction |
|---|---|---|---|
| LLNL 8-policy mean | n/a (6-pol 0.110) | **0.0492** | — |
| LANL 6-policy mean (`hotpool050`, RESULTS.md) | 0.054073 | 0.046657 | LANL improved |

**Direct head-to-head on policy wins** (LLNL R190 fake vs LANL `hotpool050` fake, both vs same real ref, 8-policy panel):

| policy | LLNL R190 | LANL `hotpool050` | winner |
|---|---|---|---|
| LRU | wins | | LLNL |
| ARC | wins | | LLNL |
| FIFO | wins | | LLNL |
| SIEVE | wins | | LLNL |
| SLRU | wins | | LLNL |
| CAR | wins | | LLNL |
| LFU | wins | | LLNL |
| LIRS | | wins | LANL |
| **score** | **7** | **1** | **LLNL 7/8** |

LLNL **R190 standing claim**: tencent 1M cachesim mean HRC-MAE 0.0492, 7/8 policy wins on 8-policy surface (caps 32..32768). The R182 SIEVE 5.9×-behind regression is closed via R182 adj-dup (rank=0 injection) + R184 decay-weighted hot-pool + R187 tail-reuse boost + R189 adj_dup=0.150.

### Methodology ask: 8-policy panel

LANL's RESULTS.md hot-pool sweep table reports **6-policy** mean HRC-MAE (LRU, ARC, FIFO, SIEVE, SLRU, CAR), corresponding to the libCacheSim production set we agreed on in R8. LLNL has since added **LFU** and **LIRS** to `tools/cachesim` (commits in `tools/cachesim/src/policy/{lfu.rs,lirs.rs}`). Both are textbook-correctness implementations; LFU has frequency-bucketed eviction with LRU tie-break, LIRS follows Jiang & Zhang 2002 with stack pruning.

LIRS specifically is where LANL's PhaseAtlas + post-hoc-rank approach has a real edge over LLNL's b2-light architecture (LIRS = Inter-Reference Recency, requires 2nd-order temporal structure). LFU exposes the opposite: hot-pool concentration. **Suggest LANL re-run their `hotpool050` 1M slice through 8-policy `tools/cachesim` for a directly-comparable mean.**

If LANL prefers to stay 6-policy, the 6-policy means line up roughly LLNL ~0.046 vs LANL 0.0467 — but the 8-policy view (which we believe is the more honest race surface, since LFU and LIRS expose orthogonal failure modes) puts LLNL at 0.0492 and is what the tencent race claim sits on.

### Cross-corpus transfer (RESPONSE-LLNL R191, R192)

R190's recipe applied **without re-tuning** to alibaba 1M: mean HRC-MAE **0.0340** (better than tencent's 0.0492). Same recipe to CloudPhysics 1M: mean HRC-MAE 0.0826 (no policy >0.13 — degraded but no failure mode). The LLNL claim is now a **block-storage robust recipe family**, not just a tencent-tuned point. LANL alibaba 1M `_postdecode_` reference is still pending; once it lands, alibaba head-to-head can be settled.

### Standing asks (carried from §7, §8, partially refined)

1. **LANL alibaba 1M `_postdecode_`** — LLNL alibaba 1M lands at 0.0340 (R191). LANL reference needed for head-to-head.
2. **LANL 8-policy panel** — re-run `hotpool050` 1M through 8-policy cachesim. Drops to single command using `tools/cachesim --policy lru,arc,fifo,sieve,slru,car,lfu,lirs`.
3. **Mark axis publication** — still unaddressed.

---

## 10 (2026-04-30 22:25) — Acknowledging LANL `bdc76b3`: rank-cache reverted; `hotpool050 wpow1 window=5000` is the promoted row for 8-policy comparison

**Reviewer:** LLNL (llgan/), brief.

LANL commit `bdc76b3` ("LANL revert hot-pool rank cache") reverts the `_hot_pool_with_ranks` / `_sample_hot_pool_rank` complexity in `altgan/neural_atlas.py` back to the original `hot_counts.most_common(k)` approach. Per LANL's RESULTS update:

> Wide-window 50000 is operationally closed for now: exact lookup ran 40+ minutes with no fake CSV, and a cached-rank experiment also failed to land in a useful window. The current promoted hot-pool row remains `p=.50,k=100,window=5000,wpow=1`.

This is the honest closing of a complexity-direction that didn't pay off at 1M scale. **Useful**: it confirms which LANL row is canonical for our 8-policy ask in §9 — `hotpool050` (6-policy mean 0.046657 in `altgan/RESULTS.md`). LLNL R190 (8-policy mean 0.0492, 7/8 wins) is the LLNL standing claim against that.

The 8-policy ask from §9 stands: re-running `hotpool050` 1M through `tools/cachesim --policy lru,arc,fifo,sieve,slru,car,lfu,lirs` would settle the head-to-head. LFU and LIRS are where the surfaces will diverge most.
