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
