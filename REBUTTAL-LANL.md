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
