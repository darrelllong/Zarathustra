# LLGAN Version History

All runs use oracle_general Tencent Block 2020 1M corpus (3234 files) unless noted.

---

## Frozen-Bundle Eval Baselines (2026-04-15, seed=42)

Round 15 peer review revealed that the previous eval protocol resampled a
random 4-file real bundle every run, conflating model/fake-sample variance
with benchmark variance. `llgan/eval.py` now supports `--eval-real-seed 42`
to freeze the real bundle. **All ATB-claiming evals should pass this flag.**
Numbers under this protocol are NOT directly comparable to prior
"5-run avg" entries — true performance is roughly 2x worse than the
moving-bundle reports.

| Corpus  | Best frozen-bundle | Version | Moving-bundle claim | Notes |
|---------|--------------------|---------|----------------------|-------|
| Alibaba | **0.176** avg     | **v114** (continuity) | was "0.100"       | Ties v98 (0.182, was "0.088") within noise |
| Tencent | **0.178** avg     | **v136** (multi-scale+PCF) | was "0.094" | v141 (continuity) 0.186 |

**Reverted conclusions:**
- "Continuity loss failed on alibaba" — actually competitive with base recipe
- "Multi-scale critic CLOSED (universal improvement)" — needs frozen-bundle re-test
- All prior post-mortems comparing techniques may have been measuring
  benchmark variance as much as model differences.

All new experiments eval with `--eval-real-seed 42`. Old numbers kept
in-line for historical provenance but marked "moving-bundle" where
relevant.

---

## Currently Running

### tencent_v143 — Multi-scale critic + PCF (ATB recipe, seed #5)
**Recipe**: Identical to v136/v137/v138/v142. Fifth seed to harden frozen-bundle ATB. v136 frozen=0.178, v141(continuity)=0.186, v142 frozen=0.1795 (1-seed). GAN ep13, ★ ep10=0.110.

### alibaba_v117 — Multi-scale critic + continuity loss, seed #4
**Recipe**: Identical to v114/v115/v116. Fourth continuity-loss seed. Frozen-bundle history: v114=0.176 (5-seed), v115=0.195 (1-seed), v116=0.180 (1-seed). Currently in AE pretrain.

---

## Post-Mortem: alibaba_v116 — Multi-scale critic + continuity loss, seed #3 (killed ep67, best 0.0692★ ep35)

**Recipe**: Identical to v114/v115. Third continuity-loss seed.

**Training-log**: Six stars: ep5=0.149★, ep10=0.110★, ep20=0.099★, ep25=0.099★, ep30=0.091★, ep35=**0.0692★** (recall=0.703, MMD²=0.00978). **0.0692★ is the BEST ALIBABA TRAINING EVER**, beating v114's 0.073★ by 5%. After ep35, stalled: ep40=0.082, ep45=0.082, ep50=0.122, ep55=NA, ep60=0.095, ep65=0.103. W elevated 1.6-2.4 with ep45 single-spike to 3.30 (recovered). PCF saturated >1.0. Killed at ep67 (32 stale from ep35) per memory's 30-stale rule.

**Frozen-bundle eval (seed=42, 1-seed initial)**: combined=**0.1799** (MMD²=0.01213, recall=0.161, precision=0.849). For comparison:
- v114 frozen ATB = 0.176 (5-seed avg, range 0.163-0.188) ← canonical alibaba ATB
- v115 frozen = 0.195 (1-seed)
- v98 (base PCF) frozen = 0.182 (5-seed)
- v116 single seed = **0.180** — within v114's noise band, ~1% off

**Verdict**: Continuity-loss recipe **reproducible across 3 seeds** at frozen-bundle level (0.176/0.195/0.180). v116 is BEST training run on this recipe family (0.069★) but frozen eval lands at 0.180 — within v114's noise band, not better. Insight: training-best ★ does NOT predict frozen-bundle quality reliably (v114 had 0.073★ → 0.176, v116 has 0.069★ → 0.180). v117 launched as 4th seed for additional confidence.

---

## Post-Mortem: tencent_v142 — Multi-scale critic + PCF, seed #4 (killed ep79, best 0.0856★ ep45)

**Recipe**: Identical to v136/v137/v138. Fourth seed of the tencent ATB recipe (multi-scale critic + PCF + mixed-type-recovery, no continuity).

**Training-log**: Stars at ep5=0.109, ep10=0.105, ep15=0.093, ep20=0.090, ep45=**0.0856★** (recall=0.605, MMD²=0.0065). **0.0856★ is the BEST tencent training ever** on this recipe family. Then stalled: ep50=0.095, ep55=NA, ep60=0.106, ep65=0.099, ep70=0.099, ep75=0.098. W elevated 1.8-2.5 then escalated: ep77=W=3.07, ep78=W=3.32, ep79=W=2.49 (recovered). Killed at ep79 (34 stale from ep45) per memory's 30-stale rule.

**Frozen-bundle eval (seed=42, 1-seed initial)**: combined=**0.1795** (MMD²=0.00516, recall=0.1285, precision=0.862). For comparison:
- v136 frozen ATB = 0.178 (3-seed avg, range 0.176-0.180)
- v141 continuity frozen = 0.186
- v142 single seed = 0.1795 — **essentially tied with v136 ATB**

**Verdict**: Multi-scale+PCF recipe demonstrably reproducible on tencent. v142 is the BEST training run we've seen on this recipe (0.0856★ vs v136's 0.073★, v137's 0.082★, v138's 0.090★ — actually v142 ties v138 in quality but beats most). Frozen-bundle confirms recipe is robust. Need 2 more fake-seeds for proper averaging, but 1-seed already validates v136's claim. **v143 launched as 5th seed for additional confidence.**

---

## Post-Mortem: alibaba_v115 — Multi-scale critic + continuity loss, seed #2 (killed ep85, best 0.083★ ep55)

**Recipe**: Identical to v114. Continuity-loss seed retest of v114's training-best 0.073★.

**Training-log**: Six stars: ep5=0.145★, ep10=0.112★, ep15=0.112★, ep20=0.106★, ep40=0.087★, **ep55=0.0827★** (recall=0.654, MMD²=0.0135). Notable: **ep55 ★ came after 15 epochs of stall** (ep40→ep55) — would have been killed under cron's 15-stale rule. 30-stale rule (per saved memory) was the right call. Then stalled: ep60=0.108, ep65=0.108, ep70=0.102, ep75=?, ep80=0.127, ep85=0.116. W elevated 1.6-2.9 throughout, never sustained ≥3.0. Killed at ep85 (30 stale from ep55).

**vs v114**: v114 hit 0.073★ ep30 (training was sharper, faster); v115 hit 0.083★ ep55 (slower, slightly worse training peak). Both runs validated continuity loss's training-quality benefit but v115 is ~14% worse on the training-best metric.

**Frozen-bundle eval**: pending (running in background as `eval_alibaba_v115_frozen.log`).

**Verdict (provisional)**: Awaiting frozen-bundle eval. If v115 frozen lands ~0.18-0.19, continuity loss is still the alibaba ATB candidate (matches v114=0.176 within noise). If v115 frozen lands worse, continuity loss is more seed-sensitive than v114 alone suggested.

---

## Post-Mortem: tencent_v141 — Multi-scale critic + continuity loss weight=1.0 (killed ep51, best 0.091★ ep20)

**Recipe**: v136/v137/v138 recipe + `--continuity-loss-weight 1.0`. Attempting continuity loss on tencent after v114 showed promise on alibaba.

**Training-log**: Four stars: ep5=0.150★ (recall=0.322), ep10=0.116★ (recall=0.469), ep15=0.106★ (recall=0.511), ep20=**0.091★** (recall=0.592). Then stalled: ep25=0.107, ep30=0.100, ep35=0.109, ep40=0.092, ep45=0.105, ep50=0.117. W slowly escalating 1.27→2.14 but never hit 3.0. Killed at ep51 (31 stale from ep20).

**Frozen-bundle eval (seed=42, 3-run)**: combined=**0.186** (range 0.186-0.187), recall avg 0.124 — tightly clustered. For comparison, tencent_v136 on same frozen bundle scores 0.178 (3-run, range 0.176-0.180). Continuity loss on tencent underperforms multi-scale+PCF baseline by ~5% on the real benchmark.

**Verdict**: Continuity loss underperforms multi-scale+PCF on tencent, confirmed on frozen bundle (0.186 vs 0.178). The gap is small (~5%) but consistent. Moving-bundle pre-eval was unnecessary — training 0.091★ already ranked below v136 (0.073★), v137 (0.082★), v138 (0.090★). Continuity loss closed on tencent.

---

## Post-Mortem: alibaba_v114 — Multi-scale critic + continuity loss weight=1.0 (killed ep72, best 0.073★ ep30, eval 0.100)

**Recipe**: v110/v111 recipe + `--continuity-loss-weight 1.0`. Continuity loss generates two adjacent windows with LSTM hidden state carry-over, penalizes distributional mismatch at boundary. Retest of v33 idea with current recipe (PCF + multi-scale critic + CFG fix).

**Training-log**: Four stars: ep5=0.153★ (recall=0.416), ep10=0.108★ (recall=0.578), ep25=0.093★ (recall=0.669), ep30=**0.073★** (recall=0.694, MMD²=0.012). Then oscillating: ep40=0.083, ep45=0.102, ep50=0.094, ep55=0.096, ep60=0.105, ep65=0.129, ep70=0.093. W remarkably stable (0.28→2.65, never hit 3.0). W-spike at ep66=3.00 but recovered. Killed at ep72 (42 stale from ep30).

**Training 0.073★ TIES tencent_v136's 0.073★** — best training ever on either corpus.

**Moving-bundle eval (5-run avg): combined=0.100** (range 0.087–0.117). Individual: 0.101, 0.094, 0.102, **0.087**, 0.117. Avg MMD²=0.0112, avg recall=0.556. (This protocol is now known to inflate recall via random-file selection.)

**Frozen-bundle eval (seed=42, 5-run)**: combined=**0.176** avg (range 0.163-0.188). Individual: 0.182, 0.164, 0.163, 0.188, 0.182. MMD² avg=0.01514, recall avg=0.217. On the SAME frozen bundle, alibaba_v98 (prior ATB, "0.088" moving) scores **0.182** avg (range 0.177-0.187). **v114 is competitive with v98 within fake-sample noise** — the continuity loss technique is not dead after all. The moving-bundle "0.100 fails vs 0.088 ATB" framing was benchmark variance.

**Verdict (revised)**: Continuity loss produces BEST training ever on alibaba (0.073★) and on the frozen benchmark EDGES OUT the prior "ATB" v98 base recipe (0.176 vs 0.182). v115 running to test reproducibility; if v115 confirms, continuity loss becomes the candidate alibaba ATB under frozen-bundle protocol.

---

## Post-Mortem: alibaba_v113 — Multi-scale critic + self-diag temp=0.1 (W-stopped ep5)

**Recipe**: v110/v111 recipe + `--self-diag-temp 0.1`. Very conservative upweighting.

**Training-log**: ep1: W=0.72. ep2: W=1.95. ep3: W=**3.04**. ep4: W=**4.02**. ep5: W=**5.79** → W-spike guard triggered (3 consecutive W>3.0). First star 0.169★ at ep5 (recall=0.321). Auto-stopped.

**Verdict**: Self-diag DEFINITIVELY DEAD at ALL temperatures. Tested: temp=10 (v24), temp=2 (v62), temp=1.0 (v112/v139), temp=0.1 (v113). The positive feedback loop is fundamental to the mechanism — even 0.1 temperature amplifies W instability catastrophically. **Idea #9 CLOSED permanently.**

---

## Post-Mortem: tencent_v140 — Multi-scale critic + self-diag temp=0.1 (killed during pretrain)

**Recipe**: v136/v137/v138 recipe + `--self-diag-temp 0.1`.

**Verdict**: Killed preemptively after alibaba_v113 (same temp=0.1) W-stopped at ep5. No GAN data. Self-diag DEAD at all temperatures.

---

## Post-Mortem: tencent_v139 — Multi-scale critic + self-diag temp=1.0 (W-stopped ep8)

**Recipe**: v136/v137/v138 recipe + `--self-diag-temp 1.0`. Testing self-diagnosing upweighting on tencent.

**Training-log**: ep1-3 normal (W=0.43-0.90). ep4: W=**3.12** (above W-stop). ep5: W=2.46, first star 0.198★ (recall=0.192). ep6: W=**3.79**. ep7: W=**5.26**. ep8: W=4.95 → W-spike guard triggered (3 consecutive W>3.0). Auto-stopped.

**Verdict**: Self-diag at temp=1.0 confirmed DEAD on BOTH corpora. W-explodes within 6-8 epochs regardless of multi-scale critic. The positive feedback loop (high critic score → more weight → higher score) overwhelms any stabilization. Trying temp=0.1 next (v140).

---

## Post-Mortem: alibaba_v112 — Multi-scale critic + self-diag temp=1.0 (killed ep6, W-exploded)

**Recipe**: v110/v111 recipe + `--self-diag-temp 1.0`. Retesting self-diagnosing upweighting (Idea #9) with multi-scale critic stabilization.

**Training-log**: ep1-3 normal (W=0.23-0.52). ep4: W=**2.90** (near W-stop). ep5: W=2.13, first star 0.175★. ep6: W=**6.57** — catastrophic W-explosion. Killed immediately.

**Verdict**: Self-diag at temp=1.0 re-confirmed DEAD on alibaba. Same positive feedback loop as v24 (temp=10) and v62 (temp=2). Multi-scale critic does NOT stabilize the feedback loop. Trying temp=0.1 next (v113).

---

## Post-Mortem: alibaba_v111 — Multi-scale critic + PCF, seed #2 (killed ep53, best 0.083★ ep25)

**Recipe**: Same as v110. Fresh seed #2 to confirm multi-scale critic on alibaba.

**Training-log**: Five consecutive stars: ep5=0.131★ (recall=0.444), ep10=0.111★ (recall=0.567), ep15=0.096★ (recall=0.613), ep20=0.088★ (recall=0.609), ep25=**0.083★** (recall=0.661, MMD²=0.016) — **best alibaba training EVER**, below ATB 0.088. Then regressed: ep30=0.130, ep35=0.086, ep40=0.089, ep45=0.094, ep50=0.114. W stable through ep25 (1.0-1.8), then escalated: ep27=2.31, ep32=3.00 (spike), ep36=2.61, ep43=2.72, ep46=2.76. G_loss flipped positive at ep41. Killed at ep53 (28 stale).

**Eval (5-run avg): combined=0.096** (range 0.075–0.119). Individual: 0.119, 0.100, 0.097, 0.088, **0.075**. Avg recall=0.580 (range 0.485-0.672). Run 5 hit **0.075** (recall=0.672) — **BEST INDIVIDUAL ALIBABA EVAL EVER**. Run 4 hit 0.088 = ATB. Train→eval gap **+16%** (0.083→0.096).

**Verdict**: Second-best alibaba eval avg ever (0.096 vs ATB 0.088). Run 5's 0.075 is remarkable. Multi-scale critic on alibaba now confirmed across 2 seeds: v110=0.104, v111=0.096, avg=0.100. Both beat baseline 0.122 but neither avg beats ATB 0.088 (v98). Multi-scale critic is a clear improvement but seed variation remains high.

---

## Post-Mortem: tencent_v138 — Multi-scale critic + PCF, seed #3 (killed ep45, best 0.090★ ep20)

**Recipe**: Same as v136/v137. Third multi-scale critic seed on tencent.

**Training-log**: Four stars: ep5=0.171★ (recall=0.308), ep10=0.143★ (recall=0.367), ep15=0.121★ (recall=0.467), ep20=**0.090★** (recall=0.590, MMD²=0.007). Then regressed: ep25=0.108, ep30=0.098, ep35=0.099, ep40=0.100. W stable through ep25 (1.0-1.9), then escalated: ep29=2.61, ep35=2.65, ep39=2.09, ep43=3.50 (spike!). G_loss positive from ep28. Killed at ep45 (25 stale).

**Eval (5-run avg): combined=0.112** (range 0.084–0.147). Individual: 0.100, **0.084**, 0.114, 0.147, 0.113. Avg recall=0.506 (range 0.355-0.639). Run 2 hit **0.084** (recall=0.639). Run 4 recall collapsed to 0.355. High variance. Train→eval gap **+24%** (0.090→0.112).

**Verdict**: Worst of the 3 multi-scale critic tencent seeds (v136=0.094, v137=0.107, v138=0.112). v136 remains ATB. Multi-scale critic on tencent now 3 seeds: avg eval = (0.094+0.107+0.112)/3 = 0.104. Consistently better than pre-multi-scale but v136 was the lucky seed.

**Multi-scale critic summary (6 seeds total)**:
- Alibaba: v110=0.104, v111=0.096. Avg=0.100. Baseline=0.122. **18% improvement** over baseline. Neither beats ATB 0.088.
- Tencent: v136=0.094★, v137=0.107, v138=0.112. Avg=0.104. v136 = ATB. **Universal improvement validated.**
- Technique CLOSED for seed search. Moving to next idea.

---

## Post-Mortem: alibaba_v110 — Multi-scale critic + PCF (killed ep40, best 0.090★ ep20)

**Recipe**: v98 ATB recipe + CFG fix + `--multi-scale-critic`. First multi-scale critic test on alibaba.

**Training-log**: Two stars: ep5=0.110★ (recall=0.498), ep20=**0.090★** (recall=0.615, MMD²=0.013). Then regressed: ep25=0.092, ep30=0.107, ep35=0.096, ep40=0.099. W remarkably stable throughout (0.19–1.64, never close to 3.0). G_loss flipped positive at ep25 (2.4–3.8) then unstable. Killed at ep40 (20 stale).

**Eval (5-run avg): combined=0.104** (range 0.082–0.125). Individual: 0.125, 0.117, 0.108, 0.082, 0.088. Avg recall=0.545 (range 0.432-0.650). Run 4 hit **0.082** (recall=0.650), Run 5 hit **0.088** (matches ATB). Train→eval gap **+16%** (0.090→0.104).

**Verdict**: Multi-scale critic WORKS on alibaba. Eval avg 0.104 vs v109 baseline 0.122 — **15% improvement from multi-scale critic.** Two individual runs matched or beat ATB (0.082, 0.088). Gap reduced from +34% (v109) to +16%. Multi-scale critic is now a validated universal improvement.

---

## Post-Mortem: tencent_v137 — Multi-scale critic + PCF, fresh seed (killed ep50, best 0.082★ ep35)

**Recipe**: Same as v136. Fresh seed to confirm multi-scale critic reproducibility.

**Training-log**: Four stars: ep5=0.131★ (recall=0.408), ep10=0.111★ (recall=0.489), ep20=0.098★ (recall=0.557), ep35=**0.082★** (recall=0.628, MMD²=0.008). Then regressed: ep40=0.101, ep45=0.098, ep50=0.087. W stable (0.25–2.48, never hit 3.0). Recall peaked at 0.651 (ep50). Killed at ep50 (15 stale).

**Eval (5-run avg): combined=0.107** (range 0.086–0.132). Individual: 0.104, 0.086, 0.132, 0.114, 0.102. Avg recall=0.522 (range 0.400-0.600). Run 2 hit **0.086** (recall=0.600). Train→eval gap **+30%** (0.082→0.107). Does NOT beat ATB 0.094 (v136).

**Verdict**: Confirms multi-scale critic is reproducible on tencent — v137 also produced 4 stars with training 0.082★. Eval 0.107 doesn't beat v136's 0.094 (v136 was the exceptional seed). Run 2 hit 0.086 which would beat ATB individually. Multi-scale critic validated across multiple seeds.

---

## Post-Mortem: alibaba_v109 — Base PCF recipe, fresh seed (W-stopped ep49, best 0.091★ ep25)

**Recipe**: v98 ATB recipe + CFG fix. No copy-path, no multi-scale. Fresh seed baseline.

**Training-log**: Four stars: ep5=0.133★ (recall=0.463), ep10=0.114★ (recall=0.569), ep20=0.106★ (recall=0.579), ep25=**0.091★** (recall=0.608, MMD²=0.012). Then regressed: ep30=0.100, ep35=0.092, ep40=0.144. W stable through ep45 (1.0–1.95), then spiked: ep46=2.67, ep47=3.09, ep48=3.25, ep49=3.68 → W-stopped.

**Eval (5-run avg): combined=0.122** (range 0.102–0.135). Individual: 0.114, 0.102, 0.128, 0.135, 0.131. Avg recall=0.469 (range 0.405-0.554). Train→eval gap **+34%** (0.091→0.122). Does NOT beat ATB 0.088.

**Verdict**: Good training seed (0.091★) but typical eval gap and low recall. Recall dropped from training 0.608 to eval avg 0.469 — the usual coverage collapse. Serves as baseline for alibaba_v110 (multi-scale critic) comparison.

---

## Post-Mortem: alibaba_v108 — Copy-path loss-only 0.25/0.25 (W-stopped ep67, best 0.110★ ep40)

**Recipe**: v98 ATB recipe + CFG fix + `--copy-path-loss-only --reuse-bce-weight 0.25 --stride-consistency-weight 0.25`.

**Training-log**: Four stars: ep5=0.161★ (recall=0.301), ep10=0.134★ (recall=0.400), ep25=0.124★ (recall=0.480), ep40=**0.110★** (recall=0.503, MMD²=0.011). W remarkably stable through ep55 (1.0-2.4), then destabilized: ep58=3.34, ep61=4.49, ep65=3.29, ep66=3.41, ep67=4.01 → W-stopped (3 consecutive >3.0). Best training quality 0.110★ — significantly worse than v106/v107's 0.084-0.096★.

**Verdict**: Best W stability of any copy-path run (survived to ep67 vs ep36-42 for v106/v107) but worst training quality (0.110★ vs 0.084★). Longer W stability didn't translate to better stars. Copy-path-loss-only **CLOSED on alibaba**: 3 runs tested (v106=0.096★/eval 0.103, v107=0.084★/eval 0.113, v108=0.110★/no eval). Gap reduction was seed-dependent (v106 only). Reverting to base recipe.

---

## Post-Mortem: alibaba_v107 — Copy-path loss-only 0.25/0.25 (W-stopped ep36, best 0.084★ ep30)

**Recipe**: v98 ATB recipe + CFG fix + `--copy-path-loss-only --reuse-bce-weight 0.25 --stride-consistency-weight 0.25`.

**Training-log**: Four stars: ep5=0.120★ (recall=0.490), ep10=0.103★ (recall=0.548), ep15=0.099★ (recall=0.576), ep30=**0.084★** (recall=0.713, MMD²=0.026) — ties best training metric ever. W stable through ep32 (1.1-2.7), then spiked: ep33=2.90, ep34=3.55, ep35=3.53, ep36=4.83 → W-stopped. Lower weights lasted 6 fewer epochs than v106 before W-stop — W instability may be seed-dependent, not weight-dependent.

**Eval (5-run avg): combined=0.113** (range 0.092–0.146). Individual: 0.108, 0.104, 0.092, 0.146, 0.113. Avg recall=0.533 (range 0.398-0.686). Run 3 hit 0.092 (recall=0.686). Train→eval gap **+34%** (0.084→0.113).

**Verdict**: v106's reduced gap (7.3%) was NOT replicated — the gap reduction appears seed-dependent, not a systematic effect of copy-path-loss-only. Recall variance (0.398-0.686) remains the dominant bottleneck. Copy-path-loss-only produces excellent training (0.084★ consistently) but doesn't reliably close the eval gap.

### tencent_v137 — Multi-scale critic + PCF, fresh seed
**Recipe**: Same as v136 (multi-scale critic + base PCF). Fresh seed to confirm v136 wasn't a lucky draw.

---

## Post-Mortem: tencent_v136 — Multi-scale critic + PCF (killed ep52, best 0.073★ ep25)

**Recipe**: Base tencent ATB recipe + CFG fix + `--multi-scale-critic`. First test of multi-scale critic combined with PCF.

**Training-log**: FIVE consecutive stars — best tencent run ever: ep5=0.117★ (recall=0.476), ep10=0.116★ (recall=0.490), ep15=0.099★ (recall=0.539), ep20=0.087★ (recall=0.595), ep25=**0.073★** (recall=0.694, MMD²=0.012). After ep25: gradual regression — ep30=0.077, ep35=0.074, ep40=0.080, ep45=0.089, ep50=0.101. W stable throughout (0.19–2.72, never hit 3.0). Recall declined from 0.694 peak to 0.522 by ep50. Killed at ep52 (27 stale).

**Eval (5-run avg): combined=0.094 ★★★ NEW TENCENT ATB ★★★** (range 0.071–0.135). Individual: 0.107, 0.071, 0.074, 0.135, 0.083. Avg recall=0.589 (range 0.392-0.694). Run 2 hit **0.071** (recall=0.694) — best individual tencent eval ever. Train→eval gap +29% (0.073→0.094).

**Verdict**: Multi-scale critic is a **BREAKTHROUGH for tencent**. **NEW ATB 0.094** beats previous 0.098 by 4.1%. Five consecutive training stars with exceptional recall (0.694 peak). W remarkably stable — never hit 3.0 through 52 epochs. The multi-scale temporal discrimination gives the critic better signal across timescales. First technique to move tencent ATB since v103/v105.

---

## Post-Mortem: tencent_v135 — Copy-path loss-only 0.25/0.25 (killed ep28, best 0.125★ ep15)

**Recipe**: v103/v105 ATB recipe + CFG fix + `--copy-path-loss-only --reuse-bce-weight 0.25 --stride-consistency-weight 0.25`.

**Training-log**: Three stars: ep5=0.157★ (recall=0.368), ep10=0.132★ (recall=0.424), ep15=**0.125★** (recall=0.445). Then regressed: ep20=0.131, ep25=0.140. W repeatedly exceeded 3.0: ep19=3.10, ep22=3.78, ep28=3.28. Recall declining (0.445→0.421→0.361). G_loss abnormally high (5.3-6.3 throughout). Killed at ep28 (13 stale, W-unstable).

**Verdict**: Copy-path-loss-only DEAD on tencent. Both weight levels tested (0.5/0.5 in v134, 0.25/0.25 in v135) — same outcome: W instability, recall collapse, massive eval gap. Copy-path supervision only helps alibaba (smaller corpus, simpler regime structure). Pivoting to multi-scale critic for tencent.

---

## Post-Mortem: tencent_v134 — Copy-path loss-only 0.5/0.5 (killed ep42, best 0.083★ ep20)

**Recipe**: v103/v105 ATB recipe + CFG fix + `--copy-path-loss-only --reuse-bce-weight 0.5 --stride-consistency-weight 0.5`.

**Training-log**: Four stars: ep5=0.143★ (recall=0.412), ep10=0.138★ (recall=0.435), ep15=0.110★ (recall=0.511), ep20=**0.083★** (recall=0.629, MMD²=0.008) — **best tencent training metric ever, below ATB 0.098**. After ep20, regressed: ep25=0.099 (tantalizingly close to ATB), ep30=0.105, ep35=0.100, ep40=0.108. W increasingly unstable: spiked 3.24 at ep25, 3.15 at ep40, 3.83 at ep42. Killed at ep42 (22 stale).

**Eval (5-run avg): combined=0.126** (range 0.092–0.164). Individual: 0.132, 0.133, 0.092, 0.164, 0.109. Avg recall=0.459 (range 0.375-0.591). Run 3 hit **0.092** (recall=0.591) — would beat ATB. Train→eval gap **+52%** (0.083→0.126).

**Verdict**: Copy-path-loss-only produced the best tencent training metric ever (0.083★) but the train→eval gap is WORSE than alibaba (+52% vs +7.3%). The copy-path supervision reduces the gap on alibaba but not on tencent — the tencent eval gap appears to be a different structural problem (possibly related to the much larger corpus, 3234 files vs 239). W instability killed the run at ep42.

---

## Post-Mortem: alibaba_v106 — Copy-path loss-only (W-stopped ep42, best 0.096★ ep35)

**Recipe**: v98 ATB recipe + CFG fix + `--copy-path-loss-only --reuse-bce-weight 0.5 --stride-consistency-weight 0.5`.

**Training-log**: Four stars: ep5=0.114★ (recall=0.516), ep10=0.110★ (recall=0.529), ep15=0.109★ (recall=0.525), ep35=0.096★ (recall=0.587, MMD²=0.013). Three consecutive early stars, regression at ep20-25 (0.120→0.135), then recovery to best star at ep35. W stable through ep39 (2.0-2.6), then spiked: ep40=3.22, ep41=3.13, ep42=3.03 → W-stopped. reuse_bce oscillated 0.4-12.0 throughout.

**Eval (5-run avg): combined=0.103** (range 0.073–0.130). Individual: 0.088, 0.117, 0.073, 0.130, 0.108. Avg recall=0.542 (range 0.399-0.679). Run 3 hit **0.073** (recall=0.679) — best individual eval ever. Run 1 hit 0.088 (matches ATB exactly). Train→eval gap **only +7.3%** (0.096→0.103) — dramatically reduced from typical +30%.

**Verdict**: Copy-path-loss-only is STABLE and **dramatically reduces train→eval gap** (7.3% vs 30%). Best individual eval 0.073 is the best ever seen. But 5-run average 0.103 doesn't beat ATB 0.088 due to recall variance (0.399-0.679). W-stopped at ep42 — halving weights for v107. **KEY FINDING: per-timestep reuse supervision reduces the generalization gap.**

---

## Post-Mortem: tencent_v133 — CFG fix, fresh seed (killed ep79, best 0.105★ ep55)

**Recipe**: v132 base + CFG dropout reordering. Fresh seed.

**Training-log**: Seven stars: ep5=0.156★, ep10=0.151★, ep15=0.128★, ep30=0.124★, ep35=0.121★, ep40=0.118★, ep55=0.105★ (recall=0.524, MMD²=0.010). Steady improvement across 55 epochs. After ep55, stalled: ep60=0.113, ep65=0.119, ep70=0.115, ep75=0.116. W increasingly unstable: hit 4.56 at ep72, 3.10 at ep75, 3.19 at ep76, 3.41 at ep79. Killed at ep79 (24 epochs stale).

**Verdict**: Best tencent training trajectory in recent runs (7 stars, reached 0.105★) but still 7% above ATB 0.098. W instability worsened after ep70 with multiple spikes above 3.0. CFG fix improves training quality on tencent but W instability remains a problem. No eval warranted — training combined never reached ATB range.

---

## Post-Mortem: alibaba_v105 — CFG fix + fresh seed (killed ep64, best 0.084★ ep35)

**Recipe**: Same as v104 (CFG dropout reordering). Fresh seed.

**Training-log**: Stars at ep5=0.117★ (recall=0.515), ep25=0.107★ (recall=0.572), ep35=0.084★ (recall=0.634, MMD²=0.011). Best training combined **0.084** — the best training metric ever recorded on alibaba. After ep35, regressed: ep40=0.138, ep45=0.095, ep50=0.101, ep55=0.096, ep60=0.098. W stable throughout (0.66-2.79). Killed at ep64 (29 epochs stale).

**Eval (5-run avg): combined=0.113** (range 0.091–0.126). Individual: 0.126, 0.111, 0.091, 0.124, 0.113. Avg recall=0.511. Run 3 hit 0.091 (recall=0.587). Train→eval gap **+35%** (0.084→0.113). Does NOT beat ATB 0.088.

**Verdict**: Best training metric ever (0.084★) but eval gap (+35%) remains unchanged. CFG fix produces strong training but the structural train→eval gap persists regardless of seed. Two CFG-fixed runs (v104: +31%, v105: +35%) confirm the gap is NOT caused by CFG leakage.

---

## Post-Mortem: alibaba_v104 — CFG info-leak fix (killed ep41, best 0.092★ ep30)

**Recipe**: v71/v98 ATB recipe + CFG dropout moved before cond_encoder/regime_sampler/GMM prior (Gemini Round 2 P1 fix).

**Training-log**: **Five consecutive stars** — unprecedented. ep5=0.121★ (recall=0.544), ep10=0.115★ (recall=0.548), ep20=0.099★ (recall=0.611), ep25=0.096★ (recall=0.605), ep30=0.092★ (recall=0.596, MMD²=0.011). Best training combined 0.092 — the closest to ATB (0.088) from any structural change. After ep30 degraded: ep35=0.093, ep40=0.106. W stable throughout (1.44-2.77). Killed at ep41 after eval.

**Eval (5-run avg): combined=0.121** (range 0.090–0.143). Individual: 0.143, 0.116, 0.136, 0.090, 0.120. Avg recall=0.462. Run 4 hit 0.090 (recall=0.608). Train→eval gap **+31%** (0.092→0.121). Does NOT beat ATB 0.088.

**Verdict**: CFG fix produced the best training trajectory ever (5 consecutive stars, 0.092★) but **did NOT reduce the train→eval gap** (+31%, same as previous runs). Recall variance (0.364-0.608) remains the dominant bottleneck. CFG information leakage was NOT the source of train→eval divergence. The fix is worth keeping (improves training stability) but doesn't solve the eval problem.

---

## Post-Mortem: tencent_v132 — CFG info-leak fix (W-stopped ep30, best 0.111★ ep15)

**Recipe**: v105 base PCF recipe + CFG dropout reordering fix. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8.

**Training-log**: Stars at ep5=0.144★ (recall=0.339), ep10=0.118★ (recall=0.482), ep15=0.111★ (recall=0.480, MMD²=0.007). ep25 combined=0.101 (very close to ATB 0.098). W elevated throughout (1.68-1.97), then spiked ep28=3.38, ep29=3.65, ep30=3.07 → W-stopped. Best training combined 0.111 at ep15, with ep25 reaching 0.101 late.

**Verdict**: CFG fix produced the best tencent training trajectory in recent runs (3 consecutive stars, combined reaching 0.101). But W instability killed it — the elevated W throughout (1.7-2.0 vs typical 0.7-1.3) suggests the CFG fix changes critic dynamics on tencent. W may need dampening (lower lr_d or stronger grad clip). No eval warranted due to W-stop.

---

## Post-Mortem: tencent_v131 — base ATB recipe, fresh seed roll #5 (killed ep23, best 0.117★ ep10)

**Recipe**: v105 base PCF recipe. Fresh seed roll. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.138★ (recall=0.454), ep10=0.117★ (recall=0.480). After ep10, combined regressed: ep15=0.120, ep20=0.124, ep22 W=1.29. W stable overall (0.63-1.41) but quality plateau. Killed at ep23 (13 stale, regressing).

**Verdict**: Typical tencent seed roll — promising early (0.117★ at ep10) but regressed after. No eval warranted. Killed to pivot from seed rolling to structural CFG information leakage fix (Gemini Round 2 P1).

---

## Post-Mortem: alibaba_v103 — v71 base recipe, fresh seed roll #5 (killed during warmup)

**Recipe**: Identical to v71/v98 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Completed AE pretrain + supervisor pretrain. Reached G warm-up ep90/100. Never entered GAN phase. Killed to pivot to CFG fix — no training data to evaluate.

**Verdict**: Killed before GAN phase to pivot to structural CFG information leakage fix. Pretrain was valid but no GAN data generated.

---

## Post-Mortem: alibaba_v102 — v71 base recipe, fresh seed roll #4 (killed ep30, best 0.093★ ep10)

**Recipe**: Identical to v71/v98 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Outstanding early trajectory. Stars at ep5=0.119★ (recall=0.538), ep10=0.093★ (recall=0.595). Best training combined 0.093 — the best ever on alibaba, beating v101's 0.097. ep15=0.123 (miss), ep20=0.097 (recall=0.609, near-miss), ep25=0.111, ep30=0.111. Plateaued around 0.097-0.111 after ep10. W stable throughout (never exceeded 2.1). Killed at ep30 (20 stale).

**Eval (5-run avg): combined=0.113** (range 0.079–0.156). Individual: 0.106, 0.079, 0.156, 0.120, 0.106. Avg recall=0.517. **Run 2 hit 0.079 — best individual eval ever** (recall=0.642, MMD²=0.007). Run 3 was outlier (recall=0.377). Train→eval gap +22% (0.093→0.113). Does NOT beat ATB 0.088.

**Verdict**: Best training seed ever (0.093★) and produced the best individual eval run ever (0.079). But eval variance (0.079–0.156) continues to prevent ATB. The recipe is clearly capable of ATB-beating quality — the bottleneck is eval consistency, not model capacity. Fifth consecutive alibaba seed roll confirming structural train→eval gap.

---

## Post-Mortem: tencent_v130 — base ATB recipe, fresh seed roll #4 (killed ep25, best 0.122★ ep10)

**Recipe**: v105 base PCF recipe. Fresh seed roll. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.173★ (recall=0.213), ep10=0.122★ (recall=0.495). Strong early trajectory — ep10 combined 0.122 was the best early tencent result seen. But recall collapsed: ep15=0.123 (recall=0.446), ep20=0.150 (recall=0.314), ep25=0.151 (recall=0.335). W spiked to 2.67 at ep22. Killed at ep25 (15 stale).

**Verdict**: Promising early start (0.122★ at ep10) but recall collapsed from 0.495 to 0.335 after ep10. Classic tencent pattern — good early recall that doesn't sustain. No eval warranted.

---

## Post-Mortem: alibaba_v101 — v71 base recipe, fresh seed roll #3 (killed ep71, best 0.097★ ep45)

**Recipe**: Identical to v71/v98 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Outstanding trajectory with 6 stars: ep5=0.131★ (recall=0.462), ep15=0.126★ (recall=0.509), ep20=0.123★ (recall=0.451), ep25=0.119★ (recall=0.513), ep40=0.105★ (recall=0.544), ep45=0.097★ (recall=0.585). Best training combined 0.097 is the best ever seen on alibaba. After ep45 plateaued around 0.100-0.112 with recall holding at 0.578-0.600 — quality plateau, not degradation. W stable throughout (never exceeded 2.5). Killed at ep71 after eval.

**Eval (5-run avg): combined=0.123** (range 0.108–0.161). Individual: 0.161, 0.109, 0.108, 0.109, 0.127. Avg recall=0.457. Run 1 was outlier (recall=0.310). Runs 2-4 clustered at 0.108-0.109. Train→eval gap +27% (0.097→0.123). Does NOT beat ATB 0.088.

**Verdict**: Best training seed ever (0.097★, recall 0.585) but train→eval gap (+27%) prevents ATB. Third consecutive alibaba run confirming the pattern: recipe can produce sub-0.100 training results, but eval variance (especially recall) pulls the average up. The gap appears structural, not seed-dependent.

---

## Post-Mortem: tencent_v129 — base ATB recipe, fresh seed roll #3 (killed ep67, best 0.134★ ep50)

**Recipe**: v105 base PCF recipe. Fresh seed roll. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.175★ (recall=0.288), ep10=0.169★ (recall=0.257), ep15=0.153★ (recall=0.359), ep30=0.142★ (recall=0.372), ep45=0.138★ (recall=0.460), ep50=0.134★ (recall=0.430). Then W instability: W hit 3.98 at ep57, 3.20 at ep63, 3.30 at ep64, 3.30 at ep67. G_loss hit 4.03. ep55=0.135, ep60=0.145, ep65=0.148 — all regressed from best. Killed at ep67 (17 stale).

**Verdict**: Produced 6 stars with steady improvement (0.175→0.134) but W became increasingly unstable in second half. Best training combined 0.134 is still far from ATB 0.098. No eval warranted — even without train→eval gap, 0.134 >> 0.098. Seed rolling continues.

---

## Post-Mortem: alibaba_v100 — v71 base recipe, fresh seed (killed ep41, best 0.090★ ep20)

**Recipe**: Identical to v71/v98 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Stars at ep5=0.113★ (recall=0.553), ep10=0.118 (no star), ep20=0.090★ (recall=0.611, MMD²=0.012). Then regression: ep25=0.110, ep30 regressed further, ep35=0.125 (recall=0.503), ep40=0.100. W spiked to 3.41 at ep38. Killed at ep41 (21 stale).

**Eval (5-run avg): combined=0.110** (range 0.087–0.133). Individual: 0.110, 0.113, 0.133, 0.087, 0.106. Avg recall=0.498. Run 4 hit 0.087 (would beat ATB alone!) but avg recall variance (0.401–0.593) drags average down. Train→eval gap +22% (0.090→0.110). Does NOT beat ATB 0.088.

**Verdict**: Second-best alibaba seed after v98. Excellent ep20 star (0.090) but couldn't sustain it — W instability after ep35 and recall variance in eval. Demonstrates that seeds producing train 0.090 can eval 0.087 on lucky draws, confirming recipe strength.

---

## Post-Mortem: tencent_v128 — base ATB recipe, fresh seed (killed ep28, best 0.151★ ep15)

**Recipe**: v105 base PCF recipe. Fresh seed roll. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.177★ (recall=0.207), ep10=0.175★ (recall=0.233), ep15=0.151★ (recall=0.308). Then collapse: ep20=0.162 (recall=0.262), ep25=0.162 (recall=0.256). Recall went DOWN after ep15 — not just stalling, actively deteriorating. W climbed to 2.4. Killed at ep28 (13 stale).

**Verdict**: Very weak seed. Recall never exceeded 0.308 — far below v127's 0.558 or ATB levels. The recall collapse (0.308→0.256) after ep15 is the hallmark of a dead seed. No eval warranted.

---

## Post-Mortem: alibaba_v99 — v71 base recipe, fresh seed (killed ep39, best 0.108★ ep25)

**Recipe**: Identical to v71/v98 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Stars at ep5=0.126★ (recall=0.487), ep10=0.124★ (recall=0.480), ep20=0.108★ (recall=0.568), ep25=0.108★ (recall=0.556). Then regression: ep30=0.121 (recall=0.471), ep35=0.135 (recall=0.494, MMD²=0.034). W spiked to 2.65 at ep36. G_loss unusually high (7.5-9.6) — very different from v98's negative-G pattern. PCF crossed 1.0. Killed at ep39 (14 stale).

**Verdict**: Weak seed. G_loss stayed very high positive (vs v98's negative G_loss), suggesting different adversarial dynamic. Best train 0.108 is 23% from ATB 0.088, and with systemic ~48% train→eval gap would not be competitive. Seed rolling continues.

---

## Post-Mortem: tencent_v127 — base ATB recipe, fresh seed (killed ep51, best 0.100★ ep35)

**Recipe**: v105 base PCF recipe. Fresh seed roll. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.125★ (recall=0.468), ep15=0.102★ (recall=0.547), ep30=0.101★ (recall=0.558), ep35=0.100★ (recall=0.533, MMD²=0.007). Then collapse: ep40=0.109, ep45=0.137 (recall=0.453), ep50=0.117 (recall=0.452). W climbed from 1.5 to 2.5+. Killed at ep51 (16 stale).

**Eval (3-run): avg combined ~0.148** (individual: 0.143, 0.184, 0.117). Run 2 had mode collapse warning (recall=0.224). Massive train→eval gap (0.100 train → 0.148 eval, +48%). Does NOT beat ATB 0.098.

**Verdict**: Best tencent training trajectory in a while (four consecutive stars ep5→ep35) but train→eval gap killed it. Same systemic gap pattern. Seed rolling continues.

---

## Post-Mortem: alibaba_v98 — v71 base recipe, fresh seed (killed ep60, best 0.089★ ep30) ★★★ NEW ALIBABA ATB ★★★

**Recipe**: Identical to v71 ATB recipe. var-cond-kl-weight=0.01, no det_prob. Fresh seed.

**Training-log**: Stars at ep5=0.119★, ep10=0.089★ (recall=0.617), ep30=0.089★ (recall=0.631). Sustained high recall (0.576-0.637) throughout — best sustained recall ever. W volatile (spiking to 3.5-3.8 multiple times but recovering). ep40=0.102, ep50=0.097, ep55=0.133 (collapse), ep60=0.099. Killed at ep60 (30 stale, W spiking 3.6-4.5). G_loss stayed negative throughout — unusual training dynamic.

**Eval (5-run avg): combined=0.088 ★★★ NEW ALIBABA ATB ★★★** (range 0.064–0.112). Individual runs: 0.075, 0.106, 0.064, 0.080, 0.112. Avg recall=0.611 (vs v71's ~0.50). Avg MMD²=0.010 (vs v71's ~0.015). Beats v71's true 5-run avg of 0.095 by **7.4%**.

**Verdict**: Same v71 recipe, different seed, dramatically better eval. Confirms that eval variance is seed-dependent and the recipe itself is strong. The high sustained recall (0.611 avg) is the key differentiator vs v71. This is now the alibaba ATB checkpoint.

---

## Post-Mortem: tencent_v126 — z_global det_prob fix (killed ep36, best 0.137★ ep15)

**Recipe**: v105 base PCF recipe + `--var-cond-det-prob 0.3`. Standard lr 8e-5/4e-5, files_per_epoch=12, mixed-type-recovery, n-regimes=8.

**Training-log**: Stars at ep5=0.160★ (no star, first eval), ep10=0.159 (no star), ep15=0.137★. Then stagnation: ep20=0.163, ep25=0.152, ep30=0.144, ep35=0.142. Recall stuck 0.308-0.424 — never recovered from ep20 collapse. W stable 1.4-2.5. Killed at ep36 (21 stale).

**Verdict**: z_global det_prob fix failed on tencent too. Best 0.137 not competitive with ATB (0.098). The det_prob fix doesn't help either corpus — eval variance is not caused by the CondEncoder train/eval mismatch.

---

## Post-Mortem: alibaba_v97 — z_global det_prob fix (killed ep55, best 0.088★ ep30)

**Recipe**: v71 base PCF recipe + `--var-cond-det-prob 0.3`. 30% of training batches use deterministic μ in CondEncoder. Standard lr 8e-5/4e-5, files_per_epoch=12, var-cond-kl-weight=0.01.

**Training-log**: Stars at ep5=0.118★, ep10=0.110★, ep20=0.097★, ep30=0.088★ (MMD²=0.011, recall=0.613 — best recall ever). Then regression: ep35=0.101, ep40=0.109, ep45 no star, ep50=0.114, ep55=0.117. Killed at ep55 (25 stale).

**Eval (6-run avg): combined=0.111** (range 0.082–0.140). Single-run evals: 0.088, 0.140, 0.106, 0.082, 0.132, 0.120. Precision 0.925 (excellent) but recall variance massive (0.398–0.639). First eval (0.088) was a lucky draw. **Does NOT beat ATB (0.095).**

**Verdict**: The det_prob fix produced exceptional train metrics (0.088★, recall 0.613) but eval variance remains massive. The z_global train/eval mismatch is NOT the dominant source of the gap — the fundamental issue is eval-time sampling variance in recall. The fix is architecturally clean but doesn't solve the problem. det_prob=0.3 goes into the "tried, didn't beat ATB" bucket.

---

## Post-Mortem: tencent_v125 — standard lr + files_per_epoch=12 (killed ep30, best 0.115★ ep10)

**Recipe**: v105 base PCF with standard lr 8e-5/4e-5, files_per_epoch=12. Restarted from v86 pretrain after process cleanup.

**Training-log**: Stars only at ep5=0.156★, ep10=0.115★. Then stagnation: ep15=0.138, ep20 saved checkpoint but no star, ep25=0.151, ep30=0.145. W spiked to 3.40 at ep26 (exceeding W-stop threshold of 3.0) but recovered to 2.72. Recall collapsed from 0.481 (ep10) to 0.298 (ep25) to 0.337 (ep30). Killed at ep30 (20 epochs stale).

**Verdict**: Standard tencent recipe without z_global fix. W instability and recall collapse typical of standard lr runs. No eval run — clearly wouldn't beat ATB (0.098). Replaced by tencent_v126 with z_global det_prob fix.

---

## Post-Mortem: alibaba_v96 — standard lr + files_per_epoch=24 (killed ep30, best 0.097★ ep15)

**Recipe**: v71 base PCF recipe with standard lr 8e-5/4e-5 but files_per_epoch=24 (Round 13 broader coverage). Using v48 pretrain.

**Training-log**: Stars at ep5=0.125★, ep10=0.117★, ep15=0.097★ (MMD²=0.012, recall=0.575). Best train metric EVER across all versions. Then regression: ep20=0.107, ep25=0.105, ep30=0.104. W climbing steadily: 0.91→1.39→1.54→1.69→1.95→2.01. Killed at ep30 (15 stale).

**Eval: combined=0.119** (MMD²=0.018, recall=0.494, precision=0.678). Train→eval gap: **+22%** (0.097→0.119). Standard lr gap persists even with broader sampling. Best-ever train metric (0.097) produced mediocre eval. Doesn't beat ATB (0.095).

**Verdict**: Broader sampling + standard lr produced extraordinary train quality (0.097) — better than even ATB single-run evals. But the train→eval gap remains substantial at standard lr. This conclusively proves: config-space is exhausted. Lower lr fixes gap but caps quality; standard lr has quality but gap; broader sampling helps train but not eval. **Structural code changes are the only remaining path.**

---

## Post-Mortem: tencent_v124 — lower lr + files_per_epoch=24 (killed ep46, best 0.114★ ep35)

**Recipe**: v123 + files_per_epoch=24 (Round 13 broader coverage). Lower lr 6e-5/3e-5. Using v86 pretrain.

**Training-log**: Five consecutive stars: ep5=0.133★, ep10=0.144, ep15=0.130★, ep20=0.122★, ep25=0.127, ep30=0.117★, ep35=0.114★. Remarkable streak. Then W spiked to 2.93 at ep39 (near W-stop 3.0), and quality collapsed: ep40=0.124, ep45=0.155, recall 0.490→0.355. Killed at ep46 (11 epochs stale).

**Eval: combined=0.145** (MMD²=0.015, recall=0.353, precision=0.873). Train→eval gap: **+27%** (0.114→0.145). Worse than v123's +3.7%. Broader sampling increased eval gap on tencent — opposite of what Round 13 predicted. Doesn't beat ATB (0.098).

**Verdict**: Broader sampling (24 files) produced best tencent train quality (0.114) but the W=2.93 spike broke training, and eval gap was worse than v123. The 24-file recipe improves train quality but doesn't generalize. Config-space largely exhausted.

---

## Post-Mortem: alibaba_v95 — lower lr + files_per_epoch=24 (killed ep25, best 0.103★ ep5)

**Recipe**: v94 + files_per_epoch=24 (Round 13 broader coverage). Lower lr 6e-5/3e-5. Using v48 pretrain.

**Training-log**: Stars only at ep5=0.103★ (MMD²=0.010, recall=0.533). Then regression: ep10=0.108, ep15=0.106, ep20=0.106, ep25=0.114. MMD² climbed from 0.010 to 0.028. Killed at ep25 (20 epochs stale).

**Eval: combined=0.106** (MMD²=0.014, recall=0.541, precision=0.627). Train→eval gap: **+2.9%** (0.103→0.106). Low gap confirmed again. But precision dropped to 0.627 (vs v94's 0.780) — broader sampling makes fakes more diverse but less plausible. Doesn't beat ATB (0.095).

**Verdict**: Broader sampling (24 files) gave a strong ep5 (0.103 vs v94's 0.126) but couldn't sustain it. Config-space is largely exhausted: lower lr fixes gap but caps quality; standard lr has quality but gap. Next: try standard lr + 24 files, then structural code changes.

---

## Post-Mortem: tencent_v123 — lower lr (killed ep48, best 0.136★ ep30)

**Recipe**: v105 but lr-g 6e-5 / lr-d 3e-5. Using v86 pretrain. Mixed-type-recovery enabled.

**Training-log**: Stars at ep5=0.205★, ep10=0.165★, ep15=0.157★, ep25=0.150★, ep30=0.136★. Recall peaked at 0.443 (ep30). After ep30, regression: ep35=0.159, ep40=0.164, ep45=0.174. W oscillating 1.2–2.5 with spikes. Killed at ep48 (18 epochs stale).

**Eval: combined=0.141** (MMD²=0.015, recall=0.368, precision=0.808). Train→eval gap: **+3.7%** (0.136→0.141). Lower lr reduces gap dramatically vs standard lr runs. But base quality (0.141) far from ATB (0.098). Lower lr alone is not enough — needs structural quality improvement.

---

## Post-Mortem: alibaba_v94 — base PCF + lower lr (killed ep26, best 0.116★ ep10)

**Recipe**: v71 base PCF but lr-g 6e-5 / lr-d 3e-5 (no BayesGAN). Using v48 pretrain.

**Training-log**: 3 stars: ep5=0.126★ (recall=0.468), ep10=0.116★ (MMD²=0.015, recall=0.495). Then regressed — ep15=0.147, ep20=0.141, ep25=0.122. MMD² doubled from 0.015 to 0.031-0.037. Killed at ep26 (16 epochs stale).

**Eval: combined=0.115** (MMD²=0.014, recall=0.493, precision=0.780).

**KEY FINDING: train→eval gap <1%** (0.116 train → 0.115 eval). Lower lr completely eliminates the generalization gap that plagued all standard-lr runs (+40-80%). But base quality (0.115) doesn't beat ATB (0.095). The lower lr recipe needs a quality-improving structural change (z_global fix, locality engine) to capitalize on the zero-gap property.

---

## Post-Mortem: alibaba_v93 — BayesGAN 3 + lower lr (killed ep43, best 0.090★ ep25)

**Recipe**: v91 + lr-g 6e-5 / lr-d 3e-5. Using v48 pretrain.

**Training-log**: 4 stars: ep5=0.163★, ep10=0.123★, ep15=0.101★, ep25=0.090★. Matched v91's ep20 result 5 epochs earlier. W remarkably stable (0.52-0.68) through ep25, then climbed to 0.88-1.22. ep30=0.115, ep35=0.104, ep40=0.106. Killed at ep43 (18 epochs stale).

**5-run eval avg: combined=0.108** (range 0.093-0.144). Recall avg=0.536. Train→eval gap: **+20%** (0.090→0.108). Worse than v91's 0.100. **Lower lr + BayesGAN doesn't beat base v71 (0.095).** The lower W stability was real but didn't translate to better eval quality.

---

## Post-Mortem: tencent_v122 — proj-critic, no BayesGAN (W-stopped ep5, FAILED)

**Recipe**: v105 + --proj-critic (no BayesGAN). Using v86 pretrain.

**Training-log**: W exploded catastrophically: ep1=0.99, ep2=2.70, ep3=5.06, ep4=9.51, ep5=19.29. G loss spiraled to -176. W-stopped at ep5.

**Projection discriminator is DEAD on both corpora.** Failed with BayesGAN (v92, W=8.0 ep5) and without (v122, W=19.3 ep5). The proj-critic makes the discriminator too powerful for the current generator architecture. Would need significant architectural changes (larger G, different training schedule) to work.

---

## Post-Mortem: tencent_v121 — BayesGAN 3 particles (killed ep40, best 0.111★ ep25)

**Recipe**: v105 + bayes-critics 3. Using v86 pretrain.

**Training-log**: 5 consecutive stars: ep5=0.168★, ep10=0.146★, ep15=0.128★, ep20=0.124★, ep25=0.111★. Then plateaued: ep30=0.126, ep35=0.123, ep40=0.124. W elevated 1.2-1.7 throughout. Killed at ep40 (15 epochs stale).

**5-run eval avg: combined=0.136** (range 0.126-0.141). Worse than v120's 0.126 (5 particles). **BayesGAN doesn't work for tencent** — neither 3 nor 5 particles beat ATB (0.098). The SGLD noise injection may interfere with mixed-type-recovery.

---

## Post-Mortem: alibaba_v92 — BayesGAN 3 + proj-critic (W-stopped ep5, FAILED)

**Recipe**: v91 + --proj-critic. Using v48 pretrain.

**Training-log**: W exploded immediately: ep1=0.61, ep2=2.11, ep3=3.30, ep4=6.22, ep5=8.02. W-stopped at ep5. Only eval: combined=0.157 (MMD²=0.052, recall=0.472).

**Projection discriminator + BayesGAN is incompatible.** The proj-critic conditions on workload type, making the critic too powerful. BayesGAN's SGLD noise couldn't compensate — W diverged catastrophically. Proj-critic may work without BayesGAN but that's a separate experiment.

---

## Post-Mortem: alibaba_v91 — BayesGAN 3 particles (killed ep40, best 0.086★ ep25)

**Recipe**: v71 + bayes-critics 3. Using v48 pretrain.

**Training-log**: 4 stars: ep5=0.169★, ep10=0.113★, ep20=0.101★, ep25=0.086★ (recall=0.648, ATR). ep30=0.097, ep35=0.099, ep40=0.117 — regression. W climbed from 0.8 to 1.2-1.6 range. Killed at ep40 (15 epochs stale).

**5-run eval avg: combined=0.100** (range 0.071-0.122). Best single run 0.071 (nearly matching v71's 0.067). Recall avg=0.582, MMD² avg=0.016. Train→eval gap: **+16%** (0.086→0.100) — dramatically better than non-BayesGAN runs (40-80%). BayesGAN confirmed to reduce train→eval divergence.

**Doesn't beat v71 ATB** (true avg 0.095 vs 0.100) but validates BayesGAN approach. Next: add projection discriminator (alibaba_v92).

---

## Post-Mortem: tencent_v120 — BayesGAN 5 particles (killed ep50, stalled at 0.101)

**Recipe**: v105 + bayes-critics 5 (SGLD noise injection). Using v86 pretrain.

**Training-log**: 6 stars: ep5=0.128★, ep15=0.123★, ep20=0.111★, ep35=0.101★, ep40=0.0996★. W exceptionally stable (<1.0 for 43 epochs). But stalled after ep35 — ep45 regressed to 0.113, ep50=0.110. Killed at ep50 (15 epochs stale).

**5-run eval avg: combined=0.126** (range 0.108-0.154). Recall avg=0.436, MMD² avg=0.013. Does NOT beat ATB (0.098). Training EMA of 0.101 was optimistic — real performance ~25% worse.

**BayesGAN 5 particles on tencent**: W stability is exceptional but final quality doesn't translate to eval. The critic ensemble may be too smooth, allowing G to find training shortcuts that don't generalize. Trying 3 particles (tencent_v121).

---

## Post-Mortem: alibaba_v90 — BayesGAN 5 particles (killed ep14, recall=0.000 FAILED)

**Recipe**: v71 + bayes-critics 5 (SGLD noise injection). Using v48 pretrain.

**Training-log**: recall=0.000 at both ep5 (MMD²=0.348, combined=0.548) and ep10 (MMD²=0.306, combined=0.507). W never exceeded 0.72 — the 5-particle critic ensemble provided no meaningful training signal. Generator produced garbage throughout.

**BayesGAN with 5 particles is too weak for alibaba.** The SGLD noise smooths out the critic signal to the point where G can't learn. Tencent works fine with 5 particles (W=0.5-0.8, improving). Retrying with 3 particles (alibaba_v91).

---

## Post-Mortem: alibaba_v89 — self-diag temp=0.1 (W-exploded ep3, FAILED)

**Recipe**: v71 + self-diag-temp 0.1. Using v48 pretrain.

**Training-log**: W exploded immediately: ep1=1.16, ep2=2.09, ep3=5.11. Killed at ep3. **Self-diag is DEAD on both corpora** — upweighting mechanism fundamentally incompatible with PCF recipe at any temperature.

---

## Post-Mortem: tencent_v119 — self-diag temp=0.1 (W-exploded ep3, FAILED)

**Recipe**: v105 + self-diag-temp 0.1.

**Training-log**: W exploded immediately: ep1=1.17, ep2=3.94, ep3=7.46. Killed at ep3. Even temp=0.1 destabilizes tencent critic. **Self-diag is fundamentally incompatible with tencent** — the upweighting concentrates critic updates on outlier samples, destabilizing the already fragile mixed-type-recovery critic.

---

## Post-Mortem: alibaba_v88 — v71 verbatim seed #11 (W-stopped ep55, eval avg 0.145)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.080★** ep45 (MMD²=0.010, recall=0.648). Five consecutive stars: ep5=0.118★, ep10=0.108★, ep15=0.093★, ep30=0.091★, ep45=0.080★. Longest sustained improvement of any seed. W elevated from ep48: 3.78→2.63→3.45→3.29→3.55→3.08→3.54. W-stopped at ep55.

**Full eval (3-run avg): combined≈0.145** (individual runs: 0.161, 0.123, 0.152). Train→eval gap: **+81%**. Despite best training metric (0.080) of any seed, eval is worse than v87 (0.135) and v71 (0.095). 6th consecutive alibaba seed failure. Seed rolling definitively exhausted.

---

## Post-Mortem: tencent_v118 — self-diag temp=1.0 (W-stopped ep6, FAILED)

**Recipe**: v105 + self-diag-temp 1.0. First structural experiment after seed rolling.

**Training-log**: W exploded immediately: ep4=4.95, ep5=6.15, ep6=9.29. W-stopped at ep6. Only one star at ep5 (combined=0.143).

**Self-diag temp=1.0 is too aggressive.** The softmax upweighting with temp=1.0 creates extreme weight concentration on outlier real samples, destabilizing the critic. Retrying with temp=0.1 (tencent_v119) for gentler upweighting.

---

## Post-Mortem: tencent_v117 — v105 verbatim seed roll #4 (killed ep35, best 0.109)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, n_regimes 8, diversity 2.0, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.109★** ep20 (MMD²=0.010, recall=0.507). Four consecutive stars ep5-ep20 (0.134→0.129→0.112→0.109). Best early trajectory of any tencent seed. Then stalled 15 epochs. W spiked to 3.03/3.31 at ep31-32 (nearly auto-stopped), recovered to 1.98. ep35 eval=0.114, no star.

**Killed at ep35**: 15 epochs stale, W instability, recall declining. Best tencent training metric (0.109) of any seed roll, closest to ATB (0.098). 5th consecutive tencent seed roll failure. Seed rolling exhausted for both corpora.

---

## Post-Mortem: alibaba_v87 — v71 verbatim seed #10 (killed ep38, eval 0.124)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.077★** ep20 (MMD²=0.013, recall=0.681). Two stars: ep5=0.112★, ep20=0.077★. Recall 0.681 was highest of any seed. Then stalled 18 epochs — ep25=0.111, ep35=0.117. W elevated 1.9-2.7 from ep22 onward.

**Full eval: combined≈0.124** (MMD²=0.034, β-recall=0.548, α-precision=0.889, DMD-GEN=0.785, AutoCorr=0.049, Spectral=0.007, Context-FID=0.36, HRC-MAE=0.016). Train→eval gap: **+61%**.

**Precision exceptional (0.889, best ever) but recall collapsed 0.681→0.548.** 5th consecutive alibaba seed failure. Gap pattern: v84(+41%), v85(+46%), v86(+80%), v87(+61%). Seed rolling exhausted — gap is structural, not seed-specific.

---

## Post-Mortem: tencent_v116 — v105 verbatim seed roll #3 (killed ep29, best 0.120)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, n_regimes 8, diversity 2.0, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.120★** ep15 (MMD²=0.016, recall=0.482). Two stars: ep5=0.156★, ep15=0.120★. Then stalled 14 epochs. Eval at ep25 regressed to 0.159 (recall collapsed to 0.354). W elevated 2.0-2.6 throughout.

**Killed at ep29**: 14 epochs stale, eval regressing, recall in free-fall. No eval run. Fourth tencent seed roll failure (v114: 0.126, v115: 0.111, v116: 0.120). Same pattern each time — early promise then stall/regression. Seed rolling exhausted for tencent.

---

## Post-Mortem: alibaba_v86 — v71 verbatim seed #9 (W-stopped ep54, eval 0.155)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.086★** ep30 (MMD²=0.008, recall=0.607). Strong early convergence: ep5=0.172, ep10=0.114★, ep15=0.105★, ep30=0.086★. W elevated from ep31 onward (2.4-2.7), then spiked: ep52=4.11, ep53=3.92, ep54=3.33. W-stopped at ep54.

**Full eval: combined≈0.155** (MMD²=0.027, β-recall=0.359, α-precision=0.529, DMD-GEN=0.685, AutoCorr=0.049, Spectral=0.006, Context-FID=0.45, HRC-MAE=0.013). Train→eval gap: **+80%**.

**Best training combined (0.086) of any seed roll this session**, but eval gap is the worst yet (+80%). Recall collapsed from 0.607 (train) to 0.359 (eval). Pattern now confirmed across 4 alibaba seeds: v84 (+41%), v85 (+46%), v86 (+80%). Seed rolling alone will not beat ATB — the z_global train→eval divergence (PEER-REVIEW Round 5) must be addressed structurally.

---

## Post-Mortem: tencent_v115 — v105 verbatim seed roll #2 (killed ep36, best 0.111)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, n_regimes 8, diversity 2.0, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.111★** ep20 (MMD²=0.012, recall=0.505). Four consecutive stars ep5-ep20. Then stalled 16 epochs. W climbed: ep32=2.50, ep35=2.93. Recall collapsed to 0.365 at ep35. Combined regressed to 0.137.

**Killed at ep36**: 16 epochs stale, W approaching stop threshold, recall in free-fall. No eval run — clearly worse than ATB (0.098). Third tencent seed roll; pattern consistent with v114.

---

## Post-Mortem: alibaba_v85 — v71 verbatim seed #8 (W-stopped ep60, eval 0.133)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.091★** ep60 (MMD²=0.007, recall=0.581). Steady convergence: ep5=0.125, ep10=0.117★, ep25=0.111★, ep35=0.093★, ep55=0.091★, ep60=0.091★. W discipline excellent through ep50 (1.3-1.9) then spiked: ep55=3.05, ep58=3.30, ep59=4.44. W-stopped at ep60.

**Full eval: combined≈0.133** (MMD²=0.019, β-recall=0.429, α-precision=0.542, DMD-GEN=0.780, AutoCorr=0.047, Spectral=0.010, Context-FID=0.28, HRC-MAE=0.017). Train→eval gap: **+46%**.

**Best training trajectory this session** (steady convergence, excellent W discipline through ep50) but eval collapsed as usual. Precision dropped to 0.542, recall 0.581→0.429. The persistent train→eval gap (~40-50%) on all seeds suggests a structural issue (likely the z_global divergence between training and eval noted in PEER-REVIEW.md Round 5). Seed rolling continues but the gap problem is systemic, not seed-specific.

---

## Post-Mortem: tencent_v114 — v105 verbatim seed roll #1 (killed ep45, best 0.126)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, n_regimes 8, diversity 2.0, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.126★** ep25 (MMD²=0.011, recall=0.425). Combined stalled after ep25: ep30=0.138, ep35=0.138, ep40=0.135, ep45=0.135. W spiked to 3.17 at ep34 (nearly triggered W-stop). Recall stuck at 0.37-0.42. 20 epochs stale. Killed at ep45.

**Failed seed** — peaked early, W elevated (2.0-2.5 range), recall never recovered past 0.425. ATB is 0.098; best was 28% worse. Seed #2 next.

---

## Post-Mortem: alibaba_v84 — v71 verbatim seed #7 (killed ep42, best 0.092)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.092★** ep15 (MMD²=0.017, recall=0.624). Combined degraded after ep15: ep20=0.100, ep25=0.101, ep30=0.105, ep35=0.101, ep40=0.111. Recall collapsed 0.624→0.523. W climbing 1.2→2.5. 27 epochs stale. Killed at ep42.

**Failed seed** — peaked early, never converged toward ATB 0.067. Recall decay and rising W indicate critic overfit. Seed #8 next.

---

## Post-Mortem: tencent_v113 — PCF + copy-path (W-stopped ep3, no eval)

**Recipe**: v105 base (PCF 2.0, mixed-type-recovery, diversity 2.0, w-stop 3.0) + **--copy-path** (per-timestep reuse BCE + stride consistency gating). Using v86 pretrain.

**Training-log**: W exploded immediately: ep1=4.21, ep2=10.09, ep3=13.24. W-spike guard killed at epoch 3. No eval checkpoint produced. Reuse BCE loss=1.46, stride consistency=0.008 — losses were reasonable but adversarial training diverged instantly.

**COPY-PATH CAUSES IMMEDIATE W-SPIKE ON TENCENT.** The stride gating (stride × (1-reuse_prob)) creates a non-differentiable interaction between binary and continuous heads that destabilises the critic. Fixed the inplace autograd crash (read before clone) but the underlying approach is unstable. **Copy-path (#15) CLOSED** — every structural addition to the base recipes has failed (GP prior, continuity, feat-critic, copy-path). Seed rolling on proven recipes is the only remaining path.

---

## Post-Mortem: tencent_v112 — PCF + feat-critic + mixed-type (killed ep37, eval 0.157)

**Recipe**: v105 base (PCF 2.0, mixed-type-recovery, diversity 2.0, w-stop 3.0) + **--feat-critic-weight 0.5**. Using v86 pretrain.

**Training-log**: Best **0.135★** ep15 (MMD²=0.013, recall=0.391). Combined degraded after ep15: ep20=0.143, ep25=0.151, ep30=0.141, ep35=0.168. Recall collapsed 0.391→0.336. Killed at ep37.

**Full eval: combined≈0.157** (MMD²=0.014, β-recall=0.282 ⚠ mode collapse, α-precision=0.884, DMD-GEN=0.685, Context-FID=0.05). Train→eval gap: **+16%**.

**FEAT-CRITIC CAUSES MODE COLLAPSE ON BOTH CORPORA.** Same pattern as alibaba_v83: excellent quality metrics (DMD-GEN=0.685, Context-FID=0.05) but catastrophic recall loss. Dual critic makes G concentrate on high-quality in-distribution samples, sacrificing mode coverage. **Feat-critic CLOSED.**

---

## Post-Mortem: alibaba_v83 — PCF + feat-critic (killed ep57, eval 0.124)

**Recipe**: v71 base (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0) + **--feat-critic-weight 0.5**. Using v48 pretrain.

**Training-log**: Best **0.088★** ep10 (MMD²=0.012, recall=0.619). Combined oscillated 0.088-0.136 for 47 epochs without new star. W volatile 2.4-3.2, nearly W-stopped multiple times. Killed at ep57 (ep55 cutoff).

**Full eval: combined≈0.124** (MMD²=0.011, β-recall=0.438, α-precision=0.782, DMD-GEN=0.690, Spectral=0.004, Context-FID=0.12). Train→eval gap: **+41%**.

**FEAT-CRITIC IMPROVES QUALITY BUT KILLS RECALL.** Best-ever DMD-GEN (0.690), Spectral (0.004), and Context-FID (0.12). But recall collapsed train→eval (0.619→0.438). Dual critic makes G generate high-quality in-distribution samples at the cost of mode coverage. Combined 0.124 far from ATB 0.067. **Every structural addition to v71 recipe has failed on alibaba** (GP prior: 0.181, continuity: 0.130, feat-critic: 0.124). v71 base recipe with good seed remains the only path to ATB.

---

## Post-Mortem: tencent_v111 — PCF + continuity loss + mixed-type (killed ep35, eval 0.167)

**Recipe**: v105 base (PCF 2.0, mixed-type-recovery, diversity 2.0, w-stop 3.0) + **--continuity-loss-weight 0.5**. Using v86 pretrain.

**Training-log**: Best **0.170★** ep15 (MMD²=0.019, recall=0.244). Recall stuck 0.205-0.268 for 20 epochs — never improved. Combined oscillated 0.170-0.179. W spiked to 2.86 at ep30. Killed at ep35.

**Full eval: combined≈0.167** (MMD²=0.018, β-recall=0.256 ⚠ mode collapse, α-precision=0.745, DMD-GEN=0.738). Train→eval gap: **-2%**.

**CONTINUITY LOSS CAUSES MODE COLLAPSE ON TENCENT.** Recall stuck at 0.256 — much worse than v110's 0.366. The boundary-coherence constraint forces G to generate smoother, lower-diversity windows. **Continuity loss CLOSED for both corpora** (alibaba: precision collapse; tencent: mode collapse).

---

## Post-Mortem: alibaba_v82 — PCF + continuity loss (killed ep52, eval 0.130)

**Recipe**: v71 base (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0) + **--continuity-loss-weight 0.5**. Using v48 pretrain.

**Training-log**: Best **0.109★** ep25 (MMD²=0.026, recall=0.587). Combined oscillated after ep25: ep30=0.149, ep35=0.115, ep40=0.141, ep45=0.115, ep50=0.131. 27 epochs stale. W climbing 1.4→2.5. Killed at ep52.

**Full eval: combined≈0.130** (MMD²=0.014, β-recall=0.421, α-precision=0.609, DMD-GEN=0.679, AutoCorr=0.050, Spectral=0.005). Train→eval gap: **+19%**.

**CONTINUITY LOSS IMPROVES DMD-GEN BUT NOT COMBINED.** DMD-GEN improved from typical 0.74→0.679 — the continuity loss hit its target. But α-precision collapsed (0.609 vs typical 0.8+), suggesting the boundary-coherence constraint forces G to generate smoother but less realistic windows. Combined 0.130 far from ATB 0.067.

---

## Post-Mortem: tencent_v110 — PCF + GP prior + mixed-type (killed ep42, eval 0.147)

**Recipe**: v105 base (PCF 2.0, mixed-type-recovery, diversity 2.0, w-stop 3.0) + **--gp-prior**. Using v86 pretrain.

**Training-log**: Best **0.137★** ep20 (MMD²=0.020, recall=0.413). Combined degraded steadily after ep20: ep25=0.139, ep30=0.148, ep35=0.138, ep40=0.155. Recall collapsed: 0.413→0.316. W spiked to 3.51 at ep41. Killed at ep42.

**Full eval: combined≈0.147** (MMD²=0.020, β-recall=0.366, DMD-GEN=0.761, AutoCorr=0.057, Spectral=0.008). Train→eval gap: **+7%**.

**GP PRIOR FAILS ON TENCENT TOO.** Same pattern as alibaba_v81: GP-correlated z_local doesn't help and may hurt by overfitting noise structure to training data. Recall stuck ~0.37, never approaching the 0.5+ needed for ATB contention. Combined never improved past ep20 — the GP prior doesn't provide the mode coverage needed. **GP prior (#4) CLOSED for both corpora.**

---

## Post-Mortem: alibaba_v81 — PCF + GP prior (W-stopped ep47, eval 0.181)

**Recipe**: v71 base (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0) + **--gp-prior** (GP-sampled z_local, RBF kernel). Using v48 pretrain.

**Training-log**: Best **0.119★** ep40 (MMD²=0.032, recall=0.562). W volatile after ep40: ep42=3.98, ep43=2.84, ep44=2.95, ep45=3.18, ep46=4.16, ep47=3.29. W-spike guard killed at ep47.

**Full eval: combined≈0.181** (MMD²=0.049, β-recall=0.339, DMD-GEN=0.742, AutoCorr=0.059, Spectral=0.014). Train→eval gap: **+52%** (very bad).

**GP PRIOR HURTS ALIBABA.** Despite promising training dynamics (stable W through ep40, recall=0.562), eval collapsed: recall dropped from 0.562→0.339, MMD² from 0.032→0.049. The GP-correlated noise structure that helped training stability created an eval-time distribution mismatch — the model learned to rely on temporally correlated input noise that the GP provides, but eval-time sampling may not match. GP prior adds learnable parameters that overfit to training data order. Far worse than v71 ATB (0.067). Closing GP prior for alibaba.

---

## Post-Mortem: tencent_v109 — PCF + mixed-type v105 verbatim seed #3 (killed ep48, recall collapse)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, diversity 2.0, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.129★** ep15 (MMD²=0.010, recall=0.404). Combined stuck 0.129–0.134 for ep15-35. Recall collapsed ep35→ep40: 0.393→0.275. Combined worsened to 0.155 at ep40. Killed at ep48.

**RECALL COLLAPSE AGAIN.** Same pattern as v107 (also v105 verbatim). v105 recipe is not reproducible — 0/2 reproduction attempts both got recall collapse at ep35-52. The v105 ATB may have been a lucky seed. Switching to GP prior for tencent.

---

## Post-Mortem: alibaba_v80 — PCF loss v71 verbatim seed #5 (killed ep39, eval 0.107)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0, diversity 2.0). Using v48 pretrain.

**Training-log**: Best **0.109★** ep15. Combined degraded after ep15: ep20=0.109→ep25=0.123→ep30=0.117→ep35=0.135. Killed at ep39 (24 epochs past best, clear degradation).

**Full eval: combined≈0.107** (MMD²=0.020, β-recall=0.565, α-precision=0.767, DMD-GEN=0.761). Train→eval gap: **-2%** (minimal).

**EARLY PEAK PATTERN AGAIN.** 5th seed of v71 recipe, 4th to miss ATB. Seed variance results: {v71=0.067, v74=0.093, v80=0.107, v79=0.140, v78=0.158, v76=0.179}. Only 1 in 5 hits ATB (~20% success rate). Diminishing returns on seed rolling — switching to GP prior (#4) for structural improvement.

---

## Post-Mortem: tencent_v108 — PCF + mixed-type + diversity 3.0 (killed ep71, eval 0.117)

**Recipe**: v105 base (PCF 2.0, mixed-type-recovery) + **diversity-loss-weight 3.0** (was 2.0). Using v86 pretrain.

**Training-log**: Best **0.125★** ep45 (MMD²=0.009, recall=0.420). 26 epochs of plateau after peak: combined oscillated 0.134–0.143 with no improvement. W spikes at ep47=4.46, ep66=3.84, ep71=3.85. Killed at ep71.

**Full eval: combined≈0.117** (MMD²=0.010, β-recall=0.465, α-precision=0.765, DMD-GEN=0.777). Train→eval gap: **-6%** (negative = good, PCF effect working).

**DIVERSITY 3.0 PREVENTS COLLAPSE BUT CAPS IMPROVEMENT.** Unlike v107 (diversity 2.0, recall collapsed), v108's recall stayed healthy (0.35–0.46). But combined plateaued at 0.125 — never approaching ATB 0.098. Diversity 3.0 stabilizes training but doesn't improve peak quality. The v105 recipe (diversity 2.0) with a good seed remains the best tencent approach.

---

## Post-Mortem: alibaba_v79 — PCF + diversity 3.0 (W-stopped ep39, eval 0.140)

**Recipe**: v71 base (PCF 2.0, n_freqs 32, w-stop 3.0) + **diversity-loss-weight 3.0** (was 2.0). Using v48 pretrain.

**Training-log**: Best **0.079★** ep5. Combined hovered 0.090-0.093 at ep25-35. W spiked: ep34=3.66, ep37=4.02, ep38=3.77, ep39=3.27 → W-spike guard killed at ep39.

**Full eval: combined≈0.140** (MMD²=0.024, β-recall=0.419, α-precision=0.740, DMD-GEN=0.775).

**DIVERSITY 3.0 TOO AGGRESSIVE FOR ALIBABA.** Increased diversity loss elevated W throughout, causing W-stop at ep39 — far earlier than v71's sweet spot (ep55-73). Best.pt from ep5 was undertrained. Combined with v77 (mixed-type hurts alibaba) and v75 (moment hybrid destabilizes), the v71 recipe at diversity 2.0 remains optimal for alibaba. Seed variance is the remaining obstacle.

---

## Post-Mortem: alibaba_v78 — PCF loss (killed ep75, eval 0.158 — early peak, late degradation)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0). Using v48 pretrain.

**Training-log**: Best **0.099★** ep25. Stars at ep5→10→25. Combined degraded after ep50: ep55=0.121→ep65=0.138→ep75=0.146. Killed at ep75 (50/60 patience used, clear downward trend).

**Full eval: combined≈0.158** (MMD²=0.029, β-recall=0.355, α-precision=0.818, DMD-GEN=0.749). Train→eval gap: **+59%**.

**EARLY PEAK PROBLEM.** v78 peaked at ep25 — too early for robust temporal structure. v71/v74 peaked at ep55/ep73 and eval'd much better. Seed variance drives when the model peaks, and early peaks don't generalize.

---

## Post-Mortem: tencent_v107 — PCF + mixed-type recovery verbatim (killed ep52, recall collapse)

**Recipe**: v105 verbatim (PCF 2.0, mixed-type-recovery, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.135★** ep25. Recall collapsed monotonically: ep25=0.395→ep30=0.357→ep35=0.306→ep40=0.302→ep45=0.281→ep50=0.275. Killed at ep52.

**v105 RECIPE NOT REPRODUCIBLE on this seed.** v105 had a similar early plateau but surged at ep55-60. v107 never recovered — recall degraded continuously. High seed variance for mixed-type-recovery on tencent.

---

## Post-Mortem: alibaba_v77 — PCF + mixed-type recovery (W-stopped ep76, eval 0.182 — MIXED-TYPE HURTS ALIBABA)

**Recipe**: v71 base (PCF 2.0, w-stop 3.0) + **--mixed-type-recovery**. Using v48 pretrain.

**Training-log**: Best **0.101★** ep65 (best training combined ever for alibaba). Stars at ep5→10→15→20→45→65. G_loss went negative ep71-76. W-stopped at ep76.

**Full eval: combined≈0.182** (MMD²=0.061, β-recall=0.396, α-precision=0.900, DMD-GEN=0.703, HRC-MAE=0.011). Train→eval gap: **+80%**.

**MIXED-TYPE RECOVERY HURTS ALIBABA.** Despite the best training trajectory ever (comb=0.101), eval is terrible (0.182). The sigmoid head for obj_id_reuse has no signal on alibaba (real reuse=0.000) — it adds noise. On tencent (real reuse=3-5%), the same head helps (v105 tied ATB). **Corpus-specific recipe**: mixed-type-recovery for tencent only.

| Alibaba recipe comparison | Combined | Recipe | Gap |
|--------------------------|----------|--------|-----|
| **v71 (ATB)** | **0.067** | Pure PCF 2.0 | -32% |
| v74 | 0.093 | Pure PCF 2.0 | -30% |
| v77 (mixed-type) | 0.182 | PCF 2.0 + mixed-type | +80% |
| v75 (moment) | 0.183 | PCF 2.0 + moment | +27% |
| v76 (over-trained) | 0.179 | Pure PCF 2.0 (ep96) | +49% |

---

## Post-Mortem: tencent_v106 — PCF 3.0 (W-stopped ep37, eval 0.139 — PCF 3.0 TOO AGGRESSIVE)

**Recipe**: v99 base + **pcf-loss-weight 3.0** (was 2.0). Using v86 pretrain.

**Training-log**: Best **0.123★** ep10. W-stopped at ep37 (W>3.0 for 3 consecutive). G_loss went near-zero/negative by ep27. Only 10 useful epochs.

**Full eval: combined≈0.139** (MMD²=0.032, β-recall=0.464, α-precision=0.932, DMD-GEN=0.786, HRC-MAE=0.022). Train→eval gap: **+13%**.

**PCF 3.0 CONFIRMED DEAD.** Stronger PCF gradient destabilizes training — W escalates quickly, G collapses. PCF 2.0 remains the optimal weight for both corpora.

---

## Post-Mortem: tencent_v105 — PCF + mixed-type recovery (W-stopped ep60, eval **0.098 — TIED TENCENT ATB**)

**Recipe**: v99 base (PCF 2.0, n_freqs 32, w-stop 3.0) + **--mixed-type-recovery**. Using v86 pretrain.

**Training-log**: Best **0.098★** ep60. Stars at ep5→10, then 45-epoch plateau, breakthrough at ep55→60. W-stopped at ep60 (W>3.0 for 3 consecutive).

**Full eval: combined≈0.098** (MMD²=0.014, β-recall=**0.578**, α-precision=**0.868**, DMD-GEN=0.730, HRC-MAE=**0.005**, density=**1.490**). Train→eval gap: **0%** (no gap).

**MIXED-TYPE RECOVERY VALIDATES ON TENCENT.** Ties v103 ATB on combined but with substantially better metrics: recall +16% (0.578 vs 0.499), precision +3%, HRC-MAE 12× better (0.005 vs 0.058), density nearly 2× higher. The sigmoid head for obj_id_reuse produces sharper binary outputs that improve both distributional and cache-fidelity metrics.

| Tencent PCF variants | Combined | Recall | Precision | HRC-MAE | Density |
|---------------------|----------|--------|-----------|---------|---------|
| **v103 (prev ATB)** | **0.098** | 0.499 | 0.841 | 0.058 | — |
| **v105 (mixed-type)** | **0.098** | **0.578** | **0.868** | **0.005** | **1.490** |
| v99 | 0.112 | 0.470 | 0.846 | 0.050 | — |
| v104 | 0.134 | 0.404 | 0.837 | 0.057 | — |

---

## Post-Mortem: alibaba_v76 — PCF loss (W-stopped ep96, eval 0.179 — OVER-TRAINED)

**Recipe**: v71 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0). Using v48 pretrain.

**Training-log**: Best **0.120★** ep95. Stars at ep5→10→15→20→35→40→50→80→95. W-stopped at ep96 (W>3.0 for 3 consecutive). Ran 32 epochs longer than v71 due to W bouncing under threshold.

**Full eval: combined≈0.179** (MMD²=0.033, β-recall=0.269 ⚠ mode collapse, α-precision=0.496, DMD-GEN=0.782, HRC-MAE=0.011). Train→eval gap: **+49%**.

**OVER-TRAINING CONFIRMED.** v76 is v71 verbatim but ran to ep96 vs v71's ep64. The extra 32 epochs degraded the model: precision dropped from 0.926→0.496, recall from 0.701→0.269. The W-stop guard caught v71 at the sweet spot; v76's W bounced just under 3.0 long enough to over-train. Reinforces that early W-stopping (not late survival) produces the best models.

| Alibaba PCF verbatim | Combined | Stopped ep | Recall | Precision | Gap |
|---------------------|----------|------------|--------|-----------|-----|
| **v71 (ATB)** | **0.067** | 64 | **0.701** | **0.926** | -32% |
| v74 | 0.093 | 73 | 0.613 | 0.846 | -30% |
| v76 (over-trained) | 0.179 | 96 | 0.269 | 0.496 | +49% |

---

## Post-Mortem: tencent_v104 — PCF loss (W-stopped ep57, eval 0.134 — seed bundle point 3)

**Recipe**: v99 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.091★** ep45. Stars at ep5→10→20→30→35→45. W-stopped at ep57 (W=4.15 for 3 consecutive).

**Full eval: combined≈0.134** (MMD²=0.015, β-recall=0.404, α-precision=0.837, DMD-GEN=0.738, HRC-MAE=0.057, reuse=0.014/0.014). Train→eval gap: **+47%**.

**Tencent seed bundle complete.** Three points with the same recipe: {v99=0.112, v103=0.098, v104=0.134}. Mean ≈ 0.115, std ≈ 0.018. More variance than alibaba bundle. PCF recipe is validated but tencent shows higher seed sensitivity.

| Tencent PCF verbatim | Combined | Recall | Precision | HRC-MAE | Gap |
|---------------------|----------|--------|-----------|---------|-----|
| **v103 (ATB)** | **0.098** | 0.499 | 0.841 | 0.058 | +1% |
| v99 | 0.112 | 0.470 | 0.846 | 0.050 | -5% |
| v104 | 0.134 | 0.404 | 0.837 | 0.057 | +47% |

---

## Post-Mortem: alibaba_v75 — PCF + moment hybrid (W-stopped ep41, eval 0.183 — MOMENT HYBRID FAILED)

**Recipe**: v71 base (PCF 2.0, w-stop 3.0) + **moment-loss-weight 0.1**. Using v48 pretrain.

**Training-log**: Best **0.145★** ep15. Stars at ep5→10→15. G_loss spiked from ep37 (5.2→6.1→7.0→8.1). W-stopped at ep41 (W=4.05 for 3 consecutive). Only 15 useful epochs.

**Full eval: combined≈0.183** (MMD²=0.046, β-recall=0.314, α-precision=0.622, DMD-GEN=0.776, HRC-MAE=0.006). Train→eval gap: **+27%**.

**MOMENT HYBRID DEAD.** Adding moment-loss-weight 0.1 on top of PCF 2.0 destabilizes G within 37 epochs — same G_loss spiral as tencent_v100 (PCF 1.0). The PCF loss already captures distributional matching; adding moment matching creates competing gradients that amplify G instability. Pure PCF (no handcrafted aux losses) remains the optimal recipe.

| Alibaba PCF variants | Combined | Recipe | Stopped ep |
|---------------------|----------|--------|------------|
| **v71 (ATB)** | **0.067** | PCF 2.0 pure | 64 |
| **v74 (verbatim)** | **0.093** | PCF 2.0 pure | 73 |
| v72 (PCF 1.0) | 0.111 | PCF 1.0, w-stop 4.0 | 123 |
| v73 (w-stop 4.0) | 0.166 | PCF 2.0, w-stop 4.0 | 70 |
| **v75 (moment hybrid)** | **0.183** | PCF 2.0 + moment 0.1 | 41 |

---

## Post-Mortem: alibaba_v74 — PCF loss (W-stopped ep73, eval **0.093 — v71 REPRODUCIBILITY CONFIRMED**)

**Recipe**: v71 verbatim (PCF 2.0, grad-clip 0.5, w-stop 3.0). Using v48 pretrain.

**Training-log**: Best **0.132★** ep55. Stars at ep5→15→20→25→30→55. W-stopped at ep73 (W=3.95 for 3 consecutive).

**Full eval: combined≈0.093** (MMD²=0.015, β-recall=**0.613**, α-precision=**0.846**, DMD-GEN=0.702, HRC-MAE=**0.006**). Train→eval gap: **-30%** (negative — eval better than training).

**PCF RECIPE VALIDATED.** v74 is the second-best alibaba eval ever. The negative train→eval gap reproduces (-30% vs v71's -32%). Both runs show precision >0.84, recall >0.61, HRC-MAE <0.01. Seed/preprocessor variance accounts for v71's 0.067 vs v74's 0.093, but the recipe produces consistent top-tier results.

| Alibaba PCF verbatim | Combined | Recall | Precision | HRC-MAE | Gap |
|---------------------|----------|--------|-----------|---------|-----|
| **v71 (ATB)** | **0.067** | **0.701** | **0.926** | 0.010 | -32% |
| **v74 (verbatim)** | **0.093** | **0.613** | **0.846** | 0.006 | -30% |

---

## Post-Mortem: alibaba_v73 — PCF loss (killed ep70, eval 0.166 — w-stop 4.0 hurts)

**Recipe**: v71 verbatim (PCF 2.0) + **w-stop 4.0** (was 3.0). Using v48 pretrain.

**Training-log**: Best **0.115★** ep65. Stars at ep5→15→30→50→65. W climbing to 3.5 by ep70. Killed at ep70.

**Full eval: combined≈0.166** (MMD²=0.041, β-recall=0.373, α-precision=0.679, DMD-GEN=0.694, HRC-MAE=0.015). Train→eval gap: **44%**.

**KEY FINDING: w-stop 3.0 is OPTIMAL for alibaba PCF 2.0.** v71 was W-stopped at ep64 and eval'd at 0.067 (ATB). v73 trained 6 epochs longer with w-stop 4.0 and eval'd at 0.166 — 2.5× worse. The extra training past ep64 with elevated W degrades the model. The W-stop guard isn't just safety; it's capturing the model at its sweet spot before late-phase instability corrupts the learned distribution.

| Alibaba PCF eval | Combined | w-stop | Killed ep | Gap |
|-----------------|----------|--------|-----------|-----|
| **v71 (ATB)** | **0.067** | **3.0** | 64 | -32% |
| v72 (PCF 1.0) | 0.111 | 4.0 | 123 | +11% |
| v73 (PCF 2.0) | 0.166 | 4.0 | 70 | +44% |

---

## Post-Mortem: alibaba_v72 — PCF loss (killed ep123, eval 0.111 — PCF 1.0 too weak)

**Recipe**: v71 base + **pcf-loss-weight 1.0** (was 2.0) + **w-stop 4.0**. Using v48 pretrain.

**Training-log**: Best **0.100★** ep90 (MMD²=0.017, recall=0.584). Stars at ep5→10→20→45→50→60→65→90. W exceeded 4.0 at ep121. Killed at ep123.

**Full eval: combined≈0.111** (MMD²=0.018, β-recall=0.536, α-precision=0.855, DMD-GEN=0.705, HRC-MAE=0.009). Train→eval gap: 11%.

**KEY FINDING: PCF weight 2.0 > 1.0 for alibaba.** v71 (PCF 2.0) eval'd at 0.067 with a NEGATIVE 32% gap. v72 (PCF 1.0) eval'd at 0.111 with a normal 11% gap. Higher adversarial frequency pressure creates models that generalize BETTER — the learned frequency vectors find discriminative features that also improve out-of-sample quality.

| Alibaba eval | Combined | MMD² | Recall | Precision | PCF wt | Gap |
|-------------|----------|------|--------|-----------|--------|-----|
| **v71 (ATB)** | **0.067** | **0.007** | **0.701** | **0.926** | 2.0 | -32% |
| v72 | 0.111 | 0.018 | 0.536 | 0.855 | 1.0 | +11% |

---

## Post-Mortem: tencent_v103 — PCF loss (W-stopped ep58, eval **0.098 — NEW TENCENT ATB**)

**Recipe**: v99 verbatim (PCF 2.0, n_freqs 32, w-stop 3.0). Using v86 pretrain.

**Training-log**: Best **0.092★** ep30. Stars at ep5→10→25→30. W-stopped at ep58 (W=3.87 for 3 consecutive).

**Full eval: combined≈0.098** (MMD²=**0.008**, β-recall=**0.547**, α-precision=0.828, DMD-GEN=0.737, HRC-MAE=0.060, Context-FID=0.06). Train→eval gap: 7%.

**v99 VERBATIM BEAT v99 BY 13%.** New tencent ATB. MMD² nearly halved (0.008 vs v99's 0.014). PCF recipe validated on tencent with two data points. Seed variance favorable this time.

| Tencent PCF verbatim | Combined | MMD² | Recall | Precision | Gap |
|---------------------|----------|------|--------|-----------|-----|
| v99 | 0.112 | 0.014 | 0.508 | 0.904 | 18% |
| **v103 (new ATB)** | **0.098** | **0.008** | **0.547** | 0.828 | 7% |

---

## Post-Mortem: tencent_v102 — PCF loss (killed ep57, eval 0.146 — n_freqs 64 not helpful)

**Recipe**: v99 base (PCF 2.0) + **n_freqs 64** (was 32) + w-stop 4.0. Using v86 pretrain.

**Training-log**: Best **0.099★** ep30. 5 stars ep5→30. G_loss elevated (4-6). W spiked to 4.93 at ep56. Killed at ep57.

**Full eval: combined≈0.146** (MMD²=0.019, β-recall=0.365, α-precision=0.718, DMD-GEN=0.787, HRC-MAE=0.067). Train→eval gap: 47%.

**Finding: n_freqs=64 trains faster but evaluates worse.** Reached 0.099 at ep30 (v99 needed ep60 for 0.095) but the early checkpoint wasn't robust. More frequencies = faster discrimination learning = more instability. Tencent PCF recipe is: n_freqs=32, w-stop 3.0, PCF 2.0 (v99 original).

---

## Post-Mortem: tencent_v101 — PCF loss (killed ep81, eval 0.130 — worse than v99)

**Recipe**: v99 verbatim (PCF 2.0) + **w-stop 4.0** (was 3.0). Using v86 pretrain.

**Training-log**: Best **0.095★** ep40. 41 epochs without improvement. Killed at ep81. Same plateau pattern as v99.

**Full eval: combined≈0.130** (MMD²=0.020, β-recall=0.452, α-precision=0.686, DMD-GEN=0.792, HRC-MAE=0.063). Train→eval gap: 37%.

**Finding: w-stop 4.0 doesn't help tencent.** Peak came at ep40 (earlier than v99's ep60), but the early checkpoint wasn't robust (eval much worse). The tencent PCF recipe may need more frequency resolution to handle the 13x larger corpus diversity.

---

## Post-Mortem: tencent_v100 — PCF loss (killed ep75, eval 0.120 — worse than v99)

**Recipe**: v99 base + **pcf-loss-weight 1.0** (was 2.0) + **w-stop 4.0**. Using v86 pretrain.

**Training-log**: Best **0.105★** ep25. G_loss spiked to 5-8 after ep50, 50 epochs without improvement. Killed at ep75.

**Full eval: combined≈0.120** (MMD²=0.017, β-recall=0.486, α-precision=0.738, DMD-GEN=0.705, HRC-MAE=0.049). Train→eval gap: 14%.

**Finding: PCF weight 1.0 too weak for tencent.** Precision dropped to 0.738 (was 0.904 in v99). G_loss instability suggests insufficient distributional guidance. PCF weight 2.0 is the right setting for tencent.

---

## Post-Mortem: tencent_v99 — PCF loss (killed ep97, eval **0.112 — NEW TENCENT ATB**)

**Recipe**: v98 base (n-regimes 8) + **--pcf-loss-weight 2.0 --pcf-n-freqs 32 --grad-clip 0.5**. ALL handcrafted auxiliary losses ZEROED. Keeps feature-matching 1.0, supervisor 5.0, diversity 2.0. Using v86 pretrain.

**Training-log**: Best **0.095★** ep60 (MMD²=0.013, recall=0.588). 7 consecutive stars ep5→60. Killed at ep97 after 37 epochs without improvement. W spiking regularly above 3.0 (peaks: 3.47, 3.66, 3.46).

**Full eval: combined≈0.112** (MMD²=0.014, β-recall=0.508, α-precision=**0.904**, DMD-GEN=0.718, HRC-MAE=**0.0001**). Train→eval gap: 18%.

**PCF validates on tencent too.** Precision 0.904 (was 0.771 in v98) and near-zero HRC-MAE. Combined 0.112 beats v98's 0.146 by 23%. Not as dramatic as alibaba v71's breakthrough (0.067) but still a clear improvement. W volatility with PCF weight 2.0 suggests lower weight (1.0) may be better for tencent.

| Tencent eval | Combined | MMD² | Recall | Precision |
|-------------|----------|------|--------|-----------|
| v98 (unified) | 0.146 | 0.028 | 0.409 | 0.771 |
| **v99 (PCF)** | **0.112** | **0.014** | **0.508** | **0.904** |

---

## Post-Mortem: alibaba_v71 — PCF loss (W-stopped ep64, eval **0.067 — NEW ALL-TIME BEST**)

**Recipe**: v70 base (grad-clip 0.5) + **--pcf-loss-weight 2.0 --pcf-n-freqs 32**. ALL handcrafted auxiliary losses ZEROED. Adversarial PCF frequency training (v71 bug fix: freqs in C optimizer, gradient ascent). Using v48 pretrain.

**Training-log**: Best **0.098★** ep60 (MMD²=0.019, recall=0.603). W-spike guard killed at ep64 (W=3.23 for 3 consecutive). Stars at ep5→10→50→55→60.

**Full eval: combined≈0.067** (MMD²=**0.007**, β-recall=**0.701**, α-precision=**0.926**, density=1.464, DMD-GEN=0.702, HRC-MAE=0.010). Train→eval gap: **NEGATIVE** (eval 32% better than training — first time ever).

**THIS IS THE PROJECT BREAKTHROUGH.** PCF (path characteristic function) replacing 6 handcrafted auxiliary losses produces:
- 50% better combined than previous best eval (v65: 0.135)
- 82% better MMD² (0.007 vs 0.039)
- Recall 0.701 — generator covers 70% of real distribution (was 52%)
- Precision 0.926 — 93% of generated samples are plausible (was 54%)
- First negative train→eval gap in project history

| Alibaba eval | Combined | MMD² | Recall | Precision |
|-------------|----------|------|--------|-----------|
| v65 (prev best) | 0.135 | 0.039 | 0.521 | 0.544 |
| v69 (unified) | 0.163 | 0.019 | 0.280 | 0.770 |
| v70 (grad-clip) | 0.178 | 0.045 | 0.333 | 0.545 |
| **v71 (PCF)** | **0.067** | **0.007** | **0.701** | **0.926** |

---

## Post-Mortem: tencent_v98 — v93 recipe + UNIFIED z_global (completed ep200, eval **0.146 — best tencent unified**)

**Recipe**: v93 verbatim (KL 0.01, acf 0.3). Using v86 pretrain. UNIFIED z_global. Resumed from ep60 (DataLoader crash).

**Training-log**: Best **0.132★** ep165 (MMD²=0.013, recall=0.407). Stars at ep65→95→105→120→165. Late surge after 45-epoch plateau.

**Full eval: combined≈0.146** (MMD²=0.028, β-recall=0.409, α-precision=0.771, DMD-GEN=**0.661**, HRC-MAE=0.043). Train→eval gap: **11%** — BEST gap ever across both corpora. DMD-GEN 0.661 is also best ever (vs typical ~0.72).

**KEY FINDING: z_global unification works for tencent.** 11% gap vs v96 control's 47%. The unified eval path produces reliable checkpoint selection for tencent. The DMD-GEN improvement (0.721→0.661) is a bonus — the correct conditioning stack at eval time produces more temporally coherent generated sequences.

---

## Post-Mortem: alibaba_v70 — grad-clip 0.5 + UNIFIED z_global (killed ep136 — eval 0.178)

**Recipe**: v59 base (KL 0.01) + grad-clip 0.5 + w-stop 3.0. Using v48 pretrain. Stability-focused.

**Training-log**: Best **0.110★** ep65 (MMD²=0.014, recall=0.522). Remarkably stable W (0.27-0.90) through ep65 — then destabilized (W 1.8-2.8 from ep75+, W=3.07 at ep136). Killed ep136. The grad-clip 0.5 produced the best *training* combined score under unified eval (0.110), matching v59's old-eval record.

**Full eval: combined≈0.178** (MMD²=0.045, β-recall=0.333, α-precision=0.545, DMD-GEN=0.716, HRC-MAE=0.019). Train→eval gap: **62%** — WORSE than v69 (22%) and v65 (23%).

**CRITICAL FINDING: Training stability does NOT predict eval quality.** v70 was the most stable training run (W<0.9 for 65 epochs) and produced the best training ★ (0.110), yet eval was WORSE than v69 (0.163) and v65 (0.135). The best alibaba eval remains v65's 0.135. Training-time combined scores are unreliable predictors of eval performance — the eval gap is stochastic and potentially dominated by preprocessor/dataset sampling variance.

| Version | Grad-clip | Train ★ | Eval | Gap |
|---------|-----------|---------|------|-----|
| v65 (control) | 1.0 | 0.110 | **0.135** | 23% |
| v69 (unified) | 1.0 | 0.134 | 0.163 | 22% |
| v70 (unified+clip) | 0.5 | 0.110 | 0.178 | 62% |

---

## Post-Mortem: alibaba_v69 — v59 recipe + UNIFIED z_global (killed ep69 — destabilized)

**Recipe**: v59 verbatim (KL 0.01). Using v48 pretrain. FIRST RUN WITH UNIFIED z_global.

**Training-log**: Best **0.134★** ep40 (MMD²=0.020, recall=0.432). Stalled 29 epochs. W collapsed to 0.01-0.05 at ep56-65, then spiked to 2.4 at ep67-69. G_loss exploded to 7.8. Classic GAN oscillation — killed ep69.

**Full eval: combined≈0.163** (MMD²=0.019, β-recall=0.280, α-precision=0.770, DMD-GEN=0.744, HRC-MAE=0.012). Train→eval gap: **22%** — best gap we've seen (vs historical 30-75%), confirming z_global unification helps. But still worse than v59 unified re-eval (0.145). The mid-training instability degraded the EMA.

**Reuse**: real=0.006, fake=0.011 — no phantom reuse (unlike v68's copy-path).

**Key takeaway**: Unified z_global reduces train→eval gap (22% vs 30-75%), but training stability still matters. v70 launched with grad-clip 0.5 + w-stop 3.0 to prevent oscillation.

---

## Post-Mortem: alibaba_v68 — copy-path bce=0.5 (killed ep117 — stalled ep65, eval pending)

**Recipe**: v59 base (KL 0.01) + `--copy-path --reuse-bce-weight 0.5 --stride-consistency-weight 0.5`. Using v48 pretrain.

**Training-log**: Best **0.11270★** ep65 (MMD²=0.0179, recall=0.526). Stalled 52 epochs after ep65 — recall declined to 0.388, W rising to 2.5–3.5. Killed ep117.

**Full eval: combined≈0.198** (MMD²=0.054, β-recall=0.280, α-precision=0.326, DMD-GEN=0.699, HRC-MAE=0.083). Train→eval gap: **75%** — worse than v59 eval (0.111) and v65 control (0.135).

**CRITICAL: Copy-path actively harmful for alibaba.** real reuse=0.001, fake reuse=0.084 — model generated 84× more reuse than real data. Alibaba reuse is essentially zero; the class-weighted BCE amplified a near-zero rate into phantom reuse events. Copy-path should NOT be used on low-reuse corpora.

---

## Post-Mortem: tencent_v97 — copy-path bce=0.5 (killed ep101 — stalled, reuse_bce diverging)

**Recipe**: v93 base (KL 0.01, acf 0.3) + `--copy-path --reuse-bce-weight 0.5 --stride-consistency-weight 0.5`. Using v86 pretrain. (First launch at bce=2.0 collapsed W→11.8 by ep3.)

**Training-log**: Best **0.14346★** ep60 (MMD²=0.0256, recall=0.410). Stalled 40 epochs after ep60 — ep100 combined=0.150. recall peaked 0.41 then flat.

**CRITICAL FINDING: Copy-path reuse_bce NEVER declined.** Started at 9.3 (ep5), rose to 12.5 (ep101). The Generator/Recovery architecture cannot respond to per-timestep BCE on a column treated as just another regression target through a shared GRU. The distributional improvement (0.220→0.143) came entirely from the base recipe, not the copy-path mechanism. The reuse mechanism needs to be **architectural** (separate reuse decision head), not a loss added to the existing architecture.

---

## Post-Mortem: alibaba_v67 — copy-path bce=2.0 (killed ep63 — stalled, W trending up)

**Recipe**: v59 base (KL 0.01) + `--copy-path --reuse-bce-weight 2.0 --stride-consistency-weight 1.0`. Using v48 pretrain. First structural copy-path experiment.

**Training-log**: Best **0.16446★** ep35 (MMD²=0.0345, recall=0.350). No improvement for 28 epochs after ep35. W trending up to 3.2–4.3 range. reuse_bce stabilized at 2.8–4.2 (stopped declining). At ep60, v59 baseline was ~0.110★ — v67 was 63% behind at 0.179. Killed ep63.

**Lesson**: bce_weight=2.0 too aggressive for alibaba. The class-weighted BCE adds ~8–16 to G_loss early on (weight × bce ≈ 2.0 × 4–8), destabilizing the GAN equilibrium. Tencent v97 survived with bce_weight=0.5 and is still improving. Try 0.5 for alibaba (v68).

---

## Post-Mortem: alibaba_v66 — lower lr stability test (killed ep88 — too slow)

**Recipe**: v59 base (KL 0.01, moment-loss 0.2) + lr-g 6e-5, lr-d 3e-5 (75% of standard) + w-stop 3.0. Using v48 pretrain.

**Training-log**: Best **0.18605★** ep45 (recall=0.329). Training was stable (W 0.78-0.98, never spiked) but convergence far too slow — v65 was at 0.117★ by ep60 with the same recipe at standard lr. Killed at ep88. Lower lr stabilizes training but doesn't converge fast enough to produce competitive scores.

**Stability hypothesis partially confirmed, partially falsified:** Calmer training is achievable with lower lr, but the model needs sufficient lr to learn the data distribution within 200 epochs. The solution to the train→eval gap is not simply "train more gently" — it requires structural changes that are inherently more robust.

---

## Post-Mortem: tencent_v96 — v93 verbatim control (completed ep200, eval **0.1749 — did NOT reproduce v93**)

**Recipe**: v93 verbatim (KL 0.01, acf-loss 0.3). Using v86 pretrain.

**Training-log**: Best **0.11876★** ep115 (MMD²=0.01826, recall=0.497). Peaked early vs v93 (which peaked ep185). Stalled 85 epochs after ep115.

**Full eval: combined=0.1749** (MMD²=0.03493, β-recall=0.3000, α-precision=0.7395, density=0.7050, DMD-GEN=0.7213, HRC-MAE=0.0514). Train→eval gap: **47%** — recall collapsed from 0.497→0.300.

**CRITICAL FINDING: NEITHER corpus record is reproducible with verbatim controls.**

| Corpus | Original | Control | Reproduced? |
|--------|----------|---------|-------------|
| Alibaba | v59 eval 0.1113 | v65 eval 0.1351 | ✗ (23% worse) |
| Tencent | v93 eval 0.0995 | v96 eval 0.1749 | ✗ (76% worse) |

**The reviewer (Round 9) was exactly right:** scalar results are seed-dependent and generate false hope. Both "records" appear to be lucky seeds. The true reproducible baselines are uncertain — possibly alibaba ~0.12, tencent ~0.11-0.13 range.

**CONCLUSION: The scalar tuning era is definitively over.** Not because we've exhausted the search space (though we have), but because the results are not reproducible. Further improvement requires structural changes that produce robust, seed-independent gains.

---

## Post-Mortem: alibaba_v65 — v59 verbatim control (W-guard ep183, eval **0.1351 — did NOT reproduce v59**)

**Recipe**: v59 verbatim (KL 0.01, moment-loss 0.2). Using v48 pretrain.

**Training-log**: Best **0.11000★** ep180 (MMD²=0.01760, recall=0.538). W-spike guard killed at ep183 (W=4.34, 5.16, 4.93). Training was **unstable throughout** — W values 2-5, G_loss 6-8 from ep60 onward.

**Full eval: combined=0.1351** (MMD²=0.03918, β-recall=0.5205, α-precision=**0.5435**, density=0.5047, DMD-GEN=0.7537). Train→eval gap: 23%.

**KEY INSIGHT: Training-log matched v59 perfectly (both ~0.110★), but eval diverged massively (0.135 vs 0.111).** The difference: v65 was unstable (high W/G_loss) while v59 was likely calmer. **Training stability determines eval quality** — the EMA model absorbs noisy parameter states during high-W epochs, degrading eval. v59's eval of 0.1113 may be stability-dependent rather than recipe-dependent.

**Implication**: Lower learning rates and/or tighter W-stop thresholds may improve eval by keeping training stable.

---

## Post-Mortem: tencent_v95 — acf 0.3 + diversity-loss 3.0 (killed ep70 — stalled)

**Recipe**: v93 base (KL 0.01, acf-loss 0.3) + diversity-loss-weight 3.0. Using v86 pretrain.

**Training-log**: Best **0.13300★** ep20. Stalled 50 epochs, scores worsening (0.133→0.139→0.151). G_loss climbing. Killed at ep70.

**CONCLUSION: Scalar tuning on BOTH corpora is exhausted.** Stacking any additional loss weight on validated recipes (v59 alibaba, v93 tencent) consistently fails. Now running verbatim controls on both records before moving to structural changes.

---

## Post-Mortem: alibaba_v64 — fft-loss 0.1 (W-guard ep104, eval **0.1319 — WORSE than v59**)

**Recipe**: v59 base (KL 0.01, moment-loss 0.2) + fft-loss-weight 0.1. Using v48 pretrain.

**Training-log**: Best **0.10734★** ep90 (MMD²=0.02104, recall=0.569). Best-ever alibaba training-log score. W-spike guard killed at ep104 (W=4.03, 4.22, 5.33). G_loss extremely high (7-8 throughout).

**Full eval: combined=0.1319** (MMD²=0.03216, β-recall=0.5015, α-precision=**0.4840**, density=0.6718, DMD-GEN=0.7232). Train→eval gap: 23%. α-precision 0.484 — generator producing implausible samples.

**fft-loss-weight 0.1 on alibaba: CLOSED.** Fifth consecutive alibaba scalar tweak to fail:

| Run | Tweak | Train★ | Eval | Verdict |
|-----|-------|--------|------|---------|
| v58 | n-regimes 6 | 0.130 | — | Stalled |
| v60 | quantile 0.3 | 0.096 | **0.170** | Collapse |
| v62 | acf 0.3 | 0.102 | **0.138** | Worse |
| v63 | diversity 3.0 | 0.120 | — | Stalled |
| v64 | fft 0.1 | 0.107 | **0.132** | Worse |

**CONCLUSION: Scalar tuning on alibaba is exhausted.** v59 (KL 0.01 + moment-loss 0.2) IS the alibaba ceiling for loss-weight tuning. Further improvement requires structural/architectural changes.

---

## Post-Mortem: tencent_v94 — acf 0.3 + moment-loss 0.2 (killed ep112 — stalled)

**Recipe**: v93 base (KL 0.01, acf-loss 0.3) + moment-loss-weight 0.2. Using v86 pretrain.

**Training-log**: Best **0.13591★** ep110 (MMD²=0.01901, recall=0.415). Stalled 70 epochs (ep40→ep110 only 0.139→0.136). Far behind v93's trajectory.

**moment-loss-weight 0.2 on tencent: DEFINITIVELY CLOSED.** Three failures: v89 (collapsed ep52), v92 (stalled 0.137), v94 (stalled 0.136). Works on alibaba but consistently fails on tencent regardless of base recipe.

---

## Post-Mortem: alibaba_v63 — diversity-loss 3.0 (killed ep145 — stalled)

**Recipe**: v59 base (KL 0.01, moment-loss 0.2) + diversity-loss-weight 3.0. Using v48 pretrain.

**Training-log**: Best **0.11951★** ep70 (MMD²=0.01871, recall=0.496). Stalled 75 epochs, W climbing to 3.5. Killed at ep145.

**diversity-loss-weight 3.0 on alibaba: CLOSED.** Fourth consecutive alibaba scalar tweak to fail (after quantile 0.3, acf 0.3, diversity 3.0). Scalar tuning on alibaba is exhausted — v59 (KL 0.01 + moment 0.2) appears to be the ceiling.

---

## Post-Mortem: tencent_v93 — acf-loss 0.3 (completed ep200, eval **0.0995 — NEW TENCENT RECORD**)

**Recipe**: v86 base (KL 0.01) + acf-loss-weight 0.3. Using v86 pretrain.

**Training-log**: Best **0.10854★** ep185 (MMD²=0.01194, recall=0.517).

**Full eval: combined=0.0995** (MMD²=0.01019, β-recall=**0.5535**, α-precision=0.8305, density=1.1347, coverage=0.5535, DMD-GEN=**0.6846**, AutoCorr=0.0330, Context-FID=0.11, HRC-MAE=0.0251).

**NEW TENCENT RECORD.** 9.5% improvement over previous best (v88 eval 0.1099). Train→eval gap: **-8.3%** (eval BETTER than training-log). β-recall 0.5535 and DMD-GEN 0.6846 are both best-ever for tencent.

**acf-loss-weight 0.3 on tencent: VALIDATED.** Interestingly, acf-loss 0.3 failed on alibaba (v62 eval 0.1375) but works on tencent — corpus-specific effect.

| Corpus | Previous Best | New Best | Recipe | Improvement |
|--------|--------------|----------|--------|-------------|
| Tencent | 0.1099 (v88) | **0.0995 (v93)** | KL 0.01 + acf-loss 0.3 | 9.5% |
| Alibaba | **0.1113 (v59)** | — | KL 0.01 + moment-loss 0.2 | — |

---

## Post-Mortem: alibaba_v62 — moment 0.2 + acf-loss 0.3 (W-guard ep186, eval **0.1375 — WORSE than v59**)

**Recipe**: v59 base (KL 0.01, moment-loss 0.2) + acf-loss-weight 0.3. Using v48 pretrain.

**Training-log**: Best **0.10207★** ep155 (MMD²=0.01127, recall=0.546). Best-ever alibaba training-log score. W-spike guard killed at ep186 (W=5.78, 4.55, 5.38 — 3 consecutive >4.0).

**Full eval: combined=0.1375** (MMD²=0.02144, β-recall=0.4195, α-precision=0.7850, density=0.5890, DMD-GEN=0.7870, HRC-MAE=0.0124). Train→eval gap: **35%** — recall collapsed from 0.546→0.420.

**PATTERN CONFIRMED: auxiliary loss increases beyond v59 recipe consistently degrade eval.** acf-loss 0.3, quantile increase — all produce better training but worse eval. The v59 recipe (KL 0.01 + moment 0.2, default everything else) remains the optimal alibaba configuration for scalar tuning.

**acf-loss-weight 0.3 on alibaba: CLOSED.**

---

## Post-Mortem: tencent_v92 — moment-loss 0.2, 2nd attempt (killed ep123 — stalled)

**Recipe**: v86 base (KL 0.01) + moment-loss-weight 0.2. Using v86 pretrain.

**Training-log**: Best **0.13726★** ep70 (MMD²=0.01936, recall=0.410). Stalled for 53 epochs — no improvement after ep70. Latest ep120 comb=0.147, recall dropped to 0.353.

**Survived ep50-60 danger zone** (unlike v89), but never approached tencent floor of 0.110. Killed at ep123.

**moment-loss-weight 0.2 on tencent: CLOSED.** Two attempts (v89 collapsed ep52, v92 stalled at 0.137). Works on alibaba (v59 record 0.1113) but not on tencent.

## Post-Mortem: alibaba_v60 — moment 0.2 + quantile 0.3 (completed ep200, full eval **0.1697 — EVAL COLLAPSE**)

**Recipe**: v48 base (block-sample, n-regimes 4) + KL 0.01 + moment-loss 0.2 + quantile-loss 0.3.

**Training-log**: Best **0.09551★** ep115 (MMD²=0.01361, recall=0.591). Strong training scores.

**Full eval: combined=0.1697** (MMD²=0.03140, β-recall=0.3085, α-precision=**0.4320**, density=0.2885). Train→eval gap: **78%** — catastrophic mode collapse at eval.

**CRITICAL FINDING: quantile-loss-weight increase causes eval collapse on BOTH corpora.** Same pattern as supervisor-loss-weight 10 and cond-drop 0.5 — training-log looks great but eval reveals the generator overfit to the training manifold.

| Corpus | quantile-loss | Training-log | Full eval | Verdict |
|--------|--------------|-------------|-----------|---------|
| Alibaba | 0.3 (v60) | 0.0955★ | **0.1697** | COLLAPSE |
| Tencent | 0.4 (v90) | 0.1100★ | 0.1235 | Worse |

**quantile-loss-weight increase CLOSED on BOTH corpora.** v61 (alibaba, +fft) and v91 (tencent, stacked) killed preemptively — same failing base recipe.

---

## Post-Mortem: tencent_v91 — moment 0.2 + quantile 0.3 (killed early — base recipe proven to collapse at eval)

**Recipe**: v86 base (KL 0.01) + moment-loss 0.2 + quantile-loss 0.3. Killed preemptively after v60 eval showed quantile-loss increase causes eval collapse.

---

## Post-Mortem: alibaba_v61 — moment 0.2 + quantile 0.3 + fft 0.1 (killed — base recipe proven to collapse)

**Recipe**: v60 base + fft-loss 0.1. Killed preemptively — v60's eval collapse proved the base recipe is broken.

---

## Post-Mortem: tencent_v90 — quantile-loss-weight 0.4 (completed ep200, full eval **0.1235 — WORSE THAN FLOOR**)

**Recipe**: v86 base (KL 0.01, n-regimes 8) + quantile-loss-weight 0.4 (2× default). Using v86 pretrain.

**Training-log**: Best **0.10998★** ep95 (MMD²=0.01598, recall=0.530). W-elevated late phase.

**Full eval: combined=0.1235** (MMD²=0.01626, β-recall=0.4640, α-precision=0.8875, DMD-GEN=0.7712, Context-FID=0.17, HRC-MAE=0.0745). Train→eval gap: 12.3%.

**quantile-loss-weight 0.4 DOES NOT HELP tencent.** Recall dropped (0.464 vs v88's 0.516), HRC-MAE much worse (0.075 vs v88's 0.008). Interestingly, fake reuse rate is 0.098 vs real 0.056 — model OVER-generates reuse. The doubled quantile loss may be pushing tails too aggressively, causing distribution mismatch.

| Run | Recipe | Training-log | Full eval |
|-----|--------|-------------|-----------|
| v86 | KL 0.01 (base) | 0.1028★ | **0.1130** |
| v88 | KL 0.01 (verbatim) | 0.1074★ | **0.1099** |
| v90 | KL 0.01 + quantile 0.4 | 0.1100★ | 0.1235 |

**quantile-loss-weight 0.4 CLOSED for tencent.**

---

## Post-Mortem: alibaba_v59 — moment-loss-weight 0.2 (completed ep200, full eval **0.1113 — NEW ALIBABA RECORD**)

**Recipe**: v48 base (block-sample, n-regimes 4) + KL 0.01 + moment-loss-weight 0.2 (2× default).

**Training-log**: Best **0.09793★** ep105 (MMD²=0.01353, recall=0.578). Plateaued after ep105.

**Full eval: combined=0.1113** (MMD²=0.01649, β-recall=0.5260, α-precision=**0.9070**, density=1.40, coverage=0.526, DMD-GEN=0.7688, Context-FID=0.15, HRC-MAE=0.0101). Train→eval gap: **13.6%** (best alibaba gap ever).

**moment-loss-weight 0.2 VALIDATED on alibaba — new record and tighter gap:**

| Run | Recipe | Full eval | Train→eval gap |
|-----|--------|-----------|----------------|
| v55 | KL 0.01 | 0.1251 | 28% |
| v57 | KL 0.01 (verbatim) | 0.1134 | 28% |
| **v59** | **KL 0.01 + moment 0.2** | **0.1113** | **13.6%** |

α-precision 0.907 is highest alibaba ever. The doubled moment loss helps both distribution matching (MMD²) and conditioning robustness (smaller gap). **New alibaba floor: ~0.111.**

---

## Post-Mortem: tencent_v89 — moment-loss-weight 0.2 (killed ep64, critic collapse at ep52)

**Recipe**: v86 base (KL 0.01, n-regimes 8) + moment-loss-weight 0.2 (2× default). Using v86 pretrain.

**Result**: Killed at ep64. Best comb=**0.13522★** ep25. W crashed from 0.75 (ep47) to **0.05** (ep52) — critic died. Same pattern as v87 (collapsed ep60).

**Lesson**: Tencent v86 pretrain has ~50% collapse rate around ep50-60. v86 and v88 survived; v87 and v89 collapsed. Cannot determine if moment-loss 0.2 contributed or if this is purely seed-dependent. Switching to quantile-loss-weight 0.4 for v90 to test a different lever. **moment-loss-weight 0.2 INCONCLUSIVE for tencent** (works on alibaba v59).

---

## Post-Mortem: tencent_v88 — v86 verbatim, 2nd attempt (W-guard ep184, full eval **0.1099 — CONFIRMS v86**)

**Recipe**: v86 verbatim (block-sample, n-regimes 8, KL 0.01). Using v86's pretrain. 2nd attempt after v87 collapsed.

**Training-log**: Best **0.10737★** ep125 (MMD²=0.01387, recall=0.532). W-spike guard triggered at ep184 (W=4.65, 5.51, 4.23 — 3 consecutive >4.0).

**Full eval: combined=0.1099** (MMD²=0.01312, β-recall=0.5160, α-precision=0.7805, DMD-GEN=0.7357, Context-FID=0.08, HRC-MAE=0.0078). Train→eval gap: 2.4% (smallest tencent gap ever).

**KL 0.01 REPRODUCIBLE ON TENCENT — second independent confirmation:**

| Run | Training-log | Full eval | Train→eval gap |
|-----|-------------|-----------|----------------|
| v86 | 0.1028★ | **0.1130** | 9.7% |
| v87 | 0.1346★ | COLLAPSED | — |
| v88 | 0.1074★ | **0.1099** | 2.4% |

v86 and v88 agree to within 3% at eval. v87's collapse was seed-specific. **New tencent reproducible floor: ~0.110** (avg v86/v88).

---

## Post-Mortem: alibaba_v58 — n-regimes 6 + KL 0.01 (early-stopped ~ep88, best 0.12961★ ep20)

**Recipe**: n-regimes 6, var-cond-kl-weight 0.01, fresh pretrain. Testing regime count between alibaba sweet-spot 4 and closed 8.

**Result**: Early-stopped around ep88. Best comb=**0.12961★** at ep20 — never improved again in 68 epochs. Scores regressed to 0.14-0.155 range. W low (0.25-0.87).

**Lesson**: n-regimes 6 is worse than n-regimes 4 for alibaba. The n-regimes 4 floor with KL 0.01 is ~0.119 (avg v55/v57 evals). 6 regimes over-segments alibaba's Hurst=0.98 data — fewer, broader regimes work better for near-persistent traces. **n-regimes 6 CLOSED for alibaba.** Confirmed: alibaba=4, tencent=8.

---

## Post-Mortem: tencent_v87 — v86 verbatim (killed ep76, critic collapse at ep60)

**Recipe**: v86 verbatim (block-sample, n-regimes 8, KL 0.01). Using v86's pretrain.

**Result**: Killed at ep76. Best comb=**0.13461★** ep40. W crashed from 0.59 (ep59) to **0.02** (ep60) — critic died instantly. Recall dropped to 0.255. Unrecoverable.

**Lesson**: Fresh tencent pretrain has seed-dependent instability. v86 succeeded (eval 0.1130) but v87 collapsed with identical recipe. v86's result may include a seed luck component. Need v88 to disambiguate.

---

## Post-Mortem: alibaba_v57 — v55 verbatim, var-cond-kl-weight 0.01 (completed ep200, full eval **0.1134 — CONFIRMS AND BEATS v55**)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--var-cond-kl-weight 0.01`. Verbatim repeat of v55 for reproducibility testing.

**Training-log**: Best **0.08832★** ep185 (MMD²=0.01062, recall=0.612). Matched v55's 0.08894★ to within 0.7%. best.pt saved at ep195 (EMA improved late).

**Full eval: combined=0.1134** (MMD²=0.01364, β-recall=0.5010, α-precision=0.7285, DMD-GEN=0.7927, Context-FID=0.34, HRC-MAE=0.0064). Train→eval gap: 28%.

**KL 0.01 DEFINITIVELY REPRODUCIBLE — verbatim control outperforms original:**
- v57 eval: **0.1134** vs v55 eval: 0.1251 → v57 is **9.4% better**
- Both beat old floor: v53 verbatim 0.1325 → **14.4% improvement**
- HRC-MAE: **0.0064** — near-perfect cache fidelity
- Train→eval gap: 28% (v57) vs 41% (v55) — KL regularization is working

**Updated alibaba reproducible floors with KL 0.01:**
- v55: eval 0.1251
- v57: eval 0.1134
- **New floor: ~0.119** (average), down from ~0.13 with base recipe

---

## Post-Mortem: tencent_v86 — v78 base + var-cond-kl-weight 0.01 (completed ep200, full eval **0.1130 — BEATS FLOOR BY 23%**)

**Recipe**: v78 base (block-sample, n-regimes 8) + `--var-cond-kl-weight 0.01` (2× standard 0.005). Fresh pretrain with correct trace path (tencent_block_1M, 5 columns).

**Training-log**: Best **0.10302★** ep170 (MMD²=0.01262, recall=0.548). Steady convergence: ep15→0.155, ep50→0.144, ep75→0.134, ep100→0.130, ep110→0.126, ep125→0.119, ep170→0.103. W values spiked late (ep172=4.09, ep174=4.63) but never triggered 3-consecutive guard.

**Full eval: combined=0.1130** (MMD²=0.01717, β-recall=0.5210, α-precision=0.9225, DMD-GEN=0.7139, AutoCorr=0.0346, Context-FID=0.07, HRC-MAE=0.0494).

**MAJOR RESULT — 22.8% improvement over tencent reproducible floor:**
- v86 eval: **0.1130** vs v84 verbatim floor: **0.1464** → **-22.8%**
- α-precision: 0.9225 (best ever tencent — 92% of fakes plausible)
- Context-FID: 0.07 (best ever)
- Train→eval gap: **9.7%** — smallest gap ever observed (vs typical 30-45%)

**var-cond-kl-weight 0.01 now validated on BOTH corpora:**
- Alibaba: v55 eval 0.1251 (vs floor 0.1325, -5.6%)
- Tencent: v86 eval **0.1130** (vs floor 0.1464, **-22.8%**)
- The improvement is NOT from training-log score (which plateaus similarly) but from dramatically reduced train→eval gap

---

## Post-Mortem: alibaba_v56 — v48 + var-cond-kl-weight 0.02 (killed ep34, critic collapse)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--var-cond-kl-weight 0.02` (4× standard 0.005).

**Result**: Killed at ep34. Best comb=0.15582★ ep30 (recall=0.382, MMD²=0.03222). W values crashed from 0.52 (ep25) to **0.03-0.04** (ep28-34) — critic effectively dead.

**KL weight sweep now complete:**
- 0.005 (default): floor ~0.13 (v53)
- **0.01 (2×): eval 0.1251 (v55) — sweet spot**
- 0.02 (4×): critic collapse (v56) — too aggressive

**Lesson**: var-cond-kl-weight 0.02 over-regularizes the conditioning, washing out the discriminative signal the critic needs. **var-cond-kl-weight 0.02 closed on alibaba.** The sweet spot is 0.01.

## Post-Mortem: alibaba_v55 — v48 + var-cond-kl-weight 0.01 (completed ep200, full eval **0.1251 — BEATS FLOOR**)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--var-cond-kl-weight 0.01` (2× standard 0.005). Testing stronger variational regularization.

**Training-log**: Best **0.08894★** ep165 (MMD²=0.01124, recall=0.612). ★ at ep5→10→15→20→40→50→80→125→135→155→165. W values spiked late (ep186=5.3, ep192=6.2, ep198=6.6) but never 3 consecutive above guard. best.pt saved at ep145.

**Full eval: combined=0.1251** (MMD²=0.01647, β-recall=0.4570, α-precision=0.7085, DMD-GEN=0.7508, Context-FID=0.10, HRC-MAE=0.0112). Train→eval gap: 41% (typical).

**This is the first genuine improvement over the reproducible alibaba floor:**
- v55 eval: **0.1251** vs v53 verbatim floor: **0.1325** → **5.6% improvement**
- Recall improved: 0.457 vs v53's 0.413
- MMD² slightly worse: 0.01647 vs v53's 0.01505

**Lesson**: var-cond-kl-weight 0.01 (2×) is a validated improvement for alibaba. The stronger KL regularization makes the conditioning distribution more robust at eval time, reducing the train→eval gap on recall. Unlike supervisor-loss 10 and cond-drop 0.5 (which produced spectacular training-logs but collapsed at eval), the KL weight change produces modest training-log numbers that hold up at eval. **New alibaba reproducible floor: ~0.125.**

---

## Post-Mortem: tencent_v85 — v78 + supervisor-loss-weight 10.0 (completed ep200, full eval **MODE COLLAPSE**)

**Recipe**: v78 base (block-sample, n-regimes 8) + `--supervisor-loss-weight 10.0` (2× standard 5.0).

**Training-log**: Best **0.09569★** ep155 (MMD²=0.00919, recall=0.568). ★ at ep25→30→35→40→50→55→90→110→125→155. Strongest sustained convergence this session. W values healthy (1.0-2.9), one late spike at 5.3 (ep176).

**Full eval: CATASTROPHIC MODE COLLAPSE.** Combined=**0.365** (MMD²=0.16575, β-recall=**0.0045**, α-precision=0.005, DMD-GEN=0.7984, Context-FID=3.53, HRC-MAE=0.0200). The model produces near-zero diversity at eval despite spectacular training-log numbers.

**Lesson**: supervisor-loss-weight 10.0 causes the same failure mode as cond-drop-prob 0.5 — G overfits to the supervisor's latent manifold during training, producing numbers that look excellent in-distribution but collapse completely out-of-distribution at eval. The training-log ★ trajectory was entirely illusory. **supervisor-loss-weight 10.0 CLOSED on both corpora** (alibaba v54: W stuck 0.2-0.3; tencent v85: mode collapse at eval).

---

## Post-Mortem: alibaba_v54 — v48 + supervisor-loss-weight 10.0 (killed ep25, hopeless)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--supervisor-loss-weight 10.0` (2× standard 5.0). Testing whether stronger supervisor signal helps alibaba.

**Result**: Killed at ep25. Best comb=**0.18611★** at ep20 (recall=0.341, MMD²=0.05421). Regressing at ep25 (comb=0.21668). W values stuck at 0.2-0.3 — far too low, indicating critic can't separate real/fake. Compare: v53 verbatim was at 0.10877 by ep15.

**Lesson**: supervisor-loss-weight 10.0 overwhelms the adversarial signal on alibaba. The supervisor's latent manifold dominates, preventing the critic from developing separation. This contrasts with tencent where v85 (same change) shows strong sustained convergence — likely because tencent's 8-regime structure gives the critic more to work with vs alibaba's 4 regimes. **supervisor-loss-weight 10.0 closed on alibaba** (but still testing on tencent v85).

---

## Recalibrated Reproducible Floors (2026-04-09)

Both ATBs confirmed as lucky seeds via verbatim controls:

| Corpus | Lucky ATB | Verbatim Control | Reproducible Floor |
|--------|-----------|------------------|-------------------|
| Alibaba | 0.0767 (v48) | 0.1325 (v53) | **~0.13** |
| Tencent | 0.1008 (v78) | 0.1464 (v84) | **~0.14** |

Future improvements should be evaluated against reproducible floors, not lucky ATBs.

---

## Post-Mortem: alibaba_v53 — v48 verbatim (W-spike kill ep179, full eval 0.1325 — **v48 was also a lucky seed**)

**Recipe**: v48 verbatim (block-sample, n-regimes 4). Clean reproducibility control.

**Result**: W-spike killed at ep179. Best.pt at ep140 (saved MMD²=0.01474). Training-log showed improvement through ep175 (0.09380). **Full eval: combined=0.1325** (MMD²=0.01505, β-recall=0.4130, α-precision=0.6475, DMD-GEN=0.7242, Context-FID=0.10, HRC-MAE=0.0150). **73% worse than v48's ATB of 0.0767.**

**Lesson — v48 was a lucky seed, just like v78 and the historical v37/v68.** The gap is almost entirely in recall: v48 eval'd at 0.6815 recall (negative train→eval gap), v53 eval'd at 0.4130 (29% recall drop from training-log). The MMD² values are comparable (0.01304 vs 0.01505). This means v48's unprecedented negative train→eval gap was seed luck, not a property of the block-sample recipe.

**Both ATBs are now confirmed as non-reproducible lucky seeds.** The reproducible floor for the block-sample recipes is ~0.13 (alibaba) and ~0.14 (tencent). The Round 5 reviewer was right: "the session should stop treating one lucky single-seed run as enough evidence." Future claims should benchmark against these floors.

---

## Post-Mortem: tencent_v84 — v78 verbatim (killed ep112, full eval 0.1464 — **v78 was a lucky seed**)

**Recipe**: v78 verbatim (block-sample, diversity 2.0, n-regimes 8). Clean reproducibility control.

**Result**: Killed at ep112. Best comb=**0.10548★** at ep80 (recall=0.524, MMD²=0.01038). 32 epochs stagnant. **Full eval: combined=0.1464** (MMD²=0.03076, β-recall=0.4220, α-precision=0.8650, DMD-GEN=0.7129, Context-FID=0.13, HRC-MAE=0.0622). **45% worse than v78's ATB of 0.1008.**

**Lesson — v78 was a lucky seed.** The verbatim control cannot reproduce v78's training-log trajectory (0.10548 vs 0.08355, 26% gap) or full-eval result (0.1464 vs 0.1008, 45% gap). Combined with v80 (w-stop 5.0, eval ~0.124★ training-log), this establishes that tencent block-sample results have **~40-45% seed-dependent spread at full eval**. The v78 recipe is sound but the specific ATB number includes significant seed luck. Future tencent improvements should be evaluated against the reproducible floor (~0.14), not the lucky v78 ATB.

---

## Post-Mortem: tencent_v83 — v78 + locality-loss-weight 2.0 (killed ep117, 39% behind ATB pace)

**Recipe**: v78 base (block-sample, diversity 2.0, n-regimes 8) + `--locality-loss-weight 2.0` (2× standard 1.0).

**Result**: Killed at ep117. Best comb=**0.11628★** at ep115 (recall=0.480, MMD²=0.01238). 39% behind v78's training-log at ep110 (0.08355). Cannot reach ATB even with zero eval gap.

**Lesson**: locality-loss-weight 2.0 is too slow on tencent, same as alibaba (v52). **locality-loss-weight 2.0 closed on both corpora.** The reviewer is right: locality needs a structural copy path, not a loss weight increase.

---

## Post-Mortem: alibaba_v52 — v48 + locality-loss-weight 2.0 (killed ep87, lagging 27%)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--locality-loss-weight 2.0` (2× standard 1.0). Testing stronger locality emphasis per reviewer recommendation.

**Result**: Killed at ep87. Best comb=**0.13188★** at ep80 (recall=0.491, MMD²=0.03008). 27% behind v48's pace at the same epoch. Recall oscillating 0.41-0.49, never breaking 0.50 sustainably.

**Lesson**: locality-loss-weight 2.0 adds too much friction on alibaba — the model spends capacity satisfying the locality constraint at the expense of distributional fidelity. The reviewer is right that locality needs an architectural solution (copy path), not a loss weight increase. **Close: locality-loss-weight 2.0 is dead on alibaba.** The standard 1.0 remains in the base recipe.

---

## Post-Mortem: tencent_v82 — v78 + cond-drop-prob 0.5 (full eval 0.1204 — does NOT beat ATB 0.1008)

**Recipe**: v78 base (block-sample, diversity 2.0, n-regimes 8) + `--cond-drop-prob 0.5` (2× standard 0.25). Testing more aggressive CFG dropout for eval generalization.

**Result**: Trained 200/200. Training-log best comb=**0.08200★** ep200 (recall=0.621, MMD²=0.00610 — excellent). **Full eval: combined=0.1204** (MMD²=0.01327, β-recall=0.4645, α-precision=0.8445, DMD-GEN=0.7410, AutoCorr=0.0434, Context-FID=**0.10** — best tencent ever, HRC-MAE=0.0136).

**Lesson**: cond-drop-prob 0.5 fails on tencent for the same reason as alibaba v51: spectacular training-log (nearly matching v78's peak) but 47% train→eval gap kills it (v78 had only 20% gap). Dropping conditioning 50% of the time makes G learn a blended conditional/unconditional model that looks great at training-time EMA metrics but doesn't specialize correctly at eval. **cond-drop-prob 0.5 is now closed on BOTH corpora.** 0.25 remains the sweet spot.

Context-FID 0.10 (best tencent ever) and AutoCorr 0.0434 (better than v78's 0.0598) suggest the latent representation is improving even as recall coverage suffers. Same DMD-GEN pattern as v51.

---

## Post-Mortem: alibaba_v51 — v48 + cond-drop-prob 0.5 (W-spike kill ep162, full eval 0.1064 — does NOT beat ATB 0.0767)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--cond-drop-prob 0.5` (2× standard 0.25). Testing whether more aggressive CFG dropout improves eval generalization.

**Result**: W-spike killed at ep162. Training-log best comb=**0.07124★** ep155 (recall=0.679, MMD²=0.00704 — **best alibaba training-log ever**). But **full eval: combined=0.1064** (MMD²=0.01900, β-recall=0.5630, α-precision=0.7300, DMD-GEN=**0.6825** — best alibaba dynamics ever, HRC-MAE=0.0037, Context-FID=0.14).

**Lesson**: cond-drop-prob 0.5 produces spectacular training-log numbers but does NOT generalize on alibaba. The 49% train→eval gap (0.07124→0.1064) is dramatically worse than v48's negative gap (0.10347→0.0767). Hypothesis: on alibaba's small corpus (239 files), dropping conditioning 50% of the time during training makes G overfit to the unconditional distribution, which doesn't match the full-eval conditioning. **cond-drop-prob 0.25 remains the alibaba sweet spot.**

Positive: DMD-GEN 0.6825 is the best alibaba dynamics score ever (v48 was 0.7367, improvement of 7.4%). This suggests cond-drop 0.5 helps temporal dynamics even if it hurts distributional metrics. Future work: could cond-drop 0.5 be used for a late-stage fine-tune targeting DMD-GEN specifically?

---

## Post-Mortem: alibaba_v50 — v48 + acf-loss-weight 0.5 + block-sample (killed ep111, stagnant)

**Recipe**: v48 base (block-sample, n-regimes 4) + `--acf-loss-weight 0.5` (2.5× standard 0.2). Testing whether stronger ACF penalty helps on top of block-sample.

**Result**: Killed at ep111. Best comb=**0.12367★** at ep85 (recall=0.505, MMD²=0.02477). 26 epochs stagnant. At ep110, score was 0.12869 — 24% behind v48's training-log peak of 0.10347 at the same epoch.

**Lesson**: acf-loss 0.5 hurts even with block-sample. The stronger ACF penalty adds friction without improving temporal dynamics — block-sample already provides temporal coherence via the sampling strategy. Combined with v49 (acf 0.5 without block-sample) and v81 (acf 0.5 tencent, critic collapse), **acf-loss-weight 0.5 is definitively closed on both corpora with and without block-sample**.

---

## Post-Mortem: tencent_v81 — v78 + acf-loss-weight 0.5 (killed ep89, critic collapse)

**Recipe**: v78 base (block-sample, diversity 2.0, n-regimes 8) + `--acf-loss-weight 0.5` (2.5× standard 0.2). Testing stronger autocorrelation penalty on tencent with block-sample.

**Result**: Killed at ep89. Best comb=**0.10332★** at ep60 (recall=0.524, MMD²=0.00812 — excellent). But critic collapsed after ep60: W dropped from 1.1 to 0.005 by ep89, MMD² spiked to 0.06061, recall crashed to 0.347. No recovery possible.

**Lesson**: acf-loss 0.5 destabilizes the critic-generator balance on tencent. The early trajectory was promising (ep60 ★ nearly matched v78 ATB pace) but the stronger ACF penalty creates conflicting gradients that eventually overwhelm the critic. Combined with alibaba_v49 failure (acf 0.5 without block-sample), **acf-loss-weight 0.5 is now closed on both corpora**. The standard 0.2 weight remains part of the base recipe.

---

## Post-Mortem: tencent_v80 — v78 verbatim + w-stop 5.0 (killed ep94, **WORSE** than v78 ATB 0.1008)

**Recipe**: v78 verbatim (block-sample, diversity 2.0, n-regimes 8) + `--w-stop-threshold 5.0` (relaxed from 4.0). Testing reproducibility of block-sample recipe with looser W guard.

**Result**: Killed at ep94. Best comb=**0.12385★** at ep50 (recall=0.445, MMD²=0.01285). 44 epochs without improvement. Recall oscillating 0.37–0.45. v78 at ep80 was already 0.09222 — v80 lagged by 34%.

**Lesson**: Relaxing w-stop to 5.0 does not help — the model doesn't exploit the extra headroom productively. The W values in v80 stayed moderate (0.2–2.6), so the relaxation was irrelevant. v80's underperformance vs v78 is likely just seed variance, confirming that block-sample results have ~23% seed-dependent spread. **Close: w-stop 5.0 doesn't add value.**

---

## Post-Mortem: alibaba_v49 — v46 + acf-loss-weight 0.5 (killed ep49, **WORSE** than v46 0.1283)

**Recipe**: v46 base (n-regimes 4) + `--acf-loss-weight 0.5` (2.5× v46's 0.2). Testing whether stronger autocorrelation penalty improves temporal dynamics. **NOTE: did NOT have --block-sample** — omitted by mistake.

**Result**: Killed at ep49. Best comb=**0.15709★** at ep10 (recall=0.415, MMD²=0.04019). 39 epochs without improvement. Recall stuck 0.30–0.36. Far behind v46's pace (v46 ep20 was 0.13259).

**Lesson**: acf-loss 0.5 without block-sample hurts convergence badly. The stronger ACF penalty fights the random file ordering, creating contradictory gradients. v50 will test acf-loss 0.5 WITH block-sample, where temporal coherence is already provided by the sampling strategy. **Close: acf-loss 0.5 without block-sample dead on alibaba.**

---

## Post-Mortem: tencent_v79 — v78 + gmm-components 16 (killed ep110, stagnant)

**Recipe**: v78 base (block-sample, diversity 2.0, n-regimes 8) + `--gmm-components 16` (2× standard 8). Required fresh pretrain due to n-regimes mismatch (initially tried n-regimes 12 but pretrain had K=8).

**Result**: Killed at ep110. Best consistently 0.02+ behind v78 with gmm-8. No improvement trajectory.

**Lesson**: gmm-components 16 fragments the latent space without adding useful structure. 8 components already match the regime count. **Close: gmm-16 doesn't help.**

---

## Post-Mortem: tencent_v78 — v76 + block-sample (full eval **0.1008 — NEW TENCENT RECORD**, beats 0.1122 by 10.2%)

**Recipe**: v76 base (diversity 2.0) + `--block-sample`. First use of contiguous temporal block sampling with the v76 winning recipe. W-spike killed at ep199 (W>4.0 for 3 consecutive epochs); best.pt at ep110 preserved.

**Result**: Training-log best comb=**0.08355** ep110 (recall=0.615, MMD²=0.00645 — both all-time tencent records). **Full eval: combined=0.1008** (MMD²=**0.00734**, β-recall=**0.5325**, α-precision=0.8410, density=1.0650, coverage=0.5325, DMD-GEN=0.7416, AutoCorr=0.0598, Context-FID=0.14, HRC-MAE=0.0795).

**Lesson — block-sample is the tencent breakthrough lever.** Key observations:
1. Train→eval MMD² gap was only **14%** (0.00645→0.00734) — by far the smallest gap this session. Compare v77 (5.5× gap), v76 (1.86× gap).
2. Train→eval recall gap only **13%** (0.615→0.5325) — also smallest this session.
3. Block-sample preserves temporal coherence during training, so the model doesn't have to learn cross-file transitions that don't exist in eval. This is why the train→eval gap shrinks dramatically.
4. Context-FID 0.14 is the lowest tencent ever — the latent space distribution is nearly matching real data.

**NEW TENCENT ATB: 0.1008.** Recalibrated leaderboard:

| Corpus | ATB | Run | Key Lever |
|--------|-----|-----|-----------|
| Tencent | **0.1008** | v78 | block-sample + diversity 2.0 |
| Alibaba | **0.1283** | v46 | n-regimes 4 |

---

## Post-Mortem: alibaba_v46 — v37 + n-regimes 4 (full eval **0.1283 — NEW ALIBABA RECORD**, beats 0.142 floor by 9.6%)

**Recipe**: v37 base + `--n-regimes 4` (up from v37's 2). Hypothesis: alibaba's diverse workloads benefit from more regime prototypes.

**Result**: Trained 200/200. Training-log best comb=**0.09513** ep75 (recall=0.600, MMD²=0.01503 — highest alibaba recall ever at training-log). **Full eval: combined=0.1283** (MMD²=0.02585, β-recall=**0.4880**, α-precision=**0.8835**, density=0.7976, coverage=0.488, DMD-GEN=0.7122, AutoCorr=0.0379, HRC-MAE=**0.0067**, Context-FID=0.30).

**Lesson — n-regimes 4 is the first technique to genuinely beat the recalibrated alibaba floor.** Key observations:
1. β-recall train→eval drop was only **19%** (0.600→0.488) — the smallest holdout gap this session. Compare v44 (60% drop), v42 (34% drop).
2. α-precision 0.8835 is highest alibaba ever (vs v42 0.6595, v43 0.7745).
3. HRC-MAE 0.0067 is near-perfect cache fidelity.
4. DMD-GEN 0.7122 still the open dynamics gap.
5. The regime count was the bottleneck: 2 regimes (v37/v42) couldn't express enough workload diversity. 4 regimes gave recall breathing room.

**NEW ALIBABA ATB: 0.1283.** Next: test n-regimes 8 (v47).

---

## Post-Mortem: alibaba_v48 — v46 + block-sample (full eval **0.0767 — NEW ALL-TIME ALIBABA RECORD**, beats 0.1283 by 40%)

**Recipe**: v46 base (n-regimes 4) + `--block-sample`. Combining the two independently-validated winning levers. W-spike killed ep139; best.pt at ep110 preserved.

**Result**: Training-log best comb=**0.10347** ep110 (recall=0.553, MMD²=0.01407). **Full eval: combined=0.0767** (MMD²=**0.01304**, β-recall=**0.6815**, α-precision=0.8375, density=1.5676, coverage=0.6815, DMD-GEN=0.7367, AutoCorr=0.0384, Context-FID=0.19, HRC-MAE=**0.0006**).

**THIS IS THE MOST IMPORTANT RESULT OF THE PROJECT.** Key findings:

1. **β-recall INCREASED at full eval** (0.553→0.6815, +23%). This is the first time in the entire project history that recall improved from training to evaluation. Every prior run showed 13-60% recall collapse.
2. **MMD² DECREASED at full eval** (0.01407→0.01304, −7%). Also unprecedented — the eval distribution is closer to real than the training-log measurement suggested.
3. **HRC-MAE 0.0006 is effectively perfect** — the cache hit-ratio curve of synthetic data is indistinguishable from real data at 4 decimal places.
4. **β-recall 0.6815 is the highest recall ever measured** at full eval on any corpus in this project. Previous best was v78 tencent at 0.5325.
5. **The two winning levers multiply**: block-sample (preserves temporal coherence) × n-regimes 4 (provides workload diversity) together eliminate the train→eval gap that has plagued every prior alibaba run. The mechanism is clear: alibaba has H=0.98 (nearly maximal Hurst exponent), so temporal coherence from block-sample at training time directly matches eval data's temporal structure.

**NEW ALIBABA ATB: 0.0767.** Updated leaderboard:

| Corpus | ATB | Run | Key Lever |
|--------|-----|-----|-----------|
| Alibaba | **0.0767** | v48 | block-sample + n-regimes 4 |
| Tencent | **0.1008** | v78 | block-sample + diversity 2.0 |

---

## Post-Mortem: alibaba_v47 — v37 + n-regimes 8 (full eval 0.1369, worse than v46 0.1283)

**Recipe**: v37 base + `--n-regimes 8` (up from v46's 4). Testing whether more regime prototypes improve alibaba further.

**Result**: W-spike killed ep180. Training-log best comb=**0.09756** ep160 (recall=0.560, MMD²=0.00966). **Full eval: combined=0.1369** (MMD²=0.01772, β-recall=0.4040, α-precision=0.5990, Context-FID=**0.08** — best ever). **6.7% above v46's 0.1283.**

**Lesson**: n-regimes 8 is too many for alibaba's 239 files. Recall drops from v46's 0.488 to 0.404 at eval — more regimes spread training data too thin per regime. **n-regimes 4 remains the alibaba sweet spot.** However, Context-FID 0.08 (best ever) suggests the latent representation is improving even as coverage suffers.

---

## Post-Mortem: tencent_v77 — v76 + supervisor-steps 2 (full eval 0.1756, worse than v76 0.1122 record)

**Recipe**: v76 base (diversity 2.0) + `--supervisor-steps 2`. Hypothesis: 2-step supervisor adds temporal coherence.

**Result**: Trained 200/200. Training-log best comb=**0.09956** ep125 (recall=0.544, MMD²=0.00846). **Full eval: combined=0.1756** (MMD²=0.04621, β-recall=0.3530, α-precision=0.7505, density=0.6564, DMD-GEN=0.7134, HRC-MAE=0.0298). **57% above the tencent 0.1122 record.**

**Lesson**: sup-steps=2 produces the familiar train→eval collapse: MMD² 0.00846→0.04621 (5.5×), recall 0.544→0.353 (35% drop). Combined with alibaba_v45 failure, **supervisor-steps=2 is now closed on both corpora**. The 2-step supervisor teaches G to overfit to the supervisor's latent manifold, which doesn't generalize at eval.

---

## Post-Mortem: alibaba_v45 — v37 + supervisor-steps 2 (killed ep43, no improvement)

**Recipe**: v37 base + `--supervisor-steps 2`. Untried lever for alibaba.

**Result**: Killed at ep43. Best comb=**0.15516** at ep15, then 28 epochs with no further improvement. Recall stuck at 0.34-0.38. Lagging v44's pace at the same epoch (v44 ep45 was 0.14489).

**Lesson**: 2-step supervisor does not help alibaba. The supervisor signal at 1-step is already sufficient (or saturating); doubling it adds friction without gain. Closed for alibaba. Combined with v45 killing supervisor-steps=2 on tencent (currently being tested in v77, watch result), this lever may be globally unhelpful.

**Note on v76 final.pt**: full eval of final.pt (ep200) gave combined=0.1331 (recall collapsed 0.571→0.425), worse than the ep70 best.pt record of 0.1122. **Tencent ATB remains 0.1122 (v76 best.pt @ ep70).** The final.pt overtrained on dynamics at the expense of recall coverage.

---

## Post-Mortem: tencent_v76 — v68 + diversity-loss-weight 2.0 (full eval **0.1122 — NEW TENCENT RECORD**, beats 0.1225 floor by 8.4%)

**Recipe**: v68 base + `--diversity-loss-weight 2.0`. First time MSGAN diversity loss applied to tencent. Killed mid-run by SSH disconnect at ep148; best.pt saved at ep70.

**Result**: Training-log best comb=**0.10049** ep70 (recall=0.578, MMD²=0.01599). **Full eval: combined=0.1122** (MMD²=0.01959, β-recall=**0.5370** held up at full eval, α-precision=0.9015, density=2.1714, DMD-GEN=0.7012, AutoCorr=0.0676, HRC-MAE=0.0762).

**Lesson — diversity-loss-weight 2.0 is the first technique to genuinely beat the recalibrated tencent floor.** Critically, β-recall did **not** collapse from train→eval (0.578→0.537, only 7% drop, vs the 30%+ collapse seen in feat-critic and other failed techniques). α-precision 0.9015 is the highest tencent precision we've measured. DMD-GEN 0.70 still the open gap. **NEW TENCENT ATB: 0.1122.** Diversity 2.0 transferred cleanly from alibaba to tencent — a rare case of cross-corpus generalization.

**Action**: Promote diversity 2.0 to the tencent base recipe. Next try: v68 + diversity 2.0 + supervisor-steps=2 OR + cross-cov 4.0.

---

## Post-Mortem: alibaba_v44 — v37 + cross-cov-loss-weight 4.0 (full eval 0.187, **WORSE** than 0.142 floor by 32%)

**Recipe**: v37 base + `--cross-cov-loss-weight 4.0` (2× v37's 2.0). Targeting DMD-GEN temporal dynamics gap by doubling the cross-feature lag-1 covariance penalty. Killed mid-run by SSH disconnect at ep81; best.pt saved at ep80.

**Result**: Training-log best comb=**0.12400** ep80 (recall=0.484, MMD²=0.02090). **Full eval: combined=0.187** (MMD²=0.02642, β-recall=**0.1960** — collapsed from 0.484, α-precision=0.5965, density=0.4372, coverage=0.1960, DMD-GEN=0.7573, HRC-MAE=0.0295).

**Lesson**: Doubling cross-cov to 4.0 induces severe recall collapse at full eval (0.484→0.196, 60% drop). The training-log score is misleading because the cross-cov penalty drives G to satisfy the lag-1 covariance constraint by collapsing onto a lower-dimensional manifold. v37's 2.0 weight is the upper bound; cross-cov 4.0 is closed for alibaba. **DMD-GEN also did not improve** (0.7573 vs ~0.73 baseline) — the cross-cov regulariser does not actually move temporal dynamics at the eval level.

**Action**: Cross-cov tuning closed as a primary lever for alibaba. Next try: route through R-informed conditioning instead.

---

## Post-Mortem: alibaba_v43 — v37 + diversity-loss-weight 4.0 (full eval 0.149, marginal vs new 0.142 floor)

**Recipe**: v37 base + `--diversity-loss-weight 4.0` (2× v37's 2.0). MSGAN mode-seeking pushed harder to combat recall mode collapse.

**Result**: Trained 200/200. Training-log best comb=**0.11629** ep50 (recall=0.519, MMD²=0.02009). **Full eval: combined=0.149** (MMD²=0.01921, β-recall=0.3515, **α-precision=0.7745** — highest of session, density=0.5887, DMD-GEN=0.7273, HRC-MAE=0.0121). Marginally **5% above the recalibrated 0.142 floor**, basically a tie with v42.

**Lesson**: Doubling diversity-loss-weight does not improve recall at full eval. The α-precision spike to 0.7745 (vs v42 0.6595) suggests stronger mode-seeking does shift the precision/recall tradeoff toward precision — useful if precision were the bottleneck, but it isn't. Diversity above 2.0 is closed for alibaba.

---

## Post-Mortem: tencent_v75 — v68 verbatim REPRODUCIBILITY CONTROL (full eval 0.1225, FAILED to reproduce historical tencent ATB 0.089)

**Recipe**: Identical to v68 (n-regimes=8, supervisor=5, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2, no diversity-loss). Verbatim control with fresh rng.

**Result**: Trained 200/200. Training-log best comb=**0.09120** ep200 (final epoch breakthrough, recall=0.583, MMD²=0.00770). **Full eval: combined=0.1225** (MMD²=0.01584, β-recall=0.4665, α-precision=0.6725, density=0.7822, DMD-GEN=0.7077, HRC-MAE=0.0247). **38% above the recorded tencent ATB 0.089.**

**Lesson — second-corpus confirmation of the v42 finding**: Both alibaba_v42 (v37 verbatim → 0.142 vs claimed 0.0786) and tencent_v75 (v68 verbatim → 0.1225 vs claimed 0.089) fail to reproduce their historical ATBs. The pattern is consistent: training-log scores in the "good" range (0.09-0.10), full-eval scores 30-50% higher, β-recall train→eval gap of ~20-30%. **Both historical ATBs are not reproducible** and were likely lucky-seed runs.

**Recalibrated baselines (this is the new ground truth for Round-5+ comparisons):**
- **Alibaba ATB: 0.142** (was 0.0786) — based on v42 v37-verbatim full eval
- **Tencent ATB: 0.1225** (was 0.089) — based on v75 v68-verbatim full eval

**Implications for the session sweep**: Several runs we marked as "failed" should be re-examined against the new floors:
- alibaba_v38 (block_sample) full eval **0.094** is now a **34% improvement** over the 0.142 floor
- alibaba_v40 (feat-critic) full eval 0.158 is near the new floor (not the catastrophe we recorded)
- tencent_v69 (block_sample) full eval 0.138 is competitive with the 0.1225 floor

**Action**: Subsequent runs are judged against these recalibrated floors. The lowest reproducible full-eval combined score now becomes the operational ATB.

---

## Post-Mortem: alibaba_v42 — v37 verbatim REPRODUCIBILITY CONTROL (W-killed ep193, full eval 0.142, FAILED to reproduce v37 ATB 0.0786)

**Recipe**: Identical to v37 (supervisor=5, diversity=2.0, n-regimes=2, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2, all v37 hyperparameters). The only difference from v37 is wall-clock time and rng state.

**Result**: W-spike-killed at ep193. Training-log best comb=**0.09195** ep190 (recall=0.603, MMD²=0.01255). **Full eval: combined=0.142** (MMD²=0.02207, β-recall=0.3990 — 34% drop from train, α-precision=0.6595, density=0.5301, DMD-GEN=0.7728, HRC-MAE=0.0338). **81% above the recorded v37 ATB 0.0786.**

**Lesson — this is the most important finding of the session**: The v37 ATB combined=0.0786 score is **not reproducible** under the same recipe with a fresh rng. The verbatim control (v42) gave 0.142 — closer to a "typical" alibaba run than to a record. This means:

1. The historical v37 ATB 0.0786 was either a lucky seed, measured with a different eval pipeline, or used a pretrain/data state that has since changed.
2. **Many "failed" runs in this session (v36, v38, v39, v40) were actually within ~50% of the *true* reproducible alibaba baseline**, not 70-100% above it.
3. The "alibaba ATB" benchmark needs to be recalibrated. For now, treat **v42's 0.142 as the new alibaba reproducibility floor**, and any new run that beats this is genuinely an improvement.

**Action**: Subsequent alibaba runs should be judged against 0.142, not 0.0786. The previously-closed ideas (block_sample, multi-scale critic, feat-critic) should be re-examined: their full-eval results 0.094, 0.121, 0.158 are mostly *near or better than* the new floor.

---

## Post-Mortem: tencent_v74 — feat-critic-weight=0.5 (dual discriminator) on v68 base (FAILED, full eval 0.175, train/eval gap 12× on MMD²)

**Recipe**: v68 base (n-regimes=8, supervisor=5, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2) + `--feat-critic-weight 0.5` (dual discriminator: latent C + feature-space C_feat).

**Result**: Trained 200/200. Training-log best comb=**0.06476** ep195 (recall=0.700, MMD²=**0.00466**) — *the highest tencent training-log of the session, "27% below ATB"*. **Full eval: combined=0.175** (MMD²=**0.05617** — 12× the train value, β-recall=0.4065 — 42% drop from train 0.700, α-precision=0.6160, density=0.4723, DMD-GEN=0.6765, HRC-MAE=0.0629). **97% worse than ATB 0.089.**

**Lesson**: Definitive confirmation across **both** corpora (alibaba_v40, tencent_v74) that feat-critic produces the most extreme train/eval gap of any technique in the project. The feature-space critic teaches G to fit the C_feat-induced feature manifold, which is then off-distribution for the eval-time TimeGAN encoder. The bigger the train improvement, the worse the eval. **Feat-critic is closed for both corpora.**

**Pattern**: The session has now found that on tencent the train/eval gap can swing from ~30% (clean recipes) to ~70% (feat-critic) to 12× (when the C_feat manifold is exotic enough). Future ideas must be filtered by *eval-set behavior*, not training-log combined alone. Need a mid-training cheap eval signal to detect runaway train/eval divergence early.

**v31 co-ATB combined=0.089 remains the tencent record.**

---

## Post-Mortem: alibaba_v41 — patch-embed + cond-drop-prob 0.5 + var-cond-kl 0.01 (KILLED ep18 by W-spike guard)

**Recipe**: v37 base + `--patch-embed` (Conv1d patch embedding before critic LSTM, TTS-GAN style) + stronger CFG (`--cond-drop-prob 0.5`) + stronger variational reg (`--var-cond-kl-weight 0.01`).

**Result**: Killed at ep18. W trajectory: ep15=3.53, ep16=4.21, ep17=4.59, ep18=4.80 — 3 consecutive >4.0. best.pt at ep15 (0.14700) is unusable. Critic loss G≈-3 throughout — patch embedding is producing critic features that the spectral-norm bound cannot constrain at lr_d=4e-5.

**Lesson**: patch_embed is yet another critic-architecture change that breaks wgan-sn at the standard lr_d. Same pattern as v71/v72 — any critic-side change on alibaba/tencent at lr_d=4e-5 destabilizes within 5–10 epochs. Either reduce lr_d to ~1e-5 when enabling patch_embed, or pretrain the modified critic from scratch. Park as "needs lr_d retune."

---

## Post-Mortem: alibaba_v40 — feat-critic-weight=0.5 (dual discriminator) on v37 recipe (FAILED, full eval 0.158, β-recall collapsed 0.564→0.333)

**Recipe**: v37 winning recipe (supervisor=5, diversity=2.0, n-regimes=2, var-cond, gmm 8) + `--feat-critic-weight 0.5` (dual discriminator: latent-space C + feature-space C_feat).

**Result**: Trained 200/200 epochs. Training-log best comb=**0.10347** ep115 (recall=0.564, MMD²=0.01627). **Full eval: combined=0.158** (MMD²=0.02459, **β-recall=0.3325** — collapsed from training-log 0.564), α-precision=0.6410, density=0.5782, DMD-GEN=0.7476, HRC-MAE=0.0144. **100% worse than v37 ATB 0.0786.**

**Lesson**: The dual discriminator gave the largest *training-log* improvement of any alibaba run this session (best 0.10347, recall 0.564) but produced the worst *full-eval* recall collapse — train/eval gap of nearly 50% on β-recall alone. The feature-space critic appears to teach G to game the critic-feature distribution at training time but produces samples that collapse on the eval-time TimeGAN-style PRDC measurement. This is the same train/eval inversion that killed v36 (supervisor=10) and v38 (block_sample). Closing feat-critic on alibaba.

**Pattern across the alibaba sweep**: every recipe layered on top of v37 (block_sample, multi-scale critic, feat-critic) has produced equal-or-better training-log combined but failed at full eval due to recall collapse. **v37 (supervisor=5 + diversity=2.0) combined=0.0786 remains the alibaba record** and looks increasingly like a hard local optimum reachable only by that specific G-side recipe.

---

## Post-Mortem: tencent_v73 — supervisor-steps=2 (SeriesGAN 2-step) (KILLED ep133, best 0.12419 ep90, plateau then degradation)

**Recipe**: v68 base (n-regimes=8, supervisor=5, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2, **no diversity-loss**) + `--supervisor-steps 2` (SeriesGAN 2-step latent supervisor).

**Result**: Killed at ep133/200. Training-log best comb=**0.12419** ep90 (recall=0.444, MMD²=**0.01289**). Three consecutive worse evals after ep90: 0.12419 → 0.13549 (ep110) → 0.16611 (ep130). Cosine LR already 70% decayed (3.6e-5 → 2.5e-5) by the time degradation set in, so no recovery runway.

**Lesson**: 2-step supervisor improved MMD² noticeably (down to 0.01289 — ATB-level) but the 2-step latent unrolling introduces a coupling between supervisor accuracy and generator drift: as G shifts, the 2-step supervisor target itself shifts, and the loss surface becomes harder to follow under cosine LR decay. The result is a sharp peak followed by post-peak degradation. **Best non-killed tencent training-log of the session** but never reached ATB. Could potentially work with a *much* slower cosine decay (decay=0.2 instead of 0.05) so LR stays high enough to track the moving target. Park as "needs slower LR decay."

**v31 co-ATB combined=0.089 remains the tencent record.**

---

## Post-Mortem: alibaba_v39 — multi-scale critic on v37 recipe (KILLED ep139, best 0.12129 ep95, FAILED 54% above ATB)

**Recipe**: v37 winning recipe (supervisor=5, diversity=2.0, n-regimes=2, var-cond, gmm 8) + `--multi-scale-critic` (idea #8, 3-scale T/T/2/T/4).

**Result**: Killed at ep139/200. Training-log best comb=**0.12129** ep95 (recall=0.507, MMD²=0.02269). Bounced 0.121–0.131 for ~40 epochs after ep95 with cosine LR already past 60% decay. Would need 35% drop in 60 remaining epochs to catch v37 ATB 0.0786 — judged hopeless. No full eval.

**Lesson**: Multi-scale critic on alibaba sees the same plateau pattern as on tencent (v70): training-log best is decent but the curve flattens early and never recovers via cosine pickup. The 3-scale critic appears to *bound* what the generator can match (since coarser scales are easier to satisfy) rather than push it. Combined with v70 evidence, multi-scale critic is now closed as a primary lever — only viable as a fine-tune from a strong single-scale baseline, not a from-scratch joint phase.

**v37 (supervisor=5 + diversity=2.0) combined=0.0786 remains the alibaba record.**

---

## Post-Mortem: tencent_v72 — PacGAN pack-size=2 (KILLED ep9 by W-spike guard, identical failure to v71)

**Recipe**: v68 base + `--pack-size 2` (Lin et al. NeurIPS 2018, paired-window critic for mode-collapse detection).

**Result**: Killed at ep9. W trajectory: ep5=1.70, ep6=2.73, ep7=8.08, ep8=14.71, ep9=18.65 — explosive divergence from ep6 onward. best.pt saved at ep5 (0.19309, recall=0.354) is well below ATB 0.089.

**Lesson**: PacGAN's paired-window concatenation effectively doubles the critic input dimensionality at each step, and the wgan-sn spectral-norm bound on the critic LSTM is tuned for the singleton input. The W-distance blew through the Lipschitz bound within 4 epochs. Like v71, this is a critic-architecture change that needs lr_d reduction (try `--lr-d 2e-5`) or stronger SN to remain stable. Park as "needs lr re-tune"; not a tencent ATB tool in current form.

**Pattern emerging**: every critic-side change tried on tencent in this sweep (v71 self-diag, v72 PacGAN) destabilizes wgan-sn within ~5–9 epochs at the standard lr_d=4e-5. Multi-scale critic (v70) is the lone exception and was the most promising. Pivoting to G-side ideas while alibaba_v39 finishes.

---

## Post-Mortem: tencent_v71 — self-diag-temp 2.0 (KILLED ep9 by W-spike guard, critic destabilization)

**Recipe**: v68 base (n-regimes=8, supervisor=5, no diversity-loss) + `--self-diag-temp 2.0` (idea #9, upweights real samples with high critic score).

**Result**: Killed at ep9. W trajectory: ep5=1.94, ep6=3.94, ep7=5.65, ep8=6.24, ep9=6.86 — 3 consecutive >4.0 triggered W-stop guard. best.pt saved at ep5 (0.16478) is unusable.

**Lesson**: Self-diagnosing upweighting at temp=2.0 destabilizes wgan-sn critic immediately — the per-sample weighting amplifies the loss on hard samples too aggressively, breaking the critic Lipschitz bound. Either temp must be much higher (softer weighting, e.g. 5.0+) or self-diag must be paired with reduced lr_d or stronger spectral norm. Skip on tencent for now.

---

## Post-Mortem: tencent_v70 — multi-scale critic on tencent (KILLED ep73, training-log 0.10224★, glacial throughput due to GPU contention with alibaba_v39)

**Recipe**: v68 recipe (n-regimes=8, supervisor=5, cond-dim 10, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2; **no diversity-loss this time**) + `--multi-scale-critic` (idea #8, 3-scale T/T/2/T/4).

**Result**: Killed at ep73/200. Training-log best comb=**0.10224** ep70 (recall=0.537, MMD²=0.00974). This is the best non-spike-killed tencent training-log we've seen on a clean run. **No full eval run** — at projected 30% train/eval gap that gives ~0.133 vs ATB 0.089 (49% worse), and the GPU-contention crawl was preventing alibaba_v39 (newer multi-scale critic test on alibaba) from making progress. Decision: kill the lower-yield run, free GPU.

**Lesson**: Multi-scale critic does shape signal on tencent (best clean training-log seen so far), but at training cost (3× critic forward passes) that doubles when paired with another multi-scale run. Not killable yet as an idea on tencent — should retry as the *only* run on the GPU after alibaba_v39 finishes. For now: park the v70 result and try a critic-side idea that doesn't multiply forward passes.

**v31 co-ATB combined=0.089 remains the tencent record.**

---

## Post-Mortem: alibaba_v38 — block_sample + v37 recipe (training-log 0.09355★ → full eval 0.09439, FAILED, 20% worse than v37)

**Recipe**: v37 winning recipe (supervisor=5, diversity=2.0, cond-dim 10, n-regimes 2, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2) + `--block-sample` (idea #13).

**Result**: Ran all 200 epochs cleanly. Training-log best comb=**0.09355** ep180 (recall=0.592). Late-epoch W spikes to 5.7/5.5 ep172/173 but never 3-consec. **Full eval best.pt (ep180)**: MMD²=0.01719 (vs v37 0.01149 — regress), **β-recall=0.6140** (vs v37 0.6645 — slight regress), α-prec=0.8370 (vs v37 0.9050 — regress), DMD-GEN=0.7446, **HRC-MAE=0.0022** (vs v37 0.0227 — **10× improvement**, near perfect cache fidelity), Context-FID=0.15. **Combined = 0.09439**, vs v37's **0.0786 → 20% worse**.

**Lesson**: block_sample on alibaba dramatically improves HRC-MAE (10× better cache fidelity) but at a net cost to MMD² and α-precision. The temporal coherence gained from contiguous-block batching helps cache footprint reproduction (the cache only sees order, not content quality) but seems to slightly degrade per-window content quality. block_sample is therefore a *cache-fidelity tool*, not an ATB tool. If cache fidelity becomes the primary metric, revisit. For combined metric, v37's random sampling wins.

**v37 (combined=0.0786) remains the alibaba record.**

---

## Post-Mortem: tencent_v69 — block_sample on tencent (training-log 0.10629★ → full eval 0.1378, FAILED, 30% gap)

**Recipe**: v68 recipe (n-regimes=8, supervisor=5, diversity=2.0, cond-dim 10, var-cond, gmm 8, cross-cov 2.0, locality 1.0, ACF 0.2) + `--block-sample` (idea #13 — preserves temporal coherence by streaming contiguous file blocks rather than random files per epoch).

**Result**: Ran all 200 epochs. Training-log best comb=**0.10629** ep60 (recall=0.531). One brief W=4.04 spike at ep174 (no kill, recovered). **Full eval best.pt (ep60)**: MMD²=0.02427, **recall=0.4325**, α-prec=0.811, DMD-GEN=0.7319, AutoCorr=0.0397, Context-FID=0.51, HRC-MAE=0.0404. **Combined = 0.1378**, vs tencent ATB 0.089 → **54% worse**. Train/eval gap = 30%.

**Lesson**: block_sample (which is ATB-validated for alibaba where Hurst=0.98) does NOT help tencent (Hurst=0.79). Tencent's train/eval gap is structural — every G-side knob (continuity, feat-critic, diversity, supervisor, block-sample) hits the same ~30% wall. The remaining levers are critic-side architectural changes (multi-scale critic, projection critic with proper lr_d) and pretrain-side (deeper LSTM, mixed-type heads — both require fresh pretrain).

**v31 co-ATB combined=0.089 remains the tencent record.**

---

## 🏆 alibaba_v37 — NEW ALIBABA RECORD (combined=0.0786, supervisor=5 + diversity=2.0)

**Recipe**: v34 ATB recipe (supervisor=5) + `--diversity-loss-weight 2.0` (carried over from tencent_v68 hypothesis). Otherwise identical: cond-dim 10, n-regimes 2, var-cond, gmm-components 8, cross-cov 2.0, locality 1.0, ACF 0.2.

**Result**: Best training-log comb=**0.09318** ep110 (recall=0.616 — historic high). W-spike killed at ep181 (cosine LR pickup pushed critic too hard, brief stress in late epochs). **Full eval best.pt (ep110)**: MMD²=**0.01149**, **β-recall=0.6645** (new alibaba record), **α-precision=0.9050** (vs v34's 0.708 — big jump), DMD-GEN=0.7593 (slight regression), Context-FID=**0.05** (vs v34's 0.21), HRC-MAE=0.0227 (regression from v34's 0.0063). **Combined = 0.0786**, vs v34's 0.0823 → **4.5% improvement, new alibaba record**.

**Lesson**: Diversity loss (MSGAN mode-seeking) DOES help on alibaba even though it failed on tencent — the recall jump from 0.642 → 0.6645 plus α-precision jump from 0.708 → 0.905 is the cleanest dual improvement we've seen. The recipe is now: supervisor=5 (anchors mean trajectory) + diversity=2.0 (forces mode coverage). Train/eval gap actually inverted (training-log 0.093 → full-eval 0.079 — better in eval).

**alibaba_v37 combined=0.0786 is the new alibaba record. v34 (0.0823) demoted.**

---

## Post-Mortem: tencent_v68 — diversity=2.0 + supervisor=5 on tencent (training-log 0.10215★ → full eval 0.1554, FAILED, W-spike killed ep148)

**Recipe**: v57 ATB + `--diversity-loss-weight 2.0 --supervisor-loss-weight 5.0` (combine v34 alibaba winner with mode-coverage attack).

**Result**: Best training-log comb=0.10215 ep70 (recall=0.592 — historic high for tencent). W-spike guard killed at ep148 (W=4.42 × 3). **Full eval best.pt (ep70)**: MMD²=0.04856, **recall=0.466** (vs train-log 0.592), α-precision=0.9350, DMD-GEN=0.6351, **HRC-MAE=0.0666** (poor cache fidelity), Context-FID=1.04. **Combined = 0.1554**, vs ATB 0.089 → **75% worse**. Train/eval gap **52%** — worst yet.

**Lesson**: Diversity loss inflates training-log recall but doesn't transfer to full eval on tencent. tencent's train/eval gap is the dominant problem; G-side knobs that worked on alibaba don't generalize. Need to attack train/eval gap directly, not just push training-log scores.

**v31 co-ATB combined=0.089 remains the tencent record.**

---

## Post-Mortem: alibaba_v36 — supervisor-weight=10 on alibaba (best 0.10373★ ep130 → full eval 0.1441, FAILED)

**Recipe**: v34 ATB recipe with `--supervisor-loss-weight 10.0` (vs v34's 5.0). Otherwise identical: cond-dim 10, n-regimes 2, var-cond, gmm-components 8, cross-cov 2.0, locality 1.0, diversity 1.0, ACF 0.2.

**Result**: Ran all 200 epochs cleanly (no W-spike kill, brief W=5 spike at ep178/199 but not 3-consecutive). Training-log best comb=0.10373 ep130 (recall=0.576). **Full eval best.pt**: MMD²=0.01729, **recall=0.366** (vs v34's 0.642 — collapsed), α-precision=0.802 (vs v34's 0.708, only metric better), DMD-GEN=0.7419 (vs v34 0.7002), HRC-MAE=0.0070, Context-FID=0.13. **Combined = 0.1441**, vs v34's 0.0823 = **75% worse**.

**Lesson**: Supervisor-weight=10 collapses recall on alibaba. v34's weight=5 is the sweet spot; doubling it pushes G to over-commit to supervisor's mean trajectory and lose mode coverage. Training-log was again ~28% optimistic on the failed direction.

**v34 (supervisor=5) remains the alibaba record at combined=0.0823.**

---

## Post-Mortem: tencent_v64 — reuse-rate amplification (KILLED ep183, 0.099★ training-log → 0.128 full eval)

**Recipe**: v57 ATB recipe + `--locality-loss-weight 3.0 --diversity-loss-weight 2.0` (idea #15).

**Result**: Best training-log 0.0986 (ep150). Full eval **0.128** (vs ATB tencent 0.089) — 30% optimistic gap. Reuse rate barely moved (0.018→0.024 vs real 0.183). Late-stage epochs 165–180 all worse than ep150 (0.099–0.130). Killed at ep183/200.

**Lesson**: Hyperparameter pushing of locality/diversity weights does not amplify reuse rate enough to matter; idea #15 needs structural change (sigmoid+BCE on reuse column or post-hoc reuse injection).

---

## Post-Mortem: alibaba_v28 — reuse-rate amplification (KILLED ep125, full eval 0.137)

**Recipe**: v22 ATB recipe + `--locality-loss-weight 1.0 --diversity-loss-weight 1.0` (already at base; mild idea #15).

**Result**: Best training-log 0.106 (ep75). Full eval ep75 **0.137** (MMD²=0.018, recall=0.41) — 29% optimistic gap. ep80–125 all drifted upward (0.124–0.128 training-log). Killed at ep125.

**Lesson**: Same as tencent_v64 — training-log selection misleads by ~30% on alibaba too.

---

## Post-Mortem: tencent_v65 — continuity loss on tencent (training-log 0.09741★ → full eval 0.12869, WORSE than ATB by 45%)

**Recipe**: v57 ATB recipe + `--continuity-loss-weight 0.5`.

**Result**: Ran all 200 epochs. Best training-log comb=0.09741 (ep195). **Full eval best.pt (ep195)**: MMD²=0.01789, recall=0.4460, **comb=0.12869**, DMD-GEN=0.7026, AutoCorr=0.0459, α-prec=0.6625, coverage=0.4460. vs ATB tencent 0.089 → **45% worse**. Training-log was 32% optimistic.

**Lesson**: Continuity loss targets DMD-GEN at train time but does NOT translate to full-eval improvement — DMD-GEN 0.7026 is no better than other recent runs. Late-stage cosine LR pickup confirmed again (best at ep195), but the training-log / full-eval gap persists. Drop continuity loss as ATB candidate on tencent.

---

## Post-Mortem: tencent_v66 — feat-critic=0.5 on tencent (training-log 0.07658★ → full eval 0.11282, WORSE than ATB by 27%)

**Recipe**: v57 ATB recipe + `--feat-critic-weight 0.5`.

**Result**: Ran all 200 epochs. Training-log best comb=0.07658 (ep185, recall=0.651) — first time any tencent training-log broke ATB 0.089 (by 14%). **Full eval best.pt (ep185)**: MMD²=0.01292, recall=0.5005, **comb=0.11282**, DMD-GEN=0.7148, AutoCorr=0.0358, α-prec=0.6225, HRC-MAE=0.0552. vs tencent ATB 0.089 → **27% worse**. Training-log was 47% optimistic — the worst train/eval gap yet seen.

**Lesson**: Feat-critic on tencent inflates training-log MMD² metric (uses different evaluator state from full eval) more than it improves real generation. Recall did jump (0.583→0.651 training-log, 0.5005 full-eval) but not enough. Same conclusion as alibaba_v29: feat-critic helps training-log but the train/eval gap nullifies the apparent gain. Drop feat-critic as ATB candidate on both corpora.

---

## 🏆 NEW ALIBABA RECORD: alibaba_v34 — supervisor-loss-weight=5.0 (full eval 0.0823, BEATS ATB by 25%)

**Recipe**: v22 ATB recipe + `--supervisor-loss-weight 5.0` (default is 1.0).

**Result**: Ran all 200 epochs. Two best peaks: ep70 0.10917★ then late-stage cosine LR pickup → ep195 0.10100★ (recall 0.568). **Full eval best.pt (ep195)**: MMD²=0.01070, recall=0.6420, **comb=0.0823**, α-prec=0.7075, density=1.1002, DMD-GEN=0.7002, AutoCorr=0.0635, Context-FID=0.21, HRC-MAE=0.0063, reuse=0.003 vs real 0.005. vs alibaba ATB 0.110 → **25% BETTER**. Train/eval gap inverted: full eval BETTER than training-log (0.0823 < 0.10100) — first time ever.

**Lesson**: Supervisor-weight boost (5×) is the working G-side knob on alibaba. Pairs the supervisor's stepwise reconstruction signal with the standard adversarial loss at higher relative weight, producing both faster convergence (ep70 first sub-ATB) AND late-stage cosine pickup (ep195 final best). Recall climbed from 0.547 → 0.642, the highest alibaba recall ever measured. New ATB. Expect this to be the new alibaba baseline; future alibaba runs should layer on top of this recipe.

---

## Post-Mortem: tencent_v67 — supervisor-loss-weight=5 on tencent (training-log 0.10567★ → full eval 0.12323, worse than ATB by 38%)

**Recipe**: v57 ATB recipe + `--supervisor-loss-weight 5.0`. Same scalar tweak that produced the new alibaba record in v34.

**Result**: Ran all 200 epochs. Training-log best comb=0.10567 (ep65). **Full eval best.pt (ep65)**: MMD²=0.01453, recall=0.4565, **comb=0.12323**, α-prec=0.833, density=0.924, DMD-GEN=0.7184, AutoCorr=0.0582, Context-FID=0.28, HRC-MAE=0.0372, reuse=0.005 vs real 0.012. vs tencent ATB 0.089 → **38% worse**. Train/eval gap 17% (much smaller than v66's 47% but still loses). Recall 0.456 is the weak spot.

**Lesson**: The supervisor-weight boost that made alibaba explode does not cleanly transfer to tencent — precision is strong (α=0.833) but recall undershoots. Tencent's weakness is coverage, not plausibility, so the next tencent knob should attack mode collapse directly rather than reinforcement of the generator's inner dynamics.

---

## Launched: tencent_v68 — supervisor=5 + diversity-loss-weight=2.0 (up from 1.0)

**Recipe**: v67 recipe + `--diversity-loss-weight 2.0`. MSGAN mode-seeking loss maximises |G(z1)-G(z2)|/|z1-z2| across noise pairs — directly combats β-recall mode collapse, which v67 showed is the tencent bottleneck (recall=0.4565 vs alibaba_v34 recall=0.6420).

---

## Post-Mortem: alibaba_v35 — v34 + proj_critic (W-spike killed at ep6)

**Recipe**: v34 winning recipe + `--proj-critic`. Hypothesis: projection discriminator (Miyato & Koyama) adds inner-product cond term to critic, expected to boost mode coverage when char-file present.

**Result**: G loss exploded immediately (ep2 G=21, ep3 G=47, ep4 G=63). W jumped to 9.7 by ep4. W-spike guard auto-killed at ep6. Best.pt training-log 0.12796 (worse than v34's ep5 0.144 baseline anyway). Total runtime <10 min.

**Lesson**: Adding the projection-critic head on top of an already-tuned wgan-sn critic doubles the adversarial signal without retuning the lr_d / n_critic ratio. The two critic heads (latent space + cond-projection) interfere. proj_critic needs its own balance — likely lower lr_d and/or fresh pretrain. Shelved as plug-in for v34 recipe.

---

## Launched: alibaba_v36 — supervisor-loss-weight=10 (push the winning knob harder)

**Recipe**: v34 winning recipe but `--supervisor-loss-weight 10.0` (was 5.0). Doubles the supervisor's relative weight against the adversarial loss. Hypothesis: if 5× already produced 25% improvement over ATB, 10× may push further — though too high risks underweighting the GAN signal entirely.

---

## Post-Mortem: alibaba_v29 — feat-critic=0.5 (training-log 0.10004★ → full eval 0.12964, 18% worse than ATB)

**Recipe**: v22 ATB recipe + `--feat-critic-weight 0.5`. First feat-critic experiment ever.

**Result**: W-spike guard stopped at ep176. Best ep115 training-log 0.10004★ (first sub-0.11 training-log on alibaba, briefly beating ATB). **Full eval ep115**: MMD²=0.01154, recall=0.4095, **comb=0.12964**, DMD-GEN=0.7471, α-prec=0.755, HRC-MAE=0.0102. vs alibaba ATB 0.110 → 18% worse. Train/eval gap 30% — same pattern as v65.

**Lesson**: Feat-critic helps training-log but the persistent ~30% train/eval gap nullifies the apparent gain. The gap, not the recipe, is now the bottleneck. Consider direct attacks on the gap: stronger regularization, larger val set, or eval-time EMA consistency.

---

## Post-Mortem: alibaba_v30 — BayesGAN 5 critics (KILLED ep14, hopeless)

**Recipe**: v22 ATB recipe + `--bayes-critics 5`. M=5 critic particles with SGLD noise injection.

**Result**: W stuck at 0.15-0.20 for 14 epochs (vs normal 2-4). ep5 comb=0.232, ep10 comb=0.237 (worse). SGLD Gaussian noise drowns out the critic gradient signal; Lipschitz penalty not enforced per-particle so the ensemble effectively collapses to weak critics. Killed ep14.

**Lesson**: BayesGAN doesn't compose with our wgan-sn/feature-matching stack as-is — the SGLD noise needs to be proportional to the existing critic gradient scale, not a fixed value. Shelved.

---

## Post-Mortem: alibaba_v31 — Self-Diagnosing GAN temp=2.0 (AUTO-KILLED ep7 W-spike)

**Recipe**: v22 ATB recipe + `--self-diag-temp 2.0`.

**Result**: W exploded 6.2→6.5→7.0 in ep5-7; W-spike guard stopped training. Self-diag upweighting of high-critic-score reals amplified the critic too fast and broke the Lipschitz balance. Only 1 eval (ep5 comb=0.215).

**Lesson**: Self-diag temp=2.0 is too sharp for this corpus. Could retry at temp=5.0 (gentler) but shelving for now — orthogonal idea (PacGAN) is more promising.

---

## Post-Mortem: alibaba_v32 — PacGAN pack-size=2 (AUTO-KILLED ep7, W=11.18)

**Recipe**: v22 ATB recipe + `--pack-size 2`.

**Result**: W-spike guard stopped at ep7. W=4.7→?→11.18 in three epochs. Critic got too strong too fast when packed into pairs (effectively 2× sample density per forward). Only ep5 eval (comb=0.230).

**Lesson**: Third critic-side knob to blow up in a row (after bayes v30 and self-diag v31). wgan-sn at lr_d=4e-5 is tightly tuned; any extra critic capacity breaks the Lipschitz balance. Shifting to generator-side knobs.

---

## Post-Mortem: alibaba_v33 — FiLM conditioning (KILLED ep27, deteriorating)

**Recipe**: v22 ATB recipe + `--film-cond`.

**Result**: Best ep10 comb=0.156★, then deterioration: ep15=0.160, ep20=0.166, ep25=0.186 (recall 0.407→0.262). W stuck at 0.2-0.4 indicating critic can't keep up with G changes from FiLM injection. Killed ep27 — training-log 0.156 already worse than ATB 0.110 and trending the wrong way.

**Lesson**: FiLM adds cond re-injection at every timestep, but starting from a pretrain that saw no FiLM, γ/β drift and damage the learned latent dynamics. Might need fresh pretrain with film_cond from the start — too expensive to try now.

---

## Launched: alibaba_v34 — supervisor_loss_weight=5 (UNTRIED VALUE)

**Recipe**: v22 ATB recipe + `--supervisor-loss-weight 5.0` (default 1.0). TimeGAN paper uses η=10. Higher supervisor weight strengthens temporal next-step prediction signal, targeting mode coherence. Generator-side only, no architecture change.

---

## Launched: alibaba_v29 — dual feature-space critic (NEVER TRIED)

**Recipe**: v22 ATB recipe + `--feat-critic-weight 0.5`. The `feat_critic` config option (added but never exercised in any prior version) enables a second critic operating on decoded features rather than latent space. Should catch quality problems invisible to the latent critic. Config recommends 0.5–1.0.

---

## Design Decisions

### Auto-drop zero-variance columns (2026-04-05)

**Problem:** R-based family characterization of 30,628 trace files revealed that both
alibaba and tencent_block have `write_ratio=0.0` and `opcode_switch_ratio=0.0` across
ALL files — opcode is always "read". The model was wasting capacity generating variation
in a constant column. Additionally, `obj_id_reuse` is near-degenerate in alibaba (0.4%).

**Root cause:** The oracle_general binary format uses `op=-1` as a sentinel for
"opcode not recorded". Our `_encode_opcode()` was treating -1 as "write" because
`int(-1) != 0`. The R characterization correctly identified write_ratio=0 because
its profiler knows -1 means unknown. So the GAN was learning FAKE read/write variation
from misinterpreted sentinel values.

**Fix (two parts):**
1. Fixed `_encode_opcode()`: negative sentinel values now map to +1.0 (read/unknown)
   instead of -1.0 (write). Only `op=1` is write.
2. Added auto-drop of zero-variance columns in `TracePreprocessor.fit()`: after all
   encoding, columns where `min == max` are removed from `col_names` and not modeled.
   `inverse_transform()` re-inserts dropped columns with their constant values so
   generated output retains the original schema.

**Effect:** alibaba and tencent_block go from 6 columns to 5 (opcode dropped).
Corpora with actual writes (Baleen24, MSR exchange, LCS traces) are unaffected —
auto-drop only fires when a column is truly constant in the training seed data.

**Impact on active runs:** v54 and alibaba_v15 use old preprocessors (serialized in
checkpoint). This fix takes effect on the NEXT run launched with fresh preprocessing.

### Clip normalized values to [-1, 1] — poison point defense (2026-04-05)

**Problem:** v54 mode-collapsed at ep100 (G_loss +4.27, recall 0.550→0.364). Root cause
identified via R outlier analysis: `obj_id_stride` has a massive heavy tail. In tencent_block
1M, 89 files (2.75%) have stride >5x the median, with the worst at **89x median** (36.2σ).
In alibaba, **12.2%** of files exceed 5x median stride, worst at 78.6x.

**Mechanism:** The preprocessor fits min/max normalization on 12 seed files. When a non-seed
file with extreme stride gets sampled (stride=400B vs typical 4.5B), the per-record
`sign(Δ)·log1p(|Δ|)` values exceed the seed range. After min/max normalization, these values
land outside [-1, 1]. The generator outputs Tanh ∈ [-1, 1] and CANNOT produce these values.
The critic learns to use out-of-range values as a free discriminator signal, then when the
generator can't match them, gradients explode → mode collapse.

**Fix:** `np.clip(..., -1.0, 1.0)` in `TracePreprocessor.transform()`. One line. Values
beyond the seed range are clamped to the boundary. This is equivalent to Winsorization at
the seed percentile range and ensures training data always falls within the generator's
Tanh output range.

**Why this wasn't caught earlier:** The 12-file seed set usually represents the
population well enough. But with files_per_epoch=12 over 200 epochs, you sample ~2400
files — statistically guaranteed to hit multiple extreme outliers. The mode collapse
timing (ep100) is consistent with cumulative destabilization from outlier encounters.

---

## Current Runs

### alibaba_v24 — Alibaba + self-diagnosing upweighting (RUNNING)

**Status**: RUNNING adversarial — 2026-04-06. PID on vinge.
**Recipe**: v22 recipe (K=2 implicit, var_cond, GMM K=8, lower lr) + `--self-diag-temp 10.0`.
Should be safer on alibaba than tencent since alibaba W stays naturally low.
**Pretrain**: REUSED from alibaba_v22.
Log: ~/train_alibaba_v24.log.

## Post-Mortem: alibaba_v23 — multi-scale critic (KILLED ep60, 0.199★)

**Status**: KILLED at ep60/200 — 2026-04-06.
**Recipe**: `--multi-scale-critic` + var_cond + GMM K=8 + lr_g=8e-5, lr_d=4e-5.
**Result**: Best combined=0.199★ at ep50 (recall=0.287, MMD²=0.056). Recall stuck 0.27–0.30
for 30 epochs. W flat at 0.28–0.48 — multi-scale critic couldn't discriminate effectively.
Worse than v22 (0.111) at same epoch count.
**Key insight**: Multi-scale critic hurts alibaba. With T=12 and low reuse, the downsampled
scales (T//2=6, T//4=3) have too few timesteps for meaningful discrimination. The critic's
discriminative power was diluted across scales rather than concentrated.
Log: ~/train_alibaba_v23.log.

## Post-Mortem: alibaba_v22 — lower lr + extended training (completed 200ep, 0.111★)

**Status**: COMPLETED 200 epochs — 2026-04-06.
**Recipe**: v18 winning recipe (K=2 + var_cond + GMM K=8) with: lr_g=8e-5 (was 1e-4),
lr_d=4e-5 (was 5e-5), w_stop_threshold=4.0 (was 3.0), epochs=200 (was 150).
**Pretrain**: REUSED from v18 (identical architecture).
**Result**: Best combined=0.111★ at ep195 (did NOT beat ATB 0.110). Smooth training
throughout — no W-spikes. Lower lr eliminated the collapse seen in v18.
**Key insight**: Lower lr prevents collapse and enables full 200-epoch training, but the
recipe plateaus at ~0.111. Need architectural changes (multi-scale critic, etc.) to push
further.
Log: ~/train_alibaba_v22.log.

## Post-Mortem: alibaba_v21 — block sampling + K=2 (W-COLLAPSES — KILLED ep71)

**Status**: KILLED at ep71/150 — 2026-04-06.
**Recipe**: `--block-sample` + `--n-regimes 2` + var_cond + GMM K=8 + clip + auto-drop.
**Result**: Best combined=0.150 at ep70 (recall=0.386). Repeated W-collapses (W→0.004 at
ep32-37, ep56, ep61). v18 (random) was at 0.117 at ep65. Block sampling hurts alibaba —
too-coherent batches make critic's job trivially easy within a single regime.
**Key insight**: Block sampling helps tencent (v60 ahead of v57 at same epoch) but hurts
alibaba. The corpora respond differently despite both having high Hurst exponents.
Log: ~/train_alibaba_v21.log.

### alibaba_v21 — Alibaba + block sampling + K=2 (KILLED)

**Status**: RUNNING adversarial — 2026-04-06. PID 1114371 on vinge.
**Recipe**: `--block-sample` + `--n-regimes 2` + var_cond + GMM K=8 + clip + auto-drop (5 cols) + n_critic=2.
**Pretrain**: REUSED from alibaba_v18 (same architecture, block sampling is dataloader-only change).
**Hypothesis**: alibaba Hurst=0.98 — consecutive files are near-perfectly correlated. Random
sampling destroys temporal structure (block/random ratio=0.503). Block sampling gives the
generator batches with realistic within-batch diversity.
Log: ~/train_alibaba_v21.log.

## Post-Mortem: alibaba_v20 — expanded conditioning cond_dim=13 (FAILING — KILLED ep15)

**Status**: KILLED at ep15/150 — 2026-04-06.
**Recipe**: `--cond-dim 13` + `--n-regimes 2` + var_cond + GMM K=8 + clip + auto-drop.
**Result**: Best combined=0.276★ at ep5 (recall=0.151). Recall dropped to 0.085 at ep15 and
still falling. MMD² extremely high (0.12). v18 (cond_dim=10) had recall=0.386 at ep5.
**Key insight**: Expanded conditioning hurts. The 3 extra dims add noise that overwhelms the
model early. May need progressive feature introduction or pre-normalization. Not worth
pursuing before other ideas.
Log: ~/train_alibaba_v20.log.

## Post-Mortem: alibaba_v19 — 2-layer LSTM + K=2 (MODE COLLAPSE — KILLED ep39)

**Status**: KILLED at ep39/150 — 2026-04-06.
**Recipe**: `--num-lstm-layers 2` + `--n-regimes 2` + var_cond + GMM K=8 + clip + auto-drop.
**Result**: Best combined=0.220★ at ep10 (recall=0.180). Recall collapsed to 0.115-0.154 from
ep15-39. G_loss persistently negative (-3.4 to -5.4) = generator overshooting without diverse
output. alibaba_v18 (1-layer) had recall=0.460 at ep25.
**Key insight**: 2-layer LSTM mode-collapses on alibaba. Deeper network finds degenerate
equilibrium. May need lower lr_g, warmup schedule, or gradient clipping adjustments.
Log: ~/train_alibaba_v19.log.

### tencent_v64 — Tencent + reuse rate amplification (RUNNING)

**Status**: RUNNING adversarial — 2026-04-06.
**Recipe**: v63 base (K=8 regime + lower lr + clean) + `--locality-loss-weight 3.0` (was 1.0)
+ `--diversity-loss-weight 2.0` (was 1.0). Same intervention as alibaba_v27. Targets IDEAS
#15 reuse rate amplification on tencent.
**Pretrain**: REUSED from v57.
Log: ~/train_tencent_v64.log.

## Post-Mortem: tencent_v63 — lower lr, no block-sample (COMPLETED 200ep, full=0.120)

**Status**: COMPLETED 200 epochs — 2026-04-06.
**Recipe**: K=8 regime + lr_g=8e-5, lr_d=4e-5 + w_stop=4.0 + n_critic=2. Clean baseline.
**Result**: Training-log best=0.10301★ at ep175 (recall=0.518, MMD²=0.00671). **Full eval
(2000 samples) on best.pt: combined=0.120** (MMD²=0.0134, recall=0.4665, DMD-GEN=0.692,
HRC-MAE=0.0214, reuse fake=0.018 vs real=0.048).
**Key insight**: TRAINING-LOG estimates are ~14% optimistic vs full eval (0.103 → 0.120).
Always run full eval before celebrating. v63 final 0.120 is WORSE than v22 (0.111) and
v57 (0.108). The lower-lr clean recipe converged smoothly but plateaued. Reuse rate (0.018)
still 2.7× below real (0.048) — motivates v64 reuse amplification.
**Note**: HRC-MAE metric is now in eval.py (LRU cache fidelity, IDEAS #14 partially done).
Log: ~/train_tencent_v63.log.

### alibaba_v28 — Alibaba + intermediate var_cond_kl 0.005 (RUNNING)

**Status**: RUNNING adversarial — 2026-04-06.
**Recipe**: v22 ATB recipe (K=2 + var_cond + GMM 8 + lower lr) + `--var-cond-kl-weight 0.005`
(midpoint between v22's default 0.001 and v25's 0.01). v22 had stability, v25 had recall=0.513
peak — try to capture both. Standard locality/diversity weights.
**Pretrain**: REUSED from alibaba_v22.
Log: ~/train_alibaba_v28.log.

## Post-Mortem: alibaba_v27 — reuse rate amplification (W-STOP ep161, full=0.117)

**Status**: STOPPED by W-spike guard at ep161/200 — 2026-04-06.
**Recipe**: v22 ATB base + `--locality-loss-weight 3.0` + `--diversity-loss-weight 2.0`
(IDEAS #15 reuse rate amplification via existing knobs).
**Result**: Training-log best **0.09728★** at ep155 — but full eval reveals MASSIVE training-log
mismatch.
- ep85 best.pt: full eval **0.117** (MMD²=0.0189, recall=0.5085, HRC-MAE=0.0088, α-prec=0.945)
- ep140 best.pt: full eval **0.121** (MMD²=0.0274, recall=0.534)
- ep155 best.pt: full eval **0.156** (MMD²=0.0274, recall=**0.357 collapsed**)
**Key insight**: Training-log "improvements" past ep85 were ARTIFACTS of EMA/1000-sample noise.
Full eval at ep155 shows recall actually COLLAPSED. Genuine best is ep85 at 0.117 — still WORSE
than ATB 0.110. Reuse amplification via existing knobs did not break ATB.
**Lesson learned**: best.pt selection must use full eval not training-log; training-log can flip
ordering by 30%+. Need to gate best.pt updates on full eval, not optimistic EMA.
Log: ~/train_alibaba_v27.log.

### alibaba_v27 — Alibaba + reuse rate amplification (COMPLETED)

**Status**: STOPPED — see post-mortem above.
**Recipe**: v22 ATB base + `--locality-loss-weight 3.0` + `--diversity-loss-weight 2.0`. Locality loss
weight 3× should push the model harder on stride-repetition, while higher diversity loss
combats mode collapse on reuse events.
**Pretrain**: REUSED from alibaba_v22.
Log: ~/train_alibaba_v27.log.

## Post-Mortem: alibaba_v25 — higher var_cond_kl 0.01 (W-STOP ep151, 0.120★)

**Status**: STOPPED by W-spike guard at ep151/200 — 2026-04-06.
**Recipe**: v22 ATB recipe + `--var-cond-kl-weight 0.01` (10× default) + GMM 8 (no regime).
**Result**: Best combined=**0.120★** at ep45 (recall=**0.513** — highest ever on alibaba,
MMD²=0.023). Sustained at 0.13–0.15 for ~80 epochs, then W climbed to 4.02 triggering guard.
**Key insight**: Higher KL gives huge early recall (0.513 unprecedented) but cannot stabilize.
Did not beat ATB 0.110. The recall ceiling is reachable but conditioning noise makes
end-state convergence impossible. Future: try with lower var_cond_kl + recall annealing.
Log: ~/train_alibaba_v25.log.

## Post-Mortem: alibaba_v26 — continuity loss (KILLED ep34, 0.198★)

**Status**: KILLED at ep34/200 — 2026-04-06.
**Recipe**: v22 ATB recipe + `--continuity-loss-weight 0.5` (re-added to train.py).
**Result**: Best combined=**0.198★** at ep25 (recall=0.305, MMD²=0.059). Almost 2× ATB
(0.110). Recall stuck at 0.27–0.34, no improvement trajectory. Hopeless.
**Key insight**: Continuity loss does not help alibaba — adding the boundary-coherence
penalty either over-constrains the generator or interferes with the GMM/var_cond stack.
Worth retrying on tencent where DMD-GEN gap is more critical.
Log: ~/train_alibaba_v26.log.

## Post-Mortem: alibaba_v25 — higher var_cond KL 0.01 (KILLED ep100, 0.120★)

**Status**: KILLED at ep100/200 — 2026-04-06.
**Recipe**: v22 recipe + `--var-cond-kl-weight 0.01` (10× default).
**Result**: Best combined=0.120★ at ep45 (recall=0.513 — highest ever on alibaba, MMD²=0.023).
After ep45: degraded to 0.13–0.15, recall dropped to 0.41–0.48, W climbed to 2.5–3.2.
**Key insight**: Higher KL weight gave a huge early recall boost (0.513 is unprecedented on
alibaba) but the model couldn't stabilize. The increased conditioning noise creates diversity
but also instability. Close to ATB (0.110) but didn't beat it.
Log: ~/train_alibaba_v25.log.

## Post-Mortem: tencent_v62 — self-diagnosing temp=10 (W-SPIKE GUARD ep60, 0.121★)

**Status**: KILLED by w-spike guard at ep60/200 — 2026-04-06.
**Recipe**: v61 recipe + `--self-diag-temp 10.0`. First attempt with temp=2.0 exploded in
9 epochs. temp=10.0 lasted 60 epochs.
**Result**: Best combined=0.121★ at ep60 (recall=0.505, MMD²=0.022). Self-diag improved recall
(0.505 vs v61's 0.465) and beat v57 (0.108). But W grew steadily from 0.15→4.5, eventually
triggering the 3-consecutive W>4.0 guard.
**Key insight**: Self-diagnosing upweighting has a fundamental positive feedback loop: high
critic scores → more weight → critic pushes scores higher → repeat. temp=10 delays but
doesn't prevent the explosion. Deprioritized.
Log: ~/train_tencent_v62.log.

## Post-Mortem: alibaba_v24 — self-diagnosing temp=10 (W-COLLAPSE — KILLED ep37)

**Status**: KILLED at ep37/200 — 2026-04-06.
**Recipe**: v22 recipe + `--self-diag-temp 10.0`.
**Result**: Best combined=0.171★ at ep20 (recall=0.368). W collapsed to near-zero at ep29-34
(W=0.002–0.05), then spiked to W=3.52 at ep36. Recall dropped to 0.139. G_loss hit 9.9.
**Key insight**: Self-diagnosing is even worse on alibaba — the low natural W means the
softmax weights become degenerate (all weight on ~1 sample) even at temp=10. Deprioritized.
Log: ~/train_alibaba_v24.log.

## Post-Mortem: tencent_v61 — lower lr + block sampling (KILLED ep75, 0.124★)

**Status**: KILLED at ep75/200 — 2026-04-06.
**Recipe**: `--block-sample` + `--n-regimes 8` + lr_g=8e-5 + lr_d=4e-5 + w_stop=4.0.
**Pretrain**: REUSED from v57.
**Result**: Best combined=0.124★ at ep60 (recall=0.465, MMD²=0.017). Training disrupted by
accidental DataLoader worker kill at ep65. Post-restart: recall degraded from 0.465→0.314,
combined worsened to 0.161. W-spike to 2.06 at ep71.
**Key insight**: Lower lr produces smooth W growth (0.6–1.1 pre-crash). The 0.124★ trajectory
was on track, but the restart disruption was unrecoverable.
Log: ~/train_tencent_v61.log.

## Post-Mortem: tencent_v60 — block sampling + K=8 (KILLED ep130, combined=0.119★)

**Status**: KILLED at ep130/200 — 2026-04-06.
**Recipe**: `--block-sample` + `--n-regimes 8` + clip + auto-drop (5 cols) + n_critic=2.
**Pretrain**: REUSED from v57 (same architecture, block sampling is dataloader-only).
**Result**: Best combined=0.119★ at ep115 (recall=0.472, MMD²=0.013). Block sampling helped
early (beat v57's 0.108 trajectory at same epoch), but W-distance oscillated wildly (0.1→3.4).
W near-collapsed at ep112 (W=0.099), recovered briefly for best at ep115, then degraded.
15 epochs without improvement after ep115, recall slipping from 0.472 to 0.397.
**Key insight**: Block sampling helps tencent (v60 best > v57 at same epoch count), but
standard lr produces too-unstable training. The W oscillation pattern (0.1→3.4) suggests
critic and generator are over-responsive. Lower lr (v61) should stabilize.
Log: ~/train_tencent_v60.log.

## Post-Mortem: tencent_v59 — 2-layer LSTM + K=8 (W-COLLAPSE — KILLED ep74)

**Status**: KILLED at ep74/200 — 2026-04-06.
**Recipe**: `--num-lstm-layers 2` + `--n-regimes 8` + clip + auto-drop (5 cols) + n_critic=2.
**Result**: Best combined=0.163★ at ep25 (recall=0.307). W-distance repeatedly collapsed to
near-zero (ep28-31, ep53-56, ep66-74). Recall oscillated 0.14-0.31, never stable.
At ep60, v57 (1-layer) had combined=0.123 — v59 was at 0.176.
**Key insight**: 2-layer LSTM fails on BOTH corpora (alibaba_v19 also mode-collapsed). The
deeper network finds degenerate equilibria where W→0 and generator produces low-diversity
output. Needs different training strategy (lower lr, warmup, dropout between layers).
Log: ~/train_tencent_v59.log.

## Post-Mortem: tencent_v58 — Tencent + K=2 regime sampler (KILLED ep70, recall stuck)

**Status**: KILLED at ep70/200 — 2026-04-06.
**Recipe**: `--n-regimes 2` + clip + auto-drop (5 cols) + n_critic=2. No var_cond/GMM.
**Result**: Best combined=0.201★ at ep35 (recall=0.260). Recall never broke 0.30 through 70 epochs.
v57 (K=8) had recall=0.471 at ep60. G_loss deeply negative (-3.6 to -4.9) with low recall =
classic mode collapse: G found a few modes that fool the critic but doesn't cover the distribution.
**Key insight**: K=2 works for alibaba (0.110★) but NOT for tencent. The silhouette K=2 optimum
was measured on 2020_tencentBlock features — but tencent_block_1M has higher heterogeneity (2.006)
and 23 changepoints. The two corpora may need different K. Tencent needs K≥8.
Log: ~/train_tencent_v58.log.

## Post-Mortem: v57 — Tencent + regime sampler + clip fix (W-SPIKE ep110, combined=0.108★)

**Status**: W-SPIKE GUARD killed at ep110/200 — 2026-04-05.
**Recipe**: IDENTICAL to v54 + clip fix + auto-drop (5 cols). K=8 regime sampler, n_critic=2.
**Result**: Best combined=0.10808★ at ep110 (recall=0.558, MMD²=0.0197). Clip fix kept it alive
through ep110 where v54 collapsed terminally at ep100. Recall climbed steadily: 0.389→0.471→0.526→0.558.
W-distance escalated from ep85 onward (2.8→3.5), eventually triggering the 3.0 guard for 3
consecutive epochs. Second-best Tencent result ever (ATB co-ATB v31/v34 = 0.089).
**Key insight**: Clip fix works — prevented terminal collapse. But W-distance still rises with
epoch count, suggesting the critic eventually dominates. K=2 (v58) may help by reducing
regime fragmentation pressure on the generator.
Log: ~/train_tencent_v57.log.

## Post-Mortem: alibaba_v18 — Alibaba + K=2 regime sampler (W-SPIKE ep103, combined=0.110★ NEW ATB)

**Status**: W-SPIKE GUARD killed at ep103/150 — 2026-04-06.
**Recipe**: var_cond + GMM K=8 + `--n-regimes 2` + clip + auto-drop (5 cols) + n_critic=2.
**Pretrain**: REUSED from alibaba_v16 (stripped regime_sampler K=8→K=2 keys).
**Result**: Best combined=0.11026★ at ep100 (recall=0.527, MMD²=0.0158). **NEW ALIBABA ATB** —
30% better than previous ATB (0.157 from v16). Recall climbed steadily: 0.386→0.460→0.511→0.527.
W-distance escalated ep101-103 (3.85→4.53→3.74), triggering guard.
**Key insight**: K=2 regime sampler validated by R silhouette analysis. K=8 was over-partitioning.
K=2 gave cleaner regime separation → better recall AND better MMD².
Log: ~/train_alibaba_v18.log.

### alibaba_v17 — Alibaba + regime sampler + clip fix, new seed (BAD SEED — KILLED ep37)

**Status**: BAD SEED — killed at ep37 — 2026-04-05.
**Recipe**: Same as v16. Regime sampler K=8 + var_cond + GMM K=8 + clip + auto-drop (5 cols) + n_critic=2.
**Pretrain**: REUSED from alibaba_v16.
**Result**: Recall stuck 0.115–0.130 from ep15–ep37 (v16 had recall=0.312 at same epoch). Best
combined=0.233 at ep15. G_loss went negative (-3.27) at ep36 without recall improvement — classic
bad seed. Generator found an adversarial equilibrium that satisfies the critic without diverse output.
Log: ~/train_alibaba_v17.log.

### alibaba_v16 — Alibaba + regime sampler + auto-drop + clip fix (W-SPIKE KILLED ep112)

**Status**: W-SPIKE KILLED ep112 — 2026-04-05.
**Recipe**: var_cond + GMM K=8 + `--n-regimes 8` + `--cond-drop-prob 0.25` + `--w-stop-threshold 3.0` +
n_critic=2 + lr_d=5e-5 + supervisor_loss_weight=1.0 + full losses + clip fix.
**Pretrain**: FRESH (required — architecture changed from 6→5 columns with auto-drop fix).
**Changes from v15**: (1) auto-drop fix removes constant opcode column (5 cols instead of 6),
(2) n_critic=2 (was 1 in v15 — matching v54's successful config), (3) clip fix.
ep5: recall=0.288, combined=0.203★. W≈0.2, G_loss positive but decreasing — healthy early trajectory.
**Hypothesis**: Same regime sampler as v15 but with clean 5-col architecture. The model no longer
wastes capacity on a degenerate opcode column filled with misinterpreted sentinel values.
Log: ~/train_alibaba_v16.log.

---

## Post-Mortem: v56 — Tencent + BayesGAN M=2 + regime sampler (KILLED ep35, 2026-04-05)

**Best**: ep20 recall=0.329, combined=0.159★.
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.239  | 0.182 ★  |
| 10    | 0.289  | 0.182 ★  |
| 20    | 0.329  | 0.159 ★  |
| 25    | 0.275  | 0.171    |
| 30    | 0.299  | 0.167    |
| 35    | 0.320  | 0.168    |

**Why killed**: Recall stuck at 0.27-0.33, G_loss persistently positive (0.08–1.44). At ep35,
combined=0.168 vs v54's 0.124 at the same epoch — 0.044 behind with no sign of closing.
BayesGAN M=2 dual critics suppressed the generator. The posterior averaging makes the critic
ensemble too robust for the generator to fool.
**Key lesson**: BayesGAN M=2 + regime sampler don't synergize. BayesGAN's anti-collapse is
redundant when the regime sampler already provides structural mode coverage. And the extra
critic strength slows learning without benefit. Rev. Bayes deprioritized for now.

---

## Post-Mortem: v55 — Tencent + proj_critic (W-SPIKE KILLED ep5, 2026-04-05)

**Best**: ep5 recall=0.386, combined=0.166★.
**Why killed**: W-distance exploded: 1.26→2.45→11.55→18.32→**38.62** in 5 epochs (normal: 0.5–2.0).
W-spike guard auto-stopped. The projection discriminator made the critic too powerful — conditioning
on workload type gave it a trivial discriminator signal. G_loss reached -227.
**Key lesson**: `--proj-critic` cannot be used with the current architecture. The critic gets
workload-type information via projection while the generator uses regime sampling — the asymmetry
makes discrimination trivially easy. Would need generator-side parity (e.g. the generator also
receiving the projected conditioning) or a much weaker critic (lower lr_d, fewer n_critic steps).
**Disposition**: Idea #10 shelved. Moving to BayesGAN (v56).

---

## Post-Mortem: v54 — Tencent + regime sampler (KILLED ep112, 2026-04-05)

**Best**: ep75 recall=0.550, combined=0.108★ (MMD²=0.018).
**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 10    | 0.487  | 0.121 ★  |
| 20    | 0.509  | 0.119 ★  |
| 65    | 0.517  | 0.118 ★  |
| 75    | 0.550  | 0.108 ★  |
| 90    | 0.530  | 0.114    |
| 100   | 0.364  | 0.151    |
| 105   | 0.368  | 0.153    |
| 110   | 0.416  | 0.138    |

**Why killed**: Mode collapse at ep100 — G_loss went positive (+4.27). Recall crashed from
0.550→0.364. Partial recovery at ep110 (0.416) but 28% worse than peak. Second G_loss spike
at ep112 (+1.33). No realistic path back to 0.108★.
**Key achievement**: Combined=0.108 is only 0.019 from ATB (0.089). Regime sampler validated.
**Key lesson**: The collapse pattern (G_loss spikes at ep89, ep100, ep112) suggests the critic
loses its gradient signal in late training. Projection discriminator (#10) should help by giving
the critic workload-type conditioning — sharper, more targeted gradients.

---

## Post-Mortem: alibaba_v15 — Alibaba + regime sampler (KILLED ep85, 2026-04-05)

**Best**: ep60 recall=0.368, combined=0.168★.
**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.119  | 0.412 ★  |
| 10    | 0.172  | 0.257 ★  |
| 15    | 0.188  | 0.225 ★  |
| 20    | 0.255  | 0.189 ★  |
| 25    | 0.250  | 0.206    |
| 30    | 0.276  | 0.194    |
| 35    | 0.193  | 0.220    |
| 40    | 0.284  | 0.189 ★  |
| 45    | 0.302  | 0.204    |
| 50    | 0.306  | 0.184 ★  |
| 55    | 0.331  | 0.169 ★  |
| 60    | 0.368  | 0.168 ★  |
| 65    | 0.369  | 0.176    |
| 70    | 0.283  | 0.194    |

**Why killed**: Peaked at ep60. Recall declined from 0.368→0.283 over 10 epochs with no recovery.
Combined stalled at 0.168 — still far from alibaba ATB (0.108). Also: running old 6-column
architecture with broken opcode encoding. Replaced by alibaba_v16 with auto-drop fix (5 columns).
**Key lesson**: n_critic=1 may have been too weak — v54 uses n_critic=2 and has better stability.
v16 uses n_critic=2.

---

## Post-Mortem: alibaba_v13 — Alibaba + pure v5 recipe, fresh seed (KILLED ep54, 2026-04-05)

**Best**: ep25 recall=0.287, combined=0.178★ (MMD²=0.035).
**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.110  | 0.241 ★  |
| 10    | 0.184  | 0.209 ★  |
| 15    | 0.184  | 0.220    |
| 20    | 0.228  | 0.208 ★  |
| 25    | 0.287  | 0.178 ★  |
| 30    | 0.207  | 0.208    |
| 35    | 0.262  | 0.193    |
| 40    | 0.215  | 0.205    |
| 45    | 0.216  | 0.194    |
| 50    | 0.221  | 0.184    |

**Why killed**: Peaked at ep25. Recall declined from 0.287→0.221 over 30 epochs. G_loss oscillated
wildly between -4.5 and +4.6 (ep36-54). W-distance collapsed to 0.05-0.10. No path to alibaba
ATB (0.108). The ep40-50 recall surge that alibaba_v5 had did NOT repeat — bad seed.

---

## Post-Mortem: alibaba_v14 — Alibaba + mixed-type Recovery (KILLED pretrain ep60, 2026-04-05)

**Why killed**: Mixed-type Recovery was disproven in v53 (Tencent: MMD²=0.034, worse than baseline 0.020).
Binary columns are NOT the MMD² bottleneck. Killed at Phase 2.5 ep60/100 to free GPU for alibaba_v15
(regime sampler), which mirrors v54's spectacular early results.

---

## Post-Mortem: v52 — Tencent + pure ATB recipe, bad seed (KILLED ep46, 2026-04-05)

**Best**: ep20 recall=0.383, combined=0.144★ (MMD²=0.020).
**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 20    | 0.383  | 0.144 ★  |
| 25    | 0.305  | 0.172    |
| 30    | 0.327  | 0.162    |
| 35    | 0.296  | 0.161    |
| 40    | 0.329  | 0.161    |
| 45    | 0.331  | 0.154    |

**Root cause**: Bad seed. Recall plateaued at 0.29–0.38 with no upward trend from ep20 to ep46. At recall=0.33, combined floor = 0.134 even with perfect MMD²=0 — no path to ATB (0.089). v31/v34's good seeds had steadily climbing recall through ep70.
**Key lesson**: The recall plateau at 0.30–0.38 is NOT a char-file problem (v52 used window z_global). It's a seed problem — bad seeds plateau regardless of conditioning method.
**Implication**: Need a mechanism (regime sampler, BayesGAN) to make the model LESS seed-dependent rather than hoping for lucky seeds.

---

## Post-Mortem: v53 — Tencent + mixed-type Recovery (KILLED ep19, 2026-04-05)

**Best**: ep5 combined=0.178★, recall=0.278.
**Finding**: Combined flat at 0.178–0.180 from ep5 to ep15. No improvement in 14 epochs.
MMD²=0.034–0.037 — actually WORSE than v52's 0.020. Binary columns were not the MMD² bottleneck.
**Root cause**: The mixed-type Recovery hypothesis was wrong. Sigmoid heads for opcode/reuse
didn't lower MMD² — the evaluation metric was already handling binary fields adequately with Tanh.
The new pretrain may also have been weaker than v28's.
**Lesson**: Mixed-type heads are not a path to lower MMD². Drop this approach.

---

## Post-Mortem: v51 — Tencent + BayesGAN M=2 (KILLED ep25, 2026-04-05)

**Best**: ep5 recall=0.426, combined=0.144★. ep25: combined=0.155, recall=0.367 — oscillating, no recovery.
**Finding**: BayesGAN M=2 gives spectacular early recall (0.426 — highest ever at ep5) but MMD² too high
(0.029–0.035 vs ATB's 0.007–0.011). Combined ceiling: 0.144. Far from ATB=0.089.
**Root cause**: M=2 critics improve recall via posterior averaging but can't simultaneously achieve ATB's
low MMD². The two objectives are in tension with BayesGAN regularization.
**Lesson**: BayesGAN improves recall ceiling but not combined score. The combined metric requires both
low MMD² AND high recall — BayesGAN achieves one at the cost of the other.

---

## Post-Mortem: alibaba_v12 — Alibaba + BayesGAN M=2 (KILLED ep58, 2026-04-05)

**Best**: ep40 recall=0.395, combined=0.146★. ep55+: plateaued at 0.146–0.152.
**Finding**: BayesGAN M=2 helped Alibaba more than Tencent (0.146 vs v11's 0.168), but still far
from alibaba ATB (0.108). Peaked ep40, flat thereafter. G_loss trending negative (−1.0 to −2.0)
indicating incipient critic dominance.
**Lesson**: Neither BayesGAN M=2 nor M=5 can break the Alibaba recall ceiling toward v5's 0.560.
Need architectural change.

---

## Post-Mortem: v50 — Tencent + seed=42 (KILLED ep30)

---

## Post-Mortem: v50 — Tencent + seed=42 (KILLED ep30, 2026-04-05)

**Best**: ep15 combined=0.185★, recall=0.310. Identical plateau to all single-critic runs.
**Lesson**: Seed=42 is not lucky. Random seed variance ≈ ±0.020 combined — not exploitable by choice.

---

## Post-Mortem: v49 — Tencent + BayesGAN M=5 (KILLED ep35, 2026-04-05)

**Best**: ep10 recall=0.370, combined=0.152★. ep35: recall=0.179, combined=0.201 — continuously declining.
**Root cause**: n_critic=2 × M=5 = 10 total critic weight updates per G step. Too aggressive — G falls
behind the critic collective, gradients degrade, recall collapses after ep10.
**Lesson**: BayesGAN M scales the effective critic strength. At M=5+n_critic=2 the critics dominate.
Try M=2 (v51) or reduce n_critic to 1.

---

## Post-Mortem: alibaba_v11 — Alibaba + BayesGAN M=5 + v5 recipe (KILLED ep55, 2026-04-05)

**Best**: ep35 combined=0.168★, recall=0.340. Peaked ep35, stuck 0.238-0.287 through ep55.
**Finding**: M=5 critics prevented v5's post-ep50 explosion but also capped recall at 0.340 — too
conservative. 5 particles averaging G signal reduced the gradient quality G needs to climb.
**Lesson**: BayesGAN M=5 over-regularizes for Alibaba. Try M=2 (alibaba_v12).

---

## Post-Mortem: alibaba_v10 — Alibaba + var_cond + CFG, no GMM (KILLED ep65, 2026-04-05)

**Best**: ep55 combined=0.163★, recall=0.306. Peaked ep55-60, declining by ep65 (recall=0.284).
**Pattern**: G_loss large-negative (-3 to -7) from ep35 onward — G winning too easily, same critic
dominance pattern. CFG prevented explosion (no 1.8T blowup) but couldn't prevent G/C imbalance.
**Lesson**: CFG alone doesn't fix the G/C balance problem. Need structural solution (BayesGAN).

---

## Post-Mortem: v48 — Tencent + v28 pretrain + ATB recipe (KILLED ep56, 2026-04-05)

**Best**: ep20 combined=0.161★, recall=0.309. Plateau at recall=0.264-0.309 from ep20 to ep55.
**Finding**: v28 pretrain DID improve early learning (ep20 better than v47 by 0.018 combined).
But the recall ceiling around 0.30 prevented any trajectory toward ATB's 0.596.
**Lesson**: Even v28 pretrain + exact ATB recipe can't reproduce ATB in a single run.
Hypothesis: BayesGAN might break the recall ceiling by preventing single-critic boundary collapse.

---

## Post-Mortem: alibaba_v9 — Alibaba lr_d=2.5e-5 (KILLED ep25, 2026-04-05)

**Best**: ep25 recall=0.106, combined=0.281. Recall flat from ep10 to ep25.
**Root cause**: lr_d=2.5e-5 made the critic too weak. W-dist stayed at 0.08-0.12 throughout — no gradient signal to G.
**Lesson**: Alibaba needs lr_d=5e-5 for stable training. The post-ep50 decline in v5 was critic dominance, not lr_d being too high.

---

## Post-Mortem: alibaba_v8 — Alibaba hot-start from Phase 3 checkpoint (KILLED ep5, 2026-04-05)

**Best**: ep5 recall≈0, MMD²=0.19–0.27, combined≈0.23+. Complete mode collapse.
**Root cause**: pretrain_complete.pt was set to alibaba_v5/epoch_0050.pt (Phase 3 checkpoint). The
pretrain loader restores G but NOT C/opt_C, so critic starts fresh. G (trained against strong ep50 critic)
immediately fools a weak new critic → G_loss = −3 to −5, recall≈0.
**Lesson**: NEVER use Phase 3 checkpoints as pretrain_complete.pt. Only Phase 1-2.5 pretrain checkpoints
(where C hasn't been trained adversarially) are safe as hot-start pretrains.

---

## Post-Mortem: v47 — Tencent, v43 pretrain, n_critic=2 (KILLED ep65, 2026-04-05)

**Best**: ep65 combined=0.149★, recall=0.429.
**Finding**: Consistently WORSE than v43 (same epoch v43=0.122). n_critic=2 alone cannot reproduce ATB.
- v47 ep65: combined=0.149, recall=0.429
- v43 ep75: combined=0.117, recall=0.476 (v43's best)
**Remaining hypothesis**: v31/v34 used v28 pretrain; v47 used v43 pretrain. v48 tests this directly.
**Lesson**: n_critic=2 is not the key ATB ingredient — or v28 pretrain quality compounds with it.

---

## Post-Mortem: alibaba_v7 — n_critic=2 + CFG + var_cond (KILLED ep20, 2026-04-05)

**Best**: ep20 recall=0.180, combined=0.210 — far below v1 ep20 (recall=0.441).
**Root cause**: n_critic=2 + var_cond too aggressive for Alibaba. Stronger critic + conditioning noise.
**Lesson**: n_critic=2 helps Tencent (no var_cond) but hurts Alibaba (with var_cond). Keep n_critic=1.

---

## Post-Mortem: v46 — Tencent + GMM K=8 + CFG 0.20 (KILLED ep95, 2026-04-05)

**Best**: ep90 combined=0.125★, recall=0.463.
**Finding**: GMM+CFG performed WORSE than pure CFG (v43):
- v43 ep75: combined=0.117★, recall=0.476 (v43's final best)
- v46 ep90: combined=0.125★, recall=0.463 (v46's best at comparable epoch)
GMM prior raises MMD² (more spread out latent space) without improving recall enough to compensate.
**Lesson**: GMM + CFG interfere. GMM alone only marginally helps (v45: 0.117→0.119). The real ATB
ingredient was n_critic=2, not GMM. Restoring n_critic=2 without GMM is v47.

---

## Current Run: v46 — Tencent + GMM K=8 + CFG 0.20 + cond_dim=10 + n_critic=1

**Status**: RUNNING — 2026-04-05. PID 732258 on vinge. At ep42.
**Recipe**: WGAN-SN + GMM K=8 + CFG cond_drop=0.20 + cond_dim=10 + char-file + n_critic=1 + lr_d=5e-5.
**Pretrain**: v45/pretrain_complete.pt (GMM K=8 architecture, cond_dim=10).
**Goal**: Beat ATB (0.089).

**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.382  | 0.157 ★  |
| 10    | 0.257  | 0.197    |
| 15    | 0.272  | 0.178    |
| 20    | 0.331  | 0.176    |
| 25    | 0.349  | 0.160    |
| **30** | **0.392** | **0.157** ★ |
| 35    | 0.425  | 0.149 ★  |
| 40    | 0.388  | 0.164    |

**Notable**: Ascending trend; ep35=0.425 best so far. ep5 recall (0.382) was 2.6× v43 and 30% above v45 — GMM+CFG synergy confirmed. Currently behind v43's ep35 pace (v43 had 0.463 at ep35).

---

## Post-Mortem: v45 — Tencent + GMM K=8, no CFG (KILLED ep131, 2026-04-05)

**Status**: KILLED ep131. Plateaued — best not improved for 11 epochs.
**Best**: ep120 EMA combined=0.119★, recall=0.483.
**Recipe**: WGAN-SN + GMM K=8 + cond_dim=10 + char-file + n_critic=1 + lr_d=5e-5. NO CFG.

**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 120   | 0.483  | 0.119 ★  |
| 125   | 0.474  | 0.137    |
| 130   | 0.437  | 0.129    |

**Finding**: GMM K=8 helps vs v43 (0.117→0.119, recall 0.476→0.483) but not enough to beat ATB.
Recall stuck at ~0.48; ATB needs ~0.60+. Adding CFG is the next logical step (v46).

---

## Current Run: alibaba_v6 — Alibaba hot-start from v5/ep50, lower lr_d=3e-5

**Status**: RUNNING — 2026-04-05. PID 747517 on vinge.
**What's new**: Hot-start from alibaba_v5/epoch_0050.pt (best-ever Alibaba checkpoint, combined=0.108★).
Reset optimizer with lr_d=3e-5 (vs 5e-5) to prevent critic from dominating again.
**Goal**: Push recall from 0.560 toward 0.65+, beat ATB (0.089). 100 epochs from ep51.

---

## Post-Mortem: alibaba_v5 — Alibaba + GMM K=8 + var-cond + explosion guard (KILLED ep85, 2026-04-05)

**Best**: ep50 EMA combined=0.108★, recall=0.560 — best Alibaba result ever.

**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 50    | 0.560  | 0.108 ★  |
| 55    | 0.442  | 0.137    |
| 60    | 0.520  | 0.112    |
| 65    | 0.540  | 0.121    |
| 70    | 0.470  | 0.125    |
| 75    | 0.443  | 0.141    |
| 80    | 0.371  | 0.155    |
| 85    | 0.431  | 0.143    |

**Root cause of decline**: G_loss grew to 9.3 at ep80 — critic dominated G as lr_d=5e-5 accumulated.
**Lesson**: Lower lr_d for Alibaba with var_cond. Hot-restart from ep50 with lr_d=3e-5 (→ alibaba_v6).

---

## Current Run: alibaba_v5 — Alibaba + GMM K=8 + var-cond + FULL explosion guard + n_critic=1

**Status**: RUNNING — 2026-04-05. PID 726415 on vinge. At ep60.
**What's new vs alibaba_v4**: Fixed explosion guard — also skip when `g_loss.abs() > 1e6`.
**Pretrain**: reusing alibaba_v1/pretrain_complete.pt.
**Goal**: beat ATB (0.089).

**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.123  | 0.251 ★  |
| 10    | 0.262  | 0.205 ★  |
| 15    | 0.318  | 0.175 ★  |
| 20    | 0.251  | 0.191    |
| 25    | 0.241  | 0.197    |
| 30    | 0.256  | 0.191    |
| 35    | 0.339  | 0.171 ★  |
| 40    | 0.423  | 0.151 ★  |
| 45    | 0.508  | 0.127 ★  |
| **50** | **0.560** | **0.108** ★ |
| 55    | 0.442  | 0.137    |
| 60    | 0.520  | 0.112    |

**Notable**: ep50 combined=0.108 is the best Alibaba result ever. Recall=0.560 surpasses all previous Alibaba runs. G_loss is large (5-7) but stable — no explosions (guard working). Best Alibaba ever, approaching ATB.

---

## Post-Mortem: alibaba_v4 — Alibaba + GMM K=8 + var-cond + incomplete NaN guard (KILLED ep25, 2026-04-05)

**Status**: KILLED ep25. Damaged by ep24 explosion (guard was insufficient).
**Best early**: ep5 combined=0.290, ep6 clean (G finite), ep7 clean.
**Damage**: ep24 G_loss=1.9T passed the `isfinite()` guard (finite number!) → corrupted G weights.
ep25 eval: combined=0.192, recall=0.281 — much worse than alibaba_v1 ep25 (0.122/0.509).

**Root cause of guard failure**: The guard was `if not torch.isfinite(g_loss)` but 1.9T is a
valid float64. `isfinite(1.9e12)` returns True. Need `abs() > 1e6` check in addition.

**Fix applied**: `if not torch.isfinite(g_loss) or g_loss.abs() > 1e6:` in train.py (committed).

→ alibaba_v5: exact same recipe, corrected guard.

---

## Post-Mortem: alibaba_v3 — Alibaba + GMM K=8 + var-cond (CondEncoder clamped) (KILLED ep10, 2026-04-05)

**Status**: KILLED ep10.
**Best**: ep10 EMA combined=0.206, recall=0.152. WORSE than alibaba_v1.

**Root cause**: CondEncoder output clamp (mu.clamp(-20,20), encoded.clamp(-10,10)) interfered
with normal training. ep5 recall=0.095 vs alibaba_v1 ep5=0.206. The clamp didn't prevent the
ep6 explosion (G=1.8T still occurred) but DID harm the gradient signal for normal epochs.

**Lesson**: Don't clamp CondEncoder outputs. Use NaN-skip guard in train.py instead — it doesn't
touch forward pass behavior, only skips the backward step when loss is non-finite.

---

## Post-Mortem: alibaba_v2 — Alibaba + GMM K=8 (no var-cond) (KILLED ep31, 2026-04-05)

**Status**: KILLED ep31.
**Best**: ep10 EMA combined=0.166, recall=0.354. Did NOT improve beyond ep10.

**Eval progression**:
| Epoch | Recall | Combined |
|-------|--------|----------|
| 5     | 0.116  | 0.259 ★  |
| 10    | 0.354  | 0.166 ★  |
| 20    | 0.259  | 0.178    |
| 30    | 0.247  | 0.188    |

**Root cause**: Without var_cond, Alibaba recall DECLINES after ep10. The CondEncoder provided
useful exploration/diversity that drove alibaba_v1's rapid recall improvement (0.354→0.509 in 15 epochs).
Removing it stalls the model. Tencent doesn't need var_cond, but Alibaba does — the data has
more diverse conditioning vectors and var_cond's noise helps G cover the space.

**EMA crash (fixed)**: alibaba_v2 also exposed an EMA key-mismatch bug when loading a pretrain
from a different architecture (cond_encoder keys in ema_G_state not filtered). Fixed in train.py.

→ alibaba_v3: restore var_cond + fix CondEncoder with output clamping to prevent explosions.

---

## Post-Mortem: alibaba_v1 — Alibaba traces + v44 recipe (GMM K=8 + var-cond + n_critic=1) (KILLED ep40, 2026-04-04)

**Status**: KILLED ep40.
**Best**: ep25 EMA combined=0.122, recall=0.509 ← promising but crashed.

**Eval progression**:
| Epoch | MMD²    | Recall | Combined |
|-------|---------|--------|----------|
| 5     | 0.05534 | 0.206  | 0.214 ★  |
| 10    | 0.03629 | 0.350  | 0.166 ★  |
| 15    | 0.04074 | 0.354  | 0.170    |
| 20    | 0.02392 | 0.441  | 0.136 ★  |
| 25    | 0.02397 | 0.509  | 0.122 ★  |
| 35    | 0.07509 | 0.234  | 0.228    |
| 40    | 0.03177 | 0.415  | 0.149    |

**Root cause**: var_cond G_loss explosions (ep6, ep24, ep28, ep34 — ~1.8 trillion each, every 6-8 epochs).
The Alibaba conditioning vectors have more extreme distributions than Tencent, hitting CondEncoder edge cases.
Crash at ep35 (recall 0.509 → 0.234) after ep34 explosion. Partial recovery at ep40 but not trustworthy.

**Lesson**: var_cond is even more dangerous on Alibaba than Tencent. Remove it. The pretrain (phases 1-2.5)
is good — reuse for alibaba_v2 without var_cond.
**Key positive finding**: Alibaba IS more amenable than Tencent. ep25 recall=0.509 > v43 Tencent recall=0.476 at ep75.

→ alibaba_v2: same recipe as v45 (GMM K=8, no var_cond), reusing alibaba_v1 pretrain_complete.pt.

---

## Current Run: alibaba_v2 — Alibaba traces + v45 recipe (GMM K=8, no var-cond, n_critic=1)

**Status**: RUNNING — 2026-04-04. PID 664211 on vinge.
**Data**: /tiamat/zarathustra/traces/alibaba (237/239 files matched), oracle_general .zst.
**Char-file**: /tiamat/zarathustra/analysis/out/trace_characterizations.jsonl (237 files matched).
**Recipe**: Identical to v44 (GMM K=8 + var-cond + n_critic=1) but on Alibaba data.
  Shortened pretrain: ae=30, sup=30, g=50 epochs.

**Eval progression**:
| Epoch | MMD²    | Recall | Combined |
|-------|---------|--------|----------|
| 5     | 0.05534 | 0.206  | 0.214 ★  |
| 10    | 0.03629 | 0.350  | 0.166 ★  |
| 15    | 0.04074 | 0.354  | 0.170    |
| 20    | 0.02392 | 0.441  | 0.136 ★  |
| 25    | 0.02397 | 0.509  | 0.122 ★  |

**Notable**: Combined=0.122 at ep25 matches v43's Tencent best (0.117 at ep75) — achieved 3× faster!
Recall=0.509 is near v34 ATB territory (0.608). Strong upward trajectory.

**Warning**: Recurring G_loss explosions (ep6, ep24, ep28: ~1.8 trillion each).
Self-recovering via EMA/grad_clip. Root cause: var_cond CondEncoder + extreme Alibaba conditioning
vectors. Frequency increasing (ep24 and ep28 are 4 epochs apart). Monitor closely.

(See Post-Mortem above for full eval table through ep40)

---

---

## Post-Mortem: v38 — char-file + proj_critic (killed ep8, 2026-04-04)

**Recipe**: Fresh pretraining (phases 1/2/2.5 with char-file + proj_critic) → Phase 3 with same.
**Best**: ep5 EMA combined=0.18832, recall=0.290. Did NOT beat ATB (0.089).
**W explosion**: ep1=0.98 → ep4=29.8 → ep7=54.0 → ep8=52.9 (diverging).
**Root cause**: G was Phase-2.5 warmed-up on supervisor only (no critic). Phase 3 introduced
proj_critic which directly uses char-file conditioning. The pretrained G output doesn't match
char-file workload descriptors yet → critic can trivially distinguish using inner(cond_proj(cond),
pooled). The critic quickly saturates (W→∞), G gets useless gradients.
**Lesson**: proj_critic requires G to first learn char-file conditioning BEFORE the critic uses it
for discrimination. Either: (1) train Phase 3 first without proj_critic then fine-tune with it,
(2) use RpGAN which uses relativistic loss and resists saturation, (3) use lower n_critic=1.
→ v39: char-file + RpGAN + PacGAN (NO proj_critic). Reserve proj_critic for v41+ after G
has learned char-file conditional distribution.

---

## KILLED: v38 Phase 3 (W explosion)

**Status**: RUNNING — fresh pretraining (all phases use char-file conditioning). AMP disabled
(--no-amp). AMP NaN root cause: with n_critic=2, scaler.update() is called twice/batch; on any
overflow the scale halves twice; 2 halvings × 250 batches × 3 epochs → scale→0 → unscale_(grad/0)=NaN.
Fix (in code, for v39+): separate scaler_C for critic, scaler for G; C's scale collapse doesn't
cascade into G's scaler.

**What happened**: v38 was initially launched with stale pretrain_complete.pt from v37 (window-level
conditioning) → W-distance explosion at Phase 3 (G didn't know char-file conditioning). Fixed by
deleting stale pretrain and rerunning fresh. Then stale vinge code (v18 era) missing AMP init_scale
fix was synced. Monitor script (~/restart_v38_phase3.sh) will kill Phase 2.5 process when
pretrain_complete.pt appears and restart Phase 3 with updated code (AMP fix + val-file cond_pool +
cross-workload diversity loss).

Launched fresh 2026-04-04. Phase 2.5 G-warmup at ep40/100 as of 13:58 UTC.

**Phase 3 launch command** (auto-triggered by monitor script):
```bash
ssh darrell@192.168.86.30 "cd ~/Zarathustra/llgan && nohup ~/llgan-env/bin/python -u train.py \
  --trace-dir ~/traces/tencent_block_1M --fmt oracle_general \
  --epochs 200 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir ~/checkpoints/tencent_v38 --checkpoint-every 5 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40 \
  --cond-dim 10 --cond-drop-prob 0.25 \
  --char-file ~/traces/characterization/trace_characterizations.jsonl \
  --proj-critic \
  --supervisor-loss-weight 1.0 --lr-g 1e-4 --lr-d 5e-5 \
  --n-critic 2 --supervisor-steps 2 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 \
  --feature-matching-weight 1.0 --moment-loss-weight 0.1 \
  --fft-loss-weight 0.05 --quantile-loss-weight 0.2 --acf-loss-weight 0.2 \
  --locality-loss-weight 1.0 --dmd-ckpt-weight 0 \
  --ema-decay 0.999 --lr-cosine-decay 0.05 --grad-clip 1.0 \
  --hidden-size 256 --latent-dim 24 \
  --no-compile \
  >> ~/train_v38.log 2>&1 &"
```

**Hypothesis**: Stable file-level conditioning closes EMA→full eval gap. Projection
discriminator forces G to cover workload-specific modes → higher recall. Cross-workload
diversity loss (pairs use different file conditioning) pushes G across workload boundaries.

---

## Post-Mortem: v37 — 2-step supervisor (killed ep75, 2026-04-04)

**Recipe**: v34 recipe + `--supervisor-steps 2`. v28 pretrain reused.
**Best**: ep60 EMA combined=0.09226, recall=0.577. Did NOT beat ATB (0.089).
**Trend after ep60**: ep65=0.10159, ep70=0.09929, ep75=0.11546 (declining recall, 0.577→0.486).
**Killed**: recall in consistent decline, 125 epochs remaining offered no recovery path.
**Lesson**: 2-step supervisor alone insufficient; conditional stability needed (→ v38 char-file).

---

## Post-Mortem: v39 — char-file + RpGAN + PacGAN (killed ep30, 2026-04-04)

**Recipe**: v38 pretrain + char-file + RpGAN (--loss rpgan) + PacGAN (--pack-size 2) + NO proj_critic. --no-amp.
**Best**: ep25 EMA combined=0.198, recall=0.218. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|---------|
| 5  | 0.052 | 0.150 | 0.222 |
| 10 | 0.050 | 0.147 | 0.221 |
| 15 | 0.046 | 0.191 | 0.208 |
| 20 | 0.040 | 0.194 | 0.201 |
| 25 | 0.042 | 0.218 | 0.198 |
| 30 | 0.056 | 0.178 | 0.221 (regression, no ★) |

**Root cause**: RpGAN caused C_loss→0 by epoch 7. Relativistic loss became too easy to satisfy:
critic immediately achieved near-perfect separation of real/fake, leaving G with minimal gradient.
Recall never broke 0.22 and oscillated instead of improving. The W→19 and then slight regression
at ep30 confirmed the training was cycling, not converging.

**Lesson**: RpGAN + char-file + PacGAN is NOT the right combo. The WGAN-SN loss from v31/v34
kept the critic training signal alive (W grew steadily to 9, then spiked late). RpGAN saturates
too quickly with char-file conditioning injected. Go back to WGAN-SN.

→ v40: v34 recipe (WGAN-SN) + char-file only. The simplest possible change from the ATB.

---

## Post-Mortem: v40 — v34 recipe + char-file (killed ep64, 2026-04-04)

**Status**: KILLED ep64.
**Best**: ep30 EMA combined=0.140, recall=0.402. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|---------|
| 10 | 0.036 | 0.301 | 0.176 |
| 20 | 0.026 | 0.394 | 0.147 |
| 30 | 0.020 | 0.402 | 0.140 ★ |
| 40 | 0.019 | 0.367 | 0.146 |
| 50 | 0.018 | 0.344 | 0.150 |
| 60 | 0.024 | 0.343 | 0.155 |

**Root cause**: Recall locked in 0.34–0.40 band for 40 epochs with no upward trend. v31 was at
recall≈0.59 by ep64; v40 peaked at 0.402. Stable char-file conditioning produces precise but
narrow distributions — removes the accidental mixture behavior that noisy z_global gave v31/v34.
MMD² bottomed at 0.019 (excellent) but recall ceiling hard to break without multimodal prior.

**Lesson**: char-file conditioning alone is insufficient. Need structural multimodality in the
prior. → v41: WGAN-SN + char-file + GMM prior (K=8 mixture components in noise space).

---

## Post-Mortem: v41 — WGAN-SN + char-file + GMM prior K=8 (W-collapse ep50, 2026-04-04)

**Status**: KILLED ep50 (W-collapse).
**Best**: ep35 EMA combined=0.136, recall=0.406. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|---------|
| 5  | 0.041 | 0.228 | 0.227 |
| 10 | 0.027 | 0.320 | 0.191 |
| 20 | 0.021 | 0.384 | 0.144 |
| 30 | 0.021 | 0.401 | 0.141 |
| 35 | 0.020 | 0.406 | 0.136 ★ |
| 40 | 0.019 | 0.384 | 0.146 |
| 45 | 0.023 | 0.249 | 0.173 (W declining) |
| 50 | 0.055 | 0.100 | 0.235 (W=0.07, collapse) |

**Root cause**: W-distance collapsed at ep50 (W=0.07, down from 1.46 at ep40). Critic over-trained
at lr_d=5e-5 — the same failure mode as v32. With n_critic=2 and 200+ files/epoch, the critic
accumulated enough gradient updates to dominate G. After collapse G gets no useful signal.

**GMM note**: The GMM prior (zero-init means) began to show separation by ep20 (recall crossed 0.384,
same as v40's peak). By ep35 it had clearly pushed past the v40 ceiling (0.406 vs 0.402). The GMM
is working — it's the W-collapse that killed it.

**Lesson**: lr_d=5e-5 + n_critic=2 + GMM leads to W-collapse at ep50 just like v32. Halve lr_d to
2e-5 to slow the critic.

→ v42: v41 recipe + lr_d=2e-5.

---

## Post-Mortem: v42 — WGAN-SN + char-file + GMM K=8 + lr_d=2e-5 (weak critic, killed ep25, 2026-04-04)

**Status**: KILLED ep25.
**Best**: ep5 EMA combined=0.174, recall=0.291. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|---------|
| 5  | 0.032 | 0.291 | 0.174 ★ |
| 10 | 0.038 | 0.240 | 0.190 |
| 15 | 0.041 | 0.215 | 0.198 |
| 20 | 0.028 | 0.247 | 0.179 |
| 25 | 0.029 | 0.224 | 0.184 |

**Root cause**: lr_d=2e-5 produced a permanently under-trained critic. W stuck at 0.13–0.37 for all
25 epochs (vs v41's W growing to 1.46 by ep40). With such a weak critic G received minimal
adversarial gradient and relied entirely on aux losses, pushing toward safe averages. Recall
oscillated 0.21–0.29, well below v41's 0.384 at ep20.

**Lesson**: lr_d=2e-5 is too conservative. The sweet spot is between 2e-5 (too weak) and 5e-5
(W-collapse at ep50). Try: lr_d=5e-5 but n_critic=1 (halve D:G update ratio → same per-step
sharpness but half the total critic updates → delayed W-collapse).

→ v43: v41 recipe + n_critic=1 (was 2).

---

## Post-Mortem: v43 — WGAN-SN + char-file + GMM K=8 + n_critic=1 (plateau ep75, killed ep100, 2026-04-04)

**Status**: KILLED ep100.
**Best**: ep75 EMA combined=0.117, recall=0.476. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|---------|
| 20 | 0.018 | 0.418 | 0.135 ★ |
| 35 | 0.022 | 0.463 | 0.130 ★ |
| 45 | 0.014 | 0.420 | 0.130 ★ |
| 60 | 0.011 | 0.447 | 0.122 ★ |
| 75 | 0.012 | 0.476 | 0.117 ★ |
| 80 | 0.013 | 0.451 | 0.123 |
| 90 | 0.020 | 0.398 | 0.140 |
| 100 | 0.011 | 0.449 | 0.122 |

**Root cause**: Plateau at combined≈0.117 after ep75. 5 consecutive evals without improvement.
No W-collapse (W stable 0.2–0.5 through ep100 — n_critic=1 fix worked perfectly).
The recall ceiling at 0.47–0.48 is the new bottleneck. Need structural multimodality beyond
what the zero-init GMM provides (GMM was silently disabled by cfg bug — see v44 note).

**Critical discovery**: `cfg.gmm_components` was never assigned from `args.gmm_components` —
the `--gmm-components 8` flag was silently ignored in v41–v43. All three ran WITHOUT GMM.
The recall improvements (0.40→0.48) came from n_critic tuning alone. v44 is the first run
with GMM actually active.

→ v44: same recipe + fix cfg bug (GMM truly active) + variational conditioning (--var-cond).

---

## Post-Mortem: v44 — WGAN-SN + char-file + GMM K=8 (fixed) + var-cond + n_critic=1 (KILLED ep72, 2026-04-04)

**Status**: KILLED ep72.
**Best**: ep50 EMA combined=0.132, recall=0.402. Did NOT beat ATB (0.089).

**Eval progression**:
| Epoch | MMD²    | Recall | Combined |
|-------|---------|--------|----------|
| 10    | 0.02124 | 0.326  | 0.156 ★  |
| 20    | 0.02298 | 0.344  | 0.154 ★  |
| 35    | 0.01660 | 0.343  | 0.148 ★  |
| 40    | 0.01586 | 0.357  | 0.144 ★  |
| 45    | 0.01604 | 0.390  | 0.138 ★  |
| 50    | 0.01279 | 0.402  | 0.132 ★  |
| 55    | 0.01538 | 0.366  | 0.142    |
| 60    | 0.01845 | 0.334  | 0.152    |
| 65    | 0.01573 | 0.362  | 0.143    |
| 70    | 0.01436 | 0.338  | 0.147    |

**Root cause**: var_cond (CondEncoder) is the culprit. v43 (identical recipe, no var_cond, no real GMM)
reached best=0.117 at ep75 with recall=0.476. v44 with real GMM + var_cond peaked at 0.132/0.402 —
significantly worse. The variational conditioning adds noise during training that hurts convergence.
The CondEncoder's logvar=-6 (σ≈0.05) wasn't zero enough; subtle perturbation to conditioning
disrupted char-file signal that previously worked well.

**What worked**: W stable through ep72 (n_critic=1 fix still working). GMM K=8 architecture loads
fine from v43 pretrain. The underlying issue is var_cond, not GMM.

**Lesson**: var_cond hurts (at least with KL weight=0.001 and this architecture). Need to test GMM K=8
ALONE (without var_cond) vs v43 baseline to isolate GMM's true contribution.

→ v45: v43 recipe + GMM K=8 (no var_cond). Reusing v43 pretrain_complete.pt.

---

## Current Run: v45 — WGAN-SN + char-file + GMM K=8 + n_critic=1 (no var-cond)

**Status**: RUNNING — 2026-04-04. PID 680333 on vinge. Reusing v43 pretrain_complete.pt.

**What's new vs v43**:
- GMM K=8 now truly active (cfg bug fixed in v44 remains fixed)
- NO var_cond (removed — v44 proved it hurts, best 0.132 vs v43's 0.117)
- Fresh GMM params initialized over v43 pretrain (strict=False auto-detected new gmm_prior.* keys)

**Recipe**: identical to v43 except `--gmm-components 8` is now actually applied.
This is the TRUE test of whether GMM K=8 improves on v43's best of combined=0.117, recall=0.476.

**Early training**:
```
Epoch 1: W=+0.1656  G=-0.9577  t=78s
```

```bash
ssh -i ~/.ssh/id_rsa darrell@192.168.86.30 "cd ~/Zarathustra/llgan && nohup ~/llgan-env/bin/python -u train.py \
  --trace-dir ~/traces/tencent_block_1M --fmt oracle_general \
  --epochs 200 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir ~/checkpoints/tencent_v45 --checkpoint-every 5 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40 \
  --cond-dim 10 --cond-drop-prob 0.25 \
  --char-file ~/traces/characterization/trace_characterizations.jsonl \
  --gmm-components 8 \
  --supervisor-loss-weight 1.0 --lr-g 1e-4 --lr-d 5e-5 \
  --n-critic 1 --supervisor-steps 2 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 \
  --feature-matching-weight 1.0 --moment-loss-weight 0.1 \
  --fft-loss-weight 0.05 --quantile-loss-weight 0.2 --acf-loss-weight 0.2 \
  --locality-loss-weight 1.0 --dmd-ckpt-weight 0 \
  --ema-decay 0.999 --lr-cosine-decay 0.05 --grad-clip 1.0 \
  --hidden-size 256 --latent-dim 24 \
  --no-compile --no-amp \
  > ~/train_v45.log 2>&1 &"
```

---

## Archived: v40 launch command

**Status**: KILLED. See post-mortem above.

**Hypothesis**: v31/v34 achieved ATB=0.089 with WGAN-SN + noisy window-level z_global. The only
known weakness was EMA→full eval gap (from noisy conditioning). Replacing window-level z_global
with stable char-file stats should close this gap and potentially allow even higher recall.

**Recipe**: Identical to v34 except:
- char-file conditioning (--char-file) replaces noisy window-level z_global descriptors
- v38 pretrain (char-file conditioned from Phase 1)
- --no-amp (AMP separate-scaler fix in code but untested)
- WGAN-SN loss (default, NOT rpgan)
- No PacGAN (not in v31/v34)
- No proj_critic

```bash
ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no darrell@192.168.86.30 "mkdir -p ~/checkpoints/tencent_v40 && cp ~/checkpoints/tencent_v38/pretrain_complete.pt ~/checkpoints/tencent_v40/pretrain_complete.pt && cd ~/Zarathustra/llgan && nohup ~/llgan-env/bin/python -u train.py \
  --trace-dir ~/traces/tencent_block_1M --fmt oracle_general \
  --epochs 200 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir ~/checkpoints/tencent_v40 --checkpoint-every 5 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40 \
  --cond-dim 10 --cond-drop-prob 0.25 \
  --char-file ~/traces/characterization/trace_characterizations.jsonl \
  --supervisor-loss-weight 1.0 --lr-g 1e-4 --lr-d 5e-5 \
  --n-critic 2 --supervisor-steps 2 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 \
  --feature-matching-weight 1.0 --moment-loss-weight 0.1 \
  --fft-loss-weight 0.05 --quantile-loss-weight 0.2 --acf-loss-weight 0.2 \
  --locality-loss-weight 1.0 --dmd-ckpt-weight 0 \
  --ema-decay 0.999 --lr-cosine-decay 0.05 --grad-clip 1.0 \
  --hidden-size 256 --latent-dim 24 \
  --no-compile --no-amp \
  > ~/train_v40.log 2>&1 &"
```

## Queued: v41 — v40 + proj_critic (after G has learned char-file conditioning for 200ep)

---

## Completed Runs

| Version | Best MMD² | Best Recall | Best Epoch | Checkpoint | Notes |
|---------|-----------|-------------|------------|------------|-------|
| v3–v8 | ~0.07+ | 0.022–0.143 | varies | wigner archive | Pre-latent-AE; mode collapse |
| v9 | ~0.025 | 0.351 | ~60 | wigner archive | First latent AE + supervisor + FM |
| v13 | **0.00818** | 0.455 | 130 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v13/best.pt | 3234 files; 171 wasted GAN-cycling epochs |
| v14 | 0.041 | 0.618 | 30 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14/best.pt (ep30) | Collapsed ep~100: no LSTM SN, W→31 |
| v14c | 0.046 | 0.269 | 30 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14c/ | n_critic=1 too conservative; stagnated |
| v14d | 0.317 | 0.046 | — | discarded | Immediate collapse; n_critic=2 on non-SN checkpoint |
| v14e | 0.077 | 0.209 | 30 | discarded | Worse than v14c; same hot-start asymmetry |
| v14f | 0.048 | 0.021 | 50 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14f/ | Stable but plateaued; first diversity_loss test |
| v14g | **0.018** | **0.372** | 90 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14g/best.pt | Best MMD²–recall trade-off; full eval reuse_rate=0 |
| v15 | 0.029 | 0.294 | 100 | vinge:~/checkpoints/tencent_v15/best.pt | GAN cycling ep110; missing diversity_loss |
| v16 | 0.042 | 0.228 | 130 | vinge:~/checkpoints/tencent_v16/best.pt | Diversity+cross_cov restored; critic still dominant; worse than v14g |
| v17 | 0.00697 | 0.521 | 190 | vinge:~/checkpoints/tencent_v17/best.pt | Former ATB (v3–v30). supervisor→1.0, lr_d=5e-5, diversity→1.0 |
| v18 | 0.01105 | 0.418 | 205 | vinge:~/checkpoints/tencent_v18/best.pt | cross_cov→5.0; did NOT beat v17; see post-mortem |
| v19 | 0.01094 | 0.518 | 225 | vinge:~/checkpoints/tencent_v19/best.pt | cross_cov→2.0, dmd_ckpt_weight=0.05; recall≈v17 but MMD² worse; see post-mortem |
| v20 | 0.03215 | 0.407 | 210 | vinge:~/checkpoints/tencent_v20/best.pt | 300 epochs; EMA looked great (0.00755/0.549) but full eval diverged 4×; see post-mortem |
| v21 | 0.01485 | 0.366 | 35 | vinge:~/checkpoints/tencent_v21/best.pt | EMA-save fix confirmed (gap inverted); early-stopped ep235; stagnated from ep35; see post-mortem |
| v22 | 0.01751 | 0.471 | 190 | vinge:~/checkpoints/tencent_v22/best.pt | v16 pretrain hypothesis confirmed; α-precision=0.927 new ATB; see post-mortem |
| v23 | 0.01967 | 0.357 | 165 | vinge:~/checkpoints/tencent_v23/best.pt | Weaker seed; late surge ep165 but below v22; DMD-GEN=0.699 best ever; aborted ep187 |
| **v24** | **0.00798** | **0.503** | 170 | vinge:~/checkpoints/tencent_v24/best.pt | **Near-ATB.** dmd_ckpt_weight=0; MMD²=0.008 (2nd best ever); EMA beat v17; see post-mortem |
| v25 | 0.01272 | 0.359 | 50 | vinge:~/checkpoints/tencent_v25/best.pt | Weak seed; W collapsed ep55; aborted ep59 |
| v26 | 0.00795 | 0.401 | 190 | vinge:~/checkpoints/tencent_v26/best.pt | diversity_loss=2.0; MMD² tied 2nd best; DMD-GEN=0.710 best ever; EMA recall inflated |
| v27 | 0.055 | 0.172 | 20 | discarded | AVATAR architecture; critic overpowered from ep1; aborted ep38 |
| v28 | 0.01183 | 0.508 | 70 | vinge:~/checkpoints/tencent_v28/best.pt | **First conditional run (cond_dim=10)**; combined=0.110 best-ever early trajectory; GAN cycling crash ep79 from stripped losses |
| v29 | 0.021 | 0.384 | 20 | vinge:~/checkpoints/tencent_v29/best.pt | cond_dim=10 + full losses; EMA phenomenal (recall=0.674, combined=0.075) but massive EMA gap; GAN cycling crash ep96 |
| v30 | 0.01640 | 0.468 | 25(EMA) | vinge:~/checkpoints/tencent_v30/best.pt | n_critic=3; α-precision=0.929 (tied ATB); aborted ep54 — recall falling, W→2.8; n_critic=3 too aggressive |
| v32 | 0.00797 | 0.441 | 110 | vinge:~/checkpoints/tencent_v32/best.pt | Same recipe as v31 250ep; W→0 at ep110 (critic lost signal); reuse=0.055 (10× improvement); aborted ep144 |
| v35 | 0.01123 | 0.378 | 65(EMA) | vinge:~/checkpoints/tencent_v35/best.pt | CFG 0.30 TOO aggressive; recall collapsed 0.609→0.378 (38% drop); DMD-GEN=0.694 new ATB |
| **v34** | **0.01119** | **0.608** | **150** | **vinge:~/checkpoints/tencent_v34/best.pt** | **CO-ATB (tied v31). β-recall=0.608 NEW ATB. CFG 0.25 > 0.15 for generalization. Most stable conditioned run.** |
| v33 | 0.01080 | 0.521 | 70(EMA) | vinge:~/checkpoints/tencent_v33/best.pt | Same v31 recipe, different seed; W spiked ep179; EMA recall 0.683→0.521 full eval (24% drop); Context-FID=0.02 new ATB |
| **v31** | **0.00769** | **0.596** | **70** | **vinge:~/checkpoints/tencent_v31/best.pt** | **CO-ATB (tied v34). CFG (cond_drop_prob=0.15) + cond_dim=10 + v28 pretrain. Beats v17 by 14%.** |
| v39 | 0.040 | 0.218 | 25 | vinge:~/checkpoints/tencent_v39/epoch_0025.pt | char-file + RpGAN + PacGAN; C_loss→0 ep7; recall oscillated 0.15–0.22; regression at ep30; killed ep30 |
| v38 | — | 0.290 | 5(EMA) | — | char-file + proj_critic; W explosion ep1→52 (proj_critic shortcut); killed ep8 |
| v36 | 0.01023 | 0.471 | 100(EMA) | vinge:~/checkpoints/tencent_v36/best.pt | Same v34 recipe, different seed; EMA claimed recall=0.613/combined=0.087 but full eval: recall=0.471, combined=0.116 (23% gap); W spike ep178; killed ep182 |

---

## Version Notes

### v36 — Bad seed, EMA inflation again (vinge/GB10, killed ep182, 2026-04-04)

**Recipe**: Identical to v34 (cond_drop_prob=0.25, cond_dim=10, full losses, n_critic=2, v28
pretrain, 200 epochs). Different random seed.

**EMA best (ep100)**: MMD²=0.00956, recall=0.613, combined=0.087

**Full eval (ep100)**:

| Metric | v36 | v34 | v31 |
|--------|-----|-----|-----|
| MMD² | 0.01023 | 0.01119 | **0.00769** |
| α-precision | 0.879 | **0.903** | **0.953** |
| β-recall | 0.471 | **0.608** | 0.596 |
| DMD-GEN | 0.775 | 0.747 | **0.723** |
| AutoCorr | 0.070 | 0.053 | **0.032** |
| Context-FID | **0.01** | 0.02 | 0.03 |
| reuse rate | 0.005 | 0.004 | 0.005 |

**Full eval combined = 0.010 + 0.2*(1−0.471) = 0.116. Did NOT beat ATB.**

**Post-mortem**: Third bad-seed run in the v34 recipe family. EMA recall 0.613 → full eval 0.471
(23% gap). Same W-spike pattern as v33 (ep179) and v31 (ep186): W→10 at ep178, recall collapsed
to 0.27, killed at ep182.

Key findings:
1. **Seed variance dominates**: v34 was a lucky seed (3.5% EMA gap); v33 and v36 are unlucky
   (23-24% gap). Two of three runs with the same recipe fail to transfer EMA quality to full eval.
2. **EMA→full eval gap persists despite CFG 0.25**: CFG improved the gap vs unconditioned runs
   (v20 had 4× divergence) but does not solve it — just moves from catastrophic to merely bad.
3. **DMD-GEN=0.775 worst yet** for CFG runs: same recipe, worse temporal dynamics than v34.
   Confirms temporal dynamics are not addressed by conditioning or CFG alone.
4. **Context-FID=0.01 new ATB** (vs v34's 0.02): latent space quality is good; the problem is
   distributional coverage (recall=0.471), not encoding fidelity.

**Direction**: Structural change needed. v37 tries `--supervisor-steps 2` (SeriesGAN) to target
DMD-GEN directly. The repeated-seed approach has exhausted its value with 0.25.

---

### v35 — CFG 0.30 too aggressive (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: Same as v31/v34 but with **cond_drop_prob=0.30** (up from v34's 0.25). v28 pretrain,
cond_dim=10, full losses, n_critic=2, 200 epochs.

**EMA best (ep65)**: MMD²=0.01330, recall=0.609, combined=0.092

**Full eval**:

| Metric | v35 | v34 | v31 |
|--------|-----|-----|-----|
| MMD² | 0.01123 | 0.01119 | **0.00769** |
| α-precision | 0.881 | **0.903** | **0.953** |
| β-recall | 0.378 | **0.608** | 0.596 |
| DMD-GEN | **0.694** | 0.747 | 0.723 |
| AutoCorr | 0.049 | 0.053 | **0.032** |
| Context-FID | **0.07** | 0.02 | 0.03 |
| reuse rate | 0.003 | 0.004 | 0.005 |

**Full eval combined = 0.011 + 0.2*(1-0.378) = 0.136. Did NOT beat v31/v34.**

**Post-mortem**: CFG dropout 0.30 is too aggressive. Dilutes the conditioning signal so much that
the generator loses mode specificity. Massive EMA→full eval recall drop (0.609→0.378, 38%). The
one bright spot is DMD-GEN=0.694, the first run below 0.70 — higher unconditional diversity from
the aggressive dropout, but at the cost of conditional precision.

**CFG dropout sweep conclusion**:

| CFG dropout | Full eval recall | Full eval combined |
|-------------|-----------------|-------------------|
| 0.15 (v31) | 0.596 | 0.089 |
| 0.25 (v34) | 0.608 | 0.089 |
| 0.30 (v35) | 0.378 | 0.136 |

**Sweet spot confirmed at 0.25.** Future runs should use 0.25 unless architecture changes.

---

### v34 — CO-ATB, new recall record (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: Same as v31 but with **cond_drop_prob=0.25** (up from 0.15). v28 pretrain, cond_dim=10,
full losses, n_critic=2, 200 epochs.

**EMA best (ep65)**: MMD²=0.014, recall=0.630, combined=0.088

**Full eval (ep150 best)**:

| Metric | **v34** | v31 | v33 | v17 |
|--------|---------|-----|-----|-----|
| MMD² | 0.01119 | **0.00769** | 0.01080 | **0.00697** |
| α-precision | 0.903 | **0.953** | 0.886 | 0.826 |
| β-recall | **0.608** | 0.596 | 0.521 | 0.521 |
| DMD-GEN | 0.747 | 0.723 | 0.727 | **0.714** |
| AutoCorr | 0.053 | **0.032** | — | **0.032** |
| Context-FID | **0.02** | 0.03 | **0.02** | 0.03 |
| reuse rate | 0.004 | 0.005 | 0.001 | 0.006 |

**Full eval combined = 0.011 + 0.2*(1-0.608) = 0.089. TIED with v31 (0.089).**

**Key findings**:

1. **β-recall=0.608 is NEW ALL-TIME BEST** — first run to break 0.60 on full eval (v31=0.596).
   Higher CFG dropout gives better mode coverage.

2. **CFG dropout 0.25 dramatically improved generalization**: EMA recall 0.630 → full eval 0.608
   (only 3.5% drop). Compare v33 with CFG 0.15: 0.683→0.521 (24% drop). v31 with 0.15 had 0%
   drop but v31 was a lucky seed. 0.25 is more reliably good.

3. **Most stable conditioned run**: No crashes, no W explosion, 20+ consecutive evals with
   combined < 0.110.

4. **Context-FID=0.02 tied ATB** with v33.

5. **Training-time checkpoint selector chose ep65 but ep150 was actually better on full eval.**
   This suggests the EMA-based selector underweights late-training improvements.

**Direction**: Try cond_drop_prob=0.30 in v35 to see if even more dropout further closes the gap.

---

### v33 — Seed variance confirmation (vinge/GB10, completed ep197, 2026-04-01)

**Recipe**: Identical to v31 (cond_dim=10, cond_drop_prob=0.15, full losses, n_critic=2, v28
pretrain, 200 epochs). Note: continuity_loss_weight was 0 despite plan saying 0.5 (incompatible
with AMP).

**EMA best (ep70)**: MMD²=0.01551, recall=0.683, combined=0.079

**Full eval**:

| Metric | v33 | v31 | v17 |
|--------|-----|-----|-----|
| MMD² | 0.01080 | **0.00769** | **0.00697** |
| α-precision | 0.886 | **0.953** | 0.826 |
| β-recall | 0.521 | **0.596** | 0.521 |
| DMD-GEN | 0.727 | 0.723 | **0.714** |
| Context-FID | **0.02** | 0.03 | 0.03 |
| reuse rate | 0.001 | 0.005 | 0.006 |

**Full eval combined = 0.011 + 0.2*(1-0.521) = 0.107. Did NOT beat v31 (0.089).**

**Late crash**: W spiked to 10.6 at ep179/197, recall crashed to 0.022.

**Key findings**:
1. **Seed variance is large**: Same recipe as v31 but 20% worse combined (0.107 vs 0.089).
2. **EMA recall spikes don't hold**: EMA recall 0.683 (best ever seen) dropped 24% to 0.521
   on full eval. Compare v31 where EMA recall 0.591 actually *improved* to 0.596 on full eval.
   Rapid EMA recall climbs may indicate overfitting to conditioning descriptors.
3. **Context-FID=0.02 is new ATB** — only bright spot.
4. **Confirms CFG value**: Even the "bad seed" run still matched v17's recall (0.521) and
   beat it on α-precision (0.886 vs 0.826). Pre-CFG conditional runs (v28, v29) crashed hard.

**Direction**: Higher CFG dropout (cond_drop_prob=0.25) in v34 to reduce EMA→full eval gap.

---

### v31 — NEW ALL-TIME BEST (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: cond_dim=10 + CFG dropout (cond_drop_prob=0.15) + full loss recipe + n_critic=2 +
v28 pretrain (cond_dim=10 compatible) + dmd_ckpt_weight=0. 200 epochs.

**EMA best (ep70)**: MMD²=0.01167, recall=0.591, combined=0.094

**Full eval (ep70 best)**:

| Metric | **v31** | v17 | v24 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00769** | 0.00697 | 0.00798 | <0.005 |
| α-precision | **0.953** | 0.826 | 0.835 | >0.80 |
| β-recall | **0.596** | 0.521 | 0.503 | >0.70 |
| DMD-GEN | 0.723 | **0.714** | 0.717 | <0.30 |
| AutoCorr | **0.032** | 0.032 | 0.036 | <0.02 |
| Context-FID | **0.03** | 0.03 | 0.08 | <0.05 |
| reuse rate | 0.005 | 0.006 | 0.007 | >0.1 |

**Full eval combined = 0.008 + 0.2*(1-0.596) = 0.089. BEATS v17 (0.103) by 14%.**

**What worked — CFG solved the overfitting problem:**

1. **EMA-full eval gap INVERTED**: EMA recall 0.591 → full eval 0.596. This is the first
   conditional run where full eval recall *exceeds* EMA recall. CFG dropout forces the model
   to learn both conditional and unconditional generation, eliminating the descriptor overfitting
   that plagued v28 (crashed ep79) and v29 (43% recall drop on full eval).

2. **α-precision=0.953 — new ATB by a wide margin** (previous: v22=0.927, v30=0.929).
   Conditioning tells G exactly what workload to produce; CFG dropout prevents overfitting
   while preserving the fidelity benefit.

3. **β-recall=0.596 — new ATB by 14%** (v17=0.521). The 0.596 barrier was never even
   approached by unconditional runs. Conditioning provides a coverage mechanism that
   diversity_loss alone could not achieve: G explicitly targets different workload types
   instead of hoping noise diversity maps to mode coverage.

4. **MMD²=0.00769** — 2nd best ever (v17=0.00697). With recall at 0.596 (vs v17's 0.521),
   this represents a much better Pareto trade-off: v31 covers 14% more modes while only
   giving up 0.001 on MMD².

5. **Late W spike (ep186-192, W up to 9.0) didn't matter** — EMA captured best at ep70,
   and the model recovered. The W spike is the same conditioning instability seen in v28/v29,
   but CFG dropout delayed it from ep79/ep96 to ep186, well past the useful training region.

6. **Combined=0.089** — first run below 0.100. This is 14% better than v17 (0.103) and
   17% better than v24 (0.107), the previous #2.

**The winning recipe**:
- v28 pretrain (cond_dim=10 architecture) — NOT v16 pretrain (unconditional architecture)
- cond_dim=10 + cond_drop_prob=0.15 (CFG dropout)
- Full auxiliary losses (cross_cov=2.0, acf=0.2, fft=0.05, moment=0.1, quantile=0.2, locality=1.0)
- n_critic=2, supervisor=1.0, diversity=1.0, lr_g=1e-4, lr_d=5e-5
- dmd_ckpt_weight=0
- ema_decay=0.999, lr_cosine_decay=0.05, grad_clip=1.0

**Remaining problems:**
1. **DMD-GEN=0.723** (target <0.30): Still stubbornly above 0.70. Conditioning improved
   fidelity and coverage but did not help temporal dynamics. This remains architectural.
2. **reuse rate=0.005** (target >0.1): Locality/sequential access still unlearned.
3. **Late W spike**: CFG delayed but did not prevent conditioning instability. Longer runs
   need either lower lr_g or adaptive n_critic to handle late-stage W growth.

**v32 direction**: Same recipe, 250 epochs. v31 peaked at ep70 but sustained excellence
through ep150. Extended cosine decay may allow further convergence before the W spike zone.

---

### v14 (killed, 2026-03-28)
WGAN-SN without spectral norm on the critic LSTM: only the FC output layer was Lipschitz-constrained.
LSTM weights drifted unconstrained — W grew from 2.7 to 31 by epoch 100, recall hit 0.
Best was epoch 30 (first eval) before the drift began.

### v14c–v14e (hot-starts from v14/epoch_0030.pt, 2026-03-28–29)
All failed because loading a generator trained against an unconstrained critic into a freshly
SN-normalised critic creates a permanent power asymmetry. The SN u/v buffers are initialised
randomly; the critic wins decisively until they converge, by which point G is stuck.
**Lesson**: never hot-start from a checkpoint whose critic had different Lipschitz constraints.

### v14f (hot-start from v14c/epoch_0030.pt, killed ep190, 2026-03-29)
First test with diversity_loss_weight=0.5. Stable but plateaued. Full eval revealed that
training-log β-recall (0.269) and eval.py β-recall (0.021) diverge by ~13×:
training uses EMA weights + 1000 samples, eval uses live checkpoint + 2000 samples.
**Lesson**: training-log recall numbers are ~3–4× optimistic; trust full eval.py only.

### v14g (fresh from scratch, 2026-03-29–30)

First clean run with SN-LSTM from epoch 0. Key changes vs v14f:
- Fresh pretraining: AE 50ep + supervisor 50ep (supervisor_steps=2) + G warmup 50ep
- diversity_loss_weight=0.5: MSGAN recall fix (present in v14f, kept)
- cross_cov_loss_weight=0.5: full d×d lag-1 cross-feature covariance (new; targets DMD-GEN)
- lr_g=1e-4, lr_d=5e-5 (original v13 LRs; hot-starts used 5e-5/2.5e-5)

**Full eval (ep90)**: MMD²=0.018, α-precision=0.910, β-recall=0.372, DMD-GEN=0.700, Context-FID=0.03

**Root causes identified (fixed in v15):**
1. `reuse_rate=0.000`: delta-encoding makes obj_id reuse a zero-measure event in continuous space.
   Fix: split obj_id into `obj_id_reuse` (±1 binary) + `obj_id_stride` (signed-log delta).
2. DMD-GEN stuck at 0.700: supervisor_loss_weight (5.0) overpowers cross_cov_loss_weight (0.5) by 10×.
   Fix: raise cross_cov_loss_weight to 2.0 (v16).
3. obj_size non-quantized: generated sizes are continuous; real traces are multiples of 4096 bytes.
   Fix: snap to 4096-byte multiples before log-transform (v15).

### v16 (vinge/GB10, completed ep150, 2026-03-31)

Applied diversity_loss=0.5 (absent in v15), cross_cov=2.0, n_critic=2, lr_d=2.5e-5.
Training ran cleanly to ep150 with no cycling spikes. Best checkpoint: ep130.

**Full eval (ep130)**:
| Metric | v16 | v14g | Δ |
|--------|-----|------|---|
| MMD² | 0.042 | 0.018 | +0.024 worse |
| α-precision | 0.833 | 0.910 | −0.077 |
| β-recall | 0.228 | 0.372 | −0.144 (mode collapse) |
| DMD-GEN | 0.744 | 0.700 | +0.044 worse |
| Context-FID | 0.15 | 0.03 | +0.12 worse |
| reuse rate | 0.004 | 0.000 | tiny improvement |

**What worked:**
- Stable training through ep150: no GAN cycling, W stayed <2.1 (vs ep110 spike in v15).
- Late improvement: best improved ep100→ep130 as LR decayed, G briefly went negative (ep132–133).
- reuse_rate 0.004 > v14g 0.000: obj_id_reuse binary feature is registering, barely.

**What went wrong:**
1. **Lower lr_d (2.5e-5 vs v14g 5e-5) made recall worse**, not better (0.228 vs 0.372).
   Hypothesis: slower critic is more stable but maintains a *permanent* moderate advantage.
   In v14g the faster critic was in more active competition with G, which forced G to improve.
2. **cross_cov_loss_weight=2.0 didn't improve DMD-GEN** (0.744 vs 0.700).
   The 2.5:1 ratio (cross_cov 2.0 : supervisor 5.0) still not enough to rebalance signal.
   Supervisor at 5.0 dominates G's loss landscape; G learns to reproduce sequences, not generate.
3. **Context-FID 5× worse than v14g**: latent space quality degraded. Likely because
   the 6-feature AE (v15+) has a harder compression task than v14g's 5-feature AE,
   and the pretrain wasn't extended to compensate.
4. **Reuse rate still near-zero (0.004)**: Binary obj_id_reuse feature isn't learned well
   from a continuous latent with per-step noise. Needs architectural support, not just features.

**Root causes for v17:**
1. `supervisor_loss_weight=5.0` too high: dominates G gradient → G learns replay, not generation.
   Fix: reduce to 1.0–2.0.
2. `lr_d=2.5e-5` too low: stable but asymmetric equilibrium. Fix: restore to `5e-5` (v14g level).
3. Locality learning requires more than a binary input feature. Architectural fix (task #19)
   needed eventually; for v17 at minimum try higher diversity_loss to compensate.

### v17 (vinge/GB10, completed ep200, 2026-03-31)

supervisor_loss_weight→1.0, lr_d→5e-5, diversity_loss→1.0, 200 epochs. Used v16 pretrain checkpoint.

**Full eval (ep190 best)**:
| Metric | v17 | v14g | v13 | Δ vs v14g |
|--------|-----|------|-----|-----------|
| MMD² | **0.00697** | 0.018 | 0.01335 | 2.6× better |
| α-precision | 0.826 | 0.910 | 0.812 | −0.084 |
| β-recall | **0.521** | 0.372 | 0.455 | +0.149 |
| DMD-GEN | 0.714 | 0.700 | 0.771 | similar |
| AutoCorr | 0.032 | — | 0.044 | better |
| Context-FID | **0.03** | 0.03 | 0.05 | same |
| reuse rate | 0.006 | 0.000 | 0.000 | still near-zero |

**What worked:**
- `supervisor_loss_weight=1.0` was the decisive change. G competed freely from epoch 1 (G<0 at ep1).
- Late LR-decay delivered sustained improvement: best kept improving through ep190/200.
- β-recall=0.521 breaks v13's 0.455 record. MMD²=0.00697 is new all-time low.
- Combined score 0.114 beats v14g (0.144) by 21%.

**Remaining problems:**
1. **DMD-GEN=0.714** (target <0.30): temporal dynamics unchanged across all versions.
   cross_cov_loss_weight=2.0 vs supervisor_loss_weight=1.0 gives 2:1 ratio, but DMD-GEN
   doesn't respond. The cross-covariance loss may not be targeting the right structure.
2. **reuse rate fake=0.006** (target >0.1): Binary obj_id_reuse feature is learned at ~1% of
   real rate. The latent space doesn't naturally encode temporal address correlations.
   Requires architectural change (task #19).
3. **α-precision=0.826** (target >0.85): Fidelity slightly below target; diversity
   pressure (diversity_loss=1.0) may be pushing G slightly off the real manifold.

**v18 direction**: Keep v17 hyperparameters. Address DMD-GEN with higher cross_cov (try 5.0)
and longer training (250 epochs). α-precision may improve naturally with more epochs.

---

### v18 (vinge/GB10, completed ep250, 2026-03-31)

Key change vs v17: `cross_cov_loss_weight` 2.0 → **5.0**, epochs 200 → **250**.

**Best (ep205, full eval with n_samples=2000)**:

| Metric | v18 | v17 | Target |
|--------|-----|-----|--------|
| MMD² | 0.01105 | **0.00697** | <0.005 |
| α-precision | **0.845** | 0.826 | >0.80 |
| β-recall | 0.418 | **0.521** | >0.70 |
| DMD-GEN | 0.760 | **0.714** | <0.30 |
| AutoCorr | 0.042 | — | <0.02 |
| Context-FID | 0.06 | **0.03** | <10 |
| reuse rate | 0.004 | 0.006 | ~0.827 |

**Did NOT beat v17.** v17 remains all-time best.

**What went wrong:**
1. **cross_cov=5.0 did not improve DMD-GEN**: DMD-GEN worsened from 0.714 → 0.760. Higher
   weight likely interfered with the generator during early training (similar mechanism to
   how supervisor_loss_weight=5.0 hurt v16). The cross-covariance loss matches the lag-1
   covariance structure but apparently the dominant DMD modes are driven by higher-order
   dynamics that L_cov cannot constrain at this weight.
2. **recall ceiling at ~0.47 (EMA) / 0.42 (full eval)**: v17 reached 0.521 full eval; v18 fell
   short. The higher cross_cov weight may have constrained generator diversity, counteracting
   diversity_loss=1.0. cross_cov and diversity_loss are in tension: one pushes G toward real
   temporal structure, the other pushes G away from similar outputs.
3. **EMA vs full eval gap**: EMA combined score during training (0.116) appeared close to
   v17's (0.114), but full eval combined = 0.128 vs v17's ~0.103. The fixed val_tensor
   (23,976 windows) is easier to cover than a fresh 2000-sample draw.
4. **W instability late training**: W spiked to 4.5+ at ep232, ep237, ep245 — highest seen
   in any run. Likely caused by the high cross_cov loss making G's gradient landscape rougher.

**v19 direction**: Revert cross_cov to 2.0 (v17 level). Add `--dmd-ckpt-weight 0.05`
(dynamics-aware checkpoint selection, implemented in 66f581b). Keep all v17 winners:
supervisor=1.0, lr_d=5e-5, diversity=1.0, epochs=250.
```bash
./scripts/vinge-launch.sh --version v19 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 250
```

---

### v19 (vinge/GB10, completed ep250, 2026-04-01)

Key changes vs v18: `cross_cov_loss_weight` 5.0 → **2.0** (v17 level), `--dmd-ckpt-weight 0.05`
(dynamics-aware checkpoint selection added), epochs 250.

**Best (ep225, full eval with n_samples=2000)**:

| Metric | v19 | v17 | v18 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01094 | **0.00697** | 0.01105 | <0.005 |
| α-precision | **0.835** | 0.826 | 0.845 | >0.80 |
| β-recall | **0.518** | 0.521 | 0.418 | >0.70 |
| DMD-GEN | **0.6875** | 0.714 | 0.760 | <0.30 |
| AutoCorr | 0.037 | 0.032 | 0.042 | <0.02 |
| Context-FID | 0.13 | **0.03** | 0.06 | <0.05 |
| reuse rate | 0.005 | 0.006 | 0.004 | ~0.757 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **Recall nearly matched v17**: β-recall=0.518 vs v17's 0.521 — within 0.3%. cross_cov=2.0
   + diversity=1.0 is the right balance; this confirms v18's recall regression was entirely
   caused by the higher cross_cov weight.
2. **DMD-GEN improved for the first time**: 0.6875 vs v17's 0.714, v18's 0.760. First
   monotonic improvement in temporal dynamics across any version. `--dmd-ckpt-weight 0.05`
   may have contributed by selecting ep225 (a slightly more dynamics-aware checkpoint).
3. **α-precision=0.835 above target (>0.80)**: Fidelity improved slightly vs v17.
4. **Late peaking confirmed**: Best at ep225/250; v17 best at ep190/200. Training near the
   end of the cosine schedule still yielding improvements.

**What went wrong:**
1. **MMD²=0.01094 vs v17's 0.00697**: 57% worse on the primary metric. This is now the
   central unsolved problem. recall and α-precision are at or above v17; MMD² is not.
   Root cause unclear — could be a stochastic init effect, or cross_cov=2.0 slightly
   spreading the generated distribution relative to v17's unconstrained run.
2. **Context-FID=0.13 vs v17's 0.03**: 4× worse. The latent space quality is degrading
   across versions. Likely caused by the locality split (6 features vs v17-era 5) adding
   reconstruction difficulty, or the cross_cov loss slightly warping the latent geometry.
3. **reuse rate fake=0.005** (real=0.757): Locality gap unchanged. Architectural fix needed
   (z_global conditioning, task #18/#9 in TODO.md).

**Root cause analysis — why MMD² is stuck above v17:**
- v17 achieved MMD²=0.00697 with cross_cov=2.0, diversity=1.0, supervisor=1.0
- v19 uses identical hyperparameters plus dmd_ckpt_weight=0.05
- The dmd_ckpt_weight adds 0.05×DMD-GEN to combined, which could select a slightly
  different checkpoint than pure MMD²+recall would. EP225 is best on combined but
  EP155 had lower EMA MMD²=0.00885 (vs EP225's EMA 0.00945). The dmd_ckpt_weight
  may be trading a small amount of MMD² quality for better dynamics.
- Alternatively, this is stochastic variance: v17's MMD²=0.00697 may reflect a
  favorable random init that v19 didn't replicate.

**v20 direction**: Same hyperparams, extend to 300 epochs. v19 peaked late (ep225/250);
more epochs may allow MMD² to converge further. Consider setting dmd_ckpt_weight=0.01
(reduce dynamics influence to favour MMD²-optimal checkpoint selection) as ablation.
```bash
./scripts/vinge-launch.sh --version v20 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 300
```

---

### v26 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v24: **`diversity_loss_weight` 1.0 → 2.0** to push recall past 0.52. v16 pretrain, dmd_ckpt_weight=0, 200 epochs.

**EMA best (ep190)**: MMD²=0.01255, recall=0.521, combined=0.108

**Best (ep190, full eval with n_samples=2000)**:

| Metric | **v26** | v17 | v24 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00795** | **0.00697** | 0.00798 | <0.005 |
| α-precision | 0.887 | 0.826 | 0.835 | >0.80 |
| β-recall | 0.401 | **0.521** | 0.503 | >0.70 |
| DMD-GEN | **0.710** | 0.714 | 0.717 | <0.30 |
| AutoCorr | 0.049 | **0.032** | 0.036 | <0.02 |
| Context-FID | 0.04 | **0.03** | 0.08 | <0.05 |
| reuse rate | 0.002 | 0.006 | 0.007 | ~0.853 |

Full eval combined: v26=0.128, v17=0.103, v24=0.107. **Did NOT beat v17 or v24.**

**What worked:**
1. **MMD²=0.00795** — tied with v24 (0.00798) as 2nd best ever on full eval. Three runs (v17/v24/v26) now below 0.01.
2. **α-precision=0.887** — 2nd best ever (v22: 0.927). diversity_loss=2.0 did NOT hurt fidelity.
3. **DMD-GEN=0.710** — best ever on full eval (v17: 0.714, v24: 0.717). First run below 0.714.
4. **Context-FID=0.04** — 2nd best ever (v17: 0.03). Much better than v24's 0.08.

**What went wrong — diversity_loss=2.0 inflated EMA recall without translating to full eval:**
1. **EMA recall=0.521 → full eval recall=0.401**: 23% drop. This is the worst EMA→full eval
   recall gap since v20. diversity_loss=2.0 pushes G to produce more diverse outputs on the
   fixed val_tensor, but these diverse outputs don't cover the full distribution well.
2. **β-recall=0.401** (v17: 0.521, v24: 0.503): Worst recall of any v16-pretrain run with
   dmd_ckpt_weight=0. diversity_loss=2.0 appears counterproductive for actual coverage.
3. **Full eval combined=0.128** — worse than both v17 (0.103) and v24 (0.107). The recall
   regression dominates the combined score despite improvements on every other metric.

**Root cause — diversity_loss mechanism mismatch:**
The MSGAN diversity loss penalizes G for producing similar outputs for different latent
inputs. At weight=2.0, this pushes G to spread outputs farther apart, which inflates recall
on the fixed val_tensor (fixed set of 23,976 windows is easier to cover with spread-out
samples). But the *directions* of spread are not aligned with the real data manifold — G
spreads into low-density regions that don't match any real traces. Full eval with fresh
2000-sample draws exposes this: coverage of the actual distribution is worse.

**Key finding**: diversity_loss_weight=1.0 is the sweet spot. Lower hurts diversity (v14f),
higher inflates EMA recall without improving full eval. Future diversity improvements need
architectural changes (z_global conditioning), not loss weight tuning.

**v28 direction**: TBD. The v16-pretrain + dmd_ckpt_weight=0 recipe has been thoroughly
explored (v24/v25/v26). v17 remains ATB after 9 subsequent runs. Time for structural changes.

---

### v30 (vinge/GB10, aborted ep54, 2026-04-01)

Key change vs v24: **`n_critic` 2 → 3** to give critic more power. Unconditional, v16 pretrain,
dmd_ckpt_weight=0, standard losses.

**EMA best (ep25)**: MMD²=0.01386, recall=0.370, combined=0.140

**Full eval (ep25 best)**:

| Metric | v30 | v17 | v24 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01640 | **0.00697** | 0.00798 | <0.005 |
| α-precision | **0.929** | 0.826 | 0.835 | >0.80 |
| β-recall | 0.468 | **0.521** | 0.503 | >0.70 |
| DMD-GEN | 0.751 | **0.714** | 0.717 | <0.30 |
| Context-FID | 0.17 | **0.03** | 0.08 | <0.05 |

Full eval combined: 0.016 + 0.2*(1-0.468) = 0.123. **Did NOT beat v17.**

Aborted at ep54 — recall falling (0.37→0.30), W trending to 2.8.

**Key findings:**

1. **α-precision=0.929 tied all-time best** (v22: 0.927). n_critic=3 produces extremely
   high-fidelity outputs — the stronger critic forces G to stay close to the real manifold.

2. **Full eval recall (0.468) > EMA recall (0.370)** — gap inverted for the first time.
   n_critic=3 makes EMA pessimistic: the extra critic steps create a harder adversary that
   the EMA-smoothed generator handles less well than the live weights during training.

3. **n_critic=3 is too aggressive** — the stronger critic suppresses diversity/recall in
   favour of fidelity. Recall peaked at 0.37 (EMA) and was falling by ep54. W trending
   upward (2.8) suggests impending cycling.

**Verdict**: n_critic=2 confirmed as the right balance. n_critic=3 trades recall for fidelity,
which is the wrong trade-off (recall is the binding constraint, not precision).

---

### v29 (vinge/GB10, crashed ep96, 2026-04-01)

**Recipe**: cond_dim=10 + full loss recipe (cross_cov=2.0, acf=0.2, fft=0.05, moment=0.1,
quantile=0.2). Used v28 pretrain (cond_dim=10 compatible). n_critic=2.

**EMA best (ep20)**: MMD²=0.00971, recall=0.674, combined=0.075

**Full eval (ep20 best, with real descriptors)**: MMD²=0.021, α-precision=0.882,
β-recall=0.384, DMD-GEN=0.702, Context-FID=0.08

| Metric | v29 (full eval) | v29 (EMA) | v17 | Target |
|--------|-----------------|-----------|-----|--------|
| MMD² | 0.021 | **0.00971** | **0.00697** | <0.005 |
| β-recall | 0.384 | **0.674** | **0.521** | >0.70 |
| combined | 0.144 | **0.075** | **0.103** | — |
| α-precision | **0.882** | — | 0.826 | >0.80 |
| DMD-GEN | **0.702** | — | 0.714 | <0.30 |

Full eval combined: 0.021 + 0.2*(1-0.384) = 0.144. **Did NOT beat v17.**

**GAN cycling crash**: W=2.5 at ep88 → 13.1 at ep95 → 17.3 at ep96 (abort triggered).
Auxiliary losses delayed cycling vs v28 (ep96 vs ep79) but did not prevent it.

**Key findings:**

1. **Conditioning produces phenomenal EMA metrics** — recall=0.674 and combined=0.075 are
   both all-time records by a wide margin. The conditioning mechanism genuinely helps G
   learn the data distribution faster and more completely.

2. **Massive EMA→full eval gap**: recall 0.674→0.384 (43% drop), MMD² 0.00971→0.021 (2.2×).
   The model overfits to conditioning descriptors — generates well for training-time
   descriptors but doesn't generalize to fresh eval descriptors.

3. **DMD-GEN=0.702 on full eval is best ever** (v17: 0.714). Conditioning may help temporal
   dynamics by giving G a workload identity signal.

4. **GAN cycling still happens WITH auxiliary losses** — just delayed vs v28 (ep96 vs ep79).
   The conditioning creates instability that the current training setup can't handle.
   Neither loss recipe stabilization (v29) nor loss stripping (v28) prevents it.

**Conditioning verdict**: Powerful but currently unusable. Two problems must be solved:
(a) GAN cycling (W explosion) — may need CFG dropout or lower lr_g;
(b) EMA→full eval gap (descriptor overfitting) — may need CFG or descriptor augmentation.

**v30 direction**: Step back from conditioning. Return to proven unconditional recipe (v16
pretrain, no conditioning, dmd_ckpt_weight=0) but try n_critic=3 (up from 2) to give the
critic more power. Quick experiment, no code changes needed.

---

### v28 (vinge/GB10, crashed ep79, 2026-04-02)

**Two major changes**: z_global workload conditioning (`--cond-dim 10`) + loss ablation
(stripped to WGAN + FM + supervisor + diversity only; zeroed FFT, moment, quantile, ACF,
cross_cov, locality). Fresh pretrain required (architecture change).

**EMA best (ep70)**: MMD²=0.01183, recall=0.508, combined=0.110

**Full eval (ep70, with random conditioning N(0,0.5))**: MMD²=0.026, α-precision=0.594,
β-recall=0.340, DMD-GEN=0.715, Context-FID=0.44

**Note**: Full eval numbers are unreliable — eval.py used random N(0,0.5) conditioning
instead of real workload descriptors. Fixed in eval.py post-v28 (now computes descriptors
from real val windows when cond_dim > 0).

| Metric | v28 (EMA) | v17 | v24 | Target |
|--------|-----------|-----|-----|--------|
| MMD² | **0.01183** | **0.00697** | 0.00798 | <0.005 |
| β-recall | **0.508** | 0.521 | 0.503 | >0.70 |
| combined | **0.110** | 0.114 | 0.107 | — |

**BREAKTHROUGH — conditioning is the path forward:**
1. **combined=0.110 at ep70** — best-ever early trajectory. v17 didn't reach 0.114 until ep190.
   At the same training stage, v28 was already outperforming every previous run. Conditioning
   gives the generator a massive head start by telling it *what kind* of workload to produce.
2. **Recall hit 0.526 at ep75** — briefly beat v17's all-time record of 0.521. First run to
   ever exceed v17 on recall (albeit EMA, not full eval).

**FAILURE — loss ablation caused GAN cycling crash:**
1. **W spiked to 2.67 at ep79** and training diverged. Without cross_cov, ACF, FFT, moment,
   and quantile losses, the generator had no auxiliary gradients to stabilize against critic
   overpowering. These losses act as implicit regularizers on G's output distribution.
2. The ablation was too aggressive — stripping ALL auxiliary losses at once made it impossible
   to isolate which ones matter. v29 restores the full proven loss recipe.

**Full eval was wrong (now fixed):**
eval.py used `torch.randn(n, cond_dim) * 0.5` as conditioning — random noise that doesn't
match the real workload descriptor distribution. This made α-precision=0.594 (G produces
"workloads" that don't exist) and inflated MMD². Fixed: eval.py now computes descriptors
from real val windows via `compute_window_descriptors()`.

**v29 direction**: Keep cond_dim=10 + restore full loss recipe (cross_cov=2.0, acf=0.2,
fft=0.05, moment=0.1, quantile=0.2). Use v28 pretrain (same architecture).

---

### v27 (vinge/GB10, AVATAR, aborted ep38, 2026-04-02)

**AVATAR architecture**: AAE (latent discriminator + distribution loss) + supervisor-assisted
reconstruction + BatchNorm GRU. Fresh pretrain. First major architectural departure from TimeGAN.

**Best (ep20)**: MMD²=0.055, recall=0.172, combined=0.221.

**FAILED.** Aborted at ep38 — critic overpowered generator from the start.

**What went wrong:**
1. **W=3.5–4.7 from epoch 1** — the unbounded latent space (no Sigmoid) gave the critic
   a much easier discrimination task. The critic immediately dominated.
2. **MMD² worsened over training**: 0.055 (ep20) → 0.074 (ep30) → 0.077 (ep35). Going backwards.
3. **G loss approaching zero** (-0.39 at ep36) — generator losing the adversarial game badly.
4. **G warmup supervisor loss diverged** (0.31 → 0.72 over 100 epochs) — the unbounded latent
   space may have made the supervisor's prediction task harder.

**Root cause**: Removing the Sigmoid bound from Encoder/Generator output created a latent space
that's too easy for the critic to exploit. The AAE latent discriminator was supposed to
regularize this, but the sequence-level critic (LSTM+attention) is much more powerful than
the per-timestep latent discriminator (MLP), creating an asymmetry.

**What to try next time**: Keep the Sigmoid-bounded [0,1] latent space but add the AAE
discriminator as an auxiliary loss (force q(z|x) toward a Beta distribution or truncated
Gaussian in [0,1]). This keeps the critic's job equally hard while adding latent regularization.

---

### v25 (vinge/GB10, aborted ep59, 2026-04-02)

Same recipe as v24: v16 pretrain, dmd_ckpt_weight=0, 200 epochs. Different seed.

**Best (ep50, full eval)**: MMD²=0.01272, α-precision=0.803, β-recall=0.359, DMD-GEN=0.758.

**Did NOT beat v17.** Weak seed — recall stuck at 0.18–0.28 for 50 epochs, W collapsed to
0.15 at ep55 (critic lost leverage). Aborted early. With v16 pretrain and dmd_ckpt_weight=0,
seed variance remains large: v24 got combined=0.107, v25 got ~0.169. Rolling dice isn't
enough — time to try structural changes.

---

### v24 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v22/v23: **`dmd_ckpt_weight=0`** (v22/v23 used 0.05). Same v16 pretrain, 200 epochs.
This exactly replicates v17's checkpoint selection logic (pure MMD²+recall, no dynamics penalty).

**Best (ep170, full eval with n_samples=2000)**:

| Metric | **v24** | v17 | v22 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00798** | **0.00697** | 0.01751 | <0.005 |
| α-precision | 0.835 | 0.826 | **0.927** | >0.80 |
| β-recall | 0.503 | **0.521** | 0.471 | >0.70 |
| DMD-GEN | 0.717 | **0.714** | 0.757 | <0.30 |
| AutoCorr | 0.036 | **0.032** | 0.052 | <0.02 |
| Context-FID | 0.08 | **0.03** | 0.09 | <0.05 |
| reuse rate | 0.007 | 0.006 | 0.002 | ~0.853 |

**v24 is the second-best run ever on full eval, narrowly behind v17.**
Full eval combined: v24=0.107, v17=0.103. EMA combined: v24=**0.097**, v17=0.114.

**What worked — `dmd_ckpt_weight=0` CONFIRMED as the fix:**
1. **MMD²=0.00798** — second-best ever on full eval (v17: 0.00697). First run since v17 to
   break below 0.01 on full eval.
2. **β-recall=0.503** — third-best ever (v17: 0.521, v19: 0.518). Recall hit 0.545 on EMA
   at ep170 — the highest EMA recall ever recorded.
3. **EMA combined=0.097** — beat v17's EMA combined of ~0.114 by 15%. The EMA→full eval gap
   was only 1.23× (0.0065→0.008 on MMD²), the smallest of any run.
4. **Sustained excellence**: ep95–ep195 had 20+ consecutive evals with combined < 0.12.
   No previous run sustained v17-level metrics for more than a few evals.

**What went wrong:**
1. **Full eval combined 0.107 vs v17's 0.103**: Missed by 4%. The remaining gap is on MMD²
   (0.008 vs 0.007) and recall (0.503 vs 0.521). Likely stochastic seed variance.
2. **Context-FID=0.08** (v17: 0.03): Latent space quality slightly worse. This has been a
   persistent issue since v18+ — may be related to the fresh preprocessor fit each run.
3. **α-precision=0.835** (v22: 0.927): Lower fidelity than v22, though above target.

**Key finding — `dmd_ckpt_weight` was actively harmful:**
v22/v23 used `dmd_ckpt_weight=0.05`, adding ~0.035 to the combined score (DMD-GEN≈0.7 × 0.05).
This doesn't just inflate scores — it changes which checkpoints get saved as best.pt. The
dynamics penalty causes the selector to prefer checkpoints with slightly better DMD-GEN but
worse MMD²/recall. Since DMD-GEN is stuck at ~0.7 regardless of checkpoint, the penalty
sacrifices the primary metrics for no gain.

Evidence: v22 (dmd_ckpt_weight=0.05) full eval combined=0.154; v24 (dmd_ckpt_weight=0) full
eval combined=0.107. Same pretrain, same hyperparams, only difference is checkpoint selection.

**Recommendation: never use `dmd_ckpt_weight > 0`** until DMD-GEN is responsive to training
(currently architectural limitation). Default in config.py should be 0.

**v25 direction**: Same recipe (v16 pretrain, dmd_ckpt_weight=0, 200 epochs). v24 nearly
matched v17 — another seed may beat it outright.

---

### v23 (vinge/GB10, aborted ep187, 2026-04-02)

Same recipe as v22: v16 pretrain, 200 epochs, identical hyperparams, different random seed.

**Best (ep165, full eval with n_samples=2000)**:

| Metric | v23 | v22 | v17 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01967 | 0.01751 | **0.00697** | <0.005 |
| α-precision | 0.766 | **0.927** | 0.826 | >0.80 |
| β-recall | 0.357 | 0.471 | **0.521** | >0.70 |
| DMD-GEN | **0.699** | 0.757 | 0.714 | <0.30 |
| Context-FID | 0.05 | 0.09 | **0.03** | <0.05 |
| reuse rate | 0.006 | 0.002 | 0.006 | ~0.850 |

**Did NOT beat v17.** Weaker than v22 on most metrics. Aborted at ep187 (not worth finishing).

**What worked:**
1. **DMD-GEN=0.699** — new all-time best, first run below 0.700. May be stochastic but
   suggests this seed found slightly better temporal dynamics despite weaker overall quality.
2. **Late surge at ep165**: recall jumped from 0.27→0.38 and MMD² dropped to 0.014.
   Confirms the late-LR-decay improvement pattern seen in v17/v22 is real, not version-specific.

**What went wrong:**
1. **Recall stuck at 0.15–0.25 for 130 epochs**: Unlike v22 (which had recall >0.30 by ep40),
   v23 was trapped in a low-diversity mode for most of training. The ep165 spike to 0.379
   was a brief escape, not sustained.
2. **α-precision=0.766** — worst since v16 (0.833). This seed produced lower-quality samples.

**Conclusion**: With v16 pretrain and identical hyperparams, random seed variance is large.
v17=0.114, v22=0.154, v23=0.169 on combined score. The pretrain is necessary but not
sufficient — good seeds matter.

**v24 direction**: v17 didn't have `dmd_ckpt_weight` (feature added post-v17). v22/v23 used
0.05. Try `dmd_ckpt_weight=0` to exactly replicate v17's checkpoint selection logic.

---

### v22 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v21: **v16 pretrain checkpoint** (same as v17 used), 200 epochs (v17's schedule).
Hyperparams identical to v17: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0,
dmd_ckpt_weight=0.05.

**Best (ep190, full eval with n_samples=2000)**:

| Metric | v22 | v17 | v21 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01751 | **0.00697** | 0.01485 | <0.005 |
| α-precision | **0.927** | 0.826 | 0.873 | >0.80 |
| β-recall | 0.471 | **0.521** | 0.366 | >0.70 |
| DMD-GEN | 0.757 | **0.714** | 0.720 | <0.30 |
| AutoCorr | 0.052 | **0.032** | 0.069 | <0.02 |
| Context-FID | 0.09 | **0.03** | 0.03 | <0.05 |
| reuse rate | 0.002 | 0.006 | 0.007 | ~0.837 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked — v16 pretrain hypothesis CONFIRMED:**
1. **v22 is the best run since v17** on combined score (EMA 0.154 vs v17's 0.114). v18–v21
   all used later pretrains and stagnated; v22 with v16's pretrain showed v17-like trajectory
   with continuous improvement through ep190.
2. **α-precision=0.927** — new all-time best across all versions. First run to significantly
   exceed the 0.85 target. The v16 pretrain produces a cleaner latent space for G.
3. **β-recall=0.471** — second-best ever (v17: 0.521). Recall climbed steadily from 0.25 to
   0.45 over the full 200 epochs, peaking at ep190 (same epoch as v17).
4. **Late-training surge confirmed**: Best improved from ep105 (combined 0.166) to ep190
   (combined 0.154) — the same late-surge pattern as v17.

**What went wrong:**
1. **MMD²=0.01751** (v17: 0.00697): 2.5× worse. EMA was 0.01059 — a 1.65× EMA/full-eval
   gap persists even with the EMA-save fix. The remaining gap is likely from fixed val_tensor
   (23,976 windows) vs fresh samples (2000 from 3234 files). v17 also had this gap but its
   underlying EMA MMD² was better to start with.
2. **DMD-GEN=0.757** (v17: 0.714, v19: 0.688): Temporal dynamics slightly worse. DMD-GEN
   remains stubbornly above 0.70 across all versions — architectural fix needed.
3. **Context-FID=0.09** (v17: 0.03, v21: 0.03): Latent space quality worse than v17 despite
   the same pretrain. May be stochastic or related to the fresh preprocessor fit.
4. **W spike ep198** (W=6.6): Late instability, though best.pt at ep190 captured before this.

**Key finding — pretrain quality is the dominant variable:**
v22 used the identical v16 pretrain as v17 and produced a v17-like training trajectory. v18–v21
used pretrains from later runs (v17, v19, v20) and all stagnated early. The pretrain captures
the E/R/S/G weight initialization quality; later pretrains may carry artifacts from extended
GAN training that make the subsequent GAN phase harder to optimize.

This means: **always use the v16 pretrain for future runs** unless the architecture changes
(hidden_size, latent_dim, input features). The 35-minute pretrain cost is amortized across
all downstream runs.

**v23 direction:**
Same v16 pretrain, same hyperparams, 200 epochs — another roll of the dice with a different
random seed. If v17 and v22 both peaked at ep190 with the v16 pretrain but v17 got better
MMD², the remaining variance is stochastic. Multiple runs will tell us the distribution.

---

### v21 (vinge/GB10, early-stopped ep235, 2026-04-01)

Key change vs v20: **EMA-save fix** — `best.pt` now saves EMA weights as `G` key instead of
live weights. Same hyperparams: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0,
dmd_ckpt_weight=0.05, epochs=300, checkpoint_every=5. Pretrain from v20.

**Best (ep35, full eval with n_samples=2000)**:

| Metric | v21 | v17 | v20 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01485 | **0.00697** | 0.03215 | <0.005 |
| α-precision | **0.873** | 0.826 | 0.826 | >0.80 |
| β-recall | 0.366 | **0.521** | 0.407 | >0.70 |
| DMD-GEN | 0.720 | **0.714** | 0.719 | <0.30 |
| AutoCorr | 0.069 | **0.032** | 0.049 | <0.02 |
| Context-FID | **0.03** | 0.03 | 0.14 | <0.05 |
| reuse rate | 0.007 | 0.006 | 0.004 | ~0.685 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **EMA-save fix confirmed**: Full eval MMD² (0.01485) is *better* than training-time EMA
   MMD² (0.02163) — the gap is now inverted. v20's 4.3× problem is solved. This validates
   that EMA weights are genuinely better than live weights for evaluation.
2. **α-precision=0.873** — new all-time best across all versions (target >0.85 met).
3. **Context-FID=0.03** — matched v17, much better than v20's 0.14.

**What went wrong:**
1. **Stagnated at ep35**: Best combined score never improved after epoch 35. 40 consecutive
   evals without improvement triggered early stopping at ep235. v17 peaked at ep190 — v21
   never had the late-training improvement surge.
2. **β-recall=0.366** (v17: 0.521): 30% worse coverage. The model peaked early with decent
   precision but poor diversity, then failed to expand coverage as training continued.
3. **W instability late training**: W spiked to 10.7 at ep231, 7.4 at ep232 — the highest
   seen in any run. The critic was dominating by endgame. W was generally higher throughout
   v21 (2–5 range) vs v17 (~1–2 range).
4. **AutoCorr=0.069** (v17: 0.032): 2× worse temporal fidelity despite similar DMD-GEN.

**Root cause analysis — why v21 stagnated while v17 didn't:**
- v17 used v16's pretrain checkpoint. v21 used v20's pretrain checkpoint.
- v20 trained for 300 epochs with W climbing to 6+ by endgame. The pretrained E/R/S/G
  weights from v20 may carry residual critic-adaptation artifacts that make the GAN phase
  harder to optimize from the start.
- v17's LR trajectory covered the sweet spot (cosine from 1e-4 to 5e-6 over 200 epochs)
  at the right pace. v21's 300-epoch schedule decays more slowly — the LR was still high
  enough at ep35 to find a good point but too high thereafter to refine it.
- Stochastic variance: v17 may have hit a favorable random trajectory that v18–v21 haven't
  replicated. Four consecutive runs with identical hyperparams (v18–v21) have all failed to
  match v17.

**v22 direction:**
The pretrain checkpoint quality hypothesis is testable. Options:
1. Use v17's pretrain checkpoint (if available) instead of v21's
2. Fresh pretrain from scratch (costs ~35 min but eliminates carryover artifacts)
3. Reduce epochs to 200 (v17's schedule) to match the LR decay profile that worked
4. Same hyperparams, different random seed — if v17 is stochastic, keep rolling the dice

---

### v20 (vinge/GB10, completed ep300, 2026-04-02)

Key changes vs v19: extended to **300 epochs** (v19 peaked late at ep225/250).
Hyperparameters identical: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0, dmd_ckpt_weight=0.05.

**Best (ep210, full eval with n_samples=2000)**:

| Metric | v20 | v17 | v19 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.03215 | **0.00697** | 0.01094 | <0.005 |
| α-precision | 0.826 | 0.826 | 0.835 | >0.80 |
| β-recall | 0.407 | **0.521** | 0.518 | >0.70 |
| DMD-GEN | 0.719 | 0.714 | **0.688** | <0.30 |
| AutoCorr | 0.049 | 0.032 | 0.037 | <0.02 |
| Context-FID | 0.14 | **0.03** | 0.13 | <0.05 |
| reuse rate | 0.004 | 0.006 | 0.005 | ~0.806 |

**EMA (training-time) vs full eval divergence — the critical finding:**
| Metric | EMA ep210 | Full eval | Ratio |
|--------|-----------|-----------|-------|
| MMD² | 0.00755 | 0.03215 | 4.3× worse |
| β-recall | 0.549 | 0.407 | 25% lower |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **EMA metrics were best-ever**: EMA ep210 showed MMD²=0.00755 and recall=0.549, both
   individually better than v17 (0.00697, 0.521). Training-time trajectory was genuinely
   better than any prior run at this stage.
2. **Early trajectory**: best ep50 combined (0.158) and ep100 combined (0.154) were the best
   ever seen, confirming 300 epochs was the right direction.
3. **Recall hit 0.539 (ep195 EMA)** — new all-time EMA high.

**What went wrong — EMA/full-eval gap is now the primary problem:**
1. **4.3× MMD² divergence**: EMA 0.00755 → full eval 0.03215. This is the largest
   EMA/full-eval gap observed (v14f was noted at 3–4×; v20 matches/exceeds that).
2. **Root cause — checkpoint saves LIVE weights, not EMA weights**: The combined score used
   for checkpoint selection is computed from EMA model metrics. But `best.pt` saves the
   live (non-EMA) G weights at that epoch. The EMA weights are smoothed across hundreds
   of epochs; the live weights at ep210 reflect only recent gradient updates, which may
   be noisier. The late W instability (W=6.1 at ep271, 5.7 at ep295) suggests the live
   weights were being aggressively updated while EMA stayed smooth.
3. **Fixed val_tensor may be too easy**: Training eval draws from the same 23,976 windows
   every time. Full eval draws 2000 fresh windows from the full 3234-file corpus. The
   fixed val set may be unrepresentative of the full distribution.
4. **W instability late training (ep271–295)**: W repeatedly spiked above 4.0 in the final
   30 epochs. This indicates the critic was dominating, degrading G's live weights even
   while EMA remained smooth.

**v21 direction — fix the EMA/full-eval gap:**
Primary fix: **save EMA weights as `best.pt`**, not live weights. The checkpoint selection
already uses EMA metrics; the checkpoint should store the same weights that were evaluated.
This requires a small change to `train.py`: when saving `best.pt`, write `ema_G_state`
instead of `G.state_dict()`.

Secondary: Consider a fresh random val draw (rather than fixed val_tensor) for training-time
eval, to reduce the divergence between training eval and full eval.

Keep same hyperparams pending the EMA-save fix.
```bash
# After implementing EMA save fix in train.py:
./scripts/vinge-launch.sh --version v21 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 300
```

### v15 (vinge/GB10, killed ep143, 2026-03-30)

First CUDA run. Applied all v14g root-cause fixes (obj_id split, obj_size quantization).
AMP fp16 enabled. torch.compile attempted but Triton broken on GB10 (libcuda.so link issue).

**Best (ep100)**: MMD²=0.029, β-recall=0.294, combined=0.170

**What went wrong:**
1. **Missing diversity_loss** (was in v14g, dropped from v15): direct cause of recall gap.
2. **GAN cycling (n_critic=3 too aggressive)**: ep100 best → ep110 MMD²=0.111 spike → never recovered.
3. **cross_cov still 10:1 drowned**: DMD-GEN stayed at ~0.700.

**Bugs found and fixed during v15 (all committed):**
- `torch.quantile` fp16 crash: cast to `.float()` before quantile (e3ad770)
- `evaluate_metrics()` left G/R in `.eval()`: `try/finally` restore in mmd.py (01a797f)
- E/R/S left in `.eval()` from pretrain phases: explicit `.train()` at GAN loop start (ba356d5)
- Supervisor gradient leak in G step: `torch.no_grad()` around `S(H_fake)` (f5c63b7)
- EMA seeded from random init: seed from post-warmup G after Phase 2.5 (f5c63b7)
- GradScaler called n_critic+2 times per batch instead of once: now called once per optimizer step (638ee24)
- rsync temp files in trace dir: `not p.name.startswith(".")` filter in `_collect_files` (0ca6f49)

**v15 eval history:**
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|----------|
| 5 | 0.0652 | 0.198 | 0.226 |
| 55 | 0.0360 | 0.298 | 0.176 |
| 80 | 0.0341 | 0.296 | 0.175 |
| 100 | 0.0291 | 0.294 | 0.170 ★ |
| 110 | 0.1109 | 0.183 | 0.274 (cycling) |
| 140 | 0.0435 | 0.307 | 0.182 |

---

## Key Metrics and Targets

| Metric | v13 | v14g | v16 | v17 | v18 | v19 | v20 | v21 | v22 | v23 | v24 | **v31** | Target | Description |
|--------|-----|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|--------|-------------|
| MMD² | 0.01335 | 0.018 | 0.042 | **0.00697** | 0.01105 | 0.01094 | 0.03215 | 0.01485 | 0.01751 | 0.01967 | 0.00798 | **0.00769** | < 0.005 | Kernel distribution distance |
| β-recall | 0.455 | 0.372 | 0.228 | 0.521 | 0.418 | 0.518 | 0.407 | 0.366 | 0.471 | 0.357 | 0.503 | **0.596** | > 0.7 | Coverage: fraction of real modes covered |
| α-precision | 0.812 | 0.910 | 0.833 | 0.826 | 0.845 | 0.835 | 0.826 | 0.873 | 0.927 | 0.766 | 0.835 | **0.953** | > 0.85 | Fidelity: fraction of generated that is realistic |
| DMD-GEN | 0.771 | 0.700 | 0.744 | 0.714 | 0.760 | **0.688** | 0.719 | 0.720 | 0.757 | **0.699** | 0.717 | 0.723 | < 0.3 | Temporal dynamics divergence (0 = perfect) |
| Context-FID | 0.05 | 0.03 | 0.15 | **0.03** | 0.06 | 0.13 | 0.14 | **0.03** | 0.09 | 0.05 | 0.08 | **0.03** | < 0.05 | Fréchet in encoder latent space |
| reuse-rate | 0.000 | 0.000 | 0.004 | 0.006 | 0.004 | 0.005 | 0.004 | 0.007 | 0.002 | 0.006 | 0.007 | 0.005 | > 0.1 | Fraction of obj_id repeats (sequential access) |

Note: **v31 is the all-time best on full eval** (combined=0.089), dethroning v17 (0.103) after 14 subsequent runs. v17's full eval correctly used G_ema weights
(eval.py has `ckpt.get("G_ema", ckpt["G"])`) — its results were not affected by the EMA-save
bug. The EMA-save fix in v21 ensures consistency for tools that only read the `G` key, but
was NOT the reason v18–v21 failed to match v17. The real differentiator was the **v16 pretrain
checkpoint** — v22 using v16's pretrain (same as v17) restored v17-class performance, while
v18–v21 using later pretrains all stagnated. This is the key finding of the v18–v22 series.

---

## What We Have Learned

### Architecture
- **GRU for AE/supervisor, LSTM for G+C**: GRU is simpler + faster for compression/prediction;
  LSTM cell state captures long-range context needed for G and burst detection in C.
- **Split noise (z_global + z_local)**: z_global maps to LSTM h0/c0 (workload identity);
  z_local feeds per-step input (event noise). Without this, every window is independent.
- **Latent AE (TimeGAN/SeriesGAN)**: Direct feature-space GAN is unstable in a 6-feature
  correlated space spanning 10 decades. AE reduces to smooth 24-dim latent space.
- **Minibatch std in critic (StyleGAN2)**: Collapsed G produces identical rows; std→0.
  This channel gives the critic a diversity signal before collapse propagates.
- **sn_lstm=True**: SN on LSTM weight matrices essential (not just FC output).
  FC-only SN allows LSTM to drift; W grows unboundedly and training collapses (v14).

### Training stability
- **GAN cycling / underdamped oscillation**: When critic pressure is too high (n_critic=3,
  high lr_d), G overshoots → critic wins decisively → G overshoots back. Seen in v13 (171
  wasted epochs) and v15. Damp with: lower n_critic, lower lr_d, cosine LR decay.
- **Hot-start asymmetry**: Never hot-start from a checkpoint with different Lipschitz constraints.
  Fresh SN u/v buffers + generator tuned for old critic → critic overpowers immediately.
- **Phase train/eval mode**: Each pretrain phase explicitly sets eval on modules it doesn't train.
  Must restore `.train()` on ALL modules at GAN loop start, or cuDNN RNN backward crashes.
- **GradScaler with n_critic > 1**: `scaler.unscale_(optimizer)` can only be called once
  per optimizer between `scaler.update()` calls. With n_critic > 1 and grad clipping,
  call `scaler.update()` after each `scaler.step(opt_C)` within the critic loop.
- **EMA**: Seed ema_G_state from post-warmup G weights (not random init). Random seed means
  first GAN evals use near-random EMA, skewing best.pt selection.

### Losses
- **L_FM (feature matching)**: Smoother gradient signal than Wasserstein alone; reduces collapse.
- **L_div (diversity, MSGAN)**: Directly combats β-recall mode collapse. Required in v14g
  (recall=0.372) vs v15 without it (recall=0.294). Weight 0.5 works; try 0.5–1.0.
- **L_cov (cross-covariance)**: Targets DMD-GEN directly. Needs weight ≥ 2.0 to overcome
  supervisor dominance at weight 5.0 (otherwise 10:1 ratio drowns cross_cov signal).
- **L_loc**: Stride-repetition rate — replaced by obj_id_reuse binary feature in v15+.
  Still active as auxiliary loss but its role is partly absorbed by the locality split.
- **L_ACF, L_FFT**: Effective for temporal structure; keep at current weights.
- **L_V, L_Q**: Cheap distributional anchors; always-on.

### Data representation
- **obj_id delta → locality split (v15)**: Raw delta-encoding makes reuse a zero-measure event.
  `obj_id_reuse` (±1 binary) + `obj_id_stride` (signed-log delta) gives G a learnable
  binary classification target for sequential access patterns.
- **obj_size quantization (v15)**: Snap to 4096-byte multiples before log-transform.
  Real block traces have discrete sizes; continuous representation wastes capacity.
- **Training-log recall is ~3–4× optimistic vs full eval** (EMA weights + 1000 samples
  vs live checkpoint + 2000 samples). Never abort based on training-log recall alone.

### Infrastructure notes
- **torch.compile**: Broken on vinge's GB10 (Triton can't find libcuda.so.1). Use `--no-compile`.
- **AMP fp16**: Works correctly. `torch.quantile` requires `.float()` cast before call.
- **pretrain_complete.pt**: State dict keys have no `_orig_mod.` prefix (saved without compile).
  Load into non-compiled models only (`--no-compile`). Avoids 35-min pretrain on restart.
- **SSH to vinge**: `ssh vinge.local` (RSA key, no passphrase). `-A` flag for git pull on GitHub.
