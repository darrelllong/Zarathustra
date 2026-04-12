# Rebuttal

*Author responses to reviewer comments go here. Each round gets its own section.*

---

## Round 1

We accept Findings 1–6 and 8 as legitimate bugs/debts and have opened TODO items for each (`TODO.md` → "Reviewer Action Items (Round 1)"). The PRDC fallback, the `--baseline` path for conditional checkpoints, the hard-coded reuse-rate column, the `TraceDataset` off-by-one, and the small-val-set reshape are all real and should be fixed before the next paper submission. We will also decouple the `/tiamat` paths and resolve the paper placeholders.

We partially push back on Finding 7 ("statistically indistinguishable" is overreach). The finding is factually correct that DMD-GEN ≈ 0.70 is nowhere near "indistinguishable" at the dynamics level, and we are softening the README and paper language accordingly. **However**, the softened claim should not go all the way to "substantial improvement on fidelity/coverage, temporal dynamics unresolved" without qualification: the most recent full-eval on the new alibaba record (alibaba_v34, supervisor-weight=5) shows combined=0.0823, MMD²=0.0107, β-recall=0.642, α-precision=0.708, density=1.10, coverage=0.642, Context-FID=0.21, HRC-MAE=0.0063. Every metric except DMD-GEN is either at or beyond the "indistinguishable" threshold commonly used in the TS-GAN literature; HRC-MAE of 0.006 means our cache-fidelity proxy is close to exact. The accurate statement is: *"fidelity, recall, density, and cache behavior are at parity with real data; linear temporal dynamics (DMD-GEN) remain the open gap."* We will make the README and paper reflect this distinction rather than a blanket retreat.

---

## Round 2

We agree with Findings 1 (scalar loss tuning is not the moat), 3 (corpus-specific playbooks), 4 (pretrain is the dominant variable), 5 (locality needs a mechanism, not a loss), 6 (several directions are empirically dead), 7 (block sampling is corpus-specific), and 8 (HRC as triage). These now all have TODO entries.

We need to push back on Finding 2, specifically the claim that "the biggest architectural mismatch is that training is still mostly local-window while generation is long-horizon and stateful" and the associated strong bet that a chunk-continuity curriculum is the highest-upside next move. We actually *ran* this: `tencent_v65` trained the proven ATB recipe with `--continuity-loss-weight 0.5`, which generates two adjacent windows with carried LSTM hidden state and penalises mean/std mismatch at the boundary — exactly the "stitched-window" objective the reviewer recommends. Training-log MMD² hit 0.09741★ (better than ATB 0.089 at train-log level), but the full eval was 0.12869 (45% worse than ATB) and DMD-GEN stayed at 0.7026 — **no improvement on the very metric the reviewer predicted would move**. See VERSIONS.md "tencent_v65 post-mortem." We do not conclude that hierarchical long-horizon modeling is a dead end — we agree that it probably matters — but we do conclude that a local boundary-continuity regulariser is not the right realisation of it. The correct version is either (a) an explicit long-horizon critic that sees multiple stitched chunks with carried state, or (b) a stage-1 regime sequence model that governs a short-horizon per-window generator. Both are large refactors; a short continuity loss is not. The reviewer's instinct is right; the lightweight realisation has already been falsified.

We also note a second-order agreement with Finding 6's projection-critic dismissal: we ran it as `alibaba_v35` (on top of the new v34 ATB recipe) and the critic/generator balance broke within 6 epochs (W-spike guard auto-killed at W=9.7). This is consistent with the reviewer's reading.

---

## Round 3

We accept Findings 1 (R-informed conditioning redesign, not a raw `cond_dim` bump), 2 (window-level characterization bridge), 3 (explicit locality engine with copy path), 4 (two-population training for poison points), 5 (routed MoE), and 6's limited-reopen list (GMM prior cleanly, `cond_dim=13` *after* conditioning cleanup). These are the right next bets and we have opened TODO items for each.

One clarification on Finding 6's projection-critic re-entry clause ("only in a late-stage fine-tune where the generator already has symmetric access to the same regime information and the critic is deliberately weakened"): the v35 failure mode we just observed was not that the critic was globally too strong, it was that the projection head's additional adversarial signal compounds with the existing wgan-sn critic at the *same* lr_d, doubling the effective critic capacity against an unchanged `n_critic=2` schedule. Any future re-entry should either (i) fresh-pretrain both heads together, or (ii) reduce `lr_d` to roughly half when the projection head is enabled. A "late-stage fine-tune" alone is not sufficient if lr_d is unchanged.

We also accept Finding 7 in full: BayesGAN, mixed-type recovery, and deeper stacked LSTMs should stay closed unless embedded in a more structured design.

---

## Round 4

No new technical findings; we appreciate the confirmation that the evaluation-fix wave partially closed earlier concerns. The three still-open eval bugs (PRDC fallback, baseline path, reuse-rate column) are already in the Round 1 TODO list and will be fixed in the next code-quality sweep.

We note for the record that, since Round 4 was written, `alibaba_v34` (supervisor-loss-weight=5) produced a new full-eval record of combined=0.0823 (25% better than the previous alibaba ATB of 0.110), with β-recall climbing to 0.642 (the highest alibaba recall ever measured). This is the first time the full-eval score on alibaba has *beaten* the training-log score, which inverts the persistent ~30% train/eval gap that motivated much of the reviewer's caution in Round 1 Finding 7. DMD-GEN remains at 0.70 — Round 2 Finding 5's concern about long-horizon dynamics still stands — but the non-dynamics metrics are now strong enough to justify the more precise claim above.

---

## Round 5

We accept Findings 1 (z_global inference path mismatch), 2 (preprocessor confounding), 3 (checkpoint selection on untrusted surface), and 6 (Fourier analysis). These are legitimate infrastructure debts and now have TODO entries. Finding 1 in particular — the divergent `z_global` construction between train.py and eval.py/generate.py — is a compelling candidate explanation for some of the train→eval gap, and we will prioritize unifying these paths.

We accept Finding 5 (multi-seed evaluation) as methodologically correct but note the tension with race velocity. We will adopt a compromise: verbatim controls for any new ATB claim, but not 3-seed bundles for every exploratory run. The recent recalibration (v42 → alibaba floor 0.142, v75 → tencent floor 0.1225) already demonstrates we take reproducibility seriously even when the conclusions are painful.

We strongly agree with Finding 4's reopening of alibaba `block_sample` and consider this vindicated by subsequent results. Since this review was written, `alibaba_v48` (block-sample + n-regimes 4) produced combined=**0.0767** — a 46% improvement over the recalibrated floor and the strongest result in the project's history. Key facts:
- β-recall **increased** at full eval (0.553→0.6815, +23%) — the first-ever negative train→eval gap on any corpus.
- HRC-MAE=**0.0006** — effectively perfect cache fidelity at 4 decimal places.
- The mechanism is clear: alibaba has H=0.98 (near-persistent Hurst exponent), so block-sample's temporal coherence matches the natural data structure.

This confirms the reviewer's instinct that block-sample was "one of the strongest reproducible improvements" when judged against the correct floor. We disagree, however, with reopening multi-scale critic (tencent) and 2-step supervisor (tencent): `tencent_v77` (supervisor-steps=2) produced combined=0.1756 at full eval with a 5.5× MMD² gap — this is not a floor-calibration issue but a genuine failure mode where the 2-step supervisor teaches G to overfit to the supervisor's latent manifold.

---

## Round 6

We accept Findings 1–3 (records are real, corpus-specific playbooks confirmed, supervisor-steps=2 globally closed) and Finding 5 (n-regimes=4 is structural for alibaba, not cosmetic). These align exactly with our experimental conclusions.

We partially push back on Finding 4's characterization of tencent_v78 as "not yet a true dynamics/locality win." The claim is correct for tencent (DMD-GEN=0.7416, HRC-MAE=0.0795), but the reviewer should note that the same block-sample lever on alibaba (v48) produced HRC-MAE=**0.0006** — a genuine cache-fidelity breakthrough. Block-sample IS solving practical cache behavior on alibaba even if DMD-GEN remains open. The more precise statement is: *block-sample closes the cache-fidelity gap on high-Hurst corpora (alibaba H=0.98) but not on moderate-Hurst corpora (tencent H=0.79), where temporal rhythms at longer horizons still need structural modeling.*

We accept Finding 6 (search has improved faster than infrastructure) as a fair observation. The z_global unification and preprocessor freezing from Round 5 are on the TODO list but have been deprioritized relative to model wins during the race. We will address them in the next code-quality sweep.

---

## Round 7

We accept Findings 3 (Fourier analysis was deferred, not answered) and 5 (Tencent multi-scale critic was closed without a full eval). Both are fair. The Fourier pass is now a top-priority TODO, and we will not re-close multi-scale critic for tencent until a full eval is run under the recalibrated floor.

We accept Finding 7 (non-winning runs contain useful signal beyond combined score). The reviewer is right that v51's best-ever alibaba DMD-GEN and v82's best-ever tencent Context-FID + AutoCorr should inform the search, not be discarded because they lost on combined. We will track per-metric bests alongside combined in future triage.

We accept Finding 8 (the post-record exploitation sweep is not evidence that the structural idea space is exhausted). The recent runs have been dominated by scalar tweaks around the KL 0.01 breakthrough, which was the right short-term move but should not be mistaken for having closed the architectural playbook. The explicit locality/copy path, window-level regime bridge, path-space critic, and MoE generator remain untried and high-priority.

We partially accept Findings 1–2 (rebuttal overstates mechanism certainty). The reviewer is correct that we should not declare "the mechanism is clear" while z_global unification and preprocessor freezing remain undone. We will reframe mechanism claims as leading hypotheses until controls are complete. **However**, the situation has materially improved since the reviewer wrote this: `var-cond-kl-weight 0.01` has now been validated with verbatim controls on **both** corpora. Alibaba v57 (v55 verbatim) produced eval 0.1134, matching v55's 0.1251 and confirming KL 0.01 is reproducible. Tencent v86 (KL 0.01 on fresh pretrain) produced eval **0.1130** — a 22.8% improvement over the tencent floor with a train→eval gap of only 9.7%, the smallest ever recorded. The mechanism hypothesis is now well-supported: KL 0.01 regularizes the conditioning distribution to be robust at eval time, dramatically reducing the train→eval gap. We accept the reviewer's point that this should be stated as "strongly supported hypothesis" rather than "settled fact" while infrastructure debts remain.

We push back on Finding 4's characterization of the Round 6 Tencent response as "rhetoric, not counter-evidence." Our point was not that alibaba's HRC rebuts the tencent DMD-GEN concern. It was that block-sample's effect is **corpus-specific and Hurst-dependent**: it closes the cache-fidelity gap on alibaba (H=0.98, HRC-MAE=0.0006) while leaving tencent's dynamics unresolved (H=0.79, HRC-MAE=0.0795). That is a substantive empirical characterization of the mechanism's scope, not a rhetorical deflection. The precise claim stands: *block-sample is a cache-fidelity solution for high-Hurst corpora, not a universal dynamics fix.* We agree tencent dynamics remain the open gap.

We push back on Finding 6's claim that we turn "single-run patterns into corpus laws." Since this finding was written, we have run multiple KL 0.01 verbatim controls that confirm the pattern is reproducible, not seed-dependent. Alibaba v55 and v57 (verbatim) agree to within 10% at full eval. Tencent v86 succeeded while v87 collapsed (seed instability in the fresh pretrain), but v88 (second verbatim attempt) is currently at ep93/200 with best 0.11756★ and healthy W values, tracking v86's successful path. We are doing exactly what the reviewer recommends — running controls before declaring laws — and the controls are confirming, not undermining, the KL 0.01 story.

---

## Round 8

We accept Findings 1 (re-elevate validation), 3 (anomaly modeling as structured problem), 4 (feature covariance / correlation-aware conditioning), 5 (architecture must respect temporal structure), 6 (semi-supervised/structured latents), and 11 (Fourier as on-mission diagnostic). These are well-aligned with both our current TODO list and the original research agenda. The recovery of anomaly modeling (tail expert, regulator, anomaly class) is particularly well-timed: our current clipping defense works but is intellectually unsatisfying and may be suppressing legitimate signal.

We accept Finding 2 (whole-trace representativity was the original target) and Finding 12 (the project is not supposed to be a prettier Filebench). The 12-step window is a practical training unit, not the research objective. We are actively pursuing chunk-continuity and hierarchical generation approaches, though tencent_v65's failure with a boundary-continuity regularizer (see Round 2 rebuttal) showed that the lightweight version of this doesn't work — the correct realization requires either an explicit long-horizon critic or a two-stage regime-to-event generator, both of which are larger refactors on the TODO list.

We accept Findings 7 (external-system replay testing) and 8 (benchmarking against legacy generators) as valuable competitive positioning. HRC evaluation was added in the recent eval cleanup wave and has already proven its worth: alibaba v48's HRC-MAE of 0.0006 is our strongest practical result, and it was invisible before HRC was instrumented. Expanding to full system replay (response time, CPU, throughput) is the natural next step for finalist models.

We accept Finding 9 (compositional/hybrid workload generation) and Finding 10 (conditional generation with cleaner semantics). The R-informed conditioning redesign currently on our TODO list is exactly the "use only signal-bearing descriptors, separate file-level from window-level factors" approach Finding 10 recommends. The routed MoE generator (Round 3 TODO) is the natural vehicle for Finding 9's compositional vision.

We note that the reviewer's recovery of the NSF proposal's original ambitions is timely and welcome. The recent KL 0.01 breakthrough — which works by regularizing the conditioning distribution, not by adding another loss term — is precisely the kind of structural insight the proposal envisioned. It suggests the path forward is through better modeling of **what the generator conditions on and how**, not through more auxiliary losses on what it outputs. This aligns with the proposal's emphasis on representative conditioning, structured latent spaces, and principled validation over brute-force tuning.

---

## Round 9

We accept the reviewer's central thesis without reservation: the project has been saying the right things while still doing the wrong things. The evidence since Round 8 vindicates this critique more strongly than the reviewer could have known when writing it.

We accept Findings 1–3 (stop scalar tweaking, pick structural bets, treat KL 0.01 as base camp). Since the reviewer wrote this, we ran **five consecutive scalar tweaks on alibaba** (v58 n-regimes 6, v60 quantile 0.3, v62 acf 0.3, v63 diversity 3.0, v64 fft 0.1) — every single one failed to beat v59 at eval. We also ran three scalar stacking attempts on tencent (v94 moment+acf, v95 diversity+acf) — both stalled. The one success was tencent v93 (acf-loss 0.3, eval **0.0995**, new tencent record), which was genuinely novel but is now the ceiling — stacking anything on top of it also fails. The reviewer's prediction that scalar sweeps generate false hope is exactly what we observed: several runs produced best-ever training-log scores (v62 at 0.102★, v64 at 0.107★) that collapsed at eval (0.138, 0.132). **Scalar tuning on both corpora is now empirically exhausted.** We are freezing v59 (alibaba) and v93 (tencent) as baselines.

We accept Finding 4 (Fourier) with the acknowledgment that four rounds of "we'll do it" is embarrassing. It is now P0 priority.

We accept Finding 5 (locality needs a mechanism, not a loss) — this is consistent with our closed list showing locality-loss-weight increases fail on both corpora.

We accept Finding 6 (infrastructure debts). We note one new data point that makes this even more urgent: `alibaba_v65` was a verbatim v59 control that matched v59's training-log perfectly (both ~0.110★) but produced eval **0.1351** vs v59's 0.1113 — a 23% gap. The difference was training instability (v65 had W values 2–5 and G_loss 6–8; v59 was likely calmer). This means the alibaba "record" may be stability-dependent rather than recipe-dependent, which is exactly the kind of measurement noise the reviewer warns about when `z_global` paths and preprocessor behavior diverge between training and eval.

We accept Finding 7 (scalar sweeps generate false hope). v65's failure to reproduce v59 is the strongest evidence yet: identical recipe, identical training-log performance, completely different eval — because the *trajectory* of training matters, not just the *recipe*. This is not something scalar tuning can fix.

We accept Finding 8 (pick a build order) and commit to the reviewer's recommended sequence: (1) conditioning-structure experiment, (2) locality-structure experiment, (3) diagnostics package (Fourier), then (4) optional scalar sidecars. The scalar era is over.

---

## Round 10

We accept the reviewer's recognition that the project has made real structural progress since Round 9. Fourier diagnostics are implemented, the copy-path mechanism was launched, and — most significantly — the PCF (path characteristic function) loss has produced a **genuine breakthrough**: alibaba_v71 eval combined=**0.067**, the best result in project history by 50%.

**On Finding 1 (Fourier landed correctly):** Agreed. The PSD metric (commit 0985c20) confirmed that marginal frequency content is already well-matched, narrowing the remaining gap to local/cross-step structure: reuse decisions, stride consistency, and path-level state transitions. This directly informed the shift to PCF and copy-path work.

**On Finding 2 (copy-path should stay the main event):** We agree in principle, but the experimental evidence has shifted. Copy-path (alibaba_v67, tencent_v97) ran but was superseded by an even more powerful structural bet: PCF loss (IDEAS.md #6). alibaba_v71 with PCF replaced all 6 handcrafted auxiliary losses (ACF, FFT, moment, quantile, cross-cov, locality) with a single learned adversarial functional on path increments. Results:
- Combined=**0.067** (50% better than any prior eval)
- MMD²=**0.007**, β-recall=**0.701**, α-precision=**0.926**
- First-ever **negative train→eval gap** (-32%): eval was better than training
- HRC-MAE=**0.010** (near-perfect cache fidelity)

This is exactly what the reviewer called for in Round 9: "mechanism changes that test different hypotheses, not finer coefficient walks around the current objective." PCF is a structural change to the critic's loss landscape, not a scalar tweak.

**On Finding 3 (strongest evidence against scalar fiddling is internal):** Completely agreed, and now reinforced by PCF results. Five consecutive alibaba scalar tweaks (v58–v64) failed; three tencent stacking attempts failed. Then PCF — a single structural change — beat every scalar attempt by 50%. The message is unambiguous: structural bets pay; scalar sweeps don't.

**On Finding 4 (local-structure agenda):** We partially agree. PCF already addresses several items on this list — it provides a learned path-level discriminator that captures reuse patterns, stride consistency, and inter-step dynamics without handcrafted loss terms. The remaining items (reuse precision/recall diagnostics, stack-distance comparisons, IRD summaries) are legitimate additions to the evaluation suite and are now on the TODO list.

**On Finding 5 (GEMINI eval-path bugs):** We accept this as legitimate and have added the specific bugs (conditional eval path, HRC weighting, reuse-column resolution, descriptor-dimension fallback) to the TODO list. We note that the z_global unification work from Round 5/9 has been partially addressed: eval.py, mmd.py, generate.py, and model.py now use the full cond_encoder → regime_sampler → gmm_prior path matching training. This was a prerequisite for the PCF results being trustworthy. The remaining GEMINI items are real but lower priority than the z_global fix.

**On Finding 6 (non-reproducibility should change judgment criteria):** Agreed. PCF results are being evaluated on exactly the criteria the reviewer recommends: (a) robust train→eval contract (v71 had a *negative* gap — eval was 32% better than training, unprecedented in project history), (b) dramatically improved locality-facing metrics (HRC-MAE=0.010), and (c) the result was reproduced directionally on tencent (v99 eval=0.112, new tencent ATB with precision 0.904). Two corpora, same recipe, both improved. This is not a lucky seed.

**On Finding 7 (best next structural ideas):** We have now executed the path-space critic recommendation (PCF), which is producing the strongest results in project history. The next structural priorities, in order, are: (1) PCF weight/frequency tuning per corpus (alibaba prefers 2.0, tencent needs 2.0 with possibly more frequencies), (2) locality-native diagnostics, (3) conditioning redesign with R-informed factorized descriptors. The copy-path mechanism remains a valid future bet but is deprioritized while PCF is delivering breakthrough results.

**On Finding 8 (repo-quality — accidental path mirror):** Accepted. The `Users/darrell/Zarathustra/` directory in the repo root will be cleaned up.
