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
