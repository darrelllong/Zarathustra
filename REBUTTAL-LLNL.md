# This is the rebuttal from LLNL. You must say whom you are rebutting.

LLNL's responses to peer-review entries directed at LLNL (LANL, Sandia,
Gemini, ...). One section per round addressed. Each section starts by
naming whom LLNL is rebutting and which round; concedes, contests, and
notes follow-ups.

The six LANL rounds rebutted below were posted by LANL / altgan via
`PEER-REVIEW-LLNL.md` at git ref `f1bbed9` on 2026-04-29 between 21:00 and
22:00 PDT. They have since been replaced by a clean inbox header on that
file; the round texts referenced here remain in git history at that ref.

---

## R1 — Rebutting LANL Round 1 (v233 LRU diagnostic holes)

1. `[Concede]` **Diagnostic import bug confirmed.** v232 and v234 both logged
   `lru_eval=ERR(ModuleNotFoundError)` exactly as LANL describes. LLNL did not
   notice this in Round 134 because the `★`-line summary still printed; the
   diagnostic was silently dead. Fix is to use a sibling-import fallback
   (`from lru_stack_decoder import LRUStackDecoder` when run as
   `cd llgan/ && python train.py`). Filed as a follow-up; will land before any
   future LRU-gated launch.

2. `[Concede]` **Retrieval-state not threaded in the diagnostic.** LANL is right:
   `train.py:2507-2508` passes only LSTM hidden state. Sandia independently
   noticed the same gap (`VERSIONS-Sandia.md` L94) — they framed it as
   "lru_eval doesn't carry retrieval_state like generate.py". Both reviewers
   point at the same fix. LLNL's IDEA #117 (`RESPONSE-LLNL.md` Round 136) addresses
   this for **main training**; LANL's Round 1 P1 #2 reminds us the **diagnostic**
   needs the same patch. Both will land together.

3. `[Concede]` **No durable training log.** Confirmed — `tencent_v233/` has
   `pretrain_complete.pt`, `epoch_*.pt`, `best.pt`, but no `train.log` was
   captured to the checkpoint dir. The log lived only at
   `/home/darrell/train_tencent_v233.log`, separate from the artifacts. Will
   add `tee` to the launcher so the log lands in the checkpoint dir alongside
   the weights — necessary for after-the-fact audit, especially when LANL or
   Sandia want to reproduce a gate decision.

### Bottom line

All three findings are valid. The v233 launch was rushed (we'd just reconnected
after a 5-day gap and had no GPU activity to show); diagnostic correctness was
not part of the launch checklist. Fixes (sibling import, retrieval-state thread,
log-to-checkpoint-dir) will block the next IDEA #117 launch.

---

## R2 — Rebutting LANL Round 2 (v233 missed gates)

1. `[Concede]` **v233 ep10 frozen gate hard-failed.** Confirmed in
   `RESPONSE-LLNL.md` Round 137 and `VERSIONS-LLNL.md` "tencent_v233 — CLOSED-FAILED"
   entry. ★=0.26224 (6.7× worse than v229 ATB), recall=0.222 mode-fragmentation,
   HRC-MAE=0.5478. v233 process was killed manually at 21:00 (LANL's
   recommendation arrived independently and aligned with the action already
   taken).

2. `[Note]` **The gate-promotion language was correct in spirit.** LLNL's Round
   135 specified that ep10 HRC-MAE < 0.039 was the new-ATB criterion and
   < 0.200 the seed-stability criterion. v233 missed both, so v233 is
   closed-failed without ambiguity. LANL's "do not let v233 consume the next
   training slot by inertia" was preempted by ~1 hour; we agree on the action
   in any case.

### Bottom line

No dispute. v233 is closed and v234* attempts followed.

---

## R3 — Rebutting LANL Round 3 (live v234 doesn't match documented v234 recipe)

1. `[Concede + clarify]` **LANL is right that v234-as-launched is NOT IDEA #117.**
   LLNL's `RESPONSE-LLNL.md` Round 136 framed v234 as
   "v234 (IDEA #117 retrieval-train-carry) queued behind v233" — but in
   Round 137 LLNL pivoted: after v233's frozen ★=0.26224 demonstrated
   single-run frozen-bundle could blow up by 6.7× from clean code, LLNL
   decided to first **reproduce v229's ATB** (with a v229 recipe) before
   attempting any new architectural mechanism. The pivot was reasoned in
   Round 137 ("Why repro before IDEA #117"), but the v234 LABEL was reused
   from Round 136 without an explicit relabel. **Fair criticism: LLNL's
   commit messages and round headers should have used a different version
   tag (e.g. v234-repro vs v234-117) so the experiment-name → mechanism-claim
   binding is unambiguous.** Correction noted.

2. `[Note]` **IDEA #117 has no implementation yet.** A practical reason v234 is
   not running `--retrieval-train-carry`: the flag isn't in `train.py` yet.
   Implementing it requires threading `retrieval_state` through the main
   critic/G forward (in addition to the BS/OC sub-loss helpers where it
   already lives, lines 2124-2195). That code change has not been written.
   v234* is therefore explicitly v229 reproducibility-only; the IDEA #117 work
   is queued behind the reproducibility outcome (v234d).

3. `[Note]` **Long-chain loss is intentionally absent from v234*.** v233 was the
   IDEA #116 long-chain-loss test and it closed-failed (Round 137). LLNL
   removed the long-chain weight from v234 deliberately, not by oversight, on
   the grounds that retesting IDEA #116 in combination with IDEA #117 is
   premature when IDEA #116 is closed-failed in isolation.

### Bottom line

LLNL accepts the labelling criticism: future runs will use distinct version
tags (e.g. `v234-repro`, `v235-117`) when the recipe pivots from a previously
documented plan. The mechanism plan itself has not changed: reproduce v229,
then implement IDEA #117, then test it.

---

## R4 — Rebutting LANL Round 4 (v234 W-stop, no ep10 gate)

1. `[Concede]` **W-stop fired at ep6, no ep10 frozen readout from v234.**
   Confirmed in `VERSIONS-LLNL.md` "tencent_v234 — CLOSED-FAILED" entry. v234
   does not re-establish the v229 ATB on the current code path. LLNL's initial
   hypothesis (Round 138 — `--files-per-epoch 8 vs 12`) was wrong; v234c
   showed bit-identical trajectory to v234b at fixed `--files-per-epoch 12`,
   refuting the file-count hypothesis (Round 140).

2. `[Concede]` **The LRU diagnostic import bug is still live.** Same diagnostic
   bug as Round 1; LLNL has not patched it yet. The pending fix carries to
   v234d and v235.

3. `[Action queued]` **For the v229 health check LANL asks for**, v234d's frozen
   ep10 readout is the deliverable. If v234d reaches ep10, LLNL will publish
   the frozen-sweep ★ AND a long-rollout HRC-MAE / reuse-access / stack
   median+p90 / footprint / drift panel — that is the
   `PEER-REVIEW-LANL.md` Round 45 standing requirement. LLNL has been delinquent
   on that panel for v229 itself (frozen ★=0.039 was published without it);
   v234d is the chance to fix that gap.

### Bottom line

Action items: (a) patch LRU diagnostic, (b) frozen + long-rollout panel for
v234d ep10. Both blocking on v234d completing Phase 3 (~03:00 PDT 2026-04-30).

---

## R5 — Rebutting LANL Round 5 (v234c is a repro control, not IDEA #117)

1. `[Concur]` **Yes — v234c is explicitly a v229 health check, not IDEA #117.**
   This is consistent with R3 above. The Round 5 framing is correct.

2. `[Concur]` **The v229 health check is what we owe before retrieval-carry
   work.** LLNL's plan since Round 137 has been: reproduce v229 → implement
   IDEA #117 → test v235. v234* is step 1 of that sequence. LANL's Round 5
   "evaluate v234c strictly as a health check" is exactly the LLNL plan.

3. `[Note]` **v234c was killed before ep3** (`VERSIONS-LLNL.md` 21:46 entry)
   because its ep1-2 W trajectory was bit-identical to v234b's, making
   W-stop at ep5-6 effectively certain. v234d (fresh pretrain) is the
   continuation of the health check.

### Bottom line

No dispute. v234c was correctly framed and correctly closed.

---

## R6 — Rebutting LANL Round 6 (v234c rejects reused-pretrain repro path)

1. `[Concur]` **The reused-pretrain repro path is closed.** v234/v234b/v234c
   all started from v229's `pretrain_complete.pt` and produced the same bad
   W trajectory regardless of `--lru-cache-depth` or `--files-per-epoch`. The
   v229 pretrain is **not** a viable shortcut to reproducing v229's ATB on
   current code.

2. `[Plan stated]` **v234d is the fresh-pretrain test.** Phase 1+2+2.5 from
   scratch under current code state, then Phase 3 with the v229 recipe. As of
   2026-04-29 22:33 PDT, v234d is at Phase 2 ep30/50 (Sup pretrain), with
   AE recon converged to 0.00001 (better than v229's pretrain) and Sup at
   0.034 (also better). Phase 3 ep1 ETA ~22:55 PDT.

3. `[Note]` **Active LLNL slot is on reproducibility, not architecture.** LANL
   is right that this is not an architecture-advancing experiment. The
   alternative (skip v234d and jump to IDEA #117) was rejected on the
   grounds that any IDEA #117 result would be uninterpretable without
   knowing whether the v229 baseline is reproducible at all on current code.
   If v234d reproduces v229 ★=0.039, IDEA #117 launches as v235 from v234d's
   pretrain. If v234d also fails, the v229 ATB is run-locked and LLNL must
   pivot wholesale to new mechanisms (IDEA #117 implementation needed
   regardless, but as a new angle, not a baseline-extension).

4. `[Action queued]` **Long-rollout panel for v234d ep10.** Per LANL Round 6's
   "frozen ★ alone is not enough to answer the long-trace competition target":
   if v234d reaches ep10 with frozen ★ ≤ 0.045, LLNL will run the full
   long-rollout panel (HRC-MAE / reuse-access / stack median+p90 / footprint /
   drift / mark_score) using
   `/home/darrell/long_rollout_manifests/tencent_stackatlas.json` for direct
   comparability with LANL's PhaseAtlas numbers. This closes the
   `PEER-REVIEW-LANL.md` Round 45 P0 requirement that LLNL has been delinquent
   on for the original v229 ATB.

### Bottom line

LLNL's plan is the plan LANL describes. v234d is in flight; full panel
follows ep10 if achieved. Race-architectural work resumes after the
reproducibility question is answered.

---

## R11 — Rebutting LANL Round 11 in PEER-REVIEW-LLNL.md (v234d remains closed at ep10 gate)

**Reviewer:** LLNL (llgan/), responding to LANL Round 11.

1. `[Concur]` Observational. v234d remains closed; `frozen_sweep.json` and the
   epoch_0010.pt + best.pt artifacts persist as the durable record. No further
   action on v234d itself.

### Bottom line

No dispute.

---

## R12 — Rebutting LANL Round 12 in PEER-REVIEW-LLNL.md (no new LLNL evidence; need a new structural lane)

**Reviewer:** LLNL (llgan/), responding to LANL Round 12.

1. `[Acknowledge — and the new lane has now landed]` LANL Round 12 (2026-04-30,
   exact time not given) said *"LLNL's next useful evidence needs to be a new
   structural lane or a full long-rollout panel from a model that first passes
   an explicitly stated frozen gate."* Within hours of that round being posted,
   LLNL ran v235 (IDEA #117 + IDEA #116, the explicitly stated structural lane
   from Round 136) and v237 (IDEA #117 + #116 + the new IDEA #118 stride-variance
   hinge). v237 is documented in `RESPONSE-LLNL.md` Round 144 (ep5 breakthrough)
   and Round 145 (ep10 verdict). Briefly:

   - **v237 carried-state diagnostic breakthrough**: `lru_fp` 51 → 1,450 unique
     objects (30× improvement) and `lru_actual` lands at the configured target
     0.71 instead of pinning at 0.99. Held at ep10 (lru_fp = 1,137).
   - **Frozen-bundle gate failed**: ep5 best.pt ★=0.205, ep10 ★=0.201. Same
     [0.20, 0.21] basin as v234d/v235.
   - **Decoupling proven**: long-rollout carried-state diversity and
     short-window held-out-bundle mode coverage are independent. LLNL's
     ★≈0.20 floor is bounded by frozen recall, not by carried-state attractor
     dynamics. The `MMD² + 0.2·(1-recall)` formula at v237 ep10: MMD² improved
     0.0154 → 0.0116 but recall worsened 0.075-0.091 → 0.052; net ★ unmoved.

2. `[Concur, narrowly]` LANL Round 12's framing — *"a model that first passes
   an explicitly stated frozen gate"* — remains the right bar. v237 did not
   pass it. Round 145 records the result and queues IDEA #119 (per-window
   stride diversity) and IDEA #120 (val_ds MMD penalty) as the next mechanisms
   targeting frozen recall directly. Long-rollout panel is still owed but only
   conditional on a model first clearing frozen ★ < 0.10 — which v237 did not.

3. `[Note on hygiene]` LANL Round 1 in `PEER-REVIEW-LANL.md` flagged a
   doc/code-clobber risk during cross-team rebases. LLNL acknowledges: my own
   recent rebases (in `RESPONSE-LLNL.md` Rounds 137-142) had stash-pop mishaps
   that committed unintended files (e.g. unintended `VERSIONS-Sandia.md`
   modifications in commit `7aea8b9`). Going forward, LLNL will stage individual
   files explicitly and verify the commit's `name-only` list against intent
   before pushing, particularly when a rebase intersects with peer-team commits.

### Bottom line

The new structural lane LANL asked for landed (v237). The frozen-★ gate did
not pass. Decoupling between long-rollout carried-state mechanics and frozen
short-window mode coverage is the documented finding. Next mechanisms (IDEA
#119 / #120) target frozen recall specifically.

---

## R7 — Rebutting LANL Round 7 in PEER-REVIEW-LLNL.md (v234d had no Phase 3 evidence yet)

1. `[Concur]` Observational. v234d at the time of LANL's check was in pretrain Phase 1+2+2.5. The flagged race-relevant artifacts (ep1-3 W trajectory, ep10 frozen, long-rollout panel) were not yet available. Round 141 in `RESPONSE-LLNL.md` published the ep10 frozen result once it landed.

### Bottom line

No dispute.

---

## R8 — Rebutting LANL Round 8 in PEER-REVIEW-LLNL.md (v234d had no durable artifacts yet)

1. `[Concede]` **Durability gap.** The v234d checkpoint dir was empty for the full ~3.5h pretrain phase because `train.py` only writes `pretrain_complete.pt` at the END of Phase 2.5. If a crash had hit during Phase 1 or Phase 2, all 3.5h would have been lost with no resumable artifact. This is the same class of issue as LANL R1's "no durable training log in checkpoint dir" — the launch path doesn't tee logs to the artifact directory and doesn't checkpoint mid-pretrain.

2. `[Action queued]` Two follow-up fixes for any future fresh-pretrain run:
   - **Mid-pretrain checkpoint emission** at end of each phase (e.g. `pretrain_phase1.pt`, `pretrain_phase2.pt`) so crashes don't waste the prior phase.
   - **Tee training log to checkpoint dir** at launch time. Same fix shape as the LRU-diagnostic log durability concern.

These are ~30 min of launcher + train.py changes; queued behind the v235 outcome.

### Bottom line

Concede. The durability fix is small and worth landing before any further fresh-pretrain attempts.

---

## R9 — Rebutting LANL Round 9 in PEER-REVIEW-LLNL.md (v234d has pretrain, no Phase 3 result)

1. `[Concur]` Observational. `pretrain_complete.pt` (37.5 MB) landed at 23:13 PDT. Phase 3 ep1 fired at ~23:48. Round 141's documented gates (ep10 frozen ★ vs v229 ★=0.039) were the right next deliverable.

### Bottom line

No dispute.

---

## R10 — Rebutting LANL Round 10 in PEER-REVIEW-LLNL.md (v234d ep10 frozen gate fails; do not launch IDEA #117 from this checkpoint)

1. `[Concede + amend]` **v234d ep10 fails the v229 reproduction gate.** Confirmed in `RESPONSE-LLNL.md` Round 141 and `VERSIONS-LLNL.md` v234d entry. Frozen ★=0.19719, β-recall=0.091. LANL's additional metrics (HRC-MAE=0.1488, DMD-GEN=0.7501, reuse real 0.022 vs fake 0.126) are useful — they were not in `frozen_sweep.json` (which only carries MMD² / recall / precision / ★). LLNL has incorporated these into the v234d entry going forward.

2. `[Contest, narrowly]` LANL's recommendation — *"Do not launch IDEA #117 from this checkpoint as though v229 reproduced"* — is well-grounded but addresses a methodological framing LLNL did not adopt. **v235 is NOT framed as a v229 reproduction continuation.** It is framed as a controlled A/B against v234d's *from-scratch baseline ★=0.197*. With identical pretrain (v234d's), identical seed=5, identical recipe minus the new flags (`--retrieval-train-carry`, `--long-chain-weight 2.0`), v235 isolates the joint effect of IDEA #117 + IDEA #116 from the v234d baseline. The race-relevant question is no longer "does v229 reproduce" (Round 141 said no); it is "does the IDEA #117 mechanism move the from-scratch ★ at all".

3. `[Concede]` That said, LANL's underlying concern is real: v234d's pretrain produces a mode-collapse-prone Phase 3 (β-recall=0.091 frozen). IDEA #117 inheriting that pretrain may be unable to recover even if the mechanism is sound; the experiment cannot distinguish "IDEA #117 doesn't work" from "v234d pretrain is bad". A clean-control v235' from a different fresh pretrain (3.5h cost) would resolve the ambiguity. **LLNL's call: let v235 reach ep10 and read the frozen ★. If ★ < 0.197 (v234d's bar), IDEA #117 helps even on this pretrain — informative. If ★ ≥ 0.197, queue v235'-from-fresh-pretrain as the disambiguation.** Cost-benefit favours sequencing this way.

4. `[Concur on the long-rollout panel requirement]` If v235 ep10 surprises with ★ ≤ 0.045 (matching the historical v229 number), LLNL still owes the full panel (HRC-MAE / reuse_access / stack median+p90 / footprint / drift / mark_score) before claiming any new tencent ATB — Darrell `PEER-REVIEW-LANL.md` Round 45 P0 standing requirement that v229 itself was delinquent on. Round 141's plan made this explicit.

### Bottom line

LLNL accepts the methodology critique with one amendment: v235 is not a continuation of a falsified v229 repro; it is a structural-mechanism A/B test against v234d. If v235 ep10 doesn't move ★ vs 0.197, LLNL will pay the 3.5h fresh-pretrain cost and rerun cleanly. The IDEA #117 implementation itself (commit `9d89806`) is unrelated to v234d's pretrain quality.
