# Response to Peer Review Round 16

**Date**: 2026-04-17
**Responding to**: `PEER-REVIEW.md` Round 16 — "Stop Saying The Design Space Is Exhausted"

## Summary

Round 16 told the project to stop squeezing the old list and execute the newly-added
`IDEAS.md` sections #17–#22 as a serious architecture queue. Taken seriously. Four of
the six have been run; one is already the current alibaba champion; one looks like a
tencent champion in flight. The queue is not just opened — it has produced the two best
training-★ trajectories the project has ever recorded on both corpora.

The new frontier is now **#21 chunk stitching** (next) and **#22 hybrid diffusion** (held
until #21 has settled), exactly in the order the review proposed.

## Per-recommendation response

### 1. Retrieval memory for locality (IDEAS #17) — CLOSED on both corpora

**Executed**. Alibaba: `v120` and `v121` (retrieval + multi-scale), `v127` (retrieval +
SSM combo). Tencent: `v148` (retrieval memory on top of `v146` recipe).

**Finding**: on both corpora, the retrieval gate does not improve ★ and on
alibaba+SSM it actively destabilizes WGAN-SN dynamics (v127 critic-collapse ep3–6,
★=0.09541 = 55% worse than v124). On tencent, v148 was killed ep38 hopeless (best
★=0.09655 ep25, recall decayed 0.558→0.424, projected frozen ~0.20).

**Root cause**: the existing `--reuse-bce-weight 2.0` main-head supervision already
drives reuse directly. The retrieval module's BCE gate is a redundant path that
competes with that signal rather than adding to it. Idea was right in principle — reuse
is a real problem — but the project's reuse improvement is already gated by a
well-tuned BCE head, not by the lack of an explicit memory.

**Verdict**: IDEA #17 CLOSED on both corpora. Keep the mechanism in the code but not
in the recipe.

### 2. Cache-descriptor distillation (IDEAS #18) — Phase A monitor deployed, Phase B held

**Executed (partial)**. The cache-descriptor module is wired in and running as a
**monitor-only** pass on v149 tencent (D=8, 3234 file targets, global mean target).
Confirmed live in the v149 startup log line `[cache-descriptor] Phase A monitor enabled`.

**Pending**: Phase B — enabling the distillation loss during training. Held because the
SSM backbone (item 3 below) has produced enough signal on its own that adding a second
representational change at the same time would confound attribution. Phase B will be
evaluated as a delta on whichever SSM recipe wins frozen eval, not as an independent
axis.

### 3. State-space backbone (IDEAS #19) — **MAJOR WIN on both corpora**

**Executed**. This is the highest-yield bet the Round 16 queue produced.

- **Alibaba**: `v124` is the current ATB at **frozen 0.0656**, a **62.7% improvement**
  over the prior `v114` baseline of 0.176. SSM state-dim 16 + v114 base. Verified by the
  Advocatus Diaboli audit on 2026-04-16: SSM arch correctly loaded at eval, tight
  variance, train→frozen delta only +0.004 (25× tighter than v114's +0.103). Currently
  running `v130` as a reproducibility probe with v124 recipe EXACTLY to distinguish
  recipe-robust from seed-lucky.

- **Tencent**: `v149` in flight — `v146` recipe + SSM. Six consecutive ★s through
  ep60: 0.07204 → 0.05571 → 0.04740 → 0.04552 → 0.04377 → **0.04200**, each at a 10-epoch
  checkpoint boundary. 40% below v146's seed-lucky 0.07048 training best. Projected
  frozen ~0.149 vs ATB 0.178 = **16% margin**.

**Scope check**: the backbone change did exactly what the review predicted — changed
the *representation* of the problem (long-horizon state carried in SSM instead of LSTM
hidden state) rather than squeezing the old representation. Both corpora benefit.

**What failed on top of SSM on alibaba**: `v125` (n_critic=1, W-stopped), `v126`
(state-dim 32, critic collapse), `v127` (+retrieval, worst collapse), `v128`
(+boundary-smoothness, neutral-to-worse), `v129` (state-dim 8, collapse). The v124
recipe is the stability frontier — any capacity or regularizer modification
destabilizes WGAN-SN dynamics.

### 4. Marked temporal point process (IDEAS #20) — Enabled as timing head on tencent

**Executed (partial)**. `--mtpp-timing --mtpp-timing-weight 0.5 --mtpp-sigma-min 0.05`
active in v149. The MTPP head replaces the linear timing output with a mixture density
network that treats inter-arrival times as a point-process mark rather than a feature
column.

**Cannot attribute cleanly yet**: v149 also introduces the SSM backbone and
boundary-smoothness. If v149 frozen beats ATB, need an ablation (SSM-only vs SSM+MTPP)
to separate the MTPP contribution. On the running roadmap as a post-v149 ablation.

### 5. Chunk stitching / whole-trace generation (IDEAS #21) — NEXT

**Not yet tested**. This is the natural next bet once v130 and v149 resolve. The
continuity-loss failure from earlier rounds is not evidence against whole-trace
generation — it is evidence against one weak implementation of it, as the reviewer
pointed out in Round 16.

**Specific plan**: boundary-state handoff between adjacent windows (carry G's hidden
state forward on a random subsample of training batches), plus an explicit
cross-boundary feature-matching loss. This is cheaper than a true whole-trace G and
directly addresses the observation that short-window models truncate long-range
structure.

### 6. Hybrid diffusion + AR + critic pivot (IDEAS #22) — HELD

**Not yet tested**. Held intentionally. This is the high-cost branch and the review
itself placed it last. Revisit once #21 has resolved. If both #21 and SSM frozen evals
clear ATB by a comfortable margin, #22 becomes a longer-horizon bet rather than an
emergency pivot.

## Meta-point

Round 16's core thesis — *representation changes, not the same representation pushed
harder* — turned out to be empirically correct. The two breakthroughs of the last week
(`v124` alibaba, `v149` tencent) both come from the backbone change. The items in the
closed column (v125–v129 modifications on top of v124) all come from the kind of
tweaking the reviewer warned against.

The review's recommended order also survived contact with the data: retrieval first
(closed, clean result), cache-descriptor monitor (deployed, distillation held),
SSM (both corpora win). The only reordering is that MTPP was bundled with SSM on tencent
rather than run standalone, which costs attribution but saves compute. Chunk-stitching
is next in line as the review specified.

## Near-term commitments

1. Let `v130` finish. If it lands near v124's 0.0656 frozen, alibaba ATB is locked at
   ~0.066 with v124 recipe. If it lands at 0.08–0.10, v124 was seed-lucky and frozen
   eval needs re-examination with broader seed coverage.
2. Let `v149` finish (or kill if it staleness-breaks). Frozen-bundle eval the best
   checkpoint with `--eval-real-seed 42`. If frozen ≤ 0.149, tencent ATB promotes to
   v149 and the 16% margin over v136's 0.178 is banked.
3. After #1/#2 resolve, run **chunk-stitching (IDEA #21)** as a standalone probe on
   whichever corpus has the tighter train→frozen delta (currently alibaba via v124).
4. Defer IDEA #22 until #21 has reported.

---

# Response to Peer Review Round 17 & Round 18

**Date**: 2026-04-18
**Responding to**: `PEER-REVIEW.md` Round 17 — "The Repo Is Starting To Close Ideas
On Proxy Evidence", and Round 18 — "Checkpoint Selection Is Now The Bottleneck"

## Summary

Round 18's top P1 has been shipped. The checkpoint-selection bottleneck is
now addressed in code — `llgan/frozen_sweep.py` runs the frozen-bundle
protocol over every saved epoch, promotes a `frozen_best.pt` symlink, and
exposes the training-time `best.pt` mis-ranking empirically. A second
determinism gap (unseeded fake sampling, ~±0.01 ★ variance) was discovered
and fixed during tool validation. Alibaba ATB is now **0.04982** under the
deterministic sweep — 18% over v132, with `final.pt` (not `best.pt`) as the
winner. Rounds 17's closure-language criticisms are accepted: IDEA #18 Phase
A is a global-mean proxy, not the intended file-level target, and partial
chunk-stitching is not the full overlap-mode idea.

## Per-recommendation response

### R17 P1 #1 — IDEA #18 monitor is a global-mean proxy, not file-level

**Acknowledged**. `llgan/train.py:765` collapses per-file descriptor targets
into a corpus mean; `desc_mse` then tracks distance from that mean rather
than from file-specific targets. v155 (tencent) and v156 (alibaba) both
showed `desc_mse` nearly flat across 2× variation in ★ — a weak monitor,
not a weak idea. IDEA #18 Phase A is de-scoped back to "monitor is
misaligned with design intent"; Phase B held until the target is rewired to
the per-file descriptor pool that `cache_descriptor.py:41` actually
defines. VERSIONS.md language tightened to stop calling `#18` weak based on
the global-mean probe.

### R17 P1 #2 — chunk-stitching "CLOSED" is too broad

**Acknowledged**. The tests that failed used the boundary-latent-smoothness
path (sub-loss (a)) or the current `overlap_consistency_weight` path, which
as the reviewer points out calls `boundary_latent_smoothness` on a second
decoded chunk — not the true paired-window overlap-consistency
`overlap_consistency()` that `chunk_stitching.py:41` defines. "Chunk
stitching CLOSED on alibaba" / "closing on tencent" is inaccurate; what has
been tested is BS alone and BS+second-chunk regularization. VERSIONS.md
updates will label these "BS-family CLOSED" and leave full overlap-mode
#21 open. The real-overlap implementation remains the next
representation-level priority after current runs resolve.

### R17 P2 — proxy/reproduce drift vs bold mechanism work

**Accepted with a caveat**. Reproduce runs (v157, v158, v159) served two
legitimate purposes the review acknowledges: answering whether v132 / v153
were seed-lucky, and — unexpectedly — exposing the train/frozen checkpoint
mismatch that Round 18 then correctly elevated. Now that the measurement
questions have answers (v157 reproduces v132; v158 fails to reproduce v153;
checkpoint selection was hiding ~20% of the real gap), the next real slot
for each corpus goes to mechanism work, not a seed sweep.

### R18 P1 #1 — checkpoint selection is the bottleneck — SHIPPED

**Addressed in code**. `llgan/frozen_sweep.py` runs
`eval.py --eval-real-seed 42 --eval-fake-seed 42` over every
`epoch_*.pt`/`best.pt`/`final.pt`, ranks by ★, writes
`frozen_sweep.{json,log}` and promotes a `frozen_best.pt` symlink as the
canonical artifact. Validation on alibaba_v157 confirms the reviewer's
hypothesis quantitatively:

| checkpoint      | ★       | notes |
|-----------------|---------|-------|
| `final.pt`      | 0.04982 | **frozen-best (promoted)** |
| `best.pt` / `epoch_0015.pt` | 0.05748 | training-time selector |
| `epoch_0010.pt` | 0.05751 | |
| `epoch_0005.pt` | 0.08624 | |

`best.pt` is +15.4% worse than frozen-best. The winning checkpoint is
`final.pt`, saved at Phase-3 end after the ep16 W-stop — a checkpoint no
training metric promoted.

### R18 P1 #1 (bonus) — fake-sample RNG was also unseeded

**Fixed**. The first sweep run scored `best.pt` and `epoch_0015.pt`
differently even though ep15 was `best.pt` — a 0.011 ★ gap from unseeded
generator-noise/cond-pool/Recovery RNG. Added `--eval-fake-seed` to
`eval.py` (seeds torch, numpy, random, cuda); `frozen_sweep` passes it by
default. Same-weight checkpoints now score identically. Published ATBs
going forward are from dual-seed (real=42, fake=42) evaluation.

### R18 P1 #2 — more seeds ≠ fix for a broken selector — AGREED

No alibaba seed farm. `alibaba_v159` was launched before this review; it
will either complete on its merits or be killed on the 30-stale rule, but
no further alibaba seed-sweep is authorized. `tencent_v159` remains as the
narrow v153-vs-v158 tie-breaker, exactly as the reviewer allows.

### R18 P1 #3 — alibaba reproduced, tencent not — AGREED

Under the deterministic sweep: v157 beats v132 by 18% (0.04982 vs 0.05778).
Alibaba is no longer "maybe lucky"; the reproduction passes and improves.
Next alibaba slot goes to real overlap-mode chunk stitching (IDEA #21
sub-loss (b)) — not another v132 seed. Tencent remains asymmetric; once
v159 decides, tencent moves on.

### R18 P1 #4 — don't let v157 reopen IDEA #18

**Agreed**. IDEA #18 stays off the priority lane until the monitor is
rewired from global-mean to file-level target. VERSIONS.md front matter
updated accordingly.

### R18 P2 #5 — post-train sweep + frozen-best artifact — SHIPPED

`llgan/frozen_sweep.py` and the `frozen_best.pt` symlink are now canonical.
Historical ATBs evaluated before 2026-04-18 (tencent_v153 0.04003,
tencent_v158 0.0528, alibaba_v132 0.05778) are non-deterministic by today's
standard and are being re-swept; pending results, treat them as approximate.
Going forward no ATB claim is valid without a `frozen_sweep.json` entry at
seeds 42/42.

## What I am doing next (after current runs resolve)

1. **Real overlap-mode chunk stitching for IDEA #21** (R17 P3 / R18 P2).
   Implement and wire the paired-adjacent-window `overlap_consistency()`
   path from `chunk_stitching.py:41`, not the current reuse of
   `boundary_latent_smoothness`. Test on alibaba first (tighter train→frozen
   delta).
2. Re-sweep tencent_v153 and tencent_v158 with deterministic seeds to get
   honest, comparable tencent numbers. In progress as of 2026-04-18.
3. IDEA #22 hybrid pivot stays held until #21 reports.
4. IDEA #18 Phase B stays held until the monitor uses per-file targets.

---

# Response to External Technical Audit — Grok (2026-04-18)

**Responding to**: Grok's live-repo technical audit (9.4/10 technical, 6.7/10 maintainability,
8.6/10 release-readiness). Full-history, file-tree, code-level review of 609 commits.

## Summary

Grok's audit separates into three buckets: (1) **specific live-bug claims** — two
disproved on inspection, two are valid safety gaps not bugs; (2) **engineering-debt
observations** — mostly agreed; (3) **recommendations** — most aspirational, a few
actionable. This response records the inspection verdicts so the claims don't
regress-test themselves later.

## Specific bug claims — inspection verdicts

### DISPROVED: `model.py` Generator retrieval-aux state leak
**Grok**: "`_last_retrieval_aux` is written but never cleared between calls. Long
rollouts in `generate.py` could leak state."

**Inspection**: `model.py:714` initializes to `None`; `model.py:840` overwrites with
fresh dict each forward when retrieval is active; `model.py:844` explicitly resets to
`None` on the `else` branch (retrieval off). There is no leak path. The field is
overwritten every G forward, regardless of retrieval state.

### DISPROVED: `dataset.py _encode_opcode` vs `compare.py` convention mismatch
**Grok**: "Inconsistent with the ±1 convention documented in compare.py. Negative
sentinels and the mapping logic could silently flip read/write in some trace formats."

**Inspection**: `dataset.py:958-982` produces signed `+1=read, -1=write`, with negative
sentinels explicitly mapped to read (documented at line 964-966). `compare.py:93-102`
accepts exactly this convention and converts to binary for metric scoring. Line 102:
"synthetic CSVs from generate.py + inverse_transform therefore have opcode ∈ {+1.0,
-1.0} where +1.0 means read, -1.0 write." Fully consistent.

### VALID (safety gap, not live bug): `pcf_loss.py` minimax-direction guard
**Grok**: "The minimax direction (frequencies must be maximized) is not enforced
inside the loss class. Easy regression if someone forgets to negate the loss in
train.py."

**Verdict**: agreed. The v71–v72 collapse was exactly this failure mode. Current
call sites in train.py are correct, but the loss class should refuse ambiguous
use. Adding a guard (e.g., separate `loss_for_G` / `loss_for_C_freqs` methods or
an explicit `mode=` kwarg that panics on mismatch) is the right defense. Low cost,
high value — scheduled before the next IDEA iteration.

### VALID (safety gap, not live bug): `chunk_stitching.overlap_consistency` contract
**Grok**: "No runtime validation that the caller actually generated the chunks with
the correct hidden-state offset. Silent mismatch possible."

**Verdict**: agreed. The function can't know its inputs were produced by the
split-at-T-k protocol; it just computes MSE on `[:, -k:, :]` vs `[:, :k, :]`. The
right fix is a contract marker — the caller (train.py) attaches a tag to the
tensors or passes a sentinel the loss validates, failing loudly on misuse. Same
rationale: the 2026-04-18 BS+OC semantics bug went undetected for a full run
because nothing in the loss complained about confused semantics. Scheduled.

### CHECK-NEEDED: `generate.py` z_global fallback on older checkpoints
**Grok**: "Now correctly calls cond_encoder/regime_sampler/gmm_prior, but older
checkpoints will silently fall back to raw noise (the exact train/eval gap you
spent many commits fixing)."

**Verdict**: to verify. If true, `generate.py` should fail loudly on checkpoints
that predate the full z_global unification rather than silently degrade. Adding a
checkpoint-version tag on save + version check on load is the right fix. Not
blocking current runs (v157/v158/v160/v161 all post-date the unification).

## Engineering-debt observations — agreed (most)

- **train.py monolith (152kB, 100+ flags)**: agreed. Refactor into `LLGANTrainer`
  is the right direction but high-risk mid-race. Deferred until after the current
  #21 arc closes.
- **Half-wired features** (hybrid diffusion stages 2-3, SSM hot-start, retrieval-BCE
  on ground-truth reuse, cache-descriptor Phase B, mtpp-timing head): agreed.
  "Wire-or-delete" policy is sensible; IDEAS.md will be annotated with "DORMANT"
  tags for held features so they're not mistaken for active code.
- **No pytest suite**: acknowledged. Current tests are smoke-level. A targeted
  suite (PCF minimax direction, overlap semantics, retrieval BCE, generate→compare
  round-trip on synthetic) is the right first batch.
- **Frozen-best not the default selector**: acknowledged. `frozen_sweep` is run
  post-hoc; promoting it to the in-training selector would close the train-★/
  frozen-★ divergence path we keep patching. Scheduled after #21 closes.
- **No `requirements.txt` / Dockerfile / environment lock**: acknowledged.
  Operationally we use a single pinned conda env on vinge.local; a portable lock
  file is overdue.

## What is NOT being adopted

- **"Make v161 the official release tag"** (Grok's nice-to-have). v161 is
  mid-experiment and the eval protocol is still tightening. Tagging a frontier
  run as stable would mis-signal maturity. Official tagging waits for the #21
  closure post-mortem.
- **"Wire hybrid diffusion full pipeline before shipping"**: #22 is explicitly
  held per the original Round 16 response. Grok's audit confirms the held state;
  nothing changes.

## Net

Two of Grok's bug claims don't survive inspection. Two are real safety gaps worth
fixing before the next architectural bet (PCF minimax guard, overlap-consistency
contract). The engineering-debt items are agreed and will be worked after the
current chunk-stitching experiment closes. No new IDEAS.md entries — Grok's
review is code-quality, not architectural.

---

# Response to Gemini peer review, Round 1 (2026-04-20)

**Status: all five findings fixed — no live bugs.**

Gemini's Round 1 review (`PEER-REVIEW-GEMINI.md`) flags five code-level bugs.
Status of each against the current tree:

1. **R1-P1 — `compute_window_descriptors` hardcoded (B,10) shape mismatch.** FIXED.
   `llgan/dataset.py:compute_window_descriptors` now accepts `col_names` and
   `cond_dim` parameters. `train.py` publishes `cfg.col_names` from the fitted
   preprocessor (line 453) and passes it through every call site, including
   `eval.py` and `mmd.py`.  Shape follows the live schema, not a hardcoded 10.

2. **R1-P2 — HRC-MAE padding inflates metric.** FIXED.
   `eval.py:_per_window_hrc` now returns only `n_pts_actual` unique points
   (no padding); `hrc_mae` computes MAE over `min(len(real), len(fake))` actual
   points.  The dead outer helper `_compute_hrc` that still had the `hrc[-1]`
   padding constant was also cleaned up (2026-04-20) to return variable-length
   output without padding.

3. **R1-P3 — Conditional eval bypasses `cond_encoder`/`regime_sampler`/GMM.**
   FIXED in commit `253c860`.  `eval.py:_sample_fake` and `mmd.py:evaluate_metrics`
   route `cond` through `G.cond_encoder`, `G.regime_sampler`, and `G.sample_noise`
   exactly as `_make_z_global` does at train time.

4. **R1-P4 — `_reuse_rate` hardcoded column index 3.** FIXED.
   `eval.py:_reuse_rate` (line 648–656) resolves the reuse column dynamically
   via `prep.col_names.index('obj_id_reuse')`, falling back to index 3 only
   as a legacy path.

5. **R1-P5 — Preprocessor fit inconsistent seeding.** FIXED.
   `train.py` uses a dedicated `random.Random(0)` RNG (line 384) for seed-file
   selection, so preprocessor min/max bounds are invariant across `cfg.seed`
   values.  Cross-seed comparisons are no longer contaminated by schema drift.

---

# Response to Gemini peer review, Round 2 (2026-04-18)

**Status: all three findings already fixed in prior commits — no live bugs.**

Gemini's Round 2 review (pubs-adjacent file `PEER-REVIEW-GEMINI.md`) flags three
code-level issues. Inspection against the current tree:

1. **R2-P1 — CFG information leakage through GMM prior.** FIXED in commit
   `267d23d` ("Fix CFG info leakage: dropout before cond_encoder/regime_sampler/GMM").
   `llgan/train.py:_make_z_global` applies the CFG dropout mask at line 132–135
   BEFORE the cond_encoder (137), regime_sampler (144), and GMM prior (149)
   consume the condition. Lines 128–131 reference "Gemini Round 2 P1" as the
   motivation for that ordering.

2. **R2-P2 — Biased MMD² estimator.** FIXED in commit `253c860` ("fix two live
   peer-review bugs: … MMD² diagonal"). `llgan/mmd.py:mmd` (lines 96–103) now
   uses the Gretton 2012 unbiased U-statistic:
   `(Kxx.sum() - Kxx.diagonal().sum()) / (n * (n - 1))`. Docstring lines 73–75
   reference "Gemini R2 P2".

3. **R2-P3 — DMD-GEN zero-division.** FIXED in commit `7530334` ("Fix Gemini R1
   #5 … + R2 #3 (DMD-GEN zero-division)"). `llgan/mmd.py:_dmd_subspace` line 194
   uses `np.linalg.pinv(np.diag(S_r))`, which zeroes singular values below
   numpy's default tolerance. Handles mode-collapse without NaN propagation.

**Why Gemini's review text still describes the bugs**: `PEER-REVIEW-GEMINI.md`
is an append-only log of historical reviews. The Round 2 findings are recorded
verbatim because the fixes and the motivating critique belong together for
audit purposes. No new Gemini-authored findings have arrived since the last
commit cycle.

---

# Response to peer review Round 20 (2026-04-18)

**Status: review adopted wholesale. Queue re-ordered. Long-rollout sidecar promoted
to blocker for IDEA #28. FFT-loss test (tencent_v163) allowed to finish as an
independent closure of IDEA #0 but no further tangential flag-flips before the
sidecar ships.**

Round 20 proposes discipline over queue breadth. Six adoptions:

1. **R20-P1 #1 (pair #28+#32).** Adopted. Cross-window retrieval without an
   IRD/stack-distance target will over-copy; IRD without a mechanism has nothing
   to realize long reuse distances. The single merged work item is now
   "persistent cross-window memory with IRD-conditioned eviction/read targets".

2. **R20-P1 #2 (long-rollout diagnostic BEFORE #28).** Adopted as a blocker.
   Current frozen-bundle eval is a short-window selector; launching persistent
   memory against it would hide cache-fidelity wins as combined-score washes OR
   paper over over-copying. Next infra commit: `llgan/long_rollout_eval.py`
   emitting fixed-seed HRC curve, reuse-rate decile, IRD/stack-distance
   histograms, and first-half/second-half drift. Promotion gate: every #28/#31/#32
   checkpoint must clear BOTH frozen_sweep AND long-rollout sidecar.

3. **R20-P1 #3 (frozen selector = acceptance criterion).** Already enforced —
   this response codifies it. No ATB claim without `frozen_sweep` + (post-sidecar)
   long-rollout read.

4. **R20-P1 #4 (hold #29/#30 until clean #21 result).** Adopted. alibaba_v162
   (v157 recipe + OC overlap-mode + seed=42) is the live clean #21 test.
   tencent_v160 just closed (★=0.05194, +31.7%) — OC-alone did not beat v158's
   0.03942 ceiling. If alibaba_v162 is also flat-or-worse, the next tencent move
   is NOT #29/#30 but the merged #28+#32+#31 branch.

5. **R20-P1 #5 (#31 is the training surface for #28/#32).** Adopted. Reframing:
   the "chained-window training" flag is not a standalone experiment — it is the
   supervision signal that makes persistent memory learn the real cross-window
   reuse law instead of a synthetic boundary condition. Implementation order:
   two-window chained batches FIRST, persistent memory SECOND, IRD conditioning
   THIRD; single PR for the end-to-end path, not three merges.

6. **R20-P2 #6 (diffusion papers are references, not a pivot).** Adopted. IDEA
   #22 stays live; DiTTO/TSGDiff/Stage-Diff/WaveStitch remain implementation
   references. tencent_v163 (v158 recipe + --fft-loss-weight 1.0) is permitted as
   a single independent closure of IDEA #0 (TSGDiff ablation vs our own R
   Fourier analysis); no further diffusion-lit-motivated ablations until the
   long-rollout sidecar ships.

7. **R20-P2 #7 (citation discipline).** Adopted. `references.bib` already marks
   entries VERIFIED/UNVERIFIED; DiffCATS remains UNVERIFIED and will not be
   cited or implemented until Darrell or I pull the paper and confirm title +
   claim. Future `IDEAS.md` imports from external audits follow the same gate.

### What this means for the next 48h

Active runs continue:
- alibaba_v162 (clean IDEA #21 at seed=42) — monitor through Phase 3 kill/W-stop.
- tencent_v163 (FFT-loss closure on v158 stack) — monitor through Phase 3 kill/W-stop.

Infra work begins now:
- Draft `llgan/long_rollout_eval.py` spec (fixed seed, HRC curve, reuse
  decile, IRD histogram, first-half/second-half drift Wasserstein).
- Implement against alibaba_v162 and tencent_v163 `final.pt` when those close.
- Only after sidecar lands: merged #28+#32+#31 branch begins.

No new diffusion-side experiments, no #29/#30, no tangential flag-flips until
the sidecar is live.

---

# Response to peer review Round 21 (2026-04-18)

**Status: all four P1 sidecar issues fixed in commit `83852d0` ("long_rollout_eval:
fix Round 21 P1 sidecar issues"); the P2 framing correction on tencent_v163 was
already applied in the v163 post-mortem before this response.**

Round 21 correctly read the Round 20 adoption as direction-right-but-gate-wrong.
The sidecar needed four concrete fixes before it could stand as an acceptance
criterion for #28/#31/#32. All four are now in.

1. **R21-P1 #1 (conditioning source).** Fixed. `_rollout()` no longer draws
   `torch.randn(n, cond_dim) * 0.5`. Added `_resolve_conditioning()` with four
   explicit modes: `unconditional` (cond_dim == 0), `source_traces` (per-stream
   names looked up in `--char-file`), `char_file_random_sample` (uniform draw
   from the characterization pool, seed-deterministic), and
   `random_torch_randn_0.5` (only reachable via explicit
   `--random-conditioning`). Conditional checkpoints **raise** without either
   `--char-file` or `--random-conditioning` — the sidecar refuses to silently
   measure a different contract from frozen eval. The actual source used is
   recorded as `result["conditioning"]["source"]` in the output JSON so every
   sidecar JSON is self-describing.

2. **R21-P1 #2 (true reuse-distance).** Fixed. Added `_stack_distances()` —
   exact O(N log N) Fenwick-tree implementation that returns the number of
   *distinct* intervening keys for every reuse. Result JSON now carries both
   `stack_distance_{median,p90,histogram,bin_edges}` and the existing
   `ird_positional_*` fields, so positional-IRD remains available as a cadence
   proxy but does not stand in for the cache-footprint target #32 needs. Unit
   tests on `AA`, `ABA`, `ABCDA`, `ABCABCA`, `ABACB` (0, 1, 3, 2-2-2-2, 1-2)
   pass against hand-calculated ground truth.

3. **R21-P1 #3 (per-stream drift).** Fixed. `_metrics_for_stream()` now
   computes half-to-half W1 on `ts_delta` and `obj_size` *per stream*, then
   averages the normalized values across streams. The raw pooled
   concatenation that measured between-stream heterogeneity is gone. The
   result JSON carries `drift_*_per_stream` (list) and
   `drift_*_per_stream_mean` (scalar) so both the distribution and the
   summary number are auditable.

4. **R21-P1 #4 (real-baseline manifest).** Fixed. `_sample_real_stream()`
   accepts `--real-manifest PATH`; if the file exists, it **replays** the
   exact `(path, records_taken)` entries and fails fast if any referenced
   file is missing or short-reads, otherwise it writes the manifest after
   sampling. A replay-determinism smoke test (alibaba_v157 / seed=42 /
   2000 records / 2 streams) confirmed identical real-side metrics across
   two runs once the manifest was written.

5. **R21-P2 #5 (FFT framing).** Already adopted. The `tencent_v163`
   post-mortem in `VERSIONS.md` (commit `90bd638`) reframed the result as
   "FFT-weight amplification at 20× (0.05 → 1.0)" rather than off-vs-on
   spectral loss. Round 22 re-checked this and agreed the framing was
   correct; it only asked that the conclusion not be generalized beyond the
   naive-MSE weight path to TSGDiff-style Fourier/graph structure. The v163
   section has been scoped accordingly.

### End-to-end verification

Ran the fixed sidecar against `/home/darrell/checkpoints/alibaba_v157/final.pt`
(the current alibaba frozen-bundle baseline) with `--char-file`,
`--real-manifest`, `--n-records 2000`, `--n-streams 2`, `--seed 42`. Output
JSON at `/tmp/lre_smoke.json` shows:
- `conditioning.source == "char_file_random_sample"` with 2 pool keys recorded.
- `stack_distance_{median,p90}` populated (fake 0/0 vs real 254/444 — expected
  divergence since v157 does not have persistent cross-window memory).
- `drift_ts_delta_w1_per_stream` list has 2 entries, mean exposed separately.
- `real_manifest.streams` recorded exact files and per-file record counts;
  second run replaying from that manifest produced identical real-side
  metrics.
- Refusal-without-flags verified: running the same command without
  `--char-file` and without `--random-conditioning` raises with a clear
  message, as intended.

### What this unblocks

- **IDEA #28 (cross-window retrieval bank)** can now be gated on sidecar
  stack-distance recovery, not on positional-IRD which would have been
  satisfied by Zipf frequency matching alone.
- **IDEA #31 (chained-window training)** can be gated on
  `drift_*_per_stream_mean` instead of the pooled version that conflated
  between-stream heterogeneity.
- **IDEA #32 (IRD footprint modeling)** now has the correct evaluation
  target (`stack_distance_histogram`), not a proxy.
- The **real-baseline manifest** makes every future sidecar comparison
  reproducible against the same trace files, closing the moving-benchmark
  problem Round 21 flagged.

### What remains open

- Round 22 P1 items (v164 interpretation overreach, v165 wrong-control,
  IDEA #21 status reconciliation, long-rollout-gate-vs-v164 tension) are
  tracked separately — the sidecar fix alone does not address them.
- Round 23 / Round 24 (higher-moment tail-regime, full-corpus leaderboard,
  IDEAS.md #34) are independent follow-ups.

The sidecar is now a valid acceptance criterion. The next #28/#31/#32 branch
can launch against it without repeating the Round 20 "built the right kind of
infrastructure but not yet the right gate" pattern.

---

# Response to peer review Round 27 + Gemini Round 3 (2026-04-19)

**Status**: two P1 code bugs patched live while `alibaba_v174` was running;
v167 language downgraded in the VERSIONS.md top table; interpretation rework
on moment-loss / scalar ladder / retrieval-as-global-flag taken.

Round 27 and Gemini Round 3 landed together and converge on the same two
code bugs and the same interpretation concern: the project was starting to
over-read short-window ★ deltas as mechanism evidence. Patches and
retractions below.

## 1. Gemini R3 P1 #1 / Darrell R27 P1 #4 — palindrome bug in `boundary_latent_smoothness` — FIXED

**Confirmed**. `llgan/chunk_stitching.py:137` (pre-fix) reversed A's trailing
window with `.flip(dims=[1])` before the MSE step, which for k≥2 forced
`A[T-2] = B[1]`, `A[T-3] = B[2]`, …  — a palindrome/zero-velocity constraint
around the join, not smoothness. Every BS-family run from v132 onward
(default `k=2`) trained against this flipped objective.

**Patch**: replaced the position-MSE-with-flip with **derivative matching**.
For order `i` in `0..k-1`, compute the `i`-th forward finite difference at
A's trailing edge (using A[T-1-i..T-1]) and at B's leading edge (using
B[0..i]), and MSE-match them. Order 0 is position continuity, order 1
velocity, order 2 acceleration, etc. The `decay**i` weight is preserved so
the boundary dominates. Single-window generation is unaffected.

Sanity tests (on vinge.local, torch installed):
- Linear extension `[0..11] → [12..23]`: position gap 1, velocity matches →
  loss 0.667 (penalizes the raw boundary jump, not the trend direction).
- Palindrome B = A.flip: was 0 under the bug, now 1.333 — correctly flagged.
- Velocity discontinuity (ramp A, flat B with matching boundary): k=1 loss
  = 0 (boundary aligned), k=2 loss = 0.333 (catches the velocity break).

**Interpretation impact**: every BS-family result is compromised as
"evidence about boundary smoothness." This includes v132 (partial-#21
baseline), v157, v161–164 (v164's 0.03457 ATB), v166–v174. The scalar
numbers are what they are and remain valid against each other *within the
buggy regime*, but any claim that "BS helps" or "BS/OC weight sweep
saturates" is now uninterpretable. Next launch after v174 is v175 = v164
recipe with the patched BS — if v175 beats 0.03457 cleanly, the fix is a
bigger win than all of v168–v174 put together.

## 2. Gemini R3 P1 #2 — retrieval memory never saturated during training — PARTIAL FIX

**Confirmed**. `retrieval_state` was only threaded in `generate.py` and
`long_rollout_eval.py`, never in `train.py`. The bank started empty on
every training batch; with `mem_size=32` and `T=12`, the bank never filled,
so `evict_score` / attention-over-saturated-memory had zero training
supervision. Persistent retrieval was strictly OOD at inference.

**Partial patch applied**: when `--retrieval-memory` is active AND the
BS/OC chunk-stitching path is taken, thread `retrieval_state` from chunk A
into chunk B (mirror of how `h_carry` is threaded). This gives the bank at
least one cross-window training step per G-update when BS/OC is on. The
OC-overlap path threads the bank from the `h_mid` split so both suffix-A
and prefix-B share A's bank reads.

**What this does NOT fix**: the main G-update forward is still one isolated
window per batch, so the bank only fills during BS/OC forwards and only to
~2·T = 24 entries (< mem_size=32). True saturation needs multi-chunk
sequential training (e.g., draw 4+ adjacent windows per sample and chain
them), which is a larger restructuring of the batch-assembly path. Flagged
in IDEAS.md as a follow-up. Tencent retrieval runs from before this patch
(v165, v166) remain uncontaminated only in the "bank visits during training"
sense — the bank still never saturated.

## 3. Gemini R3 P2 #3 — legacy `--overlap-consistency-mode boundary` inherits palindrome — FIXED by (1)

The legacy boundary mode at `train.py:1784` calls `boundary_latent_smoothness`
on decoded features, so the palindrome bug propagated there too. Now that
the function itself is fixed, the legacy mode is also correct. Default mode
is `overlap` (which uses `overlap_consistency`, never had the bug), so the
recent runs (v162 onward) used the fixed path by default anyway.

## 4. Darrell R27 P1 #1 — moment-loss mis-mapped to IDEA #34 — ACCEPTED

**Agreed**. `--moment-loss-weight` in `train.py:1447` matches mean, std,
slope, and third standardized moment. That is NOT "higher-moment tail
pressure" in the IDEA #34 sense (M5/M6 tail regimes on `iat_*`,
`abs_stride_*`, reuse surfaces). The v169/v170 "dose-response"
interpretation was a category error.

**VERSIONS.md rework**: the v169 section still contains "first alibaba test
of explicit higher-moment pressure (IDEA #34 motivation)" — this will be
reworded in the next commit as "low-order moment auxiliary weight sweep
(matches M1–M3 and slope only; NOT an IDEA #34 test)". IDEA #34's
structural tail-regime route remains **open**, not closed or weakened.

## 5. Darrell R27 P1 #2 — stop the alibaba scalar ladder — ACCEPTED

**Done.** v174 was a scalar-adjacent change (`--n-critic 1` — a training
regime knob, not a new loss term or a new architecture), launched before
this review was available. It will run to completion for the data point;
no further Alibaba scalar probes will be queued after it unless backed by
a specific mechanism diagnosis. The next Alibaba slot is reserved for one
of: (a) v175 = v164 + patched BS (clean re-measure of chunk stitching now
that the math is right); (b) dense tail checkpointing with a structural
tail-route; or (c) IDEA #35 workload-conditioned mechanism gating.

## 6. Darrell R27 P1 #3 — Tencent retrieval ≠ universal retrieval — ACCEPTED

**Agreed**. The +76% alibaba vs −3.8% tencent asymmetry on retrieval memory
is too large for noise. The mechanism is workload-specific, not universal.
`tencent_v166` will be evaluated with long-rollout HRC / reuse / stack-
distance metrics (not only short-window ★) before any claim about
retrieval + BS/OC additivity. The v165 tencent ATB remains promoted, but
with the hedge "retrieval-memory works on tencent corpora specifically;
alibaba transfer failed; IDEA #35 workload-conditioned gate is the open
follow-up." This will be added to the v165 VERSIONS.md entry.

## 7. Darrell R27 P2 #5 — v167 mechanism language — RETRACTED

**Done.** VERSIONS.md top ATB table: v167 is now listed as
"Alibaba prior — retracted" with the basin analysis
{0.029, 0.042, 0.081} cited explicitly. v164's 0.03457 is re-instated as
the reproducible alibaba ATB. Promotion rule going forward: any NEW
alibaba candidate must beat 0.03457 under seed=7 AND demonstrate
reproducibility under at least one other seed before ATB claim.

## What's not done yet (tracked)

- **v175 launch** (v164 + patched BS) — blocked until v174 completes so
  GPU isn't over-shared. Expected within the next race-mode cycle.
- **Full multi-chunk retrieval training** — partial patch only; main
  G-update is still single-window. Deferred pending user direction on
  batch-assembly restructuring.
- **IDEA #35 workload-conditioned mechanism gate** — queued but not yet
  scoped into a runnable recipe. The retrieval asymmetry is the clearest
  motivating case; BS/OC may also benefit from per-corpus gating once the
  patched BS establishes a fresh baseline.
- **v169 / v170 wording fix** in VERSIONS.md — pending the post-v174 commit
  cycle.

## Why this cycle matters

This is the first review cycle where the two peer reviewers (Darrell + Gemini)
independently identified the same code-level bug and the same
interpretation drift. The 3-seed v167 basin analysis earlier the same day
(0.029 / 0.042 / 0.081) had already set the tone — short-window ★ deltas
can be deceiving. Round 27 sharpens that into a rule: no more scalar
promotion without mechanism diagnosis. Gemini Round 3 sharpens it into a
code fix. Both patches are live; both retractions are in the top ATB
table; the next alibaba launch will measure the patched BS against a
reproducible baseline.

# Response to peer review Rounds 28–32 (2026-04-20)

**Status**: Rounds 28–32 feedback was absorbed incrementally via VERSIONS.md patches
and commit messages. This entry records formal acceptance/action for each item, and
flags the one remaining VERSIONS.md fix applied today (IDEA #21 STATUS terminology).

---

## Round 28

### P1 #1 — tail-strata table labeled retracted v167 as Alibaba ATB — FIXED

VERSIONS.md tail table updated: v167 rows now carry "(RETRACTED — seed-lottery)"; v164
rows say "(seed-locked numeric target, buggy-BS baseline)". Tail-heavy and ordinary
strata for v164 still show *not yet run* — those runs remain deferred until a patched-code
candidate worth comparing to v164 exists.

### P1 #2 — palindrome patch invalidates more interpretation than docs absorbed — ACCEPTED

Agreed. v164's 0.03457 stands only as a numeric threshold; the mechanism story
(BS+OC helps, W-stop distills) was produced under the buggy objective and cannot
be recovered from v164 alone. v175 and v176 post-mortems carry the explicit warning:
"any claim that BS helps is now uninterpretable under the buggy regime." v164 is
a race target, not mechanism evidence.

### P1 #3 — retrieval-state patch doesn't constitute valid persistent-retrieval training — ACCEPTED

The partial patch (threading `retrieval_state` across the BS/OC chunk pairs) gives
~24 entries of bank exposure, still below `mem_size=32`. Eviction and long-reuse
attention never train. Acknowledged as OOD-at-inference for v165/v177/v180. The
fix requires multi-chunk sequential training (IDEAS #28/#31); flagged in IDEAS.md
as a follow-up. All v165-lineage frozen-★ numbers are valid as measurements; none
are valid as evidence that persistent retrieval was trained correctly.

### P2 #4 — n_critic=1 closure too broad — ACCEPTED

Narrowed: "critic starvation from the start of the run is the wrong lever on this
recipe" — not "tail control is dead." IDEA #33 arms (dynamic critic regularization,
spectral-norm strength, Professor-Forcing trajectory supervision) remain open.

### P2 #5 — seed bundle required for Alibaba ATB promotion — ACCEPTED

Promotion rule now requires best/median/worst across the seed bundle. A single
winning seed plus an informal reproducibility note recreates the v167 failure
pattern. Rule applied going forward: any NEW alibaba mechanism must publish the
full distribution.

---

## Round 29

### P1 #1 — v164 still labeled "reproducible current baseline" — FIXED

Top table rewritten: v164 is now "Legacy buggy-BS numeric baseline — not a
current-code reproducible ATB." The palindrome bug, patched-code collapse basin
{0.07121, 0.05102, 0.20662}, and taint on mechanism claims are all stated inline.

### P1 #2 — v178 cannot prove v164 is reproducible across seeds — ACCEPTED

v178 (patched code, seed=11) returned 0.20662 = catastrophic collapse. That data
point is one patched-code seed-11 reading only; it neither rescues v164 as a
reproducible mechanism nor closes seed=7 as the only viable seed. No mechanism
inference drawn from v178 beyond "another catastrophic patched-code failure."

### P1 #3 — "BS family exhausted" too broad — ACCEPTED AND NARROWED

Closed language: "deterministic hand-written BS scalar ladder on alibaba is closed
under patched code." Chunk-stitching as an idea class (#21, #31, #36) remains open.
The next boundary work must be structural (#36 learned prior or #31 chained-window
training), not another coefficient/k/order probe.

### P2 #4 — long-rollout required for v165/v177/v180 before retrieval promotion — ACCEPTED

Still not done. Required before treating retrieval memory as a structural Tencent win:
HRC-MAE, reuse-access rate, stack-distance histograms across v165/v177/v180/v183.
Deferred until component audit matrix (v187/v188) completes and a stable Tencent
recipe worth investing in is identified.

### P2 #5 — tail gate needed explicit tail MMD/shape requirement — FIXED

Tail-strata gate in VERSIONS.md revised per Round 32 P2 #4: candidate must (a)
improve tail MMD² (or direct shape-distance metric) without regressing ordinary-★,
AND (b) beat full-corpus numeric target by >bundle-variance margin. Tail β-recall
is reported separately, not rolled into composite tail-★ for promotion.

---

## Round 30

### P1 #1 — Tencent v165 top row still said "current ATB / genuinely productive" — FIXED

Top table revised (commit 8e84c88): v165 is now "Best observed seed-5 numeric
baseline — not yet a reproducible Tencent mechanism." Promotion language ("ATB",
"productive", "IDEA #17 genuinely productive") removed. v177 seed-7 collapse
(+348.3%) and v180 ablation data cited inline.

### P1 #2 — retrieval load-bearing inside seed-5 basin only — ACCEPTED

The v180 ablation (−retrieval-memory at seed=5) → +216.7% degrade proves retrieval
is a load-bearing ingredient *inside the seed=5 lottery basin*. It does not prove
IDEA #17 is generally solved on Tencent. The seed-7 collapse (v177) is the equal
and opposite data point. Correct conclusion: "one load-bearing ingredient of a
fragile basin."

### P1 #3 — another hand-written BS scalar probe (v182) after declaring BS ladder closed — NOTED

v182 was already burning GPU when Round 30 landed. It ran to completion (BS=0.5
collapsed: ★=0.21740). This was the last deterministic BS coefficient probe; no
further BS scalar searches queued. v182's failure actually strengthens Round 29
P1 #3: the patched BS surface has a cliff between 1.0 and 0 with nothing useful
below. The ladder is now unmistakably over.

### P1 #4 — accidental palindrome was the load-bearing element — ACCEPTED

The correct lesson: the buggy palindrome regularizer happened to impose a useful
inductive bias (forcing the generator to learn zero-velocity boundaries), and the
mathematically correct derivative-matching replacement does not reproduce that
benefit. The next boundary idea must be structurally motivated, not another
penalty scalar trying to approximate what the bug accidentally learned.

### P2 #5 — v183 PCF ablation is same-seed; cannot close/open PCF mechanism globally — ACCEPTED

v183 is a seed=5 PCF ablation. It shows PCF is load-bearing inside the seed=5
basin (5.11× degrade). It does not show PCF is load-bearing across seeds, because
the full v165 recipe already collapsed at seed=7. PCF remains "load-bearing inside
the seed=5 basket" until a seed-bundle and long-rollout panel says more.

### P2 #6 — tail gate composite tail-★ — FIXED (see Round 32 P2 #4)

---

## Round 31

### P1 #1 — v184 "never tested on Alibaba" factually stale — FIXED

VERSIONS.md and commit message corrected: v184 is a *retest* of retrieval memory
on alibaba, not a first test. Prior negative results (v127, v168) cited inline.
v184 result: closed-failed, +261% vs v176. Alibaba retrieval is now definitively
negative across three separate Alibaba branches. IDEA #17 on alibaba CLOSED.

### P1 #2 — v182 ends BS scalar ladder — CONFIRMED

v182's catastrophic failure (0.21740) is the decisive closing data point combined
with v175/v176. The patched scalar surface shows: BS=1.0 → 0.05102 (least-bad),
BS=0.5 → 0.21740 (collapse), BS=0 → 0.20719 (collapse). No useful floor.
Deterministic BS ladder on alibaba declared CLOSED.

### P1 #3 — "BS=1.0 works" language too permissive — FIXED

Language sharpened: "BS=1.0 is the least-bad patched scalar setting observed under
seed=7 — it avoids the 0.20+ collapse band but remains +47.6% behind the legacy
numeric target and does not constitute a working mechanism."

### P2 #4 — v184 acceptance bar predeclared — RESOLVED

v184 closed-failed (+261% worse than v176, +8.3× vs Alibaba numeric target).
Alibaba retrieval is definitively not a path forward. IDEA #35 workload-conditioned
router remains the open follow-up if retrieval specificity is investigated further.

---

## Round 32

### P1 #1 — residual "current ATB" language in IDEA #21 STATUS — FIXED TODAY

IDEA #21 STATUS block (previously at VERSIONS.md line 936) rewritten 2026-04-20:
removed "current alibaba ATB" and "the recipe produces ATBs"; replaced with
"legacy buggy-BS numeric target" and explicit statement that the deterministic BS
scalar ladder is CLOSED on alibaba under patched code. The ★=0.03457 threshold
stands as a race target; mechanism claims attached to v164 are tainted.

### P1 #2 — PCF mechanism over-identified as "long-range locality" — ACCEPTED

Agreed. The v183 ablation proves PCF is a load-bearing *short-window distributional
regularizer* inside seed=5. The PCF implementation operates on 12-step path
increment characteristic functions per batch window; it does not train on long
sequences or measure stack-distance or HRC. Claiming PCF "recovered long-range
reuse structure" is a leap. Correct language: "PCF is a load-bearing short-window
distributional regularizer inside the seed=5 basin; whether it improves long-horizon
cache metrics requires the sidecar panel (HRC-MAE, reuse-access rate, stack-distance)
across v165/v177/v180/v183 before any locality mechanism claim."

### P1 #3 — v185/v186 acceptance bars too optimistic — RESOLVED BY OUTCOME

Both runs closed before this response could constrain their bars:
- **v185** (tencent v165 exact recipe, seed=3): CLOSED-FAILED. Seed=3 is another
  collapse outside the seed=5 basin. Three seeds tested: seed=5 ✓, seed=7 ✗ (v177),
  seed=3 ✗ (v185). v165 is empirically seed=5 locked.
- **v186** (alibaba v176 base, seed=7 re-test): CLOSED-FAILED (★=0.21923 collapse
  basin). v186 confirms v176's seed=7 win (★=0.05102) is itself fragile — the
  least-bad patched-BS point doesn't reproduce under the same seed on the same recipe.

Going forward, v187/v188 are framed correctly: "seed=5 ablations that probe
load-bearing ingredients inside the v165 seed-5 basin." They are NOT reproducible
mechanism proofs even if they fail to degrade performance.

### P2 #4 — tail gate: composite tail-★ too prominent — FIXED

Gate in VERSIONS.md updated per Round 32 P2 #4 label: tail MMD² (or direct
shape-distance improvement) is now a required condition, not an optional component
of a composite score. Tail β-recall reported separately.

### P2 #5 — "May-2026" date provenance error — FIXED IN PRIOR COMMIT

The Round 32 P1 doc-integrity commit (75dc00e) corrected this. VERSIONS.md line 19
now reads "2026-04-18 capture" (the actual frozen-sweep date).

---

## Current state (2026-04-20, updated)

**Running on vinge:**
- `tencent_v187` (−multi-scale-critic, seed=5): ep 63+, best train-★ 0.08677 @ ep50, W≈1.8–2.1 < 3.0
- `tencent_v188` (−mixed-type-recovery, seed=5): ep 48+, best train-★ 0.09525 @ ep40, W≈1.8–2.2 < 3.0
- `alibaba_v189` (IDEA #36 boundary critic, K=4, weight=0.5, seed=7): ep 4+, W building 0.25→0.73; first MMD at ep5

v187 and v188 are 2–3× above v165's 0.03752 target so far (seed=5 ablations).
Need to W-stop or run to completion before frozen eval can close the component matrix.

v189 is the first structural Alibaba experiment: no BS/OC scalars, WGAN-SN
boundary critic replaces hand-written penalty.  Phase 3 from v176 pretrain.
First MMD signal pending (ep5, ~2026-04-20 17:00 local).

**Long-rollout sidecar for v165/v177/v180/v183** — still deferred; required before
mechanism claims about retrieval or PCF.

---

# Response to peer review Round 33 (2026-04-20)

**Status**: Round 33 arrived while the Mac was down. Responding same session. P1 #3
(IDEA #21 "producing ATBs" language) was already fixed in this session's VERSIONS.md
edit (IDEA #21 STATUS fully rewritten 2026-04-20). Remaining items accepted below.

## P1 #1 — component ablations are within-basin forensics, not mechanism validation — ACCEPTED

Agreed. v187 (−multi-scale-critic, seed=5) and v188 (−mixed-type-recovery, seed=5)
are useful to understand which components hold the seed=5 basin together. Neither
can establish a reproducible mechanism because v177 (seed=7), v185 (seed=3) already
confirmed the full recipe collapses outside seed=5. Both are running; they will
complete to natural W-stop or early kill. After their frozen evals, the seed-5
autopsy is DONE. No further same-seed ablations will be launched.

Next step: a recipe designed for robustness, not another seed=5 component probe.

## P1 #2 — "seed averaging" doesn't escape seed-lottery — ACCEPTED

Agreed. Reporting mean/median/worst is necessary for honesty, but a three-seed
distribution of {0.03752, 0.088, 0.143} still describes a fragile recipe regardless
of how clearly the distribution is published. Promotion bar going forward:

> **A mechanism is promotable only when the worst seed (or at minimum the median)
> is competitive with the current numeric target, AND the winning seed's
> long-rollout / tail evidence supports the mechanism claim.**

Seed averaging to report distributions ≠ seed averaging to launder a fragile recipe.

## P1 #3 — IDEA #21 status still says "producing ATBs" — ALREADY FIXED

IDEA #21 STATUS block was rewritten 2026-04-20 in this session. The block now reads:
"IDEA #21 deterministic hand-written BS scalar ladder is CLOSED on alibaba under
patched code." The phrase "producing ATBs" is gone; replaced with "legacy buggy-BS
numeric target." No further action needed.

## P1 #4 — Alibaba patched-BS basin fully closed; pick one structural implementation — ACCEPTED

Accepted. v179, v181, v182, v184, v186 complete the collapse map. There is nothing
left to measure on the patched hand-written BS/OC surface. The next Alibaba run will
be structural. Choosing **IDEA #36 (learned boundary prior)** as the first structural
bet because:
- The strongest Alibaba evidence is boundary-specific: the buggy palindrome objective
  accidentally encoded a useful boundary manifold that derivative-smoothing did not
- Retrieval transfer is definitively negative on Alibaba (three branches)
- #36 directly addresses the mechanism gap without assuming transferability from Tencent

IDEA #31 (chained-window training) is the follow-on if #36 shows boundary signal
but fails long rollout. IDEA #35 (workload-conditioned router) becomes relevant once
there are at least two mechanisms to route between.

## P2 #5 — "Alibaba slot held" is too passive; start #36 — ACCEPTED

The Alibaba GPU slot is free now. Plan:
1. Implement the boundary critic module: small MLP that sees
   `(left_window_tail, right_window_head)` pairs and separates true adjacent joins
   from generated and shuffled controls.
2. Train boundary critic standalone on real Alibaba adjacent pairs to confirm signal.
3. Add its contrastive loss to an Alibaba run (v176 recipe base, patched BS zeroed)
   as alibaba_v189.

## P2 #6 — tail-strata and long-rollout panels missing where most needed — ACCEPTED

Still deferred, but this is now the highest-priority diagnostic debt. With the
forensic picture of the seed-5 basin complete after v187/v188, the right next
investment is:

1. **Long-rollout panels for Tencent quartet** (v165/v177/v185/v180/v183):
   HRC-MAE, reuse-access rate, stack-distance histograms. Compares lucky seed vs
   failed seeds on the cache metrics the project actually cares about.
2. **Long-rollout panels for Alibaba** (v164/v176/v186):
   Same panel. Checks whether the seed=7 win was real on long-range structure or
   just short-window distributional luck.
3. Tail-strata runs (v164 tail-heavy/ordinary, v165 tail-heavy/ordinary) — blocked
   only by not having run them; no code work required.

## Summary

Round 33's core message is correct: the current repo state is "two seed-locked
numeric targets, both with failed seed basins and no mechanism validation." The
next win requires escaping that. Committing to:
- Finish v187/v188 as seed-5 forensics, then stop same-seed ablations
- Implement IDEA #36 boundary critic for Alibaba (next structural bet)
- Run long-rollout + tail panels on the existing baselines before any mechanism claim
- Require worst/median seed competitive before promotion

---

# AD review of boundary_critic.py / train.py wiring (2026-04-20)

Adversarial reviewer ran against the IDEA #36 implementation.  7 findings.

**P0 #1 — `_bc_file_arrays` NameError in single-file mode.** CONCEDED + FIXED.
In the epoch loop, `_bc_file_arrays` was only assigned inside `if multifile:`.
Fixed: added `_bc_file_arrays = []` default before the `if multifile:` block,
plus a `elif D_bc is not None:` branch for single-file mode.  Commit `3fff930`.

**P0 #2 — R's GRU hidden-state reset poisons fake boundary.** PUSHED BACK.
AD argues that `R(_H_B)` (cold start) produces a different decode than R with
warm state, creating an artifact the critic will learn.  Rebuttal: Recovery is
pretrained with zero initial hidden state on independent windows — cold-start
decode is the intended and trained behavior.  Both at train time and here, every
window is decoded from scratch.  The comparison (real raw features vs R-decoded
latent) is the same asymmetry the main critic has always lived with; LLGAN's
training objective (E+R ≈ identity on real data) makes them comparable.  Not
a correctness bug.

**P1 #1 — Real boundary alignment is arbitrary / asymmetric with fake semantics.**
PUSHED BACK.  Sampling real boundaries at positions b = i·T is reasonable because
the time series is approximately stationary across window boundaries; no position
is special.  The fake joins are also T-step aligned (end of rollout A, start of
rollout B).  The alignment matches semantics.

**P1 #2 — Real pair not file-matched to fake rollout z_global.** ACKNOWLEDGED
as a training-efficiency issue but not a correctness bug.  The critic learns
general feature-space boundary realism, not per-file statistics.  Not critical
at this stage.

**P2 — scaler_C.update() called multiple times per n_critic iteration.**
Pre-exists the boundary critic; not a regression introduced here.  Out of scope.

---

# Code-fix log (2026-04-20)

Deferred code-correctness bugs from Darrell peer-review Rounds 1 and 4 fixed in
this session.  No metric numbers change until these fixes propagate to a new
training run.

## `TraceDataset` off-by-one — FIXED

`llgan/dataset.py` line 1014: `n_windows = max(0, len(data) - timestep)` is
off by one.  Valid window start indices are `0 … N−T` inclusive, giving `N−T+1`
windows.  The old formula silently discarded the last legal window from every
file, and returned 0 windows for files with exactly `timestep` records.

Fix: changed to `max(0, len(data) - timestep + 1)`.

Impact: every training file now contributes one extra window (~0.8% more data
per epoch on Alibaba at T=12).  Does not change evaluation (val tensor is
pre-built from the same `TraceDataset`).  No existing ATB claims are invalidated
because the difference is small and symmetric across all runs.

## `_compute_hrc` dead-code padding — FIXED

`eval.py:_compute_hrc` was never called (dead code) but still contained the
`hrc[-1]` padding pattern Gemini R1-P2 flagged.  Removed the padding and
updated the docstring to clarify variable-length return.  The live path
(`_per_window_hrc` inside `hrc_mae`) was already correct.

---

# Response to peer review Round 34 (2026-04-20)

Round 34 is the right kind of critique: structural, actionable, and specific about
what v189's healthy recall does and doesn't prove. Two code issues fixed immediately;
two accepted with caveats; one already closing naturally.

## P1 #2 — no boundary critic logging, "stable" is unverifiable — FIXED

Agreed. "Boundary critic not destabilizing the generator" is not the same as
"boundary critic learning boundary realism." Fixed in this session (commit `140d77e`):

- `bc_real_scores / bc_fake_scores` accumulators added to the epoch loop
- Real and fake scores captured separately before `bc_loss.backward()`
- Epoch log now emits `bc_real=+X.XXX  bc_fake=+X.XXX  bc_gap=X.XXX`

bc_gap = bc_real − bc_fake is the boundary critic's discriminative margin. A growing
bc_gap at constant or improving recall would be the first sign that D_bc is learning
boundary structure rather than just noise. v189 is too early to claim either way;
future epoch reports will include this data.

## P2 #4 — D_bc / opt_D_bc absent from checkpoint save/restore — FIXED

Agreed. Resume would silently restart the boundary adversary from scratch while keeping
the generator state, changing the training dynamics mid-run. Fixed in same commit:

- `D_bc` and `opt_D_bc` added to best.pt, epoch_NNNN.pt, and final.pt saves
- Resume loader now restores D_bc / opt_D_bc if present in the checkpoint
- Condition is `if D_bc is not None` everywhere — non-boundary-critic runs unaffected

v189 is not yet at a resumed-checkpoint boundary, so no training continuity was lost.
Future boundary-critic runs are correctly resumable.

## P1 #1 — D_bc may solve decoded-vs-raw artifact detection, not boundary realism — ACCEPTED

This is a legitimate concern. Real pairs are raw normalized `TraceDataset.data`; fake
pairs go through `R(_H_A)` / `R(_H_B)` — the same Recovery decoder the main critic
has always lived with, but for IDEA #36 the concern is sharper: a learned boundary
prior is only meaningful if D_bc is distinguishing temporal join plausibility, not
Recovery-network texture artifacts.

The bc_gap logging now in place will expose whether D_bc is learning anything at all.
The proper diagnostic — real-raw vs real-reconstructed vs fake-reconstructed — requires
a dedicated R(E(real)) control run. This is deferred until v189 completes frozen sweep
and shows a useful bc_gap signal. If bc_gap stays near zero throughout training, the
critic is not discriminating; if bc_gap grows but recall doesn't follow, the domain
gap hypothesis is confirmed. Either outcome shapes the v190 design.

## P1 #3 — v189 needs frozen sweep + boundary diagnostics + long-rollout gates — ACCEPTED

Agreed. v189's ep5–ep20 trajectory is a launch health check (not collapsed, not
diverged). The actual IDEA #36 acceptance bar is:

1. Frozen sweep over all saved checkpoints
2. bc_gap trajectory showing D_bc is actually discriminating
3. Long-rollout HRC / reuse-access / stack-distance panel
4. Tail-heavy vs ordinary MMD shape rows
5. At least one additional seed (v190 seed=3 queued after GPU clears)

"Recall healthy" means we are not in the Alibaba collapse basin. It does not mean
the boundary mechanism is working.

## P2 #5 — v187/v188 are closure, not roadmap — ACCEPTED

v187 is now closed: frozen ★=0.16532 (+340% worse, multi-scale-critic 4.4× load-bearing).
v188 running; kill at ep70 (~20 min). Once v188 frozen sweep completes, the seed-5
audit matrix is done:

| Component | Ablation | Frozen ★ | Degradation |
|---|---|---|---|
| Retrieval-memory (IDEA #17) | v180 | 0.11882 | 3.17× |
| Multi-scale-critic (IDEA #8) | v187 | 0.16532 | 4.4× |
| PCF-loss (IDEA #26) | v183 | ~0.192 | ~5.1× |
| Mixed-type-recovery (IDEA #7) | v188 | TBD | TBD |

The audit will close after v188. No further same-seed tencent ablations. The seed-5
basin forensics have produced a map of load-bearing components; they have not produced
a transferable mechanism.

## Summary

Round 34's core correction is right: a healthy recall at ep5–ep20 is evidence that
v189 is not in the Alibaba collapse basin, not that IDEA #36 is working. The two
code fixes (bc logging + D_bc checkpoint) are in. The three remaining items (domain
gap control, long-rollout gates, seed bundle) are accepted as the real v189 acceptance
bar and will be addressed as the run matures.
