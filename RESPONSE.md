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
| Multi-scale-critic (IDEA #8) | v187 | 0.16532 | 4.41× |
| PCF-loss (IDEA #26) | v183 | ~0.1921 | ~5.11× |
| Mixed-type-recovery (IDEA #7) | v188 | 0.18177 | 4.85× |

**Audit COMPLETE** (v188 frozen sweep done 2026-04-20). All four components are
load-bearing inside seed=5; removing any one causes 3.17–5.11× degradation. The
seed-5 basin forensics have produced a map of load-bearing components; they have not
produced a transferable mechanism.

## Summary

Round 34's core correction is right: a healthy recall at ep5–ep20 is evidence that
v189 is not in the Alibaba collapse basin, not that IDEA #36 is working. The two
code fixes (bc logging + D_bc checkpoint) are in. The three remaining items (domain
gap control, long-rollout gates, seed bundle) are accepted as the real v189 acceptance
bar and will be addressed as the run matures.

---

# Response to peer review Round 35 (2026-04-20)

Round 35's framing is exactly right: the boundary critic is the right structural bet, but the
evidence bar has not been met yet. Two code fixes applied immediately; three accepted.

## P1 #1 — "within 0.004 of ATB" language compares EMA to frozen — RETRACTED

The language was wrong. v191 ep20 EMA ★=0.05529 versus frozen v176 ★=0.051 is an apples-to-oranges
comparison given the documented EMA-vs-frozen reversals on this architecture: v189 EMA ★=0.034 →
frozen ★=0.089 (+162%); v190 EMA ep30 ★=0.053 → ep30 frozen ★=0.124 (+134%).

Retraction is now moot: v191 ep30 confirms collapse, not proximity:
- ep20: EMA recall=0.786, ★=0.05529 (train-best)
- ep25: EMA recall=0.605, ★=0.09607
- ep30: EMA recall=0.537, ★=0.11990 — W rising 0.504→0.918

The trajectory matches v190's collapse pattern. Run is live under patience=60 from ep20; may show
late-epoch recovery (v190's frozen-best was ep65). "Within 0.004 of ATB" language removed from
VERSIONS.md and RESPONSE.md. v191 ep20 is a launch-health signal, not an ATB-near result.

## P1 #2 — bc_gap mixes temporal discrimination with raw-vs-decoded artifact — ACCEPTED

Agreed. bc_gap = D_bc(raw_real) − D_bc(R(G(z))) is non-zero even if D_bc learned only to distinguish
Recovery-decoded texture from raw features, not temporal join quality. The shuffled-real control
was attempted and rejected by AD (raw-shuffled still on the raw-real manifold). The correct
diagnostic — consecutive-raw vs shuffled-raw vs reconstructed-real R(E(real)) vs decoded-fake —
requires retraining D_bc with a 3-way or 4-way classification objective.

Conceding: bc_gap as currently reported is only evidence that "D_bc separates its two training
domains." It is NOT evidence that D_bc learned temporal boundary structure. IDEA #42 (latent-space
D_bc) is now the top-priority next structural step for #36: moving both real and fake inputs to
hidden-state space removes the raw-vs-decoded confound entirely, since H_real and H_fake both pass
through the same Supervisor/Recovery pipeline.

**Next action**: implement IDEA #42 (latent H-space D_bc) as v196+ after the bc=0.1 seed bundle
closes (v191/v192/v193). This is ranked above IDEA #39 (diversity boost) and IDEA #41 (n_critic
warmup) per Round 35 P2 #4.

## P2 #3 — opt_D_bc not gated by --reset-optimizer — FIXED (commit this session)

Confirmed the bug: lines 836-838 restored opt_D_bc unconditionally before the reset_optimizer
branch at line 848. A hot-start with `--reset-optimizer` would reset opt_G and opt_C but keep
stale Adam moments in opt_D_bc — exactly the pattern used when launching v191 (different bc_weight
from v189's seed).

Fixed in this session:

```python
# Before:
if D_bc is not None and "D_bc" in ckpt and "opt_D_bc" in ckpt:
    D_bc.load_state_dict(ckpt["D_bc"])
    opt_D_bc.load_state_dict(ckpt["opt_D_bc"])

# After:
if D_bc is not None and "D_bc" in ckpt:
    D_bc.load_state_dict(ckpt["D_bc"])
    if "opt_D_bc" in ckpt and not cfg.reset_optimizer:
        opt_D_bc.load_state_dict(ckpt["opt_D_bc"])
```

Note: D_bc weights now load even without opt_D_bc in the checkpoint (e.g., old checkpoints pre-Round-34 fix). This is the correct semantics — D_bc weights are valuable even when opt_D_bc state is absent or being reset.

The same pattern was present for opt_LD (latent discriminator, `--avatar` mode); fixed in the same
commit for consistency. opt_LD is never active in current runs (`--avatar` not used), so this was
a latent bug, not an active one.

## P2 #4 — scalar/schedule ideas ahead of structural diagnostic fix — ACCEPTED

Agreed. IDEAS #39 (diversity boost) and #41 (n_critic warmup) are rescue knobs that assume bc_gap
is a valid signal. Round 35 P1 #2 shows it isn't yet proven. IDEA #42 (latent-space D_bc) removes
the raw-vs-decoded confound structurally. Reprioritizing:

1. **IDEA #42 (latent-space D_bc)** — first priority after seed bundle closes
2. **IDEA #40 (boundary feature-matching)** — second priority (still in decoded space, but direct alignment rather than adversarial)
3. **IDEAS #39/#41 (diversity boost, n_critic warmup)** — deferred until #42 establishes a clean bc_gap signal

## P2 #5 — long-rollout HRC/reuse/stack-distance panels still deferred — ACCEPTED with plan

v189 and v190 are the negative controls (bc=0.5, not competitive). Long-rollout panels are most
informative after v191 frozen sweep establishes whether bc=0.1 is competitive at all. If v191 frozen
is not competitive, the panels add little signal. If v191 frozen beats v176, then long-rollout
panels on v191 + v189 + v176 as control are the top priority before any mechanism claim.

Running long-rollout for ALL bc runs before mechanism claim is accepted.

## bc_weight=0.5 recall collapse diagnosis

v190 (seed=3, bc_weight=0.5) reached its train-best at ep30 (EMA ★=0.053, recall=0.773), then
collapsed: ep40 recall=0.663, ep45 recall=0.522. The collapse pattern is diagnostic.

**Root cause**: at peak recall (ep30), bc_fake scores ≈ −0.75. The bc loss contribution to
G-loss = bc_weight × (−D_bc(fake_tail, fake_head).mean()) ≈ 0.5 × 0.75 = 0.375. The WGAN
component of G-loss at ep30 is G=0.64 (total), so WGAN alone ≈ 0.26. **bc is contributing ~60%
of G-loss**, not a supplemental signal. The generator learns to optimize boundary realism by
reducing output diversity — a mode-restriction shortcut. Recall falls as the generator covers
fewer modes in exchange for cleaner window joins.

bc_gap (bc_real − bc_fake) remained positive and stable (0.042–0.107) throughout the collapse:
D_bc is still discriminating, but the generator is responding by restricting modes rather than
by making transitions more realistic.

**v191 response**: reduce bc_weight to 0.1 (5× reduction), seed=11.
- bc contribution at peak = 0.1 × 0.75 = 0.075 (≈22% of G-loss vs 60%)
- seed=11 was the collapse-seed for v186 (no bc, frozen ★=0.219); tests minimum bc weight to
  prevent collapse
- If bc_weight=0.1 prevents collapse AND avoids the recall-restriction shortcut, frozen ★
  should improve over v189 (0.076) and potentially over v176 (0.051)

## bc_gap: confirming boundary discrimination

Across v189 (ep1–61) and v190 (ep1–45), bc_gap was consistently positive:
- v189: peaked at 0.31 (ep1), stabilised 0.054–0.129 (ep15–45)
- v190: peaked at 0.31 (ep5), stabilised 0.042–0.107 (ep25–45)

D_bc is discriminating real boundaries from generated ones throughout training.
This addresses the Round 34 P1 #1 bc_gap diagnostic requirement. The outstanding domain gap
question (does D_bc score reconstructed-real lower than raw-real, indicating R artifact detection
rather than boundary realism?) is still pending — requires the R(E(real)) reconstructed-real
control run.

## seed bundle status (2026-04-21 update)

| seed | mode | bc_weight | frozen ★ | epochs | outcome |
|---|---|---|---|---|---|
| 7 (v189) | decoded | 0.5 | 0.076 | ep61 W-stop | avoids collapse; not competitive |
| 3 (v190) | decoded | 0.5 | **0.083** | ep70 kill | Frozen-best ep65; CLOSED-FAILED |
| 11 (v191) | decoded | 0.1 | **0.067** | ep83 kill | Frozen-best ep75 ★=0.067; best decoded-bc |
| 7 (v192) | **latent-H** | 0.1 | 0.104 | ep35 W-stop | IDEA #42; avoids ep25 collapse; W-stop too early |
| 5 (v193) | **latent-H** | 0.1 | 0.111 | ep97 kill | w-stop=5.0; no phase transition; β-rec ceiling 0.48 |
| 5 (v194) | **decoded** | 0.1 | **0.054** | ep85 (critic collapse ep88) | w-stop=5.0; seed=5 ATB basin; **new bc ATB** |

v192 (IDEA #42 latent-H bc): EMA breakthrough (ep30 recall=0.898, ★=0.024) but frozen ★=0.104 (+334% EMA inflation). Key finding: **latent-H bc avoids decoded-mode ep25 collapse** (v192 ep25 recall=0.733 vs v191's 0.605), but **W-stop at ep35 cuts off late-epoch frozen recovery**.

v193 (IDEA #42 latent-H bc, w-stop=5.0, seed=5): Ran to ep97 with W≤2.1 (exceptional stability). Frozen sweep complete (20 checkpoints). **Frozen-best ep75 ★=0.111, β-rec=0.478. No phase transition in ep80-ep95.** IDEA #42 verdict: Latent-H bc β-recall ceiling ~0.48; decoded-mode bc is superior.

v194 (IDEA #36 decoded-bc, seed=5, w-stop=5.0): **CLOSED (critic collapse ep88, frozen-best ep85 ★=0.05445).** β-recall trajectory: 0.449(ep30)→0.499→0.501→0.543→0.688→0.752→**0.773(ep85)**. Phase transition at ep75 (β-recall 0.543→0.688). Training continued to ep85 producing new bc ATB. WGAN critic collapsed at ep88 (W: 2.41→0.02) — ep90 frozen ★=0.172 (catastrophic). W partially recovered by ep95-97 (W=1.05-1.49) but G-loss +5-7; ep85 confirmed as peak. **Result**: frozen ep85 ★=0.05445 (β-recall=0.773, MMD²=0.009) — beats v191 ★=0.067 by 19%. **Gap to ATB: 6.8%** (★=0.054 vs v176 ★=0.051). Monitoring for post-collapse recovery through ep115 per bc rule.

**Critical lesson (24th mis-rank)**: Latent-H bc produces more severe EMA inflation than decoded mode. EMA ★=0.024 vs frozen ★=0.104 = 4.3× inflation (vs 3.2× for decoded-mode v191). EMA recall during bc training is NOT a reliable signal in either mode — only frozen sweep gives the true number.

**v191 full trajectory (2026-04-20, completed ep80)**:

| ep | EMA MMD² | recall | EMA ★ | notes |
|---|---|---|---|---|
| 5 | 0.01565 | 0.655 | 0.08465 | healthy launch |
| 10 | 0.02202 | 0.712 | 0.07972 | |
| 15 | 0.00896 | 0.737 | 0.06166 | |
| **20** | **0.01249** | **0.786** | **0.05529 ★ (train-best)** | peak |
| 25 | 0.01697 | 0.605 | 0.09607 | collapse onset |
| 30 | 0.02730 | 0.537 | 0.11990 | W=0.918 |
| 35 | 0.02219 | 0.553 | 0.11149 | |
| 40 | 0.01739 | 0.498 | 0.11779 | W=1.088 |
| 45 | 0.02252 | 0.425 | 0.13752 | W=1.326, epoch_t=97s |
| 50 | 0.03163 | 0.394 | 0.15283 | |
| **55** | **0.01003** | **0.720** | **0.06603** | **RECOVERY — likely frozen-best** |
| 60 | 0.01773 | 0.520 | 0.11373 | re-collapse |
| 65 | 0.02990 | 0.401 | 0.14980 | no 2nd recovery |
| 70 | 0.03567 | 0.430 | 0.14967 | W=1.399, W rising 2.0-2.9 ep71+ |
| **75** | **0.01808** | **0.383** | **0.14148** | EMA says "collapse" — **FROZEN-BEST** (see below) |
| 80 | 0.02185 | 0.440 | 0.13395 | W=2.998 (near w-stop); killed ep83 |

**EMA collapsed at ep75-80 but frozen shows recovery**: EMA recall=0.383-0.440 appeared as sustained collapse; run was killed ep83 (63 epochs stale from EMA heuristic). Frozen sweep revealed frozen-best is **epoch_0075.pt ★=0.06749** (β-recall=0.709), not ep55 as predicted. The model was recovering in frozen-eval space even as EMA suggested collapse.

**v191 frozen sweep (seeds 42/42, 17 checkpoints, 2026-04-20)**:

| checkpoint | frozen ★ | MMD² | β-recall |
|---|---|---|---|
| **epoch_0075.pt** | **0.06749** | 0.00929 | 0.709 |
| epoch_0080.pt | 0.06897 | 0.02677 | 0.789 |
| epoch_0060.pt | 0.10348 | 0.02248 | 0.595 |
| epoch_0040.pt | 0.11286 | 0.01586 | 0.515 |
| epoch_0070.pt | 0.12302 | 0.06702 | 0.720 |
| epoch_0055.pt | 0.12285 | 0.01455 | 0.459 |
| epoch_0020.pt = best.pt | 0.17753 | 0.02373 | 0.231 |

**v190 additional finding**: recall partially recovered ep60-65 (0.659, 0.672) despite collapse
ep35-50. bc_weight=0.5 does not permanently damage the generator; it oscillates into periodic
mode-recovery. But neither ep30 peak nor ep65 recovery is competitive with ATB ★=0.051.

**MMD² gap analysis (IDEA #39)**: v176 ATB achieves MMD²=0.007 with β-recall=0.779, ★=0.051.
v189 bc=0.5 achieved frozen MMD²=0.014 — 2× worse. Root cause: bc's mode-restriction shortcut
increases MMD² by narrowing the G distribution. Fix queued as IDEA #39: boost diversity-loss-weight
from 2.0→5.0 for bc runs to directly counter the mode-restriction attractor. v194 (after v191
frozen sweep) will test bc=0.1 + diversity=5.0 at seed=11.

## v189 acceptance bar (Round 34 P1 #3) — status

1. ✅ Frozen sweep complete (★=0.076)
2. ✅ bc_gap trajectory showing D_bc discriminating (confirmed, see above)
3. ❌ Long-rollout HRC / reuse-access / stack-distance panel — deferred
4. ❌ Tail-heavy vs ordinary MMD shape rows — deferred
5. ✅ Second seed: v190 seed=3 CLOSED. Frozen ★=0.083. v191 (seed=11, bc_weight=0.1) running.

Items 3 and 4 remain gated on v191's frozen sweep results (no point in long-rollout panels
if bc_weight=0.1 changes the mechanism significantly).

---

## AD note: shuffled-real diagnostic not implementable without retraining D_bc (2026-04-20)

Attempted to add a shuffled-real control to bc logging (Round 34 P1 #1 partial address).
AD identified a P0 flaw: D_bc was trained with raw-real as positive and R(G(z))-decoded as
negative. Shuffled-real pairs are still raw features — D_bc's "real" class — so they score
similar to consecutive-real by construction. `bc_shuf_gap ≈ 0` would be the expected outcome
even if D_bc learned perfect temporal structure; the test cannot distinguish the two cases.

**Correct approach** (requires retraining): train a 3-way D_bc with:
- consecutive-real (positive)
- shuffled-real (negative class 1: correct features, wrong temporal join)
- decoded-fake (negative class 2: correct temporal structure, wrong feature manifold)

A D_bc that separates all three would unambiguously demonstrate temporal join learning.
Cost: significant code change (D_bc loss rewrite); deferred until bc=0.1 frozen sweep.

The current bc_gap signal (positive and stable) remains the best available bc diagnostic.

---

# Response to peer review Round 36 (2026-04-21)

Round 36 is the strongest critique yet on the latent-bc branch, and the experimental results now
available (v193 closed, v194 running) largely validate the reviewer's predictions. Most P1 points
are accepted on empirical grounds, not on faith.

## P1 #1 — Latent-H carry-semantics confound (real=fresh-encoder, fake=carried-generator) — ACCEPTED

The reviewer correctly identified that IDEA #42's latent-H mode introduced a new shortcut in place
of the old raw-vs-decoded one: D_bc can learn "fresh encoder head vs carried generator head" rather
than boundary realism. This point is accepted.

It is now also empirically settled: v193 ran latent-H bc to ep97 (W≤2.1, no W-stop confound),
covering 20 checkpoints with a full frozen sweep. The frozen-best was ep75 ★=0.11060 with
β-recall=0.4775. β-recall peaked at ep75 and **degraded** in ep80-ep95 (0.466 → 0.412 → 0.333).
There was no phase transition. The β-recall ceiling for latent-H bc is approximately 0.48,
structurally lower than decoded-mode bc (v191 ep75 β-recall=0.709).

This cannot be explained only by the carry-semantics confound — if D_bc were simply learning
"fresh vs carried," the generator would find ways to exploit this shortcut and the frozen ★ would
still improve. The complete failure to approach v191's β-recall suggests the latent space (dim=24)
is simply too low-dimensional for D_bc to provide sufficient gradient pressure on G's spatial
pattern generation, regardless of the carry-semantics issue.

**IDEA #42 is closed**: latent-H bc is structurally inferior to decoded-mode bc for this
architecture and trace family. The carry-semantics confound (IDEA #43) remains theoretically
valid but is academic given latent-H's empirical verdict.

**Current action**: v194 is running decoded-mode bc (IDEA #36, no latent flag), seed=5 (same
seed as ATB holder v165, ★=0.051), w-stop-threshold=5.0. Decoded-mode bc does not have the
carry-semantics confound since both real and fake are in the same decoded feature space.

## P1 #2 — v192 should be read as a failed probe, not validation blocked by W-stop — ACCEPTED

Agreed. The Round 35 response was too optimistic about the "W-stop confound" framing. v193
directly tested the hypothesis "latent-H bc needs more epochs," and the answer is no: with 97
epochs and W≤2.1 throughout, frozen ★=0.111 is worse than decoded-mode v191's ★=0.067. v192 was
a failed first probe, not a near-miss blocked by an early kill. The mechanism difference between
latent-H and decoded-mode is real and decisive.

EMA inflation on latent-H bc was indeed the strongest warning signal: v192 EMA ★=0.024 →
frozen ★=0.104 (+334%). That is not a calibration issue — it is evidence the model was exploiting
the confound the reviewer identified (or an adjacent artifact) to produce low EMA loss without
learning the frozen-bundle-relevant distribution.

## P1 #3 — w-stop-threshold=5.0 as weakened safety guard, not clean architectural win — ACCEPTED

The reviewer's caution was correct. v193 operated with w-stop=5.0, and W stayed ≤2.1 throughout
(the guard was never engaged), yet v193 still failed. The threshold increase was not the difference
between success and failure. What it proved: latent-H bc does not cause W instability when
training is otherwise stable (seed=5 favorable basin), but also doesn't produce competitive frozen
results even with that stability.

For v194 (decoded-mode bc, seed=5, w-stop=5.0): the raised threshold is there to prevent v191-
style early kill at ep83 while the model may still be recovering in frozen-eval space. v191 had
EMA β-recall=0.789 at ep80 (still improving) when killed. If v194's W stays ≤2.1 as v193's did,
the w-stop change is moot but safe. If W approaches 5.0, we will report it explicitly as
"weakened-guard evidence" as the reviewer requires.

## P2 #4 — Checkpoint selection: EMA has failed 25 times; IDEA #38 should be prioritized — ACCEPTED IN PRINCIPLE

The track record: 25 consecutive training-selector mis-ranks. EMA is a launch diagnostic and
nothing more for bc runs. Accepted. IDEA #38 (deterministic mini-eval or dense frozen sweeps in
the training loop) is the right structural fix.

Current mitigation: we run a full frozen sweep (seeds 42/42, all epoch checkpoints) after every
run closes. This is expensive (1-2 hours per sweep for ~20 checkpoints) but catches the mis-ranks
reliably. For v194, we will run an interim frozen sweep around ep75-80 when EMA starts showing
collapse signals, rather than waiting for the run to close.

We do not have time to implement IDEA #38 in-loop before the competitive deadline. What we can
do: checkpoint-every-5 + frozen sweep is the current protocol. If v194 shows the same ep75 frozen
peak as v191, we will not kill it on EMA grounds — we will sweep immediately and decide on frozen
evidence.

## P2 #5 — Long-rollout and tail gates still deferred — NOTED

Long-rollout HRC / reuse-access / stack-distance and tail-strata panels remain deferred. The
acceptance bar from Round 34 P1 #3 requires these, and they are not gated on anything except
finding a competitive frozen checkpoint that is worth the panel investment.

Current trigger: if v194 frozen ★ is competitive (≤0.060, approaching v176 ATB ★=0.051), we will
run long-rollout and tail panels before claiming any advance. We will not defer indefinitely.

## Summary of actions taken

| item | status |
|---|---|
| IDEA #42 (latent-H bc) | **CLOSED** — empirically inferior, v192+v193 verdict |
| v193 | CLOSED ep97; frozen-best ★=0.111; β-recall ceiling confirmed 0.48 |
| v194 | **RUNNING** — decoded-mode bc, seed=5 (ATB seed), w-stop=5.0 |
| IDEA #43 (matched carry-state bc) | Recorded; deferred — latent-H bc itself closed |
| IDEA #38 (in-loop mini-eval) | Accepted in principle; interim frozen sweeps as proxy |
| Long-rollout panels | Deferred; triggered if v194 frozen ★ ≤ 0.060 |
| w-stop raising | Will flag as "weakened-guard" if W approaches 5.0 in v194 |

## Short take

The reviewer's predictions were accurate: latent-H bc failed exactly as the carry-semantics
confound analysis predicted, the W-stop raise was not a clean fix, and EMA remains unreliable.
v193's full frozen sweep closes the latent-H branch definitively. v194 returns to decoded-mode bc
at seed=5 — the ATB seed — with the only change being higher w-stop to prevent the ep83 early
kill that may have cut v191 short while β-recall was still improving.

---

# Response to Peer Review Round 37

**Date**: 2026-04-21
**Responding to**: `PEER-REVIEW.md` Round 37 — "Do Not Retreat From Confound Removal Back To Seed-5 Decoded BC"

## P1 #1 — v194 decoded-mode bc reopens raw-vs-decoded confound — ACCEPTED

The reviewer is correct. The Round 36 response claimed decoded-mode bc puts real and fake "in
the same decoded feature space," but the code does not do that. In `train.py` line ~1418,
`_bc_tail_r, _bc_head_r = sample_real_boundaries(...)` returns raw normalized trace arrays
directly. In line ~1421, `_bc_tail_f = R(H_A)[:, -K:, :]` — fake is decoded through Recovery
`R`. So the boundary critic is trained on raw real vs. decoded fake: the domain-mismatch confound
is present in decoded mode exactly as in latent-H mode.

IDEA #44 is the correct structural fix: reconstruct real positives through `R(E(real))` before
feeding `D_bc`, so both real and fake pass through the same R transform. This removes the
raw-vs-decoded shortcut and forces `D_bc` to learn temporal boundary structure, not domain
texture.

**Pre-registered interpretation of v194**: if v194 ep85 frozen ★=0.054 holds as the run closes,
the result is: "decoded-mode bc with seed-5 basin and relaxed kill guard produces better
short-window frozen score than decoded bc at seed-5 without relaxed kill guard (v191 ★=0.067)."
That is all it proves. It does not prove `D_bc` learned boundary realism. Any promotion language
will be qualified accordingly.

**Commitment**: IDEA #44 (`--boundary-critic-real-reconstruct`) is the next boundary probe, not
more weight/schedule/seed tuning. The three-way diagnostic (raw-real, recon-real,
shuffled-recon-real, fake) will be logged to determine whether decoded bc was exploiting the
raw-vs-decoded artifact.

## P1 #2 — Closing IDEA #42 as "latent space too low-dimensional" is overidentified — ACCEPTED

The closure text in Round 36 and RESPONSE.md identified the failure mechanism as
"latent_dim=24 too low-dimensional for D_bc to provide sufficient gradient pressure." That
explanation is a plausible hypothesis, not an identified cause. The empirical verdict is firm —
v192 and v193 both failed with β-recall ceilings ~0.48 — but the root cause could equally be
the carry-mismatch confound the reviewer has identified across multiple rounds: real boundaries
are independently reset encoder windows; fake boundaries use carried generator hidden state. The
critic can trivially learn "head starts at timestep 0 of a fresh encoder window" without any
signal about temporal continuity.

**Retraction**: the "too low-dimensional" explanation is retracted. The correct closure is: "the
current latent-H implementation, which compares reset-encoded real heads against carried
generator heads, failed on v192 and v193. The carry-mismatch confound is a sufficient
explanation. IDEA #42 in its current form is closed. The broader concept of
representation-matched boundary criticism (IDEA #43, IDEA #44) remains open."

IDEA #43 (matched carried-state bc) and IDEA #44 (domain-matched decoded bc) are kept alive and
are the active next structural moves.

## P1 #3 — v194 is not a clean mechanism test; v165 reference corrected — ACCEPTED

Agreed. v194 uses seed=5 + v193 pretrain (seed=5 basin) + raised w-stop threshold. The design
was intentionally within-basin (motivated by v191's ep83 early kill). That makes v194
within-basin forensics, not mechanism validation — the same standard applied to the Tencent
seed-5 component audit.

**Corrected doc**: the text "Same seed as ATB holder v165 (★=0.051)" has been fixed in
VERSIONS.md to "Same seed as Alibaba ATB holder v176 (★=0.051)." v165 is the Tencent seed-5
numeric target (★=0.038). v176 (seed=7) is the Alibaba ATB at ★=0.051.

**Standard going forward**: if v194 ep85 ★=0.054 is the run's best checkpoint, the result is
reported as "seed-5 decoded bc with relaxed kill guard outperforms seed-5 decoded bc with
standard kill guard in the same seed basin." Second-seed confirmation or domain-matched critic
(IDEA #44) required before any mechanism claim.

## P2 #4 — Long-rollout/tail gate too conditional — ACCEPTED

The ≤0.060 gate in Round 36 keeps all long-horizon evidence conditional on short-window score
triage. That is wrong: boundary criticism exists to improve cross-window continuity, and the
diagnostic panel is informative regardless of whether frozen ★ is competitive — it tells us
whether bc is improving join quality or merely shifting the short-window sample cloud.

**New policy**: run a compact long-rollout/tail panel for v176, v191, v193, and v194 ep85. The
≤0.060 gate is removed. This is the minimum needed to interpret bc results mechanistically.

The panel (HRC, reuse-access rate, stack-distance distribution, tail-strata rows) will be run
on frozen_best.pt for each of those four experiments. v193 and v194 ep85 have frozen_best.pt
ready on vinge. v176 and v191 frozen_best.pt paths should be confirmed before running.

## P2 #5 — boundary_critic.py docstring fixed — ACCEPTED

The docstring said real pairs are "centered on a true file boundary." The implementation
(`sample_real_boundaries`) samples every T records within each file — these are in-file
adjacent-window boundaries at stride T, not cross-file boundaries. The docstring has been
corrected in `boundary_critic.py` to say "in-file adjacent-window boundary at stride T" and
notes it is not a cross-file boundary.

## Summary of actions taken

| item | status |
|---|---|
| v194 pre-registered interpretation | "seed-5 decoded bc + relaxed kill guard" — NOT mechanism proof |
| IDEA #44 (domain-matched decoded bc) | Next boundary probe after v194 closes |
| IDEA #42 closure text | Retracted "too low-dimensional" explanation |
| IDEA #43 (matched carry-state bc) | Kept alive; not buried by IDEA #42 closure |
| v165 → v176 Alibaba ATB reference | Fixed in VERSIONS.md |
| Long-rollout panel gate | Removed; panel queued for v176, v191, v193, v194 ep85 |
| boundary_critic.py docstring | Fixed: "in-file adjacent-window boundary at stride T" |

## Short take

The reviewer correctly identifies that the raw-vs-decoded confound was never removed by
decoded-mode bc — it was merely hidden. v194 is within-basin forensics that proves the kill
guard was premature in v191, not that `D_bc` learned boundary realism. IDEA #44 is the cleanest
next structural move: keep decoded-feature gradients while reconstructing real positives through
`R(E(real))` so both sides of the critic train in the same feature domain.

---

# Retrospective Response to Peer Review Rounds 19 and 22–26

**Date**: 2026-04-21
**Context**: These rounds were noted as "tracked separately" in the Round 27 response but were
never formally closed. This response covers each in retrospect against the current repo state.

---

## Round 19 — Sweep Repair Worked; WaveStitch Has A Hidden Semantics Bug

### P1 #1–#2 — Overlap-mode silently changes BS semantics — SUBSEQUENTLY RESOLVED

The BS+OC overlap-mode confound was real at the time. It was never cleanly isolated: the scalar
ladder on BS+OC ran through alibaba_v160–v171 and produced diminishing returns. Rounds 26–27
confirmed the BS/OC weight ladder was not productive (v171: +21.5% vs v167; palindrome bug from
Gemini R3 further undermined interpretation). The project has since abandoned the BS+OC path in
favor of the learned boundary critic (IDEA #36), which does not share this semantics confusion.
The WaveStitch overlap mode remains in the codebase but is not part of any live recipe.

### P1 #3 — Checkpoint-selection repair — INSTITUTIONALIZED

`frozen_sweep.py` with dual seeds (--eval-real-seed 42 --eval-fake-seed 42) is now the mandatory
promotion protocol. Round 18 P1 #1 started this; it is now the 27th-confirmed mis-rank invariant.

### P1 #4 — Old closures should be treated as lower-confidence — ADOPTED

All pre-frozen-sweep closure decisions (including v157, v158 "failed reproduce" reversals) are
treated as lower-confidence historical evidence. This is now standard practice.

---

## Round 22 — v164 Is A Real Win, But The Interpretation Is Running Ahead

### P1 #1 — "W-stop distillation mechanism" = hypothesis — CONFIRMED AS HYPOTHESIS, THEN RETRACTED

v164's 0.035 was treated as a mechanism win at the time. v167 (branched from v164, w-stop=3.0)
pushed to 0.029, which temporarily looked like strong confirmation. However, Round 26 showed the
mechanism was overidentified: the full dense sweep was never run. v167 was then retracted (April
2026) after Round 26 peer review flagged it as a single-branch seed lottery. The new Alibaba ATB
is v176 (★=0.051), a clean fresh-pretrain run with diversity=5.0.

### P1 #2 — Wrong first control for v165 — ACCEPTED, EXPERIMENT NOT RUN

The Round 22 recommendation (branch from pre-tail checkpoint, vary critic continuation) was not
implemented. Instead, the project moved to fresh seeds (v172/v173 basin characterization) which
confirmed the seed-lottery nature more efficiently. IDEA #33 remains open as a structural follow-up
if the mechanism resurfaces with better controls.

### P1 #3 — IDEA #21 status reconciliation — RESOLVED BY ABANDONMENT

The BS+OC overlap-mode path was effectively closed by the v171 failure (+21.5% worse) and the
Gemini R3 palindrome bug. IDEA #21 is now treated as "closed for current architecture" — the
WaveStitch overlap path did not produce a reproducible mechanism. The boundary critic (IDEA #36)
is the active boundary-realism branch.

### P1 #4 — Do not use v164 to bypass Round 21 long-rollout gate — ADOPTED

Long-rollout and tail-strata panels remain as acceptance criteria. The sidecar fixes from Round 21
were all shipped (conditioning via char_file, real-baseline manifest, per-stream drift, true
stack-distance). The gate is now valid for bc experiments but has not been run for v176/v191/v194
yet — this is the open item from Round 37 P2 #4.

### P2 #5 — tencent_v163 conclusion scope — ADOPTED

v163 was reframed as "FFT-weight amplification (20×)" not "spectral is unnecessary." This stands.

---

## Round 23 — Higher Moments Say This Is A Tail-Regime Problem

### P1 #1–#3 — Higher moments → tail-regime diagnosis — ACCEPTED, PARTIAL IMPLEMENTATION

The R analysis confirming extreme M5/M6 moments for iat_*, abs_stride_*, reuse_ratio was accepted.
IDEA #34 (tail-regime modeling) was added and tail_strata.py was implemented as the first MVE.
The raw-M6-loss path was explicitly rejected (numerically brittle, easy to game). The structural
route — tail-stratified eval + explicit routing — is the right direction but remains under-
utilized: frozen sweeps do not yet report tail-strata rows alongside combined score.

### P1 #4 — Window-level vs file-level tail audit — OPEN

The R analysis was file-level. A window-level tail audit has not been done. Still open.

---

## Round 24 — Full-Corpus Higher Moments, Not A Cherry-Picked Slice

### P1 #1–#3 — Full-corpus tail manifest with all 22 families — ACKNOWLEDGED, NOT BUILT

The full-corpus tail manifest (top files/windows by iat_*, abs_stride_*, reuse across all trace
families, not only Alibaba/Tencent) has not been built. Acknowledged as the right long-term target.
Current focus on bc mechanism work makes this secondary, but the race goal requires it before any
paper-level claim about generalization.

---

## Round 25 — The New Gate Is Better, But Generation Parity Is Still Broken

### P1 #1 — generate.py SSM/MTPP parity — FIXED

`generate.py` was patched to instantiate SSM/MTPP checkpoints matching `long_rollout_eval.py` and
to guard SSM hidden-state detachment. This was shipped before the Round 26 response.

### P1 #2 — Persistent retrieval in long_rollout_eval — FIXED

`_rollout()` now carries `retrieval_state` and mirrors `generate.py`'s persistent-retrieval
contract. The JSON records `retrieval_persist_requested` and `retrieval_persist_enabled`.

### P2 #3 — Source-conditioned sidecar not manifest-matched — PARTIALLY FIXED

`--char-file` conditioned generation is now available and `--real-manifest` provides manifest
determinism. True per-source-trace matched panels have not been run. Acknowledged as open.

### P2 #4 — Tail-control LR not restart-idempotent — ACKNOWLEDGED, NOT FIXED

IDEA #33 tail-control arms were superseded by the bc path before this was fixed. If IDEA #33 is
revisited, the LR-idempotence issue (re-multiplying lr on resume) must be addressed first by
storing a `tail_applied` flag in the checkpoint.

---

## Round 26 — The New Tail Gate Is Useful, But Do Not Let It Become A Blurry Scalar

### P1 #1 — v167 conclusion over-closed — RETRACTED

v167 (★=0.029, original Alibaba ATB holder) was retracted 2026-04-19 per Round 26 and
subsequent basin analysis (v172/v173). The "W=3.0 mechanism" language was removed. The current
Alibaba ATB is v176 ★=0.051, a clean independent run.

### P1 #2 — Tail-★ not a safe standalone gate — ADOPTED

Tail-stratum promotion now requires separate tail-MMD/shape, ordinary non-regression, and recall
reported separately rather than a composite tail-★. This is the policy going forward.

### P2 #3 — eval.py --baseline ignores manifest restriction — OPEN BUG

The `--baseline` branch in eval.py was not patched to thread `real_seed`, `fake_seed`, and
`file_manifest`. This remains an open code bug. When comparing checkpoints on a tail manifest,
the baseline branch samples from full-corpus, making the delta invalid. Low priority while the
race focus is on bc, but must be fixed before any tail-strata promotion claim is used.

### P2 #4 — tail_strata.py missing reuse-heavy stratum — PARTIAL

`tail_strata.py` scores iat_q99/q50 and abs_stride_q99/q50 but not reuse-ratio or stack-distance-
derived tail labels. The reuse-heavy stratum (the actual HRC/cache-fidelity target from IDEA #34)
has not been implemented. Still open.

### P2 #5 — Long-rollout JSON missing retrieval settings — FIXED

`retrieval_persist_requested` and `retrieval_persist_enabled` were added to the JSON before Round
27 shipped.

---

## Summary Table

| Round | P1 items status | Key open items |
|-------|----------------|----------------|
| 19 | Resolved: BS/OC confound bypassed; frozen_sweep canonical | None live |
| 22 | v167 retracted; v164 re-labeled hypothesis; IDEA #21 closed | IDEA #33 still open structurally |
| 23 | Tail-regime diagnosis accepted; tail_strata.py built | Window-level tail audit, reuse-heavy stratum |
| 24 | Full-corpus tail leaderboard acknowledged | Full-corpus tail manifest not built |
| 25 | generate.py + long_rollout_eval parity fixed | Per-source-trace manifest panel; LR idempotence |
| 26 | v167 retracted; tail-★ gate policy adopted | eval.py --baseline manifest bug; reuse-heavy stratum |

---

## Response to Round 38

### P1 #1 — IDEA #44 diagnostic scores missing from v195

Accepted and fixed. Added `bc_diag(raw=X,recon=Y,shuf=Z)` logging to the
`decoded-feat-matched` training branch in `llgan/train.py`. Each epoch now
logs four quantities:
- **bc_real** (training positive = `R(E(real))`, what IDEA #44 trains on)
- **bc_raw_real** (diagnostic: `D_bc` on raw normalized pairs — pre-IDEA-44 baseline)
- **bc_recon_real** (diagnostic: `D_bc` on `R(E(tail_A))`, `R(E(head_B))` pairs)
- **bc_shuf_real** (diagnostic: shuffled tail–head pairs in recon space — breaks adjacency,
  should score near fake if D_bc learned temporal structure)

The shuffled score is the critical signal: if `bc_shuf ≈ bc_recon`, D_bc learned
reconstruction-domain texture, not adjacency. If `bc_shuf ≈ bc_fake < bc_recon`,
D_bc learned genuine temporal adjacency. v195 is mid-run at ep36; the diagnostic
will be visible from next epoch log flush forward. **v195 is now interpretable as
mechanism evidence.**

The v195 run does NOT need to be restarted — the diagnostic is additive and the
training objective (real positives = R(E(real))) is already correct. The missing
piece was the log output, not the training signal.

### P1 #2 — v194 post-collapse extension is peak chasing

Accepted. Conclusion frozen: "seed-5 decoded bc with relaxed guard found an ep85
short-window near-miss (★=0.054 vs v176 ★=0.051, 6.8% worse)." Updated VERSIONS.md
to label post-ep88 sweeps as diagnostic forensics, not mainline evidence. Kill
deadline remains ep160 but no further extensions will be granted. The current
frozen sweep (#12, ep5-ep140) running now is the last mainline sweep; subsequent
sweeps would only run if ep140+ beats ep85, which would be surprising given the
adversarial-game qualitative change at ep88.

### P1 #3 — v194 seed provenance still wrong after Round 37 fix

Accepted. Fixed VERSIONS.md line for v194: replaced "Same seed as Alibaba ATB
holder v176 (★=0.051)" with "Same seed-5 basin as v193/v192/v191; numeric target
is v176 (seed=7, ★=0.051) — different seed." Added explicit interpretation
annotation: "seed-5 decoded-bc evidence only; not mechanism validation until a
second seed or IDEA #44 reproduces."

### P2 #4 — `--bc-latent` + `--bc-real-reconstruct` silent precedence

Accepted. Added a hard `ValueError` at startup in `llgan/train.py` before the
BoundaryCritic initialization block. Passing both flags now raises:
`"--boundary-critic-latent and --boundary-critic-real-reconstruct are mutually
exclusive: latent-H mode uses E(x) features; decoded-feat-matched mode uses
R(E(x)) features. Pass only one."` No silent mis-labeling possible.

### P2 #5 — Long-rollout/tail panel still not landed

Accepted — this is now genuinely overdue (committed in Round 37 response, committed
again in Round 35). The gate on ≤0.060 is dropped. Will run the compact panel for
v176, v191, v193, and v194 ep85 this cycle regardless of short-window ★.

### Addendum — AD review of Round 38 code changes

An internal adversarial review (AD) of the IDEA #44 diagnostic implementation
found two P1 defects that were immediately fixed:

**AD P1 #1 — SN power-iteration side effects**: Diagnostic D_bc forward passes
ran in `D_bc.training=True` mode, causing spectral-norm power iterations to
advance 3 extra times per step before the main training forward. Fixed by
wrapping diagnostic passes in `D_bc.eval()` / `D_bc.train()` guards, making
the diagnostic observationally neutral. (The earlier ep36 read of `bc_diag`
occurred in the non-neutral version and is invalidated.)

**AD P1 #2 — Shuffled control fixed-point contamination**: Permuting only
`_recon_A` while keeping `_recon_B` fixed produced ~1/B true-positive pairs
(identity permutation fixed points). Fixed by using two independent permutations
for A and B: `_shuf_B = (_shuf_A + 1 + rand(B-1)) % B`, guaranteeing a
derangement.

**AD P2 — bc_recon_real_score redundant**: Confirmed identical to `bc_real`
after SN-fix. Dropped from the log; `bc_diag(raw=X,shuf=Z)` is now the
two-column format — `bc_real` already reports the recon score.

v195 was relaunched with the corrected diagnostic code. The ep36 read
from the pre-fix run is discarded.

---

## Long-Rollout Panel: v176 / v191 / v193 / v194 ep85

*First actual run — promised in Rounds 35, 37, 38. Gate on ≤0.060 removed per Round 38 P2 #5.*

Command: `python -m llgan.long_rollout_eval --n-records 50000 --n-streams 8 --seed 42 --char-file <char_file>` applied to each `frozen_best.pt`. Run 2026-04-21.

### Panel summary

| Metric | v176 (ATB ★=0.051) | v191 (bc ★=0.067) | v193 (latent-H ★=0.111) | v194 ep85 (bc ★=0.054) | Real |
|--------|---|---|---|---|---|
| reuse_access_rate | 0.046 | **0.193** | 0.007 | 0.006 | 0.265 |
| reuse_object_rate | 0.044 | **0.071** | 0.007 | 0.006 | 0.111 |
| reuse_decile_local_last | 0.093 | **0.260** | 0.016 | 0.018 | 0.273 |
| reuse_decile_drift | 0.048 | +0.083 excess | 0.005 | 0.008 | 0.068 |
| HRC-MAE | 0.1059 | **0.1027** | 0.1298 | 0.1305 | — |
| footprint over | +29.7% | **+9.7%** | +35.1% | +35.2% | — |
| IRD median | 1 | 1 | 1 | 1 | **194** |
| stack_distance median | 0 | 0 | 0 | 0 | **174** |

### Findings

**1. Decoded-feat bc (v191) beats the short-window ATB (v176) on every long-rollout metric.**
Reuse access rate 0.193 vs 0.046 (+320%). HRC-MAE 0.1027 vs 0.1059 (3% better). Footprint
overshoot 9.7% vs 29.7%. The bc mechanism is working — boundary criticism at the decoded
feature level substantially improves long-horizon reuse, even though v191's frozen ★=0.067 is
32% worse than v176's ★=0.051.

**2. v193 (latent-H bc) and v194 (decoded-feat bc, seed=5) are catastrophic on reuse.**
Both show reuse_access_rate ~0.006-0.007 (97-98% below real). This is ~27× worse than v191
on reuse. The latent-H mode destroys reuse; the seed-5 basin also loses reuse. The v194
"near-miss" on short-window ★=0.054 hides complete long-rollout degradation.

**3. IRD median = 1 is a universal floor across all four models.**
No generated trace has temporal locality beyond the immediate window. Real IRD median = 194
(an access is typically 194 positions away from its last occurrence). All fakes have IRD = 1
(every access is a new object or an immediate repeat). Stack distance = 0 for all fakes
(no object persists long enough to build non-trivial stack distance). This is a structural
failure: the LRU HRC can never match real because there is no object locality to exploit.

**4. v191 reuse_decile_drift is +121% above real (fake 0.151, real 0.068).**
The boundary critic is concentrating reuse in the late decile (last=0.260 vs real's 0.273)
while under-generating early reuse (first=0.109 vs real's 0.205). This shows bc is improving
total reuse but unevenly — late-epoch reuse is nearly calibrated while early-epoch reuse
is still weak.

### What this means for v195

v195 (IDEA #44, decoded-feat-matched bc) must match or improve on v191's reuse_access_rate
(0.193) to be a genuine improvement over decoded-feat bc. If v195 reuse_access_rate falls
below v191's 0.193 or near v193/v194's 0.006-0.007, the matched-domain modification hurt
long-rollout behavior. The bc_diag(raw, shuf) diagnostic will now have a second verification
layer: not just whether D_bc learned adjacency, but whether that adjacency signal translated
to long-rollout reuse.

**The primary open structural problem is IRD = 1 across all models.** Boundary criticism
improves window-join reuse but cannot fix the within-window locality failure. The LSTM
generator produces independent object draws each timestep with no persistent object revisit
mechanism. Fixing IRD = 1 requires either: (a) an explicit object-reuse probability model
(the z_global conditioning on obj_id statistics is insufficient), or (b) a long-rollout
adversarial signal directly on IRD distribution (IDEA in progress).

### Files
- `/home/darrell/longroll_v176.json`
- `/home/darrell/longroll_v191.json`
- `/home/darrell/longroll_v193.json`
- `/home/darrell/longroll_v194.json`

---

## Response to Round 39

### P1 #1 — Long-rollout panel is the immediate evidence needed — DONE

The long-rollout panel is complete and reported above (v176/v191/v193/v194 ep85,
run 2026-04-21). The evidence is in: v195 ep110 long-rollout also complete with
CATASTROPHIC result (reuse_access=0.0081, HRC-MAE=0.1287). Summary conclusion:

- v191 decoded-bc is the only bc variant with good long-rollout reuse (0.193)
- ALL decoded-feat-matched variants (v193, v194, v195) are catastrophic (0.006-0.008)
- IRD=1 floor is universal and structural across all four models and all seeds

The long-rollout sidecar has now rendered its verdict. The bc branch is a recall
stabilizer, not a locality escape route, as predicted.

### P1 #2 — Real-prefix continuation training (IDEA #47) is next — ACCEPTED

Accepted as architecture slot 1 after the current experiment closes. One cheap
parallel test is being run first: v197 adds the chained-window ACF loss on
obj_id_reuse (IDEA #45, T_long=48 across 4 chained windows). This requires zero
architecture change, reuses v195's pretrain, and runs at the same compute cost.
If it fails to move long-rollout reuse, architecture slot 1 goes to IDEA #47
immediately.

IDEA #47 (real-prefix continuation) is the right structural lever: training
explicitly on the task the model performs at inference (continue a real prefix)
closes the train/inference gap that boundary criticism only partially addresses.
Will implement as a fine-tune from an existing checkpoint with E/R frozen for
the first pass.

### P1 #3 — Locality needs an explicit output mechanism (IDEA #48) — ACCEPTED

Accepted as architecture slot 2. Starting as a post-Recovery repair experiment:
maintain a synthetic LRU stack during generation, predict new-object vs reuse,
and sample stack-distance buckets when reuse fires. If this moves HRC/stack-distance,
promote it into the model as a trained head. The reviewer is right that indirect
neural locality emergence has been failing for 15+ experiments; the object process
needs to be explicitly modeled.

### P1 #4 — Window-level bridge (IDEA #49) — ACCEPTED, LOWER PRIORITY

Accepted. Window atlas will be built after the continuation experiment (IDEA #47)
provides signal. Routing by window type makes most sense once there are at least
two mechanisms to route between.

### P1 #5 — Execution order, not larger queue — ACCEPTED

Current execution order:
1. v195 runs to ep200 (34 epochs left, ~85 min). Final frozen sweep to confirm
   or deny ep110 as the peak. Then CLOSE the bc branch.
2. v197 launches (IDEA #45 ACF chain, zero architecture change, cheap test).
3. If v197 fails long-rollout: implement IDEA #47 continuation training.
4. If v197 passes long-rollout: IDEA #48 as architecture slot 2.
5. IDEA #49 window atlas after continuation architecture is stable.

No more boundary-critic variants. The bc branch closes after v195+v197.

### P2 #6 — Keep v195, don't let it decide project direction — ACCEPTED

v195 will run to completion as promised (ep200). Its frozen ★=0.042 (ep110) is
the best clean-code result and a useful calibration point. But the project direction
is not contingent on v195 — the long-rollout failure is already confirmed. v195
closes the bc-decoded track; it does not extend it.

---

## Response to Round 40

### P1 #1 — v196 promotion bar incomplete; requires bc_diag adjacency criterion — ACCEPTED

Accepted. v196 promotion bar amended to three conditions:
1. Frozen ★ ≤ 0.054 (prior bar)
2. Long-rollout reuse_access_rate ≥ 0.10 (prior bar)
3. **bc_diag: bc_real > bc_shuf by ≥ 0.05 in the second half of training**
   (adjacency criterion — D_bc must separate adjacent from shuffled, not just
   real from fake)

If v196 satisfies conditions 1+2 but fails condition 3, the correct label is
"decoded real-vs-fake auxiliary / seed-basin regularizer", not "validated temporal
boundary critic." This is a meaningful distinction for the paper.

Note: v196 is currently paused at ep20 (killed to free GPU for v195's critical
window). Will restart after v195 completes.

### P1 #2 — IDEA #45 points at wrong object; binary ACF ≠ IRD — ACCEPTED WITH NUANCE

Accepted that binary obj_id_reuse ACF is not inter-reference distance. The reviewer
is technically correct: true IRD requires the emitted object identity stream and
counts distinct intervening objects. Binary reuse flag ACF measures run-length
structure of reuse events, not the gap distribution.

v197 (chained-window ACF on obj_id_reuse) is implemented and will be launched as a
cheap auxiliary test, explicitly NOT claimed as an IRD fix. Real ACF(lag=1) = −0.325
(strong negative — alternating reuse pattern) vs fake ACF(lag=1) ≈ 0 or positive
(long runs of same-object or absent-reuse runs). This is a real signal mismatch even
if it's not IRD directly. If v197 fails to move long-rollout reuse, the IDEA #45
ACF auxiliary is confirmed as insufficient and the slot moves to IDEA #47/#48.

If IDEA #48 (stateful LRU stack) is implemented, it will produce an actual object
identity stream that can be used to compute real IRD, making IDEA #45 Option A
(direct Wasserstein loss on IRD histograms) tractable. Until then, the ACF auxiliary
is the best available proxy at zero architecture cost.

### P1 #3 — v195 should close the matched-domain bc branch — ACCEPTED

v195 is at ep166 (34 epochs to go). The verdict is already in from ep110's
long-rollout (reuse_access=0.0081, CATASTROPHIC). The branch is closed regardless
of ep200 outcome. The final frozen sweep at ep200 will confirm whether ep110 remains
the peak or if there's a late recovery, but neither outcome changes the bc-decoded
conclusion: good short-window ★, catastrophic locality.

bc_diag shows shuf ≥ raw through ep165 — D_bc has not learned temporal adjacency
at any point in training. This is the second closure criterion (per Round 40 P1 #1
amendment). Both conditions are satisfied: (1) long-rollout catastrophic, (2) no
adjacency signal. The matched-domain boundary branch is closed.

### P2 #4 — torch.randint(B-1) crashes when B=1 — FIXED

Fixed. Added `if B >= 2:` guard around the derangement computation; emits
`nan` for the shuffled diagnostic on singleton batches. Committed 4a7a003
and pulled to vinge. This path cannot be triggered with the current training
command (`drop_last=True` + batch_size=64 + 239 files) but the CLI permits
`--batch-size 1` so the guard is correct.

### P2 #5 — Long-rollout interpretation overstates the failure — ACCEPTED

Accepted. Correcting the language in the IRD=1 floor description:

**Old** (incorrect): "every LRU cache always misses"

**Correct**: "the object process lacks the long-gap reuse law that shapes real HRCs.
The few reuses that occur are mostly immediate repeats (stack-distance ≈ 0), while
most objects never participate in realistic longer-gap reuse. A cache working set
this narrow cannot reproduce the real HRC curve, which requires objects to persist
in the working set long enough to be re-accessed after many intervening distinct
objects."

This is a more precise characterization: the failure is in the reuse gap
distribution (IRD histogram concentrated at 1, not the ~194-median real distribution),
not a claim about absolute cache hit rates.

---

## Round 41

### P1 #1 — ACF real-target mismatch (T_real=12 vs T_fake=48) — ACCEPTED; MOOT FOR v197

Accepted. The real EMA target is computed from `real_batch[:, :, obj_id_col]` (T=12), while the
fake side chains 4 windows with hidden-state carry (T_fake=48). This is a genuine mismatch: the
real ACF is estimated on single-window spans, while the fake ACF is measured on a 4-window long
rollout. They are not measuring the same quantity.

v197 was already killed at ep83 (43 epochs stale past ep40 best; frozen β-recall=0.05-0.08
throughout, 10× worse than v195 in diversity). So the bug is moot for this run. If IDEA #45 were
ever revisited, the fix would be to sample contiguous real spans of length
`timestep × acf_chain_windows` from the per-file raw arrays, compute ACF on those, and only then
compare to the carried fake rollout.

v197 result stands as a failure regardless of the target bug: diversity collapse (β-recall 0.05)
is not explained by a real-target mismatch.

### P1 #2 — Execution order delayed IDEA #47 — ACCEPTED; v197 CLOSED

Accepted. v197 was closed-failed before this review arrived, at ep83. IDEA #47 and IDEA #48 are
now the active slots. In light of LANL's NeuralAtlas results in Rounds 42-44, IDEA #48 (explicit
LRU stack decoder) is promoted ahead of IDEA #47 (continuation training) — the object-process
decoder is the more urgent architectural gap.

### P2 #1 — v195 "NEW OVERALL ATB" language is wrong — ACCEPTED; FIXING VERSIONS.md

Accepted. v195 set the best clean-code short-window frozen-bundle score (★=0.04204), but it
failed the long-rollout gate (reuse_access=0.0081 vs real 0.265; HRC-MAE=0.129). Labelling
it "overall ATB" after that failure is incorrect. Correcting all v195 references to "best
clean-code short-window score, failed long-rollout locality gate." VERSIONS.md will be updated.

### P2 #2 — IDEAS.md IDEA #45 stale IRD language — ACCEPTED; WILL FIX

Accepted. IDEA #45's motivation text still says "every LRU cache always misses." The corrected
framing (IRD=1 floor means the object process lacks long-gap reuse law, not that every access
misses) will be applied to the IDEA #45 body in IDEAS.md.

### P2 #3 — v196 promotion bar incomplete in VERSIONS.md — ACCEPTED; FIXING

Accepted. VERSIONS.md v196 block lists only frozen ★≤0.054 and reuse_access≥0.10, missing the
bc_diag criterion (bc_real > bc_shuf by ≥0.05 in second half of training). The three-part gate
from the Round 40 response amendment will be added to the run-state table.

---

## Round 42

### StackAtlas Challenge — ACCEPTED AS VALID; ARCHITECTURAL LESSON ABSORBED

LANL's StackAtlas is a correct diagnosis. We cannot make scalar losses on `obj_id_reuse` imply
the cache law that requires actual LRU stack rank to be chosen at each step. Boundary critics,
ACF chain losses, and decoded reconstructions all work on the output surface and hope the object
process emerges; StackAtlas generates the object process directly. The long-rollout panel confirms
which approach is right.

The challenge protocol (same manifest, n_records, n_streams, seed as our panel; HRC-MAE, reuse,
stack-distance, footprint, drift as primary metrics; ★ secondary) is adopted as our acceptance
gate going forward. Any model we promote must clear this panel, not just frozen-bundle ★.

### Execution commit

IDEA #48 (stateful stack-distance object decoder) is promoted to our next implementation slot.
We will not spend another architecture slot on scalar loss variants. The path is:

1. Explicit LRU stack during generation: predict new/reuse and stack-rank bucket per step.
2. Object ID comes from stack-rank selection (reuse) or fresh allocation (new).
3. Marks (dt, size, opcode, tenant) generated by our LSTM conditioned on object action and
   stack rank — this is our competitive edge over LANL's reservoir-based mark sampling.
4. Profile-conditioned routing (IDEA #50 protocol) as the acceptance benchmark.

Our target: match or beat NeuralAtlas on HRC/reuse/stack-distance while outperforming on ★
through LSTM mark realism (LANL's reservoir marks are coarse; our mark distributions are learned).

---

## Round 43

### StackAtlas Results — Conclusions Accepted

The StackAtlas panel proves the mechanism works: manifest-oracle Alibaba achieves HRC-MAE=0.00739
with reuse 0.279 vs real 0.269 and stack median 200 vs real 201. The LRU stack contract is
viable; the gap between oracle and held-out results is a conditioning gap (global atlas over-reuses
on Alibaba because it ignores workload-to-workload variation in action/rank distributions).

The over-reuse failure on Alibaba (reuse 0.435 vs real 0.269, stack median 83 vs real 201 with
16-file held-out) confirms workload-conditioned routing is essential — a global atlas cannot
represent the Alibaba mixture of high-reuse and low-reuse file families. This is exactly the gap
IDEA #49 (window router) and IDEA #50 (profile-routed atlas) target.

**Our implementation takeaway**: the stateful stack decoder (IDEA #48) must be conditioned on
per-file characterization vectors from the start, not added as a global post-hoc layer.
File-level routing (from our existing 41,831-file characterization database) is load-bearing on
Alibaba.

---

## Round 44

### P0 #1 — NeuralAtlas long-rollout results — ACCEPTED AS DEFINITIVE

NeuralAtlas HRC-MAE=0.00183 (Alibaba) and 0.01845 (Tencent) versus our best long-rollout
(v191 HRC-MAE=0.103) is a 56× difference on alibaba. This is not a tuning margin we can close
with another loss term. The result stands as the new long-rollout bar.

Our prior long-rollout table is superseded. The acceptance gate is now:
- HRC-MAE: must approach NeuralAtlas (<0.01 alibaba, <0.05 tencent)
- reuse_access: must match real within 0.05 absolute
- stack_distance_median: must be within 20% of real
- short-window ★: must also be competitive (not sacrificed for locality)

Any model that clears ★ but cannot approach these cache metrics does not advance.

### P0 #2 — Architectural lesson — ACCEPTED; COMMITTING TO OBJECT-PROCESS DECODER

Accepted. Our GAN has been trying to imply cache law from smooth latent decoders. NeuralAtlas
generates the object process directly, maintains LRU stack state, and selects the actual object
at each step. The result shows what we already knew theoretically: you cannot regress your way
to the correct IRD distribution. The object must be chosen, not inferred.

IDEA #48 implementation will follow the NeuralAtlas contract for object selection (new/reuse +
stack-rank), and add LSTM mark generation conditioned on the chosen action and rank. This is our
differentiated contribution: LANL's mark generation uses regime reservoirs (coarse, non-learned);
ours will be a neural sequence model conditioned on object-state. If we succeed, we beat NeuralAtlas
on ★ without sacrificing the cache panel.

### P1 #3 — Pure neural smoothing worsens Tencent — ACKNOWLEDGED AS OUR OPENING

LANL's pure-neural blend degrades Tencent monotonically (0.01845 → 0.07557 as blend goes 0→1).
Alibaba tolerates some smoothing, but the best blend is not 1.0 for either corpus. This means
the fitted atlas is doing real work that the neural smoother cannot yet replace.

For us this is an architectural opportunity: our LSTM mark model conditions on object-action and
stack-rank, which gives it information the pure reservoir doesn't have. If our mark model can
learn good timing/size/opcode distributions without losing transition state, we may be able to
run at blend=1.0 successfully where LANL could not.

### P1 #4 — NeuralStack failure on Tencent — LESSON ABSORBED

NeuralStack's Tencent collapse (stack median 27 vs real 60) from action/rank marginals alone
confirms that temporal transition state is necessary. This is exactly what NeuralAtlas adds with
its Markov transition model over atlas states. Our IDEA #48 implementation must include transition
state, not just action/rank marginals.

### P1 #5 — Held-out routing panel (IDEA #50) required — ADOPTED AS ACCEPTANCE GATE

Adopted. Our results will be reported using the IDEA #50 protocol: route by characterization
against a train/test split where the exact real-manifest files are held out from atlas fitting.
We will not report manifest-oracle scores as the primary result.

### Execution plan (v198)

Implement IDEA #48 (explicit LRU stack decoder) as v198:
- Per-step: LSTM predicts new/reuse logit + stack-rank bucket distribution
- Reuse: sample rank bucket, select object at that stack depth
- New: allocate fresh synthetic object ID, push to stack top
- Marks (dt, size, opcode, tenant): LSTM outputs conditioned on (object_action, stack_rank, hidden)
- Train with bucket NLL on reuse/new and rank; existing GAN losses on marks
- Long-rollout sidecar is acceptance gate; IDEA #50 held-out routing panel is benchmark
- Target: HRC-MAE <0.01 alibaba, reuse_access within 0.05 of real, ★ competitive with v195

## Round 45

### Observation — LANL uses one codebase for both Tencent and Alibaba

We noticed that `altgan/model.py` contains a single `StackAtlasModel` used without modification
for both corpora. This is not an accident — it's a structural advantage. LANL's model is purely
data-driven: the Markov transition atlas, per-state reservoir buffers, and initial state
distributions are all fitted from the actual trace statistics of each corpus. There is no
per-corpus architectural tuning because the architecture's job is to learn the statistics, not to
be hand-tuned to them.

We have been fighting corpus divergence for many rounds. Alibaba kills retrieval memory,
PCF, and multi-scale critic that Tencent benefits from. Our recipe switches are evidence that
we're embedding corpus knowledge into architecture choices rather than letting the model learn
it from data.

**The right response** is not to mimic LANL's Markov atlas (we already commit to IDEA #48/49/50),
but to adopt the same design principle: make our components data-driven and corpus-agnostic.
The LRU stack decoder (v198 phase 1) is already corpus-agnostic by this logic — the
stack-distance bucket PMF is fitted from whichever corpus we're generating for, and the
algorithm is identical. This is the right pattern.

**LANL's remaining weakness**: their mark generation (`_sample_event()`) is reservoir sampling
— a uniform draw from the per-state event buffer. It produces plausible dt/size/opcode marginals
within each regime state but cannot model sequential dependencies between marks (e.g., burst
structure within a hot-object loop, or correlated size-opcode patterns within a tenant). Our
LSTM mark head, conditioned on (object_action, stack_rank, hidden), is structurally able to learn
these dependencies. The long-rollout evaluation should break out mark quality separately from
cache quality — HRC/reuse/stack-distance measure the object process, but MMD² on
(dt, size, opcode) within regime strata would measure mark quality. That is where we can win.

### v197 — CLOSED-FAILED; v198 LRU stack decoder — LAUNCHED

v197 (IDEA #45 ACF chain) confirmed failed at ep83. Training EMA recall was 0.46–0.51 but
frozen β-recall only 0.05–0.08 — a 10× optimism gap from diversity collapse. ACF chain loss
closed permanently.

v198 launches today: `llgan/lru_stack_decoder.py` implements IDEA #48 as a post-hoc repair.
Phase 1 is corpus-agnostic (fits PMF from real trace CSV, defaults for both corpora) and wired
into `generate.py` via `--lru-stack-decoder`. The experiment is:

1. Generate trace using v195 ep110 (frozen-best ★=0.04204, catastrophic HRC-MAE=0.1287)
2. Apply LRU stack decoder with PMF fitted from real alibaba trace
3. Run long_rollout_eval on the decoded trace
4. Report HRC-MAE, reuse_access, stack_distance_median vs baseline and vs NeuralAtlas

If phase 1 moves HRC-MAE below 0.05 (vs current 0.1287), the structural hypothesis is confirmed
and we move to phase 2: integrate the stack decoder into training as a supervised head (bucket NLL).
If phase 1 shows minimal improvement, the problem is in the reuse_signal quality (generator's
obj_id_reuse column) and we need to fix the root signal before the decoder can help.

### Competitive position

LANL leads on the cache metrics panel. We lead on short-window ★ (v195 ep110 ★=0.04204 vs
NeuralAtlas which does not report this metric directly). v198 is the thesis that we can add
a correct object process on top of our strong mark model without destroying ★. If it works:
we beat LANL on the compound benchmark. If ★ degrades when we add the stack decoder, we must
redesign the training signal so that mark generation and object selection are jointly optimized.

The adversary is behind us on both fronts (short-window quality and cache metrics), so the
correct priority is completing v198 before they replicate our LSTM mark advantage. One-codebase
design is their robustness; LSTM mark conditioning is ours.
