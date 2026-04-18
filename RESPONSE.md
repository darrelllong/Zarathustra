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

