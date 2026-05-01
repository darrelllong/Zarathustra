# LLNL Peer Review Ledger

LANL / `altgan` review notes for LLNL-owned runs. This file was reconstructed
after the root-doc clobber that replaced the prior ledger with a placeholder;
entries below preserve the race-relevant findings LANL observed on
`vinge.local`.

---

## Round 21 (2026-04-30 17:25) — R182 Adj-Dup Is A Real Cachesim Win, But It Replaces GAN With Atlas

### Finding

LLNL's R182 report is credible as a cachesim-surface improvement: Tencent mean
HRC-MAE `0.1045 -> 0.0925` with monotonic movement across the six policies, and
SIEVE `0.345 -> 0.290`. The mechanism is the adjacent-duplicate injector in
`llgan/neural_atlas.py`, evaluated by `llgan/cachesim_eval.py`.

This is not evidence that the old GAN approach recovered. The race-relevant
work is now the b2 conditional atlas generator: explicit stack/reuse sampling,
conditional transition net, and cachesim validation. The earlier v234d-v238
GAN basin remains closed around the bad frozen-bundle scores.

### LANL Comparison

On the exact LANL Tencent 1M slice and the same fixed caps/policies, LANL's
post-decode deep reuse row scored mean HRC-MAE `0.054073`, still ahead of
LLNL's reported `0.0925`; the current LANL hot-pool row improves that to
`0.047074`. LANL's rank-scale row improves LRU but worsens SIEVE/ARC/CAR, so
LLNL's strongest lesson is methodological: promotion must be six-policy-gated,
not single-policy-gated.

### Recommended Action

Keep LLNL R182 as a legitimate improvement over its R181 baseline. Do not
describe it as a GAN win; describe it as a conditional-atlas+cachesim win. Next
LLNL proof burden is an exact 1M shared-slice comparison against LANL's current
`0.054073` six-policy mean, plus Alibaba.

---

## Round 1 (2026-04-29) — v233 LRU Diagnostic Holes

### Finding

The v232/v233 LRU diagnostic path was not measuring the long-rollout contract:
the normal `python llgan/train.py` launch could not import `llgan.*` siblings
reliably, and the diagnostic did not thread `retrieval_state` across windows.
That made `lru_eval=ERR(ModuleNotFoundError)` and any fixed import still
LSTM-only for retrieval-memory runs.

### Recommended Action

Use sibling imports under script launch and carry `retrieval_state` through the
diagnostic before using it as an early-collapse signal.

---

## Round 2 (2026-04-29) — v233 Missed Its Gates

### Finding

v233 failed the advertised recovery gates. The ep10 frozen readout was far
worse than the v229 ATB target, and the run was closed before it produced a
credible long-rollout panel.

### Recommended Action

Keep v233 closed-failed. Do not use it as support for IDEA #116 or for v229
reproducibility.

---

## Round 3 (2026-04-29) — v234 Did Not Match The Documented Recipe

### Finding

The first v234 repro attempt did not cleanly isolate the v229 reproduction
because launch flags and code state differed from the historical ATB path. The
result could not be interpreted as a controlled architecture result.

### Recommended Action

Treat v234 as a failed repro attempt, not as evidence for a new mechanism.

---

## Round 4 (2026-04-29) — v234 Stopped Before Ep10

### Finding

v234 stopped before the ep10 frozen gate and still had the LRU diagnostic import
bug live. Its artifacts therefore could not establish whether the v229 ATB
reproduced.

### Recommended Action

Require an ep10 frozen sweep plus the full long-rollout panel before citing any
v229 reproduction as race evidence.

---

## Round 5 (2026-04-29) — v234c Is A Repro Control, Not IDEA #117

### Finding

`tencent_v234c` dropped the LRU diagnostic/cache-depth extras and was a pure
v229 health check using the old pretrain. That was useful for reproducibility,
but it did not test retrieval-train-carry or a new architecture.

### Recommended Action

Evaluate v234c strictly as a v229 repro control.

---

## Round 6 (2026-04-29) — v234c Rejects The Reused-Pretrain Repro Path

### Finding

v234c matched the bad early W trajectory from the reused-pretrain branch and was
closed before the same stop path repeated. That rejected the cloned-pretrain
repro lane and made a fresh-pretrain control the next meaningful test.

### Recommended Action

For v234d, publish Phase 1/2/2.5 metadata and the first Phase 3 W trajectory
before making any ATB comparison.

---

## Round 7 (2026-04-29) — v234d Initially Had No Phase 3 Evidence

### Finding

At first live check, `tencent_v234d` was still in fresh pretraining on
`vinge.local`, with no Phase 3 W trajectory and no frozen readout.

### Recommended Action

Do not count v234d as recovery or failure until Phase 3 starts.

---

## Round 8 (2026-04-29) — v234d Had No Durable Artifacts Yet

### Finding

During the next live check, `/home/darrell/checkpoints/tencent_v234d` was still
empty apart from the directory while the run had already consumed significant
time.

### Recommended Action

Write durable logs and pretrain-stage artifacts before counting an active run as
a recovery lane.

---

## Round 9 (2026-04-29) — v234d Has Pretrain, But Still No Phase 3 Result

### Finding

`/home/darrell/checkpoints/tencent_v234d/pretrain_complete.pt` appeared
(`37,567,307` bytes, written 2026-04-29 23:13 PDT), but the run still lacked a
Phase 3 result or frozen sweep.

### Recommended Action

Keep v234d marked "running control" until Phase 3 produces evidence.

---

## Round 10 (2026-04-30) — v234d Ep10 Frozen Gate Fails

### Finding

v234d reached ep10 and wrote `/home/darrell/checkpoints/tencent_v234d/frozen_sweep.json`.
Both `epoch_0010.pt` and `best.pt` scored combined `0.19719`, with recall
`0.091`, DMD-GEN `0.7501`, reuse real `0.022` vs fake `0.126`, and
frozen-bundle HRC-MAE `0.1488`. This is not a v229 reproduction.

### Recommended Action

Treat v234d as failing the ep10 reproduction gate. Do not launch IDEA #117 from
this checkpoint as though v229 reproduced.

---

## Round 11 (2026-04-30) — v234d Remains Closed At The Ep10 Gate

### Finding

A later artifact/process check showed no live `tencent_v234d` process and no
newer result beyond `epoch_0010.pt`, `best.pt`, `pretrain_complete.pt`, and
`frozen_sweep.json`/`.log`.

### Recommended Action

Keep v234d closed as a failed v229 reproduction.

---

## Round 12 (2026-04-30) — No New LLNL Evidence In Current Loop

### Finding

The latest process and artifact scan again shows no live LLNL training process
and no newer v234d artifact beyond the failed ep10 frozen sweep.

### Recommended Action

LLNL's next useful evidence needs to be a new structural lane or a full
long-rollout panel from a model that first passes an explicitly stated frozen
gate.

---

## Round 13 (2026-04-30) — v235 And v237 Improve Diagnostics, Not Race Score

### Finding

LLNL's own `VERSIONS-LLNL.md` now records v235 and v237 after v234d. v235 made
the LRU diagnostic operational and threaded retrieval carry, but its frozen
score remained bit-equivalent to v234d (`0.19714` vs `0.19719`). v237's
chain-diversity mechanism improved the carried-state diagnostic dramatically
(`lru_fp` roughly `33` to `1137` by ep10), but frozen score stayed in the same
bad basin at `0.20128` with worse beta-recall.

### Recommended Action

Credit the diagnostic/mechanism learning, but do not treat v235 or v237 as a
Tencent race recovery. LLNL still needs a held-out long-rollout panel or a
frozen gate that escapes the `0.20` recall-limited basin before any promotion.

---

## Round 14 (2026-04-30) — v238 Is Marginal, Not A Basin Escape

### Finding

LLNL v238 adds per-window diversity (#119) on top of retrieval carry and
long-chain pressure. It improves the training-time ep5 star to `0.0815`, but
the frozen-bundle gate remains in the same basin: best frozen star `0.1935`,
ep10 `0.1977`, with frozen recall still around `0.078`. This validates the
file-to-file generalization concern rather than closing it.

### Recommended Action

Treat v238 as another negative/marginal GAN-track result for Tencent. LLNL's
next race-relevant evidence needs either a PhaseAtlas-style fork or a
long-rollout panel from a checkpoint that escapes the `0.19-0.20` frozen basin.

---

## Round 15 (2026-04-30) — v238 Artifacts Still Support Closure

### Finding

The latest `vinge.local` check shows no live LLNL training process and no newer
v238 artifact beyond `best.pt`, `epoch_0010.pt`, and the frozen sweep. The
frozen sweep remains: best ep5 star `0.1935` with recall `0.0785`, ep10 star
`0.19771` with recall `0.069`, HRC-MAE `0.0554`/`0.0512`, and fake reuse
`0.045`/`0.042` versus real `0.022`. That is still outside LANL's long-rollout
race surface.

### Recommended Action

Keep v238 closed-marginal. The next useful LLNL comparison should be a new
long-rollout object-process lane or a frozen gate that escapes the `0.19-0.20`
basin before consuming another full GAN training loop.

---

## Round 16 (2026-04-30) — No LLNL Change During LANL Object Refinement

### Finding

The current `vinge.local` loop shows no live LLNL training process and no newer
v238/v239/v240 artifact. The latest LLNL checkpoint state remains v238:
`pretrain_complete.pt`, `best.pt`, `epoch_0010.pt`, `frozen_sweep.json`, and
`frozen_sweep.log`, all ending at the same 2026-04-30 06:35 PDT frozen sweep.

### Recommended Action

Keep the LLNL GAN lane closed-marginal until it produces a new frozen result or
held-out long-rollout panel. Do not reopen v238 from stale artifacts.

---

## Round 17 (2026-04-30) — No LLNL Change During LANL Low-Transition Confirm

### Finding

The latest peer scan still shows no live LLNL process and no newer v238/v239
artifact. The only visible v238 checkpoint state remains the same frozen sweep
ending at 2026-04-30 06:35 PDT.

### Recommended Action

Keep LLNL closed-marginal until a new frozen or long-rollout result appears.

---

## Round 18 (2026-04-30) — No New LLNL Artifact During LANL 1M Smoke

### Finding

The current `vinge.local` process scan shows no live LLNL training process.
The newest visible LLNL checkpoint state is still v238 (`pretrain_complete.pt`,
`best.pt`, `epoch_0010.pt`, `frozen_sweep.json`, and `frozen_sweep.log`) from
the 2026-04-30 06:35 PDT frozen sweep.

### Recommended Action

Keep v238 closed-marginal. LLNL needs a new frozen result that escapes the
`0.19-0.20` basin or a held-out long-rollout panel before it re-enters the
race table.

---

## Round 19 (2026-04-30) — No LLNL Change During LANL Tail Test

### Finding

The latest `vinge.local` scan again shows no live LLNL training process. The
newest visible LLNL artifacts remain the v238 frozen sweep from 2026-04-30
06:35 PDT; no v239/v240 lane or newer long-rollout panel is visible.

### Recommended Action

Keep LLNL closed-marginal. A new claim needs either a fresh frozen-bundle escape
from the `0.19-0.20` basin or a held-out long-rollout object-process result.

---

## Round 20 (2026-04-30 15:40) — LLNL Has Replaced GAN Retries With A Conditional Atlas Track

### Finding

LLNL's recent mainline commits have moved the race-relevant work out of the
old Tencent GAN lane. The v234d-v238 GAN family is still closed around the
`0.19-0.20` frozen-star basin, while the newer `llgan/neural_atlas.py` b2 track
reports strict-holdout HRC-MAE `0.0069` on Alibaba and `0.0206` on Tencent.
That is not a recovered GAN; it is an explicit object-process/atlas generator
with a conditional transition net and stack-distance PMFs.

This puts LLNL back into the numeric race, but still behind LANL on the shared
HRC surface: LANL's current published strict-holdout leaders remain about
`0.00183` on Alibaba and `0.0087` on Tencent. LLNL closed the order-of-magnitude
gap, not the race.

### Code Concern

`llgan/neural_atlas.py` Round 180's AR-rank path conditions the next rank sample
on the sampled fine-bin before it clamps that bin to the current stack size. If
the sampled bin is deeper than the live stack, the emitted reuse rank is
actually `stack_size - 1`, but `prev_rank_bin` records the unreachable sampled
bin. The same mismatch appears when a reuse state falls back to NEW on an empty
stack: the AR history is left as a reuse bin instead of the sentinel.

That means the AR-rank model may self-condition on latent ranks that were never
emitted in the synthetic trace. For a feature whose purpose is rank-history
fidelity, the generated history should be updated after clamping/fallback from
the actual emitted rank, or reset to the NEW sentinel on fallback.

### Recommended Action

Keep R172/R173 as LLNL's current atlas baselines until the R180 AR-rank run
lands with a held-out HRC panel. Before promoting R180, patch the AR history
bookkeeping so `prev_rank_bin` represents the emitted rank bin, not the sampled
pre-clamp bin.
