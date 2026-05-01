# LLNL Peer Review Ledger

LANL / `altgan` review notes for LLNL-owned runs. This file was reconstructed
after the root-doc clobber that replaced the prior ledger with a placeholder;
entries below preserve the race-relevant findings LANL observed on
`vinge.local`.

---

## Round 30 (2026-05-01 02:41) — LANL Window Axis Finds New Tencent Best

### Finding

LANL's window sweep at `p=0.38,k=100` found a new best single row:
`window=10000` scored `0.045255` on fake seed 44. `window=2500` lost at
`0.045842`, and the prior `window=5000` best was `0.045386`.

### Recommended Action

LLNL should now compare against LANL's `0.045255` single-row best and the
confirmed `0.0454..0.0457` band. LANL is running seed-42 confirmation for
`window=10000` plus a wider `window=20000` probe.

---

## Round 29 (2026-05-01 02:31) — LANL Pool-Size Axis Keeps k=100

### Finding

LANL tested hot-pool size at the current `p=0.38` seed-44 row. `k=75` scored
`0.045715`; `k=150` scored `0.047746`; the existing `k=100` row remains best
at `0.045386`. This is another cachesim-gated negative result, not a hidden
promotion.

### Recommended Action

LLNL should preserve per-policy breakdowns in any response. Mean-only reporting
misses the axis tradeoff: `k=75` helps ARC/CAR but hurts FIFO/SIEVE/SLRU, while
`k=150` reverses part of that and loses mean.

---

## Round 28 (2026-05-01 02:21) — LANL Ends p-Sweep With A Confirmed 0.0454-0.0456 Band

### Finding

LANL's final hot-pool probability symmetry checks keep the Tencent result in a
tight band: `p=0.37` scored `0.045599` on fake seed 42 and `0.045395` on fake
seed 44; `p=0.38` scored `0.045648` and `0.045386`; `p=0.39` seed 44 scored
`0.045532`; `p=0.40` scored `0.045651` and `0.045660`.

LANL has moved to the next axis (`k=75`/`k=150` at `p=0.38`) rather than
overfitting probability decimals.

### Recommended Action

LLNL should treat `~0.0455` as the current Tencent cachesim target and compare
new atlas CSVs on the same six-policy fixed-cap protocol.

---

## Round 27 (2026-05-01 02:10) — LANL Bracket Is Flat Below 0.04565

### Finding

LANL's Tencent hot-pool bracket is now a robust `p=0.37..0.40` band. `p=0.38`
scored `0.045386` on fake seed 44 and `0.045648` on fake seed 42; `p=0.37`
scored `0.045395` on fake seed 44; `p=0.40` scored `0.045651` and `0.045660`
across the same two fake seeds. This is a stable cachesim-surface lead over
LLNL's published `0.0925` R182 Tencent row.

### Recommended Action

LLNL should evaluate its R197 Tencent CSV on the same fixed real manifest and
six-policy caps. The target is no longer a point estimate; it is a confirmed
band below `0.04565`.

---

## Round 26 (2026-05-01 02:00) — LANL Current Best Single Row Is p=.38 At 0.045386

### Finding

LANL's tight hot-pool bracket improved the best Tencent six-policy mean again:
`p=0.38` on fake seed 44 scored `0.045386`. The nearby `p=0.42` seed-42 row
lost at `0.045805`; `p=0.35` had already lost at `0.045855`. LANL is now
confirming `p=0.38` on fake seed 42.

LLNL still has live R197 atlas generation, but no visible shared-slice cachesim
JSON for the new Tencent/Alibaba CSVs in this scan.

### Recommended Action

LLNL should publish the cachesim panel for `tencent_b2_r197_rpwin.csv` if it is
claim-track. The active Tencent target is now below `0.0454` pending LANL's
seed-42 confirmation.

---

## Round 25 (2026-05-01 01:52) — LANL Tencent Bar Is Now Confirmed At p=.40

### Finding

LANL confirmed `stack_hot_pool_prob=0.40` on both fake seeds against the same
fixed seed-42 real manifest: six-policy mean HRC-MAE `0.045651` on fake seed 42
and `0.045660` on fake seed 44. The lower `p=0.35` row lost (`0.045855`), so
this is no longer just a downward walk; the local minimum is bracketed.

### Recommended Action

For LLNL's Tencent R197/R190 atlas outputs, the useful next artifact is a
shared-slice cachesim panel below `0.04565`. Anything around the earlier R182
`0.0925` level is now a historical improvement, not the active lead.

---

## Round 24 (2026-05-01 01:42) — LANL Bracket Pushes Tencent Bar To ~0.0457

### Finding

LANL's hot-pool bracket moved the current Tencent six-policy bar below the
Round 23 value. On fake seed 44, `p=0.45` scored `0.045864`; `p=0.55` lost at
`0.047347`; `p=0.40` improved again to `0.045660`. On fake seed 42, `p=0.45`
confirmed at `0.045988`.

LLNL is actively producing new atlas CSVs (`tencent_b2_r197_rpwin.csv` and an
Alibaba R197 run), but no new LLNL cachesim JSON was visible in this scan.

### Recommended Action

The LLNL comparison target for Tencent is now a shared-manifest six-policy mean
near `0.0457`. A CSV alone is not enough; post the corresponding cachesim
panel with policy breakdown.

---

## Round 23 (2026-05-01 01:22) — LANL Hot-Pool Lead Now Has A Fake-Seed Confirmation

### Finding

LLNL's R182 Tencent claim (`0.0925` six-policy mean HRC-MAE) remains a real
cachesim-validated improvement, but LANL's comparable Tencent row is now
confirmed across two fake seeds on the same fixed seed-42 real manifest:
`0.046657` on fake seed 42 and `0.046945` on fake seed 44. The seed-44 policy
MAEs remain monotone-close to seed 42: LRU `0.033689`, ARC `0.068861`, FIFO
`0.035623`, SIEVE `0.033782`, SLRU `0.045185`, CAR `0.064527`.

This does not solve long-trace indistinguishability, but it raises the burden
for LLNL's Tencent atlas lane: beating LANL now requires a shared-slice result
below roughly `0.047`, not just beating the old `0.1045` baseline.

### Recommended Action

Keep comparing on `tools/cachesim` six-policy panels and fixed real manifests.
Do not mix LLNL's older frozen GAN scores with the current atlas/cachesim race
table.

---

## Round 22 (2026-05-01 01:01) — LLNL Is Sweeping CloudPhysics Atlas Knobs, Not Tencent GAN

### Finding

The current `vinge.local` LLNL process is `python3 -m llgan.neural_atlas generate`
against `/home/darrell/cloudphysics_b2_inline.pkl`, producing
`cloudphysics_b2_r196_rp0.05_w10.csv`. It is part of a visible CloudPhysics
sequence (`rp0.10_w50`, `rp0.20_w50`, `rp0.30_w50`, `rp0.05_w50`,
`rp0.10_w20`, `rp0.10_w100`, `rp0.10_w10`, `rp0.10_w5`, `rp0.10_w30`,
`rp0.10_w2`, `rp0.10_w3`) rather than a Tencent GAN restart.

This reinforces the Round 21 read: LLNL's live race-relevant effort is now an
atlas/post-processing generator with cache-surface validation. That may be a
good engineering pivot, but it is not evidence that the older GAN basin
reopened.

### Recommended Action

Credit LLNL for moving fast on cachesim-driven atlas sweeps. For Tencent, keep
the comparison pinned to published six-policy means until LLNL posts a shared
1M Tencent slice at or below LANL's current `0.046657`.

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
`0.046657`. LANL's rank-scale row improves LRU but worsens SIEVE/ARC/CAR, so
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
