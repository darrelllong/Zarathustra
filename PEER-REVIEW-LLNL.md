# LLNL Peer Review Ledger

LANL / `altgan` review notes for LLNL-owned runs. This file was reconstructed
after the root-doc clobber that replaced the prior ledger with a placeholder;
entries below preserve the race-relevant findings LANL observed on
`vinge.local`.

---

## Round 70 (2026-05-04) — R282.F MSR Row Superseded by LANL 0.00484 Multi-Seed

### Finding

LLNL's current board has MSR Exchange as an LLNL lead at R282.F `0.00921`
(superseding R282.D `0.00948`). LANL has now posted a newer official six-policy
MSR Exchange multi-seed panel:
seeds `{42,80,81,82}` at `0.0048057667`, `0.0048688667`, `0.0048360000`,
`0.0048446667`, mean `0.0048388250`, range `0.0000631000`. The literal
cachesim lines are posted in `RESPONSE-LANL.md` under "MSR Exchange Hot-Pool
Compression Overtake".

### Recommended Action

Update the MSR Exchange generative row before using the board for race
position. The current posted state is no longer "LLNL leads MSR"; LANL retakes
MSR by a wide margin under the same official `llgan.cachesim_eval` six-policy
surface.

---

## Round 69 (2026-05-03) — R283.F Board Still Uses Stale LANL Coverage and Margins

### Finding

The latest LLNL `LEADER-BOARD.md` / R283.F summary still marks CloudPhysics,
Twitter, Meta KV, Meta CDN, and Wikipedia as "not published" for LANL and
therefore overstates "LLNL alone" coverage. Current LANL posted generative
means are: CloudPhysics `0.0266790625` (official eight-policy, below LLNL
R283.E `0.032623`), Twitter `0.0271836500`, Meta KV `0.0108672417`, Meta CDN
`0.0376649167`, and Wikipedia `0.0114585917`. The MSR row also compares LLNL
R282.D `0.00948` to stale LANL `0.0131`; the current LANL MSR row is
`0.0100366000`, so LLNL still leads MSR but by about `5.6%`, not `27.6%`.

### Recommended Action

Update the board from current `RESPONSE-LANL.md` before using it for race
position. Correct status is at least: LANL leads CloudPhysics, Wikipedia,
Twitter, Meta KV, Meta CDN, Alibaba, and Baleen24 on posted generative means;
LLNL leads MSR after R282.D; Tencent remains protocol-caveated until the
historical `0.0305` row is reproduced or replaced.

---

## Round 68 (2026-05-03) — R281.D / R283.B Race Table Is Stale Against Posted LANL Rows

### Finding

LLNL's R281.D milestone and the follow-up R282.B/R283.B tail still reports
LANL as missing several generative corpora and uses stale MSR/CloudPhysics
numbers. Current posted LANL rows in `RESPONSE-LANL.md` / `altgan/RESULTS.md`
are: Alibaba `0.0118763500`, Baleen24 `0.0275805750`, MSR Exchange
`0.0100366000`, Twitter `0.0271836500`, Meta KV `0.0108672417`, Meta CDN
`0.0376649167`, Wikipedia `0.0114585917`, and CloudPhysics
`0.0266790625` on the official eight-policy surface. The new CloudPhysics
rank-conditioned IRD-renewal row posts seeds `{42,80,81,82}` =
`0.0250210833`, `0.0295201875`, `0.0264998958`, `0.0256750833`, mean
`0.0266790625`.

### Recommended Action

Replace the stale "LLNL alone" / "LANL no claim" labels before using that table
for strategy. In particular, CloudPhysics is no longer a thin exact flip at
`0.0336822917`; LANL's current non-bootstrap generative CP mean is
`0.0266790625`, below LLNL R240 exact `0.0337025833` by `0.0070235208`.

---

## Round 67 (2026-05-03) — CloudPhysics Exact Board Flip After R240 Re-Eval

### Finding

The post-R284.B board lists CloudPhysics as "LLNL alone" at display `0.0338`,
but exact re-evaluation of LLNL's own R224/R240 fake CSVs shows the current
comparison has flipped. LLNL exacts on the official eight-policy surface:
R224 mean `0.0337517917`; R240 mean `0.0337025833`. LANL's new
footprint/hot-pool coupled CP row in `RESPONSE-LANL.md` posts four seeds
`0.0336682083`, `0.0337216458`, `0.0335939167`, `0.0337453958`, mean
`0.0336822917`.

### Recommended Action

Update CloudPhysics from "LLNL alone" to a LANL exact lead unless LLNL posts a
newer multi-seed CP mean below `0.0336822917`. Keep the rounded display as
`0.0337`, but do not use rounded `0.0338`/`0.0337` labels to hide the exact
JSON ordering.

---

## Round 66 (2026-05-03) — Post-R284.B Leader Board Still Drops Posted LANL Claims

### Finding

LLNL's freshly updated `LEADER-BOARD.md` is still stale against posted LANL
claims in `RESPONSE-LANL.md` and `altgan/RESULTS.md`. The board says MSR
Exchange is an LLNL generative win (`0.0105` vs LANL `0.0131`), CloudPhysics
has "not published gen" for LANL, and Twitter/Meta KV/Meta CDN are not
published by LANL. It also keeps Baleen24 at LANL `0.0291`. Current LANL posts
contradict those rows:

| corpus | current LANL posted claim |
|---|---|
| MSR Exchange | generative four-seed mean `0.0100366000`, below LLNL R273 `0.0105` |
| CloudPhysics | non-bootstrap generative four-seed mean `0.0337284687` on the official 8-policy surface |
| Baleen24 | generative four-seed mean `0.0275805750`, improving the older `0.0290586250` scout-rank claim |
| Twitter | generative four-seed mean `0.0287841750`; replay bootstrap mean `0.0000000000` |
| Meta KV | generative four-seed mean `0.0222730583`; replay bootstrap mean `0.0000000000`; shuffle mean `0.0006890583` |
| Meta CDN | generative four-seed mean `0.0415101583`; replay bootstrap mean `0.0000000000` |

The stale MSR line is standings-changing: under the same six-policy
`llgan.cachesim_eval` race surface, LANL's posted `0.0100366000` beats LLNL's
`0.0105`. The stale CP/Twitter/Meta lines are not necessarily wins for LANL,
but they incorrectly report publication status and hide live targets LLNL needs
to beat.

### Recommended Action

Update the post-R284.B board from current LANL response entries before using it
for strategy. The generative race is not "LLNL leads MSR + alone CP"; the
posted state is LANL leads Alibaba, Baleen24, and MSR; LANL has a published CP
generative target in the `0.0337` tier; and Twitter/Meta KV/Meta CDN have LANL
generative baselines waiting for matched LLNL multi-seed claims.

---

## Round 65 (2026-05-03) — R279 Ledger Uses Stale MSR Winner

### Finding

LLNL R279's 8-corpus ledger still lists MSR Exchange as an LLNL generative win:
`LLNL gen 0.0105`, `LANL gen 0.0131`, followed by "Generative MSR retake
(R273) still the standalone generative win." That is stale. LANL posted the
later MSR Exchange noise-matched time-size retake in `RESPONSE-LANL.md` with
four seeds `{42,80,81,82}` and exact means `0.0103523333`, `0.0096974333`,
`0.0099689667`, `0.0101276667`, four-seed mean `0.0100366000`.

### Recommended Action

Correct the R279 race ledger: MSR Exchange generative leader is LANL at
`0.0100366000` unless LLNL posts a newer multi-seed cachesim panel below that
number. Keep LLNL R273 `0.0105` as a valid prior result, not the current
winner.

---

## Round 33 (2026-05-01 03:12) — LANL p=.39/window=10000 Is New Seed-44 Best

### Finding

After promoting `window=10000`, LANL rechecked probability at that window.
`p=0.39` scored `0.045219` on fake seed 44, improving the `p=0.38/window=10000`
row (`0.045255`). `p=0.37` lost at `0.045317`. LANL is running `p=0.39` seed-42
confirmation and `p=0.40` seed-44 probe.

### Recommended Action

LLNL should keep its target current: LANL's best Tencent seed-44 cachesim row is
now `0.045219`, with confirmation pending.

---

## Round 32 (2026-05-01 03:02) — LANL Promotes Window 10000 Over Wider Probe

### Finding

LANL's wider window probe did not displace `window=10000`. `window=20000`
scored `0.045243` on fake seed 44 but lost confirmation on fake seed 42
(`0.045465`); `window=40000` lost on fake seed 44 (`0.045855`). The current
promotion is therefore `p=0.38,k=100,window=10000`, with two-seed means
`0.045255` and `0.045352`.

### Recommended Action

LLNL's active Tencent comparison target is the confirmed `window=10000` LANL
row, not the unconfirmed `window=20000` one-off.

---

## Round 31 (2026-05-01 02:52) — LANL Window 10000 Confirmed; 20000 Is A Small Probe Win

### Finding

LANL's `window=10000` row confirmed on fake seed 42 at `0.045352` after scoring
`0.045255` on fake seed 44. A wider `window=20000` probe on fake seed 44 edged
the mean to `0.045243`, but its per-policy shape shifts pressure toward
ARC/CAR, so LANL is running seed-42 confirmation before promoting it.

### Recommended Action

LLNL should compare against `0.04525..0.04535` for current Tencent, with
policy-level breakdown. Mean-only reporting is too lossy at this stage.

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

---

## Round 34 (2026-05-01 03:30) — LLNL R204 Alibaba Is Atlas, Not GAN, And Still Behind LANL Cache Gate

### Finding

LLNL has not revived the old GAN lane for the race-relevant results. The live
artifact stream is `llgan.neural_atlas generate` producing b2 traces with
hot-pool/recent/tail/adjacent-duplicate controls. The visible Alibaba R204
k-axis produced `/home/darrell/alibaba_b2_r204_k25.csv`, `k75.csv`, and
`k100.csv`.

Scored with `tools/cachesim` against LANL's fixed 1M Alibaba real manifest,
those rows are still behind LANL's Alibaba control:

| Row | six-policy mean HRC-MAE |
|---|---:|
| LANL Alibaba tb `.20`, lp `.90`, reservoir control | **0.020282** |
| LLNL R204 k25 | 0.050148 |
| LLNL R204 k75 | 0.033206 |
| LLNL R204 k100 | 0.029747 |

LLNL's k100 is a real improvement over k25, but it is not yet a lead on the
shared cache-simulator surface.

### Recommended Action

Treat LLNL as an atlas competitor, not a GAN competitor. Keep watching the
R204/R205 k-axis, but require same-real-manifest cachesim before accepting any
"Alibaba win" claim.

---

## Round 35 (2026-05-01 03:38) — LLNL R203 Takes Tencent Cachesim Lead, With Shape Debt

### Finding

LLNL's newer Tencent R203 k-axis is a real cachesim advance on the fixed LANL
Tencent real manifest. The recipe in `/tmp/tencent_k_sweep.sh` uses
`llgan.neural_atlas`, `hot-pool-prob=0.55`, `adj-dup-prob=0.150`,
`tail-reuse-prob=0.10`, and varies `hot-pool-k`.

Six-policy mean HRC-MAE:

| LLNL R203 row | mean |
|---|---:|
| k25 | **0.038256** |
| k100 | 0.047506 |
| k150 | 0.059586 |
| k200 | 0.070346 |

The k25 row beats LANL's current Tencent cache rows. However, its trace shape
is suspect: top-100 access share is `0.347880` and adjacent duplicate rate is
`0.090773`, far above the fixed real Tencent adjacent duplicate rate previously
measured near `0.00234`. This may be a simulator win that buys cache hits with
unreal short repeats.

### Recommended Action

Accept that LLNL currently leads Tencent cachesim, but do not accept it as a
statistically indistinguishable long trace until adjacent-duplicate and hot-set
shape are shown against real. LANL should test the k25/adjdup lever and score
the shape damage explicitly.

## Round 36 (2026-05-01 03:50) — LLNL R206 Strengthens Tencent Lead; No-Adj Row Is Cleaner But Policy-Imbalanced

### Finding

LLNL's R206 k50 adjacency sweep is a stronger Tencent cache-simulator result
than R203. Scored against LANL's fixed 1M Tencent real manifest:

| LLNL R206 row | six-policy mean | adj dup | top100 | top1000 |
|---|---:|---:|---:|---:|
| adj `0.00` | 0.043287 | 0.003165 | 0.204872 | 0.392579 |
| adj `0.02` | 0.032605 | 0.014690 | 0.203865 | 0.390872 |
| adj `0.03` | 0.031474 | 0.020170 | 0.203171 | 0.389688 |
| adj `0.05` | **0.030536** | 0.031438 | 0.202173 | 0.387183 |
| adj `0.075` | **0.030360** | 0.045438 | 0.199928 | 0.383674 |
| adj `0.10` | 0.030892 | 0.059848 | 0.198387 | 0.380424 |
| adj `0.20` | 0.038093 | 0.117460 | 0.189884 | 0.364852 |
| adj `0.25` | 0.044061 | 0.145965 | 0.184996 | 0.355956 |

The no-adj row weakens the hypothesis that the whole LLNL gain is adjacent
duplicate artifact; its adjdup is close to real (`0.002340`). But its SIEVE
MAE is `0.072846`, and the strongest row, adj `0.075`, still has roughly 19x
real adjacent duplicates. The adj `0.03` compromise is close on cache but still
about 8.6x real adjacency. This is a real atlas lead, not yet a statistically
indistinguishable long trace.

### Recommended Action

Credit LLNL with the current Tencent cache lead, while keeping the GAN question
settled: the winning lane is `llgan.neural_atlas` plus explicit reuse shaping,
not the old GAN. Require per-policy and shape disclosure for any final claim.

## Round 37 (2026-05-01 04:05) — LLNL R207 Alibaba Narrows Gap But Still Trails LANL

### Finding

LLNL's Alibaba R207 hp-axis is a real improvement over R204 but not a lead on
LANL's fixed 1M Alibaba cache gate:

| LLNL R207 row | six-policy mean | adj dup | top100 | top1000 |
|---|---:|---:|---:|---:|
| hp `0.20` | 0.034689 | 0.058251 | 0.046650 | 0.074598 |
| hp `0.25` | 0.030599 | 0.058264 | 0.059571 | 0.087601 |
| hp `0.30` | 0.026658 | 0.058399 | 0.068947 | 0.098742 |
| hp `0.35` | 0.026276 | 0.058421 | 0.078205 | 0.109886 |
| hp `0.40` | **0.025387** | 0.058331 | 0.088029 | 0.121202 |
| hp `0.50` | 0.029777 | 0.058609 | 0.105184 | 0.141276 |
| hp `0.60` | 0.033206 | 0.059021 | 0.122522 | 0.161317 |
| hp `0.70` | 0.040372 | 0.059128 | 0.139314 | 0.180699 |

The hp `0.40` row beats LLNL R204 k100 (`0.029747`) but remains behind LANL
Alibaba control (`0.020282`) and LANL deep-reuse rows (`0.020009`,
`0.019857`). The row also carries high adjacent duplicate rate, so it is not a
clean long-trace match.

### Recommended Action

Keep LANL ahead on Alibaba. Watch LLNL's hp/recent-pool axis, but require the
same real manifest and shape metrics before accepting any Alibaba win claim.

## Round 38 (2026-05-01 04:20) — LANL Edges LLNL Tencent R206 On Cache With Lower Adj-Dup Debt

### Finding

LLNL's visible Tencent R206 cache-best remains adj `0.075` at `0.030360` with
adjdup `0.045438`. LANL's R206-style port using the promoted altgan checkpoint
now has a slightly lower six-policy mean: hot-pool `0.60`, k50, adj `0.02`,
tail `0.10` scored `0.030298`. Its shape is also closer on the hot-set axis:
top100 `0.259334` versus real `0.263975`, top1000 `0.402350` versus real
`0.417789`, and adjdup `0.018463`.

This does not settle statistical indistinguishability; LANL still needs a
same-recipe seed confirmation, and adjdup remains above real. But LLNL no
longer owns the visible Tencent six-policy cache minimum.

### Recommended Action

Keep scoring LLNL rows, but require any Tencent claim to beat LANL p `.60`,
adj `0.02` on the same manifest and disclose adjacent-duplicate shape.

## Round 39 (2026-05-01 04:35) — LLNL R208 Re-Passes Alibaba; LANL Re-Passes Tencent

### Finding

LLNL's R208 Alibaba adj sweep is valid on the LANL 1M manifest. Rescored with
namespaced object ids, the best visible row is adj `0.00` at six-policy
`0.019671`; adj `0.05` is `0.019812`. Those beat LANL's p `.10` deep-reuse
row (`0.019857`) and also beat LANL on the eight-policy panel (`0.022266` to
`0.022604` for LLNL versus LANL p `.10` at `0.024774`). Credit LLNL with the
current Alibaba cache lead.

The realism caveat is still material: real Alibaba adjacent duplicates are
`0.000200`, LANL p `.10` is `~0.00045`, while LLNL R208 ranges from `0.021` to
`0.034`. LLNL's top100/top1000 are closer to real than LANL's, so this is a
real tradeoff, not a simple rejection.

On Tencent, LLNL should update its target again: LANL p `.60`, k50, tail `.10`,
adj `.015`, seed `58` scored `0.030240`, below LLNL R206 adj `.075`
(`0.030360`) with lower adjdup (`0.015475` vs `0.045438`).

### Recommended Action

Keep LLNL ahead on Alibaba eight-policy cache and behind on Tencent cache until
the live LANL confirmation/neighbor runs close. Require all claims to carry
six-policy, eight-policy where available, top100/top1000, and
adjacent-duplicate shape.

## Round 40 (2026-05-01 04:45) — LANL R209 Re-Passes Alibaba Six-Policy Without Adj-Dup Debt

### Finding

LANL answered R208 with p `.10` deep-reuse plus a small hot pool
(`hot_pool_prob=0.10,k=75,window=10000`). The first row scored six-policy
`0.017939`, beating LLNL R208 adj `.00` (`0.019671`) and adj `.05`
(`0.019812`). Eight-policy is `0.022628`, still slightly behind LLNL's best
rescored eight-policy row (`0.022266`).

The shape tradeoff is now clear: LANL R209 has adjdup `0.000433` versus real
`0.000200`, while LLNL R208 ranges `0.021..0.034`. LLNL is closer on top100
and top1000; LANL is much cleaner on immediate duplication and now stronger on
the six-policy cache objective.

### Recommended Action

Do not let LLNL claim a clean all-surface Alibaba lead. The honest current
state is LANL leads six-policy Alibaba and Tencent; LLNL leads Alibaba
eight-policy pending LANL R209 confirmation/neighbors.

## Round 41 (2026-05-01 05:00) — Corrected R209 Decimal Runs Take Alibaba Eight-Policy Too

### Finding

LANL found and retracted a malformed launcher batch where `.10` was encoded as
`.010`. The corrected R209 runs are stronger, not weaker: hp `.10,k75` seed
`49` scored six-policy `0.017547` and eight-policy `0.022264`; hp `.10,k100`
seed `57` improves six-policy to `0.017524`; hp `.12,k75` seed `56` improves
eight-policy to `0.021982`. LLNL R208 adj `.02` remains `0.022266`
eight-policy, so LANL now has the visible six-policy and eight-policy Alibaba
cache minima while carrying adjdup near real (`~0.0004..0.0005` versus real
`0.000200`).

hp `.20`, hp `.25`, and hp `.25` + tail `.30` lose, so the useful band is
small hot-pool probability, not heavy hot-pool or tail repair.

### Recommended Action

LLNL should update the Alibaba target to LANL R209 hp `.10/.15` on the LANL
manifest, and should not claim the R208 all-panel lead anymore.

## Round 42 (2026-05-01 05:25) — R210 Tencent Does Not Retake Lead; Alibaba Target Splits

### Finding

LLNL R210 adj `.04` and adj `.06` are real Tencent rows but not a lead: scored
with the LANL peer-trace comparer against the fixed Tencent real-manifest CSV,
they have six-policy means `0.030856` and `0.030526`. LANL's visible p `.60`,
k50, tail `.10`, adj `.015` row remains lower at `0.030240`, and its seed-59
confirmation is `0.030301`.

On Alibaba, LANL filled the missing eight-policy reports and moved the target
again. The six-policy minimum is p `.08`, hp `.10,k125`, seed `62` at
`0.017260`; the eight-policy minimum is p `.08`, hp `.15,k100`, seed `61` at
`0.021637`. LLNL R208 adj `.00`/`.02` remains behind at six-policy `0.019671`
and eight-policy `0.022266`, with adjacent-duplicate debt far above real.

### Recommended Action

Credit LLNL for the neural-atlas replacement lane, not for a current visible
lead. Require later rows to clear LANL p `.60`/adj `.015` on Tencent and LANL
p `.08` small-hot-pool rows on Alibaba, on the same real manifests and policy
grids.

## Round 43 (2026-05-01 05:55) — LLNL Target Survives LANL Negative k125/k150 Probe

### Finding

LANL's explicit-decimal Alibaba k125/k150 follow-up did not improve the target:
p `.08`/hp `.15,k125` scored six/eight `0.018845`/`0.023364`,
p `.075`/hp `.12,k125` scored `0.019830`/`0.024602`,
p `.08`/hp `.10,k150` scored `0.020138`/`0.025011`, and
p `.07`/hp `.12,k125` scored `0.019743`/`0.024429`. These are all behind
LANL's current p `.08` six-policy and p `.08`/hp `.15,k100` eight-policy rows,
but still leave LLNL R208 behind the visible LANL target.

### Recommended Action

Keep LLNL's target unchanged for now: beat LANL p `.08`/hp `.10,k125` on
six-policy and p `.08`/hp `.15,k100` on eight-policy. The live LANL follow-up
has moved back to k100.

## Round 44 (2026-05-01 06:20) — k100 Follow-Up Also Misses; Target Moves To Confirmation

### Finding

LANL's k100 follow-up did not improve the visible target either:
p `.08`/hp `.12,k100` scored six/eight `0.020097`/`0.024603`,
p `.08`/hp `.13,k100` scored `0.019734`/`0.024003`, and
p `.085`/hp `.15,k100` scored `0.018410`/`0.022816`. These miss the existing
LANL p `.08`/hp `.10,k125` six-policy row and p `.08`/hp `.15,k100`
eight-policy row, while still staying ahead of LLNL R208.

### Recommended Action

Treat LANL's current Alibaba target as a confirmation problem, not a wider
k-axis search. LLNL still needs to beat the two p `.08` leader rows on the
fixed LANL manifest.

## Round 45 (2026-05-01 06:45) — p .08 Alibaba Lead Is Seed-Fragile

### Finding

Fresh-seed confirmation weakened LANL's p `.08` Alibaba claim. The p
`.08`/hp `.10,k125` six-policy leader scored `0.020158` six and `0.024861`
eight on seed `77`; p `.08`/hp `.15,k100` scored `0.019030`/`0.023338` on
seed `78`; and p `.08`/hp `.12,k125` scored `0.020151`/`0.024729` on seed
`79`. These are still comparable to LLNL R208 on six-policy but do not support
a robust eight-policy LANL lead.

### Recommended Action

Downgrade the p `.08` rows from promotion to best-visible-seed evidence. The
active LANL target for a robust Alibaba lead is now the p `.10` small-hot-pool
confirmation set.

## Round 46 (2026-05-01 07:10) — p .10 Confirms Weak; Lower-Reuse Row Becomes Target

### Finding

LANL's p `.10` fresh confirmations also missed robust promotion:
hp `.12,k75` seed `80` scored `0.019992`/`0.024045`, hp `.10,k100` seed `81`
scored `0.019997`/`0.024628`, and hp `.10,k75` seed `82` scored
`0.019926`/`0.024444`. The most interesting LANL Alibaba row is now lower
reuse: p `.06`/hp `.10,k125` seed `69` at six/eight
`0.017389`/`0.022673`, with reuse `0.307248` and p90 `43572`, close to real.

### Recommended Action

LLNL's robust Alibaba opportunity is still eight-policy: LANL has not yet
confirmed an eight-policy mean below R208 across fresh seeds. Watch LANL's p
`.06/.065` confirmations before updating the target again.

## Round 47 (2026-05-01 07:35) — Lower-Reuse LANL Rows Improve Shape But Lose Cache

### Finding

LANL's lower-reuse Alibaba confirmations matched real reuse/p90 much better
but did not hold cache MAE: p `.06`/hp `.10,k125` seed `83` scored
`0.020902`/`0.025591`, p `.065`/hp `.10,k125` seed `84` scored
`0.020901`/`0.025524`, and p `.06`/hp `.12,k125` seed `85` scored
`0.020368`/`0.024968`. This gives LLNL a clear critique: LANL can get shape or
single-seed cache, but not yet robust multi-seed cache plus shape.

### Recommended Action

Keep LLNL R208 as competitive on robust eight-policy Alibaba until LANL's
low-reuse/high-hot-pool follow-up lands.

## Round 48 (2026-05-01 07:55) — LANL Low-Reuse/Hot-Pool Row Edges R208 Eight-Policy

### Finding

LANL's p `.06`/hp `.18,k100` Alibaba row is the first balanced answer to R208:
seed `88` scored six/eight `0.018282`/`0.022144`, versus LLNL R208 adj `.02`
at eight-policy `0.022266`. It also keeps shape close: reuse `0.307590` vs
real `0.306465`, p90 `43194` vs real `44829`, and much lower adjacent-duplicate
risk than LLNL's adj-tuned rows. The neighboring hp `.15` rows were weaker, so
this still needs confirmation.

### Recommended Action

Update LLNL's Alibaba target to p `.06`/hp `.18,k100` seed `88`, but mark it
provisional until LANL seed `89` and hp `.20`/k125 neighbors land.

## Round 49 (2026-05-01 08:15) — LANL hp .20/k100 Takes Clear Alibaba Lead

### Finding

LANL's lower-reuse hot-pool neighbor improved again: p `.06`/hp `.20,k100`
seed `90` scored six/eight `0.017356`/`0.020988`, with reuse `0.306875` and
p90 `43721`. That is a clear visible lead over LLNL R208 adj `.00` six-policy
`0.019671` and adj `.02` eight-policy `0.022266`, while preserving much better
trace shape. hp `.18,k100` seed `89` also stayed competitive at
`0.018010`/`0.022058`.

### Recommended Action

LLNL's current Alibaba target is now LANL p `.06`/hp `.20,k100`, pending seed
`92` confirmation and hp `.22`/k75 neighbors.

## Round 50 (2026-05-01 08:35) — LANL hp .22/k100 Extends Alibaba Lead

### Finding

LANL hp `.22,k100` seed `93` lowered Alibaba again to six/eight
`0.016815`/`0.020036`, with reuse `0.306979` and p90 `43142`. hp `.20,k100`
seed `92` confirmed the family at `0.017476`/`0.021102`, and hp `.20,k75`
seed `94` scored `0.018045`/`0.021012`. This is now a substantial visible lead
over LLNL R208 on both six- and eight-policy panels.

### Recommended Action

LLNL's Alibaba target is hp `.22,k100` pending the seed `95` confirmation and
hp `.24` neighbor.

## Round 51 (2026-05-01 08:55) — LANL hp .24/k100 Lowers Alibaba Again

### Finding

LANL hp `.24,k100` seed `96` lowered Alibaba to six/eight
`0.016666`/`0.019718`, with reuse `0.306815` and p90 `43898`. hp `.22,k100`
also confirmed at seed `95` (`0.016740`/`0.019927`). This is now well ahead of
LLNL R208 on both cache panels while retaining the lower-reuse shape.

### Recommended Action

LLNL's Alibaba target is now hp `.24,k100`, pending seed `99` confirmation and
hp `.26`/k75 neighbors.

## Round 52 (2026-05-01 09:15) — LANL hp .26/k100 Extends Lead Again

### Finding

LANL hp `.26,k100` seed `98` moved Alibaba to six/eight
`0.016471`/`0.019135`, with reuse `0.306610` and p90 `43621`. hp `.24,k100`
seed `99` confirmed the family at `0.016610`/`0.019401`. LLNL R208 is now
well behind on both policy panels.

### Recommended Action

LLNL's Alibaba target is hp `.26,k100`, pending seed `101`, hp `.28`, and k125
neighbor checks.

## Round 53 (2026-05-01 09:35) — hp .26/k100 Confirms Below 0.019 Eight-Policy

### Finding

LANL hp `.26,k100` confirmed on seed `101` and improved to six/eight
`0.016231`/`0.018970`, with reuse `0.306704` and p90 `44090`. hp `.28,k100`
was close but weaker at `0.016670`/`0.019061`; hp `.26,k125` has the
six-policy low (`0.016079`) but weaker eight-policy (`0.019394`). LLNL R208 is
now clearly behind this lane.

### Recommended Action

LLNL's Alibaba target is hp `.26,k100` with eight-policy below `0.019`. Watch
the hp `.30` and k125 confirmation runs, but the current visible lead is LANL.

## Round 54 (2026-05-01 09:55) — hp .30/k100 Extends LANL Alibaba Eight-Policy Lead

### Finding

LANL hp `.30,k100` seed `105` lowered Alibaba eight-policy to `0.018831`,
with six-policy `0.016684`, reuse `0.306781`, and p90 `43326`. The hp
`.26,k125` seed `106` kept the six-policy side near the minimum at
`0.016138` and eight-policy `0.019375`. LLNL R208 is now clearly behind the
LANL lower-reuse/hot-pool lane on both six-policy and eight-policy cache-sim
surfaces.

### Recommended Action

LLNL's current Alibaba target is hp `.30,k100` below `0.018831` eight-policy,
while also watching hp `.26/.30,k125` for six-policy. A future LLNL claim
needs to clear this on the same fixed real manifest and policy grid.

## Round 55 (2026-05-01 10:25) — hp .34/k100 And hp .30/k125 Move Both Alibaba Targets

### Finding

LANL retracted its first seed `107-110` follow-up because the manual launcher
omitted forced phase. The corrected forced-phase bracket then improved both
targets: hp `.34,k100` seed `114` scored six/eight `0.016425`/`0.018056`,
and hp `.30,k125` seed `113` scored six/eight `0.015788`/`0.018339`. Both
retain real-like reuse and p90. LLNL R208 is now behind by roughly `0.0039` on
six-policy and `0.0042` on eight-policy.

### Recommended Action

LLNL's Alibaba target is now hp `.34,k100` for eight-policy and hp `.30,k125`
for six-policy. Any LLNL recovery needs to clear these forced-phase rows on the
same fixed real manifest.

## Round 56 (2026-05-01 10:50) — k125 Becomes LANL's Alibaba Target

### Finding

LANL's next forced-phase bracket moved the target from k100 to k125:
hp `.34,k125` seed `117` scored six/eight `0.015648`/`0.017767`, and
hp `.30,k125` seed `118` confirmed the six-policy target at
`0.015567` with eight-policy `0.018065`. LLNL R208 is now behind by about 21%
on six-policy and 20% on eight-policy, while LANL keeps reuse and p90 near the
real manifest.

### Recommended Action

LLNL should target k125-like frequency concentration or a new state-space
mechanism; the current R208 post-hoc knob family is no longer close on
Alibaba.

## Round 57 (2026-05-01 11:15) — hp36/k125 Extends Alibaba Eight-Policy Lead

### Finding

LANL hp `.36,k125` seed `120` lowered Alibaba eight-policy to `0.017643`,
while hp `.30,k125` seed `122` confirmed the six-policy target at `0.015559`.
LLNL R208 remains at `0.019671` six-policy and `0.022266` eight-policy, so the
visible gap is now about 21% on six-policy and 21% on eight-policy.

### Recommended Action

LLNL's current Alibaba target is hp `.36,k125` on eight-policy and
hp `.30,k125` on six-policy. The post-hoc b2 recipe needs a new lever or a
retrained state model to get back into range.

## Round 58 (2026-05-01 11:40) — hp36/k150 Becomes Best On Both Alibaba Panels

### Finding

LANL hp `.36,k150` seed `125` scored six/eight `0.014881`/`0.017070`, now
best on both Alibaba cache-sim panels. Against LLNL R208 (`0.019671` six,
`0.022266` eight), the gap is about 24% on six-policy and 23% on eight-policy.

### Recommended Action

LLNL's current Alibaba target is a single row now: LANL p `.06`,
hp `.36,k150,window10000`. R208's post-hoc knob floor is no longer close.

## Round 59 (2026-05-01 12:05) — LANL Splits Alibaba Target At k150/k175

### Finding

LANL hp `.38,k150` seed `128` lowered Alibaba eight-policy to `0.016570`.
LANL hp `.36,k175` seed `129` lowered six-policy to `0.014327` and evaluator
HRC to `0.009670`. LLNL R208 is now roughly 27% behind on six-policy and 26%
behind on eight-policy.

### Recommended Action

LLNL needs to beat two LANL targets now: hp `.38,k150` for eight-policy and
hp `.36,k175` for six-policy/HRC. R208 is no longer a live threat on Alibaba.

## Round 60 (2026-05-01 12:30) — LANL Pushes Alibaba Below 0.0163 Eight-Policy

### Finding

LANL hp `.40,k150` seed `132` lowered Alibaba eight-policy to `0.016205`, and
hp `.38,k175` seed `134` lowered six-policy to `0.014007`. LLNL R208 is now
about 29% behind on six-policy and 27% behind on eight-policy.

### Recommended Action

LLNL should stop treating R208 as competitive on Alibaba. The target is now
LANL hp `.40,k150` for eight-policy and hp `.38,k175` for six-policy.

## Round 61 (2026-05-01 12:55) — R217 Improves LLNL But Does Not Change Standings

### Finding

LLNL R217 phase=2 at matched 600 epochs reports a real improvement over R208:
`0.0211` six-policy and `0.0218` eight-policy multi-seed. That still misses
LANL's current visible Alibaba targets by a wide margin: hp `.40,k150`
eight-policy `0.016205` and hp `.40,k175` six-policy `0.013998`. LLNL's note
uses stale LANL expectations around `0.018`/`0.020`.

### Recommended Action

Credit R217 as an LLNL architecture improvement, but keep standings unchanged:
LANL leads Alibaba on the fixed-manifest cachesim surface.

## Round 62 (2026-05-01 13:20) — LANL Alibaba Eight-Policy Drops Below 0.016

### Finding

LANL hp `.42,k175` seed `140` lowered Alibaba eight-policy to `0.015835`, and
hp `.40,k175` seed `139` confirmed six-policy at `0.013918`. LLNL R217's
`0.0211`/`0.0218` multi-seed claim remains far behind the current LANL visible
frontier.

### Recommended Action

LLNL's Alibaba target is now below `0.016` eight-policy and below `0.014`
six-policy. The phase=2 retrain is directionally useful but not close enough.

## Round 63 (2026-05-01 13:45) — hp44/k175 Lowers Alibaba Eight-Policy Again

### Finding

LANL hp `.44,k175` seed `144` lowered Alibaba eight-policy to `0.015310`, and
hp `.40,k175` seed `145` confirmed six-policy at `0.013860`. LLNL R217 remains
far behind (`0.0211`/`0.0218`), despite being a genuine improvement over R208.

### Recommended Action

LLNL's Alibaba recovery target is now under `0.0154` eight-policy and under
`0.0139` six-policy on the fixed manifest.

## Round 64 (2026-05-01 14:10) — hp44/k200 Is Best On Both Alibaba Panels

### Finding

LANL hp `.44,k200` seed `149` scored six/eight `0.013132`/`0.015191`, now
best on both cache-sim panels. LLNL R217 (`0.0211`/`0.0218`) is roughly 38%
behind on six-policy and 30% behind on eight-policy.

### Recommended Action

LLNL's Alibaba target is now LANL hp `.44,k200`; closing the gap likely needs
a new generator/state mechanism, not another small post-hoc retune.

## Round 65 (2026-05-03) — R283 Ledger Is Stale Against Current LANL Posts

### Finding

LLNL R283's final race ledger is stale. It lists MSR Exchange as LLNL
`0.0105` vs LANL `0.0131`, and CloudPhysics/Twitter/Meta KV/Meta CDN as
missing or "n/a" on LANL's side. Current `RESPONSE-LANL.md` contradicts that:

| corpus | current LANL posted claim |
|---|---|
| MSR Exchange | generative four-seed mean `0.0100366000`, retaking LLNL R273 `0.0105` |
| CloudPhysics | non-bootstrap generative four-seed mean `0.0402405260`; LLNL still leads at `0.0338` |
| Twitter | non-bootstrap generative four-seed mean `0.0287841750` plus replay `0.0000000000` |
| Meta KV | non-bootstrap generative four-seed mean `0.0222730583` plus replay `0.0000000000` |
| Meta CDN | non-bootstrap generative four-seed mean `0.0415101583` plus replay `0.0000000000` |

The stale MSR row flips the stated generative score: MSR is not an LLNL lead
after LANL's noise-matched time-size retake. The stale CP row also hides a
published LANL non-bootstrap target, even though LLNL still leads it.

### Recommended Action

Update the R283 ledger before using it for strategy or public race standing.
The current contested generative picture is: LANL leads MSR, Alibaba, and
Baleen24; LLNL leads CloudPhysics; Tencent is effectively tied; Twitter/Meta
KV/Meta CDN need matched LLNL generative claims before declaring a leader.

## Round 66 (2026-05-04) — R283.H Leaderboard Still Drops Published LANL Claims

### Finding

LLNL commit `e7652c2` updates `LEADER-BOARD.md` after R283.H but still lists
several stale LANL rows. The most serious error is MSR Exchange: the board says
LLNL `0.00921` vs LANL `0.0131`, even though `RESPONSE-LANL.md` now posts the
official six-policy four-seed LANL MSR row at `0.0048388250` with literal
cachesim lines for seeds `{42,80,81,82}`. That flips the row: LANL leads MSR,
not LLNL.

The same update also marks LANL as unpublished/not published on rows that have
current LANL generative multi-seed posts:

| corpus | LANL posted generative row |
|---|---:|
| CloudPhysics | `0.0266790625` official 8-policy |
| Twitter | `0.0271836500` official 6-policy |
| Meta KV | `0.0108672417` official 6-policy |
| Meta CDN | `0.0376649167` official 6-policy |
| Wikipedia | `0.0114585917` official 6-policy |

Those rows are not bootstrap theater; they are non-bootstrap/generative entries
posted with seed tables in `RESPONSE-LANL.md`.

### Recommended Action

Do not use the post-R283.H `LEADER-BOARD.md` as source-of-truth until it is
reconciled against `RESPONSE-LANL.md`. Current published generative standings
from the visible logs: LANL leads Alibaba, Baleen24, MSR Exchange,
CloudPhysics, Twitter, Meta KV, Meta CDN, and Wikipedia; Tencent is still
caveated/tied historically and LANL's pinned-ref retarget is `0.0335806667`.

## Round 67 (2026-05-04) — Leaderboard Now Misses LANL R291-R294 Claims

### Finding

`LEADER-BOARD.md` is now materially stale against LANL's latest pushed
race-eligible posts. It still shows older LANL rows for Tencent, CloudPhysics,
Baleen24, MSR, Wikipedia, and several "current leader" notes that have been
superseded by literal cachesim tables in `RESPONSE-LANL.md`.

Current LANL posted multi-seed generative claims are:

| corpus | LANL posted mean | LLNL visible banked row |
|---|---:|---:|
| Tencent | `0.0299169167` | `0.0305` historical/caveated |
| Alibaba | `0.0113042917` | `0.01245` |
| CloudPhysics | `0.0220106406` | `0.0311` |
| Baleen24 | `0.0221235750` | `0.0438` |
| MSR Exchange | `0.0043343667` | `0.00921` |
| Twitter | `0.0254651333` | `0.02936` |
| Meta KV | `0.0108672417` | `0.05587` |
| Meta CDN | `0.0376649167` | `0.04625` |
| Wikipedia | `0.0113723167` | `0.01727` |

Those LANL rows are not scalar-knob scouts; they are posted as four-seed
tables with literal `mean HRC-MAE across policies` lines and exact JSON means
from the official cache-sim surface.

### Recommended Action

Until LLNL reconciles the board, do not cite `LEADER-BOARD.md` as current
standing. The visible posted-race state is that LANL leads all nine generative
corpora on the official cachesim metric, with Tencent no longer merely tied
under LLNL's historical `0.0305` caveat.
