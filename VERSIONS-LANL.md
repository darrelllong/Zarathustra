# LANL / altgan Version Notes

This file tracks LANL-owned `altgan/` checkpoints and race-relevant updates.
The detailed LANL result ledger remains [altgan/RESULTS.md](/Users/darrell/Zarathustra/altgan/RESULTS.md).

---

## R215 Alibaba hp36 k150 Cache-Sim Target (2026-05-01)

LANL's active Alibaba lane is now lower-reuse PhaseAtlas plus higher hot-pool
pressure, evaluated through `tools/cachesim` on the fixed 1M Alibaba real
manifest. The current visible targets are:

- Six- and eight-policy: p `.06`, hp `.36,k150,window10000`, seed `125`,
  six-policy `0.014881`, eight-policy `0.017070`, evaluator HRC `0.010769`,
  reuse `0.306794`, median `254`, p90 `44358`.

LLNL R208 remains the best visible LLNL Alibaba row (`0.019671` six-policy,
`0.022266` eight-policy), so LANL currently leads the fixed-manifest Alibaba
cachesim table. The seed `107-110` manual bracket is retracted because it
omitted `--force-phase-schedule`; the corrected forced-phase bracket is
recorded here. `altgan/launch_alibaba_cachesim_bracket.py` now owns the
Alibaba launch recipe. The live bracket is hp `.36,k150` seed `127`,
hp `.38,k150` seed `128`, hp `.36,k175` seed `129`, and hp `.34,k150` seed
`130`.

---

## R208/R209 Race State: Tencent Edge, Alibaba Re-Flip (2026-05-01)

LLNL's R208 Alibaba adj-dup re-sweep is a real cache-sim advance. LANL rescored
the visible `/home/darrell/alibaba_b2_r208_adj*.csv` rows against the same
LANL 1M Alibaba real manifest and the same fixed caps:

- LLNL R208 adj `0.00`: six-policy `0.019671`, eight-policy `0.022604`.
- LLNL R208 adj `0.02`: six-policy `0.019844`, eight-policy `0.022266`.
- LLNL R208 adj `0.05`: six-policy `0.019812`, eight-policy `0.022357`.
- LANL Alibaba p `0.10` deep-reuse remains `0.019857`/`0.019892` six-policy
  and `0.024774`/`0.024839` eight-policy.
- LANL R209 p `0.10` deep-reuse plus hot-pool `0.10,k75,window10000` scored
  `0.017939` six-policy and `0.022628` eight-policy with adjdup `0.000433`.
  This was a strong single-seed re-pass on the six-policy gate without buying
  LLNL's adjacency debt. The hot-set is still low: top100 `0.014995` versus
  real `0.042228`.
- A first follow-up launcher accidentally encoded p `.10` as `.010` and
  hot-pool `.25` as `.025`; those rows are not comparable to R209 and are not
  promoted. Corrected decimal-string follow-ups confirm the useful band:
  hp `.10` seed `49` scored `0.017547` six-policy and `0.022264` eight-policy
  with adjdup `0.000404`; hp `.15` seed `50` scored `0.018764` six-policy and
  `0.022021` eight-policy with adjdup `0.000494`.
- The realism caveat now cuts both ways: real Alibaba adjdup is `0.000200`,
  LANL R209 is `0.000433`, and LLNL R208 is `0.021..0.034`. LLNL's top100 is
  closer to real, while LANL is much cleaner on immediate duplication.
- hp `.20`/`.25` and tail `.30` are worse (`0.020120`, `0.022097`, and
  `0.038220`), so tail repair is closed negative at that strength. The tighter
  bracket found hp `.12,k75` at `0.017879` six-policy and a new LANL
  eight-policy best `0.021982`, while hp `.10,k100` scored the best
  six-policy row so far: `0.017524`, median `277` versus real `276`.

Tencent moved the other way. LANL p `.60`, k50, tail `.10`, adj `.015`, fake
seed `58` scored a new visible Tencent best: six-policy `0.030240`, evaluator
HRC `0.008832`, median `81`, p90 `33757`, top100 `0.260899`, top1000
`0.402591`, adjdup `0.015475`. This beats LLNL R206 adj `.075` (`0.030360`)
with lower adjacent-duplicate debt. Confirmation seed `59` and lower-adj
neighbor adj `.010` seed `60` are running.

---

## Alibaba 1M Cachesim Gate Opened (2026-05-01)

LANL opened a 1M Alibaba cache-simulator gate with a fixed real manifest:
`/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_manifest.json`.
The first LANL control uses the existing
`alibaba_phaseatlas_marks_e20.pkl.gz` checkpoint with reservoir marks,
`transition_blend=0.2`, `local_prob_power=0.9`, and forced phase.

Current Alibaba status:
- LANL control scores six-policy cachesim mean `0.020282` on the 1M Alibaba
  manifest.
- LLNL's visible R204 Alibaba neural-atlas k-axis scores `0.050148` at `k=25`,
  `0.033206` at `k=75`, and `0.029747` at `k=100` on the same LANL real
  manifest. LLNL's active replacement for the GAN track is therefore a
  conditional neural atlas plus hand-shaped reuse controls.
- LLNL's R207 Alibaba hp-axis narrows the gap but remains behind: hp `0.40`
  scores `0.025387` on the same manifest, ahead of its R204 rows but behind
  LANL control `0.020282` and LANL deep-reuse `0.019857`/`0.019892`.
- An LLNL-shaped LANL hot-pool test (`p=0.60`, `k=25`, adjacent duplicate
  `0.15`) closes negative at `0.070192`.
- Alibaba's live LANL branch is deep new-to-reuse injection. `p=0.06`,
  `min_rank=32768`, `rank_power=2.0` matches the real reuse tail closely and
  scores `0.020009`/`0.020072` on fake seeds `44`/`42`; `p=0.10` scores
  `0.019857`/`0.019892` on fake seeds `45`/`42` but overshoots reuse and p90.
  Current cache best is `p=0.10`; current trace-shape compromise is `p=0.06`.

---

## Tencent 1M Cachesim Gate Added To LANL Evaluator (2026-04-30)

LANL now treats `tools/cachesim` as a required promotion surface for the 1M
Tencent trace. The evaluator can write the exact fake and sampled-real CSVs and
run the simulator directly via `--cachesim-bin`.

Current status:
- Best LANL six-policy fixed-cap row is the hot-pool/window branch around
  `p=0.38..0.39`, `k=100`, `window=10000`, plus deep reuse injection. The best
  single row is `0.045219`; fresh fake seeds put the robust p38 window row at
  `0.045614`/`0.045511`.
- Rank phase scale `1.2,1.2,1.3,1.3` improves the old evaluator HRC-MAE from
  `0.051810` to `0.044706`, and exactly matches stack median `84`, but worsens
  the six-policy mean to `0.055905`; it is a diagnostic row, not promoted.
- LLNL's reported R182 Tencent six-policy mean is `0.0925`, so LANL is still
  ahead on this simulator surface. The remaining LANL error is policy-shaped:
  LRU/FIFO want rank scaling, ARC/SIEVE/CAR do not.
- LLNL's newer R203/R206 Tencent atlas rows change the scoreboard. R203 k25
  scored `0.038256`; R206 k50 adj `0.075` scores `0.030360`, ahead of LANL's
  current `0.0452..0.0456` band but with adjacent duplicate rate `0.045438`
  versus real `0.002340`. R206 adj `0.03` is the current compromise
  (`0.031474`, adjdup `0.020170`); adj `0.00` is much cleaner on adjacency
  (`0.003165`) but scores `0.043287` and has bad SIEVE error. This is an atlas
  lead, not a revived GAN lead.
- LANL's first k25/adj `0.15` clone failed (`0.107924`, stack median `39`), but
  the R206-style k50/tail pair transferred: adj `0.05` scored `0.031461`, adj
  `0.00` scored `0.031040` with median exactly `84` and adjdup `0.004993`, and
  adj `0.02` improved to `0.030632` with adjdup `0.018330`. Tail `0.08` closed
  negative at `0.032613`; adj `0.03`/`0.04` also closed slightly worse
  (`0.030802`/`0.030963`). Hot-pool `0.60`/adj `0.02` now scores `0.030298`,
  edging LLNL's visible `0.030360` while carrying less adjacent-duplicate debt.
  Live follow-ups are a same-recipe confirmation and hot-pool `0.60`/adj
  `0.015`.
- `stack_adj_dup_prob` exists for controlled tests, but LANL fake already has
  more adjacent duplicates than real on this slice (`0.00427` vs `0.00234`).
- `stack_hot_pool_prob` is now the live branch. `p=0.50` raises top-100 access
  share from `0.003849` to `0.119549` and cuts SIEVE HRC-MAE to `0.033573`;
  `p=0.60` and `p=0.70` both lost to `p=0.50`, so the live branch keeps
  `p=0.50`. Frequency weight power `2.0` is closed negative; it collapsed
  stack median to `14` and raised adjacent duplicates to `0.116900`.
- `window=50000` is closed operationally negative for now: exact lookup ran
  40+ minutes with no fake CSV, and the cached-rank experiment also failed to
  land in a useful window. The implementation was reverted to exact hot-object
  lookup for the promoted `window=5000` row.
- `stack_hot_pool_max_search` now bounds exact hot-object lookup and falls back
  to the normal sampled rank when the hot object is deeper than the searched
  prefix. Seed-43 unbounded and `max_search=8192` runs were both killed at
  40+ minutes with no fake CSV; `max_search=512` was also killed at 40+
  minutes. Confirmation pivoted to fixed real manifest seed `42`; fake RNG seed
  `43` was also killed as slow, and fake RNG seed `44` is running.
- Hot-pool confirmations exposed the Python-list LRU stack as the runtime
  bottleneck. `altgan.neural_atlas` now uses `_RankedLRUStack`, an implicit
  treap with deterministic priorities, so rank moves and hot-object lookups are
  logarithmic while preserving the emitted trace contract.
- Fake-seed confirmation landed on the fixed seed-42 real manifest: `p=0.50`
  hot-pool with fast stack and fake seed `44` scored six-policy mean HRC-MAE
  `0.046945` (seed-42 row `0.046657`), evaluator HRC-MAE `0.038433`, reuse
  `0.729857` vs real `0.728415`, median `86` vs `84`, p90 `24815` vs `29150`,
  and mark score `0.036416`. This confirms the hot-pool row is not a
  single-fake-seed accident.
- Hot-pool bracket update: the current robust band is `p=0.37..0.40`.
  `p=0.38` is the best single row, scoring `0.045386` on fake seed `44` and
  `0.045648` on fake seed `42`. `p=0.37` seed `44` is essentially tied at
  `0.045395` and scored `0.045599` on seed `42`; `p=0.39` seed `44` scored
  `0.045532`; `p=0.40` confirmed at `0.045651`/`0.045660`; `p=0.35` and
  `p=0.42` both lose. The probability sweep is flat enough to stop.
- Pool-size result: keep `k=100`. At `p=0.38` on fake seed `44`, `k=75`
  regressed to `0.045715` and `k=150` regressed harder to `0.047746`; the
  original `k=100` row remains `0.045386`.
- Window result: `window=10000` improves the current best to `0.045255` on fake
  seed `44`; `window=2500` loses at `0.045842`. Confirmation `window=10000`
  seed-42 and wider probe `window=20000` seed-44 are running.
- Window follow-up: `window=10000` confirms on seed `42` at `0.045352`.
  `window=20000` edges seed `44` down to `0.045243`, but with more ARC/CAR
  pressure; seed-42 confirmation loses at `0.045465` and `window=40000`
  seed-44 loses at `0.045855`. Current promotion stays `window=10000`.
  Probability recheck at `window=10000` (`p=0.37`/`0.39`) is running.
- Probability recheck at `window=10000`: `p=0.39` is the seed-44 best at
  `0.045219`, but follow-up fake seeds `45`/`46` scored `0.045573`/`0.045627`.
  The result is not a clean promotion over `p=0.38`; `p=0.38,window=10000`
  same-seed checks on fake seeds `45`/`46` are running.

Code changes:
- `altgan.neural_atlas` keeps deep reuse boosts as post-decode trace
  corrections for transition rollout, while passing the emitted action to the
  mark runtime.
- `altgan.evaluate_neural_atlas` adds `--real-output`, `--cachesim-bin`, and
  six-policy report emission.
- `altgan.neural_atlas` adds `stack_hot_pool_*` controls for frequency-shape
  correction.
- `altgan.neural_atlas` replaces list-backed LRU generation with
  `_RankedLRUStack`; verified locally with randomized list-equivalence tests and
  `py_compile`.
- `altgan.evaluate_neural_atlas` adds `--progress-interval` so long 1M
  confirmations can emit per-stream generation progress into their logs.
- `altgan.neural_atlas` and `altgan.evaluate_neural_atlas` add
  `stack_tail_reuse_*` and `stack_recent_pool_*` controls so LANL can test the
  useful LLNL atlas levers inside the LANL evaluator and cache-sim path.

---

## Tencent Object Refinement Promotion Under Feedback 0.080 (2026-04-30)

The current LANL promotion is the e20 128-file h128 catw `0.25` neural mark
sidecar with feedback-only size log blend `0.080`, forced phase, late rank
scales `1.0,1.0,1.1,1.1`, `transition_blend=0.575`, and
`local_prob_power=0.70`.

Seeds `50-53` completed a 3x3 object micro-sweep around the current
`transition_blend=0.55`, `local_prob_power=0.8` row while holding the mark
runtime fixed. The best HRC cell was `transition_blend=0.525`,
`local_prob_power=0.825`: mean HRC-MAE `0.00873275`, fake reuse `0.61370`,
stack median `52.75`, stack p90 `165.25`, and mean mark score `0.02675281`.
The most balanced nearby cell was `transition_blend=0.575`,
`local_prob_power=0.75`: mean HRC-MAE `0.00876875`, fake reuse `0.61368`,
stack median `54`, stack p90 `170.5`, and mean mark score `0.02629462`.
The old object cell on the same seeds scored HRC-MAE `0.00925137` and mark
score `0.02670647`.

The seeds `42-45` confirmation grid changed the readout: the best HRC cell on
that panel was `transition_blend=0.575`, `local_prob_power=0.75` at mean
HRC-MAE `0.00848437`, but its mean mark score worsened to `0.02733755`. Across
the combined `42-45` plus `50-53` panels, `0.575/0.75` still wins HRC
(`0.00862656`) and improves stack p90 (`170.25` vs `168.75` for the old row),
while `0.55/0.75` is the cleaner mark compromise (`0.02651714`) with HRC
`0.00886219`. The old `0.55/0.8` row is `0.00900900` HRC and `0.02677647`
mark over the same eight seeds.

`sweep_mark_hybrids` now supports `--object-candidates` so exact object cells
can be evaluated without cross-product waste. The seeds `54-57` exact-pair
panel confirmed the shape: `0.575/0.75` won that panel on HRC at `0.00861063`,
while `0.55/0.75` won mark score at `0.02646826`. Across all twelve evaluated
seeds (`42-45`, `50-57`), `0.575/0.75` now beats the old object row on both
HRC and mark: HRC `0.00862125` vs old `0.00892654`, mark `0.02683659` vs old
`0.02701257`, with stack p90 `170.25` vs old `168.58`. The mark-favoring
`0.55/0.75` row scores HRC `0.00882121` and mark `0.02650085`.

A seeds `58-61` local-power refinement flipped the shared-seed readout again:
over the sixteen seeds where both are measured, `0.55/0.75` now edges
`0.575/0.75` on both HRC (`0.00871025` vs `0.00871844`) and mark
(`0.02671515` vs `0.02708219`). The old `0.55/0.8` row over the same sixteen
seeds is weaker at HRC `0.00890641` and mark `0.02704201`.

The new `0.575/0.70` point confirmed over seeds `42-57`; combined with the
original `58-61` panel, it has 20 seeds at mean HRC `0.00858638`, mark
`0.02694865`, reuse `0.61350`, stack median `53.75`, and stack p90 `169.8`.
On the same sixteen seeds where the current controls are fully measured
(`42-45`, `50-61`), it beats the old `0.55/0.8` row on HRC (`0.00853172` vs
`0.00890641`) and mark (`0.02683922` vs `0.02704201`), while the
mark-favoring `0.55/0.75` row remains lower on mark (`0.02671515`) but worse
on HRC (`0.00871025`).

During the `0.575/0.70` confirmation, a remote code clobber briefly reverted
`evaluate_neural_atlas.py` and removed the feedback CLI. LANL restored and
compiled the full `altgan/` tree, resumed with `--skip-existing`, and
spot-checked output metadata for feedback blend `0.08`.

A fresh seeds `62-65` interpolation panel reinforced `0.575/0.70` as the
current HRC-leading shared candidate. On the fair `42-45,50-65` seed set,
`0.575/0.70` scores HRC `0.00854965`, mark `0.02680073`, reuse `0.613864`,
stack median `53.85`, and p90 `170.3`; `0.55/0.75` scores HRC `0.00871607`
and mark `0.02678954`; old `0.55/0.8` scores HRC `0.00897950` and mark
`0.02697943`.

The interpolation panel also opened lower-transition challengers. The
confirmation run over seeds `42-61` closed them without dislodging the
promotion: `0.55/0.70` scored HRC `0.00856863`, mark `0.02690313`, reuse
`0.61286`, p90 `168.35`; `0.5625/0.70` scored HRC `0.00863293`, mark
`0.02686441`, reuse `0.61362`, p90 `169.5`. On the fair `42-45,50-65`
principal set, the promoted `0.575/0.70` row remains the HRC leader at
`0.00854965`, with mark `0.02680073`; the closest mark row, `0.55/0.75`,
scores mark `0.02678954` but weaker HRC `0.00871607`. The old `0.55/0.8`
row is worse on both HRC and mark over that fair set.

The first 1M-record, 4-stream panel exposed the next bottleneck. The promoted
row is slightly better than the old row on 1M HRC (`0.0589916` vs
`0.05981515`), but both miss the long-run reuse tail: real reuse is `0.72841`
while fake reuse stays near `0.613`, and real stack p90 is `29150` while the
unmodified fake p90 is only `170`. A tail-only rank stretch (`pivot=84`,
`scale=340`) moved fake p90 to `24224`, but worsened HRC to `0.08607485`
because total reuse remained capped. LANL then added controlled new-to-reuse
conversion in `altgan/neural_atlas.py`. The first bracket, boost probability
`0.30`, min-rank `84`, rank-power `2`, matched total reuse (`0.729707` fake
vs `0.728415` real) but still over-hit the low/mid HRC curve and left p90
short (`14132` vs `29150`). The active bracket keeps the conversion rate and
pushes injected reuses deeper (`min_rank=4096`) with `--fake-output` enabled
for `tools/cachesim`. A code follow-up makes converted events advance the
transition/mark runtime with their emitted reuse action and includes emitted
`stack_distance`/`action_class` in fake CSV rows.

Artifacts:
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_micro_seed50_53_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_micro_seed50_53_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_confirm_seed42_45_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_confirm_seed42_45_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_pairs_seed54_57_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_pairs_seed54_57_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_localrefine_seed58_61_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_localrefine_seed58_61_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_tb575_lp070_confirm_seed42_57_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_tb575_lp070_confirm_seed42_57_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_interp_seed62_65_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_interp_seed62_65_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_lowtrans_confirm_seed42_61_fb080_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_lowtrans_confirm_seed42_61_fb080_best.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_old_tb055_lp080_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_tailp84_tails340_seed42_eval_1M.json`

---

## Tencent Neural Mark Sidecar Lands (2026-04-29)

LANL completed the IDEA #53 follow-up around the strict-holdout PhaseAtlas
object-process winner. The promoted object recipe is forced phase,
`transition_blend=0.55`, `local_prob_power=0.8`, and late rank scales
`1.0,1.0,1.1,1.1`.

The mark winner is the older e20 neural-categorical sidecar rerun under the
fixed emitted-history rollout contract:
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20.pkl.gz`.
With reservoir numeric marks and neural categoricals (`mark_numeric_blend=0.0`,
log space, temp `1.0`, noise `0.05`), the four-seed confirmation over seeds
`42-45` preserved the object metrics exactly and improved mark quality:

| seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42-45 | **0.008767** | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | **0.02842** |

Paired controls:
- Reservoir marks: mean mark score `0.04684`.
- Categorical-heavy e30 sidecar: mean mark score `0.03941`.
- e20 fixed-history sidecar: mean mark score `0.02842`.

The seed-42 e20 categorical-temperature micro-sweep closed negative. Temps
`0.5`, `0.75`, `1.25`, and `1.5` all worsened mark score versus temp `1.0`;
the promoted mark setting remains `mark_categorical_temp=1.0`.

A follow-up data-size sidecar sweep promoted a narrower default-loss mark head:
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128.pkl.gz`.
The 512-file h128 sidecar closed negative on seed `42` (`mark_score=0.03838`),
and the 64-file h128 sidecar also closed negative on seed `42`
(`mark_score=0.03344`). The 128-file h128 sidecar improved every paired seed
without moving the object process. Across seeds `42-45`, it keeps mean HRC-MAE
`0.008767` and reduces mean mark score from `0.02842` to `0.02788`.

The objective-shape pass then promoted the same 128-file h128 sidecar with
lower categorical loss:
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`.
With `numeric_loss_weight=1.0` and `categorical_loss_weight=0.25`, seeds
`42-45` again keep mean HRC-MAE `0.008767`, fake reuse `0.61369` vs real
`0.61493`, stack median `53` vs `60`, and stack p90 `170` vs `174`; mean mark
score improves from the default 128-file sidecar's `0.02788` to `0.02775`.
Adjacent categorical weights closed negative on seed `42`: `0.125` scored
`0.04975`, and `0.375` scored `0.06088`.

A restored-code field-selective numeric pass first promoted emitted size-only
log blend `0.018`, then a feedback-only variant improved it further. The
current promoted runtime emits reservoir numeric marks
(`mark_numeric_blend=0.0`) but feeds the mark head size-only log blend `0.080`
for autoregressive feedback (`mark_feedback_numeric_fields=size`,
`mark_feedback_numeric_blend=0.080`). Seeds `42-45` keep the same object metrics
and reduce mean mark score to `0.02684646`; seed scores were `0.027173`,
`0.027603`, `0.026037`, and `0.026573`. Feedback-only dt and both-field
variants closed negative on seed `42`.

Emitting a tiny raw-size correction on top of that feedback setting did not
confirm. `mark_numeric_blend=0.02`, `mark_numeric_blend_space=raw`,
`mark_numeric_fields=size` with the same feedback-size `0.018` improved seed
`42` by only `0.000001`, but seeds `43-45` moved the four-seed mean to
`0.02711492`, slightly worse than the feedback-only mean `0.02710896`.
The emitted-output branch remains closed.

Fresh fake-seed robustness panels support the higher feedback setting. On
seeds `46-49`, feedback-only size `0.080` scored mean mark `0.02671222` versus
no-feedback `0.02729909`; on seeds `50-53`, it scored `0.02670647` versus
no-feedback `0.02755709`, improving every paired seed. Across seeds `42-53`,
feedback-size `0.080` averages mark `0.02675505`, with identical object
metrics for every paired comparison. Higher checks on seeds `42-45` show
`0.100` and `0.120` closing negative, so `0.080` is the current feedback
promotion.

Training-seed/data-selection variance around this promoted recipe also closed
negative. Train-seed `43` scored `0.05284` on the seed-42 gate; train-seed
`45` scored `0.03426`; and train-seed `44`, while initially promising on seed
`42` (`0.02828`), averaged `0.02850` over seeds `42-45`. The champion remains
the train-seed `42` catw `0.25` checkpoint.

Epoch-length selection around the same seed/data/objective is now bracketed and
closed except for the promoted e20 point. Seed-42 gates were negative at e10
(`0.04398`), e15 (`0.03674`), e18 (`0.04154`), e19 (`0.05914`), e21
(`0.07347`), e22 (`0.03299`), and e25 (`0.03383`). The e20 catw `0.25`
sidecar remains best on both seed-42 and the four-seed confirmation panel.

Full-field numeric blending is closed negative for this sidecar: log-space
blends `0.1`, `0.25`, and `0.5` worsened the seed-42 mark score to `0.03121`,
`0.03765`, and `0.04837` respectively. Field-specific blending is different:
feedback-only size log blend `0.080` improves the mark panel while
preserving emitted reservoir timing and size.

Hidden-size capacity is also bracketed around the champion. h256 failed the
seed-42 gate (`0.04974`), while h96 improved seed `42` but failed confirmation:
mean mark score `0.02791` over seeds `42-45`, behind h128 catw `0.25` at
`0.02775`. The h104 and h112 bridge points also failed the seed-42 gate
(`0.03533` and `0.04102`), so the current capacity branch still promotes h128.
To avoid retraining whole sidecars for every nearby epoch, `train_neural_marks`
now supports `--snapshot-epochs` for saving selected epoch checkpoints from one
run. `sweep_mark_hybrids` now also supports feedback-only numeric grids and caps
per-eval math-library threads so parallel sweeps do not oversubscribe
`vinge.local`.

Artifacts:
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_fixedhistory_seed42_cat-neural_blend-0p0_space-log_fields-both_temp-1p0_noise-0p05_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_fixedhistory_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_catw2_num05_e30_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_temp_micro_seed42_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_64files_h128_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_fieldblend_seed42_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend002_seed42_restored_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend002_confirm_restored_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend_micro_seed42_restored_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend_fine_seed42_restored_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend018_confirm_restored_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed43_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed44_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed45_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed43_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed44_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed45_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_base_seed46_49_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_feedback_size018_seed46_49_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_micro_seed46_49_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_hi_confirm_seed42_45_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_upper_seed42_45_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_upper_seed46_49_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_hi2_seed42_45_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_base_seed50_53_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_feedback080_seed50_53_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw0125_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw0375_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_trainseed43_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_trainseed44_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_trainseed45_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e10_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e15_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e18_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e19_128files_h128_catw025_snap_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e21_128files_h128_catw025_snap_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e22_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e25_128files_h128_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_numericblend_seed42_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h96_catw025_confirm_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h104_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h112_catw025_seed42_eval_100k.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h256_catw025_seed42_eval_100k.json`

Next LANL work: move beyond uniform numeric feedback; the remaining mark gap is
now mostly the reservoir dt/size W1 floor.

---

## Tencent PhaseAtlas Candidate Before Mark Sidecar (2026-04-23)

LANL's explicit object-process branch moved ahead on the long-rollout cache
panel. The promoted Tencent altgan candidate was strict-holdout PhaseAtlas with
forced phase, `transition_blend=0.55`, late rank scales
`1.0,1.0,1.1,1.1`, and fresh local-power evidence pointing to
`local_prob_power=0.8`.

The earlier `local_prob_power=0.85` branch held up on seeds `58-65` at mean
HRC-MAE `0.009288`, but a fresh micro-refinement on seeds `66-69` found
`local_prob_power=0.8` better on HRC: mean `0.009109` versus `0.009790` for
`0.85` and `0.009969` for `0.9`, with reuse and stack-distance still close to
real.

Artifacts:
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_lp085_moreseeds_summary.csv`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_localpow_micro_refine_summary.csv`

---

## R210/R209 Cachesim Loop: Peer Trace Scorer And Alibaba Split Lead (2026-05-01)

LANL added `altgan/cachesim_compare.py` so existing peer CSVs can be scored
through the same `tools/cachesim` gate as generated LANL evaluator runs. It
compares fake and real CSVs across policy/capacity grids and writes the
policy-by-policy HRC-MAE report used in the race tables.

Tencent status: LLNL R210 adj `.04` scored six-policy `0.030856` and adj
`.06` scored `0.030526` on the fixed LANL Tencent real-manifest CSV. Neither
beats LANL p `.60`, k50, tail `.10`, adj `.015` at `0.030240` or its seed-59
confirmation at `0.030301`.

Alibaba status after filling the missing eight-policy reports:

| LANL row | six-policy | eight-policy | note |
|---|---:|---:|---|
| p `.08`, hp `.10`, k125, seed `62` | **0.017260** | 0.022470 | current six-policy best |
| p `.08`, hp `.15`, k100, seed `61` | 0.017641 | **0.021637** | current eight-policy best |
| p `.08`, hp `.12`, k125, seed `68` | 0.017100 | 0.022172 | bridge candidate |
| p `.07`, hp `.10`, k125, seed `64` | 0.017426 | 0.022739 | shape-leaning lower reuse |

LLNL R208 remains behind on both visible Alibaba panels: adj `.00` six-policy
`0.019671`, adj `.02` eight-policy `0.022266`, with much higher adjacent
duplicate debt than LANL. New explicit-decimal probes are live for
p `.08`/hp `.15`/k125, p `.075`/hp `.12`/k125, p `.08`/hp `.10`/k150, and
p `.07`/hp `.12`/k125.

Those explicit-decimal probes closed negative: p `.08`/hp `.15`/k125 scored
six/eight `0.018845`/`0.023364`; p `.075`/hp `.12`/k125 scored
`0.019830`/`0.024602`; p `.08`/hp `.10`/k150 scored `0.020138`/`0.025011`;
and p `.07`/hp `.12`/k125 scored `0.019743`/`0.024429`. The next live bracket
returns to k100: p `.08`/hp `.12`, p `.08`/hp `.13`, and p `.085`/hp `.15`,
all launched with math-library thread caps after uncapped four-way launches
stalled in startup.

The k100 follow-up also closed negative: p `.08`/hp `.12,k100` scored
`0.020097`/`0.024603`, p `.08`/hp `.13,k100` scored `0.019734`/`0.024003`,
and p `.085`/hp `.15,k100` scored `0.018410`/`0.022816`. The active work is
now fresh-seed confirmation of the two actual leaders and the bridge:
p `.08`/hp `.10,k125` seed `77`, p `.08`/hp `.15,k100` seed `78`, and
p `.08`/hp `.12,k125` seed `79`.

Those p `.08` fresh-seed confirmations were weak: seed `77` scored
`0.020158`/`0.024861`, seed `78` scored `0.019030`/`0.023338`, and seed `79`
scored `0.020151`/`0.024729`. The p `.08` rows remain useful as best visible
seeds, but they are not robust enough to promote. The live confirmation pivot
returns to the p `.10` small-hot-pool family: hp `.12,k75` seed `80`, hp
`.10,k100` seed `81`, and hp `.10,k75` seed `82`.

The p `.10` confirmation set also missed on fresh seeds: hp `.12,k75` seed
`80` scored `0.019992`/`0.024045`, hp `.10,k100` seed `81` scored
`0.019997`/`0.024628`, and hp `.10,k75` seed `82` scored
`0.019926`/`0.024444`. The useful next direction is the lower-reuse shape row:
p `.06`/hp `.10,k125` seed `69` scores six/eight `0.017389`/`0.022673` while
matching reuse/p90 much more closely (`0.307248`, p90 `43572` vs real
`0.306465`, p90 `44829`). Live confirmations: p `.06`/hp `.10,k125` seed
`83`, p `.065`/hp `.10,k125` seed `84`, and p `.06`/hp `.12,k125` seed `85`.

The lower-reuse confirmations preserved shape but lost cache: seed `83` scored
`0.020902`/`0.025591` with p90 `44240`, seed `84` scored
`0.020901`/`0.025524`, and seed `85` scored `0.020368`/`0.024968` with p90
`43801`. Live follow-up tries to add hot-pool pressure without raising reuse:
p `.06`/hp `.15,k100` seed `86`, p `.06`/hp `.15,k125` seed `87`, and
p `.06`/hp `.18,k100` seed `88`.

That follow-up produced a new balanced Alibaba target. p `.06`/hp `.18,k100`
seed `88` scored six/eight `0.018282`/`0.022144`, beating LLNL R208's
eight-policy `0.022266` while keeping reuse `0.307590` and p90 `43194` close
to real. hp `.15,k100` seed `86` was weaker (`0.019406`/`0.023606`), and
hp `.15,k125` seed `87` lost (`0.019937`/`0.024477`). Live confirmations:
p `.06`/hp `.18,k100` seed `89`, p `.06`/hp `.20,k100` seed `90`, and
p `.06`/hp `.18,k125` seed `91`.

The confirmation/neighbor pass promoted hp `.20,k100`: seed `90` scored
six/eight `0.017356`/`0.020988`, reuse `0.306875`, median `240`, and p90
`43721`. Same-row hp `.18,k100` seed `89` held a good but weaker
`0.018010`/`0.022058`; hp `.18,k125` seed `91` scored `0.017992`/`0.022436`.
The live bracket confirms hp `.20,k100` seed `92` and checks hp `.22,k100`
seed `93` plus hp `.20,k75` seed `94`.

The next neighbor improved again. hp `.22,k100` seed `93` scored six/eight
`0.016815`/`0.020036`, reuse `0.306979`, median `240`, and p90 `43142`.
hp `.20,k100` seed `92` confirmed the family at `0.017476`/`0.021102`;
hp `.20,k75` seed `94` scored `0.018045`/`0.021012`. Live confirmations:
hp `.22,k100` seed `95`, hp `.24,k100` seed `96`, and hp `.22,k75` seed `97`.

hp `.24,k100` improves the target again. Seed `96` scored six/eight
`0.016666`/`0.019718`, reuse `0.306815`, median `240`, and p90 `43898`.
hp `.22,k100` confirmed at seed `95` (`0.016740`/`0.019927`); hp `.22,k75`
seed `97` scored `0.017604`/`0.020350`. Live follow-up: hp `.24,k100` seed
`99`, hp `.26,k100` seed `98`, and hp `.24,k75` seed `100`.

hp `.26,k100` improved again. Seed `98` scored six/eight
`0.016471`/`0.019135`, reuse `0.306610`, median `239`, and p90 `43621`.
hp `.24,k100` seed `99` confirmed the family at `0.016610`/`0.019401`;
hp `.24,k75` seed `100` scored `0.017711`/`0.020304`. Live follow-up:
hp `.26,k100` seed `101`, hp `.28,k100` seed `102`, and hp `.26,k125`
seed `103`.

hp `.26,k100` confirmed and improved on seed `101`: six/eight
`0.016231`/`0.018970`, reuse `0.306704`, median `240`, p90 `44090`. hp
`.28,k100` seed `102` was close but weaker (`0.016670`/`0.019061`); hp
`.26,k125` seed `103` has the current six-policy minimum (`0.016079`) but
weaker eight-policy (`0.019394`). Live tight bracket: hp `.26,k100` seed
`104`, hp `.30,k100` seed `105`, and hp `.26,k125` seed `106`.

The tight bracket moved the Alibaba eight-policy target again: hp `.30,k100`
seed `105` scored six/eight `0.016684`/`0.018831`, reuse `0.306781`, median
`237`, and p90 `43326`. The fresh hp `.26,k125` seed `106` kept the
six-policy side strong at `0.016138`/`0.019375`. LANL now leads LLNL R208 on
Alibaba by a wide cache-sim margin while retaining the lower-reuse shape. The
first seed `107-110` follow-up is retracted because it omitted
`--force-phase-schedule`.

The corrected forced-phase follow-up improved both targets. hp `.34,k100` seed
`114` scored six/eight `0.016425`/`0.018056`, reuse `0.305682`, median `235`,
and p90 `43069`. hp `.30,k125` seed `113` scored the six-policy low at
`0.015788` and eight-policy `0.018339`, with reuse `0.307460` and p90 `43533`.
Live follow-up through the launcher: hp `.34,k100` seed `115`, hp `.36,k100`
seed `116`, hp `.34,k125` seed `117`, and hp `.30,k125` seed `118`.

That follow-up moved the useful lane to k125. hp `.34,k125` seed `117` scored
six/eight `0.015648`/`0.017767`, reuse `0.306384`, median `244`, and p90
`43831`; hp `.30,k125` seed `118` confirmed the six-policy target at
`0.015567` and eight-policy `0.018065`. hp `.34,k100` seed `115` confirmed
but did not beat seed `114` (`0.016635`/`0.018159`), and hp `.36,k100` seed
`116` was weaker (`0.016904`/`0.018348`). Live k125 bracket: hp `.34,k125`
seed `119`, hp `.36,k125` seed `120`, hp `.32,k125` seed `121`, and
hp `.30,k125` seed `122`.

The k125 confirmation kept moving. hp `.36,k125` seed `120` scored six/eight
`0.015795`/`0.017643`, reuse `0.305986`, median `241`, and p90 `43424`;
hp `.30,k125` seed `122` confirmed and improved the six-policy target at
`0.015559` and eight-policy `0.018014`. hp `.32,k125` seed `121` was nearly
tied with hp `.34,k125` on eight-policy (`0.017782`). Live neighbor bracket:
hp `.36,k125` seed `123`, hp `.38,k125` seed `124`, hp `.36,k150` seed `125`,
and hp `.30,k125` seed `126`.

The next neighbor moved the target to k150. hp `.36,k150` seed `125` scored
six/eight `0.014881`/`0.017070`, evaluator HRC `0.010769`, reuse `0.306794`,
median `254`, and p90 `44358`. This is now best on both policy panels. hp
`.38,k125` seed `124` also improved eight-policy to `0.017184`, but k150 won.
Live neighbor bracket: hp `.36,k150` seed `127`, hp `.38,k150` seed `128`,
hp `.36,k175` seed `129`, and hp `.34,k150` seed `130`.
