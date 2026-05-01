# LANL / altgan Version Notes

This file tracks LANL-owned `altgan/` checkpoints and race-relevant updates.
The detailed LANL result ledger remains [altgan/RESULTS.md](/Users/darrell/Zarathustra/altgan/RESULTS.md).

---

## Tencent 1M Cachesim Gate Added To LANL Evaluator (2026-04-30)

LANL now treats `tools/cachesim` as a required promotion surface for the 1M
Tencent trace. The evaluator can write the exact fake and sampled-real CSVs and
run the simulator directly via `--cachesim-bin`.

Current status:
- Best LANL six-policy fixed-cap row remains post-decode deep reuse injection:
  mean HRC-MAE `0.054073` across LRU/ARC/FIFO/SIEVE/SLRU/CAR at caps
  `32,128,512,2048,8192`; after hot-pool redirect `p=0.50`, the best row is
  now `0.046657`.
- Rank phase scale `1.2,1.2,1.3,1.3` improves the old evaluator HRC-MAE from
  `0.051810` to `0.044706`, and exactly matches stack median `84`, but worsens
  the six-policy mean to `0.055905`; it is a diagnostic row, not promoted.
- LLNL's reported R182 Tencent six-policy mean is `0.0925`, so LANL is still
  ahead on this simulator surface. The remaining LANL error is policy-shaped:
  LRU/FIFO want rank scaling, ARC/SIEVE/CAR do not.
- `stack_adj_dup_prob` exists for controlled tests, but LANL fake already has
  more adjacent duplicates than real on this slice (`0.00427` vs `0.00234`).
- `stack_hot_pool_prob` is now the live branch. `p=0.50` raises top-100 access
  share from `0.003849` to `0.119549` and cuts SIEVE HRC-MAE to `0.033573`;
  `p=0.60` and `p=0.70` both lost to `p=0.50`, so the live branch keeps
  `p=0.50` and raises frequency weight power to `2.0`.

Code changes:
- `altgan.neural_atlas` keeps deep reuse boosts as post-decode trace
  corrections for transition rollout, while passing the emitted action to the
  mark runtime.
- `altgan.evaluate_neural_atlas` adds `--real-output`, `--cachesim-bin`, and
  six-policy report emission.
- `altgan.neural_atlas` adds `stack_hot_pool_*` controls for frequency-shape
  correction.

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
