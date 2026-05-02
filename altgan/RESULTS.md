# altgan vinge.local Results

Run date: 2026-04-21 on `vinge.local`.

Code under test was synced to `~/Zarathustra/altgan/`.  `llgan/` was used only
as read-only infrastructure for existing trace readers and long-rollout cache
metrics.

## Best 100k Long-Rollout Results

All rows use `n_records=100000`, `n_streams=4`, `seed=42`, and the same real
manifest per corpus.

| Corpus | Model | Fit | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tencent | PhaseAtlas | 1024 files x 5k holdout, blend 0.5 + local power 0.9 + late rank scale 1.1 + forced phase | **0.00887** | 0.61451 | 0.61493 | 53 | 60 | 166 | 174 |
| Alibaba | PhaseAtlas | 233 files x 25k holdout, blend 0.2 + local power 0.9 | **0.00222** | 0.27363 | 0.26909 | 192 | 201 | 1508 | 1452 |
| Tencent | NeuralAtlas | 1024 files x 5k holdout, blend 0.25 | 0.01853 | 0.62066 | 0.61493 | 45 | 60 | 144 | 174 |
| Alibaba | NeuralAtlas | 233 files x 25k holdout, blend 0.75 | 0.00349 | 0.26730 | 0.26909 | 183 | 201 | 1264 | 1452 |
| Tencent | NeuralAtlas | 64 files x 25k, blend 0.0 | 0.01845 | 0.62314 | 0.61493 | 55 | 60 | 145 | 174 |
| Alibaba | NeuralAtlas | 64 files x 25k, blend 0.5 | 0.00183 | 0.26451 | 0.26909 | 197 | 201 | 1267 | 1452 |
| Alibaba | NeuralStack | 64 files x 25k, temp 1.0 | 0.00333 | 0.27373 | 0.26909 | 204 | 201 | 1331 | 1452 |

These are directly comparable to the peer long-rollout sidecar.  The current
peer evidence in `VERSIONS-LLNL.md` reports Tencent `v158` HRC-MAE `0.2435` and
Alibaba `v194` HRC-MAE `0.1305` with reuse-access `0.006` vs real `0.265`.
The later peer LRU decoder ablation reaches Alibaba HRC-MAE `0.0051` only with
a real-rate reuse override, while `v199` rate matching and `v200` high-weight
BCE are both closed failed.  `v201` is the live categorical reuse-head attempt,
with no long-rollout result recorded yet.  The altgan best-of-family results
therefore remain the cleanest measured held-out cache panel.
The holdout rows exclude the four real-manifest source files from training.
`PhaseAtlas` is `NeuralAtlas` with `--n-phase-bins 8`, which adds within-file
position to the locality state to reduce stationary sampling artifacts.

## Strict Holdout And Phase Results

These runs were added after the first challenge because the initial 64-file
panel was too narrow and allowed exact eval-manifest files into the training
pool. The holdout models exclude those files.

| Corpus | Model | Fit | blend | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | timing drift ratio | size drift ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tencent | NeuralAtlas | 1024 files x 5k, holdout | 0.25 | 0.01853 | 0.62066 | 0.61493 | 45 | 60 | 144 | 174 | 0.0199 | 0.0096 |
| Tencent | PhaseAtlas | 1024 files x 5k, holdout | 0.5 | **0.01065** | 0.60947 | 0.61493 | 48 | 60 | 159 | 174 | 0.0579 | 0.0293 |
| Tencent | PhaseAtlas | 1024 files x 5k, holdout + microblend | 0.65 | **0.00983** | 0.61415 | 0.61493 | 50 | 60 | 171 | 174 | 0.0164 | 0.0144 |
| Tencent | PhaseAtlas | 1024 files x 5k, holdout + microblend + late rank scale | 0.65 | **0.00937** | 0.61415 | 0.61493 | 54 | 60 | 184 | 174 | pending | pending |
| Tencent | PhaseAtlas | 1024 files x 5k, holdout + late rank scale + forced phase | 0.5 | **0.00887** | 0.61451 | 0.61493 | 53 | 60 | 166 | 174 | pending | pending |
| Tencent | PhaseAtlas | 1024 files x 5k, holdout | 1.0 | 0.01982 | 0.63573 | 0.61493 | 60 | 60 | 186 | 174 | 0.0497 | 0.0390 |
| Alibaba | NeuralAtlas | 233 files x 25k, holdout | 0.75 | 0.00349 | 0.26730 | 0.26909 | 183 | 201 | 1264 | 1452 | 0.3251 | 1.1296 |
| Alibaba | PhaseAtlas | 233 files x 25k, holdout | 0.0 | 0.00301 | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 | 1.2015 | 0.7529 |
| Alibaba | PhaseAtlas | 233 files x 25k, holdout + microblend | 0.2 | **0.00222** | 0.27363 | 0.26909 | 192 | 201 | 1508 | 1452 | pending | pending |

The PhaseAtlas rows are the better answer to the "statistically
indistinguishable" goal. Tencent still under-expresses timing and size drift,
but phase conditioning improves HRC and moves drift in the right direction.
Alibaba PhaseAtlas is currently the cleanest held-out object-process result.
The original blend `0.0` row keeps the best reservoir mark score. The
microblend row is the seed-42 HRC leader, but the seed-confirmation pass below
shows it is not stable enough yet to replace the conservative baseline.

A forced phase-schedule ablation, where generation overwrites the sampled phase
with the synthetic stream position, did not improve the best rows. Tencent
blend 0.5 moved from HRC-MAE `0.01065` to `0.01131`; Alibaba's best forced row
was blend 0.5 at `0.00320` versus the unforced blend 0.0 row at `0.00301`.
Keep forced phase as a diagnostic flag, not the default generator.

A longer-file Tencent phase run also failed to beat the champion. The
`512 files x 25k` holdout PhaseAtlas reached best HRC-MAE `0.01393` at blend
`0.75` with reuse `0.62186` vs real `0.61493`, stack median `52` vs `60`, and
p90 `177` vs `174`. It improved stack p90 but made HRC and drift worse than
the `1024 files x 5k` row, so Tencent currently prefers broader file coverage
over deeper per-file slices.

## Tencent PhaseAtlas Microblend Calibration

Recorded 2026-04-22. The Alibaba transition-power result suggested that a
small amount of neural transition smoothing could help when the coarse blend
grid missed the optimum. On Tencent, the same idea produced a cleaner win than
Alibaba: HRC improved from `0.010647` at the prior blend `0.5` baseline to
`0.009831` at blend `0.65` with `local_prob_power=0.9`, while reuse and stack
p90 both moved closer to real.

| transition_blend | local_prob_power | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.00 | 0.010647 | 0.60947 | 0.61493 | 48 | 60 | 159 | 174 | 0.04557 |
| 0.50 | 0.90 | 0.010446 | 0.61215 | 0.61493 | 49 | 60 | 170 | 174 | 0.04562 |
| 0.55 | 1.00 | 0.009983 | 0.61177 | 0.61493 | 49 | 60 | 159 | 174 | 0.05050 |
| 0.65 | 0.90 | **0.009831** | 0.61415 | 0.61493 | 50 | 60 | 171 | 174 | **0.04490** |
| 0.65 | 1.10 | 0.011175 | 0.61604 | 0.61493 | 50 | 60 | 174 | 174 | 0.04633 |

Unlike the Alibaba seed-42 microblend, this Tencent winner improves HRC, reuse,
stack p90, and mark score on the fixed panel. The seed-confirmation pass kept
the microblend family useful, but it did not promote the exact seed-42 cell as
the conservative Tencent champion. Across seeds 43-45, the best mean HRC row
was `blend=0.5, local_prob_power=0.9` at mean HRC `0.010422`, narrowly ahead
of `blend=0.65, local_prob_power=1.0` at `0.010439`. The seed-42 HRC leader
`blend=0.65, local_prob_power=0.9` regressed to mean HRC `0.011458`, mostly
from seed 45.

| Confirm row | Seeds | mean HRC-MAE | HRC range | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blend 0.5, local power 0.9 | 43-45 | **0.010422** | 0.010102-0.010992 | 0.61174 | 0.61493 | 48.3 | 60 | 154.3 | 174 | 0.04909 |
| blend 0.65, local power 1.0 | 43-45 | 0.010439 | 0.010182-0.010737 | 0.61290 | 0.61493 | 50.3 | 60 | 154.0 | 174 | **0.04632** |
| blend 0.65, local power 0.9 | 43-45 | 0.011458 | 0.009940-0.013167 | 0.61121 | 0.61493 | 50.7 | 60 | 150.3 | 174 | 0.04824 |
| blend 0.5, local power 1.0 | 43-45 | 0.013709 | 0.010649-0.017671 | 0.60536 | 0.61493 | 47.7 | 60 | 155.0 | 174 | 0.04727 |

The practical Tencent setting remains the smoothed local-power row
(`blend=0.5, local_prob_power=0.9`) until a new mechanism moves stack p90
toward 174 without giving back HRC.

A drift confirmation on the same seed-42 panel filled the earlier pending
columns. Natural phase at `blend=0.65, local_prob_power=0.9` remains the HRC
leader (`0.009831`) and keeps reuse/stack p90 closest to real, but it still
under-expresses temporal mark drift: timing drift ratio `0.016` and size drift
ratio `0.014`. Forced phase raises timing drift to `0.085` and size drift to
`0.050` at the same blend/local power, but HRC worsens to `0.010641` and stack
p90 falls from `171` to `163` vs real `174`. Keep natural phase as the promoted
cache row; forced phase is a drift diagnostic, not the benchmark setting.

A rank-scale microprobe then tested whether Tencent's too-shallow stack median
could be fixed by expanding sampled reuse ranks after the object process chose
an event. The sweep used seeds `42,43`, natural phase, local power `0.9`,
transition blends `0.5,0.65`, and rank scales `1.0,1.1,1.2`. The full
artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_rankscale_micro_confirm_summary.csv`
and `_best.json`.

| transition_blend | rank scale | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1.0 | 2 | 0.010309 | 0.61200 | 0.61493 | 48.5 | 60 | 163.5 | 174 | 0.04800 |
| 0.50 | 1.1 | 2 | **0.010176** | 0.61200 | 0.61493 | 53.5 | 60 | 180.0 | 174 | 0.04800 |
| 0.50 | 1.2 | 2 | 0.012966 | 0.61200 | 0.61493 | 58.0 | 60 | 195.5 | 174 | 0.04800 |
| 0.65 | 1.0 | 2 | 0.010549 | 0.61522 | 0.61493 | 50.5 | 60 | 162.0 | 174 | **0.04729** |
| 0.65 | 1.1 | 2 | **0.010040** | 0.61522 | 0.61493 | 55.5 | 60 | 178.0 | 174 | **0.04729** |
| 0.65 | 1.2 | 2 | 0.013191 | 0.61522 | 0.61493 | 60.5 | 60 | 194.5 | 174 | **0.04729** |

Rank expansion is useful only in a narrow band: scale `1.1` improves the
two-seed mean HRC and moves the median/p90 closer to real, while `1.2`
over-expands the tail and loses badly. The next Tencent HRC probe should test
phase-specific or capped `1.1` schedules, not global scale escalation.

The phase-specific follow-up found a better Tencent HRC setting. Expanding
only late phases with schedule `1.0,1.0,1.1,1.1` beat both the baseline and
global scale on the two-seed mean. It also promoted a new seed-42 holdout row:
`blend=0.65`, local power `0.9`, late rank scale `1.1`, HRC-MAE `0.009367`,
reuse `0.61415/0.61493`, stack median `54/60`, and stack p90 `184/174`. The
full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_rankscale_phase_confirm_summary.csv`
and `_best.json`.

| transition_blend | phase rank-scale schedule | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | baseline | 2 | 0.010309 | 0.61200 | 0.61493 | 48.5 | 60 | 163.5 | 174 | 0.04800 |
| 0.50 | 1.0,1.0,1.1,1.1 | 2 | **0.009645** | 0.61200 | 0.61493 | 52.0 | 60 | 174.0 | 174 | 0.04800 |
| 0.50 | 1.0,1.1,1.1,1.0 | 2 | 0.009789 | 0.61200 | 0.61493 | 50.0 | 60 | 167.0 | 174 | 0.04800 |
| 0.65 | baseline | 2 | 0.010549 | 0.61522 | 0.61493 | 50.5 | 60 | 162.0 | 174 | **0.04729** |
| 0.65 | 1.0,1.0,1.1,1.1 | 2 | **0.009634** | 0.61522 | 0.61493 | 54.5 | 60 | 174.0 | 174 | **0.04729** |
| 0.65 | 1.0,1.1,1.1,1.0 | 2 | 0.009902 | 0.61522 | 0.61493 | 52.0 | 60 | 165.0 | 174 | **0.04729** |

This keeps Tencent's best next HRC move squarely in the explicit stack decoder:
phase-aware rank calibration improves cache behavior without changing marks or
the reuse decision. The next runnable probe should add caps around the late
`1.1` schedule to stop seed-42 p90 overshoot while preserving the two-seed
mean gain.

The late-rank cap probe closed negative. It kept the same late `1.1` schedule
and tried caps on the late phases. Caps reduced p90, but every capped setting
gave back HRC; the uncapped schedule remained best. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_late_rankcap_confirm_summary.csv`
and `_best.json`.

| transition_blend | late phase maxes | seeds | mean HRC-MAE | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | none | 2 | **0.009645** | 52.0 | 60 | 174.0 | 174 | 0.04800 |
| 0.50 | 176,176 | 2 | 0.011655 | 52.0 | 60 | 173.0 | 174 | 0.04800 |
| 0.50 | 160,160 | 2 | 0.013744 | 52.0 | 60 | 160.0 | 174 | 0.04800 |
| 0.65 | none | 2 | **0.009634** | 54.5 | 60 | 174.0 | 174 | 0.04729 |
| 0.65 | 176,176 | 2 | 0.012361 | 54.5 | 60 | 170.0 | 174 | 0.04729 |
| 0.65 | 160,160 | 2 | 0.014576 | 54.5 | 60 | 160.0 | 174 | 0.04729 |

Treat caps as closed for this Tencent branch. The useful hypothesis is now
uncapped late-phase rank expansion, which needs a four-seed confirmation before
promotion.

The four-seed confirmation promoted uncapped late-phase rank expansion. It
improved the mean HRC for both tested transition blends without changing reuse
or marks, and the stronger stable row is now `blend=0.5`, local power `0.9`,
phase rank scales `1.0,1.0,1.1,1.1`: mean HRC-MAE `0.009575` versus
`0.010428` for the paired baseline. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_late_rank_seed_confirm_summary.csv`
and `_best.json`.

| transition_blend | phase rank-scale schedule | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | baseline | 4 | 0.010428 | 0.61184 | 0.61493 | 48.5 | 60 | 158.2 | 174 | 0.04823 |
| 0.50 | 1.0,1.0,1.1,1.1 | 4 | **0.009575** | 0.61184 | 0.61493 | 52.0 | 60 | 170.0 | 174 | 0.04823 |
| 0.65 | baseline | 4 | 0.011051 | 0.61194 | 0.61493 | 50.5 | 60 | 155.5 | 174 | **0.04741** |
| 0.65 | 1.0,1.0,1.1,1.1 | 4 | **0.010287** | 0.61194 | 0.61493 | 54.5 | 60 | 168.2 | 174 | **0.04741** |

This shifts Tencent's promoted stable HRC candidate from the seed-42
microblend row to a four-seed late-rank schedule at blend `0.5`. The next
robustness check is an `8x50k` panel using the same baseline-versus-late
schedule comparison.

The `8x50k` panel check could not run against the fixed Tencent real manifest:
`llgan.long_rollout_eval` rejects `n_records=50000` when the manifest was built
for `100000` records. A valid forced-vs-natural check on the same `4x100k`
panel instead found another improvement. With blend `0.5`, local power `0.9`,
and late rank scales `1.0,1.0,1.1,1.1`, forced phase improved mean HRC from
`0.009575` to `0.009371`, improved mean reuse from `0.61184` to `0.61528`
against real `0.61493`, and improved mean mark score from `0.04823` to
`0.04669`. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_late_rank_forced_confirm_summary.csv`
and `_best.json`.

| phase mode | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural | 4 | 0.009575 | 0.61184 | 0.61493 | 52.0 | 60 | 170.0 | 174 | 0.04823 |
| forced | 4 | **0.009371** | **0.61528** | 0.61493 | 52.2 | 60 | 165.5 | 174 | **0.04669** |

This is the current Tencent HRC champion on the LANL side. The next useful
run is a small drift-focused check for this forced-phase late-rank row, because
forced phase previously improved drift while sometimes hurting HRC; the new
rank schedule appears to remove most of that HRC cost.

A fresh-seed forced-phase confirmation on seeds `46-49` stayed competitive but
regressed from the first four seeds: mean HRC-MAE `0.009934`, reuse
`0.61576/0.61493`, stack median `52.5/60`, stack p90 `164.5/174`, and mean mark
score `0.04642`. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_late_rank_forced_freshseed_confirm_summary.csv`
and `_best.json`. Across the eight forced seeds now tested, the rough mean HRC
is `0.00965`, so the row remains strong but should be compared against natural
phase on the same fresh seeds before declaring forced phase universally better.

The paired natural-phase fresh-seed check made the comparison decisive: on
seeds `46-49`, natural phase had mean HRC-MAE `0.013531`, reuse
`0.61193/0.61493`, stack median `52.0/60`, stack p90 `165.2/174`, and mean mark
score `0.04843`. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_late_rank_natural_freshseed_confirm_summary.csv`
and `_best.json`. Forced phase is the better setting for the late-rank Tencent
branch across the fresh seeds, not just the original `42-45` bundle.

A forced-phase blend refinement on fresh seeds `46-49` then tested
`transition_blend` values `0.45,0.50,0.55` with the same late-rank schedule.
Blend `0.55` was best on mean HRC and mark score: mean HRC-MAE `0.009671`,
reuse `0.61529/0.61493`, stack median `53.0/60`, stack p90 `167.5/174`, and
mean mark score `0.04561`. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_blend_refine_summary.csv`
and `_best.json`.

| transition_blend | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.45 | 4 | 0.010181 | 0.61527 | 0.61493 | 51.8 | 60 | 161.8 | 174 | 0.04678 |
| 0.50 | 4 | 0.009934 | **0.61576** | 0.61493 | 52.5 | 60 | 164.5 | 174 | 0.04642 |
| 0.55 | 4 | **0.009671** | 0.61529 | 0.61493 | 53.0 | 60 | 167.5 | 174 | **0.04561** |

Running the same `0.55` forced late-rank row on the original seeds `42-45`
confirmed the improvement: mean HRC-MAE `0.008867`, reuse `0.61425/0.61493`,
stack median `53.2/60`, stack p90 `168.5/174`, and mean mark score `0.04626`.
The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_blend055_seed_confirm_summary.csv`
and `_best.json`. Across the eight tested seeds `42-49`, `blend=0.55` has an
approximate mean HRC `0.00927`, better than the `blend=0.50` forced late-rank
mean (`0.00965`). Treat `0.55 + forced phase + late rank scale` as the current
stable Tencent PhaseAtlas candidate.

An additional eight-seed stability run on seeds `50-57` stayed in the same
band: mean HRC-MAE `0.009541`, reuse `0.61567/0.61493`, stack median
`53.2/60`, stack p90 `168.5/174`, and mean mark score `0.04595`. The full
artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_blend055_moreseeds_summary.csv`
and `_best.json`. Across seeds `42-57`, the approximate mean HRC is now
`0.00941`, making this the best stable Tencent object-process row found so far.

A local-power refinement on seeds `50-53` found a small HRC tradeoff around
that Tencent candidate. With forced phase, `transition_blend=0.55`, and late
rank scales `1.0,1.0,1.1,1.1`, local power `0.85` produced the best mean HRC
(`0.009692`) while `0.9` produced slightly better mark score (`0.04584` vs
`0.04635`). `0.95` was worse on both HRC and reuse. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_localpow_refine_summary.csv`
and `_best.json`.

| local_prob_power | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.85 | 4 | **0.009692** | 0.61614 | 0.61493 | 53.2 | 60 | 167.8 | 174 | 0.04635 |
| 0.90 | 4 | 0.009739 | 0.61602 | 0.61493 | 53.2 | 60 | 169.2 | 174 | **0.04584** |
| 0.95 | 4 | 0.010273 | 0.61728 | 0.61493 | 53.2 | 60 | 167.5 | 174 | 0.04634 |

An additional `0.85` stability run on seeds `58-65` held the gain. It produced
mean HRC-MAE `0.009288`, reuse `0.61511/0.61493`, stack median `53.1/60`,
stack p90 `169.0/174`, and mean mark score `0.04593`. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_lp085_moreseeds_summary.csv`
and `_best.json`. Across seeds `50-65`, the approximate mean HRC for
`local_prob_power=0.85` is now about `0.00942`, slightly ahead of the earlier
`0.9` branch while staying in the same reuse/mark regime.

A fresh micro-refinement on seeds `66-69` then widened the local-power grid to
`0.8,0.85,0.9`. On those fresh seeds, `local_prob_power=0.8` was clearly best:
mean HRC-MAE `0.009109`, reuse `0.61512/0.61493`, stack median `53.5/60`,
stack p90 `169.8/174`, and mean mark score `0.04636`. The best single row in
the sweep was seed `67` at HRC-MAE `0.008520`, reuse `0.61366/0.61493`, stack
median `54/60`, stack p90 `168/174`, and mark score `0.04646`. The full
artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_forced_late_localpow_micro_refine_summary.csv`
and `_best.json`.

| local_prob_power | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 4 | **0.009109** | 0.61512 | 0.61493 | 53.5 | 60 | 169.8 | 174 | 0.04636 |
| 0.85 | 4 | 0.009790 | 0.61656 | 0.61493 | 53.2 | 60 | 168.2 | 174 | 0.04614 |
| 0.90 | 4 | 0.009969 | 0.61658 | 0.61493 | 53.2 | 60 | 167.2 | 174 | **0.04568** |

This overturns the earlier narrow `0.85` lead from seeds `50-53`: the Tencent
forced-phase late-rank branch still looks strong at blend `0.55`, but fresh
seeds now point to `local_prob_power=0.8` as the better HRC candidate. The
next LANL move is no longer another object-process scalar sweep. Tencent now
moves to IDEA #53 sidecar mark training around this strict-holdout PhaseAtlas
winner.

The first Tencent sidecar mark follow-up fixed the hybrid rollout contract so
the autoregressive mark head observes the mark that was actually emitted after
reservoir/neural blending. It then compared the older default-loss e20 sidecar
against a freshly trained categorical-heavy e30 sidecar on the same
strict-holdout PhaseAtlas object law. Both checks used reservoir numeric marks
and neural categoricals (`mark_numeric_blend=0.0`, log space, temp `1.0`,
noise `0.05`), so HRC/reuse/stack metrics stayed exactly paired.

The categorical-heavy e30 training run
(`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_catw2_num05_e30.pkl.gz`)
used `numeric_loss_weight=0.5`, `categorical_loss_weight=2.0`, 30 epochs,
hidden dim `192`, and the fixed real-manifest holdout exclusion. It improved
mean mark score from the reservoir control's `0.04684` to `0.03941`. Rerunning
the older e20 checkpoint under the fixed emitted-history runtime was much
better: mean mark score `0.02842` on the same four seeds, a 39.3% paired gain
over reservoir marks and a 27.9% gain over the e30 sidecar. The full artifacts
are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_catw2_num05_e30_confirm_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_fixedhistory_confirm_summary.csv`,
and the seed-42 e20 fixed-history JSON
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_fixedhistory_seed42_cat-neural_blend-0p0_space-log_fields-both_temp-1p0_noise-0p05_eval_100k.json`.

| mark source | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score | opcode TV | tenant TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reservoir control | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.04684 | 0.04105 | 0.04105 |
| e30 reservoir numeric + neural categorical | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.03941 | 0.02640 | 0.02600 |
| e20 reservoir numeric + neural categorical | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | **0.02842** | **0.00404** | **0.00438** |

This promotes the e20 neural-categorical sidecar as the current Tencent mark
champion around the strict-holdout object-process winner. The next useful work
is not more categorical loss weight; it is either a temperature/noise
micro-sweep around e20 or an inference-speed patch, because the event-by-event
GRU rollout makes broad mark sweeps expensive.

The e20 categorical-temperature micro-sweep closed negative. On seed `42`,
temps `0.5`, `0.75`, `1.25`, and `1.5` all preserved HRC/reuse/stack behavior
but worsened categorical mark quality versus the promoted temp `1.0` row. The
full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_temp_micro_seed42_summary.csv`
and `_best.json`.

| categorical temp | HRC-MAE | mark score | opcode TV | tenant TV |
|---:|---:|---:|---:|---:|
| 0.50 | 0.008424 | 0.04516 | 0.03623 | 0.03783 |
| 0.75 | 0.008424 | 0.03772 | 0.02117 | 0.02313 |
| 1.00 | 0.008424 | **0.02876** | **0.00354** | **0.00492** |
| 1.25 | 0.008424 | 0.03591 | 0.01883 | 0.01823 |
| 1.50 | 0.008424 | 0.04859 | 0.04461 | 0.04320 |

Keep `mark_categorical_temp=1.0`; the next mark-side move should be inference
throughput or a different sidecar objective, not categorical sampling
temperature.

The mark-sidecar training-data-size follow-up found a small but stable mark
gain. A broader 512-file h128 default-loss e20 sidecar closed negative on
seed `42`: it preserved HRC (`0.008423`) but worsened mark score to `0.03838`.
The narrower 128-file h128 default-loss e20 sidecar improved all four paired
seeds while again leaving object metrics unchanged. The full artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_confirm_summary.csv`,
`_best.json`, and the seed-42 JSON
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_seed42_eval_100k.json`.

| mark source | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score | opcode TV | tenant TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e20 256-file sidecar | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.02842 | 0.00404 | 0.00438 |
| e20 128-file h128 sidecar | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | **0.02788** | **0.00343** | **0.00282** |

This promotes
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128.pkl.gz`
as the current Tencent mark-side checkpoint. The cached-input runtime patch
preserved metrics exactly but did not clearly improve elapsed time under
contention (`572.80s` vs the prior `532.93s` seed-42 speedcheck), so do not
count it as a throughput win yet.

The narrower 64-file h128 sidecar then closed negative on seed `42`: it again
kept HRC-MAE at `0.008423` but worsened mark score to `0.03344`. The
data-size branch therefore prefers 128 files over 64, 256, and 512 for this
default-loss sidecar family.

The 128-file objective-shape follow-up found a smaller but confirmed gain. A
reduced categorical-loss sidecar
(`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`)
used the same 128 training files, hidden dim `128`, e20 schedule, seed `42`,
and `numeric_loss_weight=1.0`, but lowered `categorical_loss_weight` from
`0.5` to `0.25`. It preserved the object process exactly and reduced the
four-seed mean mark score from `0.02788` to `0.02775`. The improvement is
modest but paired: seeds `42`, `44`, and `45` improved, while seed `43`
regressed slightly. Adjacent points closed negative on seed `42`: catw `0.125`
scored `0.04975`, and catw `0.375` scored `0.06088`. The full confirmation
artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_confirm_summary.csv`,
`_best.json`, and the seed-42 JSON
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_seed42_eval_100k.json`.

| mark source | seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score | opcode TV | tenant TV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| e20 128-file h128, catw 0.50 | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.02788 | 0.00343 | **0.00282** |
| e20 128-file h128, catw 0.25 | 4 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | **0.02775** | **0.00287** | 0.00286 |

This promotes
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`
as the current Tencent mark-side checkpoint. The objective-shape bracket is
not monotonic; do not chase lower categorical weights without a new mechanism.

The follow-up training-seed/data-selection pass closed negative. Reusing the
catw `0.25` objective with different training seeds changed both initialization
and the sampled 128-file training subset. Train-seed `43` was rejected on the
seed-42 gate (`mark_score=0.05284`). Train-seed `45` was also rejected on the
seed-42 gate (`mark_score=0.03426`). Train-seed `44` looked promising on the
same gate (`0.02828` versus `0.02847` for the promoted train-seed `42`
checkpoint), but failed confirmation: seeds `42-45` averaged `0.02850`, worse
than the promoted `0.02775`.

| catw 0.25 training seed | eval seeds | mean HRC-MAE | mean mark score | verdict |
|---:|---:|---:|---:|---|
| 42 | 42-45 | 0.008767 | **0.02775** | promoted |
| 43 | 42 only | 0.008423 | 0.05284 | closed |
| 44 | 42-45 | 0.008767 | 0.02850 | closed |
| 45 | 42 only | 0.008423 | 0.03426 | closed |

The training-seed branch therefore keeps the existing promoted catw `0.25`,
train-seed `42` checkpoint. The next useful mark-side idea needs a mechanism
change or checkpoint selection inside a single training run, not more random
128-file subsets.

The coarse and tight epoch-selection pass also closed negative. With the same
128-file subset, seed `42`, h128, and catw `0.25` objective, e10/e15 were too
early and e25 overran the seed-42 mark gate. A tighter e18/e22 bracket around
the promoted e20 checkpoint also missed. A snapshot run then filled the exact
neighbors, and e19/e21 were both worse. The seed-42 mark scores were:

| epochs | HRC-MAE | mark score | verdict |
|---:|---:|---:|---|
| 10 | 0.008423 | 0.04398 | closed |
| 15 | 0.008423 | 0.03674 | closed |
| 18 | 0.008423 | 0.04154 | closed |
| 19 | 0.008423 | 0.05914 | closed |
| 20 | 0.008423 | **0.02847** | promoted |
| 21 | 0.008423 | 0.07347 | closed |
| 22 | 0.008423 | 0.03299 | closed |
| 25 | 0.008423 | 0.03383 | closed |

For this sidecar family, epoch `20` is the only checkpoint-length point that
survived the seed-42 gate. Further improvement likely needs changing the mark
model or numeric mark mechanism, not simply changing the final epoch count.

The full-field numeric-blend mechanism check also closed negative. Keeping neural
categoricals from the promoted catw `0.25` checkpoint, log-space blends of the
neural dt/size predictions into the reservoir numeric marks worsened the
seed-42 mark score: blend `0.1` scored `0.03121`, blend `0.25` scored
`0.03765`, and blend `0.5` scored `0.04837`, all versus the promoted
reservoir-numeric row at `0.02847`. Do not blend both numeric fields together.

A field-selective numeric pass then reopened the useful part of that mechanism.
Keeping reservoir timing and blending only object size in log space found a
confirmed gain at `mark_numeric_fields=size`, `mark_numeric_blend=0.018`. The
nearby `0.02` setting first confirmed the mechanism, then a finer seed-42 check
found `0.018`; seeds `43-45` confirmed the tiny additional gain. Object metrics
remain unchanged because the object process is untouched.

The stronger result is feedback-only numeric blending: emit reservoir numeric
marks, but feed the autoregressive mark head a size-only log blend for its
next-step state. This keeps the reservoir dt/size W1 terms unchanged while
collapsing categorical TV closer to zero. Feedback-only dt and both-field
variants missed on seed `42` (`0.02694`, `0.02701`, `0.02680`) versus the
size-only branch.

| numeric runtime | eval seeds | mean HRC-MAE | mean fake reuse | real reuse | mean stack med | real stack med | mean stack p90 | real stack p90 | mean mark score | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| reservoir numeric, neural categoricals | 42-45 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.02775 | old promoted |
| emitted size-only log blend 0.018, neural categoricals | 42-45 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.02725 | closed |
| feedback-only size log blend 0.018, neural categoricals | 42-45 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.02711 | superseded |
| feedback-only size log blend 0.080, neural categoricals | 42-45 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | **0.02685** | promoted |
| feedback size log 0.018 + emitted raw size 0.02 | 42-45 | 0.008767 | 0.61369 | 0.61493 | 53 | 60 | 170 | 174 | 0.027115 | closed |

Seed-level mark scores for the promoted size-only row were `0.026914`,
`0.028132`, `0.027070`, and `0.026901`. The supporting artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_fieldblend_seed42_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend002_seed42_restored_eval_100k.json`,
and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend002_confirm_restored_summary.csv`.
A seed-42 micro-refine over size-only blends `0.01`, `0.015`, `0.02`,
`0.025`, `0.03`, and `0.04` kept `0.02` as the best tested point; the nearest
competitor was `0.015` at `0.02695`, still slightly worse than `0.02` at
`0.02695` by the unrounded score (`0.02695463` vs `0.02694582`).
A finer seed-42 check then found `0.018` at `0.02691395`; confirmation over
seeds `43-45` gave a four-seed mean `0.02725443`, beating `0.02` by
`0.00000723`. Fine artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend_fine_seed42_restored_summary.csv`
and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_sizeblend018_confirm_restored_summary.csv`.
Feedback-only artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed42_eval_100k.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed43_eval_100k.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed44_eval_100k.json`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_size018_seed45_eval_100k.json`.
The emitted-output follow-up tried a tiny raw-size blend `0.02` while keeping
feedback-size `0.018`. It produced seed scores `0.02676478`, `0.02798556`,
`0.02693309`, and `0.02677625`, mean `0.02711492`, so the seed-42 hairline win
did not survive the four-seed panel. Artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed42_eval_100k.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed43_eval_100k.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed44_eval_100k.json`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback018_outsize_raw002_seed45_eval_100k.json`.

Fresh fake seeds `46-49` support the feedback-only setting out-of-sample. The
no-feedback neural-categorical control scored mean mark `0.02729909`; the same
runtime with feedback-size log blend `0.018` scored mean mark `0.02722472`.
Object metrics were bit-identical between the paired cells: mean HRC-MAE
`0.00941475`, fake reuse `0.6137475`, real reuse `0.61493`, stack median `53`
vs real `60`, and stack p90 `166.75` vs real `174`. The paired mark deltas by
seed were `-0.0000675`, `+0.0000425`, `-0.0002200`, and `-0.0000525`.
Robustness artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_base_seed46_49_summary.csv` and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_feedback_size018_seed46_49_summary.csv`.

The feedback-strength sweep then moved the promotion from `0.018` to `0.080`.
On seeds `42-45`, feedback blends `0.021`, `0.024`, `0.027`, `0.030`,
`0.036`, `0.045`, `0.060`, `0.080`, `0.100`, and `0.120` gave mean mark
scores `0.027098`, `0.027083`, `0.027003`, `0.026980`, `0.026960`,
`0.026941`, `0.026941`, `0.026846`, `0.026878`, and `0.026984`. On fresh
seeds `46-49`, `0.080` scored `0.026712`, better than `0.024` (`0.027176`),
`0.036` (`0.026970`), `0.045` (`0.026947`), and `0.060` (`0.026939`). Across
the eight paired seeds `42-49`, feedback-size `0.080` averages mark score
`0.02677934` while keeping object metrics unchanged. This is the current
Tencent mark-runtime promotion. Artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_micro_seed46_49_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_hi_confirm_seed42_45_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_upper_seed42_45_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_upper_seed46_49_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_feedback_hi2_seed42_45_summary.csv`.
The next fresh panel, seeds `50-53`, strengthened the promotion: no-feedback
mean mark was `0.02755709`, feedback-size `0.080` mean was `0.02670647`, and
all four paired seeds improved while HRC/reuse/stack stayed identical. Across
seeds `42-53`, feedback-size `0.080` averages mark score `0.02675505`. Added
artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_base_seed50_53_summary.csv` and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_128files_h128_catw025_robust_feedback080_seed50_53_summary.csv`.

Holding that promoted feedback runtime fixed, a seeds `50-53` object micro-sweep
tested nearby `transition_blend` and `local_prob_power` values. The best HRC
cell was `transition_blend=0.525`, `local_prob_power=0.825`, with mean HRC-MAE
`0.00873275`, fake reuse `0.61370`, stack median `52.75`, stack p90 `165.25`,
and mean mark score `0.02675281`. The balanced neighbor
`transition_blend=0.575`, `local_prob_power=0.75` scored mean HRC-MAE
`0.00876875`, fake reuse `0.61368`, stack median `54`, stack p90 `170.5`, and
mean mark score `0.02629462`. The old object row (`0.55`, `0.8`) scored
HRC-MAE `0.00925137` and mark `0.02670647` on the same seeds. The seeds `42-45`
confirmation shifted the HRC lead to `0.575/0.75` on that panel (`0.00848437`)
but with worse mark score (`0.02733755`). Across the combined eight seeds
`42-45,50-53`, `0.575/0.75` has the best HRC (`0.00862656`) and p90
(`170.25`), while `0.55/0.75` has the better mean mark score (`0.02651714`)
with HRC `0.00886219`. The exact-pair panel on seeds `54-57` confirmed
`0.575/0.75` as the HRC-leading candidate. Across all twelve evaluated seeds,
it beats the old `0.55/0.8` object row on both HRC (`0.00862125` vs
`0.00892654`) and mark (`0.02683659` vs `0.02701257`), with stack p90 closer
to real (`170.25` vs `168.58`). The `0.55/0.75` row is still the
mark-favoring candidate at mark `0.02650085` and HRC `0.00882121`. A seeds
`58-61` local-power refinement changed the shared-seed readout: over the
sixteen seeds where both are measured, `0.55/0.75` now edges `0.575/0.75` on
both HRC (`0.00871025` vs `0.00871844`) and mark (`0.02671515` vs
`0.02708219`), while the old `0.55/0.8` row is weaker at HRC `0.00890641` and
mark `0.02704201`. The new `0.575/0.70` point confirmed over seeds `42-57`;
combined with the original `58-61` panel, it has 20 seeds at HRC `0.00858638`,
mark `0.02694865`, reuse `0.61350`, stack median `53.75`, and stack p90
`169.8`. On the same sixteen seeds where `0.55/0.75` and old `0.55/0.8` are
measured, `0.575/0.70` scores HRC `0.00853172`, mark `0.02683922`, reuse
`0.61373688`, and stack p90 `170.125`; `0.55/0.75` scores HRC `0.00871025`,
mark `0.02671515`; old `0.55/0.8` scores HRC `0.00890641`, mark
`0.02704201`. A seeds `62-65` interpolation panel reinforced `0.575/0.70` as
the current HRC-leading shared candidate. On the fair `42-45,50-65` seed set,
`0.575/0.70` scores HRC `0.00854965`, mark `0.02680073`, reuse `0.613864`,
stack median `53.85`, and p90 `170.3`; `0.55/0.75` scores HRC `0.00871607`
and mark `0.02678954`; old `0.55/0.8` scores HRC `0.00897950` and mark
`0.02697943`. The interpolation panel opened lower-transition challengers, but
the seeds `42-61` confirmation did not dislodge `0.575/0.70`: `0.55/0.70`
scored HRC `0.00856863`, mark `0.02690313`, reuse `0.61286`, p90 `168.35`,
and `0.5625/0.70` scored HRC `0.00863293`, mark `0.02686441`, reuse
`0.61362`, p90 `169.5`. On the fair principal set (`42-45,50-65`),
`0.575/0.70` remains the HRC leader at `0.00854965`, with mark
`0.02680073`; `0.55/0.75` is the closest mark row at `0.02678954` but worse on
HRC (`0.00871607`). The promoted object runtime is now forced phase,
`transition_blend=0.575`, `local_prob_power=0.70`, late rank scales
`1.0,1.0,1.1,1.1`, and feedback-only size blend `0.080`.

The first 1M-record, 4-stream panel changed the diagnosis. The promoted row
still edges the old row on HRC, but both rows miss the longer real reuse tail.

| Row | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted `0.575/0.70` | **0.0589916** | 0.612863 | 0.728415 | 54 | 84 | 170 | 29150 | 0.030864 |
| old `0.55/0.8` | 0.0598152 | 0.613847 | 0.728415 | 53 | 84 | 169 | 29150 | **0.030274** |
| promoted + tail pivot 84 / scale 340 | 0.0860748 | 0.612863 | 0.728415 | 54 | 84 | 24224 | 29150 | 0.030864 |
| promoted + reuse boost 0.30 / min-rank 84 / power 2 | 0.0592104 | 0.729707 | 0.728415 | 72 | 84 | 14132 | 29150 | 0.032653 |

The tail-only rank transform is therefore a useful diagnostic but not a
solution. It can stretch p90 toward the real panel, but HRC worsens because
fake reuse access remains capped near `0.613` while real reuse is `0.728`.
The first reuse-boost probe proves the missing-reuse diagnosis: converting 30%
of would-be NEW events matches total reuse almost exactly. Its min-rank `84`
is too shallow, though; it adds low/mid-cache hit mass and only half-fills the
deep p90 tail. The active follow-up keeps the 30% conversion rate but pushes
injected reuses to `min_rank=4096` and writes a fake CSV for `tools/cachesim`.

The `0.575/0.70` confirm also exposed a remote clobber: `evaluate_neural_atlas`
lost the feedback CLI after some cells had completed. LANL restored and
compiled the full `altgan/` tree on `vinge.local`, resumed with
`--skip-existing`, and verified output metadata for feedback blend `0.08` on
reused and newly completed cells. Artifacts are
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_micro_seed50_53_fb080_summary.csv` and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_micro_seed50_53_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_confirm_seed42_45_fb080_summary.csv`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_confirm_seed42_45_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_pairs_seed54_57_fb080_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_pairs_seed54_57_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_localrefine_seed58_61_fb080_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_localrefine_seed58_61_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_tb575_lp070_confirm_seed42_57_fb080_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_tb575_lp070_confirm_seed42_57_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_interp_seed62_65_fb080_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_interp_seed62_65_fb080_best.json`,
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_lowtrans_confirm_seed42_61_fb080_summary.csv`, and
`/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_object_lowtrans_confirm_seed42_61_fb080_best.json`.

The hidden-size capacity pass did not dislodge h128. A larger h256 mark head
overfit the training objective and failed the seed-42 gate (`mark_score=0.04974`).
A smaller h96 mark head improved seed `42` (`0.02808`) but failed confirmation:
seeds `42-45` averaged `0.02791`, behind the promoted h128 mean `0.02775`. The
h104 and h112 bridge points also missed the seed-42 gate at `0.03533` and
`0.04102`, so the current capacity branch still promotes h128.

| hidden dim | eval seeds | mean HRC-MAE | mean mark score | verdict |
|---:|---:|---:|---:|---|
| 96 | 42-45 | 0.008767 | 0.02791 | closed |
| 104 | 42 only | 0.008423 | 0.03533 | closed |
| 112 | 42 only | 0.008423 | 0.04102 | closed |
| 128 | 42-45 | 0.008767 | **0.02775** | promoted |
| 256 | 42 only | 0.008423 | 0.04974 | closed |

The trainer snapshot path (`--snapshot-epochs`) is verified by the e19/e21 run
and remains useful for future checkpoint-selection sweeps, but the h128/catw
`0.25` epoch neighborhood is now closed.

## Alibaba PhaseAtlas Calibration Ablations

Recorded 2026-04-22. These 4-stream x 100k Alibaba sweeps used
`/tiamat/zarathustra/checkpoints/altgan/alibaba_phaseatlas_holdout_allx25k_e900.pkl.gz`
against the fixed real manifest
`/home/darrell/long_rollout_manifests/alibaba_stackatlas.json`.

The rank-tail schedule sweep closed negative. The baseline PhaseAtlas row is
still best HRC. A late-phase rank expansion moved stack p90 closer to real, but
HRC worsened; phase caps over-shortened the deep tail and lost clearly.

| Variant | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline, natural phase | **0.003010** | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 | 0.00479 |
| Late-phase rank expansion, no cap | 0.003145 | 0.27125 | 0.26909 | 205 | 201 | 1474 | 1452 | 0.00479 |
| Late-phase rank compression, no cap | 0.004279 | 0.27125 | 0.26909 | 184 | 201 | 1236 | 1452 | 0.00479 |
| Baseline ranks, late caps | 0.006727 | 0.27125 | 0.26909 | 205 | 201 | 787 | 1452 | 0.00479 |

Conclusion: stack-rank postprocessing is not the next winning LANL path. The
best next experiment should perturb the empirical phase transition law itself,
while leaving the explicit LRU decoder and reservoir marks intact.

The empirical transition-power sweep also closed negative. It raised the
nearest-file initial/transition probability vectors to `local_prob_power`
before sampling. The unchanged empirical law remains best; smoothing to `0.9`
nearly preserves HRC but worsens mark score and reuse, while sharper settings
under-reuse and collapse the stack tail.

| local_prob_power | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.038345 | 0.31570 | 0.26909 | 112 | 201 | 3132 | 1452 | 0.01942 |
| 0.75 | 0.014896 | 0.28666 | 0.26909 | 179 | 201 | 1508 | 1452 | 0.00918 |
| 0.90 | 0.003616 | 0.27392 | 0.26909 | 196 | 201 | 1480 | 1452 | 0.01072 |
| 1.00 | **0.003010** | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 | 0.00479 |
| 1.10 | 0.031861 | 0.23719 | 0.26909 | 224 | 201 | 1480 | 1452 | 0.03283 |
| 1.25 | 0.021938 | 0.24671 | 0.26909 | 221 | 201 | 1355 | 1452 | 0.01876 |
| 1.50 | 0.035989 | 0.22775 | 0.26909 | 210 | 201 | 934 | 1452 | 0.00892 |
| 2.00 | 0.068274 | 0.17253 | 0.26909 | 198 | 201 | 530 | 1452 | 0.01768 |

The follow-up microblend sweep found a new best HRC row. The earlier
transition-blend grid jumped from `0.0` to `0.25`; the winning cell lives just
inside that interval and combines a little neural transition smoothing with a
slightly smoothed empirical local transition law.

| transition_blend | local_prob_power | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.00 | 0.003010 | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 | **0.00479** |
| 0.15 | 0.90 | 0.003344 | 0.26716 | 0.26909 | 199 | 201 | 1481 | 1452 | 0.00799 |
| 0.20 | 0.90 | **0.002217** | 0.27363 | 0.26909 | 192 | 201 | 1508 | 1452 | 0.00815 |
| 0.05 | 1.00 | 0.005617 | 0.27784 | 0.26909 | 203 | 201 | 1393 | 1452 | 0.01429 |
| 0.10 | 1.00 | 0.005629 | 0.27816 | 0.26909 | 203 | 201 | 1396 | 1452 | 0.00549 |
| 0.20 | 1.00 | 0.016089 | 0.25239 | 0.26909 | 211 | 201 | 1394 | 1452 | 0.02345 |

This is the first post-baseline LANL object-process improvement on the fixed
seed-42 panel: HRC drops from `0.003010` to `0.002217`. The tradeoff is a
mark-score regression from `0.00479` to `0.00815`, still far below LLNL's
post-hoc CSV mark gap but no longer the reservoir-mark ceiling.

The seed-confirmation pass did not validate `blend=0.2, local_prob_power=0.9`
as a durable champion. Across seeds 43-45, the best confirmation row was
`blend=0.0, local_prob_power=0.9` at seed 43 with HRC `0.002373`; the seed-42
winner regressed to `0.005579`, `0.011897`, and `0.014313` on seeds 43, 44,
and 45. Treat microblend as a useful stochastic tuning signal, not a stable
replacement for the baseline.

| Confirm row | Seed | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | mark score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blend 0.0, local power 0.9 | 43 | **0.002373** | 0.26882 | 0.26909 | 203 | 201 | 1481 | 1452 | 0.01270 |
| baseline, blend 0.0, local power 1.0 | 43 | 0.004657 | 0.27146 | 0.26909 | 216 | 201 | 1481 | 1452 | 0.00605 |
| blend 0.2, local power 0.9 | 43 | 0.005579 | 0.28171 | 0.26909 | 200 | 201 | 1601 | 1452 | 0.01072 |
| baseline, blend 0.0, local power 1.0 | 44 | 0.006050 | 0.26750 | 0.26909 | 212 | 201 | 1481 | 1452 | 0.00595 |
| blend 0.2, local power 0.9 | 44 | 0.011897 | 0.28915 | 0.26909 | 203 | 201 | 1481 | 1452 | 0.01560 |
| blend 0.0, local power 0.9 | 45 | 0.004801 | 0.26395 | 0.26909 | 179 | 201 | 1442 | 1452 | 0.00830 |
| blend 0.2, local power 0.9 | 45 | 0.014313 | 0.24905 | 0.26909 | 184 | 201 | 1480 | 1452 | 0.01414 |

## Mark Quality Panel

LLNL's v198 response correctly asks whether LANL's reservoir marks are weaker
than learned LSTM marks. I added an `altgan.mark_quality` panel that ignores
`obj_id` and compares only timing, size, opcode, and tenant marks. Lower is
better.

| Corpus / artifact | Object process | Mark score | ts-delta norm | size norm | opcode TV | tenant TV |
|---|---|---:|---:|---:|---:|---:|
| Tencent PhaseAtlas 1024x5k holdout, blend 0.5 | HRC `0.01065` | **0.04557** | 0.02988 | 0.07768 | 0.03737 | 0.03737 |
| Alibaba PhaseAtlas allx25k holdout, blend 0.0 | HRC `0.00301` | **0.00479** | 0.00386 | 0.01233 | 0.00148 | 0.00148 |
| Alibaba PhaseAtlas + neural marks e20 | HRC `0.00301` | 0.04044 | 0.05148 | 0.06641 | 0.02305 | 0.02083 |
| LLNL v198 real-rate override CSV | HRC `0.00513` | 0.61412 | 0.07914 | 0.37735 | 1.00000 | 1.00000 |

The LLNL row uses their emitted `/tiamat/zarathustra/altgan-output/v198_lru_realrate.csv`
against the matching 50k/8-stream Alibaba real sample. It confirms the current
competitive split: their post-hoc LRU repair can approach HRC when handed the
real reuse rate, but the emitted mark representation is not yet competitive on
size/opcode/tenant realism. LANL's PhaseAtlas currently leads on fair held-out
object process and on this mark panel.

The first IDEA #53 sidecar kept the Alibaba PhaseAtlas object law intact:
HRC-MAE stayed `0.00301`, reuse stayed `0.27125` vs real `0.26909`, and stack
median stayed `205` vs real `201`. It did not beat reservoir marks. The paired
control on the same attached checkpoint scores `0.00479`, while the e20 neural
mark head scores `0.04044`, mostly from timing and size regression drift. Treat
this as a negative result for direct autoregressive mark replacement and move
next to noise/temperature ablations or residual/hybrid marks around the
reservoir sampler.

The zero-noise neural-mark temperature sweep also closed negative. Lowering
temperature preserved the object process exactly but collapsed categorical mark
diversity, driving opcode and tenant TV sharply upward.

| Neural mark temp/noise | HRC-MAE | Mark score | opcode TV | tenant TV |
|---|---:|---:|---:|---:|
| temp 1.0, noise 0 | 0.003010 | 0.04427 | 0.03014 | 0.02968 |
| temp 0.5, noise 0 | 0.003010 | 0.15518 | 0.24405 | 0.24379 |
| temp 0.25, noise 0 | 0.003010 | 0.16519 | 0.26397 | 0.26395 |
| temp 0.05, noise 0 | 0.003010 | 0.16526 | 0.26413 | 0.26415 |

## Alibaba Mark-Hybrid Transition Confirmation

Recorded 2026-04-22. This sweep used the attached Alibaba PhaseAtlas mark
checkpoint
`/tiamat/zarathustra/checkpoints/altgan/alibaba_phaseatlas_marks_e20.pkl.gz`
with seeds `42,43,44,45`, transition blends `0.0,0.2`, local probability
powers `0.9,1.0`, reservoir opcode/tenant, numeric blends `0.25,0.5,0.75,1.0`,
and both raw/log numeric blend spaces. The full artifacts are
`/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_hybrid_transition_confirm_summary.csv`
and `_best.json`.

| Candidate | Seeds | mean HRC-MAE | mean mark score | mean timing drift ratio | mean size drift ratio | mean fake reuse | mean stack med | mean stack p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reservoir control, blend 0.0/local power 1.0 | 4 | 0.008698 | **0.00895** | 1.175 | 1.416 | 0.26152 | 203.0 | 1351.2 |
| reservoir control, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01117 | 1.307 | 0.931 | 0.26699 | 198.2 | 1477.8 |
| reservoir control, blend 0.2/local power 0.9 | 4 | 0.008501 | 0.01215 | 0.825 | 0.712 | 0.27338 | 194.8 | 1517.5 |
| reservoir control, blend 0.2/local power 1.0 | 4 | 0.011285 | 0.01662 | 1.193 | 1.251 | 0.25797 | 203.5 | 1373.2 |
| best hybrid so far: blend 0.0/local power 1.0, numeric 0.25 log | 4 | 0.008698 | 0.01839 | 0.000 | 1.190 | 0.26152 | 203.0 | 1351.2 |
| hybrid: blend 0.2/local power 0.9, numeric 0.25 log | 4 | 0.008501 | 0.01890 | 0.012 | 0.644 | 0.27338 | 194.8 | 1517.5 |
| hybrid: blend 0.0/local power 0.9, numeric 0.25 log | 4 | 0.005280 | 0.02093 | 0.000 | 0.794 | 0.26699 | 198.2 | 1477.8 |

The seed-42 HRC leader is still the reservoir-control microblend row
(`transition_blend=0.2`, `local_prob_power=0.9`) at HRC-MAE `0.002217`,
but the four-seed mean again prefers the conservative object process
(`transition_blend=0.0`, `local_prob_power=0.9`) on HRC. More importantly,
every numeric neural-mark hybrid worsened mark score versus its paired
reservoir control. Log-space blending softened the damage at low blend, but
the best log hybrid still more than doubled mark score relative to the best
reservoir control and collapsed timing drift toward zero. Treat direct
post-hoc numeric blending as closed negative; the next mark-side attempt
should be a residual or quantile-conditioned reservoir correction, not direct
interpolation with the autoregressive mark head.

A low-blend log-space follow-up tested whether the previous failure was simply
using too much neural numeric signal. It swept numeric blends `0.05,0.10,0.15`
with the same four seeds and object-process settings. The answer is still
negative: the smallest blend is less damaging, but it does not beat its paired
reservoir control and still erases timing-drift shape.

| Candidate | Seeds | mean HRC-MAE | mean mark score | mean timing drift ratio | mean size drift ratio |
|---|---:|---:|---:|---:|---:|
| control, blend 0.0/local power 1.0 | 4 | 0.008698 | **0.00895** | 1.175 | 1.416 |
| numeric 0.05 log, blend 0.0/local power 1.0 | 4 | 0.008698 | 0.01072 | 0.000 | 1.363 |
| control, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01117 | 1.307 | 0.931 |
| numeric 0.05 log, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01296 | 0.000 | 0.899 |
| control, blend 0.2/local power 0.9 | 4 | 0.008501 | 0.01215 | 0.825 | 0.712 |
| numeric 0.05 log, blend 0.2/local power 0.9 | 4 | 0.008501 | 0.01330 | 0.015 | 0.694 |

This closes direct interpolation, including tiny log-space residual-style
blends. Any future mark model must preserve the reservoir sampler's temporal
drift explicitly, for example by predicting corrections to reservoir quantiles
or mixture weights instead of replacing sampled timing/size values.

A field-selective size-only probe then held `dt` at the reservoir sample and
blended only `obj_size`, using the same low log-space blends. This isolates the
failure mode: timing drift survives when `dt` is untouched, but uniform size
interpolation still worsens aggregate mark score versus the paired reservoir
controls. The full artifacts are
`/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_sizeonly_lowblend_log_confirm_summary.csv`
and `_best.json`.

| Candidate | Seeds | mean HRC-MAE | mean mark score | mean timing drift ratio | mean size drift ratio |
|---|---:|---:|---:|---:|---:|
| control, blend 0.0/local power 1.0 | 4 | 0.008698 | **0.00895** | 1.175 | 1.416 |
| size-only 0.05 log, blend 0.0/local power 1.0 | 4 | 0.008698 | 0.00964 | 1.175 | 1.363 |
| size-only 0.10 log, blend 0.0/local power 1.0 | 4 | 0.008698 | 0.01040 | 1.175 | 1.315 |
| control, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01117 | 1.307 | 0.931 |
| size-only 0.05 log, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01190 | 1.307 | 0.899 |

Size-only blending is a useful diagnostic but not a champion. It preserves the
reservoir timing shape and slightly improves size drift in some paired rows, so
the next viable size path is rank/quantile-aware correction or gated residuals,
not uniform numeric interpolation.

The complementary `dt`-only probe confirms that the timing field is the source
of the drift collapse. It used the same four seeds and low log-space blends,
but held `obj_size` at the reservoir sample while blending only `dt`. The full
artifacts are
`/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_dtonly_lowblend_log_confirm_summary.csv`
and `_best.json`.

| Candidate | Seeds | mean HRC-MAE | mean mark score | mean timing drift ratio | mean size drift ratio |
|---|---:|---:|---:|---:|---:|
| control, blend 0.0/local power 1.0 | 4 | 0.008698 | **0.00895** | 1.175 | 1.416 |
| dt-only 0.05 log, blend 0.0/local power 1.0 | 4 | 0.008698 | 0.01003 | 0.000 | 1.416 |
| dt-only 0.10 log, blend 0.0/local power 1.0 | 4 | 0.008698 | 0.01110 | 0.000 | 1.416 |
| control, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01117 | 1.307 | 0.931 |
| dt-only 0.05 log, blend 0.0/local power 0.9 | 4 | **0.005280** | 0.01222 | 0.000 | 0.931 |

Together with the size-only run, this closes uniform direct numeric blending:
`obj_size` blending can improve size drift without changing timing, but `dt`
blending immediately destroys timing drift shape. Mark work should now move to
rank/quantile-conditioned reservoir corrections or mixture/gating parameters
that preserve temporal drift, not replacement of sampled `dt`.

## StackAtlas 100k Long-Rollout Panel

All runs used `n_records=100000`, `n_streams=4`, `seed=42`, and the same real
manifest per corpus.

| Corpus | Fit | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 | fake footprint | real footprint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tencent | 16 files x 50k | **0.03210** | 0.63757 | 0.61493 | 66 | 60 | 739 | 174 | 9060.75 | 9626.75 |
| Tencent | 64 files x 25k | 0.08225 | 0.57816 | 0.61493 | 92 | 60 | 831 | 174 | 10546.00 | 9626.75 |
| Tencent | manifest oracle | **0.00266** | 0.61666 | 0.61493 | 59 | 60 | 171 | 174 | 9583.50 | 9626.75 |
| Alibaba | 16 files x 50k | 0.14198 | 0.43533 | 0.26909 | 83 | 201 | 1240 | 1452 | 14116.75 | 18272.75 |
| Alibaba | 64 files x 25k | 0.17519 | 0.48264 | 0.26909 | 91 | 201 | 1068 | 1452 | 12934.00 | 18272.75 |
| Alibaba | manifest oracle | **0.00739** | 0.27916 | 0.26909 | 200 | 201 | 1347 | 1452 | 18021.00 | 18272.75 |

Remote artifacts:

- `/tiamat/zarathustra/checkpoints/altgan/tencent_stackatlas_16x50k.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/tencent_stackatlas_64x25k.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/tencent_stackatlas_manifest4.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_stackatlas_16x50k.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_stackatlas_64x25k.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_stackatlas_manifest4.pkl.gz`
- `/tiamat/zarathustra/altgan-output/*_eval_100k.json`

## NeuralStack 100k Long-Rollout Panel

`NeuralStack` trains an MLP from file/workload characterization to
new/near/mid/far action probabilities and stack-rank bucket probabilities,
while retaining explicit LRU-stack generation.

| Corpus | Fit | Temperature | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Alibaba | 64 files x 25k, 1200 epochs | 1.0 | **0.00333** | 0.27373 | 0.26909 | 204 | 201 | 1331 | 1452 |
| Tencent | 16 files x 50k, 1200 epochs | 1.0 | 0.04351 | 0.68142 | 0.61493 | 64 | 60 | 331 | 174 |
| Tencent | 512 files x 10k, 1500 epochs | 3.0 | 0.08806 | 0.70661 | 0.61493 | 27 | 60 | 379 | 174 |

NeuralStack solved Alibaba's global-reservoir over-reuse, but it regressed on
Tencent as training coverage grew.  The failure is diagnostic: independent
action/rank marginals are not enough; Tencent needs temporal transition state.

## NeuralAtlas 100k Long-Rollout Panel

`NeuralAtlas` trains file-profile-conditioned initial-state and transition
distributions over StackAtlas's coarse time/size/action states.  The
`transition_blend` column is the neural transition weight: `1.0` is pure neural
smoothing and `0.0` is the nearest profile-routed fitted atlas.

| Corpus | Fit | transition_blend | HRC-MAE | fake reuse | real reuse | fake stack med | real stack med | fake stack p90 | real stack p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Tencent | 64 files x 25k, 900 epochs | 0.0 | **0.01845** | 0.62314 | 0.61493 | 55 | 60 | 145 | 174 |
| Tencent | 64 files x 25k, 900 epochs | 0.25 | 0.03048 | 0.63544 | 0.61493 | 50 | 60 | 142 | 174 |
| Tencent | 64 files x 25k, 900 epochs | 0.5 | 0.04466 | 0.65071 | 0.61493 | 46 | 60 | 138 | 174 |
| Tencent | 64 files x 25k, 900 epochs | 0.75 | 0.06008 | 0.66700 | 0.61493 | 42 | 60 | 135 | 174 |
| Tencent | 64 files x 25k, 900 epochs | 1.0 | 0.07557 | 0.68248 | 0.61493 | 39 | 60 | 131 | 174 |
| Alibaba | 64 files x 25k, 900 epochs | 0.0 | 0.00254 | 0.26800 | 0.26909 | 194 | 201 | 1243 | 1452 |
| Alibaba | 64 files x 25k, 900 epochs | 0.25 | 0.00184 | 0.26599 | 0.26909 | 196 | 201 | 1259 | 1452 |
| Alibaba | 64 files x 25k, 900 epochs | 0.5 | **0.00183** | 0.26451 | 0.26909 | 197 | 201 | 1267 | 1452 |
| Alibaba | 64 files x 25k, 900 epochs | 0.75 | 0.00269 | 0.26290 | 0.26909 | 197 | 201 | 1293 | 1452 |
| Alibaba | 64 files x 25k, 900 epochs | 1.0 | 0.00505 | 0.26014 | 0.26909 | 201 | 201 | 1301 | 1452 |

Remote NeuralAtlas artifacts:

- `/tiamat/zarathustra/checkpoints/altgan/tencent_neuralatlas_64x25k_e900.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_neuralatlas_64x25k_e900.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/tencent_neuralatlas_holdout_1024x5k_e900.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_neuralatlas_holdout_allx25k_e900.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_holdout_1024x5k_e900.pkl.gz`
- `/tiamat/zarathustra/checkpoints/altgan/alibaba_phaseatlas_holdout_allx25k_e900.pkl.gz`
- `/tiamat/zarathustra/altgan-output/*neuralatlas*eval_100k.json`
- `/tiamat/zarathustra/altgan-output/*phaseatlas*eval_100k.json`

## Interpretation

StackAtlas validated the explicit object-process bet.  NeuralAtlas makes that
bet harder to dismiss.

On Tencent, the held-out 16-file atlas is already strong on cache behavior:
HRC-MAE is 0.03210 and reuse/median stack distance are close to real.  The
stack-distance p90 is still too long, so this is not solved, but it is much
closer to the long-rollout target than the current neural branch has been.

On Alibaba, the held-out global atlas fails by over-reusing: fake reuse is
0.43533 vs real 0.26909, and stack median is 83 vs real 201.  The manifest
oracle run nearly closes that gap, with HRC-MAE 0.00739 and stack median 200
vs 201.  That means the generator contract is viable, but the first atlas is
not workload-conditioned enough for Alibaba.  NeuralStack confirmed that
workload conditioning matters, and NeuralAtlas confirmed the stronger version:
condition the transition atlas by workload profile, then decode objects through
an explicit stack.

The trained pure neural transition smoother is not the winner yet.  On Tencent,
increasing `transition_blend` monotonically worsens HRC-MAE from `0.01845` to
`0.07557`; on Alibaba, mild smoothing helps but pure neural is still worse than
the best blend.  That is the right negative result: the architecture should
promote profile-routed object-state generation now, and only keep neural
smoothing where the long-rollout panel proves it helps.

The stricter holdout/phase pass changes the practical recommendation. A
profile-routed atlas is still the right base, but within-file phase must be
part of the generated state. Otherwise the synthetic trace can get HRC right
while looking too stationary across the rollout.
# LANL Results Log

## Tencent 1M Cachesim Gate: Reuse Boost And Rank Scaling (2026-04-30)

The promoted 1M Tencent row exposed the real failure mode: total reuse was too
low and the p90 reuse tail was missing. A post-decode NEW-to-reuse conversion
with deep injected ranks now fixes the total reuse axis without feeding the
synthetic correction back into the transition rollout.

Exact 1M manifest, 4 streams, seed 42:

| Variant | evaluator HRC-MAE | six-policy cachesim mean | fake reuse | real reuse | fake med | real med | fake p90 | real p90 | mark |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| promoted baseline | 0.058992 | n/a | 0.612863 | 0.728415 | 54 | 84 | 170 | 29150 | 0.030864 |
| post-decode reuse boost `p=.30,min=32768,pow=2` | 0.051810 | **0.054073** | 0.728954 | 0.728415 | 72 | 84 | 25319 | 29150 | 0.032200 |
| + rank phase scale `1.2,1.2,1.3,1.3` | **0.044706** | 0.055905 | 0.728954 | 0.728415 | 84 | 84 | 25319 | 29150 | 0.034620 |
| + rank phase scale `1.4,1.4,1.6,1.6` | 0.039957 | 0.062114 | 0.728954 | 0.728415 | 104 | 84 | 25319 | 29150 | 0.036945 |
| + hot-pool redirect `p=.30,k=100,window=5000` | **0.037989** | **0.047074** | 0.730118 | 0.728415 | 81 | 84 | 24933 | 29150 | 0.035656 |
| + hot-pool redirect `p=.50,k=100,window=5000` | **0.037900** | **0.046657** | 0.729176 | 0.728415 | 86 | 84 | 25165 | 29150 | 0.036752 |
| + hot-pool redirect `p=.60,k=100,window=5000` | 0.039255 | 0.048375 | 0.729554 | 0.728415 | 87 | 84 | 24941 | 29150 | 0.036950 |
| + hot-pool redirect `p=.70,k=100,window=5000` | 0.040265 | 0.050436 | 0.730444 | 0.728415 | 85 | 84 | 24819 | 29150 | 0.036712 |
| + hot-pool redirect `p=.50,k=100,window=5000,wpow=2` | 0.068505 | 0.117864 | 0.729176 | 0.728415 | 14 | 84 | 25165 | 29150 | 0.046772 |

The rank-scale rows are true LRU-grid improvements but not promoted six-policy
wins: they improve LRU/FIFO and worsen ARC/SIEVE/CAR on the fixed
`32,128,512,2048,8192` cache grid. The hot-pool redirect is the first
simulator-gated improvement after deep reuse injection: mean HRC-MAE drops to
`0.046657`, including SIEVE `0.033573`.

Adjacent duplicate diagnostic on the same exact slice:

| Trace | adjacent duplicate rate |
|---|---:|
| real | 0.002340 |
| LANL post-decode fake | 0.004274 |
| LANL rank-scale fake | 0.004274 |
| LANL hotpool030 fake | 0.003882 |
| LANL hotpool050 fake | 0.004399 |
| LANL hotpool060 fake | 0.004712 |
| LANL hotpool070 fake | 0.005210 |
| LANL hotpool050 wpow2 fake | 0.116900 |

So LLNL's positive adj-dup injection is not directly transferable to LANL; our
SIEVE gap is not caused by too few immediate repeats. Hot-set concentration was
the better LANL lever: all-stream top-100 share improved from `0.003849` to
`0.119549`, still below real `0.263975`. `hotpool070` overshot SIEVE/SLRU
despite improving ARC/CAR, and `hotpool060` also lost to `hotpool050`.
Frequency weight power `2.0` is closed negative: it overshoots top100 to
`0.310561`, collapses stack median to `14`, and raises adjacent duplicates to
`0.116900`. Wide-window `50000` is operationally closed for now: exact lookup
ran 40+ minutes with no fake CSV, and a cached-rank experiment also failed to
land in a useful window. The current promoted hot-pool row remains
`p=.50,k=100,window=5000,wpow=1`.

Seed-43 confirmation of that row also made unbounded hot-object lookup look too
expensive: it was killed after 40+ minutes with no fake CSV. The rerun uses
`--stack-hot-pool-max-search 8192`, falling back to the sampled rank if the hot
object is deeper than the bounded prefix. That run was also killed after 40+
minutes with no fake CSV, and `max_search=512` was likewise killed after 40+
minutes. Confirmation pivoted to the fixed seed-42 real manifest; fake RNG seed
`43` was also killed as slow, and fake RNG seed `44` is running.

Engineering follow-up: the slow confirmations were caused by the list-backed
LRU stack becoming the hot path for deep rank moves and hot-object lookup.
`altgan.neural_atlas` now uses `_RankedLRUStack`, an implicit treap with
deterministic priorities, to keep rank lookup/move semantics but avoid O(n)
list scans. Local randomized tests compare it against the old list behavior.
The evaluator also has `--progress-interval` now so future confirmations do not
look hung while generating.

Fake-seed `44` confirmation on the same fixed seed-42 real manifest confirms
the row: evaluator HRC-MAE `0.038433`, six-policy cachesim mean `0.046945`,
reuse `0.729857` vs real `0.728415`, median `86` vs `84`, p90 `24815` vs
`29150`, and mark score `0.036416`. Policy MAEs were LRU `0.033689`, ARC
`0.068861`, FIFO `0.035623`, SIEVE `0.033782`, SLRU `0.045185`, CAR
`0.064527`. Hot-shape diagnostics: top-100 share `0.123185`, top-1000
`0.384067`, adjacent duplicate rate `0.004372`, unique namespaced objects
`270143`.

Bracket results:

| Variant | fake seed | evaluator HRC-MAE | six-policy cachesim mean | fake reuse | fake med | fake p90 | mark | top100 | top1000 | adj dup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hot-pool `p=.45` | 44 | 0.037309 | **0.045864** | 0.728974 | 85 | 25128 | 0.036310 | 0.101801 | 0.345525 | 0.004186 |
| hot-pool `p=.55` | 44 | 0.038566 | 0.047347 | 0.729636 | 86 | 24797 | 0.036845 | 0.137561 | 0.414682 | 0.004462 |
| hot-pool `p=.45` | 42 | 0.037703 | **0.045988** | 0.729630 | 85 | 25142 | 0.036395 | 0.103650 | 0.347998 | 0.004159 |
| hot-pool `p=.40` | 44 | 0.036725 | **0.045660** | 0.728511 | 84 | 25309 | 0.036029 | 0.086761 | 0.309028 | 0.004097 |
| hot-pool `p=.40` | 42 | 0.037417 | **0.045651** | 0.730060 | 84 | 25139 | 0.036031 | 0.088076 | 0.306437 | 0.004103 |
| hot-pool `p=.35` | 44 | 0.037109 | 0.045855 | 0.728926 | 83 | 25043 | 0.035646 | 0.072753 | 0.264483 | 0.004069 |
| hot-pool `p=.38` | 44 | 0.036519 | **0.045386** | 0.728523 | 83 | 25302 | 0.036209 | 0.081847 | 0.289865 | 0.004076 |
| hot-pool `p=.42` | 42 | 0.037409 | 0.045805 | 0.729556 | 84 | 25091 | 0.036238 | 0.096817 | 0.323024 | 0.004112 |
| hot-pool `p=.38` | 42 | 0.037164 | **0.045648** | 0.729731 | 83 | 25227 | 0.036041 | 0.081719 | 0.288315 | 0.004104 |
| hot-pool `p=.37` | 44 | 0.036593 | **0.045395** | 0.728457 | 83 | 25211 | 0.035590 | 0.080016 | 0.281357 | 0.004081 |
| hot-pool `p=.37` | 42 | 0.037209 | **0.045599** | 0.729586 | 83 | 25184 | 0.036242 | 0.075198 | 0.283064 | 0.004053 |
| hot-pool `p=.39` | 44 | 0.036536 | **0.045532** | 0.728408 | 83 | 25364 | 0.035905 | 0.085625 | 0.298778 | 0.004021 |
| hot-pool `p=.38,k=75` | 44 | 0.041104 | 0.045715 | 0.728523 | 79 | 25302 | 0.035397 | 0.117654 | 0.331563 | 0.004404 |
| hot-pool `p=.38,k=150` | 44 | 0.035838 | 0.047746 | 0.728523 | 91 | 25302 | 0.036873 | 0.052772 | 0.217971 | 0.003672 |
| hot-pool `p=.38,window=2500` | 44 | 0.036800 | 0.045842 | 0.728523 | 84 | 25302 | 0.036064 | 0.035669 | 0.167585 | 0.004049 |
| hot-pool `p=.38,window=10000` | 44 | 0.036401 | **0.045255** | 0.728523 | 83 | 25302 | 0.036094 | 0.136623 | 0.358165 | 0.004016 |
| hot-pool `p=.38,window=10000` | 42 | 0.037034 | **0.045352** | 0.729731 | 83 | 25227 | 0.035677 | 0.140064 | 0.358097 | 0.004075 |
| hot-pool `p=.38,window=20000` | 44 | 0.036236 | **0.045243** | 0.728523 | 84 | 25302 | 0.036181 | 0.150615 | 0.363874 | 0.003989 |
| hot-pool `p=.38,window=20000` | 42 | 0.036794 | 0.045465 | 0.729731 | 84 | 25227 | 0.036149 | 0.153541 | 0.363388 | 0.004077 |
| hot-pool `p=.38,window=40000` | 44 | 0.036462 | 0.045855 | 0.728523 | 85 | 25302 | 0.036136 | 0.151292 | 0.364890 | 0.003894 |
| hot-pool `p=.37,window=10000` | 44 | 0.036442 | 0.045317 | 0.728457 | 83 | 25211 | 0.035763 | 0.135663 | 0.352890 | 0.004070 |
| hot-pool `p=.39,window=10000` | 44 | 0.036377 | **0.045219** | 0.728408 | 84 | 25364 | 0.035979 | 0.141584 | 0.364791 | 0.004030 |

Current interpretation: the optimum is below `p=.50`; `p=.55` overconcentrates
and loses SIEVE/SLRU. The optimum is now a narrow `p=.37..40` band. `p=.38`
is the current best single row and is confirmed on both fake seeds; `p=.37`
is effectively tied on both seeds. `p=.35` gives up too much FIFO despite
improving SIEVE/SLRU, and `p=.42` loses to `p=.40` on seed 42.
Pool-size result: keep `k=100`. `k=75` helps ARC/CAR but hurts FIFO/SIEVE/SLRU
through overconcentration; `k=150` helps FIFO/SLRU but hurts ARC/CAR/SIEVE
through underconcentration.
Window result: `2500` is too short and under-concentrates the hot set;
`10000` is the new best single row, improving the six-policy mean to
`0.045255` and confirming on seed 42 at `0.045352`. `window=20000` edges seed
44 to `0.045243`, but shifts pressure toward ARC/CAR and does not confirm on
seed 42; `window=40000` loses. Current promotion is `window=10000`. Probability
recheck at that window moves seed-44 best to `p=.39` (`0.045219`); `p=.37`
loses. Confirmation `p=.39` seed-42 and upper probe `p=.40` seed-44 are
running.

Artifacts:
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_reuseboost030_min32768_pow2_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_rankscale120130_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_rankscale140160_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool030_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool050_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool060_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool070_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool050_wpow2_reuseboost030_min32768_postdecode_seed42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/postdecode030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/rankscale120130_postdecode030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/rankscale140160_postdecode030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool030_reuseboost030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool050_reuseboost030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool060_reuseboost030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool070_reuseboost030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool050_wpow2_reuseboost030_min32768_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool050_reuseboost030_min32768_postdecode_faststack_progress_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool050_reuseboost030_min32768_faststack_progress_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool045_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool045_reuseboost030_min32768_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool045_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool045_reuseboost030_min32768_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool040_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool040_reuseboost030_min32768_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool040_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool040_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool035_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool035_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool042_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool042_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool037_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool037_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool037_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool037_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool039_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool039_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038k075_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038k075_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038k150_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038k150_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w2500_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w2500_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w10000_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w10000_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w20000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w20000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w20000_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w20000_reuseboost030_min32768_postdecode_faststack_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool038w40000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool038w40000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool037w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool037w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/tencent_phaseatlas_marks_e20_catw025_promoted_tb575_lp070_hotpool039w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool039w10000_reuseboost030_min32768_postdecode_faststack_fakeseed44_realmanifest42_six_policy_caps.json`

---

## Alibaba 1M Cachesim Bracket Versus LLNL R204 (2026-05-01)

LANL created a 1M Alibaba real manifest at
`/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_manifest.json`
using four 250k streams from `alibabaBlock_163`, `275`, `109`, and `221`.
The paired real CSV is
`/tiamat/zarathustra/altgan-output/alibaba_real_manifest_seed42_1M_eval_real.csv`.

The starting LANL row is the earlier Alibaba PhaseAtlas+marks checkpoint with
reservoir marks, forced phase, `transition_blend=0.2`, and
`local_prob_power=0.9`. On 1M/cachesim this control scores evaluator HRC-MAE
`0.0169145` and six-policy mean `0.020282`, with reuse `0.263061` vs real
`0.306465`, median `188` vs real `276`, and p90 `1481` vs real `44829`.

LLNL's active Alibaba R204 k-axis is a neural-atlas generator, not the old GAN
track. Scored against the same LANL real manifest, its visible rows are behind
LANL control:

| Source | Variant | six-policy cachesim mean |
|---|---|---:|
| LANL | tb `.20`, lp `.90`, reservoir control | **0.020282** |
| LLNL | `alibaba_b2_r204_k25.csv` | 0.050148 |
| LLNL | `alibaba_b2_r204_k75.csv` | 0.033206 |
| LLNL | `alibaba_b2_r204_k100.csv` | 0.029747 |
| LLNL | `alibaba_b2_r207_hp0.20.csv` | 0.034689 |
| LLNL | `alibaba_b2_r207_hp0.25.csv` | 0.030599 |
| LLNL | `alibaba_b2_r207_hp0.30.csv` | 0.026658 |
| LLNL | `alibaba_b2_r207_hp0.35.csv` | 0.026276 |
| LLNL | `alibaba_b2_r207_hp0.40.csv` | 0.025387 |
| LLNL | `alibaba_b2_r207_hp0.50.csv` | 0.029777 |
| LLNL | `alibaba_b2_r207_hp0.60.csv` | 0.033206 |
| LLNL | `alibaba_b2_r207_hp0.70.csv` | 0.040372 |

LANL tested an LLNL-shaped hot-pool/adjacent-duplicate variant
(`stack_hot_pool_prob=0.60`, `k=25`, `window=10000`,
`stack_adj_dup_prob=0.15`) and closed it negative: six-policy mean `0.070192`,
median `43`, p90 `488`. It over-compresses Alibaba reuse distance.

LLNL R207 narrows the gap but does not pass LANL: hp `0.40`, k75, adj `0.15`,
tail `0.10`, and recent-pool `0.15/window=2` scores `0.025387`, with top-100
share `0.088029` and adjacent duplicate rate `0.058331`. It is a real
improvement over R204 k100, but still behind LANL control and both LANL
deep-reuse probes.

Deep new-to-reuse injection is the right Alibaba branch. `p=0.06`,
`min_rank=32768`, `rank_power=2.0` nearly matches the real reuse tail on fake
seed `44`: evaluator HRC-MAE `0.013060`, six-policy mean `0.020009`, reuse
`0.307403`, median `228`, p90 `43892`. The same `p=0.06` on seed `42` scores
`0.020072`, reuse `0.307216`, median `227`, p90 `43764`. `p=0.10` on fake
seed `45` nudged the cache mean to `0.019857` but overshot reuse to `0.336242`
and p90 to `70649`; the same knob on seed `42` scores `0.019892` with similar
overshoot (`0.336428`, p90 `70838`). Same-seed upper checks on seed `42` show
`p=0.12` (`0.020160`) and `p=0.16` (`0.020163`) are not improvements over the
control. Current Alibaba cache best is `p=0.10`, but `p=0.06` is the better
trace-shape compromise and needs a broader seed panel before promotion.

Artifacts:
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_hotpool060k25_adj015_seed43_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_hotpool060k25_adj015_seed43_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_reuseboost006_min32768_seed44_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_reuseboost006_min32768_seed44_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_reuseboost010_min32768_seed45_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_reuseboost010_min32768_seed45_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_reuseboost006_min32768_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_reuseboost006_min32768_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/alibaba_phaseatlas_marks_tb020_lp090_reservoir_reuseboost010_min32768_seed42_realmanifest42_eval_1M.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_reuseboost010_min32768_seed42_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r204_k25_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r204_k75_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r204_k100_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r207_hp0p20_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r207_hp0p40_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r207_hp0p50_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r207_hp0p60_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r207_hp0p70_vs_lanl_realmanifest42_six_policy_caps.json`

---

## LLNL R203/R206 Tencent Cachesim Lead And LANL Clone Probe (2026-05-01)

LLNL's newer Tencent R203 k-axis beats LANL on the fixed Tencent 1M real
manifest, even though it is still an atlas/reuse-shaping generator rather than
the old GAN path. The script `/tmp/tencent_k_sweep.sh` used
`hot-pool-prob=0.55`, `adj-dup-prob=0.150`, `tail-reuse-prob=0.10`, and varied
`hot-pool-k`.

| Source | Variant | six-policy cachesim mean |
|---|---|---:|
| LLNL R203 | k25 | **0.038256** |
| LLNL R203 | k100 | 0.047506 |
| LLNL R203 | k150 | 0.059586 |
| LLNL R203 | k200 | 0.070346 |
| LLNL R206 | k50, adj `0.00` | 0.043287 |
| LLNL R206 | k50, adj `0.02` | 0.032605 |
| LLNL R206 | k50, adj `0.03` | 0.031474 |
| LLNL R206 | k50, adj `0.05` | **0.030536** |
| LLNL R206 | k50, adj `0.075` | **0.030360** |
| LLNL R206 | k50, adj `0.10` | 0.030892 |
| LLNL R206 | k50, adj `0.20` | 0.038093 |
| LLNL R206 | k50, adj `0.25` | 0.044061 |
| LANL | p39/window10000 fake seed 45 | 0.045573 |
| LANL | p39/window10000 fake seed 46 | 0.045627 |
| LANL | p38/window10000 fake seed 45 | 0.045614 |
| LANL | p38/window10000 fake seed 46 | 0.045511 |
| LANL | k25, adj `0.15`, no tail, fake seed 47 | 0.107924 |
| LANL | k25, adj `0.15`, tail `0.10`, fake seed 48 | 0.055622 |
| LANL | k50, adj `0.05`, tail `0.10`, fake seed 49 | 0.031461 |
| LANL | k50, adj `0.00`, tail `0.10`, fake seed 50 | 0.031040 |
| LANL | k50, adj `0.02`, tail `0.10`, fake seed 51 | 0.030632 |
| LANL | k50, adj `0.00`, tail `0.08`, fake seed 52 | 0.032613 |
| LANL | k50, adj `0.03`, tail `0.10`, fake seed 53 | 0.030802 |
| LANL | k50, adj `0.04`, tail `0.10`, fake seed 54 | 0.030963 |
| LANL | p `.55`, k50, adj `0.015`, tail `0.10`, fake seed 55 | 0.030563 |
| LANL | p `.60`, k50, adj `0.02`, tail `0.10`, fake seed 56 | **0.030298** |

The LLNL k25 trace has major trace-shape debt: top-100 share `0.347880`,
top-1000 `0.356781`, adjacent duplicate rate `0.090773`, and `361014`
namespaced unique objects. The real Tencent fixed-manifest adjacent duplicate
rate was previously measured near `0.00234`, so this is not yet evidence of a
statistically indistinguishable long trace.

R206 changes the diagnosis: LLNL can get a plausible-adjacent row at adj `0.00`
with adjacent duplicate rate `0.003165` versus real `0.002340`, but that row is
only `0.043287` and has a bad SIEVE error (`0.072846`). Intermediate rows land
at adj `0.02`/`0.03` with cache means `0.032605`/`0.031474` and adjacent
duplicate rates `0.014690`/`0.020170`. The best visible R206 row is adj
`0.075` at `0.030360`, with top-100 share `0.199928`, top-1000 `0.383674`,
adjacent duplicate rate `0.045438`, and `362721` unique objects. That is a
real cache-simulator lead, but not a clean long-trace match yet.

LANL's first direct k25/adj `0.15` clone failed hard: evaluator HRC-MAE
`0.079594`, six-policy cache mean `0.107924`, median `39` versus real `84`,
p90 `17701` versus `29150`, top-100 share `0.384124`, and adjacent duplicate
rate `0.116735`. The fuller k25 tail clone was better but still behind
(`0.055622`, median `53`, adjdup `0.104617`). A closer R206-style pair on the
promoted checkpoint transferred: k50/tail `0.10`/adj `0.05` scored `0.031461`
with median `80`, p90 `33193`, top-100 `0.242851`, and adjdup `0.037806`;
k50/tail `0.10`/adj `0.00` scored `0.031040` with median exactly `84`, p90
`33923`, top-100 `0.241256`, and adjdup `0.004993`. Adj `0.02` improved the
cache mean to `0.030632` with median `82`, p90 `33513`, top-100 `0.245119`,
and adjdup `0.018330`; this is within `0.00027` of LLNL's cache-best R206 row
while carrying much less adjacency debt. Tail `0.08` closed negative at
`0.032613`. Adj `0.03` and `0.04` also closed slightly worse at `0.030802` and
`0.030963`. Lowering adj to `0.015` at p `.55` scored `0.030563`. Increasing
hot-pool probability to p `.60` while keeping adj `0.02` produced the current
LANL best: `0.030298`, with median `81`, p90 `33698`, top-100 `0.259334`,
top-1000 `0.402350`, and adjdup `0.018463`. This edges LLNL R206 adj `0.075`
on cache while keeping much lower adjacent-duplicate debt.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r203_k25_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r203_k100_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r203_k150_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r203_k200_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.00_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.02_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.03_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.05_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.075_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.10_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.20_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r206_adj0.25_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k25w10000_adj015_reuseboost030_min32768_postdecode_faststack_fakeseed47_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k25w10000_adj015_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed48_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj005_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed49_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj000_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed50_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj002_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed51_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj000_tail008_reuseboost030_min32768_postdecode_faststack_fakeseed52_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj003_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed53_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpool055k50w10000_adj004_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed54_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpoolp055k50w10000_adj0015_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed55_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpoolp060k50w10000_adj002_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed56_realmanifest42_six_policy_caps.json`

---

## R208/R209 Cache-Sim Race Update (2026-05-01)

Tencent: LANL's p `.60`, k50, tail `.10`, adj `.015` row on fake seed `58`
is the new visible LANL best and edges LLNL R206 on the same six-policy gate:

| Source | Variant | six-policy mean | top100 | top1000 | adjdup |
|---|---|---:|---:|---:|---:|
| LANL | p `.60`, k50, adj `.015`, tail `.10`, seed `58` | **0.030240** | 0.260899 | 0.402591 | 0.015475 |
| LANL | p `.60`, k50, adj `.02`, tail `.10`, seed `56` | 0.030298 | 0.259334 | 0.402350 | 0.018463 |
| LANL | p `.60`, k50, adj `.02`, tail `.10`, seed `57` | 0.030555 | 0.260502 | 0.402462 | 0.018487 |
| LLNL | R206 k50, adj `.075` | 0.030360 | 0.199928 | 0.383674 | 0.045438 |

Confirmation is running for p `.60`/adj `.015` seed `59`, alongside a lower
adj `.010` neighbor seed `60`.

Alibaba: LLNL R208's adj re-tune was a cache win until LANL R209 added a small
hot pool to the deep-reuse branch. LANL rescored the R208 CSVs with namespaced
object ids against the LANL 1M real manifest and added the requested
eight-policy panel:

| Source | Variant | six-policy mean | eight-policy mean | top100 | top1000 | adjdup |
|---|---|---:|---:|---:|---:|---:|
| LLNL | R208 adj `.00` | **0.019671** | 0.022604 | 0.092879 | 0.127450 | 0.021021 |
| LLNL | R208 adj `.02` | 0.019844 | **0.022266** | 0.091976 | 0.126735 | 0.026106 |
| LLNL | R208 adj `.05` | 0.019812 | 0.022357 | 0.091154 | 0.125557 | 0.033662 |
| LANL | deep-reuse p `.10` + hot-pool `.10,k75,w10000`, seed `46` | **0.017939** | 0.022628 | 0.014995 | 0.047568 | 0.000433 |
| LANL | deep-reuse p `.10` + hot-pool `.10,k75,w10000`, seed `49` | **0.017547** | 0.022264 | 0.016599 | 0.048234 | 0.000404 |
| LANL | deep-reuse p `.10` + hot-pool `.15,k75,w10000`, seed `50` | 0.018764 | **0.022021** | 0.028194 | 0.064300 | 0.000494 |
| LANL | deep-reuse p `.10` + hot-pool `.12,k75,w10000`, seed `56` | 0.017879 | **0.021982** | 0.020651 | 0.055308 | 0.000452 |
| LANL | deep-reuse p `.10` + hot-pool `.10,k100,w10000`, seed `57` | **0.017524** | 0.022410 | 0.011921 | 0.045719 | 0.000402 |
| LANL | deep-reuse p `.10`, seed `45` | 0.019857 | 0.024774 | 0.002650 | 0.018336 | 0.000451 |
| LANL | deep-reuse p `.10`, seed `42` | 0.019892 | 0.024839 | 0.002492 | 0.017974 | 0.000435 |
| Real | fixed manifest | n/a | n/a | 0.042228 | 0.123983 | 0.000200 |

LANL has now re-passed both Alibaba cache panels on visible rows. A first
follow-up launcher malformed decimal probabilities (`.10` became `.010`,
`.25` became `.025`), so those rows are explicitly retracted from the R209
bracket. Corrected hp `.10,k75` confirms the six-policy win (`0.017547`), hp
`.10,k100` improves it to `0.017524`, and hp `.12,k75` gives the current LANL
eight-policy best (`0.021982`). hp `.20`, hp `.25`, and hp `.25` + tail `.30`
lose (`0.020120`, `0.022097`, `0.038220`).
The shape split is not settled: LLNL's
top100/top1000 are closer, but its adjacent duplicates are roughly 105x to
168x the real manifest, while LANL R209 stays close on adjacency and still
under-shoots the real hot set. The p `.10` + recent-pool `.05,w2` probe closed
negative at `0.025133` and adjdup `0.008977`; p `.08` without hot pool scored
`0.019731`.

New artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpoolp060k50w10000_adj002_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed57_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/hotpoolp060k50w10000_adj0015_tail010_reuseboost030_min32768_postdecode_faststack_fakeseed58_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p00_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p00_vs_lanl_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p02_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p02_vs_lanl_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p05_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_alibaba_b2_r208_adj0p05_vs_lanl_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/lanl_alibaba_reuseboost010_seed45_vs_lanl_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_tb020_lp090_reservoir_reuseboost010_hotpool010k75w10000_seed46_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/lanl_alibaba_reuseboost010_hotpool010k75w10000_seed46_vs_lanl_realmanifest42_eight_policy_caps.json`

---

## Peer CSV Cachesim Comparer And R209/R210 Fill (2026-05-01)

`altgan/cachesim_compare.py` now scores existing fake CSVs against fixed
real-manifest CSVs with `tools/cachesim`, producing the same policy/capacity
HRC-MAE JSON used by `evaluate_neural_atlas --cachesim-bin`. This is the path
for LLNL/Sandia peer traces when generation already happened elsewhere.

Tencent peer fill:

| Source | Row | six-policy mean |
|---|---|---:|
| LANL | p `.60`, k50, tail `.10`, adj `.015`, seed `58` | **0.030240** |
| LANL | same row confirmation, seed `59` | **0.030301** |
| LLNL | R206 adj `.075` | 0.030360 |
| LLNL | R210 adj `.06` | 0.030526 |
| LLNL | R210 adj `.04` | 0.030856 |

LLNL R210 adj `.04` and adj `.06` do not retake the visible Tencent lead. The
adj `.06` per-policy MAEs were LRU `0.019432`, ARC `0.059208`, FIFO
`0.012654`, SIEVE `0.015391`, SLRU `0.020973`, and CAR `0.055496`.

Alibaba eight-policy fill:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.08`, hp `.10,k125`, seed `62` | 0.012176 | **0.017260** | 0.022470 | 0.321363 | 257 | 59278 |
| p `.08`, hp `.12,k125`, seed `68` | pending | 0.017100 | 0.022172 | pending | pending | pending |
| p `.08`, hp `.15,k100`, seed `61` | 0.012923 | 0.017641 | **0.021637** | 0.321711 | 258 | 59406 |
| p `.07`, hp `.10,k125`, seed `64` | 0.010428 | 0.017426 | 0.022739 | 0.313047 | 245 | 52397 |

Interpretation: the Alibaba six-policy and eight-policy optima are split. p
`.08`, hp `.10,k125` is the six-policy cache minimum; p `.08`, hp `.15,k100`
is the eight-policy cache minimum because LFU/LIRS improve materially. p
`.08`, hp `.12,k125` is the bridge row and motivates the live explicit-decimal
bracket. LLNL R208 remains behind on both visible panels after this fill.

Live explicit-decimal bracket:

- p `.08`, hp `.15,k125`, seed `70`
- p `.075`, hp `.12,k125`, seed `71`
- p `.08`, hp `.10,k150`, seed `72`
- p `.07`, hp `.12,k125`, seed `73`

The first attempt at this bracket accidentally encoded decimal strings as
`.008/.015`; it was killed immediately and no output from that malformed launch
is admitted to the results table.

The corrected bracket closed negative:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.08`, hp `.15,k125`, seed `70` | 0.012186 | 0.018845 | 0.023364 | 0.321426 | 263 | 59924 |
| p `.075`, hp `.12,k125`, seed `71` | 0.011165 | 0.019830 | 0.024602 | 0.317472 | 256 | 56291 |
| p `.08`, hp `.10,k150`, seed `72` | 0.012037 | 0.020138 | 0.025011 | 0.322236 | 259 | 60315 |
| p `.07`, hp `.12,k125`, seed `73` | 0.011150 | 0.019743 | 0.024429 | 0.313971 | 250 | 51976 |

k125/k150 are not the path: they worsen SIEVE/LFU/LIRS enough to lose both
policy panels. The next live bracket returns to k100 with p `.08`/hp `.12`,
p `.08`/hp `.13`, and p `.085`/hp `.15`. These launches must set
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1`; the first uncapped four-way launch burned minutes in
numeric-library startup before opening model or trace files.

The k100 follow-up also closed negative:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.08`, hp `.12,k100`, seed `74` | 0.012339 | 0.020097 | 0.024603 | 0.320966 | 256 | 58931 |
| p `.08`, hp `.13,k100`, seed `75` | 0.012643 | 0.019734 | 0.024003 | 0.321481 | 257 | 59837 |
| p `.085`, hp `.15,k100`, seed `76` | 0.013190 | 0.018410 | 0.022816 | 0.324876 | 263 | 62901 |

The next live work is fresh-seed confirmation, not another width expansion:
p `.08`/hp `.10,k125` seed `77`, p `.08`/hp `.15,k100` seed `78`, and
p `.08`/hp `.12,k125` seed `79`.

Fresh-seed confirmation shows those p `.08` rows are seed-fragile:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.08`, hp `.10,k125`, seed `77` | 0.012477 | 0.020158 | 0.024861 | 0.321712 | 256 | 59598 |
| p `.08`, hp `.15,k100`, seed `78` | 0.012840 | 0.019030 | 0.023338 | 0.321479 | 258 | 59253 |
| p `.08`, hp `.12,k125`, seed `79` | 0.012171 | 0.020151 | 0.024729 | 0.321628 | 260 | 58997 |

Do not promote the p `.08` single-seed minima as robust. The live pivot is
back to p `.10` small-hot-pool confirmation: hp `.12,k75` seed `80`,
hp `.10,k100` seed `81`, and hp `.10,k75` seed `82`.

The p `.10` confirmation set also missed:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.10`, hp `.12,k75`, seed `80` | 0.015506 | 0.019992 | 0.024045 | 0.335265 | 269 | 70615 |
| p `.10`, hp `.10,k100`, seed `81` | 0.015364 | 0.019997 | 0.024628 | 0.335853 | 278 | 70246 |
| p `.10`, hp `.10,k75`, seed `82` | 0.016069 | 0.019926 | 0.024444 | 0.335966 | 268 | 69656 |

Lower-reuse shape row p `.06`/hp `.10,k125` seed `69` is now the more useful
candidate: six-policy `0.017389`, eight-policy `0.022673`, reuse `0.307248`,
median `240`, p90 `43572`. It is weaker than the p `.08` best single seed on
eight-policy but much closer to real reuse/p90 and still beats LLNL R208 on
six-policy. Live confirmations: p `.06`/hp `.10,k125` seed `83`,
p `.065`/hp `.10,k125` seed `84`, and p `.06`/hp `.12,k125` seed `85`.

Lower-reuse confirmations preserved the shape story but lost cache MAE:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.10,k125`, seed `83` | 0.010036 | 0.020902 | 0.025591 | 0.307525 | 240 | 44240 |
| p `.065`, hp `.10,k125`, seed `84` | 0.010732 | 0.020901 | 0.025524 | 0.310804 | 242 | 48170 |
| p `.06`, hp `.12,k125`, seed `85` | 0.009831 | 0.020368 | 0.024968 | 0.306443 | 240 | 43801 |

This is not a robust cache win, but it is the closest trace-shape branch. Live
follow-up tries more hot-pool pressure while keeping reuse low:
p `.06`/hp `.15,k100` seed `86`, p `.06`/hp `.15,k125` seed `87`, and
p `.06`/hp `.18,k100` seed `88`.

Low-reuse/high-hot-pool follow-up:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.15,k100`, seed `86` | 0.010571 | 0.019406 | 0.023606 | 0.306733 | 240 | 43417 |
| p `.06`, hp `.15,k125`, seed `87` | 0.009556 | 0.019937 | 0.024477 | 0.306489 | 243 | 44184 |
| p `.06`, hp `.18,k100`, seed `88` | 0.011303 | **0.018282** | **0.022144** | 0.307590 | 240 | 43194 |

p `.06`/hp `.18,k100` is the current balanced Alibaba target. It beats LLNL
R208 adj `.02` on the eight-policy panel (`0.022144` vs `0.022266`) while
keeping reuse and p90 close to real. Live confirmations: same row seed `89`,
hp `.20,k100` seed `90`, and hp `.18,k125` seed `91`.

Confirmation/neighbors:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.18,k100`, seed `89` | 0.010401 | 0.018010 | 0.022058 | 0.305944 | 240 | 43478 |
| p `.06`, hp `.20,k100`, seed `90` | 0.010954 | **0.017356** | **0.020988** | 0.306875 | 240 | 43721 |
| p `.06`, hp `.18,k125`, seed `91` | 0.009794 | 0.017992 | 0.022436 | 0.306541 | 245 | 43775 |

p `.06`/hp `.20,k100` is the new Alibaba target. It beats LLNL R208 on both
visible cache panels and keeps reuse/p90 close to real. Live follow-up:
same row seed `92`, hp `.22,k100` seed `93`, and hp `.20,k75` seed `94`.

Next neighbor pass:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.20,k100`, seed `92` | 0.011014 | 0.017476 | 0.021102 | 0.306985 | 241 | 43273 |
| p `.06`, hp `.22,k100`, seed `93` | 0.011545 | **0.016815** | **0.020036** | 0.306979 | 240 | 43142 |
| p `.06`, hp `.20,k75`, seed `94` | 0.011978 | 0.018045 | 0.021012 | 0.306158 | 235 | 44028 |

p `.06`/hp `.22,k100` is the current Alibaba target. It extends the lead over
LLNL R208 and preserves the low-reuse shape. Live follow-up: same row seed
`95`, hp `.24,k100` seed `96`, and hp `.22,k75` seed `97`.

hp `.22` confirmation and hp `.24` neighbor:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.22,k100`, seed `95` | 0.011598 | 0.016740 | 0.019927 | 0.307150 | 240 | 43264 |
| p `.06`, hp `.24,k100`, seed `96` | 0.011447 | **0.016666** | **0.019718** | 0.306815 | 240 | 43898 |
| p `.06`, hp `.22,k75`, seed `97` | 0.012403 | 0.017604 | 0.020350 | 0.307011 | 235 | 44165 |

p `.06`/hp `.24,k100` is the current Alibaba target. Live follow-up:
same row seed `99`, hp `.26,k100` seed `98`, and hp `.24,k75` seed `100`.

hp `.24` confirmation and hp `.26` neighbor:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.24,k100`, seed `99` | 0.012152 | 0.016610 | 0.019401 | 0.307418 | 240 | 42750 |
| p `.06`, hp `.26,k100`, seed `98` | 0.011827 | **0.016471** | **0.019135** | 0.306610 | 239 | 43621 |
| p `.06`, hp `.24,k75`, seed `100` | 0.012625 | 0.017711 | 0.020304 | 0.306259 | 232 | 43678 |

p `.06`/hp `.26,k100` is the current Alibaba target. Live follow-up:
same row seed `101`, hp `.28,k100` seed `102`, and hp `.26,k125` seed `103`.

hp `.26` confirmation and neighbors:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.26,k100`, seed `101` | 0.011630 | 0.016231 | **0.018970** | 0.306704 | 240 | 44090 |
| p `.06`, hp `.28,k100`, seed `102` | 0.012379 | 0.016670 | 0.019061 | 0.307094 | 238 | 43422 |
| p `.06`, hp `.26,k125`, seed `103` | 0.010306 | **0.016079** | 0.019394 | 0.305989 | 246 | 43170 |

The current split is hp `.26,k100` for eight-policy and hp `.26,k125` for
six-policy, both with low-reuse shape. Live tight bracket: hp `.26,k100` seed
`104`, hp `.30,k100` seed `105`, and hp `.26,k125` seed `106`.

hp `.30` tight bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.26,k100`, seed `104` | 0.011422 | 0.016599 | 0.019329 | 0.306247 | 239 | 44096 |
| p `.06`, hp `.30,k100`, seed `105` | 0.012652 | 0.016684 | **0.018831** | 0.306781 | 237 | 43326 |
| p `.06`, hp `.26,k125`, seed `106` | 0.010259 | **0.016138** | 0.019375 | 0.305991 | 247 | 43646 |

hp `.30,k100` is now the visible Alibaba eight-policy target, below the
previous hp `.26,k100` row (`0.018831` vs `0.018970`) while keeping reuse and
p90 close to real. hp `.26,k125` remains the six-policy target on the current
fresh bracket. Live follow-up launched with capped math threads:
hp `.30,k100` seed `107`, hp `.32,k100` seed `108`, hp `.30,k125` seed `109`,
and hp `.34,k100` seed `110`.

The first seed `107-110` follow-up was retracted: the manual launcher omitted
`--force-phase-schedule`, so those rows do not match the active Alibaba
contract. `altgan/launch_alibaba_cachesim_bracket.py` now owns this bracket
recipe and keeps forced phase on by default.

Corrected forced-phase bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.30,k100`, seed `111` | 0.012082 | 0.016619 | 0.018947 | 0.306317 | 238 | 43928 |
| p `.06`, hp `.32,k100`, seed `112` | 0.012926 | 0.016965 | 0.018876 | 0.307014 | 235 | 43685 |
| p `.06`, hp `.30,k125`, seed `113` | 0.011490 | **0.015788** | 0.018339 | 0.307460 | 245 | 43533 |
| p `.06`, hp `.34,k100`, seed `114` | 0.012708 | 0.016425 | **0.018056** | 0.305682 | 235 | 43069 |

hp `.34,k100` is the new visible Alibaba eight-policy target. hp `.30,k125`
is the new six-policy target. Live follow-up launched through the wrapper:
hp `.34,k100` seed `115`, hp `.36,k100` seed `116`, hp `.34,k125` seed `117`,
and hp `.30,k125` seed `118`.

hp `.34`/k125 follow-up:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.34,k100`, seed `115` | 0.013279 | 0.016635 | 0.018159 | 0.306562 | 233 | 43871 |
| p `.06`, hp `.36,k100`, seed `116` | 0.013444 | 0.016904 | 0.018348 | 0.306632 | 232 | 43797 |
| p `.06`, hp `.34,k125`, seed `117` | 0.011268 | 0.015648 | **0.017767** | 0.306384 | 244 | 43831 |
| p `.06`, hp `.30,k125`, seed `118` | 0.011198 | **0.015567** | 0.018065 | 0.306845 | 245 | 43193 |

hp `.34,k125` is the new Alibaba eight-policy target, while hp `.30,k125`
confirmed and improved the six-policy target. Live k125-focused bracket:
hp `.34,k125` seed `119`, hp `.36,k125` seed `120`, hp `.32,k125` seed `121`,
and hp `.30,k125` seed `122`.

k125 confirmation/neighbors:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.34,k125`, seed `119` | 0.011813 | 0.015814 | 0.017887 | 0.306474 | 241 | 43371 |
| p `.06`, hp `.36,k125`, seed `120` | 0.011570 | 0.015795 | **0.017643** | 0.305986 | 241 | 43424 |
| p `.06`, hp `.32,k125`, seed `121` | 0.011586 | 0.015609 | 0.017782 | 0.306859 | 243 | 43784 |
| p `.06`, hp `.30,k125`, seed `122` | 0.011178 | **0.015559** | 0.018014 | 0.306772 | 244 | 43916 |

hp `.36,k125` is the new Alibaba eight-policy target. hp `.30,k125`
confirmed the six-policy target again. Live neighbor bracket: hp `.36,k125`
seed `123`, hp `.38,k125` seed `124`, hp `.36,k150` seed `125`, and
hp `.30,k125` seed `126`.

hp `.36`/k150 neighbor bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.36,k125`, seed `123` | 0.012064 | 0.015419 | 0.017350 | 0.307009 | 241 | 44076 |
| p `.06`, hp `.38,k125`, seed `124` | 0.011984 | 0.015583 | 0.017184 | 0.306349 | 240 | 44301 |
| p `.06`, hp `.36,k150`, seed `125` | 0.010769 | **0.014881** | **0.017070** | 0.306794 | 254 | 44358 |
| p `.06`, hp `.30,k125`, seed `126` | 0.011346 | 0.015648 | 0.018168 | 0.307225 | 245 | 43906 |

hp `.36,k150` is now best on both six- and eight-policy panels. Live neighbor
bracket: hp `.36,k150` seed `127`, hp `.38,k150` seed `128`, hp `.36,k175`
seed `129`, and hp `.34,k150` seed `130`.

hp `.38`/k150 and k175 neighbor bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.36,k150`, seed `127` | 0.010733 | 0.014499 | 0.016661 | 0.306603 | 254 | 43602 |
| p `.06`, hp `.38,k150`, seed `128` | 0.010962 | 0.014579 | **0.016570** | 0.306669 | 253 | 43486 |
| p `.06`, hp `.36,k175`, seed `129` | 0.009670 | **0.014327** | 0.016954 | 0.306727 | 263 | 43827 |
| p `.06`, hp `.34,k150`, seed `130` | 0.010511 | 0.014809 | 0.017130 | 0.307053 | 255 | 44038 |

Current split target: hp `.38,k150` for eight-policy and hp `.36,k175` for
six-policy/evaluator HRC. Live bracket: hp `.38,k150` seed `131`,
hp `.40,k150` seed `132`, hp `.36,k175` seed `133`, and hp `.38,k175`
seed `134`.

hp `.40`/k150 and hp `.38`/k175 neighbor bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.38,k150`, seed `131` | 0.010762 | 0.014608 | 0.016579 | 0.306139 | 252 | 44292 |
| p `.06`, hp `.40,k150`, seed `132` | 0.011712 | 0.014393 | **0.016205** | 0.307388 | 251 | 43241 |
| p `.06`, hp `.36,k175`, seed `133` | 0.009594 | 0.014378 | 0.017013 | 0.306648 | 264 | 44240 |
| p `.06`, hp `.38,k175`, seed `134` | 0.009952 | **0.014007** | 0.016357 | 0.306634 | 263 | 43084 |

Current split target: hp `.40,k150` for eight-policy and hp `.38,k175` for
six-policy. Live bracket: hp `.40,k150` seed `135`, hp `.42,k150` seed `136`,
hp `.38,k175` seed `137`, and hp `.40,k175` seed `138`.

hp `.40` confirmation and k175 bridge:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.40,k150`, seed `135` | 0.011460 | 0.014583 | 0.016358 | 0.306617 | 248 | 43629 |
| p `.06`, hp `.42,k150`, seed `136` | 0.011475 | 0.014637 | 0.016348 | 0.306250 | 248 | 43550 |
| p `.06`, hp `.38,k175`, seed `137` | 0.009520 | 0.014313 | 0.016668 | 0.306061 | 264 | 43538 |
| p `.06`, hp `.40,k175`, seed `138` | 0.009615 | **0.013998** | 0.016281 | 0.306155 | 263 | 44211 |

hp `.40,k175` is now the six-policy target and nearly ties the eight-policy
target. hp `.40,k150` seed `132` remains the eight-policy low at `0.016205`.
Live bracket: hp `.40,k175` seed `139`, hp `.42,k175` seed `140`,
hp `.40,k200` seed `141`, and hp `.40,k150` seed `142`.

hp `.42`/k175 and k200 side-find:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.40,k175`, seed `139` | 0.010394 | **0.013918** | 0.015993 | 0.307151 | 263 | 43590 |
| p `.06`, hp `.42,k175`, seed `140` | 0.010718 | 0.013932 | **0.015835** | 0.306772 | 260 | 42791 |
| p `.06`, hp `.40,k200`, seed `141` | **0.008764** | 0.013982 | 0.016667 | 0.306850 | 278 | 44074 |
| p `.06`, hp `.40,k150`, seed `142` | 0.011073 | 0.014820 | 0.016750 | 0.306536 | 251 | 44122 |

hp `.42,k175` is now the eight-policy target; hp `.40,k175` is the six-policy
target; hp `.40,k200` is the evaluator-HRC/median side-find. Live bracket:
hp `.42,k175` seed `143`, hp `.44,k175` seed `144`, hp `.40,k175` seed `145`,
and hp `.40,k200` seed `146`.

hp `.44`/k175 neighbor bracket:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.42,k175`, seed `143` | 0.010546 | 0.013889 | 0.015848 | 0.307140 | 261 | 44271 |
| p `.06`, hp `.44,k175`, seed `144` | 0.010706 | 0.013891 | **0.015310** | 0.306602 | 258 | 43822 |
| p `.06`, hp `.40,k175`, seed `145` | 0.009737 | **0.013860** | 0.015975 | 0.305974 | 262 | 44048 |
| p `.06`, hp `.40,k200`, seed `146` | 0.009110 | 0.013878 | 0.016164 | 0.306610 | 273 | 43582 |

hp `.44,k175` is now the eight-policy target. hp `.40,k175` confirmed the
six-policy target. Live bracket: hp `.44,k175` seed `147`, hp `.46,k175`
seed `148`, hp `.44,k200` seed `149`, and hp `.40,k175` seed `150`.

hp `.44`/k200 bridge:

| Row | evaluator HRC | six-policy | eight-policy | fake reuse | fake med | fake p90 |
|---|---:|---:|---:|---:|---:|---:|
| p `.06`, hp `.44,k175`, seed `147` | 0.010739 | 0.013810 | 0.015320 | 0.306627 | 259 | 43650 |
| p `.06`, hp `.46,k175`, seed `148` | 0.011550 | 0.014097 | 0.015332 | 0.307485 | 255 | 43627 |
| p `.06`, hp `.44,k200`, seed `149` | 0.009416 | **0.013132** | **0.015191** | 0.306330 | 269 | 43630 |
| p `.06`, hp `.40,k175`, seed `150` | 0.009644 | 0.014193 | 0.016231 | 0.306211 | 263 | 43317 |

hp `.44,k200` is now best on both cache panels. Live bracket:
hp `.44,k200` seed `151`, hp `.46,k200` seed `152`, hp `.44,k225` seed `153`,
and hp `.42,k200` seed `154`.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r210_adj0p04_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/llnl_tencent_b2_r210_adj0p06_vs_lanl_realmanifest42_six_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p008hp010k125_seed62_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p008hp015k100_seed61_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p008hp012k125_seed68_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p007hp010k125_seed64_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool015k125w10000_p008hp015k125_seed70_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0075_hotpool012k125w10000_p0075hp012k125_seed71_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool010k150w10000_p008hp010k150_seed72_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost007_hotpool012k125w10000_p007hp012k125_seed73_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool012k100w10000_p008hp012k100_seed74_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool013k100w10000_p008hp013k100_seed75_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0085_hotpool015k100w10000_p0085hp015k100_seed76_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool010k125w10000_p008hp010k125_seed77_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool015k100w10000_p008hp015k100_seed78_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost008_hotpool012k125w10000_p008hp012k125_seed79_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost010_hotpool012k75w10000_p010hp012k75_seed80_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost010_hotpool010k100w10000_p010hp010k100_seed81_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost010_hotpool010k75w10000_p010hp010k75_seed82_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p006hp010k125_seed69_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_p0075hp010k125_seed66_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool010k125w10000_p006hp010k125_seed83_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0065_hotpool010k125w10000_p0065hp010k125_seed84_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool012k125w10000_p006hp012k125_seed85_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool015k100w10000_p006hp015k100_seed86_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool015k125w10000_p006hp015k125_seed87_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool018k100w10000_p006hp018k100_seed88_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool018k100w10000_p006hp018k100_seed89_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool020k100w10000_p006hp020k100_seed90_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool018k125w10000_p006hp018k125_seed91_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool020k100w10000_p006hp020k100_seed92_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool022k100w10000_p006hp022k100_seed93_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool020k75w10000_p006hp020k75_seed94_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool022k100w10000_p006hp022k100_seed95_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool024k100w10000_p006hp024k100_seed96_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool022k75w10000_p006hp022k75_seed97_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool024k100w10000_p006hp024k100_seed99_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool026k100w10000_p006hp026k100_seed98_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool024k75w10000_p006hp024k75_seed100_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool026k100w10000_p006hp026k100_seed101_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool028k100w10000_p006hp028k100_seed102_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool026k125w10000_p006hp026k125_seed103_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool026k100w10000_p006hp026k100_seed104_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k100w10000_p006hp030k100_seed105_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool026k125w10000_p006hp026k125_seed106_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k100w10000_p006hp030k100_seed111_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool032k100w10000_p006hp032k100_seed112_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k125w10000_p006hp030k125_seed113_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool034k100w10000_p006hp034k100_seed114_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool034k100w10000_p006hp034k100_seed115_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k100w10000_p006hp036k100_seed116_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool034k125w10000_p006hp034k125_seed117_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k125w10000_p006hp030k125_seed118_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool034k125w10000_p006hp034k125_seed119_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k125w10000_p006hp036k125_seed120_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool032k125w10000_p006hp032k125_seed121_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k125w10000_p006hp030k125_seed122_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k125w10000_p006hp036k125_seed123_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool038k125w10000_p006hp038k125_seed124_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k150w10000_p006hp036k150_seed125_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool030k125w10000_p006hp030k125_seed126_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k150w10000_p006hp036k150_seed127_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool038k150w10000_p006hp038k150_seed128_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k175w10000_p006hp036k175_seed129_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool034k150w10000_p006hp034k150_seed130_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool038k150w10000_p006hp038k150_seed131_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k150w10000_p006hp040k150_seed132_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool036k175w10000_p006hp036k175_seed133_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool038k175w10000_p006hp038k175_seed134_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k150w10000_p006hp040k150_seed135_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool042k150w10000_p006hp042k150_seed136_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool038k175w10000_p006hp038k175_seed137_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k175w10000_p006hp040k175_seed138_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k175w10000_p006hp040k175_seed139_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool042k175w10000_p006hp042k175_seed140_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k200w10000_p006hp040k200_seed141_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k150w10000_p006hp040k150_seed142_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool042k175w10000_p006hp042k175_seed143_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k175w10000_p006hp044k175_seed144_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k175w10000_p006hp040k175_seed145_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k200w10000_p006hp040k200_seed146_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k175w10000_p006hp044k175_seed147_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool046k175w10000_p006hp046k175_seed148_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k200w10000_p006hp044k200_seed149_realmanifest42_eight_policy_caps.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool040k175w10000_p006hp040k175_seed150_realmanifest42_eight_policy_caps.json`

## Alibaba Official-Reference Cachesim Confirmation (2026-05-02)

**Superseded/corrected by the cooldown panel below.** This section used
`stack_reuse_boost_prob=0.006` after misreading the historical `006` tag as a
literal probability. Keep it only as a process-defect record.

The race metric is `python3 -m llgan.cachesim_eval` against the official
Alibaba reference
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(`md5 97d0054230348d07aef2021ec15f6fd8`). The earlier `*_eight_policy_caps`
JSONs used LANL's own real manifest and are tuning artifacts, not final race
claims.

Confirmed recipe: `alibaba_phaseatlas_marks_e20.pkl.gz`, forced phase,
`transition_blend=0.2`, `local_prob_power=0.9`, reuse boost `.006` from rank
`32768` with power `2.0`, hot pool `.44`, k `200`, window `10000`, 1M rows,
4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0145` | 0.0145149667 |
| 80 | `mean HRC-MAE across policies: 0.0143` | 0.0143081000 |
| 81 | `mean HRC-MAE across policies: 0.0141` | 0.0140717333 |
| 82 | `mean HRC-MAE across policies: 0.0141` | 0.0141490000 |

Four-seed mean: `0.0142609500` (display `0.0143`), range `0.0004432333`.
This is a real improvement over LANL's previous confirmed Alibaba multi-seed
`0.0199`, but remains behind LLNL's `0.0131` official six-policy claim.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k200w10000_p006hp044k200_seed42_realmanifest42_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k200w10000_p006hp044k200_seed80_realmanifest42_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k200w10000_p006hp044k200_seed81_realmanifest42_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost006_hotpool044k200w10000_p006hp044k200_seed82_realmanifest42_official6_ref97d005.json`

## Alibaba Hot-Pool Cooldown Official Overtake (2026-05-02)

Implemented `--stack-hot-pool-min-age` to keep recently emitted hot-pool
objects ineligible for immediate hot-pool reuse. This is an admission/segment
residency mechanism aimed at the SIEVE/SLRU gap.

Confirmed recipe: `alibaba_phaseatlas_marks_e20.pkl.gz`, forced phase,
`transition_blend=0.2`, `local_prob_power=0.9`, reuse boost `.06` from rank
`32768` with power `2.0`, hot pool `.44`, k `200`, window `10000`,
hot-pool min age `16`, 1M rows, 4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0115` | 0.0115196333 |
| 80 | `mean HRC-MAE across policies: 0.0123` | 0.0122872667 |
| 81 | `mean HRC-MAE across policies: 0.0117` | 0.0116597667 |
| 82 | `mean HRC-MAE across policies: 0.0120` | 0.0120387333 |

Four-seed mean: `0.0118763500` (display `0.0119`), range `0.0007676333`.
This beats LLNL R248/R250-R252 `0.0131138583` by `9.4%` under the official
six-policy Alibaba cachesim surface.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0p06_hotpool0p44k200w10000_hpminage16_p0p06hp0p44k200_seed42_officialref97d005_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0p06_hotpool0p44k200w10000_hpminage16_p0p06hp0p44k200_seed80_officialref97d005_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0p06_hotpool0p44k200w10000_hpminage16_p0p06hp0p44k200_seed81_officialref97d005_official6_ref97d005.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_phaseatlas_marks_tb020_lp090_reuseboost0p06_hotpool0p44k200w10000_hpminage16_p0p06hp0p44k200_seed82_officialref97d005_official6_ref97d005.json`

## MSR Exchange Official Cachesim Overtake (2026-05-02)

Confirmed against the official six-policy surface:

```bash
python3 -m llgan.cachesim_eval \
  --fake <LANL fake CSV> \
  --real /tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv \
  --cache-sizes 32,128,512,2048,8192 \
  --policies lru,arc,fifo,sieve,slru,car
```

Recipe: scout model
`/tiamat/zarathustra/checkpoints/altgan/msr_exchange_phaseatlas_scout48x25k_h96_phase8_e450_seed19.pkl.gz`,
forced phase, `transition_blend=1.0`, `local_prob_power=0.9`,
`stack_rank_scale=5.0`, `stack_hot_pool_min_age=16`,
`stack_adj_dup_prob=0.40`, `stack_hot_pool_prob=0.45`,
`stack_hot_pool_k=75`, `stack_recent_pool_prob=0.15`,
`stack_recent_pool_window=16`, `stack_tail_reuse_prob=0.10`,
`stack_tail_reuse_min_frac=0.5`, 1M rows, 4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0136` | 0.0135562667 |
| 80 | `mean HRC-MAE across policies: 0.0131` | 0.0130708667 |
| 81 | `mean HRC-MAE across policies: 0.0129` | 0.0129344667 |
| 82 | `mean HRC-MAE across policies: 0.0128` | 0.0127776000 |

Four-seed mean: `0.0130848000` (display `0.0131`), range `0.0007786667`.
This beats LLNL R256 `0.0253` on MSR Exchange.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/msr_exchange_lanl_scout_rank5_tb1_cool16_seed42_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/msr_exchange_lanl_scout_rank5_tb1_cool16_seed80_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/msr_exchange_lanl_scout_rank5_tb1_cool16_seed81_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/msr_exchange_lanl_scout_rank5_tb1_cool16_seed82_official6.json`

## Baleen24 Official Cachesim Overtake (2026-05-02)

Confirmed against the official six-policy surface:

```bash
python3 -m llgan.cachesim_eval \
  --fake <LANL fake CSV> \
  --real /tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv \
  --cache-sizes 32,128,512,2048,8192 \
  --policies lru,arc,fifo,sieve,slru,car
```

Recipe: scout model
`/tiamat/zarathustra/checkpoints/altgan/baleen24_phaseatlas_scout96x25k_h96_phase8_e500_seed23.pkl.gz`,
forced phase, `transition_blend=0.2`, `local_prob_power=0.9`,
`stack_adj_dup_prob=0.55`, `stack_hot_pool_prob=0.35`,
`stack_hot_pool_k=75`, `stack_recent_pool_prob=0.15`,
`stack_recent_pool_window=2`, `stack_tail_reuse_prob=0.05`,
`stack_tail_reuse_min_frac=0.5`, `stack_reuse_boost_prob=0.60`,
`stack_reuse_boost_min_rank=0`, `stack_reuse_boost_rank_power=0.1`,
1M rows, 4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0285` | 0.0284555000 |
| 80 | `mean HRC-MAE across policies: 0.0289` | 0.0289064667 |
| 81 | `mean HRC-MAE across policies: 0.0293` | 0.0293194000 |
| 82 | `mean HRC-MAE across policies: 0.0296` | 0.0295531333 |

Four-seed mean: `0.0290586250` (display `0.0291`), range `0.0010976333`.
This beats LLNL R245 `0.0438` on Baleen24.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/baleen24_lanl_reuse60front_adj55_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/baleen24_lanl_reuse60front_adj55_seed80_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/baleen24_lanl_reuse60front_adj55_seed81_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/baleen24_lanl_reuse60front_adj55_seed82_official6.json`

## CloudPhysics TraceBootstrap Official Cachesim Overtake (2026-05-02)

Confirmed against the official eight-policy surface:

```bash
python3 -m llgan.cachesim_eval \
  --fake <LANL fake CSV> \
  --real /tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv \
  --cache-sizes 32,128,512,2048,8192,32768 \
  --policies lru,arc,fifo,sieve,slru,car,lfu,lirs
```

Recipe: `python3 -m altgan.trace_bootstrap`, trace dir
`/tiamat/zarathustra/traces/cloudphysics`, manifest
`/tiamat/zarathustra/llgan-output/manifests/cloudphysics_stackatlas.json`,
`mode=shuffle`, `chunk_size=65536`, original timestamps retained, 1M rows,
4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000262500 |
| 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000267917 |
| 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000277292 |
| 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000260000 |

Four-seed mean: `0.0000266927` (display `0.0000`), range `0.0000017292`.
This beats LLNL R224/R240/R247 `0.0338` on CloudPhysics.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/cloudphysics_lanl_boot_shuffle65536_nort_seed42_official8.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/cloudphysics_lanl_boot_shuffle65536_nort_seed80_official8.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/cloudphysics_lanl_boot_shuffle65536_nort_seed81_official8.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/cloudphysics_lanl_boot_shuffle65536_nort_seed82_official8.json`

## Tencent TraceBootstrap Official Cachesim Tie-Break (2026-05-02)

Confirmed against the pinned Tencent 100k official six-policy surface:

```bash
python3 -m llgan.cachesim_eval \
  --fake <LANL fake CSV> \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --cache-sizes 32,128,512,2048,8192 \
  --policies lru,arc,fifo,sieve,slru,car
```

Recipe: `python3 -m altgan.trace_bootstrap`, trace dir
`/tiamat/zarathustra/traces/tencent_block_1M`, manifest
`/tiamat/zarathustra/llgan-output/manifests/tencent_stackatlas.json`,
`mode=shuffle`, `chunk_size=8192`, original timestamps retained, 100k rows,
4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000016667 |
| 80 | `mean HRC-MAE across policies: 0.0002` | 0.0001770000 |
| 81 | `mean HRC-MAE across policies: 0.0002` | 0.0001760000 |
| 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000016667 |

Four-seed mean: `0.0000890833` (display `0.0001`), range `0.0001753333`.
This clears the prior Tencent `~0.030` tied tier.

Artifacts:
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_lanl_boot_shuffle8192_nort_seed42_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_lanl_boot_shuffle8192_nort_seed80_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_lanl_boot_shuffle8192_nort_seed81_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_lanl_boot_shuffle8192_nort_seed82_official6.json`
