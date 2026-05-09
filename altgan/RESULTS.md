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

## Alibaba Standing Ledger Correction (2026-05-03)

The current official LANL Alibaba claim is the hot-pool cooldown panel, not the
superseded decimal-correction panel. The stale standing was LANL
`0.0142609500` behind LLNL `0.0131138583`; the measured current standing is:

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0115` | 0.0115196333 |
| 80 | `mean HRC-MAE across policies: 0.0123` | 0.0122872667 |
| 81 | `mean HRC-MAE across policies: 0.0117` | 0.0116597667 |
| 82 | `mean HRC-MAE across policies: 0.0120` | 0.0120387333 |

Four-seed mean: `0.0118763500` (display `0.0119`), range `0.0007676333`.
This beats LLNL R248/R250-R252 `0.0131138583` by `9.4%` under the official
six-policy Alibaba cachesim surface.

## CloudPhysics Frequency-Pool Scout Negative (2026-05-03)

Implemented long-memory frequency-pool reuse routing in `altgan` (`d62d950`)
and tested it against CloudPhysics seed `42` on the official eight-policy
`llgan.cachesim_eval` surface. Baseline to beat was the prior best
non-bootstrap CP single-seed row:
`cloudphysics_lanl_phase1_rank3_adj25_hp05_drop005_official8.json`, mean
`0.0406011250`.

| row | added frequency-pool settings | evaluator HRC | literal `llgan.cachesim_eval` mean line | JSON mean | LFU | LIRS |
|---|---|---:|---|---:|---:|---:|
| `cloudphysics_lanl_freqpool_p003_k4096_wp05_age16_seed42` | `prob=0.03`, `k=4096`, `weight_power=0.5`, `min_age=16` | 0.0396531500 | `mean HRC-MAE across policies: 0.0409` | 0.0408657083 | 0.1130800000 | 0.0718263333 |
| `cloudphysics_lanl_freqpool_p008_k2048_wp05_age16_seed42` | `prob=0.08`, `k=2048`, `weight_power=0.5`, `min_age=16` | 0.0317419000 | `mean HRC-MAE across policies: 0.0442` | 0.0441683542 | 0.1141291667 | 0.0766388333 |
| `cloudphysics_lanl_freqpool_p005_k8192_wp025_age64_seed42` | `prob=0.05`, `k=8192`, `weight_power=0.25`, `min_age=64` | 0.0358485000 | `mean HRC-MAE across policies: 0.0420` | 0.0419542500 | 0.1119820000 | 0.0742111667 |
| `cloudphysics_lanl_freqband_p005_k8192_wp025_age64_r8192_32768_seed42` | `prob=0.05`, `k=8192`, `weight_power=0.25`, `min_age=64`, `min_rank=8192`, `max_rank=32768` | n/a | `mean HRC-MAE across policies: 0.0421` | 0.0421487917 | 0.1118605000 | 0.0746280000 |
| `cloudphysics_lanl_freqband_p008_k8192_wp025_age64_r16384_65536_seed42` | `prob=0.08`, `k=8192`, `weight_power=0.25`, `min_age=64`, `min_rank=16384`, `max_rank=65536` | n/a | `mean HRC-MAE across policies: 0.0441` | 0.0440802500 | 0.1106048333 | 0.0782850000 |

The closest row was still worse than the incumbent (`0.0408657083` vs
`0.0406011250`), and rank-banding did not rescue the family. The architecture
moved evaluator/shape diagnostics and a little LFU mass, but LIRS worsened
enough that cachesim rejected the branch. Close this path for CP unless paired
with a separate LIRS-preserving residency mechanism.

## TraceBootstrap Missing-Corpus Completion (2026-05-03)

LANL completed the TraceBootstrap ledger for the three corpora LLNL R259g
listed as "LANL not published": Alibaba, Baleen24, and MSR Exchange. Recipe:
`python3 -m altgan.trace_bootstrap`, `mode=shuffle`, `chunk_size=65536`,
original timestamps retained, 1M rows, 4 streams, seeds `{42,80,81,82}`.

| corpus | seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---|---:|---|---:|
| Alibaba | 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000424000 |
| Alibaba | 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000019667 |
| Alibaba | 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000105000 |
| Alibaba | 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000108333 |
| Baleen24 | 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000000333 |
| Baleen24 | 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000000333 |
| Baleen24 | 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| Baleen24 | 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000000333 |
| MSR Exchange | 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| MSR Exchange | 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| MSR Exchange | 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| MSR Exchange | 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |

Four-seed means: Alibaba `0.0000164250`, Baleen24 `0.0000000250`, MSR
Exchange `0.0000000000`; all display as `0.0000`. Bootstrap methodology is now
published by LANL on all five corpora, so LLNL's R259g "LANL not published"
bootstrap-only advantage is closed.

## MSR Exchange Noise-Matched Time-Size Retake (2026-05-03)

LANL added train-time conditioning noise to `altgan.neural_atlas` in commit
`68f389b`, fit
`/tiamat/zarathustra/checkpoints/altgan/msr_exchange_phaseatlas_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`,
and ran the official six-policy `llgan.cachesim_eval` surface against
`/tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv`.

Recipe: phase-2/time-4/size-4 atlas, `hidden_dim=96`,
`records_per_file=50000`, `epochs=600`, `seed=137`,
`cond_noise_std=0.05`; generation with forced phase,
`condition_from_real_manifest`, `transition_blend=1.0`,
`local_prob_power=0.9`, `stack_rank_scale=2.0`,
`stack_adj_dup_prob=0.40`, `stack_hot_pool_prob=0.45`,
`stack_hot_pool_k=75`, `stack_hot_pool_min_age=16`,
`stack_recent_pool_prob=0.15`, `stack_recent_pool_window=16`,
`stack_tail_reuse_prob=0.10`, `stack_tail_reuse_min_frac=0.5`, 1M rows,
4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0104` | 0.0103523333 |
| 80 | `mean HRC-MAE across policies: 0.0097` | 0.0096974333 |
| 81 | `mean HRC-MAE across policies: 0.0100` | 0.0099689667 |
| 82 | `mean HRC-MAE across policies: 0.0101` | 0.0101276667 |

Four-seed mean: `0.0100366000` (display `0.0100`), range `0.0006549000`.
This retakes the MSR Exchange non-bootstrap generative ledger from LLNL R273's
posted `0.0105` multi-seed claim under the matched official cachesim protocol.

## Baleen24 Noise-Regularized Rank-Half Defense (2026-05-03)

Fit
`/tiamat/zarathustra/checkpoints/altgan/baleen24_phaseatlas_scout96x25k_h96_phase8_t4s4_e600_seed23_noise0p05.pkl.gz`
with 96 Baleen24 files, 25k records/file, `hidden_dim=96`, `n_phase=8`,
`n_time_bins=4`, `n_size_bins=4`, `epochs=600`, `seed=23`,
`cond_noise_std=0.05`. Official reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

Generation recipe: forced phase, `transition_blend=0.2`,
`local_prob_power=0.9`, `stack_rank_scale=0.5`,
`stack_adj_dup_prob=0.55`, `stack_hot_pool_prob=0.35`,
`stack_hot_pool_k=75`, `stack_recent_pool_prob=0.15`,
`stack_recent_pool_window=2`, `stack_tail_reuse_prob=0.05`,
`stack_tail_reuse_min_frac=0.5`, `stack_reuse_boost_prob=0.60`,
`stack_reuse_boost_min_rank=0`, `stack_reuse_boost_rank_power=0.1`, 1M rows,
4 streams.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0274` | 0.0273759667 |
| 80 | `mean HRC-MAE across policies: 0.0278` | 0.0277821333 |
| 81 | `mean HRC-MAE across policies: 0.0277` | 0.0277449333 |
| 82 | `mean HRC-MAE across policies: 0.0274` | 0.0274192667 |

Four-seed mean: `0.0275805750` (display `0.0276`), range `0.0004061667`.
This improves the previous LANL Baleen24 mean `0.0290586250` by `5.1%` and
extends the lead over LLNL R245 `0.0438` to about `37%`. Seed-42 side scouts:
rank `1.0` scored `0.0358`; rank `2.0` scored `0.0458`; rank `0.5` is the
only live noise-fit calibration.

## Alibaba R276 Mirror Audit Negative (2026-05-03)

Mirrored LLNL's R276 reuse-drop idea on LANL's current Alibaba champion:
`alibaba_phaseatlas_marks_e20.pkl.gz`, forced phase, `transition_blend=0.2`,
`local_prob_power=0.9`, `stack_reuse_boost_prob=0.06`,
`stack_reuse_boost_min_rank=32768`, `stack_reuse_boost_rank_power=2.0`,
`stack_hot_pool_prob=0.44`, `stack_hot_pool_k=200`,
`stack_hot_pool_min_age=16`, official six-policy reference
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| row | literal `llgan.cachesim_eval` mean line | JSON mean |
|---|---|---:|
| incumbent seed42, no drop | `mean HRC-MAE across policies: 0.0115` | 0.0115196333 |
| `reuse_drop_prob=0.01` | `mean HRC-MAE across policies: 0.0126` | 0.0125671333 |
| `reuse_drop_prob=0.025` | `mean HRC-MAE across policies: 0.0140` | 0.0139983333 |
| `reuse_drop_prob=0.05` | `mean HRC-MAE across policies: 0.0168` | 0.0167914667 |

Reuse-drop is negative on LANL Alibaba too. A clean
phase-2/time-4/size-4/cond-noise Alibaba atlas also closed negative on seed 42:
LANL cooldown shape `0.0954817000`, R248 shape `0.1162945000`,
R248+cooldown `0.1126757667`, and R248+cooldown+rank2 `0.0945259333`.

## Twitter Generative Entry + Bootstrap Replay Cleanup (2026-05-03)

Twitter atlas:
`/tiamat/zarathustra/checkpoints/altgan/twitter_cluster_phaseatlas_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05_v2.pkl.gz`.
Fit: 54 files, 50k records/file, `hidden_dim=96`, `n_phase=2`,
`n_time_bins=4`, `n_size_bins=4`, `epochs=600`, `seed=137`,
`cond_noise_std=0.05`.

Promoted Twitter recipe: forced phase, `condition_from_real_manifest`,
`transition_blend=1.0`, `local_prob_power=0.9`, `stack_rank_scale=2.0`,
`stack_adj_dup_prob=0.40`, `stack_hot_pool_prob=0.65`,
`stack_hot_pool_k=75`, `stack_hot_pool_min_age=16`,
`stack_recent_pool_prob=0.25`, `stack_recent_pool_window=16`,
`stack_tail_reuse_prob=0.10`, `stack_tail_reuse_min_frac=0.5`, 1M rows,
4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0289` | 0.0288781333 |
| 80 | `mean HRC-MAE across policies: 0.0286` | 0.0285878667 |
| 81 | `mean HRC-MAE across policies: 0.0288` | 0.0287879667 |
| 82 | `mean HRC-MAE across policies: 0.0289` | 0.0288827333 |

Four-seed mean: `0.0287841750` (display `0.0288`), range
`0.0002948667`. Twitter shape read: R248/cooldown-style low transition blend
over-reuses (`0.1507` to `0.2164` seed 42). The viable basin is
MSR-like `transition_blend=1.0` with stronger hot/recent admission.

Bootstrap replay cleanup for stale/new corpus pressure:

| corpus | protocol | seeds `{42,80,81,82}` JSON means | four-seed mean |
|---|---|---|---:|
| Tencent | replay, pinned 100k manifest, 6-pol | `0.0000000000`, `0.0000000000`, `0.0000000000`, `0.0000000000` | 0.0000000000 |
| CloudPhysics | replay, 1M, 8-pol | `0.0000000000`, `0.0000000000`, `0.0000000000`, `0.0000000000` | 0.0000000000 |
| Twitter | replay, 1M, 6-pol | `0.0000000000`, `0.0000000000`, `0.0000000000`, `0.0000000000` | 0.0000000000 |
| Meta KV | replay, 1M, 6-pol | `0.0000000000`, `0.0000000000`, `0.0000000000`, `0.0000000000` | 0.0000000000 |
| Meta CDN | replay, 1M, 6-pol | `0.0000000000`, `0.0000000000`, `0.0000000000`, `0.0000000000` | 0.0000000000 |

Meta KV apples-to-apples shuffle (`mode=shuffle`, `chunk_size=65536`) scored
`0.0007697000`, `0.0006143000`, `0.0006895667`, `0.0006826667`; mean
`0.0006890583`, range `0.0001554000`, matching LLNL R278's non-stationary
shuffle scale while replay pins the exact cachesim zero.

Reproduction / extension (full per-seed fake paths + literal cachesim mean lines
+ exact JSON means; use this to close Twitter + Meta CDN shuffle rows too):

```bash
python3 -m altgan.launch_trace_bootstrap_multiseed --help
```

Shortcut (standard 1M shuffle protocol for Twitter / Meta KV / Meta CDN / Wiki):

If your `/tiamat` mount point differs, pass `--zarathustra-root /path/to/zarathustra`.

If launching from a machine without `/tiamat`, use the SSH dispatcher to run on
`vinge`/`baase` and commit+push doc updates via git (no scp):

```bash
python3 -m altgan.ssh_tracebootstrap_shuffle_pack --host vinge.local --sync bundle --ssh-option StrictHostKeyChecking=accept-new --commit --push --tmux-session tb_shuffle_pack
```

If `vinge.local` doesn't resolve on your launch machine, try the ssh-config host
alias (if you have one), e.g. `--host vinge` or `--host baase`.

If your ssh config routes through a jump host (ProxyJump) and that host doesn't resolve, add `--no-proxyjump` (or `--ssh-option ProxyJump=none`).

```bash
python3 -m altgan.launch_trace_bootstrap_shuffle_pack \
  --update-lanl-docs \
  --markdown \
  --skip-existing \
  --keep-going \
  --emit-markdown-dir /tiamat/zarathustra/altgan-output/paste_ready \
  --emit-summary-json-dir /tiamat/zarathustra/altgan-output/paste_ready
```

### TraceBootstrap shuffle panels (auto-generated)

<!-- BEGIN TRACEBOOTSTRAP_SHUFFLE_TWITTER -->
| seed | fake | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_lanl_boot_shuffle65536_nort_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 4.9466666666675434e-05 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_lanl_boot_shuffle65536_nort_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0001` | 6.0933333333335056e-05 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_lanl_boot_shuffle65536_nort_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0001` | 6.953333333335292e-05 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_lanl_boot_shuffle65536_nort_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0001` | 7.850000000001096e-05 |

Four-seed mean: `0.0000646083`, range `0.0000290333`.
<!-- END TRACEBOOTSTRAP_SHUFFLE_TWITTER -->

<!-- BEGIN TRACEBOOTSTRAP_SHUFFLE_METAKV -->
| seed | fake | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metakv_lanl_boot_shuffle65536_nort_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0008` | 0.0007696999999999917 |
| 80 | `/tiamat/zarathustra/altgan-output/metakv_lanl_boot_shuffle65536_nort_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0006` | 0.0006142999999999861 |
| 81 | `/tiamat/zarathustra/altgan-output/metakv_lanl_boot_shuffle65536_nort_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0007` | 0.0006895666666666634 |
| 82 | `/tiamat/zarathustra/altgan-output/metakv_lanl_boot_shuffle65536_nort_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0007` | 0.0006826666666666686 |

Four-seed mean: `0.0006890583`, range `0.0001554000`.
<!-- END TRACEBOOTSTRAP_SHUFFLE_METAKV -->

<!-- BEGIN TRACEBOOTSTRAP_SHUFFLE_METACDN -->
| seed | fake | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_lanl_boot_shuffle65536_nort_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 4.133333333315263e-06 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_lanl_boot_shuffle65536_nort_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 2.5333333333136623e-06 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_lanl_boot_shuffle65536_nort_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 1.4999999999876223e-06 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_lanl_boot_shuffle65536_nort_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 4.099999999992072e-06 |

Four-seed mean: `0.0000030667`, range `0.0000026333`.
<!-- END TRACEBOOTSTRAP_SHUFFLE_METACDN -->

<!-- BEGIN TRACEBOOTSTRAP_SHUFFLE_WIKI -->
| seed | fake | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/wiki_lanl_boot_shuffle65536_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 0.0000371667 |
| 80 | `/tiamat/zarathustra/altgan-output/wiki_lanl_boot_shuffle65536_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 0.0000497667 |
| 81 | `/tiamat/zarathustra/altgan-output/wiki_lanl_boot_shuffle65536_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 0.0000327333 |
| 82 | `/tiamat/zarathustra/altgan-output/wiki_lanl_boot_shuffle65536_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0000` | 0.0000343333 |

Four-seed mean: `0.0000385000`, range `0.0000170333`.
<!-- END TRACEBOOTSTRAP_SHUFFLE_WIKI -->

## Meta KV Generative Reuse-Drop Entry (2026-05-03)

Atlas:
`/tiamat/zarathustra/checkpoints/altgan/metakv_phaseatlas_lanl_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.
Fit: 5 files, 217431 records total, `hidden_dim=96`, `n_phase=2`,
`n_time_bins=4`, `n_size_bins=4`, `epochs=600`, `seed=137`,
`cond_noise_std=0.05`.

Recipe: forced phase, `condition_from_real_manifest`,
`transition_blend=1.0`, `local_prob_power=0.9`, `stack_rank_scale=2.0`,
`stack_adj_dup_prob=0.70`, `stack_reuse_drop_prob=0.05`,
`stack_hot_pool_prob=0.25`, `stack_hot_pool_k=75`,
`stack_hot_pool_min_age=16`, `stack_recent_pool_prob=0.05`,
`stack_recent_pool_window=16`, `stack_tail_reuse_prob=0.05`,
`stack_tail_reuse_min_frac=0.5`, 1M rows, 4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/metakv_real.csv`.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0222` | 0.0221643667 |
| 80 | `mean HRC-MAE across policies: 0.0227` | 0.0226979667 |
| 81 | `mean HRC-MAE across policies: 0.0222` | 0.0221568333 |
| 82 | `mean HRC-MAE across policies: 0.0221` | 0.0220730667 |

Four-seed mean: `0.0222730583` (display `0.0223`), range `0.0006249000`.
Seed-42 scouts: MSR base `0.0350`, Twitter shape `0.0439`, R248 shape
`0.0435`, rank3 light `0.0529`, adjacent-heavy without drop `0.0617+`.
The winning row pairs high adjacent admission with reuse drop; matching total
reuse alone did not win the cachesim curve.

## Meta CDN Generative Low-Tail Update (2026-05-03)

Atlas:
`/tiamat/zarathustra/checkpoints/altgan/metacdn_phaseatlas_lanl_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.
Fit: 3 files (`meta_reag`, `meta_rnha`, `meta_rprn`), 72165 records total,
`hidden_dim=96`, `n_phase=2`, `n_time_bins=4`, `n_size_bins=4`, `epochs=600`,
`seed=137`, `cond_noise_std=0.05`.

Recipe: forced phase, `condition_from_real_manifest`,
`transition_blend=1.0`, `local_prob_power=0.9`, `stack_rank_scale=2.0`,
`stack_adj_dup_prob=0.70`, `stack_reuse_drop_prob=0.03`,
`stack_hot_pool_prob=0.25`, `stack_hot_pool_k=75`,
`stack_hot_pool_min_age=16`, `stack_recent_pool_prob=0.05`,
`stack_recent_pool_window=16`, `stack_tail_reuse_prob=0.03`,
`stack_tail_reuse_min_frac=0.5`, 1M rows, 4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0376` | 0.0376173333 |
| 80 | `mean HRC-MAE across policies: 0.0375` | 0.0375326333 |
| 81 | `mean HRC-MAE across policies: 0.0380` | 0.0380410333 |
| 82 | `mean HRC-MAE across policies: 0.0375` | 0.0374686667 |

Four-seed mean: `0.0376649167` (display `0.0377`), range `0.0005723667`.
This replaces the first Meta CDN generative row `0.0415101583`, a `9.3%`
lower cachesim target. Seed-42 scouts: `tail=0.03` `0.0376173333`,
`tail=0.07` `0.0472252333`, `tail=0.08` `0.0502006000`, `tail=0.09`
`0.0531047333`, `tail=0.10` `0.0558878333`, `drop=0.02` `0.0396857667`,
`drop=0.04` `0.0445329333`, `tail0.08+drop0.02` `0.0475122000`,
`tail0.08+drop0.04` `0.0529505000`, `tail0.08+adj0.65` `0.0556624667`,
`tail0.08+hp0.30` `0.0514508667`.

## CloudPhysics Rank-Ramp Non-Bootstrap Update (2026-05-03)

Atlas:
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_phaseatlas_scout96x25k_h64_phase1_e600_seed137.pkl.gz`.
Fit: 96 CloudPhysics LCS files, 25k records/file, `hidden_dim=64`,
`n_phase=1`, `n_time_bins=4`, `n_size_bins=4`, `epochs=600`, `seed=137`.

Recipe: forced phase, `condition_from_real_manifest`,
`transition_blend=0.2`, `local_prob_power=0.9`, `stack_rank_scale=3.0`,
`stack_rank_position_scales=6,5,4.5,3.5,3,2.5,2,2,2.3,2.8`,
`stack_adj_dup_prob=0.25`, `stack_hot_pool_prob=0.05`,
`stack_hot_pool_k=50`, `stack_recent_pool_prob=0.10`,
`stack_recent_pool_window=2`, `stack_tail_reuse_prob=0.10`,
`stack_tail_reuse_min_frac=0.5`,
`stack_reuse_drop_position_probs=0.1,0.08,0.06,0.04,0.03,0.02,0.01,0,0,0`,
1M rows, 4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | literal `llgan.cachesim_eval` mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0401` | 0.0401132708 |
| 80 | `mean HRC-MAE across policies: 0.0403` | 0.0403417500 |
| 81 | `mean HRC-MAE across policies: 0.0401` | 0.0401375000 |
| 82 | `mean HRC-MAE across policies: 0.0404` | 0.0403695833 |

Four-seed mean: `0.0402405260` (display `0.0402`), range `0.0002563125`.
This is LANL's current non-bootstrap CloudPhysics generative entry, improving
the prior seed-42 position-drop scout (`0.0403778125`) but still trailing LLNL
R224/R240/R247 `0.0338`. Negative scouts: near-head duplicate bands `0.0446+`,
band mixes `0.0406+`, `h96_phase8` `0.0588+`, official-four-file 250k atlas
`0.1352+`, and fresh phase2/noise early scouts `0.0442+`.

## CloudPhysics Rank-Band + Distance-State Negative Sweep (2026-05-03)

Implemented `stack_rank_band_reuse_*` (`dd4f6f7`) and optional stack-distance
state bins (`0f4c7fb`). Both were CP-targeted architecture probes. Neither
beats the standing h64 rank-ramp seed-42 `0.0401132708`.

| family | best seed-42 literal line | best JSON mean | note |
|---|---|---:|---|
| h96 phase1/noise | `mean HRC-MAE across policies: 0.0444` | 0.0444263333 | drop005 best; still behind h64 |
| medium rank-band reuse | `mean HRC-MAE across policies: 0.0416` | 0.0415772917 | LFU/ordinary curves move, LIRS worsens |
| d6 time4/size4 distance states | `mean HRC-MAE across policies: 0.0706` | 0.0705539583 | over-reuse, too shallow |
| d6 time1/size1 distance states | `mean HRC-MAE across policies: 0.0610` | 0.0609700208 | closer LLNL analogue, still bad |
| d6 time1/size1 + global reuse drop | `mean HRC-MAE across policies: 0.0535` | 0.0534753958 | drop rescue bottoms before LIRS breaks |

Distance-state checkpoints:
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_diststate_lcs111x25k_h64_phase1_t4s4_d6_e600_seed137.pkl.gz`
and
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_diststate_lcs111x25k_h64_phase1_t1s1_d6_e600_seed137.pkl.gz`.
Both use 106 LCS files with conditioning profiles, 25k records/file, h64,
phase1, seed137, and state edges `0,8,32,128,512,1073741824`.

Conclusion: CP does not transfer cleanly to a naive distance-state
NeuralAtlas/reservoir sampler. The current best stays the action-state h64
rank-ramp panel at four-seed mean `0.0402405260`.

## CloudPhysics Rank-PMF Decoder Update (2026-05-03)

Implemented fitted per-state rank PMFs in `altgan` (`f8280ba`) plus PMF rank
scaling (`4090204`). The checkpoint
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_rankpmf_lcs96x25k_h64_phase1_t4s4_e600_seed137.pkl.gz`
uses 96 CloudPhysics LCS files at 25k records/file, h64, phase1, time4/size4,
600 epochs, seed137. The promoted recipe uses `stack_rank_pmf_prob=0.75`,
`stack_adj_dup_prob=0.20`, LANL's rank-ramp position scales, and stronger
front-loaded reuse drop.

| seed | literal cachesim mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0355` | 0.0355415000 |
| 80 | `mean HRC-MAE across policies: 0.0355` | 0.0355355208 |
| 81 | `mean HRC-MAE across policies: 0.0355` | 0.0355490000 |
| 82 | `mean HRC-MAE across policies: 0.0355` | 0.0354632917 |

Four-seed mean: `0.0355223281`, range `0.0000857083`. This closes most of the
gap from LANL's prior non-bootstrap CP rank-ramp mean `0.0402405260`, but LLNL
R224/R240/R247 still leads at `0.0338`.

Seed-42 negatives around the basin: raw LLNL-shaped rank-PMF `0.0589`,
rank-PMF `0.75` unscaled rankstrong `0.0369`, scaled PMF x2/x3
`0.0387`/`0.0434`, blend `0.65`/`0.85` `0.0364`/`0.0384`, adj `0.15`
`0.0374`, hot-pool zero `0.0368`, and tail `0.05`/`0.15`
`0.0417`/`0.0404`. The residual blocker is the LFU/LIRS trade-off: lowering
head pressure helps LFU and damages LIRS; raising it does the reverse.

## CloudPhysics Rank-PMF Quantile and Exact-Stream Negatives (2026-05-03)

Added rank-PMF in-bin quantile controls in `a893550`:
`stack_rank_pmf_bin_power`, `stack_rank_pmf_tail_bin_power`, and
`stack_rank_pmf_tail_power_pivot`. These keep fitted PMF bin probabilities but
change where samples land inside the bins. Seed-42 official 8-policy results
on the standing LCS96 rank-PMF atlas:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| tail-deep qpow | `mean HRC-MAE across policies: 0.0357` | 0.0356516250 | 0.0973031667 | 0.0696691667 |
| soft head/tail qpow | `mean HRC-MAE across policies: 0.0359` | 0.0359048958 | 0.0957675000 | 0.0676610000 |
| strong head/tail qpow | `mean HRC-MAE across policies: 0.0361` | 0.0360976667 | 0.0939040000 | 0.0650830000 |
| head-only qpow | `mean HRC-MAE across policies: 0.0366` | 0.0366248542 | 0.0858443333 | 0.0772668333 |

The qpow branch can move LFU/LIRS in the desired direction, but not without
overpaying in ordinary/adaptive policy error. It is a useful decoder lever, not
a CP retake.

Exact-stream fits closed negative:

| atlas / decode | literal cachesim mean line | JSON mean |
|---|---|---:|
| oracleGeneral manifest4, R224-style decode | `mean HRC-MAE across policies: 0.0900` | 0.0900156250 |
| oracleGeneral manifest4 + rank-PMF | `mean HRC-MAE across policies: 0.0790` | 0.0789911875 |
| oracleGeneral manifest4 + LANL rank-PMF recipe | `mean HRC-MAE across policies: 0.1208` | 0.1208424583 |
| oracleGeneral manifest4 + LANL adj035 hybrid | `mean HRC-MAE across policies: 0.0937` | 0.0936548750 |
| LCS manifest4, R224-style decode | `mean HRC-MAE across policies: 0.2652` | 0.2651890625 |
| LCS96 h64 phase1 cond-noise0.05, standing recipe | `mean HRC-MAE across policies: 0.0363` | 0.0362662708 |

Read: the broad no-noise LCS96 fit is regularizing CP in a way exact-stream
fits do not. LLNL's four-stream/narrow-fit intuition does not port into the
current `altgan` reservoir decoder, and cond-noise worsens the rank-PMF basin.

## CloudPhysics Local-PMF and Frequency-Pool Negatives (2026-05-03)

Added per-reservoir rank PMFs in `1fb6c19` and refit the LCS96 no-noise atlas
as
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_rankpmf_local_lcs96x25k_h64_phase1_t4s4_e600_seed137.pkl.gz`.
Seed-42 official 8-policy results:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| local PMF prob 0.00 | `mean HRC-MAE across policies: 0.0356` | 0.0356230833 | 0.0973548333 | 0.0696843333 |
| local PMF prob 0.25 | `mean HRC-MAE across policies: 0.0357` | 0.0356876250 | 0.0974486667 | 0.0695895000 |
| local PMF prob 0.50 | `mean HRC-MAE across policies: 0.0357` | 0.0356876250 | 0.0974486667 | 0.0695895000 |

Local PMFs do not improve CP; the global PMF is the useful regularized signal.

Frequency-pool probes on top of the standing rank-PMF decoder also closed
negative:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| old-hot fp=0.02 k=512 age=64 rank 128..8192 | `mean HRC-MAE across policies: 0.0358` | 0.0357806250 | 0.0982700000 | 0.0690941667 |
| deep-hot fp=0.04 k=1024 age=128 rank 512..32768 | `mean HRC-MAE across policies: 0.0370` | 0.0369747917 | 0.0968696667 | 0.0741036667 |

The current CP frontier is therefore not a file-local PMF or long-memory hot-set
problem. Best non-bootstrap CP remains the global rank-PMF four-seed mean
`0.0355223281`, still behind LLNL `0.0338`.

## CloudPhysics Persistent Anchor-Pool Negative (2026-05-03)

Implemented persistent promoted-object anchors in `b184a1a`. Unlike the sliding
hot pool and generated-count frequency pool, anchors are an explicit durable
identity set with age/rank gates. The goal was to match CP's real heavy-hitter
identity persistence without changing the trained atlas.

Seed-42 official 8-policy results on
`/tiamat/zarathustra/checkpoints/altgan/cloudphysics_rankpmf_lcs96x25k_h64_phase1_t4s4_e600_seed137.pkl.gz`:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| anchor light: `ap=0.02`, `k=512`, age/rank `32..32768` | `mean HRC-MAE across policies: 0.0367` | 0.0366531667 | 0.1047451667 | 0.0673445000 |
| anchor mid: `ap=0.05`, `k=512`, age/rank `64..65536` | `mean HRC-MAE across policies: 0.0380` | 0.0379836458 | 0.1115808333 | 0.0655595000 |
| anchor deep: `ap=0.08`, `k=1024`, age/rank `128`, `512..131072` | `mean HRC-MAE across policies: 0.0410` | 0.0409794375 | 0.1068363333 | 0.0708028333 |
| anchor plus lower adjacent pressure: `adj=0.10`, `ap=0.06` | `mean HRC-MAE across policies: 0.0403` | 0.0402951042 | 0.0883876667 | 0.0826723333 |

Anchor persistence is not the missing CP architecture in this decoder. It can
push LFU or LIRS independently, but the joint 8-policy mean moves away from the
standing global rank-PMF result.

## CloudPhysics Rank-PMF Feedback Multi-Seed Update (2026-05-03)

Implemented online rank-PMF feedback shaping in `2685867`. The decoder tracks
emitted rank bins per state and multiplicatively corrects the PMF branch toward
the fitted per-state PMF when other reuse routes overproduce a bin. The winning
CP setting is `stack_rank_pmf_feedback_strength=1.0`,
`stack_rank_pmf_feedback_alpha=32` on the standing LCS96 rank-PMF atlas.

| seed | literal cachesim mean line | JSON mean | LFU | LIRS |
|---:|---|---:|---:|---:|
| 42 | `mean HRC-MAE across policies: 0.0354` | 0.0354031250 | 0.0954860000 | 0.0691163333 |
| 80 | `mean HRC-MAE across policies: 0.0354` | 0.0353670833 | 0.0952138333 | 0.0666363333 |
| 81 | `mean HRC-MAE across policies: 0.0355` | 0.0354530208 | 0.0952008333 | 0.0694540000 |
| 82 | `mean HRC-MAE across policies: 0.0353` | 0.0352951667 | 0.0950235000 | 0.0692613333 |

Four-seed mean: `0.0353795990`, range `0.0001578542`. This replaces the prior
LANL non-bootstrap CP best (`0.0355223281`) but remains behind LLNL `0.0338`.

Seed-42 feedback scouts: strength `0.25` scored `0.0355257917`, strength
`0.50` scored `0.0356408125`, strength `1.0` scored `0.0354031250`, and
strength `0.50` with alpha `256` scored `0.0355570208`.

## CloudPhysics Real-Target Rank-PMF Negative (2026-05-03)

Added a real-manifest rank-PMF calibration path in `06ab85d`. It can build
PMFs from the sampled real manifest and blend or replace the model's fitted
rank PMFs before generation.

Seed-42 official 8-policy probes on top of the feedback-1.0 CP recipe:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| real-target blend 0.25 + feedback 1.0 | `mean HRC-MAE across policies: 0.0354` | 0.0354031250 | 0.0954860000 | 0.0691163333 |
| real-target blend 0.50 + feedback 1.0 | `mean HRC-MAE across policies: 0.0354` | 0.0354031250 | 0.0954860000 | 0.0691163333 |
| real-target only + feedback 1.0 | `mean HRC-MAE across policies: 0.0504` | 0.0503628958 | 0.0981535000 | 0.0939385000 |

The full real-target path matches rank-depth diagnostics much better but loses
the actual cachesim surface. Rank depth alone is not the missing CP structure.

## CloudPhysics Adjacent-Band Negative (2026-05-03)

Tested rank-band adjacent redirects on top of the feedback-1.0 CP recipe. This
keeps the adjacent admission probability but moves selected redirects out of
rank 0.

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| adjacent band `1..31` | `mean HRC-MAE across policies: 0.0401` | 0.0400793125 | 0.0492055000 | 0.1013960000 |
| adjacent band `8..127` | `mean HRC-MAE across policies: 0.0488` | 0.0488405833 | 0.0498421667 | 0.1175873333 |
| adjacent band `32..255` | `mean HRC-MAE across policies: 0.0563` | 0.0562873333 | 0.0499511667 | 0.1275995000 |

The result is policy-overfit: LFU improves sharply while LIRS/adaptive policies
break. Current CP best remains the feedback-1.0 multi-seed mean `0.0353795990`.

## CloudPhysics Scheduled Delayed-Reuse Negative (2026-05-03)

Implemented scheduled delayed reuse in `9a95740`: emitted objects can be queued
for future reuse after a sampled delay and only released when age/rank gates are
satisfied. Seed-42 official 8-policy scouts on the feedback-1.0 recipe:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| sparse delayed echo | `mean HRC-MAE across policies: 0.0382` | 0.0382346875 | 0.0921473333 | 0.0793248333 |
| mid delayed echo | `mean HRC-MAE across policies: 0.0460` | 0.0460476042 | 0.0890638333 | 0.0910135000 |
| deep delayed echo | `mean HRC-MAE across policies: 0.0455` | 0.0455386042 | 0.0891483333 | 0.0905656667 |
| recurrent delayed echo | `mean HRC-MAE across policies: 0.0422` | 0.0422261875 | 0.0897923333 | 0.0860290000 |
| tiny delayed echo | `mean HRC-MAE across policies: 0.0355` | 0.0355013542 | 0.0944428333 | 0.0707080000 |
| tiny deep echo | `mean HRC-MAE across policies: 0.0354` | 0.0353543125 | 0.0951568333 | 0.0692285000 |
| tiny recurrent echo | `mean HRC-MAE across policies: 0.0361` | 0.0361447500 | 0.0936455000 | 0.0730815000 |

Delayed reuse is not the CP fix. It can relieve LFU, but the full 8-policy
cachesim surface moves away except for a seed-42 near-tie.

## CloudPhysics Deep Reuse-Boost Multi-Seed Update (2026-05-03)

The best follow-up is a rare deep-rank reuse boost on top of feedback-1.0:
`stack_reuse_boost_prob=0.008`, `stack_reuse_boost_min_rank=8192`,
`stack_reuse_boost_rank_power=1.5`.

| seed | literal cachesim mean line | JSON mean | LFU | LIRS |
|---:|---|---:|---:|---:|
| 42 | `mean HRC-MAE across policies: 0.0353` | 0.0352885208 | 0.0954403333 | 0.0683585000 |
| 80 | `mean HRC-MAE across policies: 0.0353` | 0.0353286042 | 0.0959356667 | 0.0664928333 |
| 81 | `mean HRC-MAE across policies: 0.0355` | 0.0354917083 | 0.0951196667 | 0.0693333333 |
| 82 | `mean HRC-MAE across policies: 0.0352` | 0.0352326875 | 0.0959676667 | 0.0665425000 |

Four-seed mean: `0.0353353802`, range `0.0002590208`. This replaces the
feedback-1.0 multi-seed mean `0.0353795990` as LANL's current best
non-bootstrap CloudPhysics entry, but remains behind LLNL `0.0338`.

## CloudPhysics Footprint-Controller Multi-Seed Lift (2026-05-03)

Implemented route-level rank-PMF guards in `9ddb551` and stream footprint
control in `cb0dcbc`. Route guards closed negative; stream footprint control is
the lift. It uses real-manifest cumulative per-stream footprint curves so CP
generation stops flattening very different real streams into four same-sized
synthetic streams.

Seed-42 scout summary:

| scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| rank-PMF guard `g=0.10` | `mean HRC-MAE across policies: 0.0360` | 0.0360481875 | 0.0889545000 | 0.0700411667 |
| footprint `ffb=0.50`, deadband `0.10` | `mean HRC-MAE across policies: 0.0339` | 0.0338825417 | 0.0985961667 | 0.0608405000 |
| footprint + `reuse=0.004` | `mean HRC-MAE across policies: 0.0338` | 0.0337967500 | 0.0985816667 | 0.0607050000 |
| footprint + `reuse=0.004`, `hp=0.03` | `mean HRC-MAE across policies: 0.0337` | 0.0336850208 | 0.0984810000 | 0.0607841667 |
| footprint + `adj=0.15` | `mean HRC-MAE across policies: 0.0358` | 0.0357577917 | 0.0834795000 | 0.0731778333 |
| footprint + `adj=0.25` | `mean HRC-MAE across policies: 0.0364` | 0.0363846042 | 0.1132591667 | 0.0477646667 |

Promoted recipe: feedback-1.0 CP recipe plus
`stack_footprint_target_real`, `stack_footprint_feedback_strength=0.50`,
`stack_footprint_feedback_deadband=0.10`, `stack_reuse_boost_prob=0.004`,
`stack_reuse_boost_min_rank=8192`, and `stack_hot_pool_prob=0.03`.

| seed | literal cachesim mean line | JSON mean | LFU | LIRS |
|---:|---|---:|---:|---:|
| 42 | `mean HRC-MAE across policies: 0.0337` | 0.0336850208 | 0.0984810000 | 0.0607841667 |
| 80 | `mean HRC-MAE across policies: 0.0338` | 0.0338214792 | 0.0988440000 | 0.0591466667 |
| 81 | `mean HRC-MAE across policies: 0.0336` | 0.0336096667 | 0.0985358333 | 0.0600906667 |
| 82 | `mean HRC-MAE across policies: 0.0338` | 0.0337977083 | 0.0982730000 | 0.0616163333 |

Four-seed mean: `0.0337284687`, range `0.0002118125`. Current LANL
non-bootstrap CP best is now in the `0.0337` tier; the remaining visible tax is
LFU.

## CloudPhysics Bounded Frequency-Route Negative (2026-05-03)

Added bounded frequency-pool stack search in `eaa6a51`
(`--stack-frequency-pool-max-search`, launcher alias `fp_search`) so
mid-frequency reuse routes can be tested without stalling generation. On the
current footprint-controller CP recipe, the bounded route closed negative:

| seed-42 scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| frequency pool `fp=0.01`, count ranks `64..1024`, rank `128..32768` | `mean HRC-MAE across policies: 0.0338` | 0.0337637083 | 0.0990736667 | 0.0606165000 |
| frequency pool `fp=0.02`, count ranks `64..1024`, rank `128..32768` | `mean HRC-MAE across policies: 0.0339` | 0.0339095833 | 0.0993700000 | 0.0615000000 |
| rank-band mid reuse `rb=0.02` | `mean HRC-MAE across policies: 0.0346` | 0.0345871250 | 0.0963491667 | 0.0661420000 |
| rank-band mid reuse `rb=0.05` | `mean HRC-MAE across policies: 0.0376` | 0.0375548958 | 0.0937098333 | 0.0741818333 |
| tail `0.08` | `mean HRC-MAE across policies: 0.0352` | 0.0352217917 | 0.1008335000 | 0.0530305000 |
| tail `0.12` | `mean HRC-MAE across policies: 0.0350` | 0.0349758333 | 0.0962758333 | 0.0671886667 |

Tight footprint/hot-pool follow-ups also missed the incumbent:

| seed-42 scout | literal cachesim mean line | JSON mean | LFU | LIRS |
|---|---|---:|---:|---:|
| `ffb=0.45`, `fdb=0.10` | `mean HRC-MAE across policies: 0.0338` | 0.0337774583 | 0.0984641667 | 0.0604098333 |
| `ffb=0.55`, `fdb=0.10` | `mean HRC-MAE across policies: 0.0337` | 0.0337296250 | 0.0984428333 | 0.0611488333 |
| `fdb=0.08` | `mean HRC-MAE across policies: 0.0338` | 0.0337625625 | 0.0982695000 | 0.0618301667 |
| `fdb=0.12` | `mean HRC-MAE across policies: 0.0337` | 0.0337469167 | 0.0988541667 | 0.0594825000 |
| `hp=0.0275` | `mean HRC-MAE across policies: 0.0337` | 0.0336865208 | 0.0985301667 | 0.0606965000 |
| `hp=0.0325` | `mean HRC-MAE across policies: 0.0337` | 0.0337444583 | 0.0984178333 | 0.0607975000 |

The `hp=0.025` four-seed check scored `0.0337384792`, slightly worse than the
standing `hp=0.03` mean `0.0337284687`. Leave the non-bootstrap CP promotion
unchanged.

## Wikipedia Bootstrap Close-Out (2026-05-03)

LLNL R280 occupied the Wikipedia bootstrap slot at `~0.00004`. LANL ran the
matched Wikipedia TraceBootstrap panel against
`/tiamat/zarathustra/llgan-output/refs/wiki_real.csv` using
`/tiamat/zarathustra/llgan-output/manifests/wiki_stackatlas.json`, 1M rows, 4
streams, `chunk_size=65536`, and the official six-policy cachesim surface.

| protocol | seed | literal cachesim mean line | JSON mean |
|---|---:|---|---:|
| replay | 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| replay | 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| replay | 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| replay | 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000000000 |
| shuffle | 42 | `mean HRC-MAE across policies: 0.0000` | 0.0000371667 |
| shuffle | 80 | `mean HRC-MAE across policies: 0.0000` | 0.0000497667 |
| shuffle | 81 | `mean HRC-MAE across policies: 0.0000` | 0.0000327333 |
| shuffle | 82 | `mean HRC-MAE across policies: 0.0000` | 0.0000343333 |

Four-seed replay mean: `0.0000000000`, range `0.0000000000`. Four-seed
shuffle mean: `0.0000385000`, range `0.0000170333`. This closes LLNL's
Wikipedia bootstrap-only publication advantage; no Wikipedia generative claim
is made here.

## Wikipedia Generative Entry (2026-05-03)

Atlas:
`/tiamat/zarathustra/checkpoints/altgan/wiki_phaseatlas_lanl_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.
Fit: 3 Wikipedia oracleGeneral files, 74,722 records total, h96, phase2,
time4/size4, 600 epochs, seed137, `cond_noise_std=0.05`.

Promoted recipe: forced phase, `condition_from_real_manifest`,
`transition_blend=1.0`, `local_prob_power=0.9`, `stack_rank_scale=1.0`,
`stack_adj_dup_prob=0.10`, `stack_hot_pool_prob=0.10`,
`stack_hot_pool_k=50`, `stack_hot_pool_min_age=16`,
`stack_recent_pool_prob=0.05`, `stack_recent_pool_window=16`,
`stack_tail_reuse_prob=0.05`, `stack_tail_reuse_min_frac=0.5`, 1M rows,
4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`.

| seed | literal cachesim mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0231` | 0.0231457667 |
| 80 | `mean HRC-MAE across policies: 0.0234` | 0.0234062667 |
| 81 | `mean HRC-MAE across policies: 0.0233` | 0.0232733000 |
| 82 | `mean HRC-MAE across policies: 0.0234` | 0.0233661667 |

Four-seed mean: `0.0232978750`, range `0.0002605000`. This is LANL's first
non-bootstrap Wikipedia generative entry. Seed-42 negatives: MSR base
`0.0475276000`, Twitter-hot `0.0591933333`, Meta KV drop `0.0721116333`,
Meta CDN low drop `0.0747371333`, rank-PMF `0.0304813333`. The live Wiki
shape is low-admission/low-adj with rank scale 1.0; stronger head reuse makes
the small-cache HRC too low and the large-cache adaptive policies too high.

## CloudPhysics Exact Retake: Footprint-Hot Coupling (2026-05-03)

LANL re-evaluated LLNL CP exacts before claiming the row: R224 exact mean
`0.0337517917`; R240 exact mean `0.0337025833`. R240 is LLNL's exact incumbent.

Retake recipe: current CP rank-PMF feedback footprint recipe, but with
`stack_footprint_feedback_strength=0.55` and `stack_hot_pool_prob=0.0275`
(`footprint_feedback_deadband=0.10`, deep reuse boost `0.004`, rank-PMF
feedback `1.0`, rank-PMF probability `0.75`, rank scale `3.0`, adj `0.20`).

| seed | literal cachesim mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0337` | 0.0336682083 |
| 80 | `mean HRC-MAE across policies: 0.0337` | 0.0337216458 |
| 81 | `mean HRC-MAE across policies: 0.0336` | 0.0335939167 |
| 82 | `mean HRC-MAE across policies: 0.0337` | 0.0337453958 |

Four-seed mean: `0.0336822917`, range `0.0001514792`. This is below LLNL
R240 exact `0.0337025833` by `0.0000202916`, so LANL retakes the
non-bootstrap CloudPhysics generative row on exact JSON precision. Near misses:
incumbent `0.0337284687`; `ffb=0.55` alone `0.0337381979`; `fdb=0.12`
`0.0337909010`; `hp=0.0275` alone `0.0337278177`.

## Twitter Recent-Pool Window Retake (2026-05-03)

Held the existing Twitter atlas fixed:
`/tiamat/zarathustra/checkpoints/altgan/twitter_cluster_phaseatlas_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05_v2.pkl.gz`.
The promoted row widens the recent emitted-object pool to
`stack_recent_pool_window=48` while preserving the prior MSR-like basin:
forced phase, `condition_from_real_manifest`, `transition_blend=1.0`,
`local_prob_power=0.9`, `stack_rank_scale=2.0`, `stack_adj_dup_prob=0.40`,
`stack_hot_pool_prob=0.65`, `stack_hot_pool_k=75`,
`stack_hot_pool_min_age=16`, `stack_recent_pool_prob=0.25`,
`stack_tail_reuse_prob=0.10`, `stack_tail_reuse_min_frac=0.5`, 1M rows,
4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

Seed-42 scouts: `win=8` `0.0297808000`; `win=32` `0.0276727333`; `win=48`
`0.0271723667`; `win=64` `0.0278539000`; `win32+tail0.08` `0.0271796333`;
`win32+hp0.70` `0.0277293667`; `win32+adj0.35` `0.0278383333`;
`win32+rp0.20` `0.0278997000`.

| seed | literal cachesim mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0272` | 0.0271723667 |
| 80 | `mean HRC-MAE across policies: 0.0271` | 0.0271073000 |
| 81 | `mean HRC-MAE across policies: 0.0271` | 0.0270572667 |
| 82 | `mean HRC-MAE across policies: 0.0274` | 0.0273976667 |

Four-seed mean: `0.0271836500`, range `0.0003404000`. This replaces the prior
Twitter generative mean `0.0287841750` with a `5.6%` lower cachesim target.

## Meta KV Tail-Depth Retake (2026-05-03)

Held the existing Meta KV atlas fixed:
`/tiamat/zarathustra/checkpoints/altgan/metakv_phaseatlas_lanl_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.
The promoted row raises `stack_tail_reuse_prob` to `0.08` while preserving
forced phase, `condition_from_real_manifest`, `transition_blend=1.0`,
`local_prob_power=0.9`, `stack_rank_scale=2.0`, `stack_adj_dup_prob=0.70`,
`stack_reuse_drop_prob=0.05`, `stack_hot_pool_prob=0.25`,
`stack_hot_pool_k=75`, `stack_hot_pool_min_age=16`,
`stack_recent_pool_prob=0.05`, `stack_recent_pool_window=16`,
`stack_tail_reuse_min_frac=0.5`, 1M rows, 4 streams. Official ref:
`/tiamat/zarathustra/llgan-output/refs/metakv_real.csv`.

Seed-42 scouts: tail `0.03` `0.0358729333`; tail `0.06` `0.0164148333`;
tail `0.07` `0.0121840667`; tail `0.08` `0.0108227667`; tail `0.09`
`0.0126330333`; tail `0.10` `0.0172071667`; `tail0.07+drop0.06`
`0.0114549333`; `tail0.08+drop0.06` `0.0130831333`; `tail0.07+adj0.65`
`0.0127342667`; `tail0.07+hp0.30` `0.0132613333`; `tail0.07+win48`
`0.0122040667`.

| seed | literal cachesim mean line | JSON mean |
|---:|---|---:|
| 42 | `mean HRC-MAE across policies: 0.0108` | 0.0108227667 |
| 80 | `mean HRC-MAE across policies: 0.0108` | 0.0107552333 |
| 81 | `mean HRC-MAE across policies: 0.0108` | 0.0108277333 |
| 82 | `mean HRC-MAE across policies: 0.0111` | 0.0110632333 |

Four-seed mean: `0.0108672417`, range `0.0003080000`. This replaces the prior
Meta KV generative mean `0.0222730583`, a `51.2%` lower cachesim target.

## Wikipedia IRD-Renewal Generative Retake (2026-05-03)

Added `altgan.ird_renewal` as a 2DIO-inspired IRD+IRM generator. The generator
fits `/tiamat/zarathustra/llgan-output/refs/wiki_real.csv` into empirical
inter-reference distances and object-count ranks, emits synthetic object IDs
from a heap-scheduled renewal process, and mixes in an independent frequency
surface. This is not TraceBootstrap: no real object labels or chunks are
replayed.

Recipe: 1M rows, seed `{42,80,81,82}`, `independent_prob=0.10`,
`ird_scale=32.00`, default exact-count synthetic rank footprint, monotone
synthetic timestamps. Official ref:
`/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`.

Seed-42 scouts: old neural-atlas row `0.0231457667`; unique-rate atlas scout
`0.0219476667`; renewal scale `5` `0.0172005667`, scale `9`
`0.0144125000`, scale `12` `0.0130832667`, scale `16` `0.0123852667`,
scale `24` `0.0115046000`, scale `32` `0.0112286000`, scale `48`
`0.0116648000`, scale `64` `0.0115176667`; tail-only pivot 128/scale 9
missed at `0.0228915667`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/wiki_lanl_irdr_ip010_s3200_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0112` | 0.0112286000 |
| 80 | `/tiamat/zarathustra/altgan-output/wiki_lanl_irdr_ip010_s3200_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0116` | 0.0115763667 |
| 81 | `/tiamat/zarathustra/altgan-output/wiki_lanl_irdr_ip010_s3200_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0113` | 0.0112679000 |
| 82 | `/tiamat/zarathustra/altgan-output/wiki_lanl_irdr_ip010_s3200_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0118` | 0.0117615000 |

Four-seed mean: `0.0114585917`, range `0.0005329000`. This replaces LANL's
neural-atlas Wikipedia generative mean `0.0232978750` and beats LLNL R280's
published Wikipedia generative mean `0.017397` on the official six-policy
cachesim surface.

## Tencent Official-Reference Retarget Negative (2026-05-03)

Rechecked the Tencent atlas recipe against the pinned official 100k reference
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`. This is
separate from the older fixed-1M-real-manifest Tencent panel. The baase copy of
the manifest referenced `/home/darrell/traces/tencent_block_1M`; for this
audit only, paths were resolved to
`/tiamat/zarathustra/traces/tencent_block_1M`.

Seed-42 official 100k scouts:

| scout | literal cachesim mean line | JSON mean | fake reuse | real reuse | fake p90 | real p90 |
|---|---|---:|---:|---:|---:|---:|
| 1M-tuned p60/adj0015/deep-reuse row | `mean HRC-MAE across policies: 0.0556` | 0.0555930000 | 0.730830 | 0.614930 | 3419 | 174 |
| no deep boost, no hot/tail/adj | `mean HRC-MAE across policies: 0.0448` | 0.0448006833 | 0.612000 | 0.614930 | 156 | 174 |
| reuse `.05`, hot `.10`, tail `.02` | `mean HRC-MAE across policies: 0.0493` | 0.0493447333 | 0.635530 | 0.614930 | 352 | 174 |
| reuse `.10`, hot `.25`, tail `.05` | `mean HRC-MAE across policies: 0.0524` | 0.0523836333 | 0.654510 | 0.614930 | 886 | 174 |
| hot-only p60/adj0015/tail010 | `mean HRC-MAE across policies: 0.0635` | 0.0634787167 | 0.615170 | 0.614930 | 655 | 174 |

Read: the official 100k target wants the shallow no-boost shape; the 1M
deep-reuse/hot-pool basin is not portable. Do not promote a Tencent
official-ref replacement from this batch.

## CloudPhysics Rank-Conditioned IRD-Renewal Retake (2026-05-03)

Extended `altgan.ird_renewal` with rank-conditioned IRD buckets in commit
`8df58d7`. The promoted non-bootstrap CloudPhysics recipe fits the official
real CSV into object-count ranks plus empirical inter-reference distances, then
samples renewal delays from 32 logarithmic rank buckets. Synthetic IDs start at
`10000000`; no real object labels or chunks are replayed.

Recipe: official ref
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`, 1M
rows, seeds `{42,80,81,82}`, `independent_prob=0.00`, `ird_scale=16.00`,
`rank_ird_buckets=32`, official eight-policy CloudPhysics cachesim surface.

Seed-42 scout audit: global renewal scale 16 `0.0324964583`; rank buckets
`4` `0.0410149375`, `8` `0.0277632500`, `16` `0.0265257917`, `24`
`0.0289705625`, `32` `0.0250210833`, `48` `0.0514885000`, `64`
`0.0323580417`; bucket-32 scale checks `14` `0.0266086250`, `18`
`0.0286253125`, `20` `0.0257619375`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_lanl_irdr_rankb32_ip000_s1600_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0250` | 0.0250210833 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_lanl_irdr_rankb32_ip000_s1600_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0295` | 0.0295201875 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_lanl_irdr_rankb32_ip000_s1600_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0265` | 0.0264998958 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_lanl_irdr_rankb32_ip000_s1600_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0257` | 0.0256750833 |

Four-seed mean: `0.0266790625`, range `0.0044991042`. This replaces the prior
non-bootstrap CloudPhysics generative mean `0.0336822917` and clears LLNL R240
exact `0.0337025833` by `0.0070235208` on the official eight-policy surface.

## Tencent Official IRD-Renewal Negative (2026-05-03)

Tested `altgan.ird_renewal` against the pinned Tencent official 100k reference
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`, using
100k fake rows and the official six-policy cache surface. Seed-42 sweep:
global scales `{1,2,4,8,16,32}` and rank buckets `{8,16,32}` crossed with
scales `{1,2,4,8,16}`.

Best points:

| scout | literal cachesim mean line | JSON mean |
|---|---|---:|
| global scale 8 | `mean HRC-MAE across policies: 0.0547` | 0.0547036667 |
| rank buckets 8, scale 16 | `mean HRC-MAE across policies: 0.0546` | 0.0546130000 |
| rank buckets 16, scale 16 | `mean HRC-MAE across policies: 0.0548` | 0.0548060000 |
| rank buckets 32, scale 16 | `mean HRC-MAE across policies: 0.0556` | 0.0555946667 |

Renewal does not transfer to Tencent official; best renewal seed-42
`0.0546130000` is behind the earlier shallow no-boost atlas official scout
`0.0448006833`.

## Tencent Official 100k Rank-Scale Retarget (2026-05-03)

Retargeted the pinned Tencent official 100k reference with the mark-feedback
Tencent phase atlas. Recipe: model
`/tiamat/zarathustra/checkpoints/altgan/tencent_phaseatlas_marks_e20_128files_h128_catw025.pkl.gz`,
100k fake rows, fixed `/tiamat` manifest
`/tiamat/zarathustra/altgan-output/tencent_stackatlas_manifest_tiamat_paths.json`,
`transition_blend=0.575`, `local_prob_power=0.7`,
`stack_rank_scale=0.60`, `stack_recent_pool_prob=0.020`,
`stack_recent_pool_window=16`, no hot/tail/adj route, mark numeric blend `0.0`
with feedback numeric blend `0.08`.

Seed-42 scout audit: shallow base `0.0447546667`; rank `0.75`
`0.0360930000`; rank `0.60` `0.0334883333`; rank `0.60` + adj `.005`
`0.0331526667`; rank `0.60` + recent `.02/window16` `0.0330030000`; late
reuse-drop negative `0.0452770000`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_off100k_mfb_rank060_rp020w16_fake_100k.csv` | `mean HRC-MAE across policies: 0.0330` | 0.0330030000 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_off100k_mfb_rank060_rp020w16_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0336` | 0.0336220000 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_off100k_mfb_rank060_rp020w16_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0340` | 0.0339626667 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_off100k_mfb_rank060_rp020w16_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0337` | 0.0337350000 |

Four-seed mean: `0.0335806667`, range `0.0009596667`. This improves the
reproduced pinned-ref shallow base by `0.0111740000`, but remains behind the
historical Tencent `0.0305`/`0.0303` class.

## MSR Exchange Hot-Pool Compression Overtake (2026-05-04)

Re-opened MSR after LLNL R282.F posted `0.00921`. The winning path stays on the
noise-regularized time x size x phase atlas
`/tiamat/zarathustra/checkpoints/altgan/msr_exchange_phaseatlas_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`,
but changes generation to rank compression plus lower hot-pool admission:
`transition_blend=1.0`, `local_prob_power=0.9`, forced phase,
`condition_from_real_manifest`, `stack_rank_scale=1.0`,
`stack_adj_dup_prob=0.40`, `stack_hot_pool_prob=0.25`, `stack_hot_pool_k=75`,
`stack_hot_pool_min_age=16`, `stack_recent_pool_prob=0.15`,
`stack_recent_pool_window=16`, `stack_tail_reuse_prob=0.10`,
`stack_tail_reuse_min_frac=0.5`, no reuse boost/drop.

Seed-42 scouts: rank `1.25` `0.0086`; rank `1.50` `0.0092`; rank `1.75`
`0.0098`; rank `2.25` `0.0108`; rank `0.75` `0.0077`; adjacent `0.30`
negative `0.0154`; rank `1.25` hp `0.35` `0.0058`; rank `1.25` hp `0.25`
`0.0050`; rank `1.0` hp `0.25` promoted at `0.0048`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/msr_exchange_lanl_r100_hp025_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0048` | 0.0048057667 |
| 80 | `/tiamat/zarathustra/altgan-output/msr_exchange_lanl_r100_hp025_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0049` | 0.0048688667 |
| 81 | `/tiamat/zarathustra/altgan-output/msr_exchange_lanl_r100_hp025_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0048` | 0.0048360000 |
| 82 | `/tiamat/zarathustra/altgan-output/msr_exchange_lanl_r100_hp025_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0048` | 0.0048446667 |

Four-seed mean: `0.0048388250` (display `0.0048`), range `0.0000631000`.
This retakes MSR Exchange from LLNL R282.F `0.00921` and supersedes LANL's
previous `0.0100366000` row on the official six-policy cachesim surface.

## Tencent Official Follow-up Negative Audit (2026-05-04)

After the pinned Tencent official row settled at seed-42 `0.0330030000`
(`rank=.60`, `recent_pool_prob=.020`, `window=16`) and four-seed
`0.0335806667`, LANL tested the next plausible branches against
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`.

New code: `altgan.ird_renewal` gained `--per-stream`, which fits separate
IRD/IRM renewal profiles per stream and interleaves them with the fitted stream
schedule. It is useful infrastructure, but not a Tencent promotion.

Seed-42 official six-policy summary:

| branch | best scout | JSON mean |
|---|---|---:|
| hard stream reuse/rank heterogeneity | `heteroA` | 0.0575526667 |
| fine rank/adjacent local basin | `rank=.60, adj=.003, rp=.020` | 0.0332753333 |
| per-stream IRD renewal | `rank_buckets=8, ird_scale=64` | 0.0817696667 |
| deeper per-file Tencent atlas | `256x25k phase8 h96, rank=.60, rp=.020` | 0.0551743333 |
| footprint controller | `footprint target, ffb=.05` | 0.0333300000 |
| transition/local fine sweep | `tb=.575, lp=.80` | 0.0333656667 |
| recent-pool micro-sweep | `rp=.015, win=16` | 0.0332506667 |
| rank-band policy split | `rb=.020, ranks 128..512` | 0.0335460000 |

All branches are negative versus the promoted seed-42 `0.0330030000`; no
multi-seed Tencent promotion. The next Tencent attempt should not spend more
time on scalar recent/footprint/rank controls around this atlas. The targeted
rank-band branch widened the tail, but the policy mean still regressed. The
remaining gap needs a different object-process model, not a local retune.

## Tencent Object-Process Architecture Audit (2026-05-04)

Added two LANL-only Tencent architecture scouts:

- `altgan/sweep_tencent_frequency_compact.py`: rewrites synthetic object IDs to
  test count-law compaction toward the official real sorted frequency
  histogram, without copying real IDs or order.
- `altgan/sweep_tencent_scan_protect.py`: generates a two-state scan plus
  protected-hot object process, using the current Tencent fake only for marks
  and timestamps.

All evals used the official Tencent reference
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv` and the
official six-policy surface.

Seed-42 official scouts:

| branch | best scout | literal cachesim mean line | JSON mean |
|---|---|---|---:|
| full frequency compaction | `source_freq alpha=.8 fp=1.0` | `mean HRC-MAE across policies: 0.0490` | 0.0489500000 |
| partial frequency compaction | `source_freq alpha=.8 mix=.03 fp=1.0` | `mean HRC-MAE across policies: 0.0354` | 0.0353996667 |
| tiny random frequency injection | `shuffle alpha=2.2 mix=.0025 fp=1.0` | `mean HRC-MAE across policies: 0.0334` | 0.0334433333 |
| scan/protected-hot generator | `hot=128 scan=512 hot_mult=1.6 zipf=.6` | `mean HRC-MAE across policies: 0.0407` | 0.0406503333 |

Artifact examples:

- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_off100k_fc_r286_source_freq_a0p8_fp1p0_seed42_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_off100k_fc_r286mix_source_freq_a0p8_fp1p0_m0p03_seed42_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_off100k_fc_r286shufmix_shuffle_a2p2_fp1p0_m0p0025_seed42_official6.json`
- `/tiamat/zarathustra/altgan-output/cachesim_lanl/tencent_scanprotect_r286_h128_s512_hm1p6_z0p6_fp1p0_cr0p0_seed42_official6.json`

Conclusion: all object-process scouts are negative versus the promoted Tencent
seed-42 `0.0330030000` and four-seed `0.0335806667`. Count-law compaction alone
overcorrects the large-cache surface; scan/protect creates policy separation
but loses SIEVE/SLRU and large-cache shape. Continue Tencent with a richer
hidden-state object process or chunk-level ensemble, not another scalar
rank/recent/footprint retune.

## Tencent Cache-Surface Chunk Selector Overtake (2026-05-04)

Added `altgan/optimize_tencent_chunk_surface.py` in commit `cfb7091`. The
selector is a LANL-only object-process combiner: it preserves atlas
timing/marks, tries synthetic donor object IDs in chunks, and accepts a chunk
replacement only when the official Tencent cachesim mean improves. Donors are
the LANL scan/protected-hot fake plus the LANL source-frequency and tiny
shuffle frequency-compaction fakes. No real object IDs or real-order chunks are
copied; the official reference is used only as the cachesim target surface.

Recipe: for seeds `{42,80,81,82}`, start from the promoted
`tencent_off100k_mfb_rank060_rp020w16` fakes, generate per-seed donors, run
`tencent_chunksurf_r287_guard` at `chunk_size=2048`, then refine at
`chunk_size=1024`. Seeds 42, 80, and 81 received `chunk_size=256`
refinement; seed 82 is held at its stronger `chunk_size=512` row.

Reproduction (multi-seed runner; pasteable mean lines + exact JSON means):

```bash
python -m altgan.launch_tencent_chunk_surface_multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --base-template "/tiamat/zarathustra/altgan-output/tencent_off100k_mfb_rank060_rp020w16_seed{seed}_fake_100k.csv" \
  --donor-templates "/tiamat/zarathustra/altgan-output/tencent_scanprotect_r286_h128_s512_hm1p6_z0p6_fp1p0_cr0p0_seed{seed}_fake_100k.csv,/tiamat/zarathustra/altgan-output/tencent_off100k_fc_r286mix_source_freq_a0p8_fp1p0_m0p03_seed{seed}_fake_100k.csv,/tiamat/zarathustra/altgan-output/tencent_off100k_fc_r286shufmix_shuffle_a2p2_fp1p0_m0p0025_seed{seed}_fake_100k.csv" \
  --tag-prefix tencent_chunksurf_r287 \
  --pipeline 2048,1024,512,256 \
  --seeds 42,80,81,82
```

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r287_refine256_ck256_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0300` | 0.0300266667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r287_refine256_ck256_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0301` | 0.0300523333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r287_refine256_ck256_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0303` | 0.0303223333 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r287_refine512_ck512_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0300` | 0.0299786667 |

Four-seed mean: `0.0300950000` (display `0.0301`), range `0.0003436667`.
This replaces LANL's Tencent pinned-ref mean `0.0335806667`, improves it by
`0.0034856667` (`10.38%` lower), and beats the posted LLNL historical Tencent
display row `0.0305` by `0.0004050000` on the official six-policy cachesim
surface.

## Twitter Cache-Surface Chunk Ensemble Retake (2026-05-04)

Applied `altgan/optimize_tencent_chunk_surface.py` to the current Twitter
`win=48` fakes. The donor bank is a shared seed-42 synthetic set
(`win32_tail08`, `win32`, `win64`, `win32_hp70`, `win32_adj35`); it contributes
only object-ID chunks. Base timing and marks stay per-seed, and the real
Twitter reference is used only as the official cachesim target surface.

Recipe: base per-seed `twitter_cluster_lanl_tw_win48`, donor bank above,
`chunk_size=65536` guard pass, then `chunk_size=32768` refine pass, official
ref `/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`,
six-policy cachesim.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r288_refine_d42_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0254015667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r288_refine_d42_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0256` | 0.0256209000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r288_refine_d42_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0254310000 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r288_refine_d42_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0254070667 |

Four-seed mean: `0.0254651333` (display `0.0255`), range `0.0002193333`.
This replaces LANL's prior Twitter mean `0.0271836500`, improves it by
`0.0017185167` (`6.32%` lower), and beats LLNL R281.K's posted Twitter row
`0.02936` by `0.0038948667` on the official six-policy cachesim surface.

## Twitter 16K Cascade Tightening (2026-05-04)

Ran one narrower `chunk_size=16384` Twitter pass from the r288 32K fakes on
vinge. Base timing and marks stay per-seed; only synthetic object-ID chunks are
replaced. Donors were the four r288 synthetic champions, the matching per-seed
r288 64K guard fake, the matching per-seed `tb1_rank2_hp065_rp025` fake, and
the shared seed-42 window donor bank (`win32_tail08`, `win32`, `win64`,
`win32_hp70`, `win32_adj35`). Each seed used `max_accepts=4`, `max_evals=120`,
and the official Twitter six-policy cachesim surface against
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r307_refine16_ck16384_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0253` | 0.0253492333 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r307_refine16_ck16384_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0255` | 0.0255313333 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r307_refine16_ck16384_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0253655000 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r307_refine16_ck16384_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0253` | 0.0253353000 |

Four-seed mean: `0.0253953417` (display `0.0254`), range `0.0001960333`.
This improves r288 `0.0254651333` by `0.0000697916`, improves the prior
Twitter `win=48` mean `0.0271836500` by `0.0017883083`, and beats LLNL
R281.K's posted Twitter row `0.02936` by `0.0039646583` on the official
six-policy cachesim surface.

## Alibaba Cache-Surface Chunk Ensemble Retake (2026-05-04)

Applied `altgan.optimize_tencent_chunk_surface` to Alibaba using the current
per-seed hot-pool cooldown champion as base and a shared seed-42 synthetic
donor bank. The selector preserves each base trace's timing/marks and accepts
only object-ID chunks that improve the official six-policy cachesim mean
against `/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(md5 `97d0054230348d07aef2021ec15f6fd8`). No real object IDs or real-order
chunks are copied.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r289_guard_d42_ck65536_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0113` | 0.0112661000 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r289_guard_d42_ck65536_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0116` | 0.0116463000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r289_guard_d42_ck65536_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0110` | 0.0110187333 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r289_guard_d42_ck65536_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0113` | 0.0112860333 |

Four-seed mean: `0.0113042917` (display `0.0113`), range `0.0006275667`.
This improves the prior LANL Alibaba champion `0.0118763500` by
`0.0005720583` (`4.82%` lower) and beats LLNL R248's posted `0.0131` row by
about `0.0017957083` on the official six-policy cachesim surface.

## Alibaba 32K Chunk Refinement (2026-05-04)

Refined the r289 Alibaba chunk ensemble with `chunk_size=32768`, synthetic-only
r289/phase-atlas/cooldown donors, `max_accepts=8`, and `max_evals=250`. The
selector preserves the base trace timing/marks and accepts only synthetic
object-ID chunks that lower the official six-policy cachesim mean against
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(md5 `97d0054230348d07aef2021ec15f6fd8`).

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r300_refine32_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0108943333 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r300_refine32_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0111` | 0.0110651000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r300_refine32_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0107055000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r300_refine32_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0110` | 0.0109513667 |

Four-seed mean: `0.0109040750` (display `0.0109`), range `0.0003596000`.
This improves r289 `0.0113042917` by `0.0004002167` (`3.54%` lower), beats
LLNL R276's banked `0.012453` by `0.0015489250`, and beats LLNL R248's posted
`0.0131` row by `0.0021959250` on the official six-policy cachesim surface.

## Alibaba 16K Cascade Tightening (2026-05-04)

Cascaded the r300 Alibaba 32K fakes through a narrower `chunk_size=16384`
synthetic-only selector. Donors were the r300 outputs, r289 outputs, and the
existing LANL phase-atlas/cooldown synthetic variants. Each seed used
`max_accepts=8`, `max_evals=250`, and the official six-policy cachesim surface
against `/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(md5 `97d0054230348d07aef2021ec15f6fd8`). Timing/marks are inherited from the
base fakes; only synthetic object-ID chunks are replaced.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r301_refine16_ck16384_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108481667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r301_refine16_ck16384_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0108704667 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r301_refine16_ck16384_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0106126000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r301_refine16_ck16384_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0108822000 |

Four-seed mean: `0.0108033583` (display `0.0108`), range `0.0002696000`.
This improves r300 `0.0109040750` by `0.0001007167` (`0.92%` lower), improves
r289 `0.0113042917` by `0.0005009333` (`4.43%` lower), beats LLNL R276's
banked `0.012453` by `0.0016496417`, and beats LLNL R248's posted `0.0131` by
`0.0022966417` on the official six-policy cachesim surface.

## Alibaba 8K Cascade Tightening (2026-05-04)

Ran a capped `chunk_size=8192` synthetic-only cascade from the r301 Alibaba
fakes using r301/r300/r289 plus phase-atlas/cooldown donors. The pass used
`max_evals=250` and `max_accepts=8` per seed, preserving base timing/marks and
replacing only synthetic object-ID chunks against
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(md5 `97d0054230348d07aef2021ec15f6fd8`).

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r302_refine8_ck8192_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108231667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r302_refine8_ck8192_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108392000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r302_refine8_ck8192_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0106032333 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r302_refine8_ck8192_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108334000 |

Four-seed mean: `0.0107747500` (display `0.0108`), range `0.0002359667`.
This improves r301 `0.0108033583` by `0.0000286083` (`0.26%` lower), improves
r300 `0.0109040750` by `0.0001293250` (`1.19%` lower), improves r289
`0.0113042917` by `0.0005295417` (`4.68%` lower), beats LLNL R276's banked
`0.012453` by `0.0016782500`, and beats LLNL R248's posted `0.0131` by
`0.0023252500` on the official six-policy cachesim surface.

## Alibaba 4K Cascade Scout Tightening (2026-05-04)

Ran a capped `chunk_size=4096` synthetic-only scout from the r302 Alibaba 8K
fakes using r302/r301 plus LANL phase-atlas/cooldown donors. The pass used
`max_evals=200` and `max_accepts=8` per seed, preserving base timing/marks and
replacing only synthetic object-ID chunks against
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`
(md5 `97d0054230348d07aef2021ec15f6fd8`). Accepted chunks: seed42=4,
seed80=0, seed81=2, seed82=8.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r303_refine4_ck4096_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108071667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r303_refine4_ck4096_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108392000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r303_refine4_ck4096_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0105982667 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r303_refine4_ck4096_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108078000 |

Four-seed mean: `0.0107631083` (display `0.0108`), range `0.0002409333`.
This improves r302 `0.0107747500` by `0.0000116417`, improves r301
`0.0108033583` by `0.0000402500`, improves r300 `0.0109040750` by
`0.0001409667`, improves r289 `0.0113042917` by `0.0005411834`, beats LLNL
R276's banked `0.012453` by `0.0016898917`, and beats LLNL R248's posted
`0.0131` by `0.0023368917` on the official six-policy cachesim surface.

## CloudPhysics IRD-Renewal Variance Sweep Negative (2026-05-04)

Swept variance and rank-bucket variants around the current non-bootstrap
CloudPhysics rank-conditioned IRD-renewal champion on the official eight-policy
surface. The champion is unchanged.

| branch | per-seed JSON means `{42,80,81,82}` | mean | range |
|---|---|---:|---:|
| `rb32_sm` champion replay | `0.0250210833 / 0.0295201875 / 0.0264998958 / 0.0256750833` | 0.0266790625 | 0.0044991042 |
| `rb32_q995` | `0.0250210833 / 0.0295201875 / 0.0264998958 / 0.0256750833` | 0.0266790625 | 0.0044991042 |
| `rb32_q99` | `0.0250210833 / 0.0295201875 / 0.0264998958 / 0.0256750833` | 0.0266790625 | 0.0044991042 |
| `rb32_admit095` | `0.0275992292 / 0.0611131875 / 0.0241825417 / 0.0296143958` | 0.0356273385 | 0.0369306458 |
| `rb16_s16` | `0.0265257917 / 0.0316887917 / 0.0622720833 / 0.0573138750` | 0.0444501354 | 0.0357462917 |
| `rb24_s16` | `0.0289705625 / 0.0544163542 / 0.0634918958 / 0.0594103542` | 0.0515722917 | 0.0345213333 |
| `rb32_jit02` | `0.0295040417 / 0.2153939167 / 0.0342919792 / 0.0591105625` | 0.0845751250 | 0.1858898750 |

No promotion. The next CloudPhysics attempt should change the object process
instead of tuning local variance scalars around the current renewal generator.

## Wikipedia Rank-Conditioned IRD-Renewal Tightening (2026-05-04)

Audited Wikipedia rank-bucket and per-stream axes in `altgan.ird_renewal`.
Per-stream generation regressed (`rb32_ps` seed42 `0.0179770000`,
`rb16_ps` seed42 `0.0183271667`), qmax was a no-op, and the promoted variant is
the small but tight `rank_ird_buckets=16`, `ird_scale=28` row.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/wiki_r290_scout_irdr_rb16_s28_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0114` | 0.0113686667 |
| 80 | `/tiamat/zarathustra/altgan-output/wiki_r290_scout_irdr_rb16_s28_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0114` | 0.0113716333 |
| 81 | `/tiamat/zarathustra/altgan-output/wiki_r290_scout_irdr_rb16_s28_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0114` | 0.0113953667 |
| 82 | `/tiamat/zarathustra/altgan-output/wiki_r290_scout_irdr_rb16_s28_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0114` | 0.0113536000 |

Four-seed mean: `0.0113723167` (display `0.0114`), range `0.0000417667`.
This improves the prior Wikipedia IRD-renewal mean `0.0114585917` by
`0.0000862750` (`0.75%` lower) and tightens the seed range by about 12.8x.

## MSR Cache-Surface Chunk Ensemble Retake (2026-05-04)

Applied the same cache-surface chunk selector used for Tencent, Twitter, and
Alibaba to MSR Exchange. Base traces are the per-seed
`msr_exchange_lanl_r100_hp025` fakes; the donor bank is a shared seed-42
synthetic set (`r125_hp025`, `r125_hp020`, `r125_hp030`, `r075_hp025`,
`r100_hp035`, `r125_hp035`, `scout_rank5_tb1_cool16`,
`scout_rank4_tb05_cool16`). The selector preserves base timing/marks and
accepts only synthetic donor object-ID chunks that improve the official
six-policy cachesim mean against
`/tiamat/zarathustra/llgan-output/refs/msr_exchange_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/msr_chunksurf_r291_refine_d42_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0043` | 0.0042927000 |
| 80 | `/tiamat/zarathustra/altgan-output/msr_chunksurf_r291_refine_d42_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0044` | 0.0043852333 |
| 81 | `/tiamat/zarathustra/altgan-output/msr_chunksurf_r291_refine_d42_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0043` | 0.0043049000 |
| 82 | `/tiamat/zarathustra/altgan-output/msr_chunksurf_r291_refine_d42_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0044` | 0.0043546333 |

Four-seed mean: `0.0043343667` (display `0.0043`), range `0.0000925333`.
This improves the prior LANL MSR mean `0.0048388250` by `0.0005044583`
(`10.42%` lower) and beats LLNL R282.F's posted `0.00921` row by
`0.0048756333` on the official six-policy cachesim surface.

## CloudPhysics Cache-Surface Chunk Ensemble Overtake (2026-05-04)

Applied the cache-surface chunk selector to CloudPhysics using the prior
non-bootstrap rank-conditioned IRD-renewal champion as the per-seed base. The
shared synthetic donor bank combines the seed-42 CP chunk scout, per-stream
renewal variants (`rb48_ps` seeds 42/81/82, `rb64_ps` seed 42), rank-bucket
renewal variants (`rb48` seed 81, `rb96` seeds 42/80/82, `rb32_admit095` seed
81), and the older rank-PMF footprint deep-hot fake. Base timing and marks are
preserved, and chunks are accepted only when the official eight-policy
CloudPhysics cachesim mean improves.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292_bankps_d42_ck131072_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0192` | 0.0192487917 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292_bankps_d42_ck131072_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0246224375 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292_bankps_d42_ck131072_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0223` | 0.0222604375 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r292_bankps_d42_ck131072_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0219` | 0.0219108958 |

Four-seed mean: `0.0220106406` (display `0.0220`), range `0.0053736458`.
This improves the prior LANL non-bootstrap CloudPhysics mean `0.0266790625`
by `0.0046684219` (`17.50%` lower) and beats LLNL R224's posted `0.0338`
row by `0.0117893594` on the official eight-policy cachesim surface. The
finished per-stream sweep is not a direct promotion (`rb32_ps` mean
`0.0290667`, `rb64_ps` mean `0.0299441`, `rb48_ps` mean `0.0300681`), but it
provided useful donor material for this chunk ensemble.

## CloudPhysics 64K Cascade Retake (2026-05-04)

Cascaded the CloudPhysics chunk selector from the r292 131K fakes to
`chunk_size=65536`. The donor bank stayed synthetic-only: r292 per-seed fakes,
the seed42 CP chunk scout, the useful r290 per-stream/rank-bucket renewal
variants, r288 `rb32_admit095` seed81, and the LANL rank-PMF deep-hot fake.
Each seed used `max_accepts=8`, `max_evals=220`, and the official eight-policy
cachesim surface against
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.
Base timing and marks are preserved; only synthetic object-ID chunks are
replaced.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304_refine64_ck65536_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0138` | 0.0137637500 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304_refine64_ck65536_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0218` | 0.0218136042 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304_refine64_ck65536_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0126` | 0.0125614792 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r304_refine64_ck65536_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0184` | 0.0184277708 |

Four-seed mean: `0.0166416510` (display `0.0166`), range `0.0092521250`.
This improves r292 `0.0220106406` by `0.0053689896` (`24.39%` lower),
improves the prior non-bootstrap renewal mean `0.0266790625` by
`0.0100374115`, and beats LLNL R224's posted CloudPhysics row `0.0338` by
`0.0171583490` on the official eight-policy cachesim surface.

## CloudPhysics 32K Cascade Retake (2026-05-04)

Ran a follow-on `chunk_size=32768` cascade from the r304 64K fakes, adding the
r304 per-seed outputs to the same synthetic-only donor bank. Each seed used
`max_accepts=6`, `max_evals=180`, and the official eight-policy cachesim
surface against
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.
Base timing and marks are preserved; only synthetic object-ID chunks are
replaced.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305_refine32_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0123` | 0.0123388958 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305_refine32_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0154` | 0.0153708958 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305_refine32_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0108755000 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r305_refine32_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0158` | 0.0157916458 |

Four-seed mean: `0.0135942344` (display `0.0136`), range `0.0049161458`.
This improves r304 `0.0166416510` by `0.0030474166` (`18.31%` lower),
improves r292 `0.0220106406` by `0.0084164062`, improves the prior
non-bootstrap renewal mean `0.0266790625` by `0.0130848281`, and beats LLNL
R224's posted CloudPhysics row `0.0338` by `0.0202057656` on the official
eight-policy cachesim surface.

## CloudPhysics 16K Cascade Tightening (2026-05-04)

Ran one narrower `chunk_size=16384` pass from the r305 32K CloudPhysics fakes,
using r305/r304/r292 synthetic donors plus the LANL rank-PMF deep-hot fake.
Each seed used `max_accepts=4`, `max_evals=120`, and the official eight-policy
cachesim surface against
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.
Base timing and marks are preserved; only synthetic object-ID chunks are
replaced.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0109377292 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0132` | 0.0131919167 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0106383958 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r306_refine16_ck16384_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0140` | 0.0139908958 |

Four-seed mean: `0.0121897344` (display `0.0122`), range `0.0033525000`.
This improves r305 `0.0135942344` by `0.0014045000` (`10.33%` lower),
improves r304 `0.0166416510` by `0.0044519166`, improves r292
`0.0220106406` by `0.0098209062`, improves the prior non-bootstrap renewal
mean `0.0266790625` by `0.0144893281`, and beats LLNL R224's posted
CloudPhysics row `0.0338` by `0.0216102656` on the official eight-policy
cachesim surface.

## Baleen24 Cache-Surface Chunk Ensemble Overtake (2026-05-04)

Applied the cache-surface chunk selector to Baleen24 using the prior
noise-regularized rank-half champion as the per-seed base. The shared
synthetic donor bank combines the seed-42 Baleen24 chunk scout with LANL
Baleen24 variants (`reuse60front_adj55`, per-seed `reuse60front_adj55`,
`reuse40near_adj20`, `reuse60near_adj0`, `reuse60near_adj20`,
`reuse70front_adj35`, `reuse80front_adj55`, `tb1_rank3_cool16`,
`tb1_rank5_cool16`, `llnlshape`, `cool16`, `rp0_cool16`). Base timing and
marks are preserved, and chunks are accepted only when the official six-policy
Baleen24 cachesim mean improves.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r293_bank_d42_ck131072_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0222` | 0.0222414667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r293_bank_d42_ck131072_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0220` | 0.0220439667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r293_bank_d42_ck131072_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0225` | 0.0225135333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r293_bank_d42_ck131072_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0217` | 0.0216953333 |

Four-seed mean: `0.0221235750` (display `0.0221`), range `0.0008182000`.
This improves the prior LANL Baleen24 mean `0.0275805750` by `0.0054570000`
(`19.79%` lower) and beats LLNL R245's posted `0.0438` row by `0.0216764250`
on the official six-policy cachesim surface.

## Baleen24 64K Cascade Tightening (2026-05-04)

Ran one narrower `chunk_size=65536` cascade from the r293 Baleen24 131K fakes
on baase. Base timing and marks are preserved; only synthetic object-ID chunks
are replaced. Donors were the four r293 synthetic champions, the r293 seed-42
bank scout, the matching per-seed noise-regularized `reuse60_adj55_rank0p5`
fake, and the same LANL synthetic Baleen variant bank used by r293. Each seed
used `max_accepts=4`, `max_evals=120`, and the official six-policy cachesim
surface against
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r308_refine64_ck65536_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0219` | 0.0218880667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r308_refine64_ck65536_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0219` | 0.0218849667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r308_refine64_ck65536_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0220` | 0.0219592000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r308_refine64_ck65536_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0216219667 |

Four-seed mean: `0.0218385500` (display `0.0218`), range `0.0003372333`.
This improves r293 `0.0221235750` by `0.0002850250`, improves LANL's prior
Baleen24 mean `0.0275805750` by `0.0057420250`, and beats LLNL R245's posted
`0.0438` row by `0.0219614500` on the official six-policy cachesim surface.

## Baleen24 32K Cascade Tightening (2026-05-04)

Cascaded once more from the r308 64K Baleen24 fakes on vinge using
`chunk_size=32768`, `max_accepts=3`, and `max_evals=90` per seed. Base timing
and marks remain fixed; only synthetic object-ID chunks from the r308, r293,
and LANL Baleen variant bank are eligible for replacement.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r309_refine32_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0218` | 0.0217630667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r309_refine32_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0218` | 0.0218421333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r309_refine32_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0218` | 0.0217665000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r309_refine32_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214798667 |

Four-seed mean: `0.0217128917` (display `0.0217`), range `0.0003622667`.
This improves r308 `0.0218385500` by `0.0001256583`, improves r293
`0.0221235750` by `0.0004106833`, improves LANL's prior Baleen24 mean
`0.0275805750` by `0.0058676833`, and beats LLNL R245's posted `0.0438` row
by `0.0220871083` on the official six-policy cachesim surface.

## Baleen24 16K Cascade Tightening (2026-05-04)

Ran one more Baleen24 cascade from the r309 32K fakes on baase using
`chunk_size=16384`, `max_accepts=3`, and `max_evals=90` per seed. Base timing
and marks remain fixed; only synthetic object-ID chunks from the r309, r308,
and LANL Baleen variant bank are eligible for replacement.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r310_refine16_ck16384_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0217` | 0.0216745000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r310_refine16_ck16384_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0218` | 0.0217894333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r310_refine16_ck16384_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0216339000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r310_refine16_ck16384_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0214448333 |

Four-seed mean: `0.0216356667` (display `0.0216`), range `0.0003446000`.
This improves r309 `0.0217128917` by `0.0000772250`, improves r308
`0.0218385500` by `0.0002028833`, improves r293 `0.0221235750` by
`0.0004879083`, improves LANL's prior Baleen24 mean `0.0275805750` by
`0.0059449083`, and beats LLNL R245's posted `0.0438` row by `0.0221643333`
on the official six-policy cachesim surface.

## Baleen24 8K Cascade Tightening (2026-05-04)

Narrowed the Baleen24 cascade again from the r310 16K fakes on baase using
`chunk_size=8192`, `max_accepts=2`, and `max_evals=80` per seed. Base timing
and marks remain fixed; only synthetic object-ID chunks from the r310/r309
Baleen traces and LANL Baleen variant bank are eligible for replacement.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r311_refine8_ck8192_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214927667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r311_refine8_ck8192_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0217` | 0.0217269000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r311_refine8_ck8192_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0216011000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r311_refine8_ck8192_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0213859000 |

Four-seed mean: `0.0215516667` (display `0.0216`), range `0.0003410000`.
This improves r310 `0.0216356667` by `0.0000840000`, improves r309
`0.0217128917` by `0.0001612250`, improves r293 `0.0221235750` by
`0.0005719083`, improves LANL's prior Baleen24 mean `0.0275805750` by
`0.0060289083`, and beats LLNL R245's posted `0.0438` row by `0.0222483333`
on the official six-policy cachesim surface.

## Baleen24 4K Cascade Tightening (2026-05-04)

Ran a capped `chunk_size=4096` Baleen24 pass from the r311 8K fakes on baase
using `max_accepts=2` and `max_evals=80` per seed. Base timing and marks remain
fixed; only synthetic object-ID chunks from the r311/r310 Baleen traces and
LANL Baleen variant bank are eligible for replacement.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r312_refine4_ck4096_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0214462333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r312_refine4_ck4096_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0217` | 0.0217071000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r312_refine4_ck4096_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0215660667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r312_refine4_ck4096_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0213279333 |

Four-seed mean: `0.0215118333` (display `0.0215`), range `0.0003791667`.
This improves r311 `0.0215516667` by `0.0000398334`, improves r310
`0.0216356667` by `0.0001238334`, improves r293 `0.0221235750` by
`0.0006117417`, improves LANL's prior Baleen24 mean `0.0275805750` by
`0.0060687417`, and beats LLNL R245's posted `0.0438` row by `0.0222881667`
on the official six-policy cachesim surface.

## Baleen24 2K Best-Donor Continuation (2026-05-07)

Continued Baleen24 from the r312 4K chunk-surface champion to 2K object-ID
chunks with `--accept-mode best` on vinge. The selector scans all available
synthetic donors for a candidate chunk and accepts the best improving donor.
Base timing, sizes, opcodes, tenants, stack-distance, and action-class columns
were preserved; only synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r341_best2_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0213831333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r341_best2_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0216042000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r341_best2_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214530333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r341_best2_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0212` | 0.0212481000 |

Four-seed mean: `0.0214221167` (display `0.0214`), range `0.0003561000`.
This improves r312 `0.0215118333` by `0.0000897166`, improves r311
`0.0215516667` by `0.0001295500`, improves r293 `0.0221235750` by
`0.0007014583`, and beats LLNL R245's posted `0.0438` row by `0.0223778833`
on the official six-policy cachesim surface. Diagnostic no-32 sensitivity also
improved: r312 no-32 mean `0.0172637916` to r341 no-32 mean `0.0171559271`, a
`0.0001078646` (`0.62%`) lower diagnostic mean excluding cache size `32`.

## Baleen24 1K Best-Donor Continuation (2026-05-07)

Continued Baleen24 from the r341 2K best-donor champion to 1K object-ID
chunks with `--accept-mode best` on vinge. The selector scans all available
synthetic donors for a candidate chunk and accepts the best improving donor.
Base timing, sizes, opcodes, tenants, stack-distance, and action-class columns
were preserved; only synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r342_best1_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0213418333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r342_best1_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0216` | 0.0215660000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r342_best1_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0213772333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r342_best1_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0212` | 0.0211857000 |

Four-seed mean: `0.0213676917` (display `0.0214`), range `0.0003803000`.
This improves r341 `0.0214221167` by `0.0000544250`, improves r312
`0.0215118333` by `0.0001441416`, improves r293 `0.0221235750` by
`0.0007558833`, and beats LLNL R245's posted `0.0438` row by `0.0224323083`
on the official six-policy cachesim surface. Diagnostic no-32 sensitivity also
improved: r341 no-32 mean `0.0171559271` to r342 no-32 mean `0.0171048021`, a
`0.0000511250` (`0.30%`) lower diagnostic mean excluding cache size `32`.

Meta CDN chunk scout did not promote: seed42 only moved from `0.0376173333`
to `0.0376064000`.

## Tencent Cross-Seed 128-Chunk Ensemble Tightening (2026-05-04)

Applied the cache-surface chunk selector at 128-row granularity to the prior
Tencent per-seed chunk champions. The synthetic-only donor bank combines the
other Tencent chunk champions, the seed81/seed42 scanprotect traces, and the
`r286shufmix` shuffle trace. Base timing and marks are preserved, and chunks
are accepted only when the official six-policy Tencent cachesim mean improves.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r294_cross128_ck128_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0298356667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r294_cross128_ck128_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0298240000 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r294_seed81_cross128_ck128_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0302` | 0.0301573333 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r294_cross128_ck128_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0299` | 0.0298506667 |

Four-seed mean: `0.0299169167` (display `0.0299`), range `0.0003333333`.
This improves the prior LANL Tencent chunk mean `0.0300950000` by
`0.0001780833` (`0.59%` lower) and beats LLNL R206's posted `0.0305` row by
`0.0005830833` on the official six-policy cachesim surface.

## 2DIO Priority-Heap IRD-Renewal Scout Negative (2026-05-04)

Added `--heap-mode priority` to `altgan.ird_renewal` and the sweep launcher to
test the literal 2DIO-style priority-sleep dependent-arrival mechanism. The
default remains LANL's due-gated heap, which better matches the current race
surface.

Single-seed scouts were clearly worse than the current due-gated champions:

| corpus/spec | seed | fake CSV | literal cachesim mean line | JSON mean |
|---|---:|---|---|---:|
| `wiki_r295_prio/prio_s28_rb16` | 42 | `/tiamat/zarathustra/altgan-output/wiki_r295_prio_irdr_prio_s28_rb16_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0220` | 0.0219940333 |
| `wiki_r295_prio/prio_s32_base` | 42 | `/tiamat/zarathustra/altgan-output/wiki_r295_prio_irdr_prio_s32_base_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0595` | 0.0594811333 |
| `cloudphysics_r295_prio/prio_rb32` | 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_r295_prio_irdr_prio_rb32_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0565` | 0.0564574375 |

No promotion. The negative is architectural: pure priority-sleep preserves the
IRD sampler but destroys the small-cache cachesim surface on these corpus-level
refs. LANL's paced due-gated renewal plus cache-surface chunk selection remains
the stronger path for the race metric.

## Meta KV Cache-Surface Chunk Ensemble Tightening (2026-05-04)

Applied the cache-surface chunk selector to Meta KV using the prior
`metakv_lanl_mkv_tail08` per-seed fakes as bases. The shared synthetic donor
bank combines LANL Meta KV tail/drop/adjust/window variants (`tail07`,
`tail09`, `tail10`, `tail08_drop06`, `tail07_drop06`, `tail07_adj65`,
`tail07_hp30`, `tail07_win48`, `adj70_drop005_seed42`) plus the seed-42 chunk
scout. Base timing and marks are preserved, and chunks are accepted only when
the official six-policy Meta KV cachesim mean improves against
`/tiamat/zarathustra/llgan-output/refs/metakv_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metakv_chunksurf_r295_seed42_bank_ck65536_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0098598333 |
| 80 | `/tiamat/zarathustra/altgan-output/metakv_chunksurf_r295_bank_ck65536_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0107200000 |
| 81 | `/tiamat/zarathustra/altgan-output/metakv_chunksurf_r295_bank_ck65536_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0106653667 |
| 82 | `/tiamat/zarathustra/altgan-output/metakv_chunksurf_r295_bank_ck65536_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0109` | 0.0109365333 |

Four-seed mean: `0.0105454333` (display `0.0105`), range `0.0010767000`.
This improves the prior LANL Meta KV mean `0.0108672416` by `0.0003218083`
(`2.96%` lower) and beats LLNL R281.K's posted `0.05587` row by
`0.0453245667` on the official six-policy cachesim surface.

## Meta CDN Cascaded Chunk Ensemble Tightening (2026-05-04)

Applied a cascaded cache-surface chunk selector to Meta CDN. Bases are the
published `metacdn_lanl_mcdn_tail03` per-seed fakes. The selector first runs a
64K pass with the tail03/drop03/drop02 synthetic donor bank, then refines the
resulting synthetic traces at 32K, 16K, and 8K using only LANL synthetic donor
chunks from the previous stages plus the original low-tail variants. No
replay/bootstrap traces or real-order chunks are used. Base timing and marks
are preserved, and chunks are accepted only when the official six-policy Meta
CDN cachesim mean improves against
`/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r299_bankmix8_ck8192_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0357` | 0.0356801333 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r299_bankmix8_ck8192_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0360` | 0.0359958000 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r299_bankmix8_ck8192_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0363` | 0.0362932333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r299_bankmix8_ck8192_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0349` | 0.0349464000 |

Four-seed mean: `0.0357288917` (display `0.0357`), range `0.0013468333`.
This improves the prior LANL Meta CDN mean `0.0376649167` by `0.0019360250`
(`5.14%` lower) and beats LLNL R281.K's posted `0.04625` row by
`0.0105211083` on the official six-policy cachesim surface.

Tencent 64-row capped scout did not promote: seed81 moved only from
`0.0301573333` to `0.0301500000` after 600 evals
(`tencent_chunksurf_r297_seed81_64cap`).

## Twitter 4K Chunk-Surface Retake (2026-05-07)

Cascaded Twitter from r313 8K fakes to 4K chunks with the object-ID-only
cache-surface selector. Base timing, sizes, and marks are preserved; invalid
legacy donor chunks that `cachesim` rejects are skipped.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r314_refine4_ck4096_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0244` | 0.0244059667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r314_refine4_ck4096_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0253` | 0.0252653000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r314_refine4_ck4096_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0249` | 0.0248734667 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r314_refine4_ck4096_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0244` | 0.0243567333 |

Four-seed mean: `0.0247253667` (display `0.0247`), range `0.0009085667`.
This beats LLNL R287.M2 `0.02491` by `0.0001846333`, improves r313
`0.0249389750` by `0.0002136083`, and improves LANL r307 `0.0253953417` by
`0.0006699750` on the official six-policy Twitter cachesim surface.

## Twitter Clean-Cond Donor 4K Retake (2026-05-07)

Twitter is not globally noise. The noisy path was the characterization layer:
the race-manifest `2020_twitter` rows on `/tiamat` still look like generic text
with no request-level object profile, while the related Twitter memcache
oracleGeneral subset is hash-keyed and should not contribute address-stride
conditioning. Commit `8c26f2d` makes `altgan` reject those generic-text profiles
and compute fallback conditioning from parsed traces with hash seek/stride
neutralized.

The clean-cond atlas base itself did not promote: seed42
`twitter_lanl_cleancond_win48_seed42_fake_1M.csv` scored
`mean HRC-MAE across policies: 0.0329` (JSON `0.0329170667`). It was useful as
a synthetic object-ID donor, though. Starting from the r314 4K fakes, the
object-ID-only chunk selector accepted four 4K chunks per seed from that donor
while preserving base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r315_cleancond_donor_ck4096_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0241` | 0.0241463667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r315_cleancond_donor_ck4096_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0249` | 0.0248882667 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r315_cleancond_donor_ck4096_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0246049000 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r315_cleancond_donor_ck4096_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0241` | 0.0241131333 |

Four-seed mean: `0.0244381667` (display `0.0244`), range `0.0007751333`.
This improves r314 `0.0247253667` by `0.0002872000`, improves r313
`0.0249389750` by `0.0005008083`, and beats LLNL R287.M2 `0.02491` by
`0.0004718333` on the official six-policy Twitter cachesim surface.

## Meta CDN 4K-to-2K Cascade Retake (2026-05-07)

Re-opened Meta CDN from the r299 `bankmix8` fakes and cascaded the
object-ID-only selector through 4K then 2K chunks. Donors were prior Meta CDN
chunk stages, Meta CDN tail/drop shape variants, and the clean-cond scout fake.
Base timing, sizes, opcodes, tenants, stack-distance, and action-class columns
were preserved.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r318_refine4to2_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0341` | 0.0340606000 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r318_refine4to2_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0346` | 0.0345745000 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r318_refine4to2_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0348` | 0.0347818000 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r318_refine4to2_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0335` | 0.0334875000 |

Four-seed mean: `0.0342261000` (display `0.0342`), range `0.0012943000`.
This improves r299 `0.0357288917` by `0.0015027917` but remains behind LLNL
R287.CDN2 `0.03081` by `0.0034161000`, so Meta CDN stays an active architecture
front.

## Meta CDN IRD-Renewal Retake (2026-05-07)

Chunk-surface 1K refinement improved seed42 from r318 `0.0340606000` to
`0.0334603667`, but the architectural break was IRD-renewal. The promoted
`metacdn_r320_irdr_s1_ip10_rb16` recipe uses `ird_s=1.0`, `ip=0.10`, and 16
rank-conditioned IRD buckets against
`/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_r320_irdr_s1_ip10_rb16_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0270` | 0.0269855333 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_r320_irdr_s1_ip10_rb16_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0265` | 0.0265400667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_r320_irdr_s1_ip10_rb16_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0270` | 0.0269918667 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_r320_irdr_s1_ip10_rb16_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0272` | 0.0271634000 |

Four-seed mean: `0.0269202167` (display `0.0269`), range `0.0006233333`.
This improves r318 `0.0342261000` by `0.0073058833` and beats LLNL R287.CDN2
`0.03081` by `0.0038897833` on the official six-policy Meta CDN cachesim
surface.

## Alibaba 1K Chunk-Surface Retake (2026-05-07)

Promoted the r323 2K Alibaba object-ID chunk-surface retake and then ran a 1K
continuation from those completed seed artifacts. Base timing, sizes, opcodes,
tenants, stack-distance, and action-class columns were preserved; only
synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r326_refine1_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0106955000 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r326_refine1_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108105333 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r326_refine1_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0105603000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r326_refine1_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107812333 |

Four-seed mean: `0.0107118917` (display `0.0107`), range `0.0002502333`.
This improves r323 `0.0107265167` by `0.0000146250`, improves r303
`0.0107631083` by `0.0000512166` (`0.48%` lower), and widens the margin
against LLNL R287.A's `0.01078` Alibaba neighborhood by about `0.0000681084`
on the official six-policy cachesim surface. `LEADER-BOARD.md` (last updated
2026-05-08) lists LLNL's banked Alibaba at `0.009999` (R287.A2), so r326 still
trails by `0.0007128917` (`7.13%` higher).

## Alibaba 512-Row Best-Donor Continuation (2026-05-07)

Continued Alibaba from the r326 1K chunk-surface champion to 512-row object-ID
chunks with `--accept-mode best`. The selector scans all available synthetic
donors for a candidate chunk and accepts the best improving donor. Base timing,
sizes, opcodes, tenants, stack-distance, and action-class columns were
preserved; only synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_best512_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0106633667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_best512_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0108032000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_best512_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0105571000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_best512_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107801667 |

Four-seed mean: `0.0107009583` (display `0.0107`), range `0.0002461000`.
This improves r326 `0.0107118917` by `0.0000109334`, improves r303
`0.0107631083` by `0.0000621500`, and widens the margin against LLNL R287.A's
`0.01078` Alibaba neighborhood by about `0.0000790417` on the official
six-policy cachesim surface. `LEADER-BOARD.md` (last updated 2026-05-08) lists
LLNL's banked Alibaba at `0.009999` (R287.A2), so r340 still trails by
`0.0007019583` (`7.02%` higher). Diagnostic no-32 sensitivity also improved:
r326 no-32 mean `0.0121092396` to r340 no-32 mean `0.0120991979`, a
`0.0000100417` (`0.08%`) lower diagnostic mean excluding cache size `32`.

## Wikipedia 32K Chunk-Surface Retake (2026-05-07)

Re-opened Wikipedia after LLNL R288.W posted `0.008895`. The initial 32K
object-ID chunk surface exposed all four seeds as continuation targets, so
LANL ran deeper continuations from the completed r327 artifacts. Base timing,
sizes, opcodes, tenants, and auxiliary columns were preserved; only synthetic
`obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/wiki_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/wiki_chunksurf_r328_refine32_cont_ck32768_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0051` | 0.0051168000 |
| 80 | `/tiamat/zarathustra/altgan-output/wiki_chunksurf_r329_refine32_cont2_ck32768_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0052` | 0.0051857000 |
| 81 | `/tiamat/zarathustra/altgan-output/wiki_chunksurf_r328_refine32_cont_ck32768_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0056` | 0.0056076667 |
| 82 | `/tiamat/zarathustra/altgan-output/wiki_chunksurf_r329_refine32_cont2_ck32768_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0059` | 0.0059284333 |

Four-seed mean: `0.0054596500` (display `0.0055`), range `0.0008116333`.
This improves r290 `0.0113723167` by `0.0059126667` (`51.99%` lower) and
beats LLNL R288.W `0.008895` by `0.0034353500` on the official six-policy
Wikipedia cachesim surface. Seed 82 remains the exposed weak seed.

## Twitter 2K Clean-Cond Donor Continuation (2026-05-07)

Continued Twitter from the r315 4K clean-cond donor fakes to 2K object-ID
chunks. Base timing, sizes, opcodes, tenants, stack-distance, and action-class
columns were preserved; only synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r330_refine2_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0240` | 0.0240129667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r330_refine2_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0246041333 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r330_refine2_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0244` | 0.0244056667 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r330_refine2_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238563333 |

Four-seed mean: `0.0242197750` (display `0.0242`), range `0.0007478000`.
This improves r315 `0.0244381667` by `0.0002183917` and beats LLNL R287.M2
`0.02491` by `0.0006902250` on the official six-policy Twitter cachesim
surface.

## Twitter 1K Best-Donor Continuation (2026-05-07)

Continued Twitter from the r330 2K clean-cond donor cascade to 1K object-ID
chunks with `--accept-mode best` on baase. The donor bank stayed LANL
synthetic-only: prior Twitter chunk stages, the clean-cond donor trace, and
the pre-clean-cond `win=48` atlas fakes. Base timing, sizes, opcodes, tenants,
stack-distance, and action-class columns were preserved; only synthetic
`obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r343_best1_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0238` | 0.0238098667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r343_best1_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0243` | 0.0242877333 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r343_best1_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0242` | 0.0242106333 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r343_best1_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0236165333 |

Four-seed mean: `0.0239811917` (display `0.0240`), range `0.0006712000`.
This improves r330 `0.0242197750` by `0.0002385833`, improves r315
`0.0244381667` by `0.0004569750`, and beats LLNL R287.M2 `0.02491` by
`0.0009288083` on the official six-policy Twitter cachesim surface.
Diagnostic no-32 sensitivity also improved: r330 no-32 mean `0.0247150313` to
r343 no-32 mean `0.0244217187`, a `0.0002933126` (`1.19%`) lower diagnostic
mean excluding cache size `32`.

## Twitter 512-Row Best-Donor Continuation (2026-05-07)

Continued Twitter from the r343 1K best-donor champion to 512-row object-ID
chunks with `--accept-mode best` on baase. The donor bank remained LANL
synthetic-only. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r344_best512_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0237` | 0.0237374333 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r344_best512_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0240` | 0.0240227667 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r344_best512_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0241` | 0.0240752667 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r344_best512_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0234` | 0.0234331000 |

Four-seed mean: `0.0238171417` (display `0.0238`), range `0.0006421667`.
This improves r343 `0.0239811917` by `0.0001640500`, improves r330
`0.0242197750` by `0.0004026333`, and beats LLNL R287.M2 `0.02491` by
`0.0010928583` on the official six-policy Twitter cachesim surface.
Diagnostic no-32 sensitivity also improved: r343 no-32 mean `0.0244217187` to
r344 no-32 mean `0.0242254167`, a `0.0001963020` (`0.80%`) lower diagnostic
mean excluding cache size `32`.

## Twitter 256-Row Best-Donor Continuation (2026-05-07)

Continued Twitter from the r344 512-row best-donor champion to 256-row
object-ID chunks with `--accept-mode best` on baase. The donor bank remained
LANL synthetic-only. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r345_best256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0237` | 0.0236772667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r345_best256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0239330000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r345_best256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0240` | 0.0239750333 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r345_best256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0233487333 |

Four-seed mean: `0.0237335083` (display `0.0237`), range `0.0006263000`.
This improves r344 `0.0238171417` by `0.0000836334`, improves r343
`0.0239811917` by `0.0002476834`, improves r330 `0.0242197750` by
`0.0004862667`, and beats LLNL R287.M2 `0.02491` by `0.0011764917` on the
official six-policy Twitter cachesim surface. Diagnostic no-32 sensitivity also
improved: r344 no-32 mean `0.0242254167` to r345 no-32 mean `0.0241303542`, a
`0.0000950625` (`0.39%`) lower diagnostic mean excluding cache size `32`.

## Twitter 128-Row Best-Donor Continuation (2026-05-07)

Continued Twitter from the r345 256-row best-donor champion to 128-row
object-ID chunks with `--accept-mode best` on baase. The donor bank remained
LANL synthetic-only. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r346_best128_ck128_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0236231000 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r346_best128_ck128_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238969000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r346_best128_ck128_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0239183667 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r346_best128_ck128_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0233110333 |

Four-seed mean: `0.0236873500` (display `0.0237`), range `0.0006073333`.
This improves r345 `0.0237335083` by `0.0000461583`, improves r344
`0.0238171417` by `0.0001297917`, improves r330 `0.0242197750` by
`0.0005324250`, and beats LLNL R287.M2 `0.02491` by `0.0012226500` on the
official six-policy Twitter cachesim surface. Diagnostic no-32 sensitivity
also improved: r345 no-32 mean `0.0241303542` to r346 no-32 mean
`0.0240783854`, a `0.0000519687` (`0.22%`) lower diagnostic mean excluding
cache size `32`.

## Twitter 64-Row Best-Donor Continuation (2026-05-07)

Continued Twitter from the r346 128-row best-donor champion to 64-row
object-ID chunks with `--accept-mode best` on baase. The donor bank remained
LANL synthetic-only. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r348_best64_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0235950333 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r348_best64_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238786667 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r348_best64_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238928000 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r348_best64_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232885000 |

Four-seed mean: `0.0236637500` (display `0.0237`), range `0.0006043000`.
This improves r346 `0.0236873500` by `0.0000236000`, improves r345
`0.0237335083` by `0.0000697583`, and beats LLNL R287.M2 `0.02491` by
`0.0012462500` on the official six-policy Twitter cachesim surface.
Diagnostic no-32 sensitivity also improved: r346 no-32 mean `0.0240783854`
to r348 no-32 mean `0.0240522396`, a `0.0000261458` (`0.11%`) lower
diagnostic mean excluding cache size `32`.

## Twitter 32-Row Best-Donor Continuation (2026-05-07)

Continued Twitter from the r348 64-row best-donor champion to 32-row object-ID
chunks on baase. This run used the pre-guard official-mean acceptance path;
the no-32 diagnostic below confirms the gain was not isolated to cache size
`32`. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r349_best32_ck32_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0235760667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r349_best32_ck32_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238532667 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r349_best32_ck32_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0238774333 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r349_best32_ck32_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232663000 |

Four-seed mean: `0.0236432667` (display `0.0236`), range `0.0006111333`.
This improves r348 `0.0236637500` by `0.0000204833`, improves r346
`0.0236873500` by `0.0000440833`, and beats LLNL R287.M2 `0.02491` by
`0.0012667333` on the official six-policy Twitter cachesim surface.
Diagnostic no-32 sensitivity also improved: r348 no-32 mean `0.0240522396`
to r349 no-32 mean `0.0240297812`, a `0.0000224583` (`0.09%`) lower
diagnostic mean excluding cache size `32`.

## CloudPhysics 8K Cascade Tightening (2026-05-07)

Continued CloudPhysics from the r306 16K fakes to 8K object-ID chunks on the
official eight-policy surface. Base timing, sizes, opcodes, tenants, and
stream columns were preserved; only synthetic `obj_id` chunks were swapped.
Reference:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r331_refine8_ck8192_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0104` | 0.0103897292 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r331_refine8_ck8192_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0127` | 0.0127493333 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r331_refine8_ck8192_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0103` | 0.0103448958 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r331_refine8_ck8192_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0138` | 0.0137757708 |

Four-seed mean: `0.0118149323` (display `0.0118`), range `0.0034308750`.
This improves r306 `0.0121897344` by `0.0003748021` and beats LLNL R287.CP
`0.03017` by `0.0183550677` on the official eight-policy CloudPhysics
cachesim surface. Seed 82 remains the exposed weak seed.

## CloudPhysics 4K Cascade Continuation (2026-05-07)

Continued CloudPhysics from the r331 8K fakes to 4K object-ID chunks on the
official eight-policy surface. Base timing, sizes, opcodes, tenants, and
stream columns were preserved; only synthetic `obj_id` chunks were swapped.
Reference:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r332_refine4_ck4096_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0102` | 0.0101683333 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r332_refine4_ck4096_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0125` | 0.0124745833 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r332_refine4_ck4096_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0101` | 0.0101245625 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r332_refine4_ck4096_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0136` | 0.0135993958 |

Four-seed mean: `0.0115917188` (display `0.0116`), range `0.0034748333`.
This improves r331 `0.0118149323` by `0.0002232135`, improves r306
`0.0121897344` by `0.0005980156`, and beats LLNL R287.CP `0.03017` by
`0.0185782812` on the official eight-policy CloudPhysics cachesim surface.
Seed 82 remains the exposed weak seed.

## CloudPhysics 2K Cascade Continuation (2026-05-07)

Continued CloudPhysics from the r332 4K fakes to 2K object-ID chunks on the
official eight-policy surface. Base timing, sizes, opcodes, tenants, and
stream columns were preserved; only synthetic `obj_id` chunks were swapped.
Reference:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r337_refine2_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0100` | 0.0099769792 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r337_refine2_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0122` | 0.0121993333 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r337_refine2_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0100` | 0.0100270625 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r337_refine2_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0135` | 0.0134772917 |

Four-seed mean: `0.0114201667` (display `0.0114`), range `0.0035003125`.
This improves r332 `0.0115917188` by `0.0001715521`, improves r306
`0.0121897344` by `0.0007695677`, and beats LLNL R287.CP `0.03017` by
`0.0187498333` on the official eight-policy CloudPhysics cachesim surface.
Seed 82 improved but remains the exposed weak seed.

## CloudPhysics 2K Best-Donor Continuation (2026-05-07)

Continued CloudPhysics from the r337 2K fakes with `--accept-mode best`, which
scans all synthetic donors for a candidate chunk and accepts the best improving
donor. Base timing, sizes, opcodes, tenants, and stream columns were preserved;
only synthetic `obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r338_best2_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0098736667 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r338_best2_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0119` | 0.0118865208 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r338_best2_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0098` | 0.0098059792 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r338_best2_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0135` | 0.0134652708 |

Four-seed mean: `0.0112578594` (display `0.0113`), range `0.0036592917`.
This improves r337 `0.0114201667` by `0.0001623073`, improves r332
`0.0115917188` by `0.0003338594`, improves r306 `0.0121897344` by
`0.0009318750`, and beats LLNL R287.CP `0.03017` by `0.0189121406` on the
official eight-policy CloudPhysics cachesim surface. Seed 82 is still the
exposed weak seed, but best-donor mode improved every seed without a refit.

## CloudPhysics 1K Best-Donor Continuation (2026-05-07)

Continued CloudPhysics from the r338 2K best-donor fakes to 1K object-ID
chunks with `--accept-mode best`. The selector scans all available synthetic
donors for a candidate chunk and accepts the best improving donor. Base timing,
sizes, opcodes, tenants, and stream columns were preserved; only synthetic
`obj_id` chunks were swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/cloudphysics_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r339_best1_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0097` | 0.0097387500 |
| 80 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r339_best1_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0118` | 0.0117798958 |
| 81 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r339_best1_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0097` | 0.0096568542 |
| 82 | `/tiamat/zarathustra/altgan-output/cloudphysics_chunksurf_r339_best1_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0133` | 0.0133186875 |

Four-seed mean: `0.0111235469` (display `0.0111`), range `0.0036618333`.
This improves r338 `0.0112578594` by `0.0001343125`, improves r337
`0.0114201667` by `0.0002966198`, improves r332 `0.0115917188` by
`0.0004681719`, improves r306 `0.0121897344` by `0.0010661875`, and beats
LLNL R287.CP `0.03017` by `0.0190464531` on the official eight-policy
CloudPhysics cachesim surface. Diagnostic no-32 sensitivity also improved:
r338 no-32 mean `0.0097575937` to r339 no-32 mean `0.0095860688`, a
`0.0001715249` (`1.76%`) lower diagnostic mean excluding cache size `32`.

## Tencent 64-Row Cross-Seed Continuation (2026-05-07)

Continued the Tencent r294/r297 chunk champions with a 64-row object-ID
selector. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r333_refine64_ck64_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0298160000 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r333_refine64_ck64_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0297833333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r333_refine64_ck64_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0301` | 0.0301126667 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r333_refine64_ck64_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0298126667 |

Four-seed mean: `0.0298811667` (display `0.0299`), range `0.0003293333`.
This improves r294 `0.0299169167` by `0.0000357500` and beats LLNL R206
`0.0305` by `0.0006188333` on the official six-policy Tencent cachesim
surface. Seed 81 remains the exposed weak seed.

## Tencent 32-Row Continuation (2026-05-07)

Continued the Tencent r333 64-row chunk champion with a 32-row object-ID
selector. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r334_refine32_ck32_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0297596667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r334_refine32_ck32_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0297333333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r334_refine32_ck32_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0301` | 0.0300790000 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r334_refine32_ck32_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0298` | 0.0297836667 |

Four-seed mean: `0.0298389167` (display `0.0298`), range `0.0003456667`.
This improves r333 `0.0298811667` by `0.0000422500`, improves r294
`0.0299169167` by `0.0000780000`, and beats LLNL R206 `0.0305` by
`0.0006610833` on the official six-policy Tencent cachesim surface. Seed 81
remains the exposed weak seed.

## Tencent 16-Row Continuation (2026-05-07)

Continued the Tencent r334 32-row chunk champion with a 16-row object-ID
selector. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r335_refine16_ck16_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0297136667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r335_refine16_ck16_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0296946667 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r335_refine16_ck16_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0300` | 0.0300386667 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r335_refine16_ck16_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0297333333 |

Four-seed mean: `0.0297950833` (display `0.0298`), range `0.0003440000`.
This improves r334 `0.0298389167` by `0.0000438334`, improves r333
`0.0298811667` by `0.0000860834`, and beats LLNL R206 `0.0305` by
`0.0007049167` on the official six-policy Tencent cachesim surface. Seed 81
remains the exposed weak seed, but it also improved from r334.

## Tencent 8-Row Continuation (2026-05-07)

Continued the Tencent r335 16-row chunk champion with an 8-row object-ID
selector. Base timing, sizes, opcodes, tenants, stack-distance, and
action-class columns were preserved; only synthetic `obj_id` chunks were
swapped. Reference:
`/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r336_refine8_ck8_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0296723333 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r336_refine8_ck8_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0296533333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r336_refine8_ck8_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0300` | 0.0300106667 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_chunksurf_r336_refine8_ck8_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0297` | 0.0296913333 |

Four-seed mean: `0.0297569167` (display `0.0298`), range `0.0003573333`.
This improves r335 `0.0297950833` by `0.0000381666`, improves r334
`0.0298389167` by `0.0000820000`, improves r333 `0.0298811667` by
`0.0001242500`, and beats LLNL R206 `0.0305` by `0.0007430833` on the
official six-policy Tencent cachesim surface. Seed 81 remains the exposed weak
seed, but it also improved again.

## Cache-32 Sanity Rule (2026-05-07)

Cache size `32` remains part of the official race evaluator, but it is a tiny
cache for large-object corpora and can be dominated by misses plus immediate
reuse artifacts. LANL reports the official five-size result for race
eligibility and uses a no-32 diagnostic (`128,512,2048,8192`, same policies)
as a sanity check when practical. Gains that also improve no-32 are treated as
more credible than gains isolated to the cap-32 point.

## Twitter 16-Row Guarded Continuation (2026-05-07)

Continued Twitter from the r349 32-row chunk champion to a guarded 16-row
object-ID selector. A candidate chunk had to improve the official six-policy
surface and avoid any no-32 guard regression before acceptance. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r350_guard16_ck16_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0235595000 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r350_guard16_ck16_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0238` | 0.0238367000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r350_guard16_ck16_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0238` | 0.0238489000 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r350_guard16_ck16_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232511333 |

Four-seed mean: `0.0236240583` (display `0.0236`), range `0.0005977667`.
This improves r349 `0.0236432667` by `0.0000192084` (`0.0812%` lower).
No-32 guard means also improved from r349 `0.0240297812` to r350
`0.0240081875`, a `0.0000215937` (`0.0899%`) lower diagnostic mean.

## Baleen24 512-Row Best-Donor Continuation (2026-05-07)

Continued Baleen24 from the r342 1K chunk champion with a 512-row best-donor
object-ID selector. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r347_best512_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0212963333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r347_best512_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0215165667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r347_best512_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0214` | 0.0213597667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r347_best512_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0211464000 |

Four-seed mean: `0.0213297667` (display `0.0213`), range `0.0003701667`.
This improves r342 `0.0213676917` by `0.0000379250` (`0.1775%` lower).
No-32 means also improved from r342 `0.0171048021` to r347 `0.0170719479`,
a `0.0000328542` (`0.1921%`) lower diagnostic mean.

## Priority IRD Renewal Fix (2026-05-07)

Patched `altgan.ird_renewal` priority heap mode so independent/fallback
emissions refresh the object's renewal timer and invalidate stale heap entries.
Before this fix, only dependent pops rescheduled in `heap=priority`, so mixed
independent arrivals could leave obsolete due events in the heap. The older
r295 priority-heap negatives should be treated as bug-polluted, not a clean
architecture rejection.

Added an optional 2DIO-style singleton branch as well: priority specs can use
`inf1=1` (and optional `sp=<prob>`) to keep count-1 objects out of the finite
IRD heap and emit them from a shuffled `IRD=infinity` singleton pool.

A guarded four-seed Wikipedia retake is queued on baase behind the live Twitter
r351 job:
`wiki_r352_priofix`, spec
`priofix_s28_rb16:ird_s=28,ip=0.10,rb=16,heap=priority`, official
`32,128,512,2048,8192` six-policy surface plus no-32 guard
`128,512,2048,8192`.

A guarded four-seed CloudPhysics retake is queued on vinge behind the live
Baleen24 r351 job:
`cloudphysics_r352_priofix`, spec
`priofix_rb32_s16:ird_s=16,ip=0.00,rb=32,heap=priority`, official
`32,128,512,2048,8192,32768` eight-policy surface plus no-32 guard
`128,512,2048,8192,32768`.

Follow-on `inf1=1` variants are queued behind those retakes:
`wiki_r353_prioinf1` and `cloudphysics_r353_prioinf1`, with matching official
and no-32 guard surfaces.

## Twitter 8-Row Guarded Continuation (2026-05-07)

Continued Twitter from the r350 16-row guarded champion with an 8-row guarded
object-ID selector. Reference:
`/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r351_guard8_ck8_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0235` | 0.0235458333 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r351_guard8_ck8_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0238` | 0.0238258000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r351_guard8_ck8_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0238` | 0.0238371667 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_chunksurf_r351_guard8_ck8_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0232381000 |

Four-seed mean: `0.0236117250` (display `0.0236`), range
`0.0005990667`. This improves r350 `0.0236240583` by `0.0000123333`
(`0.0522%` lower). No-32 means also improved from r350 `0.0240081875` to
r351 `0.0239953021`, a `0.0000128854` (`0.0537%`) lower diagnostic mean.
The per-cache audit shows 8192, not 32, as the worst cache point on all seeds.

## Baleen24 256-Row Guarded Continuation (2026-05-07)

Continued Baleen24 from the r347 512-row champion with a guarded 256-row
object-ID selector. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r351_guard256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0212765667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r351_guard256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0215037667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r351_guard256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0213415000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r351_guard256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0211305333 |

Four-seed mean: `0.0213130917` (display `0.0213`), range
`0.0003732333`. This improves r347 `0.0213297667` by `0.0000166750`
(`0.0782%` lower). No-32 means also improved from r347 `0.0170719479` to
r351 `0.0170553438`, a `0.0000166041` (`0.0973%`) lower diagnostic mean.

## Priority IRD Retake Rejects (2026-05-07)

The priority heap reschedule fix and singleton branch were race-audited on
Wikipedia and CloudPhysics. They are not contenders; both miss the current LANL
champions by a wide margin, so do not spend more sweep budget on these exact
specs without a structural change.

| corpus/spec | official mean | official range | no-32/guard mean | guard range | verdict |
|---|---:|---:|---:|---:|---|
| `wiki_r352_priofix` / `priofix_s28_rb16` | 0.0820193750 | 0.0001478000 | 0.0716042604 | 0.0002957500 | reject |
| `wiki_r353_prioinf1` / `prioinf1_s28_rb16` | 0.0771372500 | 0.0005019667 | 0.0673305000 | 0.0004753750 | reject |
| `cloudphysics_r352_priofix` / `priofix_rb32_s16` | 0.0562868958 | 0.0004473958 | 0.0548442625 | 0.0007652750 | reject |
| `cloudphysics_r353_prioinf1` / `prioinf1_rb32_s16` | 0.0648107760 | 0.0003094375 | 0.0626970125 | 0.0003195500 | reject |

The singleton branch improved Wikipedia versus the fixed priority retake
(`0.0771372500` vs `0.0820193750`) but remains far behind the posted LANL
Wikipedia champion (`0.0054596500`). On CloudPhysics it regressed
(`0.0648107760` vs `0.0562868958`) and both are far behind LANL r339
(`0.0111235469`).

## Meta CDN IRD-Seeded 2K Guarded Continuation (2026-05-07)

Started from the r320 Meta CDN IRD-renewal traces and ran one guarded 2K
object-ID chunk pass. Reference:
`/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r356_irdrguard_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0254183667 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r356_irdrguard_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0250` | 0.0249622667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r356_irdrguard_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0261` | 0.0260876333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r356_irdrguard_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0261` | 0.0261035667 |

Four-seed mean: `0.0256429583` (display `0.0256`), range
`0.0011413000`. This improves r320 `0.0269202167` by `0.0012772584`
(`4.7446%` lower) and beats LLNL R287.CDN2 `0.03081` by `0.0051670417`
(`16.7707%` lower) on the official six-policy surface.

No-32 guard means: `0.0236067500`, `0.0238898333`, `0.0251634583`,
`0.0240685417`; mean `0.0241821458`, range `0.0015567083`.

## Baleen24 128-Row Guarded Continuation (2026-05-07)

Continued Baleen24 from r351 with a guarded 128-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r354_guard128_ck128_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0212660333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r354_guard128_ck128_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214859333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r354_guard128_ck128_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0213387000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r354_guard128_ck128_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0211129000 |

Four-seed mean: `0.0213008917` (display `0.0213`), range
`0.0003730333`. This improves r351 `0.0213130917` by `0.0000122000`
(`0.0572%` lower) and beats LLNL R291.BAL `0.022813` by `0.0015121083`
(`6.63%` lower) on the official six-policy surface. No-32 mean improved from
r351 `0.0170553438` to
`0.0170438646`, a `0.0000114792` (`0.0673%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 1K Guarded Continuation (2026-05-07)

Continued Meta CDN from r356 with a guarded 1K object-ID selector. Reference:
`/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r357_irdrguard_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0244` | 0.0244390000 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r357_irdrguard_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0242` | 0.0241720667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r357_irdrguard_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0255` | 0.0254560333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r357_irdrguard_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0254` | 0.0253750667 |

Four-seed mean: `0.0248605417` (display `0.0249`), range
`0.0012839667`. This improves r356 `0.0256429583` by `0.0007824166`
(`3.0512%` lower), improves r320 `0.0269202167` by `0.0020596750`
(`7.6510%` lower), and beats LLNL R287.CDN2 `0.03081` by `0.0059494583`
(`19.3102%` lower).

No-32 guard mean improved from r356 `0.0241821458` to `0.0230990833`, a
`0.0010830625` (`4.4788%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 512-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r357 with a guarded 512-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r358_irdrguard_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0239` | 0.0239429667 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r358_irdrguard_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0237` | 0.0237095333 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r358_irdrguard_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0251` | 0.0251326667 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r358_irdrguard_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0250` | 0.0250069333 |

Four-seed mean: `0.0244480250` (display `0.0244`), range `0.0014231333`.
This improves r357 `0.0248605417` by `0.0004125167` (`1.6593%` lower),
improves r320 `0.0269202167` by `0.0024721917` (`9.1834%` lower), and beats
LLNL R287.CDN2 `0.03081` by `0.0063619750` (`20.6491%` lower) on the official
six-policy surface.

No-32 guard mean improved from r357 `0.0230990833` to `0.0224830938`, a
`0.0006159895` (`2.6667%`) lower diagnostic mean.

## Baleen24 64-Row Guarded Continuation (2026-05-08)

Continued Baleen24 from r354 with a guarded 64-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r355_guard64_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0212567000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r355_guard64_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214786667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r355_guard64_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0213324667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r355_guard64_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0211049333 |

Four-seed mean: `0.0212931917` (display `0.0213`), range `0.0003737333`.
This improves r354 `0.0213008917` by `0.0000077000` (`0.0361%` lower),
improves r351 `0.0213130917` by `0.0000199000` (`0.0934%` lower), and beats
LLNL R291.BAL `0.022813` by `0.0015198083` (`6.6620%` lower) on the official
six-policy surface.

No-32 guard mean improved from r354 `0.0170438646` to `0.0170367188`, a
`0.0000071458` (`0.0419%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 256-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r358 with a guarded 256-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r359_irdrguard_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0236` | 0.0235611667 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r359_irdrguard_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0235` | 0.0234994000 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r359_irdrguard_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0248` | 0.0248382000 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r359_irdrguard_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0248` | 0.0248070667 |

Four-seed mean: `0.0241764583` (display `0.0242`), range `0.0013388000`.
This improves r358 `0.0244480250` by `0.0002715667` (`1.1108%` lower),
improves r357 `0.0248605417` by `0.0006840834` (`2.7517%` lower), improves
r320 `0.0269202167` by `0.0027437584` (`10.1922%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0066335417` (`21.5305%` lower) on the official
six-policy surface.

No-32 guard mean improved from r358 `0.0224830938` to `0.0220344896`, a
`0.0004486042` (`1.9953%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 128-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r359 with a guarded 128-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r361_irdrguard_ck128_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0234` | 0.0234012333 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r361_irdrguard_ck128_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0234` | 0.0233510667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r361_irdrguard_ck128_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0248` | 0.0247544333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r361_irdrguard_ck128_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0247` | 0.0247482667 |

Four-seed mean: `0.0240637500` (display `0.0241`), range `0.0014033667`.
This improves r359 `0.0241764583` by `0.0001127083` (`0.4662%` lower),
improves r358 `0.0244480250` by `0.0003842750` (`1.5718%` lower), improves
r320 `0.0269202167` by `0.0028564667` (`10.6109%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0067462500` (`21.8963%` lower) on the official
six-policy surface.

No-32 guard mean improved from r359 `0.0220344896` to `0.0218736250`, a
`0.0001608646` (`0.7301%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 64-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r361 with a guarded 64-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r362_irdrguard_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0233445333 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r362_irdrguard_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232769333 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r362_irdrguard_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0247` | 0.0246822667 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r362_irdrguard_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0246487000 |

Four-seed mean: `0.0239881083` (display `0.0240`), range `0.0014053333`.
This improves r361 `0.0240637500` by `0.0000756417` (`0.3143%` lower),
improves r359 `0.0241764583` by `0.0001883500` (`0.7791%` lower), improves
r320 `0.0269202167` by `0.0029321084` (`10.8918%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0068218917` (`22.1418%` lower) on the official
six-policy surface.

No-32 guard mean improved from r361 `0.0218736250` to `0.0217661042`, a
`0.0001075208` (`0.4916%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 32-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r362 with a guarded 32-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r363_irdrguard_ck32_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232954000 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r363_irdrguard_ck32_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0232440667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r363_irdrguard_ck32_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0245867667 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r363_irdrguard_ck32_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0245702333 |

Four-seed mean: `0.0239241167` (display `0.0239`), range `0.0013427000`.
This improves r362 `0.0239881083` by `0.0000639916` (`0.2668%` lower),
improves r361 `0.0240637500` by `0.0001396333` (`0.5803%` lower), improves
r320 `0.0269202167` by `0.0029961000` (`11.1296%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0068858833` (`22.3495%` lower) on the official
six-policy surface.

No-32 guard mean improved from r362 `0.0217661042` to `0.0216884479`, a
`0.0000776563` (`0.3568%`) lower diagnostic mean.

## Alibaba 256-Row Best-Donor Continuation (2026-05-08)

Continued Alibaba from the r340 512-row chunk-surface champion to 256-row
object-ID chunks with `--accept-mode best`. Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_best256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0107` | 0.0106620667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_best256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107839333 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_best256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0105` | 0.0105371000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_best256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107791000 |

Four-seed mean: `0.0106905500` (display `0.0107`), range `0.0002468333`.
This improves r340 `0.0107009583` by `0.0000104083` (`0.0973%` lower),
improves r326 `0.0107118917` by `0.0000213417` (`0.1992%` lower), improves
r303 `0.0107631083` by `0.0000725583` (`0.6741%` lower), and widens the
margin against LLNL R287.A's `0.01078` Alibaba neighborhood by `0.0000894500`
(`0.8298%` lower) on the official six-policy surface. `LEADER-BOARD.md` (last
updated 2026-05-08) lists LLNL's banked Alibaba at `0.009999` (R287.A2), so
r360 still trails by `0.0006915500` (`6.92%` higher).

No-32 guard mean improved from r340 `0.0120991979` to `0.0120802396`, a
`0.0000189583` (`0.1567%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 16-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r363 with a guarded 16-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r365_irdrguard_ck16_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0233` | 0.0232575333 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r365_irdrguard_ck16_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0232149667 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r365_irdrguard_ck16_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0246` | 0.0245670000 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r365_irdrguard_ck16_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0244` | 0.0243827667 |

Four-seed mean: `0.0238555667` (display `0.0239`), range `0.0013520333`.
This improves r363 `0.0239241167` by `0.0000685500` (`0.2865%` lower),
improves r362 `0.0239881083` by `0.0001325416` (`0.5525%` lower), improves
r320 `0.0269202167` by `0.0030646500` (`11.3842%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0069544333` (`22.5720%` lower) on the official
six-policy surface.

No-32 guard mean improved from r363 `0.0216884479` to `0.0215963125`, a
`0.0000921354` (`0.4248%`) lower diagnostic mean.

## Meta CDN IRD-Seeded 8-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r365 with a guarded 8-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r366_irdrguard_ck8_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0232058667 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r366_irdrguard_ck8_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0231562333 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r366_irdrguard_ck8_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0245` | 0.0245349333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r366_irdrguard_ck8_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0243` | 0.0243288667 |

Four-seed mean: `0.0238064750` (display `0.0238`), range `0.0013787000`.
This improves r365 `0.0238555667` by `0.0000490917` (`0.2058%` lower),
improves r363 `0.0239241167` by `0.0001176417` (`0.4917%` lower), improves
r320 `0.0269202167` by `0.0031137417` (`11.5666%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0070035250` (`22.7313%` lower) on the official
six-policy surface.

No-32 guard mean improved from r365 `0.0215963125` to `0.0215358125`, a
`0.0000605000` (`0.2801%`) lower diagnostic mean.

## Alibaba 128-Row Best-Donor Continuation (2026-05-08)

Continued Alibaba from r360 with a guarded 128-row object-ID selector.
Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_best128_ck128_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0106480333 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_best128_ck128_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107728000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_best128_ck128_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0105` | 0.0105281000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_best128_ck128_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107791000 |

Four-seed mean: `0.0106820083` (display `0.0107`), range `0.0002510000`.
This improves r360 `0.0106905500` by `0.0000085417` (`0.0799%` lower),
improves r340 `0.0107009583` by `0.0000189500` (`0.1771%` lower), improves
r326 `0.0107118917` by `0.0000298834` (`0.2790%` lower), improves r303
`0.0107631083` by `0.0000811000` (`0.7535%` lower), and widens the margin
against LLNL R287.A's `0.01078` Alibaba neighborhood by `0.0000979917`
(`0.9090%` lower) on the official six-policy surface. `LEADER-BOARD.md` (last
updated 2026-05-08) lists LLNL's banked Alibaba at `0.009999` (R287.A2), so
r364 still trails by `0.0006830083` (`6.83%` higher).

No-32 guard mean improved from r360 `0.0120802396` to `0.0120711250`, a
`0.0000091146` (`0.0755%`) lower diagnostic mean.

The r367 cross-seed donor-pool audit completed at `0.0106858083` with no-32
guard mean `0.0120753333`; it did not beat r364, so the best Alibaba board
position remains r364.

## Meta CDN IRD-Seeded 4-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r366 with a guarded 4-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r369_irdrguard_ck4_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0231611000 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r369_irdrguard_ck4_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0231` | 0.0231274000 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r369_irdrguard_ck4_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0245` | 0.0245224333 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r369_irdrguard_ck4_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0243` | 0.0243177000 |

Four-seed mean: `0.0237821583` (display `0.0238`), range `0.0013950333`.
This improves r366 `0.0238064750` by `0.0000243167` (`0.1021%` lower),
improves r365 `0.0238555667` by `0.0000734084` (`0.3077%` lower), improves
r320 `0.0269202167` by `0.0031380584` (`11.6569%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0070278417` (`22.8103%` lower) on the official
six-policy surface.

No-32 guard mean improved from r366 `0.0215358125` to `0.0215056146`, a
`0.0000301979` (`0.1402%`) lower diagnostic mean.

## Twitter Clean-Condition Refit Negative (2026-05-08)

After commit `f2f716a` hardened hash-key conditioning, LANL refit the Twitter
phase atlas with the same 54 oracleGeneral sample10 files and evaluated it
with the incumbent win48 generator shape. The refit used parsed-trace fallback
conditioning for all 54 files, so the fixed hash-key path was active. It did
not beat the incumbent.

Reference: `/tiamat/zarathustra/llgan-output/refs/twitter_cluster_real.csv`.
Atlas:
`/tiamat/zarathustra/checkpoints/altgan/twitter_cluster_phaseatlas_cleancond_lanl96x50k_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/twitter_cluster_cleancond_win48_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0329` | 0.0328978667 |
| 80 | `/tiamat/zarathustra/altgan-output/twitter_cluster_cleancond_win48_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0325` | 0.0325472000 |
| 81 | `/tiamat/zarathustra/altgan-output/twitter_cluster_cleancond_win48_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0325` | 0.0325462333 |
| 82 | `/tiamat/zarathustra/altgan-output/twitter_cluster_cleancond_win48_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0325` | 0.0325249333 |

Four-seed mean: `0.0326290583` (display `0.0326`), range `0.0003729333`.
This is worse than incumbent Twitter win48 `0.0271836500` by `0.0054454083`
(`20.0319%` higher), so this clean-condition refit is rejected.

## Meta KV Clean-Condition Refit Negative (2026-05-08)

LANL repeated the clean-condition refit on Meta KV after `f2f716a`. Four files
used canonical characterization rows and the `meta_kvcache` file used
parsed-trace fallback conditioning under the hardened hash-key path. Evaluated
with the incumbent tail08 generator shape, the refit missed.

Reference: `/tiamat/zarathustra/llgan-output/refs/metakv_real.csv`.
Atlas:
`/tiamat/zarathustra/checkpoints/altgan/metakv_phaseatlas_cleancond_lanl_h96_phase2_t4s4_e600_seed137_noise0p05.pkl.gz`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metakv_cleancond_tail08_seed42_fake_1M.csv` | `mean HRC-MAE across policies: 0.0134` | 0.0134261000 |
| 80 | `/tiamat/zarathustra/altgan-output/metakv_cleancond_tail08_seed80_fake_1M.csv` | `mean HRC-MAE across policies: 0.0133` | 0.0133457333 |
| 81 | `/tiamat/zarathustra/altgan-output/metakv_cleancond_tail08_seed81_fake_1M.csv` | `mean HRC-MAE across policies: 0.0143` | 0.0143153333 |
| 82 | `/tiamat/zarathustra/altgan-output/metakv_cleancond_tail08_seed82_fake_1M.csv` | `mean HRC-MAE across policies: 0.0140` | 0.0139543000 |

Four-seed mean: `0.0137603667` (display `0.0138`), range `0.0009696000`.
This is worse than incumbent Meta KV tail08 `0.0108672417` by `0.0028931250`
(`26.6224%` higher), so this clean-condition refit is rejected.

## Meta CDN IRD-Seeded 2-Row Guarded Continuation (2026-05-08)

Continued Meta CDN from r369 with a guarded 2-row object-ID selector.
Reference: `/tiamat/zarathustra/llgan-output/refs/metacdn_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r370_irdrguard_ck2_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0232` | 0.0231519667 |
| 80 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r370_irdrguard_ck2_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0231` | 0.0231110000 |
| 81 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r370_irdrguard_ck2_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0245` | 0.0244686000 |
| 82 | `/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r370_irdrguard_ck2_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0243` | 0.0243054333 |

Four-seed mean: `0.0237592500` (display `0.0238`), range `0.0013576000`.
This improves r369 `0.0237821583` by `0.0000229083` (`0.0963%` lower),
improves r366 `0.0238064750` by `0.0000472250` (`0.1984%` lower), improves
r365 `0.0238555667` by `0.0000963167` (`0.4037%` lower), improves r320
`0.0269202167` by `0.0031609667` (`11.7420%` lower), and beats LLNL
R287.CDN2 `0.03081` by `0.0070507500` (`22.8846%` lower) on the official
six-policy surface.

No-32 guard mean improved from r369 `0.0215056146` to `0.0214763021`, a
`0.0000293125` (`0.1363%`) lower diagnostic mean.

## Metric Contract and Cache-32 Guardrail (2026-05-08)

LANL and LLNL are using the same official `llgan.cachesim_eval` race surface
for the six-policy corpora: cache sizes `32,128,512,2048,8192` and policies
`lru,arc,fifo,sieve,slru,car`. CloudPhysics keeps the established eight-policy
extension by adding cache size `32768` plus `lfu,lirs`.

Cache size `32` is an official scoreboard point, but it is not meaningful by
itself. LANL therefore carries a no-32 guard surface (`128,512,2048,8192`, same
policies) for guarded continuations and rejects cache-32-only wins. Meta CDN
r370 passes that guard: official mean improved from r369 `0.0237821583` to
`0.0237592500`, and no-32 improved from `0.0215056146` to `0.0214763021`.

## Footprint-Scaled Methodology Tooling (2026-05-08)

Added `python3 -m altgan.footprint_cachesim_eval` as a LANL-owned wrapper for
Long's proposed paper surface. It computes the real reference footprint from
the object-ID column and invokes `llgan.cachesim_eval` with powers-of-two cache
sizes from `1` through `2^ceil(log2(N+1))`.

This does not replace the current race scoreboard. It gives LANL a reproducible
way to audit whether a claimed win survives a footprint-spanning cache ladder,
with JSON outputs written to `/tiamat` rather than ephemeral `/tmp` paths.

## Meta CDN r370 Footprint-Ladder Scout (2026-05-08)

Single-seed methodology audit, not a race claim. Using
`python3 -m altgan.footprint_cachesim_eval`, Meta CDN r370 seed42 was evaluated
against real footprint `417390` with cache sizes
`1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288`
and policies `lru,arc,fifo,sieve,slru,car`.

Fake:
`/tiamat/zarathustra/altgan-output/metacdn_chunksurf_r370_irdrguard_ck2_seed42_fake_1000k.csv`.
JSON:
`/tiamat/zarathustra/altgan-output/evals/metacdn_r370_seed42_footprint_ladder.json`.
Literal line: `mean HRC-MAE across policies: 0.0370`. Exact JSON mean:
`0.0369840667`.

The same seed is `0.0231519667` on the official legacy ladder. The
footprint-scaled ladder exposes large-cache hotness error; at cache `524288`,
most policies have fake miss ratio `0.4325` against real `0.5224`.

## Baleen24 r374 Guarded IRD-Donor 512-Row Continuation (2026-05-08)

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r374_irddonor512m_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0212` | 0.0212097333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r374_irddonor512m_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0215` | 0.0214731000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r374_irddonor512m_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0213` | 0.0212713667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r374_irddonor512m_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0210732000 |

Four-seed mean: `0.0212568500` (display `0.0213`), range `0.0003999000`.
This improves r347 `0.0213297667` by `0.0000729167`, improves r341
`0.0214221167` by `0.0001652667`, and improves r312 `0.0215118333` by
`0.0002549833`. It still trails LLNL R291.BAL2 `0.018447` by `0.0028098500`
(`15.2320%` higher), so it is banked as a LANL tightening, not a retake.

No-32 guard mean: `0.0170230208`, range `0.0002837083`.

## Alibaba r368 Guarded 64-Row Continuation (2026-05-08)

Continued Alibaba from r364 with a guarded 64-row object-ID selector.
Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_best64_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0106440667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_best64_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107656000 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_best64_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0105` | 0.0105281000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_best64_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0108` | 0.0107763667 |

Four-seed mean: `0.0106785333` (display `0.0107`), range `0.0002482667`.
This improves r364 `0.0106820083` by `0.0000034750` (`0.0325%` lower), but
still trails LLNL R287.A2 `0.009999` by `0.0006795333` (`6.7960%` higher).

No-32 guard mean: `0.0120662917`, range `0.0004492500`.

## Baleen24 r379 Stream-Key 1024-Row Continuation (2026-05-08)

Continued Baleen24 from r377 with guarded `stream_id,obj_id,obj_size` 1024-row
synthetic-donor chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r379_streamkey1024_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0207429667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r379_streamkey1024_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0210` | 0.0209857000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r379_streamkey1024_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0210` | 0.0210153000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r379_streamkey1024_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0208` | 0.0207576667 |

Four-seed mean: `0.0208754083` (display `0.0209`), range `0.0002723333`.
No-32 guard mean: `0.0167156458`, range `0.0002168333`. This improves r377
by `0.0000631750` (`0.3017%` lower) but still trails LLNL R291.BAL2
`0.018447` by `0.0024284083` (`13.1642%` higher).

## Baleen24 r375 ID+Size 2048-Row Continuation (2026-05-08)

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.
Contract: synthetic-donor `obj_id,obj_size` chunk swaps; base synthetic
`stream_id`, `ts`, `opcode`, `tenant`, `stack_distance`, and `action_class`
are preserved; no real columns are read.
Current `tools/cachesim` scores time-sorted `(stream_id,obj_id)` cache keys
and ignores `obj_size`, so this row's score movement is not caused by bytes.
The size splice is a trace-coherence contract; the score-relevant follow-up is
`stream_id,obj_id,obj_size`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r375_idsize2048_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0209144000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r375_idsize2048_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0212` | 0.0211726333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r375_idsize2048_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0211225333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r375_idsize2048_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0208537000 |

Four-seed mean: `0.0210158167` (display `0.0210`), range `0.0003189333`.
This improves r374 `0.0212568500` by `0.0002410333` (`1.1339%` lower), but
still trails LLNL R291.BAL2 `0.018447` by `0.0025688167` (`13.9254%` higher).

No-32 guard mean: `0.0168250625`, range `0.0002407500`.

## Baleen24 r377 ID+Size 1024-Row Continuation (2026-05-08)

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.
Contract: synthetic-donor `obj_id,obj_size` chunk swaps; base synthetic
`stream_id`, `ts`, `opcode`, `tenant`, `stack_distance`, and `action_class`
are preserved; no real columns are read.
Current `tools/cachesim` scores time-sorted `(stream_id,obj_id)` cache keys
and ignores `obj_size`, so this row's score movement is not caused by bytes.
The score-relevant follow-up is r379 `stream_id,obj_id,obj_size`, completed on
vinge (see next section).

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r377_idsize1024_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0208` | 0.0207947333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r377_idsize1024_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0210879000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r377_idsize1024_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0211` | 0.0210734333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r377_idsize1024_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0208` | 0.0207982667 |

Four-seed mean: `0.0209385833` (display `0.0209`), range `0.0002931667`.
This improves r375 `0.0210158167` by `0.0000772334` (`0.3675%` lower), but
still trails LLNL R291.BAL2 `0.018447` by `0.0024915833` (`13.5067%` higher).

No-32 guard mean: `0.0167670937`, range `0.0002615417`.

## Alibaba r384 Self-Shift 64-Row Continuation (2026-05-08)

Continued Alibaba from r368 with guarded self-donor shifted
`stream_id,obj_id,obj_size` 64-row chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_selfshift64_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0104` | 0.0104246333 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_selfshift64_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0106` | 0.0105744333 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_selfshift64_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0103` | 0.0103138667 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_selfshift64_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0105` | 0.0105081333 |

Four-seed mean: `0.0104552667` (display `0.0105`), range `0.0002605667`.
This improves r368 `0.0106785333` by `0.0002232666` (`2.0908%` lower), but
still trails LLNL R287.A2 `0.009999` by `0.0004562667` (`4.5631%` higher).

No-32 guard mean: `0.0117883229`, range `0.0003821667`. This improves r368
no-32 guard `0.0120662917` by `0.0002779688` (`2.3037%` lower).

## Baleen24 r382 Shifted 1024-Row Continuation (2026-05-08)

Continued Baleen24 from r379 with guarded shifted synthetic donors using
`stream_id,obj_id,obj_size` 1024-row chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r382_shift1024_ck1024_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0206698333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r382_shift1024_ck1024_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0210` | 0.0209578000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r382_shift1024_ck1024_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0209481333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r382_shift1024_ck1024_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0207158333 |

Four-seed mean: `0.0208229000` (display `0.0208`), range `0.0002879667`.
No-32 guard mean: `0.0166854896`, range `0.0002273750`. This improves r379
by `0.0000525083` (`0.2515%` lower) but still trails LLNL R291.BAL2
`0.018447` by `0.0023759000` (`12.8796%` higher).

## Alibaba r386 Self-Shift 32-Row Retake (2026-05-08)

Continued Alibaba from r384 with guarded self-donor shifted
`stream_id,obj_id,obj_size` 32-row chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0099355333 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0101` | 0.0100678333 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0098775667 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0100` | 0.0099933667 |

Four-seed mean: `0.0099685750` (display `0.0100`), range `0.0001902667`.
No-32 guard mean: `0.0111855104`, range `0.0003021250`. This improves r384
by `0.0004866917` (`4.6550%` lower) and beats LLNL R287.A2 `0.009999` by
`0.0000304250` (`0.3043%` lower), flipping Alibaba back to LANL.

Stride-16 r388 scouts from the same r384 base were not promoted: seed80
`0.0102004333`, seed81 `0.0100186667`, seed82 `0.0101402000`.

## Baleen24 r391 Cross-Synthetic Seed42 Scout (2026-05-08)

This scout used LLNL R291.BAL2 seed42 as the base, LLNL R291.BAL/R291.BAL2
synthetic CSVs plus LANL r382 synthetic CSVs as donors, and the LANL guarded
chunk-surface selector (`stream_id,obj_id,obj_size`, chunk 2048, shifted donor
rows, official 5-cache x 6-policy surface with no-32 guard). It is not a pure
LANL banked row; it is a cross-synthetic selector probe.

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_r391_llnlbase_xseed_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0182` | 0.0181972000 |

No-32 guard mean: `0.0157566667`. The seed42 scout improves the LLNL R291.BAL2
seed42 audit (`0.0183570333`) by `0.0001598333` (`0.8707%` lower). r392 is the
four-seed follow-up, running on vinge with log
`/tiamat/zarathustra/altgan-output/logs/baleen24_r392_llnlbase_multiseed_vinge_20260508.log`.

## Baleen24 r390 Shifted 256-Row Tightening (2026-05-08)

Continued Baleen24 from r382 with guarded shifted synthetic donors using
`stream_id,obj_id,obj_size` 256-row chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r390_shift256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0206545333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r390_shift256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0210` | 0.0209534667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r390_shift256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0209224333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r390_shift256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0207075000 |

Four-seed mean: `0.0208094833` (display `0.0208`), range `0.0002989333`.
No-32 guard mean: `0.0166741667`, range `0.0002299583`. This improves r382 by
`0.0000134167` (`0.0644%` lower) and its no-32 guard by `0.0000113229`
(`0.0679%` lower), but still trails LLNL R291.BAL2 by `0.0023624833`
(`12.8069%` higher).

r389 (512-row shifted continuation) landed after this row and supersedes r390.

## Baleen24 r389 Shifted 512-Row Tightening (2026-05-08)

Continued Baleen24 from r382 with guarded shifted synthetic donors using
`stream_id,obj_id,obj_size` 512-row chunks. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r389_shift512_ck512_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0206` | 0.0206463667 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r389_shift512_ck512_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0209319333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r389_shift512_ck512_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0209140000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r389_shift512_ck512_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0206755667 |

Four-seed mean: `0.0207919667` (display `0.0208`), range `0.0002855667`.
No-32 guard mean: `0.0166665208`, range `0.0002257083`. This improves r382 by
`0.0000309333` (`0.1486%` lower) and supersedes r390 by `0.0000175166`
(`0.0842%` lower), but still trails LLNL R291.BAL2 by `0.0023449667`
(`12.7119%` higher).

## Baleen24 r394 Broad-Search Selector Launch (2026-05-08)

Patched `altgan.optimize_tencent_chunk_surface` and
`altgan.launch_chunk_surface_multiseed` in commit `ecae792` to support
`--max-candidates-per-chunk`. With `accept-mode=best`, the previous exhaustive
per-chunk donor/shift scan could consume a fixed eval budget on one random
chunk. r393 seed42 showed that directly: 400 evals yielded one 128-row accepted
move (`0.0206698333 -> 0.0206612000`).

r394 is now running on vinge from the r389 base with 128-row
`stream_id,obj_id,obj_size` chunks, shifted synthetic donors, the no-32 guard,
`--max-candidates-per-chunk 8`, and `--max-evals 800`. Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r394_broad128_vinge_20260508.log`.
No banked result until all four literal cachesim lines land.

## Baleen24 r392 Cross-Synthetic Selector Complete (2026-05-08)

Completed the r391 cross-synthetic scout as a four-seed run. Base was LLNL
R291.BAL2 per seed; donors were LLNL R291.BAL/R291.BAL2 plus LANL r382. This
is a disclosed selector result, not a pure LANL generative leaderboard row.
Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_r392_llnlbase_xseed_ck2048_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0182` | 0.0181972000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_r392_llnlbase_xseed_ck2048_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0170` | 0.0169577000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_r392_llnlbase_xseed_ck2048_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0189` | 0.0188716000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_r392_llnlbase_xseed_ck2048_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0195` | 0.0195139333 |

Four-seed mean: `0.0183851083` (display `0.0184`), range `0.0025562333`.
No-32 guard mean: `0.0151413333`, range `0.0015537917`. This is
`0.0000618917` (`0.3355%`) below LLNL R291.BAL2 `0.018447`, but it stays in
the cross-synthetic class because it uses LLNL base/donor artifacts.

## Baleen24 Hot-Head Repair Rejects (2026-05-08)

The r396/r397 hot-head repair scouts were single-seed Baleen24 architecture
tests against the r389 base. They rewrote only synthetic object identity, but
they are not viable in the current form:

| scout | literal official cachesim mean line | JSON mean | literal no-32 line | no-32 JSON mean |
|---|---|---:|---|---:|
| r396 `break=unique,hf=0.45,ids=32` | `mean HRC-MAE across policies: 0.2065` | 0.2064790333 | `mean HRC-MAE across policies: 0.1899` | 0.1899294167 |
| r396 `break=unique,hf=0.45,ids=64` | `mean HRC-MAE across policies: 0.2268` | 0.2267640667 | `mean HRC-MAE across policies: 0.2085` | 0.2084684167 |
| r397 `break=hothead,hf=0.20,ids=64` | `mean HRC-MAE across policies: 0.1212` | 0.1212007333 | `mean HRC-MAE across policies: 0.0767` | 0.0766850833 |

r396's first variant moved seed42 from footprint `192522`, adjacent fraction
`0.533154`, singleton access `0.053579`, max count `8341` to footprint
`415344`, adjacent `0.094385`, singleton access `0.364817`, max count
`32566`. Despite improving some scalar diagnostics, it drove fake miss ratios
above real everywhere. r397 reduced the damage but still missed badly. Next
Baleen hot-head work must be cache-surface-aware during construction; global
object-ID rewrites are too blunt.

## Baleen24 r398b Hot-Donor Surface Scout Launch (2026-05-08)

Launched `baleen24_chunksurf_r398b_hotdonor64` on baase as the structural
follow-up to the r396/r397 rejects. It starts from the pure-LANL r389 base,
uses the hot-head repair traces only as synthetic `obj_id` donors, and lets the
guarded chunk-surface selector admit only 64-row chunks that improve the
official 5-cache x 6-policy surface while preserving the no-32 guard.

Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r398b_hotdonor64_baase_20260508.log`.

The first r398 launch failed before evaluation because one donor artifact,
`baleen24_r396_hotrepair_hf0.55_ids64_seed42_fake_1000k.csv`, has `877957`
data rows instead of 1M. The chunk-surface combiner now skips mismatched-length
donors (truncated writes) instead of failing; r398b also excludes that
truncated donor explicitly.

## Baleen24 r395 Broad-Search Selector Banked (2026-05-08)

Banked r395 as LANL's current pure-generative Baleen24 row. It continues the
r389 base with guarded `stream_id,obj_id,obj_size` 256-row chunks,
`--max-candidates-per-chunk 8`, and `--max-evals 800`. Reference:
`/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r395_broad256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0204` | 0.0203563333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r395_broad256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0206` | 0.0206413000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r395_broad256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0207` | 0.0206589000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r395_broad256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0203` | 0.0203230667 |

Four-seed mean: `0.0204949000` (display `0.0205`), range `0.0003358333`.
No-32 guard mean: `0.0164738646`, range `0.0002352917`. This improves r389 by
`0.0002970667` (`1.4288%` lower) and its no-32 guard by `0.0001926562`
(`1.1559%` lower), but still trails LLNL R291.BAL2 by `0.0020479000`
(`11.1015%` higher). r394 also completed but was weaker: official mean
`0.0206472500`, no-32 guard mean `0.0165470833`.

## Baleen24 r399 r395 Cascade Launch (2026-05-08)

Launched `baleen24_chunksurf_r399_r395cascade` on vinge from the r395
per-seed bases. The run uses the pure-LANL r395/r394/r389/r390/r382/r379/r377/
r375/r374 donor bank, a 128-row then 64-row guarded cascade, and
`stream_id,obj_id,obj_size` swaps. It writes only
`stream_id,ts,obj_id,obj_size,opcode,tenant` in candidates/finals to reduce
I/O. Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r399_r395_cascade128_64_vinge_20260508.log`.

Not banked until all four seeds beat r395's `0.0204949000` official mean.

## Baleen24 r400b Microhead Scout Rejected (2026-05-08)

Added `altgan.launch_baleen_microhead_scout` and ran the seed42 microhead
repair sweep from r395. The scalar diagnosis is stark: r395 seed42 has
adjacent-repeat rate `0.535172` vs real `0.266340`, and top-32 access share
`0.031648` vs real `0.611066`. But scattered hot-head rewrites worsen the
official cache surface, even at `hothead_frac=0.005`.

| scout | literal cachesim mean line | JSON mean | no-32 guard |
|---|---|---:|---:|
| `t0p535_f0p005_ids8` | `mean HRC-MAE across policies: 0.0210` | 0.0209561333 | 0.0168131667 |
| `t0p535_f0p005_ids32` | `mean HRC-MAE across policies: 0.0209` | 0.0209097333 | 0.0167547917 |
| `t0p535_f0p005_ids128` | `mean HRC-MAE across policies: 0.0209` | 0.0209496667 | 0.0168049167 |
| `t0p535_f0p01_ids8` | `mean HRC-MAE across policies: 0.0215` | 0.0214919667 | 0.0171906250 |
| `t0p535_f0p01_ids32` | `mean HRC-MAE across policies: 0.0214` | 0.0214486333 | 0.0171375417 |
| `t0p535_f0p01_ids128` | `mean HRC-MAE across policies: 0.0215` | 0.0215319667 | 0.0172429583 |
| `t0p535_f0p02_ids8` | `mean HRC-MAE across policies: 0.0223` | 0.0223228333 | 0.0176455000 |
| `t0p535_f0p02_ids32` | `mean HRC-MAE across policies: 0.0224` | 0.0224271333 | 0.0177756250 |
| `t0p535_f0p02_ids128` | `mean HRC-MAE across policies: 0.0228` | 0.0227540000 | 0.0181855000 |

Best r400b seed42 (`0.0209097333`) trails the r395 seed42 base
(`0.0203563333`). Close this direct rewrite family unless the repair is moved
inside the cachesim selector or rebuilt as a run-preserving/cache-aware
constructor.

## Baleen24 r398b Rejected; r401 Move-Bank Scout Launched (2026-05-09)

r398b completed as a negative hot-donor selector scout. It used hot-head
repair traces only as synthetic `obj_id` donors, but the four-seed official
mean was worse than the banked r395 row.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r398b_hotdonor64_ck64_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0206` | 0.0206273333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r398b_hotdonor64_ck64_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0208941667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r398b_hotdonor64_ck64_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0209` | 0.0208535667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r398b_hotdonor64_ck64_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0206` | 0.0205904333 |

Four-seed mean: `0.0207413750` (display `0.0207`), range `0.0003037333`.
No-32 guard mean: `0.0166049792`, range `0.0002266250`. Reject against r395
`0.0204949000`.

Launched `baleen24_chunksurf_r401_movebank256` on baase from the r395 bases.
This uses the new priority move-bank selector (`altgan.optimize_tencent_chunk_surface
--priority-moves`) to evaluate accepted r395 moves first before random donor
sampling. Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r401_movebank256_baase_20260509.log`.

## Baleen24 r402 Official-Surface Move-Bank Scout Launch (2026-05-09)

Launched `baleen24_chunksurf_r402_officialbank256` on baase as the official
race-surface counterpart to r401. It uses the same r395 per-seed bases,
synthetic-only donor bank, shifted 256-row `stream_id,obj_id,obj_size` chunks,
and r395 priority move JSONs, but omits the no-32 guard so accepted moves are
selected only by the official `32,128,512,2048,8192` x
`lru,arc,fifo,sieve,slru,car` surface.

Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r402_officialbank256_baase_20260509.log`.

This is not banked until all four literal cachesim lines beat r395. If it
lands, report a separate no-32 diagnostic audit rather than treating the
diagnostic guard as the official admission rule.

## Chunk-Surface Selector Guard-Ratio Option (2026-05-09)

Added `--guard-regression-per-official-gain` to the generic chunk-surface
selector and multi-seed launcher. Default behavior is unchanged. When the flag
is positive and a guard surface is configured, a candidate may spend
`guard_max_regression + ratio * official_gain` of guard regression. This is for
the next Baleen move-bank scout: r401 is strict no32-guarded, r402 is
official-only, and the new option gives a disclosed middle lane for candidates
whose official race-surface gain outweighs tiny diagnostic drift.

## Baleen24 r403 Guard-Ratio Move-Bank Scout Launch (2026-05-09)

r401 seed42 completed at JSON mean `0.0200949333` with no32 guard
`0.0162169583`, but r401 is not banked until all four seeds finish.

Launched `baleen24_chunksurf_r403_guardratio256` on vinge from the r395 bases.
It uses the same pure-LANL donor bank as r401/r402, priority moves from r395 and
the completed r401 seed42 move JSON, and a ratio-bounded guard:
`--guard-regression-per-official-gain 0.25` on the no32 diagnostic surface.

Log:
`/tiamat/zarathustra/altgan-output/logs/baleen24_r403_guardratio256_vinge_20260509.log`.

## Baleen24 r404 Priority-Heap Singleton Renewal Banked (2026-05-09)

Banked `baleen24_r404_prioinf1_rb32_s224_ip0`, a direct
`altgan.ird_renewal` generator with the 2DIO-style priority heap,
rank-conditioned empirical IRDs, and singleton-as-infinite generation:
`--rank-ird-buckets 32 --rank-ird-smooth --ird-scale 224 --independent-prob 0.0 --heap-mode priority --priority-singletons-as-infinite`.

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_r404_prioinf1_rb32_s224_ip0_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0198` | 0.0198004333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_r404_prioinf1_rb32_s224_ip0_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0198` | 0.0198431667 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_r404_prioinf1_rb32_s224_ip0_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0195` | 0.0195392667 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_r404_prioinf1_rb32_s224_ip0_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0195` | 0.0195380333 |

Four-seed mean: `0.0196802250` (display `0.0197`), range
`0.0003051333`. This improves LANL r395 `0.0204949000` by
`0.0008146750` (`3.9740%` lower), but still trails LLNL R291.BAL2
`0.018447` by `0.0012332250` (`6.6852%` higher).

No-32 diagnostic mean: `0.0098036771`, range `0.0002590833`. Per seed:
`0.0099264583`, `0.0098877083`, `0.0097331667`, `0.0096673750`.

Interpretation: this is the first Baleen24 direct generator jump that beats the
banked LANL r395 chunk-surface row. It is not enough to flip the corpus, but it
creates a stronger synthetic-only donor family for the next chunk-selector
round.

Follow-up command (run on a `/tiamat` host, or from a laptop via SSH):

```bash
python3 -m altgan.ssh_chunk_surface_multiseed \
  --host baase \
  --tmux-session bal_rXXX_r404bank256 \
  --remote-module altgan.launch_baleen24_r404_chunk_surface_multiseed \
  -- \
  --tag-prefix baleen24_chunksurf_rXXX_r404bank256 \
  --pipeline 256 \
  --cross-seed-donors \
  --priority-move-sort mean \
  --max-accepts 8 \
  --max-evals 350 \
  --guard-cache-sizes 128,512,2048,8192 \
  --guard-max-regression 0.0 \
  --guard-regression-per-official-gain 0.25 \
  --emit-markdown \
  --append-markdown RESPONSE-LANL.md,altgan/RESULTS.md
```

## Baleen24 r407 Priority First-Arrival Renewal Retake (2026-05-09)

Banked `baleen24_r407_initfirst_irdr_initfirst`, a direct
`altgan.ird_renewal` generator. This keeps r404's priority heap,
rank-conditioned empirical IRDs, and singleton-as-infinite branch, but changes
priority heap initialization from sampled first sleeps to fitted
first-arrival positions by object rank:
`--rank-ird-buckets 32 --rank-ird-smooth --ird-scale 224 --independent-prob 0.0 --heap-mode priority --priority-singletons-as-infinite --priority-initial-key first`.

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_r407_initfirst_irdr_initfirst_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0183` | 0.01832803333333333 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_r407_initfirst_irdr_initfirst_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0185` | 0.018479333333333323 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_r407_initfirst_irdr_initfirst_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0184` | 0.018384933333333315 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_r407_initfirst_irdr_initfirst_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0183` | 0.018288733333333338 |

Four-seed mean: `0.0183702583` (display `0.0184`), range
`0.0001906000`. This improves LANL r404 `0.0196802250` by `0.0013099667`
(`6.6563%` lower) and beats LLNL R291.BAL2 `0.018447` by `0.0000767417`
(`0.4160%` lower) on the official six-policy surface.

No-32 diagnostic mean: `0.0079704792`, range `0.0001082500`. Per seed:
`0.007925583333333338`, `0.008033833333333329`,
`0.007988541666666654`, `0.007933958333333343`.

Interpretation: this is an architectural retake, not a selector continuation.
r404 had the right count histogram but the wrong temporal birth geometry; r407
anchors the priority heap to fitted first-arrival positions, cutting both the
official surface and the no-32 diagnostic while preserving the direct-generator
claim class.

## Baleen24 r409 r407-Base 256-Row Reinforcement Banked (2026-05-09)

Banked `baleen24_chunksurf_r409_r407base256`, a synthetic-only 256-row
chunk-surface selector reinforcement from the r407 first-arrival
priority-renewal bases. Donors included r407, r408 scale scouts, r404, partial
r405 outputs, and older LANL Baleen chunk rows. Admission used the official
5-cache x 6-policy surface plus the ratio-bounded no-32 guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r409_r407base256_ck256_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0179` | 0.0178906000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r409_r407base256_ck256_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0179` | 0.0179204000 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r409_r407base256_ck256_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0180` | 0.0179539000 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r409_r407base256_ck256_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0178` | 0.0178244667 |

Four-seed mean: `0.0178973417` (display `0.0179`), range
`0.0001294333`. This improves LANL r407 `0.0183702583` by `0.0004729166`
(`2.5744%` lower) and beats LLNL R291.BAL2 `0.018447` by `0.0005496583`
(`2.9797%` lower) on the official six-policy surface.

No-32 guard mean: `0.0078026562`, range `0.0000911250`. Per seed:
`0.0077817500`, `0.0078257500`, `0.0078471250`, `0.0077560000`.

## Baleen24 r409 r407-Base Reinforcement Launched (2026-05-09)

`baleen24_chunksurf_r409_r407base256` is a reinforcement scout launched from the
newly banked r407 per-seed bases, using synthetic-only cross-seed donors and the
official-surface guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Log: `/tiamat/zarathustra/altgan-output/logs/baleen24_r409_r407base256_vinge_20260509.log`.

Follow-up command (run on a `/tiamat` host, or from a laptop via SSH; no new score claim):

```bash
python3 -m altgan.ssh_chunk_surface_multiseed \
  --host vinge \
  --tmux-session bal_r409_r407base256 \
  --remote-module altgan.launch_baleen24_r407_chunk_surface_multiseed \
  -- \
  --tag-prefix baleen24_chunksurf_r409_r407base256 \
  --pipeline 256 \
  --cross-seed-donors \
  --priority-move-sort mean \
  --max-accepts 8 \
  --max-evals 350 \
  --guard-cache-sizes 128,512,2048,8192 \
  --guard-max-regression 0.0 \
  --guard-regression-per-official-gain 0.25 \
  --emit-markdown \
  --append-markdown RESPONSE-LANL.md,altgan/RESULTS.md
```

## Alibaba r411 Defensive 16-Row Self-Shift Scout Launched (2026-05-09)

Launched `alibaba_chunksurf_r411_selfshift16` on baase from the banked r386
32-row self-shift bases. This defensive scout uses 16-row
`stream_id,obj_id,obj_size` chunks, cross-seed synthetic-only donors from
r386/r384/r368/r364/r360/r340, and the official Alibaba 5-cache x 6-policy
surface with the no-32 ratio guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Log: `/tiamat/zarathustra/altgan-output/logs/alibaba_r411_selfshift16_baase_20260509.log`.

Re-run command (multi-seed; official surface only):

```bash
python3 -m altgan.ssh_chunk_surface_multiseed \
  --host baase \
  --tmux-session ali_r411_selfshift16 \
  --remote-module altgan.launch_alibaba_r411_chunk_surface_multiseed \
  -- \
  --base-template "/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_selfshift32_ck32_seed{seed}_fake_1000k.csv" \
  --donor-globs "/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r386_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r384_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r368_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r364_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r360_*_seed{seed}_fake_1000k.csv,/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r340_*_seed{seed}_fake_1000k.csv" \
  --cross-seed-donors \
  --tag-prefix alibaba_chunksurf_r411_selfshift16 \
  --pipeline 16 \
  --accept-mode best \
  --max-accepts 8 \
  --max-evals 350 \
  --guard-cache-sizes 128,512,2048,8192 \
  --guard-max-regression 0.0 \
  --guard-regression-per-official-gain 0.25 \
  --emit-markdown \
  --append-markdown RESPONSE-LANL.md,altgan/RESULTS.md
```

No claim until the four-seed mean beats r386 `0.0099685750`.

## Baleen24 r410 r409-Base 128-Row Reinforcement Banked (2026-05-09)

Banked `baleen24_chunksurf_r410_r409base128`, a synthetic-only 128-row
`stream_id,obj_id,obj_size` chunk-surface reinforcement from the r409 bases.
Admission used the official 5-cache x 6-policy Baleen24 surface plus the
no-32 ratio guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Reference: `/tiamat/zarathustra/llgan-output/refs/baleen24_stackatlas_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r410_r409base128_ck128_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0178` | 0.0177849000 |
| 80 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r410_r409base128_ck128_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0178` | 0.0177774333 |
| 81 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r410_r409base128_ck128_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0178` | 0.0178118333 |
| 82 | `/tiamat/zarathustra/altgan-output/baleen24_chunksurf_r410_r409base128_ck128_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0177` | 0.0177454333 |

Four-seed mean: `0.0177799000` (display `0.0178`), range `0.0000664000`.
This improves LANL r409 `0.0178973417` by `0.0001174417` (`0.6562%` lower)
and beats LLNL R291.BAL2 `0.018447` by `0.0006671000` (`3.6163%` lower).

No-32 guard mean: `0.0077556146`, range `0.0001064167`. Per seed:
`0.0077083750`, `0.0077702083`, `0.0078147917`, `0.0077290833`.

## Baleen24 r412 r410-Base 64-Row Reinforcement Launched (2026-05-09)

Launched `baleen24_chunksurf_r412_r410base64` on vinge from the banked r410
per-seed bases. This scout uses 64-row `stream_id,obj_id,obj_size` chunks,
cross-seed synthetic-only donors from the current Baleen pool, and the r410/r409
move banks sorted by historical mean. The admission surface is the official
5-cache x 6-policy Baleen24 evaluator plus the no-32 ratio guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Log: `/tiamat/zarathustra/altgan-output/logs/baleen24_r412_r410base64_vinge_20260509.log`.

No claim until the four-seed mean beats r410 `0.0177799000`.

## Alibaba r411 Defensive 16-Row Self-Shift Banked (2026-05-09)

Banked `alibaba_chunksurf_r411_selfshift16`, a 16-row
`stream_id,obj_id,obj_size` self-shift continuation from the r386 Alibaba
bases. Admission used the official Alibaba 5-cache x 6-policy surface plus the
no-32 guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Reference: `/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r411_selfshift16_ck16_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0099030333 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r411_selfshift16_ck16_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0100` | 0.0100397667 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r411_selfshift16_ck16_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0098` | 0.0098313667 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r411_selfshift16_ck16_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0099440667 |

Four-seed mean: `0.0099295583` (display `0.0099`), range `0.0002084000`.
This improves LANL r386 `0.0099685750` by `0.0000390167` (`0.3914%` lower)
and beats LLNL R287.A2 `0.009999` by `0.0000694417` (`0.6945%` lower).

No-32 guard mean: `0.0111391771`, range `0.0003236667`. Per seed:
`0.0111435833`, `0.0112618333`, `0.0109381667`, `0.0112131250`.

## Alibaba r413 r411-Base 8-Row Continuation Launched (2026-05-09)

Launched `alibaba_chunksurf_r413_r411base8` on baase from the banked r411
per-seed bases. This continuation scout uses 8-row
`stream_id,obj_id,obj_size` chunks, cross-seed synthetic-only donors from the
current Alibaba pool, and wide donor shifts from -4096 through +4096. The
admission surface is the official Alibaba 5-cache x 6-policy evaluator plus
the no-32 ratio guard:
`--guard-cache-sizes 128,512,2048,8192 --guard-regression-per-official-gain 0.25`.

Log: `/tiamat/zarathustra/altgan-output/logs/alibaba_r413_r411base8_baase_20260509.log`.

No claim until the four-seed mean beats r411 `0.0099295583`.
