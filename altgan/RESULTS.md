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
peer evidence in `VERSIONS.md` reports Tencent `v158` HRC-MAE `0.2435` and
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
