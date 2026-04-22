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
| Tencent | PhaseAtlas | 1024 files x 5k holdout, blend 0.5 | **0.01065** | 0.60947 | 0.61493 | 48 | 60 | 159 | 174 |
| Alibaba | PhaseAtlas | 233 files x 25k holdout, blend 0.0 | **0.00301** | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 |
| Tencent | NeuralAtlas | 1024 files x 5k holdout, blend 0.25 | 0.01853 | 0.62066 | 0.61493 | 45 | 60 | 144 | 174 |
| Alibaba | NeuralAtlas | 233 files x 25k holdout, blend 0.75 | 0.00349 | 0.26730 | 0.26909 | 183 | 201 | 1264 | 1452 |
| Tencent | NeuralAtlas | 64 files x 25k, blend 0.0 | 0.01845 | 0.62314 | 0.61493 | 55 | 60 | 145 | 174 |
| Alibaba | NeuralAtlas | 64 files x 25k, blend 0.5 | 0.00183 | 0.26451 | 0.26909 | 197 | 201 | 1267 | 1452 |
| Alibaba | NeuralStack | 64 files x 25k, temp 1.0 | 0.00333 | 0.27373 | 0.26909 | 204 | 201 | 1331 | 1452 |

These are directly comparable to the peer long-rollout sidecar.  The current
peer evidence in `VERSIONS.md` reports Tencent `v158` HRC-MAE `0.2435` and
Alibaba `v194` HRC-MAE `0.1305` with reuse-access `0.006` vs real `0.265`.
The altgan best-of-family results are not close calls on that surface.
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
| Tencent | PhaseAtlas | 1024 files x 5k, holdout | 1.0 | 0.01982 | 0.63573 | 0.61493 | 60 | 60 | 186 | 174 | 0.0497 | 0.0390 |
| Alibaba | NeuralAtlas | 233 files x 25k, holdout | 0.75 | 0.00349 | 0.26730 | 0.26909 | 183 | 201 | 1264 | 1452 | 0.3251 | 1.1296 |
| Alibaba | PhaseAtlas | 233 files x 25k, holdout | 0.0 | **0.00301** | 0.27125 | 0.26909 | 205 | 201 | 1380 | 1452 | 1.2015 | 0.7529 |

The PhaseAtlas rows are the better answer to the "statistically
indistinguishable" goal. Tencent still under-expresses timing and size drift,
but phase conditioning improves HRC and moves drift in the right direction.
Alibaba PhaseAtlas is currently the cleanest all-around held-out result:
cache metrics are close and first-half/second-half timing and size drift are
within the same order as real.

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
