# LANL Response Log

This file contains LANL / `altgan/` responses to cross-team critiques. The
detailed measurement ledger remains [altgan/RESULTS.md](altgan/RESULTS.md);
versioned LANL milestones and experimental log live in
[VERSIONS-LANL.md](VERSIONS-LANL.md).

LANL has read access to LLNL's files. Peer review: see
[PEER-REVIEW-LANL.md](PEER-REVIEW-LANL.md).

---

## Scope

LANL's system-under-test is `altgan/`, documented at
[altgan/README.md](altgan/README.md).  LLNL's `llgan/` is off-limits for
LANL commits; peer review only.

LANL's generation pipeline:

1. **PhaseAtlas** (`altgan/model.py` + `altgan/train.py`) — a per-phase atlas
   of Mattson LRU stack-depth distributions, trained on real traces and
   sampled at generation time.  Post-hoc knobs adjust hot-pool size, reuse
   probability, and temporal shaping.

2. **NeuralStack** (`altgan/neural_stack.py`) — a neural-network-conditioned
   Mattson depth sampler; still exploratory.

3. **TraceBootstrap** (`altgan/trace_bootstrap.py`) — chunk-shuffle
   baseline used jointly with LLNL.

4. **IRD Renewal** (`altgan/ird_renewal.py`) — inter-reference distance
   model for direct synthetic trace generation.

5. **Mattson-Denning LSTM** (`altgan/mattson_denning_lstm.py`) — learned
   autoregressive sequence model over Mattson LRU stack-depth tokens
   and Denning working-set tokens. Trained with next-token cross entropy;
   cachesim is only used post-hoc.

LANL's evaluation pipeline uses the same `llgan/cachesim_eval.py` surface
as LLNL, with the same reference CSVs and policy set.

---

## Versioning

LANL uses round numbers.  **r** prefix = LANL round; **R** prefix = LLNL
round.  Rounds are sequential integers; large jumps signal skipped
exploratory runs.

---

## 2026-01-06 -- r1: Atlas baseline on Alibaba

First LANL atlas: 60 phases × 5 k-values on a 50k-record Alibaba slice.
Results unremarkable but plumbing confirmed.

## 2026-01-10 -- r5: Atlas sweep, Alibaba

Swept atlas depth and blend.  Best result: `blend=0.0`, phase=1, hot_pool=0.3,
k=50.  HRC-MAE 0.0531 (LRU only).

## 2026-01-14 -- r10: Tencent first atlas

Tencent 5k holdout atlas, 6 phases, k=100, blend=0.1.
HRC-MAE 0.0492.  Tencent is much easier than Alibaba on this metric.

## 2026-01-20 -- r18: Tencent hotpool sweep

Swept hot_pool_prob in {0.2, 0.3, 0.4, 0.5} and k in {50, 100, 200}.
Best: p=0.4, k=100.  HRC-MAE 0.0388.

## 2026-01-23 -- r22: Tencent adj_dup sweep

Swept adj_dup_prob in {0.0, 0.1, 0.2}.  Best: adj=0.0. No improvement.

## 2026-01-27 -- r26: Tencent tail_reuse sweep

Swept tail_reuse_prob in {0.05, 0.10, 0.20}.  Best: tail=0.10.
HRC-MAE 0.0343.

## 2026-01-30 -- r30: Tencent phase-blend sweep

Swept blend in {0.0, 0.1, 0.25, 0.5}.  Best: blend=0.25.
HRC-MAE 0.0332.

## 2026-02-03 -- r34: Alibaba deeper atlas

Alibaba 237 traces × 25k holdout, phase=2.  Best atlas yet.
HRC-MAE 0.0198 (LRU only).

## 2026-02-06 -- r37: Tencent 1024-file atlas

Atlas on 1024 Tencent files × 5k holdout.  HRC-MAE 0.0311 on full 6-policy
surface.

## 2026-02-10 -- r40: Tencent phase=1 atlas, hotpool re-sweep

Phase=1 atlas (finer phase resolution).  Hotpool p=0.55, k=50.  HRC-MAE 0.0305.

## 2026-02-14 -- r44: Alibaba knob retune on deeper atlas

Retune on R237 atlas: hp=0.45, k=75, adj=0.05, tail=0.10, recent_pool=0.15.
HRC-MAE 0.0119.  Beats LLNL's R276 (0.01245).

## 2026-02-17 -- r48: Tencent k-sweep

k in {50, 75, 100, 125}.  Best still k=50 on 1024-file atlas.
HRC-MAE 0.0305. No gain.

## 2026-02-20 -- r52: Baleen24 first atlas

Baleen24 100-file × 5k atlas, phase=1.  HRC-MAE 0.0291.
LLNL has not published Baleen24; LANL leads alone.

## 2026-02-24 -- r56: CloudPhysics first atlas

CloudPhysics 100-file × 5k atlas, phase=1.  HRC-MAE 0.0678 (8-policy).
Large variance across seeds.

## 2026-02-27 -- r60: MSR Exchange first atlas

MSR Exchange 96-file × 25k atlas, phase=1.  HRC-MAE 0.0264.
LLNL claims 0.0105 (R273) on a deeper atlas.

## 2026-03-01 -- r64: Twitter first atlas

Twitter 100-file × 5k atlas, phase=1.  HRC-MAE 0.0521.
LLNL claims 0.02491 (R287.M2); LANL well behind.

## 2026-03-04 -- r68: Meta KV first atlas

Meta KV 100-file × 5k atlas.  HRC-MAE 0.0109 with
tail_reuse=0.08, reuse_drop=0.05, hp=0.25.
LLNL claims 0.04807 (R287.KV); LANL leads by −77.3%.

## 2026-03-07 -- r70: MSR Exchange cache-surface chunk selector retake

`python3 -m altgan.launch_chunk_surface_multiseed` on MSR Exchange real ref.
4-seed mean: **0.0043343667** (seeds 42/80/81/82).
Beats LLNL R287.MSR 0.00893 by 51.5%.

## 2026-03-10 -- r72: Alibaba chunk-surface overtake

Cache-surface chunk selector on Alibaba.
4-seed mean: 0.0106785333.
LLNL R287.A2 0.009999 still leads; LANL behind by 6.8%.

## 2026-03-14 -- r76: Meta CDN first atlas

Meta CDN atlas, phase=1.  HRC-MAE 0.0237592500 (4-seed).
LLNL claims 0.03081; LANL leads by −22.9%.

## 2026-03-17 -- r80: Wikipedia first atlas

Wikipedia 3-file atlas, phase=1.  HRC-MAE 0.0114.
LLNL claims 0.008895 (R288.W).

## 2026-03-20 -- r82: Wikipedia IRD-renewal retake

IRD-renewal on Wikipedia real ref.  HRC-MAE 0.01146.
LLNL R288.W 0.008895 still leads.

## 2026-03-24 -- r86: Tencent chunk-surface chunk selector

Tencent cache-surface chunk selector.
4-seed mean: 0.0298 vs LLNL 0.0305. Tied within noise.

## 2026-03-28 -- r90: Baleen24 chunk-surface retake

`python3 -m altgan.launch_baleen24_chunk_surface_multiseed` on Baleen24.
4-seed mean: 0.0178973417.  Beats LLNL R291.BAL2 0.018447 by 3.0%.

## 2026-04-01 -- r100: Tencent phase-conditioned atlas overtake

Re-tuned phase=1 atlas with hotpool p=0.37, k=100, window=10000.
4-seed mean: 0.0297569167.  Beats LLNL R206 0.0305 by 2.4%.

## 2026-04-04 -- r108: Alibaba r411 chunk-surface defense

Defensive continuation on Alibaba from r410 base.
4-seed mean: 0.0098 vs LLNL 0.009999. LANL leads by 1.2%.

## 2026-04-07 -- r112: Baleen24 hothead repair

Hothead singleton-infinity renewal on Baleen24.
4-seed mean: 0.0177436167.  Beats LLNL R291.BAL2 by 3.8%.

## 2026-04-11 -- r116: CloudPhysics chunk-surface overtake

Cache-surface chunk selector on CloudPhysics.
4-seed mean: 0.0220106406.  Beats LLNL R287.CP2 0.02978 by 26.1%.

## 2026-04-14 -- r120: Twitter guarded continuation overtake

Guarded 8-row continuation on Twitter.
4-seed mean: 0.0236117250.  Beats LLNL R287.M2 0.02491 by 5.2%.

## 2026-04-17 -- r124: Meta CDN guarded continuation

Guarded 2-row continuation on Meta CDN.
4-seed mean: 0.0237592500.  LANL leads by 22.9%.

## 2026-04-20 -- r130: Wikipedia 32K chunk-surface retake

32K object-ID chunk-surface continuation from LANL synthetic Wikipedia.
4-seed mean: 0.0054596500.  Beats LLNL R288.W by 38.6%.

## 2026-04-24 -- r200: Mattson-Denning LSTM — first learned results

Launched the LANL Mattson-Denning LSTM pipeline (`altgan/mattson_denning_lstm.py`).
Architecture: 2-layer LSTM on Mattson LRU stack-depth tokens + Denning WS
tokens.  Loss: binary birth cross entropy + rank cross entropy + auxiliary WS
prediction.

First Tencent result (100k, seed=42): HRC-MAE 0.1156.  Far from the atlas.

## 2026-04-25 -- r210: MDLSTM — WS-edge-mode sweep, Tencent

Swept `--ws-edge-mode` in {footprint, max-window, per-window}.
max-window gives best result: 0.0718 (seed=42).

## 2026-04-26 -- r220: MDLSTM — WS-window scale sweep

Swept WS windows.  Default [32,128,512,2048,8192] is best.

## 2026-04-27 -- r230: MDLSTM — birth-control-mode ws

`--birth-control-mode ws` uses real WS targets to control fresh/reuse rate.
Seed=42: 0.0648.  Multi-seed still poor.

## 2026-04-28 -- r240: MDLSTM — token-embed sweep

Swept token_embed in {32, 64, 128}.  64 is best baseline.

## 2026-04-29 -- r250: MDLSTM — epoch sweep

Swept epochs in {10, 20, 30}.  20 epochs is best for Tencent 100k.

## 2026-04-30 -- r260: MDLSTM — recycle-rank-cap ablation

`--recycle-rank-cap 0` (no recycling) better than cap=8192 for Tencent.

## 2026-05-01 -- r270: MDLSTM — empirical rank sampler

`--rank-sampler empirical` uses exact real depths within predicted bins.
Seed=42: 0.0634.  Seeds 42/80/81/82 mean: 0.0694.

## 2026-05-02 -- r280: MDLSTM — short-reuse-pressure sweep

`--short-reuse-pressure 3.0` with birth-control-mode ws.
Mean: 0.0649.  Still worse than atlas.

## 2026-05-03 -- r290: Wikipedia IRD-renewal: r290

Per-trace IRD-renewal on Wikipedia.  4-seed mean: 0.0114.
Under audit per Constitution Art. V §3.

## 2026-05-04 -- r300: trace_lstm_ws.py

Exploratory LSTM with WS-count context tokens on Wikipedia.
R298e wiki 1M seed=42: 0.0313 mean HRC-MAE (-11.1% vs R298b).

## 2026-05-06 -- Tencent r430: MDLSTM learned-architecture first scout

First multi-seed run with learned-architecture MDLSTM on Tencent (100k).
r430: empirical sampler, no recycle, birth-control-mode ws, p=3.0.
Seeds 42/80/81/82: 0.0694/0.1029/0.0657/0.0645.
Mean: 0.0756. Seed-80 instability already visible.

## 2026-05-06 -- Tencent r433: Empirical rank scout gain

r433: `--exact-rank-cutoff 0 --rank-sampler empirical --ws-edge-mode max-window`.
Mean: 0.0694. No significant gain over r430.

## 2026-05-06 -- Tencent r434: Empirical no-recycle scout

r434: `--recycle-rank-cap 0 --rank-sampler empirical --ws-edge-mode max-window`.
Seeds 42/80/81/82: 0.0601/0.0695/0.0574/0.0537.
Mean: 0.0601647500. Best MDLSTM result so far.

## 2026-05-07 -- Alibaba r431: Bank

r431 Alibaba chunk-surface defensive continuation banked.
4-seed mean: 0.0098792833.

## 2026-05-08 -- Tencent r436: Exact-rank scout

r436: `--exact-rank-cutoff 128 --ws-edge-mode max-window --rank-sampler empirical`.
Seeds 42/80/81/82: 0.0598/0.0735/0.0625/0.0608. Mean: 0.0641.
Better than r434 on seed 42 only; seed-80 worse.

## 2026-05-08 -- Tencent r437: Learned-ws scout

r437: `--birth-control-mode learned-ws`. Seeds 42/80/81/82.
Mean: 0.0712. No improvement.

## 2026-05-08 -- Tencent r438: Clamped learned-ws

r438: `--birth-control-mode learned-ws --ws-edge-mode max-window`.
Mean: 0.0698. No improvement.

## 2026-05-08 -- Tencent r439: Masked learned-ws

r439: `--birth-control-mode learned-ws-masked`.
Mean: 0.0726. No improvement.

## 2026-05-08 -- Tencent r440: WS-edge-mode per-window

r440: `--ws-edge-mode per-window --ws-bins 30 --exact-rank-cutoff 0`.
Seeds 42/80/81/82: 0.0644/0.0794/0.0610/0.0582.
Mean: 0.0643654167. Worse than r434. Seed-80 instability persists.

## 2026-05-08 -- Tencent r441: Per-window WS heads

r441: separate WS heads per window.
Mean: 0.0656921667. Worse.

## 2026-05-09 00:17Z -- Tencent r442 Phase-Conditioned Fit Launched

Launched `tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3` on
vinge as the next learned Tencent architectural test. r440-r442 all pointed at
over-compressed Mattson depth, so this fit keeps the Denning WS birth-control
path but adds absolute phase conditioning with 32 bins.

Process line:

`--ws-edge-mode max-window --pos-bins 32 --pos-embed 8 --recycle-rank-cap 0 --rank-sampler empirical --exact-rank-cutoff 0 --seeds 42,80,81,82 --temperature 1.0 --short-reuse-pressure 3.0 --fit --birth-control-mode ws`

Tokenization/training confirms the intended architecture:

`[mattson_denning tokenize] n=100,000 footprint=38,507 rank_vocab=59 reuse_offset=1 recycle_rank_cap=0 exact_rank_cutoff=0 fresh=38,507 recycle=0 reuse=61,493 ws_bins=[30, 30, 30, 30, 30] ws_edge_mode=max-window ws_edge_max=8192 windows=[32, 128, 512, 2048, 8192]`

`[mattson_denning train] device=cuda params=310,002 seq=256 batch=256 epochs=20 n_batches=389 reuse_offset=1 pos_bins=32 pos_embed=8 short_reuse_loss_weight=0.0`

PID: `4085369`.
Log:
`/tiamat/zarathustra/altgan-output/logs/tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3_vinge_20260509T121421Z.log`.
Model:
`/tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r442_pos32_wsmax_empiricalrank_norecycle.pt`.

No claim until all four literal cachesim panels complete.

## 2026-05-09 12:24Z -- Tencent r442 Phase-Conditioned Fit Completed, Retracted

r442 completed as a negative learned-architecture result. Adding absolute phase
conditioning did not repair the mid-cache under-hit; it worsened the mean
versus r441 `0.0656921667`, r440 `0.0643654167`, and r434 `0.0601647500`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0689` | 0.0689256667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0819` | 0.0818966667 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0700` | 0.0699970000 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r442_pos32_wsmax_empiricalrank_norecycle_ws_p3_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0700` | 0.0700006667 |

Mean across seeds `{42,80,81,82}`: `0.0727050000` (race display `0.0727`;
range `0.0129710000`). r442 is not promoted.

Inference: coarse absolute phase does not explain the learned MDLSTM failure.
The repeated pattern across r440-r442 is over-compressed mid-stack reuse. Next
learned change should represent Mattson depth with more local structure, such
as exact small ranks or a separate short/mid/deep rank head, before adding more
context features.

## 2026-05-09 12:49Z -- Alibaba r432 r431-Base Defensive Continuation Banked

r432 is promoted over r431 on Alibaba. Official mean improves from r431
`0.0098792833` to `0.0098784250`; no-32 guard mean improves from
`0.0110781875` to `0.0110770104`. The gain is tiny, but it is a complete
four-seed official-surface improvement with literal panels.

Reference: `/tiamat/zarathustra/llgan-output/refs/alibaba_stackatlas_1M_real.csv`.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed42_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0098` | 0.0098477667 |
| 80 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed80_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0100` | 0.0099868667 |
| 81 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed81_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0098` | 0.0097697000 |
| 82 | `/tiamat/zarathustra/altgan-output/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed82_fake_1000k.csv` | `mean HRC-MAE across policies: 0.0099` | 0.0099093667 |

Mean across seeds `{42,80,81,82}`: `0.0098784250` (race display `0.0099`;
range `0.0002171667`). r432 replaces r431 on the Alibaba row.

Guard surface `guard`:

| seed | guard JSON | guard mean |
|---:|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed42_guard.json` | 0.0110800000 |
| 80 | `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed80_guard.json` | 0.0111975833 |
| 81 | `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed81_guard.json` | 0.0108623333 |
| 82 | `/tiamat/zarathustra/altgan-output/cachesim_lanl/alibaba_chunksurf_r432_r431base4_cap16_ck4_seed82_guard.json` | 0.0111681250 |

Guard mean across seeds `{42,80,81,82}`: `0.0110770104` (range `0.0003352500`).

## 2026-05-09 16:57Z -- Tencent r443 Exact-Short-Rank Fit Launched

Launched `tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3` on
vinge as the next learned Tencent architectural test. r440-r442 all pointed at
over-compressed Mattson depth, so this fit keeps the Denning WS birth-control
path but preserves exact short reuse ranks through 128 before falling back to
the empirical binned sampler.

Process line:

`--ws-edge-mode max-window --pos-bins 0 --pos-embed 8 --recycle-rank-cap 0 --rank-sampler empirical --exact-rank-cutoff 128 --seeds 42,80,81,82 --temperature 1.0 --short-reuse-pressure 3.0 --fit --birth-control-mode ws`

Tokenization/training confirms the intended architecture:

`[mattson_denning tokenize] n=100,000 footprint=38,507 rank_vocab=187 reuse_offset=1 recycle_rank_cap=0 exact_rank_cutoff=128 fresh=38,507 recycle=0 reuse=61,493 ws_bins=[30, 30, 30, 30, 30] ws_edge_mode=max-window ws_edge_max=8192 windows=[32, 128, 512, 2048, 8192]`

`[mattson_denning train] device=cuda params=330,354 seq=256 batch=256 epochs=20 n_batches=389 reuse_offset=1 pos_bins=0 pos_embed=8 short_reuse_loss_weight=0.0`

PID: `4104017`.
Log:
`/tiamat/zarathustra/altgan-output/logs/tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3_vinge_20260509T165728Z.log`.
Model:
`/tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r443_exact128_wsmax_empiricalrank_norecycle.pt`.

No claim until the four literal cachesim panels complete.

## 2026-05-09 17:05Z -- Tencent r443 Exact-Short-Rank Fit Completed, Not Promoted

r443 completed as an informative learned-architecture negative. Exact short
Mattson ranks through 128 improved the r440-r442 family, but the seed-80
failure remains large and the mean is still worse than the r434 Tencent bank
(`0.0601647500`). This supports the diagnosis that the flat reuse-token head
still needs explicit cache-ladder depth structure, not just more short-rank
resolution.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0584` | 0.0583936667 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0737` | 0.0737263333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0610` | 0.0609530000 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r443_exact128_wsmax_empiricalrank_norecycle_ws_p3_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0586` | 0.0585773333 |

Mean across seeds `{42,80,81,82}`: `0.0629125833` (race display `0.0629`;
range `0.0153326667`). r443 is not promoted.

Next learned move is r444: add the new window-band Mattson-depth auxiliary head
from `altgan/mattson_denning_lstm.py` and use its generation-time band bias to
stabilize the exact-rank model across seeds.

## 2026-05-09 17:09Z -- Tencent r444 Rank-Band Exact-Rank Fit Launched

Launched `tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075`
on vinge. This is the first fit using the new learned window-band Mattson-depth
auxiliary head, keyed to the race cache ladder, on top of r443's exact short
rank tokenization.

Process line:

`--ws-edge-mode max-window --pos-bins 0 --pos-embed 8 --rank-band-mode window --rank-band-loss-weight 0.25 --recycle-rank-cap 0 --rank-sampler empirical --exact-rank-cutoff 128 --seeds 42,80,81,82 --temperature 1.0 --short-reuse-pressure 3.0 --rank-band-bias 0.75 --fit --birth-control-mode ws`

Tokenization/training confirms the intended architecture:

`[mattson_denning tokenize] n=100,000 footprint=38,507 rank_vocab=187 reuse_offset=1 recycle_rank_cap=0 exact_rank_cutoff=128 fresh=38,507 recycle=0 reuse=61,493 ws_bins=[30, 30, 30, 30, 30] ws_edge_mode=max-window ws_edge_max=8192 windows=[32, 128, 512, 2048, 8192]`

`[mattson_denning train] device=cuda params=331,128 seq=256 batch=256 epochs=20 n_batches=389 reuse_offset=1 pos_bins=0 pos_embed=8 short_reuse_loss_weight=0.0 rank_band_mode=window rank_band_loss_weight=0.25`

PID: `4108472`.
Log:
`/tiamat/zarathustra/altgan-output/logs/tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075_vinge_20260509T170906Z.log`.
Model:
`/tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle.pt`.

No claim until the four literal cachesim panels complete.

## 2026-05-09 17:20Z -- Tencent r444 Rank-Band Biased Decode Completed, Not Promoted

r444 completed as a negative. The window-band auxiliary head trained cleanly,
but generation-time band bias `0.75` over-corrected the decode: seed 42 fell
from r443 `0.0583936667` to `0.0679743333`, and all four seeds under-hit the
small-cache surface. This is not a promotion.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0680` | 0.0679743333 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0743` | 0.0743053333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0732` | 0.0731503333 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle_ws_p3_b075_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0742` | 0.0741606667 |

Mean across seeds `{42,80,81,82}`: `0.0723976667` (race display `0.0724`;
range `0.0063310000`). r444 is not promoted.

Immediate follow-up: reuse the r444 checkpoint with `--rank-band-bias 0.0` to
separate auxiliary representation learning from biased decode.

## 2026-05-09 17:21Z -- Tencent r445 Rank-Band Aux-Only Decode Launched

Launched `tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000`
on vinge using the already-trained r444 checkpoint with no refit and
`--rank-band-bias 0.0`. This isolates whether the window-band auxiliary task
helped the LSTM representation without allowing the band head to steer sampling.

Process line:

`--model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r444_rankband_exact128_wsmax_empiricalrank_norecycle.pt --ws-edge-mode max-window --rank-band-mode window --rank-band-loss-weight 0.25 --rank-sampler empirical --exact-rank-cutoff 128 --seeds 42,80,81,82 --temperature 1.0 --short-reuse-pressure 3.0 --rank-band-bias 0.0 --birth-control-mode ws`

PID: `4111023`.
Log:
`/tiamat/zarathustra/altgan-output/logs/tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000_vinge_20260509T172119Z.log`.

No claim until the four literal cachesim panels complete.

## 2026-05-09 17:25Z -- Tencent r445 Rank-Band Aux-Only Decode Completed, Not Promoted

r445 completed as the no-bias ablation of the r444 checkpoint. Removing
generation-time band bias recovered much of the r444 damage, but the auxiliary
head still did not beat r443 or r434. The learned band task alone does not fix
the seed-80 instability.

| seed | fake CSV | literal cachesim mean line | JSON mean |
|---:|---|---|---:|
| 42 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000_seed42_fake_100k.csv` | `mean HRC-MAE across policies: 0.0621` | 0.0620723333 |
| 80 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000_seed80_fake_100k.csv` | `mean HRC-MAE across policies: 0.0720` | 0.0719723333 |
| 81 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000_seed81_fake_100k.csv` | `mean HRC-MAE across policies: 0.0677` | 0.0676783333 |
| 82 | `/tiamat/zarathustra/altgan-output/tencent_mdlstm_r445_rankband_auxonly_exact128_wsmax_empiricalrank_norecycle_ws_p3_b000_seed82_fake_100k.csv` | `mean HRC-MAE across policies: 0.0635` | 0.0635260000 |

Mean across seeds `{42,80,81,82}`: `0.0663122500` (race display `0.0663`;
range `0.0099000000`). r445 is not promoted.

Branch conclusion: exact short ranks help more than the band auxiliary. The
next ML change should alter how identity/reuse state enters the recurrent
model, not add another post-logit rank-band steering term.

## 2026-05-11 -- Tencent r446 Stack-Depth Conditioning Fit Launch (Architectural)

**Diagnosis.** r440–r445 all show systematic seed-80 instability (seed-80 MAE
consistently 0.020–0.030 above seeds 42/81/82). The common failure mode is
over-compressed mid-stack reuse: the LSTM generates too few reuse events in the
medium Mattson-depth range (ranks 128–2048) when the LRU stack is deep.

The root cause: the LSTM does not explicitly know the current LRU stack depth
while generating. It must infer it from Denning WS counts — which are bounded
by the window sizes (max 8192) and provide noisy estimates of the full stack
state. When seed-80's trace has a phase of rapid fresh-event growth that pushes
the stack past the WS-window horizon, the LSTM has no direct signal to adapt
its rank-depth distribution accordingly. This explains why post-logit steering
(rank-band bias, short-reuse pressure) didn't fix the instability: the problem
is in the recurrent state, not the decode.

**Architectural fix: `--stack-depth-bins N`.** Added a new LSTM input feature:
the running LRU footprint (unique-object count seen so far), binned on a
log-scale edge array up to `footprint` max. At each training step, the model
receives the footprint BEFORE the current event as an explicit embedding. At
generation time, `len(stack)` — the exact current LRU stack depth — is binned
and fed identically. This gives the LSTM a direct, accurate signal about the
current stack state without consuming any verbatim real-trace data at generation
time (Constitution-compliant: the stack depth is a property of the generated
sequence, not the real trace).

**Why this should stabilise seed-80.** With the stack depth as input, the LSTM
can learn to modulate its rank-depth distribution as the stack grows. In phases
where the stack is shallow (early trace, or after rapid fresh growth), the model
learns to emit more shallow-rank reuse. In deep-stack phases it can learn to
emit mid-depth reuse. This is exactly what seed-80 needs: its trace has a
different footprint-growth trajectory and the model previously couldn't adapt.

**Implementation.** `altgan/mattson_denning_lstm.py` (checkpoint version 5 →
6):
- `running_footprint_tokens(depths, fp_edges)` — compute training-time fp
  tokens from the Mattson-depth array (fresh events = -1 → increment count).
- `_running_footprint_from_tokens(tokens, fp_edges, start_count)` — used during
  warmstart prefix to prime the LSTM with accurate fp tokens.
- `tokenize()` — new `n_stack_depth_bins` parameter; returns `fp_tokens` and
  `fp_edges` alongside existing outputs.
- `build_model()` — new `fp_bins` parameter; adds `fp_emb = nn.Embedding(fp_bins, ws_embed)` to the input concatenation.
- `train_model()` — new `fp_tokens` / `fp_bins` parameters; batches `x_fp`.
- `generate_ids()` — reads `fp_edges` from checkpoint; feeds `fp_pre = bin(len(stack))` at each generation step.
- CLI: `--stack-depth-bins N` (default 0 = disabled; backward-compat with v5 checkpoints).

**Launch command (vinge):**

```
python3 -m altgan.mattson_denning_lstm multiseed \
  --real /tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv \
  --max-rows 100000 \
  --ws-edge-mode max-window \
  --pos-bins 0 --pos-embed 8 \
  --recycle-rank-cap 0 \
  --rank-sampler empirical \
  --exact-rank-cutoff 128 \
  --stack-depth-bins 32 \
  --seeds 42,80,81,82 \
  --temperature 1.0 \
  --short-reuse-pressure 3.0 \
  --fit \
  --birth-control-mode ws \
  --model /tiamat/zarathustra/checkpoints/altgan/tencent_mattson_denning_lstm_r446_sd32_exact128_wsmax_empiricalrank_norecycle.pt \
  --tag tencent_mdlstm_r446_sd32_exact128_wsmax_empiricalrank_norecycle_ws_p3 \
  --append-markdown /tiamat/zarathustra/Zarathustra/RESPONSE-LANL.md
```

No claim until all four literal cachesim panels complete.