# LLGAN Version History

All runs use oracle_general Tencent Block 2020 1M corpus (3234 files) unless noted.

## Completed Runs

| Version | Best MMD² | Best Recall | Best Epoch | Notes |
|---------|-----------|-------------|------------|-------|
| v3–v8 | ~0.07+ | 0.022–0.143 | varies | Pre-latent-AE; mode collapse |
| v9 | ~0.025 | 0.351 | ~60 | First with latent AE + supervisor + feature matching |
| v13 | **0.00818** | 0.455 | 130 | 3234 files, multi-file streaming; 171 wasted epochs (GAN cycling) |
| v14 | 0.041 | 0.618 | 30 | Collapsed epoch ~100 (W→31, recall→0) — see v14 notes |

Checkpoint locations (all on wigner):
- v9: ~/Zarathustra/llgan/checkpoints/tencent_v9/best.pt
- v13: /Volumes/Archive/Traces/checkpoints/tencent_v13/best.pt
- v14: /Volumes/Archive/Traces/checkpoints/tencent_v14/best.pt (epoch 30)
- v14c: /Volumes/Archive/Traces/checkpoints/tencent_v14c/ (in progress)

---

## In Progress

### v14c (hot-start from v14 epoch 30, 2026-03-28)

Hot-start from tencent_v14/epoch_0030.pt (MMD2=0.041, recall=0.618).

**What changed vs v14:**
- **Spectral norm on critic LSTM** (weight_ih_l0, weight_hh_l0) — v14 had SN only on
  the FC output layer; LSTM drifted unconstrained, W went to 31, recall to 0 at epoch 100.
- **Lower LR**: lr_g=5e-5, lr_d=2.5e-5 (50% reduction on lr_d)
- **n_critic=1** (was 3): one critic step per generator step to slow W drift
- **lr_cosine_decay=0.02** (was 0.05): decays to 2% of initial LR

**v14 notes (what went wrong):**
- WGAN-SN without LSTM SN: only FC output layer was Lipschitz-constrained
- W distance grew 2.7 to 16.4 by epoch 36, drifted to ~31 by epoch 100
- Recall hit 0.000 at epoch 100 — complete mode collapse
- Best checkpoint was epoch 30 (first evaluation): MMD2=0.041, recall=0.618

---

## Planned

### v15 (NVIDIA GPU, ETA ~2026-03-30)

New capabilities enabled by CUDA (not available on MPS):
- --amp: AMP fp16 gives 2-3x speed
- --compile: torch.compile gives ~20-40% additional speedup
- --cross-cov-loss-weight 1.0: full d×d lag-1 cross-feature covariance matching.
  DMD-GEN estimates linear dynamics operator A from the cross-covariance matrix; previous
  losses only matched the diagonal (per-feature ACF). Main DMD-GEN hypothesis.
- --diversity-loss-weight 1.0: MSGAN loss — maximises |G(z1)-G(z2)|/|z1-z2|; combats
  beta-recall mode collapse; requires 2nd G forward pass per step.
- --sn-lstm (now default): spectral norm on critic LSTM (fixed in v14c)

---

## Key Metrics (v13 baseline targets)

| Metric | v13 | Target | Description |
|--------|-----|--------|-------------|
| MMD2 | 0.00818 | < 0.005 | Kernel distribution distance |
| beta-recall | 0.455 | > 0.7 | Coverage: fraction of real modes covered |
| alpha-precision | 0.812 | > 0.85 | Fidelity: fraction of generated that is realistic |
| DMD-GEN | 0.771 | < 0.3 | Temporal dynamics divergence (0 = perfect) |

---

## What We Have Tried and Learned

### Architectural choices
- **GRU for AE/supervisor, LSTM for G+C**: GRU is simpler + faster for compression/prediction;
  LSTM cell state needed for G's long-range workload context and C's burst detection.
- **Split noise (z_global + z_local)**: z_global maps to LSTM h0/c0 (workload identity);
  z_local feeds per-step input (event noise). Without this, every window is independent.
- **Latent AE (TimeGAN/SeriesGAN)**: Direct feature-space GAN is unstable in a 5-feature
  correlated space spanning 10 decades. AE reduces to smooth 24-dim latent space.
- **Attention pooling in critic**: Mean pooling dilutes burst spikes. Attention up-weights
  the most discriminative timesteps.
- **Minibatch std in critic (StyleGAN2)**: Collapsed G produces identical rows, std near 0.
  This channel gives the critic a diversity signal before collapse happens.

### Training stability
- **sn_lstm=True (v14c)**: SN on LSTM weight matrices essential. SN on FC output only
  allows LSTM to drift; W grows unboundedly and training collapses.
- **n_critic=1 vs 3**: More critic steps per G step causes faster W drift on MPS.
- **Cosine LR annealing**: lr_cosine_decay=0.02-0.05 damps GAN cycling at 200+ epochs.
- **supervisor_loss_weight=5.0** (halved from 10.0): eta=10 with 100 pretrain G epochs
  caused trivial latent collapse. eta=5 with 50 pretrain epochs avoids it.
- **pretrain_g_epochs=50** (halved from 100): longer warm-up increases collapse risk.

### Losses
- **L_ACF**: Lag-1..5 autocorrelation matching. Effective for temporal structure.
- **L_loc**: Stride-repetition rate (fraction of delta-obj_id repeats). Captures
  sequential-access patterns that ACF misses.
- **L_FFT**: Frequency-domain matching. Low weight (0.05); complements ACF.
- **L_V (moment)**: Per-feature mean+std. Cheap, always-on baseline.
- **L_Q (quantile)**: p50/90/95/99 matching. Helps with heavy-tailed distributions.
- **L_FM (feature matching)**: Critic pooled-hidden-state MSE between real and fake.
  Smoother training signal than Wasserstein alone; reduces mode collapse.
- **L_cov (cross-covariance, planned v15)**: Full d×d lag-1 cross-feature covariance.
  Directly targets DMD-GEN (the linear dynamics operator is estimated from this matrix).
- **L_div (diversity, planned v15)**: MSGAN mode-seeking. Not yet tested.

---

### v14d (hot-start from v14 epoch 30, 2026-03-28 23:13)

Hot-start from tencent_v14/epoch_0030.pt (MMD2=0.041, recall=0.618).
Killed v14c after 44-epoch regression (epoch 30 best 0.046 → epoch 70: 0.062, recall 0.140).

**What changed vs v14c:**
- **n_critic=2** (was 1): v14c's G loss was persistently +2 to +4, meaning the critic
  dominated. With W max of 0.09 over 50 epochs, SN-LSTM proved we have headroom for more
  critic steps. n_critic=2 gives G stronger gradients to learn from.
- Everything else identical to v14c.

**v14c notes (what went wrong):**
- n_critic=1 was too conservative: critic stayed too strong relative to G
- G loss persistently positive (+2-4): critic correctly identified fakes but G couldn't learn
- Recall declined 0.269 (ep30) -> 0.140 (ep70) over 40 epochs: slow mode narrowing
- No W drift (max 0.09), no collapse — just stuck in local optimum
- Best epoch was 30: MMD2=0.046, recall=0.269

---

### v14d post-mortem (killed epoch 21, 2026-03-29)

Mode collapse immediately: recall 0.046-0.057 across epochs 5-20, MMD2 0.317-0.440.
n_critic=2 was too aggressive for a hot-start from a non-SN-LSTM checkpoint (v14/epoch_0030.pt).
The extra critic step caused the critic to overpower the generator before it could adapt.
SN u/v vectors were freshly initialized on old weights; 2 critic steps per G step too fast.
Killed at epoch 21.

### v14e (hot-start from v14 epoch 30, 2026-03-29, n_critic=1)

Back to v14c config (n_critic=1). The v14c 40-epoch stagnation was oscillation, not collapse.
v14c had recall 0.157-0.269 at epochs 5-30 -- vastly better than v14d's 0.044-0.057.
With 380 epochs remaining, n_critic=1 gives the best chance to break past 0.046 MMD2.

---

### v14e post-mortem (killed epoch 31, 2026-03-29)

Worse than v14c at every eval: epoch 30 MMD2=0.077/recall=0.209 vs v14c 0.046/0.269.
G loss escalated to +8-10 by epoch 17, W spike +0.63 at epoch 30.
Same root cause as v14c stagnation: hot-start from v14/epoch_0030.pt loads generator
weights tuned for unconstrained critic; fresh SN-LSTM normalization creates asymmetry.

### v14f (hot-start from v14c/epoch_0030.pt, 2026-03-29)

Key changes vs v14c/v14e:
- **Resume from tencent_v14c/epoch_0030.pt** (not v14/epoch_0030.pt). The v14c checkpoint
  has SN-LSTM u/v buffers already converged. Avoids the fresh-SN-on-old-weights asymmetry.
- **diversity-loss-weight=0.5**: MSGAN mode-seeking loss. Directly combats the recall decline
  seen in v14c (0.269->0.140 over 44 epochs). L_div = -|G(z1)-G(z2)| / |z1-z2| penalises
  the generator for producing similar outputs across different noise inputs.
- n_critic=1, all other hyperparameters identical to v14c.

Starting from v14c best: MMD2=0.046, recall=0.269.

---

### v14f post-mortem (killed epoch 190, 2026-03-29)

Stable but plateaued. Best: epoch 50 MMD2=0.048, recall=0.234 (training eval with EMA, 1000 samples).
Full eval.py on v14c/epoch_0030.pt revealed the real picture:
  MMD2=0.048, alpha-precision=0.489, beta-recall=0.021 (!!), DMD-GEN=0.703
  Context-FID=0.07 (good), AutoCorr=0.058, reuse-rate fake=0.000 vs real=0.124
Key finding: beta-recall in training log (0.269) vs eval.py (0.021) diverge because
training uses EMA weights + 1000 samples; full eval uses live checkpoint + 2000 samples.
Reuse rate=0 despite L_loc loss: generator never repeats obj_id within window.

GAN cycling pattern (underdamped oscillation) prevented sustained improvement.
W occasionally spiked to 1.19 (epoch 174) but SN-LSTM always recovered.
Killed to start fresh training.

### v14g (fresh from scratch, 2026-03-29, ~08:30)

First CLEAN run with SN-LSTM from epoch 0. All prior hot-starts carried generator weights
tuned for an unconstrained critic, creating a permanent asymmetry. Key changes:
- Fresh pretraining: AE 50ep + Supervisor 50ep (supervisor_steps=2) + G warmup 50ep
- supervisor_steps=2: 2-step temporal prediction (SeriesGAN) instead of 1-step
- cross_cov_loss_weight=0.5: full d*d lag-1 cross-feature covariance matching (DMD-GEN fix)
- diversity_loss_weight=0.5: MSGAN mode-seeking (recall fix)
- lr_g=1e-4, lr_d=5e-5 (back to original v13 LRs; hot-starts used 5e-5/2.5e-5)
- lr_cosine_decay=0.05 (faster decay than v14f's 0.02)
- Everything else same as v14f

### New: Patch embedding in critic (2026-03-29)

Implemented Conv1d(lstm_input, hidden_size, kernel=stride=3) before critic LSTM.
Folds 12-step window into 4 patch tokens. TTS-GAN (AIME 2022) style.
Enabled with --patch-embed flag (off by default; v14g does NOT use it yet).
Will test in v15 on NVIDIA where we can run full hyperparameter search.

---

## v14g Summary (completed 2026-03-30)

Best full eval (ep90): MMD²=0.018, α-precision=0.910, β-recall=0.372, DMD-GEN=0.700, Context-FID=0.03
Training eval peak (ep395): MMD²=0.113, β-recall=0.469 (EMA+1000 samples — ~3-4× optimistic vs full eval)
Checkpoint: /Volumes/Archive/Traces/checkpoints/tencent_v14g/best.pt (ep90)

**Root causes identified from generation test:**
- **reuse_rate=0.000**: delta-encoding makes obj_id reuse (delta=0) a zero-measure event
  in continuous space. Generator never outputs exactly 0.0 in the normalized delta space.
- **DMD-GEN stuck at 0.700**: cross_cov loss matches C (numerator) but not A=C·Σ⁻¹.
  Supervisor loss weight (5.0) overpowers cross_cov weight (0.5) by 10×.
- **obj_size non-quantized**: Generated sizes are continuous; real traces have discrete
  multiples of 4096 bytes.

---

## v15 (vinge/GB10 CUDA, started 2026-03-30)

First run on NVIDIA GB10 with AMP + torch.compile (2-3× speedup expected).

**P0 architectural fixes from v14g root cause analysis:**

1. **obj_id locality split** (dataset.py): Replace delta-encoded obj_id (single float)
   with two features:
   - `obj_id_reuse` (±1): +1 if same object as previous (delta=0), -1 otherwise.
     Generator can now learn reuse as a binary classification target.
   - `obj_id_stride` (signed-log delta): Seek magnitude for non-reuse accesses (0 when reuse).
   This is the direct fix for reuse_rate=0: the generator had to output exactly 0.0 in
   continuous space to produce a reuse — a zero-measure event. Now it classifies ±1.

2. **obj_size quantization** (dataset.py): Snap to nearest 4096-byte multiple before
   log-transform. Concentrates the distribution onto real discrete support.

**CUDA-only features enabled:**
- AMP fp16 (2-3× speedup)
- torch.compile (additional ~20-40%)
- 6 input features (was 5): ts, obj_size, opcode, tenant, obj_id_reuse, obj_id_stride

**Training command:**
```
~/llgan-env/bin/python train.py \
  --trace-dir ~/traces/tencent_block_1M --fmt oracle_general \
  --epochs 600 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir ~/checkpoints/tencent_v15 --checkpoint-every 10 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 60 \
  --locality-loss-weight 1.0 --acf-loss-weight 0.2 \
  --moment-loss-weight 0.1 --fft-loss-weight 0.05 \
  --quantile-loss-weight 0.2 --feature-matching-weight 1.0 \
  --cross-cov-loss-weight 0.5 --ema-decay 0.999 \
  --lr-cosine-decay 0.05 --grad-clip 1.0 --n-critic 3 \
  --hidden-size 256 --latent-dim 24 \
  --pretrain-ae-epochs 50 --pretrain-sup-epochs 50 --pretrain-g-epochs 100 \
  --supervisor-loss-weight 5.0 --lr-g 1e-4 --lr-d 5e-5
```
Log: vinge:~/train_v15.log
PID: 13560 (started 2026-03-30)
