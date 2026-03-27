# Zarathustra — TODO

Improvements derived from reading DiTTO, TTS-GAN, Diffusion-TS, NetDiffus, SeriesGAN, and
TransFusion, plus peer code review. Ordered by impact-to-effort ratio. No image-conversion
hacks — solid theory only.

Items marked ✓ are done and in the repo.

---

## Bugs fixed

- ✓ **Cross-file boundary contamination** (`train.py` `_load_epoch_dataset`)
  Concatenating raw arrays across files then windowing created sequences spanning
  last-event-of-file-A → first-event-of-file-B. Fixed: per-file `TraceDataset`,
  merged with `ConcatDataset`. No window ever crosses a file boundary.

- ✓ **Timestamp inverse_transform off-by-one** (`dataset.py`)
  `concatenate([[start], deltas[:-1]]).cumsum()` shifted the series by one position
  and duplicated the first timestamp. Fixed: `start + cumsum(deltas)`.

- ✓ **Loss named "wgan-gp" without gradient penalty** (`config.py`, `train.py`, `model.py`)
  Spectral norm applied only to the critic's final linear layer — LSTM weights are
  unconstrained. Renamed to `wgan-sn` throughout. True WGAN-GP requires second-order
  autograd (CUDA only; deferred to NVIDIA GPU task).

---

## Immediate (hours each)

- ✓ **Log-transform `ts_delta` and `obj_size`** (`dataset.py`)
- ✓ **Fourier spectral auxiliary loss** (`train.py`, `--fft-loss-weight`)
- ✓ **Moment matching loss L_V** (`train.py`, `--moment-loss-weight`)

- ✓ **Add coverage (β-recall) and α-precision metrics** to evaluation (`eval.py`)
  MMD² alone cannot detect mode collapse. Implemented full PRDC evaluation in `eval.py`
  (uses `prdc` package with numpy fallback).
  v7: MMD²=0.013, recall=0.143 (mode collapse).
  v8: lowering supervisor weight made collapse worse (recall=0.022).
  v9: critic feature matching (Salimans et al. 2016) + supervisor=10.0 + noise_dim=32.
  v9 epoch 50: MMD²=0.020, recall=0.351 ✓ — first model to exit mode-collapse zone.

- ✓ **Early stopping on best combined metric** (`train.py`)
  After 50% of training epochs, save the checkpoint with best
  `discriminative_score + λ * moment_loss` rather than the final epoch.
  SeriesGAN ablation: accounts for 43% of their discriminative improvement alone.

- ✓ **Fix generate.py: carry LSTM hidden state across window boundaries**
  Previously sampled IID windows and concatenated — hidden state reset every 12 steps,
  so long-range burst structure was incoherent. Fixed: `z_global` is fixed per stream;
  `(h_n, c_n)` passes from each window to the next. `--n-streams` for parallel traces.

---

## Medium-term (days each)

- ✓ **Fix latent design: global z + per-step innovations** (`model.py`, `train.py`)
  `z_global (batch, noise_dim)` projects via Linear → LSTM h0/c0 (workload identity).
  `z_local (batch, timestep, noise_dim)` feeds per-step innovation noise.
  Result: MMD² 0.093 at epoch 15 vs v3 best of 0.134 at epoch 55 — 30% improvement.

- ✓ **Replace raw `obj_id` regression with locality-aware representation** (`dataset.py`)
  Delta-encode obj_id without clipping, then sign(d)*log1p(|d|) to compress 10-decade
  range. Encodes: 0=repeat access, small±=sequential/strided, large±=random jump.
  Inverse: undo signed-log, then cumsum. (v5 training uses this.)

- ✓ **Replace critic mean-pooling with attention pooling** (`model.py`)
  Learned softmax-weighted sum over LSTM hidden states replaces `h.mean(dim=1)`.
  Critic can now focus on burst / regime-change timesteps instead of averaging them away.

- [ ] **Add Context-FID as primary evaluation metric** (`eval.py`)
  Train a fixed LSTM encoder on real data; embed real and generated windows; compute
  Fréchet distance in that latent space. More sensitive to temporal structure than MMD².

- ✓ **Implement latent autoencoder + supervisor (TimeGAN/SeriesGAN architecture)**
  Four-phase curriculum: (1) AE pretrain E+R, (2) supervisor pretrain S,
  (2.5) **generator warm-up** — train G via supervisor consistency only before
  introducing the critic (TimeGAN §3.3, missing from v6 → v6 completely failed),
  (3) joint GAN in latent space with supervisor consistency loss.
  Generator outputs latents [0,1]; Recovery decodes to features [-1,1].
  Supervisor loss: MSE(G(z)[:,1:,:], S(G(z))[:,:-1,:]) — temporal coherence.
  v6 stuck at MMD²≈0.97 — critic saturated before G warmed up. Fixed in v7.

- [ ] **Add dual discriminators (latent + feature space)** — *depends on latent AE above*
  Second lightweight critic on raw decoded output. Latent critic = stable early gradients;
  feature critic = catches artifacts latent critic misses.

- [ ] **Add patch embedding to critic** (`model.py`)
  `Conv1d(num_cols, embed_dim, kernel_size=3, stride=3)` before LSTM critic compresses
  12-step window into 4 patch tokens. (TTS-GAN)

- [ ] **Add workload conditioning** (`model.py`, `train.py`)
  Embed `c = [tenant_id, read_ratio, obj_size_bucket]` via MLP into generator hidden
  state and at every timestep. Classifier-free guidance style for unconditional fallback.

- [ ] **Improve validation** (`train.py`, `mmd.py`)
  Current MMD val set is 2 random files (may overlap conceptually with training) and
  MMD is flattened-window RBF — weak signal. Switch to: designated held-out files,
  replay/locality metrics (stack distance CDF, reuse distance), per-column KS tests.

---

## New ideas from literature survey (2024–2025)

### Quick wins (< 1 day each)

- [ ] **FIDE frequency inflation in FFT loss** (`train.py`) — NeurIPS 2024
  Upweight high-frequency bins in the existing FFT loss to prevent rare/extreme
  I/O events (burst opcodes, large obj_size) from being washed out.
  `weight_f = 1 + α * (|freq_f_real| / mean(|freq_real|))`. ~10 lines.

- [ ] **Adaptive Gradient Penalty (AGP)** (`train.py`) — MDPI Math 2025
  Replace fixed `gp_lambda=10` with a PI controller that adjusts λ dynamically
  based on `‖∇D‖ − 1`. Removes a fragile hyperparameter.
  `λ_{t+1} = λ_t + Kp*(‖∇D‖ − 1) + Ki*Σ(‖∇D‖ − 1)`. ~20 lines.

- [ ] **R2 regularization on fake samples** (`train.py`) — R3GAN, NeurIPS 2024
  Already have R1 via WGAN-GP (penalty on interpolated points). Add R2:
  zero-centered gradient penalty on fake samples.
  `L_R2 = λ₂ * E[‖∇_x̃ D(x̃)‖²]`. Improves mode coverage with convergence
  guarantees R1 alone can't provide. ~5 lines.

### Medium-term (1–2 days each)

- [ ] **ChronoGAN slope + skewness TS loss terms** (`train.py`) — ICMLA 2024
  Add slope (linear regression of each sequence over time) and skewness to the
  TS loss alongside existing moment matching. Skewness directly penalises
  mismatch in heavy-tailed distributions (obj_size). GRU+LSTM hybrid backbone
  is a further option but 2-day effort.

- [ ] **SeriesGAN dual discriminator + 2-step supervisor** (`model.py`, `train.py`)
  — BigData 2024 (also in Medium-term section above)
  2-step supervisor predicts `h_t` from `h_{t-2}` instead of `h_{t-1}`.
  Expected: −34% discriminative score vs TimeGAN. Already have latent AE so
  mostly wiring.

- [ ] **Relativistic paired GAN loss (RpGAN)** (`train.py`) — R3GAN, NeurIPS 2024
  Discriminator simultaneously judges "is real more real than fake?" and vice
  versa in a single forward pass. Shown to achieve full mode coverage on
  StackedMNIST where standard WGAN misses modes. Replace current WGAN objective.

- [ ] **FiLM conditioning for tenant/workload** (`model.py`) — survey, 2024
  Feature-wise Linear Modulation: learn per-condition scale+shift for generator
  hidden states. Cleaner than concatenation for conditioning on tenant_id,
  read_ratio, obj_size_bucket. Prerequisite for workload conditioning (#9 above).

### Evaluation / diagnostics

- ✓ **DMD-GEN temporal dynamics metric** (`eval.py`) — NeurIPS 2025 (arXiv 2412.11292)
  Code not yet released — implemented from paper. Runs Dynamic Mode Decomposition
  on real vs generated batches, compares eigenvector subspaces via Grassmann
  principal angles + 1-Wasserstein OT. Catches wrong autocorrelation/burst
  structure that MMD² misses. `dmdgen()` in eval.py.

- [ ] **Simulation-based evaluation** — JSSPP 2024 empirical study
  Run synthetic traces through a storage/cache simulator; compare queue depths,
  I/O latency CDFs, cache hit rates. Operational fidelity ≠ statistical fidelity
  — model may look good on MMD² but behave differently in a real simulator.

---

## Longer-term (architecture)

- [ ] **Build hierarchical two-level generator**
  `timestep=12` captures micro-patterns but not workload phases. Coarse model over
  burst envelopes / regime states; fine model over events conditioned on coarse state.
  The FFT and moment losses are compensating for this missing temporal context.

- [ ] **Train separate models per workload class** (Tencent vs Alibaba)
  Tencent: 9-day, 512B–32KB mixed, 83–94% reads.
  Alibaba: 31-day, 4KB-aligned, more variable read ratios.
  Compare per-corpus vs mixed-corpus models once Alibaba download finishes.

- [ ] **Re-train on NVIDIA GPU once box arrives**
  1. ✓ WGAN-GP enabled on MPS (create_graph=True works — CUDA-only assumption was wrong)
  2. Apply spectral norm to LSTM weight matrices (currently too slow on MPS)
  3. Increase `hidden_size`, `num_layers`, `batch_size`, `files_per_epoch`
  4. Benchmark full-corpus training time

---

## Reference papers (all in `pubs/`)

| Paper | Key contribution |
|-------|-----------------|
| Zhang et al., NAS 2024 | Our work — LLGAN baseline |
| SeriesGAN, BigData 2024 | Latent AE + supervisor + dual critic + 2-step supervisor + ℒ_TS loss |
| ChronoGAN, ICMLA 2024 | GRU+LSTM hybrid; slope/skewness/median TS loss terms |
| R3GAN, NeurIPS 2024 | RpGAN relativistic loss + R1+R2 joint regularization; full mode coverage |
| DMD-GEN, NeurIPS 2025 | Grassmannian DMD metric for temporal mode collapse; code not yet released |
| Diffusion-TS, ICLR 2024 | FFT spectral loss + trend/seasonal decomposition |
| FIDE, NeurIPS 2024 | Frequency inflation for rare/extreme value generation |
| ImagenTime, NeurIPS 2024 | Delay embedding → 2D discriminator input |
| GANs Conditioning Survey, 2024 | FiLM, projection discriminator, AdaIN for conditioning |
| Adaptive GP, MDPI Math 2025 | PI-controller dynamic λ for WGAN-GP |
| TTS-GAN, AIME 2022 | Transformer-based GAN; patch embedding for critic |
| TransFusion, 2023 | Coverage metric; diffusion+transformer for long sequences |
| NetDiffus, 2023 | Power-transform preprocessing; GASF for evaluation |
| DiTTO, 2025 | Workload conditioning; same problem, different approach |
| Paul et al., ICS 2022 | HPC trace generation predecessor |
| MEMSYS 2023 | Memory workload synthesis predecessor |
| Geminio, JSSPP 2024 | Job trace generation comparison |
