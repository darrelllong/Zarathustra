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

- [ ] **Add coverage (β-recall) and α-precision metrics** to evaluation
  MMD² alone cannot detect mode collapse. β-recall < 0.3 means the generator covers
  only a slice of the real distribution. Add PRDC metrics (via `prdc` package or
  nearest-neighbor) and report every `mmd_every` epochs.

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

- [ ] **Implement latent autoencoder + supervisor (TimeGAN/SeriesGAN architecture)**
  The single highest-leverage architectural upgrade:
  1. GRU encoder `e` + recovery `r`: compress 5 features → ~3-dim latent space
  2. GRU supervisor `s`: predicts `H_t` from `H_{t-2}` (two-step lookahead)
  3. Loss-function autoencoder `ê`: embedding-space moment matching `L_TS`
  Move generator to latent space. Four-phase curriculum training.
  SeriesGAN: **34% discriminative improvement** over TimeGAN with this.

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
  1. Enable true WGAN-GP (gradient penalty, second-order autograd)
  2. Apply spectral norm to LSTM weight matrices (currently too slow on MPS)
  3. Increase `hidden_size`, `num_layers`, `batch_size`, `files_per_epoch`
  4. Benchmark full-corpus training time

---

## Reference papers (all in `pubs/`)

| Paper | Key contribution |
|-------|-----------------|
| Zhang et al., NAS 2024 | Our work — LLGAN baseline |
| SeriesGAN, BigData 2024 | Latent AE + supervisor + dual critic + early stopping |
| Diffusion-TS, ICLR 2024 | FFT spectral loss + trend/seasonal decomposition |
| TTS-GAN, AIME 2022 | Transformer-based GAN; patch embedding for critic |
| TransFusion, 2023 | Coverage metric; diffusion+transformer for long sequences |
| NetDiffus, 2023 | Power-transform preprocessing; GASF for evaluation |
| DiTTO, 2025 | Workload conditioning; same problem, different approach |
| Paul et al., ICS 2022 | HPC trace generation predecessor |
| MEMSYS 2023 | Memory workload synthesis predecessor |
| Geminio, JSSPP 2024 | Job trace generation comparison |
