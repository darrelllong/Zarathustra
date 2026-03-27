# Zarathustra — TODO

Improvements derived from reading DiTTO, TTS-GAN, Diffusion-TS, NetDiffus, SeriesGAN, and TransFusion.
Ordered by impact-to-effort ratio. No image-conversion hacks — solid theory only.

---

## Immediate (hours each)

- [ ] **Log-transform `ts_delta` and `obj_size`** in `dataset.py` `TracePreprocessor`
  `ts_delta` and `obj_size` are power-law distributed. Applying min-max normalization directly
  causes the generator to collapse toward the mean. Replace with `log1p`/`log10` before
  normalization; invert with `expm1`/`10**x` at generation time.

- [ ] **Add coverage (β-recall) and α-precision metrics** to evaluation
  MMD² alone cannot detect mode collapse. β-recall < 0.3 means the generator is only
  reproducing a slice of the real distribution. Add PRDC metrics (via `prdc` package or
  nearest-neighbor implementation) and report every `mmd_every` epochs alongside MMD².
  *Run this first — it will tell us whether mode collapse is already the dominant problem.*

- [ ] **Add Fourier spectral auxiliary loss** to generator (`train.py`)
  `L_FFT = MSE(|FFT(real)|, |FFT(fake)|)` along the time axis, weighted ~0.05.
  Forces the generator to match frequency content: hourly/daily I/O cycles and burst patterns
  that WGAN alone ignores. Add `--fft-loss-weight` flag (default 0.05, 0 = off).

- [ ] **Add moment matching loss (L_V)** to generator (`train.py`)
  `L_V = MAE(μ_real, μ_fake) + MAE(σ_real, σ_fake)` per feature column, weighted ~0.1.
  Prevents WGAN from ignoring distributional shape — critical for `obj_size` and `ts_delta`.
  Add `--moment-loss-weight` flag.

- [ ] **Implement early stopping on best combined metric** (`train.py`)
  After 50% of training epochs, evaluate every N steps:
  `score = discriminative_score + λ * moment_loss`. Save best checkpoint, return it
  rather than the final epoch. SeriesGAN ablation: this alone accounts for 43% of their
  discriminative improvement. Add `--early-stop-after` flag.

---

## Medium-term (days each)

- [ ] **Add Context-FID as primary evaluation metric** (`eval.py`)
  Train a fixed LSTM encoder on real data; embed real and generated windows; compute Fréchet
  distance in that latent space. More sensitive to temporal structure than MMD² or raw statistics.
  Replace MMD² as the headline metric once validated.

- [ ] **Implement latent autoencoder + supervisor (TimeGAN/SeriesGAN architecture)** — *biggest win*
  The single highest-leverage architectural upgrade. Three additions:
  1. GRU encoder `e` + recovery `r`: compress 5 features → ~3-dim latent space
  2. GRU supervisor `s`: predicts `H_t` from `H_{t-2}` (two-step lookahead autoregression)
  3. Loss-function autoencoder `ê`: embedding-space moment matching loss `L_TS`

  Move generator to operate entirely in latent space. Train in four phases:
  - Pre-train `ê` (reconstruction, then freeze)
  - Pre-train latent autoencoder `e`/`r` with weak discriminator signal
  - Pre-train supervisor `s`
  - Joint adversarial training of all networks

  SeriesGAN reports **34% discriminative improvement** over TimeGAN with this structure,
  and TimeGAN already beats vanilla LSTM-GAN. Requires task above (latent AE) as prerequisite
  for the dual discriminator task below.

- [ ] **Add dual discriminators (latent + feature space)** — *depends on latent AE above*
  Add a second lightweight LSTM critic on the raw decoded feature-space output. Alternate
  gradient updates between both critics. The latent critic gives stable early gradients;
  the feature critic catches artifacts the latent critic misses.

- [ ] **Add patch embedding to critic** (`model.py`)
  Prepend `Conv1d(num_cols, embed_dim, kernel_size=3, stride=3)` to the LSTM critic.
  Compresses the 12-step window into 4 patch tokens before the LSTM, letting the critic
  reason about local temporal patterns as semantic units rather than individual timesteps.
  Inspired by TTS-GAN.

- [ ] **Add workload conditioning** (`model.py`, `train.py`)
  Embed condition vector `c = [tenant_id, read_ratio, obj_size_bucket]` via small MLP;
  inject into generator's initial hidden state and at every timestep. Enables tenant-specific
  or workload-class-specific generation without separate models. Train with both conditioned
  and unconditioned batches (classifier-free guidance style) to preserve unconditional generation.

---

## Longer-term

- [ ] **Train separate models per workload class** (Tencent vs Alibaba)
  The two corpora have fundamentally different character:
  - Tencent: 9-day traces, 512B–32KB mixed sizes, 83–94% reads
  - Alibaba: 31-day traces, 4KB-aligned sizes, more variable read ratios

  Once Alibaba download completes, train and evaluate a model per corpus and compare
  MMD², coverage, and Context-FID against a single mixed-corpus model.

- [ ] **Re-train on NVIDIA GPU once box arrives**
  Current training runs on Apple MPS (wigner.local). On NVIDIA:
  1. Enable true WGAN-GP (gradient penalty requires second-order gradients, blocked on MPS)
  2. Increase `hidden_size`, `num_layers`
  3. Scale `batch_size` and `files_per_epoch`
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
