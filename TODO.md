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

- ✓ **FIDE frequency inflation in FFT loss** (`train.py`) — NeurIPS 2024
  `weight_f = 1 + α * (|freq_f_real| / mean(|freq_real|))`. Added `--fide-alpha`
  (default 1.0). Upweights high-amplitude bins so rare I/O events are not averaged away.

- [ ] **Adaptive Gradient Penalty (AGP)** (`train.py`) — MDPI Math 2025
  Replace fixed `gp_lambda=10` with a PI controller that adjusts λ dynamically
  based on `‖∇D‖ − 1`. Removes a fragile hyperparameter.
  `λ_{t+1} = λ_t + Kp*(‖∇D‖ − 1) + Ki*Σ(‖∇D‖ − 1)`. ~20 lines.

- ✓ **R2 regularization on fake samples** (`train.py`) — R3GAN, NeurIPS 2024
  Added `--r2-lambda` (default 0; enable for v11 ablation). Zero-centered GP on fake
  samples. Complements WGAN-GP's interpolated-point penalty.

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

### From peer review literature sweep (2025)

- [ ] **WaveStitch: chunk boundary-consistency objective** (`train.py`) — SIGMOD 2026 (arXiv 2503.06231)
  WaveStitch treats long-horizon generation as a continuity problem across
  generated segments, not independent windows. The key idea is not diffusion
  but chunk stitching under explicit conditioning.
  For Zarathustra: hidden-state carry-over in generate.py is a start, but
  training has no direct pressure for chunk-to-chunk continuity. Add a boundary
  loss: penalise distribution mismatch between the last K steps of window t and
  the first K steps of window t+1. Especially valuable once hierarchical
  generation is added.

- [ ] **PCF-GAN: path-space critic** (`model.py`, `train.py`) — NeurIPS 2023
  Defines discrimination in path space (characteristic function of measures on
  path space) — much closer to what matters for workload traces than LSTM
  hidden-state pooling. Even as a metric (not a full critic replacement), a
  PCF-style evaluation is likely more sensitive to sequential law and locality
  dynamics than MMD².
  For Zarathustra: explore as a critic replacement after fixing current bugs,
  or at minimum add as a path-space evaluation metric alongside DMD-GEN.

- [ ] **High-Rank Path Development critic** (`model.py`) — NeurIPS 2024
  Extends PCF-GAN to avoid rank collapse in path features. Directly relevant
  when the model matches marginals but misses temporal filtration structure
  (exactly the DMD-GEN failure mode we see). If pursuing a critic redesign,
  path-development methods are more promising than image or diffusion machinery.
  Prerequisite: PCF-GAN exploration above.

- [ ] **Constrained TS Generation: inference-time workload control** (`generate.py`) — NeurIPS 2023
  Inference-time guidance under explicit numeric constraints — no retraining.
  Target for Zarathustra: one model, multiple workload profiles. Controllable
  knobs: read ratio, request intensity, jump-size profile, repeat rate, tenant
  mix, working-set proxy.
  Prerequisite: workload conditioning (z_global descriptors) must be in place
  first so there is something to guide.

- [ ] **Retrieval-augmented conditioning for rare regimes** (`train.py`, `generate.py`) — NeurIPS 2024
  Retrieval-Augmented Diffusion shows retrieval is valuable for rare modes that
  a generative model averages away. For Zarathustra: at train or generation time,
  retrieve a small bank of real chunks or chunk-level descriptors with similar
  burst/locality statistics and use them as context (e.g., FiLM modulation or
  prefix hidden state). Especially promising for high-burst or hotspot-heavy
  trace phases that are statistically rare but operationally critical.

- [ ] **SDformer: discrete tokenization for categorical columns** (`model.py`, `dataset.py`) — NeurIPS 2024
  Treats mixed discrete/continuous time series natively rather than forcing
  everything through a continuous regression head. For Zarathustra: tokenize
  opcode, tenant_id, and potentially bucketized locality states; use continuous
  heads only for ts_delta and obj_size. More principled than expanding the
  shared tanh output head indefinitely, and a better path than image conversion.

- [ ] **Two-timescale GDA: principled G/D learning rate schedule** (`train.py`) — JMLR 2025
  Two-Timescale Gradient Descent Ascent (JMLR 2025) shows that principled
  timescale separation between G and D is needed for convergence guarantees in
  nonconvex minimax. Currently we use n_critic steps with equal LRs — not a
  principled schedule. Make the G/D adaptation rate ratio a first-class
  hyperparameter with theory-backed defaults. Likely pays off more than adding
  another scalar regularizer.

- [ ] **Scoring rule auxiliary objective** (`train.py`, `eval.py`) — JMLR 2024
  Proper sequential scoring rules (JMLR 2024) suggest training or ranking with
  a sequence-aware objective that is neither adversarial nor MMD-only. Even if
  the final model stays GAN-based, a scoring-rule early-stopping criterion is
  more principled than MMD² alone and harder to game than a co-trained metric.
  Complementary to the multi-metric checkpoint selection item above.

### Architecture improvements (from second survey)

- [ ] **AVATAR architecture: AAE + autoregressive supervisor** (`model.py`, `train.py`) — SDM 2025 (arXiv 2501.01649)
  Adversarial autoencoder (adds discriminator on encoder posterior q(z|x) vs prior p(z))
  + autoregressive supervisor. Cleaner alternative to TimeGAN: no separate Recovery net;
  encoder and decoder share a VAE-style latent space enforced by adversarial loss.
  Priority: HIGH — directly addresses our latent space quality issues.

- ✓ **2-step supervisor** (`train.py`) — SeriesGAN, BigData 2024
  Added `--supervisor-steps 1|2` (default 1). 2-step: S(h_t) predicts h_{t+2}.
  Forces longer temporal context. DMD-GEN=0.739 on v9 is the main motivation.
  Use `--supervisor-steps 2` in v11 run.

- [ ] **TIMED-style masked attention in critic** (`model.py`) — arXiv 2509.19638
  Diffusion + autoregressive supervisor + Wasserstein critic with masked self-attention.
  The masked attention prevents critic from seeing future tokens — avoids lookahead bias.
  Adapt masked attention to our LSTM critic attention pooling layer.

- [ ] **Sig-WGAN: path signature as discriminator feature** (`model.py`, `train.py`) — Math Finance (arXiv 2006.05421)
  Replace LSTM critic with signature-based Wasserstein distance. Path signatures
  capture all iterated integrals of a time series — strong theoretical convergence
  guarantees for sequential data, stronger than WGAN-GP for recurrent generators.
  Medium complexity: `signatory` library provides GPU signature computation.

- [ ] **Stage-Diff: staged generation for workload phases** (`model.py`) — arXiv 2508.21330
  Split generation into coarse stage (regime/burst envelope) + fine stage (per-event
  conditioned on coarse). Aligns with our hierarchical generator TODO. Stage-Diff
  uses diffusion; we could use GAN with LSTM at each stage.

### Evaluation / diagnostics (from peer review)

- [ ] **External Context-FID encoder** (`eval.py`) — peer review §9
  Current Context-FID uses the checkpoint's own trained encoder → co-adapted
  with the generator, partially self-serving for model selection.
  Fix: train a separate LSTM encoder on real data only (no recovery, no GAN),
  freeze it, and use it as the FID backbone. Keep the co-trained version as a
  diagnostic but never use it as the primary checkpoint selector.
  Implementation: small standalone training script; save encoder-only .pt.

- [ ] **Multi-metric checkpoint selection** (`train.py`) — peer review §10
  `best.pt` currently saves on MMD² alone — misses mode collapse (recall) and
  temporal structure (DMD-GEN). Switch to a composite rank score:
  priority order: recall/coverage > DMD-GEN > Context-FID > MMD².
  Use weighted z-score or rank aggregation, not a hand-tuned linear formula.
  Prerequisite: DMD-GEN and PRDC must run every mmd_every epochs (currently
  only in full eval.py pass).

- [ ] **Locality-specific losses** (`train.py`, `eval.py`) — peer review §7
  The model has no direct pressure on locality structure — only generic moments
  and FFT. Add explicit decoded-sequence penalties:
  1. Histogram-matching loss on signed obj_id-delta distribution (KL or L1
     between real and fake delta histograms over each batch).
  2. Repeat-access rate: fraction of steps where signed delta ≈ 0.
  If choosing one to start: histogram-matching on signed deltas + repeat rate.
  These are the metrics most directly tied to cache-hit fidelity at replay time.

- [ ] **File-to-file fidelity tool** (`compare.py`) — NEW
  Given one real log file and one synthetic log file, compute the full metric
  suite (MMD², PRDC, DMD-GEN, AutoCorr, Context-FID) and report a human-readable
  summary. Separate from eval.py (no checkpoint needed — takes two CSV/oracleGeneral
  files directly). Use case: Prof. Amer runs compare.py on a trace pair to verify
  generation quality without needing the training code.
  Implementation: thin wrapper around eval.py metrics; `python compare.py real.csv synth.csv`.

- [ ] **Generation drift / length stress test** (`generate.py`, `compare.py`)
  Determine how long a synthetic stream can run before statistics diverge from real.
  Method: generate one stream of length L, divide into chunks of length W, compute
  rolling MMD² and AutoCorr per chunk vs real baseline.  Plot metric vs chunk index.
  Expected failure modes: timestamp delta cumsum drift, obj_size distribution shift,
  burst structure decay. Helps set a practical "max reliable stream length" bound.
  Implementation: `python generate.py --n-streams 1 --n-events 100000` then rolling
  eval on the output file.

- ✓ **Context-FID + AutoCorr metrics** (`eval.py`) — TSGBench VLDB 2024
  Fréchet distance in encoder latent space (Context-FID) + lag-1..5 ACF mismatch
  (AutoCorr). v9/best.pt baselines: Context-FID=0.27, AutoCorr=0.049.

- ✓ **DMD-GEN temporal dynamics metric** (`eval.py`) — NeurIPS 2025 (arXiv 2412.11292)
  Code not yet released — implemented from paper. Runs Dynamic Mode Decomposition
  on real vs generated batches, compares eigenvector subspaces via Grassmann
  principal angles + 1-Wasserstein OT. Catches wrong autocorrelation/burst
  structure that MMD² misses. `dmdgen()` in eval.py.
  v9/best.pt score: 0.739 (poor temporal dynamics despite MMD²=0.010, recall=0.47).

- [ ] **TSGBench evaluation suite** (`eval.py`) — VLDB 2024 (arXiv 2309.03755)
  12-metric benchmark (best paper nominee): discriminative score, predictive score,
  Context-FID, autocorrelation score, cross-correlation score, distribution metrics.
  Adopt at minimum: Context-FID (LSTM encoder → Fréchet) + autocorrelation score.
  High priority — our current eval is MMD²+PRDC+DMD-GEN; TSGBench adds temporal
  fidelity metrics directly relevant to I/O workload replay accuracy.

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
  1. WGAN-GP on MPS: FAILS for LSTM critic — `lstm_mps_backward` is not differentiable
     (create_graph=True hits second-order LSTM grad). train.py now auto-falls-back to
     wgan-sn on MPS. WGAN-GP is CUDA-only for LSTM-based critics.
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
| AVATAR, SDM 2025 | AAE + autoregressive supervisor; cleaner than TimeGAN architecture |
| TSGBench, VLDB 2024 | 12-metric evaluation benchmark; best paper nominee |
| Sig-WGAN, 2021 | Path signature as Wasserstein discriminator; strong convergence guarantees |
| SMOGAN, 2025 | GAN + MMD for imbalanced regression; relevant for heavy-tailed obj_size |
| Stage-Diff, 2025 | Staged diffusion for long sequences; validates hierarchical design |
| TIMED, 2025 | Diffusion + autoregressive supervisor + masked Wasserstein critic |
| WaveStitch, SIGMOD 2026 | Chunk-stitching under conditioning; boundary-consistency for long-horizon generation |
| PCF-GAN, NeurIPS 2023 | Path-space critic via characteristic function; stronger than LSTM pooling for sequential data |
| High-Rank Path Dev., NeurIPS 2024 | Extends PCF-GAN; avoids rank collapse, catches temporal filtration failures |
| Constrained TS Gen., NeurIPS 2023 | Inference-time numeric guidance; one model, multiple workload profiles |
| Retrieval-Augmented Diffusion, NeurIPS 2024 | Retrieval for rare modes; useful for burst/hotspot trace phases |
| SDformer, NeurIPS 2024 | Discrete tokenization for categorical data; mixed discrete/continuous outputs |
| Two-Timescale GDA, JMLR 2025 | Principled G/D learning rate separation; theory-backed convergence for nonconvex minimax |
| Scoring Rules, JMLR 2024 | Proper sequential scoring objectives; complement to adversarial training for checkpoint selection |
