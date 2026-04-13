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
  unconstrained. Renamed to `wgan-sn` throughout. True WGAN-GP is CUDA-only for
  LSTM-based critics (requires `torch.backends.cudnn.flags(enabled=False)` around
  critic forward to allow double-backward; implemented in train.py).

---

## Immediate (hours each)

- ✓ **Log-transform `ts_delta` and `obj_size`** (`dataset.py`)
- ✓ **Fourier spectral auxiliary loss** (`train.py`, `--fft-loss-weight`)
- ✓ **Moment matching loss L_V** (`train.py`, `--moment-loss-weight`)

- ✓ **Add DMD-GEN to checkpoint selection combined score** (`mmd.py`, `train.py`)
  `combined = MMD² + 0.2*(1−recall) + dmd_ckpt_weight*DMD-GEN`.
  `dmdgen` moved from `eval.py` to `mmd.py` as the canonical implementation.
  `--dmd-ckpt-weight 0.05` recommended for v19+. Default 0.0 = off (backward compatible).
  Reviewer: "Do not let the best checkpoint be chosen without a temporal law penalty."
  Adds ~5s/eval (5 DMD mini-batches). DMD-GEN stuck at ~0.71 across all versions;
  without this selector, best.pt is chosen purely on distributional metrics.

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

- ✓ **Add dual discriminators (latent + feature space)** — *depends on latent AE above*
  Second lightweight critic on raw decoded output. Latent critic = stable early gradients;
  feature critic = catches artifacts latent critic misses. Reviewer: "Once the generator
  gets good at the latent game, decoded-space artifacts become the next natural blind spot."
  Add a smaller critic on `R(H_fake)` vs `real_batch` as a supporting critic.
  Implemented: `--feat-critic-weight` (default 0.0; try 0.5–1.0). Pending first run.

- [ ] **Add patch embedding to critic** (`model.py`)
  `Conv1d(num_cols, embed_dim, kernel_size=3, stride=3)` before LSTM critic compresses
  12-step window into 4 patch tokens. (TTS-GAN)

- ✓ **Make z_global a real workload descriptor, not pure noise** (`dataset.py`, `model.py`, `train.py`)
  *Top priority per peer review.* Window-level CFG conditioning implemented in v31/v34 (cond_dim=10,
  cond_drop_prob=0.25). Full precharacterized file-level conditioning implemented for v38:
  `profile_to_cond_vector()` + `load_file_characterizations()` in `dataset.py`; `--char-file`
  flag in `train.py` and `generate.py`. Stable full-trace stats replace noisy 12-step estimates.
  Targets EMA→full eval gap and multi-corpus training quality.

- [ ] **Add training-time chunk-continuity loss** (`train.py`)
  `generate.py` carries hidden state across windows; training is still local-window only.
  Add a continuity loss penalising distributional mismatch at adjacent window boundaries.
  Or move toward a two-level generator: slow regime state + fast event generator.
  Reviewer: "The model is asked to generate long coherent traces at inference time
  without being trained on a directly chunk-continuous objective."

- ✓ **FiLM conditioning for workload** (`model.py`) — cleaner than current z_global init-only conditioning
  Feature-wise Linear Modulation: `h_t = (1 + γ(cond)) * h_t + β(cond)` at each LSTM timestep.
  Prevents conditioning signal from fading through forget gates over T=12 steps.
  Zero-init of γ, β ensures backward compatibility with existing checkpoints.
  Ref: FiLM (NeurIPS 2018). Implemented as `--film-cond` flag (not used in recent best runs).

- [ ] **Run loss-family ablation sweep** (`train.py`)
  The objective now has 10+ terms; interaction effects are a bigger risk than missing
  ingredients. Reviewer: "My suspicion is that the next large gain will come from better
  structure, not from turning on every available auxiliary term."
  Group by family: (1) adversarial + feature matching, (2) temporal (ACF + cross-cov + FFT),
  (3) locality, (4) anti-collapse (diversity + minibatch-std). Ablate one family at a time
  against a minimal baseline. Run before adding more losses.

- [ ] **Masked pretraining on real traces** (`train.py`, `dataset.py`)
  Mask random chunks or patch spans in real traces; reconstruct in latent or feature space.
  Self-supervised warm start that fits naturally before the existing AE/supervisor curriculum.
  Reviewer: "More promising than adding one more scalar GAN regularizer."
  Ref: PatchFormer (arXiv 2601.20845) — hierarchical masked reconstruction + cross-domain transfer.

- [ ] **Retrieval-conditioned rare-regime generation** (`generate.py`, `dataset.py`)
  Rare burst regimes and unusual locality phases are hard to cover with noise-only generation.
  Retrieve real chunks with similar burst/locality descriptors; use as conditioning context.
  Especially target cases where recall is decent but dynamics/locality remain wrong.
  Ref: Retrieval-Augmented Diffusion Models for Time Series Forecasting (NeurIPS 2024).

- [ ] **Chunk-stitching training curriculum** (`train.py`)
  Progressive context-length training: short windows first, then adjacent stitched windows,
  then longer contexts. Training-time analogue of generate.py's hidden-state carryover.
  Refs: WaveStitch (arXiv 2503.06231), PatchFormer (arXiv 2601.20845).

- [ ] **Improve validation** (`train.py`, `mmd.py`)
  Current MMD val set is 2 random files (may overlap conceptually with training) and
  MMD is flattened-window RBF — weak signal. Switch to: designated held-out files,
  replay/locality metrics (stack distance CDF, reuse distance), per-column KS tests.

- [ ] **Promote locality to first-class checkpoint target** (`train.py`, `eval.py`, `compare.py`)
  The `obj_id_reuse` + `obj_id_stride` split captures locality, but it is not in checkpoint
  selection or summary output. Add locality metrics (reuse rate, stride distribution) to
  checkpoint summaries and use as a tiebreaker or soft selector term.
  Reviewer: "For storage workloads, locality is not one feature among many — it is one
  of the reasons the trace matters at all. If the model gets locality wrong, good MMD²
  is still a false comfort."

---

## New ideas from literature survey (2024–2025)

### Quick wins (< 1 day each)

- ✓ **FIDE frequency inflation in FFT loss** (`train.py`) — NeurIPS 2024
  `weight_f = 1 + α * (|freq_f_real| / mean(|freq_real|))`. Added `--fide-alpha`
  (default 1.0). Upweights high-amplitude bins so rare I/O events are not averaged away.

- ✓ **Projection discriminator** (`model.py`, `train.py`) — Miyato & Koyama, ICLR 2018
  `score += inner(cond_proj(cond), pooled_features)`. Critic evaluates "is this realistic
  for this workload type?" rather than aggregate realism. --proj-critic flag. Requires
  --cond-dim > 0; best with --char-file for stable conditioning.

- ✓ **PacGAN window packing** (`train.py`) — Lin et al. NeurIPS 2018
  Critic scores packs of m windows (concatenated along time axis). Fake packs under mode
  collapse look identical; real packs are diverse. Gives explicit diversity signal beyond
  minibatch_std. --pack-size N flag (1=off, 2=pairs). Use in v39.

- [ ] **Adaptive Gradient Penalty (AGP)** (`train.py`) — MDPI Math 2025
  Replace fixed `gp_lambda=10` with a PI controller that adjusts λ dynamically
  based on `‖∇D‖ − 1`. Removes a fragile hyperparameter.
  `λ_{t+1} = λ_t + Kp*(‖∇D‖ − 1) + Ki*Σ(‖∇D‖ − 1)`. ~20 lines.

- ✓ **R2 regularization on fake samples** (`train.py`) — R3GAN, NeurIPS 2024
  Added `--r2-lambda` (default 0; enable for v11 ablation). Zero-centered GP on fake
  samples. Complements WGAN-GP's interpolated-point penalty.

### Medium-term (1–2 days each)

- ✓ **ChronoGAN slope + skewness TS loss terms** (`train.py`) — ICMLA 2024
  Added to moment loss (L_V) alongside mean/std. Slope = per-feature linear trend;
  skewness = third standardized moment (weighted 0.5×). Both activated when
  `--moment-loss-weight > 0`. GRU+LSTM hybrid backbone is a further option but 2-day effort.

- ✓ **SeriesGAN dual discriminator + 2-step supervisor** (`model.py`, `train.py`)
  — BigData 2024. Implemented: `--supervisor-steps 2` (default in all recent runs v37+).
  Dual discriminator: `--feat-critic-weight` (latent + feature-space critics).

- ✓ **Relativistic paired GAN loss (RpGAN)** (`train.py`) — R3GAN, NeurIPS 2024
  Implemented as `--loss rpgan`. D_loss = -E[log σ(C(x_r) - C(x_f))];
  G_loss = -E[log σ(C(x_f) - C(x_r))]. MPS-compatible (no second-order gradients).
  Targets β-recall mode collapse. Pending first run to evaluate.

- ✓ **FiLM conditioning for tenant/workload** (`model.py`) — survey, 2024
  Implemented as `--film-cond` flag. See Medium-term section above.

### Architecture improvements (from second survey)

- [ ] **AVATAR architecture: AAE + autoregressive supervisor** (`model.py`, `train.py`) — SDM 2025 (arXiv 2501.01649)
  Adversarial autoencoder (adds discriminator on encoder posterior q(z|x) vs prior p(z))
  + autoregressive supervisor. Cleaner alternative to TimeGAN: no separate Recovery net;
  encoder and decoder share a VAE-style latent space enforced by adversarial loss.
  Priority: HIGH — directly addresses our latent space quality issues.
  **v27 attempted full AVATAR (removed Sigmoid → unbounded latent) — FAILED.** Critic
  overpowered generator from ep1 (W=3.5–4.7). Root cause: unbounded latent space made
  critic's job trivially easy.
  **Next attempt: AVATAR-lite** — keep [0,1] Sigmoid-bounded latent space, add ONLY the
  latent discriminator + distribution loss as auxiliary losses during Phase 1 and Phase 3.
  No architecture change, just extra regularization pushing q(z|x) toward Beta/uniform in
  [0,1]. Combines cleanly with z_global conditioning and stripped losses. The `--avatar`
  flag and LatentDiscriminator class are already in the code from v27.

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

### Evaluation / diagnostics

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
  Tencent: 9-day, 512B–32KB mixed, 83–94% reads — currently training on vinge (GB10).
  Alibaba: 244 files at `/tiamat/zarathustra/traces/alibaba/` (oracle_general format, .zst).
  Compare per-corpus vs mixed-corpus models.

- [ ] **Scale up model on vinge GB10** (NVIDIA box is vinge.local — GB10, 124GB unified memory)
  1. ✓ WGAN-GP on CUDA available (cuDNN double-backward fix in train.py); MPS still falls back to wgan-sn.
  2. ✓ Spectral norm on LSTM weight matrices (weight_ih_l0 + weight_hh_l0).
  3. Try larger `hidden_size` (512), more `files_per_epoch` (24+), larger `batch_size` (128).
  4. Benchmark: v43 runs ~35–120s/epoch on GB10 with hidden_size=256, files_per_epoch=12.

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
| NetSSM, arXiv 2504.05598 | Network simulation foundation model; explicit regime/state decomposition |
| Compositional SSMs, arXiv 2504.01349 | Synthetic trace gen via compositional state-space models; most architecturally relevant |
| WaveStitch, arXiv 2503.06231 | Chunk stitching + condition-aware continuity; supports training-time continuity objective |
| PCF-GAN, NeurIPS 2023 | Path-space critic via characteristic function; more faithful sequential-law test than LSTM critic |
| High-Rank Path Dev, NeurIPS 2024 | Path signatures for deep sequential generative modeling; upgrade to path-space critic |
| Constrained TS Gen, NeurIPS 2023 | Guidance under explicit numeric constraints; foundation for controllable workload generation |
| Retrieval-Augmented Diffusion, NeurIPS 2024 | Retrieval-guided rare-mode support; transferable to burst/locality rare-regime generation |
| FreqMoE, arXiv 2501.15125 | Frequency-decomposition MoE for time series; supports regime-switching generator design |
| PatchFormer, arXiv 2601.20845 | Hierarchical masked reconstruction + cross-domain transfer; supports masked pretraining curriculum |
| Flow-Based TS Gen, arXiv 2601.22848 | Equivariance-regularised latent flow; reinforces latent-space generation without image tricks |

---

## Reviewer Action Items (Round 1 — Evaluation/Correctness Bugs)

- [ ] `[P1]` Fix PRDC fallback in `llgan/eval.py:297,321,330,337` — `r_fake` unused; "recall" and "coverage" both derived from the same `in_real_ball.any(axis=1).mean()`. Either install `prdc` as a hard dep or implement the correct fallback (recall uses real-in-fake-balls, coverage uses unique-real-in-fake-balls).
- [ ] `[P1]` Fix `eval.py --baseline` path at `llgan/eval.py:586` — passes no real windows to `_sample_fake()` for conditional checkpoints; will raise at `llgan/eval.py:432`. Thread real windows through the baseline branch.
- [ ] `[P1]` Fix reuse-rate hard-coded column index at `llgan/eval.py:539,542,546` — use `prep.col_names.index("obj_id_reuse")` the way `llgan/train.py:446` already does. Reverify recent VERSIONS.md reuse numbers after fix.
- [ ] `[P2]` Fix `TraceDataset` off-by-one at `llgan/dataset.py:988` — `len = max(0, len(data) - timestep + 1)`. Files with exactly `timestep` records currently contribute zero windows.
- [ ] `[P2]` Fix small-val-set crash in `llgan/mmd.py:113,115,250,252` — reshape using the actual returned count, not the requested `n_samples`.
- [ ] `[P2]` Decouple parser from hard-coded `/tiamat/zarathustra/traces` root in `parsers/core.py:19` and `traces/analysis/*.py:25,97`. Accept via env var or CLI.
- [ ] `[P2]` Soften "statistically indistinguishable" claims in `README.md:11,13` and `paper/main.tex:65,214` until DMD-GEN < 0.3 is actually achieved.
- [ ] `[P3]` Resolve paper placeholders (authors, hidden size, ablations, baselines, figures, final results) at `paper/main.tex:41,467,491,560`; fix `d=5` vs `d∈{5,6}` at `paper/main.tex:211`; reconcile Tencent dataset count at `paper/main.tex:451` with README counts.

## Reviewer Action Items (Round 2 — Strategic Race Focus)

- [ ] `[P1]` Build a pretrain bank: train 5–10 fresh pretrains under varied seeds/settings, rank by downstream joint-phase outcomes, then launch Phase 3 runs from the top few. Operationalize "pretrain quality is dominant variable" (VERSIONS.md:1902).
- [ ] `[P1]` Fork corpus playbooks in CLAUDE.md / README: Tencent = high-K regime (K=8) + multimodal prior; Alibaba = low-K regime (K=2) + more exploration noise. Do not unify recipes.
- [ ] `[P1]` Locality engine (explicit copy path): two-head generator that first decides reuse vs new seek, then either copies from recent-object memory or regresses a stride. Currently only supervising with L_loc scalar — reviewer is right that this is a regression, not a mechanism.
- [ ] `[P2]` Add HRC + reuse-rate to the combined checkpoint score (currently just MMD² + 0.2·(1−recall) + optional DMD-GEN weight). Use HRC-MAE as a third tiebreaker.

## Reviewer Action Items (Round 3 — Big Bets)

- [ ] `[P1]` R-informed conditioning redesign: drop `backward_seek_ratio` + `opcode_switch_ratio`; add `object_unique`, `signed_stride_lag1_autocorr`, `obj_size_std`; compress to orthogonal factor space (PCA or learned) before the generator. Do not conflate with a raw `cond_dim` bump.
- [ ] `[P1]` Window-level characterization bridge: derive per-window pseudo-labels (burst regime, locality class, object-diversity band, stride-memory pattern) from a windowed R pass, then train the regime selector supervised against those labels.
- [ ] `[P1]` Poison-point two-population training: quarantine stride-outlier files, train main G on the core distribution, then either fine-tune a tail expert or add an outlier regime to the mixture.
- [ ] `[P1]` Routed mixture-of-experts generator: small router (driven by cleaned conditioning or window pseudo-labels) picks among experts (high-reuse, burst-random, low-diversity-read; optionally corpus-specific). Stronger version of the regime sampler.
- [ ] `[P2]` Limited reopen of GMM prior under clean conditions (no `var_cond` confound). Small experiment only.
- [ ] `[P2]` Limited reopen of `cond_dim=13` *after* the conditioning redesign above — it was previously tested with the dirty feature set.

## Reviewer Action Items (Round 4 — Open From Round 1)

- See Round 1 items (PRDC, baseline, reuse-rate). Round 4 re-confirmed these are still open at eval.py L314, L539, L582.

## Reviewer Action Items (Round 5 — Reproducibility Infrastructure)

- [ ] `[P1]` Unify `z_global` inference path: build one canonical `_make_z_global()` helper and route `train.py`, `mmd.py`, `eval.py`, `model.py`, and `generate.py` through it. Currently train uses full conditioning stack (cond_encoder + regime_sampler + GMM noise) while eval/generate do raw `torch.cat([cond, noise])`. This is a plausible contributor to train/eval gaps and failed verbatim controls.
- [ ] `[P1]` Freeze per-corpus preprocessor manifest: fix the preprocessor seed and fit it on a deterministic subset, separated from the validation split. Currently preprocessor is fit on a random seed subset before val carve-out (`train.py:368,395`), which changes min/max ranges, clipping boundaries, and even which columns survive auto-drop. This confounds reproducibility controls.
- [ ] `[P1]` Improve checkpoint selection: either periodic shadow full-eval on a larger fixed file bundle, or validate that EMA-based selection on 10 held-out files rank-preserves to full eval. Current small-val selection promotes false winners (30-50% train→eval gaps documented).
- [ ] `[P2]` Fourier spectral analysis pass: run FFT on `ts_delta`, `obj_id_stride`, `obj_size`, and binary reuse on both real traces and synthetic long-rollouts. Diagnostic for burst periodicities and narrowband artifacts that MMD/PRDC/ACF/DMD-GEN may miss.
- [ ] `[P2]` Multi-seed evaluation protocol: run 3 seeds per candidate family with fixed preprocessor and held-out eval bundle, rather than single-seed runs. Prevents optimizing toward unreproducible ATBs.

## Reviewer Action Items (Round 6 — Post-Record Assessment)

- [ ] `[P2]` Keep DMD-GEN and HRC-MAE visible in triage even when combined score improves. v78 tencent set a combined record while DMD-GEN (0.7416) and HRC-MAE (0.0795) remain unresolved.
- [ ] `[P2]` Fourier analysis now even more justified: v78 may be getting the right sample cloud while missing the right temporal rhythms. Prioritize before next architectural push.

## Reviewer Action Items (Round 7 — Rebuttal Critique)

- [ ] `[P1]` Complete verbatim controls before making mechanism-level claims in the rebuttal. Do not declare "mechanism is clear" until reproducibility runs confirm the result. Frame as "leading hypothesis" until then.
- [ ] `[P1]` Run Fourier spectral analysis — this is now the third round requesting it. The latest results (v78 tencent record with poor DMD-GEN; v51 alibaba best DMD-GEN but poor combined; v82 tencent best Context-FID) create exactly the tradeoff profile where spectral diagnostics are essential.
- [ ] `[P1]` Do not close Tencent multi-scale critic without a full eval. Old v70 result had no full eval, and VERSIONS.md says it was "not killable yet." Inconsistent to close it after recalibrating other closures as too aggressive.
- [ ] `[P2]` Reframe rebuttal claims: replace "mechanism is clear" with "leading hypothesis" where z_global unification and preprocessor freezing are still undone. Accepted measurement caveats and mechanism certainty are contradictory.
- [ ] `[P2]` Stop treating combined score as the sole lens — v51 (best DMD-GEN), v82 (best Context-FID + AutoCorr) contain useful signal even though they "lost" on combined.
- [ ] `[P2]` Resume structural bets: explicit locality/copy path, window-level characterization bridge, path-space critic, MoE generator. The post-record exploitation sweep does not mean the structural design space is exhausted.

## Reviewer Action Items (Round 8 — NSF Proposal Recovery)

- [ ] `[P1]` Re-elevate validation to first-class research track: add density-sensitive metrics, clustering-style similarity, long-rollout diagnostics, Fourier/spectral checks, and external-system replay behavior to the winner-selection loop. Not paper cleanup — part of the main invention.
- [ ] `[P1]` Recover anomaly/outlier modeling as structured problem: tail expert, anomaly class, regulator, or random-timestep insertion mechanism. The NSF proposal treated outliers as necessary for realistic traces, not as cleanup noise. Stronger than just clipping.
- [ ] `[P1]` Build correlation-aware conditioning: factorized descriptors (PCA or learned factor space), remove unlearnable columns, separate file-level vs window-level signals. Old RBM work already showed reconstruction improves when correlated features are grouped.
- [ ] `[P1]` Push whole-trace generation: chunk-continuity training, long-rollout supervision, or regime-first hierarchical generator. The 12-step window is narrower than the original problem statement.
- [ ] `[P2]` Semi-supervised / structured latent modeling: disentangled factors, regime-first generation, pseudo-labeled window classes. InfoGAN-style structured latents were in the original proposal.
- [ ] `[P2]` External-system replay testing for finalists: compare generated vs real traces on actual system metrics (hit ratio, response time, CPU). "Behavioral equivalence under replay" not just "holdout similarity in embedding space."
- [ ] `[P2]` Compositional/hybrid workload generation: normal-plus-anomalous synthesis, workload-family experts, corpus-bridging. Original proposal envisioned one generator family serving multiple workload types.
- [ ] `[P3]` Benchmark against existing generators (Filebench, Impressions, VdBench) on both statistical and replay-system metrics for competitive positioning.

## Reviewer Action Items (Round 9 — "So Do It")

- [ ] `[P0]` **STOP SCALAR TUNING.** Freeze v59 (alibaba) and v93 (tencent) as reproducible baselines. No more scalar-only variants on mainline compute. All scalar sweeps are sidecars only.
- [ ] `[P0]` **RUN FOURIER NOW.** Fourth round requesting spectral analysis. Run on real traces + synthetic rollouts from v59, v93, v51 (best DMD-GEN), v82 (best Context-FID). No more deferring.
- [ ] `[P0]` **PICK A BUILD ORDER AND EXECUTE.** Recommended: (1) one conditioning-structure experiment, (2) one locality-structure experiment, (3) one diagnostics package, (4) then optional scalar sidecars.
- [ ] `[P1]` Implement conditioning-structure change: correlation-aware factorized conditioning with file-level vs window-level separation, OR window-level pseudo-label bridge. This is the highest-leverage structural bet.
- [ ] `[P1]` Implement locality-structure change: explicit reuse-vs-new decision with copy path or recent-object memory. Not another locality-loss weight — an actual mechanism.
- [ ] `[P1]` Fix infrastructure debts BEFORE making mechanism claims: z_global unification (train vs eval path divergence) and preprocessor freezing. These control whether the conditioning story is even being measured cleanly.
- [ ] `[P2]` On tencent only, reopen one critic-side structural test AFTER conditioning/locality work: full-eval multi-scale critic or path-space critic.
- [ ] `[P2]` Treat v65's failure to reproduce v59 (training-log matched but eval diverged 23%) as evidence that scalar results are stability-dependent and unreliable without verbatim controls.

## Reviewer Action Items (Round 10 — Stay On The Structural Path)

- [x] `[P0]` **Fourier/spectral diagnostics implemented.** PSD metric added to mmd.py and eval.py (commit 0985c20). Finding: PSD match is already good; remaining gap is not simple frequency content but local/cross-step structure (reuse decisions, stride consistency, IRD dynamics).
- [x] `[P0]` **Copy-path mechanism launched.** alibaba_v67 and tencent_v97 ran with per-timestep reuse BCE, stride-reuse consistency, and recovery-time stride gating. This is the structural locality bet Round 9 demanded.
- [x] `[P0]` **PCF (path-space critic) launched.** alibaba_v71 eval **0.067** — new all-time best across both corpora. PCF replaces 6 handcrafted auxiliary losses with a single learned adversarial functional on path increments (PCF-GAN, NeurIPS 2023). This is the structural critic-side bet from the TODO list.
- [ ] `[P1]` **Fix GEMINI eval blockers before drawing conclusions from structural runs.** Specific bugs:
  - [ ] Conditional eval still does raw `torch.cat([cond, noise])` instead of the encoded conditioning path used in training (eval.py L450)
  - [ ] HRC sampling uses padded cache-size pattern (eval.py L259)
  - [ ] Reuse-rate metric hardcoded to column 3 instead of using `prep.col_names` (eval.py L545)
  - [ ] Fallback descriptor path fixed at 10 dims while cond_dim can vary (dataset.py L203)
- [ ] `[P1]` **Add locality-native diagnostics.** Reuse precision/recall, reuse streak distribution, stride-on-reuse violation rate, stack-distance or IRD comparisons. These are needed to properly evaluate copy-path and PCF structural changes.
- [ ] `[P1]` **Judge structural bets on robustness, not luckiest number.** Non-reproducibility of scalar-era records means new structural work should be evaluated on: (a) more robust train→eval contract, (b) materially improved locality-facing metrics, (c) reduced seed sensitivity.
- [ ] `[P2]` **Clean up accidental absolute-path mirror.** `Users/darrell/Zarathustra/` directory tree in repo root contains duplicated files (TODO.md, llgan/train.py). Remove before it multiplies.
- [ ] `[P2]` **After copy-path/PCF stabilize, spend next structural budget on:** either path-space critic for tencent (if PCF approach needs corpus tuning) or conditioning redesign with file-level vs window-level separation.

## Reviewer Action Items (Round 11 — PCF Is A Door, Not A Destination)

- [x] `[P0]` **Freeze clean PCF recipe as baseline.** Locked: PCF 2.0, n_freqs=32, w-stop=3.0, no moment/ACF/FFT hybrid piling. All handcrafted aux losses zeroed. alibaba_v57/tencent_v86 as reproducible baselines. Done — this is the v71/v105 recipe now running.
- [x] `[P0]` **Stop scalar exploitation around PCF.** Reviewer warned against `pcf-loss-weight`, `w-stop`, `n_freqs`, and hybrid garnish drifting back to scalar fiddling. Evidence: v72 (PCF 1.0 worse), v73 (w-stop 4.0 worse), v75 (moment+PCF failed), tencent v100-v102 all underperformed clean PCF. Acknowledged — now seed-rolling on frozen recipe only.
- [ ] `[P1]` **Build generator-side locality abstraction.** Mixed-type heads + explicit reuse-vs-new routing, recent-object memory or pointer retrieval on reuse steps, stride prediction only on non-reuse steps. Copy-path was attempted (v67/v97, v113) but crashed or W-spiked — mechanism needs architectural redesign, not just loss gating.
- [ ] `[P1]` **Build conditioning-side abstraction.** File-level vs window-level split, or window-level characterization bridge. Separate slow file-level context, medium-timescale window regime, and fast per-step stochastic evolution.
- [ ] `[P1]` **Promote locality-native metric to checkpoint gate.** At least one of: reuse precision/recall, reuse-run statistics, or IRD shape should be part of accept/reject decision for generator experiments. If candidate improves combined but not reuse realism, it should not become foundation model.
- [ ] `[P2]` **Corpus-specific locality expectations.** Alibaba validates stable PCF backbone (strong distributional + cache behavior). Tencent is primary battleground for locality-native architecture. Do not force identical mechanism on both.
- [ ] `[P2]` **Resist "not more depth = do nothing architectural."** Remaining concrete abstractions: mixed-type output heads, explicit reuse-memory/pointer, locality-only or multi-branch critics, file+window latent split, window-pseudo-label routing. Deeper vanilla LSTM is least convincing unless after target-type split and locality mechanism work.
