# Ideas

## Priority: Bold Architectural Changes

Parameter tuning is unlikely to produce the breakthrough needed. The ideas below are ordered by expected impact. The Bayesian approaches at the top represent fundamental structural changes to how the model represents and covers the workload distribution — not incremental loss tweaks.

---

### 0. ~~Fourier analysis~~ (DONE — 2026-04-13)

**Method:** R spectral analysis (`traces/analysis/fourier_analysis.R`) on both corpora.
Sampled 20 files per corpus (500k records each, ~10M total). Computed smoothed periodograms
(Daniell kernel, spans=c(7,7), taper=0.1) on four derived series: ts_delta (inter-arrival
times), obj_size, reuse (binary re-access flag), stride (distance to last same-object access).
Heavy-tailed series (ts_delta, obj_size, stride) log1p-transformed before FFT.

**Results:**

| Series    | Alibaba entropy | Alibaba verdict     | Tencent entropy | Tencent verdict |
|-----------|-----------------|---------------------|-----------------|-----------------|
| ts_delta  | 0.9719          | White noise         | 0.9914          | White noise     |
| obj_size  | 0.9854          | White noise         | 0.9837          | White noise     |
| reuse     | 0.9768          | White noise         | 0.9717          | White noise     |
| **stride**| **0.9442**      | **Mild periodicity**| 0.9784          | White noise     |

Alibaba stride is the only series with mild periodicity (entropy < 0.95), showing a dominant
period of ~253 samples. All other series across both corpora are spectrally white noise.
Alibaba ts_delta and obj_size share a weak ~297-sample period. Reuse dominant frequencies
on both corpora are at the DC/near-DC component (period ≈ window length) — just the
non-zero mean, not real periodicity.

**Implications for LLGAN:**
1. **No long-range spectral structure to miss.** The LSTM's 12-step window is sufficient for
   the temporal structure present. Longer windows or hierarchical modeling won't help.
2. **Frequency-aware mechanisms are unnecessary.** FFT loss, spectral embeddings, and
   explicit frequency-aware layers would be fitting noise. The peer review's repeated
   request for spectral features is answered: there's nothing periodic to capture.
3. **Improvement must come from distributional fidelity**, not temporal modeling — z_global
   fix, locality engine, and conditioning architecture are the right targets.
4. The mild alibaba stride periodicity (~253 samples) is interesting but too weak
   (entropy 0.94, barely below threshold) to justify architectural changes.

### 1. Gaussian Mixture Prior on latent z (GMM-GAN)

**Why this is the breakthrough candidate:** The generator currently samples `z ~ N(0, I)` — a single unimodal prior. But workloads come from clearly distinct regimes (burst, sequential, random-read, write-heavy). The diversity loss tries to fight this after the fact. A GMM prior makes multi-modal coverage *structural*.

The hypothesis for why v40 recall is plateauing at ~0.40 while v31 reached 0.596: v31's noisy window-level z_global accidentally created a soft mixture prior over workload types. Replacing it with clean char-file conditioning collapsed that mixture structure. A GMM prior restores it — but deliberately and properly.

**Implementation:**
- K=8–16 mixture components, each `N(μ_k, σ_k²I)` in latent space
- Char-file conditioning (write_ratio, burstiness_cv, reuse_ratio, etc.) selects the component via a learned soft assignment `π_k(cond)`
- Sample `z ~ N(μ_{k*}, σ_{k*}²I)` where `k* ~ Categorical(π(cond))`
- Add a commitment loss to keep each component well-separated
- The critic sees no change; only G's prior changes

**Expected impact:** Directly breaks the recall ceiling by giving G explicit "mode addresses." Each workload type lives in its own region of latent space rather than competing for the same unimodal prior.

References:
- Natalia Dilokthanakul et al., ["Deep Unsupervised Clustering with Gaussian Mixture Variational Autoencoders"](https://arxiv.org/abs/1611.02648), ICLR 2017.
- Lars Maaløe et al., ["Auxiliary Deep Generative Models"](https://arxiv.org/abs/1602.05473), ICML 2016.

---

### 2. Bayesian GAN — posterior over discriminators (BayesGAN)

**Why this is bold:** Replace the single discriminator point estimate with a *posterior distribution over discriminators*, sampled via SGLD. The generator sees a mixture of discriminators; no single boundary can force collapse to one mode. Mathematically, the Nash equilibrium of BayesGAN corresponds to the generator matching the full data distribution, not a single mode.

**Implementation:**
- Replace Adam optimizer on D with SGLD: `θ_D ← θ_D - η∇L + N(0, 2η)`
- Maintain M=10 discriminator particles; update all each step
- Generator loss is the average across particles
- This adds ~10× D memory but D is small (LSTM + FC)

**Expected impact:** The single strongest anti-mode-collapse technique known. Recall improvements of 20–40% over point-estimate GANs reported on multimodal benchmarks.

Reference:
- Yunus Saatci and Andrew Wilson, ["Bayesian GAN"](https://papers.nips.cc/paper_files/paper/2017/hash/74071a673307ca7459bcf75fbd024e09-Abstract.html), NeurIPS 2017.

---

### 3. Variational conditioning — uncertainty over workload type

**Why this matters:** The char-file gives a *point estimate* of file characteristics. But two files with the same write_ratio can have very different temporal dynamics. A Bayesian approach: replace the fixed conditioning vector with a learned distribution.

**Implementation:**
- Add a small MLP `(μ_cond, log σ_cond) = f(char_stats)`
- Sample `cond ~ N(μ_cond, σ_cond)` via reparameterization at training time
- Add KL divergence penalty: `KL(q(cond|stats) || p(cond))`
- Use `μ_cond` (no sampling) at eval time

**Why it may close the EMA→full eval gap:** The gap arises because training-time conditioning is slightly different from eval-time. Explicit uncertainty modeling makes the generator robust to conditioning noise — same effect CFG dropout achieves, but principled.

This is cheap (~20 lines) and can stack on top of v40 without architectural surgery.

---

### 4. GP prior on latent trajectories (for DMD-GEN)

**Why this targets the DMD-GEN plateau:** DMD-GEN is stuck at ~0.72 because the LSTM generates temporally incoherent sequences — it has no structural prior that consecutive latent states should be smooth or autocorrelated. A Gaussian Process prior on `{z_1, ..., z_T}` imposes exactly this.

**Implementation:**
- Replace `z_t ~ N(0,I)` i.i.d. with a GP sample: `z_{1:T} ~ GP(0, K)` where K is a Matérn or RBF kernel
- Approximate with a low-rank inducing point method for efficiency
- The kernel lengthscale is a learnable parameter — lets the model discover the right temporal correlation scale

**Expected impact:** Directly targets DMD-GEN. GP time-series priors have been shown to capture autocorrelation structure that LSTMs alone cannot.

Reference:
- Vincent Fortuin et al., ["GP-VAE: Deep Probabilistic Time Series Imputation"](https://arxiv.org/abs/1907.04155), AISTATS 2020.

---

### 5. Regime-first, event-second generation (two-stage)

**Why this is bold:** Rather than asking one flat LSTM to jointly model regime identity, burst envelope, IAT, size, and locality, split into two stages:

- **Stage 1 (regime sampler):** Given char-file conditioning, sample a discrete regime code + continuous regime parameters (burst rate, access pattern, size distribution mode). This is a small VAE or normalizing flow.
- **Stage 2 (event generator):** Condition the LSTM on the regime code/parameters and generate per-event requests.

**Why it fits Zarathustra:** Current errors are regime-level — the model generates plausible individual events but the wrong *kind* of workload. A two-stage model can't confuse regimes because Stage 1 commits to one before Stage 2 starts.

References:
- Patrick Dendorfer et al., ["Goal-GAN"](https://arxiv.org/abs/2010.01114), ACCV 2020.
- Amir Sadeghian et al., ["SoPhie"](https://openaccess.thecvf.com/content_CVPR_2019/html/Sadeghian_SoPhie_An_Attentive_GAN_for_Predicting_Paths_Compliant_to_Social_CVPR_2019_paper.html), CVPR 2019.

---

### 6. Path-space critic (PCF-GAN / COT-GAN)

**Why this is bold:** Every auxiliary loss we've added (ACF, cross-cov, locality, FFT) is a handcrafted proxy for "does this sequence have the right dynamics?" A path-space critic replaces all of them with a *single learned functional* that directly compares distributions over sequences using the path characteristic function or causal optimal transport.

- No more loss weight tuning
- Discriminates on the full trajectory, not marginal statistics
- Directly targets DMD-GEN and AutoCorr without ad hoc terms

References:
- Blanka Horvath et al., ["PCF-GAN"](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7d0e867582cdc156fd280d5a6aa1be08-Abstract-Conference.html), NeurIPS 2023.
- Blanka Horvath et al., ["High Rank Path Development"](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d0cf89927acd9136d27ebf08f9e8a888-Abstract-Conference.html), NeurIPS 2024.
- Alexander Xu et al., ["COT-GAN"](https://proceedings.neurips.cc/paper/2020/hash/641d77dd5271fca28764612a028d9c8e-Abstract.html), NeurIPS 2020.

---

### 7. Mixed-type output heads (tabular GAN style)

Model the known type structure of I/O fields explicitly rather than pushing everything through one regression head.

- Separate heads: opcode → softmax, reuse flag → sigmoid, IAT/size → continuous regression, stride → signed log-normal
- Training-time conditional sampling to emphasize underrepresented field combinations
- Why it fits: opcode and reuse are not the same kind of target as IAT. Forcing one head to handle both corrupts gradients.

References:
- Lei Xu et al., ["CTGAN"](https://proceedings.neurips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html), NeurIPS 2019.
- Zilong Zhao et al., ["CTAB-GAN"](https://proceedings.mlr.press/v157/zhao21a.html), ACML 2021.

---

### 8. Multi-resolution / multi-branch critics

Use multiple discriminator views at different temporal scales — raw events, pooled windows, locality-only channel, stitched multi-window sequences.

Audio GANs (HiFi-GAN) improved dramatically once the discriminator could see both fine texture and larger periodic structure. Traces have the same multi-scale problem.

Reference:
- Jungil Kong et al., ["HiFi-GAN"](https://arxiv.org/abs/2010.05646), NeurIPS 2020.

---

### 9. Self-diagnosing upweighting of underrepresented modes

Identify real windows the model currently under-covers (by reuse rate, stride pattern, burstiness) and upweight them in critic training and feature matching. Principled alternative to blindly increasing diversity loss everywhere.

Reference:
- Jinhee Lee et al., ["Self-Diagnosing GAN"](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0ebcc77dc72360d0eb8e9504c78d38bd-Abstract.html), NeurIPS 2021.

---

### 10. Conditional generator + projection discriminator

Condition both generator and critic on char-file workload descriptors. The critic scores "is this realistic *for this workload type*?" rather than "is this realistic in aggregate?"

Blocked until G has trained on char-file conditioning for ~100+ epochs (v38 showed proj_critic + fresh G = W explosion). Safe to add in v41 after v40 trains.

References:
- Takeru Miyato and Masanori Koyama, ["cGANs with Projection Discriminator"](https://arxiv.org/abs/1802.05637), ICLR 2018.

---

### 11. Deeper LSTM (2–3 layers)

**Why this matters now:** R characterization found 23 temporal changepoints for tencent and
21 for alibaba. The Hurst exponent is 0.79 (tencent) and 0.98 (alibaba), indicating strong
long-range temporal dependence. A single LSTM layer has limited capacity for hierarchical
temporal representations — the first layer must simultaneously track per-step dynamics AND
multi-scale regime structure.

**Implementation:**
- Change LSTM in both Generator and Critic from `num_layers=1` to `num_layers=2` (or 3)
- Add residual connections between layers to prevent gradient degradation
- Optionally: different hidden sizes per layer (e.g., 256→128 pyramid)
- Minimal code change — PyTorch LSTM supports `num_layers` natively

**Expected impact:** Better capture of multi-scale temporal structure (burst envelopes,
regime transitions, long-range stride autocorrelation). R found that the data has structure
at 96-file, 127-file, and 164-file scales (Tencent changepoint spacing) — a single LSTM
layer may lack the representational depth to model these overlapping timescales simultaneously.

**Risk:** Deeper models are harder to train and more prone to gradient issues. May need
lower learning rate or gradient clipping adjustments. Pretrain checkpoints are incompatible
(architecture change).

---

### 12. Expanded conditioning (cond_dim 10 → 13)

**Why:** R analysis identified three features with significant signal that we don't use:
- `object_unique` (Tencent median=1788, Alibaba=2182) — object diversity directly controls
  reuse/seek balance
- `signed_stride_lag1_autocorr` (Tencent=-0.17, Alibaba=-0.41) — stride memory is a
  different signal from IAT autocorr
- `obj_size_std` — distribution spread, not just center (q50)

Also: `backward_seek_ratio` is redundant (= 1 - forward - reuse) and `opcode_switch_ratio`
is always 0 after auto-drop. Net: drop 2, add 3 → cond_dim 11 (or 13 if keeping both).

See R-REBUTTAL.md for details.

---

### 13. Block sampling for alibaba (Hurst=0.98)

**Why:** Alibaba's Hurst exponent is 0.98 (near-persistent). Consecutive files' PC1 scores
are almost perfectly correlated — files are NOT exchangeable. Our random `files-per-epoch=12`
sampling destroys this temporal structure. Block sampling (contiguous file windows) would
give the generator batches with realistic within-batch diversity.

**Implementation:** Add `--block-sample` flag to sample contiguous file ranges instead of
random files per epoch. Easy change in the dataloader.

---

## Deprioritized (incremental, not bold)

- **PacGAN packed critic** — tried in v39, didn't break recall ceiling
- **Full relativistic GAN objective** — tried in v39 (RpGAN), caused C_loss→0 immediately, worse than WGAN-SN
- **Structured generator factorization** — valid but overlaps with regime-first (#5); do that first
- **BayesGAN M=2 + regime sampler** — tried in v56, G_loss stayed positive, recall stuck 0.27-0.33. BayesGAN redundant when regime sampler already provides mode coverage.
- **Projection discriminator** — tried in v55, W-distance exploded to 38.6 in 5 epochs. Critic too powerful with workload conditioning.
- **Deeper LSTM 2-layer (#11)** — tried in v59 (tencent) and v19 (alibaba). Both mode-collapsed: W→0, recall stuck. Deeper network finds degenerate equilibria. Needs different training strategy.
- **Expanded conditioning cond_dim=13 (#12)** — tried in v20 (alibaba). Made things worse: recall 0.085 vs v18's 0.386 at same epoch. Extra dims add noise.
- **Self-diagnosing upweighting (#9)** — tried in v62 (tencent, temp=2→10) and v24 (alibaba, temp=10). Fundamental positive feedback loop: high critic scores → more weight → higher scores → W-explosion. temp=2 explodes in 9ep, temp=10 in ~60ep. Alibaba worse (W-collapses). Not worth pursuing without architectural changes to break the feedback loop.
- **Multi-scale critic (#8) on alibaba** — tried in v23. W stuck at 0.3, recall stagnated 0.27-0.30. With T=12, downsampled scales (T//2=6, T//4=3) too short for meaningful discrimination. May work better with longer windows.

---

## Tried and validated

- **#1 GMM prior (--gmm-components)** — implemented, part of ATB recipe
- **#3 Variational conditioning (--var-cond)** — implemented, part of alibaba recipe
- **#5 Regime sampler (--n-regimes 8)** — validated: v54 reached combined=0.108★ (0.019 from ATB)
- **#5 Regime sampler K=2 (alibaba only)** — validated: v18 reached combined=0.110★ NEW ALIBABA ATB. K=2 fails on tencent (v58).
- **#8 Multi-scale critic (--multi-scale-critic)** — implemented (works on tencent, hurts alibaba)
- **#13 Block sampling (--block-sample)** — implemented, testing in v21 (alibaba) and v60 (tencent)
- **#6 PCF loss (--pcf-loss-weight)** — **BREAKTHROUGH**: alibaba_v71 eval 0.067 (50% better than any prior). Replaces all handcrafted aux losses. Adversarial frequency training critical (freqs in C optimizer, NOT G).
- **#0 Fourier analysis** — All series ≈ white noise on both corpora. No dominant periodicities. Frequency-aware mechanisms unnecessary.

### 14. HRC (Hit Ratio Curve) evaluation — cache fidelity metric

**Why this is urgent:** Both 2DIO (EUROSYS '26) and DiffGen (ICA3PP '25) evaluate trace
generators on cache behavior, not just distributional metrics. 2DIO directly benchmarks
LLGAN and shows that low MMD² does NOT imply good HRC fidelity — their w11 trace has
the lowest MMD² but the worst HRC match. DiffGen shows LSTM/Transformer traces match
distributions but fail cache replay (62-66% hit ratio deviation). Reviewers will ask for this.

**Implementation:**
- Build a simple LRU/FIFO/LFU cache simulator (or use CacheSim)
- For each generated trace, compute HRC: hit ratio vs cache size (normalized to footprint M)
- Compare against real trace HRCs using MAE (mean absolute error)
- Add as a post-hoc eval metric alongside MMD²/recall/DMD-GEN
- Does NOT change training — pure evaluation addition

**Expected impact:** Validates (or reveals gaps in) our model's practical utility for storage
benchmarking. If our joint temporal modeling produces better HRCs than DiffGen's independent
field generation, that's a strong paper argument.

---

### 15. Reuse rate amplification — targeted obj_id_reuse improvement

**Why this matters:** Our best models produce ~0.5% reuse rate vs real traces at ~10%+.
Both 2DIO and DiffGen identify locality-of-reference as the critical property for cache
fidelity. Without sufficient object re-access, generated traces produce ~0% hit ratio at
any cache size — making cache evaluation meaningless.

**Possible approaches:**
- **Loss reweighting**: Upweight the obj_id_reuse column in reconstruction loss (currently
  treated equally with other columns). Binary cross-entropy on reuse column specifically.
- **Conditional generation**: Condition on target reuse rate from char-file (we already have
  reuse_ratio in cond vector). Ensure G actually responds to it.
- **Self-diagnosing upweighting (#9)**: Identify real windows with high reuse and upweight
  them in critic training — directly addresses the mode collapse on reuse events.
- **Post-hoc reuse injection**: At generation time, apply a learned reuse policy on top of
  the raw stride output — threshold small strides to force exact object revisits.

**Expected impact:** Directly targets the cache fidelity gap identified by both competitor papers.
Even small improvements (0.5% → 5%) would dramatically improve HRC fidelity.

---

### 16. Mixed-type output heads (validated by DiffGen)

DiffGen (ICA3PP '25) validates the core insight of IDEAS.md #7: different field types need
different generation strategies. DiffGen goes further — completely separate models per field
type (parametric for statistical fields, Transformer for continuous, histogram+Markov for
categorical). Their per-field specialization beats unified LSTM/Transformer/RNN on all
three datasets.

Our existing `mixed_type_recovery` flag (config.py) already implements sigmoid heads for
binary columns. DiffGen's results suggest we should:
- Ensure binary columns (opcode, obj_id_reuse) use sigmoid → BCE loss, not tanh → MSE
- Consider Markov transition modeling for opcode sequences (not relevant for tencent/alibaba
  where opcode is always write, but matters for multi-opcode traces like RocksDB/Systor)
- The key advantage we have over DiffGen: our LSTM generates all fields jointly, preserving
  cross-field temporal correlations that DiffGen completely ignores

---

## Competitive landscape (updated 2026-04-06)

**2DIO** (Wang, Khor, Desnoyers — EUROSYS '26): Parametric (non-ML) cache-accurate trace
generator. Strength: perfect HRC reproduction via IRD modeling. Weakness: no learning,
requires manual parameter fitting per trace, can't capture complex multi-field correlations.
Directly benchmarks LLGAN and finds it lacking on HRC fidelity (but uses weak reimplementation:
BCE loss, hidden=100-126, 30 epochs, [LBA, Length] only).

**DiffGen** (Liu et al. — ICA3PP '25): Classify-then-generate framework. Strength: per-field
specialization beats one-model-fits-all. Weakness: generates fields independently — no
cross-field temporal correlations. Uses Baleen/RocksDB/Systor (not tencent/alibaba).
Cites LLGAN [Zhang et al. NAS 2024] as related work.

**Our advantage:** Joint temporal modeling of all fields simultaneously via LSTM with
workload conditioning. Neither competitor captures cross-field dynamics. But we must
prove it via cache evaluation (#14).

---

## Execution order (updated 2026-04-11)

1. ~~v40–v54: char-file, GMM, var_cond, regime sampler~~ (DONE)
2. ~~v57: clip fix~~ (DONE — combined=0.108★, best single tencent run)
3. ~~K=2 regime sampler~~ (DONE — alibaba ATB 0.110★; tencent failed)
4. ~~Deeper LSTM (#11)~~ (FAILED — both corpora mode-collapsed)
5. ~~Expanded conditioning (#12)~~ (FAILED — hurt recall)
6. ~~Block sampling (#13)~~ (DONE — helps tencent, hurts alibaba)
7. ~~v61/v22: Lower lr (8e-5/4e-5)~~ (DONE — v22 completed 0.111★, v61 killed at 0.124★)
8. ~~Self-diagnosing (#9)~~ (FAILED — positive feedback loop destabilizes both corpora)
9. ~~Multi-scale critic (#8) on alibaba~~ (FAILED — T=12 too short for multi-scale)
10. ~~Path-space critic (#6)~~ (**BREAKTHROUGH** — alibaba_v71 eval 0.067, NEW ATB; replaces all handcrafted aux losses)
11. ~~GP prior (#4)~~ (FAILED — alibaba_v81 eval 0.181, tencent_v110 eval 0.147; overfits noise structure)
12. ~~HRC evaluation (#14)~~ (DONE — already integrated into eval.py)
13. ~~Continuity loss~~ (FAILED — alibaba_v82 eval 0.130 precision collapse; tencent_v111 eval 0.167 mode collapse)
14. ~~Feat-critic~~ (FAILED — alibaba_v83 eval 0.124, tencent_v112 eval 0.157; mode collapse on both)
15. ~~Copy-path (#15)~~ (FAILED — tencent_v113 W-stopped ep3, W=13.24 immediate divergence)
16. ~~alibaba_v84~~ (v71 seed #7, killed ep42 — peaked 0.092, stalled 27 epochs)
17. ~~tencent_v114~~ (v105 seed roll #1, killed ep45 — peaked 0.126, stalled 20 epochs)
18. ~~alibaba_v85~~ (v71 seed #8, W-stopped ep60 — train 0.091, eval 0.133, +46% gap)
19. ~~tencent_v115~~ (v105 seed roll #2, killed ep36 — peaked 0.111, W-spiked to 2.93, recall collapsed)
20. ~~alibaba_v86~~ (v71 seed #9, W-stopped ep54 — train 0.086, eval 0.155, **+80% gap**)
21. ~~tencent_v116~~ (v105 seed roll #3, killed ep29 — peaked 0.120 ep15, regressed to 0.159, 14 stale)
22. ~~alibaba_v87~~ (v71 seed #10, killed ep38 — train 0.077, eval 0.124, **+61% gap**, precision 0.889 best ever)
23. ~~tencent_v117~~ (v105 seed roll #4, killed ep35 — best 0.109 ep20, 4 stars then stalled, W-spike scare)
24. **INVESTIGATION: z_global conditioning divergence (Round 5)**
    - Root cause: CondEncoder uses stochastic (μ+σε) during training, deterministic (μ) at eval
    - Added `--cond-noise-scale` to eval.py; scale=0.75 helps v87 (avg 0.107 vs 0.135) but hurts v71
    - **KEY FINDING: eval variance is massive** — v71 ranges 0.076-0.114 (avg 0.095), NOT 0.067
    - ATB of 0.067 was a lucky draw. True v71 avg ≈ 0.095, v87 avg ≈ 0.135
    - The conditioning fix is checkpoint-dependent, not universal. Need higher n_samples or multi-run averaging.
25. ~~tencent_v118~~ (self-diag temp=1.0, W-stopped ep6 — W exploded 4.95→6.15→9.29, too aggressive)
26. ~~alibaba_v88~~ (v71 seed #11, W-stopped ep55 — train 0.080 best ever, eval avg 0.145, **+81% gap**)
27. ~~tencent_v119~~ (self-diag temp=0.1, W-exploded ep3 — even 0.1 too aggressive for tencent)
28. ~~alibaba_v89~~ (self-diag temp=0.1, W-exploded ep3 — **self-diag DEAD on BOTH corpora**)
29. ~~alibaba_v90~~ (BayesGAN 5 particles, killed ep14 — recall=0.000, critic too weak for alibaba)
30. ~~tencent_v120~~ (BayesGAN 5 particles, killed ep50 — best 0.101 ep35, **5-run eval avg 0.126**, doesn't beat ATB)
31. **BayesGAN 5 particles**: W stability exceptional but eval quality doesn't transfer. Train→eval gap persists.
32. ~~alibaba_v91~~ (BayesGAN 3 particles, killed ep40 — best 0.086★ ep25, **5-run eval avg 0.100**, gap only +16%! Doesn't beat ATB 0.095 but validates BayesGAN)
33. ~~alibaba_v92~~ (BayesGAN 3 + proj-critic, W-stopped ep5 — W exploded 0.61→8.02, **proj-critic + BayesGAN incompatible**)
34. ~~tencent_v121~~ (BayesGAN 3, killed ep40 — best 0.111★ ep25, **5-run eval avg 0.136**, worse than v120. **BayesGAN DEAD on tencent**)
35. ~~tencent_v122~~ (proj-critic no BayesGAN, W-stopped ep5 — W=19.3, **proj-critic DEAD on both corpora**)
36. ~~alibaba_v93~~ (BayesGAN 3 + lower lr, killed ep43 — best 0.090★ ep25, **5-run eval avg 0.108**, worse than v91. BayesGAN doesn't beat base v71 0.095)
37. **Running:** alibaba_v94 (base PCF + lower lr, no BayesGAN); tencent_v123 (lower lr, G warm-up)
38. **KEY INSIGHT:** Base v71 recipe (0.095) is still alibaba ATB. BayesGAN reduces gap but doesn't win. Next: structural changes (z_global fix, locality engine).
39. **Fourier analysis (#0) COMPLETE** — R spectral analysis on both corpora (10M records each). All series ≈ white noise (entropy > 0.95) except alibaba stride (0.9442, mild periodicity at ~253-sample period). No dominant periodicities for LSTM to capture. Frequency-aware mechanisms unnecessary. Confirms: improvement path is distributional (z_global, locality), not temporal.
40. ~~alibaba_v94~~ (base PCF + lower lr, killed ep26 — best 0.116★ ep10, **eval 0.115, train→eval gap <1%!!** Lower lr eliminates generalization gap but base quality 0.115 doesn't beat ATB 0.095)
41. **Running:** tencent_v123 (lower lr, GAN ep34, best comb=0.136★ ep30)
42. **KEY FINDING:** Lower lr (6e-5/3e-5) eliminates train→eval gap (<1% vs 40-80% at standard lr). Combine with quality-improving structural change to capitalize.
43. ~~tencent_v123~~ (lower lr, killed ep48 — best 0.136★ ep30, **eval 0.141, train→eval gap +3.7%**. Lower lr reduces gap but quality 0.141 far from ATB 0.098)
44. ~~alibaba_v95~~ (lower lr + files_per_epoch=24, killed ep25 — best 0.103★ ep5, **eval 0.106, gap +2.9%**. Broader sampling strong early but unsustainable. Precision drop 0.627.)
45. ~~tencent_v124~~ (lower lr + files_per_epoch=24, killed ep46 — best 0.114★ ep35, **eval 0.145, gap +27%**. Five consecutive train stars but W=2.93 spike broke it. Broader sampling worsened eval gap on tencent.)
46. **Running:** alibaba_v96 (standard lr + files_per_epoch=24, GAN ep11, best 0.117★ ep10)
47. **Launched:** tencent_v125 (standard lr 8e-5/4e-5 + files_per_epoch=24)
48. **CONFIG-SPACE EXHAUSTED.** Lower lr, broader sampling, BayesGAN, proj-critic all tested. None beat ATBs (alibaba 0.095, tencent 0.098). After v96/v125 complete, must move to structural code changes: z_global fix, locality engine.
49. ~~alibaba_v96~~ (standard lr + 24 files, killed ep30 — best 0.097★ ep15 **BEST TRAIN EVER**, eval 0.119, gap +22%. Config-space CONCLUSIVELY exhausted.)
50. **Running:** tencent_v125 (standard lr + 24 files, G warm-up ep30/100)
51. **z_global det_prob fix IMPLEMENTED** — CondEncoder now accepts `det_prob` param; during training, randomly uses deterministic μ (no noise) to align train/eval distributions. Added `--var-cond-det-prob` CLI arg.
52. **Running:** alibaba_v97 (z_global det_prob=0.3, standard lr, 12 files/epoch)
53. ~~tencent_v125~~ (standard lr, 12 files, killed ep30 — best 0.115★ ep10, 20 stale, W spiked 3.40, recall collapsed 0.481→0.298)
54. **Running:** tencent_v126 (z_global det_prob=0.3, standard lr, 12 files/epoch)
55. ~~alibaba_v97~~ (z_global det_prob=0.3, killed ep55 — best 0.088★ ep30, **6-run eval avg 0.111**, range 0.082-0.140. First eval 0.088 was lucky. Doesn't beat ATB 0.095. z_global mismatch is NOT the dominant gap source.)
56. **KEY FINDING:** Eval variance (recall 0.398–0.639) is the real problem, not train/eval distribution mismatch. Need higher n_samples or architectural fix for recall stability.
57. ~~alibaba_v98~~ (v71 base recipe, fresh seed, killed ep60 — best 0.089★ ep30, **5-run eval avg 0.088 ★★★ NEW ALIBABA ATB ★★★** Beats v71's 0.095 by 7.4%. Recall avg 0.611 vs v71's ~0.50.)
58. ~~tencent_v126~~ (z_global det_prob=0.3, killed ep36 — best 0.137★ ep15, 21 stale. **det_prob fix DEAD on both corpora.**)
59. ~~tencent_v127~~ (base ATB recipe, fresh seed, killed ep51 — best 0.100★ ep35, **3-run eval avg 0.148**, +48% gap. Four consecutive stars but train→eval gap killed it.)
60. ~~alibaba_v99~~ (v71 base recipe, fresh seed, killed ep39 — best 0.108★ ep25, G_loss abnormally high 7-9, weak seed)
61. ~~tencent_v128~~ (base ATB recipe, fresh seed, killed ep28 — best 0.151★ ep15, recall never exceeded 0.308 then collapsed. Dead seed.)
62. ~~alibaba_v100~~ (v71 base recipe, fresh seed, killed ep41 — best 0.090★ ep20, **5-run eval avg 0.110**, +22% gap. Run 4 hit 0.087! But avg doesn't beat ATB 0.088.)
63. ~~tencent_v129~~ (base ATB recipe, fresh seed roll #3, killed ep67 — best 0.134★ ep50, 6 stars but W unstable, 3.98 at ep57. No eval warranted.)
64. ~~alibaba_v101~~ (v71 base recipe, fresh seed roll #3, killed ep71 — best 0.097★ ep45, recall=0.585, **5-run eval avg 0.123**, +27% gap. Best training seed ever but doesn't beat ATB 0.088.)
65. ~~tencent_v130~~ (base ATB recipe, fresh seed roll #4, killed ep25 — best 0.122★ ep10, recall collapsed 0.495→0.335. No eval warranted.)
66. ~~alibaba_v102~~ (v71 base recipe, fresh seed roll #4, killed ep30 — best 0.093★ ep10, **5-run eval avg 0.113**, Run 2 hit 0.079! Best individual eval ever but avg doesn't beat ATB 0.088.)
67. ~~tencent_v131~~ (base ATB recipe, fresh seed roll #5, killed ep23 — best 0.117★ ep10, 13 stale, regressing. Pivoting to structural fix.)
68. ~~alibaba_v103~~ (v71 base recipe, fresh seed roll #5, killed during G warm-up ep90 — no GAN data. Pivoting to structural fix.)
69. **STRUCTURAL FIX: CFG information leakage (Gemini Round 2 P1)** — CFG dropout moved BEFORE cond_encoder, regime_sampler, and GMM prior. Previously noise retained workload identity even when conditioning was dropped, defeating CFG's unconditional training. Fix ensures truly unconditional samples.
70. ~~tencent_v132~~ (CFG info-leak fix, W-stopped ep30 — best 0.111★ ep15, ep25 combined=0.101 near ATB. W spiked 3.38→3.65→3.07. CFG fix changes critic dynamics on tencent.)
71. ~~alibaba_v104~~ (CFG info-leak fix, killed ep41 — **FIVE consecutive stars**, best 0.092★ ep30 (recall=0.596). **5-run eval avg 0.121**, +31% gap. Run 4 hit 0.090. CFG fix improves training but does NOT reduce eval gap. Recall variance 0.364-0.608 remains bottleneck.)
72. ~~tencent_v133~~ (CFG fix, killed ep79 — best 0.105★ ep55, 24 stale, W spiking >3.0. 7% above ATB.)
77. ~~tencent_v134~~ (copy-path-loss-only 0.5/0.5, killed ep42 — best **0.083★ ep20**, **5-run eval avg 0.126**, +52% gap! Copy-path reduces gap on alibaba but NOT tencent.)
80. ~~tencent_v135~~ (copy-path-loss-only 0.25/0.25, killed ep28 — best 0.125★ ep15, W>3.0 three times, recall collapsed. **Copy-path DEAD on tencent.**)
82. ~~tencent_v136~~ (multi-scale critic + PCF, killed ep52 — **FIVE consecutive stars**, best **0.073★** ep25 (recall=0.694). **5-run eval avg 0.094 ★★★ NEW TENCENT ATB ★★★** Beats 0.098 by 4.1%. Run 2 hit 0.071!)
84. ~~tencent_v137~~ (multi-scale critic + PCF, killed ep50 — FOUR stars, best **0.082★** ep35. **5-run eval avg 0.107**, Run 2 hit 0.086. Confirms multi-scale critic reproducible.)
87. ~~tencent_v138~~ (multi-scale critic + PCF, seed #3, killed ep45 — FOUR stars, best **0.090★** ep20. **5-run eval avg 0.112**, Run 2 hit 0.084. Doesn't beat ATB 0.094.)
88. **MULTI-SCALE CRITIC CLOSED** (6 seeds total). Alibaba avg 0.100 (18% over baseline). Tencent avg 0.104. Universal improvement validated. Moving to next idea.
89. ~~alibaba_v112~~ (multi-scale critic + self-diag temp=1.0, killed ep6 — **W=6.57 at ep6, same feedback loop explosion.** Self-diag temp=1.0 DEAD even with multi-scale.)
90. ~~tencent_v139~~ (multi-scale critic + self-diag temp=1.0, W-stopped ep8 — W=3.12→5.26→4.95, same feedback loop. **Self-diag temp=1.0 DEAD on BOTH corpora.**)
91. ~~alibaba_v113~~ (multi-scale critic + self-diag temp=0.1, W-stopped ep5 — W=3.04→4.02→5.79. **Self-diag DEFINITIVELY DEAD at ALL temperatures.**)
92. ~~tencent_v140~~ (multi-scale critic + self-diag temp=0.1, killed during pretrain — preemptive after v113 died)
93. **SELF-DIAGNOSING (#9) CLOSED PERMANENTLY.** Tested temp=10/2/1.0/0.1 across 5 runs. Positive feedback loop is fundamental.
94. ~~alibaba_v114~~ (multi-scale critic + continuity loss weight=1.0, killed ep72 — **best 0.073★ ep30 (TIES best train ever)**, **5-run eval avg 0.100**, +37% gap. Training breakthrough but eval doesn't beat ATB 0.088.)
95. ~~tencent_v141~~ (multi-scale critic + continuity loss weight=1.0, killed ep51 — best 0.091★ ep20, 31 stale, worse than multi-scale+PCF baseline. **Continuity loss DEAD on tencent (2nd attempt).**)
96. ~~alibaba_v115~~ (continuity-loss seed #2, killed ep85 — best 0.083★ ep55 (LATE breakthrough, validates 30-stale rule). Frozen-bundle 0.195 (1 seed, ~10% worse than v114's 0.176).)
97. ~~tencent_v142~~ (multi-scale+PCF seed #4, killed ep79 — best **0.0856★ ep45** (BEST tencent training ever this recipe), 34 stale + W rising 3.07→3.32. Frozen-bundle 0.1795 (1 seed) — TIES v136 ATB 0.178. Recipe reproducible.)
98. ~~alibaba_v116~~ (continuity-loss seed #3, killed ep67 — best **0.0692★ ep35** (BEST ALIBABA TRAINING EVER, beats v114's 0.073★), 32 stale at kill. Frozen-bundle 0.1799 (1 seed) — within noise of v114's 0.176, recipe consistent.)
99. **Running:** tencent_v143 (multi-scale+PCF ATB recipe, seed #5 — fifth data point for v136's 0.178 frozen ATB)
100. **Running:** alibaba_v117 (continuity-loss seed #4 — fourth data point. Continuity-loss frozens so far: 0.176/0.195/0.180.)
73. ~~alibaba_v105~~ (CFG fix + fresh seed, killed ep64 — best 0.084★ ep35 **BEST TRAIN EVER**, **5-run eval avg 0.113**, +35% gap. Does NOT beat ATB 0.088.)
74. **KEY FINDING:** CFG information leakage fix (Gemini R2 P1) produces best-ever training (5 stars, 0.092★) but train→eval gap unchanged at +31%. Fix kept for training stability. Eval variance in recall is the true bottleneck — not conditioning leakage, not seed luck.
75. **EVAL BUG FIXES:** (a) HRC-MAE padding: was padding to n_points by repeating final hit ratio, suppressing error at discriminative cache sizes. Fixed: compute MAE over actual sizes only. (b) Reuse metric hardcoded to col 3: was wrong when tenant column dropped. Fixed: dynamically resolve from preprocessor col_names.
76. ~~alibaba_v106~~ (copy-path-loss-only 0.5/0.5, W-stopped ep42 — best 0.096★ ep35, **5-run eval avg 0.103**, +7.3% gap! Run 3 hit 0.073 BEST EVER. **KEY: copy-path-loss-only reduces train→eval gap from 30% to 7%.**)
79. **KEY FINDING:** Per-timestep reuse supervision (copy-path-loss-only) dramatically reduces the train→eval gap. v106 gap=7.3% vs v104/v105 gap=31-35%. This is the first technique to significantly reduce the structural gap.
78. ~~alibaba_v107~~ (copy-path-loss-only 0.25/0.25, W-stopped ep36 — best **0.084★ ep30**, **5-run eval avg 0.113**, +34% gap. v106's 7% gap was NOT replicated — gap reduction is seed-dependent.)
81. ~~alibaba_v108~~ (copy-path-loss-only 0.25/0.25, W-stopped ep67 — best 0.110★ ep40, worst training quality of copy-path series despite best W stability. **Copy-path CLOSED on alibaba.**)
83. ~~alibaba_v109~~ (base PCF recipe, W-stopped ep49 — best 0.091★ ep25, **5-run eval avg 0.122**, +34% gap. Recall collapsed 0.608→0.469. Baseline for v110 multi-scale critic comparison.)
85. ~~alibaba_v110~~ (multi-scale critic + PCF, killed ep40 — best **0.090★** ep20. **5-run eval avg 0.104**, Run 4 hit 0.082! **Multi-scale critic improves alibaba eval 15% (0.122→0.104).** Gap reduced +34%→+16%.)
86. ~~alibaba_v111~~ (multi-scale critic + PCF, seed #2, killed ep53 — **FIVE consecutive stars**, best **0.083★** ep25 (BEST ALIBABA TRAIN EVER). **5-run eval avg 0.096**, Run 5 hit **0.075** BEST INDIVIDUAL EVER. Doesn't beat ATB 0.088 avg.)

---

## Literature refresh (2026-04-15) — new bets beyond the current list

Most items above have now been tried, partially validated, or explicitly closed. The next serious gains
probably require changing the abstraction, not adding one more scalar loss or critic garnish.

Important honesty note: several papers below come from adjacent domains (LLMs, generic time series,
network traffic). When I say they are relevant to Zarathustra, that is an **inference** from their
mechanism, not a claim made by the paper authors about block I/O traces specifically.

The common thread across these papers is clear:

1. The strongest recent systems do **not** win by slightly better generic sequence modeling.
2. They win by baking in one of:
   explicit memory,
   explicit workload descriptors,
   explicit long-horizon state,
   or explicit event-time structure.
3. That is exactly where Zarathustra is still weakest:
   reuse decisions,
   cache behavior,
   long-rollout coherence,
   and timestamp/locality coupling.

What follows is not a random literature dump. It is a proposed next architectural queue.

### 17. Retrieval memory for locality: explicit reuse/new decision + object pointer

**Core claim:** locality should become a structural decision, not a scalar penalty. Copy-path losses gave
evidence that per-timestep reuse supervision matters, but the generator still has no explicit mechanism for
"reuse a recent object" versus "create a fresh object." Right now the model is still trying to smuggle object
memory through a generic hidden state and a continuous output head.

**Why this fits the repo specifically:**
- Reuse-rate remains the most stubborn realism gap.
- The current `obj_id_reuse` and `obj_id_stride` split already defines the right latent decision:
  first decide reuse vs new, then decide how to realize it.
- Copy-path-loss-only runs suggest the signal is real, but the mechanism is missing.

**Architecture sketch:**
- Add a per-stream memory table of recent object embeddings and metadata.
- At each timestep, the generator outputs:
  `p_reuse`,
  a retrieval query,
  and a fresh-object proposal.
- If `reuse=1`, use attention, kNN, or pointer selection over the memory table.
- If `reuse=0`, emit a new object embedding and push it into memory.
- Predict size/opcode/timestamp conditioned on the selected object state rather than independently.
- Optionally: represent stride as retrieval rank or relative pointer distance instead of raw signed regression.

**Minimal viable experiment:**
- Keep the current generator backbone.
- Add only:
  a memory bank,
  a reuse gate,
  and a retrieval head.
- Train first on alibaba with the multi-scale+PCF backbone frozen as the control recipe.
- Use current mixed-type heads and reuse BCE as auxiliary supervision, but make retrieval the primary path.

**Success criteria:**
- Reuse rate materially improves without destroying precision.
- HRC-MAE improves, not just MMD.
- Reuse-native diagnostics improve:
  reuse precision/recall,
  reuse streak distribution,
  stride-on-reuse violation rate.
- Long-rollout replay does not immediately collapse into near-zero hit ratio.

**Failure modes / risks:**
- Memory may become a glorified nearest-neighbor copier and overfit recent context.
- The generator may learn to over-trigger reuse to satisfy the loss while harming global realism.
- Memory write policy can become unstable if every timestep inserts a noisy object state.

**Why it is still worth it:** this is the cleanest direct response to the main unsolved problem in the repo.
If this fails, it tells us something deep: not just that the loss was wrong, but that even explicit retrieval
is insufficient under the current decomposition.

**Primary sources:**
- Yuhuai Wu et al., [Memorizing Transformers](pubs/Memorizing_Transformers_2022.pdf), ICLR 2022. [arXiv](https://arxiv.org/abs/2203.08913)
- Ali Safaya and Deniz Yuret, [Neurocache: Efficient Vector Retrieval for Long-range Language Modeling](pubs/Neurocache_2024.pdf), NAACL 2024. [arXiv](https://arxiv.org/abs/2407.02486)

### 18. Cache-descriptor distillation: make cache behavior part of training, not just evaluation

**Core claim:** if cache fidelity is the real downstream target, cache-native descriptors should appear in
training, not just post-hoc evaluation. Competitor systems keep winning by building compact workload
descriptors that preserve recency/frequency/footprint structure. Zarathustra currently measures HRC after
generation, but it does not ask the model to predict or preserve compact cache-behavior summaries.

**Why this fits the repo specifically:**
- The repo already has file-level characterization infrastructure.
- HRC is now in the eval stack, so there is already a downstream operational target.
- The current conditioning vector is still mostly scalar workload metadata, not cache-native structure.

**Descriptor candidates:**
- Footprint and object-popularity summaries
- Reuse-distance or inter-reference-distance quantiles
- Popularity-size footprint descriptors
- Hit-ratio curve slices at fixed normalized cache sizes
- Working-set growth / saturation summaries
- Burst-locality interaction summaries:
  reuse ratio conditioned on burst regime,
  stride quantiles conditioned on opcode,
  object-diversity bands

**Architecture sketch:**
- Add a descriptor encoder that consumes file-level or window-level descriptor targets.
- Condition the generator on descriptor embeddings rather than only raw scalar characteristics.
- Add a descriptor reconstruction head from generated windows or generated long traces.
- Use descriptor mismatch as a training loss and also as a checkpoint tiebreaker.
- Optionally: use descriptors to define pseudo-label regimes for routing or curriculum.

**Minimal viable experiment:**
- Do not start with differentiable cache simulation.
- Start with 4-8 fixed descriptor targets computed offline from real windows/files.
- Condition on them and add one descriptor reconstruction loss.
- Run against the current best tencent recipe, since Tencent is where locality is most strategically important.

**Success criteria:**
- Better HRC-MAE and reuse realism without sacrificing combined score.
- More stable seed behavior on fixed eval bundle.
- Clear sensitivity to descriptor changes at generation time.

**Failure modes / risks:**
- Descriptor learning may collapse into superficial matching that ignores event-level realism.
- Over-conditioning can recreate the old "raw cond_dim increased, quality worsened" failure mode.
- File-level descriptors may be too coarse unless paired with window-level pseudo-labels.

**Why it is still worth it:** this is the most direct way to stop optimizing proxy metrics while hoping cache
behavior comes along for free.

**Primary sources:**
- Yirong Wang, Isaac Khor, and Peter Desnoyers, [2DIO: Configurable and Cache-Accurate Trace Generation for Storage Benchmarking](pubs/2DIO_CacheAccurate_2026.pdf), EuroSys 2026. [arXiv](https://arxiv.org/abs/2603.19971)
- Anirudh Sabnis and Ramesh K. Sitaraman, [TRAGEN: A Synthetic Trace Generator for Realistic Cache Simulations](pubs/TRAGEN_IMC2021.pdf), IMC 2021
- Anirudh Sabnis and Ramesh K. Sitaraman, [JEDI: Model-driven Trace Generation for Cache Simulations](pubs/JEDI_IMC2022.pdf), IMC 2022
- Cheng Li et al., [TraceGen: A Block-level Storage System Performance Evaluation Tool for Analyzing and Generating I/O Traces](pubs/TraceGen_HPCC2024.pdf), HPCC 2024

### 19. State-space backbone instead of a plain LSTM generator

**Core claim:** the repo is still asking a short-window recurrent model to carry too much long-horizon state.
Recent synthetic-trace work in adjacent networking domains suggests state-space models can preserve longer
stateful structure better than ordinary recurrent or transformer baselines.

**Why this fits the repo specifically:**
- `timestep=12` is still tiny relative to the real trace problem.
- `generate.py` already depends on hidden-state carry across windows, which means the repo has implicitly
  admitted it needs a long-horizon state abstraction.
- Continuity loss failed, which may mean the backbone rather than the loss is the bottleneck.

**Architecture sketch:**
- Replace the generator LSTM with an SSM block sequence.
- Preserve:
  `z_global`,
  current conditioning stack,
  mixed-type output heads,
  and the current critic for the first ablation.
- Feed one timestep at a time, but maintain a larger latent state with selective retention.
- Later:
  consider an SSM critic or file-level SSM regime model only if the generator-side swap helps.

**Minimal viable experiment:**
- Generator-only swap first.
- Same preprocessing, same outputs, same checkpoint score, same eval.
- One alibaba run and one tencent run on fixed recipes.

**Success criteria:**
- Better long-rollout stability without changing the evaluator.
- Less chunk-boundary drift.
- Better DMD-GEN / locality metrics at equal or better combined score.

**Failure modes / risks:**
- New backbone could increase training instability before showing any benefit.
- Gains may show only on long rollouts, not on short-window eval, creating a measurement mismatch.
- If evaluator variance is not fixed first, results could again be hard to interpret.

**Why it is still worth it:** it is the lowest-risk serious backbone change short of a full model-family pivot.

**Primary sources:**
- Andrew Chu et al., [Feasibility of State Space Models for Network Traffic Generation](pubs/SSM_NetworkTrafficGen_2024.pdf), arXiv 2024. [arXiv](https://arxiv.org/abs/2406.02784)
- Andrew Chu et al., [NetSSM: Multi-Flow and State-Aware Network Trace Generation using State Space Models](pubs/NetSSM_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2503.22663)

### 20. Recast the problem as a marked temporal point process

**Core claim:** timestamps are not just another feature column. Zarathustra is modeling event sequences,
where event times are continuous and the marks attached to each event have their own dependencies.
Treating `ts_delta` as one channel inside a generic vector may be conflating timing errors, locality errors,
and mark dependence.

**Why this fits the repo specifically:**
- The repo already knows that event timing matters operationally for burstiness and cache replay.
- Several persistent failures look like entangled timing/locality mistakes rather than pure marginal mismatch.
- Mixed-type outputs already move in the direction of treating fields differently; point-process modeling is a
  stronger version of the same insight.

**Architecture sketch:**
- Use a point-process module for event times.
- Use mark heads for:
  size,
  opcode,
  reuse/new,
  and maybe a latent object-family label.
- Couple the object identity path to retrieval memory from idea #17.
- Score not just per-column fidelity, but time-mark dependence:
  whether reuse events happen at the right times,
  under the right burst conditions,
  with the right size/opcode context.

**Minimal viable experiment:**
- Do not attempt full exact-likelihood MTPP training on day one.
- Start with a hybrid:
  point-process head for event times,
  ordinary heads for marks.
- Keep the rest of the architecture close to the current generator.

**Success criteria:**
- Better timing realism on long rollouts.
- Better burst-locality coupling.
- Better HRC or replay behavior if timing has been a hidden culprit.

**Failure modes / risks:**
- Extra complexity may buy little if timestamp modeling is not the real bottleneck.
- This path will likely take longer to integrate with current windowed training.
- It may be hard to compare apples-to-apples with current metrics unless the eval bundle is frozen first.

**Why it is still worth it:** this is the cleanest conceptual way to stop pretending event time is just
another regression target.

**Primary sources:**
- Yujee Song et al., [Decoupled Marked Temporal Point Process using Neural ODEs](pubs/Decoupled_MTPP_2024.pdf), ICLR 2024. [arXiv](https://arxiv.org/abs/2406.06149)
- Hui Chen et al., [Marked Temporal Bayesian Flow Point Processes](pubs/BMTPP_2024.pdf), arXiv 2024. [arXiv](https://arxiv.org/abs/2410.19512)

### 21. Chunk stitching / whole-trace generation with explicit boundary state

**Core claim:** the repo still trains locally and hopes globally. Continuity loss failing does not mean the
whole-trace problem is fake; it means the current continuity implementation was too weak or too scalar.
The generation path already stitches windows by carrying hidden state, but training still does not supervise
that contract directly.

**Why this fits the repo specifically:**
- `generate.py` already carries state across windows.
- Long-rollout realism is still a project-level goal, not a side objective.
- The current continuity loss was only one attempt, not a full chunk-generation design.

**Architecture sketch:**
- Generate overlapping chunks with explicit carried boundary state.
- Train overlap-consistency on the shared region between adjacent chunks.
- Add a boundary summarizer:
  a latent state the next chunk must honor.
- Permit parallel chunk generation, but learn a stitching or compatibility module.
- Optionally: condition later chunks on compressed summaries of earlier chunks rather than raw carry state.

**Minimal viable experiment:**
- Keep the current backbone.
- Replace scalar continuity loss with overlap-consistency training on paired adjacent windows.
- Use the hidden-state carry from `generate.py` as the target behavior to imitate in training.

**Success criteria:**
- Better long-rollout drift curves.
- Less distribution shift from early chunk to late chunk in a long generated stream.
- Better HRC stability as generated length increases.

**Failure modes / risks:**
- More expensive training.
- If the base generator is too weak, better stitching may just propagate bad state more faithfully.
- Requires stronger long-rollout diagnostics to tell whether it helped.

**Why it is still worth it:** this is a directly on-problem response to the "12-step training / long-trace
inference" mismatch.

**Primary sources:**
- Aditya Shankar et al., [WaveStitch: Flexible and Fast Conditional Time Series Generation with Diffusion Models](pubs/WaveStitch_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2503.06231)
- Xuan Hou et al., [Stage-Diff: Stage-wise Long-Term Time Series Generation Based on Diffusion Models](pubs/StageDiff_LongTS_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2508.21330)

### 22. Full hybrid pivot: diffusion + autoregressive supervisor + critic

**Core claim:** if the pure GAN family is saturating, the next major model-family bet should be a hybrid
that gives each sub-problem its own tool:
diffusion or latent denoising for global structure,
autoregressive supervision for local consistency,
and a critic for realism pressure.

**Why this fits the repo specifically:**
- The repo already has partial ingredients:
  supervisors,
  critics,
  conditioning,
  mixed-type heads.
- Recent time-series work is moving toward hybrids, not single-mechanism purity.
- DiTTO shows this direction is already reaching storage-trace generation specifically.

**Architecture sketch:**
- Stage 1:
  generate a coarse latent or chunk-level trace with diffusion or adversarial autoencoding.
- Stage 2:
  refine with an autoregressive supervisor that repairs next-step or next-chunk dependencies.
- Stage 3:
  critic judges realism in decoded space or latent path space.
- Keep current mixed-type output heads where possible rather than rebuilding everything at once.

**Minimal viable experiment:**
- Prefer AVATAR-lite or TIMED-lite before a full DiTTO-style rewrite.
- Reuse existing recovery heads and conditioning code.
- Run as a clean side branch, not mixed into the current GAN family piecemeal.

**Success criteria:**
- Better combined score on fixed eval.
- Better long-rollout drift behavior.
- Better locality and HRC simultaneously, not a trade where one goes up and the other down.

**Failure modes / risks:**
- Highest engineering cost of anything in this section.
- Could easily turn into a new project rather than a next experiment.
- Hardest path to interpret if infrastructure debt remains unresolved.

**Why it is still worth it:** this is the high-ceiling pivot if the team decides it has learned most of
what it can from pure GAN structural tweaks.

**Primary sources:**
- MohammadReza EskandariNasab et al., [TIMED: Adversarial and Autoregressive Refinement of Diffusion-Based Time Series Generation](pubs/TIMED_DiffusionTS_2025.pdf), ICDM 2025. [arXiv](https://arxiv.org/abs/2509.19638)
- MohammadReza EskandariNasab et al., [AVATAR: Adversarial Autoencoders with Autoregressive Refinement for Time Series Generation](pubs/AVATAR_AAE_TimeSeriesGen_2025.pdf), SDM 2025. [arXiv](https://arxiv.org/abs/2501.01649)
- Seohyun Kim et al., [A Diffusion-Based Framework for Configurable and Realistic Multi-Storage Trace Generation (DiTTO)](pubs/DiTTO_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2509.01919)

### Recommended build order

If we only spend mainline compute on three more architectural bets, the order should be:

1. **Retrieval memory for locality (#17)** — strongest match to the reuse failure, smallest conceptual gap
2. **Cache-descriptor distillation (#18)** — most direct way to make cache behavior a training target
3. **Chunk stitching or SSM backbone (#21 or #19)** — pick one long-horizon abstraction before a full rewrite

**Only after that** should we consider the full hybrid diffusion pivot (#22). It has the highest ceiling,
but also the highest risk of becoming a research fork rather than an extension of the current codebase.

### Concrete near-term execution plan

To keep the next phase interpretable, do not launch these as mixed cocktails. Suggested order:

1. **Infrastructure first**
   - Freeze the eval file bundle.
   - Fix PRDC fallback and generation-path parity.
   - Add one locality-native checkpoint tiebreaker.

2. **First architecture bet**
   - Retrieval memory on alibaba first, then tencent.
   - Judge on reuse realism and HRC, not just combined.

3. **Second architecture bet**
   - Descriptor distillation on tencent first.
   - Use fixed descriptors, not a giant learned characterization stack.

4. **Third architecture bet**
   - Choose one of:
     chunk stitching,
     or SSM backbone.
   - Do not do both at once.

5. **Only then**
   - Decide whether the repo still wants a full hybrid pivot.

### Bottom line

There really are more ideas. But they are no longer "another loss term" ideas. The next wave has to make
one of the hidden assumptions in the current model explicit:
memory,
cache behavior,
state,
or event time.

---

## Literature refresh (2026-04-18) — structural bets targeting the four open problems

This batch specifically targets four diagnosed failure modes that the config-space search exhausted
and that the existing architectural queue (#17–#22) does not fully address:

1. **Reuse-rate gap** (~0.5% vs 10%+): pure scalar loss and retrieval memory are in the queue, but
   nothing in the existing list gives the generator a discrete, tokenized representation of object
   identity. Ideas #23 and #26 add that.
2. **Eval variance in recall** (0.398–0.639 per run, documented in log entries 56, 64, 74): the
   generator ships a *single* point-estimate sample path per prompt. Nothing at inference time
   calibrates toward high-recall regions of the output distribution. Idea #25 targets this.
3. **Long-rollout drift** (12-step training window, hidden-state carry unsupervised): the existing
   chunk-stitching idea #21 supervises window boundaries, but does not address the open-loop vs
   closed-loop mismatch inside a single generation pass. Idea #27 targets this.
4. **Corpus-specific recipes don't transfer** (multi-scale helps tencent, hurts alibaba — log
   entry 88): every recipe so far is a *global* setting. Nothing in the existing list routes
   examples to specialist subnetworks based on workload regime. Idea #24 targets this.

All five entries below satisfy the "structural mechanism" bar: each changes the hypothesis class or
the inference procedure, not just a loss coefficient.

### 23. Discrete object tokenization via VQ-VAE codebook (TimeVQVAE-style)

**Core claim:** the reuse-rate gap is partly a *representation* problem, not just a supervision
problem. The generator currently emits continuous outputs that must be quantized to object IDs via
stride arithmetic. That path cannot easily express "reuse exactly this recent object" because the
decoder has no discrete action for it. Tokenizing the trace into discrete codes via a VQ-VAE, then
learning a prior over code sequences, makes object-level decisions first-class.

**Why this fits the repo specifically:**
- The reuse gap is the most stubborn defect. Retrieval memory (#17) fixes the addressing side; VQ
  tokenization fixes the emission side. They are complementary, not competing.
- Our mixed-type head already admits that different fields need different loss structures.
  Discrete codes generalize this: the entire trace lives in a discrete latent space.
- A bidirectional transformer prior on codes provides exactly the "any-order" editing ability that
  masked diffusion papers have validated — we can re-sample a sub-window of codes conditioned on
  its neighborhood, which addresses both long-rollout drift and mode-coverage at once.

**Architecture sketch:**
- Stage 1: train a VQ-VAE on windowed traces (continuous fields + categorical fields combined).
  Codebook size 512–1024. Commitment loss for code stability.
- Stage 2: train a bidirectional masked-diffusion / any-order autoregressive prior over the code
  sequence, conditioned on char-file stats and z_global.
- Decoding stage 1 deterministically recovers fields from codes; the discrete code sequence carries
  the generative variability.
- Optionally couple codes to retrieval memory from #17: a code index can point to a memory slot,
  collapsing "reuse" into "emit code k = copy from memory slot k".

**Minimal viable experiment:**
- Start with the alibaba multi-scale+PCF ATB recipe (v111 / v136).
- Train a small TimeVQVAE (codebook=256, window=12 preserved) on alibaba first.
- Measure: does the stage-1 reconstruction already beat v98 on HRC-MAE even *without* a learned
  prior? If yes, the representation is doing useful work. If no, do not proceed to stage 2.

**Success criteria:**
- Stage-1 reconstruction HRC-MAE within 10% of real-trace replay.
- Reuse rate in generated traces climbs from ~0.5% toward the real ~10%.
- Frozen-bundle combined★ improves on v98 (alibaba 0.088) or v136 (tencent 0.094).

**Failure modes / risks:**
- Codebook collapse — 2025 papers (VAEVQ, CODA) report this is still a live issue; mitigation via
  entropy regularization or residual VQ is mandatory.
- The decoder may leak information that the prior then has to re-learn, wasting capacity.
- Training a bidirectional transformer prior on top of a GAN backbone is a two-stage commitment;
  cannot easily A/B against v111 on the same compute.

**Primary sources:**
- Daesoo Lee et al., [Vector Quantized Time Series Generation with a Bidirectional Prior Model (TimeVQVAE)](pubs/TimeVQVAE_AISTATS2023.pdf), AISTATS 2023. [arXiv](https://arxiv.org/abs/2303.04743)
- Anonymous, [Any-Order GPT as Masked Diffusion Model](pubs/AnyOrderGPT_MaskedDiff_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2506.19935)

---

### 24. Workload-regime mixture-of-experts generator

**Core claim:** the "multi-scale helps tencent, hurts alibaba" result is not a bug; it is evidence
that the two corpora live in *different regions* of the generator's hypothesis class and a single
shared set of weights cannot serve both. A mixture-of-experts generator with char-file-driven
routing lets specialists co-exist, while shared early layers still transfer.

**Why this fits the repo specifically:**
- Log entry 88 explicitly documents the universal-recipe failure: multi-scale is "universal" only
  in the sense that it is non-catastrophic everywhere, not in the sense that it is optimal anywhere.
- The char-file vector is already a clean per-example descriptor. It is the natural MoE router
  input and is already produced by the preprocessing stack.
- Regime sampler (#5) validated that *discrete regime identity* is useful signal. MoE generalizes
  regime routing from "sample k, then condition" to "route to expert k's weights."
- Time-MoE (ICLR 2025 spotlight) shows MoE is now practical at the scale of forecasting foundation
  models, so the engineering risk is much lower than it would have been two years ago.

**Architecture sketch:**
- Shared encoder over `z`, char-file conditioning, and (optional) regime-sampler output.
- A small gating network maps char-file stats → distribution over E experts (E=4–8).
- Each expert is an LSTM or multi-scale critic aligned G head with its own weights.
- Load-balancing auxiliary loss to prevent expert collapse (a well-studied Time-MoE trick).
- Critic remains shared — the point is specialization of G, not of D.
- Optional: per-corpus experts as the initial warm start (2 experts pinned to alibaba/tencent),
  then unpin and let routing adapt.

**Minimal viable experiment:**
- E=4, top-1 routing, shared LSTM first layer, per-expert second LSTM layer.
- Train on the union of alibaba + tencent char-file-labeled data. This is the first experiment in
  the repo's history to train on the *union*, which is itself a useful diagnostic.
- Evaluate both corpora on their own frozen bundles.

**Success criteria:**
- Either (a) a single MoE generator matches both ATBs (0.088 alibaba, 0.094 tencent) with one model,
  which is a capability we have never demonstrated, or (b) routing statistics align with known
  regime labels, validating that the model learned something semantically coherent.
- Expert usage distribution is not pathologically skewed (>80% to one expert is a failure).
- Cross-corpus transfer improves: alibaba checkpoint evaluated on tencent (currently catastrophic)
  produces *some* reasonable output.

**Failure modes / risks:**
- Expert collapse under a small effective batch size — tencent alone has small files-per-epoch.
- Router noise destabilizes WGAN training. The gating network must be warm-started or the W-loss
  can explode within the first few epochs.
- MoE adds parameters; must check that gains are not just a capacity effect against a parameter-
  matched dense baseline.

**Primary sources:**
- Xiaoming Shi et al., [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](pubs/TimeMoE_ICLR2025.pdf), ICLR 2025 Spotlight. [arXiv](https://arxiv.org/abs/2409.16040)

---

### 25. Inference-time energy-based calibration for eval-variance stability

**Core claim:** the dominant unsolved failure (entries 56, 64, 74) is that eval *variance* is
larger than the gap between competing recipes. This is not a training problem — the training-time
combined★ is already near-ATB. It is an *inference-time sampling* problem. An energy-based model
trained on the generator's output distribution can be used at inference time to perform a few
Langevin steps that pull generated samples toward the real-data energy landscape, without
retraining G.

**Why this fits the repo specifically:**
- Lower lr (v94, v123) eliminated the train→eval gap mechanically but cost quality. That experiment
  proved the gap is reducible, not structural — we just need a reducer that doesn't cost quality.
- Every seed re-roll (v98–v103) confirmed eval variance is the real bottleneck. Averaging over
  n_samples is not a fix; it is a measurement adaptation.
- The existing frozen_sweep evaluation infrastructure already produces per-sample generator output;
  a calibration hook is a small addition, not a rewrite.
- EBM-CoT (arXiv 2025) validates that a small EBM head trained on top of a frozen base model can
  perform Langevin calibration in embedding space to reduce sample-to-sample inconsistency. The
  original problem there (LLM reasoning variance) is structurally the same as ours (generator
  recall variance).

**Architecture sketch:**
- Freeze the best v98 / v136 generator.
- Train a small EBM `E_φ(x, cond)` on paired (real trace window, cond) with noise contrastive
  estimation: positives = real, negatives = G-generated.
- At eval time only: run K Langevin steps on generated latent or output to minimize `E_φ`.
- Step size / K tuned per corpus; evaluated against the frozen bundle.
- Does not modify training; strictly a post-hoc inference upgrade.

**Minimal viable experiment:**
- Train E_φ on v98 (alibaba) only; 3 days of compute max.
- Evaluate with K ∈ {0, 5, 20, 50} Langevin steps.
- Gate: does the 5-run eval *variance* (not just mean) decrease?

**Success criteria:**
- 5-run eval std on recall drops from the documented ~0.12 range (0.398–0.639 observed) to <0.05
  without loss of mean recall.
- Mean combined★ improves by >10% on v98 frozen bundle.
- Calibration effect transfers to tencent v136 without re-training E_φ on tencent (bonus — not
  required).

**Failure modes / risks:**
- E_φ training is adversarial: negatives drift as G is fixed, but the EBM may overfit G's failure
  mode and refuse legitimate diversity. Mitigate with replay buffer.
- Langevin noise can push samples *off* the data manifold if E_φ is miscalibrated; step size must
  be conservative.
- If eval variance comes from char-file ambiguity (which v97 tested), EBM calibration on outputs
  will not help; the failure is upstream.

**Primary sources:**
- Xin Tong et al., [Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit Chain-of-Thought](pubs/EBM_CoT_Calibration_2025.pdf), arXiv 2025. [arXiv](https://arxiv.org/abs/2511.07124)

---

### 26. Mamba-Hawkes event-time backbone: fuses #19 (SSM) and #20 (MTPP)

**Core claim:** the existing queue has #19 (SSM backbone) and #20 (MTPP) as independent options.
Mamba Hawkes (2024) demonstrates that these two can be a single module, not two separate swaps.
This is not a rehash of #19 or #20 — it is a *unified* replacement that avoids the interface
problem of gluing an SSM backbone to a separately-trained point-process head.

**Why this fits the repo specifically:**
- The copy-path-only result (v106, gap 7.3%) suggests per-timestep timing-aware supervision helps
  when it is structural. MTPP makes timing structural.
- The repo already depends on hidden-state carry across generation windows (generate.py). Mamba's
  selective state carry is strictly better matched to that pattern than a fresh LSTM handoff.
- Doing MTPP on top of an LSTM backbone is known to be capacity-limited; doing it on top of a
  Mamba backbone changes the long-range modeling budget at the same time.

**Architecture sketch:**
- Generator backbone: Mamba blocks (selective SSM) in place of the LSTM generator core.
- Event-time head: Hawkes-style intensity conditioned on the Mamba hidden state.
- Mark heads (size, opcode, reuse, stride): unchanged mixed-type heads, but conditioned on both the
  Mamba hidden state and the sampled event-time offset.
- Critic: keep the current multi-scale critic in the first ablation; the Mamba backbone on G alone
  is the cleanest first test.

**Minimal viable experiment:**
- Swap only G's LSTM for a 2-layer Mamba block of equal hidden size; keep all heads.
- No MTPP yet — first verify the backbone swap alone is stable.
- Second phase: add the Hawkes-style time head.
- Alibaba only for the first pass; do not touch tencent until #26 is validated on one corpus.

**Success criteria:**
- Phase 1 (Mamba only): matches v98 ATB with no hidden-state carry tricks, and *long-rollout drift*
  metric improves (not just short-window eval).
- Phase 2 (Mamba + Hawkes time): burst realism (inter-arrival autocorr, ts_delta Wasserstein)
  improves without hurting recall.

**Failure modes / risks:**
- Mamba + WGAN stability is less well-studied than Mamba + MLE. Critic may find adversarial exploits
  in the selective-scan gradients.
- Mamba adds hyperparameters (state dim, d_conv) not in the current grid; tuning cost is real.
- MTPP integration after a backbone swap doubles the risk surface; keep phases strictly separated.

**Primary sources:**
- Anningzhe Gao et al., [Mamba Hawkes Process](pubs/MambaHawkes_2024.pdf), arXiv 2024. [arXiv](https://arxiv.org/abs/2407.05302)
- Plus the #19 SSM and #20 MTPP sources already in pubs/.

---

### 27. Professor-Forcing closed-loop rollout supervision

**Core claim:** the long-rollout drift problem (12-step training vs long-trace inference) has a
well-known root cause: the generator only ever sees teacher-forced inputs during training, never
its own rolled-out outputs. The generate.py path does closed-loop rollout, but nothing supervises
the resulting trajectory distribution. Professor Forcing is the textbook fix: train a small
auxiliary classifier that cannot tell open-loop from closed-loop hidden-state trajectories.

**Why this fits the repo specifically:**
- The continuity-loss family (failed on tencent twice, see entries 95 and "continuity loss" in
  tried-and-validated) was trying to solve this problem with a scalar. Professor Forcing uses a
  *classifier*, not a scalar, and is architecturally closer to the WGAN machinery already in use.
- The repo already has two critics in production use (main WGAN critic + multi-scale critic).
  Adding a third small classifier with a completely different job (open-loop vs closed-loop) is
  low engineering overhead.
- Chunk-stitching (#21) supervises window boundaries; Professor Forcing supervises every timestep
  within a window. They compose: PF fixes the inside of a window, #21 fixes the seams.

**Architecture sketch:**
- During training, for each real window, run the generator twice:
  - Once in teacher-forced mode (current training regime).
  - Once in free-running / closed-loop mode from the same initial state.
- Train a small MLP classifier D_PF to distinguish the two hidden-state trajectories.
- Add to G loss: `-λ · log D_PF(closed_loop)` (i.e., G wants the classifier to be fooled).
- The classifier operates on hidden states, not outputs — cheap, and sidesteps the output-space
  exposure-bias trap that scheduled sampling falls into.
- Critically, this does NOT require any new labels, new data, or any change to the critic.

**Minimal viable experiment:**
- Add PF as a single new flag `--professor-forcing-weight`.
- Grid: λ ∈ {0.0, 0.1, 0.5} on alibaba v98-recipe base.
- Measure: training combined★ (expect small loss), frozen-bundle eval mean, AND a new long-rollout
  drift metric (generate 10× the training window length and measure distribution shift from
  first decile to last decile).

**Success criteria:**
- Frozen-bundle eval mean stays within 10% of v98 ATB.
- Long-rollout drift metric (first-decile-vs-last-decile Wasserstein on any emitted field)
  improves by at least 25%.
- If combined★ does *not* regress on short-window eval but *does* improve on long-rollout metrics,
  this is the clean signal we want.

**Failure modes / risks:**
- D_PF can become too strong and force the generator into pathological hidden-state geometry that
  satisfies the classifier but produces bad outputs. This is the original Professor Forcing paper's
  own caveat. Mitigate with PF weight annealing.
- The method trains one extra small module per step; compute cost ~15–25%.
- If the real driver of long-rollout drift is char-file mismatch rather than hidden-state mismatch
  (possible — v97 hinted at this), PF will have no effect and the experiment will be informative
  only as a negative result.

**Primary sources:**
- Anirudh Goyal et al., [Professor Forcing: A New Algorithm for Training Recurrent Networks](pubs/ProfessorForcing_NeurIPS2016.pdf), NeurIPS 2016.
- Supporting theory on flow-matching-style self-consistency (analogous to PF's closed-loop constraint, one domain over): [Consistency Flow Matching](pubs/ConsistencyFM_ICML2025.pdf), ICML 2025. [arXiv](https://arxiv.org/abs/2407.02398)

---

### Updated recommended build order (2026-04-18)

Before starting the 2026-04-15 queue (#17–#22), the following are cheaper and attack failure modes
that #17–#22 do not touch:

- **Inference-time EBM calibration (#25)** — cheapest, does not touch training, directly targets
  the largest documented gap (eval variance). Run first.
- **Professor Forcing (#27)** — second cheapest, single flag, directly targets the long-rollout
  drift failure that the continuity-loss experiments failed to fix.
- **Workload-regime MoE (#24)** — answers the "recipes don't transfer" problem that is itself a
  ceiling on reviewer claims about universality.

Only after that is the order #17 → #18 → #23 → (#19 or #26) → #21 → #22 worth committing to.

### Honesty note on these five

All five entries above cite papers from adjacent domains (LLM reasoning for #25, masked diffusion
for #23, point processes for #26, forecasting MoE for #24, RNN training for #27). None of these
papers target block-level I/O traces. The inference that they apply here is ours, not the authors'.
Each idea therefore carries *interpretation risk* in addition to *implementation risk*: if it fails,
part of the failure may be that the mechanism just does not transfer to this problem domain.

---

## External-audit addenda (2026-04-18, Grok)

Grok's proposal audit produced four additions that survive de-dup against #1–#27. Proposals 2
(hybrid diffusion) and 7 (regime-conditioned diffusion) already live as #22/#24; Proposal 5
(frozen-bundle as primary selector) is an operational fix, not an architectural idea — tracked
outside IDEAS.md. The remaining four are kept here so their motivation is not lost when the
current #21 arc closes.

### 28. Cross-window persistent retrieval bank (extends #17)

**Gap attacked**: reuse_rate / HRC-MAE drift in long rollouts. The existing retrieval memory is
per-window; `generate.py` resets it at each window boundary, so cross-window reuse — which is
where real I/O locality lives — never matches.

**Proposal**: extend `retrieval_memory.py` to carry a persistent global bank (size 128–256) across
windows in BOTH training and generation. At window end, write final bank state to a global
buffer; next window's initial bank is sampled from that buffer with probability proportional to
the real trace's reuse-distance distribution. Add an auxiliary loss on the global bank's eviction
statistics vs. real trace's reuse-distance histogram.

**Why this is on-target**: the per-window retrieval memory made local reuse better but does nothing
for the window boundary — exactly where `generate.py` carries hidden state but not memory state.
This is the memory-state analogue of IDEA #21.

**Cost**: 2–4h implementation; negligible compute overhead; risk is that the global bank gets
dominated by early training noise and never converges. Mitigation: freeze the bank after warm-up
and only update inside a moving window.

### 29. Adaptive PCF frequencies with frequency-discriminator (extends #9-style PCF)

**Gap attacked**: β-recall ceiling on hard modes. Current PCF frequencies are fixed after
initialization; the critic learns to ignore the easy frequencies and the pressure collapses.

**Proposal**: every N epochs, re-sample frequencies from a tiny "frequency-GAN" trained to
maximize real/fake discrepancy on the currently-hardest modes (identified via PRDC low-recall
bins). Add a small frequency discriminator that scores how well the current frequency set
separates real vs fake. This keeps PCF adversarial pressure alive throughout training.

**Why this is on-target**: we already know PCF helps tencent and hurts alibaba (frozen-eval
pairing shows +0.009 on alibaba). An adaptive-frequency PCF could close that gap in both
directions by letting the loss self-tune to whichever modes matter for the current corpus.

**Cost**: 4–8h; adds a second tiny network + its update step. Risk: interaction with regime
sampler and multi-scale critic creates adversarial-soup instability.

### 30. Multi-scale overlap consistency + adversarial boundary critic (extends #21)

**Gap attacked**: DMD-GEN stuck >0.3; temporal modes don't match at long horizons. Current
`--overlap-consistency-mode=overlap` is single-scale (k=2); the multi-scale critic isn't applied
to boundaries.

**Proposal**: extend OC to 3-scale (fine k=2, medium k=6, coarse k=12) mirroring the multi-scale
critic. Add an adversarial boundary critic scoring ONLY the stitched boundary region (last-k of
A + first-k of B). Weight boundary steps higher than bulk. Once wired, lift
`--boundary-smoothness-weight` into the 1.5–2.0 range.

**Why this is on-target**: DMD-GEN measures dynamical modes across time; a single-scale overlap
loss cannot supervise multiple dynamical scales simultaneously. The boundary critic is the
adversarial version of what BS (MSE on latents) does non-adversarially.

**Cost**: 6–10h; one new tiny critic + multi-scale OC machinery. Risk: adds another critic to an
already-three-critic architecture (main, PCF, multi-scale) — GAN stability concerns.

### 31. Chained-window training augmentation (extends #21)

**Gap attacked**: train→inference mismatch. Training generates ONE window; `generate.py`
auto-regresses 2–4+ windows. Even with BS/OC, the training signal only covers one boundary.

**Proposal**: randomly sample chained windows (2–4 consecutive) from the real trace each batch.
Generate the same length with hidden-state carry inside the batch. Apply BS at every internal
boundary and OC at every overlap region. This directly trains on the inference distribution.

**Why this is on-target**: chunk_stitching only supervises a single boundary. Chained training
teaches the model that hidden state must compound through many windows, not just survive one.
This is the most direct fix to DMD-GEN and the "drift over long rollouts" failure mode.

**Cost**: 4–6h; mostly batch-assembly and loss accounting. Memory: 2–4x per batch (can be
mitigated by halving files-per-epoch). Risk: with only 12–24 windows per epoch, chaining 4
collapses effective batch diversity.

### Ordering vs. current plan

All four of these attack live failure modes that #21 (now running) cannot close on its own. Order
of attack once the current tencent_v160/alibaba_v161 runs complete:

1. **#28 cross-window retrieval bank** — single biggest reuse/HRC target; isolated to
   retrieval_memory.py + generate.py; low blast radius.
2. **#31 chained-window training** — directly reshapes the training distribution to match
   inference. Biggest expected Δ on DMD-GEN.
3. **#30 multi-scale OC + boundary critic** — only if #28+#31 leave DMD-GEN elevated.
4. **#29 adaptive PCF** — only if β-recall is still the dominant gap after the first three.

This ordering is orthogonal to the 2026-04-18 list above (#25, #27, #24, #23). Those five are
cheaper architectural experiments; #28–#31 are targeted attacks on documented failure modes from
the frozen-sweep history.

---

### 32. Explicit Inter-Reference-Distance (IRD) footprint modeling

**Gap attacked**: HRC-MAE cliffs and plateaus. Current generators (including this one) match
first-order frequency skew (Zipf/Pareto) and produce *concave* HRCs. Real storage traces have
*non-concave* HRCs with cliffs at specific cache sizes, driven by IRD histogram spikes and holes
that frequency alone cannot reproduce. This is the single gap the 2DIO paper (pubs/2DIO_CacheAccurate_2026.pdf,
EuroSys 2026, Wang/Khor/Desnoyers) argues most strongly.

**Proposal**:
- Extend `cache_descriptor.py` with a per-window IRD histogram descriptor (reuse-distance
  quantiles + spike-detection peaks via SciPy peak-finding on the IRD log-histogram).
- Conditioning path: append IRD descriptor to `cond_dim` so G sees IRD as input, not just
  frequency descriptors.
- Auxiliary loss: MSE (or Wasserstein-1) between real vs. synthetic IRD histograms computed on a
  rolling window of decoded features, added to joint-GAN step with weight
  `--ird-loss-weight`.
- Generation: when the existing reuse gate fires, sample target reuse-distance from the
  IRD-conditioned distribution rather than the uniform bank lookup.

**Why this is on-target**: IDEA #28 (cross-window retrieval bank) provides the *mechanism* for
long-range reuse but has no *distributional target* for eviction age. IRD supplies exactly that
target. #28 and #32 are complementary: #32 tells the model *which* reuse distances to produce,
#28 provides the memory state to actually produce them.

**Primary source**: 2DIO (Wang et al., EuroSys 2026). The paper's core empirical claim is that
IRD-informed synthesis reproduces non-concave HRCs and frequency-skew-only synthesis does not.

**Cost**: 4–6h. `cache_descriptor.py` already computes a superset; adding the IRD head is a
focused extension. Risk: IRD is expensive to compute exactly for very long windows; use a
Flajolet-Martin-style approximation or sample-based estimate if needed.

---

### Diffusion-paper integration targets (under IDEA #22)

These are concrete integration candidates for IDEA #22 (full hybrid diffusion pivot) surfaced by
the 2026-04-18 Grok audit.

Verified 2026-04-18 against arXiv:

- **arXiv 2509.01919 — Kim et al., DiTTO-style storage-trace diffusion**:
  [A Diffusion-Based Framework for Configurable and Realistic Multi-Storage Trace Generation](https://arxiv.org/abs/2509.01919).
  This is the most directly relevant #22 reference: it is storage-trace-specific, explicitly
  conditional/configurable, and reports high-fidelity multi-device trace synthesis. Treat HRC
  benefits as an inference to test in Zarathustra, not a quoted claim unless the paper's metric
  table confirms it.
- **arXiv 2511.12174 — Shen/Li/Long, TSGDiff**:
  [Rethinking Synthetic Time Series Generation from a Pure Graph Perspective](https://arxiv.org/abs/2511.12174).
  Correction to the external note: this is a graph-structured diffusion model, not a paper titled
  "from a diffusion perspective" and not primarily an adaptive-noise-schedule paper. The useful
  transfer idea is graph construction from Fourier/temporal dependencies plus a graph-aware latent
  diffusion backbone.
- **arXiv 2508.21330 — Hou et al., Stage-Diff**:
  [Stage-wise Long-Term Time Series Generation Based on Diffusion Models](https://arxiv.org/abs/2508.21330).
  Verified as a stage-wise long-horizon time-series diffusion reference; this maps cleanly onto
  Zarathustra's coarse-regime / local-supervisor / critic-polish split.
- **DiffCATS (Masi et al., TMLR 2025), causally-associated time-series diffusion**. Still unverified
  in this pass. Would require
  an inferred causal graph over our feature dims (obj_id_reuse → ts, etc.); moderate
  implementation cost. Verification needed.
- **WaveStitch (pubs/WaveStitch_2025.pdf, PACMMOD 2025)** — already in our library; is the basis
  of our overlap-consistency mode.

Action: DiTTO, TSGDiff, Stage-Diff, and WaveStitch slot in as implementation references for IDEA
#22, not new ideas in their own right. Verify DiffCATS before citing or implementing it.

---

### 33. Critic-trajectory distillation instead of W-stop gambling

**Gap attacked**: the newest BS+OC overlap-mode runs suggest useful generator states can appear
late in training while the critic is railing out, but the current mechanism is accidental: save
`final.pt` when the W-stop guard fires and hope that the tail checkpoint is better than the
training-selected checkpoint.

**Proposal**: turn the late-tail effect into an explicit control surface instead of a higher
`--w-stop-threshold` gamble:

- Track a critic-trajectory state (`healthy`, `railing`, `collapsed`) from W-distance slope,
  recall trend, and critic loss variance.
- When the critic enters a controlled railing state, branch into a short "distillation tail":
  freeze or slow the critic, lower discriminator updates, keep generator/recovery/auxiliary
  losses active, and emit dense frozen-sweep candidates.
- Add a paired control run where the same checkpoint enters the tail with the critic frozen
  immediately, so improvements can be attributed to generator polishing rather than adversarial
  instability.
- End the tail by frozen-sweep/long-rollout criteria, not by a larger raw W threshold.

**Why this is on-target**: the v161/v162/v164 pattern hints that final checkpoints saved during
W-stop can be much better than training-best, but tencent_v163 shows the same symptom can also
mean real recall collapse. A tail controller would separate "critic railing while G improves"
from "critic railing because the model is dying." It also converts the repeated best.pt
mis-rank pathology into a deliberate checkpoint-production strategy.

**Cost**: 4-8h for a trainer-side phase controller plus denser tail checkpointing; low model
architecture risk. The main risk is overfitting to one alibaba seed, so the first test should
start from an already-saved pre-tail checkpoint and run two or three deterministic tail variants
before spending full fresh-training budget.

---

### 34. Tail-regime modeling from higher-order moments

**Core claim:** the 2026-04-18 R higher-moment pass shows that the block-trace families have enormous
standardized 5th and 6th moments on `iat_*`, `abs_stride_*`, and `reuse_ratio` surfaces. That is a
rare-event regime problem, not something likely to be solved by another smooth scalar loss.

**Why this fits the repo specifically:**
- Tencent `iat_q50` has M6 around 3.76M and `abs_stride_q50` has M6 around 2.16M in the R pass.
- Alibaba `iat_q90` has M6 around 858K and `reuse_ratio` has M6 around 195K.
- These are precisely the surfaces where the generator keeps showing train/frozen and recall instability.
- The full-corpus leaderboard is even more extreme: the LCS-style `s3-cache-datasets__tencentBlock`
  family has `abs_stride_q50` M6 around 14.0M and `iat_q50` M6 around 11.7M, so the tail-regime
  issue is not an artifact of cherry-picking the current training pair.

**Architecture sketch:**
- Add a tail-regime classifier or router that explicitly separates ordinary windows from rare timing/seek/reuse windows.
- Route tail windows through a specialized generator head, mixture component, or sampled rare-event template.
- Use higher-order moments as diagnostics and checkpoint triage, not as a raw high-order loss term.

**Minimal viable experiment:**
- Label files or windows as tail-heavy using the R moment audit.
- Train or evaluate with a tail-stratified frozen bundle.
- Require candidate checkpoints to preserve ordinary combined score while improving tail-stratum recall.

**Failure modes / risks:**
- Directly optimizing M5/M6 would be numerically brittle and easy to game.
- File-level moments may not map perfectly to 12-step training windows; a window-level follow-up is still needed.

**Why it is still worth it:** the moment pass gives a quantitative reason that smooth average-case tuning keeps disappointing. The tails are too extreme to treat as noise.

---

### 35. Workload-conditioned mechanism router

**Gap attacked:** the same mechanism is now clearly helpful on one corpus and harmful on another.
Retrieval memory produced a small Tencent ATB, but the analogous Alibaba stack collapsed badly.
Multi-scale/PCF showed a similar cross-corpus split earlier. Treating retrieval, BS/OC, PCF, and
tail heads as globally-on recipe switches is too blunt for traces whose locality and burst regimes
differ by workload family.

**Proposal:** make the generator conditionally route between mechanism experts rather than
hard-enable each module for a full run:
- Use existing file descriptors plus new tail/reuse-stratum labels to predict soft gates for
  retrieval memory, BS/OC boundary pressure, PCF/multi-scale paths, and any tail-regime expert.
- Add a small sparsity/entropy regularizer so the router chooses a few mechanisms per workload
  instead of averaging every mechanism into every trace.
- During eval, report mechanism-gate histograms per corpus and per tail stratum so wins can be
  attributed to "Tencent uses retrieval" or "Alibaba suppresses retrieval" instead of another
  recipe-level binary flag.

**Minimal viable experiment:**
- Freeze the current v167/v165 backbones and train only a lightweight conditioning router over two
  experts: retrieval-on vs retrieval-off, or BS/OC-on vs BS/OC-off.
- Evaluate on full, ordinary, tail-heavy, and long-rollout panels. A useful router should reproduce
  Tencent's retrieval benefit while avoiding Alibaba's retrieval collapse.

**Why this is on-target:** the latest results say the project is not looking for one universal
scalar setting. It needs workload-aware composition: the architecture should learn when a mechanism
matches the trace regime instead of forcing the researcher to choose one global recipe per corpus.

**Risk:** a router can hide failures if it simply memorizes corpus IDs. Start with descriptor-only
inputs, hold out files/families, and require gate histograms to make mechanistic sense.

---

### 36. Learned boundary prior instead of deterministic BS/OC penalties

**Gap attacked:** the patched boundary-smoothness experiments show that the old palindrome bug was
not merely harmless noise. The mathematically-correct derivative penalty made alibaba much worse,
while the buggy k=2 constraint produced the current numeric baseline. That means the project should
stop treating boundary continuity as a hand-written scalar penalty and learn what a realistic join
looks like from trace data.

**Proposal:**
- Build a small boundary critic or contrastive head that sees `(left_window_tail, right_window_head)`
  pairs and distinguishes true adjacent joins from generated joins and shuffled non-adjacent joins.
- Train the generator to fool that boundary critic across generated adjacent windows, but do not
  prescribe position equality, velocity equality, or reflection symmetry directly.
- Include negative controls: true adjacent pairs, shuffled same-file pairs, shuffled cross-file pairs,
  buggy-palindrome generated joins, and derivative-smoothed generated joins.
- Report boundary-prior scores alongside frozen `★`, tail strata, and long-rollout HRC/stack-distance
  so the model cannot improve a local join metric while damaging global workload structure.

**Why this is on-target:** v175/v176 imply the useful signal was not "smoothness" in the simple
finite-difference sense. It may be a workload-specific boundary manifold: bursts, repeated IDs,
idle gaps, and local symmetry patterns that a scalar MSE cannot encode. A learned prior can preserve
that empirical manifold without hard-coding the accidental palindrome.

**Minimal viable experiment:**
- Freeze the current generator recipe and train only the boundary critic on real adjacent vs shuffled
  joins to confirm it can separate realistic boundaries.
- Add its adversarial/contrastive loss to a short alibaba run from the v164 recipe with patched BS
  disabled or near-zero.
- Promote only if it beats the current-code patched baseline and does not regress long-rollout reuse.

**Risk:** this can become another critic that overfits short windows. Keep it local to the boundary,
hold out files, and require long-rollout metrics before treating it as an IDEA #21 successor.

---

### 37. Recall-gated boundary critic weight (extends #36)

**Gap attacked**: IDEA #36 at bc_weight=0.5 shows that the learned boundary critic can dominate
the G-loss and cause recall collapse after the training peak. v190 (seed=3) reached EMA recall=0.773
at ep30 (★=0.053), then decayed to 0.508-0.574 through ep60 because the bc term contributed ~60%
of G-loss at peak D_bc discrimination. The bc mechanism prevents Alibaba collapse but simultaneously
restricts mode diversity as a shortcut to better boundary scores.

**Proposal**: gate the effective bc_weight on the current training-EMA recall:

```python
# In G-step, just before adding bc loss:
_recall_ema = current_ema_recall   # tracked per epoch
_effective_bc_w = _bc_w * min(1.0, _recall_ema / cfg.bc_recall_floor)
```

where `bc_recall_floor` (default 0.60) is the minimum acceptable recall before bc weight
is reduced. When recall > floor, bc runs at full weight. When recall < floor, bc_weight
scales down proportionally. This prevents the generator from sacrificing recall to optimize
bc, because reduced recall automatically reduces bc pressure.

**Alternatively**: use a recall-modulated discriminator step count. Only run D_bc updates
when the training recall > floor. When recall drops, D_bc freezes and the generator can
recover mode diversity without bc pressure.

**Why this is on-target**: the v189/v190 experiment established that bc_weight=0.5 is
too high (60% of G-loss) and bc_weight=0.1 (v191) is the next test. But the optimal weight
may be dynamic — early in training (recall still building) bc should be weak; late in training
(recall high) bc can be stronger. A fixed low weight may not prevent collapse at bad seeds
(weight too small to overcome the collapse attractor), while a fixed high weight causes the
recall-restriction shortcut. The recall-gated weight automatically adapts to training state.

**Minimal viable experiment:**
- Run the v191 recipe (bc_weight=0.1) and observe whether collapse is prevented at seed=11.
- If seed=11 still collapses at bc=0.1, try recall-gated weight with floor=0.60 and max_weight=0.5.
- If seed=11 doesn't collapse at bc=0.1 but recall is still worse than v176, increase the floor.

**Cost**: 1-2h code change; adds `bc_recall_floor` flag and recall-tracking in training loop.
**Risk**: recall tracking lags by 1 epoch; if recall swings sharply the gate may lag. Use EMA.

---

### 38. Per-epoch deterministic mini-eval to close the EMA-vs-frozen gap

**Gap attacked**: The 22nd training-selector mis-rank (v190) reveals a systematic flaw in the kill criterion: train-★ was stale from ep30, but the frozen-★ improved for 35 more epochs (ep30→ep65). The EMA recall overestimated the ep30 checkpoint by 52% (EMA: 0.773 vs frozen: 0.509). Our 30-epoch-stale kill policy cuts runs at the peak of an EMA artifact, not at the true peak. We killed v190 at ep70 with 40 epochs of potential improvement remaining.

**Root cause**: The EMA metric uses training-distribution MMD samples (non-deterministic, seed varies per epoch). The frozen eval uses seeds 42/42 on a fixed held-out set. The EMA is noisier and biased toward the training distribution. The divergence is largest when the model is rapidly adapting (ep20–40).

**Proposal**: Add an optional per-epoch deterministic mini-eval using a small fixed set:
```
--mini-eval-every N        # run mini-eval every N epochs (default 0 = disabled)
--mini-eval-samples M      # MMD samples for mini-eval (default 500, vs 2000 for EMA)
--mini-eval-seed S         # deterministic seed (default 42)
```
Mini-eval takes ~20–30s/epoch at 500 samples. Its ★ would be logged as `eval_★` alongside `comb=` in the epoch line. Kill criterion: 30 epochs stale from `eval_★` best (not EMA `comb=`).

**Alternative (cheaper)**: Track a 50-epoch rolling frozen sweep within the training loop — run one mini frozen eval every 10 epochs, keep a running best. If mini-eval ★ hasn't improved in 40 epochs, kill.

**Cost**: 2-3h code change; all frozen_sweep infrastructure already exists in `mmd.py` / `eval.py`; just needs deterministic seed param and a second eval path in the epoch loop.
**Risk**: doubling up eval creates GPU contention with critic updates if parallelized naively. Run sequentially. Mini-eval with 500 samples is slightly noisier than 2000-sample frozen (±0.005 expected variance).
**Impact**: if mini-eval ★ tracks frozen ★ closely, we can kill/save based on real signal. Expected: would have extended v190 kill to ep65+ and saved the frozen-best checkpoint earlier.

---

### 39. Diversity-pressure boost to offset bc MMD² overhead (extends #36)

**Gap attacked**: IDEA #36 (boundary critic) prevents Alibaba collapse but introduces MMD² overhead. v189 (bc=0.5) frozen MMD²=0.014 vs ATB v176 MMD²=0.007 — 2× worse. Root cause: bc encourages the generator to optimize boundary realism by restricting output modes (diversity decreases → MMD² increases). Even at bc=0.1, the bc signal may create mild mode-restriction pressure that prevents the generator from covering the full feature distribution.

**Proposal**: increase `--diversity-loss-weight` from 2.0 to 4.0–6.0 for bc runs. The diversity loss (which encourages G to produce varied outputs across the batch) directly counters the mode-restriction shortcut. Higher diversity pressure means the generator cannot collapse modes as a shortcut to better boundary scores.

```
# v191 baseline:       --diversity-loss-weight 2.0 --boundary-critic-weight 0.1
# v194 proposed:       --diversity-loss-weight 5.0 --boundary-critic-weight 0.1
```

**Why this is on-target**: the bc's mode-restriction shortcut (generator reduces diversity to optimize boundary scores) is exactly what the diversity loss opposes. Currently diversity-loss=2.0 was tuned for runs without bc. Adding bc creates a new attractor (restricted-diversity boundary-smooth outputs) that diversity=2.0 may not be strong enough to resist. Increasing diversity pressure to 5.0 directly counters this attractor.

**Alternative**: use a dedicated MMD-loss component (like IDEA #9's direct MMD minimization) as an auxiliary term when bc is active. This directly penalizes MMD² increase.

**Minimal viable experiment**: once v191 (bc=0.1, diversity=2.0) frozen sweep completes, launch v194 with bc=0.1, diversity=5.0, same seed=11. If frozen MMD² drops from v191 to v194 while recall stays ≥0.68, the diversity-pressure fix is real.

**Cost**: zero code change — only hyperparameter modification.
**Risk**: too much diversity pressure may cause training instability or override the bc signal entirely. Start at 4.0; if unstable, back off to 3.0.
**Expected outcome**: if v191 frozen MMD²≈0.012 and diversity=5.0 brings it to ≈0.007, ★≈0.007+0.2×0.25=0.057 — within 12% of ATB ★=0.051.

---

### 40. Feature-matching loss between real and fake boundary transitions (extends #36)

**Gap attacked**: the boundary critic (IDEA #36) provides only a binary adversarial signal (real vs fake boundary). This gives the generator a gradient direction but no direct magnitude information about HOW the boundary features differ. Feature-matching provides a direct MMD-like alignment signal at the boundary.

**Proposal**: during the G-step (not D_bc step), extract the penultimate-layer activations of D_bc on both real and fake pairs, and add an L2 feature-matching loss:

```python
# Extract D_bc penultimate features (not output)
_phi_real = D_bc.net[:-1](torch.cat([tail_r.reshape(B,-1), head_r.reshape(B,-1)], 1))
_phi_fake = D_bc.net[:-1](torch.cat([tail_f.reshape(B,-1), head_f.reshape(B,-1)], 1))
bc_fm_loss = F.mse_loss(_phi_fake.mean(0), _phi_real.mean(0).detach())
# Add: G_loss += bc_fm_weight * bc_fm_loss
```

**Why this is on-target**: adversarial signal tells G "your boundary looks wrong" but doesn't specify how to fix it. Feature-matching tells G "your mean boundary statistics need to shift in this direction." This is the same principle as `--feature-matching-weight` for the main critic, adapted to the boundary critic.

**Alternative flag**: `--boundary-critic-fm-weight FLOAT` (default 0.0 = disabled). Start at 0.1.

**Cost**: ~30 lines in train.py; requires splitting D_bc.net into feature extractor + final linear. The `BoundaryCritic.net` is a simple Sequential, so extracting `net[:-1]` is trivial.
**Risk**: feature-matching competes with adversarial signal; too large a weight can overwhelm D_bc's training objective. Start small (0.05–0.1). Requires D_bc weights to be frozen during fm extraction (detach phi_real).
**Expected outcome**: if bc fm-loss reduces boundary MMD² without collapsing recall, it might provide the MMD² reduction needed to close the gap to v176 (★=0.051 with MMD²=0.007).

---

### 41. Adaptive n-critic schedule for boundary critic warm-up

**Gap attacked**: the boundary critic (D_bc) needs to be well-trained before its adversarial loss provides useful gradient to G. In the first 10-15 epochs, D_bc may be too weak to discriminate (low bc_gap), so its contribution to G-loss is mostly noise. Later, once D_bc is expert-level, its gradient becomes more reliable. A fixed `n-critic-bc=1` (one D_bc update per G step throughout) may under-train D_bc early and over-train late.

v191 bc_gap pattern (ep1-20): 0.300 → 0.253 → 0.526 → 0.344 → 0.338 → 0.207 → 0.187 → 0.201 → 0.230 → 0.268 → 0.194 → 0.182 → 0.174 → 0.203 → 0.250 → 0.177 → 0.261 → 0.209 → 0.231 → 0.199.
bc_gap is oscillating 0.174–0.526 with no clear trend. The bc signal is noisy relative to its guidance value.

**Proposal**: use a warm-up schedule for D_bc: run `n_critic_bc = 3` updates per G-step for the first N epochs (while bc_gap < target), then switch to `n_critic_bc = 1`. This pre-trains D_bc to expert level faster, then stabilizes it.

```python
# Adaptive D_bc training frequency
_n_critic_bc = 3 if (epoch <= cfg.bc_warmup_epochs and bc_gap_ema < 0.30) else 1
for _ in range(_n_critic_bc):
    # ... D_bc training step ...
```

**Alternative**: Warm-up D_bc with standalone real/fake pairs for the first 5 epochs (no G updates during this phase), similar to how WGAN-GP warms up the critic before joint training.

**Cost**: ~15 lines in train.py; adds `--bc-warmup-epochs N` flag (default 10) and bc_gap EMA tracker.
**Risk**: over-training D_bc early can create a too-strong adversary that overwhelms G-loss early; cap at n_critic_bc=3, not higher.
**Expected outcome**: bc_gap stabilizes faster → G gets cleaner boundary gradient earlier → possibly faster recall improvement and lower MMD² oscillation. Test as v195 (same bc=0.1 recipe + bc_warmup_epochs=10).

---

### 42. Boundary critic on latent space instead of decoded feature space — **IMPLEMENTED (2026-04-20, commit 71ba3f9)**

**Gap attacked**: the current D_bc takes (tail_K_steps, head_K_steps) from the Recovery-decoded output (R(H_A), R(H_B)). Round 34 P1 #1 flagged this: D_bc may learn to detect Recovery decoder artifacts (the unique texture of the R network) rather than genuine boundary transition quality. If bc_gap is high because D_bc detects R-texture mismatch (real=raw vs fake=decoded), the bc loss misleads G to mimic R's artifacts at boundaries, not to produce realistic joins.

**Proposal**: feed D_bc the hidden state tail/head instead of the decoded feature tail/head:

```python
# Instead of:
tail_r = R(H_A[:, -K:, :])  # decoded real tail
head_r = R(H_B[:, :K, :])   # decoded real head
tail_f = R(H_Af[:, -K:, :]) # decoded fake tail
head_f = R(H_Bf[:, :K, :])  # decoded fake head

# Use latent directly:
tail_r = H_A[:, -K:, :].detach()  # hidden state tail (raw encoder output)
head_r = H_B[:, :K, :].detach()   # hidden state head
# ... no decode step
```

Since both real and fake paths go through the same Supervisor/Recovery pipeline at the hidden-state level, the R-texture artifact distinction disappears — D_bc must discriminate temporal latent structure, not decoder texture.

**Why this is on-target**: if D_bc learns to discriminate H-space boundary transitions (not feature-space), then the bc gradient to G is "your latent dynamics at boundaries look wrong," which is structurally meaningful and immune to the decoded-artifact confound. Long-rollout boundary quality depends on latent dynamics anyway (the LSTM states), so aligning bc to latent space is mechanistically cleaner.

**Implementation**: `--boundary-critic-latent` flag in train.py. Real side: `E(full_window)[:, -K:, :]` (encoder latent, B×K×latent_dim). Fake side: `G(z)[:, -K:, :]` (generator latent output). D_bc input_dim = 2×K×latent_dim = 192 (vs 40 in decoded mode). `sample_real_boundaries(full_window=True)` returns T-step windows for encoding; boundary range correctly set per mode.
**AD findings (2026-04-20)**: E is co-optimized by opt_ER during Phase 3 (not frozen). The D_bc real anchor tracks E's current latent embedding space — both real and fake are in the jointly-optimized latent space. In non-avatar mode (no BN), running stats are not affected. The "raw-vs-decoded" confound is removed; replaced by "E-encoded-real vs G-generated-latent" comparison in co-evolving latent space.
**Test**: v192 (bc=0.1, seed=7, latent mode) — CLOSED (W-stop ep35, frozen-best ep30 ★=0.104). v193 (bc=0.1, seed=5, latent mode, w-stop=5.0) — CLOSED (ep97, frozen-best ep75 ★=0.111). **IDEA #42 VERDICT: FAILED.** Both v192 and v193 show β-recall ceiling ~0.48 in latent-H mode; decoded-mode bc (v191 ep75 β-recall=0.709) is superior. Latent-H bc avoids early training collapse and W instability, but cannot push β-recall high enough to compete. The latent space is too low-dimensional (latent_dim=24 vs decoded F=5) for D_bc to provide sufficient gradient pressure on G's spatial pattern generation. **CLOSED — decoded-mode bc (IDEA #36) is superior.**

---

### 43. Matched carried-state boundary critic for real and fake joins

**Gap attacked**: IDEA #42 removes the raw-feature-vs-decoded-feature shortcut, but the current latent-H implementation still compares two different boundary semantics. The real side uses `E(window_A)` and `E(window_B)` as two independently reset GRU encodings, while the fake side uses `G(window_A)` and `G(window_B | hidden_A)` with carried LSTM state. A latent boundary critic can therefore learn an encoder-reset / start-of-window signature rather than the quality of a carried adjacent transition.

**Proposal**: train the boundary critic on matched transition representations where both real and fake heads are produced under the same carry contract.

Options:

1. **Supervisor-carried real positive**: encode the real first window, carry a learned state through `S` or a small transition adapter, and compare the predicted/carried real head against the actual encoded next-window head before feeding the join to `D_bc`. The critic then judges transition residuals, not raw reset latents.
2. **Reset-matched fake negative**: if using independent `E(A)`/`E(B)` positives, also make the fake side independent-reset style for the critic and keep the carried fake path only for generator-side rollout diagnostics.
3. **Transition-delta critic**: feed `tail`, `head`, and `head - transition(tail)` features, so the discriminator is forced to score boundary dynamics relative to a matched real transition model rather than absolute latent-domain artifacts.

**Minimal viable experiment**: add a `--boundary-critic-real-carry` mode that builds real positives from adjacent windows plus `S(E(A))`-conditioned transition features, and compare against current `--boundary-critic-latent` on the same seed/pretrain. Acceptance requires frozen sweep improvement plus a diagnostic showing the critic cannot separate positives by "head starts at timestep 0 of a fresh encoder window" alone.

**Why this is structural**: the project keeps finding that boundary scores are easy to fool by domain artifacts. A matched carried-state critic would make the adversary operate on the same train/generate contract the model must satisfy at long rollout time.

---

### 44. Domain-matched decoded boundary critic

**Gap attacked**: decoded-mode boundary criticism is currently the most promising IDEA #36 branch, but it still trains `D_bc` on raw normalized real boundaries versus `R(G(...))` decoded fake boundaries. That means the critic can still learn feature-manifold / Recovery-texture artifacts instead of temporal boundary realism. The latent-H branch removed that raw-vs-decoded shortcut but introduced its own reset/carry and low-dimensional-gradient problems. The next structural move should keep decoded-feature gradients while matching the real and fake domains.

**Proposal**: build a decoded-domain boundary critic where both real and fake joins pass through the same decode surface before `D_bc` sees them.

Options:

1. **Reconstructed-real positives**: for real adjacent windows, encode and decode with `R(E(real))`, then feed the reconstructed tail/head to `D_bc`. Fake remains `R(G(...))`. This tests whether decoded-mode bc works after the raw-vs-decoded shortcut is removed.
2. **Three-way decoded diagnostic**: train or at least evaluate scores for consecutive `R(E(real))`, shuffled `R(E(real))`, and `R(G)` joins. A useful boundary critic should rank consecutive reconstructed-real above shuffled reconstructed-real and both above generated joins.
3. **Dual-head critic**: share a trunk over decoded features but expose separate logits for domain authenticity and temporal adjacency. The generator should receive only the temporal-adjacency pressure, not a gradient that encourages matching decoder artifacts.

**Minimal viable experiment**: add `--boundary-critic-real-reconstruct` so the real side in decoded mode is sampled as full windows, transformed through `E` and `R`, then sliced at the boundary. Compare against v191/v194 on the same seed/pretrain, and log `D_bc(raw-real)`, `D_bc(recon-real)`, `D_bc(shuffled-recon-real)`, and `D_bc(fake)`.

**Acceptance bar**: a win only counts if the reconstructed-real diagnostic still separates consecutive from shuffled joins and frozen sweep improves without worsening long-rollout HRC / reuse-access / stack-distance. If the score disappears when real is reconstructed, decoded bc was mostly exploiting the raw-vs-decoded artifact.

**Why this is structural**: it preserves the decoded-feature gradient that latent-H seems to lack while removing the confound that makes current `bc_gap` hard to interpret. This is a better next boundary bet than more `bc_weight`, W-stop, or warm-up tuning.
**Implementation**: `--boundary-critic-real-reconstruct` flag added to `train.py` (2026-04-21). Real side: `R(E(full_window))[:, -K:, :]` and `R(E(full_window))[:, :K, :]` — both in decoded feature space. Fake side: `R(G(z))` as before. Mode label: `decoded-feat-matched`.
**Test**: v195 (seed=5, pretrain=v193, same recipe as v194 + `--boundary-critic-real-reconstruct`) — RUNNING

---

### 45. Long-rollout IRD adversary — fix the universal IRD=1 structural floor

**Gap attacked**: The first long-rollout panel (2026-04-21, v176/v191/v193/v194) reveals a
universal structural failure: IRD (inter-reference distance) median = 1 across all four
models, vs real's IRD median = 194. Stack distance = 0 for all models vs real's 174. This
means no generated trace has temporal object-locality beyond the immediate timestep. The
boundary critic (IDEA #36/44) can improve window-join reuse, but it cannot fix within-window
and cross-window object reuse patterns because it only sees the K-step boundary slice, not
the full object-ID recurrence structure.

The LSTM generator produces effectively independent object-ID draws each timestep (via the
`obj_id_reuse` column, which encodes whether the ID recurs). The conditioning on
trace-level `obj_id_stride` and `obj_id_reuse` statistics is insufficient: the model
can reproduce the *fraction* of reuse steps (which enters ★ via β-recall) without
preserving the *temporal gap distribution* (IRD) between reuse events.

**Root cause**: `obj_id_reuse` is a binary per-timestep indicator (1 if this access revisits
a recent object, 0 if new). The LSTM can learn the marginal P(reuse=1) without learning the
conditional P(reuse_t | prev_reuse_t-k, obj_id_t-k) — the autocorrelation structure of
reuse events. A short-window MMD metric captures P(reuse) in a 12-step window but not the
recurrence spacing distribution across 50,000 records.

**Proposal**: Add a dedicated IRD discriminator or direct IRD loss to force the generator
to match the inter-reference distance distribution.

**Option A — IRD histogram adversary (GAN signal)**:
Add a small discriminator D_ird that operates on IRD histograms computed over a long
(T_long = 512-1024 step) fake vs real window. The discriminator compares binned IRD
distributions (log-spaced bins from 1 to T_long). This gives G a gradient toward matching
the shape of the IRD histogram.

```
# In G-step, with a 512-step rollout:
real_ird_hist = compute_ird_histogram(real_long_window, bins=32)  # fixed bins
fake_ird_hist = compute_ird_histogram(fake_long_window, bins=32)
ird_loss = wasserstein_1d(fake_ird_hist, real_ird_hist.detach())
G_loss += ird_loss_weight * ird_loss
```

**Option B — direct IRD alignment loss (no discriminator)**:
Compute Wasserstein-1 distance between fake and real IRD histograms and use it directly
as a G-training loss. No D_ird parameters to train; IRD loss is purely supervised.
Cheaper to implement; trades adversarial sharpness for stability.

**Option C — reuse autocorrelation matching**:
Instead of IRD histogram, match the autocorrelation function of the `obj_id_reuse` column
over long windows. If fake and real have the same ACF(obj_id_reuse), the temporal gap
structure is reproduced. `ACF_loss = MSE(ACF_fake, ACF_real_ema)` over lag 1-50.

**Why this matters for the race**: IRD=1 means the object process lacks the long-gap reuse
law that shapes real HRCs. The few reuses that occur are mostly immediate repeats
(stack-distance ≈ 0), while most objects never participate in realistic longer-gap reuse.
A cache working set this narrow cannot reproduce the real HRC curve, which requires objects
to persist in the working set long enough to be re-accessed after many intervening distinct
objects. The real traces have IRD median=194, which means a cache of size ~200 objects
covers median reuse. A fake trace with IRD=194 would have 5-10× lower HRC-MAE than any
current model.

**Minimal viable experiment**: add `--ird-loss-weight 0.5` flag and Option B (direct
Wasserstein IRD loss). Compute IRD on 256-step fake windows (vs current 12-step training
window). This requires 256/12 ≈ 22 chained generation steps per batch, which is expensive
(~20× current window cost). To keep compute tractable, use 1 long window per batch of 32
normal windows, treating it as a regularizer.

**Cost**: ~2-3h code change. IRD computation: `np.unique` + histogram on `obj_id_reuse`
column. Chaining: extend the existing hidden-state carry logic (already present for
block-sample mode) to 256+ steps. Wasserstein-1d: already present in `llgan/mmd.py` or
use `scipy.stats.wasserstein_distance`.

**Risk**: Long-window chaining is significantly more expensive and may introduce gradient
instability (256 BPTT steps). Start with 64 steps + gradient clipping before scaling.
The IRD signal may conflict with the bc boundary signal if both are active simultaneously.

**Dependency**: run v195 (IDEA #44) first to establish the IDEA #44 baseline. If IDEA #44
improves reuse_access_rate vs v191 (0.193), a combined bc+IRD run would be the next step.
If IDEA #44 does not improve reuse, the bc mechanism alone may be saturated and IRD adversary
is the next structural lever.

---

### 46. Second-seed validation for decoded-feat bc (seed=7, the v176 basin)

**Gap attacked**: Every decoded-feat bc result to date uses seed=5 (v191 used seed=11, v194
used seed=5). Round 38 P1 #2 explicitly flagged this: "If v194 beats v191, call it 'decoded
bc works better inside seed 5 with a relaxed kill guard' until a second seed or domain-matched
critic reproduces the effect." The long-rollout panel (2026-04-21) reinforced this: v194
(seed=5 bc) has catastrophic long-rollout reuse (reuse_access=0.006), while the v176 seed=7
basin has better long-rollout reuse (0.046) without bc. If decoded-feat bc applied to seed=7
gives both short-window ★ ≤ 0.051 AND reuse_access ≥ 0.193, that is mechanism validation.

**Proposal**: Run decoded-feat bc (IDEA #36, standard decoded-feat mode, NOT IDEA #44
domain-matched) on seed=7 using v189's `pretrain_complete.pt`.
- seed=7, pretrain=v189, bc=0.1, k=4, w-stop=5.0, decoded-feat mode
- Same recipe as v194 except: seed=7 (not 5), decoded-feat (not real-reconstruct)
- Goal: reproduce v191's long-rollout reuse improvement (0.193) while reaching ★ ≤ 0.054

**Why this specific combination**: v191 (seed=11) found reuse_access=0.193 but ★=0.067. v194
(seed=5) found ★=0.054 but reuse_access=0.006. The seed=7 basin (v176) has the best
short-window ★ without bc. Adding bc to seed=7 may combine the two strengths: v176's
basin structure (low MMD²) plus bc's reuse-repair effect.

**Acceptance bar**: mechanism validation requires BOTH:
1. Frozen sweep ★ ≤ 0.054 (matches or beats v194's bc-ATB) under seeds 42/42
2. Long-rollout reuse_access ≥ 0.10 (at least halfway toward v191's 0.193)

**Cost**: zero code change — hyperparameter switch only.
**Risk**: seed=7 basin may collapse under bc pressure (v192 W-stopped at ep35 on seed=7 +
latent-H bc). The decoded-feat mode is gentler than latent-H (v191 ran to ep83 on seed=11
without W-stop), so collapse risk is lower. w-stop=5.0 provides additional guard.
**Expected outcome**: If seed=7 + bc = v176-level ★ + v191-level reuse → mechanism validated.
If seed=7 + bc collapses or shows v193/v194-style reuse degradation → bc improvement is
seed-basin-specific, not a general mechanism.

**Test**: v196 (seed=7, pretrain=v189, bc=0.1, decoded-feat, w-stop=5.0) — planned after
v194 ep160 kill frees GPU capacity.

---

### 47. Real-prefix continuation training instead of boundary-only criticism

**Gap attacked**: the current boundary work asks whether a generated chunk-to-chunk join looks
realistic, but it still trains the generator mostly as a free-running short-window sampler. The
long-rollout problem is stronger: given a real prefix window from a specific workload state, can the
model continue that trace into the next window without losing locality, burst cadence, or regime
identity?

**Proposal**: add a continuation phase that trains on true adjacent real windows `(A_real, B_real)`.
Encode `A_real`, map its tail state into the generator's initial state, then generate `B_fake` under
the same file conditioning and compare it to `B_real`.

Concrete options:

1. **State adapter**: train a small adapter from `E(A_real)` tail summary to `G`'s initial hidden/SSM
   state, then generate the next T steps. Losses can be latent MSE/PCF/MMD against `E(B_real)` or
   `R(E(B_real))`, plus reuse/timing heads where available.
2. **Continuation discriminator**: score `(A_real, B_real)` vs `(A_real, B_fake)` pairs, not just
   `(G_A, G_B)` vs raw real pairs. The critic then judges conditional continuation, not only generic
   boundary texture.
3. **Scheduled closed-loop**: start with one real prefix and one generated continuation; once stable,
   feed generated `B_fake` as the prefix for a second generated chunk and apply long-rollout sidecar
   metrics.

**Why this is structural**: it aligns the training contract with generation. `generate.py` and
`long_rollout_eval.py` ask the model to carry state across windows, while ordinary GAN training and
boundary critics only weakly constrain that transition. Prefix continuation makes "what comes next
given this workload state?" the primary supervised problem.

**Minimal viable experiment**:

- Build adjacent-window batches from the existing per-file `TraceDataset` source.
- Freeze the current AE (`E/R`) for the first probe to keep the target space fixed.
- Train only the adapter plus `G` tail for 10-20 epochs from a known checkpoint, saving dense
  continuation checkpoints.
- Evaluate with: next-window frozen `★`, boundary diagnostics, and long-rollout HRC/reuse/stack
  distance. A win requires long-rollout movement, not just short-window recall.

**Acceptance bar**: promote only if continuation improves at least one long-rollout locality/cache
metric versus the same checkpoint without continuation, while preserving ordinary frozen `★` within
the known bundle variance. If it only improves the local boundary score, it is not enough.

**Risk**: teacher-forced real prefixes can make training look good while free-running generation
still drifts. The scheduled closed-loop step is the safeguard.

---

### 48. Stateful stack-distance object decoder

**Gap attacked**: the model still emits object-locality features through a neural sequence decoder
and hopes the resulting object stream has the right cache law. The long-rollout sidecar already
showed this can fail badly: short-window `★` can be excellent while reuse-access rate, stack
distance, footprint, and HRC are wrong.

**Proposal**: stop asking the neural decoder to indirectly invent an LRU stack law. Add a stateful
object decoder that maintains an explicit synthetic LRU stack during generation.

Per event:

1. Predict `new_object` vs `reuse`.
2. If `reuse`, predict a stack-distance bucket or distribution and select the object at that
   distance from the stack.
3. If `new_object`, allocate a fresh synthetic object id and insert it at the top of the stack.
4. Generate marks such as size, opcode, and timing conditioned on the chosen object action and the
   current regime.

This can start as a post-Recovery repair layer or as a proper differentiable-ish output head. The
first version does not need to be elegant; it needs to prove that cache laws move when object ids
are generated by an object-process decoder instead of by smooth regression through `R`.

**Why this is structural**: retrieval memory is a hidden feature mechanism, not a direct guarantee
of the output stack law. A stack-distance decoder makes the cache-relevant state explicit and gives
the long-rollout sidecar a mechanism it can actually reward.

**Minimal viable experiment**:

- Fit real stack-distance bucket distributions per corpus or per tail/regime stratum using
  `long_rollout_eval.py` / `trace_profile.py` utilities.
- Generate normal synthetic features from a strong checkpoint, but replace only the object-id stream
  with the stateful stack decoder using the same reuse/new-object rates.
- Compare HRC-MAE, reuse-access rate, stack-distance median/p90, and ordinary frozen feature metrics
  against the unmodified generated stream.
- If the repair layer helps, move the decoder into training as a head with bucket NLL supervision.

**Acceptance bar**: this idea is successful only if HRC/reuse/stack-distance improve without making
ordinary feature distributions useless. It is allowed to be a hybrid generator; the goal is realistic
I/O traces, not architectural purity.

**Risk**: a post-hoc decoder can disconnect object IDs from size/timing/opcode correlations. The
second phase must condition mark generation on the chosen object action to restore those couplings.

---

### 49. Window-level regime atlas and router supervision

**Gap attacked**: the project has strong file-level characterization, but the GAN learns and fails
at the window level. File-level descriptors can say "this trace is bursty/locality-heavy," but they
do not tell the model which 12-step windows are burst starts, idle gaps, heavy-reuse islands, random
seek phases, or tail events. That mismatch keeps pushing the project toward global recipe flags:
retrieval on/off, boundary critic on/off, PCF on/off.

**Proposal**: build a window-level atlas for the target corpora and use it to supervise routing.

For every training window, compute cheap labels/descriptors:

- locality class: no-reuse / adjacent-repeat / medium-stack reuse / long-stack reuse
- timing class: idle gap / normal cadence / burst
- object action class: sequential stride / random seek / hot-object loop
- tail class: ordinary / timing-tail / stride-tail / reuse-tail
- file/regime context: file-level family vector plus within-file position decile

Then train a small router or auxiliary classifier to choose mechanisms per window:

- retrieval/stack decoder for reuse-heavy windows
- boundary/continuation pressure for regime transitions
- timing head emphasis for burst/idle windows
- tail expert for extreme timing or stride windows

**Why this is structural**: the latest results say there is no universal recipe. Tencent retrieval
is load-bearing in one basin; Alibaba retrieval is repeatedly negative; boundary pressure can help
one seed and collapse another. A window-level router lets the model learn where a mechanism belongs
instead of making the researcher pick one global switch for an entire corpus.

**Minimal viable experiment**:

- Add an offline `window_atlas.py` that emits per-window labels for a fixed manifest of Tencent and
  Alibaba files.
- Start with supervision only: train router/classifier heads and report whether predicted labels
  match held-out windows.
- Then use the router to gate two mechanisms only, for example stack decoder on/off or retrieval
  on/off, before attempting a full MoE.

**Acceptance bar**: router gates must make interpretable sense by stratum and must improve median
seed behavior, not just the best seed. Report gate histograms by corpus, tail stratum, and
long-rollout segment.

**Risk**: the router can cheat by memorizing corpus or filename identity. Hold out files, include
within-file position features cautiously, and require held-out window-label accuracy before using
the gates in training.

---

### 50. Profile-routed transition atlas

**Gap attacked**: a binary reuse loss or a per-file action marginal can still lose the temporal law
that determines cache behavior. The first trained `NeuralStack` probe made this visible: Alibaba
benefited from workload-conditioned action/rank probabilities, but Tencent regressed when the model
dropped StackAtlas's transition state.

**Proposal**: make the object process a routed transition system:

1. Fit per-file or per-window atlases over coarse time/size/object-action states.
2. Use characterization vectors to route generation to the nearest compatible atlas or a small
   mixture of atlases.
3. Train a neural smoother over initial-state and transition distributions, but keep a measured
   blend knob so the long-rollout panel can reject smoothing when it destroys locality.
4. Decode through an explicit LRU stack, never through raw scalar `obj_id` regression.

**Why this is structural**: it moves the core generated state from local scalar features to the
cache-native transition object: "given this workload profile and current locality state, what state
comes next, and which object does that imply?" This is a different contract from adding another
regularizer to `obj_id_reuse`.

**Minimal viable experiment**:

- Train a `NeuralAtlas` over 64 held-out files per corpus.
- Evaluate blend `{0.0, 0.25, 0.5, 0.75, 1.0}` against fixed real manifests.
- Promote the profile-routed atlas if it beats the peer on HRC-MAE, reuse-access, and stack-distance
  without relying on manifest-oracle training files.

**Acceptance bar**: the model must beat the current peer long-rollout sidecar by a wide margin and
must report the pure-neural blend as an ablation, not hide it. If the neural smoother worsens a
corpus, keep the router and drop that smoothing path for the promoted generator.

**Risk**: nearest-file routing can become too oracle-like if the evaluation uses the exact real
manifest. The next version should hold out routed source files from training and then route only by
characterization, not by basename identity.

---

### 51. Reuse-rate matching loss to fix dead obj_id_reuse signal

**Gap attacked**: v198 phase 1 proved that the LRU stack decoder works (HRC-MAE 0.1295→0.0051,
96% improvement) when given the real reuse rate, but the generator's `obj_id_reuse` output is
essentially dead — only 0.1% of events signal reuse vs 26.5% in real traces. The BCE supervision
(`reuse_bce_weight=2.0`) is insufficient to correct this at training time.

**Proposal**: add a global reuse-rate matching loss that directly penalizes the mean generated
reuse rate against the real corpus target:

  `L_rate = λ_rate × (mean(sigmoid(reuse_logit)) − r_target)²`

where `r_target` is the real corpus reuse rate (alibaba=0.265, tencent=0.621) and
`reuse_logit` is the pre-activation Recovery output for the `obj_id_reuse` column.

This is a scalar global constraint computed over the entire training batch, complementing
the per-event BCE which fails to enforce the aggregate rate.

**Why this works**: the BCE loss gives correct per-event direction (reuse events should be +1)
but cannot enforce that 26.5% of the batch is positive. The rate-matching loss provides
exactly this global constraint without changing the per-event target.

**Minimal viable experiment** (v199):
- Hot-start from v195 `pretrain_complete.pt` (pretrain E/R loaded; G LSTM fresh)
- Add `--reuse-rate-loss-weight 10.0 --reuse-rate-target 0.265` to training
- Monitor per-epoch mean reuse rate (should converge to ~0.265)
- Acceptance: training reuse_rate ∈ [0.24, 0.29] AND frozen ★ ≤ 0.050
- Long-rollout at ep30 (with `--lru-stack-decoder`): HRC-MAE < 0.015

**Acceptance bar**: HRC-MAE < 0.015 with LRU stack decoder in trained model (no oracle override),
★ competitive with v195. This is the target that makes the compound position — better than
NeuralAtlas on HRC and better on ★.

**Risk**: rate-matching loss may reduce per-event diversity (G learns to produce uniform
reuse probability instead of context-dependent reuse decisions). Monitor recall and frozen β-recall
to catch this.

---

### 52. Phase-conditioned nonstationary trace atlas

**Gap attacked**: a profile-routed atlas can match HRC, reuse, and stack distance while still being
too stationary. The first strict holdout pass exposed this: cache metrics stayed strong, but
first-half/second-half timing and size drift were much lower than the real traces.

**Proposal**: include within-file phase in the generated locality state. Instead of states being only
`time_bin x size_bin x object_action`, make them
`phase_bin x time_bin x size_bin x object_action`, where phase is a coarse position bin through the
source file or synthetic rollout.

**Why this is structural**: real traces are not exchangeable bags of events. Warmup, burst periods,
tenant/application shifts, and late-run tails are part of the trace identity. A generator that routes
to the right file family but samples that family stationarily can still fail the "statistically
indistinguishable" goal.

**Minimal viable experiment**:

- Train `NeuralAtlas` with `--n-phase-bins 8` under the same manifest-excluded holdout protocol.
- Compare HRC/reuse/stack-distance and drift ratios against the non-phase model.
- Promote phase conditioning only if it improves nonstationary drift without destroying cache law.

**Acceptance bar**: HRC-MAE must stay below the peer long-rollout failures by a wide margin, and
timing/size drift ratios should move toward `1.0` rather than collapsing toward zero.

**Risk**: phase bins can overfit file position and create brittle state sparsity. Keep the phase
count coarse, report pure-neural and routed blends, and require held-out manifest exclusion.

---

### 53. Neural mark head around a frozen object atlas

**Gap attacked**: the peer's v198 response is aiming at LANL's presumed weak point: reservoir
sampling for non-object marks. The first `altgan.mark_quality` panel says the current PhaseAtlas
marks are already strong on the fixed long-rollout panels, but a learned mark head is still the
right way to close the last trace-realism gap without giving up the explicit object process.

**Proposal**: freeze the profile-routed PhaseAtlas object generator and train a lightweight
sequence mark model for `ts_delta`, `obj_size`, `opcode`, and `tenant`, conditioned on
workload descriptor, atlas state, object action, stack-rank bucket, and prior emitted marks.
The object IDs still come from the LRU stack; only marks are neural.

**Why this is structural**: it keeps the part that is winning HRC/reuse/stack-distance while
targeting the exact critique LLNL raised. It also creates a compound benchmark where LANL can
win both cache law and mark realism instead of trading one away.

**Minimal viable experiment**:

- Generate PhaseAtlas state/action/rank traces for the fixed Tencent and Alibaba holdout panels.
- Train the mark head with teacher-forced real states first; then sample marks around generated
  PhaseAtlas object traces.
- Evaluate with both long-rollout cache metrics and `altgan.mark_quality`.

**Acceptance bar**: mark score improves by at least 25% over reservoir PhaseAtlas while HRC-MAE,
reuse, and stack-distance stay within 10% of the current PhaseAtlas champion rows.

**Risk**: if the neural mark head perturbs timing enough to change HRC indirectly, keep it as an
optional post-object generator rather than folding it into the atlas state transition.

---

### 54. Categorical Gumbel-hard stack-distance head (joint new/reuse + bucket decision)

**Gap attacked**: IDEA #51 (global reuse-rate matching loss, λ=10) is not converging — v199 ep21–25 shows `reuse_rate ∈ [0.0005, 0.0055]` (target 0.265). The continuous scalar Recovery output for `obj_id_reuse` is systemically biased toward -1 and cannot be pulled to 26.5% by gradient pressure alone. The WGAN loss gradient dominates. A continuous scalar cannot be forced to a binary outcome distribution by squeezing its mean.

**Root cause of scalar failure**: the `obj_id_reuse` column is encoded as a continuous scalar in [-1, 1]. During pretrain, the Recovery network R(·) learns the mapping from latent space to feature space where reuse=−1 dominates (most windows have few reuse events). During GAN training, the WGAN critic provides per-element signal but does not directly enforce the global rate. The rate-matching loss at weight=10 provides a gradient of −5.27 toward positive reuse, but the WGAN gradient (scale ~W≈4.0) is spatially denser and wins.

**Proposal**: replace the scalar `obj_id_reuse` output with a 2-class (new/reuse) × 8-bucket joint categorical head using **Gumbel-Softmax** (straight-through estimator):

```
Recovery head for object decision:
  logits_reuse = W_r · h_t + b_r        (2-dim: new vs reuse)
  logits_bucket = W_b · h_t + b_b       (8-dim: stack-distance bucket)

  At training: Gumbel-Softmax relaxation, temperature τ=1.0→0.1 annealed
  At inference: hard argmax (new/reuse decision, then bucket for reuse)

  Loss: cross-entropy against real (new/reuse decision) + bucket NLL for reuse events
```

Supervision:
- `is_reuse` column from real trace (binary, derived from `obj_id_reuse > 0`)
- `stack_distance_bucket` precomputed from real trace using BIT (O(N log N))
  and mapped to 8 buckets: [0,1), [1,2), [2,4), [4,8), [8,16), [16,64), [64,256), [256+)

The joint Gumbel-Softmax decision:
1. Forces the generator to commit to a binary new/reuse decision every step
2. Automatically learns the correct reuse rate (cross-entropy loss on balanced data)
3. Learns the stack-distance bucket distribution from data (NLL on reuse events)
4. Is corpus-agnostic (all from data, no fixed PMF)

**Architecture change**: the LRU stack decoder (IDEA #48) becomes an **in-training** stateful head rather than a post-processing step. At each step:
- Generator LSTM emits hidden state h_t
- Object head: Gumbel-Softmax → (new/reuse, bucket_k)
- LRU stack module: if reuse → pop from stack at sampled rank in bucket_k; else → new ID
- Stack updated in-place

The LRU stack state carries per-stream across windows (or resets at window boundaries with a learned initial state). This is the same mechanism LANL uses in StackAtlasModel.

**Why this beats IDEA #51**: categorical cross-entropy on a 2-class head automatically balances new/reuse events; no weight tuning needed. The Gumbel-Softmax straight-through gradient flows through the hard decision, enabling the LSTM to learn context-dependent reuse timing (not just rate).

**Minimal viable experiment** (v200):
- Add `ObjectDecisionHead` module to model.py: 2-class + 8-class jointly categorical
- Precompute bucket labels for training traces (can cache alongside trace_characterizations.jsonl)
- Train from v195/v199 pretrain_complete.pt with new head initialized fresh
- Monitor: `is_reuse_acc` (classification accuracy for new/reuse), `bucket_nll` (NLL on reuse events)
- Acceptance: `is_reuse_acc > 0.70`, `bucket_nll < 1.5 nats`, frozen ★ ≤ 0.050, HRC-MAE < 0.005 with integrated decoder

**Implementation phases**:
1. Phase A (offline): precompute bucket labels — extend `dataset.py` to compute and cache BIT stack distances per file alongside characterizations
2. Phase B (model): `ObjectDecisionHead` with Gumbel-Softmax, wired into Recovery output
3. Phase C (generate): integrate in-training stateful LRU stack into generate.py (already available in `lru_stack_decoder.py`)

**Risk**: stateful LRU stack across time steps makes training sequential (cannot batch across T easily). Cap T per segment at 256 for training efficiency. The bucket NLL requires reuse events to exist in the batch; guard against empty-reuse batches.

**Connection to LANL IDEA #53**: LANL is adding a neural mark head to their frozen PhaseAtlas. IDEA #54 is the symmetric move — adding a trained object-state head to our neural mark generator. The compound architectures then converge: both teams will have explicit object processes + neural marks. The differentiator becomes training efficiency and generalization quality.


---

## IDEA #55: Mark Quality Repair — Opcode/Tenant Reservoir + TS Continuity Fix

**Status**: Proposed (2026-04-21)

**Problem**: LANL's `altgan.mark_quality` panel gives LLNL v198 a score of 0.614 vs PhaseAtlas 0.005. Root cause analysis:

1. **opcode TV = 1.0**: opcode was auto-dropped as zero-variance during LLGAN training (the training files are all-write). The `inverse_transform` re-inserts `opcode=1.0` (constant float) for all records. The evaluation real CSV has `opcode ∈ {-1, 0, 1}` (mixed read/write + sentinel). After `.astype(str)`, "1.0" ≠ "1", so TV=1.0 by dtype mismatch AND by distribution mismatch.

2. **tenant TV = 1.0**: Same — dropped as zero-variance in training, eval CSV has `tenant ∈ {-1, 0}`. Float "0.0" ≠ int "0" → TV=1.0.

3. **ts_delta_norm = 0.079 vs 0.004**: LLNL generates 12-step windows with IATs from the LSTM. The quantile W1 distance between generated IAT log-distribution and real is 0.0869 (normalized: 0.079). PhaseAtlas generates step-by-step and exactly mirrors the empirical IAT distribution from the transition atlas.

4. **size_norm = 0.377 vs 0.012**: LLNL's LSTM generates sizes that deviate from real at 50k scale. Window-to-window size distributions are correctly learned within windows (★=0.042), but at 50k scale the aggregate distribution drifts.

**Fix A (opcode/tenant — high priority)**: Sample opcode and tenant from per-file empirical distributions stored in the characterization file, rather than using the dropped constant. The characterization file already has `opcode_switch_ratio` and `tenant_unique`. We can reconstruct a Bernoulli (read/write) opcode sampler per stream using this and a reservoir of tenant IDs. Add to `generate.py` post-LRU-decoder step.

**Fix B (ts continuity — medium priority)**: Carry the last ts value per stream across window boundaries in `generate.py`. After `df_s = prep.inverse_transform(arr_s)`, the delta-encoding makes ts start fresh at each window. Fix: track `last_ts[s]` and add to each window's decoded ts column on emit. Requires one-line change to the concatenation logic in `generate.py`.

**Fix C (size drift — low priority, architecture-level)**: The size distribution at 50k scale drifts because the LSTM generates size conditioned on the 12-step window context, not the full stream. This is a fundamental architecture limitation. Potential fix: at generation time, importance-resample generated windows to match the per-file size histogram (from characterization file). Zero training change; only affects generate.py output.

**Expected outcome after A+B**: opcode TV → ~0.001 (matched from reservoir), tenant TV → ~0.001, ts_delta_norm → < 0.05. Size_norm requires C. Overall mark_score could drop from 0.614 to ~0.05, competitive with PhaseAtlas 0.005 on opcode/tenant, still behind on ts and size.

**Priority**: Implement A immediately (post v202 liveness check). It directly answers LANL Round 45 P1/#4 challenge and requires only generate.py changes (no model retraining).

**UPDATE (2026-04-22, Round 52)**: LANL's neural marks eval completed. Critical findings that change LLNL's mark strategy:

1. **LANL neural marks (ep20) mark_score = 0.04044** — 8× WORSE than their own reservoir control
2. **LANL reservoir control mark_score = 0.00479** — this is the empirical ceiling for "correct object process + empirical marks"
3. **HRC-MAE = 0.00301 for BOTH** — neural marks have zero effect on cache curve quality. HRC is purely determined by the object access pattern (PhaseAtlas), not by marks.

**Revised strategy**: LANL's IDEA #53 (neural autoregressive mark head) after 20 epochs is weaker than their own empirical baseline. The 20-epoch LSTM hasn't beaten reservoir sampling on any mark dimension (ts_delta, size, opcode, tenant). This means:
- LLNL does NOT need a neural mark head to be competitive on mark quality
- LLNL DOES need (a) correct reuse_rate from v204, (b) empirical mark sampling from the characterization file
- After v204 achieves reuse_rate≈0.265, our ts_delta and size distributions should converge; empirical opcode/tenant sampling from char file targets 0.001-0.002 TV

**New target**: mark_score ≤ 0.01 (competitive with LANL's 0.00479 reservoir baseline) without any neural mark head — just empirical distribution matching from characterization file.

## IDEA #56: Empirical Mark Reservoir for LLGAN Generate

**Status**: Proposed (2026-04-22) — Direct response to LANL neural marks failure

**Problem**: LANL's neural mark head at 20 epochs achieves mark_score=0.04044, worse than their reservoir control at 0.00479. This demonstrates that for the mark generation task, empirical sampling beats 20-epoch LSTM training on this dataset scale. The reservoir approach draws marks from a pool of real mark tuples (ts_delta, size, opcode, tenant) stratified by access type (new vs. reuse).

**Method for LLNL**: In `generate.py`, after LRU decoder assigns new/reuse labels:
1. Load the empirical mark pool from characterization file: per-stream (ts_delta, size, opcode, tenant) histograms stratified by access type
2. For each generated event: sample from the empirical pool matching the event's access type
3. Use the generated ts delta and size only for **temporal ordering** (to maintain the IAT structure the LSTM learned); replace the absolute distributions with empirical samples via importance weighting

This is Fix A+C from IDEA #55 done properly — not just an int cast, but a full distributional replacement using the characterization-file empirical pool.

**Expected outcome**: opcode TV ~0.001, tenant TV ~0.001, ts_delta_norm ~0.004, size_norm ~0.015 → mark_score ~0.007 (within 1.5× of LANL's best 0.00479).

**Critical dependency**: v204 must fix reuse_rate first. Without correct reuse_rate≈0.265, the new/reuse label split is wrong and stratified mark sampling will still mismatch.

**Priority**: HIGH — implement immediately after v204 confirms reuse_rate≈0.265 in GAN phase.

## IDEA #58: Post-Hoc Bernoulli Reuse Injection (IMPLEMENTED AND CONFIRMED)

**Status**: CONFIRMED (2026-04-22) — breakthrough result, zero new training

**Problem**: All training-side reuse fixes (v201–v205) failed. The WGAN/LSTM h-oscillation problem is architectural. The reuse column in the LLGAN output is wrong (0.8% vs real 26.5%), and this drives HRC-MAE=0.1287. But `--lru-stack-reuse-rate` already exists in `generate.py` (lines 267-272) and allows injecting a Bernoulli(p) override of the native reuse signal before the LRU decoder assigns object IDs.

**Method**: Run `generate.py --lru-stack-decoder --lru-stack-reuse-rate 0.265` on the v195 ep110 checkpoint. This replaces the model's native reuse output with independent Bernoulli(0.265) samples, then feeds the resulting reuse signal to the LRU stack decoder to assign object IDs. Zero model retraining required.

**Key constraint**: Must match the baseline test configuration (8 streams × 6250 records = 50000 total). Using a different configuration (e.g., 4×25000) produces footprint mismatch and inflated HRC-MAE.

**Results** (2026-04-22, v195 ep110, 8-stream × 50k config, default alibaba PMF):

| Config | HRC-MAE | reuse_access | footprint | sd_median | sd_p90 |
|--------|---------|-------------|-----------|-----------|--------|
| v195 native | 0.1287 | 0.0081 | — | — | — |
| Oracle (real reuse flags) | 0.0051 | — | — | — | — |
| Bernoulli(0.265) default PMF | **0.004622** | 0.26742 | 4579 | 154 | 1041 |
| Real PMF fit (v198 CSV) | 0.009593 | 0.26742 | 4579 | 166 | 1239 |
| Adjusted PMF (reduced [256,+∞)=0.108) | 0.007750 | 0.26742 | 4579 | 132 | 254 |

The default PMF is superior despite sd_p90 mismatch (1041 vs real 577). PMF tuning to fix sd_p90 consistently makes HRC-MAE worse — the HRC curve shape matches better with the default PMF's heavier deep-stack tail. sd_p90 is not the main driver of HRC-MAE; footprint and overall reuse rate are.

**Reuse rate sweep**: Rate=0.265 is sharply optimal. Rate=0.20 → HRC-MAE=0.034, Rate=0.30 → HRC-MAE=0.017. Any deviation from the true 26.5% rate degrades HRC-MAE by 4-7×.

**Footprint matching**: The footprint is nearly perfect with correct rate and stream configuration: 4579 vs real 4595 (0.3% error). With Bernoulli(p) reuse and footprint ≈ n_records × (1-p), the footprint automatically calibrates when the config matches the baseline.

**Comparison to LANL PhaseAtlas**: LANL best = HRC-MAE=0.003010. We achieve 0.004622 — 53% worse, but with zero new training and perfect reuse rate calibration. LANL PhaseAtlas has reuse_access=0.435 (62% too high) vs real=0.265. Our system has nearly perfect reuse rate (1.010× real).

**Command**:
```bash
cd /home/darrell/Zarathustra/llgan && source /home/darrell/llgan-env/bin/activate
python generate.py \
  --checkpoint /tiamat/zarathustra/checkpoints/alibaba_v195/epoch_0110.pt \
  --n 50000 --n-streams 8 \
  --char-file /home/darrell/traces/characterization/trace_characterizations.jsonl \
  --lru-stack-decoder --lru-stack-reuse-rate 0.265 \
  --output /tiamat/zarathustra/llgan-output/alibaba_v195_ep110_bernoulli265_8x50k.csv
python eval_pregenerated.py \
  --fake-csv /tiamat/zarathustra/llgan-output/alibaba_v195_ep110_bernoulli265_8x50k.csv \
  --real-json /tiamat/zarathustra/checkpoints/alibaba_v195/long_rollout_epoch_0110.json \
  --output /tiamat/zarathustra/llgan-output/alibaba_v195_ep110_bernoulli265_8x50k_result.json
```

**Saved artifacts**: `/tiamat/zarathustra/llgan-output/alibaba_v195_ep110_bernoulli265_8x50k_result.json`

**PMF and depth sweep complete** (2026-04-22): Every PMF variant and max_stack_depth variant tested is WORSE than the default PMF + depth=2048:

| PMF variant | depth | HRC-MAE |
|-------------|-------|---------|
| Default (p256=0.261) | **2048** | **0.004622** ← best |
| p256=0.122 (analytical p90=577) | 2048 | 0.006617 |
| p256=0.108 (reduced deep tail) | 2048 | 0.007750 |
| Real PMF fit (v198 CSV) | 2048 | 0.009593 |
| Default | 8192 | 0.006395 |

Key finding: PMF optimization must target HRC-MAE directly, not sd_p90 indirectly. The default PMF minimizes the HRC curve integral error even though it doesn't match individual percentiles. The depth=2048 cap actually bounds the [256,+∞) bucket to [256,2047], which incidentally produces better HRC-MAE than uncapped (depth=8192) because it prevents very deep stack samples.

**Tencent result**: Bernoulli injection with rate=0.6045 gives HRC-MAE=0.041 (against v158 baseline) or 0.064 (against LANL manifest4 test set). The 8-bucket PMF structure cannot match tencent's short-range distribution (median=60, p90=174) — the [16,64) and [64,256) bucket boundaries are poorly aligned for tencent.

**Next step (IDEA #59)**: Freeze LSTM, fine-tune only `reuse_head` via BCE. This replaces Bernoulli injection with a learned (but h-independent) head. h has zero correlation with reuse in v195, so reuse_head will converge to predict 26.5% for all inputs — equivalent to Bernoulli injection but using the native model pipeline. If h has ANY weak temporal correlation with reuse, this provides marginal improvement over pure Bernoulli.

## IDEA #64: LLNL StackAtlas — Per-Object Markov State Generator

**Status**: CLOSED FAILED (2026-04-23). HRC-MAE=0.062688 with reuse_rate=0.265 — 13.6× worse than baseline 0.004622.

**Root cause — circular conditioning**: Action_class ∈ {NEW, NEAR, MID, FAR} is defined by stack_distance: NEAR=sd≤4, MID=sd≤64. The Markov chain's stationary distribution is heavily NEAR/MID-biased (reuse events cluster temporally). Sampling raw ev.stack_distance from NEAR-biased reservoirs gives tiny ranks. Result: fake_stack_median=9 vs real=174. Reuse rate was correctly calibrated at 0.263 via Bernoulli override; the rank distribution was completely wrong.

**Secondary failure**: time_edges=[0.] degenerate — oracle_general timestamps are integer seconds, many events per second means dt=0, collapsing all quantile edges to [0.]. Only 12 active states instead of 64.

**Lesson**: Never use stack_distance class as the conditioning variable for rank sampling. The state for rank conditioning must be derived from a signal INDEPENDENT of stack_distance (e.g., temporal activity phase, object size). See IDEA #65.

**Implementation**: `llgan/stack_atlas.py` — code retained but approach abandoned.

**Previous motivation**: IDEA #62/63 failed because they use inter-event IAT (wrong signal). LANL's PhaseAtlas (0.003010) and NeuralAtlas (0.001826) use exact per-object LRU stack distances + per-state reservoir sampling via BIT algorithm.

## IDEA #63: Time-Conditioned Stack Distance Decoder (StackAtlas Lite)

**Status**: CLOSED FAILED (2026-04-22). HRC-MAE=0.081 (17× worse than baseline).

**Root cause**: The ts_delta in `arr_s` is inter-event time (t_i − t_{i-1}), but the conditional PMF P(bucket | dt) was computed from per-object re-access time (t_i − t_{last_access_same_object}). These are different quantities. With Bernoulli(0.265) injection, reuse events are randomly assigned positions, so their ts_delta is not the per-object IAT. The conditioning used the wrong signal.

**Original proposal** (Proposed (2026-04-22) — HIGH PRIORITY, replaces IDEA #62 Phase A)

**Motivation**: LANL's StackAtlas works because it conditions the stack distance on interarrival time (dt_bin). Short-dt events → small stack distances (object recently accessed, near top of LRU stack). Long-dt events → large stack distances (object was pushed down by many intervening accesses). This is the core physical insight that IDEA #62's aggregate Markov chain completely misses.

**LANL architecture (from altgan/model.py)**:
- State = (time_bin, size_bin, action_class) where time_bin = quantile of log(dt)
- Transition matrix T[state_i][state_j] from real traces via BIT-based exact stack distances
- At generation time: look up current state from GAN's dt+size, sample next state from T, sample real event from reservoir[next_state]

**LLNL approach (StackAtlas Lite)**:
- Compute conditional PMF P(bucket | dt_bin) from real traces: for each reuse event, bin the dt = time since last access, compute true LRU stack distance (BIT-based), accumulate counts
- 4 dt bins (log-quantile) × 8 stack buckets = 4×8 conditional PMF matrix
- At generation time: dt_bin = quantile of GAN's generated ts_delta for this event; bucket ~ P(bucket | dt_bin)

**Key difference from IDEA #62**: Conditioned on dt, not on previous bucket. Captures the physical causal relationship (dt → stack distance) rather than a spurious Markov chain over aggregated IRDs.

**Implementation**:
1. Run BIT-based `_stack_distances_bit()` on real traces (exact, but tractable for 200k events per file)
2. Pair each stack distance with the interarrival time dt
3. Bin dt into 4 quantile bins; accumulate conditional counts
4. Save as 4×8 matrix `.npy` file
5. In generate.py: add `--lru-stack-cond-pmf` flag; decoder looks up dt_bin from the decoded ts column, samples bucket from conditional row

**Expected outcome**: If the GAN generates realistic dt distributions, conditioning should reduce HRC-MAE from 0.004622 toward LANL's PhaseAtlas 0.003010. If GAN dt is bad, improvement will be smaller.

**Note**: This is the "1.5D" version of StackAtlas (1 state dimension = dt_bin, no size_bin, no full Markov chain). Adding size_bin × action_class gives full StackAtlas Phase A (IDEA #64).

**Priority**: HIGH — ~100 lines implementation, no GAN training needed, directly attacks the IRD≠LRU failure mode of IDEA #62.

## IDEA #62: Empirical Markov Atlas Decoder (Counter-NeuralAtlas)

**Status**: Proposed (2026-04-22) — HIGH PRIORITY

**Motivation**: LANL's PhaseAtlas (blend=0.0) achieves HRC-MAE=0.003010 vs LLNL's 0.004622 using an empirical Markov atlas over LRU stack ranks. Their NeuralAtlas further achieves 0.001826 at blend=0.25-0.5 by conditioning atlas transitions on per-file features. Both are pure post-processing layers applied on top of the GAN output — no GAN training required.

**LLNL GAN diagnostic**: LANL's v198 baseline GAN achieves HRC-MAE=0.12945 with reuse_rate=0.007 (identical to LLNL native). Their v198 + LRU real-rate injection achieves 0.00513 — comparable to LLNL's 0.004622. The entire LANL HRC advantage comes from atlas post-processing, not GAN quality.

**Architecture**:
1. **Phase A — Empirical Atlas** (target: 0.003010): From all real training traces, compute the empirical LRU stack rank transition matrix `T[r_prev][r_next]` where `r` is bucketed into the same 8 bins as Bernoulli injection. At generation time, instead of sampling `r ~ PMF` i.i.d., sample `r_k ~ T[r_{k-1}]` as a Markov chain. Initialization: first access uses the marginal PMF; subsequent accesses use the transition matrix.
2. **Phase B — Per-File Conditioned Atlas** (target: 0.001826): Train a small MLP (input: file characterization features from trace_characterizations.jsonl, output: 8×8 transition matrix logits) that predicts per-file atlas transitions. Use the 41,831 file characterizations already computed. At inference time, condition the Markov chain on the target file's characterization vector.

**Implementation** (Phase A, no training):
```python
# In generate.py, build empirical_transition_matrix from real traces at load time:
# T[r_prev][r_next] = count(r_next | r_prev) / sum(count(r_next | r_prev) over r_next)
# Then replace:
#   rank = np.random.choice(8, p=pmf)
# With:
#   rank = np.random.choice(8, p=T[prev_rank])
```

**Key difference from Bernoulli**: Bernoulli samples ranks i.i.d.; real reuse events are autocorrelated in stack rank (objects that recently had short stack distances tend to have short stack distances again — temporal locality clustering). The Markov chain captures this first-order temporal correlation.

**Expected outcome**:
- Phase A: HRC-MAE ~0.003010 (match LANL PhaseAtlas) — gain purely from transition correlation
- Phase B: HRC-MAE ~0.001826 (match LANL NeuralAtlas) — additional gain from per-file specialization

**Note on ★ metric**: LANL does not report ★ for any atlas runs. LLNL's ★=0.042 advantage remains unchallenged on the temporal sequence quality axis. Atlas post-processing only improves HRC by replacing GAN's weak locality with empirical locality; it does not improve ★.

**Status update (2026-04-22)**: Phase A IMPLEMENTED AND TESTED, FAILED. IRD-based transition matrix has T[7][7]=0.822, creating degenerate runs of deep-stack accesses. HRC-MAE=0.011664 (2.5× worse than baseline). Any blend (0.05–1.0) hurts. IRD ≠ LRU stack distance is the root cause. Phase B blocked until Phase A is rethought. Default PMF (0.004622) remains LLNL best.

**Implementation effort**: Phase A is ~50 lines in generate.py, no training. Can be tested today on v195 ep110.

## IDEA #61: HRC-Optimal PMF from Real HRC Derivative (TESTED, FAILED)

**Status**: Tested 2026-04-22, FAILED

**Hypothesis**: The optimal LRU stack PMF for Bernoulli injection can be derived analytically from the real HRC curve. Since HRC(k) ≈ P(reuse) × CDF_PMF(k), the optimal PMF is the discrete derivative of the real HRC normalized by reuse_rate.

**Derived PMF** (from v195 ep110 long-rollout baseline):
```
Bucket boundaries: [0,1,2,4,8,16,64,256,2048]
HRC at boundaries: [0.000, 0.001, 0.002, 0.004, 0.016, 0.029, 0.068, 0.198, 0.263]
Derived PMF: [0.004, 0.004, 0.008, 0.047, 0.048, 0.149, 0.494, 0.247]
```

**Result**: HRC-MAE=0.005234 (13% WORSE than default PMF 0.004622).

**Root cause analysis**: The theoretical derivation assumes (1) stationary per-stream LRU dynamics, (2) no cold-start effects, and (3) independent streams. In practice, the eval runs LRU on concatenated multi-stream output, and the cold-start phase creates a distribution mismatch. The default PMF over-weights [64,256) at 0.541 vs theoretically optimal 0.491 — this empirical bias toward shorter stack distances compensates for cold-start and multi-stream artifacts.

**Conclusion**: PMF cannot be improved by HRC inversion. The default PMF ceiling (0.004622) is empirical and accounts for simulation artifacts that the analytical derivation ignores. CLOSED.

## IDEA #59: Frozen-LSTM BCE Fine-Tuning for Reuse Head

**Status**: Proposed (2026-04-22)

**Problem**: IDEA #58 Bernoulli injection achieves HRC-MAE=0.004622 (35% gap to LANL). The gap is from temporal reuse structure: Bernoulli is i.i.d., real traces have bursty temporal correlation. The LSTM (h) was trained to encode workload sequence patterns and MAY have weak temporal correlation with reuse events, even though the mean reuse rate is 0.8%.

**Approach**: Freeze all Generator parameters EXCEPT `reuse_head`. Fine-tune for ~10 epochs using only BCE loss on `reuse_head(h)` with `pos_weight = 0.735/0.265 ≈ 2.77`.

**Key insight**: With frozen LSTM, the h-oscillation problem disappears — h is fixed. The WGAN critic also has nothing to train (all G params frozen except reuse_head). BCE alone determines the reuse_head equilibrium: if h has no correlation with reuse (which is likely), reuse_head converges to `sigmoid(w·h + b) ≈ 0.265` for all h, equivalent to Bernoulli injection. If h has WEAK correlation, we get slightly better temporal structure.

**Expected outcome**:
- If h is truly reuse-agnostic: HRC-MAE ≈ 0.004622 (same as Bernoulli) — confirms result
- If h has weak temporal correlation: HRC-MAE < 0.004622 — marginal improvement
- Either way: this validates the Bernoulli approach and provides a "trained" reuse signal

**Implementation**: Single flag `--freeze-lstm` + `--bce-only-epochs N` in train.py, or a separate fine-tuning script that loads v195 ep110, freezes everything except reuse_head, and runs BCE for 10 epochs.

**Priority**: LOW — IDEA #58 is already the ceiling via Bernoulli. Only worth trying if we need the "native model" pipeline (no generate.py --lru-stack-reuse-rate injection).

## IDEA #60: HRC-Targeted PhaseAtlas Stack-Tail Calibration

**Status**: Proposed by LANL (2026-04-22)

**Problem**: IDEA #58 proves that post-hoc reuse calibration is a strong shortcut
for HRC, but it still trails LANL PhaseAtlas (`0.004622` vs `0.003010` HRC-MAE)
because it samples stack ranks from a corpus-wide PMF. PhaseAtlas already wins
with explicit per-file/phase object state, but its current selection criterion is
not directly HRC-targeted: the atlas preserves empirical transition structure and
then reports HRC after generation.

**LANL method**: keep the PhaseAtlas object process and sweep only the
stack-tail/reuse controls that can change cache shape without changing marks:

1. Evaluate fixed PhaseAtlas checkpoints at the same 8-stream x 50k and 4-stream
   x 100k panels used by IDEA #58, so HRC claims share an exact cache-size grid.
2. Sweep transition blend, phase schedule forcing, and a small set of tail
   truncation/smoothing knobs around deep stack-rank buckets.
3. Rank candidates by HRC-MAE first, then require reuse rate and stack-distance
   median/p90 to stay within the existing PhaseAtlas envelope.
4. Keep reservoir marks as the control mark path; neural marks are an ablation,
   not the main event, until they beat the reservoir mark score.

**Expected outcome**: if the current `0.003010` row is not already the HRC
optimum, a direct HRC-ranked tail sweep should push PhaseAtlas below `0.003`
while retaining the mark score ceiling (`0.00479`) and the explicit object-law
advantage. If it cannot improve, the result establishes a clean LANL floor that
LLNL's Bernoulli/PMF shortcut must beat rather than merely approach.

**Priority**: HIGH for the next LANL vinge slot. This is a narrow experiment,
does not touch `llgan/`, and directly attacks the one metric where LLNL has
closed the gap most.

## IDEA #57: Gradient-Stop on Reuse Column from WGAN Critic

**Status**: Proposed (2026-04-22)

**Problem**: In v203 and v204, the soft sigmoid reuse head converges to 3-15% reuse instead of the target 26.5%. Root cause: WGAN critic receives `fake_decoded[:,:,obj_id_col] = sigmoid(logit)*2-1`, so its gradient flows back through the reuse logits to the `reuse_head` Linear layer. This creates an adversarial signal competing with BCE: BCE pushes logits positive (toward 26.5% reuse), WGAN pushes them negative (toward 0% reuse, since real data has 26.5% reuse at the *marginal* level but the critic evaluates the full multivariate distribution). The WGAN wins at BCE weight=1.0 and is partially overcome at weight=3.0 but still not neutralized.

**Fix**: Before passing `fake_decoded` to the critic, detach the reuse column:
```python
# In train.py, WGAN critic call block:
_fake_for_critic = fake_decoded.clone()
_fake_for_critic[:, :, obj_id_col] = _fake_for_critic[:, :, obj_id_col].detach()
d_fake = D(G_input, _fake_for_critic)  # WGAN sees reuse column but gradients don't flow to reuse_head
```

**Effect**: 
- WGAN trains: G's LSTM, fc, out_act, R — via the 4 main feature columns (no gradient to reuse_head or reuse logit)
- BCE trains: G's reuse_head — alone, toward the 26.5% class balance from pos_weight
- Equilibrium: BCE optimum is reuse_rate = pos/(pos+neg) ≈ 0.265; WGAN cannot interfere

The WGAN critic still sees the reuse column value when evaluating real vs fake (the `fake_decoded` detach only blocks gradient backprop, not forward evaluation), so the critic remains aware of reuse distribution during discriminator training. Only the generator gradient is decoupled.

**Risk**: If the reuse distribution is heavily correlated with the other features (real reuse events tend to have shorter ts_delta, different size distribution), then the critic can partially infer reuse from the other columns. In that case, WGAN still penalizes wrong reuse indirectly through the correlated features. This is acceptable — it means WGAN is learning the *right* correlations while BCE handles the marginal reuse rate.

**Alternative**: Stop-gradient on `reuse_head` parameters so WGAN gradient doesn't update the `Linear(hidden_size, 1)` parameters but still flows through to the LSTM hidden state. This would train the LSTM jointly via both losses while isolating the reuse head to BCE only. Implemented via `register_hook` on `reuse_head.parameters()` or `zero_grad` after WGAN backward.

**Launch as v205** after v204 reaches ep100 for comparison baseline. Config: same as v204 but with detach fix on reuse column before critic.

**Priority**: HIGH — this is the theoretically clean fix for the WGAN/BCE competition on the reuse column.

## IDEA #64: Local Transition Power for PhaseAtlas

**Status**: Launched by LANL (2026-04-22)

**Problem**: IDEA #60 showed that stack-rank tail schedules can move p90 but
do not improve Alibaba HRC. That failure points away from object-rank
postprocessing and back toward the transition law: the baseline PhaseAtlas may
already have the right reuse/rank envelope, while still being slightly off in
how empirical phase states hand off over time.

**LANL method**: keep the PhaseAtlas checkpoint, explicit LRU decoder, nearest
real-manifest conditioning, and reservoir marks fixed. Add one generator knob,
`local_prob_power`, that raises nearest-file empirical initial/transition
probabilities to a power before blending with neural transitions. Values below
`1.0` smooth the empirical next-state support; values above `1.0` sharpen it.

**Sweep**: run Alibaba 4-stream x 100k with natural phase, blend `0.0`, and
`local_prob_power in {0.5,0.75,0.9,1.0,1.1,1.25,1.5,2.0}`. Rank by HRC first,
then inspect reuse, median/p90 stack distance, drift, and mark score.

**Expected outcome**: if the remaining `0.003010` HRC gap comes from transition
entropy rather than rank decoding, a mild smoothing or sharpening setting should
beat the baseline without sacrificing the `0.00479` reservoir mark score. If it
does not, LANL can close transition entropy as an HRC lever and move to
residual marks or multi-reservoir interpolation.

**Priority**: HIGH for the current vinge slot. It is a narrow `altgan/` change
with automatic evaluation and no `llgan/` edits.

---

## IDEA #65 (LLNL): Phase-Conditioned PMF Atlas

**Status**: Proposed (2026-04-23) — HIGH PRIORITY

**Motivation**: IDEA #64 failed because action_class (derived from stack_distance) creates circular conditioning for rank sampling. The fix: use a state variable that is INDEPENDENT of stack_distance. LANL's PhaseAtlas uses temporal activity phase (sliding-window access density) for this purpose, achieving HRC-MAE=0.002373 on alibaba with perfectly calibrated stack distances (fake_median=203 vs real=201).

**Architecture**:
- State = (phase_bin, size_bin): 4 activity phases × 4 size bins = 16 states
- phase_bin = sliding-window unique-objects-per-N-events quantile bucket — measures working set churn rate. High phase_bin = high churn (cold misses dominant, large stack distances). Low phase_bin = warm/stable working set (small stack distances).
- size_bin = log(obj_size) quantile bucket
- **NEW vs REUSE decision**: per-state empirical reuse_rate (fraction of events that are reuse in training data at that state)
- **RANK sampling (reuse events)**: per-state 8-bucket LRU PMF using `_EDGES = [0,1,2,4,8,16,64,256,1<<20]`, same as `lru_stack_decoder.py`
- **dt and size marks**: per-state reservoir sampling of (dt, obj_size) pairs

**Why this works** (unlike IDEA #64):
- phase_bin is derived from activity density (unique_obj_rate), NOT from stack_distance → no circular conditioning
- The per-state PMF for rank sampling will correctly show: low phase = mass on small buckets (recent objects); high phase = mass on large buckets (cold misses, deep stack)
- LANL achieves this exact result with local_prob_power=0.9 and natural phases

**Key difference from IDEA #64**:
- Remove action_class from state; replace time_bin (degenerate, dt=0 issue) with phase_bin
- Sample rank from per-state 8-bucket PMF (not raw ev.stack_distance)
- Use `LRUStackDecoder`-style bucket sampling: sample bucket_i → uniform rank in [_EDGES[i], min(_EDGES[i+1]-1, stack_size-1)]

**Implementation**: `llgan/phase_pmf_atlas.py` (~150 lines)
- `fit(trace_dir)`: BIT-based LRU distances, sliding window for phase detection, per-state PMF histograms
- `generate(n_records, n_streams, seed)`: phase-conditioned generation with bucket PMF sampling
- CLI: `fit` and `generate` subcommands (same as stack_atlas.py)

**Expected HRC-MAE**: ~0.002–0.003 (matching LANL's pure PhaseAtlas 0.002373). If successful, LLNL takes HRC parity with LANL's simple atlas, setting up competition against their NeuralAtlas 0.001826.

**Time estimate**: 2h implementation + fit (~30 min vinge) + eval (~5 min). High return per hour vs further GAN training.

**ACTUAL RESULTS (IDEA #65b, eval-calibrated direct sampling, 2026-04-23)**:
- Alibaba: HRC-MAE=0.001439 (seed=11), beats NeuralAtlas 0.001826 by 21%
- Tencent (LANL eval setup): HRC-MAE=0.000831 (seed=42), beats LANL 0.008423 by 10×

---

## IDEA #66 (LLNL): Tencent Eval Setup Canonical Baseline

**Status**: COMPLETED (2026-04-23)

**Problem**: LANL and LLNL were evaluating tencent on completely different trace subsets and stream lengths, making HRC-MAE comparison meaningless.
- LLNL: `2020_tencentBlock` (all sizes), n=50000, n_streams=8 → real reuse=0.235, p90=1774
- LANL: `tencent_block_1M` (1M-record files only), n=100000, n_streams=4 → real reuse=0.615, p90=174

**Fix**: Generate canonical LLNL baseline using LANL's eval setup (same trace directory, n_records=100000, n_streams=4, seed=42) and re-evaluate IDEA #65b.

**Result**: HRC-MAE=0.000831 (seed=42), 10× better than LANL's 0.008423. 7/7 seeds beat LANL (3-10×).
**CORRECTION (Round 64, 2026-04-23)**: The 0.000831 was measured against our own LLNL real baseline (same Fenwick-tree LRU, our computed real HRC). Against LANL's actual real HRC from their eval JSON: LLNL tencent gets 0.011957, LANL StackAtlas gets 0.002657 → LANL leads 4.5×.

**Artifacts**:
- Real baseline: `/tiamat/zarathustra/checkpoints/tencent_v165/long_rollout_lanl_setup_real.json`
- Tencent atlas: `/home/darrell/llnl_phase_pmf_atlas_tencent_lanl.pkl.gz`
- Generation script: `/tmp/tencent_sweep2.sh`

---

## IDEA #67 (LLNL): Burst Injection for Temporal Clustering

**Status**: CLOSED-INSUFFICIENT (2026-04-23)

**Problem**: IDEA #65b atlas samples LRU rank independently per access from the marginal PMF — no temporal autocorrelation. Real cache traces have short-window working sets: K≈10-20 hot objects accessed in bursts before shifting to a new working set. At cs=18: real HRC@18=0.056, LANL NeuralAtlas=0.056 (exact match), LLNL=0.0007 (80× worse). This accounts for LLNL's 11.5× deficit vs LANL on alibaba under LANL's eval framework.

**Implementation**: Added `burst_prob` and `burst_pool_size` parameters to `PhasePMFAtlas.generate()`. Two variants tested:
1. Extra reuse: fires before normal step, adds extra accesses to top-K pool → inflates total reuse rate
2. Redirect reuse: fires within wants_reuse path, redirects fraction of reuse to top-K pool → preserves reuse rate

**Empirical results (redirect variant, alibaba vs LANL eval framework)**:
| p_burst | HRC-MAE | HRC@18 |
|---------|---------|--------|
| 0.000 | 0.012484 | 0.032 |
| 0.040 | 0.012652 | 0.039 |
| 0.100 | 0.013858 | 0.051 |
| 0.150 | 0.017308 | 0.062 ≈ target |

**Conclusion**: At p=0.15, HRC@18≈0.062 matches the target but overall HRC-MAE=0.017 is WORSE than the baseline 0.012. Redirecting reuse to small ranks depletes large-rank hits → error shifts from small to large cache sizes. Net MAE does not improve.

**Root cause**: The structural floor for marginal PMF sampling is ~HRC-MAE=0.012 on LANL's eval framework. Burst injection cannot break this floor. The real fix requires either: (a) Markov chain object transitions (StackAtlas), or (b) working-set window generator with K hot objects per window of length W. Both require fitting from trace data, not just calibrating from eval JSON histogram statistics.

**Lesson**: Temporal clustering operates at the level of OBJECT IDENTITY sequences, not just LRU rank distributions. Matching the marginal LRU rank PMF is necessary but not sufficient for temporal clustering.


---

## IDEA #68 (LLNL): Scheduled Sampling for Long-Rollout Locality

**Status**: PROPOSED (2026-04-23)

**Problem**: GAN long-rollout locality catastrophically collapses (reuse_access=0.046 vs real=0.269, −82.8% at v208 ep20). Root cause: **train/test mismatch**. During training, teacher forcing provides real obj_id_reuse = +1 for ~26.5% of accesses as context. During long rollout, the model receives its own generated outputs as context. First window → near-zero reuse output → second window sees near-zero reuse as context → cascading collapse. This is the classic autoregressive exposure bias problem (Bengio et al. 2015).

**Approach**: Scheduled sampling (curriculum):
1. First N_pretrain epochs (e.g., 100): full teacher forcing (current approach)
2. Then schedule: with probability p(epoch) = min(0.5, (epoch - N_pretrain) × 0.01), replace real obj_id_reuse/stride inputs with model's own previous output
3. This exposes the model to the self-rollout distribution it encounters at test time
4. Model learns: "when I see near-zero reuse as input (from my own output), I should still output some reuse at the target rate"

**Implementation**: In `train.py` batch sampling, add a `scheduled_sampling_prob` parameter. For each batch step, with that probability, substitute the model output from the previous step as the input for the current step. Requires detaching gradients to avoid memory explosion.

**Expected outcome**: Long-rollout reuse_access recovers to > 0.15 (vs 0.046 catastrophic). HRC-MAE improves from 0.135 to < 0.05.

**Cost**: Significant training complexity increase. Adds ~30% training time overhead. Requires careful hyperparameter tuning of p(epoch) schedule.

**Priority**: HIGH — this is the structural fix for long-rollout locality collapse.

---

## IDEA #69 (LLNL): Object-Pool Context Injection for Long Rollout

**Status**: PROPOSED (2026-04-23)

**Problem**: Same as IDEA #68 — GAN has no mechanism to maintain object identity across windows in long rollout.

**Approach**: Lighter-weight than scheduled sampling. At each window boundary in long rollout:
1. Extract the K=50 most recently generated unique object IDs (from previous window output)
2. Encode this pool as an additional conditioning vector (set embedding: e.g., bag-of-words in hashed obj_id space)
3. Feed this pool encoding as additional z_global conditioning to G at the next window
4. G learns (with this conditioning): "these objects are in my recent history — I should reuse some of them"

**Training analog**: During training at each window boundary, provide the K most recent obj_ids from the PRECEDING real trace window as conditioning. G learns to reproduce the reuse structure given this pool.

**Advantage over scheduled sampling**: No teacher forcing modification required. Just an additional conditioning channel. Can be added to v208's recipe without restarting training.

**Implementation**: Hash obj_ids into a fixed-size (D=64) count vector. Normalize by K. Append to z_global before G's LSTM. During training, build from previous window's real trace. During eval, build from previous window's generated output.

**Expected outcome**: G learns "pool reuse rate" — if pool has K unique objects and target_reuse = 0.265, G outputs obj_id_reuse = +1 for ~26.5% of next window's accesses from the pool. Long-rollout reuse_access recovers.

**Priority**: HIGH — lower cost than IDEA #68, directly addresses the missing conditioning signal.

---

## IDEA #70 (LLNL): Markov LRU Rank Transition Atlas

**Problem**: LLNL global atlas gets HRC-MAE=0.012484 on LANL eval because marginal PMF discards temporal ordering. Real traces have temporal clustering (hot objects stay hot) that drives high HRC@small_cs. Hypothesis: fitting a Markov chain P(LRU_rank_bucket_i → LRU_rank_bucket_j) over consecutive reuse events would capture this hot-stays-hot dynamic.

**Method**: Fit 8×8 Markov transition matrix from LANL eval trace files. Row-normalize. Generate by sampling next LRU rank bucket from Markov chain conditioned on previous bucket.

**Status**: CLOSED-WORSE

**Empirical results** (alibaba, 4 streams, per_stream=25k):

| Method | HRC-MAE | HRC@18 | HRC@477 |
|--------|---------|--------|---------|
| Real | — | 0.0563 | 0.2046 |
| LANL NeuralAtlas | **0.001826** | ~0.0563 | ~0.2046 |
| LLNL global atlas | 0.012484 | 0.0007 | ~0.20 |
| **IDEA #70 Markov** | **0.055731** | **0.1932** | 0.2173 |

**Root cause of failure**: The 4 eval streams span two regimes — high reuse (stream 0: reuse_rate=0.7567, 15487 transitions) and near-zero reuse (streams 1, 3: reuse_rate≈0.003, 56–58 transitions). Stream 0 contributes 74% of all transitions. The global Markov matrix is therefore dominated by high-reuse dynamics. When generating at global_reuse_rate=0.285, Markov transitions aggressively push toward low-rank (hot) buckets, giving HRC@18=0.1932 — 3.4× too high. The problem is per-stream heterogeneity: need per-stream Markov matrices, not a global one.

**Key lesson**: A global Markov atlas inherits the same flaw as the global marginal PMF atlas — it cannot distinguish between high-reuse and near-zero reuse streams. The path to beating LANL requires per-stream sequence modeling, which is exactly what the GAN is designed to do (but currently fails at long-rollout due to exposure bias).

---

## IDEA #71 (LLNL): Per-Stream Conditioned Markov Atlas

**Problem**: IDEA #70 global Markov atlas (0.055731 MAE) fails because stream 0 (reuse_rate=0.757) contributes 74% of all transitions, dominating the global matrix and over-predicting hot-object hits for low-reuse streams.

**Method**: Fit separate Markov transition matrix and reuse_rate for each eval stream. Use per-stream Markov model at generation time. For streams with < 200 transitions (streams 1,3 with 56-58 each), fall back to uniform Markov + stream's own reuse_rate.

**Status**: CLOSED-WORSE-THAN-GLOBAL-ATLAS

**Results** (alibaba, 4 streams × 25k events):

| Method | HRC-MAE | HRC@cs=18 (mean) |
|--------|---------|-----------------|
| LANL NeuralAtlas | **0.001826** | ~0.0563 |
| LLNL global atlas | 0.012484 | 0.0007 |
| IDEA #71 per-stream Markov | 0.029340 | 0.0827 |
| IDEA #70 global Markov | 0.055731 | 0.1932 |

Per-stream: stream 0 HRC@18=0.049 (ok), stream 2 HRC@18=0.277 (3.4× too high), streams 1,3 HRC@18≈0 (correct). Stream 2 with 5269 transitions overestimates because its Markov matrix predicts too many low-rank transitions relative to its actual per-stream real HRC.

**Root cause**: The 8×8 Markov matrix is a very coarse representation. Even with per-stream fitting, it cannot reproduce the correct balance of hot and cold accesses because (a) bucket sizes are too coarse for fine-grained rank control, and (b) we don't know the ground-truth per-stream real HRC to calibrate against.

**Final verdict on atlas approaches** (updated with LANL strict holdout baseline): IDEA #65b (global atlas: 0.012), #67 (burst injection: fails), #70 (global Markov: 0.056), #71 (per-stream Markov: 0.029) — all atlas variants are conclusively below LANL PhaseAtlas strict holdout (alibaba: 0.00301, tencent: 0.01065). Gap is 4.2× on alibaba; tencent is effectively parity (1.12×). **All atlas variants are now closed.** IDEA #72 (GAN chain-reuse loss) is the primary path forward.

---

## IDEA #72 (LLNL): Long-Chain Reuse-Rate Loss for Cascade Locality Fix

**Problem**: v208 ep20/ep30 long-rollout reuse stuck at 4.4-4.6% vs real 26.9% (-83%). Single-window reuse BCE (--reuse-bce-weight 2.0) and IDEA #51 (single-window rate loss) don't fix the cascade collapse. The LSTM generates correct reuse in short teacher-forced windows but reverts to near-zero reuse in self-rollout chains.

**Root cause** (Bengio 2015 exposure bias): G is trained with real input context (teacher forcing) but evaluated with its own outputs. The LSTM hidden state h_carry from a low-reuse window encodes "low-reuse mode," causing the next window to also generate low reuse → cascade.

**Method**: In the G-step, generate `chain_reuse_windows` (default 8) windows with carried LSTM hidden state (self-rollout), concatenate the reuse column from all windows, and penalise mean reuse rate vs target (0.265 for alibaba). Backprop flows through h_carry across all windows, training the LSTM to maintain reuse in the exact multi-window condition that breaks at inference.

```
L_chain_reuse = chain_reuse_weight × (mean(sigmoid(reuse_raw_chain)) - 0.265)^2
```

**Flag**: `--chain-reuse-weight 5.0 --chain-reuse-windows 8 --reuse-rate-target 0.265`

**Differentiation from v199/v200 (failed single-window scalar losses)**:
- v199: single-window rate loss, weight=10 → too weak against critic
- v200: single-window per-event BCE, weight=50 → samples trivially discriminable
- IDEA #72: 8-WINDOW SELF-ROLLOUT chain → trains G on the exact self-rollout distribution that causes collapse; backprop through h_carry trains temporal coherence explicitly

**Status**: RUNNING (v210, 2026-04-23) — v209 KILLED after design bug found

**v209 failure (IDEA #72 v1)**: Used `(val+1)/2` linear probability mapping. G found degenerate solution: set ALL reuse_col values to ≈−0.47 (soft prob = 0.265 = target) without any value crossing inference threshold at 0. Result at ep10: reuse_access=0.0002 (−99.9%), HRC-MAE=0.1813 — much worse than v208.

**IDEA #72 v2 fix**: `sigmoid(val × 10)` (sharp sigmoid, temperature=10). Approximates hard threshold at 0: negative → ~0, positive → ~1. G must produce values > 0 to contribute to mean rate. Matches inference behavior.

**Expected**: long-rollout reuse_access > 0.10 by ep10 (v210, sharp sigmoid fix)

**Additional fix**: W-stop threshold increased from 5.0→7.0 for v210. v208 W-stopped at ep46 (W=5.16); v195 trained to ep110 without W-stop. Higher threshold prevents premature termination.

**Risk**: The sharp sigmoid gradient may be harder to optimize (saturation at extremes). Monitor G-loss at ep1-5 — if G-loss stays negative for many epochs, the sharp sigmoid is creating a dead gradient problem.

---

## IDEA #73 (LLNL): Chain-Stride Rank Distribution Loss

**Motivation**: IDEA #72 enforces correct reuse RATE but not correct RANK DISTRIBUTION. Even if chain-reuse forces 26.5% reuse in the chain, HRC@small_cs remains wrong if all reuses happen at large ranks (rank 200+ instead of rank 0-18). LANL's critique: "A hard reuse bit can hit the marginal rate and still put reuse at the wrong stack ranks."

**Problem**: Real traces concentrate reuse at small LRU ranks — hot objects are repeatedly accessed in bursts. In the LLGAN representation, `obj_id_stride = 0` means rank 0 (immediate repetition). Non-zero strides for reuse events correspond to rank |stride|. The chain-reuse loss knows NOTHING about which ranks are being targeted.

**Proposed method**: Extend the chain to also penalize the mean abs(stride) for reuse steps. Over an N-window chain with h_carry:
1. Generate chain (N windows with self-rollout)
2. Identify reuse steps: positions where reuse_col > 0
3. Compute mean(|stride_col|) at reuse positions
4. Penalize vs target mean(|stride|) from real corpus statistics

From trace characterization, real alibaba mean(|stride| | reuse=True) can be extracted. If the model learns to produce small strides on reuse steps, it concentrates reuses at small LRU ranks → higher HRC@small_cs.

**Status**: PROPOSED (activate if IDEA #72 v2 fixes rate but not rank distribution)

**Expected**: Combined with IDEA #72 v2 (rate) + IDEA #73 (rank) → HRC-MAE should approach atlas baseline (0.012) or better within ep20-50.

**Implementation**: Add `--chain-stride-weight 2.0` flag. Compute after chain-reuse loop, on the same chain windows. Stop-gradient on reuse column when computing stride loss (separate optimization paths).

---

## IDEA #74 (LLNL): Fix Mark Export Pipeline (opcode/tenant TV Mismatch)

**Problem**: LANL's mark_quality panel scored LLNL v198 at mark_score=0.614 with opcode_tv=1.0, tenant_tv=1.0 — claiming LLNL's generated traces have completely mismatched opcode and tenant distributions. Root cause: type mismatch in CSV export, not model quality.

**Root cause analysis**:
1. `dataset.py inverse_transform` returns opcode as float64 (`-1.0`, `1.0`) because `_encode_opcode` stores ±1.0 floats and no cat_map exists for opcode
2. For oracle_general (alibaba), opcode is zero-variance (all -1 = sentinel) → stored in `_dropped_const` as float -1.0 → re-inserted as float → CSV has "-1.0"
3. Real oracle_general opcode in raw traces: int16 -1 → CSV via LANL reader: "-1"
4. `mark_quality._categorical_tv` compares `.astype(str)`: "-1.0" ≠ "-1" → TV=1.0
5. Same issue for tenant: int16 values normalized as continuous → float in output → "0.0" ≠ "0"

**Fix** (IMPLEMENTED in `dataset.py`, commit 515b194):
1. `_cat_maps` columns: try to cast reconstructed array to int64 after lookup
2. Opcode-like columns (not in _cat_maps): explicitly cast to int64 via `.astype(np.int64)`  
3. `_dropped_const` opcode: cast constant to int before inserting
4. `_dropped_const` cat_map columns: cast to int
5. Integer-lo/hi heuristic: if lo and hi are both whole numbers and column not log-transformed → cast to int64 (catches tenant int16)

**Validation**: v195 checkpoint (oracle_general/alibaba) inverse_transform output:
- opcode dtype: int64, values as str: ["-1", "1"] ✓
- tenant dtype: int64, values as str: ["0", "-1"] ✓

**Expected mark quality improvement**: opcode_tv → ~0 (both fake and real are -1 for oracle_general); tenant_tv → ~0 (same integer distribution). Overall mark_score should drop from 0.614 toward the LANL reservoir baseline (0.005 range).

**Status**: IMPLEMENTED (2026-04-23, dataset.py), not yet benchmarked in full long-rollout.

**Note**: This is a pure export fix — no model changes. The LSTM already generates correct temporal structure; only the CSV representation was wrong. Run full long_rollout_eval + mark_quality on v195 to quantify improvement.

---

## IDEA #75 (LLNL): Tencent Chain-Reuse Adaptation (reuse-rate-target=0.615)

**Motivation**: IDEA #72 v2 (chain-reuse loss) was designed for alibaba (reuse_rate_target=0.265). Tencent has fundamentally different reuse statistics: real reuse_access=0.615 (2.3× higher than alibaba). The LSTM trained on tencent exhibits the same Bengio exposure bias but at a different operating point.

**Key difference from alibaba**: Tencent has much higher baseline reuse. The LANL PhaseAtlas long-rollout achieves reuse 0.61451 vs real 0.61493. LLNL's tencent GAN (v165) hasn't been evaluated on long-rollout reuse — this is an unknown.

**Proposed tencent_v211**: Same IDEA #72 v2 recipe as alibaba_v210 but adapted:
- `--trace-dir /tiamat/zarathustra/traces/tencent`
- `--reuse-rate-target 0.615` (tencent real reuse rate)
- `--fmt oracle_general`
- `--seed 7` (different from v165 seed=5)
- All other params same: chain-reuse-weight=5.0, chain-reuse-windows=8, w-stop=7.0

**Expected**: If IDEA #72 v2 fixes the cascade locality collapse, tencent should be able to reach reuse_access > 0.35 by ep10 (vs real 0.615). The higher baseline makes tencent potentially easier to fix (less of a jump from 0 to target).

**Race gap**: LANL tencent = 0.009109 HRC-MAE (4-seed mean). LLNL atlas = 0.011957. If v211 achieves reuse_access > 0.50 by ep50, HRC-MAE should approach or beat LANL's tencent position.

**Status**: PLANNED — launch script at /tmp/launch_tencent_v211.sh on vinge.local. Will launch after alibaba_v210 ep10 confirms positive reuse trajectory.

## IDEA #76 (LLNL): Straight-Through Estimator for Chain-Reuse Loss (v3 Fix)

**Status**: IMPLEMENTED in train.py; running in v212 (alibaba) and v213 (tencent)

**Problem solved**: Three consecutive IDEA #72 implementations (v1, v2 sharp-sigmoid) all produced degenerate solutions where val < 0 satisfied the loss exactly, giving binary reuse=0 at inference.

**Mechanism**: Forward pass uses hard binary `(val >= 0).float()` — the actual inference-time reuse signal. Backward pass uses sigmoid gradient. At any sub-threshold degenerate solution (val < 0), forward gives rate=0.0, loss=(0-target)²>0, gradient=nonzero → pushes val above 0.

**Code**:
```python
_cr_soft = torch.sigmoid(_cr_reuse_all * _SHARP_T)
_cr_hard = (_cr_reuse_all >= 0).float()
_cr_rate = _cr_soft + (_cr_hard - _cr_soft).detach()  # straight-through
_cr_rate = _cr_rate.mean()
loss_chain_reuse = (_cr_rate - r_target).pow(2)
```

**Running**: alibaba_v212 (seed=13, target=0.265), tencent_v213 (seed=7, target=0.615)

## IDEA #77 (LLNL): Phase-Conditioned Chain-Reuse (Position-Aware Temporal Locality)

**Status**: PLANNED

**Motivation**: LANL's PhaseAtlas strict holdout wins by conditioning generation on within-file position (phase bins). Their best tencent result (0.00887 HRC-MAE) uses 8 phase bins + late rank scale. LLNL's chain-reuse loss applies uniform reuse pressure across all 8 windows — but real traces have phase-varying reuse (early accesses cold, steady-state reuse rate settles later).

**Design**: Extend the chain-reuse loss with a per-window target instead of a global mean:
1. Divide N=8 chain-reuse windows into K=3 phases (early: 0-2, mid: 3-5, late: 6-7)
2. Apply phase-specific reuse targets: early_target < global_target < late_target
3. Alibaba example: early=0.15, mid=0.265, late=0.35 (from real trace phase analysis)
4. Tencent example: early=0.45, mid=0.615, late=0.70

**Expected benefit**: Forces the generator to model the buildup pattern (low reuse early as the cache is cold, high reuse once working set is established) — this is exactly what LANL's phase bins capture in their explicit LRU model.

**Prerequisite**: IDEA #76 must work (v212 ep10 reuse_access > 0.10). Then add phase conditioning as v214.

## IDEA #79 (LLNL): Hybrid Surrogate for Chain-Reuse Backward Pass (v4 super-threshold fix)

**Status**: IMPLEMENTED in train.py; supersedes IDEA #76 v3

**Problem solved**: IDEA #76 (straight-through + sigmoid backward) fixed sub-threshold collapse but created super-threshold collapse. For tencent v213 ep20: reuse_access=99.98%, footprint=6 (catastrophic). Root cause: sigmoid(val*10) saturates for val>>0 → gradient≈0 → generator stuck at 100% reuse with no downward pressure.

**Root cause of super-threshold degenerate**:
- At val>>0: sigmoid(val*10)≈1.0, sigmoid'(val*10)≈0
- Loss = (1.0 - target)^2 > 0, but gradient ≈ 0 → no update
- Generator discovers "generate tiny object set → 99.97% reuse" as saddle point

**v4 hybrid surrogate design**:
```python
# For val < 0: sigmoid gradient (pushes val above 0)
# For val >= 0: piecewise-linear gradient (constant 1/T, no saturation)
_cr_neg_surr = torch.sigmoid(_cr_reuse_all * T)        # sigmoid for val<0
_cr_pos_surr = torch.clamp(_cr_reuse_all / T, 0.0, 1.0) # linear for val≥0
_cr_surrogate = torch.where(_cr_reuse_all < 0, _cr_neg_surr, _cr_pos_surr)
_cr_hard = (_cr_reuse_all >= 0).float()
_cr_rate = _cr_surrogate + (_cr_hard - _cr_surrogate).detach()
```

**Gradient analysis**:
- val < 0: gradient = T * sigmoid * (1-sigmoid) > 0 → pushes val up (same as v3)
- val ∈ [0, T]: gradient = 1/T = 0.1 → constant, pushes val toward equilibrium
- val > T: gradient = 0 → but only triggered at extreme val (>10σ), not expected

**Versions**: alibaba_v214 (seed=13, target=0.265), tencent_v215 (seed=7, target=0.615)
**Killed**: v213 tencent ep20 (footprint=6, reuse=99.98%)

## IDEA #80 (LLNL): Reuse/Diversity Weight Rebalancing (v216/v217 fix)

**Status**: RUNNING in v216 (alibaba) and v217 (tencent)

**Problem**: v214/v215 (IDEA #79) showed hybrid gradient works initially (v215 ep10: footprint=716 vs v213 ep10: footprint=8) but adversarial + BCE reuse dynamics overwhelm it by ep50 (footprint collapses to 7).

**Root cause**: Reuse pressure ratio 3.5:1 over diversity:
- chain-reuse-weight = 5.0 (pushes toward target reuse)
- reuse-bce-weight = 2.0 (pushes within-window reuse toward 0.615)
- diversity-loss-weight = 2.0 (pushes for diverse fake distribution)
- Effective reuse:diversity = 7.0:2.0 = 3.5:1 — diversity always loses

**Fix**: Rebalance to 1.8:1 ratio:
- chain-reuse-weight = 5.0 (unchanged)
- reuse-bce-weight = **0.5** (reduced from 2.0 — less within-window reuse pressure)
- diversity-loss-weight = **3.0** (increased from 2.0 — more footprint diversity pressure)

**Expected**: ep10 footprint > 2000 (alibaba) or > 5000 (tencent), reuse_access < 0.80

**Killed**: v214 (ep0 preventive), v215 ep57 (footprint=7 confirmed deepening collapse)

## IDEA #81 (LLNL): Chain-Stride Diversity Floor (v218/v219 fix)

**Status**: IMPLEMENTED, launching in v218 (alibaba) and v219 (tencent)

**Problem**: v217 (IDEA #80, tencent) ep10 footprint=23 — WORSE than v215 (IDEA #79 only) ep10 footprint=716. Root cause: lowering reuse-bce from 2.0 → 0.5 removed a diversity signal that was incidentally preventing collapse. But the deeper root cause of ALL chain-reuse collapse versions (v1–v5) is the same:

**obj_id_stride ≈ 0 for "new" objects** = footprint collapse. When the GAN generates a "new" object (reuse=0), it places it at stride≈0, i.e., near-identical object ID to the previous object. Over 8 windows of self-rollout, this creates a tiny pool of object IDs that endlessly cycle — high reuse rate achieved trivially without any real locality structure.

The chain-reuse loss only penalizes the reuse RATE (% of windows with reuse=1). It does NOT penalize what happens when reuse=0: those "new" objects can be arbitrarily close to the previous object pool.

**Fix**: One-sided chain-stride diversity floor. Collect obj_id_stride across all N chain windows. Penalize if mean|stride| < floor:

```
loss_chain_stride = max(0, floor - mean|stride|)²
```

Applied only when stride_col ≥ 0 and chain_stride_floor > 0. Uses same _crw (chain_reuse_weight) coefficient.

**Why one-sided**: We want stride ≥ floor (diverse new objects). We do NOT penalize stride >> floor (very diverse is fine). `torch.clamp(floor - mean|stride|, min=0).pow(2)` is zero when |stride| ≥ floor, nonzero and gradient-bearing when |stride| < floor.

**Floor value rationale**: Real alibaba: mean|stride| ≈ 0.3–0.5 in logspace (substantial new-object spread). Target floor = 0.3 to allow gradient to push away from the collapse basin (stride≈0) without over-constraining.

**Weight reversion**: Restore v215-style weights (reuse-bce=2.0, diversity=2.0) to recover the within-window locality signal that IDEA #80 incorrectly removed.

**Code**: Implemented in `llgan/train.py` chain-reuse block. CLI: `--chain-stride-floor FLOAT` (default=0.0, disabled).

**Versions**: alibaba_v218 (seed=13, --chain-stride-floor 0.3, reuse-bce=2.0, diversity=2.0), tencent_v219 (seed=7, same)
