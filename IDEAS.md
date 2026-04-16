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
98. **Running:** alibaba_v116 (continuity-loss seed #3 — third data point for v114 ATB recipe; v114=0.073★/0.176frozen, v115=0.083★/0.195frozen)
99. **Running:** tencent_v143 (multi-scale+PCF ATB recipe, seed #5 — fifth data point for v136's 0.178 frozen ATB)
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
