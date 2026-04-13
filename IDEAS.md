# Ideas

## Priority: Bold Architectural Changes

Parameter tuning is unlikely to produce the breakthrough needed. The ideas below are ordered by expected impact. The Bayesian approaches at the top represent fundamental structural changes to how the model represents and covers the workload distribution — not incremental loss tweaks.

---

### 0. Fourier analysis

Have we done a Fourier transform on the time series in the trace to look for dominant frequencies? Should R do that, or some other tool?

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
18. **Running:** alibaba_v85 (v71 seed #8, combined=0.111 ep25 ★); tencent_v115 (v105 seed roll #2)
19. **Next:** Continue seed rolling; then try self-diag (#12) or BayesGAN (#9)
