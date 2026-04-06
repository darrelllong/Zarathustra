"""Default hyperparameters for LLGAN I/O workload generator."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Model
    noise_dim: int = 10
    cond_dim: int = 0               # per-window workload descriptor dim (0 = unconditional)
    cond_drop_prob: float = 0.0     # classifier-free guidance: drop conditioning with this prob
    hidden_size: int = 256
    timestep: int = 12          # window length; paper optimal: 10–15

    # Training
    loss: str = "wgan-sn"       # "wgan-sn": Wasserstein + spectral norm (output layer only)
                                # "wgan-gp": Wasserstein + gradient penalty (true Lipschitz;
                                #   CUDA only — create_graph=True through LSTM not supported on MPS)
                                # "bce": original GAN cross-entropy
                                # "rpgan": relativistic paired GAN (R3GAN, NeurIPS 2024);
                                #   MPS-compatible; improves mode coverage / β-recall
    gp_lambda: float = 10.0     # gradient penalty coefficient (WGAN-GP; standard value)
    batch_size: int = 64
    epochs: int = 200
    g_rounds: int = 3           # used only for bce mode
    n_critic: int = 3           # critic steps per generator step
    lr_g: float = 0.0001
    lr_d: float = 0.00005       # slower critic: two-timescale GDA (JMLR 2025)
    lr_cosine_decay: float = 0.05  # cosine anneal eta_min = lr * this (0 = constant LR)
    grad_clip: float = 1.0      # gradient norm clip for G and C (0 = off)
    ema_decay: float = 0.999    # EMA decay for generator weights (0 = off / use live G)
    device: str = "cuda"        # falls back to mps/cpu if unavailable

    # Data — single file
    trace_path: str = ""
    trace_format: str = "spc"   # spc | msr | k5cloud | systor | oracle_general | lcs | csv
    max_records: int = 60_000   # max records in single-file mode
    train_split: float = 0.8

    # Data — multi-file streaming
    trace_dir: str = ""          # directory of trace files (mutually exclusive with trace_path)
    files_per_epoch: int = 8     # files sampled per epoch
    records_per_file: int = 15_000  # records loaded from each file per epoch
    char_file: str = ""          # path to trace_characterizations.jsonl for precharacterized
                                 # per-file conditioning vectors; when set, replaces noisy
                                 # window-level z_global descriptors with stable full-trace stats

    # Output
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None   # specific checkpoint path; None = auto-detect latest
    reset_optimizer: bool = False       # load weights only, fresh optimizers at new LR
    sn_lstm: bool = True                # spectral norm on critic LSTM weight matrices
    patch_embed: bool = False           # Conv1d patch embedding before critic LSTM (TTS-GAN)
    film_cond: bool = False             # FiLM conditioning in G (NeurIPS 2018): applies
                                        # (1+γ(z_global))*h + β(z_global) after LSTM at each
                                        # timestep, re-injecting workload cond; requires cond_dim>0;
                                        # zero-init → backward compatible with existing checkpoints.
    proj_critic: bool = False           # projection discriminator (Miyato & Koyama, ICLR 2018);
                                        # adds inner(cond_proj(cond), pooled) to critic score;
                                        # requires cond_dim > 0 and --char-file for full benefit
    pack_size: int = 1                  # PacGAN packing (Lin et al. NeurIPS 2018): score packs of
                                        # m windows jointly; critic detects low-diversity packs
                                        # (mode collapse signature). 1=off, 2=pairs (recommended)
    checkpoint_every: int = 10
    generated_path: str = "generated.csv"
    num_generate: int = 1_000_000

    # Auxiliary generator losses
    moment_loss_weight: float = 0.1   # L_V: per-feature mean+std matching (0 = off)
    quantile_loss_weight: float = 0.2  # L_Q: per-feature quantile matching at p50/90/95/99 (0 = off)
    fft_loss_weight: float = 0.05     # L_FFT: frequency-domain matching (0 = off)
    acf_loss_weight: float = 0.1      # L_ACF: lag-1..5 per-feature autocorrelation matching (0 = off)
    cross_cov_loss_weight: float = 0.0  # L_cov: lag-1 cross-feature covariance matrix matching (0 = off).
                                         # Matches the full d×d matrix E[x_{t,i}·x_{t+1,j}] — the linear
                                         # dynamics operator that DMD-GEN estimates.  ACF only matches the
                                         # diagonal; this adds the off-diagonal cross-feature terms.
                                         # Directly targets DMD-GEN > 0.3.  Try 0.5–2.0.
    fide_alpha: float = 1.0           # FIDE: frequency inflation weight (NeurIPS 2024)
    feature_matching_weight: float = 1.0  # L_FM: critic feature matching (0 = off)
    amp: bool = True                   # AMP fp16 forward passes for 2-3× CUDA speedup (CUDA only; incompatible with wgan-gp/r1/r2)
    compile: bool = True               # torch.compile models for ~20-40% CUDA speedup (CUDA only)
    minibatch_std: bool = True         # append per-batch std channel to critic input (StyleGAN2)
    locality_loss_weight: float = 0.0  # L_loc: stride-repetition rate matching within windows (0 = off).
                                       # Measures fraction of positions whose obj_id DELTA matches a prior
                                       # delta in the window — captures sequential-access strides, not raw
                                       # object identity (delta encoding precludes direct ID comparison).
    diversity_loss_weight: float = 0.0  # L_div: MSGAN mode-seeking loss — maximises |G(z1)-G(z2)|/|z1-z2|
                                        # across random noise pairs; directly combats β-recall mode collapse.
                                        # Requires a second G forward pass per step. Try 0.5–2.0.
    continuity_loss_weight: float = 0.0  # L_cont: boundary-continuity loss for multi-window coherence.
                                          # Generates two adjacent windows with carried LSTM hidden state
                                          # and penalises mean/std mismatch at the boundary (last k ↔ first k
                                          # steps).  Directly targets DMD-GEN by training G on the same
                                          # carry-state generation task used at inference.  Try 0.5–1.0.
    r1_lambda: float = 0.0            # R1: zero-centered GP on real samples (R3GAN)
    r2_lambda: float = 0.0            # R2: zero-centered GP on fake samples (R3GAN)

    # v15 data representation
    obj_size_granularity: int = 4096  # snap obj_size to this byte multiple before encoding
                                      # (0 = off; 4096 = page-aligned; matches real block traces
                                      # where sizes are multiples of 4KB)

    # Variational conditioning (IDEAS.md idea #3)
    var_cond: bool = False               # Enable variational encoder on char-file conditioning.
                                         # Replaces fixed cond vector with N(μ(cond), σ(cond)):
                                         # samples at train time, uses μ at eval.  Makes G robust
                                         # to conditioning noise; closes EMA→full-eval gap.
                                         # Requires cond_dim > 0 and char_file.
    var_cond_kl_weight: float = 0.001    # Weight for KL(q(cond|stats) || N(0,I)) in G loss.
                                         # Small (0.001) lets σ grow slowly; try 0.01 for
                                         # stronger regularization toward unit Gaussian.

    # Regime-first two-stage generation (IDEAS.md idea #5)
    n_regimes: int = 0               # Number of workload regime prototypes (0 = off).
                                      # When > 0 and cond_dim > 0, replaces raw cond passthrough
                                      # with a Gumbel-Softmax regime sampler: K learnable regime
                                      # prototypes in cond space, hard one-hot selection per window.
                                      # G commits to one workload type before generating each window.
                                      # Compatible with v28 pretrain (only G's cond path changes).
                                      # Temperature annealed τ: 1.0 → 0.1 over training.
                                      # Recommended: 8 (same as GMM K for consistency).

    # Multi-resolution critic (IDEAS.md idea #8, HiFi-GAN style)
    multi_scale_critic: bool = False     # Replace single Critic with 3-scale MultiScaleCritic.
                                          # Each scale sees T, T//2, T//4 timesteps respectively.
                                          # G loss = mean over scales. Critic is NOT pretrained so
                                          # this is compatible with v28 pretrain (no new pretrain).
                                          # Targets DMD-GEN and AutoCorr by forcing critic to
                                          # discriminate at burst (fine), envelope (medium), and
                                          # regime (coarse) temporal scales simultaneously.

    # Mixed-type output heads in Recovery (IDEAS.md idea #7, CTGAN-style)
    mixed_type_recovery: bool = False    # Enable type-aware Recovery output heads: binary columns
                                         # (opcode, obj_id_reuse) use sigmoid→scaled-to-[-1,1] instead
                                         # of Tanh; Phase 1 reconstruction uses BCE for these columns.
                                         # Requires a new pretrain (incompatible with existing ckpts
                                         # that use the single-fc Recovery architecture).
                                         # Expected impact: lower MMD² by eliminating soft intermediate
                                         # values for binary fields during evaluation.

    # BayesGAN: posterior over discriminators (Saatci & Wilson, NeurIPS 2017)
    bayes_critics: int = 0               # Number of critic particles (0 = standard single critic).
                                         # Each particle is updated with SGLD (Adam + Gaussian noise
                                         # injection), approximating a posterior over discriminators.
                                         # G loss is averaged across all particles, preventing any
                                         # single boundary from forcing mode collapse.
                                         # Recommended: 5 (memory: M × ~1MB; GB10 has 124GB).
                                         # Requires wgan-sn. Incompatible with wgan-gp/r1/r2.

    # Deeper LSTM (IDEAS.md idea #11)
    num_lstm_layers: int = 1             # Number of LSTM layers in Generator and Critic (default 1).
                                         # 2–3 layers enable hierarchical temporal representations:
                                         # layer 1 tracks per-step dynamics, layer 2+ captures
                                         # multi-scale regime structure (burst envelopes, changepoints).
                                         # Requires fresh pretrain (architecture change).

    # GMM prior on generator noise
    gmm_components: int = 0              # K mixture components in noise prior (0 = flat N(0,I)).
                                         # When > 0 and cond_dim > 0, replaces N(0,I) with a
                                         # conditioning-aware mixture: each workload type gets its
                                         # own region of noise space.  Directly targets recall
                                         # ceiling from unimodal prior fighting multimodal data.
                                         # Recommended: 8 (try 4–16).

    # Latent autoencoder + supervisor (TimeGAN/SeriesGAN architecture)
    # Set latent_dim > 0 to enable; 0 = legacy direct-to-feature mode (v4/v5).
    latent_dim: int = 24              # latent space dimensionality
    pretrain_ae_epochs: int = 50      # Phase 1: pretrain encoder + recovery
    pretrain_sup_epochs: int = 50     # Phase 2: pretrain supervisor on real latents
    pretrain_g_epochs: int = 100      # Phase 2.5: generator warm-up via supervisor
    supervisor_loss_weight: float = 10.0  # η: supervisor term in joint generator loss
    supervisor_steps: int = 1         # 1 = 1-step supervisor; 2 = 2-step (SeriesGAN)
    lr_er: float = 0.0005             # learning rate for encoder + recovery (joint phase)

    # Dual discriminator (latent-space C + feature-space C_feat)
    feat_critic_weight: float = 0.0   # weight for feature-space critic in G loss (0 = off).
                                      # Requires latent_dim > 0.  C_feat operates on decoded
                                      # features so it penalises quality problems invisible in
                                      # latent space.  Try 0.5–1.0 alongside normal C.

    # AVATAR (Adversarial Autoencoder + Autoregressive Refinement)
    avatar: bool = False              # Enable AVATAR mode (AAE + autoregressive refinement)
    dist_loss_weight: float = 1.0     # Distribution loss weight (mean+std matching to N(0,1))
    latent_disc_weight: float = 1.0   # Latent discriminator adversarial loss weight

    # Evaluation
    mmd_every: int = 5
    mmd_samples: int = 1000
    early_stop_patience: int = 0    # evals without improvement before stopping (0 = off)
    w_stop_threshold: float = 0.0   # W-distance spike guard: stop if W > this for 3 consecutive
                                    # epochs (0 = off). Catches conditioning instability collapse
                                    # (v33: W→10.6 ep179, v36: W→10 ep178). Recommended: 3.0.
    dmd_ckpt_weight: float = 0.0    # weight for DMD-GEN in combined checkpoint score (0 = off).
                                    # combined = MMD² + 0.2*(1-recall) + dmd_ckpt_weight*DMD-GEN.
                                    # Use 0.05 to add a temporal-dynamics tiebreaker. Adds ~5s/eval
                                    # (5 mini-batches of DMD at batch_size=32).
