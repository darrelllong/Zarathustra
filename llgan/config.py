"""Default hyperparameters for LLGAN I/O workload generator."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Model
    noise_dim: int = 10
    hidden_size: int = 256
    timestep: int = 12          # window length; paper optimal: 10–15

    # Training
    loss: str = "wgan-sn"       # "wgan-sn": Wasserstein + spectral norm (output layer only)
                                # "wgan-gp": Wasserstein + gradient penalty (true Lipschitz;
                                #   works on MPS — create_graph=True confirmed supported)
                                # "bce": original GAN cross-entropy
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

    # Output
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None   # specific checkpoint path; None = auto-detect latest
    reset_optimizer: bool = False       # load weights only, fresh optimizers at new LR
    sn_lstm: bool = True                # spectral norm on critic LSTM weight matrices
    patch_embed: bool = False           # Conv1d patch embedding before critic LSTM (TTS-GAN)
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
    amp: bool = False                  # AMP fp16 forward passes for 2-3× CUDA speedup (CUDA only; incompatible with wgan-gp/r1/r2)
    compile: bool = False              # torch.compile models for ~20-40% CUDA speedup (CUDA only)
    minibatch_std: bool = True         # append per-batch std channel to critic input (StyleGAN2)
    locality_loss_weight: float = 0.0  # L_loc: stride-repetition rate matching within windows (0 = off).
                                       # Measures fraction of positions whose obj_id DELTA matches a prior
                                       # delta in the window — captures sequential-access strides, not raw
                                       # object identity (delta encoding precludes direct ID comparison).
    diversity_loss_weight: float = 0.0  # L_div: MSGAN mode-seeking loss — maximises |G(z1)-G(z2)|/|z1-z2|
                                        # across random noise pairs; directly combats β-recall mode collapse.
                                        # Requires a second G forward pass per step. Try 0.5–2.0.
    r1_lambda: float = 0.0            # R1: zero-centered GP on real samples (R3GAN)
    r2_lambda: float = 0.0            # R2: zero-centered GP on fake samples (R3GAN)

    # Latent autoencoder + supervisor (TimeGAN/SeriesGAN architecture)
    # Set latent_dim > 0 to enable; 0 = legacy direct-to-feature mode (v4/v5).
    latent_dim: int = 24              # latent space dimensionality
    pretrain_ae_epochs: int = 50      # Phase 1: pretrain encoder + recovery
    pretrain_sup_epochs: int = 50     # Phase 2: pretrain supervisor on real latents
    pretrain_g_epochs: int = 100      # Phase 2.5: generator warm-up via supervisor
    supervisor_loss_weight: float = 10.0  # η: supervisor term in joint generator loss
    supervisor_steps: int = 1         # 1 = 1-step supervisor; 2 = 2-step (SeriesGAN)
    lr_er: float = 0.0005             # learning rate for encoder + recovery (joint phase)

    # Evaluation
    mmd_every: int = 5
    mmd_samples: int = 1000
    early_stop_patience: int = 0    # evals without improvement before stopping (0 = off)
