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
    checkpoint_every: int = 10
    generated_path: str = "generated.csv"
    num_generate: int = 1_000_000

    # Auxiliary generator losses
    moment_loss_weight: float = 0.1   # L_V: per-feature mean+std matching (0 = off)
    quantile_loss_weight: float = 0.2  # L_Q: per-feature quantile matching at p50/90/95/99 (0 = off)
    fft_loss_weight: float = 0.05     # L_FFT: frequency-domain matching (0 = off)
    acf_loss_weight: float = 0.1      # L_ACF: lag-1..5 autocorrelation matching (0 = off)
    fide_alpha: float = 1.0           # FIDE: frequency inflation weight (NeurIPS 2024)
    feature_matching_weight: float = 1.0  # L_FM: critic feature matching (0 = off)
    compile: bool = False              # torch.compile models for ~20-40% CUDA speedup (CUDA only)
    minibatch_std: bool = True         # append per-batch std channel to critic input (StyleGAN2)
    locality_loss_weight: float = 0.0  # L_loc: object reuse rate matching within windows (0 = off)
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
