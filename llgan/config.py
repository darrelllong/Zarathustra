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
    loss: str = "wgan-gp"       # "wgan-gp" or "bce"
    batch_size: int = 64
    epochs: int = 200
    g_rounds: int = 3           # used only for bce mode
    n_critic: int = 5           # critic steps per generator step (wgan-gp)
    lr_g: float = 0.0001        # WGAN-GP uses lower, equal LRs
    lr_d: float = 0.0001
    device: str = "cuda"        # falls back to mps/cpu if unavailable

    # Data — single file
    trace_path: str = ""
    trace_format: str = "spc"   # spc | msr | k5cloud | systor | oracle_general | csv
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
    fft_loss_weight: float = 0.05     # L_FFT: frequency-domain matching (0 = off)

    # Evaluation
    mmd_every: int = 5
    mmd_samples: int = 1000
