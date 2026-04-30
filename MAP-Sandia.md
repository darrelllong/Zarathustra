# Sandia / newgan Cognitive Map

This file provides a cognitive map of the Sandia codebase in the newgan/ directory.
It documents the architecture, key classes, functions, and their relationships.

---

## Overview

Sandia's entry to the Zarathustra race uses a **4-phase curriculum learning** approach:

```
Phase 1: AE Pretraining      (Encoder + Recovery, MSE loss)
   |
Phase 2: Supervisor Pretrain (Teacher-forcing prediction)
   |
Phase 2.5: G Warm-up         (Imitate supervisor without Critic gradients)
   |
Phase 3: Joint GAN           (WGAN-SN + supervisor consistency loss)
```

**Competitive strategy:**
1. **Pretrain quality selection** - Build multiple pretrains, rank by downstream quality (AE loss)
2. **Cross-seed validation** - Any claim must survive 3+ seeds (42, 11, 7) before promotion
3. **Long-horizon focus** - HRC-MAE as primary checkpoint selector

---

## File Structure

```
newgan/
├── train.py          # Self-contained training script (SandiaTrainer, PretrainRanker)
├── v1_baseline.sh    # Launch script using train.py
├── generate.py       # Generation wrapper
├── run.py            # Generic runner for checkpoint generation
└── config.py         # Configuration (if needed)
```

---

## Key Classes

### PretrainRanker

**Purpose:** Rank pretrains by their downstream quality potential

**Key Methods:**
- `evaluate_pretrain(encoder, recovery, val_dataset, device)` → float
  - Evaluates pretrain quality on held-out validation
  - Returns MSE reconstruction loss (lower = better)
- `rank_pretrains()` → List[Tuple[str, float]]
  - Returns sorted list of (name, score) pairs

**Usage:** Build 3-5 pretrains, rank them, only launch Phase 3 from top pretrains

---

### SandiaTrainer

**Purpose:** Main training loop implementing the 4-phase curriculum

**Attributes:**
- `cfg` - Config object with hyperparameters
- `device` - Torch device (cuda/mps/cpu)
- `seed` - Random seed for reproducibility
- `G, C, E, R, S` - Generator, Critic, Encoder, Recovery, Supervisor models
- `best_combined` - Best validation score
- `stale_epochs` - Counter for early stopping

**Methods:**

| Method | Purpose | Key Parameters |
|--------|---------|----------------|
| `init_models(num_cols)` | Initialize all GAN models | num_cols from preprocessor |
| `pretrain_ae(train_ds, val_ds, epochs, ckpt_dir)` | Phase 1: AE training | epochs=50 |
| `pretrain_supervisor(train_ds, val_ds, epochs, ckpt_dir)` | Phase 2: Supervisor training | epochs=50 |
| `train_generator(train_ds, val_ds, epochs, ckpt_dir)` | Phase 2.5: G warmup | epochs=100 |
| `train_gan(train_ds, val_ds, epochs, ckpt_dir)` | Phase 3: Joint GAN | epochs=200 |
| `run_full_pipeline(train_ds, val_ds, ckpt_dir)` | Run complete pipeline | - |
| `_quick_val(val_tensor)` | Quick validation score | - |

**Training Loop Details:**

**Phase 1 (AE):**
- Encoder + Recovery round-trip
- MSE loss on reconstruction
- Optimizer: Adam(lr=cfg.lr_g, betas=(0.5, 0.9))

**Phase 2 (Supervisor):**
- Teacher-forcing prediction
- MSE loss on latent sequence prediction
- Encoder/Recovery frozen

**Phase 2.5 (G Warmup):**
- Generator imitates supervisor predictions
- No Critic gradients (pure MSE)
- z_global + z_local noise input

**Phase 3 (GAN):**
- WGAN-SN loss (Wasserstein with Spectral Normalization)
- Critic updates: n_critic per generator update
- Generator loss: -C(G(z)) + 0.5 * sup_loss (supervisor consistency)
-Checkpoint every cfg.checkpoint_every epochs
- Early stop after cfg.early_stop_patience stale epochs

---

## Key Functions

### Data Loading

| Function | Purpose | Returns |
|----------|---------|---------|
| `_collect_files(trace_dir, fmt)` | Find all trace files | List[Path] |
| `_load_raw_df(path, fmt, max_records)` | Load single file to DataFrame | pd.DataFrame |
| `_fit_prep_on_files(files, fmt, records_per_file, ...)` | Fit preprocessor | TracePreprocessor |
| `_load_epoch_dataset(files, fmt, records_per_file, prep, timestep, char_lookup)` | Load dataset | Tuple[ConcatDataset, Optional[np.ndarray]] |
| `load_data(trace_dir, fmt, char_file, files_per_epoch, records_per_file, timestep, val_ratio)` | Full data pipeline | Tuple[ConcatDataset, Tensor, TracePreprocessor] |

### Configuration

**Parser Flags (train.py):**
- `--trace-dir` (required): Trace directory path
- `--fmt`: Trace format (default: oracle_general)
- `--char-file`: Characterization file path
- `--epochs`: GAN epochs (default: 200)
- `--batch-size`: Batch size (default: 64)
- `--lr-g`, `--lr-d`: Generator/Critic learning rate
- `--noise-dim`, `--hidden-size`, `--latent-dim`: Model dimensions
- `--timestep`: Window length (default: 12)
- `--pretrain-ae-epochs`: AE pretrain epochs (default: 50)
- `--pretrain-sup-epochs`: Supervisor epochs (default: 50)
- `--pretrain-g-epochs`: G warmup epochs (default: 100)
- `--n-critic`: Critic updates per G update (default: 5)
- `--checkpoint-every`: Checkpoint frequency (default: 5)
- `--early-stop-patience`: Patience for early stop (default: 30)
- `--seed`: Random seed (default: 42)
- `--exp-name` (required): Experiment name
- `--checkpoint-dir`: Checkpoint directory (default: checkpoints)
- `--files-per-epoch`: Files per epoch (default: 12)
- `--records-per-file`: Records per file (default: 20000)
- `--val-ratio`: Validation ratio (default: 0.1)
- `--no-compile`, `--no-amp`: Training options

**Compatibility flags** (accepted but not yet implemented):
- `--loss`, `--cond-drop-prob`, `--var-cond`, `--var-cond-kl-weight`
- `--pcf-loss-weight`, `--pcf-n-freqs`
- `--retrieval-memory`, `--retrieval-*`
- `--mixed-type-recovery`, `--w-stop-threshold`

---

## Model Initialization

**Generator (llgan.model.Generator):**
```python
G = Generator(
    noise_dim=cfg.noise_dim,
    num_cols=num_cols,
    hidden_size=cfg.hidden_size,
    latent_dim=cfg.latent_dim,
    cond_dim=cfg.cond_dim if hasattr(cfg, 'cond_dim') else 0,
    timestep=cfg.timestep
)
```

**Critic (llgan.model.Critic):**
```python
C = Critic(
    num_cols=num_cols,
    hidden_size=cfg.hidden_size,
    use_spectral_norm=True,
    sn_lstm=True,
    minibatch_std=True,
    cond_dim=cfg.cond_dim,
    num_lstm_layers=1
)
```

**Encoder/Recovery/Supervisor:** Similar signatures with latent_dim, hidden_size, num_cols

---

## Output Artifacts

**Per-epoch checkpoints:**
- `epoch_{N:04d}.pt` - Full state dict (every N epochs)

**Best checkpoints:**
- `ae_pretrain_best.pt` - Best AE pretrain
- `supervisor_best.pt` - Best supervisor
- `g_warmup_best.pt` - Best G warmup
- `best.pt` - Best GAN checkpoint (by combined score)

**Config:**
- `config.json` - Experiment metadata (exp_name, seed, timestamp, args)

---

## Workflow

```
1. Load data
   ├── Collect files from trace_dir
   ├── Fit preprocessor on 4 seed files
   ├── Split train/val (90/10)
   ├── Subsample files_per_epoch if specified
   └── Return ConcatDataset for train, Tensor for val

2. Initialize models
   ├── Generator (G)
   ├── Critic (C)
   ├── Encoder (E)
   ├── Recovery (R)
   └── Supervisor (S)

3. Run training pipeline
   ├── Phase 1: Pretrain AE (E + R)
   ├── Phase 2: Pretrain Supervisor (S)
   ├── Phase 2.5: Warmup Generator (G)
   └── Phase 3: Joint GAN (G + C)

4. Output checkpoints and history
```

---

## Integration with llgan

**Dependencies:**
- `llgan.config.Config` - Configuration object
- `llgan.dataset` - Data loading (TraceDataset, TracePreprocessor, _READERS)
- `llgan.model` - Core models (Generator, Critic, Encoder, Recovery, Supervisor)

**Import strategy:**
```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llgan.config import Config
from llgan.dataset import TraceDataset, TracePreprocessor, _READERS
from llgan.model import Generator, Critic, Encoder, Recovery, Supervisor
```

---

## Current Status

**Fixes applied (v1.0.3):**
- Added `tempfile` import
- Fixed validation collation (ConcatDataset for train, tensor for val)
- Fixed `files_per_epoch` usage for data subsampling
- Added missing parser flags for v1_baseline.sh compatibility
- Fixed `run.py` Generator initialization

**Next steps:**
- Launch s003_tencent with fixed code
- Verify checkpoint emission
- Run cross-seed validation (seeds 42, 11, 7)
- Rank pretrains and select best for Phase 3

---

## Version Log

See `VERSIONS-Sandia.md` for detailed version history.

- **v1.0.0:** Self-contained training pipeline
- **v1.0.1:** Initial release
- **v1.0.2:** v1 baseline attempt & race entry
- **v1.0.3:** Fix Round 4/5 blockers (this version)
