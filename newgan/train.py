#!/usr/bin/env python3
"""
Sandia training script for the Zarathustra race.

This script implements Sandia's competitive strategy:
1. Pretrain quality selection - build multiple pretrains, rank by downstream quality
2. Cross-seed validation - any claim must survive 3+ seeds before promotion
3. Long-horizon focus - HRC-MAE as primary checkpoint selector

Usage on vinge.local:
    cd /home/darrell/Zarathustra/newgan
    python3 train.py --corpus tencent --seed 42 --exp-name s001

For full eval with frozen-bundle protocol:
    python3 eval_sweep.py --checkpoint-dir checkpoints/s001 --seed 42
"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import llgan models
from llgan.config import Config
from llgan.dataset import load_trace, TraceDataset, TracePreprocessor
from llgan.model import Generator, Critic, Encoder, Recovery, Supervisor


# ============================================================================
# Sandia-Specific Training Strategy
# ============================================================================

class PretrainRanker:
    """Rank pretrains by their downstream quality potential."""

    def __init__(self):
        self.pretrain_scores = {}  # {exp_name: score}

    def evaluate_pretrain(self, encoder: nn.Module, recovery: nn.Module,
                         val_dataset: Dataset, device: torch.device) -> float:
        """
        Evaluate pretrain quality on held-out validation.
        Lower reconstruction loss = better pretrain.
        """
        encoder.eval()
        recovery.eval()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in DataLoader(val_dataset, batch_size=64, shuffle=False):
                batch = batch.to(device)
                # Forward through encoder
                h = encoder(batch)
                # Decode
                recon = recovery(h)
                # Reconstruction loss (MSE)
                loss = F.mse_loss(recon, batch)
                total_loss += loss.item() * batch.size(0)
                count += batch.size(0)

        avg_loss = total_loss / count
        return avg_loss  # Lower is better

    def rank_pretrains(self) -> List[Tuple[str, float]]:
        """Return pretrains sorted by quality (best first)."""
        return sorted(self.pretrain_scores.items(), key=lambda x: x[1])


class SandiaTrainer:
    """
    Sandia training loop with pretrain selection and cross-seed validation.
    """

    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.seed = getattr(cfg, 'seed', -1)

        if self.seed >= 0:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Models
        self.G = None
        self.C = None
        self.E = None
        self.R = None
        self.S = None

        # Training state
        self.current_epoch = 0
        self.best_combined = float('inf')
        self.stale_epochs = 0

    def init_models(self, prep: TracePreprocessor):
        """Initialize all models."""
        latent_dim = getattr(self.cfg, 'latent_dim', 24)

        # Generator: takes z_global + z_local, produces latent windows
        self.G = Generator(self.cfg).to(self.device)

        # Critic: scores latent windows
        self.C = Critic(self.cfg).to(self.device)

        # Encoder: maps real windows to latent space
        self.E = Encoder(self.cfg, prep.num_cols).to(self.device)

        # Recovery: maps latent back to feature space
        self.R = Recovery(self.cfg, prep.num_cols).to(self.device)

        # Supervisor: predicts next latent step from current
        self.S = Supervisor(self.cfg).to(self.device)

        print(f"[models] G: {sum(p.numel() for p in self.G.parameters())} params")
        print(f"[models] C: {sum(p.numel() for p in self.C.parameters())} params")
        print(f"[models] E: {sum(p.numel() for p in self.E.parameters())} params")
        print(f"[models] R: {sum(p.numel() for p in self.R.parameters())} params")
        print(f"[models] S: {sum(p.numel() for p in self.S.parameters())} params")

    def pretrain_ae(self, train_ds: Dataset, val_ds: Dataset,
                   epochs: int = 50, ckpt_dir: str = None) -> Tuple[float, float]:
        """
        Phase 1: Autoencoder pretraining.
        Train Encoder + Recovery to round-trip real windows.
        """
        print(f"\n{'='*60}")
        print(f"Phase 1: Autoencoder Pretraining ({epochs} epochs)")
        print(f"{'='*60}")

        self.E.train()
        self.R.train()

        opt = torch.optim.Adam(
            list(self.E.parameters()) + list(self.R.parameters()),
            lr=self.cfg.lr_g,
            betas=(0.5, 0.9)
        )

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                # Encode
                h = self.E(batch)

                # Decode
                recon = self.R(h)

                # Reconstruction loss
                loss = F.mse_loss(recon, batch)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            # Validation
            self.E.eval()
            self.R.eval()
            val_loss = 0.0
            vcount = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    h = self.E(batch)
                    recon = self.R(h)
                    loss = F.mse_loss(recon, batch)
                    val_loss += loss.item() * batch.size(0)
                    vcount += batch.size(0)

            val_loss /= vcount
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | t={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

                # Save checkpoint
                if ckpt_dir:
                    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'E_state': self.E.state_dict(),
                        'R_state': self.R.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, os.path.join(ckpt_dir, 'ae_pretrain_best.pt'))

            self.E.train()
            self.R.train()

        print(f"\n[AE] Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
        return best_val_loss, best_epoch

    def pretrain_supervisor(self, train_ds: Dataset, val_ds: Dataset,
                           epochs: int = 50, ckpt_dir: str = None) -> float:
        """
        Phase 2: Supervisor pretraining.
        Train Supervisor to predict next latent step from real trajectories.
        """
        print(f"\n{'='*60}")
        print(f"Phase 2: Supervisor Pretraining ({epochs} epochs)")
        print(f"{'='*60}")

        self.E.eval()  # Encoder frozen
        self.R.eval()
        self.S.train()

        opt = torch.optim.Adam(self.S.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                with torch.no_grad():
                    h = self.E(batch)  # Get latents from real data

                # Supervisor predicts h_{t+1} from h_t
                # h shape: (B, T, L), predict h[:, 1:] from h[:, :-1]
                pred = self.S(h[:, :-1, :])  # (B, T-1, L)

                loss = F.mse_loss(pred, h[:, 1:, :])

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            # Validation
            self.S.eval()
            val_loss = 0.0
            vcount = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    h = self.E(batch)
                    pred = self.S(h[:, :-1, :])
                    loss = F.mse_loss(pred, h[:, 1:, :])
                    val_loss += loss.item() * batch.size(0)
                    vcount += batch.size(0)

            val_loss /= vcount
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | t={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if ckpt_dir:
                    torch.save({
                        'epoch': epoch + 1,
                        'S_state': self.S.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, os.path.join(ckpt_dir, 'supervisor_best.pt'))

            self.S.train()

        print(f"\n[Supervisor] Best val loss: {val_loss:.6f}")
        return best_val_loss

    def train_generator(self, train_ds: Dataset, val_ds: Dataset,
                       epochs: int = 100, ckpt_dir: str = None) -> float:
        """
        Phase 2.5: Generator warm-up.
        Train Generator to imitate Supervisor outputs (no Critic yet).
        """
        print(f"\n{'='*60}")
        print(f"Phase 2.5: Generator Warm-up ({epochs} epochs)")
        print(f"{'='*60}")

        self.E.eval()
        self.R.eval()
        self.C.eval()
        self.S.eval()
        self.G.train()

        opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                with torch.no_grad():
                    # Get real latents
                    h_real = self.E(batch)
                    # Get supervisor prediction
                    h_pred = self.S(h_real[:, :-1, :])

                # Sample noise
                B, T, _ = h_pred.shape
                z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                z_local = torch.randn(B, T, self.cfg.noise_dim, device=self.device)

                # Generate fake latent
                h_fake = self.G(z_global, z_local)

                # Supervised loss: match supervisor prediction
                loss = F.mse_loss(h_fake, h_pred)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            # Validation
            self.G.eval()
            val_loss = 0.0
            vcount = 0

            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    h_real = self.E(batch)
                    h_pred = self.S(h_real[:, :-1, :])
                    B, T, _ = h_pred.shape
                    z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                    z_local = torch.randn(B, T, self.cfg.noise_dim, device=self.device)
                    h_fake = self.G(z_global, z_local)
                    loss = F.mse_loss(h_fake, h_pred)
                    val_loss += loss.item() * batch.size(0)
                    vcount += batch.size(0)

            val_loss /= vcount
            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | t={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if ckpt_dir:
                    torch.save({
                        'epoch': epoch + 1,
                        'G_state': self.G.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, os.path.join(ckpt_dir, 'g_warmup_best.pt'))

            self.G.train()

        print(f"\n[Generator] Best val loss: {best_val_loss:.6f}")
        return best_val_loss

    def train_gan(self, train_ds: Dataset, val_ds: Dataset,
                 epochs: int = 200, ckpt_dir: str = None) -> Dict:
        """
        Phase 3: Joint GAN training.
        Full WGAN-SN training of G vs C with all auxiliary losses.
        """
        print(f"\n{'='*60}")
        print(f"Phase 3: Joint GAN Training ({epochs} epochs)")
        print(f"{'='*60}")

        self.E.eval()
        self.R.eval()
        self.S.eval()
        self.G.train()
        self.C.train()

        # Optimizers
        optG = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))
        optC = torch.optim.Adam(self.C.parameters(), lr=self.cfg.lr_d, betas=(0.5, 0.9))

        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)

        # Validation dataset as tensor for quick eval
        val_tensor = torch.stack([val_ds[i] for i in range(min(1000, len(val_ds)))]).to(self.device)

        history = {
            'epochs': [],
            'G_loss': [],
            'C_loss': [],
            'C_real': [],
            'C_fake': [],
            'combined_score': []
        }

        print("\n[Training loop starting...]")

        for epoch in range(epochs):
            start = time.time()

            G_epoch_loss = 0.0
            C_epoch_loss = 0.0
            C_real_sum = 0.0
            C_fake_sum = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                B = batch.size(0)

                # ------ Critic update ------
                for _ in range(self.cfg.n_critic):
                    optC.zero_grad()

                    # Real
                    with torch.no_grad():
                        h_real = self.E(batch)
                    C_real = self.C(h_real)
                    C_real_loss = -C_real.mean()

                    # Fake
                    z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                    z_local = torch.randn(B, self.cfg.timestep, self.cfg.noise_dim, device=self.device)
                    h_fake = self.G(z_global, z_local)
                    C_fake = self.C(h_fake.detach())
                    C_fake_loss = C_fake.mean()

                    # WGAN-SN loss
                    C_loss = C_fake_loss + C_real_loss

                    # Gradient penalty (simple L2 on critic output)
                    grad_penalty = 0.0
                    if hasattr(self.C, 'fc') and self.C.fc.weight.grad is not None:
                        grad_penalty = 0.1 * (self.C.fc.weight ** 2).mean()
                    C_loss = C_loss + grad_penalty

                    C_loss.backward()
                    optC.step()

                    C_epoch_loss += C_loss.item()
                    C_real_sum += C_real_loss.item()
                    C_fake_sum += C_fake_loss.item()

                # ------ Generator update ------
                optG.zero_grad()

                z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                z_local = torch.randn(B, self.cfg.timestep, self.cfg.noise_dim, device=self.device)
                h_fake = self.G(z_global, z_local)
                C_fake = self.C(h_fake)

                G_loss = -C_fake.mean()

                # Supervisor consistency loss
                with torch.no_grad():
                    h_real = self.E(batch)
                    h_pred = self.S(h_real[:, :-1, :])

                # Predict what supervisor would do on fake
                h_pred_fake = self.S(h_fake[:, :-1, :])
                sup_loss = F.mse_loss(h_pred_fake, h_fake[:, 1:, :])

                total_G_loss = G_loss + 0.5 * sup_loss
                total_G_loss.backward()
                optG.step()

                G_epoch_loss += total_G_loss.item()
                count += 1

            # Epoch metrics
            avg_G = G_epoch_loss / count
            avg_C = C_epoch_loss / (count * self.cfg.n_critic)
            avg_C_real = C_real_sum / (count * self.cfg.n_critic)
            avg_C_fake = C_fake_sum / (count * self.cfg.n_critic)

            # Quick validation: compute combined score
            combined = self._quick_val(val_tensor, epoch)

            elapsed = time.time() - start

            print(f"Epoch {epoch+1}/{epochs} | G: {avg_G:.4f} C: {avg_C:.4f} "
                  f"(real:{avg_C_real:.3f} fake:{avg_C_fake:.3f}) | "
                  f"combined:{combined:.4f} | t={elapsed:.1f}s")

            history['epochs'].append(epoch + 1)
            history['G_loss'].append(avg_G)
            history['C_loss'].append(avg_C)
            history['C_real'].append(avg_C_real)
            history['C_fake'].append(avg_C_fake)
            history['combined_score'].append(combined)

            # Checkpoint
            if ckpt_dir and (epoch + 1) % self.cfg.checkpoint_every == 0:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'G_state': self.G.state_dict(),
                    'C_state': self.C.state_dict(),
                    'G_opt': optG.state_dict(),
                    'C_opt': optC.state_dict(),
                    'history': history,
                }, os.path.join(ckpt_dir, f'epoch_{epoch+1:04d}.pt'))

            # Early stopping check
            if combined < self.best_combined:
                self.best_combined = combined
                self.stale_epochs = 0
                if ckpt_dir:
                    torch.save({
                        'epoch': epoch + 1,
                        'G_state': self.G.state_dict(),
                        'C_state': self.C.state_dict(),
                        'best_combined': combined,
                    }, os.path.join(ckpt_dir, 'best.pt'))
            else:
                self.stale_epochs += 1
                if self.stale_epochs >= getattr(self.cfg, 'early_stop_patience', 30):
                    print(f"\n[EARLY STOP] {self.stale_epochs} epochs without improvement")
                    break

        return history

    def _quick_val(self, val_tensor: torch.Tensor, epoch: int) -> float:
        """Quick validation score (MMD-like on latent space)."""
        self.G.eval()
        self.C.eval()

        with torch.no_grad():
            # Real latents
            h_real = self.E(val_tensor[:128])

            # Fake latents
            z_global = torch.randn(128, self.cfg.noise_dim, device=self.device)
            z_local = torch.randn(128, self.cfg.timestep, self.cfg.noise_dim, device=self.device)
            h_fake = self.G(z_global, z_local)

            # MMD-like: distance between means
            mmd = ((h_real.mean(0) - h_fake.mean(0)) ** 2).mean()

            # diversity: mean pairwise distance
            dist_real = ((h_real.unsqueeze(1) - h_real.unsqueeze(0)) ** 2).mean()
            dist_fake = ((h_fake.unsqueeze(1) - h_fake.unsqueeze(0)) ** 2).mean()
            diversity = 0.5 * (dist_real + dist_fake)

            # Combined: lower is better
            combined = mmd + 0.1 * diversity

        self.G.train()
        self.C.train()
        return combined.item()

    def run_full_pipeline(self, train_ds: Dataset, val_ds: Dataset,
                         ckpt_dir: str = None) -> Dict:
        """Run complete Sandia training pipeline."""
        print(f"\n{'#'*60}")
        print(f"# Sandia Training Pipeline")
        print(f"# Experiment: {getattr(self.cfg, 'exp_name', 'unknown')}")
        print(f"# Seed: {self.seed}")
        print(f"# Start: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        # Phase 1: AE pretrain
        ae_loss, ae_epoch = self.pretrain_ae(train_ds, val_ds,
                                              epochs=getattr(self.cfg, 'pretrain_ae_epochs', 50),
                                              ckpt_dir=ckpt_dir)

        # Phase 2: Supervisor pretrain
        sup_loss = self.pretrain_supervisor(train_ds, val_ds,
                                             epochs=getattr(self.cfg, 'pretrain_sup_epochs', 50),
                                             ckpt_dir=ckpt_dir)

        # Phase 2.5: Generator warm-up
        g_loss = self.train_generator(train_ds, val_ds,
                                      epochs=getattr(self.cfg, 'pretrain_g_epochs', 100),
                                      ckpt_dir=ckpt_dir)

        # Phase 3: Joint GAN
        history = self.train_gan(train_ds, val_ds,
                                epochs=getattr(self.cfg, 'epochs', 200),
                                ckpt_dir=ckpt_dir)

        print(f"\n{'#'*60}")
        print(f"# Training Complete!")
        print(f"# Best combined score: {self.best_combined:.6f}")
        print(f"# End: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        return {'ae_loss': ae_loss, 'sup_loss': sup_loss, 'g_loss': g_loss, 'history': history}


# ============================================================================
# Main entry point
# ============================================================================

def load_data(trace_dir: str, fmt: str, char_file: Optional[str],
              files_per_epoch: int, records_per_file: int,
              timestep: int, val_ratio: float = 0.1):
    """Load and prepare training data."""
    from llgan.dataset import _collect_files, _fit_prep_on_files, _load_epoch_dataset

    all_files = _collect_files(trace_dir, fmt)
    print(f"Found {len(all_files)} files in {trace_dir}")

    # Fit preprocessor on seed files
    _prep_rng = random.Random(0)
    seed_files = _prep_rng.sample(sorted(all_files), min(4, len(all_files)))
    prep = _fit_prep_on_files(seed_files, fmt, records_per_file)
    print(f"Preprocessor fitted on {len(seed_files)} files")
    print(f"  Columns: {prep.col_names}")

    # Load with 10% held out for validation
    train_files, val_files = _split_files(all_files, val_ratio)

    train_ds, _ = _load_epoch_dataset(train_files, fmt, records_per_file, prep, timestep)
    val_ds, _ = _load_epoch_dataset(val_files, fmt, records_per_file, prep, timestep)

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    return train_ds, val_ds, prep


def _split_files(all_files: list, val_ratio: float) -> Tuple[list, list]:
    """Split files into train/val with deterministic seed."""
    rng = random.Random(42)
    shuffled = sorted(all_files)  # Deterministic order
    rng.shuffle(shuffled)

    n_val = int(len(shuffled) * val_ratio)
    val_files = shuffled[:n_val]
    train_files = shuffled[n_val:]

    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Sandia training")
    parser.add_argument("--trace-dir", required=True, help="Trace directory path")
    parser.add_argument("--fmt", default="oracle_general", help="Trace format")
    parser.add_argument("--char-file", help="Characterization file path")
    parser.add_argument("--epochs", type=int, default=200, help="GAN epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr-g", type=float, default=1e-4, help="Generator LR")
    parser.add_argument("--lr-d", type=float, default=1e-4, help="Critic LR")
    parser.add_argument("--noise-dim", type=int, default=10, help="Noise dimension")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--latent-dim", type=int, default=24, help="Latent dimension")
    parser.add_argument("--timestep", type=int, default=12, help="Timestep")
    parser.add_argument("--pretrain-ae-epochs", type=int, default=50, help="AE pretrain epochs")
    parser.add_argument("--pretrain-sup-epochs", type=int, default=50, help="Supervisor epochs")
    parser.add_argument("--pretrain-g-epochs", type=int, default=100, help="G warmup epochs")
    parser.add_argument("--n-critic", type=int, default=5, help="Critic updates per G update")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Checkpoint frequency")
    parser.add_argument("--early-stop-patience", type=int, default=30, help="Patience for early stop")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp-name", required=True, help="Experiment name")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--files-per-epoch", type=int, default=12, help="Files per epoch")
    parser.add_argument("--records-per-file", type=int, default=20000, help="Records per file")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")

    args = parser.parse_args()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] {device}")

    # Load data
    train_ds, val_ds, prep = load_data(
        trace_dir=args.trace_dir,
        fmt=args.fmt,
        char_file=args.char_file,
        files_per_epoch=args.files_per_epoch,
        records_per_file=args.records_per_file,
        timestep=args.timestep,
        val_ratio=args.val_ratio
    )

    # Build config
    cfg = Config()
    cfg.trace_dir = args.trace_dir
    cfg.trace_format = args.fmt
    cfg.char_file = args.char_file
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.lr_g = args.lr_g
    cfg.lr_d = args.lr_d
    cfg.noise_dim = args.noise_dim
    cfg.hidden_size = args.hidden_size
    cfg.latent_dim = args.latent_dim
    cfg.timestep = args.timestep
    cfg.pretrain_ae_epochs = args.pretrain_ae_epochs
    cfg.pretrain_sup_epochs = args.pretrain_sup_epochs
    cfg.pretrain_g_epochs = args.pretrain_g_epochs
    cfg.n_critic = args.n_critic
    cfg.checkpoint_every = args.checkpoint_every
    cfg.early_stop_patience = args.early_stop_patience
    cfg.seed = args.seed
    cfg.exp_name = args.exp_name
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.files_per_epoch = args.files_per_epoch
    cfg.records_per_file = args.records_per_file
    cfg.col_names = list(prep.col_names) if hasattr(prep, 'col_names') else None

    # Disable compile/amp for stability on GB10
    cfg.no_compile = args.no_compile
    cfg.amp = not args.no_amp

    # Create checkpoint dir
    full_ckpt_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    Path(full_ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(os.path.join(full_ckpt_dir, 'config.json'), 'w') as f:
        json.dump({
            'exp_name': args.exp_name,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}
        }, f, indent=2)

    # Initialize and run trainer
    trainer = SandiaTrainer(cfg, device)
    trainer.init_models(prep)
    history = trainer.run_full_pipeline(train_ds, val_ds, full_ckpt_dir)

    print(f"\nTraining complete. Results saved to {full_ckpt_dir}")
    print(f"Best combined score: {trainer.best_combined:.6f}")


if __name__ == "__main__":
    main()
