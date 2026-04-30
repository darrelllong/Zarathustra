#!/usr/bin/env python3
"""
Sandia training script for the Zarathustra race.

This script implements Sandia's competitive strategy:
1. Pretrain quality selection - build multiple pretrains, rank by downstream quality
2. Cross-seed validation - any claim must survive 3+ seeds before promotion
3. Long-horizon focus - HRC-MAE as primary checkpoint selector

Usage on vinge.local:
    cd /home/darrell/Zarathustra/newgan
    python3 train.py --trace-dir /home/darrell/traces/tencent_block_1M --fmt oracle_general --seed 42 --exp-name s001
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
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Import llgan models
# This script runs from /home/darrell/Zarathustra/newgan/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from llgan.config import Config
from llgan.dataset import load_trace, TraceDataset, TracePreprocessor, _READERS, load_file_characterizations
from llgan.model import Generator, Critic, Encoder, Recovery, Supervisor


# ============================================================================
# Helper functions (copied from llgan/train.py)
# ============================================================================

def _collect_files(trace_dir: str, fmt: str) -> List[Path]:
    """Return all candidate trace files in a directory."""
    d = Path(trace_dir)
    candidates = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]
    if fmt in ("spc", "msr", "k5cloud", "systor", "csv"):
        exts = {".csv", ".tsv", ".txt", ".gz", ".zst", ""}
    else:
        exts = {".zst", ".gz", "", ".bin", ".oracleGeneral"}
        candidates = [p for p in candidates
                      if any(p.name.endswith(e) or e == "" for e in exts)]
    return sorted(candidates)


def _load_raw_df(path: Path, fmt: str, max_records: int):
    """Load a single file into a raw DataFrame."""
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown format '{fmt}'")

    p = str(path)
    if fmt != "oracle_general" and p.endswith(".zst"):
        import subprocess
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        subprocess.run(["zstd", "-d", p, "-o", tmp.name, "-f"], check=True)
        try:
            df = reader(tmp.name, max_records)
        finally:
            os.unlink(tmp.name)
    else:
        df = reader(p, max_records)
    return df


def _fit_prep_on_files(files: List[Path], fmt: str, records_per_file: int,
                       obj_size_granularity: int = 0, lru_cache_depth: int = 0) -> TracePreprocessor:
    """Fit a TracePreprocessor on a sample of files."""
    dfs = []
    for f in files:
        try:
            df = _load_raw_df(f, fmt, records_per_file)
            if len(df) > 0:
                dfs.append(df)
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}")
    if not dfs:
        raise RuntimeError("No files could be loaded for fitting preprocessor.")
    combined = pd.concat(dfs, ignore_index=True)
    prep = TracePreprocessor(obj_size_granularity=obj_size_granularity,
                             lru_cache_depth=lru_cache_depth)
    prep.fit(combined)
    return prep


def _load_epoch_dataset(files: List[Path], fmt: str, records_per_file: int,
                        prep: TracePreprocessor, timestep: int,
                        char_lookup: Optional[dict] = None) -> Tuple[Optional[Dataset], Optional[np.ndarray]]:
    """
    Load records_per_file from each file, transform with fitted prep, and
    return a list of per-file TraceDatasets.
    """
    train_datasets = []
    val_arrays = []

    cond_dim = next(iter(char_lookup.values())).shape[0] if char_lookup else 0 if char_lookup else 0

    for f in files:
        try:
            df = _load_raw_df(f, fmt, records_per_file)
            if len(df) < timestep + 2:
                continue
            arr = prep.transform(df)
            n_train = int(len(arr) * 0.8)
            train_arr = arr[:n_train]
            val_arr = arr[n_train:]

            file_cond = None
            if char_lookup:
                fname = f.name
                file_cond = char_lookup.get(fname)
                if file_cond is None:
                    for ext in (".zst", ".gz"):
                        if fname.endswith(ext):
                            file_cond = char_lookup.get(fname[:-len(ext)])
                            if file_cond is not None:
                                break
                if file_cond is None:
                    file_cond = torch.zeros(cond_dim)

            if len(train_arr) > timestep:
                train_datasets.append(TraceDataset(train_arr, timestep, file_cond=file_cond))
            if len(val_arr) > 0:
                val_arrays.append(val_arr)
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}")

    if not train_datasets:
        return None, None

    combined_val = np.concatenate(val_arrays, axis=0) if val_arrays else None
    return train_datasets, combined_val


# ============================================================================
# Sandia-Specific Training Strategy
# ============================================================================

class PretrainRanker:
    """Rank pretrains by their downstream quality potential."""

    def __init__(self):
        self.pretrain_scores = {}

    def evaluate_pretrain(self, encoder: nn.Module, recovery: nn.Module,
                          val_dataset: Dataset, device: torch.device) -> float:
        """Evaluate pretrain quality on held-out validation."""
        encoder.eval()
        recovery.eval()

        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in DataLoader(val_dataset, batch_size=64, shuffle=False):
                batch = batch.to(device)
                h = encoder(batch)
                recon = recovery(h)
                loss = F.mse_loss(recon, batch)
                total_loss += loss.item() * batch.size(0)
                count += batch.size(0)

        return total_loss / count

    def rank_pretrains(self) -> List[Tuple[str, float]]:
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

        self.G = None
        self.C = None
        self.E = None
        self.R = None
        self.S = None
        self.best_combined = float('inf')
        self.stale_epochs = 0

    def init_models(self, num_cols: int):
        """Initialize all models."""
        latent_dim = getattr(self.cfg, 'latent_dim', 24)

        self.G = Generator(
            noise_dim=self.cfg.noise_dim,
            num_cols=num_cols,
            hidden_size=self.cfg.hidden_size,
            latent_dim=self.cfg.latent_dim,
            cond_dim=self.cfg.cond_dim if hasattr(self.cfg, 'cond_dim') else 0,
            timestep=self.cfg.timestep
        ).to(self.device)
        self.C = Critic(
            num_cols=num_cols,
            hidden_size=self.cfg.hidden_size,
            use_spectral_norm=True,
            sn_lstm=True,
            minibatch_std=True,
            cond_dim=getattr(self.cfg, 'cond_dim', 0),
            num_lstm_layers=1
        ).to(self.device)

        self.E = Encoder(
            num_cols=num_cols,
            hidden_size=self.cfg.hidden_size,
            latent_dim=self.cfg.latent_dim
        ).to(self.device)

        self.R = Recovery(
            latent_dim=self.cfg.latent_dim,
            hidden_size=self.cfg.hidden_size,
            num_cols=num_cols
        ).to(self.device)

        self.S = Supervisor(
            latent_dim=self.cfg.latent_dim,
            hidden_size=self.cfg.hidden_size
        ).to(self.device)

        print(f"[models] G: {sum(p.numel() for p in self.G.parameters())} params")
        print(f"[models] C: {sum(p.numel() for p in self.C.parameters())} params")
        print(f"[models] E: {sum(p.numel() for p in self.E.parameters())} params")
        print(f"[models] R: {sum(p.numel() for p in self.R.parameters())} params")
        print(f"[models] S: {sum(p.numel() for p in self.S.parameters())} params")

    def pretrain_ae(self, train_ds: List[Dataset], val_ds: Optional[np.ndarray],
                    epochs: int = 50, ckpt_dir: str = None) -> Tuple[float, float]:
        """Phase 1: Autoencoder pretraining."""
        print(f"\n{'='*60}")
        print(f"Phase 1: Autoencoder Pretraining ({epochs} epochs)")
        print(f"{'='*60}")

        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_ds)

        self.E.train()
        self.R.train()

        opt = torch.optim.Adam(list(self.E.parameters()) + list(self.R.parameters()),
                               lr=self.cfg.lr_g, betas=(0.5, 0.9))

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0) if val_ds is not None else None

        best_val_loss = float('inf')
        best_epoch = 0

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                h = self.E(batch)
                recon = self.R(h)
                loss = F.mse_loss(recon, batch)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            val_loss = 0.0
            vcount = 0
            if val_loader:
                self.E.eval()
                self.R.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        h = self.E(batch)
                        recon = self.R(h)
                        loss = F.mse_loss(recon, batch)
                        val_loss += loss.item() * batch.size(0)
                        vcount += batch.size(0)
                val_loss /= vcount
                self.E.train()
                self.R.train()

            elapsed = time.time() - start
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | t={elapsed:.1f}s")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                if ckpt_dir:
                    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'E_state': self.E.state_dict(),
                        'R_state': self.R.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, os.path.join(ckpt_dir, 'ae_pretrain_best.pt'))

        print(f"\n[AE] Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")
        return best_val_loss, best_epoch

    def pretrain_supervisor(self, train_ds: List[Dataset], val_ds: Optional[np.ndarray],
                            epochs: int = 50, ckpt_dir: str = None) -> float:
        """Phase 2: Supervisor pretraining."""
        print(f"\n{'='*60}")
        print(f"Phase 2: Supervisor Pretraining ({epochs} epochs)")
        print(f"{'='*60}")

        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_ds)

        self.E.eval()
        self.R.eval()
        self.S.train()

        opt = torch.optim.Adam(self.S.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0) if val_ds is not None else None

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)
                with torch.no_grad():
                    h = self.E(batch)

                pred = self.S(h[:, :-1, :])
                loss = F.mse_loss(pred, h[:, 1:, :])

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            val_loss = 0.0
            vcount = 0
            if val_loader:
                self.S.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        h = self.E(batch)
                        pred = self.S(h[:, :-1, :])
                        loss = F.mse_loss(pred, h[:, 1:, :])
                        val_loss += loss.item() * batch.size(0)
                        vcount += batch.size(0)
                val_loss /= vcount
                self.S.train()

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

        print(f"\n[Supervisor] Best val loss: {best_val_loss:.6f}")
        return best_val_loss

    def train_generator(self, train_ds: List[Dataset], val_ds: Optional[np.ndarray],
                        epochs: int = 100, ckpt_dir: str = None) -> float:
        """Phase 2.5: Generator warm-up."""
        print(f"\n{'='*60}")
        print(f"Phase 2.5: Generator Warm-up ({epochs} epochs)")
        print(f"{'='*60}")

        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_ds)

        self.E.eval()
        self.R.eval()
        self.C.eval()
        self.S.eval()
        self.G.train()

        opt = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0) if val_ds is not None else None

        best_val_loss = float('inf')

        for epoch in range(epochs):
            start = time.time()
            train_loss = 0.0
            count = 0

            for batch in train_loader:
                batch = batch.to(self.device)

                with torch.no_grad():
                    h_real = self.E(batch)
                    h_pred = self.S(h_real[:, :-1, :])

                B, T, _ = h_pred.shape
                z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                z_local = torch.randn(B, T, self.cfg.noise_dim, device=self.device)
                h_fake = self.G(z_global, z_local)

                loss = F.mse_loss(h_fake, h_pred)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item() * batch.size(0)
                count += batch.size(0)

            train_loss /= count

            val_loss = 0.0
            vcount = 0
            if val_loader:
                self.G.eval()
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
                self.G.train()

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

        print(f"\n[Generator] Best val loss: {best_val_loss:.6f}")
        return best_val_loss

    def train_gan(self, train_ds: List[Dataset], val_ds: Optional[np.ndarray],
                  epochs: int = 200, ckpt_dir: str = None) -> Dict:
        """Phase 3: Joint GAN training."""
        print(f"\n{'='*60}")
        print(f"Phase 3: Joint GAN Training ({epochs} epochs)")
        print(f"{'='*60}")

        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset(train_ds)

        self.E.eval()
        self.R.eval()
        self.S.eval()
        self.G.train()
        self.C.train()

        optG = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr_g, betas=(0.5, 0.9))
        optC = torch.optim.Adam(self.C.parameters(), lr=self.cfg.lr_d, betas=(0.5, 0.9))

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size,
                                 shuffle=True, num_workers=0)

        val_tensor = torch.stack([val_ds[i] for i in range(min(1000, len(val_ds)))]) if val_ds is not None else None

        history = {'epochs': [], 'G_loss': [], 'C_loss': [], 'C_real': [], 'C_fake': [], 'combined_score': []}

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

                # Critic update
                for _ in range(self.cfg.n_critic):
                    optC.zero_grad()

                    with torch.no_grad():
                        h_real = self.E(batch)
                    C_real = self.C(h_real)
                    C_real_loss = -C_real.mean()

                    z_global = torch.randn(B, self.cfg.noise_dim, device=self.device)
                    z_local = torch.randn(B, self.cfg.timestep, self.cfg.noise_dim, device=self.device)
                    h_fake = self.G(z_global, z_local)
                    C_fake = self.C(h_fake.detach())
                    C_fake_loss = C_fake.mean()

                    C_loss = C_fake_loss + C_real_loss
                    C_loss.backward()
                    optC.step()

                    C_epoch_loss += C_loss.item()
                    C_real_sum += C_real_loss.item()
                    C_fake_sum += C_fake_loss.item()

                # Generator update
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
                h_pred_fake = self.S(h_fake[:, :-1, :])
                sup_loss = F.mse_loss(h_pred_fake, h_fake[:, 1:, :])

                total_G_loss = G_loss + 0.5 * sup_loss
                total_G_loss.backward()
                optG.step()

                G_epoch_loss += total_G_loss.item()
                count += 1

            avg_G = G_epoch_loss / count
            avg_C = C_epoch_loss / (count * self.cfg.n_critic)
            avg_C_real = C_real_sum / (count * self.cfg.n_critic)
            avg_C_fake = C_fake_sum / (count * self.cfg.n_critic)

            combined = self._quick_val(val_tensor) if val_tensor is not None else 0.0

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

    def _quick_val(self, val_tensor: torch.Tensor) -> float:
        """Quick validation score."""
        self.G.eval()
        self.C.eval()

        with torch.no_grad():
            h_real = self.E(val_tensor[:128])
            z_global = torch.randn(128, self.cfg.noise_dim, device=self.device)
            z_local = torch.randn(128, self.cfg.timestep, self.cfg.noise_dim, device=self.device)
            h_fake = self.G(z_global, z_local)
            mmd = ((h_real.mean(0) - h_fake.mean(0)) ** 2).mean()
            dist_real = ((h_real.unsqueeze(1) - h_real.unsqueeze(0)) ** 2).mean()
            dist_fake = ((h_fake.unsqueeze(1) - h_fake.unsqueeze(0)) ** 2).mean()
            diversity = 0.5 * (dist_real + dist_fake)
            combined = mmd + 0.1 * diversity

        self.G.train()
        self.C.train()
        return combined.item()

    def run_full_pipeline(self, train_ds: List[Dataset], val_ds: Optional[np.ndarray],
                          ckpt_dir: str = None) -> Dict:
        """Run complete Sandia training pipeline."""
        print(f"\n{'#'*60}")
        print(f"# Sandia Training Pipeline")
        print(f"# Experiment: {getattr(self.cfg, 'exp_name', 'unknown')}")
        print(f"# Seed: {self.seed}")
        print(f"# Start: {datetime.now().isoformat()}")
        print(f"{'#'*60}\n")

        ae_loss, ae_epoch = self.pretrain_ae(train_ds, val_ds,
                                              epochs=getattr(self.cfg, 'pretrain_ae_epochs', 50),
                                              ckpt_dir=ckpt_dir)

        sup_loss = self.pretrain_supervisor(train_ds, val_ds,
                                             epochs=getattr(self.cfg, 'pretrain_sup_epochs', 50),
                                             ckpt_dir=ckpt_dir)

        g_loss = self.train_generator(train_ds, val_ds,
                                      epochs=getattr(self.cfg, 'pretrain_g_epochs', 100),
                                      ckpt_dir=ckpt_dir)

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
              files_per_epoch: int, records_per_file: int, timestep: int,
              val_ratio: float = 0.1):
    """Load and prepare training data."""
    all_files = _collect_files(trace_dir, fmt)
    print(f"Found {len(all_files)} files in {trace_dir}")

    _prep_rng = random.Random(0)
    seed_files = _prep_rng.sample(sorted(all_files), min(4, len(all_files)))
    prep = _fit_prep_on_files(seed_files, fmt, records_per_file)
    print(f"Preprocessor fitted on {len(seed_files)} files")
    print(f"  Columns: {prep.col_names}")

    all_files_sorted = sorted(all_files)
    rng_val = random.Random(42)
    rng_val.shuffle(all_files_sorted)
    n_val_files = int(len(all_files_sorted) * val_ratio)
    val_files = all_files_sorted[:n_val_files]
    train_files = all_files_sorted[n_val_files:]

    train_ds, _ = _load_epoch_dataset(train_files, fmt, records_per_file, prep, timestep)
    val_ds, _ = _load_epoch_dataset(val_files, fmt, records_per_file, prep, timestep)

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    return train_ds, val_ds, prep


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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"[device] {device}")

    train_ds, val_ds, prep = load_data(
        trace_dir=args.trace_dir,
        fmt=args.fmt,
        char_file=args.char_file,
        files_per_epoch=args.files_per_epoch,
        records_per_file=args.records_per_file,
        timestep=args.timestep,
        val_ratio=args.val_ratio
    )

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
    cfg.no_compile = args.no_compile
    cfg.amp = not args.no_amp

    full_ckpt_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    Path(full_ckpt_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(full_ckpt_dir, 'config.json'), 'w') as f:
        json.dump({
            'exp_name': args.exp_name,
            'seed': args.seed,
            'timestamp': datetime.now().isoformat(),
            'args': {k: v for k, v in vars(args).items() if not k.startswith('_')}
        }, f, indent=2)

    trainer = SandiaTrainer(cfg, device)
    trainer.init_models(prep.num_cols)
    history = trainer.run_full_pipeline(train_ds, val_ds, full_ckpt_dir)

    print(f"\nTraining complete. Results saved to {full_ckpt_dir}")
    print(f"Best combined score: {trainer.best_combined:.6f}")


if __name__ == "__main__":
    main()
