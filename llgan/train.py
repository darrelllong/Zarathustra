"""
LLGAN training — supports BCE (paper) and WGAN-GP (default, more stable).

Usage
-----
Single file (original):
    python train.py \
        --trace /Volumes/Archive/Traces/s3-cache-datasets/tw-storage-1.oracleGeneral.zst \
        --fmt oracle_general \
        --epochs 200

Multi-file streaming (samples K files per epoch from a directory):
    python train.py \
        --trace-dir /Volumes/Archive/Traces/s3-cache-datasets \
        --fmt oracle_general \
        --files-per-epoch 8 \
        --records-per-file 15000 \
        --epochs 200

    python train.py --help
"""

import argparse
import os
import platform
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from dataset import load_trace, TracePreprocessor, TraceDataset, _READERS
from model import Generator, Critic
from mmd import evaluate_mmd


# ---------------------------------------------------------------------------
# Multi-file helpers
# ---------------------------------------------------------------------------

def _collect_files(trace_dir: str, fmt: str) -> List[Path]:
    """Return all candidate trace files in a directory."""
    d = Path(trace_dir)
    # Collect everything and filter to likely trace files by suffix
    candidates = [p for p in d.iterdir() if p.is_file()]
    # For oracle_general, prefer .zst / .gz / binary-looking files
    # For csv-like formats, prefer .csv
    # Fall back to all files so user doesn't need to rename things.
    if fmt in ("spc", "msr", "k5cloud", "systor", "csv"):
        exts = {".csv", ".tsv", ".txt", ".gz", ".zst", ""}
    else:  # oracle_general
        exts = {".zst", ".gz", "", ".bin", ".oracleGeneral"}
        # Also accept files with no extension or compound names like foo.oracleGeneral.zst
        candidates = [p for p in candidates
                      if any(p.name.endswith(e) or e == "" for e in exts)]
    return sorted(candidates)


def _load_raw_df(path: Path, fmt: str, max_records: int):
    """Load a single file into a raw DataFrame using the appropriate reader."""
    import pandas as pd
    reader = _READERS.get(fmt)
    if reader is None:
        raise ValueError(f"Unknown format '{fmt}'")

    p = str(path)
    if fmt != "oracle_general" and p.endswith(".zst"):
        import subprocess, tempfile
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        subprocess.run(["zstd", "-d", p, "-o", tmp.name, "-f"], check=True)
        try:
            df = reader(tmp.name, max_records)
        finally:
            os.unlink(tmp.name)
    else:
        df = reader(p, max_records)
    return df


def _fit_prep_on_files(
    files: List[Path],
    fmt: str,
    records_per_file: int,
) -> TracePreprocessor:
    """Fit a TracePreprocessor on a sample of files."""
    import pandas as pd
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
    prep = TracePreprocessor()
    prep.fit(combined)
    return prep


def _load_epoch_dataset(
    files: List[Path],
    fmt: str,
    records_per_file: int,
    prep: TracePreprocessor,
    timestep: int,
) -> Tuple[TraceDataset, Optional[np.ndarray]]:
    """
    Load records_per_file from each file, transform with fitted prep,
    return a concatenated TraceDataset plus a val array (last 20%).
    """
    import pandas as pd
    arrays = []
    for f in files:
        try:
            df = _load_raw_df(f, fmt, records_per_file)
            if len(df) < timestep + 1:
                continue
            arr = prep.transform(df)
            arrays.append(arr)
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}")

    if not arrays:
        return None, None

    combined = np.concatenate(arrays, axis=0)
    # 80/20 split for train/val within this epoch's data
    n_train = int(len(combined) * 0.8)
    train_arr = combined[:n_train]
    val_arr = combined[n_train:]
    return TraceDataset(train_arr, timestep), val_arr


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}  Loss: {cfg.loss}")

    multifile = bool(cfg.trace_dir)
    n_workers = 0 if platform.system() == "Darwin" else 4

    # -----------------------------------------------------------------------
    # Data setup
    # -----------------------------------------------------------------------
    if multifile:
        all_files = _collect_files(cfg.trace_dir, cfg.trace_format)
        if not all_files:
            raise RuntimeError(f"No files found in {cfg.trace_dir}")
        print(f"Trace dir: {cfg.trace_dir}  ({len(all_files)} files found)")

        # Fit preprocessor on a seed set of files (held constant across training)
        n_seed = min(max(cfg.files_per_epoch, 4), len(all_files))
        seed_files = random.sample(all_files, n_seed)
        print(f"Fitting preprocessor on {n_seed} seed files …")
        prep = _fit_prep_on_files(seed_files, cfg.trace_format, cfg.records_per_file)
        print(f"  columns ({prep.num_cols}): {prep.col_names}")
        print(f"  delta-encoded: {prep._delta_cols}")

        # Build a fixed val set from a few held-out files for MMD tracking
        val_files = random.sample(all_files, min(2, len(all_files)))
        import pandas as pd
        val_dfs = []
        for f in val_files:
            try:
                df = _load_raw_df(f, cfg.trace_format, cfg.records_per_file)
                val_dfs.append(df)
            except Exception:
                pass
        val_arr = prep.transform(pd.concat(val_dfs, ignore_index=True)) if val_dfs else None
        val_ds = TraceDataset(val_arr, cfg.timestep) if val_arr is not None else None
        val_tensor = (torch.stack([val_ds[i] for i in range(len(val_ds))])
                      if val_ds and len(val_ds) > 0 else None)
        print(f"  val windows: {len(val_ds) if val_ds else 0:,}")

        # For the first epoch, also build an initial train dataset
        train_ds, _ = _load_epoch_dataset(
            random.sample(all_files, min(cfg.files_per_epoch, len(all_files))),
            cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
        )
    else:
        print(f"Loading: {cfg.trace_path}")
        train_ds, val_ds, prep = load_trace(
            cfg.trace_path, cfg.trace_format, cfg.max_records,
            cfg.timestep, cfg.train_split,
        )
        print(f"  columns ({prep.num_cols}): {prep.col_names}")
        print(f"  delta-encoded: {prep._delta_cols}")
        print(f"  train windows: {len(train_ds):,}  val: {len(val_ds):,}")
        val_tensor = torch.stack([val_ds[i] for i in range(len(val_ds))])

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size).to(device)
    C = Critic(prep.num_cols, cfg.hidden_size).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    bce = nn.BCEWithLogitsLoss() if cfg.loss == "bce" else None

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 0
    latest = sorted(ckpt_dir.glob("epoch_*.pt"))
    if latest:
        ckpt = torch.load(latest[-1], map_location=device)
        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_C.load_state_dict(ckpt["opt_C"])
        start_epoch = ckpt["epoch"] + 1
        # Restore preprocessor from checkpoint so normalization stays consistent
        if "prep" in ckpt:
            prep = ckpt["prep"]
        print(f"Resumed from {latest[-1]} (epoch {start_epoch})")

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        G.train(); C.train()
        t0 = time.time()
        c_losses, g_losses = [], []

        # In multi-file mode: resample files each epoch for broader coverage
        if multifile:
            epoch_files = random.sample(all_files, min(cfg.files_per_epoch, len(all_files)))
            train_ds, _ = _load_epoch_dataset(
                epoch_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
            )
            if train_ds is None or len(train_ds) == 0:
                print(f"Epoch {epoch+1}: no data loaded, skipping.")
                continue

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        for real_batch in train_loader:
            real_batch = real_batch.to(device)
            B = real_batch.size(0)

            # --- Critic steps (n_critic per generator step) ---
            for _ in range(cfg.n_critic):
                z = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                fake_batch = G(z).detach()

                opt_C.zero_grad()
                if cfg.loss == "wgan-gp":
                    c_loss = (C(fake_batch) - C(real_batch)).mean()
                else:  # bce
                    real_labels = torch.ones(B, 1, device=device)
                    fake_labels = torch.zeros(B, 1, device=device)
                    c_loss = (bce(C(real_batch), real_labels) +
                              bce(C(fake_batch), fake_labels))
                c_loss.backward()
                opt_C.step()
                c_losses.append(c_loss.item())

            # --- Generator step ---
            z = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
            fake_batch = G(z)

            opt_G.zero_grad()
            if cfg.loss == "wgan-gp":
                g_loss = -C(fake_batch).mean()
            else:
                real_labels = torch.ones(B, 1, device=device)
                g_loss = bce(C(fake_batch), real_labels)
            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

        c_mean = sum(c_losses) / len(c_losses)
        g_mean = sum(g_losses) / len(g_losses)
        elapsed = time.time() - t0

        n_files_str = (f"  files={len(epoch_files)}" if multifile else
                       f"  windows={len(train_ds):,}")
        log = (f"Epoch {epoch+1:4d}/{cfg.epochs}  "
               f"C={c_mean:.4f}  G={g_mean:.4f}  t={elapsed:.1f}s"
               f"{n_files_str}")

        if val_tensor is not None and (epoch + 1) % cfg.mmd_every == 0:
            mmd_val = evaluate_mmd(G, val_tensor, cfg.mmd_samples,
                                   cfg.timestep, device)
            log += f"  MMD²={mmd_val:.5f}"
            G.train()

        print(log, flush=True)

        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "C": C.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_C": opt_C.state_dict(),
                "prep": prep,
                "config": cfg,
            }, ckpt_path)
            print(f"  → saved {ckpt_path}")

    final_path = ckpt_dir / "final.pt"
    torch.save({
        "epoch": cfg.epochs - 1,
        "G": G.state_dict(),
        "C": C.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_C": opt_C.state_dict(),
        "prep": prep,
        "config": cfg,
    }, final_path)
    print(f"Training complete → {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train LLGAN on an I/O trace")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace",     metavar="FILE",
                     help="Single trace file (original mode)")
    src.add_argument("--trace-dir", metavar="DIR",
                     help="Directory of trace files (multi-file streaming mode)")

    p.add_argument("--fmt",              default="spc",
                   choices=["spc","msr","k5cloud","systor","oracle_general","csv"])
    p.add_argument("--loss",             default="wgan-gp",
                   choices=["wgan-gp", "bce"])
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--timestep",         type=int,   default=12)
    p.add_argument("--noise-dim",        type=int,   default=10)
    p.add_argument("--hidden-size",      type=int,   default=256)
    p.add_argument("--lr-g",             type=float, default=0.0001)
    p.add_argument("--lr-d",             type=float, default=0.0001)
    p.add_argument("--n-critic",         type=int,   default=5)
    p.add_argument("--max-records",      type=int,   default=60_000,
                   help="Max records per file in single-file mode")
    # Multi-file options
    p.add_argument("--files-per-epoch",  type=int,   default=8,
                   help="Files sampled per epoch in --trace-dir mode")
    p.add_argument("--records-per-file", type=int,   default=15_000,
                   help="Records loaded from each file per epoch")
    # Output / checkpointing
    p.add_argument("--checkpoint-dir",   default="checkpoints")
    p.add_argument("--checkpoint-every", type=int,   default=10)
    p.add_argument("--mmd-every",        type=int,   default=5)
    p.add_argument("--mmd-samples",      type=int,   default=1000)
    p.add_argument("--train-split",      type=float, default=0.8)
    args = p.parse_args()

    cfg = Config()
    cfg.trace_path       = args.trace or ""
    cfg.trace_dir        = args.trace_dir or ""
    cfg.trace_format     = args.fmt
    cfg.loss             = args.loss
    cfg.epochs           = args.epochs
    cfg.batch_size       = args.batch_size
    cfg.timestep         = args.timestep
    cfg.noise_dim        = args.noise_dim
    cfg.hidden_size      = args.hidden_size
    cfg.lr_g             = args.lr_g
    cfg.lr_d             = args.lr_d
    cfg.n_critic         = args.n_critic
    cfg.max_records      = args.max_records
    cfg.files_per_epoch  = args.files_per_epoch
    cfg.records_per_file = args.records_per_file
    cfg.checkpoint_dir   = args.checkpoint_dir
    cfg.checkpoint_every = args.checkpoint_every
    cfg.mmd_every        = args.mmd_every
    cfg.mmd_samples      = args.mmd_samples
    cfg.train_split      = args.train_split
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
