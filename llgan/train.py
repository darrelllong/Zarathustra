"""
LLGAN training — supports BCE (paper) and WGAN-SN (default, more stable).

WGAN-SN: Wasserstein loss with spectral normalisation on the critic's output
linear layer. Not true WGAN-GP (no gradient penalty) — the LSTM weights are
unconstrained. True WGAN-GP requires second-order autograd (CUDA only).

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
import copy
import os
import platform
import random
import time
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from config import Config
from dataset import load_trace, TracePreprocessor, TraceDataset, _READERS
from model import Generator, Critic, Encoder, Recovery, Supervisor
from mmd import evaluate_mmd, evaluate_metrics


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
) -> Tuple[Optional[ConcatDataset], Optional[np.ndarray]]:
    """
    Load records_per_file from each file, transform with fitted prep, and
    return a ConcatDataset of per-file TraceDatasets.

    Critically: each file gets its own TraceDataset so sliding windows are
    never formed across file boundaries (last event of file A → first event
    of file B would be causally meaningless and poison a sequence model).
    """
    import pandas as pd
    train_datasets = []
    val_arrays = []

    for f in files:
        try:
            df = _load_raw_df(f, fmt, records_per_file)
            if len(df) < timestep + 2:
                continue
            arr = prep.transform(df)
            # 80/20 split within each file so val windows are also file-local
            n_train = int(len(arr) * 0.8)
            train_arr = arr[:n_train]
            val_arr   = arr[n_train:]
            if len(train_arr) > timestep:
                train_datasets.append(TraceDataset(train_arr, timestep))
            if len(val_arr) > 0:
                val_arrays.append(val_arr)
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}")

    if not train_datasets:
        return None, None

    combined_val = np.concatenate(val_arrays, axis=0) if val_arrays else None
    return ConcatDataset(train_datasets), combined_val


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

    # WGAN-GP and R2 both use create_graph=True to compute second-order gradients
    # through the critic LSTM.  MPS does not implement lstm_mps_backward as a
    # differentiable operation, so these features are CUDA-only.
    if device.type == "mps":
        if cfg.loss == "wgan-gp":
            print("[warn] wgan-gp requires second-order LSTM gradients; "
                  "MPS does not support lstm_mps_backward. Falling back to wgan-sn.")
            cfg.loss = "wgan-sn"
        if cfg.r1_lambda > 0:
            print("[warn] r1-lambda requires create_graph=True through LSTM critic; "
                  "not supported on MPS. Disabling R1.")
            cfg.r1_lambda = 0.0
        if cfg.r2_lambda > 0:
            print("[warn] r2-lambda requires create_graph=True through LSTM critic; "
                  "not supported on MPS. Disabling R2.")
            cfg.r2_lambda = 0.0

    # AMP: float16 LSTM forward passes for 2-3× speedup and halved VRAM on CUDA.
    # Incompatible with create_graph=True (wgan-gp, r1, r2) because dynamo can't
    # differentiate through autocast LSTM kernels at second order.
    use_amp = (device.type == "cuda" and cfg.amp
               and cfg.loss != "wgan-gp"
               and cfg.r1_lambda == 0.0
               and cfg.r2_lambda == 0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device: {device}  Loss: {cfg.loss}  AMP: {use_amp}")

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
        print(f"  log-transformed: {prep._log_cols}")

        # Hold out a fixed val set — remove val files from all_files so they
        # are never sampled during training.  Build per-file TraceDatasets
        # (same as training) so no window ever crosses a file boundary.
        n_val_files = min(2, max(0, len(all_files) - cfg.files_per_epoch))
        val_files = random.sample(all_files, n_val_files)
        all_files = [f for f in all_files if f not in val_files]
        val_ds, _ = _load_epoch_dataset(
            val_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
        ) if val_files else (None, [])
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
        print(f"  log-transformed: {prep._log_cols}")
        print(f"  train windows: {len(train_ds):,}  val: {len(val_ds):,}")
        val_tensor = torch.stack([val_ds[i] for i in range(len(val_ds))])

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    latent_ae = cfg.latent_dim > 0

    # In latent AE mode the critic operates on latent sequences; in legacy mode
    # it operates directly on feature sequences.
    critic_input_dim = cfg.latent_dim if latent_ae else prep.num_cols

    # Column index of obj_id in the feature vector (used by locality loss).
    obj_id_col = (prep.col_names.index("obj_id")
                  if hasattr(prep, "col_names") and "obj_id" in prep.col_names
                  else 1)

    # WGAN-GP enforces Lipschitz via gradient penalty — spectral norm is
    # redundant and can interfere with the penalty gradient computation.
    use_sn = (cfg.loss != "wgan-gp")
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=cfg.latent_dim if latent_ae else None).to(device)
    C = Critic(critic_input_dim, cfg.hidden_size,
               use_spectral_norm=use_sn,
               sn_lstm=cfg.sn_lstm,
               minibatch_std=cfg.minibatch_std,
               patch_embed=cfg.patch_embed).to(device)

    if latent_ae:
        E = Encoder(prep.num_cols, cfg.hidden_size, cfg.latent_dim).to(device)
        R = Recovery(cfg.latent_dim, cfg.hidden_size, prep.num_cols).to(device)
        S = Supervisor(cfg.latent_dim, cfg.hidden_size).to(device)
        opt_AE  = torch.optim.Adam(list(E.parameters()) + list(R.parameters()),
                                   lr=cfg.lr_er, betas=(0.9, 0.999))
        opt_S   = torch.optim.Adam(S.parameters(), lr=cfg.lr_er, betas=(0.9, 0.999))
        opt_ER  = torch.optim.Adam(list(E.parameters()) + list(R.parameters()),
                                   lr=cfg.lr_er, betas=(0.9, 0.999))
    else:
        E = R = S = opt_AE = opt_S = opt_ER = None

    # torch.compile: fuses CUDA kernels for ~20-40% speedup on NVIDIA hardware.
    # Skipped on MPS/CPU (no benefit) and when create_graph gradients are needed
    # (WGAN-GP / R1 / R2), because dynamo does not yet fully support
    # higher-order differentiation through compiled LSTM graphs.
    _can_compile = (device.type == "cuda"
                    and cfg.compile
                    and cfg.loss != "wgan-gp"
                    and cfg.r1_lambda == 0.0
                    and cfg.r2_lambda == 0.0)
    if _can_compile:
        try:
            G = torch.compile(G)
            C = torch.compile(C)
            if latent_ae:
                E = torch.compile(E)
                R = torch.compile(R)
                S = torch.compile(S)
            print("[info] torch.compile: models compiled (CUDA).")
        except Exception as exc:
            print(f"[warn] torch.compile failed, running eager: {exc}")

    # Two-timescale gradient descent-ascent (JMLR 2025): the generator uses a
    # faster learning rate (lr_g > lr_d) while the critic uses a slower one.
    # This asymmetry gives the critic time to "catch up" to each generator
    # update without oscillating wildly, which is the theoretical condition for
    # GAN convergence in the two-player minimax setting.  In practice on MPS:
    # lr_g=0.0001, lr_d=0.00005, n_critic=3 (not 5) — fewer critic steps
    # combine with slower lr_d to achieve the same effective critic-to-generator
    # update ratio without letting the unconstrained LSTM explode (v12 failure).
    # Beta1=0.5 (vs the usual 0.9) reduces gradient momentum, which helps with
    # GAN oscillations by making each update less "sticky".
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    # Cosine annealing: decay lr from initial → eta_min over cfg.epochs steps.
    # In the second half of training the generator and critic are close to
    # equilibrium — large gradient steps cause them to perpetually overshoot
    # each other (GAN cycling oscillation).  Cosine annealing damps those steps
    # smoothly without a hard schedule boundary.  EMA handles the weight-space
    # noise; cosine annealing handles the training-dynamics noise.
    # eta_min = lr * lr_cosine_decay (default 0.05 = 5% of initial LR).
    # Set lr_cosine_decay=0 to disable (constant LR, backward-compatible).
    if cfg.lr_cosine_decay > 0:
        sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_G, T_max=cfg.epochs, eta_min=cfg.lr_g * cfg.lr_cosine_decay
        )
        sched_C = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_C, T_max=cfg.epochs, eta_min=cfg.lr_d * cfg.lr_cosine_decay
        )
    else:
        sched_G = sched_C = None

    bce = nn.BCEWithLogitsLoss() if cfg.loss == "bce" else None

    # EMA (Exponential Moving Average) of generator weights.
    # GANs oscillate: the generator finds a good point, the critic adapts and
    # pushes it away, the generator moves again.  The EMA tracks the geometric
    # centre of the generator's recent trajectory rather than its instantaneous
    # position, which smooths out this cycling and consistently produces better
    # samples than the live weights (StyleGAN2, Karras et al. 2020; DDPM,
    # Ho et al. 2020; BigGAN, Brock et al. 2019 all use EMA for inference).
    # We evaluate and checkpoint using ema_G, not G directly.
    # Decay 0.999: with ~100 batches/epoch the effective window is ~1000 steps
    # (~10 epochs), long enough to smooth oscillation but short enough to track
    # genuine improvement.
    ema_G_state = copy.deepcopy(G.state_dict())
    ema_decay   = cfg.ema_decay

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_combined      = float("inf")   # combined = MMD² + 0.2*(1-recall), for best.pt
    epochs_no_improve  = 0              # consecutive MMD evals without improvement
    mmd_history: list  = []             # [(epoch, mmd, recall, combined), ...]

    # Resume
    # --resume-from <path>  : load a specific checkpoint file (overrides auto-detect)
    # --reset-optimizer     : load model weights only; fresh optimizers + schedulers
    #                         at the current --lr-g / --lr-d values.  Use with
    #                         --resume-from to hot-start from a good checkpoint with
    #                         different hyperparameters after a collapse.
    start_epoch = 0
    resume_path = None
    if cfg.resume_from:
        resume_path = Path(cfg.resume_from)
    else:
        latest = sorted(ckpt_dir.glob("epoch_*.pt"))
        if latest:
            resume_path = latest[-1]

    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        # Migrate critic state dict: old checkpoints store plain LSTM weight
        # keys (e.g. "lstm.weight_ih_l0"); new SN-LSTM models expect "_orig"
        # suffixed keys ("lstm.weight_ih_l0_orig").  Rename on load so old
        # checkpoints work with the new architecture.  strict=False skips the
        # newly added _u/_v power-iteration buffers (initialised randomly and
        # updated on the first forward pass).
        c_sd = {
            (k + "_orig" if k in ("lstm.weight_ih_l0", "lstm.weight_hh_l0") else k): v
            for k, v in ckpt["C"].items()
        }
        C.load_state_dict(c_sd, strict=False)
        if latent_ae and "E" in ckpt:
            E.load_state_dict(ckpt["E"])
            R.load_state_dict(ckpt["R"])
            S.load_state_dict(ckpt["S"])
        # Restore preprocessor from checkpoint so normalization stays consistent
        if "prep" in ckpt:
            prep = ckpt["prep"]
        # Restore EMA state if available; otherwise seed from current G weights
        if "G_ema" in ckpt:
            ema_G_state = ckpt["G_ema"]
        else:
            ema_G_state = copy.deepcopy(G.state_dict())

        if cfg.reset_optimizer:
            # Fresh optimizers at new LR — do NOT restore optimizer state, scheduler,
            # mmd_history, best_combined, or start_epoch.  Training restarts from
            # epoch 0 with the loaded model weights and the new hyperparameters.
            print(f"Hot-start from {resume_path} with reset optimizer "
                  f"(lr_g={cfg.lr_g:.2e}, lr_d={cfg.lr_d:.2e}, n_critic={cfg.n_critic})")
        else:
            opt_G.load_state_dict(ckpt["opt_G"])
            opt_C.load_state_dict(ckpt["opt_C"])
            start_epoch = ckpt["epoch"] + 1
            # Restore scheduler state if present; otherwise fast-forward to the
            # correct position so the LR matches start_epoch (handles old checkpoints
            # that predate scheduler support without silently using the wrong LR).
            if sched_G is not None:
                if "sched_G" in ckpt:
                    sched_G.load_state_dict(ckpt["sched_G"])
                    sched_C.load_state_dict(ckpt["sched_C"])
                else:
                    for _ in range(start_epoch):
                        sched_G.step()
                        sched_C.step()
            # Restore mmd_history so the curve is continuous across restarts
            if "mmd_history" in ckpt:
                mmd_history = ckpt["mmd_history"]
            # Restore best_combined so early stopping doesn't misfire
            if "combined" in ckpt:
                best_combined = ckpt["combined"]
            print(f"Resumed from {resume_path} (epoch {start_epoch})")

    # -----------------------------------------------------------------------
    # Phase 1: Autoencoder pretraining  (latent AE mode only)
    # -----------------------------------------------------------------------
    if latent_ae and start_epoch == 0:
        print(f"\n--- Phase 1: Autoencoder pretraining ({cfg.pretrain_ae_epochs} epochs) ---")
        for pre_ep in range(cfg.pretrain_ae_epochs):
            E.train(); R.train()
            if multifile:
                ep_files = random.sample(all_files, min(cfg.files_per_epoch, len(all_files)))
                pre_ds, _ = _load_epoch_dataset(
                    ep_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep)
            else:
                pre_ds = train_ds
            if pre_ds is None or len(pre_ds) == 0:
                continue
            loader = DataLoader(pre_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=n_workers, drop_last=True)
            ae_losses = []
            for xb in loader:
                xb = xb.to(device)
                opt_AE.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    loss_ae = nn.functional.mse_loss(R(E(xb)), xb)
                scaler.scale(loss_ae).backward()
                scaler.step(opt_AE)
                scaler.update()
                ae_losses.append(loss_ae.item())
            ae_mean = sum(ae_losses) / len(ae_losses)
            if (pre_ep + 1) % 10 == 0 or pre_ep == 0:
                print(f"  AE pretrain {pre_ep+1:3d}/{cfg.pretrain_ae_epochs}  "
                      f"recon={ae_mean:.5f}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Supervisor pretraining  (latent AE mode only)
    # -----------------------------------------------------------------------
    if latent_ae and start_epoch == 0:
        print(f"\n--- Phase 2: Supervisor pretraining ({cfg.pretrain_sup_epochs} epochs) ---")
        E.eval()  # freeze encoder during supervisor pretraining
        for pre_ep in range(cfg.pretrain_sup_epochs):
            S.train()
            if multifile:
                ep_files = random.sample(all_files, min(cfg.files_per_epoch, len(all_files)))
                pre_ds, _ = _load_epoch_dataset(
                    ep_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep)
            else:
                pre_ds = train_ds
            if pre_ds is None or len(pre_ds) == 0:
                continue
            loader = DataLoader(pre_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=n_workers, drop_last=True)
            sup_losses = []
            for xb in loader:
                xb = xb.to(device)
                with torch.no_grad():
                    H = E(xb)                               # (B, T, latent_dim)
                opt_S.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    S_out = S(H)                                # (B, T, latent_dim)
                    # S_out[t] predicts H[t+k] where k = supervisor_steps (1 or 2).
                    # 2-step forces longer temporal context (SeriesGAN, BigData 2024).
                    k = cfg.supervisor_steps
                    loss_sup = nn.functional.mse_loss(S_out[:, :-k, :], H[:, k:, :])
                scaler.scale(loss_sup).backward()
                scaler.step(opt_S)
                scaler.update()
                sup_losses.append(loss_sup.item())
            sup_mean = sum(sup_losses) / len(sup_losses)
            if (pre_ep + 1) % 10 == 0 or pre_ep == 0:
                print(f"  Sup pretrain {pre_ep+1:3d}/{cfg.pretrain_sup_epochs}  "
                      f"sup={sup_mean:.5f}", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2.5: Generator warm-up via supervisor  (latent AE mode only)
    #
    # Before introducing the critic, bootstrap G into the real latent space
    # using only the supervisor consistency loss.  Since S was trained on real
    # latent sequences, minimising MSE(G(z)[:,1:,:], S(G(z))[:,:-1,:]) pushes
    # G's output to obey the same autoregressive dynamics as real data.
    # Without this step the generator starts at ~0.5 everywhere (Sigmoid of
    # small random weights) while E(X_real) has rich structure — the critic
    # saturates immediately and provides no useful gradient signal.
    # (TimeGAN §3.3 "joint training initialisation")
    # -----------------------------------------------------------------------
    if latent_ae and start_epoch == 0 and cfg.pretrain_g_epochs > 0:
        print(f"\n--- Phase 2.5: Generator warm-up ({cfg.pretrain_g_epochs} epochs) ---")
        S.eval()   # supervisor is frozen; only G is updated
        for pre_ep in range(cfg.pretrain_g_epochs):
            G.train()
            if multifile:
                ep_files = random.sample(all_files, min(cfg.files_per_epoch, len(all_files)))
                pre_ds, _ = _load_epoch_dataset(
                    ep_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep)
            else:
                pre_ds = train_ds
            if pre_ds is None or len(pre_ds) == 0:
                continue
            loader = DataLoader(pre_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=n_workers, drop_last=True)
            g_sup_losses = []
            for xb in loader:
                B_pre = xb.size(0)
                z_g = torch.randn(B_pre, cfg.noise_dim, device=device)
                z_l = torch.randn(B_pre, cfg.timestep, cfg.noise_dim, device=device)
                opt_G.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    H_fake = G(z_g, z_l)
                    with torch.no_grad():
                        S_out = S(H_fake)
                    # Consistency: G(z)[t+k] ≈ S(G(z))[t]; k = supervisor_steps
                    k = cfg.supervisor_steps
                    loss_g_sup = nn.functional.mse_loss(H_fake[:, k:, :], S_out[:, :-k, :])
                scaler.scale(loss_g_sup).backward()
                scaler.step(opt_G)
                scaler.update()
                g_sup_losses.append(loss_g_sup.item())
            g_sup_mean = sum(g_sup_losses) / len(g_sup_losses)
            if (pre_ep + 1) % 10 == 0 or pre_ep == 0:
                print(f"  G warm-up  {pre_ep+1:3d}/{cfg.pretrain_g_epochs}  "
                      f"sup={g_sup_mean:.5f}", flush=True)

        print(f"\n--- Phase 3: Joint GAN training ({cfg.epochs} epochs) ---")

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        G.train(); C.train()
        t0 = time.time()
        c_losses, g_losses, w_dists = [], [], []

        # In multi-file mode: resample a fresh random subset of files each epoch.
        # This is the key mechanism behind multi-corpus generalisation: rather than
        # training on the full dataset simultaneously (which would require holding
        # 378 files × 15K records = 5M+ windows in memory), we train on a random
        # 8-file sample per epoch.  Over 300 epochs the model sees ~2400 file-epochs
        # across 378 files — approximately 6 passes through the full corpus, enough
        # for the model to encounter the diversity of tenant behaviours, burst regimes,
        # and access patterns without overfitting to any single file's particularities.
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

            # In latent AE mode the critic operates on latent sequences.
            # Encode real features once per batch; reuse across critic steps.
            if latent_ae:
                with torch.no_grad():
                    H_real = E(real_batch)              # (B, T, latent_dim)
            else:
                H_real = real_batch

            # --- Critic steps (n_critic per generator step) ---
            for _ in range(cfg.n_critic):
                z_g = torch.randn(B, cfg.noise_dim, device=device)
                z_l = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                H_fake = G(z_g, z_l).detach()

                opt_C.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    if cfg.loss in ("wgan-sn", "wgan-gp"):
                        # Separate W estimate from penalty terms so the log shows
                        # the true Wasserstein distance (W > 0 = critic winning;
                        # oscillating W = GAN cycling).
                        c_base = (C(H_fake) - C(H_real)).mean()
                        w_dists.append(-c_base.item())   # W = E[C(real)] - E[C(fake)]
                        c_loss = c_base
                        if cfg.loss == "wgan-gp":
                            # Gradient penalty: enforce 1-Lipschitz on all critic
                            # weights by penalising |∇C(x̂)| ≠ 1 at interpolations.
                            # Requires create_graph=True through the LSTM; CUDA only
                            # (MPS falls back to wgan-sn at device selection time).
                            eps = torch.rand(B, 1, 1, device=device)
                            x_hat = (eps * H_real.detach() +
                                     (1 - eps) * H_fake).requires_grad_(True)
                            d_hat = C(x_hat)
                            grads = torch.autograd.grad(
                                outputs=d_hat.sum(), inputs=x_hat,
                                create_graph=True)[0]           # (B, T, latent_dim)
                            gp = ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()
                            c_loss = c_loss + cfg.gp_lambda * gp
                        # R1 regularisation (R3GAN, NeurIPS 2024): zero-centered GP on
                        # real samples. Penalises |∇C(x_real)|² → 0, preventing the
                        # critic from memorising real data via large local gradients.
                        # Effective under wgan-sn too (not restricted to wgan-gp).
                        if cfg.r1_lambda > 0:
                            H_real_r1 = H_real.detach().clone().requires_grad_(True)
                            d_real_r1 = C(H_real_r1)
                            grads_r1 = torch.autograd.grad(
                                outputs=d_real_r1.sum(), inputs=H_real_r1,
                                create_graph=True)[0]
                            r1 = (grads_r1.norm(2, dim=(1, 2)) ** 2).mean()
                            c_loss = c_loss + cfg.r1_lambda * 0.5 * r1
                        # R2 regularisation (R3GAN, NeurIPS 2024): zero-centered GP on
                        # fake samples. Complements WGAN-GP's 1-centered interpolated-
                        # point penalty — together they provide full convergence guarantees.
                        if cfg.r2_lambda > 0:
                            H_fake_r2 = H_fake.clone().requires_grad_(True)
                            d_fake_r2 = C(H_fake_r2)
                            grads_r2 = torch.autograd.grad(
                                outputs=d_fake_r2.sum(), inputs=H_fake_r2,
                                create_graph=True)[0]
                            r2 = (grads_r2.norm(2, dim=(1, 2)) ** 2).mean()
                            c_loss = c_loss + cfg.r2_lambda * r2
                    else:  # bce
                        real_labels = torch.ones(B, 1, device=device)
                        fake_labels = torch.zeros(B, 1, device=device)
                        c_loss = (bce(C(H_real), real_labels) +
                                  bce(C(H_fake), fake_labels))
                scaler.scale(c_loss).backward()
                if cfg.grad_clip > 0:
                    # Gradient clipping is the primary Lipschitz constraint on
                    # the LSTM weights, which spectral norm does not reach.
                    # Without it, the critic's unconstrained LSTM diverged in v12:
                    # C loss drifted from -10 to -31 over 85 epochs as LSTM
                    # weights grew monotonically.  Clip at 1.0 (Pascanu et al.
                    # 2013 recommendation for RNNs) prevents this without
                    # requiring gradient penalty computation.
                    scaler.unscale_(opt_C)
                    nn.utils.clip_grad_norm_(C.parameters(), cfg.grad_clip)
                scaler.step(opt_C)
                scaler.update()
                c_losses.append(c_loss.item())

            # --- Generator step ---
            z_g = torch.randn(B, cfg.noise_dim, device=device)
            z_l = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)

            opt_G.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                H_fake = G(z_g, z_l)

                if cfg.loss in ("wgan-sn", "wgan-gp"):
                    g_score, feat_fake = C(H_fake, return_features=True)
                    g_loss = -g_score.mean()
                else:  # bce
                    real_labels = torch.ones(B, 1, device=device)
                    g_score, feat_fake = C(H_fake, return_features=True)
                    g_loss = bce(g_score, real_labels)

                # Feature matching (Salimans et al. 2016): match the mean of the
                # critic's internal representation for real vs fake.  Penalising
                # the L2 distance between batch-mean features forces the generator
                # to cover all modes the critic can "see", fixing mode collapse.
                if cfg.feature_matching_weight > 0:
                    with torch.no_grad():
                        _, feat_real = C(H_real, return_features=True)
                    loss_fm = nn.functional.mse_loss(
                        feat_fake.mean(0), feat_real.mean(0))
                    g_loss = g_loss + cfg.feature_matching_weight * loss_fm

                if latent_ae:
                    # Supervisor consistency: S(H_fake)[t] should predict H_fake[t+1].
                    # This forces the generator to produce temporally coherent latents
                    # that obey the same autoregressive dynamics as real data.
                    S_on_fake = S(H_fake)
                    k = cfg.supervisor_steps
                    loss_sup = nn.functional.mse_loss(
                        S_on_fake[:, :-k, :], H_fake[:, k:, :])
                    g_loss = g_loss + cfg.supervisor_loss_weight * loss_sup

                    # Decode generated latents to feature space for auxiliary losses
                    fake_decoded = R(H_fake)
                else:
                    fake_decoded = H_fake

                # --- Auxiliary losses (in feature space) ---
                # Moment matching (L_V): penalise differences in per-feature mean
                # and std so the generator cannot ignore distributional shape.
                if cfg.moment_loss_weight > 0:
                    loss_V = (
                        nn.functional.l1_loss(fake_decoded.mean(0), real_batch.mean(0)) +
                        nn.functional.l1_loss(fake_decoded.std(0),  real_batch.std(0))
                    )
                    g_loss = g_loss + cfg.moment_loss_weight * loss_V

                # Fourier spectral loss (L_FFT): penalise differences in frequency
                # content along the time axis — captures burst periodicity.
                # FIDE (NeurIPS 2024): upweight bins where the real spectrum has high
                # amplitude so rare/extreme I/O events (burst opcodes, large obj_size)
                # are not washed out by the flat MSE average.
                if cfg.fft_loss_weight > 0:
                    real_fft = torch.fft.rfft(real_batch.float(), dim=1).abs()
                    fake_fft = torch.fft.rfft(fake_decoded.float(), dim=1).abs()
                    if cfg.fide_alpha > 0:
                        weight_f = 1.0 + cfg.fide_alpha * (
                            real_fft / (real_fft.mean(dim=1, keepdim=True) + 1e-8))
                        loss_fft = (weight_f * (fake_fft - real_fft).pow(2)).mean()
                    else:
                        loss_fft = nn.functional.mse_loss(fake_fft, real_fft)
                    g_loss = g_loss + cfg.fft_loss_weight * loss_fft

                # Quantile matching (L_Q): penalise per-feature distribution mismatch
                # across the full range (p1…p99).  We need BOTH tails:
                #   Upper tail (p90-p99): rare long-IAT quiet periods and large I/O
                #     bursts that moment loss misses because zero-IAT mass dominates.
                #   Lower tail (p1-p10): near-zero obj_id deltas = repeat/sequential
                #     access.  Without lower quantiles, the model ignores object reuse
                #     (generates random seeks instead of the 40-50% repeat-access rate
                #     seen in real block storage).  p5 on obj_id_delta in signed-log
                #     space corresponds to deltas of a few blocks — exactly the
                #     sequential-stride pattern that drives cache hit rate.
                if cfg.quantile_loss_weight > 0:
                    q_levels = torch.tensor(
                        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
                        device=device,
                    )
                    fd_flat = fake_decoded.reshape(-1, fake_decoded.shape[-1])
                    rb_flat = real_batch.reshape(-1, real_batch.shape[-1])
                    with torch.no_grad():
                        q_real = torch.quantile(rb_flat, q_levels, dim=0)  # (9, d)
                    q_fake = torch.quantile(fd_flat, q_levels, dim=0)
                    loss_Q = nn.functional.mse_loss(q_fake, q_real)
                    g_loss = g_loss + cfg.quantile_loss_weight * loss_Q

                # Autocorrelation matching (L_ACF): penalise per-feature lag-1..5 ACF
                # mismatch between real and generated sequences.
                #
                # Why this matters: the supervisor loss enforces 1-step temporal
                # consistency (h_t → h_{t+1}) in latent space.  The FFT loss captures
                # the power spectrum.  Neither directly penalises multi-lag temporal
                # correlation structure, which is exactly what the DMD-GEN metric
                # (Grassmannian distance of dominant DMD subspaces) measures.
                # On v9/best.pt: DMD-GEN=0.674 despite MMD²=0.010 — the model learns
                # good marginal distributions but wrong sequential law.
                #
                # ACF at lag k for feature j: E[(x_t - μ)(x_{t+k} - μ)] / σ²
                # We use the unnormalised version (just the cross-product mean) since
                # the sequences are already in [-1, 1] and normalisation by σ² would
                # require computing it over the batch, adding noise to the gradient.
                # Mean is subtracted per-feature per-sequence to remove DC offset.
                if cfg.acf_loss_weight > 0:
                    # Centre each sequence so the cross-product is a pure correlation
                    # (not contaminated by mean level differences).
                    fd_c = fake_decoded - fake_decoded.mean(dim=1, keepdim=True)
                    rb_c = real_batch   - real_batch.mean(dim=1, keepdim=True)
                    acf_loss = torch.zeros(1, device=device)
                    for lag in range(1, 6):      # lags 1..5
                        fake_acf = (fd_c[:, :-lag, :] * fd_c[:, lag:, :]).mean(dim=1)
                        with torch.no_grad():
                            real_acf = (rb_c[:, :-lag, :] * rb_c[:, lag:, :]).mean(dim=1)
                        acf_loss = acf_loss + nn.functional.mse_loss(fake_acf, real_acf)
                    g_loss = g_loss + cfg.acf_loss_weight * (acf_loss / 5)

                # L_cov: lag-1 cross-feature covariance loss.
                #
                # Why this directly targets DMD-GEN:
                # Dynamic Mode Decomposition estimates a linear operator
                # A ≈ C · Σ⁻¹ where:
                #   C  = E[x_{t+1} ⊗ x_t]  (d×d lag-1 cross-covariance)
                #   Σ  = E[x_t ⊗ x_t]       (d×d auto-covariance)
                # DMD-GEN measures how far the dominant eigenvectors of A_real
                # are from A_fake on the Grassmann manifold.
                # All existing losses match per-feature statistics (L_V: diag of Σ;
                # L_ACF: diagonal of C).  None match the off-diagonal cross-feature
                # entries — e.g. how obj_size at time t predicts opcode at time t+1.
                # Matching the full d×d C matrix directly constraints A, which is
                # exactly what DMD-GEN measures.
                if cfg.cross_cov_loss_weight > 0:
                    def _lag1_cross_cov(X: torch.Tensor) -> torch.Tensor:
                        # X: (B, T, d) → (d, d) cross-covariance matrix
                        X0 = X[:, :-1, :]   # states at time t
                        X1 = X[:, 1:, :]    # states at time t+1
                        # Centre over batch and time to remove mean bias
                        X0c = X0 - X0.mean(dim=(0, 1), keepdim=True)
                        X1c = X1 - X1.mean(dim=(0, 1), keepdim=True)
                        N = X.shape[0] * (X.shape[1] - 1)
                        return torch.einsum("bti,btj->ij", X0c, X1c) / N  # (d, d)

                    fake_cov = _lag1_cross_cov(fake_decoded)
                    with torch.no_grad():
                        real_cov = _lag1_cross_cov(real_batch)
                    loss_cov = nn.functional.mse_loss(fake_cov, real_cov)
                    g_loss = g_loss + cfg.cross_cov_loss_weight * loss_cov

                # L_loc: object reuse rate matching.
                # In real block I/O ~40-50% of requests hit an object seen earlier
                # in the same short window; models trained only on Wasserstein loss
                # generate <1% reuse.  For each position t we measure the minimum
                # distance (in normalised obj_id space) to every earlier position
                # in the same window; a soft sigmoid gate at eps≈0 converts that
                # into a differentiable "reuse" indicator.  The loss penalises the
                # squared gap between generated and real mean reuse rates.
                if cfg.locality_loss_weight > 0:
                    # L_loc: stride-repetition rate matching.
                    # Measures the fraction of timesteps whose delta-encoded obj_id value
                    # matches a prior value in the same window — a proxy for sequential-
                    # access stride repetition.  (obj_id is delta-encoded + signed-log, so
                    # we cannot recover absolute identities; instead we match the statistical
                    # property that the same stride / seek-pattern recurs within a window.)
                    # Real Tencent Block data: ~20-25% stride repetition rate.
                    # Generator without L_loc: ~0% (pure random-seeking output).
                    obj_fake = fake_decoded[:, :, obj_id_col]    # (B, T)
                    obj_real = real_batch[:, :, obj_id_col]

                    def _soft_reuse_rate(obj_seq: torch.Tensor) -> torch.Tensor:
                        B, T = obj_seq.shape
                        # (B, T, T) pairwise absolute distances between delta values
                        diff = (obj_seq.unsqueeze(2) - obj_seq.unsqueeze(1)).abs()
                        # Only compare position t to EARLIER positions s < t
                        causal = torch.tril(
                            torch.ones(T, T, device=obj_seq.device), diagonal=-1
                        )
                        diff = diff + (1.0 - causal) * 1e6
                        min_dist = diff.min(dim=2).values  # (B, T): nearest prior match
                        # Soft indicator: sigmoid centred at eps with moderate slope.
                        # eps=0.04 (2% of [-1,1] range) catches near-exact repeats.
                        eps = 0.04
                        reuse = torch.sigmoid((eps - min_dist[:, 1:]) / (eps * 0.2))
                        return reuse.mean()

                    fake_reuse = _soft_reuse_rate(obj_fake)
                    with torch.no_grad():
                        real_reuse = _soft_reuse_rate(obj_real)
                    loss_loc = (fake_reuse - real_reuse).pow(2)
                    g_loss = g_loss + cfg.locality_loss_weight * loss_loc

                # L_div: MSGAN mode-seeking diversity loss.
                # For two independently sampled noise pairs, maximise the ratio of
                # output distance to input distance — directly combats mode collapse
                # (when the generator maps very different noise to the same output).
                # Reference: Mao et al., CVPR 2019, "Mode Seeking GAN".
                # Set diversity_loss_weight > 0 (try 0.5–2.0) when β-recall < 0.5.
                if cfg.diversity_loss_weight > 0:
                    B2 = B // 2
                    z_g1 = torch.randn(B2, cfg.noise_dim, device=device)
                    z_l1 = torch.randn(B2, cfg.timestep, cfg.noise_dim, device=device)
                    z_g2 = torch.randn(B2, cfg.noise_dim, device=device)
                    z_l2 = torch.randn(B2, cfg.timestep, cfg.noise_dim, device=device)
                    f1 = G(z_g1, z_l1)   # (B2, T, out_dim)
                    f2 = G(z_g2, z_l2)
                    # L2 distance in noise and output spaces
                    z_dist = (
                        (z_g1 - z_g2).pow(2).sum(-1) +
                        (z_l1 - z_l2).pow(2).sum(dim=(-1, -2))
                    ).sqrt().clamp(min=1e-8)     # (B2,)
                    x_dist = (f1 - f2).pow(2).sum(dim=(-1, -2)).sqrt()  # (B2,)
                    # Negative because we want to MAXIMISE diversity
                    loss_div = -(x_dist / z_dist).mean()
                    g_loss = g_loss + cfg.diversity_loss_weight * loss_div

            scaler.scale(g_loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(opt_G)
                nn.utils.clip_grad_norm_(G.parameters(), cfg.grad_clip)
            scaler.step(opt_G)
            scaler.update()
            g_losses.append(g_loss.item())

            # EMA update: blend live weights toward EMA state.
            # ema_G_state lives on the same device as G (deepcopy of G.state_dict()).
            # Use v.detach() — NOT v.cpu() — to avoid a device mismatch on MPS.
            with torch.no_grad():
                for k, v in G.state_dict().items():
                    ema_G_state[k].mul_(ema_decay).add_(v.detach(), alpha=1.0 - ema_decay)

            # --- Encoder + Recovery joint step (latent AE mode only) ---
            # Fine-tunes the autoencoder during GAN training to keep the latent
            # space consistent with the generator's output distribution.
            if latent_ae:
                opt_ER.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    H_real_grad = E(real_batch)
                    X_hat = R(H_real_grad)
                    # Reconstruction loss only: supervisor (S) is frozen after
                    # pretraining and has no optimizer in Phase 3.  The previous
                    # supervisor term here was a no-op — both its input
                    # (H_real_grad.detach()) and target were detached, so no
                    # gradient flowed and no parameter in opt_ER was updated by it.
                    # (Per peer review: freeze S after pretraining rather than
                    # silently running a dead term.)
                    er_loss = nn.functional.mse_loss(X_hat, real_batch)
                scaler.scale(er_loss).backward()
                scaler.step(opt_ER)
                scaler.update()

        c_mean = sum(c_losses) / len(c_losses)
        g_mean = sum(g_losses) / len(g_losses)
        w_mean = sum(w_dists)  / len(w_dists) if w_dists else float("nan")
        elapsed = time.time() - t0

        n_files_str = (f"  files={len(epoch_files)}" if multifile else
                       f"  windows={len(train_ds):,}")
        lr_now = opt_G.param_groups[0]["lr"]
        log = (f"Epoch {epoch+1:4d}/{cfg.epochs}  "
               f"W={w_mean:+.4f}  C={c_mean:.4f}  G={g_mean:.4f}  "
               f"lr={lr_now:.2e}  t={elapsed:.1f}s"
               f"{n_files_str}")

        if val_tensor is not None and (epoch + 1) % cfg.mmd_every == 0:
            # Evaluate using the EMA model, not the live G.
            # Save live weights first, then swap in EMA, then restore.
            live_G_state = copy.deepcopy(G.state_dict())
            G.load_state_dict({k: v.to(device) for k, v in ema_G_state.items()})
            mmd_val, recall_val, combined_val = evaluate_metrics(
                G, val_tensor, cfg.mmd_samples, cfg.timestep, device,
                recovery=R if latent_ae else None,
            )
            G.load_state_dict(live_G_state)   # restore live weights for training
            mmd_history.append((epoch, mmd_val, recall_val, combined_val))
            log += f"  EMA MMD²={mmd_val:.5f}  recall={recall_val:.3f}  comb={combined_val:.5f}"
            # Save best.pt on combined score (MMD² + 0.2*(1-recall)) to avoid
            # saving high-MMD²/high-recall checkpoints over lower-MMD²/mode-collapse ones.
            if combined_val < best_combined:
                best_combined     = combined_val
                epochs_no_improve = 0
                ckpt_data = {
                    "epoch": epoch,
                    "G": live_G_state,         # live weights (for resuming training)
                    "G_ema": ema_G_state,       # EMA weights (for inference/generation)
                    "C": C.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_C": opt_C.state_dict(),
                    "prep": prep,
                    "config": cfg,
                    "mmd": mmd_val,
                    "recall": recall_val,
                    "combined": combined_val,
                    "mmd_history": mmd_history,
                }
                if latent_ae:
                    ckpt_data.update({"E": E.state_dict(), "R": R.state_dict(),
                                      "S": S.state_dict()})
                torch.save(ckpt_data, ckpt_dir / "best.pt")
                log += "  ★"
            else:
                epochs_no_improve += 1
            G.train()
            if latent_ae:
                E.train(); R.train(); S.train()

        print(log, flush=True)

        # Early stopping: halt when combined score has not improved for
        # patience × mmd_every epochs.  0 = disabled.
        if (cfg.early_stop_patience > 0
                and epochs_no_improve >= cfg.early_stop_patience):
            print(f"Early stop: no improvement for "
                  f"{epochs_no_improve * cfg.mmd_every} epochs "
                  f"(patience={cfg.early_stop_patience} × mmd_every={cfg.mmd_every}).",
                  flush=True)
            break

        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            ckpt_data = {
                "epoch": epoch,
                "G": G.state_dict(),
                "G_ema": ema_G_state,
                "C": C.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_C": opt_C.state_dict(),
                "prep": prep,
                "config": cfg,
            }
            if sched_G is not None:
                ckpt_data["sched_G"] = sched_G.state_dict()
                ckpt_data["sched_C"] = sched_C.state_dict()
            if latent_ae:
                ckpt_data.update({"E": E.state_dict(), "R": R.state_dict(),
                                  "S": S.state_dict()})
            ckpt_data["mmd_history"] = mmd_history
            torch.save(ckpt_data, ckpt_path)
            print(f"  → saved {ckpt_path}")

        # Step cosine scheduler at end of epoch (after checkpoint) so the LR
        # in the saved opt state matches the epoch that was just trained.
        if sched_G is not None:
            sched_G.step()
            sched_C.step()

    final_path = ckpt_dir / "final.pt"
    final_data = {
        "epoch": cfg.epochs - 1,
        "G": G.state_dict(),
        "C": C.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_C": opt_C.state_dict(),
        "prep": prep,
        "config": cfg,
    }
    if latent_ae:
        final_data.update({"E": E.state_dict(), "R": R.state_dict(),
                           "S": S.state_dict()})
    torch.save(final_data, final_path)
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
    p.add_argument("--loss",             default="wgan-sn",
                   choices=["wgan-sn", "wgan-gp", "bce"],
                   help="wgan-sn: Wasserstein + spectral norm; "
                        "wgan-gp: Wasserstein + gradient penalty (true Lipschitz, "
                        "works on MPS); bce: original GAN cross-entropy")
    p.add_argument("--gp-lambda",        type=float, default=10.0,
                   help="Gradient penalty coefficient for wgan-gp (standard: 10)")
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--timestep",         type=int,   default=12)
    p.add_argument("--noise-dim",        type=int,   default=10)
    p.add_argument("--hidden-size",      type=int,   default=256)
    p.add_argument("--compile",          action="store_true", default=False,
                   help="torch.compile models for ~20-40%% CUDA speedup "
                        "(CUDA only; incompatible with wgan-gp/r1/r2)")
    p.add_argument("--no-minibatch-std", action="store_true", default=False,
                   help="Disable minibatch std channel in critic (on by default)")
    p.add_argument("--no-sn-lstm",       action="store_true", default=False,
                   help="Disable spectral norm on critic LSTM weight matrices "
                        "(on by default; prevents W-distance drift and mode collapse)")
    p.add_argument("--patch-embed",      action="store_true", default=False,
                   help="Conv1d patch embedding before critic LSTM: folds 12-step window "
                        "into 4 patch tokens (kernel=stride=3). TTS-GAN style. "
                        "Gives critic local-pattern inductive bias for burst detection.")
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
    p.add_argument("--resume-from",      default=None,
                   help="Path to a specific checkpoint .pt file to resume from "
                        "(overrides auto-detect of latest epoch_NNNN.pt in checkpoint-dir)")
    p.add_argument("--reset-optimizer",  action="store_true",
                   help="Load model weights from checkpoint but start with fresh "
                        "optimizers and schedulers at the current --lr-g/--lr-d values. "
                        "Use with --resume-from to hot-start after a collapse.")
    p.add_argument("--checkpoint-every", type=int,   default=10)
    p.add_argument("--mmd-every",        type=int,   default=5)
    p.add_argument("--mmd-samples",      type=int,   default=1000)
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop after N consecutive MMD evals without improvement (0=off)")
    p.add_argument("--train-split",      type=float, default=0.8)
    p.add_argument("--moment-loss-weight", type=float, default=0.1,
                   help="Weight for per-feature moment matching loss (0 = off)")
    p.add_argument("--fft-loss-weight",    type=float, default=0.05,
                   help="Weight for Fourier spectral loss (0 = off)")
    p.add_argument("--quantile-loss-weight", type=float, default=0.2,
                   help="Weight for per-feature quantile matching at p50/90/95/99 (0 = off); "
                        "directly fixes tail/burst IAT mismatch that moment loss misses")
    p.add_argument("--acf-loss-weight",    type=float, default=0.1,
                   help="Weight for lag-1..5 autocorrelation matching loss (0 = off). "
                        "Targets DMD-GEN failure: supervisor loss only enforces 1-step "
                        "prediction; ACF loss directly penalises multi-lag temporal structure.")
    p.add_argument("--cross-cov-loss-weight", type=float, default=0.0,
                   help="Weight for lag-1 cross-feature covariance loss (0 = off; try 0.5–2.0). "
                        "Matches the full d×d matrix E[x_{t,i}·x_{t+1,j}] — the linear dynamics "
                        "operator that DMD-GEN estimates.  ACF only matches the diagonal; this "
                        "adds off-diagonal cross-feature terms, directly targeting DMD-GEN.")
    p.add_argument("--fide-alpha",         type=float, default=1.0,
                   help="FIDE frequency inflation weight; 0 = flat FFT loss (NeurIPS 2024)")
    p.add_argument("--feature-matching-weight", type=float, default=1.0,
                   help="Weight for critic feature matching loss (0 = off; "
                        "fixes mode collapse by matching critic internal features)")
    p.add_argument("--grad-clip",              type=float, default=1.0,
                   help="Gradient norm clip for G and C (0 = off). Prevents "
                        "critic from dominating when LSTM weights are unconstrained.")
    p.add_argument("--ema-decay",              type=float, default=0.999,
                   help="EMA decay for generator weights (default 0.999). "
                        "EMA model is used for evaluation and generation; "
                        "smooths GAN oscillations. 0 = use live weights.")
    p.add_argument("--lr-cosine-decay",        type=float, default=0.05,
                   help="Cosine annealing: eta_min = lr * this factor over training "
                        "(default 0.05 = decay to 5%% of initial LR). "
                        "Damps GAN oscillations in the second half of training "
                        "where G and C are near equilibrium. 0 = constant LR.")
    # Latent AE + supervisor
    p.add_argument("--latent-dim",           type=int,   default=24,
                   help="Latent space dim for AE+supervisor mode; 0 = legacy direct mode")
    p.add_argument("--pretrain-ae-epochs",   type=int,   default=50,
                   help="Phase 1: autoencoder pretraining epochs")
    p.add_argument("--pretrain-sup-epochs",  type=int,   default=50,
                   help="Phase 2: supervisor pretraining epochs")
    p.add_argument("--pretrain-g-epochs",    type=int,   default=100,
                   help="Phase 2.5: generator warm-up epochs (TimeGAN supervised init)")
    p.add_argument("--supervisor-loss-weight", type=float, default=10.0,
                   help="η: supervisor consistency weight in joint training")
    p.add_argument("--supervisor-steps",   type=int,   default=1,
                   help="1 = 1-step supervisor; 2 = 2-step (SeriesGAN BigData 2024)")
    p.add_argument("--locality-loss-weight", type=float, default=0.0,
                   help="L_loc: stride-repetition rate matching within windows (0=off; try 1.0). "
                        "Matches the fraction of obj_id deltas that repeat within a window — "
                        "a proxy for sequential-access consistency.")
    p.add_argument("--diversity-loss-weight", type=float, default=0.0,
                   help="L_div: MSGAN mode-seeking loss — maximises output/noise distance ratio "
                        "across random pairs; combats mode collapse (low β-recall). "
                        "Requires a second G forward pass. Try 0.5–2.0.")
    p.add_argument("--r1-lambda",          type=float, default=0.0,
                   help="R1 zero-centered GP on real samples (R3GAN NeurIPS 2024; 0=off)")
    p.add_argument("--r2-lambda",          type=float, default=0.0,
                   help="R2 zero-centered GP on fake samples (R3GAN NeurIPS 2024; 0=off)")
    p.add_argument("--lr-er",                type=float, default=0.0005,
                   help="Learning rate for encoder + recovery")
    p.add_argument("--amp",                  action="store_true", default=False,
                   help="Enable AMP fp16 for 2-3× CUDA speedup (CUDA only; "
                        "auto-disabled with wgan-gp/r1/r2 due to create_graph incompatibility)")
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
    cfg.compile          = args.compile
    cfg.minibatch_std    = not args.no_minibatch_std
    cfg.sn_lstm          = not args.no_sn_lstm
    cfg.patch_embed      = args.patch_embed
    cfg.lr_g             = args.lr_g
    cfg.lr_d             = args.lr_d
    cfg.n_critic         = args.n_critic
    cfg.max_records      = args.max_records
    cfg.files_per_epoch  = args.files_per_epoch
    cfg.records_per_file = args.records_per_file
    cfg.checkpoint_dir   = args.checkpoint_dir
    cfg.resume_from      = args.resume_from
    cfg.reset_optimizer  = args.reset_optimizer
    cfg.checkpoint_every = args.checkpoint_every
    cfg.mmd_every              = args.mmd_every
    cfg.mmd_samples            = args.mmd_samples
    cfg.early_stop_patience    = args.early_stop_patience
    cfg.train_split         = args.train_split
    cfg.moment_loss_weight          = args.moment_loss_weight
    cfg.fft_loss_weight             = args.fft_loss_weight
    cfg.quantile_loss_weight        = args.quantile_loss_weight
    cfg.acf_loss_weight             = args.acf_loss_weight
    cfg.cross_cov_loss_weight       = args.cross_cov_loss_weight
    cfg.fide_alpha                  = args.fide_alpha
    cfg.feature_matching_weight     = args.feature_matching_weight
    cfg.grad_clip                   = args.grad_clip
    cfg.ema_decay                   = args.ema_decay
    cfg.lr_cosine_decay             = args.lr_cosine_decay
    cfg.gp_lambda                   = args.gp_lambda
    cfg.locality_loss_weight        = args.locality_loss_weight
    cfg.diversity_loss_weight       = args.diversity_loss_weight
    cfg.r1_lambda                   = args.r1_lambda
    cfg.r2_lambda                   = args.r2_lambda
    cfg.latent_dim              = args.latent_dim
    cfg.pretrain_ae_epochs      = args.pretrain_ae_epochs
    cfg.pretrain_sup_epochs     = args.pretrain_sup_epochs
    cfg.pretrain_g_epochs       = args.pretrain_g_epochs
    cfg.supervisor_loss_weight  = args.supervisor_loss_weight
    cfg.supervisor_steps        = args.supervisor_steps
    cfg.lr_er                   = args.lr_er
    cfg.amp                     = args.amp
    return cfg


if __name__ == "__main__":
    # MPS backend emits a benign buffer-resize warning on rfft; suppress it.
    warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
    cfg = parse_args()
    train(cfg)
