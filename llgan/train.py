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
        print(f"  log-transformed: {prep._log_cols}")

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

    # WGAN-GP enforces Lipschitz via gradient penalty — spectral norm is
    # redundant and can interfere with the penalty gradient computation.
    use_sn = (cfg.loss != "wgan-gp")
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=cfg.latent_dim if latent_ae else None).to(device)
    C = Critic(critic_input_dim, cfg.hidden_size,
               use_spectral_norm=use_sn).to(device)

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

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    opt_C = torch.optim.Adam(C.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    bce = nn.BCEWithLogitsLoss() if cfg.loss == "bce" else None

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_mmd = float("inf")   # track best MMD² for best.pt

    # Resume
    start_epoch = 0
    latest = sorted(ckpt_dir.glob("epoch_*.pt"))
    if latest:
        ckpt = torch.load(latest[-1], map_location=device)
        G.load_state_dict(ckpt["G"])
        C.load_state_dict(ckpt["C"])
        opt_G.load_state_dict(ckpt["opt_G"])
        opt_C.load_state_dict(ckpt["opt_C"])
        if latent_ae and "E" in ckpt:
            E.load_state_dict(ckpt["E"])
            R.load_state_dict(ckpt["R"])
            S.load_state_dict(ckpt["S"])
        start_epoch = ckpt["epoch"] + 1
        # Restore preprocessor from checkpoint so normalization stays consistent
        if "prep" in ckpt:
            prep = ckpt["prep"]
        print(f"Resumed from {latest[-1]} (epoch {start_epoch})")

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
                loss_ae = nn.functional.mse_loss(R(E(xb)), xb)
                loss_ae.backward()
                opt_AE.step()
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
                S_out = S(H)                                # (B, T, latent_dim)
                # S_out[t] should predict H[t+1]
                loss_sup = nn.functional.mse_loss(S_out[:, :-1, :], H[:, 1:, :])
                loss_sup.backward()
                opt_S.step()
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
                H_fake = G(z_g, z_l)
                with torch.no_grad():
                    S_out = S(H_fake)
                # Consistency: G(z)[t+1] ≈ S(G(z))[t]
                loss_g_sup = nn.functional.mse_loss(H_fake[:, 1:, :], S_out[:, :-1, :])
                opt_G.zero_grad()
                loss_g_sup.backward()
                opt_G.step()
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
                if cfg.loss in ("wgan-sn", "wgan-gp"):
                    c_loss = (C(H_fake) - C(H_real)).mean()
                    if cfg.loss == "wgan-gp":
                        # Gradient penalty: enforce 1-Lipschitz on all critic
                        # weights by penalising |∇C(x̂)| ≠ 1 at interpolations.
                        # Works on MPS (create_graph=True is supported since
                        # PyTorch 2.0) — previously assumed CUDA-only in error.
                        eps = torch.rand(B, 1, 1, device=device)
                        x_hat = (eps * H_real.detach() +
                                 (1 - eps) * H_fake).requires_grad_(True)
                        d_hat = C(x_hat)
                        grads = torch.autograd.grad(
                            outputs=d_hat.sum(), inputs=x_hat,
                            create_graph=True)[0]           # (B, T, latent_dim)
                        gp = ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()
                        c_loss = c_loss + cfg.gp_lambda * gp
                else:  # bce
                    real_labels = torch.ones(B, 1, device=device)
                    fake_labels = torch.zeros(B, 1, device=device)
                    c_loss = (bce(C(H_real), real_labels) +
                              bce(C(H_fake), fake_labels))
                c_loss.backward()
                opt_C.step()
                c_losses.append(c_loss.item())

            # --- Generator step ---
            z_g = torch.randn(B, cfg.noise_dim, device=device)
            z_l = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
            H_fake = G(z_g, z_l)

            opt_G.zero_grad()
            if cfg.loss == "wgan-sn":
                g_score, feat_fake = C(H_fake, return_features=True)
                g_loss = -g_score.mean()
            else:
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
                loss_sup = nn.functional.mse_loss(
                    S_on_fake[:, :-1, :], H_fake[:, 1:, :])
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
            if cfg.fft_loss_weight > 0:
                real_fft = torch.fft.rfft(real_batch.float(), dim=1).abs()
                fake_fft = torch.fft.rfft(fake_decoded.float(), dim=1).abs()
                loss_fft = nn.functional.mse_loss(fake_fft, real_fft)
                g_loss = g_loss + cfg.fft_loss_weight * loss_fft

            g_loss.backward()
            opt_G.step()
            g_losses.append(g_loss.item())

            # --- Encoder + Recovery joint step (latent AE mode only) ---
            # Fine-tunes the autoencoder during GAN training to keep the latent
            # space consistent with the generator's output distribution.
            if latent_ae:
                H_real_grad = E(real_batch)
                X_hat = R(H_real_grad)
                # Reconstruction loss
                loss_ae = nn.functional.mse_loss(X_hat, real_batch)
                # Supervisor guides embedder to produce predictable latents
                S_on_real = S(H_real_grad.detach())
                loss_sup_e = nn.functional.mse_loss(
                    S_on_real[:, :-1, :], H_real_grad[:, 1:, :].detach())
                er_loss = loss_ae + cfg.supervisor_loss_weight * loss_sup_e
                opt_ER.zero_grad()
                er_loss.backward()
                opt_ER.step()

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
                                   cfg.timestep, device,
                                   recovery=R if latent_ae else None)
            log += f"  MMD²={mmd_val:.5f}"
            # Save best.pt whenever MMD² improves — captures the best model
            # regardless of where checkpoint_every lands.
            if mmd_val < best_mmd:
                best_mmd = mmd_val
                ckpt_data = {
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "C": C.state_dict(),
                    "opt_G": opt_G.state_dict(),
                    "opt_C": opt_C.state_dict(),
                    "prep": prep,
                    "config": cfg,
                    "mmd": best_mmd,
                }
                if latent_ae:
                    ckpt_data.update({"E": E.state_dict(), "R": R.state_dict(),
                                      "S": S.state_dict()})
                torch.save(ckpt_data, ckpt_dir / "best.pt")
                log += "  ★"
            G.train()
            if latent_ae:
                E.train(); R.train(); S.train()

        print(log, flush=True)

        if (epoch + 1) % cfg.checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch+1:04d}.pt"
            ckpt_data = {
                "epoch": epoch,
                "G": G.state_dict(),
                "C": C.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_C": opt_C.state_dict(),
                "prep": prep,
                "config": cfg,
            }
            if latent_ae:
                ckpt_data.update({"E": E.state_dict(), "R": R.state_dict(),
                                  "S": S.state_dict()})
            torch.save(ckpt_data, ckpt_path)
            print(f"  → saved {ckpt_path}")

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
    p.add_argument("--moment-loss-weight", type=float, default=0.1,
                   help="Weight for per-feature moment matching loss (0 = off)")
    p.add_argument("--fft-loss-weight",    type=float, default=0.05,
                   help="Weight for Fourier spectral loss (0 = off)")
    p.add_argument("--feature-matching-weight", type=float, default=1.0,
                   help="Weight for critic feature matching loss (0 = off; "
                        "fixes mode collapse by matching critic internal features)")
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
    p.add_argument("--lr-er",                type=float, default=0.0005,
                   help="Learning rate for encoder + recovery")
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
    cfg.mmd_every           = args.mmd_every
    cfg.mmd_samples         = args.mmd_samples
    cfg.train_split         = args.train_split
    cfg.moment_loss_weight          = args.moment_loss_weight
    cfg.fft_loss_weight             = args.fft_loss_weight
    cfg.feature_matching_weight     = args.feature_matching_weight
    cfg.gp_lambda                   = args.gp_lambda
    cfg.latent_dim              = args.latent_dim
    cfg.pretrain_ae_epochs      = args.pretrain_ae_epochs
    cfg.pretrain_sup_epochs     = args.pretrain_sup_epochs
    cfg.pretrain_g_epochs       = args.pretrain_g_epochs
    cfg.supervisor_loss_weight  = args.supervisor_loss_weight
    cfg.lr_er                   = args.lr_er
    return cfg


if __name__ == "__main__":
    # MPS backend emits a benign buffer-resize warning on rfft; suppress it.
    warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
    cfg = parse_args()
    train(cfg)
