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
import math
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
from dataset import (load_trace, TracePreprocessor, TraceDataset, _READERS,
                     compute_window_descriptors, load_file_characterizations,
                     _OPCODE_COLS)
from model import Generator, Critic, MultiScaleCritic, Encoder, Recovery, Supervisor, LatentDiscriminator
from mmd import evaluate_mmd, evaluate_metrics


# ---------------------------------------------------------------------------
# PacGAN packing helper
# ---------------------------------------------------------------------------

def _pack_windows(
    x: torch.Tensor,
    pack_size: int,
    cond: Optional[torch.Tensor] = None,
) -> tuple:
    """Pack consecutive windows for PacGAN-style critics (Lin et al. NeurIPS 2018).

    Packing makes mode collapse detectable: fake packs of m windows look nearly
    identical (low intra-pack variance), while real packs are diverse.
    Complements minibatch_std with an explicit sequence-level diversity signal.

    x   : (B, T, D) → (B//m, m*T, D)
    cond: (B, C) → (B//m, C)  — take first of each pack for projection critic
    Returns (x_packed, cond_packed).
    """
    if pack_size <= 1:
        return x, cond
    B, T, D = x.shape
    n = (B // pack_size) * pack_size
    x_packed = x[:n].reshape(n // pack_size, T * pack_size, D)
    cond_packed = cond[:n:pack_size] if cond is not None else None
    return x_packed, cond_packed


# ---------------------------------------------------------------------------
# Conditioning helper
# ---------------------------------------------------------------------------

def _make_z_global(
    B: int, cfg, device: torch.device,
    real_features: Optional[torch.Tensor] = None,
    file_cond: Optional[torch.Tensor] = None,
    training: bool = True,
    G=None,
):
    """Build z_global, optionally prepending workload descriptors.

    Returns (z_global, kl_loss) where kl_loss is zero unless variational
    conditioning is active.

    Classifier-free guidance (CFG): when cfg.cond_drop_prob > 0 and training,
    randomly replace descriptor vectors with zeros with the given probability.

    Variational conditioning (IDEAS.md #3): when G.cond_encoder is set, the
    char-file vector is passed through a learned N(μ,σ²) encoder at training
    time.  At eval (training=False), the deterministic mean μ is used.  The
    returned kl_loss should be added to the generator's loss (scaled by
    cfg.var_cond_kl_weight).

    Args:
        B: batch size
        cfg: Config with noise_dim, cond_dim, cond_drop_prob, etc.
        device: target device
        real_features: (B, T, num_cols) fallback when file_cond is None.
        file_cond: (B, cond_dim) precharacterized per-file conditioning vectors.
        training: if True, apply CFG dropout and variational sampling.
        G: Generator instance; used for GMM prior and cond_encoder.

    Returns:
        z_global:  (B, noise_dim) or (B, cond_dim + noise_dim)
        kl_loss:   scalar tensor; 0 when var_cond is off
    """
    kl_loss = torch.zeros((), device=device)
    if cfg.cond_dim > 0:
        if file_cond is not None:
            cond = file_cond.to(device)
        else:
            cond = compute_window_descriptors(real_features)
        # Variational conditioning: perturb cond at training time, use μ at eval
        if G is not None and getattr(G, "cond_encoder", None) is not None:
            cond, kl_loss = G.cond_encoder(cond, training=training)
        # Regime sampler (IDEAS.md #5): map cond → discrete regime code via
        # Gumbel-Softmax.  Replaces raw cond with a regime prototype vector,
        # committing G to one workload type before generation.  Applied BEFORE
        # CFG dropout so the regime selection is stable.
        if G is not None and getattr(G, "regime_sampler", None) is not None:
            cond = G.regime_sampler(cond)
        # GMM prior: sample noise from conditioning-aware mixture.  Uses cond
        # before CFG dropout so workload→noise mapping is stable.
        if G is not None and getattr(G, "gmm_prior", None) is not None:
            noise = G.sample_noise(B, device, cond=cond)
        else:
            noise = torch.randn(B, cfg.noise_dim, device=device)
        # Classifier-free guidance: randomly drop conditioning during training
        drop_prob = getattr(cfg, "cond_drop_prob", 0.0)
        if training and drop_prob > 0:
            mask = (torch.rand(B, 1, device=device) > drop_prob).float()
            cond = cond * mask
        return torch.cat([cond, noise], dim=1), kl_loss
    return torch.randn(B, cfg.noise_dim, device=device), kl_loss


# ---------------------------------------------------------------------------
# Multi-file helpers
# ---------------------------------------------------------------------------

def _collect_files(trace_dir: str, fmt: str) -> List[Path]:
    """Return all candidate trace files in a directory."""
    d = Path(trace_dir)
    # Collect everything and filter to likely trace files by suffix
    candidates = [p for p in d.iterdir() if p.is_file() and not p.name.startswith(".")]
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
    obj_size_granularity: int = 0,
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
    prep = TracePreprocessor(obj_size_granularity=obj_size_granularity)
    prep.fit(combined)
    return prep


def _load_epoch_dataset(
    files: List[Path],
    fmt: str,
    records_per_file: int,
    prep: TracePreprocessor,
    timestep: int,
    char_lookup: Optional[dict] = None,
) -> Tuple[Optional[ConcatDataset], Optional[np.ndarray]]:
    """
    Load records_per_file from each file, transform with fitted prep, and
    return a ConcatDataset of per-file TraceDatasets.

    Critically: each file gets its own TraceDataset so sliding windows are
    never formed across file boundaries (last event of file A → first event
    of file B would be causally meaningless and poison a sequence model).

    When char_lookup is provided, each TraceDataset is tagged with the
    precharacterized per-file conditioning vector looked up by filename.
    Files not found in the lookup get a zero vector (unconditional path).
    TraceDatasets then return (window, file_cond) tuples from __getitem__.
    """
    import pandas as pd
    train_datasets = []
    val_arrays = []

    # Determine cond_dim from the lookup (all entries share the same dim)
    cond_dim = next(iter(char_lookup.values())).shape[0] if char_lookup else 0

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

            # Look up precharacterized conditioning vector for this file.
            # Try full basename first, then strip .zst/.gz.
            file_cond = None
            if char_lookup:
                fname = f.name
                file_cond = char_lookup.get(fname)
                if file_cond is None:
                    for ext in (".zst", ".gz"):
                        if fname.endswith(ext):
                            file_cond = char_lookup.get(fname[: -len(ext)])
                            if file_cond is not None:
                                break
                if file_cond is None:
                    # File not characterized: zero vector = unconditional path.
                    # Under CFG training the model already handles this case
                    # (dropout zeros are indistinguishable from unknown files).
                    file_cond = torch.zeros(cond_dim)

            if len(train_arr) > timestep:
                train_datasets.append(
                    TraceDataset(train_arr, timestep, file_cond=file_cond)
                )
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
    # Seed control for reproducibility.  Set before any random operations so
    # model init, file shuffling, and batch ordering are all deterministic.
    seed = getattr(cfg, "seed", -1)
    if seed >= 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"[seed] {seed} — training is deterministic")

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
        # rpgan uses only first-order gradients and is MPS-compatible.
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
    # proj_critic adds a gradient term O(hidden_size) larger than the base
    # WGAN gradient, pushing scaled fp16 gradients above 65504 at the default
    # init_scale=2**16=65536.  Use 2**14=16384 when proj_critic is enabled to
    # give 4× more fp16 headroom before overflow; the scaler will grow back up.
    _amp_init_scale = 2**14 if getattr(cfg, "proj_critic", False) else 2**16
    # Use SEPARATE scalers for critic (C) and generator (G) to prevent
    # scale collapse.  With n_critic > 1, the single-scaler design calls
    # scaler.update() once per critic step, so the scale halves n_critic times
    # per batch during any overflow period.  At 2 critic steps × 250 batches/epoch
    # the scale reaches 0 within 2–3 epochs (2^14 / 2^1500 ≈ 0), after which
    # unscale_() computes grad / 0 = NaN.  Separate scalers give each component
    # its own scale trajectory so G's scaler stays healthy even if C's overflows.
    scaler   = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=_amp_init_scale)  # G and pretrain
    scaler_C = torch.amp.GradScaler("cuda", enabled=use_amp, init_scale=_amp_init_scale)  # critic C only

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
        if getattr(cfg, "block_sample", False):
            all_files.sort()  # temporal order by filename
            print(f"Trace dir: {cfg.trace_dir}  ({len(all_files)} files found, block sampling)")
        else:
            print(f"Trace dir: {cfg.trace_dir}  ({len(all_files)} files found)")

        # Fit preprocessor on a seed set of files (held constant across training)
        n_seed = min(max(cfg.files_per_epoch, 4), len(all_files))
        seed_files = random.sample(all_files, n_seed)
        print(f"Fitting preprocessor on {n_seed} seed files …")
        prep = _fit_prep_on_files(seed_files, cfg.trace_format, cfg.records_per_file,
                                   obj_size_granularity=cfg.obj_size_granularity)
        print(f"  columns ({prep.num_cols}): {prep.col_names}")
        print(f"  delta-encoded: {prep._delta_cols}")
        print(f"  locality-split: {prep._obj_locality_cols}")
        print(f"  log-transformed: {prep._log_cols}")
        if getattr(prep, "_dropped_cols", []):
            print(f"  auto-dropped (zero variance): {prep._dropped_cols}")

        # Load precharacterized per-file conditioning vectors if provided.
        # Keys are filenames (basename); values are (cond_dim,) float32 tensors.
        char_lookup: Optional[dict] = None
        if cfg.char_file:
            print(f"Loading trace characterizations from {cfg.char_file} …")
            char_lookup = load_file_characterizations(cfg.char_file, cfg.cond_dim)
            n_matched = sum(1 for f in all_files if (
                char_lookup.get(f.name) is not None or
                any(char_lookup.get(f.name[:-len(e)]) is not None
                    for e in (".zst", ".gz") if f.name.endswith(e))
            ))
            print(f"  {len(char_lookup)} characterizations loaded; "
                  f"{n_matched}/{len(all_files)} training files matched")

        # Hold out a fixed val set — remove val files from all_files so they
        # are never sampled during training.  Build per-file TraceDatasets
        # (same as training) so no window ever crosses a file boundary.
        # Use 10 val files (up from 2) with a deterministic shuffle for
        # reproducible validation across restarts and version comparisons.
        all_files_sorted = sorted(all_files)   # deterministic order
        rng_val = random.Random(42)            # fixed seed → same split every run
        rng_val.shuffle(all_files_sorted)
        n_val_files = min(10, max(0, len(all_files_sorted) - cfg.files_per_epoch))
        val_files = all_files_sorted[:n_val_files]
        all_files = all_files_sorted[n_val_files:]
        val_ds, _ = _load_epoch_dataset(
            val_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
        ) if val_files else (None, [])
        val_tensor = (torch.stack([val_ds[i] for i in range(len(val_ds))])
                      if val_ds and len(val_ds) > 0 else None)
        print(f"  val windows: {len(val_ds) if val_ds else 0:,} ({n_val_files} files, seed=42)")

        # For the first epoch, also build an initial train dataset
        train_ds, _ = _load_epoch_dataset(
            random.sample(all_files, min(cfg.files_per_epoch, len(all_files))),
            cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
        )
    else:
        char_lookup = None  # single-file mode: no per-file lookup
        print(f"Loading: {cfg.trace_path}")
        train_ds, val_ds, prep = load_trace(
            cfg.trace_path, cfg.trace_format, cfg.max_records,
            cfg.timestep, cfg.train_split,
            obj_size_granularity=cfg.obj_size_granularity,
        )
        print(f"  columns ({prep.num_cols}): {prep.col_names}")
        print(f"  delta-encoded: {prep._delta_cols}")
        print(f"  locality-split: {prep._obj_locality_cols}")
        print(f"  log-transformed: {prep._log_cols}")
        if getattr(prep, "_dropped_cols", []):
            print(f"  auto-dropped (zero variance): {prep._dropped_cols}")
        print(f"  train windows: {len(train_ds):,}  val: {len(val_ds):,}")
        val_tensor = torch.stack([val_ds[i] for i in range(len(val_ds))])

    # -----------------------------------------------------------------------
    # Models
    # -----------------------------------------------------------------------
    latent_ae = cfg.latent_dim > 0

    # In latent AE mode the critic operates on latent sequences; in legacy mode
    # it operates directly on feature sequences.
    critic_input_dim = cfg.latent_dim if latent_ae else prep.num_cols

    # Column index of obj_id_reuse in the feature vector (used by locality loss).
    # v15+: obj_id is split into obj_id_reuse (+1=reuse, -1=seek) and obj_id_stride.
    obj_id_col = (prep.col_names.index("obj_id_reuse")
                  if hasattr(prep, "col_names") and "obj_id_reuse" in prep.col_names
                  else prep.col_names.index("obj_id") if "obj_id" in prep.col_names
                  else 1)
    # Column index of obj_id_stride (used by copy-path stride-reuse consistency).
    stride_col = (prep.col_names.index("obj_id_stride")
                  if hasattr(prep, "col_names") and "obj_id_stride" in prep.col_names
                  else -1)

    if cfg.copy_path:
        print(f"[copy-path] reuse_col={obj_id_col} stride_col={stride_col} "
              f"bce_w={cfg.reuse_bce_weight} stride_w={cfg.stride_consistency_weight}")

    # WGAN-GP enforces Lipschitz via gradient penalty — spectral norm is
    # redundant and can interfere with the penalty gradient computation.
    use_sn = (cfg.loss != "wgan-gp")
    _avatar = cfg.avatar
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=cfg.latent_dim if latent_ae else None,
                  avatar=_avatar,
                  cond_dim=cfg.cond_dim,
                  film_cond=getattr(cfg, "film_cond", False),
                  gmm_components=getattr(cfg, "gmm_components", 0),
                  var_cond=getattr(cfg, "var_cond", False),
                  n_regimes=getattr(cfg, "n_regimes", 0),
                  num_lstm_layers=getattr(cfg, "num_lstm_layers", 1)).to(device)
    if getattr(cfg, "n_regimes", 0) > 0 and cfg.cond_dim > 0:
        print(f"[regime-sampler] K={cfg.n_regimes} workload regimes, "
              f"τ: {G.regime_sampler.tau_start}→{G.regime_sampler.tau_end}")
    # Projection discriminator: pass cond_dim so critic adds inner(cond_proj(cond), pooled).
    # Only active when both proj_critic config flag is set AND cond_dim > 0.
    _proj_critic_dim = cfg.cond_dim if getattr(cfg, "proj_critic", False) else 0
    _critic_cls = MultiScaleCritic if getattr(cfg, "multi_scale_critic", False) else Critic
    C = _critic_cls(critic_input_dim, cfg.hidden_size,
                    use_spectral_norm=use_sn,
                    sn_lstm=cfg.sn_lstm,
                    minibatch_std=cfg.minibatch_std,
                    patch_embed=cfg.patch_embed,
                    cond_dim=_proj_critic_dim,
                    num_lstm_layers=getattr(cfg, "num_lstm_layers", 1)).to(device)
    if getattr(cfg, "multi_scale_critic", False):
        print(f"[multi-scale critic] 3-scale critic active (T, T//2, T//4)")

    # Identify binary column indices for mixed-type Recovery heads (idea #7).
    # Opcode columns (±1 read/write) and obj_id_reuse (±1 same-obj indicator)
    # should use sigmoid activation in Recovery to produce sharp ±1 values.
    _binary_cols: list = []
    if getattr(cfg, "mixed_type_recovery", False):
        for i, col in enumerate(prep.col_names):
            if col.lower() in _OPCODE_COLS or col.endswith("_reuse"):
                _binary_cols.append(i)
        if _binary_cols:
            print(f"[mixed-type] binary Recovery heads for cols: "
                  f"{[prep.col_names[i] for i in _binary_cols]} (idx {_binary_cols})")

    if latent_ae:
        E = Encoder(prep.num_cols, cfg.hidden_size, cfg.latent_dim,
                    avatar=_avatar).to(device)
        R = Recovery(cfg.latent_dim, cfg.hidden_size, prep.num_cols,
                     avatar=_avatar,
                     binary_cols=_binary_cols if _binary_cols else None,
                     copy_path=cfg.copy_path,
                     reuse_col=obj_id_col,
                     stride_col=stride_col).to(device)
        S = Supervisor(cfg.latent_dim, cfg.hidden_size,
                       avatar=_avatar).to(device)
        opt_AE  = torch.optim.Adam(list(E.parameters()) + list(R.parameters()),
                                   lr=cfg.lr_er, betas=(0.9, 0.999))
        opt_S   = torch.optim.Adam(S.parameters(), lr=cfg.lr_er, betas=(0.9, 0.999))
        opt_ER  = torch.optim.Adam(list(E.parameters()) + list(R.parameters()),
                                   lr=cfg.lr_er, betas=(0.9, 0.999))
    else:
        E = R = S = opt_AE = opt_S = opt_ER = None

    # AVATAR: latent discriminator + optimizer
    if _avatar and latent_ae:
        LD = LatentDiscriminator(cfg.latent_dim, cfg.hidden_size).to(device)
        opt_LD = torch.optim.Adam(LD.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
        bce_ld = nn.BCEWithLogitsLoss()
    else:
        LD = opt_LD = bce_ld = None

    # Dual discriminator: optional secondary critic in feature space.
    # The latent critic C operates on E/G latent sequences and provides stable
    # early gradients; C_feat catches decoded-space artifacts that the latent
    # critic can miss (e.g. wrong obj_size shape that looks fine in latent space).
    # Only meaningful in latent-AE mode (R decodes latents to features).
    _feat_cw = getattr(cfg, "feat_critic_weight", 0.0)
    if latent_ae and _feat_cw > 0:
        C_feat = Critic(prep.num_cols, cfg.hidden_size,
                        use_spectral_norm=use_sn,
                        sn_lstm=cfg.sn_lstm,
                        minibatch_std=cfg.minibatch_std,
                        patch_embed=cfg.patch_embed,
                        cond_dim=_proj_critic_dim).to(device)
        opt_C_feat = torch.optim.Adam(C_feat.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))
        print(f"[dual-critic] Feature-space critic enabled (weight={_feat_cw:.2f})")
    else:
        C_feat = opt_C_feat = None

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
            G = torch.compile(G, fullgraph=False)
            C = torch.compile(C, fullgraph=False)
            if latent_ae:
                E = torch.compile(E, fullgraph=False)
                R = torch.compile(R, fullgraph=False)
                S = torch.compile(S, fullgraph=False)
            if LD is not None:
                LD = torch.compile(LD, fullgraph=False)
            print("[info] torch.compile: models compiled (CUDA). "
                  "Triton kernels will JIT on first forward pass.")
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
    # PCF loss (path characteristic function, IDEAS.md #6):
    # Learnable frequency vectors trained ADVERSARIALLY — frequencies in C
    # optimizer (maximize PCF distance), G minimizes PCF distance.
    # v71 bug: frequencies in G optimizer collapsed to zero in 10 epochs.
    pcf_loss_fn = None
    if getattr(cfg, 'pcf_loss_weight', 0) > 0:
        from pcf_loss import PCFLoss
        pcf_loss_fn = PCFLoss(
            num_cols=prep.num_cols,
            timestep=cfg.timestep,
            n_freqs=getattr(cfg, 'pcf_n_freqs', 32),
            freq_scale=getattr(cfg, 'pcf_freq_scale', 1.0),
        ).to(device)
        print(f"[PCF] Path characteristic function loss enabled (adversarial): "
              f"n_freqs={cfg.pcf_n_freqs}")

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    # PCF frequency params go in C optimizer: trained to MAXIMIZE PCF distance.
    c_params = list(C.parameters())
    if pcf_loss_fn is not None:
        c_params += list(pcf_loss_fn.parameters())
    opt_C = torch.optim.Adam(c_params, lr=cfg.lr_d, betas=(0.5, 0.9))

    # BayesGAN (Saatci & Wilson, NeurIPS 2017): maintain a posterior over critics.
    # Particle 0 is C itself. Extra particles (1..M-1) are initialized from C
    # with small perturbations so they start diverse.  Each is updated by Adam +
    # SGLD Gaussian noise injection; G sees the average score across all particles.
    # This prevents any single discriminator boundary from forcing mode collapse.
    _n_extra = max(0, getattr(cfg, "bayes_critics", 0) - 1)
    C_extra: list = []
    opt_C_extra: list = []
    if _n_extra > 0:
        for _ in range(_n_extra):
            _Ci = Critic(critic_input_dim, cfg.hidden_size,
                         use_spectral_norm=use_sn, sn_lstm=cfg.sn_lstm,
                         minibatch_std=cfg.minibatch_std, patch_embed=cfg.patch_embed,
                         cond_dim=_proj_critic_dim).to(device)
            _Ci.load_state_dict(C.state_dict())
            with torch.no_grad():
                for _p in _Ci.parameters():
                    _p.add_(torch.randn_like(_p) * 0.01)
            C_extra.append(_Ci)
            opt_C_extra.append(torch.optim.Adam(_Ci.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9)))
        print(f"[BayesGAN] {len(C_extra)+1} critic particles "
              f"({len(C_extra)} extra; SGLD noise injection enabled)")
    C_all: list = [C] + C_extra
    opt_C_all: list = [opt_C] + opt_C_extra

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
        sched_C_extra = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                _oi, T_max=cfg.epochs, eta_min=cfg.lr_d * cfg.lr_cosine_decay)
            for _oi in opt_C_extra
        ]
    else:
        sched_G = sched_C = None
        sched_C_extra = []

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
    w_spike_epochs     = 0             # consecutive epochs with W-dist above w_stop_threshold

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
        _g_missing, _g_unexpected = G.load_state_dict(
            ckpt.get("G_live", ckpt["G"]), strict=False)
        if _g_missing:
            print(f"  New G params (freshly initialised): {_g_missing}")
        if _g_unexpected:
            print(f"  Dropped stale G params: {_g_unexpected}")
        # Migrate critic state dict: old checkpoints store plain LSTM weight
        # keys (e.g. "lstm.weight_ih_l0"); new SN-LSTM models expect "_orig"
        # suffixed keys ("lstm.weight_ih_l0_orig").  Rename on load so old
        # checkpoints work with the new architecture.  strict=False skips the
        # newly added _u/_v power-iteration buffers (initialised randomly and
        # updated on the first forward pass).
        # pretrain_complete.pt may not contain "C" (Critic trained only in Phase 3).
        if "C" in ckpt:
            c_sd = {
                (k + "_orig" if k in ("lstm.weight_ih_l0", "lstm.weight_hh_l0") else k): v
                for k, v in ckpt["C"].items()
            }
            C.load_state_dict(c_sd, strict=False)
        if latent_ae and "E" in ckpt:
            E.load_state_dict(ckpt["E"], strict=False)
            R.load_state_dict(ckpt["R"], strict=False)
            S.load_state_dict(ckpt["S"], strict=False)
        if LD is not None and "LD" in ckpt:
            LD.load_state_dict(ckpt["LD"])
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
            # mmd_history, or best_combined.  Advance start_epoch from the checkpoint
            # so Phase 1–2.5 pretrain gates (which check start_epoch == 0) are skipped
            # when hot-starting from a Phase 3 checkpoint.
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            print(f"Hot-start from {resume_path} (epoch {start_epoch}) with reset optimizer "
                  f"(lr_g={cfg.lr_g:.2e}, lr_d={cfg.lr_d:.2e}, n_critic={cfg.n_critic})")
        else:
            # pretrain_complete.pt only has opt_G (critic not trained yet).
            # Gracefully skip opt states that aren't present.
            if "opt_G" in ckpt:
                opt_G.load_state_dict(ckpt["opt_G"])
            if "opt_C" in ckpt:
                opt_C.load_state_dict(ckpt["opt_C"])
            # BayesGAN extra particles: restore if checkpoint has them, otherwise
            # seed from C (already loaded) + small perturbation (same as init above).
            if C_extra:
                _c_extra_states = ckpt.get("C_extra", [])
                _o_extra_states = ckpt.get("opt_C_extra", [])
                for _i, (_Ci, _oi) in enumerate(zip(C_extra, opt_C_extra)):
                    if _i < len(_c_extra_states):
                        _Ci.load_state_dict(_c_extra_states[_i], strict=False)
                    if _i < len(_o_extra_states):
                        _oi.load_state_dict(_o_extra_states[_i])
            # pretrain_complete.pt has no "epoch" key; leave start_epoch=0.
            if "epoch" in ckpt:
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
    # Pretrain checkpoint: skip phases 1–2.5 if a prior run already completed
    # them.  Saved after Phase 2.5 finishes.  Stores all model weights, the
    # fitted preprocessor, and the val_tensor so that normalization is
    # consistent with what E/R/S/G were trained on.
    # -----------------------------------------------------------------------
    pretrain_ckpt_path = ckpt_dir / "pretrain_complete.pt"
    pretrain_done = False
    if latent_ae and start_epoch == 0 and pretrain_ckpt_path.exists():
        print(f"Loading pretrain checkpoint ({pretrain_ckpt_path}); skipping phases 1–2.5.")
        pc = torch.load(pretrain_ckpt_path, map_location=device, weights_only=False)
        E.load_state_dict(pc["E"])
        R.load_state_dict(pc["R"])
        S.load_state_dict(pc["S"])
        # strict=False: allows new modules (cond_encoder, gmm_prior) that were
        # added after the pretrain checkpoint was saved; their weights are
        # freshly initialised by _init_weights / CondEncoder.__init__.
        missing, unexpected = G.load_state_dict(pc["G"], strict=False)
        if missing:
            print(f"  New G params (freshly initialised): {missing}")
        # Filter to only keys present in G (drops stale keys like cond_encoder.* when
        # loading an old pretrain into a model without var_cond, or vice-versa).
        _g_keys = set(G.state_dict().keys())
        ema_G_state = {k: v.to(device) for k, v in pc["G_ema"].items() if k in _g_keys}
        # New params not in pretrain EMA (e.g. gmm_prior) — seed from live G
        for k, v in G.state_dict().items():
            if k not in ema_G_state:
                ema_G_state[k] = v.clone().to(device)
        if cfg.reset_optimizer:
            print(f"  Fresh optimizer (lr_g={cfg.lr_g:.2e}, lr_d={cfg.lr_d:.2e})")
        else:
            try:
                opt_G.load_state_dict(pc["opt_G"])
            except (ValueError, KeyError):
                # G architecture changed since pretrain (e.g. cond_encoder / gmm_prior added);
                # optimizer param groups no longer match — start fresh (harmless for Phase 3).
                print("  opt_G state incompatible with new G architecture; using fresh optimizer.")
        prep = pc["prep"]
        if pc.get("val_tensor") is not None:
            val_tensor = pc["val_tensor"].to(device)
        if LD is not None and "LD" in pc:
            LD.load_state_dict(pc["LD"])
            if "opt_LD" in pc:
                opt_LD.load_state_dict(pc["opt_LD"])
        pretrain_done = True
        print("  Pretrain checkpoint loaded.")

    # -----------------------------------------------------------------------
    # Phase 1: Autoencoder pretraining  (latent AE mode only)
    # -----------------------------------------------------------------------
    if latent_ae and start_epoch == 0 and not pretrain_done:
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
                with torch.amp.autocast(device.type, enabled=use_amp):
                    H = E(xb)
                    X_hat = R(H)
                    if _binary_cols:
                        # Mixed-type reconstruction: BCE for binary columns (±1 → 0/1 target),
                        # MSE for continuous columns.  BCE targets: (xb+1)/2 ∈ {0,1}.
                        # X_hat for binary cols is in (-1,1) from 2σ-1; convert to logit space
                        # via the inverse: logit(p) = log(p/(1-p)) where p=(x+1)/2.
                        p_hat = (X_hat[..., _binary_cols] + 1.0).clamp(1e-6, 2.0 - 1e-6) / 2.0
                        logits = torch.log(p_hat / (1.0 - p_hat))
                        targets = (xb[..., _binary_cols] + 1.0) / 2.0
                        loss_bin = nn.functional.binary_cross_entropy_with_logits(logits, targets)
                        cont = [i for i in range(prep.num_cols) if i not in set(_binary_cols)]
                        loss_cont = nn.functional.mse_loss(X_hat[..., cont], xb[..., cont])
                        loss_recon = loss_cont + loss_bin
                    else:
                        loss_recon = nn.functional.mse_loss(X_hat, xb)

                    if _avatar:
                        # --- Latent discriminator step (AAE) ---
                        opt_LD.zero_grad()
                        z_prior = torch.randn_like(H)
                        d_real = LD(z_prior.detach().reshape(-1, cfg.latent_dim))
                        d_fake = LD(H.detach().reshape(-1, cfg.latent_dim))
                        ones_ld = torch.ones_like(d_real)
                        zeros_ld = torch.zeros_like(d_fake)
                        loss_disc = bce_ld(d_real, ones_ld) + bce_ld(d_fake, zeros_ld)
                    else:
                        loss_disc = None

                if _avatar:
                    scaler.scale(loss_disc).backward()
                    scaler.step(opt_LD)
                    scaler.update()

                with torch.amp.autocast(device.type, enabled=use_amp):
                    if _avatar:
                        # Encoder adversarial loss (fool the discriminator)
                        d_fake_for_E = LD(H.reshape(-1, cfg.latent_dim))
                        loss_adv_E = bce_ld(d_fake_for_E, torch.ones_like(d_fake_for_E))
                        # Distribution loss (moment matching to N(0,1))
                        loss_mean = sum(H[:, t, :].mean().abs() for t in range(H.size(1)))
                        loss_std = sum((H[:, t, :].std() - 1.0).abs() for t in range(H.size(1)))
                        loss_dist = loss_mean + loss_std
                        loss_ae = (loss_recon
                                   + cfg.latent_disc_weight * loss_adv_E
                                   + cfg.dist_loss_weight * loss_dist)
                    else:
                        loss_ae = loss_recon

                opt_AE.zero_grad()
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
    if latent_ae and start_epoch == 0 and not pretrain_done:
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
                    if _avatar:
                        # AVATAR Eq 4.4-4.5: always 1-step + 2-step
                        loss_sup_1 = nn.functional.mse_loss(S_out[:, :-1, :], H[:, 1:, :])
                        loss_sup_2 = nn.functional.mse_loss(S_out[:, :-2, :], H[:, 2:, :])
                        loss_sup = loss_sup_1 + loss_sup_2
                    else:
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
    if latent_ae and start_epoch == 0 and not pretrain_done and cfg.pretrain_g_epochs > 0:
        print(f"\n--- Phase 2.5: Generator warm-up ({cfg.pretrain_g_epochs} epochs) ---")
        S.eval()   # supervisor is frozen; only G is updated
        for pre_ep in range(cfg.pretrain_g_epochs):
            G.train()
            if multifile:
                ep_files = random.sample(all_files, min(cfg.files_per_epoch, len(all_files)))
                pre_ds, _ = _load_epoch_dataset(
                    ep_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
                    char_lookup=char_lookup)
            else:
                pre_ds = train_ds
            if pre_ds is None or len(pre_ds) == 0:
                continue
            loader = DataLoader(pre_ds, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=n_workers, drop_last=True)
            g_sup_losses = []
            for batch_pre in loader:
                # char_lookup active → DataLoader returns (window, file_cond) tuples
                if isinstance(batch_pre, (list, tuple)) and len(batch_pre) == 2:
                    xb, fc_pre = batch_pre[0].to(device), batch_pre[1].to(device)
                else:
                    xb, fc_pre = batch_pre.to(device), None
                B_pre = xb.size(0)
                xb_dev = xb
                z_g, _ = _make_z_global(B_pre, cfg, device, real_features=xb_dev,
                                        file_cond=fc_pre, G=G)
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

        # Save pretrain checkpoint so future restarts skip phases 1–2.5.
        # ema_G_state is seeded from G's post-warmup weights (not random init)
        # so the first GAN evals use a meaningful EMA baseline.
        ema_G_state = copy.deepcopy(G.state_dict())
        pretrain_ckpt = {
            "E": E.state_dict(),
            "R": R.state_dict(),
            "S": S.state_dict(),
            "G": G.state_dict(),
            "G_ema": ema_G_state,
            "opt_G": opt_G.state_dict(),
            "prep": prep,
            "val_tensor": val_tensor.cpu() if val_tensor is not None else None,
        }
        if LD is not None:
            pretrain_ckpt["LD"] = LD.state_dict()
            pretrain_ckpt["opt_LD"] = opt_LD.state_dict()
        torch.save(pretrain_ckpt, pretrain_ckpt_path)
        print(f"Pretrain checkpoint saved → {pretrain_ckpt_path}", flush=True)

    print(f"\n--- Phase 3: Joint GAN training ({cfg.epochs} epochs) ---", flush=True)

    # Build a cond_pool from char_lookup (all training-file conditioning vectors
    # stacked into a tensor).  Passed to evaluate_metrics so EMA eval uses the
    # same file-level conditioning distribution as training, instead of the
    # noisier window-level descriptors (compute_window_descriptors fallback).
    # Without this, char-file training inflates EMA recall vs full eval because
    # the EMA eval conditions G on the wrong distribution.
    # Prefer val-file conditioning vectors so EMA eval matches the conditioning
    # distribution of the val set.  Fall back to all training files if char_lookup
    # doesn't cover the val files.
    _eval_cond_pool: Optional[torch.Tensor] = None
    if char_lookup is not None and cfg.cond_dim > 0:
        if multifile:
            _val_cond_vecs = []
            for _vf in val_files:
                _vc = char_lookup.get(_vf.name)
                if _vc is None:
                    for _ext in (".zst", ".gz"):
                        if _vf.name.endswith(_ext):
                            _vc = char_lookup.get(_vf.name[:-len(_ext)])
                            break
                if _vc is not None:
                    _val_cond_vecs.append(_vc)
            if _val_cond_vecs:
                _eval_cond_pool = torch.stack(_val_cond_vecs).to(device)
            else:
                _eval_cond_pool = torch.stack(list(char_lookup.values())).to(device)
        else:
            _eval_cond_pool = torch.stack(list(char_lookup.values())).to(device)

    # -----------------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs):
        G.train(); C.train()
        if latent_ae:
            E.train(); R.train(); S.train()
        if LD is not None:
            LD.train()
        # Anneal Gumbel temperature for regime sampler (idea #5).
        if getattr(G, "regime_sampler", None) is not None:
            G.regime_sampler.anneal(epoch / max(cfg.epochs - 1, 1))
        t0 = time.time()
        c_losses, g_losses, w_dists = [], [], []
        cp_bce_losses, cp_stride_losses = [], []   # copy-path tracking
        pcf_losses = []   # PCF loss tracking

        # In multi-file mode: resample a fresh random subset of files each epoch.
        # This is the key mechanism behind multi-corpus generalisation: rather than
        # training on the full dataset simultaneously (which would require holding
        # 378 files × 15K records = 5M+ windows in memory), we train on a random
        # 8-file sample per epoch.  Over 300 epochs the model sees ~2400 file-epochs
        # across 378 files — approximately 6 passes through the full corpus, enough
        # for the model to encounter the diversity of tenant behaviours, burst regimes,
        # and access patterns without overfitting to any single file's particularities.
        if multifile:
            k = min(cfg.files_per_epoch, len(all_files))
            if getattr(cfg, "block_sample", False):
                start = random.randint(0, len(all_files) - k)
                epoch_files = all_files[start:start + k]
            else:
                epoch_files = random.sample(all_files, k)
            train_ds, _ = _load_epoch_dataset(
                epoch_files, cfg.trace_format, cfg.records_per_file, prep, cfg.timestep,
                char_lookup=char_lookup,
            )
            if train_ds is None or len(train_ds) == 0:
                print(f"Epoch {epoch+1}: no data loaded, skipping.")
                continue

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=n_workers, pin_memory=(device.type == "cuda"),
            drop_last=True,
        )

        for batch in train_loader:
            # When char_lookup is active, TraceDataset returns (window, file_cond)
            # tuples; DataLoader collates them into a 2-element list.
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                real_batch, file_cond_batch = batch[0].to(device), batch[1].to(device)
            else:
                real_batch, file_cond_batch = batch.to(device), None
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
                z_g, _ = _make_z_global(B, cfg, device, real_features=real_batch,
                                        file_cond=file_cond_batch, G=G)
                z_l = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                H_fake = G(z_g, z_l).detach()

                # PacGAN packing (computed once, shared across BayesGAN particles)
                _pack_sz = getattr(cfg, "pack_size", 1)
                H_real_c, _c_cond = _pack_windows(H_real, _pack_sz, file_cond_batch)
                H_fake_c, _    = _pack_windows(H_fake, _pack_sz, file_cond_batch)

                # BayesGAN: update all critic particles.  When bayes_critics <= 1,
                # C_all = [C] and this is identical to the previous single-critic loop.
                _w_batch: list = []
                _c_batch: list = []
                for _Ci, _opt_Ci in zip(C_all, opt_C_all):
                    _opt_Ci.zero_grad()
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        if cfg.loss == "rpgan":
                            c_real = _Ci(H_real_c, cond=_c_cond)
                            c_fake = _Ci(H_fake_c, cond=_c_cond)
                            rel = c_real - c_fake
                            c_loss = nn.functional.softplus(-rel).mean()
                            _w_batch.append(rel.mean().item())
                        elif cfg.loss in ("wgan-sn", "wgan-gp"):
                            _c_fake_scores = _Ci(H_fake_c, cond=_c_cond)
                            _c_real_scores = _Ci(H_real_c, cond=_c_cond)
                            if cfg.self_diag_temp > 0:
                                # Self-diagnosing upweighting: real samples with high
                                # critic scores (= modes G under-covers) get more weight.
                                _sd_w = torch.softmax(
                                    _c_real_scores.detach() / cfg.self_diag_temp, dim=0)
                                c_base = _c_fake_scores.mean() - (_sd_w * _c_real_scores).sum()
                            else:
                                c_base = (_c_fake_scores - _c_real_scores).mean()
                            _w_batch.append(-c_base.item())
                            c_loss = c_base
                            # Gradient penalties only on the primary critic (C); they
                            # need create_graph=True which is expensive, and spectral
                            # norm already handles Lipschitz for extra particles.
                            if _Ci is C:
                                if cfg.loss == "wgan-gp":
                                    eps = torch.rand(B, 1, 1, device=device)
                                    x_hat = (eps * H_real.detach() +
                                             (1 - eps) * H_fake).requires_grad_(True)
                                    with torch.backends.cudnn.flags(enabled=False):
                                        d_hat = _Ci(x_hat, cond=_c_cond)
                                    grads = torch.autograd.grad(
                                        outputs=d_hat.sum(), inputs=x_hat,
                                        create_graph=True)[0]
                                    gp = ((grads.norm(2, dim=(1, 2)) - 1) ** 2).mean()
                                    c_loss = c_loss + cfg.gp_lambda * gp
                                if cfg.r1_lambda > 0:
                                    H_real_r1 = H_real.detach().clone().requires_grad_(True)
                                    with torch.backends.cudnn.flags(enabled=False):
                                        d_real_r1 = _Ci(H_real_r1)
                                    grads_r1 = torch.autograd.grad(
                                        outputs=d_real_r1.sum(), inputs=H_real_r1,
                                        create_graph=True)[0]
                                    r1 = (grads_r1.norm(2, dim=(1, 2)) ** 2).mean()
                                    c_loss = c_loss + cfg.r1_lambda * 0.5 * r1
                                if cfg.r2_lambda > 0:
                                    H_fake_r2 = H_fake.clone().requires_grad_(True)
                                    with torch.backends.cudnn.flags(enabled=False):
                                        d_fake_r2 = _Ci(H_fake_r2)
                                    grads_r2 = torch.autograd.grad(
                                        outputs=d_fake_r2.sum(), inputs=H_fake_r2,
                                        create_graph=True)[0]
                                    r2 = (grads_r2.norm(2, dim=(1, 2)) ** 2).mean()
                                    c_loss = c_loss + cfg.r2_lambda * r2
                        else:  # bce
                            real_labels = torch.ones(B, 1, device=device)
                            fake_labels = torch.zeros(B, 1, device=device)
                            c_loss = (bce(_Ci(H_real_c, cond=_c_cond), real_labels[:H_real_c.size(0)]) +
                                      bce(_Ci(H_fake_c, cond=_c_cond), fake_labels[:H_fake_c.size(0)]))
                    scaler_C.scale(c_loss).backward()
                    if cfg.grad_clip > 0:
                        scaler_C.unscale_(_opt_Ci)
                        nn.utils.clip_grad_norm_(_Ci.parameters(), cfg.grad_clip)
                    scaler_C.step(_opt_Ci)
                    scaler_C.update()
                    # SGLD noise injection for particle diversity (extra particles only).
                    # After the Adam step, add ε ~ N(0, 2η·I) to perturb this particle
                    # away from the others, approximating SGLD posterior sampling.
                    if _Ci is not C and len(C_all) > 1:
                        _eta = _opt_Ci.param_groups[0]["lr"]
                        with torch.no_grad():
                            for _p in _Ci.parameters():
                                _p.add_(torch.randn_like(_p) * (_eta * 2.0) ** 0.5)
                    _c_batch.append(c_loss.item())

                # Log mean W-dist and mean c_loss across particles
                w_dists.append(sum(_w_batch) / len(_w_batch))
                c_losses.append(sum(_c_batch) / len(_c_batch))

                # --- Feature-space critic step (dual discriminator) ---
                if C_feat is not None and R is not None:
                    opt_C_feat.zero_grad()
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        with torch.no_grad():
                            feat_fake_d = R(H_fake)    # decode latents → features (detached)
                        feat_real_c, _ = _pack_windows(real_batch, _pack_sz, file_cond_batch)
                        feat_fake_c, _ = _pack_windows(feat_fake_d, _pack_sz, file_cond_batch)
                        cf_base = (C_feat(feat_fake_c, cond=_c_cond) -
                                   C_feat(feat_real_c, cond=_c_cond)).mean()
                        cf_loss = cf_base
                    scaler_C.scale(cf_loss).backward()
                    if cfg.grad_clip > 0:
                        scaler_C.unscale_(opt_C_feat)
                        nn.utils.clip_grad_norm_(C_feat.parameters(), cfg.grad_clip)
                    scaler_C.step(opt_C_feat)
                    scaler_C.update()

            # --- PCF adversarial step (frequency maximization) ---
            # Train frequency vectors to find where real and fake differ most.
            # Uses decoded features (not latents) for feature-space comparison.
            if pcf_loss_fn is not None and getattr(cfg, 'pcf_loss_weight', 0) > 0:
                opt_C.zero_grad()
                with torch.no_grad():
                    z_g_pcf, _ = _make_z_global(B, cfg, device, real_features=real_batch,
                                                file_cond=file_cond_batch, G=G)
                    z_l_pcf = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                    H_fake_pcf = G(z_g_pcf, z_l_pcf)
                    fake_decoded_pcf = R(H_fake_pcf) if latent_ae else H_fake_pcf
                # Maximize PCF distance: negate loss for gradient ascent on frequencies
                pcf_dist = pcf_loss_fn(real_batch.detach(), fake_decoded_pcf.detach())
                (-pcf_dist).backward()  # gradient ASCENT on frequency params
                if cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(pcf_loss_fn.parameters(), cfg.grad_clip)
                opt_C.step()

            # --- Generator step ---
            z_g, kl_loss = _make_z_global(B, cfg, device, real_features=real_batch,
                                          file_cond=file_cond_batch, G=G)
            z_l = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)

            opt_G.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                H_fake = G(z_g, z_l)

                if cfg.loss == "rpgan":
                    # Generator tries to make fake beat real: -E[log σ(C(x_f) - C(x_r))]
                    g_score, feat_fake = C(H_fake, return_features=True, cond=file_cond_batch)
                    with torch.no_grad():
                        c_real_g = C(H_real, cond=file_cond_batch)   # (B, 1), no grad through real
                    g_loss = nn.functional.softplus(-(g_score - c_real_g)).mean()
                elif cfg.loss in ("wgan-sn", "wgan-gp"):
                    if len(C_all) > 1:
                        # BayesGAN: G sees the average score across all critic particles.
                        # This prevents G from exploiting a single weak critic — it must
                        # satisfy the *posterior distribution* over critics.
                        # feat_fake from particle 0 (C) for feature matching below.
                        _g_scores = []
                        for _Ci in C_all:
                            _si, _ = _Ci(H_fake, return_features=True, cond=file_cond_batch)
                            _g_scores.append(_si)
                        g_score = torch.stack(_g_scores).mean(0)
                        _, feat_fake = C(H_fake, return_features=True, cond=file_cond_batch)
                    else:
                        g_score, feat_fake = C(H_fake, return_features=True, cond=file_cond_batch)
                    g_loss = -g_score.mean()
                else:  # bce
                    real_labels = torch.ones(B, 1, device=device)
                    g_score, feat_fake = C(H_fake, return_features=True, cond=file_cond_batch)
                    g_loss = bce(g_score, real_labels)

                # Feature matching (Salimans et al. 2016): match the mean of the
                # critic's internal representation for real vs fake.  Penalising
                # the L2 distance between batch-mean features forces the generator
                # to cover all modes the critic can "see", fixing mode collapse.
                if cfg.feature_matching_weight > 0:
                    with torch.no_grad():
                        _fm_scores, feat_real = C(H_real, return_features=True, cond=file_cond_batch)
                    if cfg.self_diag_temp > 0:
                        # Upweight features of under-covered real samples
                        _fm_w = torch.softmax(
                            _fm_scores.detach() / cfg.self_diag_temp, dim=0)  # (B, 1)
                        feat_real_avg = (_fm_w * feat_real).sum(0)
                    else:
                        feat_real_avg = feat_real.mean(0)
                    loss_fm = nn.functional.mse_loss(
                        feat_fake.mean(0), feat_real_avg)
                    g_loss = g_loss + cfg.feature_matching_weight * loss_fm

                if latent_ae:
                    # Supervisor consistency: S(H_fake)[t] should predict H_fake[t+1].
                    # This forces the generator to produce temporally coherent latents
                    # that obey the same autoregressive dynamics as real data.
                    # S is a frozen teacher: use no_grad so gradients only flow back
                    # to G through H_fake[:, k:, :] (the prediction target), not
                    # through S itself.  Matches the G-warmup phase pattern.
                    with torch.no_grad():
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
                # Moment matching (L_V): penalise differences in per-feature mean,
                # std, slope, and skewness (ChronoGAN, ICMLA 2024).
                # Slope and skewness directly target heavy-tailed obj_size distribution.
                if cfg.moment_loss_weight > 0:
                    loss_V = (
                        nn.functional.l1_loss(fake_decoded.mean(0), real_batch.mean(0)) +
                        nn.functional.l1_loss(fake_decoded.std(0),  real_batch.std(0))
                    )
                    # Slope: linear trend per feature per sequence
                    _T = fake_decoded.shape[1]
                    _t = torch.arange(_T, device=device, dtype=fake_decoded.dtype)
                    _t = _t - _t.mean()
                    _t_var = (_t ** 2).mean() + 1e-8
                    _x_c = fake_decoded - fake_decoded.mean(dim=1, keepdim=True)
                    _r_c = real_batch   - real_batch.mean(dim=1, keepdim=True)
                    fake_slope = (_t.view(1, _T, 1) * _x_c).mean(1) / _t_var
                    real_slope = (_t.view(1, _T, 1) * _r_c).mean(1) / _t_var
                    loss_V = loss_V + nn.functional.l1_loss(fake_slope.mean(0), real_slope.mean(0))
                    # Skewness: third standardized moment (penalises distributional tail mismatch)
                    _fx_mu = fake_decoded.mean(dim=1, keepdim=True)
                    _rx_mu = real_batch.mean(dim=1, keepdim=True)
                    _fx_s  = fake_decoded.std(dim=1, keepdim=True) + 1e-8
                    _rx_s  = real_batch.std(dim=1, keepdim=True) + 1e-8
                    fake_skew = ((fake_decoded - _fx_mu) / _fx_s).pow(3).mean(dim=1).mean(0)
                    real_skew = ((real_batch   - _rx_mu) / _rx_s).pow(3).mean(dim=1).mean(0)
                    loss_V = loss_V + 0.5 * nn.functional.l1_loss(fake_skew, real_skew)
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
                    fd_flat = fake_decoded.reshape(-1, fake_decoded.shape[-1]).float()
                    rb_flat = real_batch.reshape(-1, real_batch.shape[-1]).float()
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
                    # L_loc: reuse rate matching.
                    # v15+: obj_id_reuse is an explicit ±1 binary feature (+1=same object,
                    # -1=different object). Convert to [0,1] probability and match means.
                    # This replaces the old pairwise soft-distance computation over
                    # delta-encoded obj_id, which was unable to detect reuse because
                    # delta=0 is a zero-measure point in continuous space (v14g root cause).
                    # Real Tencent Block: ~40-50% of accesses are reuses within a window.
                    obj_fake_reuse = fake_decoded[:, :, obj_id_col]   # (B, T) in [-1, 1]
                    obj_real_reuse = real_batch[:, :, obj_id_col]
                    # (x + 1) / 2 maps ±1 → [0, 1]
                    fake_reuse = ((obj_fake_reuse + 1) / 2).mean()
                    with torch.no_grad():
                        real_reuse = ((obj_real_reuse + 1) / 2).mean()
                    loss_loc = (fake_reuse - real_reuse).pow(2)
                    g_loss = g_loss + cfg.locality_loss_weight * loss_loc

                # Copy-path: per-timestep reuse BCE + stride-reuse consistency.
                # Structural replacement for scalar locality_loss: gives per-timestep
                # gradient for reuse decisions + enforces stride=0 when reuse=+1.
                if cfg.copy_path and stride_col >= 0:
                    obj_fake_reuse_cp = fake_decoded[:, :, obj_id_col]   # (B, T) in [-1, 1]
                    obj_real_reuse_cp = real_batch[:, :, obj_id_col]
                    # Map to [0,1] for BCE
                    fake_p = (obj_fake_reuse_cp + 1.0) / 2.0
                    real_p = (obj_real_reuse_cp + 1.0) / 2.0
                    # Class-weighted BCE: upweight reuse events (minority class).
                    # pos_weight = (1 - mean_reuse_rate) / mean_reuse_rate
                    with torch.no_grad():
                        mean_reuse = real_p.mean().clamp(min=0.01)
                        pos_w = (1.0 - mean_reuse) / mean_reuse  # e.g. 0.9/0.1 = 9.0
                    weight = real_p * pos_w + (1.0 - real_p)      # per-element weight
                    loss_reuse_bce = nn.functional.binary_cross_entropy(
                        fake_p.clamp(1e-6, 1.0 - 1e-6), real_p,
                        weight=weight,
                        reduction='mean',
                    )
                    g_loss = g_loss + cfg.reuse_bce_weight * loss_reuse_bce
                    cp_bce_losses.append(loss_reuse_bce.item())

                    # Stride-reuse consistency: penalise |stride| where real says reuse.
                    stride_fake = fake_decoded[:, :, stride_col]
                    reuse_mask = (obj_real_reuse_cp > 0).float()  # 1 where reuse
                    loss_stride_cons = (stride_fake.abs() * reuse_mask).mean()
                    g_loss = g_loss + cfg.stride_consistency_weight * loss_stride_cons
                    cp_stride_losses.append(loss_stride_cons.item())

                # L_div: MSGAN mode-seeking diversity loss.
                # For two independently sampled noise pairs, maximise the ratio of
                # output distance to input distance — directly combats mode collapse
                # (when the generator maps very different noise to the same output).
                # Reference: Mao et al., CVPR 2019, "Mode Seeking GAN".
                # Set diversity_loss_weight > 0 (try 0.5–2.0) when β-recall < 0.5.
                if cfg.diversity_loss_weight > 0:
                    B2 = B // 2
                    # When char-file conditioning is available, use DIFFERENT
                    # file conditioning vectors for the two halves of the batch.
                    # This measures cross-workload diversity (does G produce
                    # different outputs for different workload types?), directly
                    # targeting the mode coverage gaps in recall.
                    _fc1 = file_cond_batch[:B2] if file_cond_batch is not None else None
                    _fc2 = file_cond_batch[B2:B2*2] if file_cond_batch is not None else None
                    z_g1, _ = _make_z_global(B2, cfg, device, real_features=real_batch[:B2],
                                             file_cond=_fc1, G=G)
                    z_l1 = torch.randn(B2, cfg.timestep, cfg.noise_dim, device=device)
                    z_g2, _ = _make_z_global(B2, cfg, device, real_features=real_batch[B2:B2*2],
                                             file_cond=_fc2, G=G)
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

                # L_cont: chunk-continuity loss (v33).  Generate two adjacent
                # windows carrying LSTM hidden state (as generate.py does at
                # inference) and penalise distributional mismatch at the
                # boundary.  Closes train/inference gap targeting DMD-GEN.
                if cfg.continuity_loss_weight > 0:
                    z_gc, _ = _make_z_global(B, cfg, device, real_features=real_batch,
                                             file_cond=file_cond_batch, G=G)
                    z_lc1 = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                    z_lc2 = torch.randn(B, cfg.timestep, cfg.noise_dim, device=device)
                    H_cont_1, h_carry = G(z_gc, z_lc1, return_hidden=True)
                    H_cont_2 = G(z_gc, z_lc2, hidden=h_carry)
                    if latent_ae:
                        feat_cont_1 = R(H_cont_1)
                        feat_cont_2 = R(H_cont_2)
                    else:
                        feat_cont_1 = H_cont_1
                        feat_cont_2 = H_cont_2
                    k_cont = max(1, min(3, cfg.timestep // 4))
                    tail = feat_cont_1[:, -k_cont:, :]
                    head = feat_cont_2[:, :k_cont, :]
                    loss_cont_mean = (tail.mean(dim=1) - head.mean(dim=1)).pow(2).mean()
                    loss_cont_std  = (tail.std(dim=1)  - head.std(dim=1)).pow(2).mean()
                    loss_cont = loss_cont_mean + loss_cont_std
                    g_loss = g_loss + cfg.continuity_loss_weight * loss_cont

                # Feature-space critic in generator loss (dual discriminator).
                # C_feat operates in decoded feature space — it sees what the
                # downstream evaluator sees, preventing the generator from hiding
                # quality problems in latent space that R glosses over.
                if C_feat is not None and R is not None and _feat_cw > 0:
                    feat_fake_g = fake_decoded   # already decoded above (or H_fake if no AE)
                    feat_fake_gc, _cf_cond = _pack_windows(feat_fake_g, _pack_sz, file_cond_batch)
                    cf_g_score = C_feat(feat_fake_gc, cond=_cf_cond)
                    g_loss = g_loss + _feat_cw * (-cf_g_score.mean())

                # L_PCF: Path Characteristic Function loss (IDEAS.md #6, PCF-GAN).
                # Single learned functional replacing handcrafted auxiliary losses.
                # Compares empirical characteristic functions of real vs fake path
                # increments at learnable frequency vectors.
                if pcf_loss_fn is not None and getattr(cfg, 'pcf_loss_weight', 0) > 0:
                    loss_pcf = pcf_loss_fn(real_batch.detach(), fake_decoded)
                    g_loss = g_loss + cfg.pcf_loss_weight * loss_pcf
                    pcf_losses.append(loss_pcf.item())

                # Variational conditioning KL loss (IDEAS.md #3): regularise the
                # cond encoder toward N(0,I).  kl_loss=0 when var_cond is off.
                _kl_w = getattr(cfg, "var_cond_kl_weight", 0.0)
                if _kl_w > 0:
                    g_loss = g_loss + _kl_w * kl_loss

            # Guard against NaN/inf/explosion G_loss (can occur with extreme conditioning
            # vectors, e.g. Alibaba traces + var_cond CondEncoder edge cases).  The
            # explosions (~1.8T) are large but *finite*, so isfinite() alone is not enough.
            # Skip the backward pass entirely — EMA weights are not touched.
            if not torch.isfinite(g_loss) or g_loss.abs() > 1e6:
                opt_G.zero_grad()
                scaler.update()
                g_losses.append(0.0)
            else:
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
                if _avatar:
                    # --- AVATAR: update latent discriminator in joint phase ---
                    opt_LD.zero_grad()
                    with torch.amp.autocast(device.type, enabled=use_amp):
                        H_real_ld = E(real_batch)
                        z_prior = torch.randn_like(H_real_ld)
                        d_real_j = LD(z_prior.detach().reshape(-1, cfg.latent_dim))
                        d_fake_j = LD(H_real_ld.detach().reshape(-1, cfg.latent_dim))
                        ones_j = torch.ones_like(d_real_j)
                        zeros_j = torch.zeros_like(d_fake_j)
                        loss_ld_j = bce_ld(d_real_j, ones_j) + bce_ld(d_fake_j, zeros_j)
                    scaler.scale(loss_ld_j).backward()
                    scaler.step(opt_LD)
                    scaler.update()

                opt_ER.zero_grad()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    H_real_grad = E(real_batch)
                    X_hat = R(H_real_grad)
                    er_loss = nn.functional.mse_loss(X_hat, real_batch)

                    if _avatar:
                        # Supervisor-assisted reconstruction (AVATAR Eq 4.10)
                        X_sup_recon = R(S(H_real_grad))
                        er_loss = er_loss + nn.functional.mse_loss(X_sup_recon, real_batch)

                        # Adversarial loss on encoder (fool latent discriminator)
                        d_fake_for_E_j = LD(H_real_grad.reshape(-1, cfg.latent_dim))
                        loss_adv_E_j = bce_ld(d_fake_for_E_j,
                                              torch.ones_like(d_fake_for_E_j))
                        # Distribution loss
                        loss_mean_j = sum(H_real_grad[:, t, :].mean().abs()
                                          for t in range(H_real_grad.size(1)))
                        loss_std_j = sum((H_real_grad[:, t, :].std() - 1.0).abs()
                                         for t in range(H_real_grad.size(1)))
                        loss_dist_j = loss_mean_j + loss_std_j
                        er_loss = (er_loss
                                   + cfg.latent_disc_weight * loss_adv_E_j
                                   + cfg.dist_loss_weight * loss_dist_j)

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
        if cp_bce_losses:
            cp_bce_m = sum(cp_bce_losses) / len(cp_bce_losses)
            cp_str_m = sum(cp_stride_losses) / len(cp_stride_losses) if cp_stride_losses else 0.0
            log += f"  reuse_bce={cp_bce_m:.4f}  stride_con={cp_str_m:.4f}"
        if pcf_losses:
            pcf_m = sum(pcf_losses) / len(pcf_losses)
            log += f"  pcf={pcf_m:.4f}"

        if val_tensor is not None and (epoch + 1) % cfg.mmd_every == 0:
            # Evaluate using the EMA model, not the live G.
            # Save live weights first, then swap in EMA, then restore.
            live_G_state = copy.deepcopy(G.state_dict())
            G.load_state_dict({k: v.to(device) for k, v in ema_G_state.items()})
            mmd_val, recall_val, combined_val = evaluate_metrics(
                G, val_tensor, cfg.mmd_samples, cfg.timestep, device,
                recovery=R if latent_ae else None,
                dmd_weight=cfg.dmd_ckpt_weight,
                cond_pool=_eval_cond_pool,
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
                    "G": {k: v.clone() for k, v in ema_G_state.items()},  # EMA weights (primary)
                    "G_live": live_G_state,     # live weights (for resuming training)
                    "G_ema": ema_G_state,       # EMA weights (backward compat)
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
                if LD is not None:
                    ckpt_data["LD"] = LD.state_dict()
                if C_extra:
                    ckpt_data["C_extra"] = [_Ci.state_dict() for _Ci in C_extra]
                    ckpt_data["opt_C_extra"] = [_oi.state_dict() for _oi in opt_C_extra]
                torch.save(ckpt_data, ckpt_dir / "best.pt")
                log += "  ★"
            else:
                epochs_no_improve += 1
            G.train()
            if latent_ae:
                E.train(); R.train(); S.train()
            if LD is not None:
                LD.train()

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

        # W-distance spike guard: halt when W-dist exceeds threshold for 3+
        # consecutive epochs.  Catches the conditioning instability collapse
        # seen in v33 (W→10.6 ep179) and v36 (W→10 ep178) which happened well
        # after the EMA best was saved — best.pt is safe to use after stopping.
        # 0 = disabled.  Recommended: 3.0 (good runs rarely exceed 2.0).
        _w_thresh = getattr(cfg, "w_stop_threshold", 0.0)
        if _w_thresh > 0 and not math.isnan(w_mean):
            if w_mean > _w_thresh:
                w_spike_epochs += 1
            else:
                w_spike_epochs = 0
            if w_spike_epochs >= 3:
                print(f"W-spike guard: W={w_mean:.2f} > {_w_thresh} for "
                      f"{w_spike_epochs} consecutive epochs. Stopping — best.pt preserved.",
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
            if LD is not None:
                ckpt_data["LD"] = LD.state_dict()
            if C_extra:
                ckpt_data["C_extra"] = [_Ci.state_dict() for _Ci in C_extra]
                ckpt_data["opt_C_extra"] = [_oi.state_dict() for _oi in opt_C_extra]
            ckpt_data["mmd_history"] = mmd_history
            torch.save(ckpt_data, ckpt_path)
            print(f"  → saved {ckpt_path}")

        # Step cosine scheduler at end of epoch (after checkpoint) so the LR
        # in the saved opt state matches the epoch that was just trained.
        if sched_G is not None:
            sched_G.step()
            sched_C.step()
            for _s in sched_C_extra:
                _s.step()

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
    if LD is not None:
        final_data["LD"] = LD.state_dict()
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
                   choices=["wgan-sn", "wgan-gp", "bce", "rpgan"],
                   help="wgan-sn: Wasserstein + spectral norm; "
                        "wgan-gp: Wasserstein + gradient penalty (true Lipschitz, CUDA only); "
                        "bce: original GAN cross-entropy; "
                        "rpgan: relativistic paired GAN (R3GAN, NeurIPS 2024; MPS-compatible, "
                        "improves mode coverage / β-recall)")
    p.add_argument("--gp-lambda",        type=float, default=10.0,
                   help="Gradient penalty coefficient for wgan-gp (standard: 10)")
    p.add_argument("--epochs",           type=int,   default=200)
    p.add_argument("--batch-size",       type=int,   default=64)
    p.add_argument("--seed",             type=int,   default=-1,
                   help="Random seed for reproducibility (-1 = random). "
                        "Sets Python/NumPy/PyTorch seeds before model init.")
    p.add_argument("--timestep",         type=int,   default=12)
    p.add_argument("--noise-dim",        type=int,   default=10)
    p.add_argument("--cond-dim",         type=int,   default=0,
                   help="Workload conditioning dim (0=unconditional; 10=per-window descriptors)")
    p.add_argument("--gmm-components",   type=int,   default=0,
                   help="GMM prior: K mixture components in noise space (0=flat N(0,I)). "
                        "When >0 and --cond-dim>0, replaces N(0,I) with a conditioning-aware "
                        "mixture so each workload type gets its own noise region. Try 8.")
    p.add_argument("--var-cond",         action="store_true", default=False,
                   help="Variational conditioning (IDEAS.md #3): replaces fixed char-file "
                        "vector with a learned N(μ,σ²) distribution. Samples cond at train "
                        "time; uses μ at eval. Adds KL(q||N(0,I)) to G loss. "
                        "Requires --cond-dim>0 and --char-file.")
    p.add_argument("--var-cond-kl-weight", type=float, default=0.001,
                   help="Weight for KL divergence term in G loss when --var-cond is set. "
                        "Small values (0.001) let σ grow slowly; larger values (0.01) "
                        "push μ toward 0 and σ toward 1 more aggressively.")
    p.add_argument("--cond-drop-prob",   type=float, default=0.0,
                   help="Classifier-free guidance: drop conditioning with this probability (0=always condition)")
    p.add_argument("--hidden-size",      type=int,   default=256)
    p.add_argument("--compile",          action="store_true", default=True,
                   help="torch.compile models for ~20-40%% CUDA speedup "
                        "(CUDA only; incompatible with wgan-gp/r1/r2; default on)")
    p.add_argument("--no-compile",       dest="compile", action="store_false",
                   help="Disable torch.compile")
    p.add_argument("--no-minibatch-std", action="store_true", default=False,
                   help="Disable minibatch std channel in critic (on by default)")
    p.add_argument("--no-sn-lstm",       action="store_true", default=False,
                   help="Disable spectral norm on critic LSTM weight matrices "
                        "(on by default; prevents W-distance drift and mode collapse)")
    p.add_argument("--patch-embed",      action="store_true", default=False,
                   help="Conv1d patch embedding before critic LSTM: folds 12-step window "
                        "into 4 patch tokens (kernel=stride=3). TTS-GAN style. "
                        "Gives critic local-pattern inductive bias for burst detection.")
    p.add_argument("--proj-critic",      action="store_true", default=False,
                   help="Projection discriminator (Miyato & Koyama, ICLR 2018): "
                        "adds inner(cond_proj(cond), critic_features) to critic score. "
                        "Conditions critic on workload descriptors so it evaluates "
                        "'is this realistic for THIS workload?' Requires --cond-dim > 0. "
                        "Best used with --char-file for stable conditioning.")
    p.add_argument("--film-cond",        action="store_true", default=False,
                   help="FiLM conditioning in G (NeurIPS 2018): applies "
                        "(1+γ(z_global))*h_t + β(z_global) after LSTM at each timestep, "
                        "re-injecting workload conditioning so it cannot fade through "
                        "the LSTM forget gate. Zero-init → backward compatible. "
                        "Requires --cond-dim > 0. Planned for v40+.")
    p.add_argument("--pack-size",        type=int, default=1,
                   help="PacGAN window packing (Lin et al. NeurIPS 2018): critic scores packs "
                        "of m consecutive windows concatenated along time axis. "
                        "1=off (default), 2=pairs. Gives critic explicit diversity signal: "
                        "fake packs look identical under mode collapse, real packs are diverse.")
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
    p.add_argument("--block-sample",     action="store_true",
                   help="Sample contiguous file blocks instead of random files per epoch. "
                        "Preserves temporal coherence for corpora with high Hurst exponent "
                        "(alibaba H=0.98). Requires filenames that sort into temporal order.")
    p.add_argument("--char-file",        default="",
                   metavar="JSONL",
                   help="Path to trace_characterizations.jsonl. When set, each "
                        "training file's z_global conditioning vector is looked "
                        "up from precharacterized full-trace statistics (write_ratio, "
                        "reuse_ratio, burstiness_cv, etc.) instead of being estimated "
                        "from noisy 12-step windows. Files not found in the lookup "
                        "receive a zero vector (unconditional path). "
                        "Use: --char-file /path/to/trace_characterizations.jsonl")
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
    p.add_argument("--avatar",              action="store_true", default=False,
                   help="Enable AVATAR mode: AAE latent discriminator + autoregressive refinement. "
                        "Replaces Sigmoid with unbounded latents targeting N(0,1).")
    p.add_argument("--dist-loss-weight",   type=float, default=1.0,
                   help="AVATAR: distribution loss weight (moment matching to N(0,1))")
    p.add_argument("--latent-disc-weight", type=float, default=1.0,
                   help="AVATAR: latent discriminator adversarial loss weight")
    p.add_argument("--locality-loss-weight", type=float, default=0.0,
                   help="L_loc: stride-repetition rate matching within windows (0=off; try 1.0). "
                        "Matches the fraction of obj_id deltas that repeat within a window — "
                        "a proxy for sequential-access consistency.")
    p.add_argument("--copy-path", action="store_true", default=False,
                   help="Copy-path mechanism: per-timestep reuse BCE + stride-reuse "
                        "consistency loss + Recovery stride gating. Structural replacement "
                        "for scalar locality_loss. Pretrain-compatible.")
    p.add_argument("--reuse-bce-weight", type=float, default=2.0,
                   help="Copy-path: per-timestep BCE weight on reuse column (default 2.0). "
                        "Class-weighted to handle seek/reuse imbalance.")
    p.add_argument("--stride-consistency-weight", type=float, default=1.0,
                   help="Copy-path: penalise |stride| when real reuse=+1 (default 1.0).")
    p.add_argument("--pcf-loss-weight",      type=float, default=0.0,
                   help="L_PCF: path characteristic function loss (PCF-GAN, NeurIPS 2023). "
                        "Single learned functional replacing ACF/FFT/moment/quantile/cross-cov/"
                        "locality losses. When > 0, set those to 0. Try 1.0–5.0.")
    p.add_argument("--pcf-n-freqs",         type=int,   default=32,
                   help="Number of learnable frequency vectors for PCF loss (default 32)")
    p.add_argument("--pcf-freq-scale",      type=float, default=0.1,
                   help="Initial scale of PCF frequency vectors (default 0.1)")
    p.add_argument("--diversity-loss-weight", type=float, default=0.0,
                   help="L_div: MSGAN mode-seeking loss — maximises output/noise distance ratio "
                        "across random pairs; combats mode collapse (low β-recall). "
                        "Requires a second G forward pass. Try 0.5–2.0.")
    p.add_argument("--feat-critic-weight", type=float, default=0.0,
                   help="Dual discriminator: weight of feature-space critic C_feat in G loss "
                        "(0=off). Requires latent_dim > 0. C_feat operates on decoded features "
                        "so it penalises quality problems invisible in latent space. Try 0.5–1.0.")
    p.add_argument("--continuity-loss-weight", type=float, default=0.0,
                   help="L_cont: boundary-continuity loss for multi-window coherence. "
                        "Generates two adjacent windows with carried LSTM hidden state "
                        "and penalises boundary mean/std mismatch. Targets DMD-GEN. "
                        "Try 0.5–1.0.")
    p.add_argument("--self-diag-temp",     type=float, default=0.0,
                   help="Self-diagnosing upweighting temperature (0=off). When > 0, real "
                        "samples with high critic scores (underrepresented modes) are "
                        "upweighted in critic training and feature matching via softmax. "
                        "Try 1.0–5.0. Compatible with existing pretrain. (NeurIPS 2021)")
    p.add_argument("--r1-lambda",          type=float, default=0.0,
                   help="R1 zero-centered GP on real samples (R3GAN NeurIPS 2024; 0=off)")
    p.add_argument("--r2-lambda",          type=float, default=0.0,
                   help="R2 zero-centered GP on fake samples (R3GAN NeurIPS 2024; 0=off)")
    p.add_argument("--dmd-ckpt-weight",    type=float, default=0.0,
                   help="Weight for DMD-GEN in checkpoint selection combined score (0=off; try 0.05). "
                        "combined = MMD² + 0.2*(1-recall) + dmd_ckpt_weight*DMD-GEN. "
                        "Adds ~5s per eval (5 DMD mini-batches). Recommended for v19+.")
    p.add_argument("--lr-er",                type=float, default=0.0005,
                   help="Learning rate for encoder + recovery")
    p.add_argument("--amp",                  action="store_true", default=True,
                   help="Enable AMP fp16 for 2-3× CUDA speedup (CUDA only; default on; "
                        "auto-disabled with wgan-gp/r1/r2 due to create_graph incompatibility)")
    p.add_argument("--no-amp",               dest="amp", action="store_false",
                   help="Disable AMP (useful for debugging)")
    p.add_argument("--num-lstm-layers",        type=int,   default=1,
                   help="Number of LSTM layers in Generator and Critic (IDEAS.md idea #11). "
                        "2-3 layers enable hierarchical temporal representations. "
                        "Requires fresh pretrain (architecture change). (default: 1)")
    p.add_argument("--n-regimes",             type=int,   default=0,
                   help="Regime-first two-stage generation (IDEAS.md idea #5): K workload "
                        "regime prototypes; Gumbel-Softmax selects one per window before G "
                        "generates. Temperature annealed τ: 1.0→0.1 over training. "
                        "Compatible with v28 pretrain. Requires cond_dim > 0. "
                        "Recommended: 8. (0 = off)")
    p.add_argument("--w-stop-threshold",     type=float, default=0.0,
                   help="W-distance spike guard: stop training if W-dist exceeds this value "
                        "for 3 consecutive epochs (0=off). Prevents the conditioning collapse "
                        "seen in v33/v36 (W→10 at ep178-179, well after EMA best was saved). "
                        "Recommended: 3.0. Good runs rarely exceed 2.0.")
    p.add_argument("--multi-scale-critic",   action="store_true", default=False,
                   help="Multi-resolution critic (IDEAS.md idea #8, HiFi-GAN style): "
                        "3 independent LSTM critics at T, T//2, T//4 temporal scales. "
                        "G loss = mean over scales. Compatible with v28 pretrain (critic "
                        "is not pretrained). Targets burst/envelope/regime discrimination.")
    p.add_argument("--mixed-type-recovery",  action="store_true", default=False,
                   help="Mixed-type output heads in Recovery (IDEAS.md idea #7): "
                        "binary columns (opcode, obj_id_reuse) use sigmoid→[-1,1] heads "
                        "with BCE reconstruction loss in Phase 1. Continuous columns keep "
                        "Tanh heads with MSE loss. Produces sharper ±1 values for binary "
                        "fields → lower MMD². Requires new pretrain (breaks v28 compat).")
    p.add_argument("--bayes-critics",        type=int, default=0,
                   help="BayesGAN: number of critic particles (0=off, 1=same as 0). "
                        "Each extra particle uses SGLD (Adam + Gaussian noise injection) "
                        "to approximate a posterior over discriminators. G loss is averaged "
                        "across all particles — no single boundary forces mode collapse. "
                        "Saatci & Wilson, NeurIPS 2017. Recommended: 5. "
                        "Requires wgan-sn (incompatible with wgan-gp/r1/r2). "
                        "Memory: M × ~1MB per particle (trivial on GB10 124GB).")
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
    cfg.cond_dim         = args.cond_dim
    cfg.cond_drop_prob   = args.cond_drop_prob
    cfg.hidden_size      = args.hidden_size
    cfg.compile          = args.compile
    cfg.minibatch_std    = not args.no_minibatch_std
    cfg.sn_lstm          = not args.no_sn_lstm
    cfg.patch_embed      = args.patch_embed
    cfg.proj_critic      = args.proj_critic
    cfg.film_cond        = args.film_cond
    cfg.pack_size        = args.pack_size
    cfg.seed             = args.seed
    cfg.lr_g             = args.lr_g
    cfg.lr_d             = args.lr_d
    cfg.n_critic         = args.n_critic
    cfg.max_records      = args.max_records
    cfg.files_per_epoch  = args.files_per_epoch
    cfg.block_sample     = args.block_sample
    cfg.records_per_file = args.records_per_file
    cfg.char_file        = args.char_file or ""
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
    cfg.copy_path                   = args.copy_path
    cfg.reuse_bce_weight            = args.reuse_bce_weight
    cfg.stride_consistency_weight   = args.stride_consistency_weight
    cfg.pcf_loss_weight             = args.pcf_loss_weight
    cfg.pcf_n_freqs                 = args.pcf_n_freqs
    cfg.pcf_freq_scale              = args.pcf_freq_scale
    cfg.diversity_loss_weight       = args.diversity_loss_weight
    cfg.feat_critic_weight          = args.feat_critic_weight
    cfg.continuity_loss_weight      = args.continuity_loss_weight
    cfg.self_diag_temp              = args.self_diag_temp
    cfg.r1_lambda                   = args.r1_lambda
    cfg.r2_lambda                   = args.r2_lambda
    cfg.dmd_ckpt_weight             = args.dmd_ckpt_weight
    cfg.gmm_components              = args.gmm_components
    cfg.var_cond                    = args.var_cond
    cfg.var_cond_kl_weight          = args.var_cond_kl_weight
    cfg.latent_dim              = args.latent_dim
    cfg.pretrain_ae_epochs      = args.pretrain_ae_epochs
    cfg.pretrain_sup_epochs     = args.pretrain_sup_epochs
    cfg.pretrain_g_epochs       = args.pretrain_g_epochs
    cfg.supervisor_loss_weight  = args.supervisor_loss_weight
    cfg.supervisor_steps        = args.supervisor_steps
    cfg.lr_er                   = args.lr_er
    cfg.amp                     = args.amp
    cfg.n_regimes               = args.n_regimes
    cfg.num_lstm_layers         = args.num_lstm_layers
    cfg.w_stop_threshold        = args.w_stop_threshold
    cfg.multi_scale_critic      = args.multi_scale_critic
    cfg.mixed_type_recovery     = args.mixed_type_recovery
    cfg.bayes_critics           = args.bayes_critics
    cfg.avatar                  = args.avatar
    cfg.dist_loss_weight        = args.dist_loss_weight
    cfg.latent_disc_weight      = args.latent_disc_weight
    # AVATAR forces 2-step supervisor
    if cfg.avatar:
        cfg.supervisor_steps = 2
    return cfg


if __name__ == "__main__":
    # MPS backend emits a benign buffer-resize warning on rfft; suppress it.
    warnings.filterwarnings("ignore", message=".*resized since it had shape.*")
    cfg = parse_args()
    train(cfg)
