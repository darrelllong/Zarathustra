"""
Generate synthetic I/O traces from a trained LLGAN checkpoint.

Usage
-----
    python generate.py \
        --checkpoint checkpoints/tencent_v4/best.pt \
        --n 1000000 \
        --output synthetic_tencent.csv

    # Generate enough windows to fill N records, 4 parallel trace streams
    python generate.py --checkpoint checkpoints/tencent_v4/best.pt \
        --n 1000000 --n-streams 4
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset import load_file_characterizations
from model import Generator, Recovery


def generate(
    checkpoint_path: str,
    n_records: int,
    output_path: str,
    n_streams: int = 1,
    device_str: str = "cuda",
    binarize_opcode: bool = True,
    char_file: str = "",
    source_trace: str = "",
    retrieval_persist_across_windows: bool = False,
    lru_stack_decoder: bool = False,
    lru_stack_corpus: str = "alibaba",
    lru_stack_real_csv: str = "",
    lru_stack_exact_fit: bool = False,
    lru_stack_pmf: str = "",
    lru_stack_reuse_rate: float = -1.0,
    lru_stack_per_stream_rates: str = "",
    lru_stack_max_depth: int = 2048,
    lru_markov_atlas: str = "",
    lru_markov_blend: float = 1.0,
    lru_cond_pmf: str = "",
    lru_cond_pmf_edges: str = "",
    lru_markov_p_rr: float = -1.0,
    lru_markov_p_mr: float = -1.0,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    prep = ckpt["prep"]

    latent_ae = "R" in ckpt  # latent AE checkpoint includes recovery weights
    latent_dim = getattr(cfg, "latent_dim", 0) if latent_ae else None

    cond_dim = getattr(cfg, "cond_dim", 0)
    # Mirror eval.py: pass every cfg-driven Generator kwarg so modern
    # checkpoints (regime sampler, GMM prior, FiLM, multi-LSTM, GP prior)
    # load cleanly. Without this, generate.py silently builds a smaller
    # Generator and load_state_dict raises on shape mismatch.
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=latent_dim, cond_dim=cond_dim,
                  film_cond=getattr(cfg, "film_cond", False),
                  gmm_components=getattr(cfg, "gmm_components", 0),
                  var_cond=getattr(cfg, "var_cond", False),
                  n_regimes=getattr(cfg, "n_regimes", 0),
                  num_lstm_layers=getattr(cfg, "num_lstm_layers", 1),
                  gp_prior=getattr(cfg, "gp_prior", False),
                  timestep=cfg.timestep,
                  retrieval_memory=getattr(cfg, "retrieval_memory", False),
                  retrieval_mem_size=getattr(cfg, "retrieval_mem_size", 32),
                  retrieval_key_dim=getattr(cfg, "retrieval_key_dim", 32),
                  retrieval_val_dim=getattr(cfg, "retrieval_val_dim", 32),
                  retrieval_decay=getattr(cfg, "retrieval_decay", 0.85),
                  retrieval_tau_write=getattr(cfg, "retrieval_tau_write", 0.5),
                  retrieval_n_warmup=getattr(cfg, "retrieval_n_warmup", 4),
                  ssm_backbone=getattr(cfg, "ssm_backbone", False),
                  ssm_state_dim=getattr(cfg, "ssm_state_dim", 16),
                  mtpp_timing=getattr(cfg, "mtpp_timing", False),
                  mtpp_sigma_min=getattr(cfg, "mtpp_sigma_min", 0.05),
                  ).to(device)
    # Prefer EMA weights for generation: they are the time-averaged model that
    # has been smoothed over recent training oscillations and consistently
    # produce better samples than the instantaneous live weights.
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        # Detect mixed-type-recovery from checkpoint keys (matches eval.py).
        r_keys = ckpt["R"].keys()
        binary_cols = None
        if any(k.startswith("fc_cont") for k in r_keys):
            binary_cols = [i for i, col in enumerate(prep.col_names)
                           if col.lower() in {"opcode", "type", "rw", "op"}
                           or col.endswith("_reuse")]
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols,
                     binary_cols=binary_cols).to(device)
        R.load_state_dict(ckpt["R"])
        R.eval()

    # Identify opcode column index (for binarization, applied in feature space)
    opcode_col = -1
    for i, col in enumerate(prep.col_names):
        if col.lower() in {"opcode", "type", "rw", "op"}:
            opcode_col = i
            break

    timestep = cfg.timestep
    # Each stream generates ceil(n_records / n_streams) records
    records_per_stream = math.ceil(n_records / n_streams)
    windows_per_stream = math.ceil(records_per_stream / timestep)

    # Collect raw (pre-inverse-transform) windows per stream separately.
    # Streams must NOT be merged before inverse_transform: timestamp and
    # obj_id reconstruction both apply cumsum, which must stay within
    # each stream's sequence boundary.
    stream_windows: list[list[np.ndarray]] = [[] for _ in range(n_streams)]

    # Build conditioning vector for generation.
    # Priority: (1) lookup source_trace in char_file, (2) random N(0,0.5).
    cond_vec: Optional[torch.Tensor] = None
    if cond_dim > 0 and char_file and source_trace:
        lookup = load_file_characterizations(char_file, cond_dim)
        key = Path(source_trace).name
        cond_vec = lookup.get(key)
        if cond_vec is None:
            for ext in (".zst", ".gz"):
                if key.endswith(ext):
                    cond_vec = lookup.get(key[: -len(ext)])
                    if cond_vec is not None:
                        break
        if cond_vec is not None:
            print(f"Using precharacterized conditioning for {key}")
            cond_vec = cond_vec.unsqueeze(0).expand(n_streams, -1).to(device)
        else:
            print(f"[warn] {key} not found in {char_file}; using random conditioning")

    with torch.no_grad():
        # Each stream is an independent long trace.
        # z_global is fixed per stream (workload identity); LSTM hidden state
        # is carried across windows so long-range burst structure is coherent.
        if cond_dim > 0:
            if cond_vec is not None:
                cond = cond_vec
            else:
                cond = torch.randn(n_streams, cond_dim, device=device) * 0.5
            # Mirror _make_z_global(training=False) from train.py:
            # cond_encoder → regime_sampler → gmm_prior then cat with noise.
            if getattr(G, 'cond_encoder', None) is not None:
                cond, _ = G.cond_encoder(cond, training=False)
            if getattr(G, 'regime_sampler', None) is not None:
                cond = G.regime_sampler(cond)
            noise = G.sample_noise(n_streams, device, cond=cond)
            z_global = torch.cat([cond, noise], dim=1)
        else:
            z_global = torch.randn(n_streams, cfg.noise_dim, device=device)
        hidden = None   # initialised from z_global on first window
        # IDEA #28: optionally thread retrieval-memory bank state across window
        # boundaries. Off by default (matches IDEA #17 per-window re-init).
        retrieval_enabled = getattr(G, "retrieval", None) is not None
        persist_retrieval = retrieval_persist_across_windows and retrieval_enabled
        if retrieval_persist_across_windows and not retrieval_enabled:
            print("[warn] --retrieval-persist-across-windows set but checkpoint "
                  "has no retrieval memory; flag ignored.")
        retrieval_state = None

        for _ in range(windows_per_stream):
            z_local = torch.randn(
                n_streams, timestep, cfg.noise_dim, device=device
            )
            if persist_retrieval:
                latent, hidden, retrieval_state = G(
                    z_global, z_local, hidden=hidden,
                    return_hidden=True,
                    retrieval_state=retrieval_state,
                    return_retrieval_state=True,
                )
                # Detach bank tensors so autograd graph doesn't grow across
                # windows (consistent with hidden-state handling below).
                retrieval_state = {
                    k: (v.detach() if torch.is_tensor(v) else v)
                    for k, v in retrieval_state.items()
                }
            else:
                latent, hidden = G(z_global, z_local, hidden=hidden,
                                   return_hidden=True)
            # Detach hidden to avoid accumulating the full computation graph.
            # SSM backbone returns (state, None); LSTM returns (h, c). Guard
            # the second element so either backbone carries cleanly across
            # windows (mirrors long_rollout_eval._rollout).
            h0 = hidden[0].detach() if hidden[0] is not None else None
            h1 = hidden[1].detach() if hidden[1] is not None else None
            hidden = (h0, h1)

            # Decode latents to feature space (latent AE mode only)
            out = R(latent) if R is not None else latent

            if binarize_opcode and opcode_col >= 0:
                out[:, :, opcode_col] = (
                    (out[:, :, opcode_col] >= 0).float() * 2 - 1
                )

            # (n_streams, timestep, num_cols) — split by stream before storing
            out_np = out.cpu().numpy()   # (n_streams, timestep, num_cols)
            for s in range(n_streams):
                stream_windows[s].append(out_np[s])  # (timestep, num_cols)

    # IDEA #48: post-hoc LRU stack decoder.
    # obj_id_reuse is only in the normalized feature space (before inverse_transform),
    # so build the decoder here and apply it per-stream before inverse_transform.
    lru_decoder_proto = None
    reuse_col_idx = -1
    if lru_stack_decoder:
        from lru_stack_decoder import LRUStackDecoder
        col_names = getattr(prep, "col_names", None)
        if col_names and "obj_id_reuse" in col_names:
            reuse_col_idx = list(col_names).index("obj_id_reuse")
        if reuse_col_idx < 0:
            print("[lru-stack] WARN: obj_id_reuse column not in preprocessor col_names; "
                  "decoder disabled")
        else:
            if lru_cond_pmf and lru_cond_pmf_edges:
                C = np.load(lru_cond_pmf)
                E = np.load(lru_cond_pmf_edges)
                print(f"[lru-stack] Cond PMF (IDEA #63) loaded: shape={C.shape}, "
                      f"n_bins={len(C)}, edges={np.round(E, 3)}")
                lru_decoder_proto = LRUStackDecoder.from_cond_pmf(
                    C, E, corpus=lru_stack_corpus, max_stack_depth=lru_stack_max_depth)
                lru_decoder_proto.print_pmf()
            elif lru_markov_atlas:
                T = np.load(lru_markov_atlas)
                print(f"[lru-stack] Markov atlas loaded from {lru_markov_atlas} "
                      f"shape={T.shape}, blend={lru_markov_blend:.2f}")
                lru_decoder_proto = LRUStackDecoder.from_markov_matrix(
                    T, corpus=lru_stack_corpus, max_stack_depth=lru_stack_max_depth)
                lru_decoder_proto.markov_blend = lru_markov_blend
                lru_decoder_proto.print_pmf()
                lru_decoder_proto.print_transition_matrix()
            elif lru_stack_pmf:
                pmf_vals = np.array([float(x) for x in lru_stack_pmf.split(",")])
                print(f"[lru-stack] Using explicit PMF: {np.round(pmf_vals, 4)}")
                lru_decoder_proto = LRUStackDecoder(
                    pmf_vals, max_stack_depth=lru_stack_max_depth,
                    markov_p_rr=lru_markov_p_rr, markov_p_mr=lru_markov_p_mr)
            elif lru_stack_real_csv:
                import pandas as _pd
                real_df = _pd.read_csv(lru_stack_real_csv)
                if "obj_id" not in real_df.columns:
                    print(f"[lru-stack] WARN: no obj_id in {lru_stack_real_csv}; "
                          "using default PMF")
                    lru_decoder_proto = LRUStackDecoder.from_default(lru_stack_corpus,
                                                                      max_stack_depth=lru_stack_max_depth)
                else:
                    print(f"[lru-stack] Fitting PMF from {lru_stack_real_csv} "
                          f"(exact={lru_stack_exact_fit})")
                    lru_decoder_proto = LRUStackDecoder.fit_from_df(
                        real_df, exact=lru_stack_exact_fit,
                        max_stack_depth=lru_stack_max_depth
                    )
            else:
                print(f"[lru-stack] Using default PMF for corpus={lru_stack_corpus!r}")
                lru_decoder_proto = LRUStackDecoder.from_default(lru_stack_corpus,
                                                                  max_stack_depth=lru_stack_max_depth)
            # IDEA #93: propagate Markov parameters to prototype (non-explicit-PMF paths)
            if lru_markov_p_rr >= 0.0 and lru_markov_p_mr >= 0.0:
                lru_decoder_proto.markov_p_rr = lru_markov_p_rr
                lru_decoder_proto.markov_p_mr = lru_markov_p_mr
            lru_decoder_proto.print_pmf()

    # IDEA #91: per-stream Bernoulli reuse rates.
    # --lru-stack-per-stream-rates "r0,r1,r2,r3" gives each stream its own rate.
    _lru_per_stream_rates: list = []
    if lru_stack_per_stream_rates:
        _lru_per_stream_rates = [float(x) for x in lru_stack_per_stream_rates.split(",")]
        if _lru_per_stream_rates:
            print(f"[lru-stack] IDEA #91 per-stream rates: {_lru_per_stream_rates}")

    # Inverse-transform each stream independently (cumsum stays within stream),
    # then label and concatenate for a single output file.
    import pandas as pd
    per_stream_dfs = []
    for s in range(n_streams):
        arr_s = np.concatenate(stream_windows[s], axis=0)[:records_per_stream]

        # Apply LRU stack decoder before inverse_transform: obj_id_reuse lives
        # in the normalized feature array. Decoder returns integer obj_ids.
        lru_obj_ids = None
        if lru_decoder_proto is not None and reuse_col_idx >= 0:
            T_copy = (lru_decoder_proto.transition_matrix.copy()
                      if lru_decoder_proto.transition_matrix is not None else None)
            C_copy = (lru_decoder_proto.cond_pmf.copy()
                      if lru_decoder_proto.cond_pmf is not None else None)
            E_copy = (lru_decoder_proto.dt_edges.copy()
                      if lru_decoder_proto.dt_edges is not None else None)
            dec = type(lru_decoder_proto)(
                lru_decoder_proto.bucket_pmf.copy(),
                lru_decoder_proto.max_stack_depth,
                transition_matrix=T_copy,
                cond_pmf=C_copy,
                dt_edges=E_copy,
                markov_p_rr=lru_decoder_proto.markov_p_rr,
                markov_p_mr=lru_decoder_proto.markov_p_mr,
            )
            dec.markov_blend = lru_decoder_proto.markov_blend
            reuse_signal = arr_s[:, reuse_col_idx]
            # IDEA #93: Markov reuse model overrides is_reuse signal inside decoder.
            # When --lru-stack-markov-prr and --lru-stack-markov-pmr are set,
            # the decoder's step() uses its internal Markov chain; reuse_signal
            # is passed but ignored. Bernoulli override also disabled when Markov active.
            if lru_markov_p_rr >= 0.0:
                if s == 0:
                    pi = lru_markov_p_mr / max(1e-9, 1 - lru_markov_p_rr + lru_markov_p_mr)
                    print(f"[lru-stack] IDEA #93 Markov reuse: p_rr={lru_markov_p_rr:.4f}, "
                          f"p_mr={lru_markov_p_mr:.4f}, π_reuse={pi:.4f}")
            # Override reuse signal with Bernoulli(p) if --lru-stack-reuse-rate
            # is set and Markov model is not active.
            # Supports comma-separated per-stream rates (IDEA #91): "r0,r1,r2,r3"
            if lru_stack_reuse_rate >= 0.0 and lru_markov_p_rr < 0.0:
                ruse_rng = np.random.default_rng(42 + s)
                reuse_signal = np.where(
                    ruse_rng.random(len(reuse_signal)) < lru_stack_reuse_rate,
                    1.0, -1.0
                )
            elif _lru_per_stream_rates and lru_markov_p_rr < 0.0:
                rate_s = _lru_per_stream_rates[s % len(_lru_per_stream_rates)]
                ruse_rng = np.random.default_rng(42 + s)
                reuse_signal = np.where(
                    ruse_rng.random(len(reuse_signal)) < rate_s,
                    1.0, -1.0
                )
            # IDEA #63: extract log1p(ts_delta) from normalized ts column
            dt_log1p_signal = None
            if dec.cond_pmf is not None and dec.dt_edges is not None:
                col_names = getattr(prep, "col_names", [])
                ts_idx = list(col_names).index("ts") if "ts" in col_names else -1
                if ts_idx >= 0:
                    ts_stats = getattr(prep, "_stats", {}).get("ts", {})
                    lo = ts_stats.get("lo", 0.0)
                    hi = ts_stats.get("hi", 1.0)
                    span = hi - lo if hi > lo else 1.0
                    # Undo min-max normalization to get log1p(ts_delta)
                    dt_log1p_signal = (arr_s[:, ts_idx] + 1) / 2 * span + lo
            lru_obj_ids = dec.decode_stream(reuse_signal, dt_log1p=dt_log1p_signal)

        df_s = prep.inverse_transform(arr_s)
        df_s.insert(0, "stream_id", s)

        if lru_obj_ids is not None and "obj_id" in df_s.columns:
            df_s["obj_id"] = lru_obj_ids
            if s == 0:
                reuse_rate = float((arr_s[:, reuse_col_idx] > 0).mean())
                print(f"[lru-stack] stream 0: {len(lru_obj_ids):,} events, "
                      f"{len(np.unique(lru_obj_ids)):,} unique objects, "
                      f"reuse_rate={reuse_rate:.3f}")

        per_stream_dfs.append(df_s)

    df = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)

    # Cast integer-category columns to int so mark_quality TV scores
    # compare "1" vs "1" (not "1.0" vs "1" from inverse_transform float output).
    for _col in ("opcode", "tenant"):
        if _col in df.columns:
            try:
                df[_col] = df[_col].round().astype(int)
            except (ValueError, TypeError):
                pass

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records ({n_streams} stream(s)) → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate I/O traces from LLGAN")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--n", type=int, default=1_000_000, help="Number of records")
    p.add_argument("--output", default="generated.csv")
    p.add_argument("--n-streams", type=int, default=1,
                   help="Number of parallel independent trace streams to generate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-binarize-opcode", action="store_true")
    p.add_argument("--cond-random", action="store_true",
                   help="Use random conditioning N(0,0.5) for conditional checkpoints (default)")
    p.add_argument("--char-file", default="",
                   metavar="JSONL",
                   help="Path to trace_characterizations.jsonl. Used with --source-trace "
                        "to condition generation on a specific workload's characteristics.")
    p.add_argument("--source-trace", default="",
                   metavar="FILE",
                   help="Trace filename (basename) whose precharacterized stats are used "
                        "as the conditioning vector. Requires --char-file.")
    p.add_argument("--retrieval-persist-across-windows", action="store_true",
                   help="IDEA #28: thread the retrieval-memory bank (K/V/T/mask) "
                        "across window boundaries during generation instead of "
                        "re-initialising each window. No-op if the checkpoint "
                        "has no retrieval memory.")
    # IDEA #48: post-hoc LRU stack decoder
    p.add_argument("--lru-stack-decoder", action="store_true",
                   help="IDEA #48: replace generated obj_id stream with an explicit "
                        "LRU stack decoder. Uses generator's obj_id_reuse signal and "
                        "a corpus-fitted stack-distance bucket PMF.")
    p.add_argument("--lru-stack-corpus", default="alibaba",
                   choices=["alibaba", "tencent"],
                   help="Corpus for default stack-distance PMF when no real CSV supplied.")
    p.add_argument("--lru-stack-real-csv", default="",
                   metavar="CSV",
                   help="Path to real trace CSV with obj_id column for PMF fitting. "
                        "If omitted, uses --lru-stack-corpus default PMF.")
    p.add_argument("--lru-stack-exact-fit", action="store_true",
                   help="Use BIT-based exact stack distances for PMF fitting (slower, "
                        "more accurate). Default: IRD approximation.")
    p.add_argument("--lru-stack-pmf", default="",
                   metavar="P0,P1,...,P7",
                   help="Explicit 8-value comma-separated PMF for stack-distance buckets "
                        "[0,1),[1,2),[2,4),[4,8),[8,16),[16,64),[64,256),[256+). "
                        "Overrides --lru-stack-corpus and --lru-stack-real-csv.")
    p.add_argument("--lru-stack-reuse-rate", type=float, default=-1.0,
                   metavar="P",
                   help="Override generator's obj_id_reuse signal with Bernoulli(P). "
                        "P=-1 (default) uses generator signal. Use P=real_reuse_rate "
                        "to ablate the reuse signal and test stack decoder alone.")
    p.add_argument("--lru-stack-max-depth", type=int, default=2048,
                   metavar="D",
                   help="LRU stack maximum depth. Objects beyond this rank are evicted. "
                        "Default 2048. Set >= footprint (n_records*(1-reuse_rate)) to "
                        "avoid capping the working set.")
    p.add_argument("--lru-markov-atlas", default="",
                   metavar="NPY",
                   help="IDEA #62: path to precomputed 8×8 Markov transition matrix "
                        "(.npy file from compute_markov_atlas.py). When set, the LRU "
                        "stack decoder samples bucket from P(next|prev) instead of the "
                        "global PMF. Requires --lru-stack-decoder.")
    p.add_argument("--lru-markov-blend", type=float, default=1.0,
                   metavar="ALPHA",
                   help="IDEA #62: blend coefficient for Markov atlas. "
                        "0.0=pure i.i.d. PMF (baseline), 1.0=pure Markov, "
                        "intermediate=weighted average. Default 1.0 (pure Markov).")
    p.add_argument("--lru-cond-pmf", default="",
                   metavar="NPY",
                   help="IDEA #63: path to (N,8) conditional PMF array "
                        "P(bucket|dt_bin) from compute_cond_pmf.py. "
                        "Requires --lru-cond-pmf-edges and --lru-stack-decoder.")
    p.add_argument("--lru-cond-pmf-edges", default="",
                   metavar="NPY",
                   help="IDEA #63: path to (N-1,) dt bin edges array "
                        "from compute_cond_pmf.py.")
    p.add_argument("--lru-stack-per-stream-rates", default="",
                   metavar="R0,R1,...",
                   help="IDEA #91: comma-separated Bernoulli reuse rates per stream. "
                        "e.g. '0.606,0.704,0.590,0.559' for 4 streams. "
                        "Overrides --lru-stack-reuse-rate when provided.")
    p.add_argument("--lru-stack-markov-prr", type=float, default=-1.0,
                   metavar="P",
                   help="IDEA #93: P(reuse | prev was reuse) for 2-state Markov reuse chain. "
                        "Set with --lru-stack-markov-pmr to enable Markov reuse model. "
                        "Replaces i.i.d. Bernoulli override. -1 (default) = disabled.")
    p.add_argument("--lru-stack-markov-pmr", type=float, default=-1.0,
                   metavar="P",
                   help="IDEA #93: P(reuse | prev was miss) for 2-state Markov reuse chain. "
                        "Stationary rate: pi=pmr/(1-prr+pmr). -1 (default) = disabled.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        args.checkpoint,
        args.n,
        args.output,
        args.n_streams,
        args.device,
        not args.no_binarize_opcode,
        char_file=args.char_file,
        source_trace=args.source_trace,
        retrieval_persist_across_windows=args.retrieval_persist_across_windows,
        lru_stack_decoder=args.lru_stack_decoder,
        lru_stack_corpus=args.lru_stack_corpus,
        lru_stack_real_csv=args.lru_stack_real_csv,
        lru_stack_exact_fit=args.lru_stack_exact_fit,
        lru_stack_pmf=args.lru_stack_pmf,
        lru_stack_reuse_rate=args.lru_stack_reuse_rate,
        lru_stack_per_stream_rates=args.lru_stack_per_stream_rates,
        lru_stack_max_depth=args.lru_stack_max_depth,
        lru_markov_atlas=args.lru_markov_atlas,
        lru_markov_blend=args.lru_markov_blend,
        lru_cond_pmf=args.lru_cond_pmf,
        lru_cond_pmf_edges=args.lru_cond_pmf_edges,
        lru_markov_p_rr=args.lru_stack_markov_prr,
        lru_markov_p_mr=args.lru_stack_markov_pmr,
    )
