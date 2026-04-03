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

import numpy as np
import torch

from model import Generator, Recovery


def generate(
    checkpoint_path: str,
    n_records: int,
    output_path: str,
    n_streams: int = 1,
    device_str: str = "cuda",
    binarize_opcode: bool = True,
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
    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size,
                  latent_dim=latent_dim,
                  cond_dim=cond_dim).to(device)
    # Prefer EMA weights for generation: they are the time-averaged model that
    # has been smoothed over recent training oscillations and consistently
    # produce better samples than the instantaneous live weights.
    g_weights = ckpt.get("G_ema", ckpt["G"])
    G.load_state_dict({k: v.to(device) for k, v in g_weights.items()})
    G.eval()

    R = None
    if latent_ae:
        R = Recovery(latent_dim, cfg.hidden_size, prep.num_cols).to(device)
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

    with torch.no_grad():
        # Each stream is an independent long trace.
        # z_global is fixed per stream (workload identity); LSTM hidden state
        # is carried across windows so long-range burst structure is coherent.
        noise = torch.randn(n_streams, cfg.noise_dim, device=device)
        if cond_dim > 0:
            # Default: generic conditioning from N(0, 0.5)
            cond = torch.randn(n_streams, cond_dim, device=device) * 0.5
            z_global = torch.cat([cond, noise], dim=1)
        else:
            z_global = noise
        hidden = None   # initialised from z_global on first window

        for _ in range(windows_per_stream):
            z_local = torch.randn(
                n_streams, timestep, cfg.noise_dim, device=device
            )
            latent, hidden = G(z_global, z_local, hidden=hidden, return_hidden=True)
            # Detach hidden to avoid accumulating the full computation graph
            hidden = (hidden[0].detach(), hidden[1].detach())

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

    # Inverse-transform each stream independently (cumsum stays within stream),
    # then label and concatenate for a single output file.
    import pandas as pd
    per_stream_dfs = []
    for s in range(n_streams):
        arr_s = np.concatenate(stream_windows[s], axis=0)[:records_per_stream]
        df_s = prep.inverse_transform(arr_s)
        df_s.insert(0, "stream_id", s)
        per_stream_dfs.append(df_s)

    df = pd.concat(per_stream_dfs, ignore_index=True).head(n_records)
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
    )
