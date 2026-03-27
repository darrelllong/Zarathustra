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

from model import Generator


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

    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    prep = ckpt["prep"]

    G = Generator(cfg.noise_dim, prep.num_cols, cfg.hidden_size).to(device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # Identify opcode column index (for binarization)
    opcode_col = -1
    for i, col in enumerate(prep.col_names):
        if col.lower() in {"opcode", "type", "rw", "op"}:
            opcode_col = i
            break

    timestep = cfg.timestep
    # Each stream generates ceil(n_records / n_streams) records
    records_per_stream = math.ceil(n_records / n_streams)
    windows_per_stream = math.ceil(records_per_stream / timestep)

    all_records = []

    with torch.no_grad():
        # Each stream is an independent long trace.
        # z_global is fixed per stream (workload identity); LSTM hidden state
        # is carried across windows so long-range burst structure is coherent.
        z_global = torch.randn(n_streams, cfg.noise_dim, device=device)
        hidden = None   # initialised from z_global on first window

        for _ in range(windows_per_stream):
            z_local = torch.randn(
                n_streams, timestep, cfg.noise_dim, device=device
            )
            out, hidden = G(z_global, z_local, hidden=hidden, return_hidden=True)
            # Detach hidden to avoid accumulating the full computation graph
            hidden = (hidden[0].detach(), hidden[1].detach())

            if binarize_opcode and opcode_col >= 0:
                out[:, :, opcode_col] = (
                    (out[:, :, opcode_col] >= 0).float() * 2 - 1
                )

            # (n_streams, timestep, num_cols) → (n_streams * timestep, num_cols)
            all_records.append(out.cpu().numpy().reshape(-1, prep.num_cols))

    arr = np.concatenate(all_records, axis=0)[:n_records]
    df = prep.inverse_transform(arr)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate I/O traces from LLGAN")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--n", type=int, default=1_000_000, help="Number of records")
    p.add_argument("--output", default="generated.csv")
    p.add_argument("--n-streams", type=int, default=1,
                   help="Number of parallel independent trace streams to generate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-binarize-opcode", action="store_true")
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
