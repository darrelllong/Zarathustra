"""
Generate synthetic I/O traces from a trained LLGAN checkpoint.

Usage
-----
    python generate.py \
        --checkpoint checkpoints/msr/final.pt \
        --n 1000000 \
        --output synthetic_msr.csv

    # Generate enough windows to fill N records
    python generate.py --checkpoint checkpoints/msr/final.pt --n 1000000
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
    batch_size: int = 512,
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
    n_windows = math.ceil(n_records / timestep)
    all_windows = []

    with torch.no_grad():
        for start in range(0, n_windows, batch_size):
            n = min(batch_size, n_windows - start)
            z = torch.randn(n, timestep, cfg.noise_dim, device=device)
            out = G(z)                          # (n, timestep, num_cols)

            if binarize_opcode and opcode_col >= 0:
                out[:, :, opcode_col] = (
                    (out[:, :, opcode_col] >= 0).float() * 2 - 1
                )

            all_windows.append(out.cpu().numpy())

    # Flatten windows → records
    arr = np.concatenate(all_windows, axis=0)  # (n_windows, timestep, num_cols)
    arr = arr.reshape(-1, prep.num_cols)[:n_records]

    df = prep.inverse_transform(arr)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} records → {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generate I/O traces from LLGAN")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--n", type=int, default=1_000_000, help="Number of records")
    p.add_argument("--output", default="generated.csv")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-binarize-opcode", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        args.checkpoint,
        args.n,
        args.output,
        args.batch_size,
        args.device,
        not args.no_binarize_opcode,
    )
