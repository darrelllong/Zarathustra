import argparse
import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from llgan.config import Config
from llgan.dataset import load_trace
from llgan.model import Generator, Critic

# Simplified run script – load a checkpoint and generate synthetic windows.
# It writes a CSV with the same column layout as the original trace.


def generate_synthetic(
    ckpt_path: str,
    num_windows: int,
    cfg: Config,
    device: torch.device,
):
    G = Generator(cfg).to(device)
    G.load_state_dict(torch.load(ckpt_path, map_location=device)["G_state"])
    G.eval()
    with torch.no_grad():
        z = torch.randn(num_windows, cfg.noise_dim, device=device)
        fake = G(z)
        # Convert to CPU numpy
        fake_np = fake.cpu().numpy()
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(fake_np)
    output_path = cfg.generated_path
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic trace written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IO trace")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-windows", type=int, default=100000, help="Number of windows to generate")
    parser.add_argument("--cfg", default="llgan/config.py", help="Config path")
    args = parser.parse_args()

    cfg = Config()
    cfg.generated_path = args.cfg  # reuse default or override if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_synthetic(args.ckpt, args.num_windows, cfg, device)


if __name__ == "__main__":
    main()
