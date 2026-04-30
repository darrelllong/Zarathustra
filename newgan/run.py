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
    prep_cols: int = None,
):
    """Generate synthetic windows from a trained checkpoint.

    Args:
        ckpt_path: Path to model checkpoint (.pt file)
        num_windows: Number of windows to generate
        cfg: Config object with model hyperparameters
        device: Torch device for generation
        prep_cols: Number of columns in preprocessor (if None, reads from checkpoint)
    """
    # Load checkpoint to get model config
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["G_state"]

    # Extract num_cols from state dict keys (first weight matrix shape)
    # Format: 'layers.0.weight' or similar
    first_weight = None
    for k in state_dict.keys():
        if 'weight' in k and state_dict[k].dim() == 2:
            first_weight = state_dict[k]
            break

    if first_weight is not None:
        # weight shape: (out_features, in_features)
        # out_features is typically num_cols * timestep for generator output
        # For latent mode, it's latent_dim * timestep
        num_cols = first_weight.shape[0] // getattr(cfg, 'timestep', 12)
    else:
        # Fallback
        num_cols = prep_cols or 5

    # Initialize Generator with correct signature
    G = Generator(
        noise_dim=getattr(cfg, 'noise_dim', 10),
        num_cols=num_cols,
        hidden_size=getattr(cfg, 'hidden_size', 256),
        latent_dim=getattr(cfg, 'latent_dim', 24),
        cond_dim=getattr(cfg, 'cond_dim', 0),
        film_cond=False,
        gmm_components=0,
        var_cond=False,
        n_regimes=0,
        num_lstm_layers=1,
        gp_prior=False,
    ).to(device)

    G.load_state_dict(state_dict)
    G.eval()
    with torch.no_grad():
        z_global = torch.randn(num_windows, getattr(cfg, 'noise_dim', 10), device=device)
        z_local = torch.randn(num_windows, getattr(cfg, 'timestep', 12), getattr(cfg, 'noise_dim', 10), device=device)
        fake = G(z_global, z_local)
        # Convert to CPU numpy
        fake_np = fake.cpu().numpy()
    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(fake_np)
    output_path = getattr(cfg, 'generated_path', 'generated.csv')
    os.makedirs(Path(output_path).parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Synthetic trace written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic IO trace")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-windows", type=int, default=100000, help="Number of windows to generate")
    parser.add_argument("--output", default="generated.csv", help="Output CSV path")
    args = parser.parse_args()

    cfg = Config()
    cfg.generated_path = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_synthetic(args.ckpt, args.num_windows, cfg, device)


if __name__ == "__main__":
    main()
