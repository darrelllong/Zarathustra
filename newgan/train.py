"""
Training script for the new synthetic IO workload generator.

The script is intentionally minimal: it loads a single trace file, builds a
`TraceDataset` from `llgan.dataset`, and runs a very small WGAN‑SN style
training loop using the `Generator` and `Critic` models from the `llgan`
package.

The goal is to get a checkpoint that can be used by ``newgan/run.py`` to
generate synthetic traces.  It is not a full‑featured training harness – for a
real race you’ll want to add checkpointing, learning‑rate schedules, etc.
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# The heavy LLGAN machinery lives in the llgan package.
# We import only what we need for this toy training script.
from llgan.config import Config
from llgan.dataset import load_trace, TraceDataset
from llgan.model import Generator, Critic

# ---------------------------------------------------------------------------
# Helper: simple WGAN‑SN training loop.
# ---------------------------------------------------------------------------

def wgan_sn_train(
    G: nn.Module,
    D: nn.Module,
    dloader: DataLoader,
    cfg: Config,
    device: torch.device,
):
    """Very small training loop – not production ready.

    Parameters
    ----------
    G, D
        Generator and Critic.
    dloader
        DataLoader over windows (each window shape ``[T, D]``).
    cfg
        Hyper‑parameters.
    device
        ``torch.device``.
    """
    # Optimizers
    optG = optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.9))
    optD = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.9))

    G.train(); D.train()

    for epoch in range(cfg.epochs):
        start = time.time()
        for i, windows in enumerate(dloader):
            # windows: [B, T, D]
            windows = windows.to(device)
            # ------------------ Critic update ------------------
            for _ in range(cfg.n_critic):
                optD.zero_grad()
                # real
                real_score = D(windows)
                # fake
                z = torch.randn(windows.size(0), cfg.noise_dim, device=device)
                fake = G(z)
                fake_score = D(fake.detach())
                # Wasserstein loss + spectral norm penalty
                loss_D = fake_score.mean() - real_score.mean()
                # Spectral norm penalty on the last linear layer
                if hasattr(D, "fc"):
                    penalty = 10.0 * (D.fc.weight.norm() ** 2)
                else:
                    penalty = 0.0
                loss_D += penalty
                loss_D.backward()
                optD.step()

            # ------------------ Generator update ------------------
            optG.zero_grad()
            fake_score = D(fake)
            loss_G = -fake_score.mean()
            loss_G.backward()
            optG.step()

        elapsed = time.time() - start
        print(
            f"Epoch {epoch+1}/{cfg.epochs} | D: {loss_D.item():.4f} G: {loss_G.item():.4f} | t={elapsed:.1f}s"
        )

        # Simple checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = cfg.checkpoint_dir
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "G_state": G.state_dict(),
                    "D_state": D.state_dict(),
                    "optimizer": optG.state_dict(),
                },
                os.path.join(ckpt_path, f"ckpt_{epoch+1}.pt"),
            )

    print("Training complete")

# ---------------------------------------------------------------------------
# Main entry‑point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train a synthetic IO generator")
    parser.add_argument("--trace", required=True, help="Path to trace file on vinge.local")
    parser.add_argument("--cfg", default="llgan/config.py", help="Path to a custom config file (optional)")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Where to write checkpoints")
    args = parser.parse_args()

    # Build a minimal Config – copy defaults and override via CLI
    cfg = Config()
    cfg.trace_path = args.trace
    cfg.epochs = args.epochs or cfg.epochs
    cfg.batch_size = args.batch_size or cfg.batch_size
    cfg.checkpoint_dir = args.checkpoint_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trace and split
    train_ds, val_ds, _ = load_trace(
        cfg.trace_path,
        cfg.trace_format,
        cfg.max_records,
        cfg.timestep,
        train_split=cfg.train_split,
    )
    dloader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    # Instantiate models
    G = Generator(cfg).to(device)
    D = Critic(cfg).to(device)

    wgan_sn_train(G, D, dloader, cfg, device)


if __name__ == "__main__":
    main()
