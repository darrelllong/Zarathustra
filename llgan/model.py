"""
LLGAN model components.

Architecture overview
---------------------
Training follows a four-phase curriculum adapted from TimeGAN/SeriesGAN:

  Phase 1 — Autoencoder pretrain:
    Encoder (E) and Recovery (R) are trained jointly on real sequences to
    build a smooth latent space. Once this space is learned, the generator
    only needs to produce points in it — a much easier task than directly
    generating raw features with their mixed scales and co-dependencies.

  Phase 2 — Supervisor pretrain:
    Supervisor (S) is trained on real latent trajectories to predict
    H_{t+1} from H_t. This distils the real temporal dynamics into S's
    weights before GAN training begins, giving the generator a meaningful
    target for its temporal structure.

  Phase 2.5 — Generator warm-up:
    Generator (G) is trained to fool S (minimise |S(G(z)) - G(z)[1:]|),
    not the critic. This seeds G into the same latent space as real data
    before the adversarial signal is introduced.

  Phase 3 — Joint GAN:
    All five components train together. G produces latent sequences that
    R decodes to features for the critic. The critic's gradient signal
    (WGAN) is now layered on top of the supervisor consistency term.

Why this phasing? Direct GAN training from scratch in a high-dimensional
feature space (5 correlated features spanning 10 decades of dynamic range)
is notoriously unstable. The AE reduces the problem to a smooth low-dim
latent space; the supervisor provides a useful training signal before the
discriminator is competent enough to give meaningful gradients.

GRU vs LSTM choices
--------------------
E, R, S use GRU: fewer parameters than LSTM, no cell state. The AE task
(compression + reconstruction) is essentially a denoising problem; GRU's
simpler gating is sufficient and faster to train. The supervisor's 1-step
prediction task also does not require LSTM's long-range cell memory.

G and C use LSTM: the generator needs to maintain workload identity over an
entire window (which regime is this trace in? bursty or idle?). LSTM's
explicit cell state is better at carrying this "global context" signal
across many timesteps without it leaking out through the forget gate.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Latent autoencoder components (TimeGAN / SeriesGAN architecture)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    GRU encoder: maps feature sequences to a smooth latent space.

    (batch, timestep, num_cols) → (batch, timestep, latent_dim) ∈ [0, 1]

    The output range [0, 1] (Sigmoid) is intentional: it matches the
    Generator's output range so that real and generated latents live in the
    same bounded space, which stabilises the supervisor and prevents the
    Recovery from needing to handle out-of-range inputs.

    AVATAR mode (avatar=True): output is unbounded, targeting N(0,1) via
    the AAE latent discriminator.  BatchNorm1d after GRU stabilises the
    latent distribution before the FC projection.
    """

    def __init__(self, num_cols: int, hidden_size: int, latent_dim: int,
                 avatar: bool = False):
        super().__init__()
        self.avatar = avatar
        self.gru = nn.GRU(num_cols, hidden_size, num_layers=1, batch_first=True)
        if avatar:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.fc  = nn.Linear(hidden_size, latent_dim)
        if not avatar:
            self.act = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        if self.avatar:
            B, T, H = h.shape
            h = self.bn(h.reshape(B * T, H)).reshape(B, T, H)
            return self.fc(h)
        return self.act(self.fc(h))


class Recovery(nn.Module):
    """
    GRU recovery: decodes latent sequences back to feature space.

    (batch, timestep, latent_dim) → (batch, timestep, num_cols) ∈ [-1, 1]

    Mixed-type output heads (IDEAS.md idea #7): binary columns (opcode,
    obj_id_reuse) use a sigmoid head → scaled to [-1, 1] via 2σ-1.  This
    produces sharp near-±1 values that match the real data distribution,
    reducing MMD² by eliminating soft intermediate values for binary fields.
    Continuous columns keep the standard Tanh head.

    When binary_cols is None (legacy), all columns share one Tanh head and
    the checkpoint is identical to the original architecture (backward compat).
    """

    def __init__(self, latent_dim: int, hidden_size: int, num_cols: int,
                 avatar: bool = False,
                 binary_cols: Optional[List[int]] = None):
        super().__init__()
        self.avatar      = avatar
        self.num_cols    = num_cols
        self.binary_cols = sorted(binary_cols) if binary_cols else []
        self.cont_cols   = [i for i in range(num_cols) if i not in set(self.binary_cols)]

        self.gru = nn.GRU(latent_dim, hidden_size, num_layers=1, batch_first=True)
        if avatar:
            self.bn = nn.BatchNorm1d(hidden_size)

        if self.binary_cols:
            # Separate heads for type-correct activations
            self.fc_cont   = nn.Linear(hidden_size, len(self.cont_cols))
            self.fc_binary = nn.Linear(hidden_size, len(self.binary_cols))
        else:
            # Legacy: single fc + Tanh (identical to original architecture)
            self.fc  = nn.Linear(hidden_size, num_cols)
            self.act = nn.Tanh()

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        if self.avatar:
            B, T, H = out.shape
            out = self.bn(out.reshape(B * T, H)).reshape(B, T, H)

        if not self.binary_cols:
            return self.act(self.fc(out))

        # Mixed-type: assemble output in original column order
        result = torch.empty(*out.shape[:2], self.num_cols,
                              dtype=out.dtype, device=out.device)
        result[..., self.cont_cols]   = torch.tanh(self.fc_cont(out))
        # Sigmoid → [0,1] → scale to [-1,1] for consistency with real data range
        result[..., self.binary_cols] = torch.sigmoid(self.fc_binary(out)) * 2.0 - 1.0
        return result


class Supervisor(nn.Module):
    """
    GRU supervisor: predicts next latent H_{t+1} from current latent H_t.

    Pre-training on real latents makes S into a learned "physics" of the
    workload's temporal dynamics: if H_t represents "idle" then S(H_t)
    should predict "still idle" or "transitioning to burst". This signal
    is then used as an auxiliary loss for the generator (§ 4.5 of the paper)
    to enforce temporal coherence in generated traces.

    The supervisor loss weight η controls how strongly temporal coherence
    is enforced relative to the adversarial signal. Too high (η ≫ 1) →
    the generator satisfies the supervisor trivially by outputting constant
    latents (the zero vector is always consistent with itself). Too low →
    the adversarial signal dominates and temporal structure is ignored.
    The v11 collapse at η=10 with 100 warm-up epochs illustrates the first
    failure mode; v13 uses η=10 but only 50 warm-up epochs to avoid it.
    """

    def __init__(self, latent_dim: int, hidden_size: int,
                 avatar: bool = False):
        super().__init__()
        self.avatar = avatar
        self.gru = nn.GRU(latent_dim, hidden_size, num_layers=1, batch_first=True)
        if avatar:
            self.bn = nn.BatchNorm1d(hidden_size)
        self.fc  = nn.Linear(hidden_size, latent_dim)
        if not avatar:
            self.act = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name: nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name: nn.init.zeros_(p)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        if self.avatar:
            B, T, H = out.shape
            out = self.bn(out.reshape(B * T, H)).reshape(B, T, H)
            return self.fc(out)
        return self.act(self.fc(out))


# ---------------------------------------------------------------------------
# AVATAR: Latent discriminator (AAE component)
# ---------------------------------------------------------------------------

class LatentDiscriminator(nn.Module):
    """AAE discriminator: distinguishes encoded latents q(z|x) from prior samples z ~ N(0,1).
    Simple MLP -- no spectral norm, no BatchNorm (per AVATAR paper S4.3)."""

    def __init__(self, latent_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) -> (N, 1) logits."""
        return self.net(z)


# ---------------------------------------------------------------------------
# Generator and Critic
# ---------------------------------------------------------------------------

class CondEncoder(nn.Module):
    """Variational encoder for char-file conditioning vectors (IDEAS.md idea #3).

    Replaces the fixed char-file conditioning vector with a learned distribution:
        μ, log_σ² = Linear(char_stats)
        cond ~ N(μ, exp(log_σ²/2))   at training time  [reparameterized]
        cond = μ                      at eval time

    Benefits:
    - Generator becomes robust to conditioning noise → closes EMA→eval gap
    - Principled uncertainty over workload type; two traces with same write_ratio
      but different temporal dynamics get different noise realizations
    - KL divergence KL(q(cond|stats) || N(0,I)) regularises the encoder

    Init: mu_net ≈ identity (passes char_stats through unchanged), logvar bias=-6
    (σ≈0.05, near-deterministic) → backward compatible with existing pretrained
    checkpoints. σ grows as training finds it can exploit conditioning uncertainty.
    """

    def __init__(self, cond_dim: int):
        super().__init__()
        self.mu_net     = nn.Linear(cond_dim, cond_dim)
        self.logvar_net = nn.Linear(cond_dim, cond_dim)
        # Identity init for mu (passes input through unchanged at start)
        nn.init.eye_(self.mu_net.weight)
        nn.init.zeros_(self.mu_net.bias)
        # Near-zero logvar: σ=exp(-3)≈0.05 — tiny perturbation initially
        nn.init.zeros_(self.logvar_net.weight)
        nn.init.constant_(self.logvar_net.bias, -6.0)

    def forward(self, cond: torch.Tensor, training: bool = True):
        """
        Args:
            cond:     (B, cond_dim) raw char-file conditioning vector
            training: sample if True; use μ deterministically if False
        Returns:
            encoded:  (B, cond_dim) — perturbed at train time, deterministic at eval
            kl:       scalar KL(N(μ,σ²) || N(0,I)) = 0.5*(σ²+μ²-1-log σ²).mean()
        """
        mu     = self.mu_net(cond)
        logvar = self.logvar_net(cond).clamp(-10, 2)   # σ in [e^-5, e^1]
        if training:
            std     = (0.5 * logvar).exp()
            encoded = mu + std * torch.randn_like(mu)
            kl      = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).mean()
        else:
            encoded = mu
            kl      = cond.new_zeros(())
        return encoded, kl


class GMMPrior(nn.Module):
    """
    Gaussian Mixture Model prior for the generator's noise vector.

    Replaces the flat N(0,I) noise prior with a conditioning-aware mixture:

        π_k = softmax(MLP(cond))          # (B, K) soft component weights
        μ_z = Σ_k π_k * μ_k               # (B, noise_dim) weighted mean
        σ_z = Σ_k π_k * exp(λ_k)          # (B, noise_dim) weighted std
        noise = μ_z + σ_z * ε,  ε ~ N(0,I)

    Each of the K mixture components has its own learnable mean μ_k and
    log-std λ_k in noise space.  A small MLP maps the conditioning vector
    to soft component weights π_k, so different workload types (burst,
    sequential, random-read, write-heavy) get their own distinct region of
    noise space.  Random ε still drives per-sample diversity within each mode.

    Why this helps: when all samples share a single N(0,I) prior, the
    diversity loss fights against the prior to spread G across workload modes.
    A GMM prior makes multi-modal coverage structural — each workload type
    has a dedicated noise region — removing the tension between quality and
    diversity.
    """

    def __init__(self, noise_dim: int, cond_dim: int, K: int = 8):
        super().__init__()
        self.noise_dim = noise_dim
        self.K = K

        # Learnable mixture component parameters.
        # Zero-init means: initial behavior is identical to N(0,I), so the
        # pretrained G sees familiar noise in early Phase 3.  Gradient updates
        # push components apart as training progresses.
        self.means    = nn.Parameter(torch.zeros(K, noise_dim))
        self.log_stds = nn.Parameter(torch.zeros(K, noise_dim))

        # Small MLP: cond → K logits (component selector)
        hidden = max(K * 2, 16)
        self.selector = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Sample noise conditioned on workload descriptor.

        Args:
            cond: (B, cond_dim) conditioning vectors
        Returns:
            noise: (B, noise_dim) — GMM-sampled, reparameterized
        """
        pi   = torch.softmax(self.selector(cond), dim=-1)   # (B, K)
        mu_z = pi @ self.means                               # (B, noise_dim)
        std_z = pi @ torch.exp(self.log_stds)                # (B, noise_dim)
        return mu_z + std_z * torch.randn_like(mu_z)


class Generator(nn.Module):
    """
    LSTM generator with split-latent noise design.

    The noise is decomposed into two orthogonal components rather than a
    single flat vector:

      z_global (batch, noise_dim)
        Encodes workload identity — which tenant, what access regime, what
        diurnal phase. Projected to the LSTM's initial (h_0, c_0) state,
        so every timestep "knows" which workload it belongs to. Two windows
        with the same z_global will have similar high-level structure (same
        burst frequency, similar working set) but different event-level noise.

      z_local (batch, timestep, noise_dim)
        Per-step innovation noise — drives the stochasticity within a single
        window. Because it is fed as the LSTM input at each step, the LSTM
        can modulate how much of the local noise bleeds through based on the
        current hidden state (i.e., the workload context set by z_global).

    This split is critical for multi-stream generation: each stream gets its
    own fixed z_global (sampled once at stream start) but independent z_local
    per window, producing traces that are globally coherent but locally varied.
    Without this split, every window would be statistically independent and
    the long-range burst structure of the real traces would be lost.

    Latent AE mode (latent_dim > 0):
      Output is (batch, timestep, latent_dim) ∈ [0, 1] via Sigmoid. The
      Recovery module then decodes to feature space. This decouples generation
      from preprocessing: the generator never sees raw features and does not
      need to learn the complex non-linear preprocessing pipeline in reverse.

    Legacy direct mode (latent_dim = None):
      Output is (batch, timestep, num_cols) ∈ (-1, 1) via Tanh. Used by v4/v5.
    """

    def __init__(
        self,
        noise_dim: int,
        num_cols: int,
        hidden_size: int,
        latent_dim: Optional[int] = None,
        avatar: bool = False,
        cond_dim: int = 0,
        film_cond: bool = False,
        gmm_components: int = 0,
        var_cond: bool = False,
    ):
        super().__init__()
        self.noise_dim   = noise_dim
        self.num_cols    = num_cols
        self.hidden_size = hidden_size
        self.latent_dim  = latent_dim
        self.avatar      = avatar
        self.cond_dim    = cond_dim
        self.film_cond   = film_cond and (cond_dim > 0)

        # Variational conditioning encoder (IDEAS.md idea #3): replaces fixed
        # char-file vector with a sampled distribution at training time.
        if var_cond and cond_dim > 0:
            self.cond_encoder = CondEncoder(cond_dim)
        else:
            self.cond_encoder = None

        # GMM prior: replaces flat N(0,I) noise with a conditioning-aware
        # mixture.  Only active when gmm_components > 0 and cond_dim > 0.
        if gmm_components > 0 and cond_dim > 0:
            self.gmm_prior = GMMPrior(noise_dim, cond_dim, K=gmm_components)
        else:
            self.gmm_prior = None

        out_dim = latent_dim if latent_dim is not None else num_cols

        # Separate projections for h_0 and c_0 give the model two degrees of
        # freedom to encode workload context: h_0 sets the initial output gate
        # bias, c_0 sets the long-term memory content. In practice this lets
        # z_global influence both "what the generator is doing now" (h) and
        # "what it will tend to do over the window" (c).
        z_input_dim = noise_dim + cond_dim
        self.z_to_h0 = nn.Linear(z_input_dim, hidden_size)
        self.z_to_c0 = nn.Linear(z_input_dim, hidden_size)

        self.lstm = nn.LSTM(
            input_size=noise_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc      = nn.Linear(hidden_size, out_dim)
        # AVATAR: unbounded output (matching Encoder's unbounded N(0,1) space)
        if avatar and latent_dim is not None:
            self.out_act = nn.Identity()
        else:
            self.out_act = nn.Sigmoid() if latent_dim is not None else nn.Tanh()

        # FiLM conditioning (Feature-wise Linear Modulation, NeurIPS 2018):
        # After LSTM, reproject z_global into scale γ and shift β and apply
        #   h_t = (1 + γ) * h_t + β
        # This reinjects the workload conditioning at every timestep so it
        # cannot fade through the LSTM forget gate.  Zero-init ensures the
        # module is initially transparent (backward compatible with old ckpts).
        if self.film_cond:
            self.film_gamma = nn.Linear(z_input_dim, hidden_size, bias=True)
            self.film_beta  = nn.Linear(z_input_dim, hidden_size, bias=True)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        else:
            self.film_gamma = None
            self.film_beta  = None

        self._init_weights()

    def sample_noise(
        self,
        B: int,
        device: torch.device,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample z_global noise, using GMM prior when available.

        Args:
            B: batch size
            device: target device
            cond: (B, cond_dim) conditioning vectors; required for GMM prior.
        Returns:
            (B, noise_dim) noise tensor
        """
        if self.gmm_prior is not None and cond is not None:
            return self.gmm_prior(cond.to(device))
        return torch.randn(B, self.noise_dim, device=device)

    def _init_weights(self):
        _skip = {"gmm_prior", "cond_encoder"}   # these have their own init
        for name, p in self.named_parameters():
            if any(s in name for s in _skip):
                continue
            if "weight" in name:
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        z_global: torch.Tensor,
        z_local: torch.Tensor,
        hidden=None,
        return_hidden: bool = False,
    ):
        """
        z_global : (batch, noise_dim) — workload code, used to set h_0/c_0.
        z_local  : (batch, timestep, noise_dim) — per-step innovation noise.
        hidden   : (h, c) tuple to continue generation across windows.
                   Pass the previous window's final hidden state here to
                   maintain temporal coherence across window boundaries
                   (used by generate.py for long trace synthesis).
        return_hidden : pass True when generating multi-window traces.
        """
        if hidden is None:
            h0 = self.z_to_h0(z_global).unsqueeze(0)   # (1, batch, hidden)
            c0 = self.z_to_c0(z_global).unsqueeze(0)
            hidden = (h0, c0)
        h, hidden_out = self.lstm(z_local, hidden)
        # FiLM: reinject workload conditioning at every timestep so it cannot
        # fade through the LSTM forget gate over the T-step window.
        # (1 + γ) * h + β; zero-init → initially transparent (identity).
        if self.film_gamma is not None and z_global is not None:
            gamma = self.film_gamma(z_global).unsqueeze(1)  # (B, 1, H)
            beta  = self.film_beta(z_global).unsqueeze(1)
            h = (1.0 + gamma) * h + beta
        out = self.out_act(self.fc(h))
        if return_hidden:
            return out, hidden_out
        return out

    @torch.no_grad()
    def generate(
        self,
        n_windows: int,
        timestep: int,
        device: torch.device,
        opcode_col: int = -1,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        if self.cond_dim > 0:
            if cond is None:
                cond = torch.randn(n_windows, self.cond_dim, device=device) * 0.5
            # Variational encoder: use μ (deterministic) at inference
            if self.cond_encoder is not None:
                cond, _ = self.cond_encoder(cond, training=False)
            noise = self.sample_noise(n_windows, device, cond=cond)
            z_global = torch.cat([cond, noise], dim=1)
        else:
            noise = self.sample_noise(n_windows, device)
            z_global = noise
        z_local  = torch.randn(n_windows, timestep, self.noise_dim, device=device)
        out = self(z_global, z_local)
        if opcode_col >= 0:
            # Hard-binarise opcode: the continuous generator output is a soft
            # interpolation between read and write; applying a threshold here
            # ensures the synthetic trace contains only valid opcodes, not
            # fractional values that have no physical meaning.
            out[:, :, opcode_col] = (out[:, :, opcode_col] >= 0).float() * 2 - 1
        return out


class Critic(nn.Module):
    """
    LSTM critic for Wasserstein training (WGAN-SN variant).

    Outputs an unbounded real-valued score — the Wasserstein distance
    estimator. No sigmoid: a sigmoid-bounded discriminator saturates when the
    generator is far from real, producing near-zero gradients and stalling
    training. Wasserstein loss provides useful gradients even when the
    distributions are disjoint (which is typical early in training).

    Lipschitz constraint: spectral normalisation on both the LSTM weight
    matrices (weight_ih_l0, weight_hh_l0) and the output FC layer.
    SN power iteration uses torch.no_grad() so it is device-agnostic
    (MPS, CUDA, CPU).  Previously only the FC layer was constrained, which
    allowed the LSTM to drift unboundedly — W reached 31 and recall hit 0
    (mode collapse) in v14 at epoch 100.  Adding sn_lstm=True fixes this.

    Temporal pooling via learned attention (not mean pooling):
    Mean pooling treats all timesteps identically. For I/O workloads this
    is wrong: a 1-timestep burst (sudden IOPS spike) is the most
    discriminative feature of a bursty workload, but mean pooling dilutes
    it across the other 11 idle timesteps. The attention module learns to
    up-weight the timesteps that best distinguish real from fake, which
    makes the critic sensitive to burst structure and regime transitions —
    exactly what the DMD-GEN metric shows we need to improve.

    Feature matching:
    The pooled hidden state (before the final linear) is returned when
    return_features=True. The generator uses this to minimise the
    discrepancy between real and fake critic representations, which
    empirically reduces mode collapse by providing a smoother training
    signal than the Wasserstein loss alone.
    """

    def __init__(self, num_cols: int, hidden_size: int,
                 use_spectral_norm: bool = True,
                 sn_lstm: bool = True,
                 minibatch_std: bool = True,
                 patch_embed: bool = False,
                 cond_dim: int = 0):
        super().__init__()
        self.minibatch_std = minibatch_std
        self.patch_embed   = patch_embed
        self.cond_dim      = cond_dim
        # One extra input feature when minibatch_std is on: the mean per-step
        # standard deviation across the batch, appended as a scalar channel.
        lstm_input = num_cols + (1 if minibatch_std else 0)

        if patch_embed:
            # Patch embedding (TTS-GAN): Conv1d with kernel=stride=3 folds the
            # 12-step window into 4 non-overlapping 3-step patch tokens.  Each
            # patch sees a short local segment (≈1 burst event at 12-step windows)
            # rather than a single raw timestep, giving the critic a richer
            # inductive bias for detecting bursty patterns.
            # Input:  (B, T=12, D) → transpose → (B, D, 12)
            # After conv: (B, hidden_size, 4) → transpose → (B, 4, hidden_size)
            # The LSTM then operates on 4 patch tokens instead of 12 raw steps,
            # reducing sequence length and encouraging attention to episode-level
            # structure rather than individual timestep noise.
            self.patch_conv = nn.Conv1d(
                lstm_input, hidden_size, kernel_size=3, stride=3
            )
            lstm_in = hidden_size
        else:
            lstm_in = lstm_input

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        if sn_lstm:
            # Apply spectral normalisation to the LSTM weight matrices.
            # This is the component that drifted unconstrained in v14,
            # causing W→31 and mode collapse at epoch 100.
            # SN power iteration runs under torch.no_grad() so it is
            # device-agnostic (MPS, CUDA, CPU all work).
            # weight_ih_l0: (4H, input_size) — input gate weights
            # weight_hh_l0: (4H, H)          — recurrent gate weights
            from torch.nn.utils import spectral_norm
            spectral_norm(self.lstm, name='weight_ih_l0')
            spectral_norm(self.lstm, name='weight_hh_l0')

        # A single linear layer maps each LSTM hidden state to a scalar
        # attention weight. Softmax across the time dimension gives a
        # probability distribution over timesteps, letting the critic
        # focus on the most informative moments in the sequence.
        self.attn = nn.Linear(hidden_size, 1)

        fc = nn.Linear(hidden_size, 1)
        if use_spectral_norm:
            from torch.nn.utils import spectral_norm
            fc = spectral_norm(fc)
        self.fc = fc

        # Projection discriminator (Miyato & Koyama, ICLR 2018):
        # score += inner(cond_proj(cond), pooled_features)
        # Conditions the critic on workload descriptors so it scores
        # "is this realistic for this workload type?" not just "is this realistic?".
        # Active when cond_dim > 0; naturally disabled (cond=None) in pretrain phases.
        if cond_dim > 0:
            self.cond_proj = nn.Linear(cond_dim, hidden_size, bias=False)
        else:
            self.cond_proj = None

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_orig" in name or ("weight" in name and "norm" not in name):
                nn.init.normal_(p, 0.0, 0.02)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x: torch.Tensor,
                return_features: bool = False,
                cond: Optional[torch.Tensor] = None):
        """x: (batch, timestep, num_cols) → (batch, 1) unbounded score.

        cond: (batch, cond_dim) workload descriptor for projection discriminator.
              When provided, adds inner(cond_proj(cond), pooled) to the score.
        """
        if self.minibatch_std:
            # Minibatch standard deviation (Karras et al., ProGAN/StyleGAN2).
            # Compute the per-feature std across the batch at each timestep,
            # average across features, and append as a single extra channel.
            # This gives the critic direct access to within-batch diversity:
            # when the generator collapses, all rows look alike, std ≈ 0, and
            # the critic learns to score low-diversity batches as fake.
            std_per_step = x.std(dim=0, keepdim=True)           # (1, T, D)
            std_scalar   = std_per_step.mean(dim=-1, keepdim=True)  # (1, T, 1)
            std_channel  = std_scalar.expand(x.shape[0], -1, -1)    # (B, T, 1)
            x = torch.cat([x, std_channel], dim=-1)             # (B, T, D+1)
        if self.patch_embed:
            # Fold raw timesteps into patch tokens before LSTM.
            # (B, T, D) → (B, D, T) → conv1d → (B, H, T//3) → (B, T//3, H)
            x = self.patch_conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.lstm(x)                                    # (B, T or T//3, H)
        attn_w = torch.softmax(self.attn(h), dim=1)            # (B, T, 1)
        pooled = (attn_w * h).sum(dim=1)                       # (B, H)
        score = self.fc(pooled)                                 # (B, 1)
        if self.cond_proj is not None and cond is not None:
            # Projection: score += (cond_proj(cond) * pooled).sum(-1, keepdim=True)
            score = score + (self.cond_proj(cond) * pooled).sum(-1, keepdim=True)
        if return_features:
            return score, pooled
        return score


# Keep old name as alias so existing code that imports Discriminator still works
Discriminator = Critic
