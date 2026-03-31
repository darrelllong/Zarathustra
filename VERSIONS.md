# LLGAN Version History

All runs use oracle_general Tencent Block 2020 1M corpus (3234 files) unless noted.

---

## Current Run: v17

**Status**: RUNNING on vinge (GB10 CUDA, AMP fp16, no compile) — ep66/200

**Current best: ep145 — MMD²=0.00727, recall=0.446, combined=0.118** ← beat this.

**What changed vs v16:**
1. `--supervisor-loss-weight 1.0` (was 5.0): removes supervisor dominance over G gradient.
2. `--lr-d 5e-5` (was 2.5e-5): restores v14g level; lower lr_d hurt recall in v16.
3. `--diversity-loss-weight 1.0` (was 0.5): more recall pressure.
4. `--epochs 200` (was 150): room for late LR-decay improvement.

**Hypothesis**: v16 proved that reducing lr_d below v14g's level makes recall *worse*, not better.
The core problem is supervisor dominance: supervisor_loss_weight=5.0 makes G learn to replay
real sequences rather than generate new ones. Reducing it is the highest-priority lever for v17.

**What to change vs v16:**
1. `--supervisor-loss-weight 1.0` (was 5.0): The 5.0 value was set for pretraining stability.
   During GAN phase it over-constrains G — G gradient is dominated by "look like the supervisor
   predicted" rather than "fool the critic." Reducing to 1.0 gives cross_cov, diversity, and
   adversarial losses room to compete.
2. `--lr-d 5e-5` (was 2.5e-5, restore to v14g level): v14g with 5e-5 got recall=0.372;
   v16 with 2.5e-5 got 0.228. Lower lr_d created a stable-but-permanent critic advantage.
3. `--diversity-loss-weight 1.0` (was 0.5): More direct pressure on recall since we expect
   supervisor reduction to destabilize slightly. Extra diversity gradient compensates.
4. `--epochs 200` (was 150): v16 improved late (ep100→130) as LR decayed. Give more room.

Everything else unchanged from v16 (n_critic=2, lr_g=1e-4, cross_cov=2.0, hidden=256, latent=24).

---

## Completed Runs

| Version | Best MMD² | Best Recall | Best Epoch | Checkpoint | Notes |
|---------|-----------|-------------|------------|------------|-------|
| v3–v8 | ~0.07+ | 0.022–0.143 | varies | wigner archive | Pre-latent-AE; mode collapse |
| v9 | ~0.025 | 0.351 | ~60 | wigner archive | First latent AE + supervisor + FM |
| v13 | **0.00818** | 0.455 | 130 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v13/best.pt | 3234 files; 171 wasted GAN-cycling epochs |
| v14 | 0.041 | 0.618 | 30 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14/best.pt (ep30) | Collapsed ep~100: no LSTM SN, W→31 |
| v14c | 0.046 | 0.269 | 30 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14c/ | n_critic=1 too conservative; stagnated |
| v14d | 0.317 | 0.046 | — | discarded | Immediate collapse; n_critic=2 on non-SN checkpoint |
| v14e | 0.077 | 0.209 | 30 | discarded | Worse than v14c; same hot-start asymmetry |
| v14f | 0.048 | 0.021 | 50 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14f/ | Stable but plateaued; first diversity_loss test |
| v14g | **0.018** | **0.372** | 90 | wigner:/Volumes/Archive/Traces/checkpoints/tencent_v14g/best.pt | Best MMD²–recall trade-off; full eval reuse_rate=0 |
| v15 | 0.029 | 0.294 | 100 | vinge:~/checkpoints/tencent_v15/best.pt | GAN cycling ep110; missing diversity_loss |
| v16 | 0.042 | 0.228 | 130 | vinge:~/checkpoints/tencent_v16/best.pt | Diversity+cross_cov restored; critic still dominant; worse than v14g |

---

## Version Notes

### v14 (killed, 2026-03-28)
WGAN-SN without spectral norm on the critic LSTM: only the FC output layer was Lipschitz-constrained.
LSTM weights drifted unconstrained — W grew from 2.7 to 31 by epoch 100, recall hit 0.
Best was epoch 30 (first eval) before the drift began.

### v14c–v14e (hot-starts from v14/epoch_0030.pt, 2026-03-28–29)
All failed because loading a generator trained against an unconstrained critic into a freshly
SN-normalised critic creates a permanent power asymmetry. The SN u/v buffers are initialised
randomly; the critic wins decisively until they converge, by which point G is stuck.
**Lesson**: never hot-start from a checkpoint whose critic had different Lipschitz constraints.

### v14f (hot-start from v14c/epoch_0030.pt, killed ep190, 2026-03-29)
First test with diversity_loss_weight=0.5. Stable but plateaued. Full eval revealed that
training-log β-recall (0.269) and eval.py β-recall (0.021) diverge by ~13×:
training uses EMA weights + 1000 samples, eval uses live checkpoint + 2000 samples.
**Lesson**: training-log recall numbers are ~3–4× optimistic; trust full eval.py only.

### v14g (fresh from scratch, 2026-03-29–30)

First clean run with SN-LSTM from epoch 0. Key changes vs v14f:
- Fresh pretraining: AE 50ep + supervisor 50ep (supervisor_steps=2) + G warmup 50ep
- diversity_loss_weight=0.5: MSGAN recall fix (present in v14f, kept)
- cross_cov_loss_weight=0.5: full d×d lag-1 cross-feature covariance (new; targets DMD-GEN)
- lr_g=1e-4, lr_d=5e-5 (original v13 LRs; hot-starts used 5e-5/2.5e-5)

**Full eval (ep90)**: MMD²=0.018, α-precision=0.910, β-recall=0.372, DMD-GEN=0.700, Context-FID=0.03

**Root causes identified (fixed in v15):**
1. `reuse_rate=0.000`: delta-encoding makes obj_id reuse a zero-measure event in continuous space.
   Fix: split obj_id into `obj_id_reuse` (±1 binary) + `obj_id_stride` (signed-log delta).
2. DMD-GEN stuck at 0.700: supervisor_loss_weight (5.0) overpowers cross_cov_loss_weight (0.5) by 10×.
   Fix: raise cross_cov_loss_weight to 2.0 (v16).
3. obj_size non-quantized: generated sizes are continuous; real traces are multiples of 4096 bytes.
   Fix: snap to 4096-byte multiples before log-transform (v15).

### v16 (vinge/GB10, completed ep150, 2026-03-31)

Applied diversity_loss=0.5 (absent in v15), cross_cov=2.0, n_critic=2, lr_d=2.5e-5.
Training ran cleanly to ep150 with no cycling spikes. Best checkpoint: ep130.

**Full eval (ep130)**:
| Metric | v16 | v14g | Δ |
|--------|-----|------|---|
| MMD² | 0.042 | 0.018 | +0.024 worse |
| α-precision | 0.833 | 0.910 | −0.077 |
| β-recall | 0.228 | 0.372 | −0.144 (mode collapse) |
| DMD-GEN | 0.744 | 0.700 | +0.044 worse |
| Context-FID | 0.15 | 0.03 | +0.12 worse |
| reuse rate | 0.004 | 0.000 | tiny improvement |

**What worked:**
- Stable training through ep150: no GAN cycling, W stayed <2.1 (vs ep110 spike in v15).
- Late improvement: best improved ep100→ep130 as LR decayed, G briefly went negative (ep132–133).
- reuse_rate 0.004 > v14g 0.000: obj_id_reuse binary feature is registering, barely.

**What went wrong:**
1. **Lower lr_d (2.5e-5 vs v14g 5e-5) made recall worse**, not better (0.228 vs 0.372).
   Hypothesis: slower critic is more stable but maintains a *permanent* moderate advantage.
   In v14g the faster critic was in more active competition with G, which forced G to improve.
2. **cross_cov_loss_weight=2.0 didn't improve DMD-GEN** (0.744 vs 0.700).
   The 2.5:1 ratio (cross_cov 2.0 : supervisor 5.0) still not enough to rebalance signal.
   Supervisor at 5.0 dominates G's loss landscape; G learns to reproduce sequences, not generate.
3. **Context-FID 5× worse than v14g**: latent space quality degraded. Likely because
   the 6-feature AE (v15+) has a harder compression task than v14g's 5-feature AE,
   and the pretrain wasn't extended to compensate.
4. **Reuse rate still near-zero (0.004)**: Binary obj_id_reuse feature isn't learned well
   from a continuous latent with per-step noise. Needs architectural support, not just features.

**Root causes for v17:**
1. `supervisor_loss_weight=5.0` too high: dominates G gradient → G learns replay, not generation.
   Fix: reduce to 1.0–2.0.
2. `lr_d=2.5e-5` too low: stable but asymmetric equilibrium. Fix: restore to `5e-5` (v14g level).
3. Locality learning requires more than a binary input feature. Architectural fix (task #19)
   needed eventually; for v17 at minimum try higher diversity_loss to compensate.

### v15 (vinge/GB10, killed ep143, 2026-03-30)

First CUDA run. Applied all v14g root-cause fixes (obj_id split, obj_size quantization).
AMP fp16 enabled. torch.compile attempted but Triton broken on GB10 (libcuda.so link issue).

**Best (ep100)**: MMD²=0.029, β-recall=0.294, combined=0.170

**What went wrong:**
1. **Missing diversity_loss** (was in v14g, dropped from v15): direct cause of recall gap.
2. **GAN cycling (n_critic=3 too aggressive)**: ep100 best → ep110 MMD²=0.111 spike → never recovered.
3. **cross_cov still 10:1 drowned**: DMD-GEN stayed at ~0.700.

**Bugs found and fixed during v15 (all committed):**
- `torch.quantile` fp16 crash: cast to `.float()` before quantile (e3ad770)
- `evaluate_metrics()` left G/R in `.eval()`: `try/finally` restore in mmd.py (01a797f)
- E/R/S left in `.eval()` from pretrain phases: explicit `.train()` at GAN loop start (ba356d5)
- Supervisor gradient leak in G step: `torch.no_grad()` around `S(H_fake)` (f5c63b7)
- EMA seeded from random init: seed from post-warmup G after Phase 2.5 (f5c63b7)
- GradScaler called n_critic+2 times per batch instead of once: now called once per optimizer step (638ee24)
- rsync temp files in trace dir: `not p.name.startswith(".")` filter in `_collect_files` (0ca6f49)

**v15 eval history:**
| Epoch | MMD² | Recall | Combined |
|-------|------|--------|----------|
| 5 | 0.0652 | 0.198 | 0.226 |
| 55 | 0.0360 | 0.298 | 0.176 |
| 80 | 0.0341 | 0.296 | 0.175 |
| 100 | 0.0291 | 0.294 | 0.170 ★ |
| 110 | 0.1109 | 0.183 | 0.274 (cycling) |
| 140 | 0.0435 | 0.307 | 0.182 |

---

## Key Metrics and Targets

| Metric | v13 | v14g | v15 | v16 | Target | Description |
|--------|-----|------|-----|-----|--------|-------------|
| MMD² | 0.00818 | 0.018 | 0.029 | 0.042 | < 0.005 | Kernel distribution distance |
| β-recall | 0.455 | 0.372 | 0.294 | 0.228 | > 0.7 | Coverage: fraction of real modes covered |
| α-precision | 0.812 | 0.910 | — | 0.833 | > 0.85 | Fidelity: fraction of generated that is realistic |
| DMD-GEN | 0.771 | 0.700 | ~0.700 | 0.744 | < 0.3 | Temporal dynamics divergence (0 = perfect) |
| Context-FID | — | 0.03 | — | 0.15 | < 0.05 | Fréchet in encoder latent space |
| reuse-rate | 0.124 | 0.000 | TBD | 0.004 | > 0.1 | Fraction of obj_id repeats (sequential access) |

Note: v14g had better MMD² than v13 despite lower β-recall. v14g is the best MMD²–recall trade-off
we have achieved. All new runs must beat v14g's combined score (0.018 MMD² + 0.2×(1-0.372) = 0.144).

---

## What We Have Learned

### Architecture
- **GRU for AE/supervisor, LSTM for G+C**: GRU is simpler + faster for compression/prediction;
  LSTM cell state captures long-range context needed for G and burst detection in C.
- **Split noise (z_global + z_local)**: z_global maps to LSTM h0/c0 (workload identity);
  z_local feeds per-step input (event noise). Without this, every window is independent.
- **Latent AE (TimeGAN/SeriesGAN)**: Direct feature-space GAN is unstable in a 6-feature
  correlated space spanning 10 decades. AE reduces to smooth 24-dim latent space.
- **Minibatch std in critic (StyleGAN2)**: Collapsed G produces identical rows; std→0.
  This channel gives the critic a diversity signal before collapse propagates.
- **sn_lstm=True**: SN on LSTM weight matrices essential (not just FC output).
  FC-only SN allows LSTM to drift; W grows unboundedly and training collapses (v14).

### Training stability
- **GAN cycling / underdamped oscillation**: When critic pressure is too high (n_critic=3,
  high lr_d), G overshoots → critic wins decisively → G overshoots back. Seen in v13 (171
  wasted epochs) and v15. Damp with: lower n_critic, lower lr_d, cosine LR decay.
- **Hot-start asymmetry**: Never hot-start from a checkpoint with different Lipschitz constraints.
  Fresh SN u/v buffers + generator tuned for old critic → critic overpowers immediately.
- **Phase train/eval mode**: Each pretrain phase explicitly sets eval on modules it doesn't train.
  Must restore `.train()` on ALL modules at GAN loop start, or cuDNN RNN backward crashes.
- **GradScaler with n_critic > 1**: `scaler.unscale_(optimizer)` can only be called once
  per optimizer between `scaler.update()` calls. With n_critic > 1 and grad clipping,
  call `scaler.update()` after each `scaler.step(opt_C)` within the critic loop.
- **EMA**: Seed ema_G_state from post-warmup G weights (not random init). Random seed means
  first GAN evals use near-random EMA, skewing best.pt selection.

### Losses
- **L_FM (feature matching)**: Smoother gradient signal than Wasserstein alone; reduces collapse.
- **L_div (diversity, MSGAN)**: Directly combats β-recall mode collapse. Required in v14g
  (recall=0.372) vs v15 without it (recall=0.294). Weight 0.5 works; try 0.5–1.0.
- **L_cov (cross-covariance)**: Targets DMD-GEN directly. Needs weight ≥ 2.0 to overcome
  supervisor dominance at weight 5.0 (otherwise 10:1 ratio drowns cross_cov signal).
- **L_loc**: Stride-repetition rate — replaced by obj_id_reuse binary feature in v15+.
  Still active as auxiliary loss but its role is partly absorbed by the locality split.
- **L_ACF, L_FFT**: Effective for temporal structure; keep at current weights.
- **L_V, L_Q**: Cheap distributional anchors; always-on.

### Data representation
- **obj_id delta → locality split (v15)**: Raw delta-encoding makes reuse a zero-measure event.
  `obj_id_reuse` (±1 binary) + `obj_id_stride` (signed-log delta) gives G a learnable
  binary classification target for sequential access patterns.
- **obj_size quantization (v15)**: Snap to 4096-byte multiples before log-transform.
  Real block traces have discrete sizes; continuous representation wastes capacity.
- **Training-log recall is ~3–4× optimistic vs full eval** (EMA weights + 1000 samples
  vs live checkpoint + 2000 samples). Never abort based on training-log recall alone.

### Infrastructure notes
- **torch.compile**: Broken on vinge's GB10 (Triton can't find libcuda.so.1). Use `--no-compile`.
- **AMP fp16**: Works correctly. `torch.quantile` requires `.float()` cast before call.
- **pretrain_complete.pt**: State dict keys have no `_orig_mod.` prefix (saved without compile).
  Load into non-compiled models only (`--no-compile`). Avoids 35-min pretrain on restart.
- **SSH to vinge**: `ssh vinge.local` (RSA key, no passphrase). `-A` flag for git pull on GitHub.
