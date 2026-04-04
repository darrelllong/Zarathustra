# LLGAN Version History

All runs use oracle_general Tencent Block 2020 1M corpus (3234 files) unless noted.

---

## Current Run: v36

**Status**: RUNNING — same v34 recipe (cond_drop_prob=0.25), different seed. v34 tied v31 at
combined=0.089; trying another seed to break through to 0.08x.

**Recipe**: Identical to v34 (cond_drop_prob=0.25, cond_dim=10, full losses, n_critic=2, v28
pretrain, 200 epochs). v35 showed 0.30 is too aggressive; 0.25 confirmed as sweet spot.

```bash
ssh 192.168.86.30 "cd ~/llgan && nohup ~/llgan-env/bin/python -u train.py \
  --trace-dir ~/traces/tencent_block_1M --fmt oracle_general \
  --epochs 200 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir ~/checkpoints/tencent_v36 --checkpoint-every 5 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40 \
  --cond-dim 10 --cond-drop-prob 0.25 \
  --supervisor-loss-weight 1.0 --lr-g 1e-4 --lr-d 5e-5 \
  --n-critic 2 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 \
  --feature-matching-weight 1.0 --moment-loss-weight 0.1 \
  --fft-loss-weight 0.05 --quantile-loss-weight 0.2 --acf-loss-weight 0.2 \
  --locality-loss-weight 1.0 --dmd-ckpt-weight 0 \
  --ema-decay 0.999 --lr-cosine-decay 0.05 --grad-clip 1.0 \
  --hidden-size 256 --latent-dim 24 \
  --no-compile \
  > ~/train_v36.log 2>&1 &"
```

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
| v17 | 0.00697 | 0.521 | 190 | vinge:~/checkpoints/tencent_v17/best.pt | Former ATB (v3–v30). supervisor→1.0, lr_d=5e-5, diversity→1.0 |
| v18 | 0.01105 | 0.418 | 205 | vinge:~/checkpoints/tencent_v18/best.pt | cross_cov→5.0; did NOT beat v17; see post-mortem |
| v19 | 0.01094 | 0.518 | 225 | vinge:~/checkpoints/tencent_v19/best.pt | cross_cov→2.0, dmd_ckpt_weight=0.05; recall≈v17 but MMD² worse; see post-mortem |
| v20 | 0.03215 | 0.407 | 210 | vinge:~/checkpoints/tencent_v20/best.pt | 300 epochs; EMA looked great (0.00755/0.549) but full eval diverged 4×; see post-mortem |
| v21 | 0.01485 | 0.366 | 35 | vinge:~/checkpoints/tencent_v21/best.pt | EMA-save fix confirmed (gap inverted); early-stopped ep235; stagnated from ep35; see post-mortem |
| v22 | 0.01751 | 0.471 | 190 | vinge:~/checkpoints/tencent_v22/best.pt | v16 pretrain hypothesis confirmed; α-precision=0.927 new ATB; see post-mortem |
| v23 | 0.01967 | 0.357 | 165 | vinge:~/checkpoints/tencent_v23/best.pt | Weaker seed; late surge ep165 but below v22; DMD-GEN=0.699 best ever; aborted ep187 |
| **v24** | **0.00798** | **0.503** | 170 | vinge:~/checkpoints/tencent_v24/best.pt | **Near-ATB.** dmd_ckpt_weight=0; MMD²=0.008 (2nd best ever); EMA beat v17; see post-mortem |
| v25 | 0.01272 | 0.359 | 50 | vinge:~/checkpoints/tencent_v25/best.pt | Weak seed; W collapsed ep55; aborted ep59 |
| v26 | 0.00795 | 0.401 | 190 | vinge:~/checkpoints/tencent_v26/best.pt | diversity_loss=2.0; MMD² tied 2nd best; DMD-GEN=0.710 best ever; EMA recall inflated |
| v27 | 0.055 | 0.172 | 20 | discarded | AVATAR architecture; critic overpowered from ep1; aborted ep38 |
| v28 | 0.01183 | 0.508 | 70 | vinge:~/checkpoints/tencent_v28/best.pt | **First conditional run (cond_dim=10)**; combined=0.110 best-ever early trajectory; GAN cycling crash ep79 from stripped losses |
| v29 | 0.021 | 0.384 | 20 | vinge:~/checkpoints/tencent_v29/best.pt | cond_dim=10 + full losses; EMA phenomenal (recall=0.674, combined=0.075) but massive EMA gap; GAN cycling crash ep96 |
| v30 | 0.01640 | 0.468 | 25(EMA) | vinge:~/checkpoints/tencent_v30/best.pt | n_critic=3; α-precision=0.929 (tied ATB); aborted ep54 — recall falling, W→2.8; n_critic=3 too aggressive |
| v32 | 0.00797 | 0.441 | 110 | vinge:~/checkpoints/tencent_v32/best.pt | Same recipe as v31 250ep; W→0 at ep110 (critic lost signal); reuse=0.055 (10× improvement); aborted ep144 |
| v35 | 0.01123 | 0.378 | 65(EMA) | vinge:~/checkpoints/tencent_v35/best.pt | CFG 0.30 TOO aggressive; recall collapsed 0.609→0.378 (38% drop); DMD-GEN=0.694 new ATB |
| **v34** | **0.01119** | **0.608** | **150** | **vinge:~/checkpoints/tencent_v34/best.pt** | **CO-ATB (tied v31). β-recall=0.608 NEW ATB. CFG 0.25 > 0.15 for generalization. Most stable conditioned run.** |
| v33 | 0.01080 | 0.521 | 70(EMA) | vinge:~/checkpoints/tencent_v33/best.pt | Same v31 recipe, different seed; W spiked ep179; EMA recall 0.683→0.521 full eval (24% drop); Context-FID=0.02 new ATB |
| **v31** | **0.00769** | **0.596** | **70** | **vinge:~/checkpoints/tencent_v31/best.pt** | **CO-ATB (tied v34). CFG (cond_drop_prob=0.15) + cond_dim=10 + v28 pretrain. Beats v17 by 14%.** |

---

## Version Notes

### v35 — CFG 0.30 too aggressive (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: Same as v31/v34 but with **cond_drop_prob=0.30** (up from v34's 0.25). v28 pretrain,
cond_dim=10, full losses, n_critic=2, 200 epochs.

**EMA best (ep65)**: MMD²=0.01330, recall=0.609, combined=0.092

**Full eval**:

| Metric | v35 | v34 | v31 |
|--------|-----|-----|-----|
| MMD² | 0.01123 | 0.01119 | **0.00769** |
| α-precision | 0.881 | **0.903** | **0.953** |
| β-recall | 0.378 | **0.608** | 0.596 |
| DMD-GEN | **0.694** | 0.747 | 0.723 |
| AutoCorr | 0.049 | 0.053 | **0.032** |
| Context-FID | **0.07** | 0.02 | 0.03 |
| reuse rate | 0.003 | 0.004 | 0.005 |

**Full eval combined = 0.011 + 0.2*(1-0.378) = 0.136. Did NOT beat v31/v34.**

**Post-mortem**: CFG dropout 0.30 is too aggressive. Dilutes the conditioning signal so much that
the generator loses mode specificity. Massive EMA→full eval recall drop (0.609→0.378, 38%). The
one bright spot is DMD-GEN=0.694, the first run below 0.70 — higher unconditional diversity from
the aggressive dropout, but at the cost of conditional precision.

**CFG dropout sweep conclusion**:

| CFG dropout | Full eval recall | Full eval combined |
|-------------|-----------------|-------------------|
| 0.15 (v31) | 0.596 | 0.089 |
| 0.25 (v34) | 0.608 | 0.089 |
| 0.30 (v35) | 0.378 | 0.136 |

**Sweet spot confirmed at 0.25.** Future runs should use 0.25 unless architecture changes.

---

### v34 — CO-ATB, new recall record (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: Same as v31 but with **cond_drop_prob=0.25** (up from 0.15). v28 pretrain, cond_dim=10,
full losses, n_critic=2, 200 epochs.

**EMA best (ep65)**: MMD²=0.014, recall=0.630, combined=0.088

**Full eval (ep150 best)**:

| Metric | **v34** | v31 | v33 | v17 |
|--------|---------|-----|-----|-----|
| MMD² | 0.01119 | **0.00769** | 0.01080 | **0.00697** |
| α-precision | 0.903 | **0.953** | 0.886 | 0.826 |
| β-recall | **0.608** | 0.596 | 0.521 | 0.521 |
| DMD-GEN | 0.747 | 0.723 | 0.727 | **0.714** |
| AutoCorr | 0.053 | **0.032** | — | **0.032** |
| Context-FID | **0.02** | 0.03 | **0.02** | 0.03 |
| reuse rate | 0.004 | 0.005 | 0.001 | 0.006 |

**Full eval combined = 0.011 + 0.2*(1-0.608) = 0.089. TIED with v31 (0.089).**

**Key findings**:

1. **β-recall=0.608 is NEW ALL-TIME BEST** — first run to break 0.60 on full eval (v31=0.596).
   Higher CFG dropout gives better mode coverage.

2. **CFG dropout 0.25 dramatically improved generalization**: EMA recall 0.630 → full eval 0.608
   (only 3.5% drop). Compare v33 with CFG 0.15: 0.683→0.521 (24% drop). v31 with 0.15 had 0%
   drop but v31 was a lucky seed. 0.25 is more reliably good.

3. **Most stable conditioned run**: No crashes, no W explosion, 20+ consecutive evals with
   combined < 0.110.

4. **Context-FID=0.02 tied ATB** with v33.

5. **Training-time checkpoint selector chose ep65 but ep150 was actually better on full eval.**
   This suggests the EMA-based selector underweights late-training improvements.

**Direction**: Try cond_drop_prob=0.30 in v35 to see if even more dropout further closes the gap.

---

### v33 — Seed variance confirmation (vinge/GB10, completed ep197, 2026-04-01)

**Recipe**: Identical to v31 (cond_dim=10, cond_drop_prob=0.15, full losses, n_critic=2, v28
pretrain, 200 epochs). Note: continuity_loss_weight was 0 despite plan saying 0.5 (incompatible
with AMP).

**EMA best (ep70)**: MMD²=0.01551, recall=0.683, combined=0.079

**Full eval**:

| Metric | v33 | v31 | v17 |
|--------|-----|-----|-----|
| MMD² | 0.01080 | **0.00769** | **0.00697** |
| α-precision | 0.886 | **0.953** | 0.826 |
| β-recall | 0.521 | **0.596** | 0.521 |
| DMD-GEN | 0.727 | 0.723 | **0.714** |
| Context-FID | **0.02** | 0.03 | 0.03 |
| reuse rate | 0.001 | 0.005 | 0.006 |

**Full eval combined = 0.011 + 0.2*(1-0.521) = 0.107. Did NOT beat v31 (0.089).**

**Late crash**: W spiked to 10.6 at ep179/197, recall crashed to 0.022.

**Key findings**:
1. **Seed variance is large**: Same recipe as v31 but 20% worse combined (0.107 vs 0.089).
2. **EMA recall spikes don't hold**: EMA recall 0.683 (best ever seen) dropped 24% to 0.521
   on full eval. Compare v31 where EMA recall 0.591 actually *improved* to 0.596 on full eval.
   Rapid EMA recall climbs may indicate overfitting to conditioning descriptors.
3. **Context-FID=0.02 is new ATB** — only bright spot.
4. **Confirms CFG value**: Even the "bad seed" run still matched v17's recall (0.521) and
   beat it on α-precision (0.886 vs 0.826). Pre-CFG conditional runs (v28, v29) crashed hard.

**Direction**: Higher CFG dropout (cond_drop_prob=0.25) in v34 to reduce EMA→full eval gap.

---

### v31 — NEW ALL-TIME BEST (vinge/GB10, completed ep200, 2026-04-01)

**Recipe**: cond_dim=10 + CFG dropout (cond_drop_prob=0.15) + full loss recipe + n_critic=2 +
v28 pretrain (cond_dim=10 compatible) + dmd_ckpt_weight=0. 200 epochs.

**EMA best (ep70)**: MMD²=0.01167, recall=0.591, combined=0.094

**Full eval (ep70 best)**:

| Metric | **v31** | v17 | v24 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00769** | 0.00697 | 0.00798 | <0.005 |
| α-precision | **0.953** | 0.826 | 0.835 | >0.80 |
| β-recall | **0.596** | 0.521 | 0.503 | >0.70 |
| DMD-GEN | 0.723 | **0.714** | 0.717 | <0.30 |
| AutoCorr | **0.032** | 0.032 | 0.036 | <0.02 |
| Context-FID | **0.03** | 0.03 | 0.08 | <0.05 |
| reuse rate | 0.005 | 0.006 | 0.007 | >0.1 |

**Full eval combined = 0.008 + 0.2*(1-0.596) = 0.089. BEATS v17 (0.103) by 14%.**

**What worked — CFG solved the overfitting problem:**

1. **EMA-full eval gap INVERTED**: EMA recall 0.591 → full eval 0.596. This is the first
   conditional run where full eval recall *exceeds* EMA recall. CFG dropout forces the model
   to learn both conditional and unconditional generation, eliminating the descriptor overfitting
   that plagued v28 (crashed ep79) and v29 (43% recall drop on full eval).

2. **α-precision=0.953 — new ATB by a wide margin** (previous: v22=0.927, v30=0.929).
   Conditioning tells G exactly what workload to produce; CFG dropout prevents overfitting
   while preserving the fidelity benefit.

3. **β-recall=0.596 — new ATB by 14%** (v17=0.521). The 0.596 barrier was never even
   approached by unconditional runs. Conditioning provides a coverage mechanism that
   diversity_loss alone could not achieve: G explicitly targets different workload types
   instead of hoping noise diversity maps to mode coverage.

4. **MMD²=0.00769** — 2nd best ever (v17=0.00697). With recall at 0.596 (vs v17's 0.521),
   this represents a much better Pareto trade-off: v31 covers 14% more modes while only
   giving up 0.001 on MMD².

5. **Late W spike (ep186-192, W up to 9.0) didn't matter** — EMA captured best at ep70,
   and the model recovered. The W spike is the same conditioning instability seen in v28/v29,
   but CFG dropout delayed it from ep79/ep96 to ep186, well past the useful training region.

6. **Combined=0.089** — first run below 0.100. This is 14% better than v17 (0.103) and
   17% better than v24 (0.107), the previous #2.

**The winning recipe**:
- v28 pretrain (cond_dim=10 architecture) — NOT v16 pretrain (unconditional architecture)
- cond_dim=10 + cond_drop_prob=0.15 (CFG dropout)
- Full auxiliary losses (cross_cov=2.0, acf=0.2, fft=0.05, moment=0.1, quantile=0.2, locality=1.0)
- n_critic=2, supervisor=1.0, diversity=1.0, lr_g=1e-4, lr_d=5e-5
- dmd_ckpt_weight=0
- ema_decay=0.999, lr_cosine_decay=0.05, grad_clip=1.0

**Remaining problems:**
1. **DMD-GEN=0.723** (target <0.30): Still stubbornly above 0.70. Conditioning improved
   fidelity and coverage but did not help temporal dynamics. This remains architectural.
2. **reuse rate=0.005** (target >0.1): Locality/sequential access still unlearned.
3. **Late W spike**: CFG delayed but did not prevent conditioning instability. Longer runs
   need either lower lr_g or adaptive n_critic to handle late-stage W growth.

**v32 direction**: Same recipe, 250 epochs. v31 peaked at ep70 but sustained excellence
through ep150. Extended cosine decay may allow further convergence before the W spike zone.

---

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

### v17 (vinge/GB10, completed ep200, 2026-03-31)

supervisor_loss_weight→1.0, lr_d→5e-5, diversity_loss→1.0, 200 epochs. Used v16 pretrain checkpoint.

**Full eval (ep190 best)**:
| Metric | v17 | v14g | v13 | Δ vs v14g |
|--------|-----|------|-----|-----------|
| MMD² | **0.00697** | 0.018 | 0.01335 | 2.6× better |
| α-precision | 0.826 | 0.910 | 0.812 | −0.084 |
| β-recall | **0.521** | 0.372 | 0.455 | +0.149 |
| DMD-GEN | 0.714 | 0.700 | 0.771 | similar |
| AutoCorr | 0.032 | — | 0.044 | better |
| Context-FID | **0.03** | 0.03 | 0.05 | same |
| reuse rate | 0.006 | 0.000 | 0.000 | still near-zero |

**What worked:**
- `supervisor_loss_weight=1.0` was the decisive change. G competed freely from epoch 1 (G<0 at ep1).
- Late LR-decay delivered sustained improvement: best kept improving through ep190/200.
- β-recall=0.521 breaks v13's 0.455 record. MMD²=0.00697 is new all-time low.
- Combined score 0.114 beats v14g (0.144) by 21%.

**Remaining problems:**
1. **DMD-GEN=0.714** (target <0.30): temporal dynamics unchanged across all versions.
   cross_cov_loss_weight=2.0 vs supervisor_loss_weight=1.0 gives 2:1 ratio, but DMD-GEN
   doesn't respond. The cross-covariance loss may not be targeting the right structure.
2. **reuse rate fake=0.006** (target >0.1): Binary obj_id_reuse feature is learned at ~1% of
   real rate. The latent space doesn't naturally encode temporal address correlations.
   Requires architectural change (task #19).
3. **α-precision=0.826** (target >0.85): Fidelity slightly below target; diversity
   pressure (diversity_loss=1.0) may be pushing G slightly off the real manifold.

**v18 direction**: Keep v17 hyperparameters. Address DMD-GEN with higher cross_cov (try 5.0)
and longer training (250 epochs). α-precision may improve naturally with more epochs.

---

### v18 (vinge/GB10, completed ep250, 2026-03-31)

Key change vs v17: `cross_cov_loss_weight` 2.0 → **5.0**, epochs 200 → **250**.

**Best (ep205, full eval with n_samples=2000)**:

| Metric | v18 | v17 | Target |
|--------|-----|-----|--------|
| MMD² | 0.01105 | **0.00697** | <0.005 |
| α-precision | **0.845** | 0.826 | >0.80 |
| β-recall | 0.418 | **0.521** | >0.70 |
| DMD-GEN | 0.760 | **0.714** | <0.30 |
| AutoCorr | 0.042 | — | <0.02 |
| Context-FID | 0.06 | **0.03** | <10 |
| reuse rate | 0.004 | 0.006 | ~0.827 |

**Did NOT beat v17.** v17 remains all-time best.

**What went wrong:**
1. **cross_cov=5.0 did not improve DMD-GEN**: DMD-GEN worsened from 0.714 → 0.760. Higher
   weight likely interfered with the generator during early training (similar mechanism to
   how supervisor_loss_weight=5.0 hurt v16). The cross-covariance loss matches the lag-1
   covariance structure but apparently the dominant DMD modes are driven by higher-order
   dynamics that L_cov cannot constrain at this weight.
2. **recall ceiling at ~0.47 (EMA) / 0.42 (full eval)**: v17 reached 0.521 full eval; v18 fell
   short. The higher cross_cov weight may have constrained generator diversity, counteracting
   diversity_loss=1.0. cross_cov and diversity_loss are in tension: one pushes G toward real
   temporal structure, the other pushes G away from similar outputs.
3. **EMA vs full eval gap**: EMA combined score during training (0.116) appeared close to
   v17's (0.114), but full eval combined = 0.128 vs v17's ~0.103. The fixed val_tensor
   (23,976 windows) is easier to cover than a fresh 2000-sample draw.
4. **W instability late training**: W spiked to 4.5+ at ep232, ep237, ep245 — highest seen
   in any run. Likely caused by the high cross_cov loss making G's gradient landscape rougher.

**v19 direction**: Revert cross_cov to 2.0 (v17 level). Add `--dmd-ckpt-weight 0.05`
(dynamics-aware checkpoint selection, implemented in 66f581b). Keep all v17 winners:
supervisor=1.0, lr_d=5e-5, diversity=1.0, epochs=250.
```bash
./scripts/vinge-launch.sh --version v19 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 250
```

---

### v19 (vinge/GB10, completed ep250, 2026-04-01)

Key changes vs v18: `cross_cov_loss_weight` 5.0 → **2.0** (v17 level), `--dmd-ckpt-weight 0.05`
(dynamics-aware checkpoint selection added), epochs 250.

**Best (ep225, full eval with n_samples=2000)**:

| Metric | v19 | v17 | v18 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01094 | **0.00697** | 0.01105 | <0.005 |
| α-precision | **0.835** | 0.826 | 0.845 | >0.80 |
| β-recall | **0.518** | 0.521 | 0.418 | >0.70 |
| DMD-GEN | **0.6875** | 0.714 | 0.760 | <0.30 |
| AutoCorr | 0.037 | 0.032 | 0.042 | <0.02 |
| Context-FID | 0.13 | **0.03** | 0.06 | <0.05 |
| reuse rate | 0.005 | 0.006 | 0.004 | ~0.757 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **Recall nearly matched v17**: β-recall=0.518 vs v17's 0.521 — within 0.3%. cross_cov=2.0
   + diversity=1.0 is the right balance; this confirms v18's recall regression was entirely
   caused by the higher cross_cov weight.
2. **DMD-GEN improved for the first time**: 0.6875 vs v17's 0.714, v18's 0.760. First
   monotonic improvement in temporal dynamics across any version. `--dmd-ckpt-weight 0.05`
   may have contributed by selecting ep225 (a slightly more dynamics-aware checkpoint).
3. **α-precision=0.835 above target (>0.80)**: Fidelity improved slightly vs v17.
4. **Late peaking confirmed**: Best at ep225/250; v17 best at ep190/200. Training near the
   end of the cosine schedule still yielding improvements.

**What went wrong:**
1. **MMD²=0.01094 vs v17's 0.00697**: 57% worse on the primary metric. This is now the
   central unsolved problem. recall and α-precision are at or above v17; MMD² is not.
   Root cause unclear — could be a stochastic init effect, or cross_cov=2.0 slightly
   spreading the generated distribution relative to v17's unconstrained run.
2. **Context-FID=0.13 vs v17's 0.03**: 4× worse. The latent space quality is degrading
   across versions. Likely caused by the locality split (6 features vs v17-era 5) adding
   reconstruction difficulty, or the cross_cov loss slightly warping the latent geometry.
3. **reuse rate fake=0.005** (real=0.757): Locality gap unchanged. Architectural fix needed
   (z_global conditioning, task #18/#9 in TODO.md).

**Root cause analysis — why MMD² is stuck above v17:**
- v17 achieved MMD²=0.00697 with cross_cov=2.0, diversity=1.0, supervisor=1.0
- v19 uses identical hyperparameters plus dmd_ckpt_weight=0.05
- The dmd_ckpt_weight adds 0.05×DMD-GEN to combined, which could select a slightly
  different checkpoint than pure MMD²+recall would. EP225 is best on combined but
  EP155 had lower EMA MMD²=0.00885 (vs EP225's EMA 0.00945). The dmd_ckpt_weight
  may be trading a small amount of MMD² quality for better dynamics.
- Alternatively, this is stochastic variance: v17's MMD²=0.00697 may reflect a
  favorable random init that v19 didn't replicate.

**v20 direction**: Same hyperparams, extend to 300 epochs. v19 peaked late (ep225/250);
more epochs may allow MMD² to converge further. Consider setting dmd_ckpt_weight=0.01
(reduce dynamics influence to favour MMD²-optimal checkpoint selection) as ablation.
```bash
./scripts/vinge-launch.sh --version v20 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 300
```

---

### v26 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v24: **`diversity_loss_weight` 1.0 → 2.0** to push recall past 0.52. v16 pretrain, dmd_ckpt_weight=0, 200 epochs.

**EMA best (ep190)**: MMD²=0.01255, recall=0.521, combined=0.108

**Best (ep190, full eval with n_samples=2000)**:

| Metric | **v26** | v17 | v24 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00795** | **0.00697** | 0.00798 | <0.005 |
| α-precision | 0.887 | 0.826 | 0.835 | >0.80 |
| β-recall | 0.401 | **0.521** | 0.503 | >0.70 |
| DMD-GEN | **0.710** | 0.714 | 0.717 | <0.30 |
| AutoCorr | 0.049 | **0.032** | 0.036 | <0.02 |
| Context-FID | 0.04 | **0.03** | 0.08 | <0.05 |
| reuse rate | 0.002 | 0.006 | 0.007 | ~0.853 |

Full eval combined: v26=0.128, v17=0.103, v24=0.107. **Did NOT beat v17 or v24.**

**What worked:**
1. **MMD²=0.00795** — tied with v24 (0.00798) as 2nd best ever on full eval. Three runs (v17/v24/v26) now below 0.01.
2. **α-precision=0.887** — 2nd best ever (v22: 0.927). diversity_loss=2.0 did NOT hurt fidelity.
3. **DMD-GEN=0.710** — best ever on full eval (v17: 0.714, v24: 0.717). First run below 0.714.
4. **Context-FID=0.04** — 2nd best ever (v17: 0.03). Much better than v24's 0.08.

**What went wrong — diversity_loss=2.0 inflated EMA recall without translating to full eval:**
1. **EMA recall=0.521 → full eval recall=0.401**: 23% drop. This is the worst EMA→full eval
   recall gap since v20. diversity_loss=2.0 pushes G to produce more diverse outputs on the
   fixed val_tensor, but these diverse outputs don't cover the full distribution well.
2. **β-recall=0.401** (v17: 0.521, v24: 0.503): Worst recall of any v16-pretrain run with
   dmd_ckpt_weight=0. diversity_loss=2.0 appears counterproductive for actual coverage.
3. **Full eval combined=0.128** — worse than both v17 (0.103) and v24 (0.107). The recall
   regression dominates the combined score despite improvements on every other metric.

**Root cause — diversity_loss mechanism mismatch:**
The MSGAN diversity loss penalizes G for producing similar outputs for different latent
inputs. At weight=2.0, this pushes G to spread outputs farther apart, which inflates recall
on the fixed val_tensor (fixed set of 23,976 windows is easier to cover with spread-out
samples). But the *directions* of spread are not aligned with the real data manifold — G
spreads into low-density regions that don't match any real traces. Full eval with fresh
2000-sample draws exposes this: coverage of the actual distribution is worse.

**Key finding**: diversity_loss_weight=1.0 is the sweet spot. Lower hurts diversity (v14f),
higher inflates EMA recall without improving full eval. Future diversity improvements need
architectural changes (z_global conditioning), not loss weight tuning.

**v28 direction**: TBD. The v16-pretrain + dmd_ckpt_weight=0 recipe has been thoroughly
explored (v24/v25/v26). v17 remains ATB after 9 subsequent runs. Time for structural changes.

---

### v30 (vinge/GB10, aborted ep54, 2026-04-01)

Key change vs v24: **`n_critic` 2 → 3** to give critic more power. Unconditional, v16 pretrain,
dmd_ckpt_weight=0, standard losses.

**EMA best (ep25)**: MMD²=0.01386, recall=0.370, combined=0.140

**Full eval (ep25 best)**:

| Metric | v30 | v17 | v24 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01640 | **0.00697** | 0.00798 | <0.005 |
| α-precision | **0.929** | 0.826 | 0.835 | >0.80 |
| β-recall | 0.468 | **0.521** | 0.503 | >0.70 |
| DMD-GEN | 0.751 | **0.714** | 0.717 | <0.30 |
| Context-FID | 0.17 | **0.03** | 0.08 | <0.05 |

Full eval combined: 0.016 + 0.2*(1-0.468) = 0.123. **Did NOT beat v17.**

Aborted at ep54 — recall falling (0.37→0.30), W trending to 2.8.

**Key findings:**

1. **α-precision=0.929 tied all-time best** (v22: 0.927). n_critic=3 produces extremely
   high-fidelity outputs — the stronger critic forces G to stay close to the real manifold.

2. **Full eval recall (0.468) > EMA recall (0.370)** — gap inverted for the first time.
   n_critic=3 makes EMA pessimistic: the extra critic steps create a harder adversary that
   the EMA-smoothed generator handles less well than the live weights during training.

3. **n_critic=3 is too aggressive** — the stronger critic suppresses diversity/recall in
   favour of fidelity. Recall peaked at 0.37 (EMA) and was falling by ep54. W trending
   upward (2.8) suggests impending cycling.

**Verdict**: n_critic=2 confirmed as the right balance. n_critic=3 trades recall for fidelity,
which is the wrong trade-off (recall is the binding constraint, not precision).

---

### v29 (vinge/GB10, crashed ep96, 2026-04-01)

**Recipe**: cond_dim=10 + full loss recipe (cross_cov=2.0, acf=0.2, fft=0.05, moment=0.1,
quantile=0.2). Used v28 pretrain (cond_dim=10 compatible). n_critic=2.

**EMA best (ep20)**: MMD²=0.00971, recall=0.674, combined=0.075

**Full eval (ep20 best, with real descriptors)**: MMD²=0.021, α-precision=0.882,
β-recall=0.384, DMD-GEN=0.702, Context-FID=0.08

| Metric | v29 (full eval) | v29 (EMA) | v17 | Target |
|--------|-----------------|-----------|-----|--------|
| MMD² | 0.021 | **0.00971** | **0.00697** | <0.005 |
| β-recall | 0.384 | **0.674** | **0.521** | >0.70 |
| combined | 0.144 | **0.075** | **0.103** | — |
| α-precision | **0.882** | — | 0.826 | >0.80 |
| DMD-GEN | **0.702** | — | 0.714 | <0.30 |

Full eval combined: 0.021 + 0.2*(1-0.384) = 0.144. **Did NOT beat v17.**

**GAN cycling crash**: W=2.5 at ep88 → 13.1 at ep95 → 17.3 at ep96 (abort triggered).
Auxiliary losses delayed cycling vs v28 (ep96 vs ep79) but did not prevent it.

**Key findings:**

1. **Conditioning produces phenomenal EMA metrics** — recall=0.674 and combined=0.075 are
   both all-time records by a wide margin. The conditioning mechanism genuinely helps G
   learn the data distribution faster and more completely.

2. **Massive EMA→full eval gap**: recall 0.674→0.384 (43% drop), MMD² 0.00971→0.021 (2.2×).
   The model overfits to conditioning descriptors — generates well for training-time
   descriptors but doesn't generalize to fresh eval descriptors.

3. **DMD-GEN=0.702 on full eval is best ever** (v17: 0.714). Conditioning may help temporal
   dynamics by giving G a workload identity signal.

4. **GAN cycling still happens WITH auxiliary losses** — just delayed vs v28 (ep96 vs ep79).
   The conditioning creates instability that the current training setup can't handle.
   Neither loss recipe stabilization (v29) nor loss stripping (v28) prevents it.

**Conditioning verdict**: Powerful but currently unusable. Two problems must be solved:
(a) GAN cycling (W explosion) — may need CFG dropout or lower lr_g;
(b) EMA→full eval gap (descriptor overfitting) — may need CFG or descriptor augmentation.

**v30 direction**: Step back from conditioning. Return to proven unconditional recipe (v16
pretrain, no conditioning, dmd_ckpt_weight=0) but try n_critic=3 (up from 2) to give the
critic more power. Quick experiment, no code changes needed.

---

### v28 (vinge/GB10, crashed ep79, 2026-04-02)

**Two major changes**: z_global workload conditioning (`--cond-dim 10`) + loss ablation
(stripped to WGAN + FM + supervisor + diversity only; zeroed FFT, moment, quantile, ACF,
cross_cov, locality). Fresh pretrain required (architecture change).

**EMA best (ep70)**: MMD²=0.01183, recall=0.508, combined=0.110

**Full eval (ep70, with random conditioning N(0,0.5))**: MMD²=0.026, α-precision=0.594,
β-recall=0.340, DMD-GEN=0.715, Context-FID=0.44

**Note**: Full eval numbers are unreliable — eval.py used random N(0,0.5) conditioning
instead of real workload descriptors. Fixed in eval.py post-v28 (now computes descriptors
from real val windows when cond_dim > 0).

| Metric | v28 (EMA) | v17 | v24 | Target |
|--------|-----------|-----|-----|--------|
| MMD² | **0.01183** | **0.00697** | 0.00798 | <0.005 |
| β-recall | **0.508** | 0.521 | 0.503 | >0.70 |
| combined | **0.110** | 0.114 | 0.107 | — |

**BREAKTHROUGH — conditioning is the path forward:**
1. **combined=0.110 at ep70** — best-ever early trajectory. v17 didn't reach 0.114 until ep190.
   At the same training stage, v28 was already outperforming every previous run. Conditioning
   gives the generator a massive head start by telling it *what kind* of workload to produce.
2. **Recall hit 0.526 at ep75** — briefly beat v17's all-time record of 0.521. First run to
   ever exceed v17 on recall (albeit EMA, not full eval).

**FAILURE — loss ablation caused GAN cycling crash:**
1. **W spiked to 2.67 at ep79** and training diverged. Without cross_cov, ACF, FFT, moment,
   and quantile losses, the generator had no auxiliary gradients to stabilize against critic
   overpowering. These losses act as implicit regularizers on G's output distribution.
2. The ablation was too aggressive — stripping ALL auxiliary losses at once made it impossible
   to isolate which ones matter. v29 restores the full proven loss recipe.

**Full eval was wrong (now fixed):**
eval.py used `torch.randn(n, cond_dim) * 0.5` as conditioning — random noise that doesn't
match the real workload descriptor distribution. This made α-precision=0.594 (G produces
"workloads" that don't exist) and inflated MMD². Fixed: eval.py now computes descriptors
from real val windows via `compute_window_descriptors()`.

**v29 direction**: Keep cond_dim=10 + restore full loss recipe (cross_cov=2.0, acf=0.2,
fft=0.05, moment=0.1, quantile=0.2). Use v28 pretrain (same architecture).

---

### v27 (vinge/GB10, AVATAR, aborted ep38, 2026-04-02)

**AVATAR architecture**: AAE (latent discriminator + distribution loss) + supervisor-assisted
reconstruction + BatchNorm GRU. Fresh pretrain. First major architectural departure from TimeGAN.

**Best (ep20)**: MMD²=0.055, recall=0.172, combined=0.221.

**FAILED.** Aborted at ep38 — critic overpowered generator from the start.

**What went wrong:**
1. **W=3.5–4.7 from epoch 1** — the unbounded latent space (no Sigmoid) gave the critic
   a much easier discrimination task. The critic immediately dominated.
2. **MMD² worsened over training**: 0.055 (ep20) → 0.074 (ep30) → 0.077 (ep35). Going backwards.
3. **G loss approaching zero** (-0.39 at ep36) — generator losing the adversarial game badly.
4. **G warmup supervisor loss diverged** (0.31 → 0.72 over 100 epochs) — the unbounded latent
   space may have made the supervisor's prediction task harder.

**Root cause**: Removing the Sigmoid bound from Encoder/Generator output created a latent space
that's too easy for the critic to exploit. The AAE latent discriminator was supposed to
regularize this, but the sequence-level critic (LSTM+attention) is much more powerful than
the per-timestep latent discriminator (MLP), creating an asymmetry.

**What to try next time**: Keep the Sigmoid-bounded [0,1] latent space but add the AAE
discriminator as an auxiliary loss (force q(z|x) toward a Beta distribution or truncated
Gaussian in [0,1]). This keeps the critic's job equally hard while adding latent regularization.

---

### v25 (vinge/GB10, aborted ep59, 2026-04-02)

Same recipe as v24: v16 pretrain, dmd_ckpt_weight=0, 200 epochs. Different seed.

**Best (ep50, full eval)**: MMD²=0.01272, α-precision=0.803, β-recall=0.359, DMD-GEN=0.758.

**Did NOT beat v17.** Weak seed — recall stuck at 0.18–0.28 for 50 epochs, W collapsed to
0.15 at ep55 (critic lost leverage). Aborted early. With v16 pretrain and dmd_ckpt_weight=0,
seed variance remains large: v24 got combined=0.107, v25 got ~0.169. Rolling dice isn't
enough — time to try structural changes.

---

### v24 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v22/v23: **`dmd_ckpt_weight=0`** (v22/v23 used 0.05). Same v16 pretrain, 200 epochs.
This exactly replicates v17's checkpoint selection logic (pure MMD²+recall, no dynamics penalty).

**Best (ep170, full eval with n_samples=2000)**:

| Metric | **v24** | v17 | v22 | Target |
|--------|---------|-----|-----|--------|
| MMD² | **0.00798** | **0.00697** | 0.01751 | <0.005 |
| α-precision | 0.835 | 0.826 | **0.927** | >0.80 |
| β-recall | 0.503 | **0.521** | 0.471 | >0.70 |
| DMD-GEN | 0.717 | **0.714** | 0.757 | <0.30 |
| AutoCorr | 0.036 | **0.032** | 0.052 | <0.02 |
| Context-FID | 0.08 | **0.03** | 0.09 | <0.05 |
| reuse rate | 0.007 | 0.006 | 0.002 | ~0.853 |

**v24 is the second-best run ever on full eval, narrowly behind v17.**
Full eval combined: v24=0.107, v17=0.103. EMA combined: v24=**0.097**, v17=0.114.

**What worked — `dmd_ckpt_weight=0` CONFIRMED as the fix:**
1. **MMD²=0.00798** — second-best ever on full eval (v17: 0.00697). First run since v17 to
   break below 0.01 on full eval.
2. **β-recall=0.503** — third-best ever (v17: 0.521, v19: 0.518). Recall hit 0.545 on EMA
   at ep170 — the highest EMA recall ever recorded.
3. **EMA combined=0.097** — beat v17's EMA combined of ~0.114 by 15%. The EMA→full eval gap
   was only 1.23× (0.0065→0.008 on MMD²), the smallest of any run.
4. **Sustained excellence**: ep95–ep195 had 20+ consecutive evals with combined < 0.12.
   No previous run sustained v17-level metrics for more than a few evals.

**What went wrong:**
1. **Full eval combined 0.107 vs v17's 0.103**: Missed by 4%. The remaining gap is on MMD²
   (0.008 vs 0.007) and recall (0.503 vs 0.521). Likely stochastic seed variance.
2. **Context-FID=0.08** (v17: 0.03): Latent space quality slightly worse. This has been a
   persistent issue since v18+ — may be related to the fresh preprocessor fit each run.
3. **α-precision=0.835** (v22: 0.927): Lower fidelity than v22, though above target.

**Key finding — `dmd_ckpt_weight` was actively harmful:**
v22/v23 used `dmd_ckpt_weight=0.05`, adding ~0.035 to the combined score (DMD-GEN≈0.7 × 0.05).
This doesn't just inflate scores — it changes which checkpoints get saved as best.pt. The
dynamics penalty causes the selector to prefer checkpoints with slightly better DMD-GEN but
worse MMD²/recall. Since DMD-GEN is stuck at ~0.7 regardless of checkpoint, the penalty
sacrifices the primary metrics for no gain.

Evidence: v22 (dmd_ckpt_weight=0.05) full eval combined=0.154; v24 (dmd_ckpt_weight=0) full
eval combined=0.107. Same pretrain, same hyperparams, only difference is checkpoint selection.

**Recommendation: never use `dmd_ckpt_weight > 0`** until DMD-GEN is responsive to training
(currently architectural limitation). Default in config.py should be 0.

**v25 direction**: Same recipe (v16 pretrain, dmd_ckpt_weight=0, 200 epochs). v24 nearly
matched v17 — another seed may beat it outright.

---

### v23 (vinge/GB10, aborted ep187, 2026-04-02)

Same recipe as v22: v16 pretrain, 200 epochs, identical hyperparams, different random seed.

**Best (ep165, full eval with n_samples=2000)**:

| Metric | v23 | v22 | v17 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01967 | 0.01751 | **0.00697** | <0.005 |
| α-precision | 0.766 | **0.927** | 0.826 | >0.80 |
| β-recall | 0.357 | 0.471 | **0.521** | >0.70 |
| DMD-GEN | **0.699** | 0.757 | 0.714 | <0.30 |
| Context-FID | 0.05 | 0.09 | **0.03** | <0.05 |
| reuse rate | 0.006 | 0.002 | 0.006 | ~0.850 |

**Did NOT beat v17.** Weaker than v22 on most metrics. Aborted at ep187 (not worth finishing).

**What worked:**
1. **DMD-GEN=0.699** — new all-time best, first run below 0.700. May be stochastic but
   suggests this seed found slightly better temporal dynamics despite weaker overall quality.
2. **Late surge at ep165**: recall jumped from 0.27→0.38 and MMD² dropped to 0.014.
   Confirms the late-LR-decay improvement pattern seen in v17/v22 is real, not version-specific.

**What went wrong:**
1. **Recall stuck at 0.15–0.25 for 130 epochs**: Unlike v22 (which had recall >0.30 by ep40),
   v23 was trapped in a low-diversity mode for most of training. The ep165 spike to 0.379
   was a brief escape, not sustained.
2. **α-precision=0.766** — worst since v16 (0.833). This seed produced lower-quality samples.

**Conclusion**: With v16 pretrain and identical hyperparams, random seed variance is large.
v17=0.114, v22=0.154, v23=0.169 on combined score. The pretrain is necessary but not
sufficient — good seeds matter.

**v24 direction**: v17 didn't have `dmd_ckpt_weight` (feature added post-v17). v22/v23 used
0.05. Try `dmd_ckpt_weight=0` to exactly replicate v17's checkpoint selection logic.

---

### v22 (vinge/GB10, completed ep200, 2026-04-02)

Key change vs v21: **v16 pretrain checkpoint** (same as v17 used), 200 epochs (v17's schedule).
Hyperparams identical to v17: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0,
dmd_ckpt_weight=0.05.

**Best (ep190, full eval with n_samples=2000)**:

| Metric | v22 | v17 | v21 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01751 | **0.00697** | 0.01485 | <0.005 |
| α-precision | **0.927** | 0.826 | 0.873 | >0.80 |
| β-recall | 0.471 | **0.521** | 0.366 | >0.70 |
| DMD-GEN | 0.757 | **0.714** | 0.720 | <0.30 |
| AutoCorr | 0.052 | **0.032** | 0.069 | <0.02 |
| Context-FID | 0.09 | **0.03** | 0.03 | <0.05 |
| reuse rate | 0.002 | 0.006 | 0.007 | ~0.837 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked — v16 pretrain hypothesis CONFIRMED:**
1. **v22 is the best run since v17** on combined score (EMA 0.154 vs v17's 0.114). v18–v21
   all used later pretrains and stagnated; v22 with v16's pretrain showed v17-like trajectory
   with continuous improvement through ep190.
2. **α-precision=0.927** — new all-time best across all versions. First run to significantly
   exceed the 0.85 target. The v16 pretrain produces a cleaner latent space for G.
3. **β-recall=0.471** — second-best ever (v17: 0.521). Recall climbed steadily from 0.25 to
   0.45 over the full 200 epochs, peaking at ep190 (same epoch as v17).
4. **Late-training surge confirmed**: Best improved from ep105 (combined 0.166) to ep190
   (combined 0.154) — the same late-surge pattern as v17.

**What went wrong:**
1. **MMD²=0.01751** (v17: 0.00697): 2.5× worse. EMA was 0.01059 — a 1.65× EMA/full-eval
   gap persists even with the EMA-save fix. The remaining gap is likely from fixed val_tensor
   (23,976 windows) vs fresh samples (2000 from 3234 files). v17 also had this gap but its
   underlying EMA MMD² was better to start with.
2. **DMD-GEN=0.757** (v17: 0.714, v19: 0.688): Temporal dynamics slightly worse. DMD-GEN
   remains stubbornly above 0.70 across all versions — architectural fix needed.
3. **Context-FID=0.09** (v17: 0.03, v21: 0.03): Latent space quality worse than v17 despite
   the same pretrain. May be stochastic or related to the fresh preprocessor fit.
4. **W spike ep198** (W=6.6): Late instability, though best.pt at ep190 captured before this.

**Key finding — pretrain quality is the dominant variable:**
v22 used the identical v16 pretrain as v17 and produced a v17-like training trajectory. v18–v21
used pretrains from later runs (v17, v19, v20) and all stagnated early. The pretrain captures
the E/R/S/G weight initialization quality; later pretrains may carry artifacts from extended
GAN training that make the subsequent GAN phase harder to optimize.

This means: **always use the v16 pretrain for future runs** unless the architecture changes
(hidden_size, latent_dim, input features). The 35-minute pretrain cost is amortized across
all downstream runs.

**v23 direction:**
Same v16 pretrain, same hyperparams, 200 epochs — another roll of the dice with a different
random seed. If v17 and v22 both peaked at ep190 with the v16 pretrain but v17 got better
MMD², the remaining variance is stochastic. Multiple runs will tell us the distribution.

---

### v21 (vinge/GB10, early-stopped ep235, 2026-04-01)

Key change vs v20: **EMA-save fix** — `best.pt` now saves EMA weights as `G` key instead of
live weights. Same hyperparams: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0,
dmd_ckpt_weight=0.05, epochs=300, checkpoint_every=5. Pretrain from v20.

**Best (ep35, full eval with n_samples=2000)**:

| Metric | v21 | v17 | v20 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.01485 | **0.00697** | 0.03215 | <0.005 |
| α-precision | **0.873** | 0.826 | 0.826 | >0.80 |
| β-recall | 0.366 | **0.521** | 0.407 | >0.70 |
| DMD-GEN | 0.720 | **0.714** | 0.719 | <0.30 |
| AutoCorr | 0.069 | **0.032** | 0.049 | <0.02 |
| Context-FID | **0.03** | 0.03 | 0.14 | <0.05 |
| reuse rate | 0.007 | 0.006 | 0.004 | ~0.685 |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **EMA-save fix confirmed**: Full eval MMD² (0.01485) is *better* than training-time EMA
   MMD² (0.02163) — the gap is now inverted. v20's 4.3× problem is solved. This validates
   that EMA weights are genuinely better than live weights for evaluation.
2. **α-precision=0.873** — new all-time best across all versions (target >0.85 met).
3. **Context-FID=0.03** — matched v17, much better than v20's 0.14.

**What went wrong:**
1. **Stagnated at ep35**: Best combined score never improved after epoch 35. 40 consecutive
   evals without improvement triggered early stopping at ep235. v17 peaked at ep190 — v21
   never had the late-training improvement surge.
2. **β-recall=0.366** (v17: 0.521): 30% worse coverage. The model peaked early with decent
   precision but poor diversity, then failed to expand coverage as training continued.
3. **W instability late training**: W spiked to 10.7 at ep231, 7.4 at ep232 — the highest
   seen in any run. The critic was dominating by endgame. W was generally higher throughout
   v21 (2–5 range) vs v17 (~1–2 range).
4. **AutoCorr=0.069** (v17: 0.032): 2× worse temporal fidelity despite similar DMD-GEN.

**Root cause analysis — why v21 stagnated while v17 didn't:**
- v17 used v16's pretrain checkpoint. v21 used v20's pretrain checkpoint.
- v20 trained for 300 epochs with W climbing to 6+ by endgame. The pretrained E/R/S/G
  weights from v20 may carry residual critic-adaptation artifacts that make the GAN phase
  harder to optimize from the start.
- v17's LR trajectory covered the sweet spot (cosine from 1e-4 to 5e-6 over 200 epochs)
  at the right pace. v21's 300-epoch schedule decays more slowly — the LR was still high
  enough at ep35 to find a good point but too high thereafter to refine it.
- Stochastic variance: v17 may have hit a favorable random trajectory that v18–v21 haven't
  replicated. Four consecutive runs with identical hyperparams (v18–v21) have all failed to
  match v17.

**v22 direction:**
The pretrain checkpoint quality hypothesis is testable. Options:
1. Use v17's pretrain checkpoint (if available) instead of v21's
2. Fresh pretrain from scratch (costs ~35 min but eliminates carryover artifacts)
3. Reduce epochs to 200 (v17's schedule) to match the LR decay profile that worked
4. Same hyperparams, different random seed — if v17 is stochastic, keep rolling the dice

---

### v20 (vinge/GB10, completed ep300, 2026-04-02)

Key changes vs v19: extended to **300 epochs** (v19 peaked late at ep225/250).
Hyperparameters identical: supervisor=1.0, lr_d=5e-5, diversity=1.0, cross_cov=2.0, dmd_ckpt_weight=0.05.

**Best (ep210, full eval with n_samples=2000)**:

| Metric | v20 | v17 | v19 | Target |
|--------|-----|-----|-----|--------|
| MMD² | 0.03215 | **0.00697** | 0.01094 | <0.005 |
| α-precision | 0.826 | 0.826 | 0.835 | >0.80 |
| β-recall | 0.407 | **0.521** | 0.518 | >0.70 |
| DMD-GEN | 0.719 | 0.714 | **0.688** | <0.30 |
| AutoCorr | 0.049 | 0.032 | 0.037 | <0.02 |
| Context-FID | 0.14 | **0.03** | 0.13 | <0.05 |
| reuse rate | 0.004 | 0.006 | 0.005 | ~0.806 |

**EMA (training-time) vs full eval divergence — the critical finding:**
| Metric | EMA ep210 | Full eval | Ratio |
|--------|-----------|-----------|-------|
| MMD² | 0.00755 | 0.03215 | 4.3× worse |
| β-recall | 0.549 | 0.407 | 25% lower |

**Did NOT beat v17.** v17 remains all-time best.

**What worked:**
1. **EMA metrics were best-ever**: EMA ep210 showed MMD²=0.00755 and recall=0.549, both
   individually better than v17 (0.00697, 0.521). Training-time trajectory was genuinely
   better than any prior run at this stage.
2. **Early trajectory**: best ep50 combined (0.158) and ep100 combined (0.154) were the best
   ever seen, confirming 300 epochs was the right direction.
3. **Recall hit 0.539 (ep195 EMA)** — new all-time EMA high.

**What went wrong — EMA/full-eval gap is now the primary problem:**
1. **4.3× MMD² divergence**: EMA 0.00755 → full eval 0.03215. This is the largest
   EMA/full-eval gap observed (v14f was noted at 3–4×; v20 matches/exceeds that).
2. **Root cause — checkpoint saves LIVE weights, not EMA weights**: The combined score used
   for checkpoint selection is computed from EMA model metrics. But `best.pt` saves the
   live (non-EMA) G weights at that epoch. The EMA weights are smoothed across hundreds
   of epochs; the live weights at ep210 reflect only recent gradient updates, which may
   be noisier. The late W instability (W=6.1 at ep271, 5.7 at ep295) suggests the live
   weights were being aggressively updated while EMA stayed smooth.
3. **Fixed val_tensor may be too easy**: Training eval draws from the same 23,976 windows
   every time. Full eval draws 2000 fresh windows from the full 3234-file corpus. The
   fixed val set may be unrepresentative of the full distribution.
4. **W instability late training (ep271–295)**: W repeatedly spiked above 4.0 in the final
   30 epochs. This indicates the critic was dominating, degrading G's live weights even
   while EMA remained smooth.

**v21 direction — fix the EMA/full-eval gap:**
Primary fix: **save EMA weights as `best.pt`**, not live weights. The checkpoint selection
already uses EMA metrics; the checkpoint should store the same weights that were evaluated.
This requires a small change to `train.py`: when saving `best.pt`, write `ema_G_state`
instead of `G.state_dict()`.

Secondary: Consider a fresh random val draw (rather than fixed val_tensor) for training-time
eval, to reduce the divergence between training eval and full eval.

Keep same hyperparams pending the EMA-save fix.
```bash
# After implementing EMA save fix in train.py:
./scripts/vinge-launch.sh --version v21 --supervisor-loss-weight 1.0 --lr-d 5e-5 \
  --diversity-loss-weight 1.0 --cross-cov-loss-weight 2.0 --dmd-ckpt-weight 0.05 --epochs 300
```

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

| Metric | v13 | v14g | v16 | v17 | v18 | v19 | v20 | v21 | v22 | v23 | v24 | **v31** | Target | Description |
|--------|-----|------|-----|-----|-----|-----|-----|-----|-----|-----|-----|---------|--------|-------------|
| MMD² | 0.01335 | 0.018 | 0.042 | **0.00697** | 0.01105 | 0.01094 | 0.03215 | 0.01485 | 0.01751 | 0.01967 | 0.00798 | **0.00769** | < 0.005 | Kernel distribution distance |
| β-recall | 0.455 | 0.372 | 0.228 | 0.521 | 0.418 | 0.518 | 0.407 | 0.366 | 0.471 | 0.357 | 0.503 | **0.596** | > 0.7 | Coverage: fraction of real modes covered |
| α-precision | 0.812 | 0.910 | 0.833 | 0.826 | 0.845 | 0.835 | 0.826 | 0.873 | 0.927 | 0.766 | 0.835 | **0.953** | > 0.85 | Fidelity: fraction of generated that is realistic |
| DMD-GEN | 0.771 | 0.700 | 0.744 | 0.714 | 0.760 | **0.688** | 0.719 | 0.720 | 0.757 | **0.699** | 0.717 | 0.723 | < 0.3 | Temporal dynamics divergence (0 = perfect) |
| Context-FID | 0.05 | 0.03 | 0.15 | **0.03** | 0.06 | 0.13 | 0.14 | **0.03** | 0.09 | 0.05 | 0.08 | **0.03** | < 0.05 | Fréchet in encoder latent space |
| reuse-rate | 0.000 | 0.000 | 0.004 | 0.006 | 0.004 | 0.005 | 0.004 | 0.007 | 0.002 | 0.006 | 0.007 | 0.005 | > 0.1 | Fraction of obj_id repeats (sequential access) |

Note: **v31 is the all-time best on full eval** (combined=0.089), dethroning v17 (0.103) after 14 subsequent runs. v17's full eval correctly used G_ema weights
(eval.py has `ckpt.get("G_ema", ckpt["G"])`) — its results were not affected by the EMA-save
bug. The EMA-save fix in v21 ensures consistency for tools that only read the `G` key, but
was NOT the reason v18–v21 failed to match v17. The real differentiator was the **v16 pretrain
checkpoint** — v22 using v16's pretrain (same as v17) restored v17-class performance, while
v18–v21 using later pretrains all stagnated. This is the key finding of the v18–v22 series.

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
