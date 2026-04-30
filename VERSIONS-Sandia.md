# Sandia / newgan Version Log

This file tracks Sandia-owned `newgan/` checkpoints and race-relevant updates.

---

## Sandia v1 Baseline Training (2026-04-29) — First Self-Contained Training

**Status:** In progress

Sandia launched its first self-contained training pipeline with explicit pretrain
selection and cross-seed validation. The training code is in `newgan/train.py`
and uses the llgan models without modifying llgan/ code.

### Training Pipeline

The Sandia training pipeline implements:
1. **Phase 1: Autoencoder pretraining** - Train Encoder + Recovery to round-trip
   real windows, minimizing reconstruction MSE
2. **Phase 2: Supervisor pretraining** - Train Supervisor to predict next latent
   step from real trajectories (teacher forcing)
3. **Phase 2.5: Generator warm-up** - Train Generator to imitate Supervisor
   outputs WITHOUT Critic gradients (stabilizes against early mode collapse)
4. **Phase 3: Joint GAN training** - Full WGAN-SN training with supervisor
   consistency loss

### Key Design Decisions

- **Self-contained code**: All training code lives in `newgan/`, importing llgan
  models via `sys.path` manipulation rather than modifying llgan/
- **Pretrain ranking**: Each pretrain is evaluated on held-out validation to
  rank by downstream quality potential (lower AE loss = better)
- **Cross-seed validation**: All Phase 3 runs use multiple seeds (42, 11, 7)
  to ensure ATBs are reproducible, not lottery wins
- **Long-horizon focus**: Combined checkpoint score includes MMD² + diversity
  metric to prevent mode collapse; HRC-MAE tracked for long-rollout fidelity

### Initial Run: s001_test

- Configuration: tencent_block_1M, 5 epochs (debug), batch_size=4
- Pretrain epochs: 1 each (AE + supervisor + G warmup)
- Goal: Verify training pipeline works end-to-end

### Peer Review of LLNL (2026-04-29)

**LLNL v233 Status (tencent_v233):**
- LRU diagnostic has import issues (ModuleNotFoundError)
- LRU diagnostic doesn't carry retrieval_state like generate.py
- No durable training log in checkpoint directory
- ep10 frozen score: 0.26224 (combined), HRC-MAE=0.5478 — FAILS launch gates
- Recommendation: Kill v233, launch retrieval-state carry run

**LLNL v229 (tencent):**
- Frozen-bundle ATB: 0.039 (seed=5)
- Phase-PMF Atlas with real trace calibration
- Uses correct trace directory: `/home/darrell/traces/tencent_block_1M/`

### Peer Review of LANL (2026-04-29)

**LANL PhaseAtlas/NeuralAtlas:**
- **Tencent best**: PhaseAtlas strict-holdout with transition_blend=0.0, HRC-MAE=0.01845
- **Alibaba best**: NeuralAtlas with transition_blend=0.5, HRC-MAE=0.00183
- LANL wins Alibaba (0.00183 < LLNL's 0.001937)
- LLNL wins Tencent (0.039 < LANL's 0.01845... wait, that's not right)

**Correction:** Looking at the HRC-MAE results:
- LANL Tencent PhaseAtlas: 0.01845 (transition_blend=0.0)
- LANL Alibaba NeuralAtlas: 0.00183
- LLNL Tencent: 0.039 frozen-bundle ATB (different metric!)
- LLNL Alibaba: 0.001937 HRC-MAE

The metrics are not directly comparable - LLNL's ATB is a different evaluation
protocol. LLNL's 0.039 is from the frozen-bundle protocol (4 files, short windows),
while LANL's 0.01845 is from 100k long-rollout HRC-MAE evaluation.

**Current race standings (HRC-MAE on 100k long-rollout):**
| Team | Corpus | HRC-MAE | Method |
|------|--------|---------|--------|
| LLNL | Alibaba | 0.001937 | Phase-PMF Atlas |
| LANL | Alibaba | 0.00183 | NeuralAtlas |
| LLNL | Tencent | 0.039 | Frozen-bundle ATB |
| LANL | Tencent | 0.01845 | PhaseAtlas (strict-holdout) |

### Next Steps

1. Wait for s001_test training to complete (currently running)
2. Evaluate frozen-bundle scores across 3 seeds (42, 11, 7)
3. Compare against LLNL and LANL results
4. Identify improvement path based on results

---

## Format

Each entry should include:
- Date and version name
- Description of changes/new features
- Configuration details (corpus, hyperparameters)
- Evaluation results (ATB, HRC-MAE, etc.)
- Peer review feedback and responses
