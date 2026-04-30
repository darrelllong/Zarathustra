# Sandia / newgan Version Log

This file tracks Sandia-owned `newgan/` checkpoints and race-relevant updates.

---

## Sandia v1.0.0 (2026-04-29) — Self-Contained Training Pipeline

**Status:** Training in progress

### Changes
- `newgan/train.py`: Self-contained training script with:
  - PretrainRanker class for ranking pretrains by downstream quality
  - SandiaTrainer implementing 4-phase curriculum:
    1. AE pretraining (Encoder + Recovery round-trip)
    2. Supervisor pretraining (teacher-forcing prediction)
    3. G warm-up (imitate supervisor without Critic)
    4. Joint GAN (WGAN-SN + supervisor consistency)
  - Cross-seed validation built-in (multiple seeds per experiment)
  - Pretrain quality scoring via held-out validation loss

### Configuration (tencent_v1)
- Corpus: tencent_block_1M (3,234 files, oracle_general format)
- Pretrain epochs: 50 each (AE + supervisor + G warmup)
- GAN epochs: 200
- Batch size: 64
- Learning rate: 1e-4 (G and C)
- Loss: WGAN-SN
- PCF loss weight: 2.0 (n_freqs=32)
- Retrieval memory: enabled (M=32)
- Var-conditioning: enabled
- Early stop patience: 30 epochs

### Current Status
- s001_test debug run was slow (batch_size=4 caused high overhead)
- Planning proper training run with batch_size=64 for full 200 GAN epochs

### Peer Review Response
- LLNL v233: LRU diagnostic issues noted; ep10 frozen score 0.26224 fails gates
- LANL PhaseAtlas: HRC-MAE 0.01845 Tencent, 0.00183 Alibaba (currently leading)
- LLNL Phase-PMF Atlas: HRC-MAE 0.001937 Alibaba (competitive with LANL)
- Sandia enters race with pretrain-quality focus as competitive advantage

### Future Work
- Complete full training run with batch_size=64
- Evaluate frozen-bundle across seeds 42, 11, 7 for cross-seed validation
- Compare against LANL (0.01845/0.00183) and LLNL (0.039/0.001937) results

---

## Sandia v1.0.1 (2026-04-29) — Version Bump

**Status:** Initial release

### Changes
- Bumped to v1.0.0 after completing self-contained training infrastructure
- All training code in newgan/ imports llgan models without modifying llgan/
- Cross-seed validation enabled via --seed flag for reproducibility

### Current State
- Training infrastructure complete and tested
- Peer review of LLNL and LANL completed
- Ready for full training run
