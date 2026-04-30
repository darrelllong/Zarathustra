# Sandia / newgan Version Log

This file tracks Sandia-owned `newgan/` checkpoints and race-relevant updates.

---

## Sandia v1.0.0 (2026-04-29) — Self-Contained Training Pipeline

**Status:** Training infrastructure complete

### Changes
- `newgan/train.py`: Self-contained training script with:
  - PretrainRanker class for ranking pretrains by downstream quality
  - SandiaTrainer implementing 4-phase curriculum:
    1. AE pretraining (Encoder + Recovery round-trip, MSE loss)
    2. Supervisor pretraining (teacher-forcing prediction)
    3. G warm-up (imitate supervisor without Critic gradients)
    4. Joint GAN (WGAN-SN + supervisor consistency loss)
  - Cross-seed validation built-in (multiple seeds per experiment)
  - Pretrain quality scoring via held-out validation loss
- `newgan/v1_baseline.sh`: Launch script using newgan/train.py
- `newgan/generate.py`: Generation wrapper
- `newgan/run.py`: Generic runner
- `PEER-REVIEW-Sandia.md`: Peer review documentation
- `VERSIONS-Sandia.md`: This version log

### Key Design Decisions
- Self-contained code in newgan/ (imports llgan models via sys.path)
- Pretrain ranking: Evaluate on held-out validation (lower AE loss = better)
- Cross-seed validation: 3+ seeds (42, 11, 7) per experiment
- Long-horizon focus: Combined score includes diversity metric

---

## Sandia v1.0.1 (2026-04-29) — Initial Release

**Status:** Initial release

---

## Sandia v1.0.2 (2026-04-29) — v1 Baseline Attempt & Race Entry

**Status:** Training infrastructure verified; not yet on race board

### Changes
- Updated VERSIONS-Sandia.md with correct race numbers per Round 6:
  - LANL Tencent: 0.01845 → 0.008735 (PhaseAtlas + neural marks)
  - LANL Alibaba: 0.00183 → 0.00301 (retracted strict-holdout)
  - LLNL Alibaba: 0.001937 HRC-MAE (lead over LANL +35%)
  - LLNL Tencent: 0.039 frozen-bundle ATB (different metric)

### Race Dashboard (Current)
| Team | Corpus | metric | value | method |
|------|--------|--------|-------|--------|
| LLNL | Alibaba | frozen ★ | 0.001937 | v195 ep110, seed=11 |
| LLNL | Tencent | frozen ★ | 0.039 | v229 ep10, seed=5 |
| LANL | Alibaba | HRC-MAE | 0.00301 | PhaseAtlas strict-holdout |
| LANL | Tencent | HRC-MAE | 0.008735 | PhaseAtlas + neural marks |
| Sandia | - | - | - | Not yet on board |

### v1 Baseline Attempt (s001_test, s002_tencent)
- Attempted training with newgan/train.py
- Initial debug run (s001_test): batch_size=4, 5 epochs, ~25 min
- Full run (s002_tencent): batch_size=64, 10 epochs (5 pretrain + 5 GAN)
- Training script identified issues per Round 4/5:
  - Parser flags not matching v1_baseline.sh
  - `files_per_epoch` ignored
  - Validation collation needs ConcatDataset
  - `newgan/run.py` Generator init issue

---

## Sandia v1.0.3 (2026-04-29) — Fix Round 4/5 Blockers

**Status:** Code fixes applied; ready for next training run

### Changes
- `newgan/train.py`:
  - Added missing `tempfile` import (line 66)
  - Changed `_load_epoch_dataset` to return `ConcatDataset` for train, tensor for val
  - Updated `load_data()` to properly use `files_per_epoch` for subsampling
  - Added missing parser flags: `--cond-drop-prob`, `--var-cond`, `--var-cond-kl-weight`, `--pcf-loss-weight`, `--pcf-n-freqs`, `--retrieval-memory`, `--retrieval-*`, `--mixed-type-recovery`, `--w-stop-threshold` (all accepted but not yet implemented for v1 compatibility)
- `newgan/run.py`:
  - Fixed `Generator()` initialization to use correct positional/keyword args
  - Updated `generate_synthetic()` to extract `num_cols` from checkpoint state dict
  - Changed default output path from `llgan/config.py` to `generated.csv`

### Round 8/9 Response
- s001_test/s002_tencent both failed to emit `.pt` files due to the bugs above
- With fixes applied, next run should complete and produce checkpoints
- Ready to launch s003_tencent with proper checkpoint emission

### Race Status
- LLNL leads Alibaba: 0.001937 vs LANL's 0.00301 (+35%)
- LANL leads Tencent: 0.008735
- Sandia: Still not on board; need to produce first checkpoint

---

## Race Notes

- LLNL leads Alibaba (0.001937 vs LANL's 0.00301)
- LANL leads Tencent on HRC-MAE (0.008735)
- Sandia's competitive advantage: pretrain-quality selection
- All ATB claims must use `--eval-real-seed 42` (Round 15 protocol)
