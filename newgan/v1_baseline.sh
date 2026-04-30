#!/bin/bash
# Sandia v1 baseline - first Sandia-run training with cross-seed validation
# Uses newgan/train.py (Sandia's training code)
# Launch on vinge.local

set -e

# Remote paths
REMOTE_HOME="/home/darrell"
REMOTE_CODE="$REMOTE_HOME/llgan"
REMOTE_TRACES="$REMOTE_HOME/traces"
REMOTE_CHECKPOINTS="$REMOTE_HOME/checkpoints"
REMOTE_DATA="/tiamat/zarathustra"

# Experiment parameters
EXP_NAME="s001_tencent_v1"
TRACE_DIR="$REMOTE_HOME/traces/tencent_block_1M"
CHAR_FILE="$REMOTE_DATA/analysis/out/trace_characterizations.jsonl"
CHECKPOINT_DIR="$REMOTE_CHECKPOINTS/$EXP_NAME"

# Recipe parameters - Sandia's approach
# - Full training pipeline: AE pretrain + supervisor + G warmup + GAN
# - Cross-seed validation: run with seeds {42, 11, 7}
# - WGAN-SN loss for stability on GB10
# - PCF loss for path-space adversarial training
# - Retrieval memory for long-horizon reuse prediction

python3 -u $REMOTE_CODE/newgan/train.py \
    --trace-dir "$TRACE_DIR" \
    --fmt oracle_general \
    --char-file "$CHAR_FILE" \
    --loss wgan-sn \
    --epochs 200 \
    --batch-size 64 \
    --timestep 12 \
    --noise-dim 10 \
    --hidden-size 256 \
    --latent-dim 24 \
    --pretrain-ae-epochs 50 \
    --pretrain-sup-epochs 50 \
    --pretrain-g-epochs 100 \
    --files-per-epoch 12 \
    --records-per-file 20000 \
    --lr-g 0.0001 \
    --lr-d 0.0001 \
    --n-critic 5 \
    --checkpoint-every 5 \
    --early-stop-patience 30 \
    --cond-drop-prob 0.25 \
    --var-cond \
    --var-cond-kl-weight 0.001 \
    --pcf-loss-weight 2.0 \
    --pcf-n-freqs 32 \
    --retrieval-memory \
    --retrieval-mem-size 32 \
    --retrieval-key-dim 32 \
    --retrieval-val-dim 32 \
    --retrieval-decay 0.85 \
    --retrieval-tau-write 0.5 \
    --retrieval-n-warmup 4 \
    --mixed-type-recovery \
    --no-compile \
    --seed 42 \
    --exp-name "$EXP_NAME" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "Training complete. Checkpoint saved to: $CHECKPOINT_DIR"
