#!/bin/bash
# Sandia baseline for Baleen24
set -e

# Remote paths
REMOTE_HOME="/home/darrell"
REMOTE_CODE="$REMOTE_HOME/llgan"
REMOTE_TRACES="$REMOTE_HOME/traces"
REMOTE_CHECKPOINTS="$REMOTE_HOME/checkpoints"
REMOTE_DATA="/tiamat/zarathustra"

# Experiment parameters
EXP_NAME="s001_baleen24_v1"
TRACE_DIR="$REMOTE_HOME/traces/baleen24"
CHAR_FILE="$REMOTE_DATA/analysis/out/trace_characterizations.jsonl"
CHECKPOINT_DIR="$REMOTE_CHECKPOINTS/$EXP_NAME"

# Training command
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

EOF