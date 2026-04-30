#!/bin/bash
# Sandia v1 baseline - reproduction of v165 tencent recipe with deterministic controls
# Launch on vinge.local

set -e

# Remote paths
REMOTE_HOME="/home/darrell"
REMOTE_CODE="$REMOTE_HOME/llgan"
REMOTE_TRACES="$REMOTE_HOME/traces"
REMOTE_CHECKPOINTS="$REMOTE_HOME/checkpoints"
REMOTE_DATA="/tiamat/zarathustra"

# Experiment parameters
EXP_NAME="tencent_v1"
TRACE_DIR="$REMOTE_HOME/traces/tencent_block_1M"
CHAR_FILE="$REMOTE_DATA/analysis/out/trace_characterizations.jsonl"
CHECKPOINT_DIR="$REMOTE_CHECKPOINTS/$EXP_NAME"

# Recipe parameters (v165 = v158 + retrieval memory)
# - Multi-scale critic (disabled per feedback - not universal)
# - PCF 2.0 loss weight
# - Retrieval memory enabled
# - n_regimes=8 (from v158)
# - var_cond (enabled)
# - batch_size=64
# - files_per_epoch=12
# - records_per_file=20000

python3 -u $REMOTE_CODE/train.py \
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
    --pretrain-ae-epochs 0 \
    --pretrain-sup-epochs 0 \
    --pretrain-g-epochs 0 \
    --files-per-epoch 12 \
    --records-per-file 20000 \
    --lr-g 0.0001 \
    --lr-d 0.0001 \
    --n-critic 5 \
    --checkpoint-every 5 \
    --mmd-every 5 \
    --mmd-samples 1000 \
    --early-stop-patience 30 \
    --cond-drop-prob 0.25 \
    --var-cond \
    --var-cond-kl-weight 0.001 \
    --gumbel-reuse \
    --gumbel-reuse-weight 1.0 \
    --gumbel-tau-start 1.0 \
    --gumbel-tau-end 0.5 \
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
    --boundary-smoothness-weight 0.5 \
    --overlap-consistency-weight 0.5 \
    --overlap-consistency-mode overlap \
    --overlap-consistency-k 2 \
    --no-compile \
    --seed 42 \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    2>&1 | tee "$CHECKPOINT_DIR/train.log"

echo "Training complete. Running frozen_sweep..."
python3 -u $REMOTE_CODE/frozen_sweep.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --trace-dir "$TRACE_DIR" \
    --fmt oracle_general \
    --char-file "$CHAR_FILE" \
    --eval-real-seed 42 \
    --eval-fake-seed 42
