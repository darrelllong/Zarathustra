#!/usr/bin/env bash
# vinge-launch.sh — launch a training run on vinge with standard defaults
#
# Why: the training command has ~20 flags, many of which are version-specific.
# Typing it by hand every time leads to mistakes (wrong flag, wrong version,
# wrong checkpoint dir, forgetting --no-compile). This script sets safe defaults
# and makes version-specific overrides explicit.
#
# Usage:
#   ./scripts/vinge-launch.sh --version v17 [extra train.py flags...]
#   ./scripts/vinge-launch.sh --version v16 --n-critic 1 --epochs 100
#   ./scripts/vinge-launch.sh --help
#
# The script:
#   1. Kills any existing training for this version (safe — doesn't kill others)
#   2. Creates the checkpoint directory if needed
#   3. Launches training in the background with nohup
#   4. Waits 10 seconds and shows the first lines of the log
#
# Defaults (as of v16; override with extra flags):
#   --epochs 150 --files-per-epoch 12 --records-per-file 15000
#   --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40
#   --n-critic 2 --lr-g 1e-4 --lr-d 2.5e-5
#   --hidden-size 256 --latent-dim 24
#   --supervisor-loss-weight 5.0 --ema-decay 0.999 --lr-cosine-decay 0.05
#   --grad-clip 1.0 --no-compile (torch.compile broken on GB10)
#   All auxiliary losses at their v16 values

set -euo pipefail

VINGE="vinge.local"
PYTHON="~/llgan-env/bin/python"
TRAIN="~/llgan/train.py"
TRACE_DIR="~/traces/tencent_block_1M"
LOG_DIR="~"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

[[ "${1:-}" == "--help" ]] && usage

VERSION=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$VERSION" ]]; then
    echo "Error: --version is required (e.g., --version v17)" >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

CKPT_DIR="~/checkpoints/tencent_${VERSION}"
LOG_FILE="${LOG_DIR}/train_${VERSION}.log"

echo "=== Launching training $VERSION on vinge ==="
echo "  Checkpoint dir: $CKPT_DIR"
echo "  Log: $LOG_FILE"
echo "  Extra args: ${EXTRA_ARGS[*]:-none}"
echo

# Build the command. All defaults are here; pass extra flags after to override.
# Note: --no-compile is always set. torch.compile requires Triton to build CUDA
# kernels via gcc, and the libcuda.so.1 symlink is broken on the GB10 system.
# AMP fp16 still works (it uses cuBLAS directly, not Triton).
CMD="$PYTHON -u $TRAIN \
  --trace-dir $TRACE_DIR --fmt oracle_general \
  --epochs 150 --files-per-epoch 12 --records-per-file 15000 \
  --checkpoint-dir $CKPT_DIR --checkpoint-every 5 \
  --mmd-every 5 --mmd-samples 2000 --early-stop-patience 40 \
  --locality-loss-weight 1.0 --acf-loss-weight 0.2 \
  --moment-loss-weight 0.1 --fft-loss-weight 0.05 \
  --quantile-loss-weight 0.2 --feature-matching-weight 1.0 \
  --cross-cov-loss-weight 2.0 --diversity-loss-weight 0.5 \
  --ema-decay 0.999 --lr-cosine-decay 0.05 \
  --grad-clip 1.0 --n-critic 2 \
  --hidden-size 256 --latent-dim 24 \
  --pretrain-ae-epochs 50 --pretrain-sup-epochs 50 --pretrain-g-epochs 100 \
  --supervisor-loss-weight 5.0 --lr-g 1e-4 --lr-d 2.5e-5 \
  --no-compile \
  ${EXTRA_ARGS[*]:-}"

ssh "$VINGE" "
    set -e

    # Kill any existing training for this version only.
    # Using the checkpoint dir name as the match pattern is safe: it uniquely
    # identifies this version's process without risk of killing unrelated python jobs.
    EXISTING=\$(pgrep -f 'tencent_${VERSION}' 2>/dev/null || true)
    if [[ -n \"\$EXISTING\" ]]; then
        echo 'Killing existing $VERSION training (PIDs: '\$EXISTING')'
        kill \$EXISTING || true
        sleep 2
    fi

    mkdir -p $CKPT_DIR

    nohup $CMD > $LOG_FILE 2>&1 &
    PID=\$!
    echo \"Launched PID \$PID\"
    sleep 10
    echo
    echo '=== First lines of $LOG_FILE ==='
    tail -20 $LOG_FILE
"
