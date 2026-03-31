#!/usr/bin/env bash
# vinge-check.sh — show the current training status on vinge
#
# Why: checking training manually requires sshing to vinge, knowing log paths,
# and grepping for the right info each time. Mistakes happen (wrong log file,
# missing PID, forgetting to check for crashes). This script does it correctly
# every time with a single command.
#
# Usage:
#   ./scripts/vinge-check.sh [version]      # default: shows latest log
#   ./scripts/vinge-check.sh v16            # check a specific version
#   ./scripts/vinge-check.sh --help
#
# What it shows:
#   - Whether the training process is alive
#   - Last 40 lines of the training log (MMD evals, epoch progress)
#   - Best checkpoint so far (if any)

set -euo pipefail

VINGE="vinge.local"
REMOTE_HOME="\$HOME"          # expands on vinge, not locally
LOG_DIR="${REMOTE_HOME}"
CKPT_DIR="${REMOTE_HOME}/checkpoints"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

[[ "${1:-}" == "--help" ]] && usage

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
    # Find the most recently modified training log
    LOG_FILE=$(ssh "$VINGE" "ls -t \$HOME/train_v*.log 2>/dev/null | head -1")
    if [[ -z "$LOG_FILE" ]]; then
        echo "No training logs found on vinge." >&2
        exit 1
    fi
    # Extract version from filename (e.g., /home/darrell/train_v16.log → v16)
    VERSION=$(basename "$LOG_FILE" .log | sed 's/train_//')
else
    LOG_FILE="${LOG_DIR}/train_${VERSION}.log"
fi

echo "=== vinge training status: $VERSION ==="
echo

# Check if the training process is alive.
# pgrep searches by pattern in the full command line. The checkpoint dir name
# uniquely identifies which version is running, even if multiple pythons exist.
PID=$(ssh "$VINGE" "pgrep -f 'tencent_${VERSION}' 2>/dev/null | head -1 || true")
if [[ -n "$PID" ]]; then
    echo "Process: RUNNING (PID $PID)"
else
    echo "Process: NOT RUNNING"
fi
echo

# Show the tail of the log. 40 lines gives enough context to see the last
# few epochs, any recent MMD eval, and any crash traceback.
echo "=== Last 40 lines of $LOG_FILE ==="
ssh "$VINGE" "tail -40 $LOG_FILE 2>/dev/null || echo '(log not found)'"
echo

# Show best checkpoint info. The best.pt always has the epoch and metrics
# saved in config; use Python to extract rather than guessing from log.
BEST_PT="${CKPT_DIR}/tencent_${VERSION}/best.pt"
echo "=== Best checkpoint: $BEST_PT ==="
ssh "$VINGE" "
cd \$HOME/llgan 2>/dev/null || cd \$HOME/Zarathustra/llgan
\$HOME/llgan-env/bin/python3 -c \"
import torch, sys
try:
    ck = torch.load('${CKPT_DIR}/tencent_${VERSION}/best.pt', map_location='cpu', weights_only=False)
    print(f'  epoch:    {ck.get(\\\"epoch\\\", \\\"?\\\") + 1}')
    print(f'  MMD²:     {ck.get(\\\"mmd\\\", \\\"?\\\"):.5f}')
    print(f'  recall:   {ck.get(\\\"recall\\\", \\\"?\\\"):.3f}')
    print(f'  combined: {ck.get(\\\"combined\\\", \\\"?\\\"):.5f}')
except FileNotFoundError:
    print('  (not saved yet)')
except Exception as e:
    print(f'  error: {e}')
\" 2>/dev/null
" || echo "  (could not read checkpoint)"
