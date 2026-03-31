#!/usr/bin/env bash
# vinge-pull.sh — pull the latest code from GitHub onto vinge and sync to ~/llgan
#
# Why: vinge has two separate locations for the code — the git repo
# (~/Zarathustra) and the working copy used for training (~/llgan).
# Pulling on vinge requires SSH agent forwarding (-A) for GitHub access;
# without it you get "Permission denied (publickey)". After pulling, the
# ~/llgan working copy needs to be updated from the repo. This script
# does all three steps correctly in sequence.
#
# Usage:
#   ./scripts/vinge-pull.sh        # pull main branch
#   ./scripts/vinge-pull.sh --help
#
# After this runs, the training code in ~/llgan on vinge matches the local
# git HEAD. Restart any running training job manually if needed.

set -euo pipefail

VINGE="vinge.local"

[[ "${1:-}" == "--help" ]] && { sed -n '2,/^$/p' "$0" | sed 's/^# \?//'; exit 0; }

echo "=== Pulling latest code onto vinge ==="

# -A forwards your local SSH agent so vinge can authenticate to GitHub.
# Without agent forwarding, git pull fails with "Permission denied (publickey)"
# because vinge doesn't have a private key that GitHub trusts.
ssh -A "$VINGE" "
    set -e
    cd ~/Zarathustra
    git pull
    echo
    echo '=== Syncing ~/Zarathustra/llgan/ → ~/llgan/ ==='
    # ~/llgan is the directory where training is actually run from.
    # It was created before the git repo existed on vinge and is kept
    # separate so training runs don't depend on the git working tree.
    cp ~/Zarathustra/llgan/*.py ~/llgan/
    echo 'Done. Files synced:'
    ls -la ~/llgan/*.py | awk '{print \$9, \$5}'
    echo
    echo 'Git log (last 3):'
    git -C ~/Zarathustra log --oneline -3
"
