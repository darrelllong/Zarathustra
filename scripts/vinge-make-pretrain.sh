#!/usr/bin/env bash
# vinge-make-pretrain.sh — copy a pretrain checkpoint from one version to another
#
# Why: the 200-epoch pretrain (AE 50ep + supervisor 50ep + G warmup 100ep)
# takes ~35 minutes on the GB10. When only changing GAN-phase hyperparameters
# (n_critic, lr_d, loss weights), the pretrained E/R/S/G weights are reusable.
# Saving them as pretrain_complete.pt lets train.py skip phases 1–2.5 entirely.
#
# IMPORTANT — compatibility constraint:
#   The source and destination must share the same:
#     - Number of input features (currently 6: ts, obj_size, opcode, tenant, obj_id_reuse, obj_id_stride)
#     - hidden_size and latent_dim
#     - --no-compile flag (state dict keys have no _orig_mod. prefix)
#   If any of these change, start fresh — do not reuse the pretrain checkpoint.
#
# Usage:
#   ./scripts/vinge-make-pretrain.sh --from v15 --to v17
#   ./scripts/vinge-make-pretrain.sh --from v15 --to v17 --epoch 0010
#   ./scripts/vinge-make-pretrain.sh --help
#
# --from: source version (pretrain_complete.pt from that version's checkpoint dir,
#         or epoch_NNNN.pt if pretrain_complete.pt doesn't exist)
# --to:   destination version (creates that version's checkpoint dir)
# --epoch: which epoch checkpoint to use as source (default: pretrain_complete.pt,
#          fall back to epoch_0010.pt)

set -euo pipefail

VINGE="vinge.local"

usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

[[ "${1:-}" == "--help" ]] && usage

FROM_VERSION=""
TO_VERSION=""
EPOCH="0010"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)  FROM_VERSION="$2"; shift 2 ;;
        --to)    TO_VERSION="$2";   shift 2 ;;
        --epoch) EPOCH="$2";        shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$FROM_VERSION" || -z "$TO_VERSION" ]]; then
    echo "Error: --from and --to are required." >&2
    echo "Run with --help for usage." >&2
    exit 1
fi

echo "=== Creating pretrain checkpoint: $FROM_VERSION → $TO_VERSION ==="

ssh "$VINGE" "
    set -e

    FROM_DIR=\$HOME/checkpoints/tencent_${FROM_VERSION}
    TO_DIR=\$HOME/checkpoints/tencent_${TO_VERSION}
    mkdir -p \$TO_DIR

    # Prefer pretrain_complete.pt if it exists (it has exactly the right keys).
    # Fall back to epoch_NNNN.pt which has the same weights plus C and optimizer
    # for the critic (which we don't need for the destination).
    if [[ -f \$FROM_DIR/pretrain_complete.pt ]]; then
        SRC=\$FROM_DIR/pretrain_complete.pt
        echo 'Source: pretrain_complete.pt (ideal)'
    elif [[ -f \$FROM_DIR/epoch_${EPOCH}.pt ]]; then
        SRC=\$FROM_DIR/epoch_${EPOCH}.pt
        echo \"Source: epoch_${EPOCH}.pt (extracting E/R/S/G/G_ema/opt_G/prep)\"
    else
        echo \"Error: neither pretrain_complete.pt nor epoch_${EPOCH}.pt found in \$FROM_DIR\" >&2
        exit 1
    fi

    cd \$HOME/llgan
    \$HOME/llgan-env/bin/python3 -c \"
import torch, sys, os
sys.path.insert(0, os.path.expanduser('~') + '/llgan')

src = torch.load('\$SRC', map_location='cpu', weights_only=False)
print('Source keys:', list(src.keys()))

# Extract only what train.py expects in pretrain_complete.pt.
# The format is documented in train.py around the 'pretrain_complete.pt' save block.
# Keys must NOT have the _orig_mod. prefix — that prefix appears only when
# torch.compile wraps the module. We always use --no-compile on this system.
dst = {
    'E':      src['E'],
    'R':      src['R'],
    'S':      src['S'],
    'G':      src['G'],
    'G_ema':  src['G_ema'],
    'opt_G':  src['opt_G'],
    'prep':   src['prep'],
    'val_tensor': None,  # will be rebuilt from trace files on next run
}

# Sanity check: make sure keys look right (no _orig_mod. prefix)
first_E_key = next(iter(dst['E'].keys()))
if first_E_key.startswith('_orig_mod.'):
    print('ERROR: keys have _orig_mod. prefix — checkpoint was saved with torch.compile.')
    print('Cannot use with --no-compile. Start pretrain fresh instead.')
    sys.exit(1)

out = os.path.expanduser('~') + '/checkpoints/tencent_${TO_VERSION}/pretrain_complete.pt'
torch.save(dst, out)
print(f'Saved: {out}')
print(f'E sample key: {first_E_key}')
\"
    echo 'Done. Ready to use with: ./scripts/vinge-launch.sh --version ${TO_VERSION}'
"
