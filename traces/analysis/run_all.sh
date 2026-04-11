#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=${1:-/tiamat/zarathustra/analysis/out}
SAMPLE_RECORDS=${SAMPLE_RECORDS:-8192}
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
mkdir -p "$OUT_DIR"
python3 "$SCRIPT_DIR/trace_inventory.py" --out-dir "$OUT_DIR"
python3 "$SCRIPT_DIR/characterize_traces.py" --inventory "$OUT_DIR/trace_inventory.jsonl" --out "$OUT_DIR/trace_characterizations.jsonl" --sample-records "$SAMPLE_RECORDS"
python3 "$SCRIPT_DIR/trace_rollup.py" --inventory "$OUT_DIR/trace_inventory.jsonl" --characterizations "$OUT_DIR/trace_characterizations.jsonl" --out-dir "$OUT_DIR"
