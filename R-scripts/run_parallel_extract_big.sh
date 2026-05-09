#!/usr/bin/env bash
# Parallel runner using extract_extended_features_big.R (1800s/series budget,
# 1M record cap, larger subsamples). Same chunking as run_parallel_extract.sh.
set -euo pipefail
APPEND=${1:?usage: APPEND_DIR [N_WORKERS] [MAX_RECORDS]}
WORKERS=${2:-8}
MAX=${3:-1000000}
MANIFEST="$APPEND/sample_manifest.csv"
[[ -f $MANIFEST ]] || { echo "missing $MANIFEST"; exit 2; }

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
SCRIPT="$SCRIPT_DIR/extract_extended_features_big.R"
[[ -f $SCRIPT ]] || { echo "missing $SCRIPT"; exit 2; }

WORK="$APPEND/_workers"; mkdir -p "$WORK"
rm -f "$WORK"/*.csv "$WORK"/*.log 2>/dev/null

awk -v W="$WORKERS" 'NR==1{hdr=$0; for(i=0;i<W;i++) print hdr > ("'"$WORK"'/chunk."i".csv"); next}
                     {f=(NR-2)%W; print >> ("'"$WORK"'/chunk."f".csv")}' "$MANIFEST"

PIDS=()
for i in $(seq 0 $((WORKERS-1))); do
  CHUNK="$WORK/chunk.$i.csv"; WD="$WORK/out.$i"; mkdir -p "$WD/per_trace"
  Rscript "$SCRIPT" "$CHUNK" "$WD" "$MAX" > "$WORK/log.$i" 2>&1 &
  PIDS+=($!)
done
echo "spawned ${#PIDS[@]} big workers"

FAIL=0
for p in "${PIDS[@]}"; do if ! wait "$p"; then FAIL=$((FAIL+1)); fi; done
echo "big workers done; failures=$FAIL"

mkdir -p "$APPEND/per_trace"
for i in $(seq 0 $((WORKERS-1))); do
  cp -n "$WORK/out.$i/per_trace/"*.json "$APPEND/per_trace/" 2>/dev/null || true
done
{
  hdr_done=0
  for i in $(seq 0 $((WORKERS-1))); do
    F="$WORK/out.$i/new_features.csv"
    [[ -f $F ]] || continue
    if [[ $hdr_done -eq 0 ]]; then cat "$F"; hdr_done=1
    else tail -n +2 "$F"; fi
  done
} > "$APPEND/new_features.csv"
echo "merged feature rows:"; wc -l "$APPEND/new_features.csv"
