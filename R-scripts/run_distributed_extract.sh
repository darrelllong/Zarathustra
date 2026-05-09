#!/usr/bin/env bash
# Split the sample_manifest.csv into halves and run extract_extended_features.R
# on vinge and baase in parallel against the same /tiamat NFS-shared storage.
# Final: merge per-trace JSON + concatenate new_features.csv.
#
# Usage: run_distributed_extract.sh <append_run_dir> [MAX_RECORDS]
set -euo pipefail
APPEND=${1:?usage: run_distributed_extract.sh APPEND_DIR [MAX_RECORDS]}
MAX=${2:-5000000}
MANIFEST="$APPEND/sample_manifest.csv"
[[ -f $MANIFEST ]] || { echo "missing $MANIFEST"; exit 2; }

VINGE_DIR="$APPEND/_vinge"
BAASE_DIR="$APPEND/_baase"
mkdir -p "$VINGE_DIR" "$BAASE_DIR"

# Split: keep header, alternate lines.
awk 'NR==1{print > "/tmp/_man.h"; next}
     {if(NR%2==0) print > "/tmp/_man.v.body"; else print > "/tmp/_man.b.body"}' "$MANIFEST"
cat /tmp/_man.h /tmp/_man.v.body > "$VINGE_DIR/sample_manifest.csv"
cat /tmp/_man.h /tmp/_man.b.body > "$BAASE_DIR/sample_manifest.csv"
rm -f /tmp/_man.h /tmp/_man.v.body /tmp/_man.b.body

V_LOG="$VINGE_DIR/extract.log"
B_LOG="$BAASE_DIR/extract.log"
SCRIPT="/tiamat/zarathustra/r-analysis-src/R-scripts/extract_extended_features.R"

ssh vinge.local "Rscript $SCRIPT $VINGE_DIR/sample_manifest.csv $VINGE_DIR $MAX" \
    > "$V_LOG" 2>&1 &
VPID=$!
ssh baase.local "Rscript $SCRIPT $BAASE_DIR/sample_manifest.csv $BAASE_DIR $MAX" \
    > "$B_LOG" 2>&1 &
BPID=$!

wait $VPID; V_RC=$?
wait $BPID; B_RC=$?
echo "vinge exit=$V_RC  baase exit=$B_RC"

# Merge outputs.
cp -r "$VINGE_DIR/per_trace/." "$APPEND/per_trace/" 2>/dev/null || true
cp -r "$BAASE_DIR/per_trace/." "$APPEND/per_trace/" 2>/dev/null || true
mkdir -p "$APPEND/per_trace"
mv -n "$VINGE_DIR/per_trace/"*.json "$APPEND/per_trace/" 2>/dev/null || true
mv -n "$BAASE_DIR/per_trace/"*.json "$APPEND/per_trace/" 2>/dev/null || true

# Concatenate new_features.csv (keep one header).
{
  if [[ -f "$VINGE_DIR/new_features.csv" ]]; then
    cat "$VINGE_DIR/new_features.csv"
  fi
  if [[ -f "$BAASE_DIR/new_features.csv" ]]; then
    tail -n +2 "$BAASE_DIR/new_features.csv"
  fi
} > "$APPEND/new_features.csv"

echo "merged feature rows:"
wc -l "$APPEND/new_features.csv"
