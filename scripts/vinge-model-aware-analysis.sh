#!/usr/bin/env bash
set -euo pipefail

VINGE="vinge.local"
LOCAL_ROOT="/Users/darrell/Zarathustra"
REMOTE_SRC="/tiamat/zarathustra/r-analysis-src"
REMOTE_OUT_BASE="/tiamat/zarathustra/r-output"
REMOTE_VENV="/tiamat/zarathustra/r-analysis-venv"
RUN_ID="${1:-model_aware_$(date +%Y%m%d_%H%M%S)}"
REMOTE_RUN_DIR="${REMOTE_OUT_BASE}/${RUN_ID}"
REMOTE_DUCKDB_THREADS_CMD='if pgrep -af train.py >/dev/null 2>&1; then echo 1; else echo 4; fi'

echo "== syncing analysis scripts to ${VINGE}:${REMOTE_SRC} =="
ssh "$VINGE" "mkdir -p '$REMOTE_SRC' '$REMOTE_RUN_DIR'"
rsync -az \
  "${LOCAL_ROOT}/parsers" \
  "${LOCAL_ROOT}/R-scripts" \
  "${LOCAL_ROOT}/traces/analysis/normalize_characterizations.py" \
  "$VINGE:$REMOTE_SRC/"

echo "== installing R dependencies =="
ssh "$VINGE" "nice -n 10 ionice -c3 bash -lc 'Rscript $REMOTE_SRC/R-scripts/install_packages.R'"

echo "== ensuring isolated Python parquet support =="
ssh "$VINGE" "nice -n 10 ionice -c3 bash -lc '
  if [ ! -x \"$REMOTE_VENV/bin/python\" ]; then
    python3 -m venv \"$REMOTE_VENV\"
  fi
  \"$REMOTE_VENV/bin/pip\" install --quiet --upgrade pip >/tmp/duckdb_install.log 2>&1
  \"$REMOTE_VENV/bin/pip\" install --quiet duckdb >>/tmp/duckdb_install.log 2>&1 || {
    tail -50 /tmp/duckdb_install.log
    exit 1
  }
'"

echo "== normalizing characterization inputs =="
ssh "$VINGE" "nice -n 10 ionice -c3 bash -lc '
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  export TRACE_PARSER_DUCKDB_THREADS=\$($REMOTE_DUCKDB_THREADS_CMD)
  PYTHONPATH=$REMOTE_SRC $REMOTE_VENV/bin/python $REMOTE_SRC/normalize_characterizations.py --out-dir $REMOTE_RUN_DIR/normalized
'"

echo "== running corpus analysis =="
ssh "$VINGE" "nice -n 10 ionice -c3 bash -lc 'Rscript $REMOTE_SRC/R-scripts/run_corpus_analysis.R $REMOTE_RUN_DIR/normalized/trace_characterizations.normalized.jsonl $REMOTE_RUN_DIR/results $REMOTE_RUN_DIR/repo_sync'"

echo "== running model-aware synthesis =="
ssh "$VINGE" "nice -n 10 ionice -c3 bash -lc 'Rscript $REMOTE_SRC/R-scripts/run_model_aware_analysis.R $REMOTE_RUN_DIR/results $REMOTE_RUN_DIR/repo_sync /home/darrell'"

echo "== syncing repo summary artifacts back =="
mkdir -p "${LOCAL_ROOT}/characterizations/families"
rsync -az "$VINGE:$REMOTE_RUN_DIR/repo_sync/" "${LOCAL_ROOT}/characterizations/"

echo "== complete =="
echo "remote run dir: ${REMOTE_RUN_DIR}"
