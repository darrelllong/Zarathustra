#!/bin/bash
# Run desnoyers IRM and 2DIO baselines across all 9 race corpora.
# Generates single-seed (42) traces, evaluates on the official cachesim surface,
# and dumps a summary table.
set -eo pipefail
cd /home/darrell/Zarathustra
PY=~/llgan-env/bin/python

REFS=/tiamat/zarathustra/llgan-output/refs
OUT=/tiamat/zarathustra/llgan-output/desnoyers
mkdir -p $OUT

# Per-corpus: name, real CSV, surface (cache-sizes, policies)
declare -A CORPORA
CORPORA[alibaba]="$REFS/alibaba_stackatlas_1M_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"
CORPORA[tencent]="$REFS/tencent_stackatlas_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"
CORPORA[cloudphysics]="$REFS/cloudphysics_stackatlas_real.csv|32,128,512,2048,8192,32768|lru,arc,fifo,sieve,slru,car,lfu,lirs"
CORPORA[baleen24]="$REFS/baleen24_stackatlas_real.csv|32,128,512,2048,8192,32768|lru,arc,fifo,sieve,slru,car,lfu,lirs"
CORPORA[msr]="$REFS/msr_exchange_stackatlas_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"
CORPORA[metakv]="$REFS/metakv_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"
CORPORA[metacdn]="$REFS/metacdn_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"
CORPORA[wiki]="$REFS/wiki_real.csv|32,128,512,2048,8192|lru,arc,fifo,sieve,slru,car"

run_model() {
  local model=$1   # irm or irm_2dio
  local corpus=$2
  IFS='|' read -r real cache_sizes policies <<< "${CORPORA[$corpus]}"
  if [ ! -f "$real" ]; then
    echo "  $corpus: SKIP (no real CSV)" ; return
  fi
  local pkl=$OUT/${corpus}_${model}.pkl.gz
  local fake=$OUT/${corpus}_${model}_seed42.csv
  local cs=$OUT/${corpus}_${model}_seed42.json
  echo "  fitting ${corpus}/${model}..."
  $PY -m desnoyers.${model} fit --real $real --output $pkl --max-rows 1000000 2>&1 | tail -1
  echo "  generating..."
  $PY -m desnoyers.${model} generate --model $pkl --output $fake --n 1000000 --seed 42 2>&1 | tail -1
  echo "  evaluating..."
  $PY -m llgan.cachesim_eval --fake $fake --real $real \
    --cache-sizes $cache_sizes --policies $policies --out $cs 2>&1 | grep "mean HRC-MAE"
}

echo "=== Desnoyers IRM + 2DIO baselines, single seed=42 ==="
for corpus in alibaba tencent cloudphysics baleen24 msr metakv metacdn wiki; do
  echo
  echo "## $corpus"
  for m in irm irm_2dio; do
    run_model $m $corpus
  done
done

echo
echo "=== Summary ==="
$PY << 'EOF'
import json, os, glob
OUT = "/tiamat/zarathustra/llgan-output/desnoyers"
print(f"{'corpus':<14s}  {'IRM':>10s}  {'2DIO':>10s}  {'best LLNL':>12s}  {'LANL banked':>12s}")
LLNL_BANKED = {
    "alibaba": 0.009999, "tencent": "—", "cloudphysics": 0.02978,
    "baleen24": 0.018447, "msr": 0.00893, "metakv": 0.04807,
    "metacdn": 0.03081, "wiki": 0.008895
}
LANL_BANKED = {
    "alibaba": 0.01076, "tencent": 0.03010, "cloudphysics": 0.0267,
    "baleen24": 0.0276, "msr": 0.00484, "metakv": 0.0109,
    "metacdn": 0.0377, "wiki": 0.01146
}
for corpus in ("alibaba","tencent","cloudphysics","baleen24","msr","metakv","metacdn","wiki"):
    irm_f = f"{OUT}/{corpus}_irm_seed42.json"
    d2io_f = f"{OUT}/{corpus}_irm_2dio_seed42.json"
    irm_v = json.load(open(irm_f))['mean_hrc_mae'] if os.path.exists(irm_f) else None
    d2io_v = json.load(open(d2io_f))['mean_hrc_mae'] if os.path.exists(d2io_f) else None
    irm_s = f"{irm_v:.5f}" if irm_v is not None else "—"
    d2io_s = f"{d2io_v:.5f}" if d2io_v is not None else "—"
    llnl_s = f"{LLNL_BANKED[corpus]:.5f}" if isinstance(LLNL_BANKED[corpus], float) else LLNL_BANKED[corpus]
    lanl_s = f"{LANL_BANKED[corpus]:.5f}"
    print(f"{corpus:<14s}  {irm_s:>10s}  {d2io_s:>10s}  {llnl_s:>12s}  {lanl_s:>12s}")
EOF
