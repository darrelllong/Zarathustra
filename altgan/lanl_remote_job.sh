#!/usr/bin/env bash
set -euo pipefail

ROOT="${LANL_ZARATHUSTRA_ROOT:-$HOME/LANL/Zarathustra}"
OUT_ROOT="${ALTGAN_OUT_ROOT:-/tiamat/zarathustra/altgan-output}"
CKPT_ROOT="${ALTGAN_CKPT_ROOT:-/tiamat/zarathustra/checkpoints/altgan}"
VENV_PY="${ALTGAN_PY:-/tiamat/zarathustra/altgan-venv/bin/python}"
REAL_TENCENT="${TENCENT_REAL:-/tiamat/zarathustra/llgan-output/refs/tencent_stackatlas_real.csv}"

usage() {
  cat <<'EOF'
usage:
  altgan/lanl_remote_job.sh pull
  altgan/lanl_remote_job.sh status <pattern> [log_path]
  altgan/lanl_remote_job.sh kill <pattern>
  altgan/lanl_remote_job.sh launch-mdlstm-tencent <tag> <model_file> <fit|nofit> <birth|nobirth> <seed> [epochs] [footprint|ws] [short_reuse_pressure]

Remote LANL runner. Keep local SSH invocations simple so the local sandbox sees
only `ssh -i ... host /path/to/altgan/lanl_remote_job.sh ...`; all chaining,
redirection, nohup, and backgrounding happens on the remote host.
EOF
}

pull_repo() {
  cd "$ROOT"
  git pull --ff-only
}

status_job() {
  local pattern="${1:?pattern required}"
  local log_path="${2:-}"
  pgrep -af "$pattern" || true
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    tail -160 "$log_path"
  fi
}

kill_job() {
  local pattern="${1:?pattern required}"
  pkill -f "$pattern" || true
  sleep 1
  pgrep -af "$pattern" || true
}

launch_mdlstm_tencent() {
  local tag="${1:?tag required}"
  local model_file="${2:?model_file required}"
  local fit_mode="${3:?fit|nofit required}"
  local birth_mode="${4:?birth|nobirth required}"
  local seed="${5:?seed required}"
  local epochs="${6:-20}"
  local control_mode="${7:-footprint}"
  local short_reuse_pressure="${8:-0}"

  pull_repo
  mkdir -p "$OUT_ROOT/logs" "$CKPT_ROOT"

  local model_path="$CKPT_ROOT/$model_file"
  local log_path="$OUT_ROOT/logs/${tag}_vinge_20260509.log"
  local -a cmd=(
    "$VENV_PY" -u -m altgan.mattson_denning_lstm multiseed
    --real "$REAL_TENCENT"
    --model "$model_path"
    --tag "$tag"
    --output-root "$OUT_ROOT"
    --max-rows 100000
    --n-records 100000
    --rank-bins 64
    --ws-bins 32
    --ws-windows 32,128,512,2048,8192
    --hidden 128
    --token-embed 64
    --ws-embed 16
    --seq-len 256
    --batch 256
    --epochs "$epochs"
    --lr 0.001
    --seed "$seed"
    --seeds 42,80,81,82
    --temperature 1.0
    --short-reuse-pressure "$short_reuse_pressure"
    --cache-sizes 32,128,512,2048,8192
    --policies lru,arc,fifo,sieve,slru,car
  )

  if [[ "$fit_mode" == "fit" ]]; then
    cmd+=(--fit)
  elif [[ "$fit_mode" != "nofit" ]]; then
    echo "fit mode must be fit or nofit" >&2
    exit 2
  fi

  if [[ "$birth_mode" == "nobirth" ]]; then
    cmd+=(--no-birth-control)
  elif [[ "$birth_mode" != "birth" ]]; then
    echo "birth mode must be birth or nobirth" >&2
    exit 2
  fi

  if [[ "$control_mode" == "footprint" || "$control_mode" == "ws" ]]; then
    cmd+=(--birth-control-mode "$control_mode")
  else
    echo "control mode must be footprint or ws" >&2
    exit 2
  fi

  cd "$ROOT"
  nohup "${cmd[@]}" > "$log_path" 2>&1 < /dev/null &
  local pid=$!
  echo "PID:$pid"
  echo "LOG:$log_path"
  echo "MODEL:$model_path"
}

main() {
  local action="${1:-}"
  shift || true
  case "$action" in
    pull) pull_repo ;;
    status) status_job "$@" ;;
    kill) kill_job "$@" ;;
    launch-mdlstm-tencent) launch_mdlstm_tencent "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
