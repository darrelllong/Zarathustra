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
  altgan/lanl_remote_job.sh launch-chunksurf <log_tag> <module> [--tmux <session>] -- <args...>
  altgan/lanl_remote_job.sh launch-mdlstm-tencent <tag> <model_file> <fit|nofit> <birth|nobirth> <seed> [epochs] [footprint|ws|learned-ws|learned-ws-masked] [short_reuse_pressure] [short_reuse_loss_weight] [recycle_rank_cap] [uniform|empirical] [exact_rank_cutoff]

Remote LANL runner. Keep local SSH invocations simple so the local sandbox sees
only `ssh -i ... host /path/to/altgan/lanl_remote_job.sh ...`; all chaining,
redirection, nohup, and backgrounding happens on the remote host.
EOF
}

pull_repo() {
  cd "$ROOT"
  git pull --ff-only
}

launch_chunksurf() {
  local log_tag="${1:?log_tag required}"
  local module="${2:?module required}"
  shift 2

  local tmux_session=""
  if [[ "${1:-}" == "--tmux" ]]; then
    tmux_session="${2:-}"
    shift 2
  fi
  if [[ "${1:-}" == "--" ]]; then
    shift
  fi

  pull_repo
  mkdir -p "$OUT_ROOT/logs"

  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local log_path="$OUT_ROOT/logs/${log_tag}_${ts}.log"
  local -a cmd=("$VENV_PY" -u -m "$module" "$@")

  cd "$ROOT"
  if [[ -n "$tmux_session" ]]; then
    local cmd_str=""
    printf -v cmd_str '%q ' "${cmd[@]}"
    tmux new-session -d -s "$tmux_session" bash -lc "cd \"$ROOT\" && $cmd_str > \"$log_path\" 2>&1"
    echo "TMUX:$tmux_session"
    echo "LOG:$log_path"
    return 0
  fi

  nohup "${cmd[@]}" > "$log_path" 2>&1 < /dev/null &
  local pid=$!
  echo "PID:$pid"
  echo "LOG:$log_path"
}

status_job() {
  local pattern="${1:?pattern required}"
  local log_path="${2:-}"
  pgrep -af "$pattern" | awk '$0 !~ /lanl_remote_job\.sh status/ {print}' || true
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    tail -160 "$log_path"
  fi
}

kill_job() {
  local pattern="${1:?pattern required}"
  local -a pids=()
  while read -r pid _rest; do
    [[ -z "$pid" ]] && continue
    pids+=("$pid")
  done < <(pgrep -af "$pattern" | awk '$0 !~ /lanl_remote_job\.sh kill/ {print}')
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" || true
  fi
  sleep 1
  pgrep -af "$pattern" | awk '$0 !~ /lanl_remote_job\.sh kill/ {print}' || true
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
  local short_reuse_loss_weight="${9:-0}"
  local recycle_rank_cap="${10:-0}"
  local rank_sampler="${11:-uniform}"
  local exact_rank_cutoff="${12:-0}"

  pull_repo
  mkdir -p "$OUT_ROOT/logs" "$CKPT_ROOT"

  local model_path="$CKPT_ROOT/$model_file"
  local ts
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  local host_tag
  host_tag="$(hostname -s 2>/dev/null || hostname || echo host)"
  host_tag="${host_tag//[^A-Za-z0-9._-]/_}"
  local log_path="$OUT_ROOT/logs/${tag}_${host_tag}_${ts}.log"
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
    --short-reuse-loss-weight "$short_reuse_loss_weight"
    --recycle-rank-cap "$recycle_rank_cap"
    --rank-sampler "$rank_sampler"
    --exact-rank-cutoff "$exact_rank_cutoff"
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

  if [[ "$control_mode" == "footprint" || "$control_mode" == "ws" || "$control_mode" == "learned-ws" || "$control_mode" == "learned-ws-masked" ]]; then
    cmd+=(--birth-control-mode "$control_mode")
  else
    echo "control mode must be footprint, ws, learned-ws, or learned-ws-masked" >&2
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
    launch-chunksurf) launch_chunksurf "$@" ;;
    launch-mdlstm-tencent) launch_mdlstm_tencent "$@" ;;
    -h|--help|help|"") usage ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
