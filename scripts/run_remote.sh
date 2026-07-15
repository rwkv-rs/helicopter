#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.remote}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

load_env_file() {
  local path="$1"
  local raw line key value
  [[ -f "$path" ]] || return 0

  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(trim "$raw")"
    [[ -z "$line" || "$line" == \#* ]] && continue
    if [[ "$line" == export\ * ]]; then
      line="$(trim "${line#export }")"
    fi
    [[ "$line" == *=* ]] || continue

    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [[ -n "${!key+x}" ]] && continue

    if [[ ${#value} -ge 2 && "${value:0:1}" == "${value: -1}" ]] &&
       [[ "${value:0:1}" == "'" || "${value:0:1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "$key=$value"
  done <"$path"
}

if [[ -f "$ENV_FILE" ]]; then
  load_env_file "$ENV_FILE"
fi

REMOTE_SSH_HOST="${REMOTE_SSH_HOST:-rwkv-sha-pro6000x8}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/caizus/Projects/MachineLearning/helicopter}"
REMOTE_VENV="${REMOTE_VENV:-$REMOTE_ROOT/.venv}"
REMOTE_HTTP_PROXY="${REMOTE_HTTP_PROXY:-}"
REMOTE_HTTPS_PROXY="${REMOTE_HTTPS_PROXY:-$REMOTE_HTTP_PROXY}"
REMOTE_NO_PROXY="${REMOTE_NO_PROXY:-localhost,127.0.0.1,::1}"
REMOTE_RUN_LOG_DIR="${REMOTE_RUN_LOG_DIR:-logs/remote}"
REMOTE_COLLECT_PATHS="${REMOTE_COLLECT_PATHS:-logs reports/validation outputs runs wandb tensorboard}"
REMOTE_RESULT_ROOT="${REMOTE_RESULT_ROOT:-$ROOT}"
PREPARE_REMOTE="${PREPARE_REMOTE:-1}"
SYNC_REMOTE="${SYNC_REMOTE:-1}"
INSTALL_REMOTE="${INSTALL_REMOTE:-1}"
COLLECT_REMOTE_RESULTS="${COLLECT_REMOTE_RESULTS:-1}"

usage() {
  cat <<'EOF'
usage: scripts/run_remote.sh [options] [--] command...

Prepare rwkv-sha-pro6000x8 with scripts/install_remote.sh, run command in the
remote repository, then copy configured result paths back.

Options:
  --no-prepare    skip scripts/install_remote.sh
  --no-sync       prepare without rsync
  --no-install    prepare without running scripts/install_local.sh remotely
  --no-collect    skip copying result paths back
  --collect       copy result paths back after the command
  -h, --help      show this help

Examples:
  scripts/run_remote.sh -- helicopter takeoff --dataset gsm8k g1g-1.5b grpo
  INSTALL_REMOTE=0 scripts/run_remote.sh -- python -m unittest tests.test_cli
EOF
}

REMOTE_COMMAND_ARGS=()
while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --no-prepare)
      PREPARE_REMOTE=0
      ;;
    --no-sync)
      SYNC_REMOTE=0
      ;;
    --no-install)
      INSTALL_REMOTE=0
      ;;
    --no-collect)
      COLLECT_REMOTE_RESULTS=0
      ;;
    --collect)
      COLLECT_REMOTE_RESULTS=1
      ;;
    --)
      shift
      REMOTE_COMMAND_ARGS=("$@")
      break
      ;;
    *)
      REMOTE_COMMAND_ARGS=("$@")
      break
      ;;
  esac
  shift
done

print_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  print_cmd "$@"
  [[ "${DRY_RUN:-0}" == "1" ]] || "$@"
}

die() {
  echo "error: $*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require_local_tools() {
  have ssh || die "ssh is required"
  have rsync || die "rsync is required"
}

remote_command_text() {
  local text
  if ((${#REMOTE_COMMAND_ARGS[@]})); then
    printf -v text '%q ' "${REMOTE_COMMAND_ARGS[@]}"
    printf '%s' "${text% }"
  elif [[ -n "${REMOTE_COMMAND:-}" ]]; then
    printf '%s' "$REMOTE_COMMAND"
  else
    return 1
  fi
}

remote_run_env_args() {
  local args=(
    "VENV=$REMOTE_VENV"
    "REMOTE_VENV=$REMOTE_VENV"
    "HELICOPTER_VENV=$REMOTE_VENV"
    "HELICOPTER_PYTHON=$REMOTE_VENV/bin/python"
    "HTTP_PROXY=$REMOTE_HTTP_PROXY"
    "HTTPS_PROXY=$REMOTE_HTTPS_PROXY"
    "http_proxy=$REMOTE_HTTP_PROXY"
    "https_proxy=$REMOTE_HTTPS_PROXY"
    "NO_PROXY=$REMOTE_NO_PROXY"
    "no_proxy=$REMOTE_NO_PROXY"
    "ALL_PROXY="
    "all_proxy="
  )
  local key
  for key in \
    PYTHON_VERSION PYTHONPATH PYPI_INDEX_URL UV_INDEX_URL HF_ENDPOINT UV_LINK_MODE \
    CARGO_REGISTRY_MIRROR CUDA_HOME CUDA_PATH INSTALL_COMPONENTS VLLM_BUILD_PROFILE HELICOPTER_VERL_PATH RWKV_LM_PATH HELICOPTER_RWKV_LM_PATH \
    HELICOPTER_VLLM_RWKV_PATH VLLM_RWKV_PATH WEIGHT_PATH DATASETS_PATH \
    HELICOPTER_NUM_NODES HELICOPTER_NUM_DEVICES HELICOPTER_TENSOR_PARALLEL_SIZE \
    HELICOPTER_INFER_WKV_MODE HELICOPTER_INFER_EMB_DEVICE \
    HELICOPTER_INFER_ALLOW_FP16_ACCUMULATION HELICOPTER_TAKEOFF_WKV_MODE \
    HELICOPTER_TAKEOFF_EMB_DEVICE HELICOPTER_TAKEOFF_ALLOW_FP16_ACCUMULATION \
    PYTORCH_CUDA_ALLOC_CONF WANDB_API_KEY WANDB_PROJECT WANDB_ENTITY; do
    [[ -n "${!key+x}" ]] || continue
    args+=("$key=${!key}")
  done

  printf ' %q' "${args[@]}"
}

remote_command_script() {
  local command_text="$1"
  cat <<EOF
set -euo pipefail
export PATH="$(printf '%q' "$REMOTE_VENV")/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
mkdir -p $(printf '%q' "$REMOTE_RUN_LOG_DIR")
run_log="$(printf '%q' "$REMOTE_RUN_LOG_DIR")/\$(date +%Y%m%d-%H%M%S).log"
printf 'Remote command: %s\n' $(printf '%q' "$command_text")
set +e
bash -lc $(printf '%q' "$command_text") 2>&1 | tee "\$run_log"
status=\${PIPESTATUS[0]}
set -e
printf '%s\n' "\$status" >"\$run_log.exit"
exit "\$status"
EOF
}

prepare_remote() {
  [[ "$PREPARE_REMOTE" == "1" ]] || return 0

  run env \
    "ENV_FILE=$ENV_FILE" \
    "SYNC_REMOTE=$SYNC_REMOTE" \
    "INSTALL_REMOTE=$INSTALL_REMOTE" \
    "DRY_RUN=${DRY_RUN:-0}" \
    "$ROOT/scripts/install_remote.sh"
}

run_remote_command() {
  local command_text quoted_root script
  command_text="$(remote_command_text)" || die "missing remote command; pass one after --"

  quoted_root="$(printf '%q' "$REMOTE_ROOT")"
  script="$(remote_command_script "$command_text")"
  print_cmd ssh "$REMOTE_SSH_HOST" \
    "cd $quoted_root && env$(remote_run_env_args) bash -lc $(printf '%q' "$script")"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0

  ssh "$REMOTE_SSH_HOST" \
    "cd $quoted_root && env$(remote_run_env_args) bash -lc $(printf '%q' "$script")"
}

collect_remote_results() {
  [[ "$COLLECT_REMOTE_RESULTS" == "1" ]] || return 0

  local rel remote_path local_parent test_command remote_source
  for rel in $REMOTE_COLLECT_PATHS; do
    [[ -n "$rel" ]] || continue
    [[ "$rel" != /* ]] || die "REMOTE_COLLECT_PATHS must be relative paths; got $rel"

    remote_path="$REMOTE_ROOT/$rel"
    local_parent="$REMOTE_RESULT_ROOT/$(dirname "$rel")"
    test_command="test -e $(printf '%q' "$remote_path")"
    print_cmd ssh "$REMOTE_SSH_HOST" "$test_command"
    if [[ "${DRY_RUN:-0}" != "1" ]] && ! ssh "$REMOTE_SSH_HOST" "$test_command"; then
      echo "Skipping missing remote result path: $REMOTE_SSH_HOST:$remote_path"
      continue
    fi

    run mkdir -p "$local_parent"
    remote_source="$REMOTE_SSH_HOST:$remote_path"
    run rsync -a "$remote_source" "$local_parent/"
  done
}

require_local_tools
prepare_remote

remote_status=0
run_remote_command || remote_status=$?
collect_remote_results

if ((remote_status != 0)); then
  echo "Remote command failed with exit code $remote_status; collected configured result paths when available" >&2
  exit "$remote_status"
fi
