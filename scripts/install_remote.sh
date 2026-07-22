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
REMOTE_CUDA_HOME="${REMOTE_CUDA_HOME:-/usr/local/cuda}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_COMPONENTS="${INSTALL_COMPONENTS:-rwkv-lm,dev}"
UPDATE_UV="${UPDATE_UV:-0}"
UV_UPGRADE="${UV_UPGRADE:-0}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
VLLM_BUILD_PROFILE="${VLLM_BUILD_PROFILE:-rwkv}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-0.11.2.dev278+gdbc3d9991}"
VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
VLLM_REBUILD="${VLLM_REBUILD:-auto}"
VERL_REINSTALL="${VERL_REINSTALL:-auto}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
BUILD_TMPDIR="${BUILD_TMPDIR:-$REMOTE_ROOT/.tmp}"
REMOTE_HTTP_PROXY="${REMOTE_HTTP_PROXY:-}"
REMOTE_HTTPS_PROXY="${REMOTE_HTTPS_PROXY:-$REMOTE_HTTP_PROXY}"
REMOTE_NO_PROXY="${REMOTE_NO_PROXY:-localhost,127.0.0.1,::1}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
UV_INDEX_URL="${UV_INDEX_URL:-$PYPI_INDEX_URL}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
UV_LINK_MODE="${UV_LINK_MODE:-copy}"
CARGO_REGISTRY_MIRROR="${CARGO_REGISTRY_MIRROR:-sparse+https://rsproxy.cn/index/}"
SYNC_REMOTE="${SYNC_REMOTE:-1}"
INSTALL_REMOTE="${INSTALL_REMOTE:-1}"
REMOTE_REQUIRED_DIRS="${REMOTE_REQUIRED_DIRS:-/home/caizus/Projects /home/caizus/Weights /home/caizus/Datasets}"

die() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/install_remote.sh [options]

Prepare the remote SSH host environment. For running remote commands and
copying results back, use scripts/run_remote.sh.

Options:
  --no-sync       skip rsync to the remote repository
  --no-install    skip scripts/install_local.sh on the remote host
  -h, --help      show this help

Examples:
  scripts/install_remote.sh
  SYNC_REMOTE=0 scripts/install_remote.sh
EOF
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --no-sync)
      SYNC_REMOTE=0
      ;;
    --no-install)
      INSTALL_REMOTE=0
      ;;
    *)
      die "unknown install_remote.sh option: $1; use scripts/run_remote.sh to run remote commands"
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

have() {
  command -v "$1" >/dev/null 2>&1
}

component_enabled() {
  local requested="$1"
  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    [[ "$component" == "$requested" ]] && return 0
  done
  return 1
}

validate_install_config() {
  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  ((${#components[@]} > 0)) || die "INSTALL_COMPONENTS must select at least one dependency group"
  for component in "${components[@]}"; do
    case "$component" in
      dev | vllm-rwkv | verl-rwkv | rwkv-lm | verl-liger) ;;
      full) die "INSTALL_COMPONENTS=full is disabled; select explicit dependency groups" ;;
      *) die "unknown INSTALL_COMPONENTS entry '$component'; use a comma-separated subset of dev,vllm-rwkv,verl-rwkv,rwkv-lm,verl-liger" ;;
    esac
  done

  case "${INSTALL_PROFILE:-}" in
    "" | rwkv) ;;
    full) die "INSTALL_PROFILE=full is disabled; use INSTALL_COMPONENTS" ;;
    *) die "INSTALL_PROFILE=${INSTALL_PROFILE} is disabled; use INSTALL_COMPONENTS" ;;
  esac
  case "${HELICOPTER_VLLM_BUILD_PROFILE:-}" in
    "") ;;
    full) die "HELICOPTER_VLLM_BUILD_PROFILE=full is disabled; use VLLM_BUILD_PROFILE=rwkv" ;;
    *) die "HELICOPTER_VLLM_BUILD_PROFILE is unsupported; use VLLM_BUILD_PROFILE=rwkv" ;;
  esac
  [[ "$VLLM_BUILD_PROFILE" == "rwkv" ]] ||
    die "VLLM_BUILD_PROFILE=$VLLM_BUILD_PROFILE is disabled; only rwkv is supported"
}

require_local_tools() {
  have ssh || die "ssh is required"
  have rsync || die "rsync is required"
}

verify_remote_tools() {
  local script required_dir
  script='set -euo pipefail
command -v git
command -v uv || command -v curl
command -v python3'

  if [[ "$INSTALL_REMOTE" == "1" ]] &&
     { component_enabled vllm-rwkv || component_enabled rwkv-lm; }; then
    script="$script
command -v cc
command -v c++"
  fi
  if [[ "$INSTALL_REMOTE" == "1" ]] && component_enabled vllm-rwkv; then
    script="$script
nvidia-smi -L | wc -l
test -x $(printf '%q' "$REMOTE_CUDA_HOME/bin/nvcc")"
  fi

  for required_dir in $REMOTE_REQUIRED_DIRS; do
    script="$script
test -d $(printf '%q' "$required_dir")"
  done

  run ssh "$REMOTE_SSH_HOST" "bash -lc $(printf '%q' "$script")"
}

sync_remote_repo() {
  [[ "$SYNC_REMOTE" == "1" ]] || return 0

  run ssh "$REMOTE_SSH_HOST" "mkdir -p $(printf '%q' "$REMOTE_ROOT")"
  run rsync -a --delete \
    --exclude '.git/' \
    --exclude '.git' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.env.local' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.cache/' \
    --exclude '.tmp/' \
    --exclude '.deps/' \
    --exclude '*.so' \
    --exclude '/src/infer/vllm-rwkv/vllm/_build_profile.json' \
    --exclude 'build/' \
    --exclude 'dist/' \
    --exclude '*.egg-info/' \
    --exclude 'target/' \
    --exclude 'node_modules/' \
    --exclude '/logs/' \
    --exclude '/runs/' \
    --exclude '/outputs/' \
    --exclude '/checkpoints/' \
    --exclude '/wandb/' \
    --exclude '/tensorboard/' \
    --exclude '/weights/' \
    --exclude '/models/' \
    --exclude '/datasets/' \
    --exclude '/data/' \
    "$ROOT/" "$REMOTE_SSH_HOST:$REMOTE_ROOT/"

  write_remote_source_revision "$ROOT" "$REMOTE_ROOT/.helicopter-source-revision"
  write_remote_source_revision \
    "$ROOT/src/infer/vllm-rwkv" \
    "$REMOTE_ROOT/src/infer/vllm-rwkv/.helicopter-source-revision"
}

local_git_revision() {
  local path="$1"
  local revision status

  revision="$(git -C "$path" rev-parse --verify HEAD 2>/dev/null)" || return 0
  status="$(git -C "$path" status --porcelain --untracked-files=all 2>/dev/null || true)"
  if [[ -n "$status" ]]; then
    revision="$revision-dirty"
  fi
  printf '%s\n' "$revision"
}

write_remote_source_revision() {
  local local_path="$1"
  local remote_marker="$2"
  local revision remote_dir

  revision="$(local_git_revision "$local_path")"
  [[ -n "$revision" ]] || return 0

  remote_dir="$(dirname "$remote_marker")"
  run ssh "$REMOTE_SSH_HOST" \
    "mkdir -p $(printf '%q' "$remote_dir") && printf '%s\n' $(printf '%q' "$revision") > $(printf '%q' "$remote_marker")"
}

remote_env_args() {
  local args=(
    "PYTHON_VERSION=$PYTHON_VERSION"
    "VENV=$REMOTE_VENV"
    "REMOTE_VENV=$REMOTE_VENV"
    "HELICOPTER_VENV=$REMOTE_VENV"
    "HELICOPTER_PYTHON=$REMOTE_VENV/bin/python"
    "CUDA_HOME=$REMOTE_CUDA_HOME"
    "CUDA_PATH=$REMOTE_CUDA_HOME"
    "INSTALL_COMPONENTS=$INSTALL_COMPONENTS"
    "INSTALL_SYSTEM_DEPS=0"
    "UPDATE_UV=$UPDATE_UV"
    "UV_UPGRADE=$UV_UPGRADE"
    "RUN_PIP_CHECK=$RUN_PIP_CHECK"
    "UV_SYNC_INEXACT=$UV_SYNC_INEXACT"
    "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
    "VLLM_BUILD_PROFILE=$VLLM_BUILD_PROFILE"
    "VLLM_VERSION_OVERRIDE=$VLLM_VERSION_OVERRIDE"
    "VLLM_USE_PRECOMPILED=$VLLM_USE_PRECOMPILED"
    "VLLM_REBUILD=$VLLM_REBUILD"
    "VERL_REINSTALL=$VERL_REINSTALL"
    "CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    "BUILD_TMPDIR=$BUILD_TMPDIR"
    "HTTP_PROXY=$REMOTE_HTTP_PROXY"
    "HTTPS_PROXY=$REMOTE_HTTPS_PROXY"
    "http_proxy=$REMOTE_HTTP_PROXY"
    "https_proxy=$REMOTE_HTTPS_PROXY"
    "NO_PROXY=$REMOTE_NO_PROXY"
    "no_proxy=$REMOTE_NO_PROXY"
    "ALL_PROXY="
    "all_proxy="
    "PYPI_INDEX_URL=$PYPI_INDEX_URL"
    "UV_INDEX_URL=$UV_INDEX_URL"
    "HF_ENDPOINT=$HF_ENDPOINT"
    "UV_LINK_MODE=$UV_LINK_MODE"
    "CARGO_REGISTRY_MIRROR=$CARGO_REGISTRY_MIRROR"
  )

  printf ' %q' "${args[@]}"
}

install_remote_env() {
  [[ "$INSTALL_REMOTE" == "1" ]] || return 0

  local quoted_root
  quoted_root="$(printf '%q' "$REMOTE_ROOT")"

  run ssh "$REMOTE_SSH_HOST" \
    "cd $quoted_root && env$(remote_env_args) bash scripts/install_local.sh"
}

validate_install_config
require_local_tools
verify_remote_tools
sync_remote_repo
install_remote_env

echo "Remote environment ready: $REMOTE_SSH_HOST:$REMOTE_ROOT"
