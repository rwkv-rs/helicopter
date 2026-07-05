#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-$ROOT/.venv}"
UV="${UV:-uv}"
INSTALL_PROFILE="${INSTALL_PROFILE:-rwkv}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
UPDATE_UV="${UPDATE_UV:-1}"
UV_UPGRADE="${UV_UPGRADE:-1}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"
CLEAN_SUBMODULE_VENVS="${CLEAN_SUBMODULE_VENVS:-1}"
CLEAN_VLLM_CMAKE_CACHE="${CLEAN_VLLM_CMAKE_CACHE:-1}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
VLLM_VERSION_OVERRIDE="${VLLM_VERSION_OVERRIDE:-}"
VLLM_REBUILD="${VLLM_REBUILD:-auto}"
VERL_REINSTALL="${VERL_REINSTALL:-auto}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-RelWithDebInfo}"
BUILD_TMPDIR="${BUILD_TMPDIR:-}"
UV_INDEX_URL="${UV_INDEX_URL:-${PYPI_INDEX_URL:-}}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
CARGO_REGISTRY_MIRROR="${CARGO_REGISTRY_MIRROR:-}"
CARGO_REGISTRY_MIRROR_NAME="${CARGO_REGISTRY_MIRROR_NAME:-rsproxy-sparse}"

VLLM="$ROOT/src/infer/vllm-rwkv"
RWKV_LM="$ROOT/src/train/rwkv-lm"
VERL="$ROOT/src/train/verl-rwkv"
STAMP_DIR="$VENV/.helicopter-stamps"
VLLM_STAMP="$STAMP_DIR/vllm-native.sha256"

export PATH="$VENV/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

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

die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

version_at_least() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}

configure_network() {
  [[ -n "$HF_ENDPOINT" ]] && export HF_ENDPOINT
  [[ -n "${UV_LINK_MODE:-}" ]] && export UV_LINK_MODE

  if [[ -n "$CARGO_REGISTRY_MIRROR" ]]; then
    export CARGO_HOME="${CARGO_HOME:-$VENV/.cargo}"
    mkdir -p "$CARGO_HOME"
    cat >"$CARGO_HOME/config.toml" <<EOF
[source.crates-io]
replace-with = "$CARGO_REGISTRY_MIRROR_NAME"

[source.$CARGO_REGISTRY_MIRROR_NAME]
registry = "$CARGO_REGISTRY_MIRROR"
EOF
  fi
}

configure_build_dirs() {
  if [[ -n "$BUILD_TMPDIR" ]]; then
    mkdir -p "$BUILD_TMPDIR"
    export TMPDIR="$BUILD_TMPDIR"
  fi
}

clean_submodule_venvs() {
  [[ "$CLEAN_SUBMODULE_VENVS" == "1" ]] || return 0

  local env_dir
  for env_dir in "$VLLM/.venv" "$VERL/.venv" "$RWKV_LM/.venv"; do
    [[ -e "$env_dir" ]] || continue
    [[ "$env_dir" == "$ROOT"/src/*/.venv ]] || die "refusing to remove unexpected venv path: $env_dir"
    run rm -rf "$env_dir"
  done
}

clean_vllm_cmake_cache() {
  [[ "$CLEAN_VLLM_CMAKE_CACHE" == "1" ]] || return 0
  [[ -d "$VLLM/.deps" ]] || return 0

  local subbuild_dir
  while IFS= read -r subbuild_dir; do
    [[ -n "$subbuild_dir" ]] || continue
    [[ "$subbuild_dir" == "$VLLM/.deps/"*-subbuild ]] ||
      die "refusing to remove unexpected CMake cache path: $subbuild_dir"
    run rm -rf "$subbuild_dir"
  done < <(find "$VLLM/.deps" -maxdepth 1 -type d -name '*-subbuild' -print | LC_ALL=C sort)
}

ensure_uv() {
  if ! have "$UV"; then
    have curl || die "uv is missing and curl is not available to install it"
    run sh -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    have "$UV" || UV="$(command -v uv || true)"
    [[ "${DRY_RUN:-0}" == "1" || -n "$UV" ]] || die "uv installation finished but uv is still not on PATH"
  fi

  if [[ "$UPDATE_UV" == "1" ]]; then
    run "$UV" self update || warn "uv self update failed; continuing with installed uv"
  fi
}

install_system_deps() {
  [[ "$INSTALL_SYSTEM_DEPS" == "1" ]] || return 0
  have apt-get || die "INSTALL_SYSTEM_DEPS=1 currently supports apt-get only"
  run sudo apt-get update
  run sudo apt-get install -y --no-install-recommends \
    build-essential curl git ninja-build pkg-config
}

check_compiler_env() {
  local missing=()
  have cc || missing+=("cc")
  have c++ || missing+=("c++")

  if ((${#missing[@]})); then
    install_system_deps
    missing=()
    have cc || missing+=("cc")
    have c++ || missing+=("c++")
  fi

  ((${#missing[@]} == 0)) || die "missing C/C++ build tools: ${missing[*]}"
}

check_native_env() {
  local missing=()
  have cmake || missing+=("cmake")
  have ninja || missing+=("ninja")
  ((${#missing[@]} == 0)) || die "missing native build tools after uv sync: ${missing[*]}"

  local cmake_version
  cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
  version_at_least "$cmake_version" "3.26" || die "cmake >= 3.26 is required; found $cmake_version"

  if have g++; then
    local gcc_version
    gcc_version="$(g++ -dumpfullversion -dumpversion)"
    version_at_least "$gcc_version" "11.3" || die "g++ >= 11.3 is required; found $gcc_version"
  fi

  if [[ "${VLLM_REQUIRE_RUST_FRONTEND:-0}" == "1" ]]; then
    have rustc || die "rustc is required when VLLM_REQUIRE_RUST_FRONTEND=1"
    have cargo || die "cargo is required when VLLM_REQUIRE_RUST_FRONTEND=1"
  fi
}

check_cuda_env() {
  [[ "$VLLM_TARGET_DEVICE" == "cuda" ]] || return 0

  if ! have nvcc && [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
  fi

  have nvcc || die "nvcc is required for VLLM_TARGET_DEVICE=cuda; set CUDA_HOME or install the CUDA toolkit"

  if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
    export CUDA_HOME CUDA_PATH="$CUDA_HOME"
  fi

  have nvidia-smi || warn "nvidia-smi is not on PATH; nvcc exists, so build can continue"
}

configure_cuda_arch_list() {
  [[ "$VLLM_TARGET_DEVICE" == "cuda" ]] || return 0
  [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] || return 0
  [[ -x "$VENV/bin/python" ]] || return 0

  local arch_list
  arch_list="$("$VENV/bin/python" - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit(0)

capabilities = {
    torch.cuda.get_device_capability(index)
    for index in range(torch.cuda.device_count())
}
supported_arches = set(torch.cuda.get_arch_list())

if capabilities == {(12, 1)} and "sm_121" not in supported_arches and "sm_120" in supported_arches:
    print("12.0+PTX")
PY
)"

  if [[ -n "$arch_list" ]]; then
    export TORCH_CUDA_ARCH_LIST="$arch_list"
    echo "Using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST for CUDA 12.1 devices supported by sm_120"
  fi
}

sync_uv_env() {
  local sync_args=(sync)
  [[ -n "$UV_INDEX_URL" ]] && sync_args+=(--index-url "$UV_INDEX_URL")
  [[ "$UV_SYNC_INEXACT" == "1" ]] && sync_args+=(--inexact)
  sync_args+=(--project "$ROOT" --python "$PYTHON_VERSION" --no-default-groups --group rwkv)
  [[ "$UV_UPGRADE" == "1" ]] && sync_args+=(--upgrade)

  case "$INSTALL_PROFILE" in
    rwkv) ;;
    full) sync_args+=(--group full) ;;
    *) die "unknown INSTALL_PROFILE=$INSTALL_PROFILE; use rwkv or full" ;;
  esac

  run "$UV" "${sync_args[@]}"
}

vllm_native_fingerprint() {
  {
    printf 'VLLM_TARGET_DEVICE=%s\n' "$VLLM_TARGET_DEVICE"
    printf 'VLLM_VERSION_OVERRIDE=%s\n' "$VLLM_VERSION_OVERRIDE"
    printf 'CMAKE_BUILD_TYPE=%s\n' "$CMAKE_BUILD_TYPE"
    printf 'TORCH_CUDA_ARCH_LIST=%s\n' "${TORCH_CUDA_ARCH_LIST:-}"
    "$VENV/bin/python" - <<'PY'
import platform
import sys

import torch

print(f"python={sys.version}")
print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
PY
    find "$VLLM/CMakeLists.txt" "$VLLM/setup.py" "$VLLM/cmake" "$VLLM/csrc" \
      -type f -print 2>/dev/null | LC_ALL=C sort | while IFS= read -r path; do
        sha256sum "$path"
      done
  } | sha256sum | awk '{print $1}'
}

vllm_native_ready() {
  "$VENV/bin/python" - <<'PY' >/dev/null
import vllm
import vllm._C_stable_libtorch
import vllm.rwkv7_ops
PY
}

verl_ready() {
  "$VENV/bin/python" - <<'PY' >/dev/null
import verl
PY
}

install_vllm_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  mkdir -p "$STAMP_DIR"
  local fingerprint
  fingerprint="$(vllm_native_fingerprint)"

  if [[ "$VLLM_REBUILD" != "1" && -f "$VLLM_STAMP" ]] &&
     [[ "$(cat "$VLLM_STAMP")" == "$fingerprint" ]] &&
     vllm_native_ready; then
    echo "vLLM native extensions are already built for this source and environment; reusing existing install"
    return 0
  fi

  run env \
    VLLM_TARGET_DEVICE="$VLLM_TARGET_DEVICE" \
    VLLM_VERSION_OVERRIDE="$VLLM_VERSION_OVERRIDE" \
    VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}" \
    CMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    "${pip[@]}" --no-deps --no-build-isolation -e "$VLLM" --torch-backend=auto

  vllm_native_ready
  fingerprint="$(vllm_native_fingerprint)"
  printf '%s\n' "$fingerprint" >"$VLLM_STAMP"
}

install_rwkv_lm_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  if [[ -f "$RWKV_LM/pyproject.toml" || -f "$RWKV_LM/setup.py" ]]; then
    run "${pip[@]}" --no-deps -e "$RWKV_LM"
  else
    echo "rwkv-lm has no local package metadata; dependencies are covered by pyproject.toml"
  fi
}

install_verl_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  if [[ "$VERL_REINSTALL" != "1" ]] && verl_ready; then
    echo "verl editable package is already installed; reusing existing install"
    return 0
  fi
  run "${pip[@]}" --no-deps -e "$VERL"
}

check_python_packages() {
  [[ "$RUN_PIP_CHECK" == "1" ]] || return 0

  print_cmd "$UV" pip check --project "$ROOT" --python "$VENV/bin/python"
  [[ "${DRY_RUN:-0}" == "1" ]] && return 0

  local check_output filtered_output
  if check_output="$("$UV" pip check --project "$ROOT" --python "$VENV/bin/python" 2>&1)"; then
    printf '%s\n' "$check_output"
    return 0
  fi

  filtered_output="$(printf '%s\n' "$check_output" |
    grep -v -F 'The package `nvidia-cusparselt-cu13` was built for a different platform' |
    grep -v -E '^(Checked [0-9]+ packages in .+|Found 1 incompatibility)$' || true)"
  if [[ -z "$filtered_output" ]] &&
     [[ "$check_output" == *'The package `nvidia-cusparselt-cu13` was built for a different platform'* ]]; then
    printf '%s\n' "$check_output" >&2
    warn "ignoring uv platform-tag check for nvidia-cusparselt-cu13; NVIDIA publishes the aarch64 wheel with a manylinux2014_sbsa tag"
    return 0
  fi

  printf '%s\n' "$check_output" >&2
  return 1
}

configure_network
configure_build_dirs
clean_submodule_venvs
ensure_uv
check_compiler_env
sync_uv_env
check_native_env
check_cuda_env
configure_cuda_arch_list
clean_vllm_cmake_cache
install_vllm_package
install_rwkv_lm_package
install_verl_package
check_python_packages

clean_submodule_venvs

echo "Environment ready: $VENV"
