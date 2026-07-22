#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV="${VENV:-$ROOT/.venv}"
UV="${UV:-uv}"
UV_VERSION="${UV_VERSION:-0.11.14}"
INSTALL_COMPONENTS="${INSTALL_COMPONENTS:-rwkv-hf,rwkv-lm,dev}"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
UPDATE_UV="${UPDATE_UV:-0}"
UV_UPGRADE="${UV_UPGRADE:-1}"
RUN_PIP_CHECK="${RUN_PIP_CHECK:-1}"
UV_SYNC_INEXACT="${UV_SYNC_INEXACT:-1}"
CLEAN_SUBMODULE_VENVS="${CLEAN_SUBMODULE_VENVS:-1}"
CLEAN_VLLM_CMAKE_CACHE="${CLEAN_VLLM_CMAKE_CACHE:-1}"
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
# Any2RWKV BF16 conversion/distillation only needs rwkv-hf + rwkv-lm. This
# vLLM build setting belongs solely to the independently selected vllm-rwkv group.
VLLM_BUILD_PROFILE="${VLLM_BUILD_PROFILE:-rwkv}"
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
RWKV_HF="$ROOT/src/train/rwkv-hf"
ANY2RWKV="$ROOT/src/train/any2rwkv"
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

component_enabled() {
  local expected="$1" component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    [[ "$component" == "$expected" ]] && return 0
  done
  return 1
}

validate_install_components() {
  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  ((${#components[@]} > 0)) || die "INSTALL_COMPONENTS must select at least one dependency group"
  for component in "${components[@]}"; do
    case "$component" in
      dev | vllm-rwkv | verl-rwkv | rwkv-lm | rwkv-hf | lighteval | verl-liger) ;;
      full)
        die "INSTALL_COMPONENTS=full is disabled; select explicit dependency groups"
        ;;
      *)
        die "unknown INSTALL_COMPONENTS entry '$component'; use a comma-separated subset of dev,vllm-rwkv,verl-rwkv,rwkv-lm,rwkv-hf,lighteval,verl-liger"
        ;;
    esac
  done
  if component_enabled verl-rwkv && component_enabled lighteval; then
    die "verl-rwkv and lighteval are separate environments because their latex2sympy2-extended requirements conflict"
  fi

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

native_component_enabled() {
  component_enabled vllm-rwkv || component_enabled rwkv-lm || component_enabled rwkv-hf
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
  for env_dir in "$VLLM/.venv" "$VERL/.venv" "$RWKV_LM/.venv" "$RWKV_HF/.venv" "$ANY2RWKV/.venv"; do
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
  [[ "$UV_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] ||
    die "UV_VERSION must be an exact release such as 0.11.14"

  local actual_version="" install_uv=0
  if have "$UV"; then
    actual_version="$("$UV" --version | awk '{print $2}')"
  fi
  if [[ "$actual_version" != "$UV_VERSION" || "$UPDATE_UV" == "1" ]]; then
    install_uv=1
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      print_cmd bash -o pipefail -c "curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_NO_MODIFY_PATH=1 sh"
    else
      local standalone_installed=0
      if have curl; then
        print_cmd bash -o pipefail -c "curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_NO_MODIFY_PATH=1 sh"
        if bash -o pipefail -c "curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh | env UV_NO_MODIFY_PATH=1 sh"; then
          standalone_installed=1
        fi
      fi
      if [[ "$standalone_installed" == "0" ]]; then
        have "$UV" ||
          die "the pinned uv installer is unavailable and no existing uv can use the configured Python index"
        warn "the pinned uv standalone installer is unreachable; using the configured Python index"
        local tool_install=("$UV" tool install --force)
        [[ -n "$UV_INDEX_URL" ]] && tool_install+=(--index-url "$UV_INDEX_URL")
        tool_install+=("uv@$UV_VERSION")
        run "${tool_install[@]}"
      fi
    fi
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    [[ "${DRY_RUN:-0}" == "1" ]] || hash -r
    [[ ! -x "$HOME/.local/bin/uv" ]] || UV="$HOME/.local/bin/uv"
  fi
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    have "$UV" || UV="$(command -v uv || true)"
    [[ -n "$UV" ]] || die "uv installation finished but uv is still not on PATH"
    actual_version="$("$UV" --version | awk '{print $2}')"
    [[ "$actual_version" == "$UV_VERSION" ]] ||
      die "uv version mismatch: expected $UV_VERSION, found $actual_version"
    [[ "$install_uv" == "0" ]] || printf 'Pinned uv ready: %s\n' "$actual_version"
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
  native_component_enabled || return 0
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
  native_component_enabled || return 0
  local missing=()
  component_enabled vllm-rwkv && ! have cmake && missing+=("cmake")
  have ninja || missing+=("ninja")
  ((${#missing[@]} == 0)) || die "missing native build tools after uv sync: ${missing[*]}"

  if component_enabled vllm-rwkv; then
    local cmake_version
    cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"
    version_at_least "$cmake_version" "3.26" || die "cmake >= 3.26 is required; found $cmake_version"
  fi

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
  native_component_enabled || return 0
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
  native_component_enabled || return 0
  [[ "$VLLM_TARGET_DEVICE" == "cuda" ]] || return 0
  [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]] || return 0

  local arch_list=""
  if [[ -x "$VENV/bin/python" ]]; then
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
elif len(capabilities) == 1:
    major, minor = next(iter(capabilities))
    print(f"{major}.{minor}")
PY
)"
  elif have nvidia-smi; then
    arch_list="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
      awk 'NF { seen[$1] = 1 } END { if (length(seen) == 1) for (arch in seen) print arch }')"
  fi

  if [[ -n "$arch_list" ]]; then
    export TORCH_CUDA_ARCH_LIST="$arch_list"
    echo "Using TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST for native dependency builds"
  fi
}

sync_uv_env() {
  local sync_args=(sync)
  [[ -n "$UV_INDEX_URL" ]] && sync_args+=(--index-url "$UV_INDEX_URL")
  [[ "$UV_SYNC_INEXACT" == "1" ]] && sync_args+=(--inexact)
  sync_args+=(--project "$ROOT" --python "$PYTHON_VERSION" --no-default-groups)
  [[ "$UV_UPGRADE" == "1" ]] && sync_args+=(--upgrade)

  local component
  local -a components=()
  IFS=, read -r -a components <<<"$INSTALL_COMPONENTS"
  for component in "${components[@]}"; do
    sync_args+=(--group "$component")
  done

  run "$UV" "${sync_args[@]}"
}

vllm_native_fingerprint() {
  {
    printf 'VLLM_TARGET_DEVICE=%s\n' "$VLLM_TARGET_DEVICE"
    printf 'VLLM_BUILD_PROFILE=%s\n' "$VLLM_BUILD_PROFILE"
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
import vllm._rapid_sampling
import vllm.rwkv7_ops
from vllm.build_profile import get_build_profile_metadata

metadata = get_build_profile_metadata()
assert metadata.profile == "rwkv", metadata
assert "_C_stable_libtorch" not in metadata.configured_targets, metadata
PY
}

verl_ready() {
  "$VENV/bin/python" - <<'PY' >/dev/null
import verl
PY
}

rwkv_hf_ready() {
  "$VENV/bin/python" - <<'PY' >/dev/null
import rwkv7_hf
PY
}

any2rwkv_ready() {
  "$VENV/bin/python" - "$ANY2RWKV" <<'PY' >/dev/null
import pathlib
import sys

import any2rwkv

expected = pathlib.Path(sys.argv[1]).resolve()
actual = pathlib.Path(any2rwkv.__file__).resolve()
if expected not in actual.parents:
    raise SystemExit(f"any2rwkv import resolved outside product package: {actual}")
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
    VLLM_BUILD_PROFILE="$VLLM_BUILD_PROFILE" \
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

install_rwkv_hf_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  # uv sync has already installed the locked build backend into the workspace
  # venv. Reuse it so editable installation remains offline/reproducible when
  # the remote package index is unavailable.
  run "${pip[@]}" --no-deps --no-build-isolation -e "$RWKV_HF"
  [[ "${DRY_RUN:-0}" == "1" ]] || rwkv_hf_ready
}

install_any2rwkv_package() {
  local pip=( "$UV" pip install )
  [[ -n "$UV_INDEX_URL" ]] && pip+=(--index-url "$UV_INDEX_URL")
  pip+=(--project "$ROOT" --python "$VENV/bin/python" )

  run "${pip[@]}" --no-deps --no-build-isolation -e "$ANY2RWKV"
  [[ "${DRY_RUN:-0}" == "1" ]] || any2rwkv_ready
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

validate_install_components
configure_network
configure_build_dirs
clean_submodule_venvs
ensure_uv
check_compiler_env
check_cuda_env
configure_cuda_arch_list
sync_uv_env
check_native_env
component_enabled vllm-rwkv && clean_vllm_cmake_cache
component_enabled vllm-rwkv && install_vllm_package
component_enabled rwkv-lm && install_rwkv_lm_package
if component_enabled rwkv-hf; then
  install_rwkv_hf_package
  install_any2rwkv_package
fi
component_enabled verl-rwkv && install_verl_package
check_python_packages

clean_submodule_venvs

echo "Environment ready: $VENV"
