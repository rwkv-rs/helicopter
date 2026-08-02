#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_root="${project_root}/src/infer/vllm-rwkv"
model_path="${project_root}/models/rwkv7/rwkv7-g1i_preview5445-1.5b-20260729-ctx16384.pth"
model_path="${RWKV_MODEL_PATH:-${model_path}}"

export PATH="${vllm_root}/.venv-rwkv/bin:/usr/local/cuda-13.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"
export TMPDIR=/tmp
export TMP=/tmp
export TEMP=/tmp
export CUDA_HOME=/usr/local/cuda-13.0
export VLLM_RWKV7_WKV_MODE="${VLLM_RWKV7_WKV_MODE:-fp16}"
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"
export VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}"
export HF_HUB_DISABLE_TELEMETRY=1

exec "${vllm_root}/.venv-rwkv/bin/vllm" serve "${model_path}" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name rwkv7-g1i-1.5b \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  "$@"
