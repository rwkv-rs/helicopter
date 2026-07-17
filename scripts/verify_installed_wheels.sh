#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIGHTEVAL_REQUIREMENT="lighteval @ git+https://github.com/huggingface/lighteval.git@64f4f5ae173626509fad6e477ca4ee56ebb26129"
mkdir -p "${ROOT}/.tmp"
TEMP_ROOT="$(mktemp -d "${ROOT}/.tmp/wheel-smoke.XXXXXX")"
trap 'rm -rf "${TEMP_ROOT}"' EXIT

uv build --wheel --out-dir "${TEMP_ROOT}/dist" "${ROOT}"
uv build --wheel --out-dir "${TEMP_ROOT}/dist" "${ROOT}/src/eval/lighteval"

ROOT_WHEEL="$(find "${TEMP_ROOT}/dist" -maxdepth 1 -name 'helicopter-*.whl' -print -quit)"
if [[ -z "${ROOT_WHEEL}" ]]; then
  echo "helicopter wheel was not built" >&2
  exit 1
fi

uv venv --python 3.12 "${TEMP_ROOT}/base"
uv pip install --python "${TEMP_ROOT}/base/bin/python" "${ROOT_WHEEL}"
"${TEMP_ROOT}/base/bin/helicopter" --help >/dev/null
if "${TEMP_ROOT}/base/bin/python" -c 'import helicopter_lighteval' 2>/dev/null; then
  echo "base wheel unexpectedly imports helicopter_lighteval" >&2
  exit 1
fi

uv venv --python 3.12 "${TEMP_ROOT}/eval"
uv pip install \
  --python "${TEMP_ROOT}/eval/bin/python" \
  --find-links "${TEMP_ROOT}/dist" \
  "${ROOT_WHEEL}[eval]" \
  "${LIGHTEVAL_REQUIREMENT}"
"${TEMP_ROOT}/eval/bin/helicopter" eval --help >/dev/null
"${TEMP_ROOT}/eval/bin/python" - <<'PY'
from helicopter_lighteval import run_evaluation
from helicopter_lighteval.scoreboard import publish_manifest
assert callable(run_evaluation)
assert callable(publish_manifest)
PY
