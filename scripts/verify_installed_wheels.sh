#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/.tmp"
TEMP_ROOT="$(mktemp -d "${ROOT}/.tmp/wheel-smoke.XXXXXX")"
trap 'rm -rf "${TEMP_ROOT}"' EXIT

uv build --wheel --out-dir "${TEMP_ROOT}/dist" "${ROOT}"

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
