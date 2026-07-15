#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
if "${TEMP_ROOT}/base/bin/python" -c 'import lighteval_runner' 2>/dev/null; then
  echo "base wheel unexpectedly imports lighteval_runner" >&2
  exit 1
fi

for profile in eval full; do
  uv venv --python 3.12 "${TEMP_ROOT}/${profile}"
  uv pip install \
    --python "${TEMP_ROOT}/${profile}/bin/python" \
    --find-links "${TEMP_ROOT}/dist" \
    "${ROOT_WHEEL}[${profile}]"
  "${TEMP_ROOT}/${profile}/bin/helicopter" eval --help >/dev/null
  "${TEMP_ROOT}/${profile}/bin/python" - <<'PY'
from importlib.resources import files

root = files("lighteval_runner")
assert root.joinpath("tasks/assets/function_calling_v1.jsonl").is_file()
assert root.joinpath("tasks/assets/coding_v1.jsonl").is_file()
PY
done
