#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${SCOREBOARD_POSTGRES_IMAGE:-postgres:17}"
MIRROR_IMAGE="${SCOREBOARD_POSTGRES_MIRROR_IMAGE:-docker.m.daocloud.io/library/postgres:17}"
PORT="${SCOREBOARD_POSTGRES_TEST_PORT:-55432}"
CONTAINER="helicopter-scoreboard-postgres-$$"

docker_command=(docker)
if ! docker info >/dev/null 2>&1; then
  sudo -n docker info >/dev/null
  docker_command=(sudo -n docker)
fi

if ! "${docker_command[@]}" image inspect "$IMAGE" >/dev/null 2>&1; then
  "${docker_command[@]}" pull "$MIRROR_IMAGE"
  "${docker_command[@]}" tag "$MIRROR_IMAGE" "$IMAGE"
fi

cleanup() {
  "${docker_command[@]}" stop "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"${docker_command[@]}" run --rm -d --name "$CONTAINER" \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p "127.0.0.1:${PORT}:5432" "$IMAGE" >/dev/null

for _ in $(seq 1 30); do
  if "${docker_command[@]}" exec "$CONTAINER" pg_isready -U postgres >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
"${docker_command[@]}" exec "$CONTAINER" pg_isready -U postgres >/dev/null

SCOREBOARD_TEST_POSTGRES_URL="postgresql://postgres@127.0.0.1:${PORT}/postgres" \
  "$ROOT/.venv/bin/python" -m pytest -q "$ROOT/tests"
