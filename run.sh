#!/usr/bin/env bash

set -Eeuo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ -x "$repo_dir/.venv/bin/python" ]]; then
  python_bin="$repo_dir/.venv/bin/python"
elif ! python_bin="$(command -v python3)"; then
  echo "python3 is required to start Luxo" >&2
  exit 127
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required to start the renderer" >&2
  exit 127
fi

if [[ ! -d "$repo_dir/renderer" ]]; then
  echo "renderer directory is missing" >&2
  exit 1
fi

core_pid=""
renderer_pid=""

terminate_children() {
  local child_pid
  for child_pid in "$core_pid" "$renderer_pid"; do
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
      kill -TERM "$child_pid" 2>/dev/null || true
    fi
  done
  for child_pid in "$core_pid" "$renderer_pid"; do
    if [[ -n "$child_pid" ]]; then
      wait "$child_pid" 2>/dev/null || true
    fi
  done
}

handle_signal() {
  local exit_code="$1"
  trap - EXIT INT TERM
  terminate_children
  exit "$exit_code"
}

trap terminate_children EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

(
  cd -- "$repo_dir"
  exec "$python_bin" -m core.main
) &
core_pid="$!"

npm run dev --prefix "$repo_dir/renderer" &
renderer_pid="$!"

while kill -0 "$core_pid" 2>/dev/null && kill -0 "$renderer_pid" 2>/dev/null; do
  sleep 0.2
done

set +e
if ! kill -0 "$core_pid" 2>/dev/null; then
  wait "$core_pid"
  exit_code="$?"
else
  wait "$renderer_pid"
  exit_code="$?"
fi
set -e

exit "$exit_code"
