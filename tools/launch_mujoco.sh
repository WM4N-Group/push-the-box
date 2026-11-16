#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Install first: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements-mujoco.txt" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Run the trampoline explicitly: mjpython's shebang breaks on paths with spaces.
  exec "$PYTHON" "$ROOT/.venv/bin/mjpython" "$ROOT/tools/mujoco_push_scene.py" "$@"
fi
exec "$PYTHON" "$ROOT/tools/mujoco_push_scene.py" "$@"
