#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACK_PID=""

cleanup() {
  if [[ -n "$BACK_PID" ]]; then
    kill "$BACK_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

echo "Starting Offline Notebook LM (backend + desktop)..."
echo

UV_BIN="${UV_BIN:-$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")}"

if [ -x "$UV_BIN" ]; then
  echo "[dev] Using uv for backend ($UV_BIN)"
  (
    cd "$ROOT_DIR/backend"
    "$UV_BIN" run uvicorn notebooklm_backend.app:create_app --factory --host 127.0.0.1 --port 8000
  ) & BACK_PID=$!
else
  echo "[dev] Using python venv for backend"
  if [ ! -d "$ROOT_DIR/backend/.venv" ]; then
    python3 -m venv "$ROOT_DIR/backend/.venv"
    source "$ROOT_DIR/backend/.venv/bin/activate"
    pip install -e ".[dev]"
  else
    source "$ROOT_DIR/backend/.venv/bin/activate"
  fi
  (
    cd "$ROOT_DIR/backend"
    uvicorn notebooklm_backend.app:create_app --factory --host 127.0.0.1 --port 8000
  ) & BACK_PID=$!
fi

echo "[dev] Backend started at http://127.0.0.1:8000 (pid=$BACK_PID)"

cd "$ROOT_DIR/apps/desktop"
npm run dev

