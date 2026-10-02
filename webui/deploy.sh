#!/usr/bin/env bash
#
# Start the Mimosa Observatory locally: backend (FastAPI) + frontend (Vite).
# This is the supported way to run the tool — a single-operator, localhost UI
# with no authentication. Do not expose it on a shared or public network.
# See webui/README.md for details.
#
# Usage:
#   ./deploy.sh            install deps if needed, then run both servers
#   ./deploy.sh --check    run preflight + dependency install only, then exit
#
# Ports/hosts (override via environment):
#   MIMOSA_BACKEND_HOST   (default 127.0.0.1)
#   MIMOSA_BACKEND_PORT   (default 8848)
#   MIMOSA_FRONTEND_PORT  (default 5173)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$HERE/backend"
FRONTEND_DIR="$HERE/frontend"

BACKEND_HOST="${MIMOSA_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${MIMOSA_BACKEND_PORT:-8848}"
FRONTEND_PORT="${MIMOSA_FRONTEND_PORT:-5173}"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

log() { printf '\033[1;33m==>\033[0m %s\n' "$1"; }
die() { printf '\033[1;31mError:\033[0m %s\n' "$1" >&2; exit 1; }

command -v uv  >/dev/null 2>&1 || die "uv not found — install from https://docs.astral.sh/uv/"
command -v npm >/dev/null 2>&1 || die "npm not found — Node >= 20 is required"

# Non-fatal: observability works without it, but setup/refine/classify/launch
# need a Python that can import Mimosa.
mimosa_python="${MIMOSA_PYTHON:-${MIMOSA_ROOT:-$HERE/..}/.venv/bin/python}"
[ -x "$mimosa_python" ] || log "Note: Mimosa venv not found at $mimosa_python — observability works, but refine/classify/launch stay disabled (set MIMOSA_PYTHON)."

log "Installing backend dependencies (uv sync)"
( cd "$BACKEND_DIR" && uv sync )

if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  log "Installing frontend dependencies (npm install)"
  ( cd "$FRONTEND_DIR" && npm install )
fi

if [ "$CHECK_ONLY" = "1" ]; then
  log "Preflight OK. Run ./deploy.sh (without --check) to start the servers."
  exit 0
fi

pids=()
cleanup() {
  log "Shutting down"
  if [ "${#pids[@]}" -gt 0 ]; then
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  fi
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

log "Backend  -> http://$BACKEND_HOST:$BACKEND_PORT"
( cd "$BACKEND_DIR" && exec uv run uvicorn app.main:app --host "$BACKEND_HOST" --port "$BACKEND_PORT" ) &
pids+=("$!")

log "Frontend -> http://localhost:$FRONTEND_PORT   (open this)"
( cd "$FRONTEND_DIR" && exec env MIMOSA_API="http://$BACKEND_HOST:$BACKEND_PORT" \
    npm run dev -- --port "$FRONTEND_PORT" --strictPort ) &
pids+=("$!")

wait
