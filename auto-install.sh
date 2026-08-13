#!/usr/bin/env bash
#
# auto-install.sh — one-shot installer + launcher for Mimosa-AI.
#
# Brings up everything Mimosa needs and then opens the Observatory web UI:
#
#   1. Preflight   — git / uv / Python 3.12 / Node 20+ / Docker present, Docker daemon running
#   2. API key     — make sure at least one LLM provider key is available
#   3. Toolomics   — clone + ./start.sh (MCP tools, ports 5000-5200, via Docker)
#   4. Perspicacité— clone + uv run perspicacite serve (literature grounding, :5468)
#   5. Mimosa      — uv sync (creates .venv); optionally `uv tool install` the CLI
#   6. Observatory — webui/deploy.sh (backend :8848 + frontend :5173)
#
# Toolomics and Perspicacité are cloned next to this checkout (see MIMOSA_WORKDIR)
# and left running in the background; Mimosa auto-discovers both over their ports.
#
# Usage:
#   ./auto-install.sh                 full install, then launch the web UI
#   ./auto-install.sh --yes           non-interactive (accept every default)
#   ./auto-install.sh --no-webui      install/start deps only, then exit
#   ./auto-install.sh --help          show all flags
#
# Common overrides (environment variables):
#   MIMOSA_WORKDIR         where sibling repos are cloned (default: parent of this repo)
#   PERSPICACITE_PORT      Perspicacité port                (default: 5468)
#   TOOLOMICS_PORT_MIN/MAX Toolomics MCP scan range         (default: 5000 / 5200)
#   MIMOSA_BACKEND_PORT    Observatory backend              (default: 8848)
#   MIMOSA_FRONTEND_PORT   Observatory frontend             (default: 5173)

set -euo pipefail

# --------------------------------------------------------------------------- #
# Paths & defaults
# --------------------------------------------------------------------------- #
MIMOSA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${MIMOSA_WORKDIR:-$(dirname "$MIMOSA_DIR")}"

TOOLOMICS_REPO="https://github.com/HolobiomicsLab/Toolomics.git"
PERSPICACITE_REPO="https://github.com/HolobiomicsLab/Perspicacite-AI.git"
TOOLOMICS_DIR="$WORKDIR/Toolomics"
PERSPICACITE_DIR="$WORKDIR/Perspicacite-AI"

PERSPICACITE_PORT="${PERSPICACITE_PORT:-5468}"
TOOLOMICS_PORT_MIN="${TOOLOMICS_PORT_MIN:-5000}"
TOOLOMICS_PORT_MAX="${TOOLOMICS_PORT_MAX:-5200}"
BACKEND_PORT="${MIMOSA_BACKEND_PORT:-8848}"
FRONTEND_PORT="${MIMOSA_FRONTEND_PORT:-5173}"

# LLM providers Mimosa recognises (any one is enough).
API_KEYS="ANTHROPIC_API_KEY OPENAI_API_KEY MISTRAL_API_KEY DEEPSEEK_API_KEY HF_TOKEN OPENROUTER_API_KEY"
PROJECT_ENV="$MIMOSA_DIR/.env"
USER_ENV="${XDG_CONFIG_HOME:-$HOME/.config}/mimosa/.env"

# Flags (defaults)
ASSUME_YES=0
INSTALL_CLI=ask
LAUNCH_WEBUI=1
DO_TOOLOMICS=1
DO_PERSPICACITE=1
OPEN_BROWSER=1

# --------------------------------------------------------------------------- #
# Pretty output
# --------------------------------------------------------------------------- #
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[1;31m'; GRN=$'\033[1;32m'
  YEL=$'\033[1;33m'; BLU=$'\033[1;34m'; RST=$'\033[0m'
else
  BOLD=; DIM=; RED=; GRN=; YEL=; BLU=; RST=
fi

step() { printf '\n%s==>%s %s%s%s\n' "$BLU" "$RST" "$BOLD" "$1" "$RST"; }
ok()   { printf '  %s✓%s %s\n' "$GRN" "$RST" "$1"; }
info() { printf '  %s·%s %s\n' "$DIM" "$RST" "$1"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$RST" "$1" >&2; }
die()  { printf '\n%sError:%s %s\n' "$RED" "$RST" "$1" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Yes/no prompt honouring --yes (returns 0 for yes).
confirm() {
  local prompt="$1" default="${2:-y}" reply
  if [ "$ASSUME_YES" = 1 ]; then return 0; fi
  if [ ! -t 0 ]; then [ "$default" = y ]; return; fi
  local hint="[Y/n]"; [ "$default" = n ] && hint="[y/N]"
  printf '  %s?%s %s %s ' "$YEL" "$RST" "$prompt" "$hint" >&2
  read -r reply || true
  reply="${reply:-$default}"
  case "$reply" in [Yy]*) return 0 ;; *) return 1 ;; esac
}

# TCP port check (no external tools — bash /dev/tcp).
port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && { exec 3>&- 3<&-; return 0; }
  return 1
}

# Block until a port accepts connections, or time out.
wait_for_port() {
  local port="$1" label="$2" timeout="${3:-120}" waited=0
  printf '  %s·%s waiting for %s on :%s ' "$DIM" "$RST" "$label" "$port"
  while ! port_open "$port"; do
    if [ "$waited" -ge "$timeout" ]; then printf '\n'; return 1; fi
    printf '.'; sleep 2; waited=$((waited + 2))
  done
  printf ' %sup%s\n' "$GRN" "$RST"
}

# Is anything listening anywhere in the Toolomics range?
range_open() {
  local p
  for ((p = TOOLOMICS_PORT_MIN; p <= TOOLOMICS_PORT_MAX; p++)); do
    port_open "$p" && return 0
  done
  return 1
}

# Block until any port in the Toolomics range accepts connections, or time out.
wait_for_range() {
  local label="$1" timeout="${2:-300}" waited=0
  printf '  %s·%s waiting for %s on :%s-%s ' "$DIM" "$RST" "$label" "$TOOLOMICS_PORT_MIN" "$TOOLOMICS_PORT_MAX"
  while ! range_open; do
    if [ "$waited" -ge "$timeout" ]; then printf '\n'; return 1; fi
    printf '.'; sleep 3; waited=$((waited + 3))
  done
  printf ' %sup%s\n' "$GRN" "$RST"
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
usage() {
  # Print the header comment block (everything after the shebang, up to the
  # first non-comment line), stripped of the leading "# ".
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -y|--yes)          ASSUME_YES=1 ;;
    --cli)             INSTALL_CLI=yes ;;
    --no-cli)          INSTALL_CLI=no ;;
    --no-webui)        LAUNCH_WEBUI=0 ;;
    --skip-toolomics)  DO_TOOLOMICS=0 ;;
    --skip-perspicacite) DO_PERSPICACITE=0 ;;
    --no-open)         OPEN_BROWSER=0 ;;
    -h|--help)         usage ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #
printf '%s\n' "$BOLD"
printf '  ┌────────────────────────────────────────────┐\n'
printf '  │   Mimosa-AI · auto-install                  │\n'
printf '  └────────────────────────────────────────────┘\n'
printf '%s' "$RST"
info "Mimosa checkout : $MIMOSA_DIR"
info "Sibling repos in: $WORKDIR"

# --------------------------------------------------------------------------- #
# 1. Preflight
# --------------------------------------------------------------------------- #
step "1/6  Checking prerequisites"

have git || die "git not found — install the Xcode command-line tools or Git."

if ! have uv; then
  if confirm "uv not found. Install it now (astral.sh installer)?"; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1090
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"
    have uv || die "uv installed but not on PATH — open a new shell and re-run."
  else
    die "uv is required — https://docs.astral.sh/uv/"
  fi
fi
ok "uv    $(uv --version 2>/dev/null | awk '{print $2}')"

# Python 3.12 — required by Mimosa; uv installs it on demand without root.
if ! uv python find 3.12 >/dev/null 2>&1; then
  info "Python 3.12 not found — installing via uv"
  uv python install 3.12 || die "could not install Python 3.12 — install it manually (https://www.python.org or 'brew install python@3.12')."
fi
ok "python $(uv python find 3.12) ($(uv run --python 3.12 python -V 2>/dev/null || echo 3.12))"

have npm || die "npm not found — install Node.js 20+ (https://nodejs.org or 'brew install node')."
node_major="$(node -v 2>/dev/null | sed 's/^v//' | cut -d. -f1)"
[ -n "$node_major" ] && [ "$node_major" -ge 20 ] 2>/dev/null \
  || die "Node.js 20+ required (found $(node -v 2>/dev/null || echo none))."
ok "node  $(node -v)"

# --------------------------------------------------------------------------- #
# 2. Docker
# --------------------------------------------------------------------------- #
if [ "$DO_TOOLOMICS" = 1 ]; then
  step "2/6  Ensuring Docker is running"
  have docker || die "docker not found — install Docker Desktop (https://docker.com/products/docker-desktop)."

  if docker info >/dev/null 2>&1; then
    ok "Docker daemon is running"
  else
    info "Docker daemon not responding — trying to start it"
    case "$(uname -s)" in
      Darwin) open -a Docker 2>/dev/null || open -a "Docker Desktop" 2>/dev/null || true ;;
      Linux)  (sudo systemctl start docker 2>/dev/null || systemctl --user start docker 2>/dev/null || true) ;;
    esac
    waited=0
    printf '  %s·%s waiting for Docker daemon ' "$DIM" "$RST"
    until docker info >/dev/null 2>&1; do
      if [ "$waited" -ge 120 ]; then
        printf '\n'; die "Docker did not become ready — start Docker Desktop manually and re-run."
      fi
      printf '.'; sleep 3; waited=$((waited + 3))
    done
    printf ' %sready%s\n' "$GRN" "$RST"
  fi
else
  step "2/6  Docker  (skipped — --skip-toolomics)"
fi

# --------------------------------------------------------------------------- #
# 3. API key
# --------------------------------------------------------------------------- #
step "3/6  Checking LLM API key"

# Look in the live environment first, then the two dotenv files Mimosa loads.
api_key_present() {
  local key val f
  for key in $API_KEYS; do
    val="$(printf '%s' "${!key:-}")"
    [ -n "$val" ] && { echo "$key (environment)"; return 0; }
  done
  for f in "$PROJECT_ENV" "$USER_ENV"; do
    [ -f "$f" ] || continue
    for key in $API_KEYS; do
      if grep -Eq "^[[:space:]]*(export[[:space:]]+)?$key[[:space:]]*=[[:space:]]*['\"]?[^'\"[:space:]]+" "$f"; then
        echo "$key ($f)"; return 0
      fi
    done
  done
  return 1
}

if found="$(api_key_present)"; then
  ok "Found API key: $found"
else
  warn "No LLM provider key found."
  info "Supported: ANTHROPIC / OPENAI / MISTRAL / DEEPSEEK / HF_TOKEN / OPENROUTER"
  if [ "$ASSUME_YES" = 0 ] && [ -t 0 ]; then
    printf '  %s?%s Which provider var? (blank to skip) ' "$YEL" "$RST" >&2
    read -r kv || true
    if [ -n "${kv:-}" ]; then
      printf '  %s?%s Paste the value for %s: ' "$YEL" "$RST" "$kv" >&2
      read -rs val || true; printf '\n' >&2
      if [ -n "${val:-}" ]; then
        printf '%s=%s\n' "$kv" "$val" >> "$PROJECT_ENV"
        chmod 600 "$PROJECT_ENV" 2>/dev/null || true
        ok "Wrote $kv to $PROJECT_ENV"
      fi
    fi
  fi
  api_key_present >/dev/null || warn "Continuing without a key — the web UI's observability works, but launching/refining runs will be disabled until you add one to $PROJECT_ENV."
fi

# --------------------------------------------------------------------------- #
# Helper: clone or refresh a repo
# --------------------------------------------------------------------------- #
clone_repo() {
  local url="$1" dir="$2" name="$3"
  if [ -d "$dir/.git" ]; then
    ok "$name already cloned ($dir)"
  else
    info "Cloning $name"
    git clone --depth 1 "$url" "$dir"
    ok "$name cloned"
  fi
}

# --------------------------------------------------------------------------- #
# 4. Toolomics
# --------------------------------------------------------------------------- #
if [ "$DO_TOOLOMICS" = 1 ]; then
  step "4/6  Toolomics (MCP tools)"
  clone_repo "$TOOLOMICS_REPO" "$TOOLOMICS_DIR" "Toolomics"
  [ -x "$TOOLOMICS_DIR/start.sh" ] || chmod +x "$TOOLOMICS_DIR/start.sh" 2>/dev/null || true

  if range_open; then
    ok "Something is already listening in :$TOOLOMICS_PORT_MIN-$TOOLOMICS_PORT_MAX — assuming Toolomics is up"
  else
    info "Starting Toolomics in the background (docker compose, first run can take several minutes)"
    # start.sh supervises its MCP servers forever and never returns on its own —
    # it must run detached, or every step below would block waiting on it.
    ( cd "$TOOLOMICS_DIR" && nohup ./start.sh </dev/null > toolomics.log 2>&1 &
      echo $! > .toolomics.pid )
    if wait_for_range "Toolomics" 600; then
      ok "Toolomics MCP servers are up (logs: $TOOLOMICS_DIR/toolomics.log)"
    else
      warn "No listener detected in :$TOOLOMICS_PORT_MIN-$TOOLOMICS_PORT_MAX after 10min — containers may still be building; see $TOOLOMICS_DIR/toolomics.log. Mimosa will discover them once ready."
    fi
  fi
else
  step "4/6  Toolomics  (skipped — --skip-toolomics)"
fi

# --------------------------------------------------------------------------- #
# 5. Perspicacité
# --------------------------------------------------------------------------- #
if [ "$DO_PERSPICACITE" = 1 ]; then
  step "5/6  Perspicacité (literature grounding)"
  clone_repo "$PERSPICACITE_REPO" "$PERSPICACITE_DIR" "Perspicacité"

  if port_open "$PERSPICACITE_PORT"; then
    ok "Perspicacité already running on :$PERSPICACITE_PORT"
  else
    info "Installing Perspicacité deps (uv sync)"
    ( cd "$PERSPICACITE_DIR" && uv sync )
    info "Launching Perspicacité in the background (perspicacite serve)"
    ( cd "$PERSPICACITE_DIR" && nohup uv run perspicacite serve --port "$PERSPICACITE_PORT" > perspicacite.log 2>&1 &
      echo $! > .perspicacite.pid )
    if wait_for_port "$PERSPICACITE_PORT" "Perspicacité" 90; then
      ok "Perspicacité is up (logs: $PERSPICACITE_DIR/perspicacite.log)"
    else
      warn "Perspicacité didn't answer on :$PERSPICACITE_PORT within 90s — see $PERSPICACITE_DIR/perspicacite.log"
    fi
  fi
else
  step "5/6  Perspicacité  (skipped — --skip-perspicacite)"
fi

# --------------------------------------------------------------------------- #
# 6. Mimosa itself (+ optional CLI)
# --------------------------------------------------------------------------- #
step "6/6  Mimosa core"
info "Installing Mimosa deps (uv sync — creates .venv for the web UI bridge)"
( cd "$MIMOSA_DIR" && uv sync --python 3.12 )
ok "Mimosa environment ready"

case "$INSTALL_CLI" in
  yes) do_cli=1 ;;
  no)  do_cli=0 ;;
  ask) if confirm "Install the 'mimosa' CLI system-wide (uv tool install)?" n; then do_cli=1; else do_cli=0; fi ;;
esac
if [ "${do_cli:-0}" = 1 ]; then
  info "Installing the mimosa CLI (uv tool install)"
  uv tool install --force "$MIMOSA_DIR"
  ok "'mimosa' command installed — run 'mimosa' from anywhere"
else
  info "Skipped system-wide CLI (run Mimosa with: cd $MIMOSA_DIR && uv run main.py)"
fi

# --------------------------------------------------------------------------- #
# Summary + launch
# --------------------------------------------------------------------------- #
printf '\n%s  Setup complete.%s\n' "$GRN" "$RST"
[ "$DO_TOOLOMICS" = 1 ]    && info "Toolomics     → MCP tools on :$TOOLOMICS_PORT_MIN-$TOOLOMICS_PORT_MAX (Docker)"
[ "$DO_PERSPICACITE" = 1 ] && info "Perspicacité  → http://localhost:$PERSPICACITE_PORT"

if [ "$LAUNCH_WEBUI" = 0 ]; then
  step "Web UI launch skipped (--no-webui)"
  info "Start it later with: cd $MIMOSA_DIR/webui && ./deploy.sh"
  exit 0
fi

step "Launching the Observatory web UI"
info "Backend  → http://127.0.0.1:$BACKEND_PORT"
info "Frontend → http://localhost:$FRONTEND_PORT   (open this)"
info "Press Ctrl-C to stop the web UI (Toolomics & Perspicacité keep running)."

# The web UI's launch/refine bridge shells into Mimosa via this interpreter.
export MIMOSA_ROOT="$MIMOSA_DIR"
export MIMOSA_PYTHON="$MIMOSA_DIR/.venv/bin/python"
export MIMOSA_BACKEND_PORT="$BACKEND_PORT"
export MIMOSA_FRONTEND_PORT="$FRONTEND_PORT"

# Open the browser once the frontend is actually serving.
if [ "$OPEN_BROWSER" = 1 ] && have open; then
  ( wait_for_port "$FRONTEND_PORT" "frontend" 120 >/dev/null 2>&1 \
      && open "http://localhost:$FRONTEND_PORT" ) &
fi

[ -x "$MIMOSA_DIR/webui/deploy.sh" ] || chmod +x "$MIMOSA_DIR/webui/deploy.sh" 2>/dev/null || true
cd "$MIMOSA_DIR/webui"
exec ./deploy.sh
