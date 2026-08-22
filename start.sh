#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
ENV_FILE="$PROJECT_DIR/.env"
HOST="${ADA_HOST:-0.0.0.0}"
PORT="${ADA_PORT:-8000}"
VIDEO_MODE="${ADA_VIDEO_MODE:-activity}"
KIOSK_URL="http://localhost:$PORT"
CHROMIUM_PROFILE="${ADA_CHROMIUM_PROFILE:-$PROJECT_DIR/.chromium-kiosk}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "ADA Pi virtual environment is missing." >&2
  echo "Run: python3 -m venv '$PROJECT_DIR/.venv'" >&2
  echo "Then: '$PROJECT_DIR/.venv/bin/python' -m pip install -r '$PROJECT_DIR/backend/requirements.txt'" >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ADA Pi configuration is missing: $ENV_FILE" >&2
  echo "Copy .env.example to .env and add GEMINI_API_KEY." >&2
  exit 1
fi

if command -v chromium >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium)"
elif command -v chromium-browser >/dev/null 2>&1; then
  CHROMIUM_BIN="$(command -v chromium-browser)"
else
  echo "Chromium is not installed or is not on PATH." >&2
  exit 1
fi

cd "$PROJECT_DIR"
ADA_VIDEO_MODE="$VIDEO_MODE" "$VENV_PYTHON" -m uvicorn backend.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --env-file "$ENV_FILE" &
SERVER_PID=$!
BROWSER_PID=""

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$BROWSER_PID" ]] && kill -0 "$BROWSER_PID" 2>/dev/null; then
    kill "$BROWSER_PID" 2>/dev/null || true
  fi
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
  [[ -z "$BROWSER_PID" ]] || wait "$BROWSER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Do not open Chromium until the backend is accepting requests.
READY=false
for _ in {1..100}; do
  if "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlopen('$KIOSK_URL', timeout=.2)" >/dev/null 2>&1; then
    READY=true
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ADA Pi backend stopped before Chromium could start." >&2
    exit 1
  fi
  sleep .1
done
if [[ "$READY" != true ]]; then
  echo "Timed out waiting for ADA Pi backend at $KIOSK_URL." >&2
  exit 1
fi

"$CHROMIUM_BIN" \
  --kiosk \
  --no-first-run \
  --disable-session-crashed-bubble \
  --autoplay-policy=no-user-gesture-required \
  --user-data-dir="$CHROMIUM_PROFILE" \
  "$KIOSK_URL" &
BROWSER_PID=$!

# Closing either side closes the other. This also handles the on-screen Exit
# button, which asks the backend to stop and lets this supervisor close Chromium.
set +e
wait -n "$SERVER_PID" "$BROWSER_PID"
STATUS=$?
set -e
exit "$STATUS"
