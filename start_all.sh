#!/usr/bin/env zsh
set -euo pipefail

# start_all.sh — start ComfyUI, Jazzy voice backend, and the avatar frontend
# Usage:
#   chmod +x start_all.sh
#   ./start_all.sh

PROJECT_ROOT="${PROJECT_ROOT:-${0:A:h}}"
COMFY_DIR="${COMFY_DIR:-$PROJECT_ROOT/ComfyUI-source}"
APP_DIR="${APP_DIR:-$PROJECT_ROOT/kindred-avatar}"
BACKEND_DIR="${BACKEND_DIR:-$APP_DIR}"
FRONTEND_DIR="${FRONTEND_DIR:-$APP_DIR/frontend}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"

COMFY_LOG="$LOG_DIR/comfyui-startup.log"
BACKEND_LOG="$LOG_DIR/jazzy-backend.log"
FRONTEND_LOG="$LOG_DIR/jazzy-frontend.log"

ensure_port_free() {
  lsof -ti:"$1" | xargs kill -9 2>/dev/null || true
}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[start_all] Missing canonical Python environment: $PYTHON_BIN" >&2
  echo "[start_all] Follow $PROJECT_ROOT/how_to_run.md to create it." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "[start_all] Starting Jazzy stack..."

# Start ComfyUI
if [ -d "$COMFY_DIR" ]; then
  echo "[start_all] Starting ComfyUI from: $COMFY_DIR"
  ensure_port_free 8188
  nohup "$PYTHON_BIN" "$COMFY_DIR/main.py" --listen 127.0.0.1 --port 8188 > "$COMFY_LOG" 2>&1 &
  echo "[start_all] ComfyUI launched (logs -> $COMFY_LOG)"
else
  echo "[start_all] ComfyUI directory not found at $COMFY_DIR — skipping ComfyUI"
fi

# Start backend (server_voice.py)
SERVER_PY=""
if [ -f "$BACKEND_DIR/backend/server_voice.py" ]; then
  SERVER_PY="$BACKEND_DIR/backend/server_voice.py"
else
  SERVER_PY=$(find "$BACKEND_DIR" -maxdepth 4 -type f -name server_voice.py -print -quit || true)
fi

if [ -n "$SERVER_PY" ]; then
  echo "[start_all] Starting backend server: $SERVER_PY"
  ensure_port_free 8075
  nohup "$PYTHON_BIN" "$SERVER_PY" > "$BACKEND_LOG" 2>&1 &
  echo "[start_all] Backend launched (logs -> $BACKEND_LOG)"
else
  echo "[start_all] server_voice.py not found in $BACKEND_DIR — skipping backend"
fi

# Serve frontend (simple static server)
if [ -d "$FRONTEND_DIR" ]; then
  echo "[start_all] Serving frontend from: $FRONTEND_DIR on port 8070"
  ensure_port_free 8070
  (cd "$FRONTEND_DIR" && nohup "$PYTHON_BIN" -m http.server 8070 > "$FRONTEND_LOG" 2>&1 &)
  echo "[start_all] Frontend available at http://127.0.0.1:8070 (logs -> $FRONTEND_LOG)"
else
  echo "[start_all] Frontend dir not found at $FRONTEND_DIR — skipping frontend"
fi

cat <<'EOF'

Optional local-model examples / notes:

- Ollama (if installed locally):
    # run Ollama server (if you use Ollama for LLMs)
    # ollama serve &

- Local LLM HTTP server placeholder (example environment variable):
    # export OLLAMA_HOST=http://127.0.0.1:11434
    # export OLLAMA_MODEL=ggml-jazzy-7b

- WhisperLiveKit / local transcription:
    # Start WhisperLiveKit according to its README if you run transcription locally

Remember to install the local runtimes you plan to use and set any API_HOST/API_KEY env vars
before starting the stack if you want the backend to connect to them automatically.

EOF

echo "[start_all] Jazzy UI:      http://127.0.0.1:8070"
echo "[start_all] Jazzy backend: ws://127.0.0.1:8075"
echo "[start_all] ComfyUI:       http://127.0.0.1:8188"

echo "[start_all] Done — check logs:"
echo "  ComfyUI:  $COMFY_LOG"
echo "  Backend:  $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"

echo "[start_all] To make this script executable: chmod +x start_all.sh"
echo "[start_all] Then run: ./start_all.sh"

exit 0
