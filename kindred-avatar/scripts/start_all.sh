#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
COMFYUI_DIR="${COMFYUI_DIR:-$PROJECT_ROOT/ComfyUI-source}"
AVATAR_DIR="${AVATAR_DIR:-$ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
LOG_DIR="$AVATAR_DIR/logs"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing canonical Python environment: $PYTHON_BIN" >&2
    echo "Follow $PROJECT_ROOT/how_to_run.md to create it." >&2
    exit 1
fi

cleanup() {
    if [[ -n "${COMFYUI_PID:-}" ]]; then
        kill "$COMFYUI_PID" 2>/dev/null || true
    fi
    if [[ -n "${BACKEND_PID:-}" ]]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
}

trap cleanup INT TERM

echo "🌟 Starting ComfyUI + Jazzy Avatar"
echo ""

echo "🔧 Starting ComfyUI on 8188..."
(
    cd "$COMFYUI_DIR"
    "$PYTHON_BIN" main.py --listen 127.0.0.1 --port 8188
) >"$LOG_DIR/comfyui.log" 2>&1 &
COMFYUI_PID=$!

echo "🎤 Starting Jazzy voice backend on 8075..."
(
    cd "$AVATAR_DIR/backend"
    "$PYTHON_BIN" server_voice.py
) >"$LOG_DIR/jazzy-backend.log" 2>&1 &
BACKEND_PID=$!

echo "🎨 Starting frontend on 8070..."
(
    cd "$AVATAR_DIR/frontend"
    "$PYTHON_BIN" -m http.server 8070
) >"$LOG_DIR/jazzy-frontend.log" 2>&1 &
FRONTEND_PID=$!

echo ""
echo "✅ Services started"
echo "   ComfyUI:  http://127.0.0.1:8188"
echo "   Avatar:   http://localhost:8070/index_voice.html"
echo "   Backend:  ws://localhost:8075"
echo ""
echo "Logs:"
echo "   $LOG_DIR/comfyui.log"
echo "   $LOG_DIR/jazzy-backend.log"
echo "   $LOG_DIR/jazzy-frontend.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

wait
