#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "Missing canonical Python environment: $PYTHON_BIN" >&2
    echo "Follow $PROJECT_ROOT/how_to_run.md to create it." >&2
    exit 1
fi

echo "🌟 Starting Jazzy Avatar with Voice Support..."
echo ""

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Warning: Ollama doesn't appear to be running"
    echo "   Start it with: ollama serve"
    echo ""
fi

# Start backend with voice support
echo "🔧 Starting WebSocket server with WhisperLiveKit..."
cd "$SCRIPT_DIR/backend"
"$PYTHON_BIN" server_voice.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Start frontend server
echo "🎨 Starting frontend server..."
cd "$SCRIPT_DIR/frontend"
"$PYTHON_BIN" -m http.server 8070 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ Jazzy Avatar with Voice is ready!"
echo ""
echo "📍 Open in your browser: http://localhost:8070/index_voice.html"
echo "🔌 WebSocket server: ws://localhost:8075"
echo ""
echo "Features:"
echo "  🎤 Click 'Voice' button to speak to Jazzy"
echo "  💬 Or type messages in the chat"
echo "  🎨 Click 'Image' to generate with ComfyUI (coming soon)"
echo "  🤖 3D robot model with animations"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 Jazzy Avatar with Voice stopped'; exit" INT

wait
