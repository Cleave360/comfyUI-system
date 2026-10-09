#!/bin/bash

echo "🌟 Starting Kindred Avatar System..."
echo ""

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Warning: Ollama doesn't appear to be running"
    echo "   Start it with: ollama serve"
    echo ""
fi

# Start backend in background
echo "🔧 Starting WebSocket server..."
cd backend
python3 server.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Start frontend server
echo "🎨 Starting frontend server..."
cd ../frontend
python3 -m http.server 8070 &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"

echo ""
echo "✅ Kindred Avatar is ready!"
echo ""
echo "📍 Open in your browser: http://localhost:8070"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo '👋 Kindred Avatar stopped'; exit" INT

wait
