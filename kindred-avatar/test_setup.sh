#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

echo "🧪 Testing Kindred Avatar Voice Setup"
echo ""

# Test 1: Check if venv exists
echo "1️⃣ Checking Python virtual environment..."
if [ -x "$PYTHON_BIN" ]; then
    echo "   ✅ Canonical environment found: $PYTHON_BIN"
else
    echo "   ❌ Canonical environment not found: $PYTHON_BIN"
    echo "   Follow $PROJECT_ROOT/how_to_run.md to create it."
    exit 1
fi

# Test 2: Check if packages are installed
echo ""
echo "2️⃣ Checking Python packages..."
cd "$SCRIPT_DIR/backend"
"$PYTHON_BIN" << 'EOF'
import sys
errors = []

try:
    import whisperlivekit
    print("   ✅ whisperlivekit")
except ImportError as e:
    print(f"   ❌ whisperlivekit: {e}")
    errors.append("whisperlivekit")

try:
    import websockets
    print("   ✅ websockets")
except ImportError as e:
    print(f"   ❌ websockets: {e}")
    errors.append("websockets")

try:
    import requests
    print("   ✅ requests")
except ImportError as e:
    print(f"   ❌ requests: {e}")
    errors.append("requests")

if errors:
    print(f"\n   Installing missing packages: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n   All packages installed!")
    sys.exit(0)
EOF

# Test 3: Check if Ollama is running
echo ""
echo "3️⃣ Checking Ollama..."
if pgrep -x "ollama" > /dev/null; then
    echo "   ✅ Ollama is running"
else
    echo "   ⚠️  Ollama not running - start it with: ollama serve"
fi

# Test 4: Check if 3D model exists
echo ""
echo "4️⃣ Checking 3D model..."
cd "$SCRIPT_DIR/frontend"
if [ -f "robot.gltf" ]; then
    echo "   ✅ robot.gltf found"
else
    echo "   ⚠️  robot.gltf not found - will use sphere fallback"
fi

echo ""
echo "🎉 Setup check complete!"
echo ""
echo "To start the server:"
echo "   ./start_voice.sh"
echo ""
echo "Then open: http://localhost:8070/index_voice.html"
