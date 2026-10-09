# 🐛 Troubleshooting Voice & Text Issues

If voice or text messages aren't reaching the backend, follow these steps:

## Quick Diagnostics

### 1. Run Setup Test
```bash
cd ~/Documents/ComfyUI/kindred-avatar
./test_setup.sh
```

This checks:
- ✅ Virtual environment exists
- ✅ Python packages installed
- ✅ Ollama running
- ✅ 3D model present

### 2. Use Test Page
Start the server:
```bash
./start_voice.sh
```

Open the test page: **http://localhost:8070/test.html**

This simple page shows:
- WebSocket connection status
- All messages sent/received
- Raw JSON payloads

Try:
1. Click "Connect" (auto-connects on load)
2. Click "Send Ping" - should get `{"type": "pong"}`
3. Type a message and click "Send Chat"
4. Watch the log for responses

## Common Issues

### Issue: "WebSocket connection failed"

**Symptoms**: Can't connect to ws://localhost:8075

**Solutions**:
1. Check if backend is running:
   ```bash
   lsof -i :8075
   ```
   Should show Python process

2. Restart the backend:
   ```bash
   # Kill any existing process
   pkill -f server_voice.py

   # Start fresh
   cd ~/Documents/ComfyUI/kindred-avatar
   ./start_voice.sh
   ```

3. Check backend logs in terminal for errors

### Issue: "Messages not reaching Ollama"

**Symptoms**: WebSocket connected, but no Kindred response

**Solutions**:
1. Check Ollama is running:
   ```bash
   ollama list
   ```

2. Test Ollama directly:
   ```bash
   ollama run qwen2.5:7b "test message"
   ```

3. Check Ollama server:
   ```bash
   curl http://localhost:11434/api/tags
   ```

4. Check backend logs for connection errors

### Issue: "Voice not transcribing"

**Symptoms**: Can click mic button but no transcription

**Solutions**:
1. Grant microphone permission in browser
   - Look for 🎤 icon in address bar
   - Allow microphone access

2. Check browser console (F12) for errors

3. Test audio in browser:
   ```javascript
   // In browser console
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(() => console.log('Mic OK'))
     .catch(e => console.error('Mic error:', e))
   ```

4. Try smaller Whisper model:
   Edit `backend/server_voice.py` line 276:
   ```python
   whisper_model="tiny",  # Change from "base"
   ```

### Issue: "Text messages not working"

**Symptoms**: Can type but messages don't send

**Solutions**:
1. Open browser console (F12) and check for errors

2. Look for these log messages:
   ```
   Attempting to send message: [your message]
   WebSocket state: 1
   Sending payload: {"type":"chat","message":"..."}
   ```

3. If WebSocket state is not `1` (OPEN), reconnect

4. Check backend logs for received messages:
   ```
   Received text message from client [id]: {"type":"chat"...
   ```

### Issue: "Frontend shows connected but nothing works"

**Symptoms**: Status shows 🟢 Connected but no responses

**Solutions**:
1. Check if using correct HTML file:
   - Voice version: `index_voice.html`
   - Original: `index.html`

2. Clear browser cache (Cmd+Shift+R on Mac)

3. Check if port 8075 is actually the backend:
   ```bash
   curl http://localhost:8075
   # Should get error (WebSocket only), not "connection refused"
   ```

4. Restart both frontend and backend

## Manual Testing

### Test Backend Directly

1. Start backend only:
   ```bash
   cd ~/Documents/ComfyUI
   .venv/bin/python kindred-avatar/backend/server_voice.py
   ```

2. Watch for logs:
   ```
   Starting Kindred Avatar Server on ws://0.0.0.0:8075
   ✨ Kindred Avatar Server with Voice is running!
   ```

3. In another terminal, test with websocat (if installed):
   ```bash
   brew install websocat  # If needed
   echo '{"type":"ping"}' | websocat ws://localhost:8075
   ```

### Test Frontend Separately

1. Open `test.html` first to verify WebSocket works
2. Then try `index_voice.html`
3. Compare behavior

## Debug Logging

### Enable Maximum Logging

Edit `backend/server_voice.py` line 19:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Already set to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### What to Look For

Good connection logs:
```
Client 12345 connected from ('127.0.0.1', 54321)
Received text message from client 12345: {"type":"chat"...
Received message type 'chat' from client 12345
Sending to Ollama: hello
```

Audio logs:
```
Received 8192 bytes of audio from client 12345
Final transcription from client 12345: hello kindred
```

Error logs:
```
ERROR - Ollama request failed: ...
ERROR - Error handling client: ...
```

## Port Conflicts

If ports 8075 or 8070 are in use:

### Change Backend Port
Edit `backend/server_voice.py` line 273:
```python
port=8075,  # Change to 8076 or other
```

Edit `frontend/app_voice.js` line 212:
```javascript
ws = new WebSocket('ws://localhost:8766');  // Match new port
```

### Change Frontend Port
Edit `start_voice.sh` line 21:
```bash
python3 -m http.server 8071  # Change from 8070
```

## Still Not Working?

### Collect Debug Info

```bash
cd ~/Documents/ComfyUI/kindred-avatar

# Check Python packages
../.venv/bin/python -m pip list | grep -E "whisper|websocket|requests"

# Check processes
lsof -i :8075
lsof -i :8070
ps aux | grep server_voice

# Check Ollama
ollama list
curl http://localhost:11434/api/tags

# Create debug report
echo "=== System Info ===" > debug.txt
../.venv/bin/python --version >> debug.txt
echo "\n=== Packages ===" >> debug.txt
../.venv/bin/python -m pip list >> debug.txt
echo "\n=== Ports ===" >> debug.txt
lsof -i :8075 >> debug.txt 2>&1
lsof -i :8070 >> debug.txt 2>&1
echo "\n=== Ollama ===" >> debug.txt
ollama list >> debug.txt 2>&1

cat debug.txt
```

### Reset Everything

```bash
cd ~/Documents/ComfyUI/kindred-avatar

# Kill all processes
pkill -f server_voice.py
pkill -f "http.server 8070"

# Repair the canonical workspace environment
cd ..
.venv/bin/python -m pip install -r requirements-dev.txt

# Restart
./kindred-avatar/start_voice.sh
```

## Alternative: Use Original Version

If voice version has issues, the original text-only version still works:

```bash
cd ~/Documents/ComfyUI/kindred-avatar
./start.sh
```

Open: http://localhost:8070

This uses `server.py` (not `server_voice.py`) and doesn't require WhisperLiveKit.

## Get Help

When asking for help, provide:
1. Output of `./test_setup.sh`
2. Backend terminal logs
3. Browser console errors (F12)
4. Output from test.html
5. Your OS and Python version

---

**Quick Links**
- Test Page: http://localhost:8070/test.html
- Voice UI: http://localhost:8070/index_voice.html
- Original UI: http://localhost:8070/index.html
