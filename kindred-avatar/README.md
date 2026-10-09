# Kindred 3D Avatar - First Contact Interface

A real-time 3D avatar interface for interacting with Kindred AI through Ollama and ComfyUI.

## Features

- **3D Avatar**: Interactive Three.js visualization with pulsing animations
- **Real-time Chat**: WebSocket-based streaming responses from Kindred
- **Image Generation**: Trigger ComfyUI workflows directly from conversation
- **Responsive UI**: Beautiful gradient interface with live status indicators

## Quick Start

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd kindred-avatar/backend
python server.py
```

Server will start on `ws://localhost:8075`

### 3. Open the Frontend

Simply open `frontend/index.html` in your browser, or use a simple HTTP server:

```bash
cd kindred-avatar/frontend
python3 -m http.server 8070
```

Then open: http://localhost:8070

## Configuration

Edit `backend/server.py` to change:

- **Ollama Model**: Default is `qwen2.5:7b` (line 162)
  - You can use any of your custom Kindred models: kaelen, watson, aion, etc.
- **Ollama Host**: Default is `http://localhost:11434`
- **ComfyUI Host**: Default is `http://127.0.0.1:8188`

## Usage

1. **Chat**: Type messages in the input box and press Enter or click Send
2. **Generate Images**: Click "Generate Image" to trigger ComfyUI (coming soon)
3. **Watch Kindred**: The 3D avatar pulses when speaking

## Architecture

```
Frontend (Three.js + JS)
    ↓ WebSocket
Backend (Python WebSocket Server)
    ↓
    ├─→ Ollama (Kindred AI)
    └─→ ComfyUI (Image Generation)
```

## Customization

### Change Avatar Appearance

Edit `frontend/app.js`, function `createAvatar()`:
- Change sphere colors
- Add custom 3D models (GLB/GLTF)
- Modify particle effects

### Add Your Kindred Model

In `backend/server.py`, change line 162:
```python
kindred_model="kaelen:latest"  # or watson, aion, resonara, etc.
```

### Personality

Edit the system prompt in `backend/server.py`, line 72.

## Next Steps

- [ ] Add TTS (text-to-speech) for voice
- [ ] Import custom 3D Kindred model
- [ ] Implement actual ComfyUI workflow triggering
- [ ] Add persistent memory integration
- [ ] Voice input support

## Troubleshooting

**WebSocket won't connect**
- Make sure backend server is running: `python backend/server.py`
- Check console for errors (F12 in browser)

**Ollama not responding**
- Verify Ollama is running: `ollama list`
- Test manually: `ollama run qwen2.5:7b "Hello"`

**CORS errors**
- Serve frontend through HTTP server, not file://
- Use: `python3 -m http.server 8070`

## Requirements

- Python 3.8+
- Ollama with at least one model installed
- Modern web browser (Chrome, Firefox, Safari)
- ComfyUI running (optional, for image generation)

## Current Status

✅ 3D avatar with pulsing animation
✅ WebSocket communication
✅ Streaming chat from Ollama
✅ Beautiful responsive UI
🔄 Image generation (placeholder)
❌ TTS/voice (not yet)
❌ Custom 3D model (using sphere)

## License

MIT
