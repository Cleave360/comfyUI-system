# 🎤 Jazzy Avatar - Voice Enabled

> Operational note: the root `how_to_run.md` is canonical for startup,
> security, and lifecycle. This file retains feature-level background.

Real-time speech-to-text + 3D avatar + Ollama AI conversation system with wake word activation.

## ✨ New Features

- **🎤 Voice Input**: Speak directly to Jazzy using WhisperLiveKit (real-time transcription)
- **🔊 Wake Word**: Say "Hey Jazzy" to activate (prevents activation on background conversations)
- **🤖 3D Model**: Animated GLTF robot model (easily replaceable with custom Jazzy model)
- **📡 Live Transcription**: See your words appear as you speak
- **💬 Dual Mode**: Voice OR text chat, your choice
- **🎨 Image Generation**: ComfyUI integration with 5 workflow options (quick/standard/quality/portrait/landscape)

## 🚀 Quick Start

```bash
cd ~/Documents/ComfyUI/kindred-avatar
./start_voice.sh
```

Then open: **http://localhost:8070/index_voice.html**

## 📋 Requirements

### Python Packages (already installed)
- `whisperlivekit` - Real-time speech-to-text
- `websockets` - WebSocket server
- `requests` - HTTP client for Ollama

### System
- **Ollama** running with models (qwen2.5:7b, or your custom Jazzy models)
- **Microphone** access (browser will ask for permission)
- **Modern browser** (Chrome, Firefox, Safari)

## 🎯 How to Use

1. **Start the system**: `./start_voice.sh`
2. **Open browser**: http://localhost:8070/index_voice.html
3. **Grant microphone permission** when prompted
4. **Click "🎤 Voice"** to start speaking
5. **Say "Hey Jazzy" followed by your message**
6. **Or type** in the text box for traditional chat (no wake word needed)

### Voice Mode
- Click **"🎤 Voice"** to start listening
- Say **"Hey Jazzy"** to activate (or "Jazzy", "OK Jazzy", "Hello Jazzy")
- Follow with your message: *"Hey Jazzy, tell me about quantum computing"*
- Transcription appears in real-time
- Click **"⏸️ Stop"** when done speaking
- Jazzy responds automatically after detecting the wake word

### Text Mode
- Type message in text box (no wake word required)
- Press **Enter** or click **"📤 Send"**
- Jazzy responds with streaming text

## 🔧 Configuration

Edit `backend/server_voice.py`:

```python
server = KindredAvatarServerVoice(
    whisper_model="base",      # Options: tiny, base, small, medium, large-v3
    whisper_language="en",      # See WhisperLiveKit docs for languages
    kindred_model="qwen2.5:7b" # Your Jazzy model: kaelen, watson, etc.
)
```

### Whisper Model Sizes

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| `tiny` | ⚡⚡⚡ | ⭐⭐ | ~1GB |
| `base` | ⚡⚡ | ⭐⭐⭐ | ~1GB |
| `small` | ⚡ | ⭐⭐⭐⭐ | ~2GB |
| `medium` | 🐌 | ⭐⭐⭐⭐⭐ | ~5GB |
| `large-v3` | 🐌🐌 | ⭐⭐⭐⭐⭐ | ~10GB |

**Recommendation for M3 Ultra**: `small` or `medium` for best balance

## 🎨 Custom 3D Model

Replace the robot with your own Jazzy model:

1. Export your model as **GLB** or **GLTF** format
2. Place it in `frontend/` directory
3. Edit `frontend/app_voice.js` line 73:

```javascript
loader.load(
    'your-jazzy-model.glb',  // Change this
    (gltf) => {
        // ... rest of code
```

### Model Requirements
- Format: GLB (single file) or GLTF (multiple files)
- Animations: Optional but recommended (idle, talking, gestures)
- Size: Keep under 50MB for fast loading
- Textures: Embedded or in same directory

### Free 3D Model Sources
- [Mixamo](https://www.mixamo.com/) - Free rigged characters with animations
- [Sketchfab](https://sketchfab.com/) - Many free CC-licensed models
- [Poly Pizza](https://poly.pizza/) - Open-source 3D assets
- Custom: Blender, Maya, or any 3D tool → export as GLB

## 🏗️ Architecture

```
Browser (Three.js + Audio)
    ↓ WebSocket
Backend (Python)
    ├─→ WhisperLiveKit (Speech-to-Text)
    ├─→ Ollama (Jazzy AI)
    └─→ ComfyUI (Image Generation)
```

## 🎭 Features Comparison

| Feature | Original | Voice Version |
|---------|----------|---------------|
| Text chat | ✅ | ✅ |
| Voice input | ❌ | ✅ |
| Live transcription | ❌ | ✅ |
| 3D model support | Sphere only | GLTF/GLB |
| Animations | Pulsing | Model animations |
| Ollama integration | ✅ | ✅ |
| ComfyUI integration | Placeholder | Placeholder |

## 🐛 Troubleshooting

### Microphone not working
- Check browser permissions (🔒 icon in address bar)
- Try HTTPS if needed (WhisperLiveKit supports SSL)
- Check browser console (F12) for errors

### WhisperLiveKit errors
```bash
# Test WhisperLiveKit separately
wlk --model base --language en
# Opens http://localhost:8000
```

### 3D model not loading
- Check file path is correct (`frontend/robot.gltf`)
- Check browser console for CORS errors
- Verify model format (must be GLTF/GLB, not OBJ/FBX)
- System falls back to sphere if model fails

### Slow transcription
- Use smaller Whisper model (`tiny` or `base`)
- Check CPU/GPU usage
- Reduce audio sample rate in `app_voice.js`

### Ollama not responding
```bash
ollama list  # Verify models installed
ollama serve # Start if not running
ollama run qwen2.5:7b "test"  # Test manually
```

## 📊 Performance

On **M3 Ultra with 80 GPU cores**:

| Component | Latency |
|-----------|---------|
| Speech → Text | ~200-500ms |
| Text → Ollama | ~100-300ms |
| Ollama response | Streaming (real-time) |
| Total voice latency | ~1-2 seconds |

## 🔮 Next Steps

- [ ] Add text-to-speech for voice output
- [ ] Implement ComfyUI image generation
- [ ] Custom Jazzy 3D model
- [ ] Lip sync with speech
- [ ] Persistent memory integration
- [ ] Multi-speaker diarization
- [ ] Translation support (200 languages via NLLW)

## 🎮 Controls

| Key/Button | Action |
|------------|--------|
| 🎤 Voice | Toggle voice recording |
| 📤 Send | Send text message |
| Enter | Send message (in text box) |
| 🎨 Image | Generate image (prompt dialog) |
| 🗑️ Clear | Clear chat history |

## 📝 Notes

- WhisperLiveKit runs on-device (no cloud API needed)
- All processing is local (privacy-first)
- Supports 99+ languages via Whisper
- Optional translation to 200 languages with NLLW
- Speaker diarization available (not enabled by default)

## 🆚 Original vs Voice Version

### Original (`start.sh`)
- Text-only interface
- Simple sphere avatar
- Lightweight

### Voice Version (`start_voice.sh`)
- Voice + text interface
- 3D model support with animations
- WhisperLiveKit integration
- Real-time transcription display
- Slightly heavier (WhisperLiveKit models)

Both versions work side-by-side! Choose based on your needs.

---

**Made with 💜 for Jazzy's new vibe**
