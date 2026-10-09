# 🎤 Voice Not Working? Quick Fixes

Based on your test results, here are the most common issues and solutions:

## ✅ What's Working
- Microphone permission: Working
- Audio capture: Working
- WebSocket connection: Working
- Backend: Running correctly

## ❌ Issue: Full Voice Pipeline

The audio is being sent to the backend, but WhisperLiveKit might not be transcribing. Here are the solutions:

### Fix 1: Restart with Fresh Server

Sometimes WhisperLiveKit needs a clean restart:

```bash
# Kill everything
pkill -f server_voice.py
pkill -f "http.server"

# Wait 2 seconds
sleep 2

# Start fresh
cd ~/Documents/ComfyUI/kindred-avatar
./start_voice.sh
```

Then test again at: http://localhost:8070/voice_test.html

### Fix 2: Check Backend Terminal

Look for these messages in the terminal where server is running:

**Good signs:**
```
Received audio chunk #10 (8192 bytes) from client...
Started transcription handler for client...
Transcription response for client...
Final transcription from client: [your words]
```

**Bad signs:**
```
Error processing audio chunk...
Error in transcription handler...
```

If you see errors, that tells us what's wrong.

### Fix 3: Speak Longer and Louder

WhisperLiveKit needs a few seconds of audio to transcribe:

1. Click "Start Voice Test"
2. Wait for "Recording - speak now!"
3. **Speak clearly for 5-10 seconds** (not just 1-2 seconds)
4. Example: "Hello Kindred, this is a test of the voice recognition system, can you hear me?"
5. Keep speaking until you see transcription appear

### Fix 4: Try Smaller Model

The `base` model might be slow. Try `tiny` for faster response:

Edit `backend/server_voice.py` line 276:
```python
whisper_model="tiny",  # Change from "base"
```

Restart server.

### Fix 5: Check System Audio Settings

macOS sometimes mutes input:
1. System Settings → Sound → Input
2. Make sure correct microphone is selected
3. Check input level meter moves when you speak
4. Increase input volume if needed

### Fix 6: Use Different Browser

Safari sometimes has audio issues. Try:
- Chrome (best for WebAudio)
- Firefox
- Edge

### Fix 7: Manual Test

Test WhisperLiveKit directly:

```bash
# In another terminal
cd ~/Documents/ComfyUI
.venv/bin/wlk --model tiny --language en
```

Then open http://localhost:8000 and test voice there. If it works there but not in your app, it's a WebSocket issue.

## Debug Checklist

Run through voice_test.html and check:

- [ ] Test 1: ✅ Microphone permission granted
- [ ] Test 2: Audio level shows > 5% when speaking
- [ ] Test 2: Waveform visualization moves when speaking
- [ ] Test 3: ✅ WebSocket connects and gets pong
- [ ] Test 4: "Sent audio chunk" appears in log repeatedly
- [ ] Test 4: Backend terminal shows "Received audio chunk"
- [ ] Test 4: Transcription appears (even if wrong/partial)

## Expected Timeline

When working correctly:
1. Click "Start Voice Test" - immediate
2. Start speaking - audio chunks sent every ~100ms
3. After 2-3 seconds - first partial transcription
4. Stop speaking - final transcription within 1-2 seconds

If nothing appears after 10 seconds of speaking, there's an issue.

## Still Not Working?

Let me know:
1. What you see in browser console (F12)
2. What you see in backend terminal
3. How long you spoke
4. What the audio level % reached in Test 2

I can help debug further!
