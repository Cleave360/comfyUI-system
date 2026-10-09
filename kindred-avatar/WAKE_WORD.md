# Wake Word Configuration

## Overview
Kindred now uses wake word detection to prevent activation from background conversations. This is especially useful in multi-user or noisy environments.

## Wake Words
The following phrases will activate Kindred:
- **"Kindred"**
- **"Hey Kindred"**
- **"OK Kindred"**
- **"Hello Kindred"**

The wake word is case-insensitive and will be automatically removed from your message.

## How It Works

### With Wake Word (Voice Mode)
```
You say: "Hey Kindred, what's the weather like?"
Kindred hears: "what's the weather like?"
Status: ✅ ACTIVATED
```

### Without Wake Word (Voice Mode)
```
You say: "I think we should go to the store"
Kindred: [Ignores - no wake word detected]
Status: ⏭️ IGNORED
```

### Text Mode
Wake words are **NOT required** in text chat mode - type your message normally.

## Configuration

### Enable/Disable Wake Word
Edit `backend/server_voice.py` around line 62:

```python
# Wake word configuration
self.wake_words = ["kindred", "hey kindred", "ok kindred", "hello kindred"]
self.require_wake_word = True  # Set to False to disable wake word requirement
```

To disable wake word detection entirely:
```python
self.require_wake_word = False
```

### Add Custom Wake Words
```python
self.wake_words = [
    "kindred",
    "hey kindred",
    "ok kindred",
    "hello kindred",
    "yo kindred",           # Add custom
    "assistant",            # Add custom
    "computer"              # Add custom
]
```

### Change Wake Word Sensitivity
The wake word detection uses simple substring matching. The word/phrase just needs to appear anywhere in the transcription.

For more sophisticated detection (e.g., requiring wake word at the start):
```python
# In handle_transcription_results(), replace:
has_wake_word = any(wake_word in text_lower for wake_word in self.wake_words)

# With:
has_wake_word = any(text_lower.startswith(wake_word) for wake_word in self.wake_words)
```

## Visual Indicators

### Frontend Hint
The UI displays: **"💬 Say 'Hey Kindred' to activate"**

This hint is always visible in the top-left corner when using voice mode.

### Console Logs
When wake word is detected:
```
🎤 Wake word detected! Message: 'what's the weather like?'
```

When speech is ignored (no wake word):
```
⏭️ Ignoring speech without wake word: 'random conversation'
```

## Use Cases

### ✅ Good Use Cases for Wake Word
- **Multi-user environments** - Prevents accidental activation
- **Open mic scenarios** - Only responds when explicitly called
- **Background conversations** - Won't interrupt when people are talking nearby
- **Shared spaces** - Multiple people can be present without triggering

### ❌ When to Disable Wake Word
- **Private/quiet environments** - You're the only user
- **Accessibility needs** - Wake word might be difficult to say
- **Quick testing** - Faster iteration without saying wake word
- **Push-to-talk mode** - Button press already gates input

## Examples

### Correct Usage
```
"Hey Kindred, tell me a joke"
"Kindred, what time is it?"
"OK Kindred, generate an image of a sunset"
"Hello Kindred, how are you today?"
```

### Will Be Ignored
```
"I was thinking about kindred spirits" (not addressing the AI)
"What should we do today?"
"The weather is nice"
```

## Troubleshooting

### Wake Word Not Detected
**Problem**: You're saying the wake word but Kindred doesn't respond

**Solutions**:
1. Check console logs - is transcription working?
2. Try saying the wake word more clearly/loudly
3. Check `self.require_wake_word = True` in server_voice.py
4. Verify wake word is in `self.wake_words` list
5. Test with text chat to confirm Ollama is responding

### False Activations
**Problem**: Kindred activates when you don't want it to

**Solutions**:
1. Use more unique wake phrases: "yo kindred assistant"
2. Require wake word at start of sentence (see config above)
3. Increase silence detection threshold (min_chunk_size)
4. Add voice authentication (advanced - not yet implemented)

### Wake Word Removed Incorrectly
**Problem**: Part of your message gets removed along with wake word

**Example**: "Hey Kindred, hello world" → "world" (loses "hello")

**Solution**:
This happens because "hello kindred" is also a wake word. Use:
```python
# Sort wake words by length (longest first) to remove the most specific match
self.wake_words = sorted(
    ["kindred", "hey kindred", "ok kindred", "hello kindred"],
    key=len,
    reverse=True
)
```

## Advanced Features

### Per-User Wake Words (Future)
```python
# Store user-specific wake words
client_data["wake_words"] = ["kindred", "hey kindred"]
```

### Voice Fingerprinting (Future)
Combine wake word with voice recognition to only respond to authorized users.

### Dynamic Wake Words (Future)
Allow users to set custom wake words via UI.

### Wake Word Confirmation (Future)
Visual/audio feedback when wake word is detected:
```python
await websocket.send(json.dumps({
    "type": "wake_word_detected",
    "timestamp": time.time()
}))
```

## Performance Impact

- **Overhead**: Minimal (~5ms per transcription)
- **False Positives**: Rare with unique wake words
- **False Negatives**: Can occur with unclear audio
- **Latency**: No additional delay beyond transcription

## Comparison with Alternatives

### Push-to-Talk Button
- ✅ No false activations
- ✅ Clear activation point
- ❌ Requires button press (less natural)
- ❌ Not hands-free

### Wake Word (Current Implementation)
- ✅ Hands-free operation
- ✅ Natural conversation flow
- ⚠️ Possible false activations
- ⚠️ Requires saying wake word

### Continuous Listening (No Gate)
- ✅ Most natural (no trigger needed)
- ❌ High false activation rate
- ❌ Privacy concerns
- ❌ Not suitable for shared spaces

## References

- Voice assistants: Alexa ("Alexa"), Google ("Hey Google"), Siri ("Hey Siri")
- Wake word detection libraries: Porcupine, Snowboy (deprecated)
- Voice activity detection: WhisperLiveKit VAD, WebRTC VAD
