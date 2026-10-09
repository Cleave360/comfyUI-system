// Jazzy Avatar Frontend - Three.js + WebSocket with Voice Support

// Three.js scene components
let scene, camera, renderer, avatar, light, mixer, clock;
let particles = [];
let isAnimating = false;
let animationScale = 1.0;
let isModelLoaded = false;

// WebSocket and audio
let ws;
let isRecording = false;
let mediaRecorder;
let audioContext;
let audioStream;
let processor;
let analyser;
let waveformDataArray;
let waveformCanvas;
let waveformCtx;
let waveformContainer;
let waveformAnimationId;
let waveformResizeAttached = false;

// State management
let currentState = 'idle'; // idle, listening, thinking, speaking

// Voice settings
let voiceSettings = {
    selectedVoice: null,
    pitch: 1.1,
    speed: 1.0
};
let availableVoices = [];

// Audio feedback
function playWakeWordBeep() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 800; // 800Hz tone
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.15);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.15);

    console.log('🔔 Wake word beep played');
}

// State indicator functions
function setState(newState) {
    currentState = newState;
    const indicator = document.getElementById('state-indicator');

    // Remove all state classes
    indicator.classList.remove('listening', 'thinking', 'speaking');

    // Update indicator based on state
    switch(newState) {
        case 'listening':
            indicator.textContent = '🎤';
            indicator.classList.add('listening');
            break;
        case 'thinking':
            indicator.textContent = '🤔';
            indicator.classList.add('thinking');
            break;
        case 'speaking':
            indicator.textContent = '💬';
            indicator.classList.add('speaking');
            break;
        case 'idle':
        default:
            indicator.textContent = '';
            indicator.style.opacity = '0';
            break;
    }

    console.log(`🔄 State changed: ${newState}`);
}

function ensureWaveformCanvas() {
    if (!waveformCanvas) {
        waveformCanvas = document.getElementById('waveform');
        waveformCtx = waveformCanvas?.getContext('2d');
        waveformContainer = document.getElementById('waveform-container');
    }
    if (!waveformResizeAttached && waveformCanvas) {
        resizeWaveformCanvas();
        window.addEventListener('resize', resizeWaveformCanvas);
        waveformResizeAttached = true;
    }
}

function resizeWaveformCanvas() {
    if (!waveformCanvas || !waveformCtx) return;
    const { width, height } = waveformCanvas.getBoundingClientRect();
    waveformCanvas.width = width || 0;
    waveformCanvas.height = height || 0;
}

function startWaveformAnimation() {
    ensureWaveformCanvas();
    if (!waveformCanvas || !waveformCtx || !analyser || !waveformDataArray) return;
    resizeWaveformCanvas();
    waveformContainer?.classList.add('active');
    const draw = () => {
        waveformAnimationId = requestAnimationFrame(draw);
        analyser.getByteTimeDomainData(waveformDataArray);
        waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
        waveformCtx.lineWidth = 2;
        waveformCtx.strokeStyle = '#aa88ff';
        waveformCtx.shadowColor = 'rgba(170, 136, 255, 0.6)';
        waveformCtx.shadowBlur = 10;
        waveformCtx.beginPath();
        const sliceWidth = waveformCanvas.width / waveformDataArray.length;
        let x = 0;
        for (let i = 0; i < waveformDataArray.length; i++) {
            const v = waveformDataArray[i] / 128.0;
            const y = (v * waveformCanvas.height) / 2;
            if (i === 0) {
                waveformCtx.moveTo(x, y);
            } else {
                waveformCtx.lineTo(x, y);
            }
            x += sliceWidth;
        }
        waveformCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
        waveformCtx.stroke();
    };
    draw();
}

function stopWaveformAnimation() {
    waveformContainer?.classList.remove('active');
    if (waveformAnimationId) {
        cancelAnimationFrame(waveformAnimationId);
        waveformAnimationId = null;
    }
    if (waveformCtx && waveformCanvas) {
        waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
}

// Initialize Three.js scene
function initThreeJS() {
    const canvas = document.querySelector('#avatar-canvas');

    // Scene
    scene = new THREE.Scene();
    scene.background = null;

    // Camera
    camera = new THREE.PerspectiveCamera(
        50,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 1.5, 5);
    camera.lookAt(0, 1, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true
    });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x8866ff, 0.5);
    scene.add(ambientLight);

    light = new THREE.PointLight(0xaa88ff, 1, 100);
    light.position.set(0, 3, 5);
    light.castShadow = true;
    scene.add(light);

    // Clock for animations
    clock = new THREE.Clock();

    // Try to load 3D model, fallback to sphere
    loadAvatarModel();

    // Create particle system
    createParticles();

    // Animation loop
    animate();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);
}

function loadAvatarModel() {
    const loader = new THREE.GLTFLoader();

    loader.load(
        'robot.gltf', // Change this to your Jazzy model
        (gltf) => {
            avatar = gltf.scene;
            avatar.position.set(0, 0, 0);
            avatar.scale.set(1.5, 1.5, 1.5);

            // Setup animations if available
            if (gltf.animations && gltf.animations.length > 0) {
                mixer = new THREE.AnimationMixer(avatar);
                // Play idle animation (usually first one)
                const action = mixer.clipAction(gltf.animations[0]);
                action.play();
            }

            // Enable shadows
            avatar.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });

            scene.add(avatar);
            isModelLoaded = true;
            console.log('✅ 3D model loaded successfully');
        },
        (progress) => {
            console.log('Loading model...', (progress.loaded / progress.total * 100).toFixed(2) + '%');
        },
        (error) => {
            console.warn('Failed to load 3D model, using sphere fallback:', error);
            createFallbackAvatar();
        }
    );
}

function createFallbackAvatar() {
    // Fallback sphere avatar
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshPhongMaterial({
        color: 0x8866ff,
        emissive: 0x442288,
        shininess: 100,
        transparent: true,
        opacity: 0.9
    });

    avatar = new THREE.Mesh(geometry, material);
    avatar.position.set(0, 1, 0);
    avatar.castShadow = true;
    scene.add(avatar);
    isModelLoaded = true;
    console.log('✅ Using sphere fallback avatar');
}

function createParticles() {
    const particleCount = 500;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i += 3) {
        positions[i] = (Math.random() - 0.5) * 10;
        positions[i + 1] = (Math.random() - 0.5) * 10;
        positions[i + 2] = (Math.random() - 0.5) * 10;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
        color: 0xaa88ff,
        size: 0.05,
        transparent: true,
        opacity: 0.6,
        blending: THREE.AdditiveBlending
    });

    const particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);
    particles.push(particleSystem);
}

function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Update animation mixer
    if (mixer) {
        mixer.update(delta);
    }

    // Pulsing effect when Jazzy speaks
    if (isAnimating && avatar) {
        const pulse = Math.sin(Date.now() * 0.005) * 0.1;
        avatar.scale.setScalar(1.5 + pulse * animationScale);

        // Pulse light intensity
        light.intensity = 1 + pulse;
    }

    // Rotate particles slowly
    particles.forEach(p => {
        p.rotation.y += 0.0005;
        p.rotation.x += 0.0002;
    });

    // Gentle avatar rotation
    if (avatar && !isAnimating) {
        avatar.rotation.y += 0.002;
    }

    renderer.render(scene, camera);
}

function onWindowResize() {
    const canvas = document.querySelector('#avatar-canvas');
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

function startPulseAnimation() {
    isAnimating = true;
    animationScale = 0.15;
}

function stopPulseAnimation() {
    isAnimating = false;
    if (avatar) {
        avatar.scale.set(1.5, 1.5, 1.5);
    }
}

// WebSocket connection
function connectWebSocket() {
    const token = new URLSearchParams(window.location.search).get('token');
    const wsPort = new URLSearchParams(window.location.search).get('ws_port') || '8075';
    const wsUrl = new URL(`ws://${window.location.hostname || '127.0.0.1'}:${wsPort}`);
    if (token) wsUrl.searchParams.set('token', token);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('✅ Connected to Jazzy Avatar Server');
        updateStatus('connected', '🟢 Connected');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateStatus('error', '🔴 Connection Error');
    };

    ws.onclose = () => {
        console.log('Disconnected from server');
        updateStatus('disconnected', '🟡 Disconnected');

        // Auto-reconnect after 3 seconds
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            connectWebSocket();
        }, 3000);
    };
}

function handleWebSocketMessage(data) {
    console.log('Received message:', data);

    switch (data.type) {
        case 'token':
            // Streaming token from Jazzy (voice server)
            if (currentState !== 'speaking') {
                setState('speaking');
            }
            appendToKindredMessage(data.content);
            if (!isAnimating) startPulseAnimation();
            break;

        case 'chat_token':
            // Streaming token from original server
            if (currentState !== 'speaking') {
                setState('speaking');
            }
            appendToKindredMessage(data.token);
            if (!isAnimating) startPulseAnimation();
            break;

        case 'complete':
            // Response complete (voice server)
            setState('idle');
            stopPulseAnimation();
            finalizeKindredMessage();
            break;

        case 'chat_complete':
            // Complete message (original server - non-streaming)
            setState('idle');
            stopPulseAnimation();
            finalizeKindredMessage();
            break;

        case 'transcription':
            // Voice transcription
            displayTranscription(data.text, data.is_final, data.confidence);
            if (data.is_final && data.text) {
                // User finished speaking, now Jazzy is thinking
                setState('thinking');
            }
            break;

        case 'wake_word_detected':
            // Play confirmation beep when wake word is detected
            playWakeWordBeep();
            break;

        case 'system':
            // System message
            displaySystemMessage(data.content);
            break;

        case 'error':
            // Error message
            displayError(data.content);
            setState('idle');
            stopPulseAnimation();
            break;

        case 'image_generation_started':
            displaySystemMessage(`🎨 Generating image: ${data.prompt}`);
            setState('thinking');
            startPulseAnimation();
            break;

        case 'image_generation_complete':
            setState('idle');
            stopPulseAnimation();
            displaySystemMessage(data.message);
            if (data.image_url) {
                displayGeneratedImage(data.image_url, data.prompt);
            }
            break;

        default:
            console.warn('Unknown message type:', data.type, data);
    }
}

// Chat UI functions
let currentKindredMessage = null;

function displaySystemMessage(content) {
    const messagesDiv = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system-message';
    msgDiv.textContent = content;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function displayError(content) {
    const messagesDiv = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message error-message';
    msgDiv.textContent = `❌ ${content}`;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function displayTranscription(text, isFinal, confidence) {
    const transcriptionDiv = document.getElementById('transcription');

    // Show text with optional confidence
    let displayText = text;
    if (confidence !== undefined && confidence !== null) {
        const confidencePercent = Math.round(confidence * 100);
        const confidenceColor = confidence > 0.8 ? '#00ff88' : confidence > 0.6 ? '#ffaa00' : '#ff6666';
        displayText = `${text} <span style="font-size: 12px; color: ${confidenceColor}; margin-left: 8px;">${confidencePercent}%</span>`;
    }

    transcriptionDiv.innerHTML = displayText;
    transcriptionDiv.style.opacity = text ? '1' : '0';

    if (isFinal && text) {
        // Display as user message
        displayUserMessage(text);
        // Clear transcription
        setTimeout(() => {
            transcriptionDiv.innerHTML = '';
            transcriptionDiv.style.opacity = '0';
        }, 500);
    }
}

function displayUserMessage(content) {
    const messagesDiv = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.textContent = content;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function displayKindredMessage(content) {
    const messagesDiv = document.getElementById('messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message kindred-message';
    msgDiv.textContent = content;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function displayGeneratedImage(imageUrl, prompt) {
    const messagesDiv = document.getElementById('messages');
    const imgContainer = document.createElement('div');
    imgContainer.className = 'message kindred-message';
    imgContainer.style.padding = '10px';

    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = prompt || 'Generated image';
    img.style.maxWidth = '100%';
    img.style.borderRadius = '8px';
    img.style.cursor = 'pointer';
    img.onclick = () => window.open(imageUrl, '_blank');

    imgContainer.appendChild(img);
    messagesDiv.appendChild(imgContainer);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;

    console.log('🖼️ Displayed generated image:', imageUrl);
}

function appendToKindredMessage(token) {
    if (!currentKindredMessage) {
        const messagesDiv = document.getElementById('messages');
        currentKindredMessage = document.createElement('div');
        currentKindredMessage.className = 'message kindred-message';
        currentKindredMessage.fullText = ''; // Store full text for TTS
        messagesDiv.appendChild(currentKindredMessage);
    }

    currentKindredMessage.textContent += token;
    currentKindredMessage.fullText += token;
    const messagesDiv = document.getElementById('messages');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function finalizeKindredMessage() {
    if (currentKindredMessage && currentKindredMessage.fullText) {
        // Speak the complete response
        speak(currentKindredMessage.fullText);
    }
    currentKindredMessage = null;
}

function updateStatus(status, text) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = text;
    statusDiv.className = 'status ' + status;
}

// Text-to-speech
let speechSynthesis = window.speechSynthesis;
let currentUtterance = null;
let kindredVoice = null;
let isSpeaking = false;
let speechQueue = [];

// Initialize TTS voices
function initTTS() {
    const voices = speechSynthesis.getVoices();
    availableVoices = voices;

    // Populate voice selector
    const voiceSelect = document.getElementById('voice-select');
    voiceSelect.innerHTML = '';

    voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = voice.name;
        option.textContent = `${voice.name} (${voice.lang})`;
        voiceSelect.appendChild(option);
    });

    // Try to find a nice voice (prefer Samantha, Alex, or any female voice on Mac)
    kindredVoice = voices.find(v => v.name.includes('Samantha')) ||
                   voices.find(v => v.name.includes('Fiona')) ||
                   voices.find(v => v.name.includes('Karen')) ||
                   voices.find(v => v.name.includes('female')) ||
                   voices[0]; // Fallback to first available

    if (kindredVoice) {
        voiceSettings.selectedVoice = kindredVoice;
        voiceSelect.value = kindredVoice.name;
        document.getElementById('voice-name').textContent = kindredVoice.name.split(' ')[0];
    }

    console.log('🔊 TTS Voice selected:', kindredVoice?.name || 'default');
}

// Speak text using TTS
function speak(text) {
    if (!text || text.trim().length === 0) return;

    // Cancel any ongoing speech
    if (isSpeaking) {
        speechSynthesis.cancel();
        isSpeaking = false;
    }

    const cleanText = text.trim();
    currentUtterance = new SpeechSynthesisUtterance(cleanText);

    if (voiceSettings.selectedVoice) {
        currentUtterance.voice = voiceSettings.selectedVoice;
    }
    currentUtterance.rate = voiceSettings.speed;
    currentUtterance.pitch = voiceSettings.pitch;
    currentUtterance.volume = 1.0;

    currentUtterance.onstart = () => {
        console.log('🔊 Speaking:', cleanText.substring(0, 50) + (cleanText.length > 50 ? '...' : ''));
        isSpeaking = true;
        setState('speaking');
        startPulseAnimation();
    };

    currentUtterance.onend = () => {
        console.log('🔇 Finished speaking');
        isSpeaking = false;
        setState('idle');
        stopPulseAnimation();
        currentUtterance = null;
    };

    currentUtterance.onerror = (event) => {
        console.error('TTS error:', event);
        isSpeaking = false;
        setState('idle');
        stopPulseAnimation();
        currentUtterance = null;
    };

    speechSynthesis.speak(currentUtterance);
}

// Voice input functions
async function initAudio() {
    try {
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true
            }
        });

        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        const source = audioContext.createMediaStreamSource(audioStream);

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        waveformDataArray = new Uint8Array(analyser.fftSize);

        // Use ScriptProcessor for audio streaming
        const bufferSize = 4096;
        processor = audioContext.createScriptProcessor(bufferSize, 1, 1);

        processor.onaudioprocess = (e) => {
            if (isRecording && ws && ws.readyState === WebSocket.OPEN) {
                const inputData = e.inputBuffer.getChannelData(0);
                // Convert Float32Array to Int16Array for WhisperLiveKit
                const int16Data = new Int16Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                }
                ws.send(int16Data.buffer);
                console.log('Sent audio chunk:', int16Data.length, 'samples');
            }
        };

        source.connect(analyser);
        analyser.connect(processor);
        processor.connect(audioContext.destination);
        ensureWaveformCanvas();

        console.log('✅ Audio initialized');
        return true;
    } catch (error) {
        console.error('Failed to initialize audio:', error);
        displayError('Microphone access denied. Text chat still available.');
        return false;
    }
}

function toggleRecording() {
    const recordBtn = document.getElementById('record-btn');

    if (!isRecording) {
        // Start recording
        if (!audioContext) {
            initAudio().then(success => {
                if (success) startRecording();
            });
        } else {
            startRecording();
        }
    } else {
        // Stop recording
        stopRecording();
    }
}

function startRecording() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume().catch((error) => {
            console.error('Failed to resume audio context:', error);
        });
    }

    if (audioStream) {
        audioStream.getAudioTracks().forEach((track) => {
            track.enabled = true;
        });
    }

    isRecording = true;
    setState('listening');
    const recordBtn = document.getElementById('record-btn');
    const cancelBtn = document.getElementById('cancel-recording');
    recordBtn.textContent = '⏸️ Stop';
    recordBtn.classList.add('recording');
    cancelBtn.classList.add('visible');
    startWaveformAnimation();
    displaySystemMessage('🎤 Listening...');
    console.log('Started recording - audio will stream to server');
}

function stopRecording() {
    isRecording = false;
    setState('idle');
    const recordBtn = document.getElementById('record-btn');
    const cancelBtn = document.getElementById('cancel-recording');
    recordBtn.textContent = '🎤 Voice';
    recordBtn.classList.remove('recording');
    cancelBtn.classList.remove('visible');
    stopWaveformAnimation();
}

function cancelRecording() {
    if (isRecording) {
        stopRecording();
        displaySystemMessage('❌ Recording cancelled');

        // Send cancel message to server if needed
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'cancel_recording'
            }));
        }
    }
}

// Text input functions
function sendMessage() {
    const input = document.getElementById('message-input');
    const message = input.value.trim();

    console.log('Attempting to send message:', message);
    console.log('WebSocket state:', ws ? ws.readyState : 'null');

    if (message && ws && ws.readyState === WebSocket.OPEN) {
        displayUserMessage(message);
        const payload = JSON.stringify({
            type: 'chat',
            message: message
        });
        console.log('Sending payload:', payload);
        ws.send(payload);
        input.value = '';
    } else {
        console.error('Cannot send: message empty or WebSocket not open');
    }
}

function generateImage() {
    const prompt = prompt('Enter image prompt:', 'a mystical AI entity in purple light');

    if (prompt && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'generate_image',
            prompt: prompt
        }));
    }
}

function clearChat() {
    document.getElementById('messages').innerHTML = '';
    displaySystemMessage('Chat cleared');
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Three.js
    initThreeJS();

    // Initialize TTS
    initTTS();
    // Re-initialize when voices are loaded (some browsers load voices asynchronously)
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = initTTS;
    }

    // Connect to WebSocket server
    connectWebSocket();

    // UI event listeners
    document.getElementById('send-btn').addEventListener('click', sendMessage);
    document.getElementById('record-btn').addEventListener('click', toggleRecording);
    document.getElementById('generate-btn').addEventListener('click', generateImage);
    document.getElementById('clear-btn').addEventListener('click', clearChat);

    // Voice settings listeners
    document.getElementById('voice-settings-toggle').addEventListener('click', () => {
        const settingsPanel = document.getElementById('voice-settings');
        settingsPanel.classList.toggle('visible');
    });

    document.getElementById('close-settings-btn').addEventListener('click', () => {
        document.getElementById('voice-settings').classList.remove('visible');
    });

    document.getElementById('voice-select').addEventListener('change', (e) => {
        const selectedVoiceName = e.target.value;
        voiceSettings.selectedVoice = availableVoices.find(v => v.name === selectedVoiceName);
        document.getElementById('voice-name').textContent = selectedVoiceName.split(' ')[0];
        console.log('🔊 Voice changed to:', selectedVoiceName);
    });

    document.getElementById('pitch-slider').addEventListener('input', (e) => {
        voiceSettings.pitch = parseFloat(e.target.value);
        document.getElementById('pitch-value').textContent = voiceSettings.pitch.toFixed(1);
    });

    document.getElementById('speed-slider').addEventListener('input', (e) => {
        voiceSettings.speed = parseFloat(e.target.value);
        document.getElementById('speed-value').textContent = voiceSettings.speed.toFixed(1);
    });

    document.getElementById('test-voice-btn').addEventListener('click', () => {
        const testText = "Hello, I am Jazzy. This is how I sound with these settings.";
        speak(testText);
    });

    document.getElementById('cancel-recording').addEventListener('click', cancelRecording);

    document.getElementById('message-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Welcome message
    setTimeout(() => {
        displaySystemMessage('✨ Welcome! I am Jazzy. Speak or type to begin our conversation.');
    }, 500);
});
