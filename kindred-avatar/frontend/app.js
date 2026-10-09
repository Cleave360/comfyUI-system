// Kindred Avatar Frontend - Three.js + WebSocket

// Three.js scene setup
let scene, camera, renderer, avatar, light;
let isAnimating = false;
let animationScale = 1.0;

function initThreeJS() {
    const canvas = document.querySelector('#avatar-canvas');

    // Scene
    scene = new THREE.Scene();
    scene.background = null; // Transparent for gradient background

    // Camera
    camera = new THREE.PerspectiveCamera(
        75,
        canvas.clientWidth / canvas.clientHeight,
        0.1,
        1000
    );
    camera.position.z = 5;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    canvas.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    light = new THREE.PointLight(0xffffff, 1, 100);
    light.position.set(5, 5, 5);
    scene.add(light);

    // Create simple avatar (sphere for now - we'll upgrade this later)
    createAvatar();

    // Handle window resize
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

function createAvatar() {
    // Simple glowing sphere as Kindred avatar
    const geometry = new THREE.SphereGeometry(1.5, 64, 64);

    // Gradient material
    const material = new THREE.MeshPhongMaterial({
        color: 0x764ba2,
        emissive: 0x667eea,
        emissiveIntensity: 0.5,
        shininess: 100,
    });

    avatar = new THREE.Mesh(geometry, material);
    scene.add(avatar);

    // Add particles/glow effect
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 1000;
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i++) {
        positions[i] = (Math.random() - 0.5) * 10;
    }

    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const particleMaterial = new THREE.PointsMaterial({
        color: 0xffffff,
        size: 0.05,
        transparent: true,
        opacity: 0.6,
    });

    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);
}

function animate() {
    requestAnimationFrame(animate);

    // Rotate avatar slowly
    avatar.rotation.y += 0.005;

    // Pulsing animation when speaking
    if (isAnimating) {
        animationScale += Math.sin(Date.now() * 0.01) * 0.002;
        avatar.scale.set(animationScale, animationScale, animationScale);

        // Clamp scale
        animationScale = Math.max(0.95, Math.min(1.05, animationScale));
    } else {
        // Return to normal size
        animationScale += (1.0 - animationScale) * 0.1;
        avatar.scale.set(animationScale, animationScale, animationScale);
    }

    // Rotate particles
    scene.children.forEach(child => {
        if (child instanceof THREE.Points) {
            child.rotation.y += 0.001;
        }
    });

    renderer.render(scene, camera);
}

function onWindowResize() {
    const canvas = document.querySelector('#avatar-canvas');
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
}

// WebSocket connection
let ws;
let currentKindredMessage = "";

function connectWebSocket() {
    const token = new URLSearchParams(window.location.search).get('token');
    const wsPort = new URLSearchParams(window.location.search).get('ws_port') || '8075';
    const wsUrl = new URL(`ws://${window.location.hostname || '127.0.0.1'}:${wsPort}`);
    if (token) wsUrl.searchParams.set('token', token);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('Connected to Kindred server');
        updateConnectionStatus('Connected', 'green');
        updateStatus('Ready to chat');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('Error', 'red');
    };

    ws.onclose = () => {
        console.log('Disconnected from server');
        updateConnectionStatus('Disconnected', 'orange');
        setTimeout(connectWebSocket, 3000); // Reconnect after 3s
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'chat_token':
            // Stream incoming tokens
            if (!currentKindredMessage) {
                addMessage('', 'kindred', true); // Create message element
                isAnimating = true;
            }
            currentKindredMessage = data.full_text;
            updateLastMessage(currentKindredMessage);
            break;

        case 'chat_complete':
            currentKindredMessage = "";
            isAnimating = false;
            updateStatus('Ready');
            break;

        case 'status':
            updateStatus(data.status);
            if (data.status === 'thinking') {
                isAnimating = true;
            }
            break;

        case 'image_generated':
            addMessage(`🖼️ ${data.message}`, 'kindred');
            isAnimating = false;
            break;

        case 'error':
            addMessage(`⚠️ Error: ${data.message}`, 'kindred');
            isAnimating = false;
            updateStatus('Error');
            break;
    }
}

function sendMessage(message) {
    if (!message.trim()) return;

    addMessage(message, 'user');

    ws.send(JSON.stringify({
        type: 'chat',
        message: message
    }));

    document.getElementById('message-input').value = '';
}

function generateImage() {
    const prompt = prompt || "a beautiful fantasy landscape";

    ws.send(JSON.stringify({
        type: 'generate_image',
        prompt: prompt
    }));

    updateStatus('Generating image...');
}

function addMessage(text, sender, isStreaming = false) {
    const chatContainer = document.getElementById('chat-container');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.textContent = text;

    if (isStreaming) {
        messageDiv.id = 'streaming-message';
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateLastMessage(text) {
    const streamingMsg = document.getElementById('streaming-message');
    if (streamingMsg) {
        streamingMsg.textContent = text;
    }
}

function updateStatus(status) {
    const indicator = document.getElementById('status-indicator');
    indicator.textContent = status;

    if (status.toLowerCase().includes('thinking')) {
        indicator.classList.add('thinking');
    } else {
        indicator.classList.remove('thinking');
    }
}

function updateConnectionStatus(status, color) {
    const statusEl = document.getElementById('connection-status');
    statusEl.textContent = `Status: ${status}`;
    statusEl.style.color = color;
}

function clearChat() {
    document.getElementById('chat-container').innerHTML = '';
    currentKindredMessage = "";
}

// Event listeners
document.getElementById('send-btn').addEventListener('click', () => {
    const input = document.getElementById('message-input');
    sendMessage(input.value);
});

document.getElementById('message-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage(e.target.value);
    }
});

document.getElementById('gen-image-btn').addEventListener('click', generateImage);
document.getElementById('clear-btn').addEventListener('click', clearChat);

// Initialize
initThreeJS();
connectWebSocket();

// Welcome message
setTimeout(() => {
    addMessage("Hello! I'm Kindred. I can chat with you and help generate images. What would you like to create today?", 'kindred');
}, 500);
