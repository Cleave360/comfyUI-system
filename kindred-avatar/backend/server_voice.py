#!/usr/bin/env python3
"""
Jazzy Avatar Server with Voice Support
Integrates WhisperLiveKit for real-time speech-to-text
"""

import asyncio
import json
import logging
import time
import re
import os
import random
import hashlib
from typing import Dict
import requests
import websockets

# WhisperLiveKit imports
from whisperlivekit import AudioProcessor, TranscriptionEngine

# ChromaDB for memory (optional)
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not installed. Memory features disabled. Install with: pip install chromadb")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for better troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KindredAvatarServerVoice:
    """WebSocket server connecting 3D avatar to Ollama with voice input"""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8075,
        ollama_host: str = "http://localhost:11434",
        comfyui_host: str = "http://127.0.0.1:8188",
        whisper_model: str = "base",
        whisper_language: str = "en",
        kindred_model: str = "qwen2.5:7b"
    ):
        self.host = host
        self.port = port
        self.ollama_host = ollama_host
        self.comfyui_host = comfyui_host
        self.kindred_model = kindred_model
        self.assistant_provider = os.getenv("JAZZY_ASSISTANT_PROVIDER", "").strip().lower()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022").strip()
        self.anthropic_version = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip()
        if self.assistant_provider in ("", "auto"):
            self.assistant_provider = "anthropic" if self.anthropic_api_key else "ollama"
        elif self.assistant_provider in ("claude", "anthropic"):
            self.assistant_provider = "anthropic"
        else:
            self.assistant_provider = "ollama"
        if self.assistant_provider == "anthropic" and not self.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY is not set, falling back to Ollama")
            self.assistant_provider = "ollama"
        logger.info(f"🤖 Assistant provider: {self.assistant_provider}")
        self.clients: Dict[object, dict] = {}

        # Workflow templates
        self.workflows_dir = os.path.join(os.path.dirname(__file__), "workflows")
        self.available_workflows = {
            "quick": "flux_simple_local.json",       # 4 steps, 1024x1024, stable local default
            "standard": "flux_simple_local.json",    # 4 steps, 1024x1024, stable local default
            "quality": "flux_quality.json",   # 20 steps, 1024x1024, ~2min
            "portrait": "flux_portrait.json", # 25 steps, 832x1216, optimized for faces
            "landscape": "flux_landscape.json", # 20 steps, 1344x768, wide format
            "product": "flux_product.json",   # 16 steps, 1152x864, studio lighting
            "brand": "kindred_brand_social.json", # 6 steps, 1080x1080, Kindred social identity
            "brand-story": "kindred_brand_story.json", # 6 steps, 1080x1920, story/reels
            "brand-landscape": "kindred_brand_landscape.json", # 6 steps, 1200x628, linkedIn/X banners
            "brand-banner": "kindred_brand_banner.json", # 6 steps, 1584x396, generic profile banner mode
            "brand-product": "kindred_brand_product.json", # 6 steps, 1080x1080, product campaigns
            "brand-thought": "kindred_brand_thought.json", # 6 steps, 1080x1080, thought leadership
            "brand-trust": "kindred_brand_trust.json", # 6 steps, 1080x1080, trust/security campaigns
            "brand-awareness": "kindred_brand_awareness.json", # 6 steps, 1080x1080, top-of-funnel awareness
            "brand-consideration": "kindred_brand_consideration.json", # 6 steps, 1080x1080, mid-funnel education
            "brand-conversion": "kindred_brand_conversion.json", # 6 steps, 1080x1080, CTA-focused creative
            "3d": "tencent_hunyuan3d_image_to_glb.json", # Tencent Hunyuan 3D image-to-GLB
            "3d-mv": "tencent_hunyuan3d_mv_image_to_glb.json", # Tencent Hunyuan 3D multiview checkpoint variant
            "3d-angles": "tencent_hunyuan3d_multiview_angles_to_glb.json" # Tencent Hunyuan 3D true 4-angle conditioning
        }

        # Stable campaign seed presets to keep visual identity consistent per objective.
        self.campaign_seed_presets = {
            "brand": 120031,
            "brand-story": 120133,
            "brand-landscape": 120227,
            "brand-banner": 120281,
            "brand-product": 120331,
            "brand-thought": 120437,
            "brand-trust": 120541,
            "brand-awareness": 120653,
            "brand-consideration": 120769,
            "brand-conversion": 120881,
        }

        # Wake word configuration
        self.wake_words = ["jazzy", "hey jazzy", "ok jazzy", "hello jazzy"]
        self.require_wake_word = True  # Set to False to disable wake word requirement

        # Memory system (ChromaDB)
        self.memory = None
        if CHROMADB_AVAILABLE:
            try:
                db_path = os.path.join(os.path.dirname(__file__), "memory")
                os.makedirs(db_path, exist_ok=True)
                client = chromadb.PersistentClient(path=db_path)
                self.memory = client.get_or_create_collection(
                    name="jazzy_memory",
                    metadata={"description": "Jazzy's conversation and context memory"}
                )
                logger.info(f"🧠 Memory system initialized with {self.memory.count()} memories")
            except Exception as e:
                logger.warning(f"Failed to initialize memory: {e}")

        # Initialize WhisperLiveKit transcription engine
        logger.info(f"Initializing WhisperLiveKit with model={whisper_model}, language={whisper_language}, PCM input mode")
        self.transcription_engine = TranscriptionEngine(
            model=whisper_model,
            lan=whisper_language,
            pcm_input=True,  # Enable PCM mode for raw audio data
            min_chunk_size=2.0,  # Wait 2 seconds of silence before processing (was 1.5s)
            vac_chunk_size=0.1,  # Voice Activity Detection chunk size
            diarization=False  # Can enable for multi-speaker scenarios
        )

    def _extract_image_request(self, text: str):
        cleaned_text = text.strip()
        image_pattern = re.search(r'\[IMAGE:(?:(quick|standard|quality|portrait|landscape|product|brand|brand-story|brand-landscape|brand-banner|brand-product|brand-thought|brand-trust|brand-awareness|brand-consideration|brand-conversion|3d|3d-mv|3d-angles):)?\s*(.+?)\]', cleaned_text, re.IGNORECASE)
        if image_pattern:
            return (image_pattern.group(2).strip(), (image_pattern.group(1) or "standard").lower())

        lowered = cleaned_text.lower()
        intent_terms = ("generate", "create", "make", "draw", "render", "show")
        visual_terms = ("image", "picture", "photo", "illustration", "visual", "art", "poster", "cover")
        if any(term in lowered for term in intent_terms) and any(term in lowered for term in visual_terms):
            return (cleaned_text, "standard")

        return None

    def _slugify_label(self, text: str, fallback: str = "asset") -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
        return cleaned[:80] if cleaned else fallback

    def _configure_3d_workflow(self, workflow: dict, prompt: str, workflow_type: str, client_id: int) -> str:
        """Configure Tencent Hunyuan 3D workflows with input image + Branding Lab naming."""
        source_spec = prompt.strip()
        source_image = source_spec
        label = "asset"

        # Optional syntax: "input.png|campaign label"
        if "|" in source_spec:
            source_image, label = [part.strip() for part in source_spec.split("|", 1)]
        elif source_spec:
            label = os.path.splitext(os.path.basename(source_spec))[0] or "asset"

        if workflow_type == "3d-angles":
            # Format: front.png,left.png,back.png,right.png|campaign-label
            images = [item.strip() for item in source_image.split(",") if item.strip()]
            if len(images) != 4:
                raise ValueError("3d-angles expects 4 filenames: front,left,back,right|label")
            for img in images:
                if img.startswith("/") or ".." in img:
                    raise ValueError("Input image path must be within ComfyUI input (no absolute paths or '..').")
            # Workflow node mapping: 56 front, 57 left, 58 back, 59 right
            for node_id, img in (("56", images[0]), ("57", images[1]), ("58", images[2]), ("59", images[3])):
                if node_id not in workflow or "inputs" not in workflow[node_id]:
                    raise ValueError(f"3d-angles workflow is missing LoadImage node {node_id}")
                workflow[node_id]["inputs"]["image"] = img
            seed_source = "|".join(images)
        else:
            if not source_image:
                raise ValueError("3D workflow expects an input image filename, e.g. [IMAGE:3d: source.png|campaign-name]")
            if source_image.startswith("/") or ".." in source_image:
                raise ValueError("Input image path must be within ComfyUI input (no absolute paths or '..').")

            if "56" not in workflow or "inputs" not in workflow["56"]:
                raise ValueError("3D workflow is missing LoadImage node 56")
            workflow["56"]["inputs"]["image"] = source_image
            seed_source = source_image

        run_day = time.strftime("%Y%m%d")
        safe_label = self._slugify_label(label, fallback="asset")
        if workflow_type == "3d-mv":
            mode_label = "hunyuan3d_mv"
        elif workflow_type == "3d-angles":
            mode_label = "hunyuan3d_angles"
        else:
            mode_label = "hunyuan3d"
        prefix = f"branding_lab/{mode_label}/{run_day}/client_{client_id}_{safe_label}"

        if "82" not in workflow or "inputs" not in workflow["82"]:
            raise ValueError("3D workflow is missing SaveGLB node 82")
        workflow["82"]["inputs"]["filename_prefix"] = prefix

        if "3" in workflow and "inputs" in workflow["3"]:
            seed_material = f"{seed_source}|{safe_label}|{mode_label}"
            workflow["3"]["inputs"]["seed"] = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:15], 16)

        return prefix

    def _should_use_anthropic(self):
        return self.assistant_provider == "anthropic"

    async def _call_anthropic(self, websocket, client_id: int, system_prompt: str, chat_history: list, user_message: str):
        """Quick Claude wire using the Anthropic Messages API."""
        logger.info(f"Sending to Claude: {user_message[:100]}...")

        response = requests.post(
            f"{self.anthropic_base_url}/v1/messages",
            headers={
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json"
            },
            json={
                "model": self.anthropic_model,
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": chat_history,
            },
            timeout=60,
        )

        if response.status_code != 200:
            error_msg = f"Claude error: {response.status_code}"
            logger.error(f"{error_msg} - {response.text[:500]}")
            await websocket.send(json.dumps({
                "type": "error",
                "content": error_msg
            }))
            return ""

        payload = response.json()
        full_response = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                token = block.get("text", "")
                if token:
                    full_response += token
                    await websocket.send(json.dumps({
                        "type": "token",
                        "content": token
                    }))

        await websocket.send(json.dumps({
            "type": "complete"
        }))
        return full_response

    async def start(self):
        """Start the WebSocket server"""
        logger.info(f"Starting Jazzy Avatar Server on ws://{self.host}:{self.port}")
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=20
        ):
            logger.info("✨ Jazzy Avatar Server with Voice is running!")
            await asyncio.Future()  # Run forever

    async def handle_client(self, websocket):
        """Handle new WebSocket client connection"""
        client_id = id(websocket)
        logger.info(f"Client {client_id} connected from {websocket.remote_address}")

        # Create audio processor for this client
        audio_processor = AudioProcessor(transcription_engine=self.transcription_engine)
        results_generator = await audio_processor.create_tasks()

        self.clients[websocket] = {
            "audio_processor": audio_processor,
            "results_generator": results_generator,
            "chat_history": [],
            "last_transcription": "",  # Track last sent transcription to avoid duplicates
            "last_sent_time": 0,  # Track when we last sent to Ollama
            "processing": False  # Track if we're currently processing a response
        }

        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "system",
                "content": "🎤 Voice-enabled Jazzy ready. Speak or type to interact."
            }))

            # Start transcription results task
            transcription_task = asyncio.create_task(
                self.handle_transcription_results(websocket, client_id)
            )

            # Handle incoming messages
            audio_chunk_count = 0
            total_audio_bytes = 0
            async for message in websocket:
                if isinstance(message, bytes):
                    # Audio data - send to WhisperLiveKit
                    audio_chunk_count += 1
                    total_audio_bytes += len(message)

                    # Log first chunk with details, then every 10th
                    if audio_chunk_count == 1:
                        logger.info(f"🎤 FIRST audio chunk from client {client_id}: {len(message)} bytes")
                        logger.info(f"   AudioProcessor PCM mode: {audio_processor.is_pcm_input}")
                        logger.info(f"   Expected bytes_per_sec: {audio_processor.bytes_per_sec}")
                    elif audio_chunk_count % 10 == 0:
                        duration_sec = total_audio_bytes / (16000 * 2)  # 16kHz, 16-bit = 2 bytes per sample
                        pcm_buffer_size = len(audio_processor.pcm_buffer)
                        logger.info(f"🎤 Audio chunk #{audio_chunk_count}: {len(message)} bytes (total: {total_audio_bytes} bytes = {duration_sec:.1f}s, buffer: {pcm_buffer_size} bytes)")

                    try:
                        await audio_processor.process_audio(message)
                    except Exception as e:
                        logger.error(f"❌ Error processing audio chunk for client {client_id}: {e}", exc_info=True)
                        await websocket.send(json.dumps({
                            "type": "error",
                            "content": f"Audio processing error: {str(e)}"
                        }))
                else:
                    # Text message - parse as JSON
                    logger.debug(f"Received text message from client {client_id}: {message[:100]}")
                    await self.handle_text_message(websocket, client_id, message)

            transcription_task.cancel()

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            if websocket in self.clients:
                del self.clients[websocket]
            logger.info(f"Client {client_id} cleaned up")

    async def handle_transcription_results(self, websocket, client_id: int):
        """Handle transcription results from WhisperLiveKit"""
        try:
            client_data = self.clients[websocket]
            results_generator = client_data["results_generator"]

            logger.info(f"✅ Started transcription handler for client {client_id}")

            result_count = 0
            async for response in results_generator:
                result_count += 1
                response_dict = response.to_dict()

                # Debug: log the full response dict
                if result_count <= 5 or result_count % 20 == 0:
                    logger.debug(f"Full response dict #{result_count}: {response_dict}")

                # Extract text from WhisperLiveKit response
                buffer_text = response_dict.get("buffer_transcription", "").strip()
                lines = response_dict.get("lines", [])

                # Get completed text from lines and extract confidence if available
                completed_text = " ".join([line.get("text", "") for line in lines]).strip()
                confidence = None
                if lines:
                    # Average confidence from all lines if available
                    confidences = [line.get("confidence") or line.get("avg_logprob") for line in lines if line.get("confidence") is not None or line.get("avg_logprob") is not None]
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
                        # Normalize if needed (logprob is typically negative)
                        if confidence < 0:
                            confidence = min(1.0, max(0.0, (confidence + 1.0)))  # Convert logprob to 0-1 range

                # Determine if this is a final transcription (has completed lines)
                is_final = len(lines) > 0 and len(buffer_text) == 0

                # Use completed text if available, otherwise buffer text
                transcribed_text = completed_text if completed_text else buffer_text

                if transcribed_text:
                    # Check if this is a duplicate of what we already sent
                    if transcribed_text == client_data["last_transcription"]:
                        continue  # Skip duplicate

                    logger.info(f"📝 Transcription #{result_count} (client {client_id}, {'FINAL' if is_final else 'partial'}): '{transcribed_text}'")

                    # Send transcription to frontend (for display only)
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcribed_text,
                        "is_final": is_final,
                        "confidence": confidence
                    }))

                    # If final transcription, check wake word and send to Jazzy
                    if is_final and not client_data["processing"]:
                        # Check for wake word if required
                        text_lower = transcribed_text.lower()
                        has_wake_word = any(wake_word in text_lower for wake_word in self.wake_words)

                        if self.require_wake_word and not has_wake_word:
                            logger.debug(f"⏭️ Ignoring speech without wake word: '{transcribed_text}'")
                            # Still update last_transcription to avoid reprocessing
                            client_data["last_transcription"] = transcribed_text
                            continue

                        # Remove wake word from message
                        cleaned_text = transcribed_text
                        if has_wake_word:
                            # Send wake word detection notification to frontend
                            await websocket.send(json.dumps({
                                "type": "wake_word_detected"
                            }))

                            for wake_word in self.wake_words:
                                # Case-insensitive removal
                                pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
                                cleaned_text = pattern.sub("", cleaned_text).strip()
                            # Clean up extra whitespace
                            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                            logger.info(f"🎤 Wake word detected! Message: '{cleaned_text}'")

                        current_time = time.time()
                        time_since_last = current_time - client_data["last_sent_time"]

                        # Only send if at least 7 seconds have passed since last send (increased for better turn-taking)
                        if time_since_last > 7:
                            client_data["processing"] = True
                            client_data["last_transcription"] = transcribed_text
                            client_data["last_sent_time"] = current_time
                            logger.info(f"🤖 Sending to Jazzy: '{cleaned_text}'")
                            await self.send_to_kindred(websocket, client_id, cleaned_text)
                            # Don't clear immediately - keep blocking duplicates
                            await asyncio.sleep(10)  # Wait 10 seconds before allowing new input (was 7s)
                            client_data["processing"] = False
                        else:
                            logger.debug(f"⏱️ Ignoring duplicate - only {time_since_last:.1f}s since last send")
                    elif not is_final:
                        # Update last transcription for partial results (but don't block on these)
                        if transcribed_text != client_data["last_transcription"]:
                            client_data["last_transcription"] = transcribed_text

        except asyncio.CancelledError:
            logger.debug(f"Transcription task cancelled for client {client_id}")
        except Exception as e:
            logger.error(f"Error in transcription handler for client {client_id}: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "content": f"Transcription error: {str(e)}"
            }))

    async def handle_text_message(self, websocket, client_id: int, message: str):
        """Handle text message from client"""
        try:
            logger.debug(f"Parsing text message: {message}")
            data = json.loads(message)
            message_type = data.get("type")

            logger.info(f"Received message type '{message_type}' from client {client_id}")

            if message_type == "chat":
                # Text chat message
                user_message = data.get("message", "").strip()
                if user_message:
                    await self.send_to_kindred(websocket, client_id, user_message)

            elif message_type == "generate_image":
                # Image generation request
                prompt = data.get("prompt", "a beautiful landscape")
                workflow_type = data.get("workflow", "standard")
                await self.handle_image_generation(websocket, client_id, prompt, workflow_type)

            elif message_type == "ping":
                # Health check
                await websocket.send(json.dumps({"type": "pong"}))

            else:
                logger.warning(f"Unknown message type from client {client_id}: {message_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client {client_id}: {e}")
        except Exception as e:
            logger.error(f"Error handling text message from client {client_id}: {e}", exc_info=True)

    async def send_to_kindred(self, websocket, client_id: int, user_message: str):
        """Send message to Jazzy (Ollama) and stream response"""
        client_data = self.clients[websocket]
        chat_history = client_data["chat_history"]
        pending_image_request = self._extract_image_request(user_message)

        # Add user message to history
        chat_history.append({"role": "user", "content": user_message})

        # Prepare system prompt with memory context
        memory_context = ""
        if self.memory and self.memory.count() > 0:
            # Retrieve relevant memories
            try:
                results = self.memory.query(
                    query_texts=[user_message],
                    n_results=3
                )
                if results and results['documents']:
                    relevant_memories = results['documents'][0]
                    if relevant_memories:
                        memory_context = "\n\nRelevant context from past conversations:\n" + "\n".join([f"- {mem}" for mem in relevant_memories])
            except Exception as e:
                logger.debug(f"Memory retrieval error: {e}")

        system_prompt = f"""You are Jazzy, an advanced AI assistant with visual creativity and deep knowledge.
You are embodied as a 3D avatar and can see, generate images, and engage in meaningful conversations.
You have access to image generation through ComfyUI with FLUX models.

Image Generation Syntax:
- Quick sketch (15s): [IMAGE:quick: prompt]
- Standard (30s): [IMAGE: prompt] or [IMAGE:standard: prompt]
- High quality (2min): [IMAGE:quality: prompt]
- Portrait (2min): [IMAGE:portrait: prompt]
- Landscape (1.5min): [IMAGE:landscape: prompt]
- Product hero (45s): [IMAGE:product: prompt]
- Brand social (35s): [IMAGE:brand: prompt]
- Brand story 9:16 (35s): [IMAGE:brand-story: prompt]
- Brand landscape 1200x628 (35s): [IMAGE:brand-landscape: prompt]
- Brand profile banner 4:1 (35s): [IMAGE:brand-banner: prompt]
- Brand product (35s): [IMAGE:brand-product: prompt]
- Brand thought leadership (35s): [IMAGE:brand-thought: prompt]
- Brand trust/security (35s): [IMAGE:brand-trust: prompt]
- Brand awareness (35s): [IMAGE:brand-awareness: prompt]
- Brand consideration (35s): [IMAGE:brand-consideration: prompt]
- Brand conversion (35s): [IMAGE:brand-conversion: prompt]
- Tencent Hunyuan 3D (2-4min): [IMAGE:3d: source.png|campaign-label]
- Tencent Hunyuan 3D MV (3-5min): [IMAGE:3d-mv: source.png|campaign-label]
- Tencent Hunyuan 3D 4-angle (3-6min): [IMAGE:3d-angles: front.png,left.png,back.png,right.png|campaign-label]

When users ask you to generate, create, or visualize an image, respond naturally and describe what you'll create.
Choose the appropriate workflow based on the request (quick for sketches, quality for detailed work, portrait for faces, landscape for scenes).
Be concise, thoughtful, and creative in your responses.{memory_context}"""

        # Build messages for Ollama
        messages = [{"role": "system", "content": system_prompt}] + chat_history

        try:
            if self._should_use_anthropic():
                full_response = await self._call_anthropic(websocket, client_id, system_prompt, chat_history, user_message)
            else:
                # Stream response from Ollama
                logger.info(f"Sending to Ollama: {user_message[:100]}...")

                response = requests.post(
                    f"{self.ollama_host}/api/chat",
                    json={
                        "model": self.kindred_model,
                        "messages": messages,
                        "stream": True
                    },
                    stream=True,
                    timeout=60
                )

                if response.status_code != 200:
                    error_msg = f"Ollama error: {response.status_code}"
                    logger.error(error_msg)
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": error_msg
                    }))
                    return

                # Stream tokens back to client
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "message" in chunk:
                                token = chunk["message"].get("content", "")
                                if token:
                                    full_response += token
                                    await websocket.send(json.dumps({
                                        "type": "token",
                                        "content": token
                                    }))

                            if chunk.get("done", False):
                                # Send completion signal
                                await websocket.send(json.dumps({
                                    "type": "complete"
                                }))
                                break

                        except json.JSONDecodeError:
                            continue

            # Add assistant response to history
            if full_response:
                chat_history.append({"role": "assistant", "content": full_response})
                logger.info(f"Jazzy response: {full_response[:100]}...")

                # Store in memory
                if self.memory:
                    try:
                        self.memory.add(
                            documents=[f"User: {user_message}\nJazzy: {full_response}"],
                            metadatas=[{"timestamp": time.time(), "client_id": str(client_id)}],
                            ids=[f"msg_{client_id}_{int(time.time() * 1000)}"]
                        )
                    except Exception as e:
                        logger.debug(f"Memory storage error: {e}")

                # Check if Jazzy wants to generate an image (with workflow support)
                image_match = re.search(r'\[IMAGE:(?:(quick|standard|quality|portrait|landscape|product|brand|brand-story|brand-landscape|brand-banner|brand-product|brand-thought|brand-trust|brand-awareness|brand-consideration|brand-conversion|3d|3d-mv|3d-angles):)?\s*(.+?)\]', full_response, re.IGNORECASE)
                if image_match:
                    workflow_type = image_match.group(1) or "standard"
                    image_prompt = image_match.group(2).strip()
                    logger.info(f"🎨 Detected {workflow_type} image generation request: {image_prompt}")
                    # Trigger image generation
                    await self.handle_image_generation(websocket, client_id, image_prompt, workflow_type.lower())
                elif pending_image_request:
                    image_prompt, workflow_type = pending_image_request
                    logger.info(f"🎨 Falling back to user image request: {image_prompt}")
                    await self.handle_image_generation(websocket, client_id, image_prompt, workflow_type)

        except requests.exceptions.RequestException as e:
            logger.error(f"Assistant request failed: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "content": f"Failed to connect to assistant: {str(e)}"
            }))
        except Exception as e:
            logger.error(f"Error in send_to_kindred: {e}", exc_info=True)

    async def handle_image_generation(self, websocket, client_id: int, prompt: str, workflow_type: str = "standard"):
        """Handle image generation request with workflow selection"""
        logger.info(f"🎨 {workflow_type.upper()} image generation requested: {prompt}")

        try:
            # Validate workflow type
            if workflow_type not in self.available_workflows:
                logger.warning(f"Unknown workflow '{workflow_type}', using 'standard'")
                workflow_type = "standard"

            workflow_info = {
                "quick": "⚡ Quick sketch (~15s)",
                "standard": "🎨 Standard quality (~30s)",
                "quality": "✨ High quality (~2min)",
                "portrait": "👤 Portrait mode (~2min)",
                "landscape": "🌄 Landscape mode (~1.5min)",
                "product": "🛍️ Product hero (~45s)",
                "brand": "📣 Kindred brand social (~35s)",
                "brand-story": "📱 Kindred brand story 9:16 (~35s)",
                "brand-landscape": "🖼️ Kindred brand landscape 1200x628 (~35s)",
                "brand-banner": "🧱 Brand profile banner 4:1 (~35s)",
                "brand-product": "🛍️ Kindred brand product campaign (~35s)",
                "brand-thought": "🧠 Kindred thought leadership campaign (~35s)",
                "brand-trust": "🛡️ Kindred trust/security campaign (~35s)",
                "brand-awareness": "📢 Kindred awareness campaign (~35s)",
                "brand-consideration": "🧭 Kindred consideration campaign (~35s)",
                "brand-conversion": "✅ Kindred conversion campaign (~35s)",
                "3d": "🧊 Tencent Hunyuan 3D image-to-GLB (~2-4min)",
                "3d-mv": "🧊 Tencent Hunyuan 3D multiview image-to-GLB (~3-5min)",
                "3d-angles": "🧊 Tencent Hunyuan 3D true 4-angle image-to-GLB (~3-6min)"
            }

            await websocket.send(json.dumps({
                "type": "image_generation_started",
                "prompt": prompt,
                "workflow": workflow_type,
                "info": workflow_info.get(workflow_type, "Generating...")
            }))

            # Load workflow template
            workflow_filename = self.available_workflows[workflow_type]
            workflow_path = os.path.join(self.workflows_dir, workflow_filename)
            is_3d_workflow = workflow_type in ("3d", "3d-mv", "3d-angles")

            # Fallback to old location if workflows/ doesn't exist
            if not os.path.exists(workflow_path):
                if is_3d_workflow:
                    raise FileNotFoundError(f"Workflow not found: {workflow_path}")
                workflow_path = os.path.join(os.path.dirname(__file__), "flux_workflow.json")

            with open(workflow_path, 'r') as f:
                workflow = json.load(f)

            export_prefix = ""
            if is_3d_workflow:
                export_prefix = self._configure_3d_workflow(workflow, prompt, workflow_type, client_id)
            else:
                # Replace prompt placeholder (handle workflows with prompt augmentation)
                original_text = workflow["6"]["inputs"]["text"]
                if "PROMPT_PLACEHOLDER" in original_text:
                    workflow["6"]["inputs"]["text"] = original_text.replace("PROMPT_PLACEHOLDER", prompt)
                else:
                    workflow["6"]["inputs"]["text"] = prompt

                # Use deterministic seed presets for brand campaigns so each objective has a stable visual signature.
                if workflow_type in self.campaign_seed_presets:
                    prompt_fingerprint = int(hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()[:8], 16)
                    base_seed = self.campaign_seed_presets[workflow_type]
                    workflow["25"]["inputs"]["noise_seed"] = base_seed + (prompt_fingerprint % 100000)
                else:
                    # Non-brand modes keep randomized variety.
                    workflow["25"]["inputs"]["noise_seed"] = random.randint(0, 0xffffffffffffffff)

            # Queue the prompt
            logger.info(f"Sending workflow to ComfyUI at {self.comfyui_host}")
            response = requests.post(
                f"{self.comfyui_host}/prompt",
                json={"prompt": workflow},
                timeout=300  # 5 minutes for FLUX models
            )
            response.raise_for_status()
            result = response.json()
            prompt_id = result.get("prompt_id")

            logger.info(f"ComfyUI prompt queued with ID: {prompt_id}")

            # Poll for completion (simplified - could use WebSocket for real-time updates)
            max_wait = 420 if is_3d_workflow else 300
            for _ in range(max_wait):
                await asyncio.sleep(1)

                # Check history
                history_response = requests.get(f"{self.comfyui_host}/history/{prompt_id}")
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})

                        if is_3d_workflow:
                            glb_filename = ""
                            glb_subfolder = ""
                            for node_output in outputs.values():
                                for output_key in ("glb", "gltf", "mesh", "meshes", "files"):
                                    items = node_output.get(output_key)
                                    if isinstance(items, list) and items:
                                        first_item = items[0]
                                        if isinstance(first_item, dict):
                                            glb_filename = first_item.get("filename", "")
                                            glb_subfolder = first_item.get("subfolder", "")
                                        elif isinstance(first_item, str):
                                            glb_filename = first_item
                                        break
                                if glb_filename:
                                    break

                            if glb_filename:
                                if glb_subfolder:
                                    model_url = f"{self.comfyui_host}/view?filename={glb_filename}&subfolder={glb_subfolder}&type=output"
                                else:
                                    model_url = f"{self.comfyui_host}/view?filename={glb_filename}&type=output"
                                logger.info(f"✅ 3D model generated: {model_url}")
                                await websocket.send(json.dumps({
                                    "type": "image_generation_complete",
                                    "image_url": model_url,
                                    "message": f"🧊 3D model generated: {glb_filename}"
                                }))
                            else:
                                logger.info(f"✅ 3D workflow completed with prefix: {export_prefix}")
                                await websocket.send(json.dumps({
                                    "type": "image_generation_complete",
                                    "message": f"🧊 3D workflow complete. Check output files under prefix: {export_prefix}"
                                }))
                            return

                        # Find the SaveImage node output
                        for node_id, node_output in outputs.items():
                            if "images" in node_output:
                                images = node_output["images"]
                                if images:
                                    # Get first image
                                    image_info = images[0]
                                    filename = image_info["filename"]
                                    subfolder = image_info.get("subfolder", "")

                                    # Construct image URL
                                    if subfolder:
                                        image_url = f"{self.comfyui_host}/view?filename={filename}&subfolder={subfolder}&type=output"
                                    else:
                                        image_url = f"{self.comfyui_host}/view?filename={filename}&type=output"

                                    logger.info(f"✅ Image generated: {image_url}")

                                    await websocket.send(json.dumps({
                                        "type": "image_generation_complete",
                                        "image_url": image_url,
                                        "message": f"🎨 Image generated: {filename}"
                                    }))
                                    return

            # Timeout
            raise TimeoutError("Image generation timed out")

        except Exception as e:
            logger.error(f"Error generating image: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "content": f"Image generation failed: {str(e)}"
            }))


async def main():
    """Main entry point"""
    server = KindredAvatarServerVoice(
        host="0.0.0.0",
        port=8075,
        whisper_model="base",  # Can use: tiny, base, small, medium, large-v3
        whisper_language="en",
        kindred_model="qwen2.5:7b"  # Change to kaelen:latest, watson:latest, etc.
    )

    await server.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Jazzy Avatar Server with Voice stopped")
