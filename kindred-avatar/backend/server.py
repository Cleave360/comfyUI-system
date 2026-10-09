"""
Kindred Avatar WebSocket Server

Connects the 3D avatar frontend to:
- Ollama (for Kindred AI responses)
- ComfyUI (for image generation)
- TTS (text-to-speech for voice)
"""

import asyncio
import json
import websockets
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KindredAvatarServer:
    def __init__(
        self,
        ollama_host="http://localhost:11434",
        comfyui_host="http://127.0.0.1:8188",
        kindred_model="qwen2.5:7b"
    ):
        self.ollama_host = ollama_host
        self.comfyui_host = comfyui_host
        self.kindred_model = kindred_model
        self.active_connections = set()

    async def handle_client(self, websocket):
        """Handle a WebSocket client connection"""
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

        try:
            async for message in websocket:
                data = json.loads(message)
                await self.process_message(websocket, data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")
        finally:
            self.active_connections.remove(websocket)

    async def process_message(self, websocket, data):
        """Process incoming messages from the avatar"""
        msg_type = data.get("type")

        if msg_type == "chat":
            # User sent a text message to Kindred
            await self.handle_chat(websocket, data.get("message", ""))

        elif msg_type == "generate_image":
            # User requested image generation
            await self.handle_image_generation(websocket, data.get("prompt", ""))

        elif msg_type == "ping":
            # Health check
            await websocket.send(json.dumps({"type": "pong"}))

    async def handle_chat(self, websocket, user_message):
        """Send message to Kindred (Ollama) and stream response"""
        try:
            # Send "thinking" indicator
            await websocket.send(json.dumps({
                "type": "status",
                "status": "thinking"
            }))

            # System prompt for Kindred persona
            system_prompt = """You are Kindred, a creative and thoughtful AI assistant who helps with image generation and creative projects. You are embodied in a 3D avatar and can see, create, and interact with visual content. Be concise, friendly, and helpful. When appropriate, suggest generating images to visualize ideas."""

            # Call Ollama API
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.kindred_model,
                "prompt": user_message,
                "system": system_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 256,
                }
            }

            response_text = ""

            # Stream the response
            response = requests.post(url, json=payload, stream=True, timeout=120)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    response_text += token

                    # Stream tokens back to frontend
                    await websocket.send(json.dumps({
                        "type": "chat_token",
                        "token": token,
                        "full_text": response_text
                    }))

                    if chunk.get("done", False):
                        break

            # Send complete message
            await websocket.send(json.dumps({
                "type": "chat_complete",
                "message": response_text
            }))

            logger.info(f"Kindred response: {response_text[:100]}...")

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Chat error: {str(e)}"
            }))

    async def handle_image_generation(self, websocket, prompt):
        """Trigger ComfyUI image generation"""
        try:
            await websocket.send(json.dumps({
                "type": "status",
                "status": "generating_image",
                "message": f"Generating: {prompt[:50]}..."
            }))

            # For now, just acknowledge
            # TODO: Implement actual ComfyUI workflow trigger
            await asyncio.sleep(2)  # Simulate generation time

            await websocket.send(json.dumps({
                "type": "image_generated",
                "message": "Image generation would happen here",
                "image_url": "/placeholder.jpg"
            }))

            logger.info(f"Image generation requested: {prompt}")

        except Exception as e:
            logger.error(f"Error in image generation: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Generation error: {str(e)}"
            }))

    async def start(self, host="localhost", port=8075):
        """Start the WebSocket server"""
        logger.info(f"Starting Kindred Avatar Server on ws://{host}:{port}")
        logger.info(f"Using Ollama model: {self.kindred_model}")
        logger.info(f"Ollama host: {self.ollama_host}")
        logger.info(f"ComfyUI host: {self.comfyui_host}")

        async with websockets.serve(self.handle_client, host, port):
            await asyncio.Future()  # Run forever


async def main():
    server = KindredAvatarServer(
        ollama_host="http://localhost:11434",
        comfyui_host="http://127.0.0.1:8188",
        kindred_model="qwen2.5:7b"  # Use your preferred Kindred model
    )
    await server.start(host="localhost", port=8075)


if __name__ == "__main__":
    asyncio.run(main())
