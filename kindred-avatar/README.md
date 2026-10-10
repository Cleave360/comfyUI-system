# Jazzy Avatar

Jazzy is the browser, WebSocket, voice, chat, and governed ComfyUI workflow
surface for this repository. The canonical operational guide is the root
[`how_to_run.md`](../how_to_run.md).

## Run

From the repository root:

```bash
./scripts/bootstrap.py --repair-links
cp .env.example .env
# Set ADAPTIVE_API_KEY in .env
./start_all.sh
```

Open `http://127.0.0.1:8070/index_voice.html`. Use `./status_all.sh` and
`./stop_all.sh` for lifecycle management. The managed backend listens on
`ws://127.0.0.1:8075`; Adaptive uses port 8080 for its API and 8765 for its
streaming UI.

## Features

- WhisperLiveKit voice input and wake-word handling
- Streaming Ollama or Anthropic chat
- Image, brand, video, and Hunyuan 3D workflow dispatch through ComfyUI
- Adaptive context envelope and accepted-start audit gate before dispatch
- Optional ChromaDB memory
- Three.js sphere fallback when no licensed GLTF asset is installed

## Configuration and security

Use the root `.env`, based on `.env.example`. Services bind to loopback by
default. A non-loopback `JAZZY_HOST` requires `JAZZY_WS_TOKEN`; open the browser
with `?token=...`. `JAZZY_ALLOWED_ORIGINS` and `JAZZY_MAX_MESSAGE_BYTES` further
bound the WebSocket surface.

Image and 3D generation requires Adaptive at `ADAPTIVE_BASE` with a valid
`ADAPTIVE_API_KEY`. Agent or service principals also require `JAZZY_LEASE_ID`.
The dispatch fails closed if the start audit append isn't accepted.

## Avatar asset

No redistributable robot model is included. The UI intentionally falls back to
an animated sphere. To use a custom avatar, place a licensed `robot.gltf` and
its referenced assets in `frontend/` locally. Those paths remain ignored until
their provenance and redistribution rights are documented.

## Troubleshooting

- Inspect `logs/jazzy-backend.log` and run `./status_all.sh`.
- Confirm Ollama is running when `JAZZY_ASSISTANT_PROVIDER=ollama`.
- Confirm Adaptive credentials before testing generation.
- Serve the frontend through the managed HTTP server; don't open it as `file://`.

This component is covered by the repository MIT licence.
