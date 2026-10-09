# Security Posture

**Threat Model & Permissions**:
- **Execution Level**: ComfyUI runs within an isolated virtual environment (`.venv`), isolating destructive module updates from the main macOS `brew` or `python` systems.
- **Node Provenance**: All models and custom nodes require an automated pre-flight and sandbox test by the LLM system before being installed.
- **Network Boundaries**: ComfyUI listens exclusively on local `0.0.0.0` / `127.0.0.1` interfaces, isolating the 8188 endpoint from incoming external internet queries to prevent unverified payloads unless authenticated through the Kindred Swarm websocket protocols.
