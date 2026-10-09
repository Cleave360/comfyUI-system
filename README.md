# Kindred ComfyUI System

Local Apple Silicon creative-generation orchestration for ComfyUI, the Jazzy
voice/avatar interface, and Kindred image, video, brand, and 3D workflows.

## Runtime layout

- `ComfyUI-source/` — local upstream ComfyUI checkout, intentionally not vendored
- `kindred-avatar/` — WebSocket voice backend and browser frontend
- `ComfyUI-workflows/` — portable API and UI workflow definitions
- `user/default/workflows/` — user-facing ComfyUI workflow library
- `custom_nodes/` — workspace-owned custom nodes
- `.venv/` — the single local Python 3.12 environment, ignored by Git

Models, inputs, outputs, logs, credentials, local user state, and agent
transcripts are intentionally excluded from version control.

## Quick start

See [how_to_run.md](how_to_run.md) for environment setup, service startup,
tests, MPS verification, and endpoint checks.

The standard local endpoints are:

- ComfyUI: `http://127.0.0.1:8188`
- Jazzy frontend: `http://127.0.0.1:8070/index_voice.html`
- Jazzy WebSocket backend: `ws://127.0.0.1:8075`

Port `8765` is reserved for the Adaptive Layer.

## Governance

Repository identity and execution policy are defined by `governance.toml` and
`governance.policy.toml`. Local `.kindred/` audit and execution evidence is
append-only and must never be committed.
