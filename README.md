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

```bash
./scripts/bootstrap.py --repair-links
cp .env.example .env
# Set ADAPTIVE_API_KEY in .env, then:
./start_all.sh
```

See [how_to_run.md](how_to_run.md) for the complete setup, lifecycle,
governance, tests, MPS verification, and endpoint checks.

The standard local endpoints are:

- ComfyUI: `http://127.0.0.1:8188`
- Jazzy frontend: `http://127.0.0.1:8070/index_voice.html`
- Jazzy WebSocket backend: `ws://127.0.0.1:8075`

The Adaptive API runs on port `8080`; port `8765` is reserved for its streaming
UI. Neither port is owned by this workspace.

## Governance

Repository identity and execution policy are defined by `governance.toml` and
`governance.policy.toml`. Local `.kindred/` audit and execution evidence is
append-only and must never be committed.

Every ComfyUI workflow submitted by Jazzy is fail-closed on an accepted
Adaptive `comfy.workflow.start` audit event. Chat and health checks remain
available when Adaptive is offline, but workflow dispatch does not.

## Repository contracts

- Pinned upstream revisions and runtime layout: `config/system.toml`
- Local model inventory: `config/models.lock.json`
- Components and trust boundaries: `docs/ARCHITECTURE.md`
- Reproducible Apple Silicon baseline: `docs/APPLE_SILICON_BASELINE.md`
- Security policy and reporting: `SECURITY.md`
- Contribution and release process: `CONTRIBUTING.md` and `docs/RELEASING.md`
