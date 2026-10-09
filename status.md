# Status

## Current verified state — 2026-10-09

- One Python 3.12 environment at `.venv`; dependency lock and `pip check` pass.
- ComfyUI and ComfyUI-Manager are pinned by commit in `config/system.toml`.
- Workspace-owned custom nodes are tracked and linked into the upstream checkout.
- MPS tensor execution and live ComfyUI `darwin / mps` device reporting pass.
- Managed endpoints are ComfyUI 8188, Jazzy WebSocket 8075, and frontend 8070.
  Adaptive remains independent on 8765.
- Jazzy workflow dispatch is fail-closed on Adaptive start-audit acceptance.
- Repository tests: 8 passed. Pinned upstream tests: 308 passed, 1 skipped,
  1 known FP8-on-CPU test deselected.
- Model lock: 53 artifacts / 466,485,856,755 bytes with SHA-256 hashes.

The services were stopped cleanly after live verification; this file does not
claim they are currently running. Model MPS compatibility and licences remain
artifact-specific and are not inferred from basic runtime proof.
