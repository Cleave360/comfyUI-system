# Architecture and trust boundaries

The outer repository owns reproducible orchestration, Jazzy, portable workflows,
custom nodes, governance integration, and operational documentation. It pins but
does not vendor the upstream ComfyUI and ComfyUI-Manager Git checkouts.

```text
Browser :8070
   │ WebSocket :8075 (origin limit; token required off-loopback)
   ▼
Jazzy voice/chat backend ───────► Ollama or Anthropic
   │
   │ complete execution envelope + accepted start audit
   ▼
Adaptive Layer :8765
   │ acceptance
   ▼
ComfyUI :8188 ───────► local models / input / output / workflows
   │
   └── terminal audit to Adaptive; durable local replay evidence on failure
```

The accepted start event is an admission boundary, not a sandbox. ComfyUI and
third-party nodes execute as the local user. All public-facing deployment needs
an authenticated reverse proxy, TLS, containment, secret management, and an
explicit egress policy beyond what this local stack supplies.

## Durable and generated state

- Git: orchestration code, configs, owned nodes, workflows, tests, docs.
- Nested checkouts: pinned upstream source, ignored by the outer repository.
- `.kindred/`: append-only local governance evidence, never committed.
- `.runtime/`: process state, never committed.
- `models/`, `input/`, `output/`, `logs/`: local artifacts, never committed.
- `config/models.lock.json`: content inventory only; binaries remain local.

## Proof boundaries

Repository tests establish config, gate, manifest, and lifecycle behavior at the
component level. A live smoke establishes local endpoint readiness. Neither is
proof of internet-safe deployment, custom-node isolation, model licensing, or
every workflow's MPS compatibility.
