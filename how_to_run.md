# Running the Kindred ComfyUI Stack

This workspace uses one canonical virtual environment at `.venv` with Python 3.12. Do not create environments under `ComfyUI-source/` or `kindred-avatar/backend/`. Model weights, inputs, outputs, and runtime user data remain local and are intentionally excluded from Git.

## One-time setup

The bootstrap script checks out the exact ComfyUI and ComfyUI-Manager revisions
from `config/system.toml`, creates the single environment, installs dependencies,
and creates safe links to workspace models, inputs, outputs, user workflows, and
owned custom nodes. It refuses to replace non-equivalent data or update a dirty
upstream checkout.

`requirements-dev.txt` records the dependency inputs;
`requirements-lock.txt` pins the verified macOS/Python 3.12 environment used by
bootstrap. Refresh the lock only after dependency, test, and MPS validation.

```bash
cd ~/Documents/ComfyUI
./scripts/bootstrap.py --repair-links
cp .env.example .env
```

Set `ADAPTIVE_API_KEY` in `.env`. The default Adaptive endpoint is
`http://127.0.0.1:8765`. Do not commit `.env`.

`pyexpat` is part of Python's standard library; it is not a separate project dependency. Verify it with:

```bash
.venv/bin/python -c 'import xml.parsers.expat as expat; print(expat.EXPAT_VERSION)'
```

## Start the complete stack

```bash
cd ~/Documents/ComfyUI
./start_all.sh
```

The launcher uses only the root `.venv` and starts:

- ComfyUI at `http://127.0.0.1:8188`
- Jazzy avatar frontend at `http://127.0.0.1:8070/index_voice.html`
- Jazzy WebSocket backend at `ws://127.0.0.1:8075`

Port `8765` is reserved for the Adaptive Layer and is not touched by this workspace.

Logs are written under `logs/` and are intentionally ignored by Git.
Process ownership is recorded under `.runtime/`; the launcher never kills an
unrelated process merely because it owns a configured port. Startup fails and
rolls back its own processes if any service misses its readiness check.

The first voice-backend start downloads and warms the Whisper `base` model
(about 145 MB). ChromaDB memory is optional; the backend starts without it.

## Lifecycle commands

```bash
./status_all.sh
./stop_all.sh
./scripts/stack_manager.py restart
```

If a configured port is occupied, identify its owner yourself; the manager will
report the conflict without terminating it.

## Network and governance

All services bind to loopback by default. If `JAZZY_HOST` is changed to a
non-loopback address, `JAZZY_WS_TOKEN` is mandatory and the browser URL must
include it:

```bash
http://host:8070/index_voice.html?token=YOUR_TOKEN
```

`JAZZY_ALLOWED_ORIGINS` limits browser origins and
`JAZZY_MAX_MESSAGE_BYTES` bounds WebSocket messages. Workflow execution uses
the context in `governance.toml`. Human-local dispatch is the default; agent or
service principals must also set `JAZZY_LEASE_ID`. If Adaptive does not accept
the start event, the workflow is not sent to ComfyUI. Local append-only replay
evidence is written beneath ignored `.kindred/audit/`.

## Verification

Check dependencies and tests:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
./scripts/bootstrap.py --check --skip-deps
.venv/bin/python scripts/model_manifest.py check
```

To refresh content hashes after model changes, run
`.venv/bin/python scripts/model_manifest.py scan --hash`. The lock also reports
interrupted cache downloads; delete or resume them deliberately rather than
treating them as usable weights. Unknown source/licence values require review
before a model is redistributed.

The pinned upstream suite can be run separately with:

```bash
.venv/bin/python -m pytest ComfyUI-source/tests-unit -q -k 'not mixed_precision_load'
```

The excluded upstream FP8 CPU test calls an operator unavailable on CPU; it is
not evidence of an MPS workflow failure.

Verify Apple Metal execution:

```bash
.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(); x=torch.arange(6, dtype=torch.float32, device="mps").reshape(2,3); print((x @ x.T).cpu())'
```

Check listening ports after startup:

```bash
lsof -nP -iTCP:8188 -iTCP:8070 -iTCP:8075 -sTCP:LISTEN
```

If ComfyUI starts but a workflow fails, confirm that it uses FP16 or BF16 weights. FP8 model compatibility is not guaranteed on MPS.
