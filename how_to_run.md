# Running the Kindred ComfyUI Stack

This workspace uses one canonical virtual environment at `.venv` with Python 3.12. Do not create environments under `ComfyUI-source/` or `kindred-avatar/backend/`. Model weights, inputs, outputs, and runtime user data remain local and are intentionally excluded from Git.

## One-time setup

The orchestration repository does not vendor the large upstream ComfyUI checkout.
After cloning this repository, create it once:

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git ComfyUI-source
```

From the repository root, create the shared environment and install both
ComfyUI and avatar dependencies:

```bash
cd ~/Documents/ComfyUI
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
```

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

The first voice-backend start downloads and warms the Whisper `base` model
(about 145 MB). ChromaDB memory is optional; the backend starts without it.

## Start components separately

ComfyUI only:

```bash
cd ~/Documents/ComfyUI
./ComfyUI-source/start.sh
```

Avatar voice backend and frontend only:

```bash
cd ~/Documents/ComfyUI
./kindred-avatar/start_voice.sh
```

## Verification

Check dependencies and tests:

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest ComfyUI-source/tests-unit -q
```

Verify Apple Metal execution:

```bash
.venv/bin/python -c 'import torch; assert torch.backends.mps.is_available(); x=torch.arange(6, dtype=torch.float32, device="mps").reshape(2,3); print((x @ x.T).cpu())'
```

Check listening ports after startup:

```bash
lsof -nP -iTCP:8188 -iTCP:8070 -iTCP:8075 -sTCP:LISTEN
```

If ComfyUI starts but a workflow fails, confirm that it uses FP16 or BF16 weights. FP8 model compatibility is not guaranteed on MPS.
