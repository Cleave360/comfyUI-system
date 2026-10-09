# Status

**Build State**:
- **ComfyUI Core**: Installed locally in `ComfyUI-source` and run through the single workspace environment at `.venv` (Python 3.12).
- **Hardware Integration**: Metal Performance Shaders (MPS) verified on 2026-10-09 using the canonical Python 3.12 environment and PyTorch 2.5.1. `torch.backends.mps.is_available()` returned `True`, and a matrix multiplication completed on an `mps` tensor with the expected result. This verifies local PyTorch MPS execution; it does not prove every model or custom node is MPS-compatible.
- **Model Arsenal**: FP16 FLUX.1 [schnell/dev], Qwen BF16 variants, and core CLIP encoders properly placed and symbolic links set up.
- **Manager**: ComfyUI-Manager installed; debugging cache/server fault.

**Configured Runtime Components**:
- ComfyUI Server (`http://127.0.0.1:8188`)
- Jazzy frontend (`http://127.0.0.1:8070/index_voice.html`)
- Jazzy WebSocket backend (`ws://127.0.0.1:8075`)
- Port `8765` is reserved for the Adaptive Layer.

These are configured endpoints, not a claim that the services are currently running. Pre-flight scripts, the model puller, and memory monitoring are scaffolded locally.
