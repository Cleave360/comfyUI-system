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

Set `ADAPTIVE_API_KEY` in `.env`. The default Adaptive API endpoint is
`http://127.0.0.1:8080`; its streaming UI uses port `8765`. Do not commit `.env`.

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

Port `8765` is reserved for the Adaptive Layer streaming UI and is not touched
by this workspace. Governed audit events go to the Adaptive API on port `8080`.

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

## Apple Silicon image benchmarks

Inspect the reproducible benchmark profiles without starting ComfyUI:

```bash
.venv/bin/python scripts/benchmark_apple_silicon.py inspect
```

Start the stack (and the Adaptive Layer at the configured `ADAPTIVE_BASE`),
then run a cold-first plus two warm renders with a deterministic seed sequence:

```bash
.venv/bin/python scripts/benchmark_apple_silicon.py run --profile interactive
.venv/bin/python scripts/benchmark_apple_silicon.py run --profile brand-social
.venv/bin/python scripts/benchmark_apple_silicon.py run --profile property-quality
.venv/bin/python scripts/benchmark_apple_silicon.py run --profile qwen-quality --timeout 1800
```

The profiles exercise 512x512 FLUX Schnell interaction latency, the 1080x1080
Kindred social workflow, a 1024x1024 FLUX Dev quality render, and the proven
1024x576/20-step Qwen-Image 2512 BF16 production route. The canonical
1328x1328/50-step Qwen template is a separate maximum-quality workload and is
not the routine benchmark.
Each prompt is admitted through the same fail-closed Adaptive governance gate
as Jazzy. JSON reports and generated benchmark images are written beneath
ignored `reports/benchmarks/` and `output/benchmarks/`. The report records the
Python, PyTorch/MPS, workflow, timing, and memory context while deliberately
excluding ComfyUI's launch arguments, which may contain secrets.

The default base seed is 360; successive runs use 361 and 362 so ComfyUI cannot
serve graph-cache hits as false warm-render measurements. Use `--prompt`,
`--seed`, `--width`, `--height`, and `--steps` only for named experiments;
retain the resulting JSON report so comparisons remain auditable.

The production environment remains pinned to PyTorch 2.5.1. A controlled
PyTorch 2.14.1 Qwen/MPS comparison produced flat grey frames and slower
generation, so `config/torch-modern-overrides.txt` is experimental evidence,
not an upgrade instruction. Do not apply it to `.venv`; see
`docs/APPLE_SILICON_BASELINE.md` for the measurements and rejection decision.

For a bounded Qwen module profile, start a separate server. The selected
diffusion forward is deliberately synchronized, so do not treat that run's
wall time as a production benchmark:

```bash
mkdir -p reports/runtime/qwen-profiler/user
KINDRED_QWEN_PROFILE_FORWARD=3 \
  .venv/bin/python scripts/profile_qwen_mps_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory reports/runtime/qwen-profiler/user
```

Run `qwen-quality` against port 8288, then inspect
`reports/profiles/qwen_mps_modules.json`. See `docs/QWEN_MPS_PROFILE.md` for the
measured hotspot ranking and proof boundaries.

To capture one real MLP block, add these variables to the isolated profiler
command (the resulting tensor file is large and Git-ignored):

```bash
KINDRED_QWEN_PROFILE_FORWARD=3 \
KINDRED_QWEN_CAPTURE_BLOCK=30 \
KINDRED_QWEN_CAPTURE_PATH=reports/captures/qwen_mlp_block30.pt \
KINDRED_QWEN_PROFILE_PATH=reports/profiles/qwen_mlp_capture_profile.json \
  .venv/bin/python scripts/profile_qwen_mps_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory "$PWD/reports/runtime/qwen-profiler/user"
```

The retained-weight MLX comparison uses the separate ignored
`reports/venvs/qwen-mlx` environment so production `.venv` stays pinned:

```bash
reports/venvs/qwen-mlx/bin/python scripts/benchmark_qwen_mlp.py \
  --capture reports/captures/qwen_mlp_block30.pt \
  --output reports/benchmarks/qwen_mlp_block30.json
```

`scripts/qwen_cfg_batch_server.py` is an experimental isolated-server harness,
not a production launch path. Its `baseline`, `padded`, and `padded_serial`
modes record transformer forward count, batch size, and context length. The
pad-to-48 result was slower and failed fixed-seed parity; do not enable it in
the normal stack. Full commands, hashes, and results are in
`docs/QWEN_MPS_PROFILE.md`.

For the larger retained-runtime experiment, use
`KINDRED_QWEN_CAPTURE_KIND=block` with
`KINDRED_QWEN_CAPTURE_BLOCK_COUNT=1` or `2`. Use `block_boundary` with a count of
60 to capture only the first input and final output, without duplicating the
model weights. Benchmark the exact ComfyUI comparator first, then supply its
report to MLX:

```bash
.venv/bin/python scripts/benchmark_qwen_block_torch.py \
  --capture reports/captures/qwen_blocks0_59_boundary.pt \
  --model models/diffusion_models/qwen_image_2512_bf16.safetensors \
  --output reports/benchmarks/qwen_blocks0_59_torch.json

reports/venvs/qwen-mlx/bin/python scripts/benchmark_qwen_block.py \
  --capture reports/captures/qwen_blocks0_59_boundary.pt \
  --torch-baseline reports/benchmarks/qwen_blocks0_59_torch.json \
  --model models/diffusion_models/qwen_image_2512_bf16.safetensors \
  --model-sha256 cbf55390fff27dbc785046d7007b04e0c5dd7421e7ef128f2831eacb53a8e075 \
  --skip-int8 --output reports/benchmarks/qwen_blocks0_59.json
```

PyTorch 2.12 reopens zero-copy Torch MPS/MLX interoperability. Keep that newer
runtime isolated from production and verify the bridge before using it:

```bash
UV_CACHE_DIR=reports/.uv-cache uv venv --python 3.12 \
  reports/venvs/qwen-mlx-torch212
UV_CACHE_DIR=reports/.uv-cache uv pip install \
  --python reports/venvs/qwen-mlx-torch212/bin/python \
  -r requirements-qwen-mlx-torch212.txt

git clone https://github.com/ml-explore/mlx-examples.git \
  reports/vendor/mlx-examples
git -C reports/vendor/mlx-examples checkout \
  796f5b53cab69a3d48a44233ce21aae889e94a08

MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_mps_mlx_zero_copy.py \
  --warmups 3 --iterations 9 --handoff-iterations 50
```

For the Qwen VAE comparison, first launch the one-shot capture server on an
experimental port and dispatch one governed `qwen-quality` render to it. Stop
the server after `reports/captures/qwen_vae_1024x576.pt` appears:

```bash
.venv/bin/python scripts/capture_qwen_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory "$PWD/reports/runtime/qwen-vae/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_apple_silicon.py run \
  --profile qwen-quality --base-url http://127.0.0.1:8288 \
  --runs 1 --seed 7460 --width 1024 --height 576 --steps 20 \
  --timeout 1800 --no-cold-first \
  --output reports/benchmarks/qwen_vae_capture_run.json
```

Then run the warm native control and complete strict-precision MLX bridge:

```bash
.venv/bin/python scripts/benchmark_qwen_vae_torch.py \
  --warmups 2 --iterations 7

MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_mlx.py --warmups 2 --iterations 7
```

The measured component decoder passed the isolated gates at 2.0676x native VAE
speed after matching ComfyUI's BF16 input boundary. It is not bit-exact.

For the end-to-end opt-in trial, create the runtime directory and launch the
isolated server with the exact eight-run mode sequence:

```bash
mkdir -p reports/runtime/qwen-mlx-vae-e2e/user
MLX_ENABLE_TF32=0 \
KINDRED_QWEN_MLX_VAE_RUN_MODES=stock,stock,stock,mlx,mlx,stock,stock,mlx \
KINDRED_QWEN_MLX_VAE_REPORT="$PWD/reports/profiles/qwen_mlx_vae_e2e_full_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/qwen_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/qwen-mlx-vae-e2e/user"
```

In another terminal, run the governed alternating comparison:

```bash
ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_qwen_mlx_vae_e2e.py \
  --base-url http://127.0.0.1:8288 --seeds 7460,7461,7462 \
  --steps 20 --cooldown-seconds 5 --timeout 1800 \
  --server-report reports/profiles/qwen_mlx_vae_e2e_full_server.json \
  --output reports/benchmarks/qwen_mlx_vae_e2e_full.json
```

The three-seed sweep passed the fixed quality and direct VAE speed gates. The
isolated requirements now install stable-ABI TorchAudio 2.11 rather than
borrowing production's incompatible binary. Keep this server experimental: it
still borrows other production dependencies read-only and does not establish a
full-workflow speedup. See `docs/QWEN_MPS_PROFILE.md` for the exact boundary.

To reproduce the temporal decoder trial, first collect a second real 20-step
endpoint with the capture server. Use an absolute capture path; the launcher
also resolves a relative path against the repository root:

```bash
mkdir -p reports/runtime/qwen-vae-video-capture/user
KINDRED_QWEN_VAE_CAPTURE="$PWD/reports/captures/qwen_vae_video_endpoint_b.pt" \
  .venv/bin/python scripts/capture_qwen_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/qwen-vae-video-capture/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_apple_silicon.py run \
  --profile qwen-quality --base-url http://127.0.0.1:8288 \
  --runs 1 --seed 7463 --width 1024 --height 576 --steps 20 \
  --timeout 1800 --no-cold-first \
  --output reports/benchmarks/qwen_vae_video_capture_run.json
```

After stopping the capture server, run the official and bounded-cache MLX
comparisons in the isolated Torch 2.12 environment:

```bash
MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 1,2,4 --warmups 1 --iterations 4 \
  --output reports/benchmarks/qwen_vae_video_full.json

MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 1,2,4 --warmups 1 --iterations 4 \
  --mlx-cache-policy eager_bounded \
  --output reports/benchmarks/qwen_vae_video_bounded.json
```

The bounded policy eliminates retained MLX allocator cache and preserves the
official MLX output exactly, but it does not reduce the measured 36.89 GiB
transient peak for thirteen frames. The experiment therefore remains rejected
under its predeclared 20 GiB memory gate.

The activation-lifetime and spatial-tail candidates use the same harness:

```bash
MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 1,2,4 --warmups 1 --iterations 4 \
  --mlx-cache-policy op_eager_bounded \
  --output reports/benchmarks/qwen_vae_video_op_eager_full.json

MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 1,2,4 --warmups 1 --iterations 4 \
  --mlx-cache-policy spatial_tiled_bounded --spatial-tile-halo 32 \
  --output reports/benchmarks/qwen_vae_video_spatial_tiled_full.json
```

Both remain experimental. The tiled path is valid only for the pinned Wan
decoder layout and fails closed if the final stage is not the expected three
attention-free residual blocks. It preserved the official MLX output exactly
and reduced thirteen-frame peak memory to 24.74 GiB, still above the unchanged
20 GiB gate.

To reproduce the per-operation attribution and the first candidate that passes
the original memory gate:

```bash
MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 4 --warmups 1 --iterations 2 \
  --mlx-cache-policy cache_consuming_chunked_resample --trace-memory \
  --output reports/benchmarks/qwen_vae_video_chunked_resample_trace.json

MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --latent-frames 1,2,4 --warmups 1 --iterations 4 \
  --mlx-cache-policy cache_consuming_chunked_resample \
  --output reports/benchmarks/qwen_vae_video_cache_chunked_full.json
```

Tracing resets peak counters around each explicit MLX evaluation and is for
attribution, not authoritative timing. The untraced full sweep is the gate
result: 19.908 GiB peak and 2.101x speedup at thirteen output frames, reproduced
independently at the same peak and 2.105x. Keep the candidate isolated until a
real video model supplies native temporal latents and the server path passes a
repeated endurance run.

To reproduce the native Wan VACE acceptance, use the isolated newer-Torch
environment. Production PyTorch 2.5.1 cannot execute UniPC's
`torch.linalg.solve` on MPS; do not patch the sampler or enable process-wide CPU
fallback. Port 8288 must be free:

```bash
mkdir -p reports/runtime/wan-native-video/user
KINDRED_WAN_VAE_CAPTURE="$PWD/reports/captures/wan_vace_1.3b_native_video_torch212.pt" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/capture_wan_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/wan-native-video/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_apple_silicon.py run \
  --profile wan-native-video --base-url http://127.0.0.1:8288 \
  --runs 1 --seed 770 --timeout 1800 --sample-interval 2 \
  --no-cold-first \
  --output reports/benchmarks/wan_native_video_torch212_capture_run.json
```

Stop the capture server before the component comparison so its loaded models
do not contaminate memory. Then run:

```bash
MLX_ENABLE_TF32=0 reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/benchmark_qwen_vae_video.py \
  --native-capture reports/captures/wan_vace_1.3b_native_video_torch212.pt \
  --weights models/vae/wan_2.1_vae.safetensors \
  --warmups 1 --iterations 4 \
  --mlx-cache-policy cache_consuming_chunked_resample --trace-memory \
  --output reports/benchmarks/wan_vae_native_video_cache_chunked.json
```

This passed the native-latent component gates twice at 2.434x and 2.666x, with
the same 13.596 GiB MLX peak, identical output hashes, and zero retained MLX
cache. Production remains pinned to PyTorch 2.5.1 pending full dependency,
audio, image-workflow, and repeated-server validation of a newer runtime.

For the governed alternating full-stack trial, start the isolated server with
the exact ten-run sequence. This includes two warm-ups, two stock controls, and
three alternating stock/MLX pairs:

```bash
mkdir -p reports/runtime/wan-mlx-vae-e2e/user
MLX_ENABLE_TF32=0 \
KINDRED_WAN_MLX_VAE_RUN_MODES=stock,mlx,stock,stock,stock,mlx,mlx,stock,stock,mlx \
KINDRED_WAN_MLX_VAE_REPORT="$PWD/reports/profiles/wan_mlx_vae_e2e_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/wan_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/wan-mlx-vae-e2e/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_wan_mlx_vae_e2e.py \
  --base-url http://127.0.0.1:8288 --seeds 770,771,772 \
  --steps 4 --cooldown-seconds 2 --timeout 1800 --sample-interval 2 \
  --server-report reports/profiles/wan_mlx_vae_e2e_server.json \
  --output reports/benchmarks/wan_mlx_vae_e2e_full.json
```

The first acceptance passed every fixed gate, including exact control,
same-seed latent hashes, all-frame and temporal quality, three-of-three
end-to-end wins, 13.596 GiB peak, and zero retained MLX cache. Timing became
strongly non-stationary under the shared machine load, so use the 4.011-second
least-contended pair saving rather than the inflated 72.273-second median as
the practical reference. Stop port 8288 after the run; this remains an isolated
experimental server. Audio nodes require the current pinned TorchAudio 2.11.

To reproduce the endurance diagnosis, use eight identical modes, then run two
warm-ups and six fixed-seed measurements. The passing isolated candidate uses
the MLX decoder and clears only Torch's reclaimable MPS cache after each decode:

```bash
mkdir -p reports/runtime/wan-sampler-mlx-empty-cache/user
MLX_ENABLE_TF32=0 \
KINDRED_WAN_MLX_VAE_RUN_MODES=mlx,mlx,mlx,mlx,mlx,mlx,mlx,mlx \
KINDRED_WAN_MPS_CLEANUP_POLICY=torch_empty_cache \
KINDRED_WAN_MLX_VAE_REPORT="$PWD/reports/profiles/wan_sampler_mlx_empty_cache_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/wan_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/wan-sampler-mlx-empty-cache/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_wan_mps_sampler_endurance.py \
  --base-url http://127.0.0.1:8288 --seed 773 \
  --warmups 2 --runs 6 --steps 4 --cooldown-seconds 2 \
  --timeout 1800 --sample-interval 2 \
  --server-report reports/profiles/wan_sampler_mlx_empty_cache_server.json \
  --output reports/benchmarks/wan_sampler_endurance_mlx_empty_cache.json
```

That candidate passed the fixed stability and memory-growth gates with a
24.129-second median total, 1.165 total p95/median, and 56 MiB Torch driver
growth. For diagnosis, set all modes to `stock`, set the cleanup policy to
`none` or `torch_empty_cache`, or use a 30-second cooldown. The complete
five-configuration comparison and evidence hashes are in `docs/QWEN_MPS_PROFILE.md`.
This is not the production startup path.

To reproduce the longer mixed-workload soak, configure seven MLX Wan decodes:
one video warm-up and six measured videos. The client first runs a governed
audio smoke, then alternates full 20-step Qwen images and four-step Wan videos:

```bash
mkdir -p reports/runtime/mixed-image-video-soak/user
MLX_ENABLE_TF32=0 \
KINDRED_WAN_MLX_VAE_RUN_MODES=mlx,mlx,mlx,mlx,mlx,mlx,mlx \
KINDRED_WAN_MPS_CLEANUP_POLICY=torch_empty_cache \
KINDRED_WAN_MLX_VAE_REPORT="$PWD/reports/profiles/mixed_image_video_soak_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/wan_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/mixed-image-video-soak/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_mixed_image_video_soak.py \
  --base-url http://127.0.0.1:8288 --pairs 6 \
  --qwen-steps 20 --wan-steps 4 --cooldown-seconds 2 \
  --timeout 1800 --sample-interval 2 \
  --server-report reports/profiles/mixed_image_video_soak_server.json \
  --output reports/benchmarks/mixed_image_video_soak.json
```

The first soak passed audio, output determinism, drift, memory, and accounting
gates but failed p95 timing stability after a large second-pair excursion. Use
`--reanalyze-only` to recompute decoded-pixel hashes and gates from retained
outputs without rerunning the workflows. Production remains unchanged.

To reproduce the follow-up process-isolation experiment, run one persistent
worker per workload. The Qwen worker clears reclaimable Torch MPS cache after
each image decode; the Wan worker retains the passing MLX VAE plus post-decode
cache boundary:

```bash
mkdir -p reports/runtime/isolated-soak-qwen/user \
  reports/runtime/isolated-soak-wan/user

MLX_ENABLE_TF32=0 \
KINDRED_WAN_MLX_VAE_RUN_MODES=mlx \
KINDRED_WAN_MPS_CLEANUP_POLICY=none \
KINDRED_TORCH_EMPTY_CACHE_ALL_VAE=1 \
KINDRED_WAN_MLX_VAE_REPORT="$PWD/reports/profiles/isolated_soak_qwen_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/wan_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/isolated-soak-qwen/user"

MLX_ENABLE_TF32=0 \
KINDRED_WAN_MLX_VAE_RUN_MODES=mlx,mlx,mlx,mlx,mlx,mlx,mlx \
KINDRED_WAN_MPS_CLEANUP_POLICY=torch_empty_cache \
KINDRED_TORCH_EMPTY_CACHE_ALL_VAE=0 \
KINDRED_WAN_MLX_VAE_REPORT="$PWD/reports/profiles/isolated_soak_wan_server.json" \
  reports/venvs/qwen-mlx-torch212/bin/python \
  scripts/wan_mlx_vae_server.py \
  --listen 127.0.0.1 --port 8289 --disable-all-custom-nodes \
  --disable-auto-launch \
  --user-directory "$PWD/reports/runtime/isolated-soak-wan/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_isolated_image_video_soak.py \
  --image-url http://127.0.0.1:8288 --video-url http://127.0.0.1:8289 \
  --pairs 6 --qwen-steps 20 --wan-steps 4 --cooldown-seconds 2 \
  --timeout 1800 --sample-interval 2 \
  --image-server-report reports/profiles/isolated_soak_qwen_server.json \
  --video-server-report reports/profiles/isolated_soak_wan_server.json \
  --baseline-report reports/benchmarks/mixed_image_video_soak.json \
  --output reports/benchmarks/isolated_image_video_soak.json
```

The first complete run rejected process isolation as a promotion candidate.
Outputs remained valid and deterministic, but Qwen total p95/median was 2.373,
Wan total p95/median was 1.401, and Qwen Torch driver allocation grew by
4.141 GiB. The second measured Qwen render took 318.972 seconds before later
runs stabilized near 134 seconds. Stop both experimental ports after capture;
production ports 8188, 8075, and 8070 are not part of this experiment.

### Live GPU experiment dashboard and recovery policy

The read-only dashboard displays workflow duration, probe latency, available
unified memory, recovery waits, current run state, and measured outputs/hour.
It binds only to loopback, scans supported JSON reports every two seconds, and
does not expose process arguments or arbitrary report fields:

```bash
.venv/bin/python scripts/gpu_metrics_gui.py \
  --listen 127.0.0.1 --port 8090 \
  --report-dir reports/benchmarks
```

Open `http://127.0.0.1:8090`. A single report can be selected with repeated
`--report` arguments instead of directory discovery.

The recovery-policy screen compares fixed 2, 10, and 30-second gaps against an
adaptive MPS readiness gate. Each policy gets a fresh isolated Torch 2.12/MLX
server on port 8288. After two warm-ups it measures four identical Qwen/Wan
pairs. A retained 4096x4096 FP16 matrix multiplication is sampled every two
seconds; the adaptive policy requires three consecutive samples below a robust
idle threshold before dispatch. Fixed policies run the same probe so probe
overhead is comparable, but never let it shorten their configured delay.

Set `ADAPTIVE_BASE` and `ADAPTIVE_API_KEY` in the ignored root `.env`, start
the dashboard, then run:

```bash
.venv/bin/python scripts/run_recovery_policy_matrix.py \
  --pairs 4 --qwen-steps 20 --wan-steps 4 --port 8288
```

Reports are written incrementally beneath `reports/benchmarks/`. The matrix
runner records failures and always stops its owned server between policies.
The initial order is a screening design and is explicitly recorded as an order
confound. A passing adaptive result must be repeated in a counterbalanced order
before promotion. This experiment does not change the production server.

The first four-policy screen completed all 40 governed workflows. Adaptive
passed every predeclared gate at 30.582 outputs/hour, 1.033 Qwen p95/median,
1.001 Wan p95/median, and a 4.091-second median recovery wait. Fixed 10 seconds
ranked fractionally higher at 30.763 outputs/hour; fixed 30 seconds produced the
lowest raw render times but only 29.348 outputs/hour after its waits. Fixed 2
seconds was unstable at 27.121 outputs/hour and 1.443 Qwen p95/median. Treat
this as positive screening evidence for recovery-aware dispatch, not promotion
proof, because each arm had only four measured pairs and policy order was not
counterbalanced.

New reports also record Apple's public `ProcessInfo.thermalState` as nominal,
fair, serious, or critical, plus Low Power Mode. The dashboard treats serious
as the performance-impact boundary and critical as significant impact. This is
system thermal pressure, not a GPU-only thermometer. Apple provides no supported
GPU junction-temperature or fixed throttle-temperature API, so the report and
GUI leave both Celsius fields explicitly unavailable instead of guessing.

To confirm the close adaptive-versus-fixed-10s result without the original arm
order, run the predeclared AB/BA experiment:

```bash
.venv/bin/python scripts/run_counterbalanced_recovery.py \
  --pairs 4 --qwen-steps 20 --wan-steps 4 --port 8288
```

Round one runs fixed 10 seconds then adaptive; round two reverses the order.
Every arm gets a fresh server and the same seeds. The aggregate comparison
requires valid deterministic outputs, no serious/critical thermal sample,
100% adaptive readiness, adaptive throughput within 2% of fixed 10 seconds,
a shorter adaptive wait, adaptive per-workload p95/median at most 1.35, and no
more than 1.10x spread between the two round-specific throughput ratios.

The 2026-10-12 confirmation completed all 40 workflows, but did **not** pass
the complete promotion gate. Pooled adaptive throughput was 30.423
outputs/hour versus 27.977 for fixed 10 seconds (1.087x), and its median wait
fell from 10.032 to 4.089 seconds. All outputs were valid and deterministic,
adaptive readiness was 100%, adaptive Qwen and Wan p95/median were 1.033 and
1.000, and no sample reached Apple's `serious` or `critical` thermal state.
The round-specific throughput ratios were 1.145 and 1.029, however, giving a
1.113 order-interaction spread against the predeclared 1.10 maximum. Keep
adaptive experimental: the result supports its throughput and wait-time value,
but not an order-independent production promotion. The next confirmation
should use more randomized/repeated blocks rather than changing this gate.

The follow-up is frozen as six balanced randomized paired blocks. A recorded
seed shuffles three fixed-first and three adaptive-first blocks. Every arm gets
a fresh server, two warm-ups, and two measured Qwen/Wan pairs:

```bash
.venv/bin/python scripts/run_randomized_recovery_blocks.py \
  --qwen-steps 20 --wan-steps 4 --port 8288 --random-seed 20261012
```

This is a roughly multi-hour experiment producing 12 arm reports plus a
manifest and comparison beneath ignored `reports/`. In addition to the earlier
gates, adaptive must be non-inferior within 2% in both arm-order strata and
both chronological halves. Order-stratum and time-half geometric-mean ratio
spreads must each be at most 1.10. The runner preserves a failed comparison as
evidence and always stops its owned experimental server.

The seeded 2026-10-12 run completed all 72 workflows and all 48 measured
outputs were valid and deterministic. Adaptive pooled 27.336 outputs/hour
versus 25.886 for fixed 10 seconds (1.056x), while median recovery wait fell
from 10.032 to 4.092 seconds. Adaptive Qwen and Wan p95/median were 1.053 and
1.025, readiness was 100%, and no sample reached `serious` or `critical`.

The overall gate nevertheless failed. Adaptive/fixed block ratios ranged from
0.931 to 1.304. Fixed-first and adaptive-first geometric means were 1.126 and
0.982, producing a 1.147 order spread; first-half and second-half means were
1.120 and 0.988, producing a 1.134 temporal spread. Both exceeded 1.10. Keep
adaptive experimental: it showed better pooled throughput, shorter waits, and
flatter tails, but not an order- and time-independent advantage.

The MLX-examples follow-up retained a read-only residual probe and a deliberately
isolated cache experiment. Start the probe on a free experimental port, then run
one governed `qwen-quality` render against it:

```bash
KINDRED_QWEN_RESIDUAL_REPORT=reports/profiles/qwen_residual_probe.json \
  .venv/bin/python scripts/qwen_residual_probe_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory "$PWD/reports/runtime/qwen-residual/user"
```

The quality-changing cache harness requires explicit sigma values. The only
measured candidate so far is the conservative single-step trial below:

```bash
KINDRED_QWEN_CACHE_SKIP_SIGMAS=0.2562 \
KINDRED_QWEN_CACHE_REPORT=reports/profiles/qwen_residual_cache_one_step.json \
  .venv/bin/python scripts/qwen_residual_cache_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory "$PWD/reports/runtime/qwen-residual/user"
```

Do not substitute this harness for the normal ComfyUI server. It deliberately
changes the numerical output. The initial two-image check and subsequent failed
promotion sweep are documented in `docs/QWEN_MPS_PROFILE.md`.

For a controlled stock/cache sweep, provide the exact run-mode sequence when
starting the isolated server. The benchmark inserts a latent pass-through nonce
to force KSampler execution while retaining model and conditioning caches:

```bash
KINDRED_QWEN_CACHE_SKIP_SIGMAS=0.2562 \
KINDRED_QWEN_CACHE_RUN_MODES=stock,stock,stock,cache,cache,stock,stock,cache,cache,stock \
KINDRED_QWEN_CACHE_REPORT=reports/profiles/qwen_residual_cache_sweep_server.json \
  .venv/bin/python scripts/qwen_residual_cache_server.py \
  --listen 127.0.0.1 --port 8288 --disable-all-custom-nodes \
  --disable-auto-launch --user-directory "$PWD/reports/runtime/qwen-residual-sweep/user"

ADAPTIVE_BASE=http://127.0.0.1:8080 ADAPTIVE_API_KEY=YOUR_LOCAL_KEY \
  .venv/bin/python scripts/benchmark_qwen_residual_cache.py \
  --base-url http://127.0.0.1:8288 --seeds 5400,5401,5402,5403 \
  --cooldown-seconds 60 \
  --cache-report reports/profiles/qwen_residual_cache_sweep_server.json \
  --output reports/benchmarks/qwen_residual_cache_sweep.json
```

The measured sweep failed its predeclared quality gate and is retained as
negative evidence. Do not enable residual reuse in the production server.

The full-stack MLX result is diagnostic and rejected for production; see the
profile document for its performance and accumulated-error boundaries.

Check listening ports after startup:

```bash
lsof -nP -iTCP:8188 -iTCP:8070 -iTCP:8075 -sTCP:LISTEN
```

If ComfyUI starts but a workflow fails, confirm that it uses FP16 or BF16 weights. FP8 model compatibility is not guaranteed on MPS.
