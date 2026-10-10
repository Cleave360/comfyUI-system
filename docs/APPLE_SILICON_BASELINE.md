# Apple Silicon Image Baseline

Measured on 2026-10-10 using the governed benchmark harness in
`scripts/benchmark_apple_silicon.py`.

## Runtime

- Apple M3 Ultra, 512 GiB unified memory
- macOS 26.2, arm64
- Python 3.12.11
- PyTorch 2.5.1 with MPS built and available
- ComfyUI 0.7.0
- One cold model load followed by two warm generations
- Deterministic seeds 360, 361, and 362; changing the seed forces real graph
  execution while preserving repeatability across environments
- Every prompt admitted by Adaptive at `127.0.0.1:8080` before ComfyUI queueing

## Results

| Profile | Workload | Cold | Warm 1 | Warm 2 | Warm median |
|---|---|---:|---:|---:|---:|
| `interactive` | FLUX Schnell, 512x512, 2 steps | 24.159 s | 3.025 s | 3.019 s | 3.022 s |
| `brand-social` | FLUX Schnell, 1080x1080, 6 steps | 46.281 s | 25.162 s | 26.172 s | 25.667 s |
| `property-quality` | FLUX Dev, 1024x1024, 20 steps | 92.612 s | 69.450 s | 76.484 s | 72.967 s |
| `qwen-quality` | Qwen-Image 2512 BF16, 1024x576, 20 steps | 113.729 s | 72.451 s | 72.458 s | 72.454 s |

The first experimental pass reused seed 360 for every run. ComfyUI correctly
served the second and third prompts from its graph cache, producing roughly
one-second completion times. Those values are cache latency, not render
performance, and are excluded from this baseline. The harness now advances the
seed for every timed run.

## Evidence boundaries

The JSON reports and PNG outputs remain in ignored local directories:
`reports/benchmarks/` and `output/benchmarks/`. Their report SHA-256 values are:

- interactive: `2483822bd71177873b98678a5b7ce1c1bf6900c472a5080f13ceaf91adc71272`
- brand-social: `dc97c5d77f4e75c40c739bf7811154f750822c31aecec0930fddb39fd0c9d9f8`
- property-quality: `2bafcc71852179c675886abdefe2bb3b1ac16b481901835d2920bc6ca8fbf94f`
- qwen-quality: `4517e3ef2c4e2cf5ee711d7556dbaead4bbf3d16ca42f4989ea8c22608e8e319`

A separate maximum-quality probe used the canonical 1328x1328/50-step Qwen
template. Its first cold image completed successfully in 1,208.675 seconds
(20:08), including 19:01 of sampling at roughly 22.8 seconds per step. The
resulting PNG has SHA-256
`3203148dda9bcbd363192536266b4b3e50e0121a50034c7e39cbb46d58e9fc3a`.
The automatically queued second run was deliberately interrupted after this
established that the template is a distinct maximum-quality workload, not the
remembered 2-3 minute production route. Adaptive recorded both the successful
first run and governed interruption.

ComfyUI's `/system_stats` reports system-wide unified-memory availability, not
per-workflow MPS allocation. It is useful for pressure monitoring but does not
prove kernel-level memory traffic or zero-copy behavior. Any MLX/MPS claim must
therefore be supported by a separate profiler capture.

## Decision

Use `qwen-quality` as the primary PyTorch-version comparison: it exercises the
40.9 GB BF16 model plus the full 16.6 GB encoder and matches the established
production route. Keep FLUX Dev and Schnell as regression guards. Do not start
an MLX port until a profiler identifies a stable hot operation that is outside
model load, text encoding, and output serialization.

## PyTorch 2.14.1 comparison

PyTorch 2.14.1, TorchVision 0.29.1, and TorchAudio 2.11.0 were tested in an
isolated overlay environment on 2026-10-10. The production `.venv` and its
PyTorch 2.5.1 installation were not modified. The same Qwen workflow, prompt,
Adaptive admission path, and MPS device were used. The cold-first pass reused
the production seeds exactly; the resident-model replication used new seeds to
force uncached execution.

| Runtime | Pass | Cold / first | Warm observations | Warm median | Output |
|---|---|---:|---:|---:|---|
| PyTorch 2.5.1 | baseline, seeds 360-362 | 113.729 s | 72.451, 72.458 s | 72.454 s | valid images |
| PyTorch 2.14.1 | cold-first, seeds 360-362 | 119.792 s | 75.593, 103.634 s | 89.614 s | flat grey frames |
| PyTorch 2.14.1 | resident model, seeds 1360-1362 | 196.214 s | 160.969, 126.764 s | 143.867 s | flat grey frames |

All six 2.14.1 outputs were exactly uniform RGB `(128, 127, 125)` with zero
per-channel standard deviation. This is a correctness failure, independent of
the timing regression. The initial 2.14.1 warm median was 23.68% slower than
production; the resident-model replication was 98.56% slower. These figures
describe this Qwen/MPS route only and do not establish a general PyTorch or MPS
performance ranking.

The candidate is rejected. Keep PyTorch 2.5.1 as the verified production
runtime. The comparison reports are locally retained at
`reports/benchmarks/20261010T093232Z_qwen-quality.json` and
`reports/benchmarks/20261010T093900Z_qwen-quality.json`, with SHA-256 values
`0a37cd89df4a07f330dd5e52455f86c9b6586b3678e313f7d18958807da8fb7a` and
`780839787c6b4e31030a9c2bcf06c548e74a4401f3db3f1a792d8cd9a52db5ed`.

The harness now measures saved-image channel variance. A flat output is
recorded as `invalid_output`, reported to Adaptive as a failed run, and causes
the command to exit non-zero after preserving the JSON report.
