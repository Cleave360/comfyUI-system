# Qwen-Image MPS Profile

Measured on 2026-10-10 with the verified PyTorch 2.5.1 runtime on Apple M3
Ultra. The workload was the governed `qwen-quality` profile: Qwen-Image 2512
BF16, 1024x576, 20 diffusion steps, CFG 4.

## Architecture finding

The MinerU large-vocabulary accelerator is valuable prior art, but its direct
hook is not present in this workflow. ComfyUI's Qwen 2.5-VL component is a
prompt encoder. Its `lm_head` is disabled in
`ComfyUI-source/comfy/text_encoders/llama.py`, and it returns hidden states once
before sampling. There is no repeated vocabulary projection or top-k operation.

The repeated path is `QwenImageTransformer2DModel`: 60 diffusion-transformer
blocks executed for each of 20 sampler steps. At 1024x576 the measured forward
had:

- latent shape `[1, 16, 1, 72, 128]`
- 2,304 image tokens after 2x2 patching
- 37 text tokens
- hidden width 3,072
- 24 attention heads of width 128
- image and text MLP expansion width 12,288

## Method and boundary

`scripts/profile_qwen_mps_server.py` launches an isolated ComfyUI server and
instruments one selected diffusion forward. It synchronizes MPS before and
after measured module boundaries. This makes asynchronous GPU time visible but
perturbs the selected step, so the profile supports hotspot ranking, not
end-to-end speed claims. The uninstrumented production baseline remains
72.454 seconds warm median.

Two governed renders completed with valid, non-flat images. The fine-grained
profile used PyTorch 2.5.1/MPS and measured one 2,630.590 ms forward:

| Operation family | Time | Share of profiled forward |
|---|---:|---:|
| Image MLP expand and contract linears | 973.854 ms | 37.02% |
| Image Q/K/V and attention-output linears | 595.599 ms | 22.64% |
| Scaled-dot-product attention | 275.925 ms | 10.49% |
| Text attention and MLP linears | 249.395 ms | 9.48% |
| Image and text modulation linears | 54.616 ms | 2.08% |
| Remaining norms, RoPE, concatenation, activation, gating, and framing | 481.201 ms | 18.29% |

The two image MLP matrices were individually stable: expand median 7.359 ms
(p95 8.305 ms) and contract median 7.631 ms (p95 8.636 ms) across 60 blocks.
Image Q/K/V projections contributed 463.806 ms across 180 calls. The actual
attention reduction was only about one tenth of the forward, so replacing the
attention backend alone has a limited Amdahl ceiling.

Local ignored evidence:

- `reports/profiles/qwen_mps_modules.json`, SHA-256
  `0f3c80d997cd4b255e15407aca2830661379a3415e06f09c7e0ef82390d0d597`
- `reports/profiles/qwen_mps_linears.json`, SHA-256
  `156969a4fd156701e28f105bac4dd6ac7cec958f13a6ffe97cf785943a504c58`
- Fine-profile benchmark report, SHA-256
  `19883a5794f12823d8bf80e33e31a9fdcc115609c9a94c2b366636a6a084a6e2`

## Interpretation

This is a large-matrix/weight-streaming problem, not a vocabulary-selection
problem. The model contains roughly 38.1 GiB of BF16 weights. Analytically, the
60 blocks account for about 38 GiB of linear weights per diffusion forward;
that is too large for on-chip cache and is revisited across 20 steps. This is a
model-shape calculation, not a profiler measurement of physical DRAM traffic.

The Silicon lessons still transfer in three ways: retain converted weights,
measure the complete bridge rather than an isolated kernel, and keep exact
BF16 and approximate INT8 claims separate. A per-token `lm_head` monkey patch
does not transfer. Bridging hundreds of individual PyTorch layers to MLX is
also unlikely to be economical; a candidate must cover at least a complete MLP
or transformer block with retained weights.

## Real-shape MLP experiment

The follow-up experiment captured block 30 at diffusion forward 3, including
real image/text hidden states, reference outputs, and both MLP weights and
biases. The image input and output are `[1, 2304, 3072]`; the text pair is
`[1, 37, 3072]`. Expansion and contraction weights are `[12288, 3072]` and
`[3072, 12288]`. The capture is 315 MiB and is intentionally ignored by Git.

`scripts/benchmark_qwen_mlp.py` retained the converted weights in MLX and ran
seven measured iterations after two warmups:

| Complete MLP | Image median | Text median | Image relative L2 | Result |
|---|---:|---:|---:|---|
| PyTorch 2.5.1 MPS BF16 | 15.468 ms | 0.829 ms | 0 (exact reference) | Baseline |
| MLX 0.32.2 BF16 | 16.526 ms | 0.904 ms | 0.002284 | Slower, non-exact |
| MLX INT8 affine, group 32 | 17.211 ms | 0.903 ms | 0.010399 | Slower, approximate |

This runtime cannot provide the proposed zero-copy bridge. PyTorch 2.5.1
reports `Cannot pack tensors on mps:0` when exporting MPS DLPack, and it reports
`Unsupported device_type: 8` when importing an MLX Metal allocation. The
required copied image-MLP bridge measured 22.326 ms median, versus 15.468 ms
for native PyTorch MPS. Therefore neither MLX candidate qualifies for
integration into the diffusion loop.

The two dependency-independent image/text MLP branches were also issued on
separate MLX streams. Their combined median fell from 17.254 ms sequential to
16.732 ms concurrent: a 1.0312x speedup with exact concurrent-versus-sequential
outputs. This is real overlap, but the text branch is too small to provide a
material end-to-end gain. Attention must finish before either MLP begins, so a
10-15 ms stagger cannot safely overlap joint attention with the image MLP
without changing the transformer dependency graph.

Local ignored evidence:

- `reports/captures/qwen_mlp_block30.pt`, SHA-256
  `7db359a415895a3a308441da91923dc14aa0ae22be93a66f2c6c6cdfac72373a`
- `reports/profiles/qwen_mlp_capture_profile.json`, SHA-256
  `b3373290c67cb1471eb986d45fff4da3f81d09b960b666d04a82ad4be9a958b1`
- `reports/benchmarks/qwen_mlp_block30.json`, SHA-256
  `64977d2e3f19c8c6a79d0c30e2ab4839c27834895eff14768b1c1607eb97897d`

## CFG batching experiment

The stock workflow cannot concatenate positive and negative Qwen conditioning:
their text lengths are 37 and 42. The baseline therefore executes 120 batch-1
transformer forwards for 20 diffusion steps. `scripts/qwen_cfg_batch_server.py`
tested a bounded pad-to-48 intervention with an explicit joint-attention key
mask. It mechanically achieved the intended 60 batch-2 forwards, but failed
both promotion gates:

| Mode | Transformer forwards | Batch | Governed timing | Same-seed parity |
|---|---:|---:|---:|---|
| Stock baseline | 120 | 1 | 97.605 s warm median | Reference |
| Padded CFG batch | 60 | 2 | 140.361 s warm median | Failed |
| Padded serial control | 40 for one run | 1 | 98.683 s | Exact match to padded batch |

The baseline and padded rows each contain three valid renders. Timings varied
materially under sustained unified-memory load, but the candidate is slower
even against the slow baseline sample. For seeds 5360-5362, padded output mean
absolute RGB differences from baseline were 19.005, 13.383, and 10.649. The
seed-5360 serial padded output was pixel-exact to its batch-2 padded output and
not exact to baseline. That isolates the drift to the changed padded tensor
shape/mask path, rather than batch-2 arithmetic. It may be shape-dependent
BF16 kernel rounding amplified over diffusion, so this evidence does not claim
the logical mask is incorrect.

The padded CFG candidate is rejected: halving forward-call count does not halve
work, batch 2 is slower on this MPS workload, and exact same-seed parity is
lost. The production workflow and PyTorch 2.5.1 environment remain unchanged.

Local ignored evidence:

- Baseline benchmark/profile SHA-256: `5ecaee826ea82e6687a6c5150a02d59bbb34549aeaed9382ad3b0fa7edc466a5` /
  `63bb76c24194c7d8e343784db0c9c13bc386f47c51e7cb4e64fc35b40b8007c2`
- Padded benchmark/profile SHA-256: `11067f0670163f5f222ea8652d1ee6995e50b0ad5a6bfe6d8a035549f628e9a5` /
  `bbb4f71ca486b5d3326e2b108d2a196803960a0e5e7ba898c6d20ac8ba1364d0`
- Serial-control benchmark/profile SHA-256: `da25bc4e243cb82e18d689a6629291bb27a23c3e8a524bc2a35257afeb495e3a` /
  `7c7553fa76a78889c1bc61761d0af06abf5bc5053f160cc4182891aea96dffd6`

## Decision

All five initial checks are complete. The capture and comparison infrastructure
is retained for future PyTorch/MLX releases, but none of the initial candidates
should be enabled in production.

## Complete-block and retained-stack experiment

The next experiment tested the larger fused region directly. The profiler can
now capture either a complete real block, a contiguous block range, or only the
input/output boundaries of the 60-block stack. The boundary form avoids making
a second 38 GiB copy of the model: the benchmark loads block weights directly
from the verified `qwen_image_2512_bf16.safetensors` artifact.

The authoritative comparator reconstructs ComfyUI's actual
`QwenImageTransformerBlock`, including its selected MPS attention backend. It
reproduced every captured BF16 output exactly. `scripts/benchmark_qwen_block.py`
then retained all MLX weights, used fused MLX scaled-dot-product attention, and
tested eager and whole-graph-compiled execution. All timings synchronize the
device and exclude model loading and compilation:

| Real region | Exact PyTorch MPS | MLX BF16 eager | MLX BF16 compiled | Compiled speedup |
|---|---:|---:|---:|---:|
| Block 30 | 28.831 ms | 29.768 ms | 27.772 ms | 1.0381x |
| Blocks 30-31 | 57.121 ms | 58.839 ms | 55.070 ms | 1.0372x |
| Blocks 0-59 | 1702.056 ms | 1896.641 ms | 1688.995 ms | 1.0077x |

The small compiled advantage did not scale. A complete 60-block transformer
pass saves only 13.061 ms in this optimistic retained-weight measurement. The
stock CFG workflow performs 40 such passes for 20 diffusion steps, so the
analytical ceiling is about 0.522 seconds per image before any PyTorch/MLX
handoff, input/output projections, scheduler, VAE, or text-encoder cost.

Numerical drift also accumulated across the residual stack:

| Region | Image relative L2 | Text relative L2 | Image cosine | Text cosine |
|---|---:|---:|---:|---:|
| Block 30, compiled BF16 | 0.003435 | 0.003225 | 0.999994 | 0.999995 |
| Blocks 30-31, compiled BF16 | 0.005384 | 0.002574 | 0.999986 | 0.999997 |
| Blocks 0-59, compiled BF16 | 0.014441 | 0.053315 | 0.999943 | 0.998690 |

Compiled INT8 was also tested at one and two blocks. It gained only 1.0187x at
one block and regressed to 0.9215x across two, with larger error, so the 60-block
INT8 run was deliberately skipped.

The full retained MLX port is therefore rejected before end-to-end integration:
its best-case performance gain is below one percent at the tested stack, bridge
cost is excluded, and it already fails exact parity with material accumulated
error. This is a component result, not an image-quality claim; no MLX output was
promoted into the production diffusion loop.

Local ignored evidence:

- One-block capture / exact Torch / MLX SHA-256:
  `77ed512e9cfc2e28fb72a39d1afdf1ce088b6fc272e92ed9e62de2d508e29fc1` /
  `8557aac3a7518e1f0edeea06cbe9d5a7b612a05a372df91f47a44131a79a3e1b` /
  `618452dec1fcb459f09365330e4fd5bd66b382dbf2bef8a9d839326533740214`
- Two-block capture / exact Torch / MLX SHA-256:
  `e33c5537ba0c0edce5bb964947920ae3335380ee5fbb49f0ddda18f36cf36aac` /
  `2f500373b7ce30f549dacd14704e798e577715f6c7093c2c448de2d7c6d01f0b` /
  `6b3c838545149c26dfc31e5d75d7c85531737ec80ae22b4a1717150cdedcb4cd`
- Full-stack boundary / exact Torch / MLX SHA-256:
  `30a58e4d8fcc143c7bd748ce67a6312694eca82d1fdc6d7591318649c4789d9c` /
  `73b2e504dd5dd7ee2b973b9e26240e6f65f6d091de3c163b24cf93ce034011de` /
  `20792b34a106cbd9c9e148e36fe9cb1620370a4fc7a00a0cf9b2d34f573734e9`

The remaining credible paths are PyTorch-native MPS improvements that preserve
the exact execution graph, or an explicitly quality-changing route such as a
distilled/fewer-step Qwen workflow. The latter must be evaluated as a separate
quality-versus-latency product decision, not described as a kernel speedup.

## MLX examples review and residual-cache probe

The current [`ml-explore/mlx-examples`](https://github.com/ml-explore/mlx-examples)
implementations were reviewed after the retained-stack result. Their
[FLUX layers](https://github.com/ml-explore/mlx-examples/blob/main/flux/flux/layers.py)
use fused MLX attention and a small shapeless-compiled RoPE helper, but the
pipeline does not compile the entire denoising model. The
[WAN 2.1 pipeline](https://github.com/ml-explore/mlx-examples/blob/main/video/wan2.1/wan/pipeline.py)
adds three relevant techniques: compiling the complete flow model with its
state, using asynchronous evaluation around the sampler update, and TeaCache
residual reuse. The first two do not remove Qwen's dominant 60-block dependency
chain; our retained whole-stack compile already measured their plausible
compute ceiling. TeaCache is materially different because it avoids selected
transformer passes rather than making each pass marginally cheaper.

`scripts/qwen_residual_probe_server.py` therefore measured, without changing
the render, the image and text residual across the complete 60-block stack for
all 40 positive/negative forwards in one governed 20-step render. Consecutive
image residuals within each conditioning stream had:

- cosine similarity from 0.998565 to 0.999953;
- median relative L2 change of 0.1397;
- minimum relative L2 change of 0.01868 at sigma 0.256198.

That minimum was consistent across the 37-token and 42-token conditioning
streams. It justified one bounded intervention, not a general cache threshold.
`scripts/qwen_residual_cache_server.py` reused the preceding residual only at
sigma 0.2562. It skipped exactly two transformer forwards (one per stream),
each taking about 0.03 seconds for framing instead of running 60 blocks. A warm
candidate render completed in 69.445 seconds, consistent with removing roughly
one of 20 sampler steps from the established 72.454-second warm baseline, but a
single non-interleaved run is not a promotion-quality speed measurement.

The intervention is not numerically exact. Against stock same-seed renders:

| Seed | RGB MAE | RMSE | PSNR | Pixels changed |
|---:|---:|---:|---:|---:|
| 5363 | 2.703 | 5.052 | 34.06 dB | 94.39% |
| 5364 | 3.131 | 5.593 | 33.18 dB | 95.80% |

Both candidate images remained valid and looked closely matched on inspection,
but this is only two-image evidence. The cache remains an isolated experiment.
It is not enabled in production, and the upstream WAN polynomial/threshold must
not be copied because its calibration is model- and resolution-specific.

Local ignored evidence:

- Residual probe JSON SHA-256:
  `4a504d2719eb39bddca2fa80c41597a5254738ac03e7265e5eb29bceb17d7fb3`
- Seed-5363 cache profile / benchmark SHA-256:
  `1fa6bb6f696fb40bb132ac9d8f2b7971b9c3ded8812bb64f5dd7339ec15db37d` /
  `3836117bf151e94992b7026309100e372664fce836df5e4c8629009c10047033`
- Seed-5364 candidate / stock benchmark SHA-256:
  `9b89abb546c159a8866a20183342d25f31983cef509a980416ba2e48000c326d` /
  `87f0993a2a2eb80f0bf06814c9c611acece65af53f6510acba736fe7ec9fd3eb`

### Interleaved promotion sweep

A follow-up sweep removed ComfyUI graph-cache ambiguity with a latent
pass-through nonce directly upstream of KSampler. It preserved the loaded model
and conditioning caches while forcing every same-seed sampler pair to execute.
The experiment used one exact stock/stock determinism control, four stock/cache
pairs with alternating order, a single cached sigma (`0.2562`), and a 60-second
cooldown between renders. Promotion thresholds were declared before execution:

- stock/stock RGB MAE no greater than 0.01;
- every candidate RGB MAE no greater than 4.0;
- every candidate PSNR at least 32.0 dB;
- paired median saving at least 2.0 seconds;
- exactly two skipped transformer forwards per cache render.

The mechanical and determinism gates passed. The stock control was pixel exact,
all ten images were valid, mode order matched the declared sequence, and every
cache render skipped exactly two complete transformer forwards. The paired
median saving was 9.543 seconds, but individual differences ranged from
`-13.100` to `+19.118` seconds because sustained MPS throughput still varied
substantially despite cooldowns. Treat that timing as directional, not a stable
production speed estimate.

The quality gate failed:

| Seed | RGB MAE | RMSE | PSNR | Pixels changed | Seconds saved |
|---:|---:|---:|---:|---:|---:|
| 5400 | 3.287 | 5.972 | 32.61 dB | 96.22% | -13.100 |
| 5401 | 2.912 | 4.841 | 34.43 dB | 96.30% | +15.064 |
| 5402 | 3.644 | 6.546 | **31.81 dB** | 96.19% | +19.118 |
| 5403 | 3.383 | 6.235 | 32.23 dB | 95.32% | +4.021 |

Seed 5402 fell below the predeclared 32.0 dB floor. The worst pair remained
visually close on manual inspection, but the experiment is rejected under its
own acceptance criteria. Residual caching stays disabled in production. A
future trial would need a more conservative reuse construction or a separately
approved quality threshold; moving the gate after observing these results would
not be valid promotion evidence.

Local ignored evidence:

- Final sweep report SHA-256:
  `aa3b158bc83d3947f2e38c54991dccdda658da540625de1ad98355f2eb345cbd`
- Server mode/skip ledger SHA-256:
  `91595a7456fe39e68d837dd6d516a27a2411bc343042b4fcc72c3d8ea2ed7b7a`

## PyTorch 2.12 zero-copy and native MLX VAE experiment

The bridge assumption changed with PyTorch 2.12. In an isolated environment
(`torch==2.12.1`, `mlx==0.32.3`) ordinary MPS tensors imported into MLX with
`mx.asarray(..., copy=False)`, and MLX arrays imported back with
`torch.as_tensor(...)`, retained shared Metal storage. Mutation probes passed
in both directions for float32, float16, and bfloat16, including a strided
Torch view. For the real `[1,2304,3072]` Qwen image-MLP input, the median
Torch-to-MLX-to-Torch handoff was 3.625 microseconds.

That mechanical result did not rescue the Qwen MLP candidate. On the same real
capture, native Torch 2.12 MPS BF16 took 14.716 ms while the complete zero-copy
MLX round trip took 16.357 ms, a 0.8997x result. Its numerical quality gate
passed (`0.002284` relative L2), but the speed gate failed. The useful result is
that the bridge itself is no longer the blocker in PyTorch 2.12; candidate
regions still need faster MLX kernels.

The Qwen image VAE is not a Stable Diffusion or FLUX 2D `AutoencoderKL`. The
local 242 MiB checkpoint is a Wan 2.1 causal 3D VAE with 16 latent channels,
and Apple's current `mlx-examples/video/wan2.1/wan/vae.py` is the matching
reference architecture. The experiment pinned `mlx-examples` commit
`796f5b53cab69a3d48a44233ce21aae889e94a08` and captured the exact real
`[1,16,1,72,128]` latent and PyTorch output from a governed 1024x576 Qwen run.

The fair comparator excludes weight loading and compilation, synchronizes each
iteration, and includes the normal ComfyUI CPU input and CPU output boundaries:

| Decoder | Warm median (7 runs) | Relative speed | Output parity |
|---|---:|---:|---|
| ComfyUI PyTorch 2.5.1 MPS BF16 | 1028.395 ms | 1.0000x | Exact capture match |
| Official MLX Wan VAE, strict precision | 497.382 ms | **2.0676x** | MAE 0.573 RGB levels; 44.624 dB PSNR |

The corrected MLX path first rounds the raw latent to BF16, matching ComfyUI's
native VAE boundary, before converting it back to the normalized form expected
by the official MLX wrapper. It saved 531.013 ms per image in the isolated
component run. It passed all
predeclared trial gates: exact shape, finite output, mean error no greater than
one RGB level, PSNR at least 40 dB, speedup at least 1.05x, and peak MLX memory
below 12 GiB. Observed warm MLX peak memory was 9.01 GiB. The output is not
bit-exact: 99.967% of float pixels differ and the maximum single-channel error
was 0.21691, despite the low aggregate error. This is therefore a strong
component result, not yet a production promotion or a perceptual-quality claim.

At the current roughly 72-second full workflow, the direct still-image saving
is under one second. The result becomes more important for multi-frame/video
decoding and establishes the first measured MLX region with a material speed
advantage. Production remains on PyTorch 2.5.1.

### Isolated end-to-end decoder hook

`scripts/qwen_mlx_vae_server.py` runs ComfyUI under the ignored Torch 2.12/MLX
environment and borrows only non-Torch dependencies from production `.venv`.
It routes only a 16-channel `WanVAE`, fails closed on tiled/options-based
decoding, records every mode and tensor hash, and returns CPU output exactly as
the normal ComfyUI boundary does. The production server and environment are
not modified. Audio extras are unavailable in this experimental runtime because
the production torchaudio binary is intentionally blocked as ABI-incompatible;
the Qwen image workflow does not use those nodes.

A two-step same-process negative control established that VAE error is
latent-distribution dependent: the candidate reached only 1.566 RGB MAE and
37.23 dB, failing the fixed 1.0/40 dB gate. Eager BF16 and FP32 variants were
worse, so the quality threshold was not relaxed.

The decisive governed sweep used three 20-step seeds, alternating stock/MLX
order, plus an exact stock/stock control. A cache-bust node forced the sampler
to execute on every render. All eight images were valid and all promotion
gates passed:

| Seed | RGB MAE | PSNR | Maximum error | Changed pixels |
|---:|---:|---:|---:|---:|
| 7460 | 0.5320 | 44.44 dB | 54 | 49.91% |
| 7461 | 0.5354 | 44.68 dB | 37 | 50.52% |
| 7462 | 0.4429 | 46.24 dB | 50 | 50.09% |

The stock/stock control was pixel-exact. Median direct VAE time was 1.573
seconds for native Torch and 0.792 seconds for MLX, a 0.781-second saving and
1.9868x speedup inside the full server. System load varied sharply during the
sweep, so total-render differences ranged from -41.247 to +0.020 seconds and
are explicitly not a speed gate. This evidence promotes the decoder only to an
opt-in isolated candidate. It does not authorize replacing the production
PyTorch 2.5.1 server or claim a measured full-workflow speedup.

Local ignored evidence:

- Zero-copy bridge report SHA-256:
  `071eaa9ca0406fbac883045ab659be0eb03d580a578d5b495ef53e15680107b9`
- Real VAE capture SHA-256:
  `c9094cfe641b6165f2d361c3ea780aa40a777b5f4a9de7df86a039127cb8d273`
- Native Torch VAE report SHA-256:
  `1fc0f7b630eb8a6c24307cad9a126fdf81baaf4de8a92ae3be821e1133e4eeed`
- MLX VAE report SHA-256:
  `b3948cc8efb177395beaf4dbde274f074298046f2e284ed23b447b04c859666f`
- End-to-end sweep report SHA-256:
  `3ad3b16536a08db3d11cfc9de287442b25336fa3af45d365d678db99834f1145`
- End-to-end server ledger SHA-256:
  `f9bbfc129b52d9541fefcd0ab999a03c7c4d779db7b23a62d1c0a85dc09e9bc3`

## Temporal Qwen/Wan VAE scaling experiment

The next trial tested whether the native MLX decoder advantage becomes more
material for video-shaped data. Two distinct 1024x576 latent endpoints came
from governed 20-step Qwen renders. Intermediate latent frames were linear
interpolations between those real endpoints. This construction tests decoder
scaling and temporal arithmetic; it is not evidence that a video diffusion
model generated coherent motion.

The causal Wan decoder maps 1, 2, and 4 latent frames to 1, 5, and 13 RGB
frames. Each decoder was warmed once and measured four times with alternating
order. Weight loading was excluded; the normal CPU input/output boundary and
the Torch MPS/MLX zero-copy bridge were included. Before the run, the trial
required every sequence to remain below 1.0 global RGB MAE, every frame below
1.25 RGB MAE and above 40 dB PSNR, temporal-difference MAE below 1.5 RGB
levels, at least 1.25x speedup at the longest sequence, and no more than 20 GiB
MLX peak memory.

| Latent / output frames | Torch MPS median | MLX median | Speedup | Global RGB MAE | MLX peak |
|---:|---:|---:|---:|---:|---:|
| 1 / 1 | 1.788 s | 0.449 s | 3.986x | 0.574 | 9.01 GiB |
| 2 / 5 | 7.390 s | 2.147 s | 3.442x | 0.316 | 26.75 GiB |
| 4 / 13 | 18.260 s | 6.124 s | 2.982x | 0.262 | 36.89 GiB |

All structure, finite-value, per-frame, aggregate-quality, temporal-difference,
and speed gates passed. The worst frame was 0.574 RGB MAE at 44.62 dB, and the
worst temporal-difference MAE was 0.661 RGB levels. The memory gate failed at
five and thirteen output frames. The official MLX path also retained 82.53 GiB
in its allocator cache after the complete sweep, despite only 0.237 GiB active
at the end.

A bounded-cache follow-up eagerly materialized every causal feature cache and
called `mx.clear_cache()` inside the measured decode boundary. Its outputs were
byte-identical to the official MLX mode at all three sequence lengths. It
reduced retained allocator cache from 82.53 GiB to zero and still delivered a
2.086x paired speedup for thirteen frames, but transient peak memory remained
36.89 GiB. Therefore the cache-residency issue is mitigated while the transient
working set is not. The original 20 GiB gate remains failed, and neither mode
is promoted into production.

Three activation-lifetime interventions then used the same inputs, quality
gates, and CPU boundaries. Every candidate produced byte-identical MLX output
at 1, 5, and 13 frames and cleared retained allocator cache to zero:

| Thirteen-frame MLX policy | Median | Speedup | Peak memory | Change vs official |
|---|---:|---:|---:|---:|
| Official whole-frame compiled graph | 6.124 s | 2.982x | 36.89 GiB | baseline |
| Layer-boundary materialization | 6.346 s | 1.946x | 30.45 GiB | -17.5% |
| Residual operation boundaries | 6.188 s | 1.993x | 25.05 GiB | -32.1% |
| Two-way spatial tile on final high-resolution tail | 6.686 s | 2.005x | 24.74 GiB | -32.9% |

The operation-level candidate removed most of the avoidable peak while keeping
roughly 2x throughput. Tiling the attention-free final stage with a 32-pixel
halo saved only another 0.30 GiB, so that tail is not the dominant remaining
allocation. The five-frame tiled case narrowly missed the gate at 20.14 GiB;
the thirteen-frame case missed it at 24.74 GiB. This falsifies final-stage-only
tiling as a sufficient solution.

The next useful diagnostic is per-stage peak attribution across the earlier
upsampling stages and causal feature caches. A lower-resolution video tier can
also be tested as a separately declared product profile, but must not revise
this experiment's gate after observation.

Local ignored evidence:

- Second governed endpoint capture SHA-256:
  `a97178183a125c5456a8a0223bb98097b4bd06d0ed8c83950762bf96e351fe2c`
- Governed capture render report SHA-256:
  `89f1ec3b7a371e9e6082e89826387c1180d159ed23c1246a1761be477c1981ef`
- Official temporal sweep SHA-256:
  `0cf0074abc0d93f8e240635b0a024ce145b73fa37b52d6693d74cdf9d34f5531`
- Bounded-cache temporal sweep SHA-256:
  `833980aa5f4afaee8ae4b81fa701bd193af46f00bf853eb504bcc94203b7c9f5`
- Layer-boundary sweep SHA-256:
  `875ec451cfa1a35bba15932ec1f30f1df5ca02fb3d4132ccc856c1a1831e2062`
- Residual operation-boundary sweep SHA-256:
  `3f7c2db5b0d1e7ae79dbcab572f4c7fa491aa313d687763b23d51d241880106b`
- Spatial-tail tiled sweep SHA-256:
  `effffbd08d457bccda2aadbaebbd8f9f7df47e6b065bdc5a03fa8d68520aefd1`

### Per-operation attribution and bounded passing candidate

The operation-eager path was instrumented by resetting MLX peak telemetry
immediately before each explicit evaluation boundary. This perturbs scheduling,
so a separate untraced sweep remains the authoritative timing and gate result.
The trace identified `frame3.stage3.residual2.conv2` as the original 25.045 GiB
peak: 19.972 GiB was already active before that convolution completed. The
cause was lifecycle overlap, not one exceptionally large output tensor—the
complete previous temporal-cache list stayed referenced while the replacement
list was built.

The cache-consuming candidate clears each old cache-list entry immediately
after its single use. It preserved the official MLX output hashes and reduced
the thirteen-frame peak to 20.579 GiB. The new maximum moved to the final
`upsample2d`, which expanded four temporal slices together from
`[1,4,288,512,192]` to `[1,4,576,1024,96]`.

The final candidate processes that purely spatial resample one temporal slice
at a time, while leaving both causal 3D upsamplers unchanged. Two independent
four-iteration sweeps passed every original gate without changing thresholds:

| Output frames | Torch median | MLX median | Speedup | MLX peak | Global RGB MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.174 s | 0.697 s | 1.684x | 7.55 GiB | 0.574 |
| 5 | 4.720 s | 2.620 s | 1.801x | 16.95 GiB | 0.316 |
| 13 | 11.717 s | 5.577 s | **2.101x** | **19.908 GiB** | 0.262 |

The repeat measured 5.565 seconds, 2.105x, and the identical 19.908 GiB peak.
All three MLX output hashes were byte-identical to the official reference;
retained allocator cache was zero; the worst frame remained 0.574 RGB MAE at
44.62 dB and worst temporal-difference MAE remained 0.661 RGB levels. The
candidate therefore qualifies as a reproducible component-level pass under the
predeclared 20 GiB, quality, and speed gates.

This does not prove motion quality, a complete video-generation workflow, or
production-server stability. Integration remains isolated until a real video
latent and repeated server workload reproduce the result.

Local ignored evidence:

- Original operation trace SHA-256:
  `56088f21762be312c6361bc84ed80ccd5e6c3c0ee5e5fc04eae1d2b491442acf`
- Consume-and-release trace SHA-256:
  `73869c3b0b4021b6e533d0f23e4ce415e29fa8edb5a0a113673f6d12c54da359`
- Chunked-resample trace SHA-256:
  `987841b69a5a719dc6753b0bcaec4906c275ea69ae89cdd26381c8248da7e02e`
- Decisive full sweep SHA-256:
  `63846ae6588e46276e98d7fae89e2cd29c93f6ea790f3033626b8ce66ed2e799`
- Independent repeat SHA-256:
  `8012335a019936d7889d3e0d78cdf7874cdae0037e5dbcf3bdbe8a52b354bb0d`

## Native Wan video acceptance and UniPC runtime result

The real-video acceptance used the installed Wan VACE 1.3B model, its CausVid
LoRA, FP16 UMT5 encoder, `wan_2.1_vae.safetensors`, and the committed 832x480
reference image. The minimal governed workflow generated 17 frames in four
diffusion steps. Its post-diffusion latent was genuinely temporal:
`[1,16,5,60,104]`, rather than an interpolation constructed by the benchmark.

The first attempt exposed a runtime boundary. ComfyUI's unchanged UniPC code
already calls `torch.linalg.solve`; production PyTorch 2.5.1 lacks the
corresponding MPS implementation and failed after two sampling steps. A direct
2x2 control reproduced the failure. The same control returned the correct
solution under the isolated PyTorch 2.12.1 environment, and the unchanged
UniPC workflow then completed all four steps. It produced 17 finite, non-flat
frames in 56.209 seconds. This supports an isolated runtime-upgrade candidate,
not a ComfyUI sampler fork or an in-place production upgrade. The experimental
runtime deliberately blocks the ABI-incompatible production torchaudio binary.

The captured native latent and exact Wan weights then passed the original
decoder gates with the cache-consuming, spatial-resample-chunked MLX path:

| Run | Torch 2.12 MPS | MLX | Speedup | RGB MAE | MLX peak | Retained cache |
|---|---:|---:|---:|---:|---:|---:|
| Decisive | 10.301 s | 4.232 s | **2.434x** | 0.156 | **13.596 GiB** | 0 B |
| Independent repeat | 11.274 s | 4.230 s | **2.666x** | 0.156 | **13.596 GiB** | 0 B |

Both runs produced identical Torch and MLX hashes across repeats. The Torch
benchmark output was byte-identical to the output captured inside the server.
Minimum per-frame PSNR was 61.65 dB and worst temporal-difference MAE was 0.187
RGB levels. This closes the native-latent component gate and improves the peak
from 19.908 GiB for the synthetic 1024x576/13-frame trial to 13.596 GiB at the
real workflow's 832x480/17-frame tier. It still does not establish an MLX
end-to-end server speedup: diffusion and text encoding remained PyTorch, while
the MLX comparison was performed after capture.

Local ignored evidence:

- Native Wan capture SHA-256:
  `68feda22d97e4e37c2d09d49a489053e82e4317829da7b9abd0aa703cf415f62`
- Governed workflow report SHA-256:
  `23739633c8107cc16fb89492c3e913bd3ae966e3c736ea8397cb86d147a1021b`
- Native decoder report SHA-256:
  `2701d0d54bab3d35f58f893af17196ef9fc0bb5af30c0a30e724efc737f69c3a`
- Independent repeat SHA-256:
  `abfd9133e030feb0e37711decc92e4d98d31c102d8288d669dcdd0d5f121a312`

### Alternating full-stack Wan server acceptance

The native component candidate was then installed only in an isolated ComfyUI
server and evaluated in one ten-run process. The sequence was fixed before the
run: stock/MLX warm-ups, an exact stock/stock control, and three same-seed
stock/MLX pairs with pair order alternating. A latent pass-through nonce forced
the sampler and decoder to execute on every run. Adaptive admitted all ten
workflows; every run produced 17 finite, non-flat 832x480 frames.

All predeclared gates passed:

- The stock/stock control was pixel-exact across all 17 frames.
- Server modes exactly matched the requested ten-mode sequence and there was
  exactly one genuine `[1,16,5,60,104]` temporal decode per workflow.
- Stock and MLX received byte-identical latent input for each of the three
  paired seeds.
- Global RGB MAE was 0.135-0.151 levels, minimum per-frame PSNR was 55.62 dB,
  and worst temporal-difference MAE was 0.260 levels.
- All three MLX workflows finished faster. End-to-end savings were 4.011,
  100.512, and 72.273 seconds, passing the two-second median and two-of-three
  win gates.
- Paired direct VAE medians were 34.483 seconds for Torch and 7.515 seconds for
  MLX (4.588x). MLX peak memory was 13.596 GiB with zero retained cache.
- Manual midpoint comparison found no visible stock/MLX divergence; inspection
  of first/middle/last frames confirmed coherent forward motion.

The large median timing improvement must not be treated as expected production
throughput. The shared machine became strongly non-stationary during the run:
diffusion steps rose from roughly 4-7 seconds to 10-25 seconds, and stock VAE
decode reached 64.660 seconds before partially recovering. Reported free unified
memory remained above 353 GiB, so simple memory exhaustion does not explain the
slowdown. Alternating order and three pair wins make the direction credible;
the least-contended measured pair is the useful magnitude anchor at 4.011
seconds saved end-to-end (42.134 to 38.123 seconds). Production remains
unchanged until the newer Torch/audio dependency set and long-running runtime
stability are resolved.

Local ignored evidence:

- Full-stack benchmark SHA-256:
  `efb896141c0d73f6d6bc62bce2b9fe597021b2e6be07a23fd4d1069209ca4a8d`
- Full-stack server ledger SHA-256:
  `8ac767e6dd11ece6cccb636672277db3a9f641d1073dcdc0a9e73f76c16d4bdb`
