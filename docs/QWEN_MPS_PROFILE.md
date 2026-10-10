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
