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

## Next experiment

Build a captured real-shape microbenchmark from one representative block:

1. Capture image hidden states and the two BF16 MLP weights at a fixed block and
   diffusion step, with hashes and a PyTorch MPS reference output.
2. Compare PyTorch MPS against retained-weight MLX BF16 for the complete
   `linear -> GELU(tanh) -> linear` MLP, including synchronization and DLPack
   handoff costs.
3. Add MLX INT8 affine group-size 32 as a separate approximate candidate.
4. Require numerical gates at the block output and latent checkpoints, then a
   fixed prompt/seed image-quality suite. Token top-k consensus is not an
   applicable quality metric for image diffusion.
5. Promote only a candidate that improves the governed end-to-end warm median,
   preserves valid outputs, and passes the image-quality gate.

Image MLP is the first target because it is the largest measured family and
offers a materially larger ceiling than scaled-dot-product attention. A full
MLX Qwen-Image port becomes justified only if the block-level bridge cost or
framework crossings erase the isolated MLP gain.
