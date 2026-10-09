# Project Scope

## Purpose
The Kindred ComfyUI ecosystem is designed to establish a local, 512GB-optimized image and video generation node for the Kindred Agent network. It serves as the physical "hands and eyes" using advanced visual models (FLUX.1 [dev], SDXL).

## In Scope
- Setup and optimization of ComfyUI for Apple M3 Ultra logic processors.
- Parallel loading of massive FP16 models.
- Communication endpoints (MCP and rust-bridge) for LLM automation.
- Hardware-specific constraint bypassing (memory caps, FP8 quantization limitations).
- Self-healing workflows for custom node management.

## Out of Scope
- Cloud outsourcing of generation tasks.
- Non-Apple Silicon hardware support in this branch.
- Standard minimal-memory (8GB) PC paradigms.
