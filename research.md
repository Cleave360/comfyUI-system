# Research & Methodologies

## Verified findings

- PyTorch 2.5.1 reports MPS available on this host, a matrix multiplication
  completed on an MPS tensor, and the live ComfyUI endpoint reported `mps`.
- This does not prove that every model, dtype, operator, or custom node works on
  MPS. In particular, the upstream mixed-precision CPU test calls an FP8 scaled
  matrix operator unavailable on CPU.
- FP16/BF16 filenames are compatibility hints, not execution proof. The model
  lock therefore records MPS status conservatively.
- Disabling PyTorch MPS memory limits globally is not part of the supported
  startup procedure; it can destabilize the host. Memory tuning must be scoped,
  measured, and workload-specific.
