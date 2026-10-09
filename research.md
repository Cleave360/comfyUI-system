# Research & Methodologies

**Findings & Discoveries**:
- **Apple M3 Ultra MPS Constraints**: The Metal backend does not natively support FP8 bit-datatypes owing to missing Tensor Cores native to Nvidia architectures. Instead, we have validated running FP16 or BF16, exploiting the massive 512GB unified RAM bus rather than attempting to compress data (which slows down logic processing).
- **Process Memory Bounding**: macOS historically attempts to isolate and throttle single application memory grabs beyond a 70% bounds check. Injecting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` globally via plist overrides this kernel limit, vastly augmenting capability.
