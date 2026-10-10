from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_block_torch_benchmark_uses_actual_comfyui_block():
    source = (ROOT / "scripts" / "benchmark_qwen_block_torch.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "qwen_block_torch_benchmark.v1" in source
    assert "QwenImageTransformerBlock" in source
    assert "strict=True" in source
