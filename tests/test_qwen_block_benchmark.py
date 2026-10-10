from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_block_benchmark_is_syntax_valid_and_records_boundaries():
    source = (ROOT / "scripts" / "benchmark_qwen_block.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "qwen_block_benchmark.v1" in source
    assert "framework_bridge_excluded" in source
    assert "scaled_dot_product_attention" in source
    assert "mlx_int8_group32" in source
    assert "mlx_compiled_bf16" in source
