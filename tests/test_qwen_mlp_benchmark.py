from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_mlp_benchmark_is_syntax_valid_and_records_proof_boundaries():
    source = (ROOT / "scripts" / "benchmark_qwen_mlp.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "qwen_mlp_benchmark.v1" in source
    assert "zero_copy_supported" in source
    assert "mlx_int8_group32" in source
    assert "mlx_branch_concurrency" in source
