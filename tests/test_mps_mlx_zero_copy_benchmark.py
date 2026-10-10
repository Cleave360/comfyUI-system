from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_zero_copy_benchmark_is_syntax_valid_and_governed():
    source = (ROOT / "scripts" / "benchmark_mps_mlx_zero_copy.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "kindred.comfyui.mps_mlx_zero_copy.v1" in source
    assert "copy=False" in source
    assert "mutation_visible" in source
    assert "bridge_speedup_at_least_1_02x" in source
    assert "bridge_relative_l2_at_most_0_0025" in source
    assert "production_environment_modified\": False" in source


def test_zero_copy_benchmark_uses_atomic_report_write():
    source = (ROOT / "scripts" / "benchmark_mps_mlx_zero_copy.py").read_text(
        encoding="utf-8"
    )
    assert "temporary.replace(destination)" in source
    assert "torch.mps.synchronize()" in source
    assert "mx.eval(" in source
