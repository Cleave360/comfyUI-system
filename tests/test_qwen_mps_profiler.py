from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_profiler_wrapper_is_syntax_valid_and_bounded():
    path = ROOT / "scripts" / "profile_qwen_mps_server.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "TARGET_FORWARD" in source
    assert "CAPTURE_BLOCK" in source
    assert "CAPTURE_KIND" in source
    assert "CAPTURE_BLOCK_COUNT" in source
    assert "qwen_mlp_capture.v1" in source
    assert "qwen_block_capture.v1" in source
    assert "qwen_block_range_capture.v1" in source
    assert "qwen_block_boundary_capture.v1" in source
    assert 'state["active"] = False' in source
    assert "mps_synchronized_boundaries" in source
