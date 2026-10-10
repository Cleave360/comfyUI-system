from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_profiler_wrapper_is_syntax_valid_and_bounded():
    path = ROOT / "scripts" / "profile_qwen_mps_server.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "TARGET_FORWARD" in source
    assert 'state["active"] = False' in source
    assert "mps_synchronized_boundaries" in source
