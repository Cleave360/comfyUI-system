from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_cfg_batch_server_is_syntax_valid_and_fail_closed():
    source = (ROOT / "scripts" / "qwen_cfg_batch_server.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "qwen_cfg_batch_profile.v1" in source
    assert "exceeding configured pad length" in source
    assert "attention_mask.masked_fill_" in source
    assert "padded_serial" in source
    assert "keep_qwen_cfg_serial" in source
