from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_vae_capture_server_is_bounded_and_atomic():
    source = (ROOT / "scripts" / "capture_qwen_vae_server.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "kindred.comfyui.qwen_vae_capture.v1" in source
    assert "not captured" in source
    assert 'latent_channels", None) == 16' in source
    assert "temporary.replace(CAPTURE_PATH)" in source
    assert "torch.mps.synchronize()" in source
    assert "CAPTURE_PATH = ROOT / CAPTURE_PATH" in source
