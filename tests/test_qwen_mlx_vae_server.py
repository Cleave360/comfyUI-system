from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_mlx_vae_server_is_isolated_bounded_and_audited():
    source = (ROOT / "scripts" / "qwen_mlx_vae_server.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "kindred.comfyui.qwen_mlx_vae_server.v1" in source
    assert 'value not in {"stock", "mlx"}' in source
    assert "mx.asarray(source_mps, copy=False)" in source
    assert "torch.as_tensor(pixels)" in source
    assert "pixels_torch.cpu()" in source
    assert "samples_in.to(torch.bfloat16).float()" in source
    assert '"input_sha256": tensor_sha256(samples_in)' in source
    assert "temporary.replace(capture_path)" in source
    assert '"production_modified": False' in source
    assert "temporary.replace(REPORT_PATH)" in source
    assert "KindredCacheBustLatent" in source
    assert 'torchaudio_package_version != "2.11.0"' in source
    assert '"torchaudio": torchaudio_package_version' in source


def test_qwen_mlx_vae_server_fails_closed_outside_target_architecture():
    source = (ROOT / "scripts" / "qwen_mlx_vae_server.py").read_text(encoding="utf-8")
    assert 'type(getattr(self, "first_stage_model", None)).__name__ == "WanVAE"' in source
    assert "expected BCTHW Wan latent" in source
    assert "does not support VAE options or tiled decoding" in source
    assert "run-mode sequence exhausted" in source


def test_qwen_mlx_experimental_runtime_is_pinned():
    requirements = (ROOT / "requirements-qwen-mlx-torch212.txt").read_text(
        encoding="utf-8"
    )
    assert "torch==2.12.1" in requirements
    assert "torchaudio==2.11.0" in requirements
    assert "torchvision==0.27.1" in requirements
    assert "mlx==0.32.3" in requirements
