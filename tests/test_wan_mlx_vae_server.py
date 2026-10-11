from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    value = (ROOT / "scripts" / "wan_mlx_vae_server.py").read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_wan_mlx_server_is_temporal_isolated_and_bounded():
    value = source()
    assert "kindred.comfyui.wan_mlx_vae_server.v1" in value
    assert 'value not in {"stock", "mlx"}' in value
    assert '"decoder_policy": "cache_consuming_chunked_resample"' in value
    assert "operation_eager=True, consume_cache=True" in value
    assert "chunk_spatial_resample=True" in value
    assert "mx.asarray(source_mps, copy=False)" in value
    assert "torch.as_tensor(pixels)" in value
    assert "mx.clear_cache()" in value
    assert '"production_modified": False' in value


def test_wan_mlx_server_fails_closed_and_records_memory():
    value = source()
    assert "samples_in.shape[2] > 1" in value
    assert "requires a genuine temporal latent" in value
    assert "does not support VAE options or tiled decoding" in value
    assert "run-mode sequence exhausted" in value
    assert '"input_sha256": tensor_sha256(samples_in)' in value
    assert '"mlx_peak_memory_bytes"' in value
    assert '"mlx_cache_memory_bytes"' in value
    assert "temporary.replace(REPORT_PATH)" in value
    assert "KindredCacheBustLatent" in value
