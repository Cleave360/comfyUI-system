from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source(name: str) -> str:
    value = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_native_qwen_vae_benchmark_is_synchronized_and_atomic():
    value = source("benchmark_qwen_vae_torch.py")
    assert "kindred.comfyui.qwen_vae_torch.v1" in value
    assert "torch.mps.synchronize()" in value
    assert "model_loading_excluded_from_measured_iterations" in value
    assert "args.output = args.output.resolve()" in value
    assert '"pixel_error_vs_capture": pixel_error' in value
    assert "temporary.replace(destination)" in value


def test_mlx_qwen_vae_benchmark_has_zero_copy_and_quality_gates():
    value = source("benchmark_qwen_vae_mlx.py")
    assert "kindred.comfyui.qwen_vae_mlx_zero_copy.v1" in value
    assert "mx.asarray(source_mps, copy=False)" in value
    assert "torch.as_tensor(pixels)" in value
    assert "pixels_torch.cpu()" in value
    assert "cpu_input_and_cpu_output_match_comfy_boundary" in value
    assert "mae_rgb255_at_most_1" in value
    assert "psnr_at_least_40_db" in value
    assert "speedup_at_least_1_05x" in value
    assert "warm_peak_memory_at_most_12_gib" in value
    assert '"production_environment_modified": False' in value
    assert "temporary.replace(destination)" in value
