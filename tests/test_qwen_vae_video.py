from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    value = (ROOT / "scripts" / "benchmark_qwen_vae_video.py").read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_video_benchmark_declares_scope_and_fixed_gates():
    value = source()
    assert "kindred.comfyui.qwen_vae_video.v1" in value
    assert '"video_generation_quality_claimed": False' in value
    assert '"global_rgb_mae_max_each": 1.0' in value
    assert '"per_frame_rgb_mae_max_each": 1.25' in value
    assert '"per_frame_psnr_db_min_each": 40.0' in value
    assert '"temporal_delta_rgb_mae_max_each": 1.5' in value
    assert '"largest_sequence_speedup_min": 1.25' in value
    assert '"mlx_peak_memory_gib_max": 20.0' in value


def test_video_benchmark_uses_real_endpoints_and_zero_copy():
    value = source()
    assert "qwen_vae_1024x576.pt" in value
    assert "qwen_vae_video_endpoint_b.pt" in value
    assert "mx.asarray(source_mps, copy=False)" in value
    assert "torch.as_tensor(pixels)" in value
    assert '"endpoint_latents_are_real_qwen_captures": True' in value
    assert '"mode_order": "stock,mlx then mlx,stock alternating"' in value
    assert "temporary.replace(destination)" in value


def test_video_benchmark_checks_output_count_and_temporal_delta():
    value = source()
    assert "expected_output_frames = 1 + 4 * (latent_count - 1)" in value
    assert '"temporal_delta": temporal' in value
    assert '"output_frame_count_expected"' in value
    assert '"temporal_delta_rgb_mae"' in value


def test_video_benchmark_has_bounded_temporal_cache_candidate():
    value = source()
    assert '"op_eager_bounded"' in value
    assert "mx.eval(output, *feature_cache)" in value
    assert "mx.clear_cache()" in value
    assert '"cache_release_in_timed_boundary"' in value
    assert "def layer_eager_decoder_step" in value
    assert 'evaluated(f"{frame}.head_conv", value, *new_cache)' in value
    assert "def operation_eager_residual" in value
    assert 'evaluated(f"{label}.conv2", shortcut, residual, new_first, new_second)' in value
    assert '"spatial_tiled_bounded"' in value
    assert "def spatial_tiled_decoder_step" in value
    assert 'parser.add_argument("--spatial-tile-halo", type=int, default=32)' in value
    assert "seven-pixel tail radius" in value
    assert 'parser.add_argument("--trace-memory", action="store_true")' in value
    assert "def summarize_memory_trace" in value
    assert '"summary_by_peak"' in value
    assert '"per_operation_memory_trace": args.trace_memory' in value
    assert '"cache_consuming_bounded"' in value
    assert "feature_cache[index] = None" in value
    assert "consume_cache=True" in value
    assert '"cache_consuming_chunked_resample"' in value
    assert "def temporal_chunked_spatial_upsample" in value
    assert 'layer.mode != "upsample2d"' in value
    assert "chunk_spatial_resample=True" in value
