from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    value = (ROOT / "scripts" / "benchmark_wan_mlx_vae_e2e.py").read_text(
        encoding="utf-8"
    )
    ast.parse(value)
    return value


def test_wan_e2e_has_predeclared_full_stack_gates():
    value = source()
    assert "kindred.comfyui.wan_mlx_vae_e2e.v1" in value
    assert '"candidate_global_rgb_mae_max_each": 1.0' in value
    assert '"candidate_temporal_delta_rgb_mae_max_each": 1.5' in value
    assert '"median_vae_speedup_min": 1.5' in value
    assert '"median_end_to_end_seconds_saved_min": 2.0' in value
    assert '"end_to_end_pair_wins_min": 2' in value
    assert '"mlx_peak_memory_gib_max": 20.0' in value
    assert '"mlx_cache_memory_bytes_max": 0' in value


def test_wan_e2e_warms_controls_alternates_and_checks_server_evidence():
    value = source()
    assert '"warmup_stock"' in value
    assert '"warmup_mlx"' in value
    assert '"control_a"' in value
    assert '"control_b"' in value
    assert '("stock", "mlx") if index % 2 == 0 else ("mlx", "stock")' in value
    assert "KindredCacheBustLatent" in value
    assert '"paired_input_hashes"' in value
    assert '"genuine_temporal_latents"' in value
    assert '"seventeen_frame_outputs"' in value
    assert '"mode_sequence"' in value
    assert '"server_torch_2_12"' in value
    assert "temporary.replace(path)" in value
