from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wan_endurance_has_fixed_stability_and_memory_gates():
    source = (ROOT / "scripts" / "benchmark_wan_vae_endurance.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "kindred.comfyui.wan_vae_endurance.v1" in source
    assert '"stock_p95_to_median_max": 1.30' in source
    assert '"mlx_p95_to_median_max": 1.30' in source
    assert '"torch_driver_growth_gib_max": 1.0' in source
    assert '"mlx_active_growth_mib_max": 64.0' in source
    assert '"mlx_cache_memory_bytes_max": 0' in source
    assert 'choices=("none", "torch_empty_cache")' in source


def test_wan_endurance_alternates_and_checks_determinism():
    source = (ROOT / "scripts" / "benchmark_wan_vae_endurance.py").read_text(
        encoding="utf-8"
    )
    assert '("stock", "mlx") if index % 2 == 0 else ("mlx", "stock")' in source
    assert '"stock_output_deterministic"' in source
    assert '"mlx_output_deterministic"' in source
    assert '"stock_capture_replay_exact"' in source
    assert '"last_half_to_first_half_median"' in source
    assert "torch.mps.current_allocated_memory()" in source
    assert "torch.mps.driver_allocated_memory()" in source
    assert "temporary.replace(path)" in source
