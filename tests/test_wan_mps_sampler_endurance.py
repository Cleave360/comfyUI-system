from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_server_traces_sampler_and_supports_bounded_cleanup_policy():
    value = (ROOT / "scripts" / "wan_mlx_vae_server.py").read_text(encoding="utf-8")
    ast.parse(value)
    assert 'CLEANUP_POLICY not in {"none", "torch_empty_cache"}' in value
    assert "nodes.common_ksampler = routed_common_ksampler" in value
    assert '"sampler_records": state["sampler_records"]' in value
    assert 'record["memory_before_cleanup"]' in value
    assert "torch.mps.empty_cache()" in value


def test_sampler_endurance_has_fixed_drift_and_memory_gates():
    value = (
        ROOT / "scripts" / "benchmark_wan_mps_sampler_endurance.py"
    ).read_text(encoding="utf-8")
    ast.parse(value)
    assert "kindred.comfyui.wan_mps_sampler_endurance.v1" in value
    assert '"sampler_p95_to_median_max": 1.35' in value
    assert '"sampler_last_half_to_first_half_median_max": 1.25' in value
    assert '"torch_driver_growth_gib_max": 1.0' in value
    assert '"process_rss_growth_gib_max": 1.0' in value
    assert '"production_modified": False' in value
