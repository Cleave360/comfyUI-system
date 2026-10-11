from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    value = (
        ROOT / "scripts" / "benchmark_mixed_image_video_soak.py"
    ).read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_mixed_soak_is_governed_alternating_and_audio_gated():
    value = source()
    assert "kindred.comfyui.mixed_image_video_soak.v1" in value
    assert '"strict_alternation": True' in value
    assert "qwen-quality" in value
    assert "wan-native-video" in value
    assert "benchmark:mixed-soak-audio-smoke" in value
    assert '"audio_nodes_available"' in value
    assert '"audio_workflow_success"' in value
    assert '"torchaudio_stable_abi"' in value
    assert '"production_modified": False' in value


def test_mixed_soak_has_fixed_stability_memory_and_determinism_gates():
    value = source()
    assert '"sampler_p95_to_median_max_each": 1.35' in value
    assert '"sampler_last_half_to_first_half_median_max_each": 1.25' in value
    assert '"torch_driver_growth_gib_max": 1.0' in value
    assert '"process_rss_growth_gib_max": 2.0' in value
    assert '"image_deterministic"' in value
    assert '"video_deterministic"' in value
    assert "add_cache_bust(workflow, index + 1)" in value
