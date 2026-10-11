from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source() -> str:
    value = (
        ROOT / "scripts" / "benchmark_isolated_image_video_soak.py"
    ).read_text(encoding="utf-8")
    ast.parse(value)
    return value


def test_isolated_soak_uses_two_workers_and_same_mixed_sequence():
    value = source()
    assert "kindred.comfyui.isolated_image_video_soak.v1" in value
    assert '"separate_processes": True' in value
    assert '"qwen-quality": args.image_url' in value
    assert '"wan-native-video": args.video_url' in value
    assert "runs = sequence(args.pairs)" in value
    assert '"separate_worker_pids"' in value
    assert '"image_worker_has_no_wan_decode"' in value


def test_isolated_soak_has_fixed_comparative_stability_and_memory_gates():
    value = source()
    assert '"sampler_p95_to_median_max_each": 1.35' in value
    assert '"baseline_to_isolated_median_speedup_min_each": 1.20' in value
    assert '"combined_process_rss_gib_max": 96.0' in value
    assert '"baseline_median_speedup"' in value
    assert '"combined_process_rss"' in value
    assert '"image_deterministic"' in value
    assert '"video_deterministic"' in value
    assert '"production_modified": False' in value
