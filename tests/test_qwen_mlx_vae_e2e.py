from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_mlx_vae_e2e_has_fixed_interleaved_gates():
    source = (ROOT / "scripts" / "benchmark_qwen_mlx_vae_e2e.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "kindred.comfyui.qwen_mlx_vae_e2e.v1" in source
    assert '"candidate_rgb_mae_max_each": 1.0' in source
    assert '"candidate_psnr_db_min_each": 40.0' in source
    assert '"median_vae_seconds_saved_min": 0.5' in source
    assert '("stock", "mlx") if index % 2 == 0 else ("mlx", "stock")' in source
    assert "end_to_end_timing_is_directional_not_a_gate" in source
    assert 'parser.add_argument("--steps", type=int, default=20)' in source


def test_qwen_mlx_vae_e2e_forces_execution_and_checks_server_evidence():
    source = (ROOT / "scripts" / "benchmark_qwen_mlx_vae_e2e.py").read_text(encoding="utf-8")
    assert "KindredCacheBustLatent" in source
    assert '"mode_sequence": observed_modes == expected_modes' in source
    assert '"one_decode_per_run": len(records) == len(sequence)' in source
    assert '"finite_cpu_outputs"' in source
    assert '"server_torch_2_12"' in source
    assert "temporary.replace(path)" in source
