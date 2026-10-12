from __future__ import annotations

import ast
import json
from pathlib import Path

from scripts.gpu_metrics_gui import DashboardState, ROOT, normalize_report


def test_recovery_benchmark_is_governed_and_incremental():
    source = (ROOT / "scripts" / "benchmark_recovery_policy.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert "base.GovernanceGate(ROOT)" in source
    assert "wait_for_recovery" in source
    assert "fixed_recovery_window" in source
    assert '"live_recovery_samples"' in source
    assert '"production_modified": False' in source


def test_comparison_requires_all_four_predeclared_policies():
    source = (ROOT / "scripts" / "compare_recovery_policies.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert '{"fixed-2s", "fixed-10s", "fixed-30s", "adaptive"}' in source
    assert '"adaptive_profile_p95_to_median_max": 1.35' in source
    assert '"adaptive_throughput_floor": "fixed-2s"' in source


def test_dashboard_normalizes_recovery_report_without_exposing_raw_payload():
    payload = {
        "schema_version": "kindred.comfyui.mps_recovery_policy.v1",
        "created_at": "2026-10-11T00:00:00+00:00",
        "status": {"phase": "complete", "completed": True},
        "method": {"policy": "fixed", "fixed_cooldown_seconds": 10},
        "calibration": {"median_ms": 2, "threshold_ms": 3},
        "runs": [{
            "run": 1, "label": "image_01", "profile": "qwen-quality",
            "total_seconds": 100, "status": "success", "memory_samples": [{
                "elapsed_seconds": 1, "ram_free_bytes": 10 * 1024**3,
                "devices": [{"vram_free_bytes": 10 * 1024**3}],
            }],
        }],
        "recovery_events": [],
        "analysis": {"outputs_per_hour": 20},
        "secret": "must not pass through",
    }
    result = normalize_report(ROOT / "reports" / "benchmarks" / "sample.json", payload)
    assert result is not None
    assert result["policy"] == "fixed 10s"
    assert result["runs"][0]["memory"][0]["system_free_gib"] == 10.0
    assert "secret" not in result


def test_dashboard_is_loopback_only_and_has_live_api():
    source = (ROOT / "scripts" / "gpu_metrics_gui.py").read_text(encoding="utf-8")
    html = (ROOT / "docs" / "gpu_metrics_dashboard.html").read_text(encoding="utf-8")
    assert 'args.listen not in {"127.0.0.1", "localhost", "::1"}' in source
    assert 'path == "/api/data"' in source
    assert "setInterval(refresh,2000)" in html
    assert "Workflow duration" in html
    assert "Recovery probe latency" in html
    assert "Thermal pressure" in html
    assert "no public GPU °C cutoff" in html


def test_dashboard_discovers_new_reports_in_watched_directory(tmp_path):
    state = DashboardState([], tmp_path)
    assert state.report_paths() == []
    report = tmp_path / "new.json"
    report.write_text("{}", encoding="utf-8")
    assert state.report_paths() == [report]


def test_dashboard_caches_unchanged_reports(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.gpu_metrics_gui.ROOT", tmp_path)
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "schema_version": "kindred.comfyui.mps_recovery_policy.v1",
        "created_at": "2026-10-11T00:00:00+00:00",
        "method": {"policy": "adaptive"},
    }), encoding="utf-8")
    state = DashboardState([], tmp_path)
    calls = 0
    original = Path.read_text

    def counted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    assert len(state.snapshot()["reports"]) == 1
    assert len(state.snapshot()["reports"]) == 1
    assert calls == 1


def test_matrix_uses_fresh_server_and_predeclared_screening_order():
    source = (ROOT / "scripts" / "run_recovery_policy_matrix.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert '("fixed-2s", "fixed", 2.0)' in source
    assert '("fixed-10s", "fixed", 10.0)' in source
    assert '("fixed-30s", "fixed", 30.0)' in source
    assert '("adaptive", "adaptive", 0.0)' in source
    assert '"fresh_server_per_policy": True' in source
    assert '"order_is_a_screening_confound": True' in source
