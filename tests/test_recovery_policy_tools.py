from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

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


def test_counterbalance_runner_predeclares_ab_ba_fresh_server_design():
    source = (ROOT / "scripts" / "run_counterbalanced_recovery.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert '("round1-fixed10", 1, 1, "fixed", 10.0)' in source
    assert '("round1-adaptive", 1, 2, "adaptive", 0.0)' in source
    assert '("round2-adaptive", 2, 1, "adaptive", 0.0)' in source
    assert '("round2-fixed10", 2, 2, "fixed", 10.0)' in source
    assert '"design": "AB/BA"' in source
    assert '"fresh_server_per_arm": True' in source


def test_counterbalance_comparison_has_noninferiority_order_and_thermal_gates():
    source = (ROOT / "scripts" / "compare_counterbalanced_recovery.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert '"adaptive_throughput_noninferiority_margin": 0.02' in source
    assert '"order_interaction_ratio_spread_max": 1.10' in source
    assert '"no_serious_or_critical_thermal_state"' in source
    assert '"adaptive_wait_shorter_than_fixed_10s"' in source


def test_counterbalance_comparison_executes_on_valid_ab_ba_reports(tmp_path):
    arms = {
        "round1-fixed10": (1, 1, "fixed", 10.0, 30.0),
        "round1-adaptive": (1, 2, "adaptive", 0.0, 30.2),
        "round2-adaptive": (2, 1, "adaptive", 0.0, 30.1),
        "round2-fixed10": (2, 2, "fixed", 10.0, 30.0),
    }
    paths = {}
    for arm, (round_number, position, policy, cooldown, throughput) in arms.items():
        path = tmp_path / f"{arm}.json"
        runs = [
            {"profile": "qwen-quality", "total_seconds": 100, "status": "success"},
            {"profile": "wan-native-video", "total_seconds": 40, "status": "success"},
            {"profile": "qwen-quality", "total_seconds": 100, "status": "success"},
            {"profile": "wan-native-video", "total_seconds": 40, "status": "success"},
        ]
        wait = 4.0 if policy == "adaptive" else 10.0
        events = [
            {"wait_seconds": wait, "ready": True, "samples": [{"thermal_state": "nominal"}]}
            for _ in range(3)
        ]
        path.write_text(json.dumps({
            "schema_version": "kindred.comfyui.mps_recovery_policy.v1",
            "status": {"completed": True},
            "method": {
                "arm_label": arm,
                "counterbalance_round": round_number,
                "counterbalance_position": position,
                "policy": policy,
                "fixed_cooldown_seconds": cooldown,
            },
            "runs": runs,
            "recovery_events": events,
            "analysis": {
                "elapsed_seconds": 480,
                "outputs_per_hour": throughput,
                "image_deterministic": True,
                "video_deterministic": True,
            },
        }), encoding="utf-8")
        paths[arm] = path
    output = tmp_path / "comparison.json"
    command = [sys.executable, str(ROOT / "scripts" / "compare_counterbalanced_recovery.py")]
    for arm, path in paths.items():
        command.extend([f"--{arm}", str(path)])
    command.extend(["--output", str(output)])
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["overall_pass"] is True
    assert comparison["gates"]["adaptive_throughput_noninferior_within_2_percent"] is True


def test_randomized_block_runner_is_balanced_seeded_and_fresh_server():
    source = (ROOT / "scripts" / "run_randomized_recovery_blocks.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert "DEFAULT_SEED = 20261012" in source
    assert "BLOCKS = 6" in source
    assert "PAIRS_PER_ARM = 2" in source
    assert '["fixed-adaptive"] * 3 + ["adaptive-fixed"] * 3' in source
    assert '"design": "balanced randomized paired blocks"' in source
    assert '"fresh_server_per_arm": True' in source


def test_randomized_block_comparator_predeclares_stratified_gates():
    source = (ROOT / "scripts" / "compare_randomized_recovery_blocks.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert '"adaptive_noninferiority_margin": 0.02' in source
    assert '"adaptive_noninferior_in_both_order_strata"' in source
    assert '"order_interaction_spread_max_1_10"' in source
    assert '"adaptive_noninferior_in_both_time_halves"' in source
    assert '"temporal_interaction_spread_max_1_10"' in source
    assert '"no_serious_or_critical_thermal_state"' in source


def test_randomized_block_comparator_executes_balanced_manifest(tmp_path):
    schedule = [
        "fixed-adaptive", "fixed-adaptive", "adaptive-fixed",
        "adaptive-fixed", "fixed-adaptive", "adaptive-fixed",
    ]
    arms = []
    for block, sequence in enumerate(schedule, start=1):
        for position, policy in enumerate(sequence.split("-"), start=1):
            slug = f"block{block}-{policy}"
            path = tmp_path / f"{slug}.json"
            wait = 4.0 if policy == "adaptive" else 10.0
            throughput = 30.6 if policy == "adaptive" else 30.0
            runs = [
                {"profile": "qwen-quality", "total_seconds": 100, "status": "success"},
                {"profile": "wan-native-video", "total_seconds": 40, "status": "success"},
                {"profile": "qwen-quality", "total_seconds": 100, "status": "success"},
                {"profile": "wan-native-video", "total_seconds": 40, "status": "success"},
                {"profile": "qwen-quality", "total_seconds": 100, "status": "success"},
                {"profile": "wan-native-video", "total_seconds": 40, "status": "success"},
            ]
            events = [
                {"wait_seconds": wait, "ready": True, "samples": [{"thermal_state": "nominal"}]}
                for _ in range(5)
            ]
            path.write_text(json.dumps({
                "schema_version": "kindred.comfyui.mps_recovery_policy.v1",
                "status": {"completed": True},
                "method": {
                    "arm_label": slug,
                    "counterbalance_round": block,
                    "counterbalance_position": position,
                    "policy": policy,
                    "fixed_cooldown_seconds": 0.0 if policy == "adaptive" else 10.0,
                },
                "runs": runs,
                "recovery_events": events,
                "analysis": {
                    "elapsed_seconds": 470 if policy == "adaptive" else 480,
                    "outputs_per_hour": throughput,
                    "image_deterministic": True,
                    "video_deterministic": True,
                },
            }), encoding="utf-8")
            arms.append({
                "name": slug, "block": block, "sequence": sequence,
                "position": position, "policy": policy, "report": str(path),
                "status": "complete",
            })
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": "kindred.comfyui.mps_recovery_randomized_blocks.v1",
        "method": {
            "design": "balanced randomized paired blocks",
            "blocks": 6,
            "pairs_per_arm": 2,
        },
        "arms": arms,
    }), encoding="utf-8")
    output = tmp_path / "comparison.json"
    completed = subprocess.run([
        sys.executable, str(ROOT / "scripts" / "compare_randomized_recovery_blocks.py"),
        "--manifest", str(manifest), "--output", str(output),
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    comparison = json.loads(output.read_text(encoding="utf-8"))
    assert comparison["overall_pass"] is True
    assert comparison["paired_block_ratios"]["count"] == 6
    assert comparison["gates"]["adaptive_noninferior_in_both_order_strata"] is True
    assert comparison["gates"]["adaptive_noninferior_in_both_time_halves"] is True


def test_shadow_replay_requires_three_consecutive_ready_samples():
    from scripts.analyze_recovery_shadow import shadow_release

    samples = [
        {"latency_ms": 20, "wait_elapsed_seconds": 0},
        {"latency_ms": 40, "wait_elapsed_seconds": 2},
        {"latency_ms": 20, "wait_elapsed_seconds": 4},
        {"latency_ms": 19, "wait_elapsed_seconds": 6},
        {"latency_ms": 18, "wait_elapsed_seconds": 8},
    ]
    result = shadow_release(samples, threshold_ms=30, consecutive_required=3)
    assert result["resolved"] is True
    assert result["sample_index"] == 4
    assert result["release_seconds"] == 8


def test_shadow_analyzer_is_counterfactual_and_does_not_modify_dispatch():
    source = (ROOT / "scripts" / "analyze_recovery_shadow.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source)
    assert '"dispatch_modified": False' in source
    assert '"counterfactual_only": True' in source
    assert 'report.get("recovery_events", [])[1:]' in source
    assert "does not prove what" in source
