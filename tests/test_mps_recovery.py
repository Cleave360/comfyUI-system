from __future__ import annotations

from scripts.mps_recovery import (
    AppleThermalProbe,
    fixed_recovery_window,
    recovery_threshold,
    summarize_runs,
    wait_for_recovery,
)


def test_apple_thermal_probe_has_explicit_supported_boundary():
    probe = AppleThermalProbe()
    sample = probe.measure()
    assert sample["thermal_state"] in {
        "nominal", "fair", "serious", "critical", "unknown", "unavailable",
    }
    assert probe.metadata()["performance_impact_boundary"] == "serious"
    assert probe.metadata()["gpu_throttle_temperature_celsius"] is None


class Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


def measurements(values: list[float]):
    iterator = iter(values)
    return lambda: {"latency_ms": next(iterator)}


def test_recovery_threshold_uses_median_multiplier_or_mad_margin():
    result = recovery_threshold([10.0, 10.1, 10.2, 10.3, 40.0])
    assert result["median_ms"] == 10.2
    assert result["threshold_ms"] == 12.75
    assert result["p95_ms"] == 40.0


def test_adaptive_wait_requires_consecutive_ready_samples():
    clock = Clock()
    result = wait_for_recovery(
        measurements([2.0, 1.1, 1.4, 1.0, 0.9]),
        threshold_ms=1.2, consecutive_required=2,
        interval_seconds=1.0, max_wait_seconds=10.0,
        sleep=clock.sleep, monotonic=clock.monotonic,
    )
    assert result["ready"] is True
    assert result["wait_seconds"] == 4.0
    assert [item["ready"] for item in result["samples"]] == [False, True, False, True, True]


def test_fixed_window_observes_but_never_shortens_cooldown():
    clock = Clock()
    result = fixed_recovery_window(
        measurements([0.8, 0.9, 1.0]),
        cooldown_seconds=2.0, interval_seconds=1.0, threshold_ms=1.2,
        sleep=clock.sleep, monotonic=clock.monotonic,
    )
    assert result["wait_seconds"] == 2.0
    assert len(result["samples"]) == 3


def test_summary_reports_throughput_and_profile_stability():
    runs = [
        {"profile": "warmup", "total_seconds": 5, "started_monotonic": 0, "completed_monotonic": 5},
        {"profile": "warmup", "total_seconds": 5, "started_monotonic": 6, "completed_monotonic": 11},
        {"profile": "qwen-quality", "total_seconds": 100, "started_monotonic": 20, "completed_monotonic": 120},
        {"profile": "wan-native-video", "total_seconds": 40, "started_monotonic": 130, "completed_monotonic": 170},
    ]
    events = [{"wait_seconds": 1}, {"wait_seconds": 9}, {"wait_seconds": 10}]
    result = summarize_runs(runs, events)
    assert result["measured_runs"] == 2
    assert result["outputs_per_hour"] == 48.0
    assert result["total_recovery_wait_seconds"] == 19.0
