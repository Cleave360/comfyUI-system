#!/usr/bin/env python3
"""Aggregate the predeclared AB/BA adaptive-versus-fixed-10s confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from mps_recovery import percentile


SCHEMA = "kindred.comfyui.mps_recovery_policy.v1"
EXPECTED = {
    "round1-fixed10": (1, 1, "fixed"),
    "round1-adaptive": (1, 2, "adaptive"),
    "round2-adaptive": (2, 1, "adaptive"),
    "round2-fixed10": (2, 2, "fixed"),
}


def load_report(path: Path, expected_arm: str) -> dict[str, Any]:
    report = json.loads(path.resolve().read_text(encoding="utf-8"))
    if report.get("schema_version") != SCHEMA:
        raise SystemExit(f"not a recovery-policy report: {path}")
    if not report.get("status", {}).get("completed"):
        raise SystemExit(f"incomplete recovery-policy report: {path}")
    round_number, position, policy = EXPECTED[expected_arm]
    method = report.get("method", {})
    actual = (
        method.get("arm_label"), method.get("counterbalance_round"),
        method.get("counterbalance_position"), method.get("policy"),
    )
    expected = (expected_arm, round_number, position, policy)
    if actual != expected:
        raise SystemExit(f"arm metadata mismatch for {path}: {actual} != {expected}")
    if policy == "fixed" and float(method.get("fixed_cooldown_seconds", -1)) != 10.0:
        raise SystemExit(f"fixed counterbalance arm is not 10 seconds: {path}")
    return report


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for report in reports for run in report["runs"][2:]]
    waits = [
        float(event["wait_seconds"])
        for report in reports for event in report["recovery_events"][1:]
    ]
    elapsed = sum(float(report["analysis"]["elapsed_seconds"]) for report in reports)
    profiles = {}
    for profile in ("qwen-quality", "wan-native-video"):
        values = [float(run["total_seconds"]) for run in measured if run["profile"] == profile]
        median = statistics.median(values)
        p95 = percentile(values, 0.95)
        profiles[profile] = {
            "runs": len(values),
            "median_seconds": round(median, 3),
            "min_seconds": round(min(values), 3),
            "max_seconds": round(max(values), 3),
            "p95_seconds": round(p95, 3),
            "p95_to_median": round(p95 / median, 6),
        }
    thermal_states = [
        sample.get("thermal_state", "unavailable")
        for report in reports
        for event in report["recovery_events"]
        for sample in event.get("samples", [])
    ]
    counts = {
        state: thermal_states.count(state)
        for state in ("nominal", "fair", "serious", "critical", "unavailable", "unknown")
        if thermal_states.count(state)
    }
    return {
        "measured_runs": len(measured),
        "elapsed_seconds": round(elapsed, 3),
        "outputs_per_hour": round(len(measured) * 3600 / elapsed, 3),
        "median_recovery_wait_seconds": round(statistics.median(waits), 3),
        "total_recovery_wait_seconds": round(sum(waits), 3),
        "profiles": profiles,
        "all_runs_valid": all(run["status"] == "success" for report in reports for run in report["runs"]),
        "all_outputs_deterministic": all(
            report["analysis"]["image_deterministic"]
            and report["analysis"]["video_deterministic"]
            for report in reports
        ),
        "recovery_ready_fraction": round(
            sum(bool(event["ready"]) for report in reports for event in report["recovery_events"])
            / sum(len(report["recovery_events"]) for report in reports),
            6,
        ),
        "thermal_state_counts": counts,
        "thermal_performance_boundary_crossed": any(
            state in {"serious", "critical"} for state in thermal_states
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in EXPECTED:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = {
        arm: load_report(getattr(args, arm.replace("-", "_")), arm)
        for arm in EXPECTED
    }
    adaptive = aggregate([arms["round1-adaptive"], arms["round2-adaptive"]])
    fixed = aggregate([arms["round1-fixed10"], arms["round2-fixed10"]])
    round_ratios = {
        "round1_adaptive_after_fixed": round(
            arms["round1-adaptive"]["analysis"]["outputs_per_hour"]
            / arms["round1-fixed10"]["analysis"]["outputs_per_hour"], 6,
        ),
        "round2_adaptive_before_fixed": round(
            arms["round2-adaptive"]["analysis"]["outputs_per_hour"]
            / arms["round2-fixed10"]["analysis"]["outputs_per_hour"], 6,
        ),
    }
    ratio_values = list(round_ratios.values())
    throughput_ratio = adaptive["outputs_per_hour"] / fixed["outputs_per_hour"]
    gates = {
        "all_outputs_valid": adaptive["all_runs_valid"] and fixed["all_runs_valid"],
        "all_outputs_deterministic": (
            adaptive["all_outputs_deterministic"] and fixed["all_outputs_deterministic"]
        ),
        "adaptive_recovery_ready": adaptive["recovery_ready_fraction"] == 1.0,
        "no_serious_or_critical_thermal_state": not (
            adaptive["thermal_performance_boundary_crossed"]
            or fixed["thermal_performance_boundary_crossed"]
        ),
        "adaptive_throughput_noninferior_within_2_percent": throughput_ratio >= 0.98,
        "adaptive_wait_shorter_than_fixed_10s": (
            adaptive["median_recovery_wait_seconds"]
            < fixed["median_recovery_wait_seconds"]
        ),
        "adaptive_qwen_p95_ratio_max_1_35": (
            adaptive["profiles"]["qwen-quality"]["p95_to_median"] <= 1.35
        ),
        "adaptive_wan_p95_ratio_max_1_35": (
            adaptive["profiles"]["wan-native-video"]["p95_to_median"] <= 1.35
        ),
        "order_interaction_ratio_spread_max_1_10": (
            max(ratio_values) / min(ratio_values) <= 1.10
        ),
    }
    output = {
        "schema_version": "kindred.comfyui.mps_recovery_counterbalanced_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria_predeclared": {
            "design": "AB/BA with fresh server per arm",
            "round1": ["fixed-10s", "adaptive"],
            "round2": ["adaptive", "fixed-10s"],
            "adaptive_throughput_noninferiority_margin": 0.02,
            "profile_p95_to_median_max": 1.35,
            "order_interaction_ratio_spread_max": 1.10,
            "thermal_boundary": "no serious or critical samples",
        },
        "arms": {
            name: {
                "outputs_per_hour": report["analysis"]["outputs_per_hour"],
                "source": str(getattr(args, name.replace("-", "_")).resolve()),
            }
            for name, report in arms.items()
        },
        "policies": {"adaptive": adaptive, "fixed-10s": fixed},
        "adaptive_vs_fixed_10s_throughput_ratio": round(throughput_ratio, 6),
        "round_throughput_ratios": round_ratios,
        "order_interaction_ratio_spread": round(max(ratio_values) / min(ratio_values), 6),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(output, indent=2))
    return 0 if output["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
