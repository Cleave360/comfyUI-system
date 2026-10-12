#!/usr/bin/env python3
"""Compare completed fixed-cooldown and adaptive MPS recovery reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


SCHEMA = "kindred.comfyui.mps_recovery_policy.v1"


def policy_name(report: dict[str, Any]) -> str:
    method = report["method"]
    if method["policy"] == "adaptive":
        return "adaptive"
    seconds = float(method["fixed_cooldown_seconds"])
    return f"fixed-{seconds:g}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded: dict[str, Any] = {}
    for path in args.reports:
        report = json.loads(path.resolve().read_text(encoding="utf-8"))
        if report.get("schema_version") != SCHEMA:
            raise SystemExit(f"not a recovery-policy report: {path}")
        if not report.get("status", {}).get("completed"):
            raise SystemExit(f"recovery-policy report is incomplete: {path}")
        name = policy_name(report)
        if name in loaded:
            raise SystemExit(f"duplicate policy report: {name}")
        loaded[name] = {"path": str(path.resolve()), "report": report}
    required = {"fixed-2s", "fixed-10s", "fixed-30s", "adaptive"}
    missing = sorted(required - loaded.keys())
    if missing:
        raise SystemExit(f"missing required policies: {', '.join(missing)}")

    policies = {}
    for name, item in loaded.items():
        analysis = item["report"]["analysis"]
        policies[name] = {
            "source": item["path"],
            "outputs_per_hour": analysis["outputs_per_hour"],
            "median_run_seconds": analysis["median_run_seconds"],
            "p95_run_seconds": analysis["p95_run_seconds"],
            "median_recovery_wait_seconds": analysis["median_recovery_wait_seconds"],
            "total_recovery_wait_seconds": analysis["total_recovery_wait_seconds"],
            "profiles": analysis["profiles"],
            "all_runs_valid": analysis["all_runs_valid"],
            "image_deterministic": analysis["image_deterministic"],
            "video_deterministic": analysis["video_deterministic"],
            "recovery_ready_fraction": analysis["recovery_ready_fraction"],
        }
    fixed2 = policies["fixed-2s"]
    adaptive = policies["adaptive"]
    gates = {
        "all_outputs_valid": all(value["all_runs_valid"] for value in policies.values()),
        "all_outputs_deterministic": all(
            value["image_deterministic"] and value["video_deterministic"]
            for value in policies.values()
        ),
        "adaptive_recovery_ready": adaptive["recovery_ready_fraction"] == 1.0,
        "adaptive_throughput_not_worse_than_fixed_2s": (
            adaptive["outputs_per_hour"] >= fixed2["outputs_per_hour"]
        ),
        "adaptive_qwen_p95_ratio_max_1_35": (
            adaptive["profiles"]["qwen-quality"]["p95_to_median"] <= 1.35
        ),
        "adaptive_wan_p95_ratio_max_1_35": (
            adaptive["profiles"]["wan-native-video"]["p95_to_median"] <= 1.35
        ),
    }
    ranking = sorted(
        policies,
        key=lambda name: (-policies[name]["outputs_per_hour"], policies[name]["p95_run_seconds"]),
    )
    output = {
        "schema_version": "kindred.comfyui.mps_recovery_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "criteria_predeclared": {
            "required_policies": sorted(required),
            "adaptive_profile_p95_to_median_max": 1.35,
            "adaptive_throughput_floor": "fixed-2s",
            "quality": "all valid and deterministic",
        },
        "policies": policies,
        "ranking_by_outputs_per_hour": ranking,
        "adaptive_vs_fixed_2s_throughput_ratio": round(
            adaptive["outputs_per_hour"] / fixed2["outputs_per_hour"], 6,
        ),
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
