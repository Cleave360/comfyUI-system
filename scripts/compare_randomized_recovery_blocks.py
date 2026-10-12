#!/usr/bin/env python3
"""Aggregate the predeclared balanced randomized-block recovery experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compare_counterbalanced_recovery import aggregate


REPORT_SCHEMA = "kindred.comfyui.mps_recovery_policy.v1"
MANIFEST_SCHEMA = "kindred.comfyui.mps_recovery_randomized_blocks.v1"


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def load_arm(path: Path, block: int, position: int, policy: str) -> dict[str, Any]:
    report = json.loads(path.resolve().read_text(encoding="utf-8"))
    if report.get("schema_version") != REPORT_SCHEMA:
        raise SystemExit(f"not a recovery-policy report: {path}")
    if not report.get("status", {}).get("completed"):
        raise SystemExit(f"incomplete recovery-policy report: {path}")
    expected_label = f"block{block}-{policy}"
    method = report.get("method", {})
    actual = (
        method.get("arm_label"), method.get("counterbalance_round"),
        method.get("counterbalance_position"), method.get("policy"),
    )
    expected = (expected_label, block, position, policy)
    if actual != expected:
        raise SystemExit(f"arm metadata mismatch for {path}: {actual} != {expected}")
    expected_cooldown = 10.0 if policy == "fixed" else 0.0
    if float(method.get("fixed_cooldown_seconds", -1)) != expected_cooldown:
        raise SystemExit(f"cooldown mismatch for {path}")
    return report


def summarize_ratios(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "geometric_mean": round(geometric_mean(values), 6),
        "median": round(statistics.median(values), 6),
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SystemExit(f"not a randomized-block manifest: {manifest_path}")
    method = manifest.get("method", {})
    if method.get("design") != "balanced randomized paired blocks":
        raise SystemExit("manifest design mismatch")
    if method.get("blocks") != 6 or method.get("pairs_per_arm") != 2:
        raise SystemExit("manifest does not match the predeclared 6x2 design")

    arm_entries = manifest.get("arms", [])
    if len(arm_entries) != 12 or not all(arm.get("status") == "complete" for arm in arm_entries):
        raise SystemExit("manifest must contain twelve complete arms")

    blocks: dict[int, dict[str, Any]] = {}
    adaptive_reports = []
    fixed_reports = []
    for entry in arm_entries:
        block = int(entry["block"])
        position = int(entry["position"])
        policy = str(entry["policy"])
        report_path = Path(entry["report"])
        report = load_arm(report_path, block, position, policy)
        block_reports = blocks.setdefault(block, {})
        if policy in block_reports:
            raise SystemExit(f"duplicate {policy} arm in block {block}")
        block_reports[policy] = report
        block_reports[f"{policy}_position"] = position
        block_reports[f"{policy}_source"] = str(report_path.resolve())
        (adaptive_reports if policy == "adaptive" else fixed_reports).append(report)

    if sorted(blocks) != list(range(1, 7)):
        raise SystemExit("blocks must be numbered one through six")

    adaptive = aggregate(adaptive_reports)
    fixed = aggregate(fixed_reports)
    block_rows = []
    order_groups = {"adaptive-first": [], "fixed-first": []}
    first_half = []
    second_half = []
    for block in sorted(blocks):
        reports = blocks[block]
        if "adaptive" not in reports or "fixed" not in reports:
            raise SystemExit(f"block {block} does not contain both policies")
        ratio = (
            float(reports["adaptive"]["analysis"]["outputs_per_hour"])
            / float(reports["fixed"]["analysis"]["outputs_per_hour"])
        )
        order = "adaptive-first" if reports["adaptive_position"] == 1 else "fixed-first"
        order_groups[order].append(ratio)
        (first_half if block <= 3 else second_half).append(ratio)
        block_rows.append({
            "block": block,
            "order": order,
            "adaptive_outputs_per_hour": reports["adaptive"]["analysis"]["outputs_per_hour"],
            "fixed_10s_outputs_per_hour": reports["fixed"]["analysis"]["outputs_per_hour"],
            "adaptive_vs_fixed_10s_ratio": round(ratio, 6),
            "adaptive_source": reports["adaptive_source"],
            "fixed_10s_source": reports["fixed_source"],
        })

    all_ratios = [row["adaptive_vs_fixed_10s_ratio"] for row in block_rows]
    order_summaries = {name: summarize_ratios(values) for name, values in order_groups.items()}
    half_summaries = {
        "blocks_1_to_3": summarize_ratios(first_half),
        "blocks_4_to_6": summarize_ratios(second_half),
    }
    order_values = [summary["geometric_mean"] for summary in order_summaries.values()]
    half_values = [summary["geometric_mean"] for summary in half_summaries.values()]
    pooled_ratio = adaptive["outputs_per_hour"] / fixed["outputs_per_hour"]
    order_spread = max(order_values) / min(order_values)
    temporal_spread = max(half_values) / min(half_values)
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
        "adaptive_pooled_noninferior_within_2_percent": pooled_ratio >= 0.98,
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
        "adaptive_noninferior_in_both_order_strata": all(value >= 0.98 for value in order_values),
        "order_interaction_spread_max_1_10": order_spread <= 1.10,
        "adaptive_noninferior_in_both_time_halves": all(value >= 0.98 for value in half_values),
        "temporal_interaction_spread_max_1_10": temporal_spread <= 1.10,
    }
    output = {
        "schema_version": "kindred.comfyui.mps_recovery_randomized_blocks_comparison.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "criteria_predeclared": {
            "blocks": 6,
            "balanced_orders": True,
            "pairs_per_arm": 2,
            "fresh_server_per_arm": True,
            "adaptive_noninferiority_margin": 0.02,
            "profile_p95_to_median_max": 1.35,
            "order_interaction_spread_max": 1.10,
            "temporal_interaction_spread_max": 1.10,
            "thermal_boundary": "no serious or critical samples",
        },
        "blocks": block_rows,
        "policies": {"adaptive": adaptive, "fixed-10s": fixed},
        "adaptive_vs_fixed_10s_pooled_ratio": round(pooled_ratio, 6),
        "paired_block_ratios": summarize_ratios(all_ratios),
        "order_strata": order_summaries,
        "order_interaction_spread": round(order_spread, 6),
        "time_halves": half_summaries,
        "temporal_interaction_spread": round(temporal_spread, 6),
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
