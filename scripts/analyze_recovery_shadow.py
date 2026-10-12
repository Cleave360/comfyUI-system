#!/usr/bin/env python3
"""Replay adaptive readiness decisions from fixed-policy probe telemetry."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
from typing import Any


REPORT_SCHEMA = "kindred.comfyui.mps_recovery_policy.v1"


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def shadow_release(
    samples: list[dict[str, Any]], threshold_ms: float, consecutive_required: int,
) -> dict[str, Any]:
    consecutive = 0
    for index, sample in enumerate(samples):
        if float(sample["latency_ms"]) <= threshold_ms:
            consecutive += 1
        else:
            consecutive = 0
        if consecutive >= consecutive_required:
            return {
                "resolved": True,
                "sample_index": index,
                "release_seconds": round(float(sample["wait_elapsed_seconds"]), 3),
                "release_latency_ms": round(float(sample["latency_ms"]), 6),
            }
    return {
        "resolved": False,
        "sample_index": None,
        "release_seconds": None,
        "release_latency_ms": None,
    }


def load_fixed_reports(manifest_path: Path) -> list[tuple[str, Path]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reports = [
        (str(arm["name"]), Path(arm["report"]))
        for arm in manifest.get("arms", [])
        if arm.get("policy") == "fixed" and arm.get("status") == "complete"
    ]
    if not reports:
        raise SystemExit("manifest contains no complete fixed-policy arms")
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    rows = []
    unresolved = 0
    serious_or_critical = 0
    for arm_name, report_path in load_fixed_reports(manifest_path):
        report_path = report_path.resolve()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("schema_version") != REPORT_SCHEMA:
            raise SystemExit(f"not a recovery-policy report: {report_path}")
        if not report.get("status", {}).get("completed"):
            raise SystemExit(f"incomplete report: {report_path}")
        method = report.get("method", {})
        if method.get("policy") != "fixed":
            raise SystemExit(f"shadow input is not fixed policy: {report_path}")
        threshold = float(report["calibration"]["threshold_ms"])
        consecutive_required = int(method.get("ready_consecutive", 3))
        runs_by_label = {run["label"]: run for run in report.get("runs", [])}
        for event in report.get("recovery_events", [])[1:]:
            samples = event.get("samples", [])
            result = shadow_release(samples, threshold, consecutive_required)
            actual_wait = float(event["wait_seconds"])
            if result["resolved"]:
                saved = max(0.0, actual_wait - float(result["release_seconds"]))
            else:
                saved = None
                unresolved += 1
            thermal_boundary = any(
                sample.get("thermal_state") in {"serious", "critical"}
                for sample in samples
            )
            serious_or_critical += int(thermal_boundary)
            next_run = runs_by_label.get(event.get("before_label"), {})
            rows.append({
                "arm": arm_name,
                "event_index": event.get("event_index"),
                "after_label": event.get("after_label"),
                "before_label": event.get("before_label"),
                "next_profile": next_run.get("profile"),
                "next_run_seconds_observed_after_fixed_wait": next_run.get("total_seconds"),
                "threshold_ms": round(threshold, 6),
                "consecutive_required": consecutive_required,
                "actual_fixed_wait_seconds": round(actual_wait, 3),
                "shadow": result,
                "potential_wait_saved_seconds": None if saved is None else round(saved, 3),
                "thermal_performance_boundary_crossed": thermal_boundary,
            })

    resolved_rows = [row for row in rows if row["shadow"]["resolved"]]
    release_times = [float(row["shadow"]["release_seconds"]) for row in resolved_rows]
    saved_times = [float(row["potential_wait_saved_seconds"]) for row in resolved_rows]
    output = {
        "schema_version": "kindred.comfyui.mps_recovery_shadow.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path),
        "method": {
            "dispatch_modified": False,
            "counterfactual_only": True,
            "measured_events_only": True,
            "decision": "first configured consecutive samples at or below each arm calibration threshold",
            "limitation": (
                "Observed next-run timing followed the full fixed wait; it does not prove what "
                "would have happened after the earlier shadow release."
            ),
        },
        "events": rows,
        "summary": {
            "events": len(rows),
            "resolved_events": len(resolved_rows),
            "unresolved_events": unresolved,
            "resolved_fraction": round(len(resolved_rows) / len(rows), 6),
            "median_shadow_release_seconds": round(statistics.median(release_times), 3),
            "minimum_shadow_release_seconds": round(min(release_times), 3),
            "maximum_shadow_release_seconds": round(max(release_times), 3),
            "total_potential_wait_saved_seconds": round(sum(saved_times), 3),
            "median_potential_wait_saved_seconds": round(statistics.median(saved_times), 3),
            "serious_or_critical_events": serious_or_critical,
        },
    }
    atomic_write(args.output.resolve(), output)
    print(json.dumps(output["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
