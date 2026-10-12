#!/usr/bin/env python3
"""Measure fixed or probe-driven recovery between governed Qwen/Wan runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_apple_silicon as base
from benchmark_mixed_image_video_soak import output_digest, sequence
from benchmark_wan_mlx_vae_e2e import add_cache_bust, write_report
from mps_recovery import (
    MPSLatencyProbe,
    fixed_recovery_window,
    recovery_threshold,
    summarize_runs,
    utc_now,
    wait_for_recovery,
)


ROOT = SCRIPT_DIR.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8288")
    parser.add_argument("--policy", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--wan-steps", type=int, default=4)
    parser.add_argument("--probe-matrix-size", type=int, default=4096)
    parser.add_argument("--probe-iterations", type=int, default=1)
    parser.add_argument("--calibration-samples", type=int, default=9)
    parser.add_argument("--probe-interval-seconds", type=float, default=2.0)
    parser.add_argument("--ready-consecutive", type=int, default=3)
    parser.add_argument("--max-recovery-seconds", type=float, default=120.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 2:
        raise SystemExit("--pairs must be at least two")
    if args.policy == "fixed" and args.cooldown_seconds < 0:
        raise SystemExit("--cooldown-seconds cannot be negative")
    args.output = args.output.resolve()

    base.load_dotenv()
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    response = session.get(f"{base_url}/system_stats", timeout=15)
    response.raise_for_status()
    workflows = {
        profile: base.load_workflow(profile)
        for profile in ("qwen-quality", "wan-native-video")
    }
    gate = base.GovernanceGate(ROOT)
    probe = MPSLatencyProbe(
        matrix_size=args.probe_matrix_size, iterations=args.probe_iterations,
    )
    calibration_records: list[dict[str, Any]] = []
    for index in range(args.calibration_samples):
        sample = probe.measure()
        sample["calibration_index"] = index
        calibration_records.append(sample)
        if index + 1 < args.calibration_samples:
            time.sleep(args.probe_interval_seconds)
    calibration = {
        "samples": calibration_records,
        **recovery_threshold([item["latency_ms"] for item in calibration_records]),
    }
    run_sequence = sequence(args.pairs)
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.mps_recovery_policy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": {"phase": "ready", "completed": False},
        "method": {
            "policy": args.policy,
            "fixed_cooldown_seconds": args.cooldown_seconds,
            "probe_interval_seconds": args.probe_interval_seconds,
            "ready_consecutive": args.ready_consecutive,
            "max_recovery_seconds": args.max_recovery_seconds,
            "probe_matrix_size": args.probe_matrix_size,
            "probe_iterations": args.probe_iterations,
            "thermal_telemetry": probe.thermal.metadata(),
            "warmups_excluded": 2,
            "pairs": args.pairs,
            "qwen_steps": args.qwen_steps,
            "wan_steps": args.wan_steps,
            "sequence": run_sequence,
            "production_modified": False,
        },
        "calibration": calibration,
        "live_recovery_samples": [],
        "recovery_events": [],
        "runs": [],
    }
    write_report(args.output, report)
    experiment_started = time.monotonic()

    for index, item in enumerate(run_sequence):
        if index:
            report["status"] = {
                "phase": "recovery_wait",
                "completed": False,
                "next_label": item["label"],
            }
            write_report(args.output, report)

            def persist_sample(_sample: dict[str, Any]) -> None:
                report["live_recovery_samples"].append(dict(_sample))
                write_report(args.output, report)

            if args.policy == "adaptive":
                event = wait_for_recovery(
                    probe.measure,
                    threshold_ms=calibration["threshold_ms"],
                    consecutive_required=args.ready_consecutive,
                    interval_seconds=args.probe_interval_seconds,
                    max_wait_seconds=args.max_recovery_seconds,
                    on_sample=persist_sample,
                )
            else:
                event = fixed_recovery_window(
                    probe.measure,
                    cooldown_seconds=args.cooldown_seconds,
                    interval_seconds=args.probe_interval_seconds,
                    threshold_ms=calibration["threshold_ms"],
                    on_sample=persist_sample,
                )
            event.update({
                "event_index": len(report["recovery_events"]),
                "after_label": run_sequence[index - 1]["label"],
                "before_label": item["label"],
                "policy": args.policy,
            })
            report["recovery_events"].append(event)
            report["live_recovery_samples"] = []
            write_report(args.output, report)

        profile = item["profile"]
        steps = args.qwen_steps if profile == "qwen-quality" else args.wan_steps
        workflow = base.configure_workflow(
            workflows[profile], prompt=base.PROFILES[profile]["prompt"],
            seed=item["seed"], prefix=f"benchmarks/recovery/{args.policy}/{item['label']}",
            steps=steps,
        )
        add_cache_bust(workflow, index + 1)
        report["status"] = {
            "phase": "workflow",
            "completed": False,
            "current_label": item["label"],
            "current_profile": profile,
        }
        write_report(args.output, report)
        print(f"run {index + 1}/{len(run_sequence)}: {item['label']}", flush=True)
        started_monotonic = time.monotonic()
        result = base.run_once(
            session, base_url, workflow, gate,
            profile=f"recovery:{args.policy}:{profile}", run_number=index + 1,
            seed=item["seed"], timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        result.update({
            "label": item["label"],
            "profile": profile,
            "pixel_sha256": output_digest(result),
            "started_monotonic": started_monotonic,
            "completed_monotonic": time.monotonic(),
            "completed_at": utc_now(),
        })
        report["runs"].append(result)
        write_report(args.output, report)

    report["status"] = {"phase": "complete", "completed": True}
    report["analysis"] = summarize_runs(report["runs"], report["recovery_events"])
    report["analysis"]["experiment_wall_seconds"] = round(
        time.monotonic() - experiment_started, 3,
    )
    measured = report["runs"][2:]
    report["analysis"]["all_runs_valid"] = all(
        item["status"] == "success" for item in report["runs"]
    )
    report["analysis"]["image_deterministic"] = len({
        item["pixel_sha256"] for item in measured
        if item["profile"] == "qwen-quality"
    }) == 1
    report["analysis"]["video_deterministic"] = len({
        item["pixel_sha256"] for item in measured
        if item["profile"] == "wan-native-video"
    }) == 1
    report["analysis"]["recovery_ready_fraction"] = round(
        sum(bool(item["ready"]) for item in report["recovery_events"])
        / max(len(report["recovery_events"]), 1), 6,
    )
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["all_runs_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
