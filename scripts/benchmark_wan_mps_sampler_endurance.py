#!/usr/bin/env python3
"""Measure repeated full-stack Wan sampler drift in one isolated process."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_apple_silicon as base
from benchmark_wan_mlx_vae_e2e import add_cache_bust, write_report


ROOT = SCRIPT_DIR.parent
CRITERIA = {
    "sampler_p95_to_median_max": 1.35,
    "sampler_last_half_to_first_half_median_max": 1.25,
    "total_p95_to_median_max": 1.35,
    "total_last_half_to_first_half_median_max": 1.25,
    "torch_driver_growth_gib_max": 1.0,
    "process_rss_growth_gib_max": 1.0,
}


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def stability(values: list[float]) -> dict[str, float]:
    midpoint = len(values) // 2
    median = statistics.median(values)
    return {
        "median_seconds": median,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "p95_seconds": p95(values),
        "p95_to_median": p95(values) / median,
        "last_half_to_first_half_median": (
            statistics.median(values[midpoint:]) / statistics.median(values[:midpoint])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8288")
    parser.add_argument("--seed", type=int, default=773)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 1 or args.runs < 4:
        raise SystemExit("at least one warmup and four measured runs are required")
    args.server_report = args.server_report.resolve()
    args.output = args.output.resolve()

    base.load_dotenv()
    source = base.load_workflow("wan-native-video")
    prompt = base.PROFILES["wan-native-video"]["prompt"]
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    response = session.get(f"{base_url}/system_stats", timeout=15)
    response.raise_for_status()
    gate = base.GovernanceGate(ROOT)
    count = args.warmups + args.runs
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.wan_mps_sampler_endurance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "server": base_url,
            "same_process": True,
            "fixed_seed": args.seed,
            "warmups_excluded": args.warmups,
            "measured_runs": args.runs,
            "diffusion_steps": args.steps,
            "cooldown_seconds": args.cooldown_seconds,
            "criteria_predeclared": CRITERIA,
            "production_modified": False,
        },
        "runs": [],
    }
    write_report(args.output, report)

    for index in range(count):
        if index and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        label = f"{'warmup' if index < args.warmups else 'measured'}_{index + 1:02d}"
        workflow = base.configure_workflow(
            source, prompt=prompt, seed=args.seed,
            prefix=f"benchmarks/wan-sampler-endurance/{label}", steps=args.steps,
        )
        add_cache_bust(workflow, index + 1)
        print(f"run {index + 1}/{count}: {label}", flush=True)
        result = base.run_once(
            session, base_url, workflow, gate,
            profile="wan-mps-sampler-endurance", run_number=index + 1,
            seed=args.seed, timeout=args.timeout, sample_interval=args.sample_interval,
        )
        result["label"] = label
        report["runs"].append(result)
        write_report(args.output, report)

    server = json.loads(args.server_report.read_text(encoding="utf-8"))
    samplers = server["sampler_records"]
    decodes = server["records"]
    measured_slice = slice(args.warmups, count)
    measured_runs = report["runs"][measured_slice]
    measured_samplers = samplers[measured_slice]
    sampler_values = [item["seconds"] for item in measured_samplers]
    total_values = [item["total_seconds"] for item in measured_runs]
    sampler_stability = stability(sampler_values)
    total_stability = stability(total_values)
    baseline_memory = samplers[args.warmups]["memory_before"]
    end_memory = decodes[-1]["memory_after_cleanup"]
    memory_growth = {
        key: end_memory[key] - baseline_memory[key]
        for key in baseline_memory
    }
    gates = {
        "one_sampler_per_run": len(samplers) == count,
        "one_decode_per_run": len(decodes) == count,
        "all_runs_valid": all(item["status"] == "success" for item in report["runs"]),
        "fixed_sampler_output": len({item["output_sha256"] for item in decodes}) == 1,
        "sampler_p95_stability": sampler_stability["p95_to_median"]
        <= CRITERIA["sampler_p95_to_median_max"],
        "sampler_drift": sampler_stability["last_half_to_first_half_median"]
        <= CRITERIA["sampler_last_half_to_first_half_median_max"],
        "total_p95_stability": total_stability["p95_to_median"]
        <= CRITERIA["total_p95_to_median_max"],
        "total_drift": total_stability["last_half_to_first_half_median"]
        <= CRITERIA["total_last_half_to_first_half_median_max"],
        "torch_driver_growth": memory_growth["torch_driver_allocated_bytes"]
        <= CRITERIA["torch_driver_growth_gib_max"] * 1024**3,
        "process_rss_growth": memory_growth["process_rss_bytes"]
        <= CRITERIA["process_rss_growth_gib_max"] * 1024**3,
    }
    report["server_method"] = server["method"]
    report["analysis"] = {
        "sampler_stability": sampler_stability,
        "total_stability": total_stability,
        "decode_seconds": [item["seconds"] for item in decodes[measured_slice]],
        "memory_growth": memory_growth,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
