#!/usr/bin/env python3
"""Compare process-isolated Qwen/Wan workers with the mixed-process soak."""

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
from benchmark_mixed_image_video_soak import output_digest, sequence, stability
from benchmark_wan_mlx_vae_e2e import add_cache_bust, write_report


ROOT = SCRIPT_DIR.parent
CRITERIA = {
    "sampler_p95_to_median_max_each": 1.35,
    "sampler_last_half_to_first_half_median_max_each": 1.25,
    "total_p95_to_median_max_each": 1.35,
    "total_last_half_to_first_half_median_max_each": 1.25,
    "baseline_to_isolated_median_speedup_min_each": 1.20,
    "torch_driver_growth_gib_max_each": 1.0,
    "process_rss_growth_gib_max_each": 2.0,
    "combined_process_rss_gib_max": 96.0,
    "mlx_peak_memory_gib_max": 20.0,
    "mlx_cache_memory_bytes_max": 0,
}


def worker_growth(
    samplers: list[dict[str, Any]], end_memory: dict[str, int],
) -> dict[str, int]:
    baseline = samplers[1]["memory_before"]
    return {key: end_memory[key] - baseline[key] for key in baseline}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-url", default="http://127.0.0.1:8288")
    parser.add_argument("--video-url", default="http://127.0.0.1:8289")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--wan-steps", type=int, default=4)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--image-server-report", type=Path, required=True)
    parser.add_argument("--video-server-report", type=Path, required=True)
    parser.add_argument(
        "--baseline-report", type=Path,
        default=ROOT / "reports" / "benchmarks" / "mixed_image_video_soak.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 4:
        raise SystemExit("--pairs must be at least four")
    for name in ("image_server_report", "video_server_report", "baseline_report", "output"):
        setattr(args, name, getattr(args, name).resolve())

    base.load_dotenv()
    workflows = {
        profile: base.load_workflow(profile)
        for profile in ("qwen-quality", "wan-native-video")
    }
    urls = {
        "qwen-quality": args.image_url.rstrip("/"),
        "wan-native-video": args.video_url.rstrip("/"),
    }
    sessions = {profile: requests.Session() for profile in urls}
    audio_nodes = {}
    for profile, url in urls.items():
        response = sessions[profile].get(f"{url}/system_stats", timeout=15)
        response.raise_for_status()
        node_response = sessions[profile].get(f"{url}/object_info/EmptyAudio", timeout=15)
        node_response.raise_for_status()
        audio_nodes[profile] = "EmptyAudio" in node_response.json()

    gate = base.GovernanceGate(ROOT)
    runs = sequence(args.pairs)
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.isolated_image_video_soak.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "separate_processes": True,
            "strict_alternation": True,
            "warmups_excluded": 2,
            "measured_pairs": args.pairs,
            "qwen_steps": args.qwen_steps,
            "wan_steps": args.wan_steps,
            "cooldown_seconds": args.cooldown_seconds,
            "worker_urls": urls,
            "criteria_predeclared": CRITERIA,
            "production_modified": False,
        },
        "audio_nodes": audio_nodes,
        "runs": [],
    }
    write_report(args.output, report)

    for index, item in enumerate(runs):
        if index and args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        profile = item["profile"]
        steps = args.qwen_steps if profile == "qwen-quality" else args.wan_steps
        workflow = base.configure_workflow(
            workflows[profile], prompt=base.PROFILES[profile]["prompt"],
            seed=item["seed"], prefix=f"benchmarks/isolated-soak/{item['label']}",
            steps=steps,
        )
        add_cache_bust(workflow, index + 1)
        print(
            f"run {index + 1}/{len(runs)}: {item['label']} -> {urls[profile]}",
            flush=True,
        )
        result = base.run_once(
            sessions[profile], urls[profile], workflow, gate,
            profile=f"isolated-soak:{profile}", run_number=index + 1,
            seed=item["seed"], timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        result["label"] = item["label"]
        result["profile"] = profile
        result["pixel_sha256"] = output_digest(result)
        report["runs"].append(result)
        write_report(args.output, report)

    workers = {
        "qwen-quality": json.loads(args.image_server_report.read_text(encoding="utf-8")),
        "wan-native-video": json.loads(args.video_server_report.read_text(encoding="utf-8")),
    }
    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    by_profile: dict[str, Any] = {}
    memory_growth: dict[str, Any] = {}
    for profile in ("qwen-quality", "wan-native-video"):
        profile_runs = [item for item in report["runs"][2:] if item["profile"] == profile]
        sampler_records = workers[profile]["sampler_records"]
        total_stats = stability([item["total_seconds"] for item in profile_runs])
        sampler_stats = stability([item["seconds"] for item in sampler_records[1:]])
        baseline_stats = baseline["analysis"]["profiles"][profile]
        by_profile[profile] = {
            "total": total_stats,
            "sampler": sampler_stats,
            "pixel_hashes": [item["pixel_sha256"] for item in profile_runs],
            "baseline_to_isolated_total_median_speedup": (
                baseline_stats["total"]["median_seconds"] / total_stats["median_seconds"]
            ),
            "baseline_to_isolated_sampler_median_speedup": (
                baseline_stats["sampler"]["median_seconds"] / sampler_stats["median_seconds"]
            ),
        }
        if profile == "wan-native-video":
            end_memory = workers[profile]["records"][-1]["memory_after_cleanup"]
        else:
            end_memory = workers[profile]["non_target_decode_records"][-1][
                "memory_after_cleanup"
            ]
        memory_growth[profile] = worker_growth(sampler_records, end_memory)

    qwen_samplers = workers["qwen-quality"]["sampler_records"]
    wan_samplers = workers["wan-native-video"]["sampler_records"]
    wan_decodes = workers["wan-native-video"]["records"]
    combined_max_rss = (
        max(item["memory_after"]["process_rss_bytes"] for item in qwen_samplers)
        + max(item["memory_after"]["process_rss_bytes"] for item in wan_samplers)
    )
    gates = {
        "audio_nodes_available": all(audio_nodes.values()),
        "torchaudio_stable_abi": all(
            worker["runtime"].get("torchaudio") == "2.11.0"
            for worker in workers.values()
        ),
        "separate_worker_pids": workers["qwen-quality"]["runtime"]["process_id"]
        != workers["wan-native-video"]["runtime"]["process_id"],
        "one_sampler_per_worker_workflow": len(qwen_samplers) == args.pairs + 1
        and len(wan_samplers) == args.pairs + 1,
        "one_wan_decode_per_video": len(wan_decodes) == args.pairs + 1,
        "image_worker_has_no_wan_decode": len(workers["qwen-quality"]["records"]) == 0,
        "one_image_decode_per_image": len(
            workers["qwen-quality"]["non_target_decode_records"]
        ) == args.pairs + 1,
        "all_runs_valid": all(item["status"] == "success" for item in report["runs"]),
        "image_deterministic": len(set(by_profile["qwen-quality"]["pixel_hashes"])) == 1,
        "video_deterministic": len(set(by_profile["wan-native-video"]["pixel_hashes"])) == 1,
        "sampler_p95_stability": all(
            value["sampler"]["p95_to_median"]
            <= CRITERIA["sampler_p95_to_median_max_each"]
            for value in by_profile.values()
        ),
        "sampler_drift": all(
            value["sampler"]["last_half_to_first_half_median"]
            <= CRITERIA["sampler_last_half_to_first_half_median_max_each"]
            for value in by_profile.values()
        ),
        "total_p95_stability": all(
            value["total"]["p95_to_median"]
            <= CRITERIA["total_p95_to_median_max_each"]
            for value in by_profile.values()
        ),
        "total_drift": all(
            value["total"]["last_half_to_first_half_median"]
            <= CRITERIA["total_last_half_to_first_half_median_max_each"]
            for value in by_profile.values()
        ),
        "baseline_median_speedup": all(
            value["baseline_to_isolated_total_median_speedup"]
            >= CRITERIA["baseline_to_isolated_median_speedup_min_each"]
            for value in by_profile.values()
        ),
        "torch_driver_growth": all(
            value["torch_driver_allocated_bytes"]
            <= CRITERIA["torch_driver_growth_gib_max_each"] * 1024**3
            for value in memory_growth.values()
        ),
        "process_rss_growth": all(
            value["process_rss_bytes"]
            <= CRITERIA["process_rss_growth_gib_max_each"] * 1024**3
            for value in memory_growth.values()
        ),
        "combined_process_rss": combined_max_rss
        <= CRITERIA["combined_process_rss_gib_max"] * 1024**3,
        "mlx_peak_memory": max(item["mlx_peak_memory_bytes"] for item in wan_decodes)
        <= CRITERIA["mlx_peak_memory_gib_max"] * 1024**3,
        "mlx_cache_memory": max(item["mlx_cache_memory_bytes"] for item in wan_decodes)
        <= CRITERIA["mlx_cache_memory_bytes_max"],
    }
    report["worker_runtime"] = {
        profile: worker["runtime"] for profile, worker in workers.items()
    }
    report["analysis"] = {
        "profiles": by_profile,
        "memory_growth": memory_growth,
        "combined_max_process_rss_bytes": combined_max_rss,
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
