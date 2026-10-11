#!/usr/bin/env python3
"""Run a governed mixed Qwen-image/Wan-video endurance soak."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any
from uuid import uuid4

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_apple_silicon as base
from benchmark_wan_mlx_vae_e2e import add_cache_bust, output_paths, write_report


ROOT = SCRIPT_DIR.parent
CRITERIA = {
    "sampler_p95_to_median_max_each": 1.35,
    "sampler_last_half_to_first_half_median_max_each": 1.25,
    "total_p95_to_median_max_each": 1.35,
    "total_last_half_to_first_half_median_max_each": 1.25,
    "torch_driver_growth_gib_max": 1.0,
    "process_rss_growth_gib_max": 2.0,
    "mlx_peak_memory_gib_max": 20.0,
    "mlx_cache_memory_bytes_max": 0,
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


def output_digest(result: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    paths = output_paths(result)
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            digest.update(str(rgb.size).encode())
            digest.update(rgb.tobytes())
    if not paths:
        raise ValueError("workflow produced no local image outputs")
    return digest.hexdigest()


def wait_for_history(
    session: requests.Session, base_url: str, prompt_id: str, timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = session.get(f"{base_url}/history/{prompt_id}", timeout=15)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.25)
    raise TimeoutError(f"audio smoke prompt {prompt_id} timed out")


def audio_smoke(
    session: requests.Session, base_url: str, gate: base.GovernanceGate,
) -> dict[str, Any]:
    required_nodes = ("EmptyAudio", "SaveAudio", "AudioEncoderLoader")
    object_info = {}
    for node in required_nodes:
        response = session.get(f"{base_url}/object_info/{node}", timeout=15)
        response.raise_for_status()
        object_info[node] = node in response.json()
    workflow = {
        "1": {
            "class_type": "EmptyAudio",
            "inputs": {"duration": 0.25, "sample_rate": 16000, "channels": 1},
        },
        "2": {
            "class_type": "SaveAudio",
            "inputs": {
                "audio": ["1", 0],
                "filename_prefix": "benchmarks/audio-runtime-smoke",
            },
        },
    }
    context = gate.begin(workflow, "benchmark:mixed-soak-audio-smoke")
    prompt_id = ""
    try:
        response = session.post(
            f"{base_url}/prompt",
            json={"prompt": workflow, "client_id": f"kindred-audio-smoke-{uuid4()}"},
            timeout=30,
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        entry = wait_for_history(session, base_url, prompt_id, 120)
        status = entry.get("status", {})
        success = status.get("status_str") in (None, "success") and status.get(
            "completed"
        ) is not False
        audio_outputs = [
            item
            for output in entry.get("outputs", {}).values()
            if isinstance(output, dict)
            for item in output.get("audio", [])
            if isinstance(item, dict)
        ]
        gate.finish(
            context, success=success and bool(audio_outputs),
            detail=f"audio_smoke={success}; prompt_id={prompt_id}; outputs={len(audio_outputs)}",
        )
        return {
            "required_nodes": object_info,
            "prompt_id": prompt_id,
            "success": success,
            "outputs": audio_outputs,
        }
    except Exception as exc:
        gate.finish(context, success=False, detail=f"audio_smoke_error={exc}")
        raise


def sequence(pairs: int) -> list[dict[str, Any]]:
    values = [
        {"label": "warmup_image", "profile": "qwen-quality", "seed": 9100},
        {"label": "warmup_video", "profile": "wan-native-video", "seed": 9200},
    ]
    for index in range(pairs):
        values.extend([
            {"label": f"image_{index + 1:02d}", "profile": "qwen-quality", "seed": 9101},
            {"label": f"video_{index + 1:02d}", "profile": "wan-native-video", "seed": 9201},
        ])
    return values


def analyze(
    report: dict[str, Any], server: dict[str, Any], runs: list[dict[str, Any]],
    pairs: int,
) -> dict[str, Any]:
    samplers = server["sampler_records"]
    decodes = server["records"]
    for result in report["runs"]:
        result["pixel_sha256"] = output_digest(result)
    measured_samplers = samplers[2:]
    by_profile: dict[str, Any] = {}
    for profile in ("qwen-quality", "wan-native-video"):
        indices = [
            index for index, item in enumerate(runs[2:], start=2)
            if item["profile"] == profile
        ]
        profile_runs = [report["runs"][index] for index in indices]
        profile_samplers = [samplers[index] for index in indices]
        by_profile[profile] = {
            "total": stability([item["total_seconds"] for item in profile_runs]),
            "sampler": stability([item["seconds"] for item in profile_samplers]),
            "pixel_hashes": [item["pixel_sha256"] for item in profile_runs],
        }

    baseline_memory = measured_samplers[0]["memory_before"]
    end_memory = decodes[-1]["memory_after_cleanup"]
    memory_growth = {
        key: end_memory[key] - baseline_memory[key] for key in baseline_memory
    }
    mlx_records = [item for item in decodes if item["mode"] == "mlx"]
    audio = report["audio_smoke"]
    gates = {
        "audio_nodes_available": all(audio["required_nodes"].values()),
        "audio_workflow_success": audio["success"] and bool(audio["outputs"]),
        "torchaudio_stable_abi": server["runtime"].get("torchaudio") == "2.11.0",
        "one_sampler_per_workflow": len(samplers) == len(runs),
        "one_wan_decode_per_video": len(decodes) == pairs + 1,
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
        "torch_driver_growth": memory_growth["torch_driver_allocated_bytes"]
        <= CRITERIA["torch_driver_growth_gib_max"] * 1024**3,
        "process_rss_growth": memory_growth["process_rss_bytes"]
        <= CRITERIA["process_rss_growth_gib_max"] * 1024**3,
        "mlx_peak_memory": max(item["mlx_peak_memory_bytes"] for item in mlx_records)
        <= CRITERIA["mlx_peak_memory_gib_max"] * 1024**3,
        "mlx_cache_memory": max(item["mlx_cache_memory_bytes"] for item in mlx_records)
        <= CRITERIA["mlx_cache_memory_bytes_max"],
    }
    return {
        "profiles": by_profile,
        "memory_growth": memory_growth,
        "mlx_peak_memory_bytes": max(item["mlx_peak_memory_bytes"] for item in mlx_records),
        "gates": gates,
        "overall_pass": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8288")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--wan-steps", type=int, default=4)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reanalyze-only", action="store_true")
    args = parser.parse_args()
    if args.pairs < 4:
        raise SystemExit("--pairs must be at least four")
    args.server_report = args.server_report.resolve()
    args.output = args.output.resolve()
    runs = sequence(args.pairs)
    if args.reanalyze_only:
        report = json.loads(args.output.read_text(encoding="utf-8"))
        server = json.loads(args.server_report.read_text(encoding="utf-8"))
        report["analysis"] = analyze(report, server, runs, args.pairs)
        write_report(args.output, report)
        print(json.dumps(report["analysis"], indent=2), flush=True)
        return 0 if report["analysis"]["overall_pass"] else 2

    base.load_dotenv()
    workflows = {
        profile: base.load_workflow(profile)
        for profile in ("qwen-quality", "wan-native-video")
    }
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    response = session.get(f"{base_url}/system_stats", timeout=15)
    response.raise_for_status()
    gate = base.GovernanceGate(ROOT)
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.mixed_image_video_soak.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "server": base_url,
            "same_process": True,
            "strict_alternation": True,
            "warmups_excluded": 2,
            "measured_pairs": args.pairs,
            "qwen_steps": args.qwen_steps,
            "wan_steps": args.wan_steps,
            "cooldown_seconds": args.cooldown_seconds,
            "criteria_predeclared": CRITERIA,
            "sequence": runs,
            "production_modified": False,
        },
        "audio_smoke": audio_smoke(session, base_url, gate),
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
            seed=item["seed"], prefix=f"benchmarks/mixed-soak/{item['label']}",
            steps=steps,
        )
        add_cache_bust(workflow, index + 1)
        print(f"run {index + 1}/{len(runs)}: {item['label']}", flush=True)
        result = base.run_once(
            session, base_url, workflow, gate,
            profile=f"mixed-soak:{profile}", run_number=index + 1,
            seed=item["seed"], timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        result["label"] = item["label"]
        result["profile"] = profile
        result["pixel_sha256"] = output_digest(result)
        report["runs"].append(result)
        write_report(args.output, report)

    server = json.loads(args.server_report.read_text(encoding="utf-8"))
    report["server_runtime"] = server["runtime"]
    report["server_method"] = server["method"]
    report["analysis"] = analyze(report, server, runs, args.pairs)
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
