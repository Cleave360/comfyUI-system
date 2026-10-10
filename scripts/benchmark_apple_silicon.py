#!/usr/bin/env python3
"""Benchmark production ComfyUI image workflows on Apple Silicon.

Runs are admitted through the repository's Adaptive governance gate before a
prompt reaches ComfyUI. Reports contain timings and memory observations, but
never copy ComfyUI's launch argv because it may contain credentials.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any
from uuid import uuid4

import requests


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "kindred-avatar" / "backend"
sys.path.insert(0, str(BACKEND))

from governance import GovernanceGate  # noqa: E402


PROFILES = {
    "interactive": {
        "workflow": "kindred-avatar/backend/workflows/flux_quick.json",
        "description": "FLUX Schnell, 512x512, 2 steps; Jazzy interaction latency",
        "prompt": "A refined studio portrait of Jazzy, a warm futuristic AI host, clean neutral background",
    },
    "brand-social": {
        "workflow": "kindred-avatar/backend/workflows/kindred_brand_social.json",
        "description": "FLUX Schnell, 1080x1080, 6 steps; Branding Lab social asset",
        "prompt": "A governed AI command centre represented as a luminous architectural system",
    },
    "property-quality": {
        "workflow": "kindred-avatar/backend/workflows/flux_quality.json",
        "description": "FLUX Dev, 1024x1024, 20 steps; Property Social quality render",
        "prompt": "Editorial photograph of a beautifully renovated British townhouse interior, natural daylight, accurate materials",
    },
}


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"").strip("'"))


def load_workflow(profile: str) -> dict[str, Any]:
    path = ROOT / PROFILES[profile]["workflow"]
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        raise ValueError(f"workflow is not a non-empty API graph: {path}")
    invalid = [node_id for node_id, node in workflow.items() if not isinstance(node, dict) or not node.get("class_type")]
    if invalid:
        raise ValueError(f"workflow must use ComfyUI API format; invalid nodes: {invalid[:5]}")
    return workflow


def configure_workflow(
    source: dict[str, Any], *, prompt: str, seed: int, prefix: str,
    width: int | None = None, height: int | None = None, steps: int | None = None,
) -> dict[str, Any]:
    workflow = deepcopy(source)
    prompt_nodes: list[dict[str, Any]] = []
    for node in workflow.values():
        node_type = node.get("class_type")
        inputs = node.setdefault("inputs", {})
        if node_type == "CLIPTextEncode" and isinstance(inputs.get("text"), str):
            prompt_nodes.append(node)
        elif node_type == "RandomNoise" and "noise_seed" in inputs:
            inputs["noise_seed"] = seed
        elif node_type == "EmptyLatentImage":
            if width is not None:
                inputs["width"] = width
            if height is not None:
                inputs["height"] = height
        elif node_type == "BasicScheduler" and steps is not None:
            inputs["steps"] = steps
        elif node_type == "SaveImage":
            inputs["filename_prefix"] = prefix

    replaced = False
    for node in prompt_nodes:
        text = node["inputs"]["text"]
        if "PROMPT_PLACEHOLDER" in text:
            node["inputs"]["text"] = text.replace("PROMPT_PLACEHOLDER", prompt)
            replaced = True
    if not replaced:
        if not prompt_nodes:
            raise ValueError("workflow has no CLIPTextEncode prompt node")
        prompt_nodes[0]["inputs"]["text"] = prompt
    return workflow


def sanitized_stats(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep memory/device evidence while excluding sensitive process argv."""
    result = deepcopy(payload)
    system = result.get("system")
    if isinstance(system, dict):
        system.pop("argv", None)
    return result


def memory_observation(stats: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    system = stats.get("system", {}) if isinstance(stats, dict) else {}
    devices = stats.get("devices", []) if isinstance(stats, dict) else []
    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "ram_free_bytes": system.get("ram_free"),
        "ram_total_bytes": system.get("ram_total"),
        "devices": [
            {
                "name": device.get("name"),
                "type": device.get("type"),
                "vram_free_bytes": device.get("vram_free"),
                "vram_total_bytes": device.get("vram_total"),
            }
            for device in devices if isinstance(device, dict)
        ],
    }


def output_files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for output in entry.get("outputs", {}).values():
        if not isinstance(output, dict):
            continue
        for item in output.get("images", []):
            if isinstance(item, dict):
                files.append({key: item.get(key) for key in ("filename", "subfolder", "type")})
    return files


def run_once(
    session: requests.Session, base_url: str, workflow: dict[str, Any], gate: GovernanceGate,
    *, profile: str, run_number: int, seed: int, timeout: float, sample_interval: float,
) -> dict[str, Any]:
    context = gate.begin(workflow, f"benchmark:{profile}")
    started_wall = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    prompt_id = ""
    samples: list[dict[str, Any]] = []
    try:
        response = session.post(
            f"{base_url}/prompt",
            json={"prompt": workflow, "client_id": f"kindred-benchmark-{uuid4()}"},
            timeout=30,
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        submitted = time.monotonic()
        deadline = started + timeout
        entry: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            history_response = session.get(f"{base_url}/history/{prompt_id}", timeout=15)
            history_response.raise_for_status()
            history = history_response.json()
            if prompt_id in history:
                entry = history[prompt_id]
                break
            stats_response = session.get(f"{base_url}/system_stats", timeout=15)
            stats_response.raise_for_status()
            samples.append(memory_observation(sanitized_stats(stats_response.json()), time.monotonic() - started))
            time.sleep(sample_interval)
        if entry is None:
            raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout:.0f}s")
        status = entry.get("status", {})
        if status.get("status_str") not in (None, "success") or status.get("completed") is False:
            raise RuntimeError(f"ComfyUI reported unsuccessful history status: {status}")
        elapsed = time.monotonic() - started
        gate.finish(context, success=True, detail=f"benchmark={profile}; prompt_id={prompt_id}; seconds={elapsed:.3f}")
        return {
            "run": run_number,
            "seed": seed,
            "prompt_id": prompt_id,
            "started_at": started_wall,
            "submit_seconds": round(submitted - started, 3),
            "total_seconds": round(elapsed, 3),
            "outputs": output_files(entry),
            "memory_samples": samples,
            "status": "success",
        }
    except Exception as exc:
        gate.finish(context, success=False, detail=f"benchmark={profile}; prompt_id={prompt_id}; error={exc}")
        raise


def runtime_metadata() -> dict[str, Any]:
    torch_data: dict[str, Any]
    try:
        import torch
        torch_data = {
            "version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
        }
    except ImportError:
        torch_data = {"available": False}
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch_data,
    }


def inspect_profile(name: str) -> dict[str, Any]:
    workflow = load_workflow(name)
    nodes = [node for node in workflow.values() if isinstance(node, dict)]
    return {
        "name": name,
        **PROFILES[name],
        "models": [node["inputs"] for node in nodes if node.get("class_type") in {"UNETLoader", "CheckpointLoaderSimple"}],
        "latent": [node["inputs"] for node in nodes if node.get("class_type") == "EmptyLatentImage"],
        "scheduler": [node["inputs"] for node in nodes if node.get("class_type") == "BasicScheduler"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    inspect_cmd = sub.add_parser("inspect", help="show benchmark profiles without contacting ComfyUI")
    inspect_cmd.add_argument("--profile", choices=sorted(PROFILES))
    run = sub.add_parser("run", help="execute a governed local benchmark")
    run.add_argument("--profile", choices=sorted(PROFILES), required=True)
    run.add_argument("--base-url", default="http://127.0.0.1:8188")
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--seed", type=int, default=360)
    run.add_argument("--prompt")
    run.add_argument("--width", type=int)
    run.add_argument("--height", type=int)
    run.add_argument("--steps", type=int)
    run.add_argument("--timeout", type=float, default=900)
    run.add_argument("--sample-interval", type=float, default=1.0)
    run.add_argument("--no-cold-first", action="store_true")
    run.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    if args.command == "inspect":
        names = [args.profile] if args.profile else sorted(PROFILES)
        print(json.dumps([inspect_profile(name) for name in names], indent=2))
        return 0
    if args.runs < 1 or args.sample_interval <= 0 or args.timeout <= 0:
        raise SystemExit("runs, sample-interval, and timeout must be positive")

    load_dotenv()
    source = load_workflow(args.profile)
    prompt = args.prompt or PROFILES[args.profile]["prompt"]
    base_url = args.base_url.rstrip("/")
    session = requests.Session()
    availability_response = session.get(f"{base_url}/system_stats", timeout=15)
    availability_response.raise_for_status()
    gate = GovernanceGate(ROOT)
    if not args.no_cold_first:
        free_response = session.post(
            f"{base_url}/free", json={"unload_models": True, "free_memory": True}, timeout=30,
        )
        free_response.raise_for_status()
        # ComfyUI handles the free request asynchronously between worker loops.
        time.sleep(1)
    initial_response = session.get(f"{base_url}/system_stats", timeout=15)
    initial_response.raise_for_status()
    initial_stats = sanitized_stats(initial_response.json())

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runs = []
    for index in range(1, args.runs + 1):
        run_seed = args.seed + index - 1
        prefix = f"benchmarks/{args.profile}/{run_stamp}_run_{index:02d}"
        workflow = configure_workflow(
            source, prompt=prompt, seed=run_seed, prefix=prefix,
            width=args.width, height=args.height, steps=args.steps,
        )
        print(f"run {index}/{args.runs}: {args.profile}", flush=True)
        runs.append(run_once(
            session, base_url, workflow, gate, profile=args.profile, run_number=index,
            seed=run_seed, timeout=args.timeout, sample_interval=args.sample_interval,
        ))

    timings = [item["total_seconds"] for item in runs]
    report = {
        "schema_version": "kindred.comfyui.apple_silicon_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": inspect_profile(args.profile),
        "prompt": prompt,
        "seed": args.seed,
        "seed_strategy": "base_seed_plus_zero_based_run_index",
        "cold_first": not args.no_cold_first,
        "governance": {"admission": "Adaptive GovernanceGate", "direct_ungoverned_dispatch": False},
        "runtime": runtime_metadata(),
        "comfyui_system_stats": initial_stats,
        "summary": {
            "runs": len(runs),
            "median_seconds": round(statistics.median(timings), 3),
            "min_seconds": round(min(timings), 3),
            "max_seconds": round(max(timings), 3),
            "warm_median_seconds": round(statistics.median(timings[1:]), 3) if len(timings) > 1 else None,
        },
        "results": runs,
    }
    destination = args.output or ROOT / "reports" / "benchmarks" / f"{run_stamp}_{args.profile}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
