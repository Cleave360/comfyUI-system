#!/usr/bin/env python3
"""Run governed alternating stock/MLX full-stack Wan video evaluation."""

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

import numpy as np
from PIL import Image
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_apple_silicon as base


ROOT = SCRIPT_DIR.parent
DEFAULT_SEEDS = (770, 771, 772)
CRITERIA = {
    "control_rgb_mae_max": 0.01,
    "candidate_global_rgb_mae_max_each": 1.0,
    "candidate_per_frame_rgb_mae_max_each": 1.25,
    "candidate_per_frame_psnr_db_min_each": 40.0,
    "candidate_temporal_delta_rgb_mae_max_each": 1.5,
    "median_vae_speedup_min": 1.5,
    "median_end_to_end_seconds_saved_min": 2.0,
    "end_to_end_pair_wins_min": 2,
    "mlx_peak_memory_gib_max": 20.0,
    "mlx_cache_memory_bytes_max": 0,
}


def output_paths(result: dict[str, Any]) -> list[Path]:
    root = (ROOT / "output").resolve()
    paths = []
    for item in result["outputs"]:
        path = (root / str(item.get("subfolder") or "") / item["filename"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"output path escapes output directory: {path}")
        paths.append(path)
    return paths


def load_video(result: dict[str, Any]) -> np.ndarray:
    frames = []
    for path in output_paths(result):
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB"), dtype=np.float64))
    if not frames:
        raise ValueError("run has no image frames")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("video frames do not share one shape")
    return np.stack(frames)


def delta_metrics(delta: np.ndarray) -> dict[str, float | str]:
    absolute = np.abs(delta)
    mse = float(np.mean(delta * delta))
    return {
        "rgb_mae": float(np.mean(absolute)),
        "rgb_rmse": math.sqrt(mse),
        "psnr_db": "inf" if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        "max_absolute_error": float(np.max(absolute)),
        "changed_pixel_fraction": float(np.mean(np.any(delta != 0, axis=-1))),
    }


def video_metrics(stock: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    stock_video = load_video(stock)
    candidate_video = load_video(candidate)
    if stock_video.shape != candidate_video.shape:
        raise ValueError(f"video shapes differ: {stock_video.shape} != {candidate_video.shape}")
    per_frame = [
        delta_metrics(stock_video[index] - candidate_video[index])
        for index in range(stock_video.shape[0])
    ]
    temporal = []
    for index in range(1, stock_video.shape[0]):
        stock_delta = stock_video[index] - stock_video[index - 1]
        candidate_delta = candidate_video[index] - candidate_video[index - 1]
        temporal.append(delta_metrics(stock_delta - candidate_delta))
    return {
        "shape": list(stock_video.shape),
        "global": delta_metrics(stock_video - candidate_video),
        "per_frame": per_frame,
        "temporal_delta": temporal,
    }


def sequence_for(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    sequence = [
        {"label": "warmup_stock", "mode": "stock", "seed": seeds[0] - 2},
        {"label": "warmup_mlx", "mode": "mlx", "seed": seeds[0] - 2},
        {"label": "control_a", "mode": "stock", "seed": seeds[0] - 1},
        {"label": "control_b", "mode": "stock", "seed": seeds[0] - 1},
    ]
    for index, seed in enumerate(seeds):
        modes = ("stock", "mlx") if index % 2 == 0 else ("mlx", "stock")
        for mode in modes:
            sequence.append({"label": f"seed_{seed}_{mode}", "mode": mode, "seed": seed})
    return sequence


def add_cache_bust(workflow: dict[str, Any], nonce: int) -> None:
    samplers = [node for node in workflow.values() if node.get("class_type") == "KSampler"]
    if len(samplers) != 1:
        raise ValueError(f"expected exactly one KSampler, found {len(samplers)}")
    sampler = samplers[0]
    latent_input = sampler["inputs"]["latent_image"]
    node_id = "kindred_cache_bust"
    if node_id in workflow:
        raise ValueError(f"workflow already contains reserved node id {node_id}")
    workflow[node_id] = {
        "class_type": "KindredCacheBustLatent",
        "inputs": {"samples": latent_input, "nonce": nonce},
    }
    sampler["inputs"]["latent_image"] = [node_id, 0]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8288")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if len(seeds) != 3 or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain exactly three unique integer seeds")
    if args.cooldown_seconds < 0 or args.steps < 1:
        raise SystemExit("--cooldown-seconds must be non-negative and --steps positive")
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
    sequence = sequence_for(seeds)
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.wan_mlx_vae_e2e.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "server": base_url,
            "same_process": True,
            "same_seed_pairs": True,
            "pair_order_alternates": True,
            "warmup_runs_excluded_from_analysis": True,
            "stock_stock_control": True,
            "cache_bust_scope": "latent pass-through immediately upstream of KSampler",
            "cooldown_seconds_between_runs": args.cooldown_seconds,
            "diffusion_steps": args.steps,
            "criteria_predeclared": CRITERIA,
            "sequence": sequence,
        },
        "runs": [],
    }
    write_report(args.output, report)

    for index, item in enumerate(sequence, start=1):
        if index > 1 and args.cooldown_seconds:
            print(f"cooldown {args.cooldown_seconds:.0f}s before run {index}", flush=True)
            time.sleep(args.cooldown_seconds)
        prefix = f"benchmarks/wan-mlx-vae-e2e/{index:02d}_{item['label']}"
        workflow = base.configure_workflow(
            source, prompt=prompt, seed=item["seed"], prefix=prefix, steps=args.steps,
        )
        add_cache_bust(workflow, index)
        print(f"run {index}/{len(sequence)}: {item['label']}", flush=True)
        result = base.run_once(
            session, base_url, workflow, gate,
            profile=f"wan-mlx-vae-e2e:{item['mode']}", run_number=index,
            seed=item["seed"], timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        result["label"] = item["label"]
        result["mode"] = item["mode"]
        report["runs"].append(result)
        write_report(args.output, report)

    by_label = {run["label"]: run for run in report["runs"]}
    control = video_metrics(by_label["control_a"], by_label["control_b"])
    pairs = []
    for seed in seeds:
        stock = by_label[f"seed_{seed}_stock"]
        mlx = by_label[f"seed_{seed}_mlx"]
        pairs.append({
            "seed": seed,
            "stock_seconds": stock["total_seconds"],
            "mlx_seconds": mlx["total_seconds"],
            "end_to_end_seconds_saved": round(stock["total_seconds"] - mlx["total_seconds"], 3),
            "video": video_metrics(stock, mlx),
        })

    server_report = json.loads(args.server_report.read_text(encoding="utf-8"))
    records = server_report["records"]
    observed_modes = [record["mode"] for record in records]
    expected_modes = [item["mode"] for item in sequence]
    records_by_label = {
        item["label"]: record for item, record in zip(sequence, records, strict=False)
    }
    measured_records = [
        records_by_label[f"seed_{seed}_{mode}"]
        for seed in seeds for mode in ("stock", "mlx")
    ]
    stock_vae = [record["seconds"] for record in measured_records if record["mode"] == "stock"]
    mlx_vae = [record["seconds"] for record in measured_records if record["mode"] == "mlx"]
    median_vae_speedup = statistics.median(stock_vae) / statistics.median(mlx_vae)
    end_to_end_saved = [pair["end_to_end_seconds_saved"] for pair in pairs]
    e2e_wins = sum(value > 0 for value in end_to_end_saved)
    mlx_records = [record for record in records if record["mode"] == "mlx"]

    quality_pass = True
    for pair in pairs:
        video = pair["video"]
        frame_mae = [item["rgb_mae"] for item in video["per_frame"]]
        frame_psnr = [
            math.inf if item["psnr_db"] == "inf" else float(item["psnr_db"])
            for item in video["per_frame"]
        ]
        temporal_mae = [item["rgb_mae"] for item in video["temporal_delta"]]
        quality_pass = quality_pass and (
            video["global"]["rgb_mae"] <= CRITERIA["candidate_global_rgb_mae_max_each"]
            and max(frame_mae) <= CRITERIA["candidate_per_frame_rgb_mae_max_each"]
            and min(frame_psnr) >= CRITERIA["candidate_per_frame_psnr_db_min_each"]
            and max(temporal_mae) <= CRITERIA["candidate_temporal_delta_rgb_mae_max_each"]
        )

    paired_input_hashes = all(
        records_by_label[f"seed_{seed}_stock"]["input_sha256"]
        == records_by_label[f"seed_{seed}_mlx"]["input_sha256"]
        for seed in seeds
    )
    gates = {
        "determinism": control["global"]["rgb_mae"] <= CRITERIA["control_rgb_mae_max"],
        "quality": quality_pass,
        "vae_speedup": median_vae_speedup >= CRITERIA["median_vae_speedup_min"],
        "end_to_end_median_saved": statistics.median(end_to_end_saved) >= CRITERIA[
            "median_end_to_end_seconds_saved_min"
        ],
        "end_to_end_pair_wins": e2e_wins >= CRITERIA["end_to_end_pair_wins_min"],
        "mode_sequence": observed_modes == expected_modes,
        "one_decode_per_run": len(records) == len(sequence),
        "paired_input_hashes": paired_input_hashes,
        "genuine_temporal_latents": all(
            record["input_shape"][0:3] == [1, 16, 5] for record in records
        ),
        "seventeen_frame_outputs": all(record["output_shape"][1] == 17 for record in records),
        "finite_cpu_outputs": all(
            record["all_finite"] and record["output_device"] == "cpu" for record in records
        ),
        "valid_outputs": all(run["status"] == "success" for run in report["runs"]),
        "mlx_peak_memory": max(record["mlx_peak_memory_bytes"] for record in mlx_records)
        <= CRITERIA["mlx_peak_memory_gib_max"] * 1024**3,
        "mlx_cache_memory": max(record["mlx_cache_memory_bytes"] for record in mlx_records)
        <= CRITERIA["mlx_cache_memory_bytes_max"],
        "server_torch_2_12": str(server_report["runtime"]["torch"]).startswith("2.12."),
    }
    report["server_runtime"] = server_report["runtime"]
    report["analysis"] = {
        "control": control,
        "pairs": pairs,
        "vae_timing": {
            "stock_median_seconds": statistics.median(stock_vae),
            "mlx_median_seconds": statistics.median(mlx_vae),
            "median_speedup": median_vae_speedup,
            "median_seconds_saved": statistics.median(stock_vae) - statistics.median(mlx_vae),
        },
        "end_to_end_timing": {
            "median_seconds_saved": statistics.median(end_to_end_saved),
            "min_seconds_saved": min(end_to_end_saved),
            "max_seconds_saved": max(end_to_end_saved),
            "pair_wins": e2e_wins,
        },
        "memory": {
            "mlx_peak_memory_bytes": max(record["mlx_peak_memory_bytes"] for record in mlx_records),
            "mlx_max_active_memory_bytes": max(record["mlx_active_memory_bytes"] for record in mlx_records),
            "mlx_max_cache_memory_bytes": max(record["mlx_cache_memory_bytes"] for record in mlx_records),
        },
        "server_evidence": {
            "observed_modes": observed_modes,
            "record_count": len(records),
            "paired_input_hashes": paired_input_hashes,
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
