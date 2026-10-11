#!/usr/bin/env python3
"""Run an interleaved same-seed end-to-end Qwen MLX VAE evaluation."""

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
DEFAULT_SEEDS = (7460, 7461, 7462)
CRITERIA = {
    "control_rgb_mae_max": 0.01,
    "candidate_rgb_mae_max_each": 1.0,
    "candidate_psnr_db_min_each": 40.0,
    "median_vae_seconds_saved_min": 0.5,
}


def output_path(result: dict[str, Any]) -> Path:
    item = result["outputs"][0]
    root = (ROOT / "output").resolve()
    path = (root / str(item.get("subfolder") or "") / item["filename"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"output path escapes output directory: {path}")
    return path


def image_metrics(stock: Path, candidate: Path) -> dict[str, float]:
    with Image.open(stock) as stock_image, Image.open(candidate) as candidate_image:
        stock_array = np.asarray(stock_image.convert("RGB"), dtype=np.float64)
        candidate_array = np.asarray(candidate_image.convert("RGB"), dtype=np.float64)
    if stock_array.shape != candidate_array.shape:
        raise ValueError(f"image shapes differ: {stock_array.shape} != {candidate_array.shape}")
    delta = stock_array - candidate_array
    absolute = np.abs(delta)
    mse = float(np.mean(delta * delta))
    return {
        "rgb_mae": float(np.mean(absolute)),
        "rgb_rmse": math.sqrt(mse),
        "psnr_db": math.inf if mse == 0 else 20 * math.log10(255 / math.sqrt(mse)),
        "max_absolute_error": float(np.max(absolute)),
        "changed_pixel_fraction": float(np.mean(np.any(delta != 0, axis=2))),
    }


def sequence_for(seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    sequence = [
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


def finite_metrics(metrics: dict[str, float]) -> dict[str, float | str]:
    return {key: ("inf" if math.isinf(value) else value) for key, value in metrics.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8288")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--server-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must contain unique integer seeds")
    if args.cooldown_seconds < 0 or args.steps < 1:
        raise SystemExit("--cooldown-seconds must be non-negative and --steps positive")

    base.load_dotenv()
    source = base.load_workflow("qwen-quality")
    prompt = base.PROFILES["qwen-quality"]["prompt"]
    session = requests.Session()
    response = session.get(f"{args.base_url.rstrip('/')}/system_stats", timeout=15)
    response.raise_for_status()
    gate = base.GovernanceGate(ROOT)
    sequence = sequence_for(seeds)
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.qwen_mlx_vae_e2e.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "server": args.base_url.rstrip("/"),
            "same_process": True,
            "same_seed_pairs": True,
            "pair_order_alternates": True,
            "cache_bust_scope": "latent pass-through immediately upstream of KSampler",
            "cooldown_seconds_between_runs": args.cooldown_seconds,
            "diffusion_steps": args.steps,
            "criteria_predeclared": CRITERIA,
            "sequence": sequence,
            "end_to_end_timing_is_directional_not_a_gate": True,
        },
        "runs": [],
    }
    write_report(args.output, report)

    for index, item in enumerate(sequence, start=1):
        if index > 1 and args.cooldown_seconds:
            print(f"cooldown {args.cooldown_seconds:.0f}s before run {index}", flush=True)
            time.sleep(args.cooldown_seconds)
        prefix = f"benchmarks/qwen-mlx-vae-e2e/{index:02d}_{item['label']}"
        workflow = base.configure_workflow(
            source, prompt=prompt, seed=item["seed"], prefix=prefix, steps=args.steps,
        )
        add_cache_bust(workflow, index)
        print(f"run {index}/{len(sequence)}: {item['label']}", flush=True)
        result = base.run_once(
            session, args.base_url.rstrip("/"), workflow, gate,
            profile=f"qwen-mlx-vae-e2e:{item['mode']}", run_number=index,
            seed=item["seed"], timeout=args.timeout,
            sample_interval=args.sample_interval,
        )
        result["label"] = item["label"]
        result["mode"] = item["mode"]
        report["runs"].append(result)
        write_report(args.output, report)

    by_label = {run["label"]: run for run in report["runs"]}
    control = finite_metrics(image_metrics(
        output_path(by_label["control_a"]), output_path(by_label["control_b"]),
    ))
    pairs = []
    for seed in seeds:
        stock = by_label[f"seed_{seed}_stock"]
        mlx = by_label[f"seed_{seed}_mlx"]
        pairs.append({
            "seed": seed,
            "stock_seconds": stock["total_seconds"],
            "mlx_seconds": mlx["total_seconds"],
            "end_to_end_seconds_saved": round(stock["total_seconds"] - mlx["total_seconds"], 3),
            "image": finite_metrics(image_metrics(output_path(stock), output_path(mlx))),
        })

    server_report = json.loads(args.server_report.read_text(encoding="utf-8"))
    records = server_report["records"]
    observed_modes = [record["mode"] for record in records]
    expected_modes = [item["mode"] for item in sequence]
    stock_vae = [record["seconds"] for record in records if record["mode"] == "stock"]
    mlx_vae = [record["seconds"] for record in records if record["mode"] == "mlx"]
    median_vae_saved = statistics.median(stock_vae) - statistics.median(mlx_vae)

    quality_pass = all(
        pair["image"]["rgb_mae"] <= CRITERIA["candidate_rgb_mae_max_each"]
        and pair["image"]["psnr_db"] != "inf"
        and pair["image"]["psnr_db"] >= CRITERIA["candidate_psnr_db_min_each"]
        for pair in pairs
    )
    gates = {
        "determinism": control["rgb_mae"] <= CRITERIA["control_rgb_mae_max"],
        "quality": quality_pass,
        "vae_speed": median_vae_saved >= CRITERIA["median_vae_seconds_saved_min"],
        "mode_sequence": observed_modes == expected_modes,
        "one_decode_per_run": len(records) == len(sequence),
        "finite_cpu_outputs": all(
            record["all_finite"] and record["output_device"] == "cpu" for record in records
        ),
        "valid_outputs": all(run["status"] == "success" for run in report["runs"]),
        "server_torch_2_12": str(server_report["runtime"]["torch"]).startswith("2.12."),
    }
    end_to_end_saved = [pair["end_to_end_seconds_saved"] for pair in pairs]
    report["server_runtime"] = server_report["runtime"]
    report["analysis"] = {
        "control": control,
        "pairs": pairs,
        "vae_timing": {
            "stock_median_seconds": statistics.median(stock_vae),
            "mlx_median_seconds": statistics.median(mlx_vae),
            "median_seconds_saved": median_vae_saved,
        },
        "end_to_end_timing_directional": {
            "median_seconds_saved": statistics.median(end_to_end_saved),
            "min_seconds_saved": min(end_to_end_saved),
            "max_seconds_saved": max(end_to_end_saved),
        },
        "server_evidence": {
            "observed_modes": observed_modes,
            "record_count": len(records),
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    write_report(args.output, report)
    print(json.dumps(report["analysis"], indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
