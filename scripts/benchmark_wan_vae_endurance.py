#!/usr/bin/env python3
"""Diagnose long-run Torch MPS versus MLX Wan VAE decoder stability."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable

import mlx.core as mx
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import benchmark_qwen_vae_video as base


ROOT = SCRIPT_DIR.parent
CRITERIA = {
    "global_rgb_mae_max": 1.0,
    "per_frame_rgb_mae_max": 1.25,
    "per_frame_psnr_db_min": 40.0,
    "temporal_delta_rgb_mae_max": 1.5,
    "median_speedup_min": 1.5,
    "stock_p95_to_median_max": 1.30,
    "mlx_p95_to_median_max": 1.30,
    "stock_last_half_to_first_half_median_max": 1.30,
    "mlx_last_half_to_first_half_median_max": 1.30,
    "torch_driver_growth_gib_max": 1.0,
    "process_rss_growth_gib_max": 1.0,
    "mlx_peak_memory_gib_max": 20.0,
    "mlx_active_growth_mib_max": 64.0,
    "mlx_cache_memory_bytes_max": 0,
}


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def memory_snapshot() -> dict[str, int]:
    process = psutil.Process()
    virtual = psutil.virtual_memory()
    return {
        "torch_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "torch_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "torch_recommended_max_bytes": torch.mps.recommended_max_memory(),
        "mlx_active_memory_bytes": mx.get_active_memory(),
        "mlx_cache_memory_bytes": mx.get_cache_memory(),
        "process_rss_bytes": process.memory_info().rss,
        "system_available_bytes": virtual.available,
    }


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def timed(decode: Callable[[torch.Tensor], torch.Tensor], latent: torch.Tensor):
    torch.mps.synchronize()
    started = time.perf_counter()
    output = decode(latent)
    torch.mps.synchronize()
    return time.perf_counter() - started, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path,
        default=ROOT / "reports" / "captures" / "wan_vace_1.3b_native_video_torch212.pt",
    )
    parser.add_argument(
        "--weights", type=Path,
        default=ROOT / "models" / "vae" / "wan_2.1_vae.safetensors",
    )
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument(
        "--cleanup-policy", choices=("none", "torch_empty_cache"), default="none",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "benchmarks" / "wan_vae_endurance.json",
    )
    args = parser.parse_args()
    args.capture = args.capture.resolve()
    args.weights = args.weights.resolve()
    args.output = args.output.resolve()
    if args.pairs < 4:
        raise SystemExit("--pairs must be at least four")
    if not args.capture.is_file() or not args.weights.is_file():
        raise SystemExit("native Wan capture and VAE weights are required")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is required")

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    latent = payload["samples"].float()
    captured_output = payload["output"].float()
    if tuple(latent.shape[:3]) != (1, 16, 5):
        raise SystemExit("capture must contain one five-frame Wan latent")

    base.configure_comfy_imports()
    global psutil
    import psutil
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    import comfy.utils

    state = comfy.utils.load_torch_file(str(args.weights), safe_load=True)
    torch_vae = comfy.sd.VAE(sd=state)
    WanVAE = base.load_official_wan_vae_class()
    mlx_vae = WanVAE()
    mlx_weights = WanVAE.sanitize(mx.load(str(args.weights)))
    mlx_vae.load_weights(list(mlx_weights.items()), strict=False)
    mx.eval(mlx_vae.parameters())

    def stock_decode(value: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch_vae.decode(value).detach().cpu()

    def mlx_decode(value: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            rounded = value.to(torch.bfloat16).float()
            normalized = (
                (rounded - base.MEAN.to(rounded.device)) / base.STD.to(rounded.device)
            ).permute(0, 2, 3, 4, 1).contiguous()
        source_mps = normalized[0].to("mps")
        torch.mps.synchronize()
        source_mlx = mx.asarray(source_mps, copy=False)
        decoded = base.layer_eager_decode(
            mlx_vae, source_mlx, operation_eager=True, consume_cache=True,
            chunk_spatial_resample=True,
        )
        pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
        mx.eval(pixels)
        pixels_torch = torch.as_tensor(pixels)
        torch.mps.synchronize()
        output = pixels_torch.cpu().unsqueeze(0)
        del decoded, pixels, pixels_torch, source_mlx, source_mps
        mx.clear_cache()
        return output

    # Compile and load both paths before establishing the memory baseline.
    warmups = []
    for mode, decode in (("stock", stock_decode), ("mlx", mlx_decode)):
        seconds, output = timed(decode, latent)
        warmups.append({
            "mode": mode,
            "seconds": seconds,
            "output_sha256": base.tensor_sha256(output),
        })
        del output
        gc.collect()
    baseline_memory = memory_snapshot()

    sequence = []
    for index in range(args.pairs):
        sequence.extend(("stock", "mlx") if index % 2 == 0 else ("mlx", "stock"))
    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.wan_vae_endurance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mps_available": torch.backends.mps.is_available(),
            "mlx_device": str(mx.default_device()),
        },
        "method": {
            "capture": str(args.capture),
            "capture_sha256": base.file_sha256(args.capture),
            "weights": str(args.weights),
            "weights_sha256": base.file_sha256(args.weights),
            "warmups": warmups,
            "pairs": args.pairs,
            "sequence": sequence,
            "cleanup_policy": args.cleanup_policy,
            "criteria_predeclared": CRITERIA,
            "production_modified": False,
        },
        "baseline_memory": baseline_memory,
        "records": [],
    }
    atomic_write(args.output, report)

    outputs: dict[str, torch.Tensor] = {}
    hashes: dict[str, list[str]] = {"stock": [], "mlx": []}
    for index, mode in enumerate(sequence):
        decode = stock_decode if mode == "stock" else mlx_decode
        if mode == "mlx":
            mx.reset_peak_memory()
        if hasattr(torch.mps, "reset_peak_memory_stats"):
            torch.mps.reset_peak_memory_stats()
        before = memory_snapshot()
        seconds, output = timed(decode, latent)
        output_hash = base.tensor_sha256(output)
        after = memory_snapshot()
        record = {
            "index": index,
            "mode": mode,
            "seconds": seconds,
            "output_sha256": output_hash,
            "all_finite": bool(torch.isfinite(output).all()),
            "memory_before": before,
            "memory_after": after,
            "mlx_peak_memory_bytes": mx.get_peak_memory() if mode == "mlx" else None,
        }
        report["records"].append(record)
        hashes[mode].append(output_hash)
        outputs[mode] = output
        atomic_write(args.output, report)
        print(
            f"run={index + 1}/{len(sequence)} mode={mode} seconds={seconds:.3f} "
            f"torch_driver_gib={after['torch_driver_allocated_bytes'] / 1024**3:.3f} "
            f"mlx_cache_gib={after['mlx_cache_memory_bytes'] / 1024**3:.3f}",
            flush=True,
        )
        if args.cleanup_policy == "torch_empty_cache":
            torch.mps.empty_cache()
        gc.collect()

    timings = {
        mode: [record["seconds"] for record in report["records"] if record["mode"] == mode]
        for mode in ("stock", "mlx")
    }
    stability = {}
    for mode, values in timings.items():
        midpoint = len(values) // 2
        stability[mode] = {
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
            "p95_seconds": p95(values),
            "p95_to_median": p95(values) / statistics.median(values),
            "last_half_to_first_half_median": (
                statistics.median(values[midpoint:]) / statistics.median(values[:midpoint])
            ),
        }

    quality = base.image_metrics(outputs["mlx"], outputs["stock"])
    per_frame_mae = [item["rgb_mae"] for item in quality["per_frame"]]
    per_frame_psnr = [
        math.inf if item["psnr_db"] == "inf" else float(item["psnr_db"])
        for item in quality["per_frame"]
    ]
    temporal_mae = [item["rgb_mae"] for item in quality["temporal_delta"]]
    end_memory = memory_snapshot()
    mlx_records = [record for record in report["records"] if record["mode"] == "mlx"]
    gates = {
        "stock_output_deterministic": len(set(hashes["stock"])) == 1,
        "mlx_output_deterministic": len(set(hashes["mlx"])) == 1,
        "stock_capture_replay_exact": hashes["stock"][0] == base.tensor_sha256(captured_output),
        "all_outputs_finite": all(record["all_finite"] for record in report["records"]),
        "quality_global": quality["global"]["rgb_mae"] <= CRITERIA["global_rgb_mae_max"],
        "quality_per_frame_mae": max(per_frame_mae) <= CRITERIA["per_frame_rgb_mae_max"],
        "quality_per_frame_psnr": min(per_frame_psnr) >= CRITERIA["per_frame_psnr_db_min"],
        "quality_temporal": max(temporal_mae) <= CRITERIA["temporal_delta_rgb_mae_max"],
        "median_speedup": stability["stock"]["median_seconds"] / stability["mlx"]["median_seconds"]
        >= CRITERIA["median_speedup_min"],
        "stock_p95_stability": stability["stock"]["p95_to_median"]
        <= CRITERIA["stock_p95_to_median_max"],
        "mlx_p95_stability": stability["mlx"]["p95_to_median"]
        <= CRITERIA["mlx_p95_to_median_max"],
        "stock_drift": stability["stock"]["last_half_to_first_half_median"]
        <= CRITERIA["stock_last_half_to_first_half_median_max"],
        "mlx_drift": stability["mlx"]["last_half_to_first_half_median"]
        <= CRITERIA["mlx_last_half_to_first_half_median_max"],
        "torch_driver_growth": end_memory["torch_driver_allocated_bytes"]
        - baseline_memory["torch_driver_allocated_bytes"]
        <= CRITERIA["torch_driver_growth_gib_max"] * 1024**3,
        "process_rss_growth": end_memory["process_rss_bytes"] - baseline_memory["process_rss_bytes"]
        <= CRITERIA["process_rss_growth_gib_max"] * 1024**3,
        "mlx_peak_memory": max(record["mlx_peak_memory_bytes"] for record in mlx_records)
        <= CRITERIA["mlx_peak_memory_gib_max"] * 1024**3,
        "mlx_active_growth": end_memory["mlx_active_memory_bytes"]
        - baseline_memory["mlx_active_memory_bytes"]
        <= CRITERIA["mlx_active_growth_mib_max"] * 1024**2,
        "mlx_cache_memory": end_memory["mlx_cache_memory_bytes"]
        <= CRITERIA["mlx_cache_memory_bytes_max"],
    }
    report["analysis"] = {
        "stability": stability,
        "median_speedup": stability["stock"]["median_seconds"] / stability["mlx"]["median_seconds"],
        "quality": quality,
        "end_memory": end_memory,
        "memory_growth": {
            key: end_memory[key] - baseline_memory[key] for key in end_memory
        },
        "gates": gates,
        "overall_pass": all(gates.values()),
    }
    atomic_write(args.output, report)
    print(json.dumps({
        "stability": stability,
        "median_speedup": report["analysis"]["median_speedup"],
        "memory_growth": report["analysis"]["memory_growth"],
        "gates": gates,
        "overall_pass": report["analysis"]["overall_pass"],
    }, indent=2), flush=True)
    return 0 if report["analysis"]["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
