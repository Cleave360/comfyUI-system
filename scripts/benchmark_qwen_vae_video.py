#!/usr/bin/env python3
"""Compare native Torch MPS and zero-copy MLX on temporal Qwen/Wan latents.

The input endpoints are real captured Qwen image latents. Intermediate latent
frames are deterministic linear interpolations, so this is a decoder scaling
and temporal-consistency experiment rather than a video-generation benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Any, Callable

import mlx.core as mx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
MLX_EXAMPLES = ROOT / "reports" / "vendor" / "mlx-examples"
WAN_MODULES = MLX_EXAMPLES / "video" / "wan2.1" / "wan"
DEFAULT_CAPTURES = (
    ROOT / "reports" / "captures" / "qwen_vae_1024x576.pt",
    ROOT / "reports" / "captures" / "qwen_vae_video_endpoint_b.pt",
)
MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]).view(1, 16, 1, 1, 1)
STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]).view(1, 16, 1, 1, 1)
CRITERIA = {
    "global_rgb_mae_max_each": 1.0,
    "per_frame_rgb_mae_max_each": 1.25,
    "per_frame_psnr_db_min_each": 40.0,
    "temporal_delta_rgb_mae_max_each": 1.5,
    "largest_sequence_speedup_min": 1.25,
    "mlx_peak_memory_gib_max": 20.0,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "median_seconds": statistics.median(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "p95_seconds": ordered[p95_index],
    }


def build_temporal_latent(first: torch.Tensor, last: torch.Tensor, frames: int) -> torch.Tensor:
    if frames < 1:
        raise ValueError("frames must be positive")
    expected = (1, 16, 1)
    if tuple(first.shape[:3]) != expected or first.shape != last.shape:
        raise ValueError("captured endpoints must have matching [1,16,1,H,W] shapes")
    if frames == 1:
        return first.clone()
    values = []
    for index in range(frames):
        alpha = index / (frames - 1)
        values.append(first * (1.0 - alpha) + last * alpha)
    return torch.cat(values, dim=2)


def image_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate_np = candidate.detach().float().cpu().numpy().astype(np.float64)
    reference_np = reference.detach().float().cpu().numpy().astype(np.float64)
    if candidate_np.shape != reference_np.shape or candidate_np.ndim != 5:
        raise ValueError("candidate and reference must share [B,T,H,W,C] shape")

    def metrics(delta: np.ndarray) -> dict[str, float | str]:
        absolute = np.abs(delta)
        mse = float(np.mean(np.square(delta)))
        return {
            "rgb_mae": float(np.mean(absolute)) * 255.0,
            "rgb_rmse": math.sqrt(mse) * 255.0,
            "psnr_db": "inf" if mse == 0 else 10.0 * math.log10(1.0 / mse),
            "max_rgb_error": float(np.max(absolute)) * 255.0,
        }

    frame_metrics = [
        metrics(candidate_np[:, index] - reference_np[:, index])
        for index in range(candidate_np.shape[1])
    ]
    temporal = []
    for index in range(1, candidate_np.shape[1]):
        candidate_delta = candidate_np[:, index] - candidate_np[:, index - 1]
        reference_delta = reference_np[:, index] - reference_np[:, index - 1]
        temporal.append(metrics(candidate_delta - reference_delta))
    return {
        "global": metrics(candidate_np - reference_np),
        "per_frame": frame_metrics,
        "temporal_delta": temporal,
    }


def load_official_wan_vae_class():
    package = types.ModuleType("kindred_mlx_wan_video_reference")
    package.__path__ = [str(WAN_MODULES)]
    sys.modules[package.__name__] = package
    for name in ("vae_layers", "vae"):
        qualified = f"{package.__name__}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, WAN_MODULES / f"{name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import official MLX module: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return sys.modules[f"{package.__name__}.vae"].WanVAE


def eager_cache_decode(model: Any, latent: mx.array) -> mx.array:
    """Evaluate temporal feature caches per step to bound lazy graph retention."""
    latent = latent[None]
    scale = 1.0 / model.std
    latent = latent / scale.reshape(1, 1, 1, 1, model.z_dim) + model.mean.reshape(
        1, 1, 1, 1, model.z_dim,
    )
    decoded_input = model.conv2(latent)
    mx.eval(decoded_input)
    feature_cache = [None] * model.decoder.num_cache_slots
    outputs = []
    for index in range(decoded_input.shape[1]):
        frame = decoded_input[:, index:index + 1]
        output, feature_cache = model._compiled_decode(frame, feature_cache)
        mx.eval(output, *feature_cache)
        outputs.append(output)
    return mx.clip(mx.concatenate(outputs, axis=1), -1.0, 1.0)[0]


def configure_comfy_imports() -> None:
    production_sites = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
    if len(production_sites) != 1:
        raise RuntimeError("expected exactly one production site-packages directory")
    sys.path.append(str(production_sites[0]))
    sys.path.insert(0, str(SOURCE))
    sys.modules["torchaudio"] = None
    sys.argv = [sys.argv[0]]
    os.chdir(SOURCE)


def timed_decode(decode: Callable[[torch.Tensor], torch.Tensor], latent: torch.Tensor) -> tuple[float, torch.Tensor]:
    torch.mps.synchronize()
    started = time.perf_counter()
    output = decode(latent)
    torch.mps.synchronize()
    return time.perf_counter() - started, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-capture", type=Path, default=DEFAULT_CAPTURES[0])
    parser.add_argument("--last-capture", type=Path, default=DEFAULT_CAPTURES[1])
    parser.add_argument(
        "--weights", type=Path,
        default=SOURCE / "models" / "vae" / "qwen_image_vae.safetensors",
    )
    parser.add_argument("--latent-frames", default="1,2,4")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument(
        "--mlx-cache-policy", choices=("official", "eager_bounded"), default="official",
        help="Official reference decode or eager temporal-cache materialization plus cache release.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "benchmarks" / "qwen_vae_video.json",
    )
    args = parser.parse_args()
    paths = (args.first_capture.resolve(), args.last_capture.resolve(), args.weights.resolve())
    args.output = args.output.resolve()
    latent_frames = tuple(int(value) for value in args.latent_frames.split(",") if value.strip())
    if not all(path.is_file() for path in paths) or not (WAN_MODULES / "vae.py").is_file():
        raise SystemExit("two captures, VAE weights, and the pinned MLX reference are required")
    if args.warmups < 0 or args.iterations < 2 or args.iterations % 2:
        raise SystemExit("warmups must be non-negative; iterations must be an even integer >= 2")
    if not latent_frames or tuple(sorted(set(latent_frames))) != latent_frames:
        raise SystemExit("--latent-frames must be unique, positive, ascending integers")
    if latent_frames[0] < 1:
        raise SystemExit("--latent-frames values must be positive")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is required")

    first_capture = torch.load(paths[0], map_location="cpu", weights_only=False)
    last_capture = torch.load(paths[1], map_location="cpu", weights_only=False)
    first = first_capture["samples"].float()
    last = last_capture["samples"].float()
    if tensor_sha256(first) == tensor_sha256(last):
        raise SystemExit("captures must contain distinct real latents")

    configure_comfy_imports()
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    import comfy.utils

    state = comfy.utils.load_torch_file(str(paths[2]), safe_load=True)
    torch_vae = comfy.sd.VAE(sd=state)
    WanVAE = load_official_wan_vae_class()
    mlx_vae = WanVAE()
    mlx_weights = WanVAE.sanitize(mx.load(str(paths[2])))
    mlx_vae.load_weights(list(mlx_weights.items()), strict=False)
    mx.eval(mlx_vae.parameters())

    def stock_decode(latent: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch_vae.decode(latent).detach().cpu()

    def mlx_decode(latent: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            rounded = latent.to(torch.bfloat16).float()
            mean = MEAN.to(rounded.device, torch.float32)
            std = STD.to(rounded.device, torch.float32)
            normalized = ((rounded - mean) / std).permute(0, 2, 3, 4, 1).contiguous()
        outputs = []
        for sample in normalized:
            source_mps = sample.to("mps")
            torch.mps.synchronize()
            source_mlx = mx.asarray(source_mps, copy=False)
            decoded = (
                mlx_vae.decode(source_mlx)
                if args.mlx_cache_policy == "official"
                else eager_cache_decode(mlx_vae, source_mlx)
            )
            pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
            mx.eval(pixels)
            pixels_torch = torch.as_tensor(pixels)
            torch.mps.synchronize()
            outputs.append(pixels_torch.cpu())
        result = torch.stack(outputs)
        if args.mlx_cache_policy == "eager_bounded":
            del decoded, pixels, pixels_torch, source_mlx, source_mps
            mx.clear_cache()
        return result

    report: dict[str, Any] = {
        "schema_version": "kindred.comfyui.qwen_vae_video.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mps_available": torch.backends.mps.is_available(),
            "mlx_device": str(mx.default_device()),
        },
        "reference": {
            "mlx_examples_commit": subprocess.check_output(
                ["git", "-C", str(MLX_EXAMPLES), "rev-parse", "HEAD"], text=True,
            ).strip(),
            "module": "video/wan2.1/wan/vae.py",
        },
        "artifacts": {
            "first_capture": str(paths[0]),
            "first_capture_sha256": file_sha256(paths[0]),
            "last_capture": str(paths[1]),
            "last_capture_sha256": file_sha256(paths[1]),
            "weights": str(paths[2]),
            "weights_sha256": file_sha256(paths[2]),
        },
        "method": {
            "endpoint_latents_are_real_qwen_captures": True,
            "intermediate_latents": "deterministic linear interpolation between real endpoints",
            "video_generation_quality_claimed": False,
            "decoder_scaling_and_temporal_consistency_only": True,
            "latent_frames": latent_frames,
            "warmups_per_decoder_and_length": args.warmups,
            "measured_iterations_per_decoder_and_length": args.iterations,
            "mode_order": "stock,mlx then mlx,stock alternating",
            "cpu_input_and_cpu_output_match_comfy_boundary": True,
            "torch_to_mlx": "mx.asarray(torch_mps_tensor, copy=False)",
            "mlx_to_torch": "torch.as_tensor(mlx_array)",
            "mlx_cache_policy": args.mlx_cache_policy,
            "cache_release_in_timed_boundary": args.mlx_cache_policy == "eager_bounded",
            "criteria_predeclared": CRITERIA,
            "production_environment_modified": False,
        },
        "results": [],
    }
    atomic_write_json(args.output, report)

    for latent_count in latent_frames:
        latent = build_temporal_latent(first, last, latent_count)
        expected_output_frames = 1 + 4 * (latent_count - 1)
        for _ in range(args.warmups):
            stock_decode(latent)
            mlx_decode(latent)

        timings: dict[str, list[float]] = {"stock": [], "mlx": []}
        outputs: dict[str, torch.Tensor] = {}
        mx.reset_peak_memory()
        if hasattr(torch.mps, "reset_peak_memory_stats"):
            torch.mps.reset_peak_memory_stats()
        for pair_index in range(args.iterations // 2):
            order = ("stock", "mlx") if pair_index % 2 == 0 else ("mlx", "stock")
            for mode in order:
                seconds, output = timed_decode(
                    stock_decode if mode == "stock" else mlx_decode, latent,
                )
                timings[mode].append(seconds)
                outputs[mode] = output
                gc.collect()
            reverse = tuple(reversed(order))
            for mode in reverse:
                seconds, output = timed_decode(
                    stock_decode if mode == "stock" else mlx_decode, latent,
                )
                timings[mode].append(seconds)
                outputs[mode] = output
                gc.collect()

        quality = image_metrics(outputs["mlx"], outputs["stock"])
        stock_summary = summarize(timings["stock"])
        mlx_summary = summarize(timings["mlx"])
        speedup = stock_summary["median_seconds"] / mlx_summary["median_seconds"]
        per_frame_mae = [float(item["rgb_mae"]) for item in quality["per_frame"]]
        per_frame_psnr = [
            math.inf if item["psnr_db"] == "inf" else float(item["psnr_db"])
            for item in quality["per_frame"]
        ]
        temporal_mae = [float(item["rgb_mae"]) for item in quality["temporal_delta"]]
        result = {
            "latent_frames": latent_count,
            "expected_output_frames": expected_output_frames,
            "input_shape": list(latent.shape),
            "input_sha256": tensor_sha256(latent),
            "output_shape": list(outputs["stock"].shape),
            "stock": {
                **stock_summary,
                "median_output_frames_per_second": expected_output_frames / stock_summary["median_seconds"],
                "output_sha256": tensor_sha256(outputs["stock"]),
            },
            "mlx": {
                **mlx_summary,
                "median_output_frames_per_second": expected_output_frames / mlx_summary["median_seconds"],
                "output_sha256": tensor_sha256(outputs["mlx"]),
            },
            "speedup": speedup,
            "quality": quality,
            "mlx_peak_memory_bytes": mx.get_peak_memory(),
            "mlx_active_memory_bytes": mx.get_active_memory(),
            "mlx_cache_memory_bytes": mx.get_cache_memory(),
            "gates": {
                "output_shape_exact": outputs["stock"].shape == outputs["mlx"].shape,
                "output_frame_count_expected": outputs["stock"].shape[1] == expected_output_frames,
                "all_pixels_finite": bool(
                    torch.isfinite(outputs["stock"]).all() and torch.isfinite(outputs["mlx"]).all()
                ),
                "global_rgb_mae": quality["global"]["rgb_mae"] <= CRITERIA["global_rgb_mae_max_each"],
                "per_frame_rgb_mae": max(per_frame_mae) <= CRITERIA["per_frame_rgb_mae_max_each"],
                "per_frame_psnr": min(per_frame_psnr) >= CRITERIA["per_frame_psnr_db_min_each"],
                "temporal_delta_rgb_mae": (
                    not temporal_mae
                    or max(temporal_mae) <= CRITERIA["temporal_delta_rgb_mae_max_each"]
                ),
            },
        }
        result["gates"]["quality_and_structure"] = all(result["gates"].values())
        report["results"].append(result)
        atomic_write_json(args.output, report)
        print(
            f"latent={latent_count} output={outputs['stock'].shape[1]} "
            f"stock={stock_summary['median_seconds']:.3f}s "
            f"mlx={mlx_summary['median_seconds']:.3f}s speedup={speedup:.3f}x "
            f"mae={quality['global']['rgb_mae']:.3f}",
            flush=True,
        )

    largest = report["results"][-1]
    report["gates"] = {
        "all_sequence_quality_and_structure": all(
            result["gates"]["quality_and_structure"] for result in report["results"]
        ),
        "largest_sequence_speedup": largest["speedup"] >= CRITERIA["largest_sequence_speedup_min"],
        "mlx_peak_memory": max(
            result["mlx_peak_memory_bytes"] for result in report["results"]
        ) <= CRITERIA["mlx_peak_memory_gib_max"] * 1024**3,
    }
    report["overall_pass"] = all(report["gates"].values())
    atomic_write_json(args.output, report)
    print(json.dumps({"gates": report["gates"], "overall_pass": report["overall_pass"]}, indent=2))
    print(f"report: {args.output}")
    return 0 if report["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
