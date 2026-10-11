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
import mlx.nn as nn
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


def operation_eager_residual(
    layer: Any,
    value: mx.array,
    first_cache: Any,
    second_cache: Any,
    create_cache_entry: Callable[..., mx.array],
):
    """Bound residual-block intermediates without changing its arithmetic."""
    shortcut = layer.shortcut(value) if layer.shortcut is not None else value
    mx.eval(shortcut)
    residual = nn.silu(layer.norm1(value))
    mx.eval(residual)
    cache_input = residual
    residual = layer.conv1(residual, first_cache)
    new_first = create_cache_entry(cache_input, first_cache)
    mx.eval(shortcut, residual, new_first)
    residual = nn.silu(layer.norm2(residual))
    mx.eval(residual)
    cache_input = residual
    residual = layer.conv2(residual, second_cache)
    new_second = create_cache_entry(cache_input, second_cache)
    mx.eval(shortcut, residual, new_first, new_second)
    output = shortcut + residual
    mx.eval(output, new_first, new_second)
    return output, new_first, new_second


def layer_eager_decoder_step(
    decoder: Any,
    value: mx.array,
    feature_cache: list[Any],
    operation_eager: bool = False,
):
    """Run one latent frame while materializing after each decoder layer."""
    cache_index = 0
    new_cache = []
    layers_module = sys.modules[decoder.conv1.__class__.__module__]
    create_cache_entry = layers_module.create_cache_entry

    def residual(layer: Any, value: mx.array, first_cache: Any, second_cache: Any):
        if operation_eager:
            return operation_eager_residual(
                layer, value, first_cache, second_cache, create_cache_entry,
            )
        return layer(value, first_cache, second_cache)

    cache_input = value
    value = decoder.conv1(value, feature_cache[cache_index])
    # Use the reference helper through the bound module's package so cache
    # construction remains identical to the pinned Apple implementation.
    new_cache.append(create_cache_entry(cache_input, feature_cache[cache_index]))
    cache_index += 1
    mx.eval(value, *new_cache)

    value, first, second = residual(
        decoder.middle_res1,
        value, feature_cache[cache_index], feature_cache[cache_index + 1],
    )
    new_cache.extend((first, second))
    cache_index += 2
    mx.eval(value, *new_cache)

    value = decoder.middle_attn(value)
    mx.eval(value, *new_cache)

    value, first, second = residual(
        decoder.middle_res2,
        value, feature_cache[cache_index], feature_cache[cache_index + 1],
    )
    new_cache.extend((first, second))
    cache_index += 2
    mx.eval(value, *new_cache)

    for stage in decoder.upsamples:
        for layer in stage:
            layer_name = type(layer).__name__
            if layer_name == "ResidualBlock":
                value, first, second = residual(
                    layer,
                    value, feature_cache[cache_index], feature_cache[cache_index + 1],
                )
                new_cache.extend((first, second))
                cache_index += 2
            elif layer_name == "AttentionBlock":
                value = layer(value)
            elif layer_name == "Resample":
                value, cache = layer(value, feature_cache[cache_index])
                if cache is not None:
                    new_cache.append(cache)
                    cache_index += 1
            else:
                raise RuntimeError(f"unexpected decoder layer: {layer_name}")
            mx.eval(value, *new_cache)

    value = decoder.head_norm(value)
    value = nn.silu(value)
    if operation_eager:
        mx.eval(value, *new_cache)
    cache_input = value
    value = decoder.head_conv(value, feature_cache[cache_index])
    new_cache.append(create_cache_entry(cache_input, feature_cache[cache_index]))
    mx.eval(value, *new_cache)
    return value, new_cache


def layer_eager_decode(model: Any, latent: mx.array, operation_eager: bool = False) -> mx.array:
    """Decode with layer-sized graphs to test activation lifetime as the peak source."""
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
        output, feature_cache = layer_eager_decoder_step(
            model.decoder,
            decoded_input[:, index:index + 1],
            feature_cache,
            operation_eager=operation_eager,
        )
        outputs.append(output)
    return mx.clip(mx.concatenate(outputs, axis=1), -1.0, 1.0)[0]


def spatial_tiled_decoder_step(
    decoder: Any,
    value: mx.array,
    state: dict[str, Any] | None,
    halo: int,
):
    """Tile only the attention-free high-resolution decoder tail."""
    final_stage = decoder.upsamples[-1]
    if len(final_stage) != 3 or any(type(layer).__name__ != "ResidualBlock" for layer in final_stage):
        raise RuntimeError("spatial tiling requires the pinned three-residual final Wan stage")
    prefix_slot_count = decoder.num_cache_slots - 7
    previous_prefix = state["prefix"] if state is not None else [None] * prefix_slot_count
    cache_index = 0
    new_prefix = []
    layers_module = sys.modules[decoder.conv1.__class__.__module__]
    create_cache_entry = layers_module.create_cache_entry

    cache_input = value
    value = decoder.conv1(value, previous_prefix[cache_index])
    new_prefix.append(create_cache_entry(cache_input, previous_prefix[cache_index]))
    cache_index += 1
    mx.eval(value, *new_prefix)

    for layer in (decoder.middle_res1,):
        value, first, second = operation_eager_residual(
            layer, value, previous_prefix[cache_index], previous_prefix[cache_index + 1],
            create_cache_entry,
        )
        new_prefix.extend((first, second))
        cache_index += 2
    value = decoder.middle_attn(value)
    mx.eval(value, *new_prefix)
    value, first, second = operation_eager_residual(
        decoder.middle_res2, value,
        previous_prefix[cache_index], previous_prefix[cache_index + 1],
        create_cache_entry,
    )
    new_prefix.extend((first, second))
    cache_index += 2

    for stage in decoder.upsamples[:-1]:
        for layer in stage:
            layer_name = type(layer).__name__
            if layer_name == "ResidualBlock":
                value, first, second = operation_eager_residual(
                    layer, value,
                    previous_prefix[cache_index], previous_prefix[cache_index + 1],
                    create_cache_entry,
                )
                new_prefix.extend((first, second))
                cache_index += 2
            elif layer_name == "AttentionBlock":
                value = layer(value)
            elif layer_name == "Resample":
                prior = previous_prefix[cache_index] if hasattr(layer, "time_conv") else None
                value, cache = layer(value, prior)
                if cache is not None:
                    new_prefix.append(cache)
                    cache_index += 1
            else:
                raise RuntimeError(f"unexpected decoder layer: {layer_name}")
            mx.eval(value, *new_prefix)
    if cache_index != prefix_slot_count:
        raise RuntimeError(f"expected {prefix_slot_count} prefix caches, used {cache_index}")

    height = value.shape[2]
    split = height // 2
    cores = ((0, split), (split, height))
    previous_tiles = state["tiles"] if state is not None else [[None] * 7 for _ in cores]
    output_tiles = []
    new_tiles = []
    for tile_index, (core_start, core_end) in enumerate(cores):
        tile_start = max(0, core_start - halo)
        tile_end = min(height, core_end + halo)
        tile = value[:, :, tile_start:tile_end]
        tile_cache = previous_tiles[tile_index]
        tile_new_cache = []
        tile_cache_index = 0
        for layer in final_stage:
            tile, first, second = operation_eager_residual(
                layer, tile,
                tile_cache[tile_cache_index], tile_cache[tile_cache_index + 1],
                create_cache_entry,
            )
            tile_new_cache.extend((first, second))
            tile_cache_index += 2
        tile = nn.silu(decoder.head_norm(tile))
        mx.eval(tile, *tile_new_cache)
        cache_input = tile
        tile = decoder.head_conv(tile, tile_cache[tile_cache_index])
        tile_new_cache.append(create_cache_entry(cache_input, tile_cache[tile_cache_index]))
        mx.eval(tile, *tile_new_cache)
        local_start = core_start - tile_start
        local_end = local_start + core_end - core_start
        output_tiles.append(tile[:, :, local_start:local_end])
        new_tiles.append(tile_new_cache)
    output = mx.concatenate(output_tiles, axis=2)
    mx.eval(output)
    return output, {"prefix": new_prefix, "tiles": new_tiles}


def spatial_tiled_decode(model: Any, latent: mx.array, halo: int) -> mx.array:
    latent = latent[None]
    scale = 1.0 / model.std
    latent = latent / scale.reshape(1, 1, 1, 1, model.z_dim) + model.mean.reshape(
        1, 1, 1, 1, model.z_dim,
    )
    decoded_input = model.conv2(latent)
    mx.eval(decoded_input)
    state = None
    outputs = []
    for index in range(decoded_input.shape[1]):
        output, state = spatial_tiled_decoder_step(
            model.decoder, decoded_input[:, index:index + 1], state, halo,
        )
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
        "--mlx-cache-policy",
        choices=(
            "official", "eager_bounded", "layer_eager_bounded", "op_eager_bounded",
            "spatial_tiled_bounded",
        ),
        default="official",
        help="Official reference decode or eager temporal-cache materialization plus cache release.",
    )
    parser.add_argument("--spatial-tile-halo", type=int, default=32)
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
    if args.spatial_tile_halo < 7:
        raise SystemExit("--spatial-tile-halo must be at least the seven-pixel tail radius")
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
            if args.mlx_cache_policy == "official":
                decoded = mlx_vae.decode(source_mlx)
            elif args.mlx_cache_policy == "eager_bounded":
                decoded = eager_cache_decode(mlx_vae, source_mlx)
            elif args.mlx_cache_policy == "layer_eager_bounded":
                decoded = layer_eager_decode(mlx_vae, source_mlx)
            elif args.mlx_cache_policy == "op_eager_bounded":
                decoded = layer_eager_decode(mlx_vae, source_mlx, operation_eager=True)
            else:
                decoded = spatial_tiled_decode(mlx_vae, source_mlx, args.spatial_tile_halo)
            pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
            mx.eval(pixels)
            pixels_torch = torch.as_tensor(pixels)
            torch.mps.synchronize()
            outputs.append(pixels_torch.cpu())
        result = torch.stack(outputs)
        if args.mlx_cache_policy != "official":
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
            "spatial_tile_halo": args.spatial_tile_halo,
            "cache_release_in_timed_boundary": args.mlx_cache_policy != "official",
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
