#!/usr/bin/env python3
"""Benchmark Apple's MLX Wan VAE with zero-copy Torch MPS boundaries."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

import mlx.core as mx
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
MLX_EXAMPLES = ROOT / "reports" / "vendor" / "mlx-examples"
WAN_MODULES = MLX_EXAMPLES / "video" / "wan2.1" / "wan"
MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
       3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))
    return {
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def atomic_write_json(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_official_wan_vae_class():
    """Import only Apple's VAE files, without unrelated pipeline dependencies."""
    package = types.ModuleType("kindred_mlx_wan_reference")
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


def reference_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(MLX_EXAMPLES), "rev-parse", "HEAD"], text=True,
    ).strip()


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(delta)))
    return {
        "mae_float01": float(np.mean(np.abs(delta))),
        "mae_rgb255": float(np.mean(np.abs(delta))) * 255.0,
        "rmse_float01": math.sqrt(mse),
        "max_abs_float01": float(np.max(np.abs(delta))),
        "psnr_db": float("inf") if mse == 0 else 10.0 * math.log10(1.0 / mse),
        "changed_fraction": float(np.mean(delta != 0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", type=Path,
        default=ROOT / "reports" / "captures" / "qwen_vae_1024x576.pt",
    )
    parser.add_argument(
        "--weights", type=Path,
        default=SOURCE / "models" / "vae" / "qwen_image_vae.safetensors",
    )
    parser.add_argument(
        "--torch-report", type=Path,
        default=ROOT / "reports" / "benchmarks" / "qwen_vae_torch251.json",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "benchmarks" / "qwen_vae_mlx_zero_copy.json",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1:
        raise SystemExit("warmups must be non-negative and iterations must be positive")
    required = (args.capture, args.weights, args.torch_report, WAN_MODULES / "vae.py")
    if not all(path.is_file() for path in required):
        raise SystemExit("capture, weights, Torch report, and pinned MLX reference are required")
    if not torch.backends.mps.is_available():
        raise SystemExit("Torch MPS is required for the zero-copy boundary test")

    captured = torch.load(args.capture, map_location="cpu", weights_only=False)
    native = json.loads(args.torch_report.read_text(encoding="utf-8"))
    samples = captured["samples"].float()
    mean = torch.tensor(MEAN).view(1, 16, 1, 1, 1)
    std = torch.tensor(STD).view(1, 16, 1, 1, 1)
    normalized = ((samples - mean) / std).permute(0, 2, 3, 4, 1)[0].contiguous()
    source_cpu = normalized

    WanVAE = load_official_wan_vae_class()
    vae = WanVAE()
    weights = WanVAE.sanitize(mx.load(str(args.weights)))
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())

    def decode_bridge():
        source_mps = source_cpu.to("mps")
        latent_mlx = mx.asarray(source_mps, copy=False)
        decoded = vae.decode(latent_mlx)
        pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
        mx.eval(pixels)
        pixels_torch = torch.as_tensor(pixels)
        torch.mps.synchronize()
        return pixels_torch.cpu()

    mx.reset_peak_memory()
    started = time.perf_counter()
    output = decode_bridge()
    cold_ms = (time.perf_counter() - started) * 1000
    cold_peak = mx.get_peak_memory()
    for _ in range(args.warmups):
        output = decode_bridge()

    mx.reset_peak_memory()
    timings = []
    for _ in range(args.iterations):
        torch.mps.synchronize()
        started = time.perf_counter()
        output = decode_bridge()
        timings.append((time.perf_counter() - started) * 1000)
    warm_peak = mx.get_peak_memory()

    candidate = output[0].float().cpu().numpy()
    reference = captured["output"][0, 0].float().numpy()
    metrics = error_metrics(candidate, reference)
    finite = bool(np.isfinite(candidate).all())
    native_median = float(native["warm_decode"]["median_ms"])
    mlx_summary = summarize(timings)
    speedup = native_median / float(mlx_summary["median_ms"])

    # Declared before the governed comparison: the candidate must preserve
    # shape and finite pixels, stay within one RGB level mean error and 40 dB
    # PSNR, beat the warm native decoder by 5%, and remain below 12 GiB peak.
    gates = {
        "output_shape_exact": list(output.shape) == list(captured["output"].shape[1:]),
        "all_pixels_finite": finite,
        "mae_rgb255_at_most_1": metrics["mae_rgb255"] <= 1.0,
        "psnr_at_least_40_db": metrics["psnr_db"] >= 40.0,
        "speedup_at_least_1_05x": speedup >= 1.05,
        "warm_peak_memory_at_most_12_gib": warm_peak <= 12 * 1024**3,
    }
    gates["overall_trial_gate"] = all(gates.values())

    report = {
        "schema_version": "kindred.comfyui.qwen_vae_mlx_zero_copy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mlx_device": str(mx.default_device()),
            "mlx_enable_tf32": False,
        },
        "reference": {
            "mlx_examples_commit": reference_commit(),
            "module": "video/wan2.1/wan/vae.py",
            "weights_loaded": len(weights),
            "strict_false_reason": "official module stores mean/std as arrays absent from checkpoint",
        },
        "artifacts": {
            "capture": str(args.capture),
            "capture_sha256": file_sha256(args.capture),
            "weights": str(args.weights),
            "weights_sha256": file_sha256(args.weights),
            "torch_report": str(args.torch_report),
            "torch_report_sha256": file_sha256(args.torch_report),
        },
        "method": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "input_normalization": "Wan21 process_in inverse of captured process_out",
            "torch_to_mlx": "mx.asarray(torch_mps_tensor, copy=False)",
            "mlx_to_torch": "torch.as_tensor(mlx_array)",
            "complete_round_trip_timed": True,
            "cpu_input_and_cpu_output_match_comfy_boundary": True,
            "model_loading_and_compilation_excluded_from_warm_iterations": True,
            "production_environment_modified": False,
        },
        "input_shape": list(source_cpu.shape),
        "output_shape": list(output.shape),
        "output_torch_device": str(output.device),
        "cold_decode_round_trip_ms": round(cold_ms, 3),
        "cold_peak_memory_bytes": cold_peak,
        "warm_decode_round_trip": mlx_summary,
        "warm_peak_memory_bytes": warm_peak,
        "native_torch_mps_median_ms": native_median,
        "speedup_vs_native": round(speedup, 4),
        "pixel_error": metrics,
        "gates": gates,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
