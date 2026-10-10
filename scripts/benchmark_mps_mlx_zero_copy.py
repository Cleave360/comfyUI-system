#!/usr/bin/env python3
"""Verify Torch MPS/MLX zero-copy aliasing and benchmark a real Qwen MLP bridge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
import torch

from benchmark_qwen_mlp import (
    error_metrics,
    file_sha256,
    mlx_mlp_bf16,
    retained_mlx_weights,
    summarize,
    torch_mlp,
)


ROOT = Path(__file__).resolve().parents[1]
DTYPES = {
    "float32": (torch.float32, mx.float32),
    "float16": (torch.float16, mx.float16),
    "bfloat16": (torch.bfloat16, mx.bfloat16),
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * fraction) - 1))
    return ordered[index]


def microseconds(values: list[float]) -> dict[str, float]:
    return {
        "median_us": round(statistics.median(values), 3),
        "min_us": round(min(values), 3),
        "max_us": round(max(values), 3),
        "p95_us": round(percentile(values, 0.95), 3),
    }


def torch_to_mlx_alias(dtype_name: str) -> dict[str, object]:
    torch_dtype, _ = DTYPES[dtype_name]
    source = torch.arange(8, device="mps", dtype=torch_dtype)
    torch.mps.synchronize()
    imported = mx.asarray(source, copy=False)
    mx.eval(imported)
    source.add_(10)
    torch.mps.synchronize()
    observed = np.asarray(imported.astype(mx.float32)).tolist()
    expected = [float(value) for value in range(10, 18)]
    return {
        "copy_false_succeeded": True,
        "mutation_visible": observed == expected,
        "observed": observed,
    }


def mlx_to_torch_alias(dtype_name: str) -> dict[str, object]:
    _, mlx_dtype = DTYPES[dtype_name]
    source = mx.arange(8, dtype=mlx_dtype)
    mx.eval(source)
    imported = torch.as_tensor(source)
    imported.add_(20)
    torch.mps.synchronize()
    observed = np.asarray(source.astype(mx.float32)).tolist()
    expected = [float(value) for value in range(20, 28)]
    return {
        "device": str(imported.device),
        "mps_device": imported.device.type == "mps",
        "mutation_visible": observed == expected,
        "observed": observed,
    }


def strided_alias_probe() -> dict[str, object]:
    base = torch.arange(64, device="mps", dtype=torch.float32).reshape(8, 8)
    source = base[:, ::2]
    torch.mps.synchronize()
    try:
        imported = mx.asarray(source, copy=False)
        mx.eval(imported)
        base.add_(100)
        torch.mps.synchronize()
        observed = np.asarray(imported)
        expected = np.arange(64, dtype=np.float32).reshape(8, 8)[:, ::2] + 100
        return {
            "copy_false_succeeded": True,
            "mutation_visible": bool(np.array_equal(observed, expected)),
            "shape": list(imported.shape),
            "torch_stride": list(source.stride()),
        }
    except Exception as exc:  # A negative stride/layout result is useful evidence.
        return {
            "copy_false_succeeded": False,
            "mutation_visible": False,
            "error": f"{type(exc).__name__}: {exc}",
            "torch_stride": list(source.stride()),
        }


def handoff_timings(shape: tuple[int, ...], iterations: int) -> dict[str, object]:
    torch_source = torch.zeros(shape, device="mps", dtype=torch.bfloat16)
    torch.mps.synchronize()
    mlx_source = mx.zeros(shape, dtype=mx.bfloat16)
    mx.eval(mlx_source)

    torch_to_mlx = []
    mlx_to_torch = []
    round_trip = []
    for _ in range(iterations):
        torch.mps.synchronize()
        started = time.perf_counter_ns()
        imported_mlx = mx.asarray(torch_source, copy=False)
        mx.eval(imported_mlx)
        torch_to_mlx.append((time.perf_counter_ns() - started) / 1000)

        mx.eval(mlx_source)
        started = time.perf_counter_ns()
        imported_torch = torch.as_tensor(mlx_source)
        torch.mps.synchronize()
        mlx_to_torch.append((time.perf_counter_ns() - started) / 1000)

        torch.mps.synchronize()
        started = time.perf_counter_ns()
        bridged_mlx = mx.asarray(torch_source, copy=False)
        mx.eval(bridged_mlx)
        bridged_torch = torch.as_tensor(bridged_mlx)
        torch.mps.synchronize()
        round_trip.append((time.perf_counter_ns() - started) / 1000)

    bridged_torch.add_(1)
    torch.mps.synchronize()
    alias_value = float(torch_source.reshape(-1)[0].item())
    return {
        "shape": list(shape),
        "logical_mib": round(torch_source.numel() * torch_source.element_size() / 1024**2, 3),
        "torch_to_mlx": microseconds(torch_to_mlx),
        "mlx_to_torch": microseconds(mlx_to_torch),
        "round_trip": microseconds(round_trip),
        "round_trip_mutation_visible": alias_value == 1.0,
    }


def time_torch_mps(fn, warmups: int, iterations: int):
    output = None
    for _ in range(warmups):
        output = fn()
        torch.mps.synchronize()
    timings = []
    for _ in range(iterations):
        torch.mps.synchronize()
        started = time.perf_counter()
        output = fn()
        torch.mps.synchronize()
        timings.append((time.perf_counter() - started) * 1000)
    return timings, output


def real_qwen_bridge(capture: Path, warmups: int, iterations: int) -> dict[str, object]:
    payload = torch.load(capture, map_location="cpu", weights_only=False)
    tensors = payload["tensors"]
    reference = tensors["image_mlp.output"].float().numpy()
    x_mps = tensors["image_mlp.input"].to("mps")
    expand_weight = tensors["image_mlp.expand_weight"].to("mps")
    expand_bias = tensors["image_mlp.expand_bias"].to("mps")
    contract_weight = tensors["image_mlp.contract_weight"].to("mps")
    contract_bias = tensors["image_mlp.contract_bias"].to("mps")

    native_timings, native_output = time_torch_mps(
        lambda: torch_mlp(
            x_mps, expand_weight, expand_bias, contract_weight, contract_bias,
        ),
        warmups,
        iterations,
    )

    mlx_weights = retained_mlx_weights(tensors, "image_mlp")

    def zero_copy_bridge():
        mlx_input = mx.asarray(x_mps, copy=False)
        mlx_output = mlx_mlp_bf16(mlx_input, mlx_weights)
        mx.eval(mlx_output)
        torch_output = torch.as_tensor(mlx_output)
        torch.mps.synchronize()
        return torch_output

    bridge_timings, bridge_output = time_torch_mps(
        zero_copy_bridge, warmups, iterations,
    )
    native_candidate = native_output.float().cpu().numpy()
    bridge_candidate = bridge_output.float().cpu().numpy()
    native_median = statistics.median(native_timings)
    bridge_median = statistics.median(bridge_timings)
    return {
        "capture": {
            "path": str(capture),
            "sha256": file_sha256(capture),
            "metadata": payload["metadata"],
        },
        "shape": list(x_mps.shape),
        "native_torch_mps_bf16": {
            **summarize(native_timings),
            "error": error_metrics(native_candidate, reference),
        },
        "zero_copy_mlx_bf16_round_trip": {
            **summarize(bridge_timings),
            "error": error_metrics(bridge_candidate, reference),
            "torch_output_device": str(bridge_output.device),
        },
        "speedup_vs_native": round(native_median / bridge_median, 4),
        "milliseconds_saved": round(native_median - bridge_median, 3),
    }


def atomic_write_json(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        default=ROOT / "reports" / "captures" / "qwen_mlp_block30.pt",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=9)
    parser.add_argument("--handoff-iterations", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "benchmarks" / "mps_mlx_zero_copy_torch212.json",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1 or args.handoff_iterations < 1:
        raise SystemExit("warmups must be non-negative and iterations must be positive")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")
    if not args.capture.is_file():
        raise SystemExit(f"capture not found: {args.capture}")

    aliases = {}
    for dtype_name in DTYPES:
        aliases[dtype_name] = {
            "torch_to_mlx": torch_to_mlx_alias(dtype_name),
            "mlx_to_torch": mlx_to_torch_alias(dtype_name),
        }
    aliases["strided_float32"] = strided_alias_probe()

    alias_gate = all(
        result["torch_to_mlx"]["copy_false_succeeded"]
        and result["torch_to_mlx"]["mutation_visible"]
        and result["mlx_to_torch"]["mps_device"]
        and result["mlx_to_torch"]["mutation_visible"]
        for name, result in aliases.items()
        if name in DTYPES
    )

    handoffs = {
        "small_control": handoff_timings((8,), args.handoff_iterations),
        "qwen_image_mlp_input": handoff_timings(
            (1, 2304, 3072), args.handoff_iterations,
        ),
    }
    qwen = real_qwen_bridge(args.capture, args.warmups, args.iterations)

    # Declared before the governed run: mechanical aliasing must pass for all
    # production-relevant dtypes. Promotion additionally requires at least a
    # 2% complete-bridge speedup and no worse than the established MLX BF16
    # image-MLP relative-L2 boundary.
    gates = {
        "bidirectional_alias_all_required_dtypes": alias_gate,
        "real_shape_round_trip_alias": handoffs["qwen_image_mlp_input"]["round_trip_mutation_visible"],
        "bridge_speedup_at_least_1_02x": qwen["speedup_vs_native"] >= 1.02,
        "bridge_relative_l2_at_most_0_0025": (
            qwen["zero_copy_mlx_bf16_round_trip"]["error"]["relative_l2"] <= 0.0025
        ),
    }
    gates["overall_promotion"] = all(gates.values())

    report = {
        "schema_version": "kindred.comfyui.mps_mlx_zero_copy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mps_available": torch.backends.mps.is_available(),
            "mlx_device": str(mx.default_device()),
        },
        "method": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "handoff_iterations": args.handoff_iterations,
            "torch_mps_synchronized": True,
            "mlx_outputs_evaluated": True,
            "production_environment_modified": False,
        },
        "aliases": aliases,
        "handoffs": handoffs,
        "qwen_image_mlp": qwen,
        "gates": gates,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"report: {args.output}")
    return 0 if alias_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
