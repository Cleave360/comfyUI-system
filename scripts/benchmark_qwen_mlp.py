#!/usr/bin/env python3
"""Benchmark a captured Qwen-Image MLP on PyTorch MPS and Apple MLX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.nn.functional as F

try:
    import mlx.core as mx
except ImportError:
    mx = None


ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "median_ms": round(statistics.median(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
    }


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate64 = candidate.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    difference = candidate64 - reference64
    denominator = np.linalg.norm(reference64.reshape(-1))
    cosine_denominator = np.linalg.norm(candidate64.reshape(-1)) * denominator
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
        "cosine": float(np.dot(candidate64.reshape(-1), reference64.reshape(-1)) / cosine_denominator),
    }


def torch_mlp(x, expand_weight, expand_bias, contract_weight, contract_bias):
    return F.linear(
        F.gelu(F.linear(x, expand_weight, expand_bias), approximate="tanh"),
        contract_weight,
        contract_bias,
    )


def mlx_gelu_tanh(x):
    coefficient = 0.7978845608028654
    return 0.5 * x * (1.0 + mx.tanh(coefficient * (x + 0.044715 * x * x * x)))


def mlx_mlp_bf16(x, weights):
    hidden = x @ weights["expand_weight"].T + weights["expand_bias"]
    return mlx_gelu_tanh(hidden) @ weights["contract_weight"].T + weights["contract_bias"]


def mlx_mlp_int8(x, weights):
    hidden = mx.quantized_matmul(
        x,
        weights["expand_weight_q"],
        scales=weights["expand_scales"],
        biases=weights["expand_biases"],
        transpose=True,
        group_size=32,
        bits=8,
    ) + weights["expand_bias"]
    return mx.quantized_matmul(
        mlx_gelu_tanh(hidden),
        weights["contract_weight_q"],
        scales=weights["contract_scales"],
        biases=weights["contract_biases"],
        transpose=True,
        group_size=32,
        bits=8,
    ) + weights["contract_bias"]


def time_torch(fn, warmups: int, iterations: int) -> tuple[list[float], torch.Tensor]:
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


def time_mlx(fn, warmups: int, iterations: int) -> tuple[list[float], object]:
    output = None
    for _ in range(warmups):
        output = fn()
        if isinstance(output, tuple):
            mx.eval(*output)
        else:
            mx.eval(output)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = fn()
        if isinstance(output, tuple):
            mx.eval(*output)
        else:
            mx.eval(output)
        timings.append((time.perf_counter() - started) * 1000)
    return timings, output


def mlx_from_torch_cpu(tensor: torch.Tensor):
    return mx.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def retained_mlx_weights(tensors: dict[str, torch.Tensor], stream: str) -> dict[str, object]:
    result = {}
    for name in ("expand_weight", "expand_bias", "contract_weight", "contract_bias"):
        result[name] = mlx_from_torch_cpu(tensors[f"{stream}.{name}"])
    mx.eval(*result.values())
    return result


def quantize_mlx_weights(weights: dict[str, object]) -> dict[str, object]:
    result = dict(weights)
    for name in ("expand", "contract"):
        quantized, scales, biases = mx.quantize(
            weights[f"{name}_weight"], group_size=32, bits=8, mode="affine",
        )
        result[f"{name}_weight_q"] = quantized
        result[f"{name}_scales"] = scales
        result[f"{name}_biases"] = biases
    mx.eval(*result.values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("iterations must be positive and warmups non-negative")
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")
    if mx is None:
        raise SystemExit("MLX is required; use the isolated qwen-mlx environment")

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    tensors = payload["tensors"]
    reference = {
        stream: tensors[f"{stream}.output"].float().numpy()
        for stream in ("image_mlp", "text_mlp")
    }

    torch_data = {}
    torch_outputs = {}
    for stream in ("image_mlp", "text_mlp"):
        x = tensors[f"{stream}.input"].to("mps")
        ew = tensors[f"{stream}.expand_weight"].to("mps")
        eb = tensors[f"{stream}.expand_bias"].to("mps")
        cw = tensors[f"{stream}.contract_weight"].to("mps")
        cb = tensors[f"{stream}.contract_bias"].to("mps")
        timings, output = time_torch(
            lambda: torch_mlp(x, ew, eb, cw, cb), args.warmups, args.iterations,
        )
        candidate = output.float().cpu().numpy()
        torch_data[stream] = {**summarize(timings), "error": error_metrics(candidate, reference[stream])}
        torch_outputs[stream] = output
        del x, ew, eb, cw, cb

    mlx_inputs = {
        stream: mlx_from_torch_cpu(tensors[f"{stream}.input"])
        for stream in ("image_mlp", "text_mlp")
    }
    mlx_weights = {
        stream: retained_mlx_weights(tensors, stream)
        for stream in ("image_mlp", "text_mlp")
    }
    mlx_bf16 = {}
    for stream in ("image_mlp", "text_mlp"):
        timings, output = time_mlx(
            lambda stream=stream: mlx_mlp_bf16(mlx_inputs[stream], mlx_weights[stream]),
            args.warmups,
            args.iterations,
        )
        candidate = np.array(output.astype(mx.float32))
        mlx_bf16[stream] = {**summarize(timings), "error": error_metrics(candidate, reference[stream])}

    quantized_weights = {
        stream: quantize_mlx_weights(mlx_weights[stream])
        for stream in ("image_mlp", "text_mlp")
    }
    mlx_int8 = {}
    for stream in ("image_mlp", "text_mlp"):
        timings, output = time_mlx(
            lambda stream=stream: mlx_mlp_int8(mlx_inputs[stream], quantized_weights[stream]),
            args.warmups,
            args.iterations,
        )
        candidate = np.array(output.astype(mx.float32))
        mlx_int8[stream] = {**summarize(timings), "error": error_metrics(candidate, reference[stream])}

    sequential_timings, sequential_outputs = time_mlx(
        lambda: (
            mlx_mlp_bf16(mlx_inputs["image_mlp"], mlx_weights["image_mlp"]),
            mlx_mlp_bf16(mlx_inputs["text_mlp"], mlx_weights["text_mlp"]),
        ),
        args.warmups,
        args.iterations,
    )
    image_stream = mx.new_stream(mx.gpu)
    text_stream = mx.new_stream(mx.gpu)

    def concurrent_branches():
        with mx.stream(image_stream):
            image_output = mlx_mlp_bf16(mlx_inputs["image_mlp"], mlx_weights["image_mlp"])
        with mx.stream(text_stream):
            text_output = mlx_mlp_bf16(mlx_inputs["text_mlp"], mlx_weights["text_mlp"])
        return image_output, text_output

    concurrent_timings, concurrent_outputs = time_mlx(
        concurrent_branches, args.warmups, args.iterations,
    )
    concurrency = {
        "sequential": summarize(sequential_timings),
        "concurrent": summarize(concurrent_timings),
        "speedup": round(statistics.median(sequential_timings) / statistics.median(concurrent_timings), 4),
        "image_error_vs_sequential": error_metrics(
            np.array(concurrent_outputs[0].astype(mx.float32)),
            np.array(sequential_outputs[0].astype(mx.float32)),
        ),
        "text_error_vs_sequential": error_metrics(
            np.array(concurrent_outputs[1].astype(mx.float32)),
            np.array(sequential_outputs[1].astype(mx.float32)),
        ),
    }

    bridge = {"zero_copy_supported": False}
    try:
        torch.utils.dlpack.to_dlpack(torch_outputs["text_mlp"].float().contiguous())
        bridge["zero_copy_supported"] = True
    except RuntimeError as exc:
        bridge["torch_mps_to_mlx_error"] = str(exc)
    try:
        probe = mx.arange(4, dtype=mx.float32)
        mx.eval(probe)
        torch.utils.dlpack.from_dlpack(probe)
    except RuntimeError as exc:
        bridge["mlx_to_torch_mps_error"] = str(exc)

    x_mps = tensors["image_mlp.input"].to("mps")

    def copied_bridge():
        torch.mps.synchronize()
        cpu_input = x_mps.float().cpu().to(torch.bfloat16).contiguous()
        mlx_input = mlx_from_torch_cpu(cpu_input)
        mlx_output = mlx_mlp_bf16(mlx_input, mlx_weights["image_mlp"])
        cpu_output = np.array(mlx_output.astype(mx.float32))
        mps_output = torch.from_numpy(cpu_output).to("mps", dtype=torch.bfloat16)
        torch.mps.synchronize()
        return mps_output

    copied_timings = []
    copied_output = None
    for _ in range(max(1, min(3, args.iterations))):
        started = time.perf_counter()
        copied_output = copied_bridge()
        copied_timings.append((time.perf_counter() - started) * 1000)
    bridge["copied_image_mlp"] = {
        **summarize(copied_timings),
        "error": error_metrics(copied_output.float().cpu().numpy(), reference["image_mlp"]),
    }

    report = {
        "schema_version": "kindred.comfyui.qwen_mlp_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "path": str(args.capture),
            "sha256": file_sha256(args.capture),
            "metadata": payload["metadata"],
        },
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "mlx": "0.32.2",
            "mps_available": torch.backends.mps.is_available(),
        },
        "method": {"warmups": args.warmups, "iterations": args.iterations},
        "pytorch_mps_bf16": torch_data,
        "mlx_bf16": mlx_bf16,
        "mlx_int8_group32": mlx_int8,
        "mlx_branch_concurrency": concurrency,
        "bridge": bridge,
    }
    destination = args.output or ROOT / "reports" / "benchmarks" / "qwen_mlp_block30.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
