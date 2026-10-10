#!/usr/bin/env python3
"""Benchmark a captured Qwen-Image transformer block on MPS and retained MLX."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from safetensors import safe_open

try:
    import mlx.core as mx
except ImportError:
    mx = None


HEADS = 24
HEAD_DIM = 128
NORM_EPS = 1e-6


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


def error_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    candidate64 = candidate.astype(np.float64, copy=False)
    reference64 = reference.astype(np.float64, copy=False)
    difference = candidate64 - reference64
    denominator = np.linalg.norm(reference64.reshape(-1))
    cosine_denominator = np.linalg.norm(candidate64.reshape(-1)) * denominator
    return {
        "exact": bool(np.array_equal(candidate, reference)),
        "max_abs": float(np.max(np.abs(difference))),
        "mean_abs": float(np.mean(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference.reshape(-1)) / denominator),
        "cosine": float(np.dot(candidate64.reshape(-1), reference64.reshape(-1)) / cosine_denominator),
    }


def mlx_from_torch(tensor: torch.Tensor):
    return mx.from_dlpack(torch.utils.dlpack.to_dlpack(tensor.contiguous()))


def mlx_linear(x, weights, prefix, quantized):
    if quantized:
        quantized_weight, scales, biases = weights[f"{prefix}.quantized"]
        output = mx.quantized_matmul(
            x, quantized_weight, scales=scales, biases=biases, transpose=True,
            group_size=32, bits=8,
        )
    else:
        output = x @ weights[f"{prefix}.weight"].T
    return output + weights[f"{prefix}.bias"]


def mlx_rope(x, frequencies):
    original_dtype = x.dtype
    paired = x.astype(frequencies.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    output = frequencies[..., 0] * paired[..., 0] + frequencies[..., 1] * paired[..., 1]
    return output.reshape(x.shape).astype(original_dtype)


def mlx_modulate(x, params):
    shift, scale, gate = mx.split(params, 3, axis=-1)
    return shift[:, None, :] + x * (1 + scale[:, None, :]), gate[:, None, :]


def mlx_gelu_tanh(x):
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def mlx_mlp(x, weights, prefix, quantized):
    hidden = mlx_gelu_tanh(mlx_linear(x, weights, f"{prefix}.net.0.proj", quantized))
    return mlx_linear(hidden, weights, f"{prefix}.net.2", quantized)


def mlx_block(inputs, weights, quantized=False):
    image = inputs["hidden_states"]
    text = inputs["encoder_hidden_states"]
    temb = inputs["temb"]
    frequencies = inputs["image_rotary_emb"]

    silu_temb = temb * mx.sigmoid(temb)
    image_params = mlx_linear(silu_temb, weights, "img_mod.1", quantized)
    text_params = mlx_linear(silu_temb, weights, "txt_mod.1", quantized)
    image_mod1, image_mod2 = mx.split(image_params, 2, axis=-1)
    text_mod1, text_mod2 = mx.split(text_params, 2, axis=-1)
    image_attn_input, image_gate1 = mlx_modulate(
        mx.fast.layer_norm(image, None, None, NORM_EPS), image_mod1,
    )
    text_attn_input, text_gate1 = mlx_modulate(
        mx.fast.layer_norm(text, None, None, NORM_EPS), text_mod1,
    )

    def projection(source, prefix, norm_prefix):
        value = mlx_linear(source, weights, prefix, quantized)
        value = value.reshape(value.shape[0], value.shape[1], HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        if norm_prefix is not None:
            value = mx.fast.rms_norm(value, weights[f"{norm_prefix}.weight"], NORM_EPS)
        return value

    image_q = projection(image_attn_input, "attn.to_q", "attn.norm_q")
    image_k = projection(image_attn_input, "attn.to_k", "attn.norm_k")
    image_v = projection(image_attn_input, "attn.to_v", None)
    text_q = projection(text_attn_input, "attn.add_q_proj", "attn.norm_added_q")
    text_k = projection(text_attn_input, "attn.add_k_proj", "attn.norm_added_k")
    text_v = projection(text_attn_input, "attn.add_v_proj", None)
    query = mlx_rope(mx.concatenate((text_q, image_q), axis=2), frequencies)
    key = mlx_rope(mx.concatenate((text_k, image_k), axis=2), frequencies)
    value = mx.concatenate((text_v, image_v), axis=2)
    attended = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=HEAD_DIM ** -0.5,
    )
    attended = attended.transpose(0, 2, 1, 3).reshape(image.shape[0], text.shape[1] + image.shape[1], -1)
    text_attn = mlx_linear(attended[:, :text.shape[1]], weights, "attn.to_add_out", quantized)
    image_attn = mlx_linear(attended[:, text.shape[1]:], weights, "attn.to_out.0", quantized)
    image = image + image_gate1 * image_attn
    text = text + text_gate1 * text_attn
    image_mlp_input, image_gate2 = mlx_modulate(
        mx.fast.layer_norm(image, None, None, NORM_EPS), image_mod2,
    )
    text_mlp_input, text_gate2 = mlx_modulate(
        mx.fast.layer_norm(text, None, None, NORM_EPS), text_mod2,
    )
    image = image + image_gate2 * mlx_mlp(image_mlp_input, weights, "img_mlp", quantized)
    text = text + text_gate2 * mlx_mlp(text_mlp_input, weights, "txt_mlp", quantized)
    return text, image


def mlx_block_chain(inputs, block_weights, quantized=False):
    text = inputs["encoder_hidden_states"]
    image = inputs["hidden_states"]
    for weights in block_weights:
        text, image = mlx_block(
            {
                "hidden_states": image,
                "encoder_hidden_states": text,
                "temb": inputs["temb"],
                "image_rotary_emb": inputs["image_rotary_emb"],
            },
            weights,
            quantized=quantized,
        )
    return text, image


def time_mlx(fn, warmups, iterations):
    output = None
    for _ in range(warmups):
        output = fn()
        mx.eval(*output)
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        output = fn()
        mx.eval(*output)
        timings.append((time.perf_counter() - started) * 1000)
    return timings, output


def output_errors(output, reference):
    return {
        name: error_metrics(candidate, reference[name])
        for name, candidate in {
            "encoder_hidden_states": output[0],
            "hidden_states": output[1],
        }.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--torch-baseline", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("iterations must be positive and warmups non-negative")
    if mx is None or not torch.backends.mps.is_available():
        raise SystemExit("MLX and PyTorch MPS are required")

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    schema = payload["metadata"]["schema_version"]
    if schema == "kindred.comfyui.qwen_block_capture.v1":
        captured_blocks = [payload]
    elif schema == "kindred.comfyui.qwen_block_range_capture.v1":
        captured_blocks = [payload["blocks"][key] for key in sorted(payload["blocks"], key=int)]
    elif schema == "kindred.comfyui.qwen_block_boundary_capture.v1":
        if args.model is None or not args.model_sha256:
            raise SystemExit("--model and --model-sha256 are required for a boundary capture")
        captured_blocks = [payload]
    else:
        raise SystemExit("capture is not a Qwen transformer block or range")
    reference = {
        name: value.float().numpy()
        for name, value in (payload if schema.endswith("boundary_capture.v1") else captured_blocks[-1])["outputs"].items()
    }
    torch_report = json.loads(args.torch_baseline.read_text(encoding="utf-8"))
    if torch_report.get("schema_version") != "kindred.comfyui.qwen_block_torch_benchmark.v1":
        raise SystemExit("PyTorch baseline report has the wrong schema")
    capture_sha = file_sha256(args.capture)
    if torch_report["capture"]["sha256"] != capture_sha:
        raise SystemExit("PyTorch baseline and MLX benchmark use different captures")
    torch_result = torch_report["pytorch_mps_bf16"]
    torch_median = torch_result["median_ms"]

    mlx_inputs = {
        name: mlx_from_torch(value)
        for name, value in (payload if schema.endswith("boundary_capture.v1") else captured_blocks[0])["inputs"].items()
        if isinstance(value, torch.Tensor)
    }
    if schema.endswith("boundary_capture.v1"):
        model_file = safe_open(args.model, framework="pt", device="cpu")
        mlx_weights = []
        for offset in range(payload["metadata"]["block_count"]):
            index = payload["metadata"]["block"] + offset
            prefix = f"transformer_blocks.{index}."
            mlx_weights.append({
                key.removeprefix(prefix): mlx_from_torch(model_file.get_tensor(key))
                for key in model_file.keys()
                if key.startswith(prefix)
            })
    else:
        mlx_weights = [
            {name: mlx_from_torch(value) for name, value in captured["state_dict"].items()}
            for captured in captured_blocks
        ]
    mx.eval(*mlx_inputs.values(), *(value for weights in mlx_weights for value in weights.values()))
    mlx_timings, mlx_output = time_mlx(
        lambda: mlx_block_chain(mlx_inputs, mlx_weights), args.warmups, args.iterations,
    )
    mlx_arrays = tuple(np.array(value.astype(mx.float32)) for value in mlx_output)

    quantized_weights = None
    int8_timings = None
    int8_arrays = None
    if not args.skip_int8:
        quantized_weights = []
        for block in mlx_weights:
            quantized_block = dict(block)
            for name, value in list(block.items()):
                if name.endswith(".weight") and value.ndim == 2:
                    prefix = name.removesuffix(".weight")
                    quantized_block[f"{prefix}.quantized"] = mx.quantize(
                        value, group_size=32, bits=8, mode="affine",
                    )
            quantized_weights.append(quantized_block)
        quantized_values = [
            item
            for block in quantized_weights
            for value in block.values()
            for item in (value if isinstance(value, tuple) else (value,))
        ]
        mx.eval(*quantized_values)
        int8_timings, int8_output = time_mlx(
            lambda: mlx_block_chain(mlx_inputs, quantized_weights, quantized=True), args.warmups, args.iterations,
        )
        int8_arrays = tuple(np.array(value.astype(mx.float32)) for value in int8_output)

    def compile_block(weights, quantized):
        def execute(image, text, temb, frequencies):
            return mlx_block_chain(
                {
                    "hidden_states": image,
                    "encoder_hidden_states": text,
                    "temb": temb,
                    "image_rotary_emb": frequencies,
                },
                weights,
                quantized=quantized,
            )
        return mx.compile(execute)

    compiled_bf16 = compile_block(mlx_weights, False)
    compiled_bf16_timings, compiled_bf16_output = time_mlx(
        lambda: compiled_bf16(
            mlx_inputs["hidden_states"], mlx_inputs["encoder_hidden_states"],
            mlx_inputs["temb"], mlx_inputs["image_rotary_emb"],
        ),
        args.warmups,
        args.iterations,
    )
    compiled_bf16_arrays = tuple(np.array(value.astype(mx.float32)) for value in compiled_bf16_output)

    compiled_int8_timings = None
    compiled_int8_arrays = None
    if quantized_weights is not None:
        compiled_int8 = compile_block(quantized_weights, True)
        compiled_int8_timings, compiled_int8_output = time_mlx(
            lambda: compiled_int8(
                mlx_inputs["hidden_states"], mlx_inputs["encoder_hidden_states"],
                mlx_inputs["temb"], mlx_inputs["image_rotary_emb"],
            ),
            args.warmups,
            args.iterations,
        )
        compiled_int8_arrays = tuple(np.array(value.astype(mx.float32)) for value in compiled_int8_output)

    report = {
        "schema_version": "kindred.comfyui.qwen_block_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {
            "path": str(args.capture),
            "sha256": capture_sha,
            "metadata": payload["metadata"],
        },
        "runtime": {
            "python": __import__("sys").version.split()[0],
            "torch": torch.__version__,
            "mlx": importlib.metadata.version("mlx"),
            "mps_available": torch.backends.mps.is_available(),
        },
        "method": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "retained_weights": True,
            "framework_bridge_excluded": True,
            "pytorch_baseline_report": str(args.torch_baseline),
            "mlx_attention": "fast.scaled_dot_product_attention",
            "int8": "affine group_size=32 for every 2D linear weight",
            "block_count": len(mlx_weights),
            "model": {
                "path": str(args.model),
                "declared_sha256": args.model_sha256,
                "size_bytes": args.model.stat().st_size,
            } if args.model else None,
        },
        "pytorch_mps_bf16": torch_result,
        "mlx_bf16": {
            **summarize(mlx_timings),
            "speedup_vs_pytorch": round(torch_median / statistics.median(mlx_timings), 4),
            "error": output_errors(mlx_arrays, reference),
        },
        "mlx_int8_group32": None if int8_timings is None else {
            **summarize(int8_timings),
            "speedup_vs_pytorch": round(torch_median / statistics.median(int8_timings), 4),
            "error": output_errors(int8_arrays, reference),
        },
        "mlx_compiled_bf16": {
            **summarize(compiled_bf16_timings),
            "speedup_vs_pytorch": round(torch_median / statistics.median(compiled_bf16_timings), 4),
            "error": output_errors(compiled_bf16_arrays, reference),
        },
        "mlx_compiled_int8_group32": None if compiled_int8_timings is None else {
            **summarize(compiled_int8_timings),
            "speedup_vs_pytorch": round(torch_median / statistics.median(compiled_int8_timings), 4),
            "error": output_errors(compiled_int8_arrays, reference),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
