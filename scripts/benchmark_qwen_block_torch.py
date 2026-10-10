#!/usr/bin/env python3
"""Benchmark a captured Qwen block with ComfyUI's actual PyTorch MPS class."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ComfyUI-source"))

from comfy.ldm.qwen_image.model import QwenImageTransformerBlock  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise SystemExit("iterations must be positive and warmups non-negative")
    if not torch.backends.mps.is_available():
        raise SystemExit("PyTorch MPS is required")

    payload = torch.load(args.capture, map_location="cpu", weights_only=False)
    schema = payload["metadata"]["schema_version"]
    if schema == "kindred.comfyui.qwen_block_capture.v1":
        captured_blocks = [payload]
    elif schema == "kindred.comfyui.qwen_block_range_capture.v1":
        captured_blocks = [payload["blocks"][key] for key in sorted(payload["blocks"], key=int)]
    elif schema == "kindred.comfyui.qwen_block_boundary_capture.v1":
        if args.model is None:
            raise SystemExit("--model is required for a boundary capture")
        captured_blocks = [payload]
    else:
        raise SystemExit("capture is not a Qwen transformer block or range")
    blocks = []
    model_file = safe_open(args.model, framework="pt", device="cpu") if schema.endswith("boundary_capture.v1") else None
    for offset, captured in enumerate(captured_blocks if model_file is None else range(payload["metadata"]["block_count"])):
        block = QwenImageTransformerBlock(
            dim=3072,
            num_attention_heads=24,
            attention_head_dim=128,
            dtype=torch.bfloat16,
            device="mps",
            operations=torch.nn,
        )
        if model_file is None:
            state_dict = captured["state_dict"]
        else:
            index = payload["metadata"]["block"] + offset
            prefix = f"transformer_blocks.{index}."
            state_dict = {
                key.removeprefix(prefix): model_file.get_tensor(key)
                for key in model_file.keys()
                if key.startswith(prefix)
            }
        block.load_state_dict(state_dict, strict=True)
        block.eval()
        blocks.append(block)
    inputs = {
        name: value.to("mps") if isinstance(value, torch.Tensor) else value
        for name, value in (payload if schema.endswith("boundary_capture.v1") else captured_blocks[0])["inputs"].items()
    }

    def execute():
        text = inputs["encoder_hidden_states"]
        image = inputs["hidden_states"]
        for block in blocks:
            text, image = block(
                hidden_states=image,
                encoder_hidden_states=text,
                encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
                temb=inputs["temb"],
                image_rotary_emb=inputs["image_rotary_emb"],
                timestep_zero_index=inputs["timestep_zero_index"],
                transformer_options=inputs["transformer_options"],
            )
        return text, image

    output = None
    with torch.inference_mode():
        for _ in range(args.warmups):
            output = execute()
            torch.mps.synchronize()
        timings = []
        for _ in range(args.iterations):
            torch.mps.synchronize()
            started = time.perf_counter()
            output = execute()
            torch.mps.synchronize()
            timings.append((time.perf_counter() - started) * 1000)

    names = ("encoder_hidden_states", "hidden_states")
    errors = {}
    for name, candidate in zip(names, output, strict=True):
        errors[name] = error_metrics(
            candidate.float().cpu().numpy(),
            (payload if schema.endswith("boundary_capture.v1") else captured_blocks[-1])["outputs"][name].float().numpy(),
        )
    report = {
        "schema_version": "kindred.comfyui.qwen_block_torch_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "capture": {"path": str(args.capture), "sha256": file_sha256(args.capture)},
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "implementation": "ComfyUI QwenImageTransformerBlock",
        },
        "method": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "block_count": len(blocks),
            "model": str(args.model) if args.model else None,
        },
        "pytorch_mps_bf16": {**summarize(timings), "error": errors},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
