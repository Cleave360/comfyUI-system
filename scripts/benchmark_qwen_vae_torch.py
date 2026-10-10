#!/usr/bin/env python3
"""Benchmark the native ComfyUI Wan/Qwen VAE on a captured real latent."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        default=ROOT / "reports" / "captures" / "qwen_vae_1024x576.pt",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=SOURCE / "models" / "vae" / "qwen_image_vae.safetensors",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "benchmarks" / "qwen_vae_torch251.json",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.iterations < 1:
        raise SystemExit("warmups must be non-negative and iterations must be positive")
    if not args.capture.is_file() or not args.weights.is_file():
        raise SystemExit("capture and VAE weights are required")

    # ComfyUI owns its CLI parser. Hide this benchmark's already-parsed options.
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)
    import comfy.options

    comfy.options.enable_args_parsing()
    import utils.extra_config  # noqa: F401
    import torch
    import comfy.sd
    import comfy.utils

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is required")
    captured = torch.load(args.capture, map_location="cpu", weights_only=False)
    samples = captured["samples"]
    state = comfy.utils.load_torch_file(str(args.weights), safe_load=True)
    vae = comfy.sd.VAE(sd=state)

    cold_started = time.perf_counter()
    output = vae.decode(samples)
    torch.mps.synchronize()
    cold_ms = (time.perf_counter() - cold_started) * 1000
    for _ in range(args.warmups):
        output = vae.decode(samples)
        torch.mps.synchronize()

    timings = []
    for _ in range(args.iterations):
        torch.mps.synchronize()
        started = time.perf_counter()
        output = vae.decode(samples)
        torch.mps.synchronize()
        timings.append((time.perf_counter() - started) * 1000)

    exact_capture_match = bool(torch.equal(output.cpu(), captured["output"]))
    report = {
        "schema_version": "kindred.comfyui.qwen_vae_torch.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "device": "mps",
            "vae_dtype": str(vae.vae_dtype),
        },
        "artifacts": {
            "capture": str(args.capture),
            "capture_sha256": file_sha256(args.capture),
            "weights": str(args.weights),
            "weights_sha256": file_sha256(args.weights),
        },
        "method": {
            "warmups": args.warmups,
            "iterations": args.iterations,
            "device_synchronized": True,
            "model_loading_excluded_from_measured_iterations": True,
        },
        "input_shape": list(samples.shape),
        "output_shape": list(output.shape),
        "output_device": str(output.device),
        "cold_decode_ms": round(cold_ms, 3),
        "warm_decode": summarize(timings),
        "exact_capture_match": exact_capture_match,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
