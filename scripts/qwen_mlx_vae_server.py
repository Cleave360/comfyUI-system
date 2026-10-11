#!/usr/bin/env python3
"""Run an isolated ComfyUI server with an opt-in MLX Qwen VAE decoder.

This harness must be launched with the ignored Torch 2.12/MLX environment. It
borrows only non-Torch packages from the production virtual environment and
never changes the normal ComfyUI server or its pinned dependencies.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
MLX_EXAMPLES = ROOT / "reports" / "vendor" / "mlx-examples"
WAN_MODULES = MLX_EXAMPLES / "video" / "wan2.1" / "wan"
REPORT_PATH = Path(os.environ.get(
    "KINDRED_QWEN_MLX_VAE_REPORT",
    ROOT / "reports" / "profiles" / "qwen_mlx_vae_server.json",
))
WEIGHTS_PATH = Path(os.environ.get(
    "KINDRED_QWEN_MLX_VAE_WEIGHTS",
    SOURCE / "models" / "vae" / "qwen_image_vae.safetensors",
))
CAPTURE_DIR = os.environ.get("KINDRED_QWEN_MLX_VAE_CAPTURE_DIR", "").strip()
RUN_MODES = tuple(
    value.strip() for value in os.environ.get(
        "KINDRED_QWEN_MLX_VAE_RUN_MODES", "mlx",
    ).split(",") if value.strip()
)
if not RUN_MODES or any(value not in {"stock", "mlx"} for value in RUN_MODES):
    raise SystemExit("KINDRED_QWEN_MLX_VAE_RUN_MODES must contain only stock or mlx")
if not WEIGHTS_PATH.is_file():
    raise SystemExit(f"Qwen VAE weights not found: {WEIGHTS_PATH}")

# Keep the isolated environment's Torch/torchvision first. Append the production
# environment only as a fallback for ComfyUI's pure-Python dependencies.
production_sites = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
if len(production_sites) != 1:
    raise SystemExit("expected exactly one production site-packages directory")
sys.path.append(str(production_sites[0]))
sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

# Do not let the fallback environment expose a torchaudio binary built for
# production Torch 2.5.1. The Qwen image route does not use audio nodes.
sys.modules["torchaudio"] = None
import main as comfy_main  # noqa: E402 - lets ComfyUI own its Torch import order
import mlx.core as mx  # noqa: E402
import torch  # noqa: E402
from comfy import sd  # noqa: E402
import nodes  # noqa: E402


MEAN = torch.tensor([
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]).view(1, 16, 1, 1, 1)
STD = torch.tensor([
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]).view(1, 16, 1, 1, 1)


class KindredCacheBustLatent:
    """Pass a latent unchanged while varying its ComfyUI cache signature."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "nonce": ("INT", {"default": 0})}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "Kindred/Testing"

    def execute(self, samples, nonce):
        del nonce
        return (samples,)


nodes.NODE_CLASS_MAPPINGS["KindredCacheBustLatent"] = KindredCacheBustLatent
nodes.NODE_DISPLAY_NAME_MAPPINGS["KindredCacheBustLatent"] = "Kindred Cache Bust Latent (Test)"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_official_wan_vae_class():
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


state: dict[str, Any] = {
    "model": None,
    "model_load_seconds": None,
    "decode_index": 0,
    "records": [],
}


def mode_for_decode(index: int) -> str:
    if len(RUN_MODES) == 1:
        return RUN_MODES[0]
    if index >= len(RUN_MODES):
        raise RuntimeError("Qwen MLX VAE run-mode sequence exhausted")
    return RUN_MODES[index]


def mlx_model():
    if state["model"] is None:
        started = time.perf_counter()
        WanVAE = load_official_wan_vae_class()
        model = WanVAE()
        weights = WanVAE.sanitize(mx.load(str(WEIGHTS_PATH)))
        model.load_weights(list(weights.items()), strict=False)
        mx.eval(model.parameters())
        state["model"] = model
        state["model_load_seconds"] = time.perf_counter() - started
    return state["model"]


def write_report() -> None:
    report = {
        "schema_version": "kindred.comfyui.qwen_mlx_vae_server.v1",
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
            "weights": str(WEIGHTS_PATH),
            "weights_sha256": file_sha256(WEIGHTS_PATH),
        },
        "method": {
            "experimental": True,
            "production_modified": False,
            "run_modes": RUN_MODES,
            "torch_to_mlx": "mx.asarray(torch_mps_tensor, copy=False)",
            "mlx_to_torch": "torch.as_tensor(mlx_array)",
            "output_device": "cpu",
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "") == "1",
        },
        "model_load_seconds": state["model_load_seconds"],
        "records": state["records"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


def mlx_decode(samples_in: torch.Tensor) -> torch.Tensor:
    if samples_in.ndim != 5 or samples_in.shape[1] != 16:
        raise ValueError(f"expected BCTHW Wan latent, received {tuple(samples_in.shape)}")
    mean = MEAN.to(samples_in.device, torch.float32)
    std = STD.to(samples_in.device, torch.float32)
    # Native ComfyUI casts the raw, unscaled latent to the VAE's BF16 dtype
    # before decode. Reproduce that rounding before converting back to the
    # normalized representation expected by Apple's high-level MLX wrapper.
    rounded = samples_in.to(torch.bfloat16).float()
    normalized = ((rounded - mean) / std).permute(0, 2, 3, 4, 1).contiguous()
    outputs = []
    model = mlx_model()
    for sample in normalized:
        source_mps = sample.to("mps")
        torch.mps.synchronize()
        latent_mlx = mx.asarray(source_mps, copy=False)
        decoded = model.decode(latent_mlx)
        pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
        mx.eval(pixels)
        pixels_torch = torch.as_tensor(pixels)
        torch.mps.synchronize()
        outputs.append(pixels_torch.cpu())
    return torch.stack(outputs)


original_decode = sd.VAE.decode


def routed_decode(self, samples_in, vae_options={}):
    is_target = (
        getattr(self, "latent_channels", None) == 16
        and type(getattr(self, "first_stage_model", None)).__name__ == "WanVAE"
    )
    if not is_target:
        return original_decode(self, samples_in, vae_options=vae_options)
    index = state["decode_index"]
    mode = mode_for_decode(index)
    state["decode_index"] += 1
    torch.mps.synchronize()
    started = time.perf_counter()
    peak_before = mx.get_peak_memory()
    if mode == "mlx":
        if vae_options:
            raise ValueError("MLX VAE trial does not support VAE options or tiled decoding")
        output = mlx_decode(samples_in)
    else:
        output = original_decode(self, samples_in, vae_options=vae_options)
    torch.mps.synchronize()
    record = {
        "decode_index": index,
        "mode": mode,
        "seconds": round(time.perf_counter() - started, 6),
        "input_shape": list(samples_in.shape),
        "input_device": str(samples_in.device),
        "input_sha256": tensor_sha256(samples_in),
        "output_shape": list(output.shape),
        "output_device": str(output.device),
        "output_sha256": tensor_sha256(output),
        "all_finite": bool(torch.isfinite(output).all()),
        "mlx_peak_memory_bytes": mx.get_peak_memory() if mode == "mlx" else peak_before,
    }
    if CAPTURE_DIR:
        capture_dir = Path(CAPTURE_DIR)
        capture_dir.mkdir(parents=True, exist_ok=True)
        capture_path = capture_dir / f"decode_{index:02d}_{mode}.pt"
        temporary = capture_path.with_suffix(capture_path.suffix + ".tmp")
        torch.save({
            "mode": mode,
            "samples": samples_in.detach().cpu(),
            "output": output.detach().cpu(),
        }, temporary)
        temporary.replace(capture_path)
        record["capture"] = str(capture_path)
    state["records"].append(record)
    write_report()
    print(
        f"Kindred Qwen VAE mode={mode} decode={index} seconds={record['seconds']:.6f}",
        flush=True,
    )
    return output


sd.VAE.decode = routed_decode
write_report()
event_loop, _, start_all = comfy_main.start_comfyui()
try:
    event_loop.run_until_complete(start_all())
except KeyboardInterrupt:
    pass
finally:
    comfy_main.cleanup_temp()
