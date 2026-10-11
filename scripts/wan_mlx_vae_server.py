#!/usr/bin/env python3
"""Run isolated ComfyUI with alternating stock/MLX temporal Wan VAE decode."""

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
    "KINDRED_WAN_MLX_VAE_REPORT",
    ROOT / "reports" / "profiles" / "wan_mlx_vae_server.json",
))
WEIGHTS_PATH = Path(os.environ.get(
    "KINDRED_WAN_MLX_VAE_WEIGHTS",
    ROOT / "models" / "vae" / "wan_2.1_vae.safetensors",
))
RUN_MODES = tuple(
    value.strip() for value in os.environ.get(
        "KINDRED_WAN_MLX_VAE_RUN_MODES", "mlx",
    ).split(",") if value.strip()
)
CLEANUP_POLICY = os.environ.get("KINDRED_WAN_MPS_CLEANUP_POLICY", "none").strip()
ALL_VAE_CLEANUP = os.environ.get("KINDRED_TORCH_EMPTY_CACHE_ALL_VAE", "0").strip()
if not REPORT_PATH.is_absolute():
    REPORT_PATH = ROOT / REPORT_PATH
if not WEIGHTS_PATH.is_absolute():
    WEIGHTS_PATH = ROOT / WEIGHTS_PATH
if not RUN_MODES or any(value not in {"stock", "mlx"} for value in RUN_MODES):
    raise SystemExit("KINDRED_WAN_MLX_VAE_RUN_MODES must contain only stock or mlx")
if CLEANUP_POLICY not in {"none", "torch_empty_cache"}:
    raise SystemExit(
        "KINDRED_WAN_MPS_CLEANUP_POLICY must be none or torch_empty_cache"
    )
if ALL_VAE_CLEANUP not in {"0", "1"}:
    raise SystemExit("KINDRED_TORCH_EMPTY_CACHE_ALL_VAE must be 0 or 1")
if not WEIGHTS_PATH.is_file():
    raise SystemExit(f"Wan VAE weights not found: {WEIGHTS_PATH}")

# Keep isolated Torch packages first and borrow only pure-Python production
# dependencies. TorchAudio 2.11 uses PyTorch's stable ABI for Torch >= 2.11.
production_sites = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
if len(production_sites) != 1:
    raise SystemExit("expected exactly one production site-packages directory")
sys.path.append(str(production_sites[0]))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(SOURCE)

torch_package_version = importlib.metadata.version("torch").split("+", 1)[0]
torchaudio_package_version = importlib.metadata.version("torchaudio").split("+", 1)[0]
torch_major_minor = tuple(int(value) for value in torch_package_version.split(".")[:2])
if torchaudio_package_version != "2.11.0" or torch_major_minor < (2, 11):
    raise SystemExit(
        "isolated audio runtime requires torchaudio 2.11.0 and torch >= 2.11: "
        f"torch={torch_package_version}, torchaudio={torchaudio_package_version}"
    )

import main as comfy_main  # noqa: E402 - ComfyUI owns Torch initialization
import mlx.core as mx  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from comfy import sd  # noqa: E402
import nodes  # noqa: E402
from benchmark_qwen_vae_video import layer_eager_decode  # noqa: E402


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
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def load_official_wan_vae_class():
    package = types.ModuleType("kindred_mlx_wan_server_reference")
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
    "non_target_decode_records": [],
    "sampler_records": [],
}


def memory_snapshot() -> dict[str, int]:
    return {
        "torch_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "torch_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "torch_recommended_max_bytes": torch.mps.recommended_max_memory(),
        "mlx_active_memory_bytes": mx.get_active_memory(),
        "mlx_cache_memory_bytes": mx.get_cache_memory(),
        "process_rss_bytes": psutil.Process().memory_info().rss,
        "system_available_bytes": psutil.virtual_memory().available,
    }


def mode_for_decode(index: int) -> str:
    if len(RUN_MODES) == 1:
        return RUN_MODES[0]
    if index >= len(RUN_MODES):
        raise RuntimeError("Wan MLX VAE run-mode sequence exhausted")
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
        "schema_version": "kindred.comfyui.wan_mlx_vae_server.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "process_id": os.getpid(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchaudio": torchaudio_package_version,
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
            "decoder_policy": "cache_consuming_chunked_resample",
            "torch_to_mlx": "mx.asarray(torch_mps_tensor, copy=False)",
            "mlx_to_torch": "torch.as_tensor(mlx_array)",
            "output_device": "cpu",
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "") == "1",
            "cleanup_policy": CLEANUP_POLICY,
            "all_vae_cleanup": ALL_VAE_CLEANUP == "1",
        },
        "model_load_seconds": state["model_load_seconds"],
        "sampler_records": state["sampler_records"],
        "non_target_decode_records": state["non_target_decode_records"],
        "records": state["records"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


def mlx_decode(samples_in: torch.Tensor) -> torch.Tensor:
    if samples_in.ndim != 5 or tuple(samples_in.shape[:2]) != (1, 16):
        raise ValueError(f"expected one BCTHW Wan latent, received {tuple(samples_in.shape)}")
    if samples_in.shape[2] <= 1:
        raise ValueError("full-stack Wan trial requires a genuine temporal latent")
    mean = MEAN.to(samples_in.device, torch.float32)
    std = STD.to(samples_in.device, torch.float32)
    rounded = samples_in.to(torch.bfloat16).float()
    normalized = ((rounded - mean) / std).permute(0, 2, 3, 4, 1).contiguous()
    sample = normalized[0]
    source_mps = sample.to("mps")
    torch.mps.synchronize()
    latent_mlx = mx.asarray(source_mps, copy=False)
    decoded = layer_eager_decode(
        mlx_model(), latent_mlx, operation_eager=True, consume_cache=True,
        chunk_spatial_resample=True,
    )
    pixels = mx.clip((decoded.astype(mx.float32) + 1.0) / 2.0, 0.0, 1.0)
    mx.eval(pixels)
    pixels_torch = torch.as_tensor(pixels)
    torch.mps.synchronize()
    output = pixels_torch.cpu().unsqueeze(0)
    del decoded, pixels, pixels_torch, latent_mlx, source_mps
    mx.clear_cache()
    return output


original_decode = sd.VAE.decode
original_common_ksampler = nodes.common_ksampler


def routed_common_ksampler(*args, **kwargs):
    index = len(state["sampler_records"])
    before = memory_snapshot()
    torch.mps.synchronize()
    started = time.perf_counter()
    output = original_common_ksampler(*args, **kwargs)
    torch.mps.synchronize()
    record = {
        "sampler_index": index,
        "seconds": round(time.perf_counter() - started, 6),
        "memory_before": before,
        "memory_after": memory_snapshot(),
    }
    state["sampler_records"].append(record)
    write_report()
    print(
        f"Kindred Wan sampler={index} seconds={record['seconds']:.6f}",
        flush=True,
    )
    return output


def routed_decode(self, samples_in, vae_options={}):
    is_target = (
        getattr(self, "latent_channels", None) == 16
        and type(getattr(self, "first_stage_model", None)).__name__ == "WanVAE"
        and samples_in.ndim == 5
        and samples_in.shape[2] > 1
    )
    if not is_target:
        torch.mps.synchronize()
        started = time.perf_counter()
        output = original_decode(self, samples_in, vae_options=vae_options)
        torch.mps.synchronize()
        record = {
            "decode_index": len(state["non_target_decode_records"]),
            "vae_class": type(getattr(self, "first_stage_model", None)).__name__,
            "seconds": round(time.perf_counter() - started, 6),
            "memory_before_cleanup": memory_snapshot(),
        }
        if ALL_VAE_CLEANUP == "1":
            torch.mps.empty_cache()
        record["memory_after_cleanup"] = memory_snapshot()
        state["non_target_decode_records"].append(record)
        write_report()
        return output
    index = state["decode_index"]
    mode = mode_for_decode(index)
    state["decode_index"] += 1
    if mode == "mlx" and vae_options:
        raise ValueError("MLX Wan VAE trial does not support VAE options or tiled decoding")
    if mode == "mlx":
        mx.reset_peak_memory()
    torch.mps.synchronize()
    started = time.perf_counter()
    output = (
        mlx_decode(samples_in)
        if mode == "mlx"
        else original_decode(self, samples_in, vae_options=vae_options)
    )
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
        "mlx_peak_memory_bytes": mx.get_peak_memory() if mode == "mlx" else None,
        "mlx_active_memory_bytes": mx.get_active_memory() if mode == "mlx" else None,
        "mlx_cache_memory_bytes": mx.get_cache_memory() if mode == "mlx" else None,
    }
    state["records"].append(record)
    record["memory_before_cleanup"] = memory_snapshot()
    if CLEANUP_POLICY == "torch_empty_cache":
        torch.mps.empty_cache()
    record["memory_after_cleanup"] = memory_snapshot()
    write_report()
    print(
        f"Kindred Wan VAE mode={mode} decode={index} seconds={record['seconds']:.6f}",
        flush=True,
    )
    return output


nodes.common_ksampler = routed_common_ksampler
sd.VAE.decode = routed_decode
write_report()
event_loop, _, start_all = comfy_main.start_comfyui()
try:
    event_loop.run_until_complete(start_all())
except KeyboardInterrupt:
    pass
finally:
    comfy_main.cleanup_temp()
