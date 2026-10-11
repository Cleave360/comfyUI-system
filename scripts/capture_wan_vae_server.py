#!/usr/bin/env python3
"""Launch ComfyUI and capture one genuine temporal Wan VAE decode."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
CAPTURE_PATH = Path(
    os.environ.get(
        "KINDRED_WAN_VAE_CAPTURE",
        ROOT / "reports" / "captures" / "wan_vace_1.3b_native_video.pt",
    )
)
if not CAPTURE_PATH.is_absolute():
    CAPTURE_PATH = ROOT / CAPTURE_PATH

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

# The newer-Torch experiment keeps its own Torch packages first, then borrows
# ComfyUI's pure-Python dependencies from the production environment.
PRODUCTION_VENV = (ROOT / ".venv").resolve()
if not Path(sys.executable).resolve().is_relative_to(PRODUCTION_VENV):
    production_sites = sorted((PRODUCTION_VENV / "lib").glob("python*/site-packages"))
    if len(production_sites) != 1:
        raise SystemExit("expected exactly one production site-packages directory")
    sys.path.append(str(production_sites[0]))
    torch_version = importlib.metadata.version("torch").split("+", 1)[0]
    torchaudio_version = importlib.metadata.version("torchaudio").split("+", 1)[0]
    torch_major_minor = tuple(int(value) for value in torch_version.split(".")[:2])
    if torchaudio_version != "2.11.0" or torch_major_minor < (2, 11):
        raise SystemExit(
            "isolated audio runtime requires torchaudio 2.11.0 and torch >= 2.11: "
            f"torch={torch_version}, torchaudio={torchaudio_version}"
        )

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import utils.extra_config  # noqa: E402,F401
import torch  # noqa: E402
from comfy import sd  # noqa: E402


original_decode = sd.VAE.decode
captured = False


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def capture_decode(self, samples_in, vae_options={}):
    global captured
    synchronize()
    started = time.perf_counter()
    output = original_decode(self, samples_in, vae_options=vae_options)
    synchronize()
    elapsed = time.perf_counter() - started
    vae_class = type(self.first_stage_model).__name__
    is_temporal_wan = (
        vae_class == "WanVAE"
        and getattr(self, "latent_channels", None) == 16
        and samples_in.ndim == 5
        and samples_in.shape[2] > 1
    )
    if not captured and is_temporal_wan:
        payload = {
            "metadata": {
                "schema_version": "kindred.comfyui.wan_vae_capture.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__,
                "vae_class": vae_class,
                "vae_dtype": str(self.vae_dtype),
                "decode_seconds": elapsed,
                "input_shape": list(samples_in.shape),
                "output_shape": list(output.shape),
                "genuine_temporal_latent": True,
            },
            "samples": samples_in.detach().cpu(),
            "output": output.detach().cpu(),
        }
        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CAPTURE_PATH.with_suffix(CAPTURE_PATH.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(CAPTURE_PATH)
        captured = True
        print(f"Kindred Wan VAE capture: {CAPTURE_PATH}")
    return output


sd.VAE.decode = capture_decode
runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
