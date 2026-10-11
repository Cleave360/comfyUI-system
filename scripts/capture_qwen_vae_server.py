#!/usr/bin/env python3
"""Launch ComfyUI and capture one real Qwen VAE input/output pair."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import runpy
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
CAPTURE_PATH = Path(
    os.environ.get(
        "KINDRED_QWEN_VAE_CAPTURE",
        ROOT / "reports" / "captures" / "qwen_vae_1024x576.pt",
    )
)
if not CAPTURE_PATH.is_absolute():
    CAPTURE_PATH = ROOT / CAPTURE_PATH

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

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
    if not captured and getattr(self, "latent_channels", None) == 16:
        payload = {
            "metadata": {
                "schema_version": "kindred.comfyui.qwen_vae_capture.v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "torch": torch.__version__,
                "vae_class": type(self.first_stage_model).__name__,
                "vae_dtype": str(self.vae_dtype),
                "decode_seconds": elapsed,
                "input_shape": list(samples_in.shape),
                "output_shape": list(output.shape),
            },
            "samples": samples_in.detach().cpu(),
            "output": output.detach().cpu(),
        }
        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = CAPTURE_PATH.with_suffix(CAPTURE_PATH.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(CAPTURE_PATH)
        captured = True
        print(f"Kindred Qwen VAE capture: {CAPTURE_PATH}")
    return output


sd.VAE.decode = capture_decode
runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
