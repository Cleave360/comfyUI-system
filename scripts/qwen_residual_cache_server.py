#!/usr/bin/env python3
"""Launch an isolated ComfyUI server with a bounded Qwen residual-cache trial.

This experimental harness reuses the previous complete-transformer residual at
explicitly selected sigma values. It is quality-changing and must not be used
as the production ComfyUI launch path.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
REPORT_PATH = Path(os.environ.get(
    "KINDRED_QWEN_CACHE_REPORT",
    ROOT / "reports" / "profiles" / "qwen_residual_cache.json",
))
TARGET_SIGMAS = tuple(
    float(value) for value in os.environ.get("KINDRED_QWEN_CACHE_SKIP_SIGMAS", "").split(",")
    if value.strip()
)
RUN_MODES = tuple(
    value.strip() for value in os.environ.get("KINDRED_QWEN_CACHE_RUN_MODES", "cache").split(",")
    if value.strip()
)
SIGMA_TOLERANCE = float(os.environ.get("KINDRED_QWEN_CACHE_SIGMA_TOLERANCE", "0.0001"))
if not TARGET_SIGMAS:
    raise SystemExit("KINDRED_QWEN_CACHE_SKIP_SIGMAS must contain at least one sigma")
if not RUN_MODES or any(value not in {"stock", "cache"} for value in RUN_MODES):
    raise SystemExit("KINDRED_QWEN_CACHE_RUN_MODES must contain only stock or cache")

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import utils.extra_config  # noqa: E402,F401 - reserve ComfyUI's utils package before torch imports

import torch  # noqa: E402
from comfy.ldm.qwen_image import model as qwen_model  # noqa: E402
import nodes  # noqa: E402


class KindredCacheBustLatent:
    """Pass a latent through unchanged while varying its cache signature."""

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


state: dict[str, Any] = {
    "active": False,
    "forward": 0,
    "skip": False,
    "sigma": None,
    "previous_sigma": None,
    "run_index": -1,
    "mode": RUN_MODES[0],
    "branch": "unknown",
    "key": None,
    "input_img": None,
    "input_txt": None,
    "cached": {},
    "records": [],
}


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def selected(sigma: float | None) -> bool:
    return sigma is not None and any(abs(sigma - target) <= SIGMA_TOLERANCE for target in TARGET_SIGMAS)


def mode_for_run(index: int) -> str:
    if len(RUN_MODES) == 1:
        return RUN_MODES[0]
    if index >= len(RUN_MODES):
        raise RuntimeError("Qwen residual cache run-mode sequence exhausted")
    return RUN_MODES[index]


def write_report() -> None:
    report = {
        "schema_version": "kindred.comfyui.qwen_residual_cache.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "method": {
            "intervention": True,
            "quality_changing": True,
            "target_sigmas": TARGET_SIGMAS,
            "sigma_tolerance": SIGMA_TOLERANCE,
            "run_modes": RUN_MODES,
            "description": "Reuses the preceding same-shape CFG-branch transformer residual.",
        },
        "records": state["records"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


original_block_forward = qwen_model.QwenImageTransformerBlock.forward
original_model_forward = qwen_model.QwenImageTransformer2DModel._forward


def cached_block_forward(self, *args, **kwargs):
    block_index = getattr(self, "_kindred_cache_block_index", -1)
    last_block = getattr(self, "_kindred_cache_last_block", -2)
    img = kwargs.get("hidden_states", args[0] if args else None)
    txt = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)

    if state["active"] and block_index == 0:
        state["input_img"] = img.detach().clone()
        state["input_txt"] = txt.detach().clone()
        state["key"] = f"branch={state['branch']};img={img.shape[1]};txt={txt.shape[1]}"
        state["skip"] = (
            state["mode"] == "cache"
            and selected(state["sigma"])
            and state["key"] in state["cached"]
        )

    if state["active"] and state["skip"]:
        if block_index == 0:
            cached = state["cached"][state["key"]]
            return txt + cached["txt"], img + cached["img"]
        return txt, img

    result = original_block_forward(self, *args, **kwargs)
    if state["active"] and block_index == last_block:
        output_txt, output_img = result
        state["cached"][state["key"]] = {
            "img": (output_img - state["input_img"]).detach().clone(),
            "txt": (output_txt - state["input_txt"]).detach().clone(),
        }
    return result


def cached_model_forward(self, *args, **kwargs):
    state["forward"] += 1
    x = args[0] if args else kwargs.get("x")
    timesteps = args[1] if len(args) > 1 else kwargs.get("timesteps")
    state["sigma"] = scalar(timesteps.reshape(-1)[0]) if isinstance(timesteps, torch.Tensor) else None
    control = kwargs.get("control")
    ref_latents = args[4] if len(args) > 4 else kwargs.get("ref_latents")
    if control is not None or ref_latents is not None or x is None or x.shape[0] != 1:
        raise RuntimeError("Qwen residual cache trial supports only batch-1 text-to-image without ControlNet")
    new_run = state["previous_sigma"] is None or (
        state["sigma"] is not None
        and state["previous_sigma"] is not None
        and state["sigma"] > state["previous_sigma"] + SIGMA_TOLERANCE
    )
    if new_run:
        state["cached"].clear()
        state["run_index"] += 1
        state["mode"] = mode_for_run(state["run_index"])
    state["previous_sigma"] = state["sigma"]
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        transformer_options = args[6] if len(args) > 6 else {}
    branch = transformer_options.get("cond_or_uncond", [])
    state["branch"] = ",".join(str(value) for value in branch) if branch else "unknown"
    last_block = len(self.transformer_blocks) - 1
    for index, block in enumerate(self.transformer_blocks):
        block._kindred_cache_block_index = index
        block._kindred_cache_last_block = last_block

    state["active"] = True
    state["skip"] = False
    started = time.perf_counter()
    try:
        result = original_model_forward(self, *args, **kwargs)
    finally:
        state["active"] = False
    state["records"].append({
        "forward": state["forward"],
        "run_index": state["run_index"],
        "mode": state["mode"],
        "branch_key": state["key"],
        "sigma": state["sigma"],
        "skipped": state["skip"],
        "wall_seconds": round(time.perf_counter() - started, 6),
    })
    write_report()
    return result


qwen_model.QwenImageTransformerBlock.forward = cached_block_forward
qwen_model.QwenImageTransformer2DModel._forward = cached_model_forward

runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
