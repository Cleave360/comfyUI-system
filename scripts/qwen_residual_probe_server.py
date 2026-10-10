#!/usr/bin/env python3
"""Launch ComfyUI with a read-only Qwen transformer residual probe.

The probe measures how much the complete 60-block transformer residual changes
between consecutive diffusion forwards for the same CFG branch. It never reuses
a residual and therefore cannot change the generated image.
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
    "KINDRED_QWEN_RESIDUAL_REPORT",
    ROOT / "reports" / "profiles" / "qwen_residual_probe.json",
))

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.ldm.qwen_image import model as qwen_model  # noqa: E402


state: dict[str, Any] = {
    "active": False,
    "forward": 0,
    "input_img": None,
    "input_txt": None,
    "metadata": {},
    "previous": {},
    "records": [],
}


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def comparison(current: torch.Tensor, previous: torch.Tensor | None) -> dict[str, float] | None:
    if previous is None:
        return None
    current_f = current.float()
    previous_f = previous.float()
    delta = current_f - previous_f
    eps = torch.finfo(torch.float32).eps
    current_l1 = current_f.abs().mean().clamp_min(eps)
    current_l2 = torch.linalg.vector_norm(current_f).clamp_min(eps)
    cosine = torch.nn.functional.cosine_similarity(
        current_f.reshape(1, -1), previous_f.reshape(1, -1), dim=1
    )
    return {
        "reuse_relative_l1": scalar(delta.abs().mean() / current_l1),
        "reuse_relative_l2": scalar(torch.linalg.vector_norm(delta) / current_l2),
        "cosine_similarity": scalar(cosine[0]),
    }


def branch_key(metadata: dict[str, Any], img: torch.Tensor, txt: torch.Tensor) -> str:
    branch = metadata.get("cond_or_uncond")
    branch_label = ",".join(str(value) for value in branch) if branch else "unknown"
    return f"branch={branch_label};img={img.shape[1]};txt={txt.shape[1]}"


def write_report() -> None:
    report = {
        "schema_version": "kindred.comfyui.qwen_residual_probe.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "method": {
            "intervention": False,
            "description": (
                "Measures consecutive complete-transformer residual similarity "
                "within each CFG branch; no residual is reused."
            ),
        },
        "records": state["records"],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(REPORT_PATH)


original_block_forward = qwen_model.QwenImageTransformerBlock.forward
original_model_forward = qwen_model.QwenImageTransformer2DModel._forward


def probed_block_forward(self, *args, **kwargs):
    block_index = getattr(self, "_kindred_probe_block_index", -1)
    if state["active"] and block_index == 0:
        img = kwargs.get("hidden_states", args[0] if args else None)
        txt = kwargs.get("encoder_hidden_states", args[1] if len(args) > 1 else None)
        state["input_img"] = img.detach().clone()
        state["input_txt"] = txt.detach().clone()

    result = original_block_forward(self, *args, **kwargs)

    if state["active"] and block_index == getattr(self, "_kindred_probe_last_block", -2):
        output_txt, output_img = result
        input_img = state["input_img"]
        input_txt = state["input_txt"]
        img_residual = (output_img - input_img).detach()
        txt_residual = (output_txt - input_txt).detach()
        key = branch_key(state["metadata"], output_img, output_txt)
        previous = state["previous"].get(key)
        state["records"].append({
            "forward": state["forward"],
            "branch_key": key,
            "cond_or_uncond": state["metadata"].get("cond_or_uncond"),
            "sigma": state["metadata"].get("sigma"),
            "image": comparison(img_residual, None if previous is None else previous["img"]),
            "text": comparison(txt_residual, None if previous is None else previous["txt"]),
        })
        state["previous"][key] = {
            "img": img_residual.clone(),
            "txt": txt_residual.clone(),
        }
    return result


def probed_model_forward(self, *args, **kwargs):
    state["forward"] += 1
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        transformer_options = args[6] if len(args) > 6 else {}
    timesteps = args[1] if len(args) > 1 else kwargs.get("timesteps")
    sigma = None
    if isinstance(timesteps, torch.Tensor) and timesteps.numel():
        sigma = scalar(timesteps.reshape(-1)[0])
    state["metadata"] = {
        "cond_or_uncond": list(transformer_options.get("cond_or_uncond", [])),
        "sigma": sigma,
    }
    last_block = len(self.transformer_blocks) - 1
    for index, block in enumerate(self.transformer_blocks):
        block._kindred_probe_block_index = index
        block._kindred_probe_last_block = last_block

    state["active"] = True
    started = time.perf_counter()
    try:
        return original_model_forward(self, *args, **kwargs)
    finally:
        state["active"] = False
        if state["records"]:
            state["records"][-1]["wall_seconds"] = round(time.perf_counter() - started, 6)
            write_report()


qwen_model.QwenImageTransformerBlock.forward = probed_block_forward
qwen_model.QwenImageTransformer2DModel._forward = probed_model_forward

runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
