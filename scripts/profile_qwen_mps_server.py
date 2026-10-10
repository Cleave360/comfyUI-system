#!/usr/bin/env python3
"""Launch ComfyUI with a bounded Qwen-Image MPS module profiler.

The profiler instruments exactly one diffusion-model forward pass selected by
KINDRED_QWEN_PROFILE_FORWARD (default: 3). It synchronizes MPS at measured
boundaries, so the selected step is diagnostic rather than a benchmark result.
All other diffusion steps execute without profiler synchronization.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
TARGET_FORWARD = int(os.environ.get("KINDRED_QWEN_PROFILE_FORWARD", "3"))
CAPTURE_BLOCK = int(os.environ.get("KINDRED_QWEN_CAPTURE_BLOCK", "-1"))
CAPTURE_BLOCK_COUNT = int(os.environ.get("KINDRED_QWEN_CAPTURE_BLOCK_COUNT", "1"))
CAPTURE_KIND = os.environ.get("KINDRED_QWEN_CAPTURE_KIND", "mlp")
PROFILE_PATH = Path(os.environ.get(
    "KINDRED_QWEN_PROFILE_PATH",
    ROOT / "reports" / "profiles" / "qwen_mps_modules.json",
))
CAPTURE_PATH = Path(os.environ.get(
    "KINDRED_QWEN_CAPTURE_PATH",
    ROOT / "reports" / "captures" / "qwen_mlp_block.pt",
))
if CAPTURE_KIND not in {"mlp", "block", "block_boundary"}:
    raise SystemExit("KINDRED_QWEN_CAPTURE_KIND must be mlp, block, or block_boundary")
if CAPTURE_BLOCK_COUNT < 1:
    raise SystemExit("KINDRED_QWEN_CAPTURE_BLOCK_COUNT must be positive")

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402
from comfy.ldm.qwen_image import model as qwen_model  # noqa: E402


state: dict[str, Any] = {
    "forward": 0,
    "active": False,
    "events": [],
    "captures": {},
    "block_captures": {},
    "block_boundary": {},
}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def synchronize() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(label: Callable[[Any], str]):
    def decorate(original):
        def wrapped(self, *args, **kwargs):
            if not state["active"]:
                return original(self, *args, **kwargs)
            category = label(self)
            block_index = getattr(self, "_kindred_block_index", None)
            capture_block = (
                CAPTURE_KIND in {"block", "block_boundary"}
                and category == "transformer_block"
                and CAPTURE_BLOCK <= block_index < CAPTURE_BLOCK + CAPTURE_BLOCK_COUNT
            )
            if capture_block:
                names = (
                    "hidden_states", "encoder_hidden_states", "encoder_hidden_states_mask",
                    "temb", "image_rotary_emb", "timestep_zero_index", "transformer_options",
                )
                inputs = {
                    name: kwargs.get(name, args[index] if index < len(args) else None)
                    for index, name in enumerate(names)
                }
            synchronize()
            started = time.perf_counter()
            result = original(self, *args, **kwargs)
            synchronize()
            stream = getattr(self, "_kindred_stream", None)
            if CAPTURE_KIND == "mlp" and block_index == CAPTURE_BLOCK and stream in {"image_mlp", "text_mlp"}:
                state["captures"][f"{stream}.input"] = args[0].detach().cpu()
                state["captures"][f"{stream}.output"] = result.detach().cpu()
                state["captures"][f"{stream}.expand_weight"] = self.net[0].proj.weight.detach().cpu()
                state["captures"][f"{stream}.expand_bias"] = self.net[0].proj.bias.detach().cpu()
                state["captures"][f"{stream}.contract_weight"] = self.net[2].weight.detach().cpu()
                state["captures"][f"{stream}.contract_bias"] = self.net[2].bias.detach().cpu()
            if capture_block and CAPTURE_KIND == "block":
                state["block_captures"][block_index] = {
                    "inputs": {name: cpu_copy(value) for name, value in inputs.items()},
                    "outputs": {
                        "encoder_hidden_states": cpu_copy(result[0]),
                        "hidden_states": cpu_copy(result[1]),
                    },
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in self.state_dict().items()
                    },
                }
            elif capture_block and CAPTURE_KIND == "block_boundary":
                if block_index == CAPTURE_BLOCK:
                    state["block_boundary"]["inputs"] = {
                        name: cpu_copy(value) for name, value in inputs.items()
                    }
                if block_index == CAPTURE_BLOCK + CAPTURE_BLOCK_COUNT - 1:
                    state["block_boundary"]["outputs"] = {
                        "encoder_hidden_states": cpu_copy(result[0]),
                        "hidden_states": cpu_copy(result[1]),
                    }
            state["events"].append({
                "category": category,
                "block": getattr(self, "_kindred_block_index", None),
                "milliseconds": round((time.perf_counter() - started) * 1000, 3),
            })
            return result
        return wrapped
    return decorate


def summarize(events: list[dict[str, Any]], total_ms: float) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for event in events:
        grouped[event["category"]].append(event["milliseconds"])
    aggregates = {
        category: {
            "calls": len(values),
            "total_ms": round(sum(values), 3),
            "mean_ms": round(sum(values) / len(values), 3),
            "max_ms": round(max(values), 3),
        }
        for category, values in sorted(grouped.items())
    }
    block_ms = aggregates.get("transformer_block", {}).get("total_ms", 0.0)
    nested_ms = sum(
        aggregates.get(category, {}).get("total_ms", 0.0)
        for category in ("joint_attention", "image_mlp", "text_mlp")
    )
    aggregates["block_other_derived"] = {
        "total_ms": round(block_ms - nested_ms, 3),
        "note": "block total minus attention and both MLP streams",
    }
    aggregates["model_other_derived"] = {
        "total_ms": round(total_ms - block_ms, 3),
        "note": "model forward total minus transformer blocks",
    }
    attention_linear_categories = (
        "image_qkv_linear", "text_qkv_linear", "image_attention_out_linear",
        "text_attention_out_linear",
    )
    attention_linear_ms = sum(
        aggregates.get(category, {}).get("total_ms", 0.0)
        for category in attention_linear_categories
    )
    aggregates["attention_non_linear_derived"] = {
        "total_ms": round(
            aggregates.get("joint_attention", {}).get("total_ms", 0.0)
            - attention_linear_ms
            - aggregates.get("scaled_dot_product_attention", {}).get("total_ms", 0.0),
            3,
        ),
        "note": "joint attention minus measured linears and scaled-dot-product attention",
    }
    return aggregates


def register_module_timer(module, category: str, block_index: int) -> None:
    def before(_module, _args):
        if state["active"]:
            synchronize()
            _module._kindred_profile_started = time.perf_counter()

    def after(_module, _args, _output):
        if state["active"]:
            synchronize()
            state["events"].append({
                "category": category,
                "block": block_index,
                "milliseconds": round(
                    (time.perf_counter() - _module._kindred_profile_started) * 1000,
                    3,
                ),
            })

    module.register_forward_pre_hook(before)
    module.register_forward_hook(after)


original_attention = qwen_model.Attention.forward
original_feed_forward = qwen_model.FeedForward.forward
original_block = qwen_model.QwenImageTransformerBlock.forward
original_model_forward = qwen_model.QwenImageTransformer2DModel._forward
original_sdpa = qwen_model.optimized_attention_masked

qwen_model.Attention.forward = timed(lambda _self: "joint_attention")(original_attention)
qwen_model.FeedForward.forward = timed(
    lambda self: getattr(self, "_kindred_stream", "mlp")
)(original_feed_forward)
qwen_model.QwenImageTransformerBlock.forward = timed(lambda _self: "transformer_block")(original_block)


def profiled_sdpa(*args, **kwargs):
    if not state["active"]:
        return original_sdpa(*args, **kwargs)
    synchronize()
    started = time.perf_counter()
    result = original_sdpa(*args, **kwargs)
    synchronize()
    state["events"].append({
        "category": "scaled_dot_product_attention",
        "block": None,
        "milliseconds": round((time.perf_counter() - started) * 1000, 3),
    })
    return result


qwen_model.optimized_attention_masked = profiled_sdpa


def profiled_model_forward(self, *args, **kwargs):
    state["forward"] += 1
    if state["forward"] != TARGET_FORWARD:
        return original_model_forward(self, *args, **kwargs)

    for index, block in enumerate(self.transformer_blocks):
        block._kindred_block_index = index
        block.attn._kindred_block_index = index
        block.img_mlp._kindred_block_index = index
        block.img_mlp._kindred_stream = "image_mlp"
        block.txt_mlp._kindred_block_index = index
        block.txt_mlp._kindred_stream = "text_mlp"
        for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
            register_module_timer(projection, "image_qkv_linear", index)
        for projection in (block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj):
            register_module_timer(projection, "text_qkv_linear", index)
        register_module_timer(block.attn.to_out[0], "image_attention_out_linear", index)
        register_module_timer(block.attn.to_add_out, "text_attention_out_linear", index)
        register_module_timer(block.img_mlp.net[0].proj, "image_mlp_expand_linear", index)
        register_module_timer(block.img_mlp.net[2], "image_mlp_contract_linear", index)
        register_module_timer(block.txt_mlp.net[0].proj, "text_mlp_expand_linear", index)
        register_module_timer(block.txt_mlp.net[2], "text_mlp_contract_linear", index)
        register_module_timer(block.img_mod[1], "image_modulation_linear", index)
        register_module_timer(block.txt_mod[1], "text_modulation_linear", index)

    state["events"] = []
    state["active"] = True
    synchronize()
    started = time.perf_counter()
    try:
        result = original_model_forward(self, *args, **kwargs)
        synchronize()
    finally:
        total_ms = (time.perf_counter() - started) * 1000
        state["active"] = False

    x = args[0] if args else kwargs.get("x")
    context = args[2] if len(args) > 2 else kwargs.get("context")
    report = {
        "schema_version": "kindred.comfyui.qwen_mps_profile.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "method": {
            "target_diffusion_forward": TARGET_FORWARD,
            "mps_synchronized_boundaries": True,
            "warning": "diagnostic synchronization perturbs this forward; do not use as an end-to-end benchmark",
        },
        "model": {
            "layers": len(self.transformer_blocks),
            "inner_dim": self.inner_dim,
            "attention_heads": self.transformer_blocks[0].num_attention_heads,
            "attention_head_dim": self.transformer_blocks[0].attention_head_dim,
            "latent_shape": list(x.shape) if x is not None else None,
            "context_shape": list(context.shape) if context is not None else None,
        },
        "total_ms": round(total_ms, 3),
        "aggregates": summarize(state["events"], total_ms),
        "events": state["events"],
    }
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Kindred Qwen MPS profile written: {PROFILE_PATH}", flush=True)
    if CAPTURE_BLOCK >= 0:
        if CAPTURE_KIND == "mlp":
            expected = {
                f"{stream}.{name}"
                for stream in ("image_mlp", "text_mlp")
                for name in ("input", "output", "expand_weight", "expand_bias", "contract_weight", "contract_bias")
            }
            missing = sorted(expected - state["captures"].keys())
            if missing:
                raise RuntimeError(f"Qwen MLP capture incomplete: {missing}")
            payload = {
                "metadata": {
                    "schema_version": "kindred.comfyui.qwen_mlp_capture.v1",
                    "torch": torch.__version__,
                    "profile_forward": TARGET_FORWARD,
                    "block": CAPTURE_BLOCK,
                },
                "tensors": state["captures"],
            }
        elif CAPTURE_KIND == "block":
            expected_blocks = set(range(CAPTURE_BLOCK, CAPTURE_BLOCK + CAPTURE_BLOCK_COUNT))
            missing_blocks = sorted(expected_blocks - state["block_captures"].keys())
            if missing_blocks:
                raise RuntimeError(f"Qwen transformer block captures missing: {missing_blocks}")
            metadata = {
                "schema_version": (
                    "kindred.comfyui.qwen_block_capture.v1"
                    if CAPTURE_BLOCK_COUNT == 1
                    else "kindred.comfyui.qwen_block_range_capture.v1"
                ),
                "torch": torch.__version__,
                "profile_forward": TARGET_FORWARD,
                "block": CAPTURE_BLOCK,
                "block_count": CAPTURE_BLOCK_COUNT,
            }
            if CAPTURE_BLOCK_COUNT == 1:
                payload = {"metadata": metadata, **state["block_captures"][CAPTURE_BLOCK]}
            else:
                payload = {
                    "metadata": metadata,
                    "blocks": {
                        str(index): state["block_captures"][index]
                        for index in sorted(state["block_captures"])
                    },
                }
        else:
            if set(state["block_boundary"]) != {"inputs", "outputs"}:
                raise RuntimeError("Qwen transformer block boundary capture was not recorded")
            payload = {
                "metadata": {
                    "schema_version": "kindred.comfyui.qwen_block_boundary_capture.v1",
                    "torch": torch.__version__,
                    "profile_forward": TARGET_FORWARD,
                    "block": CAPTURE_BLOCK,
                    "block_count": CAPTURE_BLOCK_COUNT,
                },
                **state["block_boundary"],
            }
        CAPTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, CAPTURE_PATH)
        print(f"Kindred Qwen {CAPTURE_KIND} capture written: {CAPTURE_PATH}", flush=True)
    return result


qwen_model.QwenImageTransformer2DModel._forward = profiled_model_forward

runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
