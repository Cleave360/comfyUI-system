#!/usr/bin/env python3
"""Launch an isolated ComfyUI server for baseline or padded Qwen CFG batching."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ComfyUI-source"
MODE = os.environ.get("KINDRED_QWEN_CFG_MODE", "baseline")
PAD_TOKENS = int(os.environ.get("KINDRED_QWEN_CFG_PAD_TOKENS", "40"))
REPORT_PATH = Path(os.environ.get(
    "KINDRED_QWEN_CFG_REPORT",
    ROOT / "reports" / "profiles" / f"qwen_cfg_{MODE}.json",
))
if MODE not in {"baseline", "padded", "padded_serial"}:
    raise SystemExit("KINDRED_QWEN_CFG_MODE must be baseline, padded, or padded_serial")

sys.path.insert(0, str(SOURCE))
os.chdir(SOURCE)

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.conds  # noqa: E402
import comfy.model_base  # noqa: E402
import comfy.samplers  # noqa: E402
from comfy.ldm.qwen_image import model as qwen_model  # noqa: E402


state = {"forward_calls": 0, "batch_sizes": [], "context_lengths": []}


def write_report() -> None:
    report = {
        "schema_version": "kindred.comfyui.qwen_cfg_batch_profile.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "mode": MODE,
        "pad_tokens": PAD_TOKENS if MODE != "baseline" else None,
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
        "forward_calls": state["forward_calls"],
        "batch_size_counts": dict(Counter(map(str, state["batch_sizes"]))),
        "context_length_counts": dict(Counter(map(str, state["context_lengths"]))),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


original_model_forward = qwen_model.QwenImageTransformer2DModel._forward


def counted_model_forward(self, *args, **kwargs):
    x = args[0] if args else kwargs["x"]
    context = args[2] if len(args) > 2 else kwargs["context"]
    result = original_model_forward(self, *args, **kwargs)
    state["forward_calls"] += 1
    state["batch_sizes"].append(int(x.shape[0]))
    state["context_lengths"].append(int(context.shape[1]))
    write_report()
    return result


qwen_model.QwenImageTransformer2DModel._forward = counted_model_forward


if MODE != "baseline":
    original_extra_conds = comfy.model_base.QwenImage.extra_conds
    original_attention = qwen_model.Attention.forward

    def padded_extra_conds(self, **kwargs):
        cross_attn = kwargs.get("cross_attn")
        if cross_attn is None:
            return original_extra_conds(self, **kwargs)
        token_count = cross_attn.shape[1]
        if token_count > PAD_TOKENS:
            raise RuntimeError(
                f"Qwen context has {token_count} tokens, exceeding configured pad length {PAD_TOKENS}",
            )
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                (cross_attn.shape[0], token_count), device=cross_attn.device, dtype=torch.bool,
            )
        else:
            attention_mask = attention_mask.to(torch.bool)
        padding = PAD_TOKENS - token_count
        if padding:
            cross_attn = F.pad(cross_attn, (0, 0, 0, padding))
            attention_mask = F.pad(attention_mask, (0, padding), value=False)
        patched = dict(kwargs)
        patched["cross_attn"] = cross_attn
        output = original_extra_conds(self, **patched)
        output["attention_mask"] = comfy.conds.CONDRegular(attention_mask)
        return output

    def masked_attention(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_mask=None,
        image_rotary_emb=None,
        transformer_options={},
    ):
        if encoder_hidden_states_mask is not None and attention_mask is None:
            valid_text = encoder_hidden_states_mask.to(torch.bool)
            valid_image = torch.ones(
                (hidden_states.shape[0], hidden_states.shape[1]),
                device=hidden_states.device,
                dtype=torch.bool,
            )
            valid_keys = torch.cat((valid_text, valid_image), dim=1)
            attention_mask = torch.zeros(
                (hidden_states.shape[0], 1, 1, valid_keys.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            attention_mask.masked_fill_(~valid_keys[:, None, None, :], -torch.finfo(hidden_states.dtype).max)
        return original_attention(
            self,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
            transformer_options=transformer_options,
        )

    comfy.model_base.QwenImage.extra_conds = padded_extra_conds
    qwen_model.Attention.forward = masked_attention

    if MODE == "padded_serial":
        original_can_concat_cond = comfy.samplers.can_concat_cond

        def keep_qwen_cfg_serial(first, second):
            # Sampler candidate selection compares the first condition with
            # itself. Preserve that match so the candidate batch is never
            # empty; only prevent distinct Qwen CFG branches from merging.
            if first is second:
                return True
            if "c_crossattn" in first.conditioning and "c_crossattn" in second.conditioning:
                return False
            return original_can_concat_cond(first, second)

        comfy.samplers.can_concat_cond = keep_qwen_cfg_serial


write_report()
runpy.run_path(str(SOURCE / "main.py"), run_name="__main__")
