from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_wan_vae_capture_server_requires_genuine_temporal_wan_input():
    source = (ROOT / "scripts" / "capture_wan_vae_server.py").read_text(
        encoding="utf-8"
    )
    ast.parse(source)
    assert "kindred.comfyui.wan_vae_capture.v1" in source
    assert 'vae_class == "WanVAE"' in source
    assert "samples_in.ndim == 5" in source
    assert "samples_in.shape[2] > 1" in source
    assert '"genuine_temporal_latent": True' in source
    assert "temporary.replace(CAPTURE_PATH)" in source


def test_wan_native_video_workflow_is_a_small_real_temporal_graph():
    path = ROOT / "ComfyUI-workflows" / "wan-vace-1.3b-native-video-api.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    by_type = {node["class_type"]: node for node in workflow.values()}
    assert by_type["UNETLoader"]["inputs"]["unet_name"] == (
        "wan2.1_vace_1.3B_fp16.safetensors"
    )
    assert by_type["VAELoader"]["inputs"]["vae_name"] == "wan_2.1_vae.safetensors"
    assert by_type["WanVaceToVideo"]["inputs"]["length"] == 17
    assert by_type["KSampler"]["inputs"]["steps"] == 4
    assert by_type["KSampler"]["inputs"]["sampler_name"] == "uni_pc"
    assert "VAEDecode" in by_type
