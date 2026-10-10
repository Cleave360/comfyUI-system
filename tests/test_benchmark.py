from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_apple_silicon as benchmark


def test_profiles_are_valid_api_workflows():
    for profile in benchmark.PROFILES:
        workflow = benchmark.load_workflow(profile)
        assert all(node.get("class_type") for node in workflow.values())


def test_configure_workflow_is_deterministic_and_does_not_mutate_source():
    source = benchmark.load_workflow("brand-social")
    before = json.dumps(source, sort_keys=True)
    configured = benchmark.configure_workflow(
        source,
        prompt="A test campaign",
        seed=1234,
        prefix="benchmarks/test",
        width=640,
        height=384,
        steps=3,
    )
    assert json.dumps(source, sort_keys=True) == before
    assert configured["25"]["inputs"]["noise_seed"] == 1234
    assert configured["27"]["inputs"] | {"width": 640, "height": 384} == configured["27"]["inputs"]
    assert configured["17"]["inputs"]["steps"] == 3
    assert configured["9"]["inputs"]["filename_prefix"] == "benchmarks/test"
    assert "A test campaign" in configured["6"]["inputs"]["text"]


def test_system_stats_redacts_argv_without_mutating_payload():
    payload = {
        "system": {"argv": ["main.py", "--token", "secret"], "ram_free": 10},
        "devices": [{"name": "mps", "vram_free": 20}],
    }
    sanitized = benchmark.sanitized_stats(payload)
    assert "argv" not in sanitized["system"]
    assert payload["system"]["argv"][-1] == "secret"
    assert sanitized["devices"] == payload["devices"]


def test_qwen_quality_profile_uses_full_bf16_production_route():
    workflow = benchmark.load_workflow("qwen-quality")
    configured = benchmark.configure_workflow(
        workflow, prompt="Qwen benchmark", seed=999, prefix="benchmarks/qwen",
    )
    assert configured["37"]["inputs"]["unet_name"] == "qwen_image_2512_bf16.safetensors"
    assert configured["38"]["inputs"]["clip_name"] == "qwen_2.5_vl_7b.safetensors"
    assert configured["3"]["inputs"]["steps"] == 20
    assert configured["3"]["inputs"]["seed"] == 999
    assert configured["58"]["inputs"]["width"] == 1024
    assert configured["58"]["inputs"]["height"] == 576
    assert configured["81"]["inputs"]["text"] == "Qwen benchmark"


def test_output_validation_rejects_flat_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    output = tmp_path / "output" / "benchmarks"
    output.mkdir(parents=True)
    Image.new("RGB", (16, 16), (128, 127, 125)).save(output / "flat.png")
    varied = Image.new("RGB", (16, 16), (0, 0, 0))
    varied.putpixel((0, 0), (255, 255, 255))
    varied.save(output / "varied.png")

    flat = benchmark.validate_output_images([
        {"filename": "flat.png", "subfolder": "benchmarks", "type": "output"},
    ])
    nonflat = benchmark.validate_output_images([
        {"filename": "varied.png", "subfolder": "benchmarks", "type": "output"},
    ])

    assert flat[0]["valid"] is False
    assert flat[0]["channel_stddev"] == [0.0, 0.0, 0.0]
    assert nonflat[0]["valid"] is True
