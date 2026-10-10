from __future__ import annotations

import json
from pathlib import Path
import sys


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
