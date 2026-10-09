from __future__ import annotations

import json
from pathlib import Path
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "kindred-avatar" / "backend"))

import bootstrap
import model_manifest
import stack_manager
from network_policy import authenticated_path, validate_binding


def test_system_manifest_and_reserved_ports():
    manifest = bootstrap.load_manifest()
    assert manifest["system"]["python"] == "3.12"
    assert manifest["runtime"]["jazzy_port"] == 8075
    assert 8765 not in {value for key, value in manifest["runtime"].items() if key.endswith("_port")}
    assert len(manifest["upstream"]["comfyui"]["revision"]) == 40
    assert len(manifest["upstream"]["comfyui_manager"]["revision"]) == 40


def test_governance_policy_is_enforced():
    with (ROOT / "governance.policy.toml").open("rb") as handle:
        policy = tomllib.load(handle)
    assert policy["execution"] == {
        "governed_execution_required": True,
        "allow_raw_shell_in_dev": False,
        "require_envelope": True,
        "require_audit_append": True,
    }
    assert policy["leases"]["require_lease_for_agent_exec"] is True
    assert policy["leases"]["require_lease_for_service_exec"] is True


def test_model_manifest_matches_local_inventory():
    locked = json.loads((ROOT / "config" / "models.lock.json").read_text())
    if model_manifest.MODELS.exists():
        assert model_manifest.check_manifest(False) == 0
    else:
        assert locked["schema_version"] == "comfyui.models.lock.v1"
        assert locked["artifact_count"] == len(locked["artifacts"])
    assert all(item["source"] and item["license"] for item in locked["artifacts"])
    assert isinstance(locked["incomplete_downloads"], list)


def test_port_probe_refuses_an_existing_listener(monkeypatch):
    class OccupiedSocket:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def settimeout(self, timeout): return None
        def connect_ex(self, address): return 0

    monkeypatch.setattr(stack_manager.socket, "socket", OccupiedSocket)
    with pytest.raises(RuntimeError, match="already in use"):
        stack_manager.assert_port_free("127.0.0.1", 8075)


def test_runtime_defaults_are_loopback_and_do_not_use_adaptive_port():
    specs = stack_manager.services({})
    assert {item["port"] for item in specs} == {8070, 8075, 8188}
    assert all(item["host"] == "127.0.0.1" for item in specs)


def test_non_loopback_requires_token_and_token_comparison():
    with pytest.raises(RuntimeError, match="JAZZY_WS_TOKEN"):
        validate_binding("0.0.0.0", "")
    validate_binding("0.0.0.0", "strong-token")
    assert authenticated_path("/?token=strong-token", "strong-token")
    assert not authenticated_path("/?token=wrong", "strong-token")
