from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kindred-avatar" / "backend"))

from governance import GovernanceError, GovernanceGate


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"accepted": True}


class RejectedResponse:
    status_code = 400
    reason = "Bad Request"

    def raise_for_status(self):
        raise requests.HTTPError("400 Client Error")

    def json(self):
        return {"code": "INVALID_EVENT_FIELD", "message": "schema_version exceeds max length 16"}


def configured_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("ADAPTIVE_BASE", "http://adaptive.test")
    monkeypatch.setenv("ADAPTIVE_API_KEY", "secret")
    gate = GovernanceGate(ROOT)
    gate.local_ledger = tmp_path / "audit.jsonl"
    return gate


def test_begin_appends_valid_envelope_before_dispatch(monkeypatch, tmp_path):
    gate = configured_gate(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr("governance.requests.post", lambda *args, **kwargs: calls.append((args, kwargs)) or Response())
    context = gate.begin({"6": {"inputs": {"text": "hello"}}}, "standard")
    payload = calls[0][1]["json"]
    assert payload["layer"] == "dispatch"
    assert payload["event"]["event_type"] == "comfy.workflow.start"
    assert payload["event"]["schema_version"] == "v1.1"
    assert len(payload["event"]["command_hash"]) == 64
    assert payload["envelope"]["principal_type"] == "human"
    gate.finish(context, success=True, detail="prompt_id=test")
    assert len(gate.local_ledger.read_text().splitlines()) == 2


def test_begin_fails_closed_without_adaptive_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv("ADAPTIVE_BASE", raising=False)
    monkeypatch.delenv("ADAPTIVE_API_KEY", raising=False)
    gate = GovernanceGate(ROOT)
    gate.local_ledger = tmp_path / "audit.jsonl"
    with pytest.raises(GovernanceError, match="failed closed"):
        gate.begin({"workflow": 1}, "standard")
    record = json.loads(gate.local_ledger.read_text())
    assert record["central_append"] == "failed"


def test_begin_preserves_structured_adaptive_rejection(monkeypatch, tmp_path):
    gate = configured_gate(monkeypatch, tmp_path)
    monkeypatch.setattr("governance.requests.post", lambda *args, **kwargs: RejectedResponse())
    with pytest.raises(GovernanceError, match="400 INVALID_EVENT_FIELD"):
        gate.begin({"workflow": 1}, "standard")
    record = json.loads(gate.local_ledger.read_text())
    assert "schema_version exceeds max length 16" in record["detail"]


def test_agent_dispatch_requires_lease(monkeypatch, tmp_path):
    gate = configured_gate(monkeypatch, tmp_path)
    monkeypatch.setenv("JAZZY_PRINCIPAL_TYPE", "agent")
    monkeypatch.delenv("JAZZY_LEASE_ID", raising=False)
    with pytest.raises(GovernanceError, match="requires JAZZY_LEASE_ID"):
        gate.begin({"workflow": 1}, "standard")


def test_agent_instance_also_requires_lease(monkeypatch, tmp_path):
    gate = configured_gate(monkeypatch, tmp_path)
    monkeypatch.setenv("JAZZY_AGENT_INSTANCE_ID", "agent-instance")
    monkeypatch.delenv("JAZZY_LEASE_ID", raising=False)
    with pytest.raises(GovernanceError, match="requires JAZZY_LEASE_ID"):
        gate.begin({"workflow": 1}, "standard")


def test_terminal_append_failure_is_marked_for_replay(monkeypatch, tmp_path):
    gate = configured_gate(monkeypatch, tmp_path)
    calls = 0

    def append(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return Response()
        raise OSError("adaptive became unavailable")

    monkeypatch.setattr("governance.requests.post", append)
    context = gate.begin({"workflow": 1}, "standard")
    gate.finish(context, success=True, detail="prompt_id=test")
    records = [json.loads(line) for line in gate.local_ledger.read_text().splitlines()]
    assert records[-1]["central_append"] == "replay_required"
    assert records[-1]["payload"]["event"]["event_type"] == "comfy.workflow.finish"
