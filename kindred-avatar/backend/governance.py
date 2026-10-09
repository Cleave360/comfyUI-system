"""Fail-closed Adaptive audit gate for ComfyUI workflow dispatches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any
from uuid import uuid4

import requests


HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class GovernanceError(RuntimeError):
    """Raised when a governed side effect cannot be admitted."""


@dataclass(frozen=True)
class DispatchContext:
    request_id: str
    envelope: dict[str, Any]
    command_hash: str
    command_preview: str
    working_directory: str


class GovernanceGate:
    """Bind workflow execution to a context envelope and Adaptive audit record."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.governance = self._load_toml(self.repo_root / "governance.toml")
        self.policy = self._load_toml(self.repo_root / "governance.policy.toml")
        execution = self.policy.get("execution", {})
        required_controls = ("governed_execution_required", "require_envelope", "require_audit_append")
        if self.policy.get("schema_version") != "governance.policy.v1" or not all(
            execution.get(control) is True for control in required_controls
        ):
            raise GovernanceError("repository policy does not require governed envelope and audit enforcement")
        adaptive = self.governance.get("adaptive", {})
        self.base_url = os.getenv(adaptive.get("base_url_env", "ADAPTIVE_BASE"), "").rstrip("/")
        self.api_key = os.getenv(adaptive.get("api_key_env", "ADAPTIVE_API_KEY"), "")
        self.timeout = float(os.getenv("JAZZY_AUDIT_TIMEOUT_SECONDS", "5"))
        self.local_ledger = self.repo_root / ".kindred" / "audit" / "comfyui_workflow_dispatch.jsonl"

    @staticmethod
    def _load_toml(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    def _envelope(self, run_id: str) -> dict[str, str]:
        principal_type = os.getenv("JAZZY_PRINCIPAL_TYPE", "human").strip().lower()
        if principal_type not in {"human", "agent", "service"}:
            raise GovernanceError(f"invalid JAZZY_PRINCIPAL_TYPE: {principal_type}")
        envelope = {
            "envelope_version": "v1",
            "tenant_id": self.governance["org"]["tenant_id"],
            "workspace_id": self.governance["repo"]["workspace_id"],
            "project_id": self.governance["repo"]["project_id"],
            "run_id": run_id,
            "principal_id": os.getenv("JAZZY_PRINCIPAL_ID", "local-operator"),
            "principal_type": principal_type,
            "ui_instance_id": self.governance["kindred"]["default_ui_instance_id"],
        }
        optional = {
            "lease_id": os.getenv("JAZZY_LEASE_ID", ""),
            "agent_instance_id": os.getenv("JAZZY_AGENT_INSTANCE_ID", ""),
            "agent_key_id": os.getenv("JAZZY_AGENT_KEY_ID", ""),
        }
        envelope.update({key: value for key, value in optional.items() if value})
        if (principal_type in {"agent", "service"} or envelope.get("agent_instance_id")) and not envelope.get("lease_id"):
            raise GovernanceError("agent or service workflow dispatch requires JAZZY_LEASE_ID")
        return envelope

    @staticmethod
    def _canonical_hash(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _local_append(self, payload: dict[str, Any], central: str, detail: str = "") -> None:
        self.local_ledger.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "central_append": central,
            "detail": detail[:500],
            "payload": payload,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        with self.local_ledger.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_central(self, payload: dict[str, Any]) -> None:
        if not self.base_url or not self.api_key:
            raise GovernanceError("ADAPTIVE_BASE and ADAPTIVE_API_KEY are required for governed dispatch")
        response = requests.post(
            f"{self.base_url}/v1/audit/append",
            headers={"x-api-key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            body = {}
        if isinstance(body, dict) and body.get("accepted") is False:
            raise GovernanceError(f"Adaptive rejected audit append: {body}")

    def _payload(self, context: DispatchContext, event_type: str, status: str, decision: str, detail: str = "") -> dict[str, Any]:
        if not HEX_64.fullmatch(context.command_hash):
            raise GovernanceError("invalid command hash")
        event = {
            "schema_version": "execution.audit.v1",
            "event_id": str(uuid4()),
            "event_ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "source_system": "comfyui-system",
            "executor": "jazzy-websocket",
            "command_hash": context.command_hash,
            "command_preview": context.command_preview[:200],
            "working_directory": context.working_directory,
            "policy_decision": decision,
            "status": status,
        }
        if detail:
            event["payload"] = {"detail": detail[:500]}
        return {"request_id": context.request_id, "layer": "dispatch", "envelope": context.envelope, "event": event}

    def begin(self, workflow: dict[str, Any], workflow_type: str) -> DispatchContext:
        run_id = str(uuid4())
        context = DispatchContext(
            request_id=f"comfy-{uuid4()}",
            envelope=self._envelope(run_id),
            command_hash=self._canonical_hash(workflow),
            command_preview=f"ComfyUI workflow dispatch: {workflow_type}",
            working_directory=str(self.repo_root),
        )
        payload = self._payload(context, "comfy.workflow.start", "success", "allow")
        try:
            self._append_central(payload)
        except Exception as exc:
            self._local_append(payload, "failed", str(exc))
            raise GovernanceError(f"governance admission failed closed: {exc}") from exc
        try:
            self._local_append(payload, "accepted")
        except OSError:
            pass
        return context

    def finish(self, context: DispatchContext, *, success: bool, detail: str = "") -> None:
        event_type = "comfy.workflow.finish" if success else "comfy.workflow.error"
        status = "success" if success else "error"
        payload = self._payload(context, event_type, status, "allow", detail)
        try:
            self._append_central(payload)
        except Exception as exc:
            try:
                self._local_append(payload, "replay_required", str(exc))
            except Exception as local_exc:
                raise GovernanceError(
                    f"terminal audit failed and durable replay evidence could not be written: {local_exc}"
                ) from exc
            return
        try:
            self._local_append(payload, "accepted")
        except OSError:
            pass
