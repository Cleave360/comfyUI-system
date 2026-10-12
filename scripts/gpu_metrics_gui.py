#!/usr/bin/env python3
"""Serve a read-only live dashboard for governed GPU benchmark reports."""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "docs" / "gpu_metrics_dashboard.html"
SUPPORTED = {
    "kindred.comfyui.mps_recovery_policy.v1",
    "kindred.comfyui.mixed_image_video_soak.v1",
    "kindred.comfyui.isolated_image_video_soak.v1",
}


def gib(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(value / 1024**3, 4)


def normalize_memory(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for sample in samples:
        devices = sample.get("devices", [])
        device = devices[0] if devices and isinstance(devices[0], dict) else {}
        result.append({
            "elapsed_seconds": sample.get("elapsed_seconds"),
            "system_free_gib": gib(sample.get("ram_free_bytes")),
            "device_free_gib": gib(device.get("vram_free_bytes")),
        })
    return result


def normalize_report(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    schema = payload.get("schema_version")
    if schema not in SUPPORTED:
        return None
    method = payload.get("method", {})
    policy = method.get("policy")
    if policy == "fixed":
        policy = f"fixed {float(method.get('fixed_cooldown_seconds', 0)):g}s"
    elif policy == "adaptive":
        policy = "adaptive probe"
    elif schema == "kindred.comfyui.mixed_image_video_soak.v1":
        policy = f"same process {float(method.get('cooldown_seconds', 0)):g}s"
    else:
        policy = "isolated workers"
    runs = []
    for item in payload.get("runs", []):
        runs.append({
            "run": item.get("run"),
            "label": item.get("label", f"run {item.get('run', '?')}"),
            "profile": item.get("profile", "unknown"),
            "total_seconds": item.get("total_seconds"),
            "status": item.get("status"),
            "started_at": item.get("started_at"),
            "completed_at": item.get("completed_at"),
            "memory": normalize_memory(item.get("memory_samples", [])),
        })
    recovery = []
    for event in payload.get("recovery_events", []):
        recovery.append({
            "event_index": event.get("event_index"),
            "after_label": event.get("after_label"),
            "before_label": event.get("before_label"),
            "wait_seconds": event.get("wait_seconds"),
            "ready": event.get("ready"),
            "timed_out": event.get("timed_out"),
            "samples": [
                {
                    "wait_elapsed_seconds": sample.get("wait_elapsed_seconds"),
                    "latency_ms": sample.get("latency_ms"),
                    "ready": sample.get("ready"),
                    "system_available_gib": gib(sample.get("system_available_bytes")),
                    "torch_driver_gib": gib(sample.get("torch_driver_allocated_bytes")),
                    "thermal_state": sample.get("thermal_state"),
                    "thermal_boundary_reached": sample.get(
                        "thermal_performance_boundary_reached"
                    ),
                    "low_power_mode": sample.get("low_power_mode"),
                }
                for sample in event.get("samples", [])
            ],
        })
    live_samples = [
        {
            "wait_elapsed_seconds": sample.get("wait_elapsed_seconds"),
            "latency_ms": sample.get("latency_ms"),
            "ready": sample.get("ready"),
            "system_available_gib": gib(sample.get("system_available_bytes")),
            "torch_driver_gib": gib(sample.get("torch_driver_allocated_bytes")),
            "thermal_state": sample.get("thermal_state"),
            "thermal_boundary_reached": sample.get(
                "thermal_performance_boundary_reached"
            ),
            "low_power_mode": sample.get("low_power_mode"),
        }
        for sample in payload.get("live_recovery_samples", [])
    ]
    calibration = payload.get("calibration", {})
    analysis = dict(payload.get("analysis", {}))
    if "outputs_per_hour" not in analysis:
        measured = [item for item in runs if not str(item["label"]).startswith("warmup_")]
        if measured:
            first = measured[0]
            last = measured[-1]
            try:
                started = datetime.fromisoformat(first["started_at"])
                last_started = datetime.fromisoformat(last["started_at"])
                elapsed = (last_started - started).total_seconds() + float(last["total_seconds"])
            except (TypeError, ValueError):
                elapsed = sum(float(item["total_seconds"]) for item in measured)
            if elapsed > 0:
                analysis["outputs_per_hour"] = round(len(measured) * 3600 / elapsed, 3)
    return {
        "id": path.name,
        "path": str(path.relative_to(ROOT)),
        "schema_version": schema,
        "created_at": payload.get("created_at"),
        "policy": policy,
        "status": payload.get("status", {"phase": "complete", "completed": True}),
        "method": method,
        "calibration": {
            "median_ms": calibration.get("median_ms"),
            "p95_ms": calibration.get("p95_ms"),
            "threshold_ms": calibration.get("threshold_ms"),
        },
        "runs": runs,
        "recovery_events": recovery,
        "live_recovery_samples": live_samples,
        "analysis": analysis,
    }


class DashboardState:
    def __init__(self, reports: list[Path], report_dir: Path | None = None) -> None:
        self.reports = reports
        self.report_dir = report_dir
        self._cache: dict[Path, tuple[int, int, dict[str, Any] | None, str | None]] = {}
        self._lock = threading.Lock()

    def report_paths(self) -> list[Path]:
        values = set(self.reports)
        if self.report_dir is not None:
            values.update(self.report_dir.glob("*.json"))
        return sorted(values, reverse=True)

    def snapshot(self) -> dict[str, Any]:
        normalized: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        with self._lock:
            paths = self.report_paths()
            for path in paths:
                try:
                    stat = path.stat()
                    cached = self._cache.get(path)
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if cached is None or cached[:2] != signature:
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        item = normalize_report(path, payload)
                        cached = (*signature, item, None)
                        self._cache[path] = cached
                    item, error = cached[2], cached[3]
                    if item is not None:
                        normalized.append(item)
                    if error is not None:
                        errors.append({"path": str(path), "error": error})
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    message = str(exc)
                    self._cache[path] = (0, 0, None, message)
                    errors.append({"path": str(path), "error": message})
            current = set(paths)
            self._cache = {
                path: value for path, value in self._cache.items() if path in current
            }
        normalized.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return {"reports": normalized, "errors": errors}


def handler_factory(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path == "/":
                self.send_bytes(DASHBOARD.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/data":
                data = json.dumps(state.snapshot(), separators=(",", ":")).encode()
                self.send_bytes(data, "application/json; charset=utf-8")
                return
            if path == "/health":
                self.send_bytes(b'{"status":"ok"}', "application/json; charset=utf-8")
                return
            self.send_bytes(b"not found", "text/plain; charset=utf-8", 404)

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


def discover_reports(explicit: list[Path], report_dir: Path) -> list[Path]:
    report_root = (ROOT / "reports").resolve()
    values = [path.resolve() for path in explicit]
    if not values:
        values = sorted(report_dir.resolve().glob("*.json"), reverse=True)
    for path in values:
        if not path.is_relative_to(report_root):
            raise SystemExit(f"report must remain beneath {report_root}: {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument(
        "--report-dir", type=Path,
        default=ROOT / "reports" / "benchmarks",
    )
    args = parser.parse_args()
    if args.listen not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("GPU metrics GUI is restricted to loopback")
    reports = discover_reports(args.report, args.report_dir)
    dynamic_dir = None if args.report else args.report_dir.resolve()
    server = ThreadingHTTPServer(
        (args.listen, args.port),
        handler_factory(DashboardState(reports, dynamic_dir)),
    )
    print(f"GPU metrics GUI: http://{args.listen}:{args.port}")
    if dynamic_dir is None:
        print(f"Watching {len(reports)} report paths")
    else:
        print(f"Watching report directory: {dynamic_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
