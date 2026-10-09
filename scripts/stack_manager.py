#!/usr/bin/env python3
"""Start, stop, and inspect the local ComfyUI/Jazzy stack safely."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Callable
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
STATE_PATH = RUNTIME / "stack.json"
LOG_DIR = ROOT / "logs"
PYTHON = ROOT / ".venv" / "bin" / "python"


def load_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def environment() -> dict[str, str]:
    env = os.environ.copy()
    for key, value in load_dotenv().items():
        env.setdefault(key, value)
    return env


def read_state() -> dict:
    if not STATE_PATH.exists():
        return {"services": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"services": {}}


def write_state(state: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE_PATH)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def assert_port_free(host: str, port: int) -> None:
    with socket.socket() as probe:
        probe.settimeout(0.25)
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(f"{host}:{port} is already in use; no process was killed")


def http_health(url: str) -> bool:
    try:
        with urlopen(url, timeout=2) as response:
            return 200 <= response.status < 400
    except Exception:
        return False


async def _websocket_health(url: str, origin: str) -> bool:
    try:
        import websockets
        async with websockets.connect(url, origin=origin, open_timeout=2, close_timeout=1) as ws:
            await ws.send(json.dumps({"type": "ping"}))
            while True:
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if message.get("type") == "pong":
                    return True
    except Exception:
        return False


def websocket_health(url: str, origin: str) -> bool:
    return asyncio.run(_websocket_health(url, origin))


def services(env: dict[str, str]) -> list[dict]:
    comfy_host = env.get("COMFYUI_HOST", "127.0.0.1")
    comfy_port = int(env.get("COMFYUI_PORT", "8188"))
    jazzy_host = env.get("JAZZY_HOST", "127.0.0.1")
    jazzy_port = int(env.get("JAZZY_PORT", "8075"))
    frontend_host = env.get("JAZZY_FRONTEND_HOST", "127.0.0.1")
    frontend_port = int(env.get("JAZZY_FRONTEND_PORT", "8070"))
    token = env.get("JAZZY_WS_TOKEN", "")
    ws_url = f"ws://{jazzy_host}:{jazzy_port}" + (f"?{urlencode({'token': token})}" if token else "")
    origin = f"http://{frontend_host}:{frontend_port}"
    return [
        {"name": "comfyui", "host": comfy_host, "port": comfy_port,
         "cwd": ROOT / "ComfyUI-source",
         "command": [str(PYTHON), "main.py", "--listen", comfy_host, "--port", str(comfy_port)],
         "log": LOG_DIR / "comfyui-startup.log",
         "health": lambda: http_health(f"http://{comfy_host}:{comfy_port}/system_stats"), "timeout": 180},
        {"name": "jazzy", "host": jazzy_host, "port": jazzy_port,
         "cwd": ROOT / "kindred-avatar" / "backend", "command": [str(PYTHON), "server_voice.py"],
         "log": LOG_DIR / "jazzy-backend.log", "health": lambda: websocket_health(ws_url, origin), "timeout": 180},
        {"name": "frontend", "host": frontend_host, "port": frontend_port,
         "cwd": ROOT / "kindred-avatar" / "frontend",
         "command": [str(PYTHON), "-m", "http.server", str(frontend_port), "--bind", frontend_host],
         "log": LOG_DIR / "jazzy-frontend.log",
         "health": lambda: http_health(f"http://{frontend_host}:{frontend_port}/"), "timeout": 30},
    ]


def wait_ready(name: str, process: subprocess.Popen, check: Callable[[], bool], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"{name} exited before readiness (code {code})")
        if check():
            return
        time.sleep(1)
    raise RuntimeError(f"{name} did not become ready within {timeout}s")


def stop_service(name: str, entry: dict, grace: int = 15) -> None:
    pid = int(entry.get("pid", 0))
    if not pid or not alive(pid):
        return
    expected = entry.get("command", [])
    if expected:
        observed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False, capture_output=True, text=True,
        ).stdout
        signature = next((Path(part).name for part in expected if part.endswith(".py")), "http.server")
        if signature not in observed:
            raise RuntimeError(f"refusing to stop {name}: pid {pid} no longer matches saved command")
    print(f"stopping {name} (pid {pid})")
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(0.25)
    if alive(pid):
        os.killpg(pid, signal.SIGKILL)


def stop() -> int:
    state = read_state()
    for name, entry in reversed(list(state.get("services", {}).items())):
        stop_service(name, entry)
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    return 0


def start() -> int:
    if not PYTHON.exists():
        raise RuntimeError("missing .venv; run ./scripts/bootstrap.py first")
    current = read_state()
    running = [name for name, item in current.get("services", {}).items() if alive(int(item.get("pid", 0)))]
    if running:
        raise RuntimeError(f"stack state already has running services: {', '.join(running)}")
    env = environment()
    specs = services(env)
    for spec in specs:
        assert_port_free(spec["host"], spec["port"])
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    state = {"started_at": time.time(), "services": {}}
    started: list[tuple[str, subprocess.Popen]] = []
    try:
        for spec in specs:
            log_handle = spec["log"].open("a", encoding="utf-8")
            process = subprocess.Popen(spec["command"], cwd=spec["cwd"], env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            log_handle.close()
            started.append((spec["name"], process))
            state["services"][spec["name"]] = {
                "pid": process.pid, "host": spec["host"], "port": spec["port"],
                "log": str(spec["log"].relative_to(ROOT)), "command": spec["command"]}
            write_state(state)
            print(f"waiting for {spec['name']} on {spec['host']}:{spec['port']} (pid {process.pid})")
            wait_ready(spec["name"], process, spec["health"], spec["timeout"])
            print(f"ready: {spec['name']}")
    except Exception:
        for name, process in reversed(started):
            stop_service(name, {"pid": process.pid})
        if STATE_PATH.exists():
            STATE_PATH.unlink()
        raise
    print("stack ready: frontend http://127.0.0.1:8070 | Jazzy ws://127.0.0.1:8075 | ComfyUI http://127.0.0.1:8188")
    return 0


def status() -> int:
    env = environment()
    state = read_state()
    configured = {item["name"]: item for item in services(env)}
    failed = False
    for name in ("comfyui", "jazzy", "frontend"):
        entry = state.get("services", {}).get(name, {})
        pid = int(entry.get("pid", 0))
        is_alive = bool(pid and alive(pid))
        is_ready = bool(is_alive and configured[name]["health"]())
        print(f"{name}: {'ready' if is_ready else 'not ready'}" + (f" (pid {pid})" if pid else ""))
        failed |= not is_ready
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "restart", "status"))
    args = parser.parse_args()
    try:
        if args.action == "start": return start()
        if args.action == "stop": return stop()
        if args.action == "restart":
            stop()
            return start()
        return status()
    except (RuntimeError, OSError) as exc:
        print(f"stack: ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
