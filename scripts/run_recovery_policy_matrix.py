#!/usr/bin/env python3
"""Run the four predeclared MPS recovery policies against fresh servers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SERVER_PYTHON = ROOT / "reports" / "venvs" / "qwen-mlx-torch212" / "bin" / "python"
CLIENT_PYTHON = ROOT / ".venv" / "bin" / "python"
POLICIES = (
    ("fixed-2s", "fixed", 2.0),
    ("adaptive", "adaptive", 0.0),
    ("fixed-30s", "fixed", 30.0),
    ("fixed-10s", "fixed", 10.0),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def wait_ready(base_url: str, process: subprocess.Popen, timeout: float = 300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"isolated server exited early with {process.returncode}")
        try:
            with urlopen(f"{base_url}/system_stats", timeout=5) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"isolated server was not ready within {timeout:g}s")


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8288)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--wan-steps", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "benchmarks")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "reports" / "runtime")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "reports" / "profiles")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    if args.pairs < 2:
        raise SystemExit("--pairs must be at least two")
    for executable in (SERVER_PYTHON, CLIENT_PYTHON):
        if not executable.is_file():
            raise SystemExit(f"required Python is missing: {executable}")

    output_dir = args.output_dir.resolve()
    runtime_root = (args.runtime_dir / f"recovery-policy-{args.run_id}").resolve()
    profile_dir = args.profile_dir.resolve()
    matrix_path = output_dir / f"recovery_policy_matrix_{args.run_id}.json"
    comparison_path = output_dir / f"recovery_policy_comparison_{args.run_id}.json"
    base_url = f"http://127.0.0.1:{args.port}"
    matrix = {
        "schema_version": "kindred.comfyui.mps_recovery_matrix.v1",
        "created_at": utc_now(),
        "status": {"phase": "starting", "completed": False},
        "method": {
            "policy_order": [item[0] for item in POLICIES],
            "fresh_server_per_policy": True,
            "order_is_a_screening_confound": True,
            "pairs": args.pairs,
            "qwen_steps": args.qwen_steps,
            "wan_steps": args.wan_steps,
            "port": args.port,
            "production_modified": False,
        },
        "policies": [],
    }
    atomic_write(matrix_path, matrix)
    report_paths: list[Path] = []

    for slug, policy, cooldown in POLICIES:
        report_path = output_dir / f"recovery_policy_{args.run_id}_{slug}.json"
        server_report = profile_dir / f"recovery_policy_{args.run_id}_{slug}_server.json"
        log_path = runtime_root / f"{slug}.log"
        user_dir = runtime_root / slug / "user"
        user_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        phase = {
            "name": slug,
            "policy": policy,
            "cooldown_seconds": cooldown,
            "report": str(report_path),
            "server_report": str(server_report),
            "log": str(log_path),
            "started_at": utc_now(),
            "status": "starting_server",
        }
        matrix["policies"].append(phase)
        matrix["status"] = {"phase": "policy", "current": slug, "completed": False}
        atomic_write(matrix_path, matrix)
        environment = os.environ.copy()
        environment.update({
            "MLX_ENABLE_TF32": "0",
            "KINDRED_WAN_MLX_VAE_RUN_MODES": "mlx",
            "KINDRED_WAN_MPS_CLEANUP_POLICY": "torch_empty_cache",
            "KINDRED_TORCH_EMPTY_CACHE_ALL_VAE": "1",
            "KINDRED_WAN_MLX_VAE_REPORT": str(server_report),
        })
        server_command = [
            str(SERVER_PYTHON), str(ROOT / "scripts" / "wan_mlx_vae_server.py"),
            "--listen", "127.0.0.1", "--port", str(args.port),
            "--disable-all-custom-nodes", "--disable-auto-launch",
            "--user-directory", str(user_dir),
        ]
        process = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    server_command, cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
                wait_ready(base_url, process)
                phase["status"] = "benchmarking"
                phase["server_pid"] = process.pid
                atomic_write(matrix_path, matrix)
                command = [
                    str(CLIENT_PYTHON),
                    str(ROOT / "scripts" / "benchmark_recovery_policy.py"),
                    "--base-url", base_url,
                    "--policy", policy,
                    "--cooldown-seconds", str(cooldown),
                    "--pairs", str(args.pairs),
                    "--qwen-steps", str(args.qwen_steps),
                    "--wan-steps", str(args.wan_steps),
                    "--output", str(report_path),
                ]
                completed = subprocess.run(command, cwd=ROOT, env=environment)
                phase["client_returncode"] = completed.returncode
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"{slug} benchmark exited with {completed.returncode}"
                    )
                phase["status"] = "complete"
                phase["completed_at"] = utc_now()
                report_paths.append(report_path)
        except Exception as exc:
            phase["status"] = "failed"
            phase["error"] = f"{type(exc).__name__}: {exc}"
            matrix["status"] = {"phase": "failed", "current": slug, "completed": False}
            atomic_write(matrix_path, matrix)
            raise
        finally:
            if process is not None:
                stop_server(process)
            atomic_write(matrix_path, matrix)

    compare_command = [
        str(CLIENT_PYTHON), str(ROOT / "scripts" / "compare_recovery_policies.py"),
        *[str(path) for path in report_paths], "--output", str(comparison_path),
    ]
    comparison = subprocess.run(compare_command, cwd=ROOT)
    matrix["comparison"] = {
        "path": str(comparison_path),
        "returncode": comparison.returncode,
        "gate_passed": comparison.returncode == 0,
    }
    matrix["status"] = {"phase": "complete", "completed": True}
    matrix["completed_at"] = utc_now()
    atomic_write(matrix_path, matrix)
    print(f"matrix: {matrix_path}")
    print(f"comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
