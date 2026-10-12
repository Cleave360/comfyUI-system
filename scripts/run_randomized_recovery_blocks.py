#!/usr/bin/env python3
"""Run the predeclared balanced randomized-block MPS recovery experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import subprocess

from run_recovery_policy_matrix import (
    CLIENT_PYTHON,
    ROOT,
    SERVER_PYTHON,
    atomic_write,
    stop_server,
    utc_now,
    wait_ready,
)


DEFAULT_SEED = 20261012
BLOCKS = 6
PAIRS_PER_ARM = 2


def balanced_schedule(seed: int) -> list[str]:
    schedule = ["fixed-adaptive"] * 3 + ["adaptive-fixed"] * 3
    random.Random(seed).shuffle(schedule)
    return schedule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8288)
    parser.add_argument("--qwen-steps", type=int, default=20)
    parser.add_argument("--wan-steps", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "benchmarks")
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "reports" / "runtime")
    parser.add_argument("--profile-dir", type=Path, default=ROOT / "reports" / "profiles")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    for executable in (SERVER_PYTHON, CLIENT_PYTHON):
        if not executable.is_file():
            raise SystemExit(f"required Python is missing: {executable}")

    schedule = balanced_schedule(args.random_seed)
    output_dir = args.output_dir.resolve()
    runtime_root = (args.runtime_dir / f"recovery-randomized-{args.run_id}").resolve()
    profile_dir = args.profile_dir.resolve()
    manifest_path = output_dir / f"recovery_randomized_blocks_{args.run_id}.json"
    comparison_path = output_dir / f"recovery_randomized_blocks_comparison_{args.run_id}.json"
    base_url = f"http://127.0.0.1:{args.port}"
    manifest = {
        "schema_version": "kindred.comfyui.mps_recovery_randomized_blocks.v1",
        "created_at": utc_now(),
        "status": {"phase": "starting", "completed": False},
        "method": {
            "design": "balanced randomized paired blocks",
            "blocks": BLOCKS,
            "pairs_per_arm": PAIRS_PER_ARM,
            "random_seed": args.random_seed,
            "sequence_schedule": schedule,
            "fresh_server_per_arm": True,
            "warmups_per_arm": 2,
            "qwen_steps": args.qwen_steps,
            "wan_steps": args.wan_steps,
            "port": args.port,
            "production_modified": False,
        },
        "arms": [],
    }
    atomic_write(manifest_path, manifest)

    for block, sequence in enumerate(schedule, start=1):
        policies = sequence.split("-")
        for position, policy in enumerate(policies, start=1):
            cooldown = 10.0 if policy == "fixed" else 0.0
            slug = f"block{block}-{policy}"
            report_path = output_dir / f"recovery_randomized_blocks_{args.run_id}_{slug}.json"
            server_report = profile_dir / f"recovery_randomized_blocks_{args.run_id}_{slug}_server.json"
            log_path = runtime_root / f"{slug}.log"
            user_dir = runtime_root / slug / "user"
            user_dir.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            arm = {
                "name": slug,
                "block": block,
                "sequence": sequence,
                "position": position,
                "policy": policy,
                "cooldown_seconds": cooldown,
                "report": str(report_path),
                "server_report": str(server_report),
                "log": str(log_path),
                "started_at": utc_now(),
                "status": "starting_server",
            }
            manifest["arms"].append(arm)
            manifest["status"] = {"phase": "arm", "current": slug, "completed": False}
            atomic_write(manifest_path, manifest)
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
                    arm["status"] = "benchmarking"
                    arm["server_pid"] = process.pid
                    atomic_write(manifest_path, manifest)
                    command = [
                        str(CLIENT_PYTHON), str(ROOT / "scripts" / "benchmark_recovery_policy.py"),
                        "--base-url", base_url,
                        "--policy", policy,
                        "--cooldown-seconds", str(cooldown),
                        "--pairs", str(PAIRS_PER_ARM),
                        "--qwen-steps", str(args.qwen_steps),
                        "--wan-steps", str(args.wan_steps),
                        "--arm-label", slug,
                        "--counterbalance-round", str(block),
                        "--counterbalance-position", str(position),
                        "--output", str(report_path),
                    ]
                    completed = subprocess.run(command, cwd=ROOT, env=environment)
                    arm["client_returncode"] = completed.returncode
                    if completed.returncode != 0:
                        raise RuntimeError(f"{slug} benchmark exited with {completed.returncode}")
                    arm["status"] = "complete"
                    arm["completed_at"] = utc_now()
            except Exception as exc:
                arm["status"] = "failed"
                arm["error"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = {"phase": "failed", "current": slug, "completed": False}
                atomic_write(manifest_path, manifest)
                raise
            finally:
                if process is not None:
                    stop_server(process)
                atomic_write(manifest_path, manifest)

    compare_command = [
        str(CLIENT_PYTHON), str(ROOT / "scripts" / "compare_randomized_recovery_blocks.py"),
        "--manifest", str(manifest_path), "--output", str(comparison_path),
    ]
    comparison = subprocess.run(compare_command, cwd=ROOT)
    manifest["comparison"] = {
        "path": str(comparison_path),
        "returncode": comparison.returncode,
        "gate_passed": comparison.returncode == 0,
    }
    manifest["status"] = {"phase": "complete", "completed": True}
    manifest["completed_at"] = utc_now()
    atomic_write(manifest_path, manifest)
    print(f"manifest: {manifest_path}")
    print(f"comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
