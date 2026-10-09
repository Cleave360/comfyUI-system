#!/usr/bin/env python3
"""Create or verify a transparent inventory of local model artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOCK = ROOT / "config" / "models.lock.json"
METADATA = ROOT / "config" / "model_metadata.toml"
MODEL_SUFFIXES = {".bin", ".ckpt", ".gguf", ".onnx", ".pt", ".pth", ".safetensors", ".sft"}


def precision_for(name: str) -> str:
    lowered = name.lower()
    for token in ("fp8_e4m3fn", "fp8", "bf16", "fp16", "int8", "int4"):
        if token in lowered:
            return token
    return "unknown"


def mps_status(precision: str) -> str:
    if precision.startswith("fp8"):
        return "unsupported_or_operator_dependent"
    if precision in {"fp16", "bf16"}:
        return "candidate_not_workflow_verified"
    return "unknown_not_verified"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()


def scan(with_hashes: bool) -> dict:
    catalog = {}
    if METADATA.exists():
        with METADATA.open("rb") as handle:
            catalog = tomllib.load(handle).get("models", {})
    previous = {}
    if with_hashes and LOCK.exists():
        try:
            previous = {
                item["path"]: item
                for item in json.loads(LOCK.read_text(encoding="utf-8")).get("artifacts", [])
            }
        except (json.JSONDecodeError, KeyError):
            previous = {}
    artifacts = []
    if MODELS.exists():
        for path in sorted(MODELS.rglob("*")):
            if not path.is_file() or ".cache" in path.parts or path.suffix.lower() not in MODEL_SUFFIXES:
                continue
            stat = path.stat()
            precision = precision_for(path.name)
            relative_path = path.relative_to(ROOT).as_posix()
            declared = catalog.get(relative_path, {})
            old = previous.get(relative_path, {})
            reusable_hash = (
                old.get("size_bytes") == stat.st_size
                and old.get("mtime_ns") == stat.st_mtime_ns
                and old.get("sha256")
            )
            artifact = {
                "path": relative_path,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": (reusable_hash or digest(path)) if with_hashes else None,
                "source": declared.get("source", "unknown_local_artifact"),
                "license": declared.get("license", "unknown_review_required"),
                "notes": declared.get("notes", ""),
                "precision": precision,
                "mps_compatibility": mps_status(precision),
            }
            artifacts.append(artifact)
    incomplete_downloads = []
    if MODELS.exists():
        incomplete_downloads = [
            {"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size}
            for path in sorted(MODELS.rglob("*.incomplete")) if path.is_file()
        ]
    return {
        "schema_version": "comfyui.models.lock.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hashes_complete": with_hashes,
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(item["size_bytes"] for item in artifacts),
        "incomplete_downloads": incomplete_downloads,
        "artifacts": artifacts,
    }


def write_manifest(with_hashes: bool) -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(scan(with_hashes), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LOCK.relative_to(ROOT)}")
    return 0


def check_manifest(verify_hashes: bool) -> int:
    if not LOCK.exists():
        print("model manifest is missing", file=sys.stderr)
        return 1
    locked = json.loads(LOCK.read_text(encoding="utf-8"))
    errors = []
    for artifact in locked.get("artifacts", []):
        path = ROOT / artifact["path"]
        if not path.is_file():
            errors.append(f"missing: {artifact['path']}")
            continue
        if path.stat().st_size != artifact["size_bytes"]:
            errors.append(f"size mismatch: {artifact['path']}")
        expected = artifact.get("sha256")
        if verify_hashes and expected and digest(path) != expected:
            errors.append(f"checksum mismatch: {artifact['path']}")
    current_paths = {
        path.relative_to(ROOT).as_posix() for path in MODELS.rglob("*")
        if path.is_file() and ".cache" not in path.parts and path.suffix.lower() in MODEL_SUFFIXES
    }
    locked_paths = {item["path"] for item in locked.get("artifacts", [])}
    errors.extend(f"untracked model: {path}" for path in sorted(current_paths - locked_paths))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"model manifest PASS ({len(locked_paths)} artifacts)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--hash", action="store_true", dest="with_hashes")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--hash", action="store_true", dest="verify_hashes")
    args = parser.parse_args()
    return write_manifest(args.with_hashes) if args.action == "scan" else check_manifest(args.verify_hashes)


if __name__ == "__main__":
    raise SystemExit(main())
