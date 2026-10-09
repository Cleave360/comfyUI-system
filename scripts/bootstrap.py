#!/usr/bin/env python3
"""Idempotently bootstrap the pinned Kindred ComfyUI workspace."""

from __future__ import annotations

import argparse
import filecmp
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "system.toml"


class BootstrapError(RuntimeError):
    pass


def run(*args: str, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else ""


def load_manifest() -> dict:
    with MANIFEST_PATH.open("rb") as handle:
        manifest = tomllib.load(handle)
    if manifest.get("schema_version") != "comfyui.system.v1":
        raise BootstrapError("unsupported config/system.toml schema")
    return manifest


def git_value(repo: Path, *args: str) -> str:
    return run("git", *args, cwd=repo, capture=True)


def ensure_checkout(path: Path, spec: dict, *, check: bool, update: bool) -> bool:
    url = spec["url"]
    revision = spec["revision"]
    created = False
    if not path.exists():
        if check:
            raise BootstrapError(f"missing checkout: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", url, str(path), cwd=ROOT)
        created = True

    if not (path / ".git").exists():
        raise BootstrapError(f"existing path is not a Git checkout: {path}")

    origin = git_value(path, "remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git").lower() != url.rstrip("/").removesuffix(".git").lower():
        raise BootstrapError(f"unexpected origin for {path}: {origin}")

    head = git_value(path, "rev-parse", "HEAD")
    if head == revision:
        return created
    if check or (not update and not created):
        raise BootstrapError(f"{path} is at {head}; expected {revision}")
    if not created and git_value(path, "status", "--porcelain"):
        raise BootstrapError(f"refusing to update dirty checkout: {path}")
    run("git", "fetch", "origin", revision, cwd=path)
    run("git", "checkout", "--detach", revision, cwd=path)
    return created


def directory_equivalent(left: Path, right: Path) -> bool:
    def inventory(root: Path) -> dict[Path, Path]:
        return {
            path.relative_to(root): path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.name != ".DS_Store"
        }

    left_files = inventory(left)
    right_files = inventory(right)
    if left_files.keys() != right_files.keys():
        return False
    return all(filecmp.cmp(path, right_files[relative], shallow=False) for relative, path in left_files.items())


def safe_replace_with_link(
    link_path: Path,
    target: Path,
    *,
    check: bool,
    repair: bool,
    fresh_checkout: bool = False,
) -> None:
    expected = Path(os.path.relpath(target, link_path.parent))
    if link_path.is_symlink():
        actual = Path(os.readlink(link_path))
        if actual == expected or link_path.resolve() == target.resolve():
            return
        if check or not repair:
            raise BootstrapError(f"wrong symlink {link_path} -> {actual}; expected {expected}")
        link_path.unlink()
    elif link_path.exists():
        equivalent = link_path.is_dir() and target.is_dir() and directory_equivalent(link_path, target)
        empty_or_placeholders = link_path.is_dir() and all(
            child.name.startswith("put_") or child.name == ".DS_Store" for child in link_path.iterdir()
        )
        if not (equivalent or empty_or_placeholders or fresh_checkout):
            raise BootstrapError(f"refusing to replace non-equivalent path: {link_path}")
        if check or not repair:
            raise BootstrapError(f"{link_path} should be a symlink; rerun with --repair-links")
        shutil.rmtree(link_path)
    elif check:
        raise BootstrapError(f"missing symlink: {link_path}")

    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(expected, target_is_directory=True)


def ensure_runtime_links(manifest: dict, *, check: bool, repair: bool, fresh_checkout: bool = False) -> None:
    comfy = ROOT / "ComfyUI-source"
    for source_name, root_relative in manifest["links"].items():
        target = ROOT / root_relative
        if not check:
            target.mkdir(parents=True, exist_ok=True)
        safe_replace_with_link(
            comfy / source_name,
            target,
            check=check,
            repair=repair,
            fresh_checkout=fresh_checkout,
        )

    for node_name in manifest["custom_nodes"]["owned"]:
        source = ROOT / "custom_nodes" / node_name
        if not source.is_dir():
            raise BootstrapError(f"missing tracked custom node: {source}")
        safe_replace_with_link(
            comfy / "custom_nodes" / node_name,
            source,
            check=check,
            repair=repair,
        )


def ensure_environment(*, check: bool, install_deps: bool) -> None:
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        if check:
            raise BootstrapError("missing .venv")
        interpreter = shutil.which("python3.12")
        if not interpreter:
            raise BootstrapError("python3.12 is required")
        run(interpreter, "-m", "venv", str(ROOT / ".venv"), cwd=ROOT)
    version = run(str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')", capture=True)
    if version != "3.12":
        raise BootstrapError(f"canonical environment must use Python 3.12, found {version}")
    if install_deps and not check:
        run(str(python), "-m", "pip", "install", "--upgrade", "pip", cwd=ROOT)
        run(str(python), "-m", "pip", "install", "-r", "requirements-lock.txt", cwd=ROOT)
    run(
        str(python),
        "-c",
        "import pytest, requests, websockets, xml.parsers.expat",
        cwd=ROOT,
    )
    run(str(python), "-m", "pip", "check", cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify without changing the workspace")
    parser.add_argument("--update", action="store_true", help="move clean checkouts to pinned revisions")
    parser.add_argument("--repair-links", action="store_true", help="replace safe equivalent directories with links")
    parser.add_argument("--skip-deps", action="store_true", help="do not install Python dependencies")
    args = parser.parse_args()

    try:
        manifest = load_manifest()
        comfy_created = ensure_checkout(
            ROOT / "ComfyUI-source",
            manifest["upstream"]["comfyui"],
            check=args.check,
            update=args.update,
        )
        manager_path = ROOT / "ComfyUI-source" / "custom_nodes" / "ComfyUI-Manager"
        ensure_checkout(manager_path, manifest["upstream"]["comfyui_manager"], check=args.check, update=args.update)
        ensure_runtime_links(
            manifest,
            check=args.check,
            repair=args.repair_links or comfy_created,
            fresh_checkout=comfy_created,
        )
        ensure_environment(check=args.check, install_deps=not args.skip_deps)
    except (BootstrapError, subprocess.CalledProcessError) as exc:
        print(f"bootstrap: ERROR: {exc}", file=sys.stderr)
        return 1
    print("bootstrap: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
