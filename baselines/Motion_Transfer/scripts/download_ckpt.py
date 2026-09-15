#!/usr/bin/env python3
"""Download and verify assets needed by MotionTransfer inference."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from torch.hub import download_url_to_file
from tqdm import tqdm

from model_assets import (
    REPO_ROOT,
    expand_names,
    load_manifest,
    torch_checkpoint_path,
    verify_file,
)


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def _git_has_commit(repository: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repository,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def prepare_common_source(spec: dict[str, Any], *, force: bool) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download CommonSource.")

    target = REPO_ROOT / spec["local_dir"]
    revision = spec["revision"]
    if target.exists() and not (target / ".git").is_dir():
        raise RuntimeError(
            f"CommonSource target exists but is not a Git checkout: {target}"
        )

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"Cloning CommonSource from {spec['repo_url']}")
        _run_git(["clone", spec["repo_url"], str(target)])

    remote = _run_git(["config", "--get", "remote.origin.url"], cwd=target)
    actual_origin = remote.rstrip("/").removesuffix(".git")
    expected_origin = spec["repo_url"].rstrip("/").removesuffix(".git")
    if actual_origin != expected_origin:
        raise RuntimeError(f"Unexpected CommonSource origin: {remote}")

    head = _run_git(["rev-parse", "HEAD"], cwd=target)
    if head != revision:
        tqdm.write(f"Checking out CommonSource revision {revision}")
        if not _git_has_commit(target, revision):
            _run_git(["fetch", "origin", revision], cwd=target)
        checkout_args = ["checkout", "--detach"]
        if force:
            checkout_args.append("--force")
        _run_git([*checkout_args, revision], cwd=target)

    failures = []
    for relative_path, file_spec in spec["files"].items():
        ok, message = verify_file(
            target / relative_path, file_spec, verify_hash=True, progress=False
        )
        if not ok:
            failures.append(message)

    if failures and force:
        _run_git(["checkout", "--force", revision], cwd=target)
        failures = []
        for relative_path, file_spec in spec["files"].items():
            ok, message = verify_file(
                target / relative_path, file_spec, verify_hash=True, progress=False
            )
            if not ok:
                failures.append(message)

    if failures:
        raise RuntimeError("CommonSource integrity check failed:\n  " + "\n  ".join(failures))
    tqdm.write(f"Ready: {target} @ {revision}")


def download_hf_file(name: str, spec: dict[str, Any], *, force: bool) -> None:
    destination = REPO_ROOT / spec["local_path"]
    if destination.exists():
        ok, message = verify_file(destination, spec, progress=True)
        if ok:
            tqdm.write(message)
            return
        if not force:
            raise RuntimeError(f"{message}\nRe-run with --force to replace this file.")
        destination.unlink()

    local_dir = REPO_ROOT / spec["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    tqdm.write(
        f"Downloading {name}: {spec['repo_id']}::{spec['filename']} "
        f"@ {spec['revision']}"
    )
    path = Path(
        hf_hub_download(
            repo_id=spec["repo_id"],
            filename=spec["filename"],
            repo_type="model",
            revision=spec["revision"],
            local_dir=str(local_dir),
            force_download=force,
        )
    )
    ok, message = verify_file(path, spec, progress=True)
    if not ok:
        raise RuntimeError(message)
    if path.resolve() != destination.resolve():
        raise RuntimeError(f"Unexpected download path: expected {destination}, got {path}")
    tqdm.write(message)


def download_raft(spec: dict[str, Any], *, force: bool) -> None:
    destination = torch_checkpoint_path(spec["cache_filename"])
    if destination.exists():
        ok, message = verify_file(destination, spec, progress=True)
        if ok:
            tqdm.write(message)
            return
        if not force:
            raise RuntimeError(f"{message}\nRe-run with --force to replace this file.")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Downloading RAFT-Large from {spec['url']}")
    download_url_to_file(
        spec["url"],
        str(destination),
        hash_prefix=spec["sha256"],
        progress=True,
    )
    ok, message = verify_file(destination, spec, progress=True)
    if not ok:
        raise RuntimeError(message)
    tqdm.write(message)


def download_hf_snapshot(name: str, spec: dict[str, Any], *, force: bool) -> None:
    required_files = list(spec["files"])
    tqdm.write(
        f"Downloading {name}: {spec['repo_id']} @ {spec['revision']} "
        f"({len(required_files)} files)"
    )
    snapshot_path = Path(
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="model",
            revision=spec["revision"],
            allow_patterns=required_files,
            force_download=force,
            max_workers=4,
        )
    )

    failures = []
    for relative_path, file_spec in spec["files"].items():
        ok, message = verify_file(
            snapshot_path / relative_path,
            file_spec,
            verify_hash=True,
            progress=True,
        )
        if ok:
            tqdm.write(message)
        else:
            failures.append(message)
    if failures:
        raise RuntimeError(f"{name} integrity check failed:\n  " + "\n  ".join(failures))
    tqdm.write(f"Ready: {snapshot_path}")


def main() -> None:
    manifest = load_manifest()
    valid_names = sorted(
        set(manifest["groups"])
        | set(manifest["files"])
        | set(manifest["snapshots"])
        | {"common_source"}
    )

    parser = argparse.ArgumentParser(
        description="Download pinned MotionTransfer assets and verify size/SHA256."
    )
    parser.add_argument(
        "--name",
        default="all",
        choices=valid_names,
        help="Download all assets, one method, or one named asset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing asset if its integrity check fails.",
    )
    args = parser.parse_args()

    for name in expand_names(manifest, args.name):
        if name == "common_source":
            prepare_common_source(manifest["common_source"], force=args.force)
        elif name == "raft":
            download_raft(manifest["files"][name], force=args.force)
        elif name in manifest["files"]:
            download_hf_file(name, manifest["files"][name], force=args.force)
        elif name in manifest["snapshots"]:
            download_hf_snapshot(name, manifest["snapshots"][name], force=args.force)
        else:
            raise AssertionError(f"Unhandled asset: {name}")

    tqdm.write("All requested assets passed size and SHA256 verification.")


if __name__ == "__main__":
    main()
