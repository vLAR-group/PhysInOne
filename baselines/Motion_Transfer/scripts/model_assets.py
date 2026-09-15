#!/usr/bin/env python3
"""Shared manifest and integrity helpers for model setup scripts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "download_manifest.json"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported manifest schema: {manifest.get('schema_version')}")
    return manifest


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    progress_bar = tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"SHA256 {path.name}",
        disable=not progress,
    )
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
                progress_bar.update(len(chunk))
    finally:
        progress_bar.close()
    return digest.hexdigest()


def verify_file(
    path: Path,
    spec: dict[str, Any],
    *,
    verify_hash: bool = True,
    progress: bool = False,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing: {path}"

    expected_size = int(spec["size"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, (
            f"size mismatch: {path} (expected {expected_size}, got {actual_size})"
        )

    if verify_hash:
        expected_hash = str(spec["sha256"]).lower()
        actual_hash = sha256_file(path, progress=progress)
        if actual_hash != expected_hash:
            return False, (
                f"SHA256 mismatch: {path} (expected {expected_hash}, got {actual_hash})"
            )

    return True, f"ready: {path} ({format_bytes(actual_size)})"


def torch_checkpoint_path(filename: str) -> Path:
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
    return torch_home / "hub" / "checkpoints" / filename


def expand_names(manifest: dict[str, Any], name: str) -> list[str]:
    groups = manifest["groups"]
    if name in groups:
        return list(groups[name])

    valid_assets = {
        "common_source",
        *manifest["files"].keys(),
        *manifest["snapshots"].keys(),
    }
    if name in valid_assets:
        return [name]

    valid = sorted(set(groups) | valid_assets)
    raise ValueError(f"Unknown asset name: {name}. Valid names: {', '.join(valid)}")
