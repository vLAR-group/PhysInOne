#!/usr/bin/env python3
"""Verify the MotionTransfer environment and all required model assets."""
from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from packaging.requirements import Requirement

from model_assets import (
    REPO_ROOT,
    expand_names,
    load_manifest,
    torch_checkpoint_path,
    verify_file,
)


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, message: str) -> None:
        print(f"[OK]   {message}")

    def skip(self, message: str) -> None:
        print(f"[SKIP] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"[FAIL] {message}")


def _load_environment() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "PyYAML is not installed; create the Conda environment first."
        ) from error

    with (REPO_ROOT / "environment.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _check_version(reporter: Reporter, package: str, requirement: str) -> None:
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        reporter.fail(f"package missing: {package}{requirement}")
        return

    if version not in Requirement(f"{package}{requirement}").specifier:
        reporter.fail(f"{package} version {version}; expected {requirement}")
    else:
        reporter.ok(f"{package}=={version}")


def verify_environment(reporter: Reporter, *, skip_cuda: bool) -> None:
    environment = _load_environment()
    conda_dependencies = [
        item for item in environment["dependencies"] if isinstance(item, str)
    ]
    pip_dependencies = next(
        item["pip"] for item in environment["dependencies"] if isinstance(item, dict)
    )

    expected_python = next(
        item.split("=", 1)[1] for item in conda_dependencies if item.startswith("python=")
    )
    actual_python = ".".join(map(str, sys.version_info[:3]))
    if actual_python != expected_python:
        reporter.fail(f"Python {actual_python}; expected {expected_python}")
    else:
        reporter.ok(f"Python {actual_python}")

    conda_import_names = {
        "pip": "pip",
        "numpy": "numpy",
        "pyyaml": "PyYAML",
        "pytorch": "torch",
        "torchvision": "torchvision",
        "torchaudio": "torchaudio",
    }
    for item in conda_dependencies:
        name, version = item.split("=", 1)
        package = conda_import_names.get(name)
        if package:
            _check_version(reporter, package, f"=={version}")

    for item in pip_dependencies:
        if item.startswith("--"):
            continue
        requirement = Requirement(item)
        _check_version(reporter, requirement.name, str(requirement.specifier))

    for module_name in (
        "accelerate",
        "cv2",
        "diffusers",
        "open_clip",
        "pytorch_lightning",
        "rp",
        "safetensors",
        "transformers",
        "xformers.ops",
    ):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            reporter.fail(f"cannot import {module_name}: {type(error).__name__}: {error}")
        else:
            reporter.ok(f"import {module_name}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        reporter.fail("ffmpeg is not on PATH")
    else:
        result = subprocess.run(
            [ffmpeg, "-version"], check=False, capture_output=True, text=True
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        expected_ffmpeg = next(
            item.split("=", 1)[1]
            for item in conda_dependencies
            if item.startswith("ffmpeg=")
        )
        if result.returncode != 0 or f"ffmpeg version {expected_ffmpeg}" not in first_line:
            reporter.fail(
                f"FFmpeg does not match {expected_ffmpeg}: {first_line or ffmpeg}"
            )
        else:
            reporter.ok(first_line)

    try:
        import torch
    except Exception as error:
        reporter.fail(f"cannot import torch for CUDA checks: {error}")
        return

    expected_cuda = next(
        item.split("=", 1)[1]
        for item in conda_dependencies
        if item.startswith("pytorch-cuda=")
    )
    if torch.version.cuda != expected_cuda:
        reporter.fail(
            f"PyTorch CUDA build is {torch.version.cuda}; expected {expected_cuda}"
        )
    else:
        reporter.ok(f"PyTorch CUDA build {expected_cuda}")

    if skip_cuda:
        reporter.skip("CUDA device checks disabled by --skip_cuda")
        return

    if not torch.cuda.is_available():
        reporter.fail("CUDA is not available to PyTorch; check NVIDIA driver and CUDA visibility")
        return

    device_count = torch.cuda.device_count()
    reporter.ok(f"CUDA devices visible: {device_count}")
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = properties.total_memory / (1024**3)
        reporter.ok(f"cuda:{index}: {properties.name}, {memory_gib:.1f} GiB")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            versions = sorted(set(result.stdout.split()))
            reporter.ok(f"NVIDIA driver: {', '.join(versions)}")
        else:
            reporter.fail("nvidia-smi could not query the NVIDIA driver")
    else:
        reporter.fail("nvidia-smi is not on PATH")


def _verify_common_source(
    reporter: Reporter,
    spec: dict[str, Any],
    *,
    verify_hash: bool,
) -> None:
    target = REPO_ROOT / spec["local_dir"]
    if not (target / ".git").is_dir():
        reporter.fail(f"CommonSource checkout missing: {target}")
        return

    result = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if result.returncode != 0 or head != spec["revision"]:
        reporter.fail(f"CommonSource revision {head or 'unknown'}; expected {spec['revision']}")
    else:
        reporter.ok(f"CommonSource revision {head}")

    for relative_path, file_spec in spec["files"].items():
        ok, message = verify_file(
            target / relative_path,
            file_spec,
            verify_hash=verify_hash,
            progress=verify_hash,
        )
        (reporter.ok if ok else reporter.fail)(message)


def _verify_snapshot(
    reporter: Reporter,
    name: str,
    spec: dict[str, Any],
    *,
    verify_hash: bool,
) -> None:
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=spec["repo_id"],
                repo_type="model",
                revision=spec["revision"],
                allow_patterns=list(spec["files"]),
                local_files_only=True,
            )
        )
    except Exception as error:
        reporter.fail(
            f"{name} pinned snapshot missing: {spec['repo_id']} @ {spec['revision']} ({error})"
        )
        return

    reporter.ok(f"{name} snapshot: {snapshot_path}")
    for relative_path, file_spec in spec["files"].items():
        ok, message = verify_file(
            snapshot_path / relative_path,
            file_spec,
            verify_hash=verify_hash,
            progress=verify_hash,
        )
        (reporter.ok if ok else reporter.fail)(message)


def verify_assets(
    reporter: Reporter,
    manifest: dict[str, Any],
    *,
    method: str,
    verify_hash: bool,
) -> None:
    for name in expand_names(manifest, method):
        if name == "common_source":
            _verify_common_source(
                reporter, manifest["common_source"], verify_hash=verify_hash
            )
        elif name == "raft":
            spec = manifest["files"][name]
            path = torch_checkpoint_path(spec["cache_filename"])
            ok, message = verify_file(
                path, spec, verify_hash=verify_hash, progress=verify_hash
            )
            (reporter.ok if ok else reporter.fail)(message)
        elif name in manifest["files"]:
            spec = manifest["files"][name]
            path = REPO_ROOT / spec["local_path"]
            ok, message = verify_file(
                path, spec, verify_hash=verify_hash, progress=verify_hash
            )
            (reporter.ok if ok else reporter.fail)(message)
        elif name in manifest["snapshots"]:
            _verify_snapshot(
                reporter,
                name,
                manifest["snapshots"][name],
                verify_hash=verify_hash,
            )
        else:
            reporter.fail(f"unhandled manifest asset: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify environment, CUDA, pinned sources, and model assets."
    )
    parser.add_argument(
        "--method",
        choices=["all", "motionpro", "gowiththeflow"],
        default="all",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Check paths and sizes without reading every file for SHA256.",
    )
    parser.add_argument(
        "--skip_cuda",
        action="store_true",
        help="Skip physical GPU checks (for CI/diagnostics only, not inference readiness).",
    )
    args = parser.parse_args()

    reporter = Reporter()
    print("== Environment ==")
    verify_environment(reporter, skip_cuda=args.skip_cuda)
    print("\n== Model assets ==")
    verify_assets(
        reporter,
        load_manifest(),
        method=args.method,
        verify_hash=not args.quick,
    )

    print("\n== Summary ==")
    if reporter.failures:
        print(f"Setup verification failed with {len(reporter.failures)} problem(s).")
        print(
            "Run `python scripts/download_ckpt.py --name "
            + args.method
            + "` for missing assets."
        )
        raise SystemExit(1)
    print("Setup verification passed.")


if __name__ == "__main__":
    main()
