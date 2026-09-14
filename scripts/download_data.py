#!/usr/bin/env python3
"""Download public PhysInOne evaluation data and validation 3D assets."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import stat
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Sequence
from urllib.parse import quote, unquote


REPOSITORY = "vLAR/PhysInOne"
LIST_ROOT = Path(__file__).resolve().parent / "download_lists"
CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "PhysInOne-public-downloader/1.0"


@dataclass(frozen=True)
class TaskSpec:
    manifest: str
    remote_prefix: str
    local_root: str
    extract_by_default: bool = False


TASKS = {
    "video_generation": TaskSpec(
        "video_generation.txt",
        "Leaderboard/Video%20Generation/",
        "leaderboard/video_generation",
    ),
    "future_prediction": TaskSpec(
        "future_prediction.txt",
        "Leaderboard/Future%20Prediction/",
        "leaderboard/future_prediction",
    ),
    "physical_properties_estimation": TaskSpec(
        "physical_properties_estimation.txt",
        "Leaderboard/Physical%20Properties%20Estimation/",
        "leaderboard/physical_properties_estimation",
    ),
    "motion_transfer": TaskSpec(
        "motion_transfer.txt",
        "Leaderboard/Motion%20Transfer/",
        "leaderboard/motion_transfer",
    ),
    "3d_assets": TaskSpec(
        "3d_assets.txt",
        "Assets/PhysicBenchmark/",
        "assets/PhysicBenchmark",
        extract_by_default=True,
    ),
}

ALIASES = {
    "video": "video_generation",
    "vg": "video_generation",
    "future": "future_prediction",
    "fp": "future_prediction",
    "physical_properties": "physical_properties_estimation",
    "physical_property_estimation": "physical_properties_estimation",
    "ppe": "physical_properties_estimation",
    "motion": "motion_transfer",
    "mt": "motion_transfer",
    "assets": "3d_assets",
    "3d": "3d_assets",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_message(exc: BaseException) -> str:
    message = " ".join(str(exc).replace("\t", " ").splitlines()).strip()
    return message or type(exc).__name__


def normalize_task(value: str) -> str:
    slug = value.strip().lower().replace("-", "_").replace(" ", "_")
    slug = ALIASES.get(slug, slug)
    if slug not in TASKS and slug != "all":
        choices = ", ".join([*TASKS, "all"])
        raise argparse.ArgumentTypeError(
            f"unknown task {value!r}; choose one of: {choices}"
        )
    return slug


def selected_tasks(values: Sequence[str]) -> list[str]:
    if not values or "all" in values:
        return list(TASKS)
    return list(dict.fromkeys(values))


def read_manifest(spec: TaskSpec) -> list[str]:
    path = LIST_ROOT / spec.manifest
    if not path.is_file():
        raise RuntimeError(f"missing download manifest: {path}")
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise RuntimeError(f"empty download manifest: {path}")
    if len(entries) != len(set(entries)):
        raise RuntimeError(f"duplicate paths in download manifest: {path}")
    invalid = [entry for entry in entries if not entry.startswith(spec.remote_prefix)]
    if invalid:
        raise RuntimeError(f"invalid path in {path.name}: {invalid[0]}")
    for entry in entries:
        validate_remote_path(entry)
    return entries


def validate_remote_path(encoded_path: str) -> None:
    decoded = unquote(encoded_path)
    path = PurePosixPath(decoded)
    if path.is_absolute() or not path.parts:
        raise RuntimeError(f"unsafe remote path: {encoded_path}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe remote path: {encoded_path}")
    if "\\" in decoded or re.match(r"^[A-Za-z]:", decoded):
        raise RuntimeError(f"unsafe remote path: {encoded_path}")


def local_path_for(output_dir: Path, spec: TaskSpec, remote_path: str) -> Path:
    relative = unquote(remote_path[len(spec.remote_prefix) :])
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise RuntimeError(f"unsafe relative path: {remote_path}")
    if any(any(character.isspace() for character in part) for part in parts):
        raise RuntimeError(f"local path would contain whitespace: {remote_path}")
    return output_dir.joinpath(*PurePosixPath(spec.local_root).parts, *parts)


def filter_entries(entries: Iterable[str], scene_filters: Sequence[str]) -> list[str]:
    filters = [item.casefold() for item in scene_filters if item.strip()]
    if not filters:
        return list(entries)
    return [
        entry
        for entry in entries
        if any(item in unquote(entry).casefold() for item in filters)
    ]


def build_url(repository: str, revision: str, remote_path: str) -> str:
    encoded_revision = quote(revision, safe="")
    encoded_path = quote(unquote(remote_path), safe="/._-~")
    return (
        f"https://huggingface.co/datasets/{repository}/resolve/"
        f"{encoded_revision}/{encoded_path}?download=true"
    )


@dataclass(frozen=True)
class DownloadResult:
    task: str
    remote_path: str
    local_path: str
    status: str
    bytes_written: int = 0
    error: str = ""


def download_once(
    url: str,
    destination: Path,
    timeout: float,
    resume: bool,
) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    if not resume and partial.exists():
        partial.unlink()

    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            content_range = exc.headers.get("Content-Range", "")
            match = re.search(r"\*/(\d+)$", content_range)
            if match and int(match.group(1)) == offset:
                os.replace(partial, destination)
                return "resumed", 0
        raise

    with response:
        status_code = getattr(response, "status", response.getcode())
        append = bool(offset and status_code == 206)
        mode = "ab" if append else "wb"
        starting_size = offset if append else 0
        expected_raw = response.headers.get("Content-Length")
        expected = int(expected_raw) if expected_raw and expected_raw.isdigit() else None
        written = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
        if expected is not None and written != expected:
            raise IOError(
                f"incomplete response: expected {expected} bytes, received {written}"
            )
        if partial.stat().st_size != starting_size + written:
            raise IOError("downloaded file size changed unexpectedly")

    os.replace(partial, destination)
    return ("resumed" if append else "downloaded"), written


def download_file(
    task: str,
    remote_path: str,
    destination: Path,
    revision: str,
    timeout: float,
    retries: int,
    resume: bool,
    force: bool,
    skip_removed_archive: bool,
    repository: str = REPOSITORY,
) -> DownloadResult:
    if skip_removed_archive and not destination.exists() and marker_matches(destination):
        return DownloadResult(
            task, remote_path, str(destination), "extracted_zip_removed"
        )
    if destination.is_file() and not force:
        return DownloadResult(task, remote_path, str(destination), "reused")
    if destination.exists() and not destination.is_file():
        return DownloadResult(
            task,
            remote_path,
            str(destination),
            "failed",
            error="download target exists and is not a file",
        )
    if force:
        partial = destination.with_name(f"{destination.name}.part")
        if partial.exists():
            partial.unlink()

    url = build_url(repository, revision, remote_path)
    error = ""
    for attempt in range(retries + 1):
        try:
            status, written = download_once(url, destination, timeout, resume)
            return DownloadResult(
                task, remote_path, str(destination), status, bytes_written=written
            )
        except Exception as exc:  # continue independent downloads and report every failure
            error = safe_message(exc)
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    return DownloadResult(
        task, remote_path, str(destination), "failed", error=error
    )


def is_safe_zip_member(info: zipfile.ZipInfo) -> bool:
    raw = info.filename
    normalized = raw.replace("\\", "/")
    path = PurePosixPath(normalized)
    if raw != normalized or path.is_absolute() or not path.parts:
        return False
    if any(part in {"", ".", ".."} for part in path.parts):
        return False
    if re.match(r"^[A-Za-z]:", normalized):
        return False
    mode = (info.external_attr >> 16) & 0o170000
    return mode != stat.S_IFLNK


def extraction_marker(archive: Path) -> Path:
    return archive.parent / ".physinone_extract_status" / f"{archive.name}.json"


def marker_matches(archive: Path, destination_required: Optional[Path] = None) -> bool:
    marker = extraction_marker(archive)
    if not marker.is_file():
        return False
    if destination_required is not None and not destination_required.is_dir():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if archive.is_file():
        return state.get("archive_size") == archive.stat().st_size
    return state.get("archive_removed") is True


def write_extraction_marker(archive: Path, file_count: int, removed: bool = False) -> None:
    marker = extraction_marker(archive)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "archive": archive.name,
        "archive_size": archive.stat().st_size if archive.is_file() else None,
        "archive_removed": removed,
        "files": file_count,
        "status": "complete",
        "verified_at": utc_now(),
    }
    marker.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def inspect_archive(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    unsafe = [info.filename for info in members if not is_safe_zip_member(info)]
    if unsafe:
        raise RuntimeError(f"unsafe ZIP member: {unsafe[0]}")
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"ZIP CRC check failed at {bad}")
    return members


def extract_scene_archive(archive_path: Path) -> tuple[str, int, int]:
    destination = archive_path.with_suffix("")
    if marker_matches(archive_path, destination):
        return "already_extracted", 0, 0
    if destination.exists():
        raise RuntimeError(
            f"extraction target exists without a matching marker: {destination}"
        )
    temporary = archive_path.parent / (
        f".{archive_path.stem}.extracting-{os.getpid()}-{threading.get_ident()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = inspect_archive(archive)
            roots = {PurePosixPath(info.filename).parts[0] for info in members}
            if roots != {archive_path.stem}:
                raise RuntimeError(
                    f"unexpected ZIP roots {sorted(roots)}; expected {archive_path.stem!r}"
                )
            archive.extractall(temporary)
            extracted = temporary / archive_path.stem
            if not extracted.is_dir():
                raise RuntimeError("archive did not create the expected scene directory")
            extracted.replace(destination)
        file_count = sum(1 for info in members if not info.is_dir())
        write_extraction_marker(archive_path, file_count)
        return "extracted", file_count, 0
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def merge_tree(source_root: Path, destination_root: Path) -> tuple[int, int]:
    added = 0
    reused = 0
    for source in sorted(source_root.rglob("*")):
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file():
                raise RuntimeError(f"extraction conflict: {destination}")
            if source.stat().st_size != destination.stat().st_size or not filecmp.cmp(
                source, destination, shallow=False
            ):
                raise RuntimeError(f"extraction would overwrite different data: {destination}")
            source.unlink()
            reused += 1
        else:
            source.replace(destination)
            added += 1
    return added, reused


def extract_asset_archive(archive_path: Path) -> tuple[str, int, int]:
    if marker_matches(archive_path):
        return "already_extracted", 0, 0
    temporary_root = archive_path.parent / ".physinone_extracting"
    temporary = temporary_root / (
        f"{archive_path.stem}-{os.getpid()}-{threading.get_ident()}"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = inspect_archive(archive)
            archive.extractall(temporary)
        added, reused = merge_tree(temporary, archive_path.parent)
        file_count = sum(1 for info in members if not info.is_dir())
        write_extraction_marker(archive_path, file_count)
        return "extracted", added, reused
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if temporary_root.exists() and not any(temporary_root.iterdir()):
            temporary_root.rmdir()


@dataclass(frozen=True)
class ExtractionResult:
    task: str
    archive: str
    status: str
    added: int = 0
    reused: int = 0
    error: str = ""


class RunLogs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.downloads = root / "downloads.tsv"
        self.extractions = root / "extractions.tsv"
        self.failures = root / "failures.tsv"
        self._lock = threading.Lock()
        self.downloads.write_text(
            "task\tremote_path\tlocal_path\tstatus\tbytes_written\ttimestamp\n",
            encoding="utf-8",
        )
        self.extractions.write_text(
            "task\tarchive\tstatus\tadded\treused\ttimestamp\n",
            encoding="utf-8",
        )
        self.failures.write_text(
            "stage\ttask\tpath\terror\ttimestamp\n", encoding="utf-8"
        )

    def record_download(self, result: DownloadResult) -> None:
        now = utc_now()
        with self._lock:
            if result.error:
                with self.failures.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"download\t{result.task}\t{result.remote_path}\t"
                        f"{result.error}\t{now}\n"
                    )
            else:
                with self.downloads.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{result.task}\t{result.remote_path}\t{result.local_path}\t"
                        f"{result.status}\t{result.bytes_written}\t{now}\n"
                    )

    def record_extraction(self, result: ExtractionResult) -> None:
        now = utc_now()
        with self._lock:
            if result.error:
                with self.failures.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"extract\t{result.task}\t{result.archive}\t"
                        f"{result.error}\t{now}\n"
                    )
            else:
                with self.extractions.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{result.task}\t{result.archive}\t{result.status}\t"
                        f"{result.added}\t{result.reused}\t{now}\n"
                    )


def should_extract(task: str, requested: Optional[bool]) -> bool:
    if requested is not None:
        return requested
    return TASKS[task].extract_by_default


def extract_download(
    task: str,
    archive: Path,
    delete_after: bool,
) -> ExtractionResult:
    try:
        if task == "3d_assets":
            status, added, reused = extract_asset_archive(archive)
        else:
            status, added, reused = extract_scene_archive(archive)
        if delete_after and archive.is_file():
            file_count = added + reused
            archive.unlink()
            write_extraction_marker(archive, file_count, removed=True)
        return ExtractionResult(task, str(archive), status, added, reused)
    except Exception as exc:
        return ExtractionResult(
            task, str(archive), "failed", error=safe_message(exc)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download public PhysInOne Leaderboard data and validation 3D assets "
            "with Python standard-library networking."
        )
    )
    parser.add_argument(
        "--task",
        nargs="+",
        type=normalize_task,
        default=["all"],
        help=(
            "Task(s): video_generation, future_prediction, "
            "physical_properties_estimation, motion_transfer, 3d_assets, or all."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("PhysInOne_data"),
        help="Local output root without spaces (default: ./PhysInOne_data).",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Keep paths containing this scene name or six-character ID; repeatable.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent downloads (default: 4)."
    )
    parser.add_argument(
        "--revision", default="main", help="Dataset revision (default: main)."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries after each failure (default: 3)."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Network timeout in seconds (default: 60).",
    )
    extraction = parser.add_mutually_exclusive_group()
    extraction.add_argument(
        "--extract", dest="extract", action="store_true", help="Safely extract ZIP files."
    )
    extraction.add_argument(
        "--no-extract",
        dest="extract",
        action="store_false",
        help="Keep all ZIP files compressed.",
    )
    parser.set_defaults(extract=None)
    parser.add_argument(
        "--delete-zip-after-extract",
        action="store_true",
        help="Delete a ZIP only after successful extraction.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Restart incomplete .part files instead of resuming them.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace completed local downloads."
    )
    parser.add_argument(
        "--list-only", action="store_true", help="Print selected public URLs and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show counts and output paths without downloading."
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.retries < 0:
        raise SystemExit("--retries cannot be negative")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")
    if args.delete_zip_after_extract and args.extract is False:
        raise SystemExit("--delete-zip-after-extract cannot be used with --no-extract")
    if any(character.isspace() for character in str(args.output_dir)):
        raise SystemExit("--output-dir must not contain whitespace")

    tasks = selected_tasks(args.task)
    output_dir = args.output_dir.expanduser().resolve()
    selections: list[tuple[str, str, Path]] = []
    counts: dict[str, int] = {}
    for task in tasks:
        spec = TASKS[task]
        entries = filter_entries(read_manifest(spec), args.scene)
        counts[task] = len(entries)
        selections.extend(
            (task, entry, local_path_for(output_dir, spec, entry))
            for entry in entries
        )

    for task in tasks:
        print(f"{task}: {counts[task]} file(s)")
    print(f"total: {len(selections)} file(s)")
    if args.scene and not selections:
        print("No files matched --scene.", file=sys.stderr)
        return 2
    if args.list_only:
        for _, remote_path, _ in selections:
            print(build_url(REPOSITORY, args.revision, remote_path))
        return 0
    if args.dry_run:
        for task in tasks:
            print(f"{task} -> {output_dir / TASKS[task].local_root}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = RunLogs(output_dir / "_download_logs" / run_id)
    results: list[DownloadResult] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_file,
                task,
                remote_path,
                destination,
                args.revision,
                args.timeout,
                args.retries,
                not args.no_resume,
                args.force,
                (
                    args.delete_zip_after_extract
                    and should_extract(task, args.extract)
                    and remote_path.lower().endswith(".zip")
                ),
            ): (task, remote_path)
            for task, remote_path, destination in selections
        }
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            logs.record_download(result)
            label = result.status if not result.error else f"failed: {result.error}"
            print(f"[{completed}/{total}] {result.task}: {label}: {result.remote_path}")

    extraction_results: list[ExtractionResult] = []
    successful = {
        (result.task, result.remote_path): Path(result.local_path)
        for result in results
        if not result.error
    }
    for task, remote_path, destination in selections:
        if not remote_path.lower().endswith(".zip"):
            continue
        if not should_extract(task, args.extract):
            continue
        archive = successful.get((task, remote_path), destination)
        marker = extraction_marker(archive)
        if not archive.is_file() and marker_matches(archive):
            result = ExtractionResult(task, str(archive), "already_extracted_zip_removed")
        elif not archive.is_file():
            result = ExtractionResult(
                task,
                str(archive),
                "failed",
                error="archive is unavailable because its download failed",
            )
        else:
            result = extract_download(task, archive, args.delete_zip_after_extract)
        extraction_results.append(result)
        logs.record_extraction(result)
        label = result.status if not result.error else f"failed: {result.error}"
        print(f"[extract] {task}: {label}: {archive}")

    download_failures = sum(1 for result in results if result.error)
    extraction_failures = sum(1 for result in extraction_results if result.error)
    summary = {
        "repository": REPOSITORY,
        "revision": args.revision,
        "tasks": tasks,
        "selected_files": len(selections),
        "successful_downloads": len(results) - download_failures,
        "download_failures": download_failures,
        "completed_extractions": len(extraction_results) - extraction_failures,
        "extraction_failures": extraction_failures,
        "output_dir": str(output_dir),
        "log_dir": str(logs.root),
        "finished_at": utc_now(),
    }
    (logs.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if download_failures or extraction_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
