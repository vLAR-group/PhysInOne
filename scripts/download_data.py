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
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Hashable, Iterable, Optional, Sequence, TextIO
from urllib.parse import quote, unquote


REPOSITORY = "vLAR/PhysInOne"
LIST_ROOT = Path(__file__).resolve().parent / "download_lists"
CHUNK_SIZE = 8 * 1024 * 1024
USER_AGENT = "PhysInOne-public-downloader/1.0"


def format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(max(0.0, value))
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{amount:.0f} {unit}"
            return f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{amount:.1f} TiB"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    value = int(seconds + 0.5)
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class DownloadProgress:
    """Thread-safe aggregate progress bar with a clean non-TTY fallback."""

    def __init__(
        self,
        total_files: int,
        label: str = "Downloading",
        *,
        show_bytes: bool = True,
        stream: Optional[TextIO] = None,
        refresh_interval: float = 0.2,
        log_interval: float = 10.0,
    ) -> None:
        self.total_files = max(0, int(total_files))
        self.label = label
        self.show_bytes = show_bytes
        self.stream = stream or sys.stderr
        self.interactive = bool(
            getattr(self.stream, "isatty", lambda: False)()
            and not bool(getattr(self.stream, "closed", False))
        )
        self.refresh_interval = max(0.05, refresh_interval)
        self.log_interval = max(1.0, log_interval)
        self.started_at = time.monotonic()
        self._last_render = 0.0
        self._completed = 0
        self._failed = 0
        self._network_bytes = 0
        self._states: dict[Hashable, list[Optional[int]]] = {}
        self._samples: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()
        self._closed = False
        self._dirty = True

    def begin_file(
        self,
        key: Hashable,
        existing_bytes: int = 0,
        total_bytes: Optional[int] = None,
    ) -> None:
        with self._lock:
            self._states[key] = [
                max(0, int(existing_bytes)),
                None if total_bytes is None else max(0, int(total_bytes)),
            ]
            self._dirty = True
            self._render_locked()

    def advance(self, key: Hashable, amount: int) -> None:
        increment = max(0, int(amount))
        if not increment:
            return
        with self._lock:
            state = self._states.setdefault(key, [0, None])
            state[0] = int(state[0] or 0) + increment
            self._network_bytes += increment
            self._dirty = True
            current = time.monotonic()
            self._samples.append((current, self._network_bytes))
            cutoff = current - 12.0
            while len(self._samples) > 2 and self._samples[0][0] < cutoff:
                self._samples.popleft()
            self._render_locked(current)

    def complete(
        self,
        key: Hashable,
        status: str,
        local_path: str = "",
        failed: bool = False,
    ) -> None:
        with self._lock:
            state = self._states.setdefault(key, [0, None])
            if not failed and local_path:
                path = Path(local_path)
                try:
                    size = path.stat().st_size if path.is_file() else None
                except OSError:
                    size = None
                if size is not None:
                    state[0] = size
                    state[1] = size
            self._completed += 1
            if failed:
                self._failed += 1
            self._dirty = True
            self._render_locked(force=self._completed == self.total_files)

    def finish(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.total_files:
                if self._dirty:
                    self._render_locked(force=True)
                if self.interactive:
                    self.stream.write("\n")
                    self.stream.flush()
            self._closed = True

    def _speed_locked(self, current: float) -> tuple[float, float]:
        elapsed = max(current - self.started_at, 1e-6)
        average = self._network_bytes / elapsed
        if len(self._samples) >= 2:
            first_time, first_bytes = self._samples[0]
            window = max(current - first_time, 1e-6)
            recent = (self._network_bytes - first_bytes) / window
        else:
            recent = average
        return recent, average

    def _byte_totals_locked(self) -> tuple[int, Optional[int]]:
        current_bytes = sum(int(state[0] or 0) for state in self._states.values())
        known = [int(state[1]) for state in self._states.values() if state[1] is not None]
        if not known:
            return current_bytes, None
        average_size = sum(known) / len(known)
        estimated = int(sum(known) + average_size * (self.total_files - len(known)))
        return current_bytes, max(current_bytes, estimated)

    def _line_locked(self, current: float) -> str:
        ratio = self._completed / self.total_files if self.total_files else 1.0
        width = shutil.get_terminal_size(fallback=(120, 24)).columns
        bar_width = 10 if width < 120 else min(24, max(12, width // 10))
        filled = min(bar_width, int(ratio * bar_width))
        if filled < bar_width and self._completed < self.total_files:
            bar = "=" * filled + ">" + "." * (bar_width - filled - 1)
        else:
            bar = "=" * filled + "." * (bar_width - filled)
        elapsed = max(current - self.started_at, 1e-6)
        headline = f"{self.label} [{bar}] {self._completed}/{self.total_files}"
        if self._failed:
            headline += f" fail:{self._failed}"
        parts = [headline]
        if self.show_bytes:
            current_bytes, estimated_bytes = self._byte_totals_locked()
            recent, average = self._speed_locked(current)
            current_text = format_bytes(current_bytes).replace(" ", "")
            if estimated_bytes is None:
                byte_text = current_text
                eta = None
            else:
                total_text = format_bytes(estimated_bytes).replace(" ", "")
                byte_text = f"{current_text}/~{total_text}"
                eta = (
                    max(0, estimated_bytes - current_bytes) / recent
                    if recent > 0
                    else None
                )
            recent_text = format_bytes(recent).replace(" ", "")
            parts.extend([byte_text, f"{recent_text}/s", f"ETA {format_duration(eta)}"])
            if width >= 120:
                average_text = format_bytes(average).replace(" ", "")
                parts.append(f"avg {average_text}/s")
        else:
            rate = self._completed / elapsed
            remaining = max(0, self.total_files - self._completed)
            eta = remaining / rate if rate > 0 else None
            parts.extend([f"{rate:.1f} files/s", f"ETA {format_duration(eta)}"])
        return " | ".join(parts)

    def _render_locked(self, current: Optional[float] = None, force: bool = False) -> None:
        if self._closed:
            return
        timestamp = current if current is not None else time.monotonic()
        interval = self.refresh_interval if self.interactive else self.log_interval
        if not force and timestamp - self._last_render < interval:
            return
        line = self._line_locked(timestamp)
        if self.interactive:
            width = shutil.get_terminal_size(fallback=(120, 24)).columns
            clipped = line[: max(1, width - 1)]
            self.stream.write("\r\033[2K" + clipped)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
        self._last_render = timestamp
        self._dirty = False


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
    progress: Optional[DownloadProgress] = None,
    progress_key: object = "",
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
        if progress is not None:
            total_bytes = starting_size + expected if expected is not None else None
            progress.begin_file(progress_key, starting_size, total_bytes)
        written = 0
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                written += len(chunk)
                if progress is not None:
                    progress.advance(progress_key, len(chunk))
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
    progress: Optional[DownloadProgress] = None,
    progress_key: object = "",
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
            status, written = download_once(
                url, destination, timeout, resume, progress, progress_key
            )
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
    progress = DownloadProgress(len(selections), "Downloading")
    try:
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
                    REPOSITORY,
                    progress,
                    index,
                ): index
                for index, (task, remote_path, destination) in enumerate(selections)
            }
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                results.append(result)
                logs.record_download(result)
                progress.complete(
                    key, result.status, result.local_path, failed=bool(result.error)
                )
    finally:
        progress.finish()

    extraction_results: list[ExtractionResult] = []
    successful = {
        (result.task, result.remote_path): Path(result.local_path)
        for result in results
        if not result.error
    }
    extraction_plan = [
        item
        for item in selections
        if item[1].lower().endswith(".zip") and should_extract(item[0], args.extract)
    ]
    if extraction_plan:
        extraction_progress = DownloadProgress(
            len(extraction_plan), "Extracting", show_bytes=False
        )
        try:
            for index, (task, remote_path, destination) in enumerate(extraction_plan):
                archive = successful.get((task, remote_path), destination)
                marker = extraction_marker(archive)
                if not archive.is_file() and marker_matches(archive):
                    result = ExtractionResult(
                        task, str(archive), "already_extracted_zip_removed"
                    )
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
                extraction_progress.complete(
                    index, result.status, result.archive, failed=bool(result.error)
                )
        finally:
            extraction_progress.finish()

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
