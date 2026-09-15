#!/usr/bin/env python3
"""Download selected public PhysInOne scene archives from dataset shards."""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Optional, Sequence
from urllib.parse import quote, urlparse

from download_data import DownloadResult, download_file, utc_now


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_SELECTION = Path("selected_cases.json")
DEFAULT_REPO_MAP = SCRIPT_ROOT / "download_lists" / "repo_map.json"


def load_json(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_repository(value: str) -> str:
    value = value.strip().rstrip("/")
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "datasets":
            value = f"{parts[1]}/{parts[2]}"
        else:
            raise ValueError(f"cannot parse dataset repository URL: {value}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
        raise ValueError(f"invalid dataset repository name: {value}")
    return value


def load_repo_map(path: Path) -> dict[str, str]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError("repository map must be a JSON object")
    return {str(key): normalize_repository(str(value)) for key, value in raw.items()}


def safe_component(value: object, field: str) -> str:
    text = str(value)
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"unsafe {field}: {text!r}")
    if any(character.isspace() for character in text):
        raise ValueError(f"{field} contains whitespace: {text!r}")
    return text


def encoded_remote_path(case: dict) -> str:
    raw = str(case.get("hf_zip_path", ""))
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"unsafe dataset path: {raw!r}")
    if any(part in {"", ".", ".."} for part in path.parts) or "\\" in raw:
        raise ValueError(f"unsafe dataset path: {raw!r}")
    return quote(raw, safe="/._-~")


def output_path(output_dir: Path, case: dict, keep_structure: bool) -> Path:
    remote = PurePosixPath(str(case["hf_zip_path"]))
    filename = safe_component(remote.name, "archive name")
    if not keep_structure:
        return output_dir / filename
    return output_dir.joinpath(
        safe_component(case["part_id"], "part ID"),
        safe_component(case["split"], "split"),
        safe_component(case["activity_type"], "activity type"),
        filename,
    )


class Logs:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.success = root / "downloads.tsv"
        self.failures = root / "failures.tsv"
        self._lock = threading.Lock()
        self.success.write_text(
            "case_id\trepository\tremote_path\tlocal_path\tstatus\tbytes_written\ttimestamp\n",
            encoding="utf-8",
        )
        self.failures.write_text(
            "case_id\trepository\tremote_path\terror\ttimestamp\n",
            encoding="utf-8",
        )

    def record(self, case_id: str, repository: str, result: DownloadResult) -> None:
        now = utc_now()
        with self._lock:
            if result.error:
                with self.failures.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{case_id}\t{repository}\t{result.remote_path}\t"
                        f"{result.error}\t{now}\n"
                    )
            else:
                with self.success.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"{case_id}\t{repository}\t{result.remote_path}\t"
                        f"{result.local_path}\t{result.status}\t"
                        f"{result.bytes_written}\t{now}\n"
                    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download selected public PhysInOne scene archives."
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=DEFAULT_SELECTION,
        help="Selection JSON produced by filter_cases.py (default: selected_cases.json).",
    )
    parser.add_argument(
        "--repo-map",
        "--repo_map",
        dest="repo_map",
        type=Path,
        default=DEFAULT_REPO_MAP,
        help="Dataset shard map supplied with this repository.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("PhysInOne_data/full_dataset"),
        help="Whitespace-free local output directory.",
    )
    parser.add_argument(
        "--revision", default="main", help="Dataset revision (default: main)."
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent downloads (default: 4)."
    )
    parser.add_argument(
        "--retries", type=int, default=3, help="Retries after each failure (default: 3)."
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="Request timeout in seconds."
    )
    parser.add_argument(
        "--keep-shard-structure",
        "--keep_shard_structure",
        dest="keep_shard_structure",
        action="store_true",
        help="Preserve part/split/activity directories instead of a flat output.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Restart incomplete .part files instead of resuming them.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace completed local files."
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Print the planned transfers without downloading.",
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
    if any(character.isspace() for character in str(args.output_dir)):
        raise SystemExit("--output-dir must not contain whitespace")

    selection = load_json(args.selection)
    if not isinstance(selection, dict) or not isinstance(selection.get("cases"), list):
        raise ValueError("selection JSON must contain a list named 'cases'")
    cases = selection["cases"]
    repository_map = load_repo_map(args.repo_map)
    missing = sorted(
        {
            str(case.get("part_id", ""))
            for case in cases
            if str(case.get("part_id", "")) not in repository_map
        }
    )
    if missing:
        raise KeyError("part IDs missing from repository map: " + ", ".join(missing))

    output_dir = args.output_dir.expanduser().resolve()
    planned = []
    for case in cases:
        part_id = str(case["part_id"])
        planned.append(
            (
                str(case.get("case_id", "")),
                repository_map[part_id],
                encoded_remote_path(case),
                output_path(output_dir, case, args.keep_shard_structure),
            )
        )

    print(f"cases: {len(planned)}")
    if args.dry_run:
        for case_id, repository, remote_path, destination in planned[:20]:
            print(f"{case_id}: {repository}/{remote_path} -> {destination}")
        if len(planned) > 20:
            print(f"... and {len(planned) - 20} more")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = Logs(output_dir / "_download_logs" / run_id)
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_file,
                "full_dataset",
                remote_path,
                destination,
                args.revision,
                args.timeout,
                args.retries,
                not args.no_resume,
                args.overwrite,
                False,
                repository,
            ): (case_id, repository)
            for case_id, repository, remote_path, destination in planned
        }
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            case_id, repository = futures[future]
            result = future.result()
            results.append(result)
            logs.record(case_id, repository, result)
            label = result.status if not result.error else f"failed: {result.error}"
            print(f"[{completed}/{total}] {label}: {case_id}")

    failures = sum(1 for result in results if result.error)
    summary = {
        "selected_cases": len(planned),
        "successful_downloads": len(results) - failures,
        "download_failures": failures,
        "output_dir": str(output_dir),
        "log_dir": str(logs.root),
        "finished_at": utc_now(),
    }
    (logs.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
