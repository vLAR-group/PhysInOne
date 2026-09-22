#!/usr/bin/env python3
"""
Validate PhysInOne submission ZIP files against a generated JSON manifest.

Validation checklist performed by this script:

1. Submission and manifest paths:
   - Submission directory exists and is a directory.
   - Manifest JSON exists and can be opened.
   - Manifest root is a JSON object.
   - Every manifest entry has exactly [camera_name, frame_count].
   - Every camera_name is a string beginning with "CineCamera_".
   - Every frame_count is an integer, is not a boolean, and is non-negative.

2. Submission ZIP inventory:
   - Discover ZIP files recursively below the submission directory.
   - ZIP filename stems match manifest scene names exactly.
   - Report missing scene ZIP files.
   - Report extra scene ZIP files not listed in the manifest.
   - Reject duplicate ZIP files with the same filename stem, even in different
     subdirectories.

3. ZIP archive integrity and layout:
   - The ZIP archive can be opened.
   - Every ZIP member passes ZipFile.testzip() CRC/integrity checking.
   - The archive does not wrap its contents in a same-named outer directory.
   - The expected camera directory is directly at the ZIP root.
   - For the static track, the expected camera comes from the manifest.
   - For the moving track, the expected camera is CineCamera_Moving.
   - Any additional root-level CineCamera_* directories are reported as
     warnings.

4. RGB frame files:
   - At least one .jpg exists below <expected_camera>/rgb/.
   - Each JPG member can be read from the ZIP archive.
   - Its actual content format is JPEG, not merely a .jpg filename.
   - Pillow can verify the JPEG file structure.
   - Pillow can fully load the JPEG pixel data.
   - OpenCV can decode the same bytes, matching the evaluator's image loader.
   - OpenCV returns a non-empty three-channel image with positive dimensions.
   - The number of JPG files equals the manifest frame_count.

5. Result reporting:
   - Display passed, failed, missing, extra, error, and warning counts.
   - Display all details with --verbose, or at most five errors/warnings by
     default.
   - Return exit code 0 only when there are no errors; otherwise return 1.

Expected manifest format:
{
  "scene_name_1": ["CineCamera_8", 250],
  "scene_name_2": ["CineCamera_1", 120]
}

Expected static-track ZIP structure:
scene_name.zip
└── CineCamera_8/
    └── rgb/
        ├── 0001.jpg
        ├── 0002.jpg
        └── ...

The ZIP must not wrap its contents in a directory named after the ZIP. Every
JPG under the expected camera's rgb directory must be readable, and the JPG
count must equal the frame number in the manifest. ZIP files are discovered
recursively below the submission directory. Their filenames must match the
manifest scene names exactly: no missing, extra, or duplicate scene ZIPs are
allowed, regardless of which subfolder contains them.

Usage:
    python check_submission.py SUBMISSION_DIR --manifest MANIFEST_JSON
    python check_submission.py SUBMISSION_DIR static --manifest MANIFEST_JSON
    python check_submission.py SUBMISSION_DIR moving --manifest MANIFEST_JSON

For the moving track, CineCamera_Moving is checked instead of the camera name
in the manifest, while the manifest frame number is still used.
"""

import argparse
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

try:
    import cv2
    import numpy as np
except ImportError:
    print(
        "ERROR: OpenCV and NumPy are required to validate images with the "
        "same decoder used by the evaluator. Install them with: "
        "pip install opencv-python numpy",
        file=sys.stderr,
    )
    raise SystemExit(2)

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:
    print(
        "ERROR: Pillow is required to verify JPG files. Install it with: "
        "pip install Pillow",
        file=sys.stderr,
    )
    raise SystemExit(2)

DEFAULT_JSON_PATH = "./dataset/selected.json"
MOVING_FOLDER_NAME = "CineCamera_Moving"
CAMERA_PREFIX = "CineCamera_"
RGB_FOLDER_NAME = "rgb"

# ANSI colors are enabled for interactive terminals. FORCE_COLOR=1 is useful
# for testing or terminals whose TTY status cannot be detected; NO_COLOR
# follows the standard convention for disabling colored output.
COLOR_ENABLED = (
    "NO_COLOR" not in os.environ
    and (sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1")
)
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def color(text: str, *styles: str) -> str:
    """Apply ANSI styles when color output is enabled."""
    if not COLOR_ENABLED:
        return text
    return f"{''.join(styles)}{text}{RESET}"


@dataclass(frozen=True)
class ManifestEntry:
    camera_name: str
    frame_count: int


def print_progress_bar(
    iteration: int,
    total: int,
    prefix: str = "Progress",
    suffix: str = "",
    length: int = 40,
    fill: str = "█",
    start_time: float | None = None,
) -> None:
    """Print a dynamic progress bar with elapsed time and estimated time left."""
    if total == 0:
        sys.stdout.write(
            f"\r{color(prefix, CYAN, BOLD)} | "
            f"{color('(No items to check)', YELLOW)} |\n"
        )
        sys.stdout.flush()
        return

    percent = f"{100 * iteration / total:.1f}"
    filled_length = length * iteration // total
    bar = color(fill * filled_length, GREEN) + color(
        "-" * (length - filled_length), DIM
    )

    timing = ""
    if start_time is not None:
        elapsed_seconds = max(0.0, time.monotonic() - start_time)
        if iteration > 0:
            eta_seconds = elapsed_seconds / iteration * (total - iteration)
            timing = (
                f"Elapsed: {format_duration(elapsed_seconds)} | "
                f"ETA: {format_duration(eta_seconds)} | "
            )
        else:
            timing = "Elapsed: 00:00 | ETA: calculating... | "

    sys.stdout.write(
        f"\r{color(prefix, CYAN, BOLD)} |{bar}| "
        f"{color(percent + '%', CYAN)} ({iteration}/{total}) "
        f"{color(timing, YELLOW)}"
        f"{color(suffix, MAGENTA)}"
    )
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def format_duration(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_part = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"
    return f"{minutes:02d}:{seconds_part:02d}"


def scene_code(scene_or_zip_name: str) -> str:
    """Extract CODE from **__bg*__CODE_trajectory.zip."""
    stem = Path(scene_or_zip_name).stem
    if stem.endswith("_trajectory"):
        stem = stem.removesuffix("_trajectory")

    if "__" in stem:
        code = stem.rsplit("__", 1)[1]
        if code:
            return code

    # Fallback for names that do not follow the expected convention.
    return stem


def load_manifest(json_path: Path) -> tuple[dict[str, ManifestEntry], list[str]]:
    """Load and validate scene -> [camera name, frame count] entries."""
    try:
        with json_path.open("r", encoding="utf-8") as file:
            raw_manifest = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        return {}, [f"failed to load manifest {json_path}: {error}"]

    if not isinstance(raw_manifest, dict):
        return {}, [
            "manifest root must be a JSON object, got "
            f"{type(raw_manifest).__name__}"
        ]

    manifest: dict[str, ManifestEntry] = {}
    errors: list[str] = []

    for raw_scene_name, value in raw_manifest.items():
        scene_name = str(raw_scene_name)

        if not isinstance(value, list) or len(value) != 2:
            errors.append(
                f"manifest key '{scene_name}' must have [camera_name, frame_number]"
            )
            continue

        camera_name, frame_count = value
        if not isinstance(camera_name, str) or not camera_name.startswith(CAMERA_PREFIX):
            errors.append(
                f"manifest key '{scene_name}' has invalid camera name: {camera_name!r}"
            )
            continue

        # bool is a subclass of int, so reject it explicitly.
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            errors.append(
                f"manifest key '{scene_name}' has a non-integer frame number: "
                f"{frame_count!r}"
            )
            continue
        if frame_count < 0:
            errors.append(
                f"manifest key '{scene_name}' has a negative frame number: "
                f"{frame_count}"
            )
            continue

        manifest[scene_name] = ManifestEntry(camera_name, frame_count)

    return manifest, errors


def normalize_member_name(name: str) -> str:
    """Normalize a ZIP member path to forward-slash form without leading './'."""
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.lstrip("/")


def member_parts(name: str) -> tuple[str, ...]:
    """Return normalized, non-empty path parts for a ZIP member."""
    normalized = normalize_member_name(name)
    return tuple(part for part in PurePosixPath(normalized).parts if part not in ("", "."))


def top_level_entries(zf: zipfile.ZipFile) -> set[str]:
    """Return the top-level file and directory names in a ZIP."""
    entries: set[str] = set()
    for info in zf.infolist():
        parts = member_parts(info.filename)
        if parts:
            entries.add(parts[0])
    return entries


def direct_camera_folders(zf: zipfile.ZipFile) -> set[str]:
    """Return CineCamera_* directory names located directly at the ZIP root."""
    cameras: set[str] = set()
    for info in zf.infolist():
        parts = member_parts(info.filename)
        if parts and parts[0].startswith(CAMERA_PREFIX):
            cameras.add(parts[0])
    return cameras


def jpg_members_under_camera_rgb(
    zf: zipfile.ZipFile, camera_name: str
) -> list[zipfile.ZipInfo]:
    """Return JPG files anywhere below CAMERA/rgb/ inside the ZIP."""
    images: list[zipfile.ZipInfo] = []

    for info in zf.infolist():
        if info.is_dir():
            continue

        parts = member_parts(info.filename)
        if len(parts) < 3:
            continue
        if parts[0] != camera_name or parts[1] != RGB_FOLDER_NAME:
            continue
        if PurePosixPath(parts[-1]).suffix.lower() != ".jpg":
            continue

        images.append(info)

    return images


def verify_jpg(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str | None:
    """Validate a JPG with both Pillow and OpenCV.

    Pillow can accept some JPEG streams that OpenCV cannot decode. The
    evaluator uses ``cv2.imread``-style decoding, so checking only Pillow is
    insufficient. This function deliberately validates both decoders.
    """
    try:
        image_bytes = zf.read(info)
        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "JPEG":
                return f"content format is {image.format!r}, not JPEG"
            image.verify()

        # verify() checks file integrity but does not decode all pixel data.
        # Reopen and load the image to ensure it can actually be decoded.
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()

        # The evaluator reads extracted frames with OpenCV. Decode the exact
        # bytes from the ZIP, which is stricter and more representative than
        # checking only the Pillow decoder above.
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            return "OpenCV could not decode the JPEG (cv2.imdecode returned None)"
        if decoded.ndim != 3 or decoded.shape[2] != 3:
            return f"OpenCV decoded an unexpected shape: {decoded.shape}"
        if decoded.shape[0] <= 0 or decoded.shape[1] <= 0:
            return f"OpenCV decoded an empty image: {decoded.shape}"
    except (
        OSError,
        RuntimeError,
        EOFError,
        ValueError,
        cv2.error,
        zipfile.BadZipFile,
        UnidentifiedImageError,
    ) as error:
        return str(error)

    return None


def validate_zip(
    zip_path: Path,
    manifest_entry: ManifestEntry,
    track: str,
    display_path: str | None = None,
) -> tuple[list[str], list[str]]:
    """Validate one ZIP's structure, JPG readability, and frame count."""
    errors: list[str] = []
    warnings: list[str] = []
    zip_stem = zip_path.stem
    zip_label = display_path or zip_path.name
    expected_camera = (
        MOVING_FOLDER_NAME if track == "moving" else manifest_entry.camera_name
    )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                errors.append(
                    f"[CORRUPT ZIP]      '{zip_label}' has a bad member: "
                    f"'{bad_member}'"
                )
                return errors, warnings

            top_entries = top_level_entries(zf)

            # Rule 1: the ZIP must not contain zip_stem/... as an outer wrapper.
            if zip_stem in top_entries:
                errors.append(
                    f"[OUTER FOLDER]     '{zip_label}' contains a same-named "
                    f"outer folder '{zip_stem}/'; {expected_camera}/ must be "
                    "directly at the ZIP root"
                )
                return errors, warnings

            cameras = direct_camera_folders(zf)
            if expected_camera not in cameras:
                errors.append(
                    f"[CAMERA FOLDER]    '{zip_label}' is missing root-level "
                    f"'{expected_camera}/'; found: {sorted(cameras) or '<none>'}"
                )
                return errors, warnings

            unexpected_cameras = sorted(camera for camera in cameras if camera != expected_camera)
            if unexpected_cameras:
                warnings.append(
                    f"[EXTRA CAMERA]     '{zip_label}' also contains: "
                    f"{unexpected_cameras}"
                )

            jpg_members = jpg_members_under_camera_rgb(zf, expected_camera)
            if not jpg_members:
                errors.append(
                    f"[MISSING JPG]      '{zip_label}' has no .jpg files under "
                    f"'{expected_camera}/{RGB_FOLDER_NAME}/'"
                )
                return errors, warnings

            unreadable: list[tuple[str, str]] = []
            for info in jpg_members:
                reason = verify_jpg(zf, info)
                if reason is not None:
                    unreadable.append((info.filename, reason))

            for filename, reason in unreadable:
                errors.append(
                    f"[UNREADABLE JPG]   '{zip_label}!{filename}' -> {reason}"
                )

            actual_count = len(jpg_members)
            if actual_count != manifest_entry.frame_count:
                errors.append(
                    f"[FRAME COUNT]      '{zip_label}' -> expected "
                    f"{manifest_entry.frame_count} JPG files in "
                    f"'{expected_camera}/{RGB_FOLDER_NAME}/', found {actual_count}"
                )

    except (OSError, zipfile.BadZipFile) as error:
        errors.append(f"[CORRUPT ZIP]      '{zip_label}' -> {error}")

    return errors, warnings


def check_submission(
    submission_dir: Path,
    track: str,
    json_path: Path,
    verbose: bool = False,
) -> bool:
    """Validate all required submission ZIPs and print a summary."""
    if not submission_dir.is_dir():
        print(color(f"✗ Submission directory does not exist: {submission_dir}", RED, BOLD))
        return False
    if not json_path.is_file():
        print(color(f"✗ Manifest JSON does not exist: {json_path}", RED, BOLD))
        return False

    manifest, manifest_errors = load_manifest(json_path)
    if manifest_errors:
        print(
            color(
                f"✗ The manifest is invalid ({len(manifest_errors)} problem(s)):",
                RED,
                BOLD,
            )
        )
        for error in manifest_errors:
            print(color(f"  - {error}", RED))
        return False

    errors: list[str] = []
    warnings: list[str] = []
    zip_paths = sorted(
        (
            path
            for path in submission_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".zip"
        ),
        key=lambda path: str(path.relative_to(submission_dir)),
    )
    zip_paths_by_stem: dict[str, list[Path]] = {}
    for path in zip_paths:
        zip_paths_by_stem.setdefault(path.stem, []).append(path)

    present_zips: dict[str, Path] = {}
    duplicate_stems: set[str] = set()
    for stem, paths in zip_paths_by_stem.items():
        if len(paths) > 1:
            duplicate_stems.add(stem)
            relative_paths = [
                str(path.relative_to(submission_dir)) for path in paths
            ]
            errors.append(
                f"[DUPLICATE ZIP]    scene '{stem}' has multiple ZIP files: "
                f"{relative_paths}"
            )
        else:
            present_zips[stem] = paths[0]

    expected_scene_names = set(manifest)
    submitted_scene_names = set(zip_paths_by_stem)
    missing_scene_names = sorted(expected_scene_names - submitted_scene_names)
    extra_scene_names = sorted(submitted_scene_names - expected_scene_names)

    total_checks = len(manifest)
    passed_scenes = 0
    failed_scenes = 0
    validation_start_time = time.monotonic()

    print(color("PhysInOne submission validation", CYAN, BOLD))
    print(
        color(
            f"Checking {total_checks} expected scene(s) against "
            f"{len(zip_paths)} ZIP file(s)...",
            CYAN,
        )
    )
    if total_checks == 0:
        print_progress_bar(
            0,
            0,
            prefix="Progress",
            start_time=validation_start_time,
        )

    for index, (scene_name, manifest_entry) in enumerate(manifest.items(), 1):
        if scene_name in duplicate_stems:
            failed_scenes += 1
            print_progress_bar(
                index,
                total_checks,
                suffix=f"Code: {scene_code(scene_name):<16}",
                start_time=validation_start_time,
            )
            continue

        zip_path = present_zips.get(scene_name)
        if zip_path is None:
            failed_scenes += 1
            errors.append(
                f"[MISSING ZIP]      key '{scene_name}' -> expected "
                f"'{scene_name}.zip'"
            )
            print_progress_bar(
                index,
                total_checks,
                suffix=f"Code: {scene_code(scene_name):<16}",
                start_time=validation_start_time,
            )
            continue

        relative_zip_path = str(zip_path.relative_to(submission_dir))
        zip_errors, zip_warnings = validate_zip(
            zip_path,
            manifest_entry,
            track,
            display_path=relative_zip_path,
        )
        errors.extend(zip_errors)
        warnings.extend(zip_warnings)

        if zip_errors:
            failed_scenes += 1
        else:
            passed_scenes += 1

        print_progress_bar(
            index,
            total_checks,
            suffix=f"Code: {scene_code(zip_path.name):<16}",
            start_time=validation_start_time,
        )

    for stem in sorted(zip_paths_by_stem):
        if stem not in manifest:
            relative_paths = [
                str(path.relative_to(submission_dir))
                for path in zip_paths_by_stem[stem]
            ]
            errors.append(
                f"[EXTRA ZIP]        {relative_paths} are not listed in the manifest"
            )

    print(color("Summary:", BOLD), end=" ")
    print(
        f"{color(str(passed_scenes) + ' passed', GREEN)}, "
        f"{color(str(failed_scenes) + ' failed', RED)}, "
        f"{color(str(len(missing_scene_names)) + ' missing', RED)}, "
        f"{color(str(len(extra_scene_names)) + ' extra', RED)}, "
        f"{color(str(len(errors)) + ' error(s)', RED)}, "
        f"{color(str(len(warnings)) + ' warning(s)', YELLOW)}."
    )

    if errors:
        detail_limit = len(errors) if verbose else min(5, len(errors))
        print(
            "\n"
            + color(
                f"Error details (showing {detail_limit} of {len(errors)}):",
                RED,
                BOLD,
            )
        )
        for error in errors[:detail_limit]:
            print(color(f"  - {error}", RED))
        if detail_limit < len(errors):
            print(color(
                f"  ... {len(errors) - detail_limit} more error(s) hidden; "
                "rerun with --verbose to show all.",
                DIM,
            ))
    if warnings:
        detail_limit = len(warnings) if verbose else min(5, len(warnings))
        print(
            "\n"
            + color(
                f"Warning details (showing {detail_limit} of {len(warnings)}):",
                YELLOW,
                BOLD,
            )
        )
        for warning in warnings[:detail_limit]:
            print(color(f"  - {warning}", YELLOW))
        if detail_limit < len(warnings):
            print(color(
                f"  ... {len(warnings) - detail_limit} more warning(s) hidden; "
                "rerun with --verbose to show all.",
                DIM,
            ))
    if not errors and not warnings:
        print(color("\n✓ RESULT: VALID — all checks passed. You can submit!", GREEN, BOLD))
    elif not errors:
        print(
            color(
                "\n! RESULT: VALID WITH WARNINGS — review them before submitting.",
                YELLOW,
                BOLD,
            )
        )
    else:
        print(
            color(
                "\n✗ RESULT: INVALID — do not submit yet; fix the errors and run again.",
                RED,
                BOLD,
            )
        )
    return not errors


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Root directory containing submission ZIP files at any depth",
    )
    parser.add_argument(
        "track",
        nargs="?",
        choices=("static", "moving"),
        default="static",
        help="Submission track (default: static)",
    )
    parser.add_argument(
        "--manifest",
        "--json-path",
        dest="json_path",
        type=Path,
        default=Path(os.environ.get("SUBMISSION_MANIFEST", DEFAULT_JSON_PATH)),
        help=(
            "Generated scene manifest path; default: SUBMISSION_MANIFEST or "
            f"{DEFAULT_JSON_PATH}"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show every error and warning instead of the compact default report",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored console output",
    )
    return parser.parse_args()


def main() -> int:
    global COLOR_ENABLED
    args = parse_args()
    if args.no_color:
        COLOR_ENABLED = False
    valid = check_submission(
        args.submission_dir,
        args.track,
        args.json_path,
        verbose=args.verbose,
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
