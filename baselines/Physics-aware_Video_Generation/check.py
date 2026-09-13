#!/usr/bin/env python3
"""
check.py

Validate a submission directory against the expected JSON manifest for the 
PhysInOne Leaderboard. Includes a real-time progress bar.

Usage:
    python check.py <submission_dir> [track]

    submission_dir : path to the folder containing *.zip files (e.g., ./submission)
    track          : "static" (default) or "moving"
"""

import os
import sys
import json
import zipfile
from pathlib import Path

# Default paths matching the PhysInOne workflow
DEFAULT_JSON_PATH = "./dataset/selected_cinecamera.json"
MOVING_FOLDER_NAME = "CineCamera_Moving"


# ==========================================
# Progress Bar Implementation
# ==========================================
def print_progress_bar(iteration, total, prefix='Progress', suffix='', length=40, fill='█'):
    """Prints a dynamic progress bar to the console."""
    if total == 0:
        sys.stdout.write(f'\r{prefix} | (No items to check) |\n')
        sys.stdout.flush()
        return
        
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total}) {suffix}')
    sys.stdout.flush()
    
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


# ==========================================
# Core Checking Logic
# ==========================================
def _top_level_entries(zf: zipfile.ZipFile) -> set:
    """Return the set of top-level names (files or dirs) inside a zip."""
    entries = set()
    for name in zf.namelist():
        if not name: continue
        top = name.split("/", 1)[0]
        if top: entries.add(top)
    return entries

def _is_dir_in_zip(zf: zipfile.ZipFile, dirname: str) -> bool:
    """Check whether `dirname` exists as a directory entry inside the zip."""
    for name in zf.namelist():
        if name == dirname + "/" or name.startswith(dirname + "/"):
            return True
    return False

def check_zip_legal(zip_path: Path):
    """Rule 1: A legal zip must NOT wrap contents in a same-named outer folder."""
    stem = zip_path.stem
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            tops = _top_level_entries(zf)
    except zipfile.BadZipFile as e:
        return False, f"corrupt or unreadable zip ({e})"

    if stem in tops:
        return False, f"contains an outer parent folder named '{stem}'"
    return True, None

def check_inner_folder(zip_path: Path, expected_folder: str):
    """Rule 2: The zip must directly contain the expected subfolder."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            if not _is_dir_in_zip(zf, expected_folder):
                tops = _top_level_entries(zf)
                return False, (
                    f"missing expected folder '{expected_folder}'; "
                    f"top-level entries found: {sorted(tops) or '<empty zip>'}"
                )
    except zipfile.BadZipFile as e:
        return False, f"corrupt or unreadable zip ({e})"
    return True, None


def check_submission(submission_dir: str, track: str, json_path: str) -> bool:
    submission_dir = Path(submission_dir)
    json_path = Path(json_path)

    # ---- pre-flight ---------------------------------------------------------
    if not submission_dir.is_dir():
        print(f"ERROR: submission directory does not exist: {submission_dir}")
        return False
    if not json_path.is_file():
        print(f"ERROR: manifest JSON not found: {json_path}")
        return False

    # ---- load manifest ------------------------------------------------------
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"ERROR: failed to load manifest {json_path}: {e}")
        return False

    if not isinstance(manifest, dict):
        print(f"ERROR: manifest root must be a JSON object, got {type(manifest).__name__}")
        return False

    # ---- resolve expected inner-folder name per key -------------------------
    track_is_moving = track.strip().lower() == "moving"
    def expected_folder_for(value) -> str:
        return MOVING_FOLDER_NAME if track_is_moving else str(value)

    # ---- enumerate zips present in submission -------------------------------
    print("Scanning submission directory for zip files...")
    present_zips = {p.stem: p for p in submission_dir.iterdir()
                    if p.is_file() and p.suffix.lower() == ".zip"}
    print(f"Found {len(present_zips)} zip files. Starting validation...\n")

    errors = []
    warnings = []
    total_checks = len(manifest)

    # ---- Rule 1 + Rule 2: every manifest key must have a legal zip ----------
    for i, (key, value) in enumerate(manifest.items(), 1):
        print_progress_bar(
            i, total_checks, 
            prefix='Validating', 
            suffix=f'-> {key}.zip'
        )

        if key not in present_zips:
            errors.append(f"[MISSING ZIP]      key '{key}' -> expected '{key}.zip'")
            continue

        zip_path = present_zips[key]

        # Rule 1: structural legality
        legal, reason = check_zip_legal(zip_path)
        if not legal:
            errors.append(f"[ILLEGAL STRUCT]   '{key}.zip' -> {reason}")
            continue 

        # Rule 2: inner folder
        expected = expected_folder_for(value)
        found, reason = check_inner_folder(zip_path, expected)
        if not found:
            errors.append(f"[MISSING FOLDER]   '{key}.zip' -> {reason}")

    # ---- Rule 3: warn about extra zips not in the manifest ------------------
    print("Checking for extra/unexpected zip files...")
    for stem, path in sorted(present_zips.items()):
        if stem not in manifest:
            warnings.append(f"[EXTRA ZIP]        '{stem}.zip' is not listed in the manifest")

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Submission dir : {submission_dir.resolve()}")
    print(f"Manifest       : {json_path.resolve()}  ({len(manifest)} keys)")
    print(f"Track          : {track}  "
          f"(inner folder expected: "
          f"{MOVING_FOLDER_NAME if track_is_moving else '<value from JSON>'})")
    print(f"Zips found     : {len(present_zips)}")
    print("=" * 70)

    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   {e}")
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   {w}")
    if not errors and not warnings:
        print("\n✅ All checks passed — submission is valid.")
    elif not errors:
        print("\n🟡 Submission is structurally valid but has warnings above.")

    print("=" * 70)
    return len(errors) == 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if "--help" in sys.argv or "-h" in sys.argv else 1

    # Match the exact CLI usage from the documentation:
    # python check.py <submission_dir> [track]
    submission_dir = sys.argv[1]
    track = sys.argv[2] if len(sys.argv) > 2 else "static"
    
    # Allow overriding the manifest path via environment variable if needed, 
    # but default to the standard PhysInOne path.
    json_path = os.environ.get("SUBMISSION_MANIFEST", DEFAULT_JSON_PATH)

    ok = check_submission(submission_dir, track, json_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())