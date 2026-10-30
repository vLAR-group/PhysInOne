#!/usr/bin/env python3
"""Check future-prediction submission ZIPs locally. No GT, network or GPU needed."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import zipfile
import zlib

DEFAULT_MANIFEST = Path(__file__).parent / "dataset/submission_manifest.json"
EXTENSIONS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
MAX_MEMBER_BYTES = 64 * 1024 * 1024


def dependencies():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Install dependencies: pip install pillow numpy opencv-python-headless") from exc
    return Image, np, cv2


def load_manifest(path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected manifest schema_version 1")
    scenes = data.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Manifest must contain a non-empty scenes object")
    for scene, entry in scenes.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene) or not isinstance(entry, dict):
            raise ValueError(f"Invalid scene entry: {scene!r}")
        for key in ("training_cameras", "novel_cameras", "frame_stems"):
            values = entry.get(key)
            pattern = r"[0-9]+" if key == "frame_stems" else r"CineCamera_[0-9]+"
            if (not isinstance(values, list) or not values
                    or any(not isinstance(v, str) or not re.fullmatch(pattern, v) for v in values)
                    or len(values) != len(set(values))):
                raise ValueError(f"Invalid {key}: {scene}")
        if set(entry["training_cameras"]) & set(entry["novel_cameras"]):
            raise ValueError(f"Overlapping training and novel views: {scene}")
        size = entry.get("image_size")
        if (not isinstance(size, list) or len(size) != 2
                or any(type(v) is not int or not 0 < v <= 16384 for v in size)):
            raise ValueError(f"Invalid image_size: {scene}")
    return data


def member_path(info):
    name = info.filename
    parts = PurePosixPath(name).parts
    if (not name or "\\" in name or "\x00" in name or name.startswith("/")
            or any(p == ".." or ":" in p for p in parts)
            or stat.S_ISLNK(info.external_attr >> 16)):
        raise ValueError(f"Unsafe ZIP member: {name!r}")
    if info.flag_bits & 1:
        raise ValueError(f"Encrypted ZIP member: {name}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"ZIP member exceeds 64 MiB safety limit: {name}")
    if not parts:
        raise ValueError("Empty ZIP member path")
    return "/".join(parts)


def validate_zip(path, entry):
    Image, np, cv2 = dependencies()
    result = {"zip": str(path), "errors": [], "warnings": [], "images": 0,
              "novel_views": False, "camera_summary": {}}
    errors, warnings = result["errors"], result["warnings"]
    try:
        with zipfile.ZipFile(path) as archive:
            members = {}
            for info in archive.infolist():
                name = member_path(info)
                if name in members:
                    raise ValueError(f"Duplicate ZIP member: {name}")
                members[name] = info
            # Accept direct camera folders or one outer folder, as the FP Worker does.
            roots = {n.split("/")[0] for n in members}
            if not any(n.startswith("CineCamera") for n in roots) and len(roots) == 1:
                wrapper = next(iter(roots))
                members = {n[len(wrapper)+1:]: i for n, i in members.items() if n.startswith(wrapper+"/")}
            cameras = {n.split("/")[0] for n in members if n.split("/")[0].startswith("CineCamera")}
            train, novel = set(entry["training_cameras"]), set(entry["novel_cameras"])
            if cameras not in (train, train | novel):
                errors.append(f"Camera mismatch: training={sorted(train)}, optional novel={sorted(novel)}, found={sorted(cameras)}")
            result["novel_views"] = cameras == train | novel
            selected = train | (novel if cameras & novel else set())
            frames = set(entry["frame_stems"])
            images = defaultdict(dict)
            for name, info in members.items():
                if info.is_dir():
                    continue
                parts = name.split("/")
                suffix = PurePosixPath(name).suffix.lower()
                stem = PurePosixPath(name).stem
                if (len(parts) != 3 or parts[0] not in selected or parts[1] != "rgb"
                        or suffix not in EXTENSIONS or stem not in frames):
                    errors.append(f"Unexpected file (submit target RGB only): {name}")
                    continue
                if stem in images[parts[0]]:
                    errors.append(f"Duplicate frame: {parts[0]}/rgb/{stem} has multiple image files")
                images[parts[0]][stem] = (name, info)
            for camera in sorted(selected):
                missing = frames - set(images[camera])
                if missing:
                    errors.append(f"Missing {len(missing)} target frame(s) in {camera}: {', '.join(sorted(missing)[:5])}")
            if errors:
                return result
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC check failed: {bad}")
            for camera in sorted(selected):
                hashes = set()
                flat = Counter()
                for stem in entry["frame_stems"]:
                    name, info = images[camera][stem]
                    try:
                        raw = archive.read(info)
                        if not raw:
                            raise ValueError("empty image")
                        with Image.open(io.BytesIO(raw)) as im:
                            if im.format != EXTENSIONS[PurePosixPath(name).suffix.lower()]:
                                raise ValueError(f"Image content {im.format} does not match extension")
                            if list(im.size) != entry["image_size"]:
                                raise ValueError(f"Resolution {im.size}; expected {entry['image_size']}")
                            im.verify()
                        with Image.open(io.BytesIO(raw)) as im:
                            im.load()
                            rgb = np.asarray(im.convert("RGB"))
                            hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
                            if not np.any(rgb):
                                flat["black"] += 1
                            elif np.all(rgb == 255):
                                flat["white"] += 1
                        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if decoded is None or decoded.shape != rgb.shape:
                            raise ValueError("OpenCV could not decode an RGB image of the expected size")
                        result["images"] += 1
                    except (OSError, ValueError, RuntimeError, EOFError, cv2.error, zipfile.BadZipFile) as exc:
                        errors.append(f"Unreadable image {name}: {exc}")
                if flat:
                    warnings.append(f"{camera}: {dict(flat)} uniform black/white frame(s)")
                if len(hashes) == 1 and len(frames) > 1:
                    warnings.append(f"{camera}: all future frames have identical decoded RGB pixels")
                result["camera_summary"][camera] = {"unique_decoded_frames": len(hashes), "uniform_frames": dict(flat)}
    except (OSError, ValueError, RuntimeError, EOFError, zipfile.BadZipFile, NotImplementedError, zlib.error) as exc:
        errors.append(str(exc))
    return result


def check_submission(directory, manifest, progress=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Submission directory does not exist: {directory}")
    found = defaultdict(list)
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".zip":
            found[path.stem].append(path)
    scenes = manifest["scenes"]
    report = {"release": manifest.get("release"), "scenes": {}, "errors": [], "warnings": []}
    extra = sorted(set(found) - set(scenes))
    if extra:
        report["errors"].append(f"Unexpected scene ZIPs: {', '.join(extra)}")
    for index, (scene, entry) in enumerate(scenes.items(), 1):
        paths = found.get(scene, [])
        if len(paths) != 1:
            message = "Missing scene ZIP" if not paths else "Duplicate scene ZIPs: " + ", ".join(map(str, paths))
            result = {"errors": [message], "warnings": [], "images": 0, "novel_views": False}
        else:
            result = validate_zip(paths[0], entry)
        report["scenes"][scene] = result
        if progress:
            progress(index, len(scenes), scene, not result["errors"])
    good = [r for r in report["scenes"].values() if not r["errors"]]
    if good and 0 < sum(r["novel_views"] for r in good) < len(good):
        report["warnings"].append("Mixed novel-view coverage: aggregate novel-view metrics will be unavailable.")
    report["summary"] = {
        "expected_scenes": len(scenes), "passed": len(good), "failed": len(scenes)-len(good),
        "missing": sum(not found.get(s) for s in scenes), "extra": len(extra),
        "errors": len(report["errors"])+sum(len(r["errors"]) for r in report["scenes"].values()),
        "warnings": len(report["warnings"])+sum(len(r["warnings"]) for r in report["scenes"].values()),
        "images_checked": sum(r["images"] for r in report["scenes"].values()),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path, help="Folder containing one prediction ZIP per scene")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Official manifest (bundled by default)")
    parser.add_argument("--report", type=Path, help="Write detailed JSON results to a new file")
    parser.add_argument("--verbose", action="store_true", help="Show every error and warning")
    parser.add_argument("--fail-on-warning", action="store_true", help="Return failure for content warnings too")
    args = parser.parse_args()
    start = time.monotonic()
    def progress(done, total, scene, passed):
        elapsed = time.monotonic()-start
        eta = elapsed / done * (total-done)
        line = f"[{done}/{total}] {'PASS' if passed else 'FAIL'} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s | {scene}"
        if sys.stdout.isatty():
            print("\r\033[K"+line, end="\n" if done==total else "", flush=True)
        else:
            print(line, flush=True)
    try:
        dependencies()
        if args.report and args.report.exists():
            raise ValueError(f"Report already exists; choose a new path: {args.report}")
        manifest = load_manifest(args.manifest)
        print(f"Future Prediction | manifest {manifest.get('release', 'custom')} | {len(manifest['scenes'])} scenes")
        report = check_submission(args.submission_dir, manifest, progress)
        print("Summary: " + json.dumps(report["summary"]))
        for kind in ("errors", "warnings"):
            details = list(report[kind])
            details.extend(f"{scene}: {message}" for scene, result in report["scenes"].items() for message in result[kind])
            for message in details if args.verbose else details[:5]:
                print(f"{kind.upper()}: {message}")
            if not args.verbose and len(details)>5:
                print(f"... {len(details)-5} more {kind}; use --verbose or --report")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
        failed = report["summary"]["errors"] or (args.fail_on_warning and report["summary"]["warnings"])
        print("INVALID" if failed else "VALID WITH WARNINGS" if report["summary"]["warnings"] else "VALID")
        return 1 if failed else 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
