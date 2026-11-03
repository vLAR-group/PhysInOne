#!/usr/bin/env python3
"""Validate Physical Properties Estimation scene ZIPs offline, without GT or a GPU."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time
import zipfile
import zlib

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "dataset/submission_manifest.json"
EXTENSIONS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_MEMBERS = 4096
JSON_FILES = {"materials_configs.json", "initvelocity.json"}
MATERIAL_FIELDS = {
    "ELASTIC": {"E", "nu"},
    "NEWTONIAN": {"mu", "kappa"},
    "NON_NEWTONIAN": {"mu", "kappa", "yield", "visc"},
    "PLASTICINE": {"E", "nu", "yield"},
    "SAND": {"friction_angle"},
}


def dependencies():
    try:
        import cv2
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise ValueError("Install dependencies: pip install pillow numpy opencv-python-headless") from exc
    return Image, np, cv2


def strict_json(text):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=unique_keys, parse_constant=invalid_constant)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def load_manifest(path):
    data = strict_json(Path(path).read_text(encoding="utf-8-sig"))
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise ValueError("Expected manifest schema_version 1")
    scenes = data.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("Manifest must contain a non-empty scenes object")
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict) or set(defaults) - {"frame_stems", "image_size"}:
        raise ValueError("Manifest defaults may contain only frame_stems and image_size")
    for scene, entry in scenes.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", scene) or not isinstance(entry, dict):
            raise ValueError(f"Invalid scene entry: {scene!r}")
        entry = {**defaults, **entry}
        scenes[scene] = entry
        if not isinstance(entry.get("material"), str) or entry["material"] not in MATERIAL_FIELDS:
            raise ValueError(f"Invalid material in manifest: {scene}")
        for key in ("training_cameras", "novel_cameras", "frame_stems"):
            values = entry.get(key)
            pattern = r"[0-9]{4}" if key == "frame_stems" else r"CineCamera_[0-9]+"
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


def validate_parameters(material, velocity, expected_material):
    errors = []
    if not isinstance(material, dict) or set(material) != {expected_material}:
        errors.append(f"materials_configs.json must contain exactly one material: {expected_material}")
    else:
        values = material[expected_material]
        required = MATERIAL_FIELDS[expected_material]
        if not isinstance(values, dict):
            errors.append("Material parameters must be a JSON object")
        else:
            missing, extra = required - set(values), set(values) - required
            if missing:
                errors.append(f"Missing material parameter(s): {', '.join(sorted(missing))}")
            if extra:
                errors.append(f"Unexpected material parameter(s): {', '.join(sorted(extra))}")
            for key, value in values.items():
                if not number(value):
                    errors.append(f"Material parameter {key} must be a finite number, not a string/list/bool")
                elif key in {"E", "mu", "kappa", "yield", "visc"} and value <= 0:
                    errors.append(f"Material parameter {key} must be > 0 for log10 scoring")
                elif key == "nu" and not -1 < value < 0.5:
                    errors.append("Poisson ratio nu must satisfy -1 < nu < 0.5")
                elif key == "friction_angle" and not 0 <= value < 90:
                    errors.append("friction_angle must be in degrees with 0 <= value < 90")
    if not isinstance(velocity, dict) or set(velocity) != {"vx", "vy", "vz"}:
        errors.append("initvelocity.json must contain exactly vx, vy and vz")
    elif any(not number(value) for value in velocity.values()):
        errors.append("Initial velocity components must be finite numbers, not strings/lists/bools")
    return errors


def member_path(info):
    name = info.filename
    parts = name.rstrip("/").split("/")
    if (not name or "\\" in name or "\x00" in name
            or any(part in {"", ".", ".."} or ":" in part for part in parts)
            or stat.S_ISLNK(info.external_attr >> 16)):
        raise ValueError(f"Unsafe ZIP member: {name!r}")
    if info.flag_bits & 1:
        raise ValueError(f"Encrypted ZIP member: {name}")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError(f"ZIP member exceeds 64 MiB limit: {name}")
    return "/".join(parts)


def payload_members(archive, scene):
    if len(archive.infolist()) > MAX_MEMBERS:
        raise ValueError(f"ZIP has more than {MAX_MEMBERS} members")
    members = {}
    for info in archive.infolist():
        name = member_path(info)
        if name in members:
            raise ValueError(f"Duplicate ZIP member: {name}")
        members[name] = info
    prefix = "property_estimation/"
    if (not members or any(
            not name.startswith(prefix) and not (name == prefix.rstrip("/") and info.is_dir())
            for name, info in members.items())):
        raise ValueError(
            "ZIP root must contain only property_estimation/; bare payloads and scene wrappers "
            f"are not supported. Package with: python -m zipfile -c {scene}.zip property_estimation"
        )
    payload = {name[len(prefix):]: info for name, info in members.items()
               if name.startswith(prefix)}
    if any(name.split("/")[0] in {scene, "property_estimation"} for name in payload):
        raise ValueError("Place parameter JSONs, novel/ and reinit/ directly inside property_estimation/")
    return payload


def read_parameter_json(archive, info):
    if info.file_size > MAX_JSON_BYTES:
        raise ValueError("Parameter JSON exceeds 1 MiB limit")
    return strict_json(archive.read(info).decode("utf-8-sig"))


def validate_zip(path, entry):
    Image, np, cv2 = dependencies()
    result = {"zip": str(path), "errors": [], "warnings": [], "images": 0,
              "parameters_valid": False, "camera_summary": {}}
    errors, warnings = result["errors"], result["warnings"]
    groups = {"novel": set(entry["novel_cameras"]),
              "reinit": set(entry["training_cameras"]) | set(entry["novel_cameras"])}
    frames = set(entry["frame_stems"])
    try:
        with zipfile.ZipFile(path) as archive:
            members = payload_members(archive, Path(path).stem)
            images = defaultdict(dict)
            directories = set(groups)
            for group, cameras in groups.items():
                directories.update(f"{group}/{cam}" for cam in cameras)
                directories.update(f"{group}/{cam}/rgb" for cam in cameras)
            for name, info in members.items():
                if info.is_dir():
                    if name not in directories:
                        errors.append(f"Unexpected directory: {name}")
                    continue
                if name in JSON_FILES:
                    continue
                parts = name.split("/")
                suffix, stem = PurePosixPath(name).suffix.lower(), PurePosixPath(name).stem
                if (len(parts) != 4 or parts[0] not in groups or parts[1] not in groups[parts[0]]
                        or parts[2] != "rgb" or suffix not in EXTENSIONS or stem not in frames):
                    errors.append(f"Unexpected file: {name}")
                    continue
                key = (parts[0], parts[1])
                if stem in images[key]:
                    errors.append(f"Duplicate frame: {parts[0]}/{parts[1]}/rgb/{stem}")
                images[key][stem] = (name, info)
            for group, cameras in groups.items():
                for camera in sorted(cameras):
                    missing = frames - set(images[(group, camera)])
                    if missing:
                        errors.append(f"Missing {len(missing)} frame(s) in {group}/{camera}: "
                                      + ", ".join(sorted(missing)[:5]))
            parameters = {}
            for filename in sorted(JSON_FILES):
                if filename not in members or members[filename].is_dir():
                    errors.append(f"Missing parameter file: {filename}")
                else:
                    try:
                        parameters[filename] = read_parameter_json(archive, members[filename])
                    except (ValueError, UnicodeError) as exc:
                        errors.append(f"Invalid {filename}: {exc}")
            if len(parameters) == 2:
                parameter_errors = validate_parameters(parameters["materials_configs.json"],
                    parameters["initvelocity.json"], entry["material"])
                errors.extend(parameter_errors)
                result["parameters_valid"] = not parameter_errors
            if errors:
                return result
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC check failed: {bad}")
            for group, cameras in groups.items():
                for camera in sorted(cameras):
                    hashes, flat = set(), Counter()
                    for stem in entry["frame_stems"]:
                        name, info = images[(group, camera)][stem]
                        try:
                            raw = archive.read(info)
                            if not raw:
                                raise ValueError("empty image")
                            with Image.open(io.BytesIO(raw)) as im:
                                if im.format != EXTENSIONS[PurePosixPath(name).suffix.lower()]:
                                    raise ValueError(f"Image content {im.format} does not match extension")
                                if list(im.size) != entry["image_size"]:
                                    raise ValueError(f"Resolution {im.size}; expected {entry['image_size']}")
                                if im.mode != "RGB":
                                    raise ValueError(f"Expected RGB image, found mode {im.mode}")
                                im.verify()
                            with Image.open(io.BytesIO(raw)) as im:
                                im.load()
                                rgb = np.asarray(im)
                                hashes.add(hashlib.sha256(rgb.tobytes()).hexdigest())
                                if not np.any(rgb):
                                    flat["black"] += 1
                                elif np.all(rgb == 255):
                                    flat["white"] += 1
                            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if decoded is None or decoded.shape != rgb.shape:
                                raise ValueError("OpenCV could not decode RGB at the expected size")
                            result["images"] += 1
                        except (OSError, ValueError, RuntimeError, EOFError, cv2.error,
                                Image.DecompressionBombError, zipfile.BadZipFile) as exc:
                            errors.append(f"Unreadable image {name}: {exc}")
                    label = f"{group}/{camera}"
                    if flat:
                        warnings.append(f"{label}: {dict(flat)} uniform black/white frame(s)")
                    if len(hashes) == 1 and len(frames) > 1:
                        warnings.append(f"{label}: all frames have identical decoded RGB pixels")
                    result["camera_summary"][label] = {
                        "unique_decoded_frames": len(hashes), "uniform_frames": dict(flat)}
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
    report = {"task": "physical-properties-estimation",
              "scenes": {}, "errors": [], "warnings": []}
    extra = sorted(set(found) - set(scenes))
    if extra:
        report["errors"].append(f"Unexpected scene ZIPs: {', '.join(extra)}")
    for index, (scene, entry) in enumerate(scenes.items(), 1):
        paths = found.get(scene, [])
        if len(paths) != 1:
            message = "Missing scene ZIP" if not paths else "Duplicate scene ZIPs: " + ", ".join(map(str, paths))
            result = {"errors": [message], "warnings": [], "images": 0,
                      "parameters_valid": False, "camera_summary": {}}
        else:
            result = validate_zip(paths[0], entry)
        report["scenes"][scene] = result
        if progress:
            progress(index, len(scenes), scene, not result["errors"])
    good = [r for r in report["scenes"].values() if not r["errors"]]
    report["summary"] = {
        "expected_scenes": len(scenes), "passed": len(good), "failed": len(scenes)-len(good),
        "missing": sum(not found.get(s) for s in scenes), "extra": len(extra),
        "errors": len(report["errors"])+sum(len(r["errors"]) for r in report["scenes"].values()),
        "warnings": len(report["warnings"])+sum(len(r["warnings"]) for r in report["scenes"].values()),
        "parameter_sets_valid": sum(r["parameters_valid"] for r in report["scenes"].values()),
        "images_checked": sum(r["images"] for r in report["scenes"].values()),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_dir", type=Path, help="Folder containing one prediction ZIP per scene")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                        help="Submission manifest (bundled SinglePhysics 60-case release by default)")
    parser.add_argument("--report", type=Path, help="Write detailed JSON results to a new file")
    parser.add_argument("--verbose", action="store_true", help="Show every error and warning")
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat content warnings as failures")
    args = parser.parse_args(argv)
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
        print(f"Physical Properties Estimation | {len(manifest['scenes'])} scenes")
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
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
        failed = report["summary"]["errors"] or (args.fail_on_warning and report["summary"]["warnings"])
        print("INVALID" if failed else "VALID WITH WARNINGS" if report["summary"]["warnings"] else "VALID")
        return 1 if failed else 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
