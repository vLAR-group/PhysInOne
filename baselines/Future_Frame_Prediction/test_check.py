"""Offline regression tests: python -m unittest discover -s this_directory."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

from PIL import Image

SPEC = importlib.util.spec_from_file_location("fp_check", Path(__file__).with_name("check.py"))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = {"training_cameras": ["CineCamera_2", "CineCamera_7"],
                      "novel_cameras": ["CineCamera_0", "CineCamera_4"],
                      "frame_stems": ["0075", "0076", "0077"], "image_size": [16, 16]}
        self.manifest = {"schema_version": 1, "release": "fixture", "scenes": {"scene_trajectory": self.entry}}

    def members(self, novel=False, ext=".png", flat=False):
        cameras = self.entry["training_cameras"] + (self.entry["novel_cameras"] if novel else [])
        result = {}
        for camera in cameras:
            for i, stem in enumerate(self.entry["frame_stems"]):
                im = Image.new("RGB", (16, 16), (255, 255, 255) if flat else (30+i*40, 60, 100))
                buf = io.BytesIO()
                im.save(buf, format=CHECK.EXTENSIONS[ext])
                result[f"{camera}/rgb/{stem}{ext}"] = buf.getvalue()
        return result

    def write_zip(self, members, name="scene_trajectory.zip"):
        path = self.root / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, raw in members.items():
                archive.writestr(member, raw)
        return path

    def check(self, members):
        return CHECK.validate_zip(self.write_zip(members), self.entry)

    def test_png_training_only(self):
        result = self.check(self.members())
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["images"], 6)
        self.assertFalse(result["novel_views"])

    def test_jpeg_both_groups(self):
        result = self.check(self.members(novel=True, ext=".jpeg"))
        self.assertEqual(result["errors"], [])
        self.assertTrue(result["novel_views"])

    def test_mixed_extensions(self):
        members = self.members()
        jpeg = self.members(ext=".jpg")
        del members["CineCamera_2/rgb/0075.png"]
        members["CineCamera_2/rgb/0075.jpg"] = jpeg["CineCamera_2/rgb/0075.jpg"]
        self.assertEqual(self.check(members)["errors"], [])

    def test_wrapper(self):
        self.assertEqual(self.check({"scene_trajectory/"+k: v for k, v in self.members().items()})["errors"], [])

    def test_missing_and_wrong_frame_ids(self):
        members = self.members()
        members["CineCamera_2/rgb/0000.png"] = members.pop("CineCamera_2/rgb/0075.png")
        result = self.check(members)
        self.assertTrue(any("Missing" in e for e in result["errors"]))
        self.assertTrue(any("Unexpected" in e for e in result["errors"]))

    def test_partial_novel_rejected(self):
        members = {k: v for k, v in self.members(novel=True).items() if not k.startswith("CineCamera_4/")}
        self.assertTrue(self.check(members)["errors"])

    def test_empty_novel_rejected(self):
        members = self.members()
        members["CineCamera_0/"] = b""
        self.assertTrue(self.check(members)["errors"])

    def test_extras_rejected(self):
        for name in ("weights.pt", "caption.txt", "CineCamera_Moving/rgb/0075.png"):
            with self.subTest(name=name):
                members = self.members()
                members[name] = b"extra"
                self.assertTrue(self.check(members)["errors"])

    def test_duplicate_stem_rejected(self):
        members = self.members()
        members["CineCamera_2/rgb/0075.jpg"] = self.members(ext=".jpg")["CineCamera_2/rgb/0075.jpg"]
        self.assertTrue(any("Duplicate frame" in e for e in self.check(members)["errors"]))

    def test_corrupt_image_and_wrong_extension(self):
        for raw in (b"", b"broken image", self.members()["CineCamera_2/rgb/0075.png"]):
            with self.subTest(size=len(raw)):
                members = self.members(ext=".jpg")
                members["CineCamera_2/rgb/0075.jpg"] = raw
                self.assertTrue(self.check(members)["errors"])

    def test_wrong_resolution(self):
        self.entry["image_size"] = [32, 32]
        self.assertTrue(self.check(self.members())["errors"])

    def test_truncated_jpeg(self):
        members = self.members(ext=".jpg")
        key = "CineCamera_2/rgb/0075.jpg"
        members[key] = members[key][:-20]
        self.assertTrue(self.check(members)["errors"])

    def test_broken_deflate_returns_error(self):
        path = self.write_zip(self.members())
        with patch.object(zipfile.ZipFile, "testzip", side_effect=zlib.error("invalid compressed data")):
            self.assertTrue(CHECK.validate_zip(path, self.entry)["errors"])

    def test_content_warning(self):
        result = self.check(self.members(flat=True))
        self.assertEqual(result["errors"], [])
        self.assertTrue(any("white" in w for w in result["warnings"]))
        self.assertTrue(any("identical" in w for w in result["warnings"]))

    def test_unsafe_paths(self):
        for name in ("../evil", "/absolute", "C:\\evil"):
            with self.subTest(name=name):
                members = self.members()
                members[name] = b"evil"
                self.assertTrue(self.check(members)["errors"])

    def test_symlink_rejected(self):
        path = self.write_zip(self.members())
        with zipfile.ZipFile(path, "a") as archive:
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = 0o120777 << 16
            archive.writestr(info, "outside")
        self.assertTrue(CHECK.validate_zip(path, self.entry)["errors"])

    def test_duplicate_zip_member(self):
        path = self.write_zip(self.members())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(path, "a") as archive:
                archive.writestr("CineCamera_2/rgb/0075.png", b"duplicate")
        self.assertTrue(CHECK.validate_zip(path, self.entry)["errors"])

    def test_crc_failure(self):
        members = self.members()
        path = self.write_zip(members)
        raw = path.read_bytes()
        needle = next(iter(members.values()))
        start = raw.index(needle)
        corrupted = bytearray(raw)
        corrupted[start+40] ^= 1
        path.write_bytes(corrupted)
        self.assertTrue(CHECK.validate_zip(path, self.entry)["errors"])

    def test_inventory_missing_extra_duplicate(self):
        self.assertEqual(CHECK.check_submission(self.root, self.manifest)["summary"]["missing"], 1)
        self.write_zip(self.members())
        self.write_zip(self.members(), "unrelated.zip")
        (self.root/"nested").mkdir()
        self.write_zip(self.members(), "nested/scene_trajectory.zip")
        report = CHECK.check_submission(self.root, self.manifest)
        self.assertEqual(report["summary"]["extra"], 1)
        self.assertTrue(report["scenes"]["scene_trajectory"]["errors"])

    def test_mixed_novel_warning(self):
        self.write_zip(self.members())
        self.write_zip(self.members(novel=True), "another.zip")
        self.manifest["scenes"]["another"] = copy.deepcopy(self.entry)
        report = CHECK.check_submission(self.root, self.manifest)
        self.assertEqual(report["summary"]["errors"], 0)
        self.assertTrue(report["warnings"])

    def test_manifest_validation(self):
        path = self.root/"manifest.json"
        path.write_text(json.dumps(self.manifest))
        self.assertEqual(CHECK.load_manifest(path), self.manifest)
        self.entry["novel_cameras"] = ["CineCamera_2"]
        path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            CHECK.load_manifest(path)
        path.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaises(ValueError):
            CHECK.load_manifest(path)

    def test_cli_exit_report_read_only(self):
        archive = self.write_zip(self.members(flat=True))
        before = archive.read_bytes()
        manifest = self.root/"manifest.json"
        manifest.write_text(json.dumps(self.manifest))
        cmd = [sys.executable, str(Path(CHECK.__file__)), str(self.root), "--manifest", str(manifest)]
        self.assertEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(cmd+["--fail-on-warning"], capture_output=True).returncode, 1)
        report = self.root/"report.json"
        self.assertEqual(subprocess.run(cmd+["--report", str(report)], capture_output=True).returncode, 0)
        self.assertEqual(subprocess.run(cmd+["--report", str(report)], capture_output=True).returncode, 2)
        self.assertEqual(archive.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
