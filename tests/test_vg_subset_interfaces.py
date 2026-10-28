"""Offline CLI/index-routing tests; no models, GPU, or network required."""
import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VG = ROOT / 'baselines/Physics-aware_Video_Generation'
spec = importlib.util.spec_from_file_location('vg_submission_check', VG / 'check.py')
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)


def isolated_functions(path, names, namespace, parser_only=False):
    """Load actual source functions without importing diffusion dependencies."""
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(tree.body) == len(names)
    if parser_only:
        parser = next(n for n in tree.body if n.name == 'parse_args')
        # Retain the real argument definitions and the availability gate, but
        # stop before hardware/configuration handling (which is not under test).
        stop = next(i for i,n in enumerate(parser.body) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'args.batch_size <= 0')
        parser.body = parser.body[:stop] + [ast.Return(ast.Name('args', ast.Load()))]
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace


class SubsetInterfaceTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {}, clear=True)
        patch.start(); self.addCleanup(patch.stop)

    def parse_check(self, *args):
        with mock.patch.object(sys, 'argv', ['check.py', '/tmp/submissions', *args]):
            return checker.parse_args()

    def test_checker_defaults_and_large_index(self):
        self.assertEqual(self.parse_check().subset, 'mini')
        self.assertEqual(self.parse_check().json_path.name, 'selected_mini.json')
        args = self.parse_check('--subset', 'large')
        self.assertEqual(args.subset, 'large')
        self.assertEqual(args.json_path, VG / 'dataset/selected_full.json')

    def test_checker_rejects_full(self):
        with contextlib.redirect_stderr(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as raised:
                self.parse_check('--subset', 'full')
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid choice: 'full'", out.getvalue())

    def test_explicit_manifest_then_environment_override(self):
        with mock.patch.dict(os.environ, {'SUBMISSION_MANIFEST': '/tmp/env.json'}):
            self.assertEqual(self.parse_check('--subset', 'large').json_path, Path('/tmp/env.json'))
            self.assertEqual(self.parse_check('--subset', 'large', '--manifest', '/tmp/custom.json').json_path,
                             Path('/tmp/custom.json'))

    def test_checker_rejects_small(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse_check('--subset', 'small')

    def test_checker_main_uses_selected_large_not_mini(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); submissions = root / 'zips'; submissions.mkdir()
            mini = root / 'selected_mini.json'
            mini.write_text(json.dumps({'different_mini_scene': ['CineCamera_2', 1]}))
            (root / 'selected_full.json').write_text(json.dumps({'large_scene': ['CineCamera_5', 1]}))
            image = io.BytesIO(); Image.new('RGB', (16, 16), 'red').save(image, format='JPEG')
            with zipfile.ZipFile(submissions / 'large_scene.zip', 'w') as archive:
                archive.writestr('CineCamera_5/rgb/0000.jpg', image.getvalue())
            with mock.patch.object(checker, 'DEFAULT_JSON_PATH', str(mini)), \
                 mock.patch.object(sys, 'argv', ['check.py', str(submissions), 'static', '--subset', 'large']), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(checker.main(), 0)

    def test_missing_large_index_never_falls_back_to_mini(self):
        with tempfile.TemporaryDirectory() as directory:
            mini = Path(directory) / 'selected_mini.json'; mini.write_text('{}')
            with mock.patch.object(checker, 'DEFAULT_JSON_PATH', str(mini)):
                args = self.parse_check('--subset', 'large')
                self.assertFalse(args.json_path.exists())
                self.assertNotEqual(args.json_path, mini)

    def test_dataset_large_maps_to_legacy_filename(self):
        ns = isolated_functions(VG / 'dataset/PhysInOne_Dataset.py', {'leaderboard_selection_path'},
                                {'Path': Path, 'SELECTED': '/example/selected_full.json'})
        choose = ns['leaderboard_selection_path']
        self.assertEqual(choose('large'), '/example/selected_full.json')
        with self.assertRaises(ValueError): choose('full')
        self.assertEqual(choose('mini'), '/example/selected_mini.json')
        with self.assertRaises(ValueError): choose('small')

    def test_inference_parser_uses_same_subset_contract(self):
        ns = isolated_functions(VG / 'inference.py', {'parse_args'},
                                {'argparse': argparse, 'sys': sys}, parser_only=True)
        argv = ['inference.py', '--config', 'unused.yaml', '--data_path', '/tmp/unused', '--leaderboard']
        with mock.patch.object(sys, 'argv', argv):
            self.assertEqual(ns['parse_args']().subset, 'mini')
        for value in ('large', 'full', 'small'):
            with self.subTest(value=value), mock.patch.object(sys, 'argv', argv + ['--subset', value]), \
                 contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised: ns['parse_args']()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('large Video Generation subset is not available yet' if value == 'large' else 'invalid choice',
                              err.getvalue())


if __name__ == '__main__':
    unittest.main()
