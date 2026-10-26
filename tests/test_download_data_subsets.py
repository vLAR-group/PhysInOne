"""Offline regression tests for the public downloader's VG subset contract."""
import contextlib
from collections import defaultdict
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('public_download_data', ROOT / 'scripts/download_data.py')
downloader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = downloader
SPEC.loader.exec_module(downloader)


class VideoSubsetTests(unittest.TestCase):
    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = downloader.main(list(args))
        return code, output.getvalue()

    def test_default_equals_explicit_mini(self):
        args = ('--task', 'video-generation', '--output-dir', './data', '--list-only')
        default = self.run_cli(*args)
        explicit = self.run_cli(*args, '--subset', 'mini')
        self.assertEqual(default, explicit)
        self.assertEqual(default[0], 0)
        self.assertIn('video_generation (subset=mini): 4005 file(s)', default[1])
        self.assertEqual(sum(line.startswith('https://') for line in default[1].splitlines()), 4005)

    def test_full_fails_before_io(self):
        for task in ('video-generation', 'all', 'vg'):
            with self.subTest(task=task), mock.patch.object(downloader, 'read_manifest') as read, \
                 mock.patch.object(downloader.urllib.request, 'urlopen') as network, \
                 mock.patch.object(Path, 'mkdir') as mkdir, contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as raised:
                    downloader.main(['--task', task, '--subset', 'full', '--dry-run'])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn('not available yet', err.getvalue())
                read.assert_not_called(); network.assert_not_called(); mkdir.assert_not_called()

    def test_explicit_subset_rejected_for_unrelated_task(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                downloader.main(['--task', 'future-prediction', '--subset', 'mini', '--dry-run'])
        self.assertEqual(raised.exception.code, 2)

    def test_other_tasks_keep_their_manifests(self):
        for task in ('future_prediction', 'physical_properties_estimation', 'motion_transfer', '3d_assets'):
            with self.subTest(task=task):
                expected = len(downloader.read_manifest(downloader.TASKS[task]))
                code, text = self.run_cli('--task', task, '--dry-run')
                self.assertEqual(code, 0)
                self.assertIn(f'{task}: {expected} file(s)', text)

    def test_all_and_multiple_tasks_use_mini(self):
        for tasks in [('all',), ('video-generation', 'motion-transfer')]:
            code, text = self.run_cli('--task', *tasks, '--subset', 'mini', '--dry-run')
            self.assertEqual(code, 0)
            self.assertIn('video_generation (subset=mini): 4005 file(s)', text)

    def test_manifest_whitelist_and_compatibility_alias(self):
        spec = downloader.TASKS['video_generation']
        entries = downloader.read_manifest(spec)
        scenes = defaultdict(set)
        for path in entries:
            relative = path[len(spec.remote_prefix):]
            category, scene, member = relative.split('/', 2)
            self.assertIn(category, ('SinglePhysics', 'DoublePhysics', 'TriplePhysics'))
            scenes[scene].add(member)
        self.assertEqual(len(scenes), 801)
        for files in scenes.values():
            static = [p.split('/')[0] for p in files if p.endswith('/rgb/0000.jpg') and not p.startswith('CineCamera_Moving/')]
            self.assertEqual(len(static), 1)
            self.assertEqual(files, {static[0] + '/rgb/0000.jpg', 'CineCamera_Moving/rgb/0000.jpg',
                                    'blender_' + static[0] + '.json', 'blender_CineCamera_Moving.json', 'caption.txt'})
        self.assertEqual((downloader.LIST_ROOT / spec.manifest).read_bytes(),
                         (downloader.LIST_ROOT / 'video_generation.txt').read_bytes())

    def test_single_scene_filter_still_selects_five_files(self):
        code, text = self.run_cli('--task', 'video-generation', '--scene', 'opkpjr', '--dry-run')
        self.assertEqual(code, 0)
        self.assertIn('total: 5 file(s)', text)


if __name__ == '__main__':
    unittest.main()
