"""Check user-facing launch commands without installing models or using an API."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ple_launcher', Path(__file__).resolve().parents[1] / 'scripts/run.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_evaluation_flags_are_forwarded_when_input_is_omitted(self):
        for forwarded in (['--tasks', 'P21', '--dry-run'],
                          ['--output', 'runs/custom', '--dry-run']):
            with self.subTest(forwarded=forwarded), patch.object(launcher.sys, 'platform', 'linux'), \
                    patch.object(launcher.os, 'chdir'), \
                    patch.object(launcher, 'evaluation_input', return_value=['--video-root', 'videos']) as inputs, \
                    patch.object(launcher, 'evaluation_python', return_value='/ready/bin/python'), \
                    patch.object(launcher, 'run') as run:
                self.assertEqual(launcher.main(['evaluate', *forwarded]), 0)
                inputs.assert_called_once_with('videos')
                self.assertEqual(run.call_args.args[0][-len(forwarded):], forwarded)

    def test_evaluation_prefers_manifest_and_accepts_paths_with_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            videos = Path(directory) / 'my videos'
            videos.mkdir()
            self.assertEqual(launcher.evaluation_input(videos), ['--video-root', videos])
            manifest = videos / 'manifest.json'
            manifest.write_text('[]')
            self.assertEqual(launcher.evaluation_input(videos), ['--manifest', manifest])
            self.assertEqual(launcher.evaluation_input(manifest), ['--manifest', manifest])
            with patch.object(launcher.sys, 'platform', 'linux'), patch.object(launcher.os, 'chdir'), \
                    patch.object(launcher, 'evaluation_python', return_value='/ready/bin/python') as provision, \
                    patch.object(launcher, 'run') as run:
                launcher.main(['evaluate', str(videos), '--tasks', 'P21', '--dry-run'])
                provision.assert_called_once_with(True)
                self.assertIn(manifest, run.call_args.args[0])
                self.assertEqual(run.call_args.args[0][-3:], ['--tasks', 'P21', '--dry-run'])

    def test_seedance_key_never_enters_config_or_command(self):
        key = 'fixture-private-credential'
        env = {'SEEDANCE_API_KEY': key, 'SEEDANCE_BASE_URL': 'https://provider.example/v1'}
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, env, clear=True), \
                patch.object(launcher.sys, 'platform', 'linux'), patch.object(launcher.os, 'chdir'), \
                patch.object(launcher, 'ROOT', Path(directory)), \
                patch.object(launcher, 'environment', return_value='/ready/bin/python'), \
                patch.object(launcher, 'run') as run:
            launcher.main(['seedance', '--tasks', 'P21', '--seeds', '42', '--dry-run'])
            contents = (Path(directory) / 'runs/launch/seedance-2.5.json').read_text()
            self.assertNotIn(key, contents)
            self.assertNotIn(key, str(run.call_args))
            profile = json.loads(contents)['models']['seedance-2.5']
            self.assertEqual(profile['options']['api_key_env'], 'SEEDANCE_API_KEY')
            self.assertEqual(profile['options']['request_format'], 'litellm_json')

    def test_invalid_input_fails_before_environment_installation(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher.sys, 'platform', 'linux'), \
                patch.object(launcher.os, 'chdir'), patch.object(launcher, 'evaluation_python') as provision:
            with self.assertRaises(ValueError):
                launcher.main(['evaluate', str(Path(directory) / 'missing')])
            provision.assert_not_called()

    def test_evaluation_setup_is_cached_and_missing_weights_trigger_repair(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), \
                patch.object(launcher, 'ROOT', Path(directory)), \
                patch.object(launcher, 'run') as run:
            root = Path(directory)
            (root / 'setup.sh').write_text('installer fixture')
            (root / 'requirements.txt').write_text('dependencies fixture')
            python = root / '.venv/eval/bin/python'
            python.parent.mkdir(parents=True)
            expected = ('Qwen3.6-27B/config.json', 'grounding-dino-tiny/config.json',
                        'sam2.1-hiera-small/sam2.1_hiera_small.pt',
                        'sam2.1-hiera-small-transformers/config.json',
                        'sam2.1-hiera-large-transformers/config.json',
                        'cotracker3/scaled_offline.pth', 'cotracker3/source/hubconf.py')
            for name in expected:
                path = root / 'models' / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('model fixture')
            with patch.object(launcher, 'environment', return_value=python):
                self.assertEqual(launcher.evaluation_python(False), python)
                self.assertEqual(run.call_count, 1)
                self.assertTrue(run.call_args.kwargs['env']['PATH'].startswith(str(python.parent)))
                launcher.evaluation_python(False)
                self.assertEqual(run.call_count, 1)
                (root / 'models' / expected[0]).unlink()
                launcher.evaluation_python(False)
                self.assertEqual(run.call_count, 2)


if __name__ == '__main__':
    unittest.main()
