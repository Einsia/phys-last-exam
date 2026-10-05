"""Generation integration tests use explicit synthetic fixtures, not benchmark samples."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import base64
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
import generate
from generation.backends import REGISTRY, Seedance25Video
from generation.config import MODELS, SCENES, default_config, resolve_profile
from generation.media import inspect_video, prepare_video
from generation.worker import seedance_generate


FIXTURE = '''import argparse, json, pathlib, time
import av, numpy as np
p=argparse.ArgumentParser()
p.add_argument('--request',required=True)
a=p.parse_args()
r=json.loads(pathlib.Path(a.request).read_text())
if r['seed'] == 99: raise SystemExit(3)
if r['seed'] == 98: time.sleep(30)
output=pathlib.Path(r['output'])
with av.open(str(output),'w',format='mp4') as c:
    s=c.add_stream('libx264', rate=10)
    s.width,s.height,s.pix_fmt=112,64,'yuv420p'
    for i in range(4):
        x=np.full((64,112,3),i*50,dtype=np.uint8)
        f=av.VideoFrame.from_ndarray(x,format='rgb24')
        for packet in s.encode(f): c.mux(packet)
    for packet in s.encode(): c.mux(packet)
output.with_suffix('.received.json').write_text(json.dumps(r))
'''


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'fixture generator.py'
        self.script.write_text(FIXTURE)
        self.output, self.work = self.root / 'videos', self.root / 'work'
        self.config = self.root / 'config.json'
        self.config.write_text(json.dumps({'models': {'fixture': {
            'backend': 'command', 'command': [sys.executable, str(self.script), '--request', '{request}'],
            'num_frames': 4, 'cwd': str(self.root)}}}))

    def run_cli(self, *extra):
        with redirect_stdout(io.StringIO()):
            return generate.main(['--models', 'fixture', '--config', str(self.config),
                                  '--output', str(self.output), '--work-dir', str(self.work), *extra])

    def test_all_eight_models_forty_tasks_four_seeds_and_special_names(self):
        profiles = {k: resolve_profile(k, v, self.root) for k, v in default_config(self.root)['models'].items()}
        jobs = generate.plan_jobs(evaluate.discover_tasks(), MODELS, profiles, [42, 43, 44, 45], self.output, 'fixture')
        self.assertEqual(len(jobs), 1280)
        self.assertEqual(len({j['video'] for j in jobs}), 1280)
        for job in jobs:
            if job['task'] in SCENES:
                self.assertTrue(Path(job['video']).stem.startswith(SCENES[job['task']] + '_seed'))
        self.assertEqual({j['task'] for j in jobs if j['requires_annotation']}, {'P37','P38','P39','P41','P43','P49'})

    def test_dry_run_never_starts_a_worker_or_checks_a_paid_api(self):
        with patch('generate.check_profile', side_effect=AssertionError('must not check credentials')), \
             patch('generate.subprocess.Popen', side_effect=AssertionError('must not launch')):
            self.assertEqual(self.run_cli('--tasks', 'P19', '--seeds', '42', '--dry-run'), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(evaluate.read(self.work / 'plan.json')['jobs']), 1)

    def test_python_symlink_keeps_its_model_virtual_environment(self):
        environment = self.root / 'model-env'
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        profile = resolve_profile('cog', dict(backend='cogvideox1.5-5b-i2v', options=dict(
            proj='.', model_dir='.', python_bin='model-env/bin/python')), self.root)
        prefix = subprocess.check_output([profile['options']['python_bin'], '-c', 'import sys; print(sys.prefix)'], text=True).strip()
        self.assertEqual(Path(prefix), environment)

    def test_generation_export_manifest_evaluator_and_resume(self):
        self.assertEqual(self.run_cli('--tasks', 'P3', 'P19', 'P37', '--seeds', '42'), 0)
        manifest = evaluate.read(self.output / 'manifest.json')
        self.assertEqual(len(manifest), 3)
        self.assertTrue(all(row['image'].startswith('.inputs/') and row['prompt'].startswith('.inputs/') for row in manifest))
        annotated = next(row for row in manifest if row['task'] == 'P37')
        self.assertTrue(annotated['annotation'].startswith('.inputs/'))
        template = evaluate.read(self.output / annotated['annotation'])
        self.assertEqual(template['annotation_type'], 'task_first_frame_template')
        self.assertNotIn('source_video_sha256', template)
        jobs, duplicates = evaluate.prepare_jobs(manifest, self.output, evaluate.discover_tasks())
        self.assertFalse(duplicates)
        self.assertTrue(all(not j['input_errors'] for j in jobs))
        self.assertEqual(len(evaluate.scan_videos(self.output, evaluate.discover_tasks())), 3)
        calibrated = self.output / 'fixture/P3_gpt_01_modern_seed42.mp4'
        info = inspect_video(calibrated)
        self.assertEqual((info['width'], info['height'], info['frames']), (1344, 768, 4))
        meta = evaluate.read(calibrated.with_name(calibrated.stem + '_config.json'))
        self.assertEqual(meta['media']['transform'], 'resize_xy')
        self.assertEqual(meta['media']['raw']['frames'], meta['media']['exported']['frames'])
        before = {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*.mp4')}
        self.assertEqual(self.run_cli('--tasks', 'P3', 'P19', 'P37', '--seeds', '42', '--resume'), 0)
        self.assertEqual(before, {str(p): p.stat().st_mtime_ns for p in self.output.rglob('*.mp4')})
        self.assertTrue(all(r['resumed'] for r in evaluate.read(self.work / 'summary.json')['records']))
        self.assertEqual(evaluate.main(['--manifest', str(self.output / 'manifest.json'), '--output', str(self.root / 'eval'), '--dry-run']), 0)
        portable = self.root / 'moved videos'
        shutil.move(self.output, portable)
        moved, _ = evaluate.prepare_jobs(manifest, portable, evaluate.discover_tasks())
        self.assertTrue(all(not j['input_errors'] for j in moved))
        self.assertTrue(all(Path(j['image']).is_relative_to(portable) for j in moved))
        self.assertTrue(Path(next(j for j in moved if j['task'] == 'P37')['annotation']).is_relative_to(portable))

    def test_changed_settings_cannot_relabel_an_existing_video_or_manifest(self):
        self.assertEqual(self.run_cli('--tasks', 'P19', '--seeds', '42'), 0)
        before = (self.output / 'manifest.json').read_bytes()
        config = evaluate.read(self.config)
        config['models']['fixture']['num_frames'] = 5
        self.config.write_text(json.dumps(config))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli('--tasks', 'P19', '--seeds', '42', '--resume')
        self.assertEqual(before, (self.output / 'manifest.json').read_bytes())

    def test_failed_and_timed_out_generations_remain_in_the_evaluation_manifest(self):
        self.assertEqual(self.run_cli('--tasks', 'P19', '--seeds', '99', '42'), 2)
        self.assertEqual(len(evaluate.read(self.output / 'manifest.json')), 2)
        self.assertFalse((self.output / 'fixture/g2_P19_seed99.mp4').exists())
        self.assertTrue((self.output / 'fixture/g2_P19_seed42.mp4').is_file())
        self.assertEqual(self.run_cli('--tasks', 'P19', '--seeds', '98', '--timeout', '0.2'), 2)
        self.assertEqual(len(evaluate.read(self.output / 'manifest.json')), 3)

    def test_non_video_file_is_never_exported(self):
        raw = self.root / 'bad.mp4'
        raw.write_text('{"error":"provider failed"}')
        with self.assertRaises(Exception):
            prepare_video(raw, self.root / 'published.mp4', 'P19')
        self.assertFalse((self.root / 'published.mp4').exists())

    def test_wrong_video_container_is_not_accepted_as_mp4(self):
        import av
        import numpy as np
        raw = self.root / 'actually-matroska.mp4'
        with av.open(str(raw), 'w', format='matroska') as container:
            stream = container.add_stream('libx264', rate=10)
            stream.width, stream.height, stream.pix_fmt = 112, 64, 'yuv420p'
            for _ in range(2):
                frame = av.VideoFrame.from_ndarray(np.zeros((64,112,3), dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with self.assertRaisesRegex(ValueError, 'MP4 container'):
            prepare_video(raw, self.root / 'published.mp4', 'P19')
        self.assertFalse((self.root / 'published.mp4').exists())

    def test_command_argument_quotes_and_shell_syntax_are_passed_literally(self):
        from generation.worker import run
        profile = resolve_profile('custom', {'backend':'command', 'command':[sys.executable, 'adapter.py',
            '--image','{image}','--prompt','{prompt}','--seed','{seed}','--output','{output}']}, self.root)
        prompt = 'a "quoted" prompt; $(touch NEVER_EXECUTE) `echo literal` {braces}'
        request = dict(profile=profile, prompt=prompt, seed=42, image='first frame.png', raw_video=str(self.root / 'raw.mp4'))
        with patch('generation.worker.subprocess.run') as runner:
            run(request, self.root / 'request.json')
        command = runner.call_args.args[0]
        self.assertEqual(command[command.index('--prompt') + 1], prompt)
        self.assertNotIn('shell', runner.call_args.kwargs)

    def test_seedance_poll_failure_resumes_same_job_and_never_sends_seed(self):
        model = Seedance25Video(base_url='https://video.example', api_key_env='TEST_ONLY_KEY')
        state = self.root / 'remote.json'
        request = dict(signature='same-input', image='frame.png', prompt='A ball falls.', num_frames=120, seed=42)
        with patch.object(model, '_create_job', return_value=({'id':'job-1'}, 'job-1', '')) as create, \
             patch.object(model, '_poll_job', side_effect=[TimeoutError(), ({'id':'changed-id'}, '')]), \
             patch.object(model, '_download_result') as download:
            with self.assertRaises(TimeoutError):
                seedance_generate(model, request, self.root / 'raw.mp4', state)
            result = seedance_generate(model, request, self.root / 'raw.mp4', state)
            self.assertEqual(create.call_count, 1)
            self.assertEqual(download.call_args.args[0]['id'], 'job-1')
            self.assertFalse(result.meta['seed_sent'])
        state.write_text(json.dumps(dict(signature='same-input', status='submitting')))
        with patch.object(model, '_create_job') as create, self.assertRaisesRegex(RuntimeError, 'unknown'):
            seedance_generate(model, request, self.root / 'raw.mp4', state)
        create.assert_not_called()

    def test_seedance_http_contract_upload_poll_download_and_key_not_persisted(self):
        frame = self.root / 'frame.png'
        frame.write_bytes(b'fixture image bytes')
        for protocol in ('aihubmix_json', 'multipart'):
            with self.subTest(protocol=protocol):
                model = Seedance25Video(base_url='https://video.example/v1', api_key_env='TEST_ONLY_KEY', request_format=protocol)
                created = MagicMock(status_code=200)
                created.json.return_value = {'id': 'job / 1', 'status': 'queued'}
                completed = MagicMock(status_code=200)
                completed.json.return_value = {'id': 'provider-id', 'status': 'completed'}
                download = MagicMock(status_code=200)
                download.__enter__.return_value = download
                download.iter_content.return_value = [b'\x00\x00\x00\x20ftyp', b'fixture video bytes']
                def submit(url, **kwargs):
                    self.assertEqual(url, 'https://video.example/v1/videos')
                    self.assertEqual(kwargs['headers']['Authorization'], 'Bearer fixture-api-key')
                    body = kwargs.get('json', kwargs.get('data'))
                    self.assertEqual(body['prompt'], 'A ball falls.\nNo cuts.')
                    self.assertNotIn('seed', body)
                    self.assertNotIn('num_frames', body)
                    if protocol == 'aihubmix_json':
                        self.assertEqual(body['duration'], 5)
                        self.assertEqual(body['frame_images'][0]['image_url']['url'], 'data:image/png;base64,' + base64.b64encode(frame.read_bytes()).decode())
                    else:
                        self.assertEqual(body['seconds'], '5')
                        self.assertEqual(kwargs['files']['input_reference'][1].read(), frame.read_bytes())
                        self.assertNotIn('Content-Type', kwargs['headers'])
                    return created
                state, output = self.root / (protocol + '.json'), self.root / (protocol + '.mp4')
                request = dict(signature=protocol, image=str(frame), prompt='A ball falls.\nNo cuts.', num_frames=120, seed=42)
                with patch.dict(os.environ, {'TEST_ONLY_KEY': 'fixture-api-key'}), \
                     patch('generation.backends.requests.post', side_effect=submit) as post, \
                     patch('generation.backends.requests.get', side_effect=[completed, download]) as get:
                    result = seedance_generate(model, request, output, state)
                self.assertEqual(post.call_count, 1)
                self.assertEqual([call.args[0] for call in get.call_args_list], [
                    'https://video.example/v1/videos/job%20%2F%201',
                    'https://video.example/v1/videos/job%20%2F%201/content'])
                self.assertTrue(output.read_bytes().endswith(b'fixture video bytes'))
                self.assertFalse(result.meta['seed_sent'])
                self.assertNotIn('fixture-api-key', state.read_text() + json.dumps(asdict(result)))

    def test_native_adapter_calls_preserve_image_prompt_seed_and_frame_constraints(self):
        profiles = default_config(self.root)['models']
        for name in MODELS:
            if name == 'seedance-2.5':
                continue
            with self.subTest(model=name):
                model = REGISTRY[name](**profiles[name]['options'])
                output = self.root / (name + '.mp4')
                def fake_run(command, *, log_path=None):
                    self.assertIn('a prompt with spaces', command)
                    self.assertIn('43', command)
                    if name.startswith('lingbot'):
                        (output.with_suffix('') / 'base.mp4').write_bytes(b'fixture')
                    else:
                        output.write_bytes(b'fixture')
                    return 0, '', 0
                # LingBot uses a JSON prompt file; inspect that instead of argv.
                def lingbot_run(command, *, log_path=None):
                    payload=evaluate.read(output.with_suffix('') / 'prompt.json')
                    self.assertEqual(payload['caption'], 'a prompt with spaces')
                    self.assertEqual(payload['seed'], 43)
                    (output.with_suffix('') / 'base.mp4').write_bytes(b'fixture')
                    return 0, '', 0
                with patch.object(model, '_exists'), patch.object(Path, 'exists', return_value=True), \
                     patch.object(model, '_run', side_effect=lingbot_run if name.startswith('lingbot') else fake_run):
                    result=model.generate(str(self.root / 'frame.png'), 'a prompt with spaces', str(output), num_frames=81 if name!='minimax-h3' else 124, seed=43)
                self.assertEqual(result.path, str(output))


if __name__ == '__main__':
    unittest.main()
