"""Batch orchestration tests; fixture scores are not benchmark measurements."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate as batch


FIXTURE_EVALUATOR = '''
import argparse, hashlib, json, pathlib, sys, time
p=argparse.ArgumentParser()
for name in ('video','image','output','sample-id','model','fixture-mode'):
    p.add_argument('--'+name)
p.add_argument('--seed', type=int)
a, rest=p.parse_known_args()
if a.fixture_mode == 'crash':
    raise SystemExit(3)
if a.fixture_mode == 'slow':
    time.sleep(30)
rejected=a.fixture_mode == 'reject'
measured=not rejected
score=.3 if rejected else 1.
physics=None if rejected else .5
task=pathlib.Path(__file__).resolve().parents[1].name
metric=lambda key,defined: dict(id=key,defined=defined,status=('measured' if measured else 'not_run') if defined else 'not_applicable',measurement_attempted=defined and measured,raw_value=.5 if defined and measured else None,physics_score=.5 if defined and measured else None)
data=dict(schema_version='vdmbench.evaluation.v4',task=dict(id=task),sample=dict(id=a.sample_id,model=a.model,video_sha256=hashlib.sha256(pathlib.Path(a.video).read_bytes()).hexdigest(),image_sha256=hashlib.sha256(pathlib.Path(a.image).read_bytes()).hexdigest()),status='consistency_rejected' if rejected else 'complete',consistency=dict(status='evaluated',score=score,threshold=.8,passed=measured,details={}),physics=dict(attempted=measured,status='complete' if measured else 'not_run',score=physics,judgment='physics_fail' if measured else 'evidence_insufficient',defined_metrics=1,measured_metrics=int(measured),coverage=float(measured)),metrics=dict(M1=metric('M1',True),M2=metric('M2',False)),score=dict(total=.15*score+(.85*physics if measured else 0)),failures=[],provenance={})
if a.fixture_mode == 'wrong-input':
    data['sample']['video_sha256']='wrong'
data['sample']['seed'] = a.seed
if a.fixture_mode == 'wrong-seed':
    data['sample']['seed'] = 999
pathlib.Path(a.output).write_text(json.dumps(data))
raise SystemExit(1 if rejected else 0)
'''


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.folder = self.root / 'P19'
        evaluator = self.folder / 'evaluator'
        evaluator.mkdir(parents=True)
        (evaluator / 'evaluate.py').write_text(FIXTURE_EVALUATOR)
        self.image = self.folder / 'first_frame.png'
        self.image.write_bytes(b'fixture image')
        (self.folder / 'prompt.txt').write_text('Fixture generation prompt.')
        self.tasks = {'P19': dict(path=str(self.folder), difficulty='easy', group='g2')}
        self.video = self.root / 'g2_P19_seed42.mp4'
        self.video.write_bytes(b'fixture video')
        self.validator = batch.load_module('batch_test_schema', batch.ROOT / 'easy/P19/evaluator/_shared/unified_evaluators/schema_v4.py').validate
        self.output = self.root / 'output'
        self.config = dict(python=sys.executable, gate={}, measurement_device=None, timeout=10, task_fingerprints={'P19': 'fixture'})

    def tearDown(self):
        self.temporary.cleanup()

    def job(self, mode=None):
        row = dict(task='P19', model='test-model', video=str(self.video), seed=42)
        if mode:
            row['backend_args'] = ['--fixture-mode', mode]
        return batch.prepare_jobs([row], self.root, self.tasks)[0][0]

    def run_job(self, job, resume=False, config=None):
        return batch.evaluate_job(job, self.output, config or self.config, resume, self.validator,
                                  set(), threading.Lock(), threading.Event())

    def test_all_forty_current_task_packages(self):
        tasks = batch.discover_tasks()
        self.assertEqual(len(tasks), 40)
        self.assertEqual({d: sum(v['difficulty'] == d for v in tasks.values()) for d in ('easy', 'medium', 'hard')},
                         {'easy': 15, 'medium': 15, 'hard': 10})

    def test_directory_scan_deduplicates_only_identical_sample_inputs(self):
        root = self.root / 'videos'
        directory = root / 'lingbot' / 'gpt'
        directory.mkdir(parents=True)
        (directory / self.video.name).write_bytes(self.video.read_bytes())
        nested = directory / self.video.stem
        nested.mkdir()
        (nested / 'base.mp4').write_bytes(self.video.read_bytes())
        rows = batch.scan_videos(root, self.tasks)
        jobs, duplicates = batch.prepare_jobs(rows, root, self.tasks)
        self.assertEqual((len(jobs), len(duplicates)), (1, 1))
        self.assertEqual(jobs[0]['seed'], 42)
        generation_config = directory / (self.video.stem + '_config.json')
        generation_config.write_text(json.dumps({'arguments': {'prompt': 'Actual generation prompt.'}}))
        (nested / 'base_config.json').write_bytes(generation_config.read_bytes())
        rows = batch.scan_videos(root, self.tasks)
        jobs, duplicates = batch.prepare_jobs(rows, root, self.tasks)
        self.assertEqual((len(jobs), len(duplicates)), (1, 1))
        self.assertEqual(jobs[0]['files']['prompt']['sha256'], batch.sha(generation_config))
        (nested / 'base.mp4').write_bytes(b'different video')
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'):
            batch.prepare_jobs(rows, root, self.tasks)

    def test_relative_manifest_and_annotation_paths(self):
        annotation = self.root / 'annotation.json'
        annotation.write_text('{}')
        prompt = self.root / 'actual.txt'
        prompt.write_text('The actual per-video prompt.')
        rows = [dict(task='P19', model='test-model', video=self.video.name,
                     annotation=annotation.name, prompt=prompt.name, sample_id='custom_42')]
        jobs, _ = batch.prepare_jobs(rows, self.root, self.tasks)
        job = jobs[0]
        self.assertEqual(job['video'], str(self.video))
        self.assertEqual(job['prompt'], str(prompt))
        self.assertEqual(job['annotation'], str(annotation))
        command = batch.command_for(job, self.output, sys.executable, {'backend': 'http', 'model': 'vlm'})
        self.assertEqual(command[command.index('--annotation') + 1], str(annotation))
        self.assertEqual(command[command.index('--video') + 1], str(self.video))

    def test_unknown_names_and_output_overrides_are_rejected(self):
        with self.assertRaises(ValueError):
            batch.make_job(dict(task='P19', model='../escape', video=self.video.name), self.root, self.tasks)
        with self.assertRaisesRegex(ValueError, 'cannot override'):
            batch.make_job(dict(task='P19', model='model', video=self.video.name,
                                backend_args=['--output=/tmp/other.json']), self.root, self.tasks)

    def test_completed_and_gate_rejected_results_are_both_valid(self):
        completed = self.run_job(self.job())
        rejected = self.run_job(self.job('reject'))
        self.assertEqual(completed['status'], 'complete')
        self.assertAlmostEqual(completed['total'], .575)
        self.assertEqual(rejected['status'], 'consistency_rejected')
        self.assertAlmostEqual(rejected['total'], .045)
        self.assertIsNone(rejected['physics_score'])
        self.assertEqual(rejected['exit_code'], 1)

    def test_errors_never_enter_means_as_zero(self):
        completed = self.run_job(self.job())
        crash = self.run_job(self.job('crash'))
        wrong_input = self.run_job(self.job('wrong-input'))
        wrong_seed = self.run_job(self.job('wrong-seed'))
        self.assertEqual(crash['status'], 'execution_error')
        self.assertEqual(wrong_input['status'], 'execution_error')
        self.assertEqual(wrong_seed['status'], 'execution_error')
        self.assertIsNone(crash['total'])
        self.assertIsNone(wrong_input['total'])
        values = batch.aggregate([completed, crash, wrong_input, wrong_seed])
        self.assertEqual((values['videos'], values['scored'], values['errors']), (4, 1, 3))
        self.assertAlmostEqual(values['total_mean'], .575)

    def test_resume_checks_inputs_settings_and_result_integrity(self):
        job = self.job()
        first = self.run_job(job)
        self.assertTrue(self.run_job(job, resume=True)['resumed'])
        Path(first['result_path']).write_text('{}')
        second = self.run_job(job, resume=True)
        self.assertFalse(second['resumed'])
        self.assertNotEqual(second['result_path'], first['result_path'])
        self.video.write_bytes(b'changed video')
        third = self.run_job(self.job(), resume=True)
        self.assertFalse(third['resumed'])
        fourth = self.run_job(self.job(), resume=True, config={**self.config, 'task_fingerprints': {'P19': 'new code'}})
        self.assertFalse(fourth['resumed'])
        self.assertTrue(Path(first['result_path']).exists())

    def test_missing_input_and_timeout_are_explicit_errors(self):
        self.video.unlink()
        missing = self.run_job(self.job())
        self.assertEqual(missing['status'], 'input_error')
        self.assertIsNone(missing['total'])
        self.video.write_bytes(b'fixture video')
        timed_out = self.run_job(self.job('slow'), config={**self.config, 'timeout': .2})
        self.assertEqual(timed_out['status'], 'execution_error')
        self.assertIn('exceeded', timed_out['error'])
        self.assertIsNone(timed_out['total'])

    def test_cli_continues_after_a_failed_video_and_writes_summary(self):
        shared = self.folder / 'evaluator/_shared/unified_evaluators'
        shared.mkdir(parents=True)
        actual_shared = batch.ROOT / 'easy/P19/evaluator/_shared/unified_evaluators'
        for name in ('consistency.py', 'schema_v4.py'):
            (shared / name).write_bytes((actual_shared / name).read_bytes())
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps([
            dict(task='P19', model='test-model', sample_id='good', video=str(self.video)),
            dict(task='P19', model='test-model', sample_id='bad', video=str(self.video), backend_args=['--fixture-mode', 'crash'])]))
        argv = ['--manifest', str(manifest), '--output', str(self.output), '--workers', '2', '--require-all-tasks']
        with patch.object(batch, 'discover_tasks', return_value=self.tasks), redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(argv), 2)
        summary = batch.read(self.output / 'summary.json')
        self.assertEqual((summary['finished'], summary['overall']['errors']), (2, 1))
        self.assertAlmostEqual(summary['overall']['total_mean'], .575)
        self.assertTrue((self.output / 'by_model.csv').is_file())

    def test_require_all_tasks_detects_partial_input_before_inference(self):
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps([dict(task='P19', model='test-model', video=str(self.video))]))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            batch.main(['--manifest', str(manifest), '--output', str(self.output), '--require-all-tasks', '--dry-run'])
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_quickstart_models_and_explicit_overrides_reach_child_process(self):
        shared = self.folder / 'evaluator/_shared/unified_evaluators'
        shared.mkdir(parents=True)
        actual = batch.ROOT / 'easy/P19/evaluator/_shared/unified_evaluators'
        for name in ('consistency.py', 'schema_v4.py'):
            (shared / name).write_bytes((actual / name).read_bytes())
        # Record the actual command/environment received by the child process.
        evaluator = self.folder / 'evaluator/evaluate.py'
        evaluator.write_text(FIXTURE_EVALUATOR.replace(
            "raise SystemExit(1 if rejected else 0)",
            "import os\npathlib.Path(a.output).with_name('received.json').write_text(json.dumps(dict(argv=sys.argv, models=os.environ.get('FINAL_MODELS_DIR'))))\nraise SystemExit(1 if rejected else 0)"))
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps([dict(task='P19', model='test-model', video=str(self.video))]))
        cases = [
            ({}, [], str(batch.ROOT / 'models'), str(batch.ROOT / 'models/Qwen3.6-27B'), 'local', 'cuda:0'),
            ({'FINAL_MODELS_DIR': 'custom-models', 'VLM_MODEL': 'existing-server', 'VLM_BACKEND': 'http',
              'VLM_BASE_URL': 'http://localhost:8000/v1'}, [], str(Path('custom-models').resolve()), 'existing-server', 'http', 'cuda:0'),
            ({'VLM_MODEL': 'environment-model'}, ['--consistency-model', '/explicit/model', '--measurement-device', 'cpu'],
             str(batch.ROOT / 'models'), '/explicit/model', 'local', 'cpu'),
        ]
        for index, (environment, flags, root, model, backend, device) in enumerate(cases):
            output = self.root / f'case{index}'
            with self.subTest(index=index), patch.dict(batch.os.environ, environment, clear=True), \
                    patch.object(batch, 'discover_tasks', return_value=self.tasks), redirect_stdout(io.StringIO()):
                self.assertEqual(batch.main(['--manifest', str(manifest), '--output', str(output), *flags]), 0)
            config = batch.read(next((output / 'configurations').glob('*.json')))
            received = batch.read(next(output.glob('test-model/P19/*/attempt*/received.json')))
            self.assertEqual(received['models'], root)
            self.assertEqual(received['argv'][received['argv'].index('--consistency-model') + 1], model)
            self.assertEqual(config['gate']['backend'], backend)
            self.assertEqual(config['measurement_device'], device)


if __name__ == '__main__':
    unittest.main()
