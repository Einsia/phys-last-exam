"""Batch entrypoints must resolve data/metadata.json from the task root."""
import contextlib
import datetime
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BatchSchemaPaths(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.root = Path(self.workspace.name)
        for group, task_id in [('g2', 'P19'), ('g7', 'P7')]:
            task = self.root / group / task_id
            (task / 'data').mkdir(parents=True)
            (task / 'evaluator').mkdir()
            (task / 'evaluator' / 'evaluate.py').write_text('# original backend\n')
            (task / 'data' / 'metadata.json').write_text(json.dumps({'samples': [{
                'sample_id': 'sample_00', 'model': 'minimax_h3',
                'video_path': 'output_videos/minimax_h3/sample_00.mp4',
            }]}))

    def test_batch_resolves_task_group_and_video_outside_data(self):
        runner = script('run_all_eval')
        commands = []

        def evaluate(command, **kwargs):
            commands.append(command)
            output = Path(command[command.index('--output') + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                'metrics': {}, 'verbose': {'M1': {
                    '_provenance': {'run_started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()},
                    '_scoring_summary': {'score_status': 'complete'},
                }},
            }))
            return subprocess.CompletedProcess(command, 0)

        with patch.object(runner, 'ROOT', self.root), patch.object(runner.subprocess, 'run', side_effect=evaluate), contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(['--groups', 'g2,g7', '--samples', 'sample_00',
                                '--model', 'minimax_h3', '--output-root', str(self.root / 'results'),
                                '--log-dir', str(self.root / 'logs')])
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 2)
        for group, task_id in [('g2', 'P19'), ('g7', 'P7')]:
            task = self.root / group / task_id
            command = next(cmd for cmd in commands if cmd[1] == str(task / 'evaluator' / 'evaluate.py'))
            self.assertEqual(command[command.index('--video') + 1], str(task / 'output_videos/minimax_h3/sample_00.mp4'))
            self.assertEqual(command[command.index('--output') + 1], str(self.root / 'results' / group / task_id / 'eval_results/minimax_h3/result_sample_00.json'))

    def test_installer_writes_entries_at_task_root(self):
        installer = script('install_unified_entries')
        schema = self.root / 'shared' / 'physeval' / 'schema.py'
        schema.parent.mkdir(parents=True)
        schema.write_text('# schema placeholder\n')
        with patch.object(installer, 'ROOT', self.root), contextlib.redirect_stdout(io.StringIO()):
            installer.main()
        for group, task_id in [('g2', 'P19'), ('g7', 'P7')]:
            task = self.root / group / task_id
            entry = (task / 'evaluator' / 'evaluate.py').read_text()
            shell = (task / 'scripts' / 'run_eval.sh').read_text()
            self.assertIn(f"main('{task_id}',", entry)
            self.assertIn(f'--groups {group} --tasks {task_id}', shell)
            self.assertEqual((task / 'evaluator' / 'measure_backend.py').read_text(), '# original backend\n')
            self.assertFalse((task / 'data' / 'evaluator').exists())


if __name__ == '__main__':
    unittest.main()
