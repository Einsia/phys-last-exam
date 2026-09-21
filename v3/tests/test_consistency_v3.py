"""Gate control flow and score accounting; no model download or GPU required."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from unified_evaluators import consistency, runtime
from unified_evaluators.contract import finalize, metric
from scripts.finalize_all_eval import rich_metrics


def decision(score=0.8, passed=True, status='evaluated'):
    return {'score': score, 'passed': passed, 'status': status, 'threshold': 0.8,
            'reason': 'scene continuity judgment', 'issues': []}


class ScoringTests(unittest.TestCase):
    def result(self, gate=None, attempted=True, metrics=None, error=None):
        metrics = metrics or {'M1': metric(2.0, 0.5, True), 'M2': metric(defined=False)}
        return finalize('P19', {'sample_id': 'example', 'video_path': 'example.mp4'},
                        metrics, {k: {'defined': m['extract_success'] is not None} for k, m in metrics.items()},
                        [], {'physics_attempted': attempted, **({'runtime_error': error} if error else {})},
                        consistency=gate or decision(), physics_attempted=attempted)

    def test_consistency_replaces_recognition_bonus(self):
        result = self.result()
        summary = result['verbose']['M1']['_scoring_summary']
        self.assertAlmostEqual(summary['score'], 0.15 * 0.8 + 0.85 * 0.5)
        self.assertEqual(result['metrics']['M1']['metric'], 0.5)
        self.assertEqual(summary['recognition_weight'], 0.0)
        self.assertEqual(summary['physics_score'], 0.5)

    def test_failed_physics_keeps_consistency_and_does_not_redistribute(self):
        result = self.result(metrics={'M1': metric(2., 1., True), 'M2': metric()})
        summary = result['verbose']['M1']['_scoring_summary']
        self.assertEqual(summary['physics_score'], 0.5)
        self.assertAlmostEqual(summary['score'], 0.12 + 0.425)
        self.assertEqual(summary['score_status'], 'partial')
        result = self.result(metrics={'M1': metric(), 'M2': metric(defined=False)})
        self.assertAlmostEqual(result['verbose']['M1']['_scoring_summary']['score'], 0.12)

    def test_rejection_keeps_fractional_consistency_only(self):
        result = self.result(decision(0.2, False), attempted=False)
        summary = result['verbose']['M1']['_scoring_summary']
        self.assertAlmostEqual(summary['score'], 0.03)
        self.assertIsNone(summary['physics_score'])
        self.assertEqual(summary['score_status'], 'consistency_rejected')
        self.assertFalse(result['verbose']['M1']['measurement_attempted'])
        self.assertIsNone(result['metrics']['M1']['metric'])

    def test_transport_error_is_not_a_zero_score_or_pass(self):
        result = self.result(decision(None, None, 'error'), attempted=False)
        self.assertIsNone(result['verbose']['M1']['_scoring_summary']['score'])
        self.assertEqual(result['verbose']['M1']['_scoring_summary']['score_status'], 'consistency_error')

    def test_backend_error_keeps_consistency_but_is_execution_error(self):
        result = self.result(metrics={'M1': metric(), 'M2': metric()}, error='backend broke')
        self.assertAlmostEqual(result['verbose']['M1']['_scoring_summary']['score'], 0.12)
        self.assertEqual(result['verbose']['M1']['_scoring_summary']['score_status'], 'execution_error')

    def test_physics_without_gate_pass_is_rejected(self):
        with self.assertRaises(ValueError):
            self.result(decision(0.2, False), attempted=True)

    def test_finalization_round_trip_cannot_restore_v2_bonus(self):
        first = self.result()
        second = finalize('P19', {'sample_id': 'example', 'video_path': 'example.mp4'}, rich_metrics(first),
                          first['verbose'], [], first['verbose']['M1']['_provenance'])
        self.assertEqual(first, second)


class ResponseTests(unittest.TestCase):
    def test_default_threshold_is_moderate(self):
        self.assertEqual(consistency.Settings().threshold, 0.8)

    def test_strict_numbers(self):
        for value in [True, None, '0.8', -0.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                consistency.parse_judgment(json.dumps({'score': value, 'reason': 'x', 'issues': []}))

    def test_fenced_json_and_minimum_fields(self):
        self.assertEqual(consistency.parse_judgment('```json\n{"score":0.7,"reason":"x","issues":[]}\n```')['score'], 0.7)
        for text in ['{}', '{"score":0.7,"reason":""}', 'not JSON', '[]']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                consistency.parse_judgment(text)

    def test_fresh_decision_replaces_old_pass_on_api_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'consistency'
            directory.mkdir()
            (directory / 'consistency.json').write_text(json.dumps(decision(1., True)))
            with patch.object(consistency, 'sample_frames', return_value=[]), \
                 patch.object(consistency, '_http_reply', side_effect=TimeoutError('timeout')):
                result = consistency.evaluate('video', None, 'prompt', 'P19', temp,
                                              consistency.Settings(backend='http', base_url='http://localhost/v1'))
            self.assertEqual(result['status'], 'error')
            self.assertIsNone(result['score'])
            self.assertIsNone(result['passed'])
            self.assertIsNone(json.loads((directory / 'consistency.json').read_text())['passed'])

    def test_threshold_is_configurable_and_boundary_passes(self):
        for threshold, passed in [(0.3, True), (0.8, True), (0.9, False)]:
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as temp:
                with patch.object(consistency, 'sample_frames', return_value=[]), \
                     patch.object(consistency, '_local_reply', return_value='{"score":0.8,"reason":"recognizable","issues":[]}'):
                    result = consistency.evaluate('video', None, 'prompt', 'P19', temp,
                                                  consistency.Settings(threshold=threshold))
                self.assertIs(result['passed'], passed)


class TemporalCoherencePromptTests(unittest.TestCase):
    def test_physical_action_prompt_and_reference_cannot_become_gate_instructions(self):
        action = 'The ball MUST fall with acceleration 9.81, never remain static.'
        items = [
            {'label': 'Reference image of the intended setup', 'path': '/reference.jpg', 'role': 'reference'},
            {'label': 'Video frame at 3.000 seconds', 'path': '/frame.jpg', 'role': 'video'},
        ]
        messages = consistency.messages_for(items, 'P1', action)
        text = json.dumps(messages)
        self.assertNotIn(action, text)
        self.assertNotIn('/reference.jpg', text)
        self.assertIn('3.000 seconds', text)
        self.assertIn('reference_comparison', text)
        self.assertIn('temporal visual coherence', text)
        self.assertIn('U-shaped tube turning into a rectangular tank', text)


class ControlFlowTests(unittest.TestCase):
    def setup_paths(self, directory, group='g2', task_id='P19'):
        root = Path(directory)
        task = root / group / task_id
        task.mkdir(parents=True)
        video = root / 'input.mp4'
        video.write_bytes(b'test video; decoding is mocked')
        output = root / 'result.json'
        return task, video, output

    def test_reject_skips_backend_and_track_preparation(self):
        for gate, code in [(decision(0.2, False), 1), (decision(None, None, 'error'), 2)]:
            with self.subTest(gate=gate), tempfile.TemporaryDirectory() as temp:
                task, video, output = self.setup_paths(temp, 'g3', 'P3')
                with patch.object(runtime.consistency_gate, 'evaluate', return_value=gate), \
                     patch.object(runtime, 'prepare_track_cache') as cache, \
                     patch.object(runtime, 'backend_command') as command, \
                     patch.object(runtime.subprocess, 'run') as process, \
                     patch.object(runtime, 'ensure_visual_evidence', side_effect=lambda evidence, *args: evidence):
                    status = runtime.main('P3', task, ['--video', str(video), '--output', str(output)])
                self.assertEqual(status, code)
                cache.assert_not_called(); command.assert_not_called(); process.assert_not_called()
                result = json.loads(output.read_text())
                self.assertFalse(result['verbose']['M1']['_provenance']['physics_attempted'])
                self.assertIsNone(result['verbose']['M1']['_provenance']['raw_result_path'])
                self.assertIsNone(result['verbose']['M1']['raw_result_path'])

    def test_pass_runs_backend_once_and_consumes_gate_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            task, video, output = self.setup_paths(temp)
            forwarded = []
            def command(task, row, raw, debug, forward):
                forwarded.extend(forward)
                return ['python', 'backend', '--output', str(raw)]
            def process(cmd, **kwargs):
                Path(cmd[cmd.index('--output') + 1]).write_text('{}')
                return SimpleNamespace(returncode=0)
            with patch.object(runtime.consistency_gate, 'evaluate', return_value=decision()), \
                 patch.object(runtime, 'backend_command', side_effect=command), \
                 patch.object(runtime.subprocess, 'run', side_effect=process) as backend, \
                 patch.object(runtime, 'classical_result', return_value=(
                     {'M1': metric(1., 0.5, True), 'M2': metric(defined=False)},
                     {'M1': {'defined': True}, 'M2': {'defined': False}})), \
                 patch.object(runtime, 'ensure_visual_evidence', side_effect=lambda evidence, *args: evidence):
                status = runtime.main('P19', task, ['--video', str(video), '--output', str(output),
                                      '--consistency-threshold', '0.4', '--consistency-frames', '6'])
            self.assertEqual(status, 0)
            backend.assert_called_once()
            self.assertEqual(forwarded, [])
            result = json.loads(output.read_text())
            self.assertTrue(result['verbose']['M1']['_scoring_summary']['physics_attempted'])
            self.assertAlmostEqual(result['verbose']['M1']['_scoring_summary']['score'], 0.545)


if __name__ == '__main__':
    unittest.main()
