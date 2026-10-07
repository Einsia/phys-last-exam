"""Keep scoring semantics intact when result explanations are in English."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'easy/P1/evaluator/_shared'))
from unified_evaluators.contract import physeval_result
from reliability.status import classify_metric_status

spec = importlib.util.spec_from_file_location('english_result_schema', ROOT / 'hard/P25/evaluator/utils/physeval/schema.py')
schema = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schema
spec.loader.exec_module(schema)


class EnglishScoringTests(unittest.TestCase):
    def test_fraction_rule_fallback_keeps_direct_physics_score(self):
        result = schema.Result('P25', 'video.mp4', 'first_frame.png', 'Fixture prompt.', 'fixture', 42)
        metric = result.add('M1', 'Observed fraction', non_residual=True)
        metric.succeed(0.35)
        raw = result.to_json()
        # Older results may omit the explicit type and precomputed score.
        # The translated rule must still identify a direct fraction mapping.
        raw['verbose']['M1'].pop('non_residual', None)
        raw['verbose']['M1'].pop('physics_score', None)
        slots, _ = physeval_result(raw)
        self.assertAlmostEqual(slots['M1']['physics_score'], 0.35)
        self.assertAlmostEqual(slots['M1']['proxy_score'], 0.15 + 0.85 * 0.35)
        self.assertIsNone(slots['M2']['extract_success'])

    def test_english_diagnostics_do_not_turn_missing_measurements_into_failed_events(self):
        metric = {'extract_success': False, 'physics_score': None}
        block = {'failure_reason': 'No scoreable physical-phenomenon evidence was obtained.'}
        status, reason = classify_metric_status(metric, block, physics_attempted=True)
        self.assertEqual(status, 'evidence_insufficient')
        self.assertIn('not reliably extracted', reason)
        failed, _ = classify_metric_status(metric, {'event_completed': False}, physics_attempted=True)
        self.assertEqual(failed, 'task_failed')


if __name__ == '__main__':
    unittest.main()
