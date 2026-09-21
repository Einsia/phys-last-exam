import unittest

from reliability.protocol import get_protocol
from reliability.status import classify_metric_status, classify_result_status


class ReliabilityStatusTests(unittest.TestCase):
    def test_extraction_failure_is_evidence_insufficient(self):
        status, reason = classify_metric_status(
            {'extract_success': False, 'physics_score': None},
            {'failure_reason': 'P28_thread_ball_tracks_incomplete'},
            physics_attempted=True,
        )
        self.assertEqual(status, 'evidence_insufficient')
        self.assertIn('不能判定', reason)

    def test_explicit_event_failure_is_task_failed(self):
        status, _ = classify_metric_status(
            {'extract_success': False, 'physics_score': None},
            {'event_completed': False, 'event_evidence': 'ice remains in final frames'},
            physics_attempted=True,
        )
        self.assertEqual(status, 'task_failed')

    def test_measurable_pass_and_fail(self):
        passed, _ = classify_metric_status({'extract_success': True, 'physics_score': .71}, {}, physics_attempted=True, threshold=.65)
        failed, _ = classify_metric_status({'extract_success': True, 'physics_score': .64}, {}, physics_attempted=True, threshold=.65)
        self.assertEqual(passed, 'physics_pass')
        self.assertEqual(failed, 'physics_fail')

    def test_result_keeps_gate_rejection_out_of_task_failure(self):
        status, _, details = classify_result_status(
            {'M1': {'extract_success': False, 'physics_score': None}},
            {'M1': {'failure_reason': 'consistency gate rejected video'}},
            physics_attempted=False,
        )
        self.assertEqual(status, 'evidence_insufficient')
        self.assertEqual(details['M1'][0], 'evidence_insufficient')

    def test_control_thresholds_are_task_specific(self):
        self.assertEqual(get_protocol('P21')['pass_threshold'], .65)
        self.assertEqual(get_protocol('P48')['pass_threshold'], .50)


if __name__ == '__main__':
    unittest.main()
