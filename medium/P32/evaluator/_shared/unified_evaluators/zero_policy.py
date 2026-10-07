"""Operational zero for a phenomenon not resolved by a completed extraction.

This policy is explicit: a zero is an assessment outcome, never a fabricated
physical measurement or a claim that an unobserved event certainly did not
occur. Runtime failures and consistency rejection cannot enter this path.
"""
from copy import deepcopy

VERSION = 'not_observed_zero_v1'


def apply(data):
    if data['provenance'].get('unobserved_zero_policy') != VERSION:
        return data
    if (data['consistency'].get('passed') is not True
            or not data['physics']['attempted']
            or data['status'] in ('execution_error', 'consistency_error')):
        return data
    missing = [k for k, m in data['metrics'].items() if m['status'] == 'unmeasurable']
    if not missing:
        return data
    provenance = data['provenance']
    if not provenance.get('backend_completed') or not provenance.get('raw_result_path'):
        raise ValueError('Zero policy requires a completed, recorded physical extraction')
    if any(not data['metrics'][k]['evidence'] for k in missing):
        raise ValueError('Zero policy requires retained observation evidence')
    data = deepcopy(data)
    for key in missing:
        metric = data['metrics'][key]
        original_reason = metric['reason'] or 'Required physical phenomenon was not resolved.'
        metric.update(status='not_observed', physics_score=0.0, raw_value=None,
                      evaluation_source='unobserved_zero_policy',
                      reason="No scoreable physical-phenomenon evidence was obtained for this metric; assign zero under the policy. Extraction diagnostics: " + original_reason)
        metric['normalization']['zero_policy'] = {
            'version': VERSION, 'assigned_score': 0.0,
            'extraction_reason': original_reason, 'raw_measurement_available': False,
            'meaning': 'No scored physical demonstration; this is not a measured numerical violation.'}
    for failure in data['failures']:
        if failure['metric_id'] in missing and failure['code'] == 'unmeasurable':
            failure.update(code='not_observed_zero', message=data['metrics'][failure['metric_id']]['reason'])
    defined = [m for m in data['metrics'].values() if m['defined']]
    if any(m['physics_score'] is None for m in defined):
        raise ValueError('Completed physical assessment still has an unscored metric')
    physical_score = sum(m['physics_score'] for m in defined) / len(defined)
    data['status'] = 'scored_with_zero'
    data['physics'].update(
        status='scored_with_zero', score=physical_score,
        scored_metrics=len(defined), zero_policy_metrics=missing,
        missing_metric_policy=VERSION,
        reason="Evaluation completed; " + ", ".join(missing) + ": no scoreable physical-phenomenon evidence was obtained; assign zero under the policy.")
    data['score'].update(physics_contribution=.85*physical_score,
                         total=.15*data['consistency']['score']+.85*physical_score)
    return data
