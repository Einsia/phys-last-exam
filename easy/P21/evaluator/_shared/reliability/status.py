"""Explainable measurement-state classification.

The old V3 status (`complete`/`partial`/`unavailable`) describes extraction
coverage, but it cannot say whether an event failed or simply was not
measurable.  This module adds a separate four-way judgment state while keeping
the old score and runtime status backward compatible.

No extraction failure is treated as a task failure unless the result contains
explicit event/condition evidence.  This is the key safety rule of the new
contract.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


MEASUREMENT_STATUSES = (
    'task_failed',
    'physics_pass',
    'physics_fail',
    'evidence_insufficient',
)

# This is deliberately an operational default, separate from the continuous
# score.  Task-specific calibrated thresholds can override it in a block or
# result metadata without changing the ranking score.
DEFAULT_PASS_THRESHOLD = 0.80

_EVENT_KEYS = {
    'event_status', 'task_status', 'condition_status', 'event_completed',
    'task_completed', 'conditions_valid', 'failure_class', 'judgment_status',
}
_EXPLICIT_FAILURE_VALUES = {
    'failed', 'failure', 'not_completed', 'incomplete', 'invalid',
    'invalid_condition', 'conditions_invalid', 'task_failed', 'event_failed',
}
_EXPLICIT_SUCCESS_VALUES = {'completed', 'complete', 'passed', 'valid', 'true', True}
# Keep this deliberately narrow.  Strings such as
# ``P31_thread_ball_tracks_incomplete`` describe an extractor, not an event.
_EVENT_WORDS = re.compile(
    r'\b(event|task)\s+(?:was\s+)?(?:not\s+completed|did\s+not\s+occur|failed)\b|'
    r'\b(?:condition|conditions)\s+(?:are\s+)?(?:invalid|not\s+satisfied|failed)\b', re.I)


def _walk(value: Any, prefix: str = ''):
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f'{prefix}.{key}' if prefix else str(key)
            yield path, key, item
            yield from _walk(item, path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f'{prefix}[{index}]')


def explicit_event_evidence(block: Mapping[str, Any] | None):
    """Return explicit event/condition evidence, never inferred from nulls."""
    if not isinstance(block, Mapping):
        return None
    found = []
    for path, key, value in _walk(block):
        key_text = str(key).lower()
        if key_text in _EVENT_KEYS:
            normalized = str(value).strip().lower() if not isinstance(value, bool) else value
            if normalized in _EXPLICIT_FAILURE_VALUES or value is False:
                found.append({'path': path, 'value': value, 'kind': 'explicit_failure'})
            elif normalized in _EXPLICIT_SUCCESS_VALUES:
                found.append({'path': path, 'value': value, 'kind': 'explicit_success'})
        if key_text in {'failure_reason', 'reason', 'status_reason'} and isinstance(value, str):
            if _EVENT_WORDS.search(value):
                found.append({'path': path, 'value': value, 'kind': 'explicit_text'})
    return found or None


def _finite_score(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0 <= value <= 1 else None


def classify_metric_status(metric: Mapping[str, Any], block: Mapping[str, Any] | None,
                           *, physics_attempted: bool, threshold: float = DEFAULT_PASS_THRESHOLD):
    """Classify one metric without conflating missing measurements and events."""
    if not physics_attempted:
        return 'evidence_insufficient', "Physical measurement was not performed because the consistency gate or runtime status did not permit it; task failure is not inferred."
    evidence = explicit_event_evidence(block)
    if evidence and any(item['kind'] in {'explicit_failure', 'explicit_text'} for item in evidence):
        return 'task_failed', "The result contains explicit evidence of an incomplete event or invalid experimental conditions."
    flag = metric.get('extract_success')
    if flag is not True:
        return 'evidence_insufficient', "Key physical quantities were not reliably extracted; extraction failure alone does not establish that the event did not occur."
    score = _finite_score(metric.get('physics_score'))
    if score is None:
        return 'evidence_insufficient', "Measurement markers are present, but no valid pure physics score is available."
    if score >= threshold:
        return 'physics_pass', f'Measurement is valid; pure physics score {score:.4f} meets threshold {threshold:.2f}.'
    return 'physics_fail', f'Measurement is valid; pure physics score {score:.4f} is below threshold {threshold:.2f}.'


def classify_result_status(metrics: Mapping[str, Mapping[str, Any]], blocks: Mapping[str, Mapping[str, Any]],
                           *, physics_attempted: bool, threshold: float = DEFAULT_PASS_THRESHOLD):
    """Classify the video-level physical judgment and return audit details."""
    details = {}
    for key, metric in metrics.items():
        defined = metric.get('extract_success') is not None
        if not defined:
            continue
        details[key] = classify_metric_status(metric, blocks.get(key),
                                              physics_attempted=physics_attempted,
                                              threshold=threshold)
    if not details:
        return 'evidence_insufficient', "No defined physical metric is available for assessment.", details
    statuses = [item[0] for item in details.values()]
    if 'task_failed' in statuses:
        status = 'task_failed'
        reason = "At least one metric has explicit evidence of an incomplete event or invalid conditions."
    elif all(item == 'physics_pass' for item in statuses):
        status = 'physics_pass'
        reason = "All defined and measurable metrics meet the fixed physical pass thresholds."
    elif all(item in {'physics_pass', 'physics_fail'} for item in statuses) and 'physics_fail' in statuses:
        status = 'physics_fail'
        reason = "All defined metrics are measurable, but at least one physical constraint does not meet the pass threshold."
    else:
        status = 'evidence_insufficient'
        reason = "Evidence is insufficient for at least one required metric; extraction failure is not treated as task failure."
    return status, reason, details


def status_summary(status: str, details: Mapping[str, Any], *, defined_count: int,
                   reliable_count: int):
    return {
        'measurement_status': status,
        'measurement_statuses': list(MEASUREMENT_STATUSES),
        'measurement_status_reason': {
            'task_failed': "Explicit evidence supports an incomplete event or invalid experimental conditions.",
            'physics_pass': "Key quantities are measurable and satisfy the fixed physical constraints.",
            'physics_fail': "Key quantities are measurable but violate the fixed physical constraints.",
            'evidence_insufficient': "Insufficient evidence; extraction failure does not establish that the event did not occur.",
        }.get(status, "Undefined"),
        'measurement_coverage': reliable_count / defined_count if defined_count else 0.0,
        'reliably_judged': status in {'task_failed', 'physics_pass', 'physics_fail'},
        'defined_metric_count': defined_count,
        'reliably_measured_metric_count': reliable_count,
    }
