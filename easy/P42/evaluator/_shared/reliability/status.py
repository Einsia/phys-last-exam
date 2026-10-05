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
# ``P28_thread_ball_tracks_incomplete`` describe an extractor, not an event.
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
        return 'evidence_insufficient', '物理测量未执行；一致性门控或运行状态未允许测量，未推断任务失败。'
    evidence = explicit_event_evidence(block)
    if evidence and any(item['kind'] in {'explicit_failure', 'explicit_text'} for item in evidence):
        return 'task_failed', '结果包含明确的事件未完成或实验条件无效证据。'
    flag = metric.get('extract_success')
    if flag is not True:
        return 'evidence_insufficient', '关键物理量未可靠提取；仅凭提取失败不能判定事件未发生。'
    score = _finite_score(metric.get('physics_score'))
    if score is None:
        return 'evidence_insufficient', '测量标记存在但没有有效的纯物理分数。'
    if score >= threshold:
        return 'physics_pass', f'测量有效，纯物理分数 {score:.4f} 达到阈值 {threshold:.2f}。'
    return 'physics_fail', f'测量有效，纯物理分数 {score:.4f} 低于阈值 {threshold:.2f}。'


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
        return 'evidence_insufficient', '没有定义的物理指标可用于判断。', details
    statuses = [item[0] for item in details.values()]
    if 'task_failed' in statuses:
        status = 'task_failed'
        reason = '至少一个指标有明确的事件未完成或条件无效证据。'
    elif all(item == 'physics_pass' for item in statuses):
        status = 'physics_pass'
        reason = '所有已定义且可测指标均达到冻结的物理通过阈值。'
    elif all(item in {'physics_pass', 'physics_fail'} for item in statuses) and 'physics_fail' in statuses:
        status = 'physics_fail'
        reason = '所有已定义指标均可测，但至少一个物理约束未达到通过阈值。'
    else:
        status = 'evidence_insufficient'
        reason = '至少一个必要指标证据不足；未把提取失败直接解释为任务失败。'
    return status, reason, details


def status_summary(status: str, details: Mapping[str, Any], *, defined_count: int,
                   reliable_count: int):
    return {
        'measurement_status': status,
        'measurement_statuses': list(MEASUREMENT_STATUSES),
        'measurement_status_reason': {
            'task_failed': '事件未完成或实验条件无效，且有明确证据支持。',
            'physics_pass': '关键量可测且满足冻结物理约束。',
            'physics_fail': '关键量可测但违反冻结物理约束。',
            'evidence_insufficient': '证据不足；不能从提取失败推断事件未发生。',
        }.get(status, '未定义'),
        'measurement_coverage': reliable_count / defined_count if defined_count else 0.0,
        'reliably_judged': status in {'task_failed', 'physics_pass', 'physics_fail'},
        'defined_metric_count': defined_count,
        'reliably_measured_metric_count': reliable_count,
    }
