"""Versioned, dependency-free physical measurements and recognition scoring.

G8/G9 retain their existing measurement eligibility and non-residual mappings.
M2 is not defined by these task evaluator specifications. Its measurement stays
null and is excluded from the total's denominator. Defined but unmeasurable
metrics retain their weight and contribute zero.
"""
from copy import deepcopy
import math

LEGACY_VERSION = 'physical-bench-proxy-v2-recognition015-arithmetic-g8g9'
VERSION = LEGACY_VERSION + '-defined-metrics'
RECOGNITION_WEIGHT = 0.15
PHYSICS_WEIGHT = 0.85
METRIC_KEYS = ('M1', 'M2')
SCALES = {'P41': ('error_half_score', 1.0),
          'P43': ('time_error_half_score_sec', 0.5),
          'P47': ('error_at_zero', 1.0),
          'P49': ('error_at_zero', 1.0)}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def residual_score(error, scale):
    if not finite(scale) or scale <= 0:
        raise ValueError('Residual scale must be finite and positive')
    if not finite(error):
        raise ValueError('Missing, NaN and infinite residuals are not measurements')
    # Avoid overflow for large, finite errors without introducing a cutoff.
    e = abs(error)
    return 1.0 / (1.0 + e / scale) if e <= scale else (scale / e) / (1.0 + scale / e)


def metric_score(physics_score, measurable):
    if not measurable:
        return 0.0
    if not finite(physics_score) or not 0 <= physics_score <= 1:
        raise ValueError('A measurable metric requires a finite physics score in [0, 1]')
    return RECOGNITION_WEIGHT + PHYSICS_WEIGHT * physics_score


def aggregate(metrics):
    defined = [key for key in METRIC_KEYS if metrics[key]['extract_success'] is not None]
    successful = [key for key in defined if metrics[key]['extract_success'] is True]
    weights = {key: 1.0 / len(defined) if key in defined else 0.0 for key in METRIC_KEYS}
    complete = bool(defined) and len(successful) == len(defined)
    return {
        'version': VERSION,
        'score': sum(weights[key] * metrics[key]['proxy_score'] for key in defined),
        'weights': weights,
        'score_status': 'complete' if complete else 'partial' if successful else 'unavailable',
        'overall_proxy_valid': complete,
        'defined_metrics': defined,
        'not_applicable_metrics': [key for key in METRIC_KEYS if key not in defined],
        'coverage_scope': 'defined_metrics',
        'missing_metric_policy': 'defined_metric_failure_zero_no_redistribution',
        'not_applicable_metric_policy': 'excluded_from_denominator',
        'interpretation': '分数包含可测量性和物理表现；不是物理正确概率。只对已定义指标等权平均；complete 表示已定义指标全部可测，partial 表示部分可测且不强制清零。',
    }


def measurement_and_physics(task, verbose, previous_score):
    m = verbose.get('measurements', {})
    thresholds = verbose.get('thresholds', {})
    info = {'type': 'existing_non_residual_mapping', 'scale': None,
            'formula': verbose.get('score_details', {}).get('formula')}
    if task in SCALES:
        key, default = SCALES[task]
        scale = thresholds.get(key, default)
        error = m.get('delta_time_sec') if task == 'P43' else m.get('raw_m1_error')
        if task == 'P43' and not all(finite(m.get(k)) for k in ('N_COM_cross', 'N_tip')):
            raise ValueError('Missing independently measured event frames')
        value = m['N_COM_cross'] - m['N_tip'] if task == 'P43' else error
        info = {'type': 'residual', 'error': error, 'scale': scale,
                'scale_parameter': key, 'formula': '1 / (1 + abs(error) / scale)',
                'error_unit': 's' if task == 'P43' else 'dimensionless',
                'metric_unit': 'frames' if task == 'P43' else 'dimensionless',
                'scale_source': 'existing task engineering parameter; numeric value retained',
                'finite_error_cutoff': False}
        if task == 'P47' and m.get('error_kind') == 'unbounded_straight_partition':
            # Explicitly proved straight-partition limit, not a generic NaN/Inf.
            value = {'error_kind': 'unbounded_straight_partition', 'value': None}
            info['type'] = 'measured_limit'
            return value, 0.0, info
        return value, residual_score(error, scale), info
    if not finite(previous_score) or not 0 <= previous_score <= 1:
        raise ValueError('Missing or invalid task physics score')
    if task == 'P34':
        d1, d2 = m.get('delta_1_deg'), m.get('delta_2_deg')
        if not finite(d1) or not finite(d2):
            raise ValueError('Missing signed compass deflections')
        value = {'delta_1_deg': d1, 'delta_2_deg': d2,
                 'signed_sum_deg': d1 + d2, 'opposite_direction': d1 * d2 < 0}
    elif task == 'P37':
        value = {'height_ratio': m.get('height_ratio'),
                 'h_open_diameter': m.get('h_open_diameter'),
                 'h_closed_diameter': m.get('h_closed_diameter')}
        if not all(finite(value[k]) for k in ('h_open_diameter', 'h_closed_diameter')):
            raise ValueError('Missing ring heights')
    elif task == 'P38':
        value = {'N_solid': m.get('N_solid'), 'N_slotted': m.get('N_slotted'),
                 'oscillation_count_ratio': m.get('oscillation_count_ratio')}
        if not all(finite(value[k]) for k in ('N_solid', 'N_slotted')):
            raise ValueError('Missing oscillation counts')
    elif task == 'P39':
        value = previous_score  # Event F1 is itself the original physical statistic.
        if not all(finite(m.get(k)) for k in ('TP', 'FP', 'FN')):
            raise ValueError('Missing event matching counts')
    else:
        raise ValueError(f'No scoring definition for {task}')
    return value, float(previous_score), info


def score_result(result):
    """Finalize a fresh extractor result without changing measurement eligibility.

    Idempotent for already finalized output; historical values are retained under
    normalization_source and verbose.M1.raw_measurement.
    """
    if result.get('normalization_source', {}).get('version') == VERSION:
        return result
    if result.get('normalization_source', {}).get('version') == LEGACY_VERSION:
        # Previously normalized metrics already contain the recognition reward.
        # Only reaggregate; never treat their physical measurements as scores.
        result.setdefault('aggregation_history', []).append({
            'reason': 'User clarification 2026-09-11: average defined metrics only; undefined M2 stays null',
            'previous_proxy': deepcopy(result['proxy']),
        })
        result['proxy'] = aggregate(result['metrics'])
        result['normalization_source']['version'] = VERSION
        return result
    previous = deepcopy(result['metrics'])
    result['normalization_source'] = {
        'version': VERSION, 'previous_metrics': previous,
        'recognition_weight': RECOGNITION_WEIGHT, 'physics_weight': PHYSICS_WEIGHT,
        'eligibility_policy': 'retain extractor per-metric reliability checks; reject non-finite required values',
    }
    for key in METRIC_KEYS:
        entry = result['metrics'][key]
        success = entry.get('extract_success')
        details = result['verbose'].get(key)
        value = physics = None
        if details is not None:
            details['raw_measurement'] = deepcopy(details.get('measurements', {}))
            details['extractor_metric_before_normalization'] = deepcopy(entry.get('metric'))
        if success is True:
            if key != 'M1':
                raise ValueError('G8/G9 task specifications currently define M1 only')
            try:
                value, physics, info = measurement_and_physics(result['task_id'], details, entry.get('metric'))
            except ValueError as exc:
                success = False
                details['status'] = 'extraction_failed'
                details['reason'] = f'Invalid required physical measurement: {exc}'
                details['normalization_error'] = str(exc)
            else:
                details['normalization'] = info
        entry.clear()
        entry.update(extract_success=success, metric=value if success is True else None,
                     physics_score=physics if success is True else None,
                     recognition_score=RECOGNITION_WEIGHT if success is True else 0.0,
                     proxy_score=metric_score(physics, success is True), proxy_valid=success is True,
                     metric_status='measured' if success is True else 'not_applicable' if success is None else 'unmeasurable')
        if details is not None:
            details['scoring'] = {'formula': '0.15 + 0.85 * physics_score if measurable else 0',
                                  'physics_score': entry['physics_score'],
                                  'recognition_score': entry['recognition_score'],
                                  'proxy_score': entry['proxy_score'], 'recognition_awarded_once': True}
    result['proxy'] = aggregate(result['metrics'])
    result['schema_version'] = 'physical-bench-evaluation-v2'
    return result
