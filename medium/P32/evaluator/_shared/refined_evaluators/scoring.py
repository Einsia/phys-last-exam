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
SCALES = {'P36': ('error_half_score', 1.0),
          'P10': ('time_error_half_score_sec', 0.5),
          'P39': ('error_at_zero', 1.0),
          'P38': ('error_at_zero', 1.0)}


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
        'interpretation': "Scores combine measurability and physical performance; they are not probabilities of physical correctness. Average only defined metrics. complete means all defined metrics are measurable; partial means some are measurable without forcing the total to zero.",
    }


def measurement_and_physics(task, verbose, previous_score):
    m = verbose.get('measurements', {})
    thresholds = verbose.get('thresholds', {})
    info = {'type': 'existing_non_residual_mapping', 'scale': None,
            'formula': verbose.get('score_details', {}).get('formula')}
    if task=='P10' and verbose.get('score_details',{}).get('version')=='p10_forced_support_v4':
        error=m.get('raw_m1_error');scale=thresholds.get('geometry_error_half_score',.02)
        if not all(finite(m.get(k)) for k in ('pivot_drift_p95_px','contact_gap_p95_px','observed_rotation_deg')):
            raise ValueError('Missing independently observed support geometry')
        return error,residual_score(error,scale),{'type':'residual','error':error,'scale':scale,
          'formula':'1/(1+resolved_geometry_error/geometry_error_half_score)',
          'error_unit':'relative to observed block geometry','version':'p10_forced_support_v4'}
    if task in SCALES:
        key, default = SCALES[task]
        scale = thresholds.get(key, default)
        error = m.get('delta_time_sec') if task == 'P10' else m.get('raw_m1_error')
        if task == 'P10' and not all(finite(m.get(k)) for k in ('N_COM_cross', 'N_tip')):
            raise ValueError('Missing independently measured event frames')
        value = m['N_COM_cross'] - m['N_tip'] if task == 'P10' else error
        info = {'type': 'residual', 'error': error, 'scale': scale,
                'scale_parameter': key, 'formula': '1 / (1 + abs(error) / scale)',
                'error_unit': 's' if task == 'P10' else 'dimensionless',
                'metric_unit': 'frames' if task == 'P10' else 'dimensionless',
                'scale_source': 'existing task engineering parameter; numeric value retained',
                'finite_error_cutoff': False}
        if task == 'P39' and m.get('error_kind') == 'unbounded_straight_partition':
            # Explicitly proved straight-partition limit, not a generic NaN/Inf.
            value = {'error_kind': 'unbounded_straight_partition', 'value': None}
            info['type'] = 'measured_limit'
            return value, 0.0, info
        return value, residual_score(error, scale), info
    if not finite(previous_score) or not 0 <= previous_score <= 1:
        raise ValueError('Missing or invalid task physics score')
    if task == 'P32':
        separation=m.get('final_direction_separation_deg')
        if not finite(separation) or not 0 <= separation <= 180:
            raise ValueError('Missing measured final compass direction separation')
        expected=(1-math.cos(math.radians(separation)))/2
        if abs(previous_score-expected)>1e-9:
            raise ValueError('P32 score disagrees with measured final directions')
        value={'final_direction_separation_deg':separation}
        info.update(type='final_direction_alignment',version='opinion_v2_final_direction',
                    formula='(1-cos(final_direction_separation_deg*pi/180))/2',
                    metric_unit='degrees',first_frame_direction_used=False)
    elif task == 'P33':
        value = {'height_ratio': m.get('height_ratio'),
                 'h_open_diameter': m.get('h_open_diameter'),
                 'h_closed_diameter': m.get('h_closed_diameter')}
        if not all(finite(value[k]) for k in ('h_open_diameter', 'h_closed_diameter')):
            raise ValueError('Missing ring heights')
    elif task == 'P34':
        value = {'N_solid': m.get('N_solid'), 'N_slotted': m.get('N_slotted'),
                 'oscillation_count_ratio': m.get('oscillation_count_ratio')}
        if not all(finite(value[k]) for k in ('N_solid', 'N_slotted')):
            raise ValueError('Missing oscillation counts')
    elif task == 'P30':
        if verbose.get('score_details', {}).get('version') == 'p30_entry_light_v5':
            entered, light = m.get('magnet_entered'), m.get('lamp_lit_during_entry')
            if not isinstance(entered, bool) or 'lamp_lit_during_entry' not in m:
                raise ValueError('Missing observed magnet-entry/lamp booleans')
            if not isinstance(light, bool) and not (light is None and entered and
                    m.get('lamp_state_during_entry') == 'unknown'):
                raise ValueError('Unresolved lamp evidence must be explicitly marked unknown')
            if light and not entered:
                raise ValueError('Lamp-during-entry evidence requires observed magnet entry')
            expected = .5 * entered + .5 * (light is True)
            if abs(previous_score - expected) > 1e-12:
                raise ValueError('P30 score disagrees with its two observed half-point events')
            value = {'magnet_entered': entered, 'confirmed_component_count':
                     int(entered) + int(light is True)}
            if light is not None:
                value['lamp_lit_during_entry'] = light
            info.update(type='two_observed_events', version='p30_entry_light_v5',
                        formula='0.5 * confirmed_component_count',
                        lamp_state=m.get('lamp_state_during_entry'),
                        score_is_confirmed_evidence_lower_bound=light is None)
        elif verbose.get('score_details',{}).get('version')=='p30_induction_timing_v4':
            if not finite(m.get('reference_activity_time_centroid_sec')) or not finite(m.get('stationary_emission_fraction')):
                raise ValueError('Missing observed induction timing evidence')
            value={'absolute_timing_error_sec':m.get('absolute_timing_error_sec'),
                   'resolved_timing_error_sec':m.get('resolved_timing_error_sec'),
                   'emission_detected':m.get('emission_detected'),
                   'stationary_emission_fraction':m.get('stationary_emission_fraction')}
            info.update(type='induction_timing_proxy',version='p30_induction_timing_v4')
        else:
            value = previous_score
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
