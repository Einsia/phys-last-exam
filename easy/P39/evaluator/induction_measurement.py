"""Keep observed absence of lamp emission distinct from missing evidence.

Undefined emission times stay null in measurements. The raw metric records
observed activity and absence without inventing a finite timing error.
"""
from refined_evaluators.scoring import measurement_and_physics as _base_measurement


def measurement_and_physics(task, verbose, previous_score):
    value, physics, info = _base_measurement(task, verbose, previous_score)
    if task != 'P39' or verbose.get('score_details', {}).get('version') != 'p39_induction_timing_v4':
        return value, physics, info
    observed = verbose.get('measurements', {})
    if observed.get('emission_detected') is False:
        if physics != 0.0:
            raise ValueError('Observed absence of emission requires zero physical score')
        value = {
            'emission_detected': False,
            'reference_activity_time_centroid_sec': observed['reference_activity_time_centroid_sec'],
            'stationary_emission_fraction': observed['stationary_emission_fraction'],
        }
        info = dict(info, timing_error_applicable=False,
                    observed_zero_reason='Magnet activity was observed but lamp emission was absent.')
    return value, physics, info


def install():
    from refined_evaluators import scoring
    scoring.measurement_and_physics = measurement_and_physics
