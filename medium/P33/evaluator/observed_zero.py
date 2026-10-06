"""Retain independently observed zero outcomes when their optional ratio is undefined."""
from refined_evaluators.scoring import measurement_and_physics as _base


def measurement_and_physics(task, verbose, previous_score):
    value, physics, info=_base(task,verbose,previous_score)
    m=verbose.get('measurements',{})
    if task=='P33' and m.get('height_ratio') is None:
        if physics!=0.0 or not (str(verbose.get('reason','')).startswith('zero_closed_height')):
            raise ValueError('Undefined ratio lacks verified observed-zero evidence')
        value={k:m[k] for k in ['h_open_diameter', 'h_closed_diameter']}
        info=dict(info,ratio_applicable=False,observed_zero_preserved=True)
    return value,physics,info


def install():
    from refined_evaluators import scoring
    scoring.measurement_and_physics=measurement_and_physics
