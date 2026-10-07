"""G3/G7 V2 mappings, with the frozen task-specific scales and validity gates."""
import math
from .contract import KEYS, clip, gm, measurement_block, metric, present, q

G3_SCALES = {'P3':0.15, 'P1':0.18, 'P8':0.25, 'P14':0.20, 'P17':0.12}
G3_LOGIC = {
    'P3': ("Measure the ranges of both complete parabolic trajectories and compute R30/R60 - 1.", "Fit the 30-degree and 60-degree trajectories separately, normalize residuals by ball diameter, and compare initial speeds."),
    'P1': ("Estimate restitution independently from bounce heights and timing; check agreement between estimates and that passive-collision restitution does not exceed 1.", "Continuously measure energy violations without external energy input using relative increases in successive bounce heights, and report restitution variability."),
    'P8': ("Measure cumulative ball-center displacement, marker rotation, and ball radius to compute the integral form of v/(omega*R)-1.", "Estimate instantaneous contact-point velocity residual from the difference between ball-center speed and omega*R."),
    'P14': ("Track both pendulum bobs and suspension points, estimate periods from peak intervals, measure lengths, and compute abs((T1/T2)^2/(L1/L2)-1).", "Compute period coefficients of variation for the short and long pendulums separately, then aggregate pure physics scores."),
    'P17': ("Extract rays from temporal brightness differences, independently fit the waterline, measure ray angles relative to the normal, and compute sin(theta_i)/sin(theta_t)-n_water.", "Measure the separation between incident and refracted ray intersections with the waterline and normalize by the fixed vessel width."),
}


def component(value, scale=None, *, physics=None, formula=None):
    score = q(value, scale) if scale is not None else physics
    return {'raw_measurement':value, 'scale':scale, 'physics_score':score,
            'formula':formula or '1/(1+abs(e)/a)'}


def score_g3(task, values, validity, measurements=None, reason=None):
    components = {'M1':{'error':component(values['M1'], G3_SCALES[task])}}
    aux = values['M2']
    if task == 'P3':
        scales = {'trajectory_30deg_parabola_rmse_d':0.30, 'trajectory_60deg_parabola_rmse_d':0.30, 'initial_speed_magnitude_error':0.20}
        components['M2'] = {k:component((aux or {}).get(k), a) for k,a in scales.items()}
    elif task == 'P1':
        aux = aux or {}
        components['M1']['passive_restitution_excess']=component(aux.get('passive_restitution_excess'),0.15)
        components['M2'] = {
            'height_increase_excess': component(aux.get('height_increase_excess'),0.15),
            **({'adjacent_restitution_coefficient_cv': component(aux['adjacent_restitution_coefficient_cv'],0.35)}
               if present(aux.get('adjacent_restitution_coefficient_cv')) else {})}
    elif task == 'P14':
        components['M2'] = {k:component((aux or {}).get(k),0.20) for k in ('short_pendulum_period_cv','long_pendulum_period_cv')}
    else:
        components['M2'] = {'error':component(aux,0.35 if task == 'P8' else 0.025)}
    metrics, blocks = {}, {}
    for i,key in enumerate(KEYS):
        physics = gm(c['physics_score'] for c in components[key].values())
        metrics[key] = metric(values[key], physics, validity.get(key, False))
        blocks[key] = measurement_block(G3_LOGIC[task][i], values[key],
            steps=["Decode this video and obtain trajectories or geometry that pass validity checks.", "Recompute physical quantities using the fixed event selection, fitting, and scale definitions for the task.", "Normalize residuals and add the recognition reward once per metric."],
            measurements=measurements, normalization={'components':components[key], 'aggregation':'geometric mean' if len(components[key]) > 1 else 'single component'}, reason=reason)
    return metrics, blocks


def g3_result(raw):
    task, m = raw['task_id'], raw.get('metrics', {})
    measurements = raw.get('measurements', {})
    status = raw.get('status') if isinstance(raw.get('status'),dict) else raw.get('statuses', raw)
    valid = status.get('metric_validity', raw.get('metric_validity'))
    if task == 'P17' and valid is not None:
        valid = {'M1':valid.get('M1_snell_residual',False), 'M2':valid.get('M2_intersection_consistency',False)}
    if valid is None:
        valid = {key: bool(status.get('extract_success', raw.get('extract_success')) and status.get('measurement_valid', raw.get('measurement_valid',False))) for key in KEYS}
    if task == 'P3':
        aux = {}
        for slot, ball in measurements.get('balls',{}).items():
            angle = int(ball['angle_target_deg'])
            aux[f'trajectory_{angle}deg_parabola_rmse_d'] = m.get('M2',{}).get('per_ball',{}).get(slot,{}).get('parabola_rmse_d')
        aux['initial_speed_magnitude_error'] = m.get('initial_speed_error')
        values = {'M1':m.get('M1'), 'M2':aux}
    elif task == 'P1':
        heights = measurements.get('rebound_heights_px', [])
        checks = [right < left for left,right in zip(heights,heights[1:])]
        values = {'M1':m.get('M1_height_time_restitution_consistency'), 'M2':{
            'height_decrease_checks':checks, 'height_decrease_fraction':sum(checks)/len(checks) if checks else None,
            'height_increase_excess':sum(max(0.0,right/left-1.0) for left,right in zip(heights,heights[1:]))/len(checks) if checks and all(h>0 for h in heights) else None,
            'passive_restitution_excess':sum(max(0.0,math.sqrt(right/left)-1.0) for left,right in zip(heights,heights[1:]))/len(checks) if checks and all(h>0 for h in heights) else None,
            **({'adjacent_restitution_coefficient_cv':m['M2_restitution_coefficient_cv']} if present(m.get('M2_restitution_coefficient_cv')) else {})}}
    elif task == 'P8':
        values = {'M1':m.get('M1_rolling_ratio_signed'), 'M2':m.get('M2_contact_velocity_nmae')}
    elif task == 'P14':
        pendulums = raw.get('pendulums', {})
        values = {'M1':m.get('m1_abs_residual'), 'M2':{f'{name}_pendulum_period_cv':(pendulums.get(name,{}).get('period') or {}).get('period_cv') for name in ('short','long')}}
        measurements = {'pendulums':pendulums, 'camera_audit':raw.get('camera_audit')}
    else:
        values = {'M1':m.get('M1_snell_residual_signed'), 'M2':m.get('M2_intersection_disagreement_normalized')}
        measurements = {'geometry':raw.get('geometry'), 'temporal':raw.get('temporal'), 'structure':raw.get('structure')}
    if task == 'P1' and measurements.get('single_motion_no_repeated_bounce') is True:
        metrics={key:metric({'observed_outcome':'single_motion_no_repeated_bounce'},.1,True) for key in KEYS}
        blocks={key:measurement_block(G3_LOGIC[task][i],metrics[key]['raw_value'] if 'raw_value' in metrics[key] else {'observed_outcome':'single_motion_no_repeated_bounce'},
            measurements=measurements,steps=["Continuous tracking confirms a single motion segment without multiple comparable bounces."],
            normalization={'policy':'opinion_v2_observed_single_motion','physics_score':.1},reason="A single motion segment was measured; the evaluation specification assigns a physics score of 0.1.") for i,key in enumerate(KEYS)}
        return metrics,blocks
    return score_g3(task, values, valid, measurements,
        raw.get('failure_reason') or status.get('measurement_invalid_reasons') or raw.get('errors'))


G7_LOGIC = {
    'P9': ("Estimate a shared motion direction from both trajectories and compare equal-distance travel-time ratios with sqrt(10/7) at multiple virtual reference positions.", "M2 is omitted under the current evaluation specification; rotation and no-slip behavior are not measured and do not enter the denominator."),
    'P13': ("Estimate both pendulum periods from turning-point intervals and compare their ratio with the existing finite-amplitude theoretical ratio.", "Use the clipped value of (T30/T15-1)/(r_theory-1) to measure the direction of the period relationship."),
    'P11': ("Fit the incline direction and accelerations along the ascending and descending trajectories; compare the measured acceleration ratio with friction-inclusive theory.", "M2 is omitted under evaluation specification v2 and excluded from the scoring denominator."),
    'P20': ("Independently fit each transmitted ray and medium interface, estimate refractive indices using Snell geometry, and compute their coefficient of variation; ordinary total internal reflection supplies only a lower bound and is not treated as a critical angle.", "M2 is omitted under evaluation specification v2 and excluded from the scoring denominator."),
    'P28': ("First confirm sustained ice shrinkage and liquid formation on both sides. Within the same frame, compare liquid levels relative to each vessel base and normalized by vessel height. Record a temporary lead when the crushed-ice vessel exceeds the localization error for at least 0.25 seconds. Distinguish an observed absence of a lead from insufficient evidence. This metric is not complete-melting time or melted mass.", "Two-dimensional projected area cannot identify three-dimensional mass; area ratios are retained only as diagnostics and excluded from physical scoring."),
}


def ratio(a,b):
    return float(a)/float(b) if present(a) and present(b) and float(b) > 0 else None


def g7_result(raw):
    task, m, measures = raw['task_id'], raw.get('metrics',{}), raw.get('measurements',{})
    first = m.get('M1')
    base_ok = bool(raw.get('extract_success'))
    if task == 'P9':
        aux = {}
        c2 = {}
    elif task == 'P13':
        declared = measures.get('T30_greater_than_T15', m.get('M2_T30_gt_T15'))
        aux = {'T30_gt_T15':declared}
        r = ratio(measures.get('T30_s'),measures.get('T15_s'))
        expected = measures.get('expected_period_ratio_exact')
        score = clip((r-1)/(expected-1)) if r is not None and present(expected) and expected > 1 else None
        c2 = {'period_direction':component({'measured_ratio':r,'expected_ratio':expected},physics=score,formula='clip((T30/T15-1)/(r_theory-1))')}
    elif task == 'P11':
        aux = {'constant_acceleration_fit_up':m.get('M2_up_fit_relative_rmse'), 'constant_acceleration_fit_down':m.get('M2_down_fit_relative_rmse'), 't_down_gt_t_up':measures.get('t_down_gt_t_up',m.get('M2_t_down_gt_t_up'))}
        r = ratio(measures.get('t_down_s'),measures.get('t_up_s'))
        expected = measures.get('expected_acceleration_ratio')
        margin = clip((r-1)/(math.sqrt(expected)-1)) if r is not None and present(expected) and expected > 1 else None
        c2 = {'up_fit':component(aux['constant_acceleration_fit_up'],0.10), 'down_fit':component(aux['constant_acceleration_fit_down'],0.10),
              'time_margin':component({'time_ratio':r,'expected_acceleration_ratio':expected},physics=margin,formula='clip((t_down/t_up-1)/(sqrt(expected_acceleration_ratio)-1))')}
    elif task == 'P20':
        residuals = measures.get('per_ray_snell_residual',m.get('per_ray_snell_residual'))
        aux = {'per_ray_snell_residual':residuals}
        c2 = {f'ray_{i}':component(v,0.06665) for i,v in enumerate(residuals or [],1)}
    else:
        aux = {'initial_mass_visual_validity':m.get('M2_initial_projected_area_log_error')}
        c2 = {}  # Projected area cannot establish equal three-dimensional mass.
    if task == 'P28':
        lead = measures.get('surface_lead')
        if lead is not None:
            observed = lead.get('lead_observed')
            base_ok = base_ok and bool(lead.get('usable')) and isinstance(observed, bool)
            first = {'lead_observed': observed, 'decision': lead.get('decision'),
                     'rule_version': lead.get('rule_version'),
                     'lead_segment_count': len(lead.get('lead_segments', [])),
                     'common_valid_frames': lead.get('common_valid_frames', 0)}
            c1 = {'sustained_liquid_level_lead':component(first, physics=float(observed) if base_ok else None,
                    formula='1 if sustained normalized right-minus-left lead exceeds localization error; 0 if adequately observed without lead; null if unobservable')}
        else:
            # Explicit backward compatibility for archived raw records only.
            block, crushed = measures.get('block_water_surface_half_rise_time_s'), measures.get('crushed_water_surface_half_rise_time_s')
            violation = max(0.0, crushed/block-1.0) if present(block) and present(crushed) and block > 0 and crushed > 0 else None
            first = {'t_block':block,'t_crushed':crushed,'relative_order_violation':violation}
            c1 = {'relative_order_violation':component(violation,0.20,formula='legacy archived rule: 1/(1+max(0,t_crushed/t_block-1)/0.20)')}
    else:
        c1 = {'error':component(first, {'P9':0.10,'P13':0.05,'P11':0.10,'P20':0.05}[task])}
    if task == 'P11' and measures.get('observed_outcome') == 'ascent_only':
        base_ok=True;first={'observed_outcome':'ascent_only'}
        c1={'observed_ascent_only':component(first,physics=.1,formula='opinion_v2_ascent_only_score_0.1')}
    metrics, blocks = {}, {}
    for i,(key,value,components) in enumerate((('M1',first,c1),('M2',aux,c2))):
        if task in ('P9','P11','P20') and key == 'M2':
            metrics[key] = metric(defined=False)
            blocks[key] = {'defined':False,'principle':G7_LOGIC[task][i],
                'measurement_steps':[], 'measurements':{},
                'normalization':{'policy':'removed_by_review; excluded_from_denominator'},'evidence':[]}
            continue
        if task == 'P28' and key == 'M2':
            metrics[key] = metric(defined=False)
            blocks[key] = {'defined':False,'principle':G7_LOGIC[task][i],
                'measurement_steps':["Retain projected-area diagnostics without inferring three-dimensional mass from two-dimensional area."],
                'measurements':{'projected_area_diagnostic':value,'equal_initial_mass':'declared task prerequisite; not identifiable from this view'},
                'normalization':{'policy':'not_applicable; excluded from denominator'},'evidence':[]}
            continue
        physics = gm(c['physics_score'] for c in components.values())
        normalization = {'components':components,'aggregation':'geometric mean' if len(components) > 1 else 'single component'}
        if task == 'P20' and key == 'M2':
            coverage = min(len(components)/3,1)
            physics = physics*coverage if physics is not None else None
            normalization.update(coverage=coverage, aggregation='coverage * geometric mean of measured Snell ray scores')
        metrics[key] = metric(value,physics,base_ok)
        blocks[key] = measurement_block(G7_LOGIC[task][i],value,
            steps=["Decode frame by frame and apply the original geometry or motion extraction for this task.", "Retain the original measurement validity checks; physical errors alone do not remove the recognition reward.", "Map pure physics scores using fixed parameters and add the recognition reward once per metric."],
            measurements=measures,normalization=normalization,reason=("Recognized but measurement is unstable: insufficient reliable shared intervals or unstable trajectories." if task=='P9' and measures.get('identification_status')=='identified_measurement_unstable' else raw.get('failure_reason')))
        if task == 'P28' and key == 'M1' and measures.get('surface_lead') is not None:
            blocks[key]['failure_reason'] = None
            blocks[key]['reason'] = measures['surface_lead']['reason_zh']
            blocks[key]['measurement_steps'] = [
                "Confirm sustained ice shrinkage on both sides separately; compare only intervals with a visible liquid layer.",
                "Within the same frame, measure levels relative to each vessel base and normalize by vessel height.",
                "Record a temporary liquid-level lead when it exceeds the sum of localization errors for at least 0.25 seconds.",
                "Distinguish no observed lead during at least 0.50 seconds of continuous readable evidence from insufficient observations; do not interpolate gaps."]
    return metrics, blocks
