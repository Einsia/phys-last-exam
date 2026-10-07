"""P10 support geometry measured only on actual visible quadrilaterals.

Missing poses and camera references remain missing. All sufficiently long
visible intervals are retained, irrespective of their physical score.
"""
import warnings
import math
import cv2
import numpy as np
from refined_evaluators.common import ExtractionError, write_json
from refined_evaluators.numeric import soft_error_score
from refined_evaluators.tasks import refine_solid_region, csv_file, plot
from observed_quad import quad_observation

VERSION = 'p10_observed_support_v4'


def normalized_measurement(verbose, previous_score):
    """Publish the observed event condition as part of the physical mapping."""
    from refined_evaluators.scoring import finite, residual_score
    m = verbose.get('measurements', {})
    needed = ('raw_m1_error','pivot_drift_p95_px','contact_gap_p95_px','observed_rotation_deg')
    if not all(finite(m.get(key)) and m[key] >= 0 for key in needed):
        raise ValueError('Missing independently observed support geometry')
    if len(m.get('scored_frames', [])) < 4 or not m.get('observed_intervals'):
        raise ValueError('Missing observed support intervals')
    event = m['observed_rotation_deg'] >= 5.
    if m.get('rotation_observed') is not event:
        raise ValueError('Rotation event flag disagrees with observed angles')
    scale = verbose['thresholds']['geometry_error_half_score']
    error = m['raw_m1_error']
    score = residual_score(error, scale) if event else 0.
    if not finite(previous_score) or not math.isclose(previous_score, score, abs_tol=1e-12):
        raise ValueError('Support score disagrees with independently recorded quantities')
    return error, score, {'type':'observed_event_conditioned_residual', 'error':error, 'scale':scale,
        'formula':'observed_rotation * 1/(1+resolved_geometry_error/geometry_error_half_score)',
        'observed_rotation':event, 'error_unit':'relative to observed block geometry', 'version':VERSION}


def observed_camera(xy, visibility, groups):
    ids = groups.get('fixed_reference', [])
    if not ids:
        return np.zeros((len(xy), 2)), np.zeros(len(xy))
    displacement = xy[:, ids].copy() - xy[0, ids]
    displacement[~visibility[:, ids]] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        offset = np.nanmedian(displacement, axis=1)
        residual = np.nanmedian(np.linalg.norm(displacement-offset[:, None], axis=2), axis=1)
    return offset, residual


def visible_intervals(valid, times, min_seconds=.3):
    ids = np.flatnonzero(valid)
    if not len(ids):
        return []
    dt = float(np.median(np.diff(times)))
    breaks = (np.diff(ids) != 1) | (np.diff(times[ids]) > 1.5*dt+1e-6)
    return [part for part in np.split(ids, np.flatnonzero(breaks)+1)
            if len(part) >= 4 and times[part[-1]]-times[part[0]] >= min_seconds-1e-9]


def measure(quads, times, offset, reference_residual, ground, config):
    quads, times, offset, reference_residual = [np.asarray(x, float) for x in
                                               (quads, times, offset, reference_residual)]
    valid = np.isfinite(quads).all((1, 2)) & np.isfinite(offset).all(1) & np.isfinite(reference_residual)
    spans = visible_intervals(valid, times)
    if not spans:
        raise ExtractionError('No continuous observed block/camera interval of at least 0.3 seconds')
    ids = np.concatenate(spans)
    first = int(ids[0])
    q = quads-offset[:, None, :]
    diagonal = float(np.linalg.norm(q[first, 0]-q[first, 2]))
    if diagonal < 10:
        raise ExtractionError('Block geometry too small for support measurement')
    ground = np.asarray(ground, float)
    axis = ground[-1]-ground[0]
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    support = q[:, 2]
    displacement = np.linalg.norm(support-support[first], axis=1)
    contact = np.abs((support-ground[0]) @ normal)
    edges = np.linalg.norm(np.roll(q, -1, axis=1)-q, axis=2)
    rigid = np.max(np.abs(edges/edges[first]-1), axis=1)
    noise = max(float(config['geometry_noise_px']), float(np.percentile(reference_residual[ids], 95)))
    drift_px = float(np.percentile(displacement[ids], 95))
    contact_px = float(np.percentile(contact[ids], 95))
    pivot_error = max(0., drift_px-2*noise)/diagonal
    contact_error = max(0., contact_px-2*noise)/diagonal
    shape_error = max(0., float(np.percentile(rigid[ids], 95))-2*noise/max(float(np.min(edges[first])), 1.))
    error = max(pivot_error, contact_error, shape_error)
    theta = np.full(len(times), np.nan)
    rotations = []
    for span in spans:
        vectors = (q[span, 0]+q[span, 1]-q[span, 2]-q[span, 3])/2
        angles = np.degrees(np.unwrap(np.arctan2(vectors[:, 0], -vectors[:, 1])))
        theta[span] = angles
        rotations.append(float(np.ptp(angles)))
    rotation = max(rotations)
    event_observed = rotation >= 5.
    score = soft_error_score(error, config['geometry_error_half_score']) if event_observed else 0.
    measurements = {
        'pivot_drift_p95_px':drift_px, 'contact_gap_p95_px':contact_px,
        'initial_block_diagonal_px':diagonal, 'geometry_noise_px':noise,
        'pivot_drift_relative_resolved':pivot_error, 'contact_gap_relative_resolved':contact_error,
        'rigid_shape_error':shape_error, 'raw_m1_error':error,
        'observed_rotation_deg':rotation, 'rotation_observed':event_observed,
        'support_drift_max_px':float(np.max(displacement[ids])),
        'actual_pose_coverage':float(valid.mean()), 'scored_pose_coverage':float(len(ids)/len(times)),
        'reference_frame':first, 'scored_frames':ids.tolist(),
        'observed_intervals':[[int(s[0]), int(s[-1])] for s in spans],
        'interval_rotations_deg':rotations, 'missing_poses_interpolated':False,
        'hidden_rotation_measured':False,
    }
    if not event_observed:
        measurements['observed_event_zero_reason'] = "No toppling rotation of at least 5 degrees is observed in continuous visible intervals; retain measured displacement and corners and assign a physics score of 0. Motion during unobserved intervals is unknown."
    # Diagnostics preserve holes; no fabricated corner enters an overlay.
    diagnostic = {'valid':valid, 'theta':theta, 'displacement':displacement,
                  'contact':contact, 'rigid':rigid, 'support':support, 'noise':noise}
    return score, measurements, diagnostic


def evaluate(context):
    c = context
    offset, residual = observed_camera(c.xy, c.vis, c.groups)
    quads, fit_errors, previous = [], [], None
    for frame, mask in zip(c.frames, c.masks[:, 0]):
        observation = quad_observation(refine_solid_region(frame, mask), previous)
        if observation is None:
            quads.append(np.full((4, 2), np.nan)); fit_errors.append(None)
        else:
            previous = observation[0]
            quads.append(previous); fit_errors.append(observation[1])
    quads = np.asarray(quads)
    write_json(c.out/'corners.json', {'corners_TL_TR_BR_BL':quads,
        'fit_area_relative_errors':fit_errors, 'missing_poses_interpolated':False})
    write_json(c.out/'camera_motion.json', {'offset_xy':offset, 'reference_residual_px':residual})
    c.annotations['block'] = {'observed_quads':quads, 'trajectory_xy':quads.mean(1)}
    score, measurements, data = measure(quads, c.t, offset, residual, c.a['geometry']['ground_points'], c.cfg)
    c.m.update(
        principle="Under continuous external pushing, use observed corners to check support contact, ground distance, and rigid-body shape; hidden intervals remain unknown.",
        measurements=measurements,
        score_details={'version':VERSION,
            'formula':'observed_rotation * 1/(1+max(resolved_pivot_drift, resolved_contact_error, rigid_shape_error)/geometry_error_half_score)',
            'range':[0,1], 'half_score_relative_error':c.cfg['geometry_error_half_score'],
            'selection':'all continuous observed intervals >= 0.3 seconds and >= 4 frames; no score-based interval selection'},
        uncertainty={'pixel_noise_floor':data['noise'], 'resolution_allowance_px':2*data['noise'],
                     'kind':'image measurement resolution; missing intervals are not interpolated'},
        applicability={'driving':'continuous external actuator as specified by the task',
            'free_instability_onset_required':False, 'ground_geometry':'observed static first-frame support line',
            'limitations':'checks observed support kinematics only; hidden motion, forces, friction and 3-D depth are not measured'},
        reason=measurements.get('observed_event_zero_reason') or "Measure support and rigid-body relationships across all continuous visible intervals; retain observed coverage and missing intervals without requiring 95% corner coverage over the whole clip.")
    csv_file(c.out/'pose_events.csv', {'frame':np.arange(len(c.t)), 'time_sec':c.t,
        'theta_deg':data['theta'], 'pivot_drift_px':data['displacement'],
        'contact_gap_px':data['contact'], 'rigid_shape_relative_change':data['rigid']})
    plot(c.out/'pose_events.png', c.t, {'pivot drift / diagonal':data['displacement']/measurements['initial_block_diagonal_px'],
        'contact gap / diagonal':data['contact']/measurements['initial_block_diagonal_px'],
        'shape change':data['rigid']}, 'Relative observed geometry error')
    write_json(c.out/'events.json', {'support_position_px':data['support'], 'rotation_deg':data['theta'],
        'observed_intervals':measurements['observed_intervals'], 'missing_poses_interpolated':False})
    c.calculation = [f'Observed pose coverage={measurements["actual_pose_coverage"]:.5f}; scored intervals={measurements["observed_intervals"]}.',
        f'Observed rotation={measurements["observed_rotation_deg"]:.5f} degrees; support geometry error={measurements["raw_m1_error"]:.8f}; score={score:.8f}.']
    return score


def install():
    """Register this backend for P10; leave all other task implementations intact."""
    from refined_evaluators import tasks, export, scoring
    tasks.TASKS['P10'] = evaluate
    previous_mapping = scoring.measurement_and_physics

    def mapping(task, verbose, score):
        if task == 'P10' and verbose.get('score_details', {}).get('version') == VERSION:
            return normalized_measurement(verbose, score)
        return previous_mapping(task, verbose, score)
    scoring.measurement_and_physics = mapping
    previous_render = export.render_geometry

    def render(canvas, context, index):
        canvas = previous_render(canvas, context, index)
        if context.task == 'P10':
            quad = context.annotations.get('block', {}).get('observed_quads')
            if quad is not None and np.isfinite(quad[index]).all():
                cv2.polylines(canvas, [np.rint(quad[index]).astype(np.int32)], True, (0,255,255), 2)
        return canvas
    export.render_geometry = render
