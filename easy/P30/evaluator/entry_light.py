"""P30: independently observed insertion and lamp response, two half points."""
import cv2
import numpy as np
from refined_evaluators.common import ExtractionError, write_json
from refined_evaluators.vision import camera_motion

VERSION = 'p30_entry_light_v5'


def sustained_intervals(active, t, minimum):
    """Return actual consecutive observations, without filling missing frames."""
    active = np.asarray(active, bool)
    t = np.asarray(t, float)
    bounds = np.flatnonzero(np.diff(np.r_[False, active, False]))
    return [(int(a), int(b - 1)) for a, b in zip(bounds[::2], bounds[1::2])
            if t[b - 1] - t[a] >= minimum - 1e-9]


def observe_entry(front, rear, aperture_overlap, valid, t, entry, exit_, hysteresis, hold):
    """Observe a leading edge entering the opening; complete transit is optional."""
    front, rear, valid = np.asarray(front), np.asarray(rear), np.asarray(valid, bool)
    outside = valid & (front < entry - hysteresis)
    inside = valid & aperture_overlap & (front >= entry + hysteresis) & (rear <= exit_)
    entries = []
    for start, stop in sustained_intervals(inside, t, hold):
        prior = np.flatnonzero(outside[:start])
        if len(prior):
            entries.append({'start_frame': start, 'end_frame': stop,
                            'observed_entry_bracket_sec': [float(t[prior[-1]]), float(t[start])],
                            'inside_interval_sec': [float(t[start]), float(t[stop])]})
    if not entries:
        if valid.mean() < .8:
            raise ExtractionError('Magnet entry not observed and silhouette coverage is insufficient to prove absence')
        if not outside.any() and inside.any():
            raise ExtractionError('Magnet already inside coil: entry is left-censored')
    return entries, inside


def evaluate(c):
    from refined_evaluators.tasks import csv_file, plot
    c.m['principle'] = "Check only two visible events: magnet entry into the coil earns 0.5, and a visibly lit lamp during entry earns another 0.5; exit from the other end is not required."
    c.m['score_details'] = {'version': VERSION, 'formula': '0.5 * magnet_entered + 0.5 * lamp_lit_during_entry',
        'range': [0, 1], 'component_weights': {'magnet_entered': .5, 'lamp_lit_during_entry': .5},
        'full_transit_required': False, 'field_model_used': False,
        'new_light_onset_required': False, 'initial_dark_state_required': False,
        'lamp_window': 'Observed magnet-coil overlap interval plus configured onset tolerance; no centroid matching.'}
    t = np.asarray(c.t, float)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ExtractionError('Too few valid PTS observations for entry/light events')
    drift, camera_residual = camera_motion(c.xy, c.vis, c.groups)
    g = c.a['geometry']
    plane = np.asarray(g['entry_plane'], float)
    tangent = plane[1] - plane[0]
    opening_length = np.linalg.norm(tangent)
    if opening_length < 5:
        raise ExtractionError('Unresolved coil opening geometry')
    tangent /= opening_length
    normal = np.array([tangent[1], -tangent[0]])
    direction = np.asarray(g.get('pass_direction', [1., 0.]), float)
    if normal @ direction < 0:
        normal *= -1
    entry = float(plane.mean(0) @ normal)
    exit_ = float(np.asarray(g['exit_plane'], float).mean(0) @ normal)
    if exit_ <= entry:
        raise ExtractionError('Coil entry/exit geometry disagrees with pass direction')
    transverse = np.sort(plane @ tangent)
    names = [o['name'] for o in c.a['objects']]
    magnet = names.index('magnet')
    first_area = int(np.count_nonzero(c.masks[0, magnet]))
    if first_area < 8:
        raise ExtractionError('Initial magnet silhouette not observed')
    front, rear, centers, valid, overlap = [], [], [], [], []
    for i, mask in enumerate(c.masks[:, magnet]):
        yy, xx = np.where(mask)
        if len(xx) < max(8, .1 * first_area) or len(xx) > 5 * first_area:
            front.append(np.nan); rear.append(np.nan); centers.append([np.nan, np.nan])
            valid.append(False); overlap.append(False); continue
        points = np.c_[xx, yy].astype(float) - drift[i]
        axial = points @ normal
        across = points @ tangent
        lower, upper = np.percentile(axial, [2, 98])
        low_transverse, high_transverse = np.percentile(across, [2, 98])
        front.append(float(upper)); rear.append(float(lower)); centers.append(np.median(points, axis=0))
        valid.append(True)
        overlap.append(bool(high_transverse >= transverse[0] and low_transverse <= transverse[1]))
    front, rear = np.asarray(front), np.asarray(rear)
    centers, valid, overlap = np.asarray(centers), np.asarray(valid), np.asarray(overlap)
    hysteresis = max(1., c.cfg['crossing_hysteresis_coil_length_ratio'] * (exit_ - entry))
    entries, inside = observe_entry(front, rear, overlap, valid, t, entry, exit_,
                                    hysteresis, c.cfg['crossing_hold_sec'])
    lamp, background, neighborhood, core_bright_fraction, core_p99 = [], [], [], [], []
    lamp_box = np.asarray(c.a['objects'][names.index('lamp')]['box'], float)
    for i, frame in enumerate(c.frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        dx, dy = np.rint(drift[i]).astype(int)
        rois = {}
        for values, key in [(lamp, 'lamp_emission_box'), (background, 'lamp_background_box')]:
            x1, y1, x2, y2 = np.asarray(g[key], int) + [dx, dy, dx, dy]
            if x1 < 0 or y1 < 0 or x2 > gray.shape[1] or y2 > gray.shape[0] or x2 <= x1 or y2 <= y1:
                raise ExtractionError('Lamp/background photometry ROI is not completely visible')
            rois[key] = gray[y1:y2, x1:x2]
            values.append(float(np.mean(rois[key])))
        # The surrounding ring is measured in the same frame and excludes the
        # bulb itself. It supplies evidence for a lamp that was already on in
        # frame zero; no dark-to-light transition is required.
        lx1, ly1, lx2, ly2 = np.rint(lamp_box + [dx, dy, dx, dy]).astype(int)
        pad = max(4, int(round(.35 * max(lx2-lx1, ly2-ly1))))
        rx1, ry1 = max(0, lx1-pad), max(0, ly1-pad)
        rx2, ry2 = min(gray.shape[1], lx2+pad), min(gray.shape[0], ly2+pad)
        ring = gray[ry1:ry2, rx1:rx2]
        outside_bulb = np.ones(ring.shape, bool)
        outside_bulb[max(0, ly1-ry1):min(ring.shape[0], ly2-ry1),
                     max(0, lx1-rx1):min(ring.shape[1], lx2-rx1)] = False
        if outside_bulb.sum() < 16:
            raise ExtractionError('Insufficient visible lamp neighborhood for same-frame light evidence')
        neighbor = max(float(np.median(ring[outside_bulb])), background[-1])
        core = rois['lamp_emission_box']
        neighborhood.append(neighbor)
        core_bright_fraction.append(float(np.mean(core >= max(160., neighbor + 25.))))
        core_p99.append(float(np.percentile(core, 99)))
    lamp, background, neighborhood = map(np.asarray, (lamp, background, neighborhood))
    core_bright_fraction, core_p99 = map(np.asarray, (core_bright_fraction, core_p99))
    brightness = lamp - background
    initial_outside = valid & (front < entry - hysteresis) & (t <= t[0] + .5)
    baseline_ids = np.flatnonzero(initial_outside)
    baseline_available = bool(len(baseline_ids))
    baseline = float(np.median(brightness[baseline_ids])) if baseline_available else float(brightness[0])
    noise = max(1., 1.4826 * float(np.median(abs(brightness[baseline_ids] - baseline)))) if baseline_available else 1.
    threshold = baseline + c.cfg['brightness_on_sigma'] * noise
    temporal_lit = (brightness > threshold) if baseline_available else np.zeros(len(t), bool)
    same_frame_lit = core_bright_fraction >= .25
    lit = temporal_lit | same_frame_lit
    # Bright, low-contrast or tiny-highlight observations may be glass/reflection
    # or an unresolved filament. Preserve them as unknown, never as an unlit lamp.
    clear_dark = ((core_p99 < 80.) | ((lamp < neighborhood - 15.) &
                                    (core_p99 < neighborhood + 5.))) & ~lit
    unknown = ~lit & ~clear_dark
    light_intervals = sustained_intervals(lit, t, c.cfg['min_event_duration_sec'])
    unknown_intervals = sustained_intervals(unknown, t, c.cfg['min_event_duration_sec'])
    tolerance = float(c.cfg['onset_tolerance_sec'])
    matching, uncertain = [], []
    for intervals, matches in [(light_intervals, matching), (unknown_intervals, uncertain)]:
        for start, stop in intervals:
            for event in entries:
                lo, hi = event['inside_interval_sec']
                if t[stop] >= lo - tolerance and t[start] <= hi + tolerance:
                    matches.append({'light_start_sec': float(t[start]), 'light_end_sec': float(t[stop]),
                                    'entry_window_sec': [lo - tolerance, hi + tolerance]})
    entered = bool(entries)
    lamp_during_entry = True if matching else None if entered and uncertain else False
    # Each confirmed event earns its own half point. An unresolved lamp must
    # not erase an independently observed insertion or be labelled dark.
    score = .5 * entered + .5 * (lamp_during_entry is True)
    measurements = {'magnet_entered': entered, 'lamp_lit_during_entry': lamp_during_entry,
        'lamp_state_during_entry': 'lit' if matching else 'unknown' if entered and uncertain else 'dark' if entered else 'not_applicable',
        'entry_component_score': .5 if entered else 0.,
        'lamp_component_score': None if lamp_during_entry is None else .5 if lamp_during_entry else 0.,
        'entry_events': entries, 'lamp_matches': matching, 'lamp_uncertain_intervals': uncertain,
        'lamp_lit_observed_anywhere': bool(light_intervals),
        'same_frame_lit_fraction': float(same_frame_lit.mean()),
        'temporal_brightening_fraction': float(temporal_lit.mean()),
        'lamp_unknown_fraction': float(unknown.mean()),
        'entry_plane_axial_px': entry, 'exit_plane_axial_px': exit_,
        'magnet_silhouette_valid_fraction': float(valid.mean()),
        'brightness_baseline': baseline, 'temporal_baseline_available': baseline_available,
        'brightness_noise': noise, 'brightness_on_threshold': threshold}
    c.m['measurements'] = measurements
    c.m['uncertainty'] = {'entry_hysteresis_px': hysteresis,
        'score_is_confirmed_evidence_lower_bound': lamp_during_entry is None,
        'score_interval': [score, score + (.5 if lamp_during_entry is None else 0.)],
        'entry_timing_brackets_sec': [e['observed_entry_bracket_sec'] for e in entries],
        'lamp_timing_tolerance_sec': tolerance,
        'camera_reference_residual_max_px': float(np.nanmax(camera_residual))}
    c.m['applicability'] = {'checks': 'Visible magnet insertion and visibly lit lamp during entry; the lamp may have been on from the start.',
        'not_measured': 'Magnetic flux, induced voltage/current, full passage and lamp time centroid.',
        'lamp_calibration': 'Same-frame luminous-core contrast/area relative to the bulb neighborhood, or observed background-corrected brightening. No new onset or initial dark state required. Unresolved brightness/reflections remain unknown.',
        'identity': 'Reviewed first-frame magnet identity propagated by the segmentation model; no trajectory inferred across missing masks.'}
    c.m['reason'] = f'Magnet entered the coil: {entered}; visibly lit lamp during entry: {lamp_during_entry}; each event contributes 0.5.'
    c.annotations['magnet'] = {'trajectory_xy': centers + drift}
    c.annotations['events'] = {'crossings': [{'id': 'enter', 'time_sec': e['inside_interval_sec'][0]} for e in entries]}
    csv_file(c.out/'motion_brightness.csv', {'frame': np.arange(len(t)), 'time_sec': t,
        'magnet_center_x': centers[:, 0], 'magnet_front_axial_px': front,
        'magnet_rear_axial_px': rear, 'silhouette_valid': valid,
        'overlap_with_coil': inside, 'lamp_corrected': brightness,
        'lamp_above_threshold': temporal_lit, 'same_frame_lamp_lit': same_frame_lit,
        'lamp_state_unknown': unknown, 'lamp_neighborhood': neighborhood,
        'lamp_core_bright_fraction': core_bright_fraction, 'lamp_core_p99': core_p99})
    plot(c.out/'entry_light_events.png', t, {'magnet overlaps coil': inside.astype(float),
        'lamp visibly lit': lit.astype(float), 'lamp state unknown': unknown.astype(float)}, 'Observed binary events')
    write_json(c.out/'brightness_calibration.json', {'baseline': baseline, 'noise': noise,
        'baseline_frames': baseline_ids, 'temporal_baseline_available': baseline_available,
        'on_threshold': threshold, 'same_frame_rule': {'core_min_luminance': 160.,
            'core_min_neighbor_contrast': 25., 'minimum_bright_core_fraction': .25},
        'initial_dark_or_new_light_onset_required': False})
    write_json(c.out/'events.json', measurements)
    c.calculation = [f'Entry events measured from the observed leading silhouette: {entries}',
        f'Lamp threshold={threshold:.6f}; matched light intervals={matching}',
        f'P30 M1 components: entry={measurements["entry_component_score"]}; lamp={measurements["lamp_component_score"]}; score={score}.',
        'Partial insertion suffices. Full transit, field-model fitting, timing centroids and stationary-emission penalties are not part of this rule.']
    return score
