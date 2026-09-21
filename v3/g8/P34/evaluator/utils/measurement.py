"""Independent red-pole tracking, circular-dial calibration and M1 statistics."""
import math
import numpy as np
import cv2
from .common import ExtractionError


def _contours(binary):
    return cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]


def circular_calibration(ellipse, pivot):
    """Metric rectification of a circle whose physical centre is independently observed.

    A conic alone is insufficient for general perspective rectification. Its polar
    of the projected physical centre is the plane's vanishing line. Rectify that
    line, then whiten the resulting centred ellipse. The remaining rotation is
    constant and does not change signed angular differences.
    """
    (cx, cy), (d1, d2), angle = ellipse
    a = math.radians(angle)
    rotation = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    quadratic = rotation @ np.diag([4 / d1**2, 4 / d2**2]) @ rotation.T
    offset = np.array([cx, cy]) - pivot
    conic = np.zeros((3, 3))
    conic[:2, :2] = quadratic
    conic[:2, 2] = conic[2, :2] = -quadratic @ offset
    conic[2, 2] = offset @ quadratic @ offset - 1
    projective = np.eye(3)
    projective[2, :2] = conic[:2, 2] / conic[2, 2]
    inv = np.linalg.inv(projective)
    affine_conic = inv.T @ conic @ inv
    metric = affine_conic[:2, :2] / -affine_conic[2, 2]
    values, vectors = np.linalg.eigh(metric)
    if np.any(values <= 0):
        raise ExtractionError('Invalid circle/physical-centre calibration')
    whitening = np.eye(3)
    whitening[:2, :2] = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    translate = np.eye(3)
    translate[:2, 2] = -pivot
    return whitening @ projective @ translate


def transform_points(points, homography):
    points = np.atleast_2d(points)
    q = np.column_stack([points, np.ones(len(points))]) @ homography.T
    if np.any(np.abs(q[:, 2]) < 1e-8):
        raise ExtractionError('Point on rectification horizon')
    return q[:, :2] / q[:, 2, None]


def dial_geometry(frame, mask, cfg):
    ys, xs = np.where(mask)
    if len(xs) < 100:
        raise ExtractionError('Compass mask missing or too small')
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max()+1, ys.max()+1
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 1] < cfg.dial_white_saturation_max) & (hsv[..., 2] > cfg.dial_white_value_min)
             & mask[y0:y1, x0:x1]).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours = sorted(_contours(white), key=cv2.contourArea, reverse=True)
    if not contours or len(contours[0]) < 30:
        raise ExtractionError('No reliable dial boundary')
    # Tick marks cut inward from the white rim. The convex hull recovers the
    # outside dial boundary without fitting those dark, radial indentations.
    contour = cv2.convexHull(contours[0]).astype(np.float32)
    if len(contour) < 12:
        raise ExtractionError('Insufficient distributed points on the dial rim')
    ellipse_local = cv2.fitEllipse(contour)
    (cx, cy), diameters, orientation = ellipse_local
    if min(diameters) < 30 or min(diameters)/max(diameters) < .60:
        raise ExtractionError('Dial too small or strongly foreshortened')
    rotation_angle = math.radians(orientation)
    rotation = np.array([[math.cos(rotation_angle), -math.sin(rotation_angle)],
                         [math.sin(rotation_angle), math.cos(rotation_angle)]])
    points = (contour[:, 0] - [cx, cy]) @ rotation
    radial_error = np.abs(np.sqrt(np.sum((points / (np.array(diameters)/2))**2, axis=1))-1)
    residual = float(np.quantile(radial_error, .95))
    if residual > cfg.max_ellipse_residual:
        raise ExtractionError(f'Irregular dial boundary (relative residual={residual:.4f})')
    radius = min(diameters)/2
    yy, xx = np.indices(white.shape)
    central = (xx-cx)**2 + (yy-cy)**2 < (.26*radius)**2
    # The gold/brass pivot is a separate visual landmark, not the box centre.
    gold = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 45) & (hsv[..., 1] >= 65) & (hsv[..., 2] >= 50) & central
    gold_contours = sorted(_contours(gold), key=cv2.contourArea, reverse=True)
    if not gold_contours or cv2.contourArea(gold_contours[0]) < 4:
        raise ExtractionError('Cannot independently locate the physical pivot')
    g = gold_contours[0]
    moment = cv2.moments(g)
    pivot = np.array([moment['m10']/moment['m00']+x0, moment['m01']/moment['m00']+y0])
    pivot_area = cv2.contourArea(g)
    if pivot_area > .08 * math.pi*radius**2:
        raise ExtractionError('Pivot candidate too large')
    centre = np.array([cx+x0, cy+y0])
    if np.linalg.norm(pivot-centre) / radius > cfg.max_pivot_offset_fraction:
        raise ExtractionError('Pivot inconsistent with dial geometry')
    ellipse = (tuple(centre), diameters, orientation)
    H = circular_calibration(ellipse, pivot)
    # Contour round-trip verifies the calibration is numerically well-conditioned.
    circle_points = transform_points(contour[:, 0] + [x0, y0], H)
    circle_residual = float(np.quantile(np.abs(np.linalg.norm(circle_points, axis=1)-1), .95))
    if circle_residual > 2*cfg.max_ellipse_residual:
        raise ExtractionError('Circle rectification failed boundary validation')
    return {'pivot': pivot, 'radius': radius, 'ellipse': ellipse, 'H': H,
            'ellipse_residual': residual, 'rectified_circle_residual': circle_residual,
            'pivot_area': pivot_area}


def red_pole(frame, mask, geom, cfg, prediction=None):
    pivot, radius = geom['pivot'], geom['radius']
    x0, y0 = np.maximum(0, (pivot-1.15*radius).astype(int))
    x1, y1 = np.minimum(frame.shape[1::-1], (pivot+1.15*radius+1).astype(int))
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    yy, xx = np.indices(hsv.shape[:2])
    dist = np.hypot(xx+x0-pivot[0], yy+y0-pivot[1])
    red = (((hsv[..., 0] <= 12) | (hsv[..., 0] >= 170)) & (hsv[..., 1] >= 85) &
           (hsv[..., 2] >= 55) & (dist > .10*radius) & (dist < 1.05*radius) & mask[y0:y1, x0:x1])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    candidates = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < cfg.min_red_pixels:
            continue
        ys, xs = np.where(labels == label)
        points = np.column_stack([xs+x0, ys+y0]).astype(np.float32)
        vx, vy, lx, ly = cv2.fitLine(points, cv2.DIST_HUBER, 0, .01, .01).ravel()
        unit = np.array([vx, vy])
        if np.dot(unit, points.mean(axis=0)-pivot) < 0:
            unit = -unit
        normal = np.array([-unit[1], unit[0]])
        line_distance = abs(np.dot(pivot-[lx, ly], normal))
        residual = float(np.sqrt(np.mean(((points-[lx, ly]) @ normal)**2)))
        projections = (points-pivot) @ unit
        length = float(np.quantile(projections, .99))
        if not .4*radius < length < 1.1*radius or line_distance > cfg.max_needle_line_residual_px:
            continue
        if residual > cfg.max_needle_line_residual_px or np.ptp(projections) < .30*radius:
            continue
        tip = pivot + unit*length
        # Colour determines pole identity. Flow only disambiguates multiple red candidates.
        flow_distance = float(np.linalg.norm(tip-prediction)) if prediction is not None else None
        candidates.append({'tip': tip, 'red_pixels': len(points), 'line_residual_px': residual,
                           'pivot_line_distance_px': float(line_distance), 'flow_distance_px': flow_distance})
    if not candidates:
        raise ExtractionError('No reliable red pole attached to the pivot')
    if len(candidates) > 1:
        if prediction is None:
            raise ExtractionError('Ambiguous red-pole candidates')
        candidates = [c for c in candidates if c['flow_distance_px'] < .20*radius]
        if len(candidates) != 1:
            raise ExtractionError('Optical flow cannot resolve red-pole ambiguity')
    obs = candidates[0]
    # Use the reference calibration; freeze its arbitrary in-plane rotation.
    rectified = transform_points([pivot, obs['tip']], geom['H'])
    dx, dy = rectified[1]-rectified[0]
    obs['angle_deg'] = math.degrees(math.atan2(-dy, dx))
    obs['raw_angle_deg'] = math.degrees(math.atan2(-(obs['tip'][1]-pivot[1]), obs['tip'][0]-pivot[0]))
    return obs


def match_static(reference, current, exclusion=None):
    """Independent visual check of camera motion, without moving-needle features."""
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    allowed = None if exclusion is None else (~exclusion).astype(np.uint8)*255
    kp1, desc1 = orb.detectAndCompute(ref, allowed)
    kp2, desc2 = orb.detectAndCompute(cur, allowed)
    if desc1 is None or desc2 is None:
        return {'reliable': False, 'reason': 'No static feature descriptors'}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc1, desc2, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < .72*n.distance]
    if len(good) < 15:
        return {'reliable': False, 'reason': 'Too few static feature matches', 'matches': len(good)}
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp2[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2)
    if matrix is None or int(inliers.sum()) < 15:
        return {'reliable': False, 'reason': 'Static registration failed'}
    error = np.linalg.norm(src @ matrix[:, :2].T + matrix[:, 2] - dst, axis=1)
    good_error = float(np.median(error[inliers.ravel().astype(bool)]))
    return {'reliable': good_error < 1.5, 'inliers': int(inliers.sum()),
            'reprojection_error_px': good_error,
            'rotation_deg': math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])),
            'scale': float(np.hypot(matrix[0, 0], matrix[1, 0])),
            'shift_px': float(np.linalg.norm(matrix[:, 2])), 'matrix': matrix.tolist()}


def track_needles(frames, times, masks, cfg):
    calibrations, calibration_errors = [], []
    for j in range(2):
        try:
            calibrations.append(dial_geometry(frames[0], masks[0, j], cfg))
            calibration_errors.append(None)
        except ExtractionError as e:
            calibrations.append(None)
            calibration_errors.append(str(e))
    rows = []
    previous_gray, previous_tips = None, [None, None]
    exclusion = cv2.dilate(np.any(masks[0], axis=0).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    reference_area = np.sum(masks[0], axis=(1, 2))
    for i, (frame, t) in enumerate(zip(frames, times)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        camera = match_static(frames[0], frame, exclusion)
        camera_ok = (camera['reliable'] and abs(camera['rotation_deg']) <= cfg.max_camera_rotation_deg
                     and camera['shift_px'] <= cfg.max_camera_shift_px and abs(camera['scale']-1) <= .005)
        row = {'frame_index': i, 'time_sec': float(t), 'camera': camera, 'left': {}, 'right': {}}
        overlap = np.count_nonzero(masks[i, 0] & masks[i, 1]) / max(1, min(np.sum(masks[i], axis=(1, 2))))
        for j, name in enumerate(('left', 'right')):
            try:
                geom = calibrations[j]
                if geom is None:
                    raise ExtractionError(calibration_errors[j])
                if not camera_ok:
                    raise ExtractionError('Camera motion exceeds calibrated-frame bounds or registration is unreliable')
                if overlap > .05:
                    raise ExtractionError('Compass masks merged or overlapped')
                area_ratio = masks[i, j].sum()/max(reference_area[j], 1)
                if not .65 < area_ratio < 1.4:
                    raise ExtractionError('Compass mask area changed excessively')
                observed = dial_geometry(frame, masks[i, j], cfg)
                if np.linalg.norm(observed['pivot']-geom['pivot']) > cfg.max_pivot_drift_fraction*geom['radius']:
                    raise ExtractionError('Physical pivot moved or compass identity switched')
                prediction = None
                if previous_gray is not None and previous_tips[j] is not None:
                    origin = np.float32(previous_tips[j]).reshape(1, 1, 2)
                    dest, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, origin, None, winSize=(21,21), maxLevel=2)
                    if status is not None and status[0, 0]:
                        back, valid, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, dest, None, winSize=(21,21), maxLevel=2)
                        if valid[0, 0] and np.linalg.norm(back-origin) < 1.5:
                            prediction = dest[0, 0]
                current_geom = {**geom, 'pivot': observed['pivot']}
                obs = red_pole(frame, masks[i, j], current_geom, cfg, prediction)
                row[name] = {'valid': True, 'reason': None, 'pivot': observed['pivot'], **obs}
                previous_tips[j] = obs['tip']
            except ExtractionError as e:
                row[name] = {'valid': False, 'reason': str(e), 'angle_deg': None, 'raw_angle_deg': None,
                             'pivot': None, 'tip': None}
                previous_tips[j] = None
        rows.append(row)
        previous_gray = gray
    return rows, calibrations, calibration_errors


def unwrap_track(times, angles, cfg):
    """Do not bridge gaps whose direction/turn count cannot be observed."""
    result = np.full(len(angles), np.nan)
    valid = np.flatnonzero(np.isfinite(angles))
    if len(valid) < 3 or len(valid)/len(angles) < cfg.min_valid_fraction:
        raise ExtractionError('Insufficient valid needle observations')
    if times[valid[0]]-times[0] > cfg.max_gap_sec or times[-1]-times[valid[-1]] > cfg.max_gap_sec:
        raise ExtractionError('Needle track does not cover start/end')
    result[valid[0]] = angles[valid[0]]
    for previous, current in zip(valid[:-1], valid[1:]):
        dt = times[current]-times[previous]
        if dt > cfg.max_gap_sec + 1e-9 or cfg.max_speed_deg_sec * dt >= 180:
            raise ExtractionError('Unobserved gap makes angle unwrapping ambiguous')
        step = (angles[current]-angles[previous]+180) % 360 - 180
        if abs(step) > cfg.max_step_deg or abs(step)/dt > cfg.max_speed_deg_sec:
            raise ExtractionError('Discontinuous angle: possible pole switch or tracking failure')
        result[current] = result[previous]+step
    return result


def measure_image_reference(image, masks, calibrations, first_row, cfg):
    """Measure the registered supplied still, independently checking both red poles.

    The caller must first validate static scene correspondence and warp the image
    into the video's reference coordinates. Static matching alone is insufficient:
    a different time point of the same apparatus can match the background.
    """
    result = {'accepted': False, 'needles': {}, 'reason': None}
    try:
        for j, name in enumerate(('left', 'right')):
            geometry = calibrations[j]
            first = first_row[name]
            if geometry is None or not first['valid']:
                raise ExtractionError(f'{name}: first video frame has no reliable red-pole observation')
            observed = dial_geometry(image, masks[j], cfg)
            drift = float(np.linalg.norm(observed['pivot']-geometry['pivot']))
            if drift > cfg.max_pivot_drift_fraction*geometry['radius']:
                raise ExtractionError(f'{name}: supplied still pivot does not align with video')
            obs = red_pole(image, masks[j], {**geometry, 'pivot': observed['pivot']}, cfg)
            difference = (obs['angle_deg']-first['angle_deg']+180) % 360-180
            result['needles'][name] = {**obs, 'pivot': observed['pivot'],
                                     'difference_from_video_start_deg': float(difference)}
            if abs(difference) > cfg.initial_reference_tolerance_deg:
                raise ExtractionError(f'{name}: supplied still needle direction differs from video start')
        result['accepted'] = True
    except ExtractionError as e:
        result['reason'] = str(e)
    return result


def initial_reference(rows, times, tracks, cfg, image_reference=None):
    """Define the observed initial direction without requiring a stationary pre-roll."""
    details = {'source': 'video_first_frame', 'frame_index': 0, 'time_sec': float(times[0]),
               'stationarity_required': False, 'needles': {},
               'image_reference_check': image_reference}
    use_image = image_reference is not None and image_reference.get('accepted') is True
    for j, name in enumerate(('left', 'right')):
        if not rows[0][name]['valid'] or not np.isfinite(tracks[j][0]):
            raise ExtractionError(f'{name}: no reliable initial red-pole observation in video frame 0')
        angle = float(tracks[j][0])
        if use_image:
            measured = float(image_reference['needles'][name]['angle_deg'])
            difference = (measured-angle+180) % 360-180
            if not np.isfinite(measured) or abs(difference) > cfg.initial_reference_tolerance_deg:
                raise ExtractionError(f'{name}: supplied still cannot join the observed angle trajectory')
            # Put the independently observed still angle on the video's unwrap branch.
            angle += difference
        details['needles'][name] = {'angle_deg': angle, 'valid': True}
    if use_image:
        details['source'] = 'matched_first_frame_image'
        details['image_path'] = image_reference.get('image_path')
    return details


def final_stable_window(times, angle_tracks, cfg):
    if times[-1]-times[0] < cfg.stable_window_sec:
        raise ExtractionError('Video too short to establish a final stable window')
    start = max(0, int(np.searchsorted(times, times[-1]-cfg.stable_window_sec, side='right'))-1)
    indices = np.arange(start, len(times))
    details = {'start_time_sec': float(times[indices[0]]), 'end_time_sec': float(times[indices[-1]]),
               'start_frame': int(indices[0]), 'end_frame': int(indices[-1]), 'needles': {}}
    ok = True
    for name, track in zip(('left', 'right'), angle_tracks):
        valid_idx = indices[np.isfinite(track[indices])]
        vals = track[valid_idx]
        span = float(times[valid_idx[-1]]-times[valid_idx[0]]) if len(vals) else 0
        angle_range = float(np.ptp(vals)) if len(vals) else None
        valid = (len(vals) >= 3 and len(vals)/len(indices) >= cfg.min_valid_fraction
                 and span >= cfg.stable_window_sec-1e-9 and angle_range <= cfg.stable_angle_range_deg)
        details['needles'][name] = {'valid_frames': len(vals), 'valid_duration_sec': span,
            'angle_range_deg': angle_range, 'median_deg': float(np.median(vals)) if len(vals) else None,
            'mad_deg': float(np.median(np.abs(vals-np.median(vals)))) if len(vals) else None, 'stable': bool(valid)}
        ok &= valid
    details['stable'] = bool(ok)
    return details


def decide(delta1, delta2, cfg):
    """Continuous score; threshold conditions are explanatory diagnostics only."""
    if not math.isfinite(delta1) or not math.isfinite(delta2):
        raise ExtractionError('Cannot score non-finite deflections')
    tolerance = max(cfg.symmetry_abs_tol_deg, cfg.symmetry_rel_tol * max(abs(delta1), abs(delta2)))
    conditions = {'left_min_deflection': abs(delta1) >= cfg.min_deflection_deg,
                  'right_min_deflection': abs(delta2) >= cfg.min_deflection_deg,
                  'opposite_direction': delta1*delta2 < 0,
                  'symmetric_magnitude': abs(delta1+delta2) <= tolerance}
    # Normalize by observed motion rather than converting tolerance checks to 0/1.
    scale = max(abs(delta1), abs(delta2))
    if scale == 0:
        symmetry_error, symmetry_score, motion_score = 1.0, 0.0, 0.0
    else:
        left, right = delta1/scale, delta2/scale
        symmetry_error = min(1.0, max(0.0, abs(left+right)/(abs(left)+abs(right))))
        symmetry_score = 1.0-symmetry_error
        motion_score = min(1.0, min(abs(delta1),abs(delta2))/cfg.min_deflection_deg)
    metric = float(min(1.0,max(0.0,symmetry_score*motion_score)))
    details = {
        'version': 'signed_deflection_balance_v1', 'range': [0.0,1.0], 'higher_is_better': True,
        'formula': '(1 - abs(delta_1 + delta_2) / (abs(delta_1) + abs(delta_2))) * min(1, min(abs(delta_1), abs(delta_2)) / min_deflection_deg)',
        'zero_motion_policy': 'If both deflections are zero, the score is 0.0.',
        'normalized_symmetry_error': float(symmetry_error), 'symmetry_score': float(symmetry_score),
        'motion_score': float(motion_score)}
    return metric, conditions, tolerance, details


def summarize(rows, times, cfg, verbose, image_reference=None):
    tracks, failures = [], []
    for name in ('left', 'right'):
        angles = np.array([r[name]['angle_deg'] if r[name]['valid'] else np.nan for r in rows], dtype=float)
        try:
            unwrapped = unwrap_track(times, angles, cfg)
        except ExtractionError as e:
            failures.append(f'{name}: {e}')
            unwrapped = np.full(len(rows), np.nan)
        tracks.append(unwrapped)
        for row, value in zip(rows, unwrapped):
            row[name]['unwrapped_angle_deg'] = float(value) if np.isfinite(value) else None
    verbose['quality'] = {name+'_valid_fraction': float(np.mean([r[name]['valid'] for r in rows])) for name in ('left','right')}
    if failures:
        raise ExtractionError('; '.join(failures))
    pre = initial_reference(rows, times, tracks, cfg, image_reference)
    post = final_stable_window(times, tracks, cfg)
    verbose['initial_reference'] = pre
    verbose['windows'] = {'post': post}
    for k, name in enumerate(('left', 'right'), start=1):
        verbose['measurements'][f'theta_{k}_pre_deg'] = pre['needles'][name]['angle_deg']
        if post['needles'][name]['stable']:
            verbose['measurements'][f'theta_{k}_post_deg'] = post['needles'][name]['median_deg']
    if not post['stable']:
        raise ExtractionError('Final stable window absent; motion may be truncated')
    m = verbose['measurements']
    d1, d2 = m['theta_1_post_deg']-m['theta_1_pre_deg'], m['theta_2_post_deg']-m['theta_2_pre_deg']
    metric, conditions, tolerance, score_details = decide(d1, d2, cfg)
    m.update(delta_1_deg=d1, delta_2_deg=d2, opposite_direction=conditions['opposite_direction'],
             absolute_deflection_sum_deg=abs(d1+d2), applied_symmetry_tolerance_deg=tolerance)
    verbose['conditions'] = conditions
    verbose['score_details'] = score_details
    verbose['status'] = 'scored'
    verbose['reason'] = (f"Normalized M1 score = {score_details['symmetry_score']:.6f} symmetry "
                         f"* {score_details['motion_score']:.6f} motion = {metric:.6f}. "
                         'Threshold conditions are diagnostics, not a binary metric.')
    return metric
