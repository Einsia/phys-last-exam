"""Frame-zero rod identities and observed dark ridges on photographic floors.

The initializer contains no shadow directions. Rod registration and every
shadow sample come from the measured frame. A ridge must be darker than BOTH
sides and visibly connected to its rod; illumination gradients are not ridges.
No common-light-source geometry is used to find or choose a shadow.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def fingerprint(frame):
    value = np.ascontiguousarray(frame)
    h = hashlib.sha256(str(value.shape).encode('ascii'))
    h.update(value.tobytes())
    return h.hexdigest()


def load_initialization(path, clip):
    data = json.loads(Path(path).read_text())
    h, w = clip[0].shape[:2]
    if (data.get('task_id') != 'P19'
            or data.get('coordinate_frame') != 'decoded_video_frame_0'
            or data.get('annotation_type') != 'reviewed_frame0_rods_v1'
            or data.get('size_wh') != [w, h]
            or data.get('source_frame0_sha256') != fingerprint(clip[0])
            or data.get('source_video_sha256') != hashlib.sha256(Path(clip.path).read_bytes()).hexdigest()
            or not data.get('review', {}).get('accepted')):
        raise ValueError('P19 rod initialization is not reviewed or does not match this video/frame')
    rods = data.get('objects', [])
    if len(rods) < 3 or len({r['id'] for r in rods}) != len(rods):
        raise ValueError('P19 needs at least three distinct first-frame rod identities')
    for rod in rods:
        x0, y0, x1, y1 = rod['bbox_xyxy']
        x, y = rod['foot_xy']
        if not (0 <= x0 < x < x1 <= w and 0 <= y0 < y < y1 <= h):
            raise ValueError('P19 first-frame rod box/foot is outside the decoded frame')
    return data


def gradient(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                         cv2.Sobel(gray, cv2.CV_32F, 0, 1))


def track_feet(frames, objects):
    """Keep only features observed in every intervening frame, with FB checks."""
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    points, initial, identities, counts = [], [], [], []
    for j, obj in enumerate(objects):
        x, y = obj['foot_xy']
        height = y-obj['bbox_xyxy'][1]
        rx, ry = max(10, round(.09*height)), max(8, round(.05*height))
        mask = np.zeros(previous.shape, np.uint8)
        cv2.rectangle(mask, (x-rx, y-ry), (x+rx, y+ry), 255, -1)
        found = cv2.goodFeaturesToTrack(previous, 20, .01, 3, mask=mask, blockSize=3)
        counts.append(0 if found is None else len(found))
        if found is not None:
            for p in found[:, 0]:
                points.append(p); initial.append(p); identities.append(j)
    points = np.array(points, np.float32).reshape(-1, 1, 2)
    initial, identities = np.array(initial), np.array(identities)
    params = dict(winSize=(21, 21), maxLevel=3,
                  minEigThreshold=1e-4,
                  criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, .01))
    for frame in frames[1:]:
        if not len(points):
            break
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nxt, good, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, **params)
        back, backward, _ = cv2.calcOpticalFlowPyrLK(current, previous, nxt, None, **params)
        keep = (good[:, 0] > 0) & (backward[:, 0] > 0) & (np.linalg.norm(back[:, 0]-points[:, 0], axis=1) < 1.5)
        points, initial, identities = nxt[keep], initial[keep], identities[keep]
        previous = current
    found = {}
    for j, obj in enumerate(objects):
        keep = identities == j
        if np.count_nonzero(keep) < max(4, .4*counts[j]):
            continue
        differences = points[keep, 0]-initial[keep]
        shift = np.median(differences, axis=0)
        residual = float(np.percentile(np.linalg.norm(differences-shift, axis=1), 75))
        tolerance = max(2., .006*frames[0].shape[0])
        if residual > tolerance:
            continue
        found[obj['id']] = {'shift': shift, 'continuously_observed_features': int(np.count_nonzero(keep)),
                            'initial_features': counts[j], 'translation_residual_px': residual,
                            'max_translation_residual_px': tolerance,
                            'interpolated_frames': 0}
    return found


def relocate_rods(first, frame, objects, tracked=None):
    """Independently match each actual first-frame shaft, with bounded motion."""
    old, new = gradient(first), gradient(frame)
    h, w = old.shape
    rods, diagnostics = [], []
    for obj in objects:
        box = np.array(obj['bbox_xyxy'], int)
        x0, y0, x1, y1 = box
        fx, fy = obj['foot_xy']
        height = fy-y0
        rx, ry = max(10, round(.075*height)), max(7, round(.045*height))
        bx0, by0, bx1, by1 = max(0, fx-rx), max(0, fy-ry), min(w, fx+rx+1), min(h, fy+ry+1)
        template = old[y0:y1, x0:x1]
        mx, my = max(8, round(.055*w)), max(8, round(.055*h))
        sx, sy = max(0, x0-mx), max(0, y0-my)
        ex, ey = min(w, x1+mx), min(h, y1+my)
        search = new[sy:ey, sx:ex]
        if float(np.std(template)) < .25:
            diagnostics.append({'id': obj['id'], 'reason': 'first-frame shaft has no resolved edges'})
            continue
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        base_response = cv2.matchTemplate(new, old[by0:by1, bx0:bx1], cv2.TM_CCOEFF_NORMED)
        yy, xx = np.indices(response.shape, dtype=np.float32)
        coordinates = np.stack([xx+sx+bx0-x0, yy+sy+by0-y0], axis=-1)
        base_response = _read(base_response, coordinates)
        response = .7*response + .3*base_response
        _, confidence, _, peak = cv2.minMaxLoc(response)
        dx, dy = sx+peak[0]-x0, sy+peak[1]-y0
        record = {'id': obj['id'], 'template_ncc': float(confidence), 'shift_xy': [int(dx), int(dy)]}
        diagnostics.append(record)
        flow = (tracked or {}).get(obj['id'])
        if flow:
            dx, dy = flow['shift']
            record.update({k: v for k, v in flow.items() if k != 'shift'})
            record['shift_xy'] = [float(dx), float(dy)]
            record['localization'] = 'continuous observed pedestal features'
        elif confidence < .45:
            record['reason'] = 'measured-frame shaft registration unresolved'
            continue
        else:
            record['localization'] = 'joint shaft and pedestal gradient registration'
        foot = np.array(obj['foot_xy'], float) + [dx, dy]
        moved = box + [dx, dy, dx, dy]
        rods.append({'id': obj['id'], 'foot': foot, 'bbox': moved,
                     'size': float(height), 'template_ncc': float(confidence)})
        record['measured_foot_xy'] = foot.tolist()
    return rods, diagnostics


def _read(lum, points):
    return cv2.remap(lum, points[..., 0].astype(np.float32),
                     points[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def relocate_pedestals(frame, expected_count):
    """Re-localize interchangeable rods after a large view change.

    Each proposal needs a resolved dark circular pedestal AND a thin bright
    or colored shaft ending at that pedestal. Dark shadow lines cannot supply
    the shaft. No shadow angle or expected lamp position is consulted.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = gray.shape
    color = frame.max(2).astype(np.float32)-frame.min(2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, .07*h,
                              param1=70, param2=14,
                              minRadius=max(3, round(.006*h)), maxRadius=round(.04*h))
    detected = cv2.createLineSegmentDetector().detect(gray)[0]
    if circles is None or detected is None:
        return [], [{'localization': 'global pedestal/shaft', 'reason': 'no joint circular pedestal and shaft proposals'}]
    lines = detected.reshape(-1, 4)
    proposals = []
    for x, y, radius in circles[0]:
        base = np.array([x, y], float)
        x0, y0 = max(0, int(x-radius)), max(0, int(y-radius))
        x1, y1 = min(w, int(x+radius)+1), min(h, int(y+radius)+1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = (xx-x)**2+(yy-y)**2 < (.75*radius)**2
        if not inside.any():
            continue
        center = float(np.median(gray[y0:y1, x0:x1][inside]))
        angles = np.arange(0, 2*np.pi, .1)
        ring = base+np.c_[np.cos(angles), np.sin(angles)]*(1.7*radius)
        dark_fraction = float(np.mean(_read(gray.astype(np.float32), ring[None])[0]-center > 20))
        if dark_fraction < .65:
            continue
        candidates = []
        for line in lines:
            a, b = line[:2].astype(float), line[2:].astype(float)
            length = np.linalg.norm(a-b)
            if length < max(2.5*radius, .035*h) or min(np.linalg.norm(a-base), np.linalg.norm(b-base)) > 1.8*radius:
                continue
            if np.linalg.norm(a-base) > np.linalg.norm(b-base):
                a, b = b, a
            direction = (b-a)/length
            normal = np.array([-direction[1], direction[0]])
            points = a+(b-a)*np.linspace(.15, .85, 30)[:, None]
            for offset in (-4., -2., 0., 2., 4.):
                middle = points+offset*normal
                left, right = middle+6*normal, middle-6*normal
                def contrast(image):
                    return float(np.median(_read(image, middle[None])[0]-np.maximum(
                        _read(image, left[None])[0], _read(image, right[None])[0])))
                light, colored = contrast(gray.astype(np.float32)), contrast(color)
                strength = max(light, colored)
                if strength >= 6.:
                    candidates.append((strength*np.sqrt(length), a+offset*normal, b+offset*normal,
                                       float(length), light, colored))
        if not candidates:
            continue
        strength, a, b, length, light, colored = max(candidates, key=lambda item: item[0])
        proposals.append({'foot': base, 'bbox': np.r_[np.minimum(base, b)-6, np.maximum(base, b)+6],
                          'size': length, 'shaft_segment': np.array([a, b]),
                          'pedestal_radius': float(radius), 'proposal_strength': float(strength),
                          'bright_shaft_contrast': light, 'colored_shaft_contrast': colored,
                          'dark_annulus_fraction': dark_fraction})
    # More unexplained apparatus than expected is ambiguous; do not choose a
    # favorable subset by its resulting geometry or physical score.
    if not 3 <= len(proposals) <= expected_count:
        return [], [{'localization': 'global pedestal/shaft', 'reason': 'ambiguous observed rod count',
                     'proposal_count': len(proposals), 'expected_count_from_frame0': expected_count}]
    proposals.sort(key=lambda item: float(item['foot'][0]))
    diagnostics = []
    for i, rod in enumerate(proposals):
        rod['id'] = f'observed_rod_{i+1}'
        diagnostics.append({'id': rod['id'], 'localization': 'current-frame circular pedestal and observed thin shaft',
                            'measured_foot_xy': rod['foot'].tolist(),
                            'observed_shaft_xy': rod['shaft_segment'].tolist(),
                            'pedestal_radius_px': rod['pedestal_radius'],
                            'bright_shaft_contrast': rod['bright_shaft_contrast'],
                            'colored_shaft_contrast': rod['colored_shaft_contrast'],
                            'dark_annulus_fraction': rod['dark_annulus_fraction'],
                            'identities': 'interchangeable rods; no cross-frame physical quantities inferred'})
    return proposals, diagnostics


def shadow_rays(frame, rods):
    """Read continuous, rod-attached local darkness; retain its actual samples."""
    lum = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    lum = cv2.GaussianBlur(lum, (3, 3), 0)
    h, w = lum.shape
    excluded = np.zeros((h, w), np.uint8)
    rod_bodies = np.zeros((h, w), np.uint8)
    for rod in rods:
        x0, y0, x1, _ = map(int, rod['bbox'])
        fx, fy = map(float, rod['foot'])
        if 'shaft_segment' in rod:
            top = rod['shaft_segment'][1]
            direction = top-rod['foot']
            direction /= np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            half = max(4., .02*rod['size'])
            polygon = np.array([rod['foot']+half*normal, top+half*normal,
                                top-half*normal, rod['foot']-half*normal])
            cv2.fillConvexPoly(rod_bodies, np.rint(polygon).astype(int), 1)
            cv2.line(excluded, (round(fx), round(fy)), tuple(np.rint(top).astype(int)), 1, 3)
            cv2.circle(excluded, (round(fx), round(fy)), round(rod['pedestal_radius']), 1, -1)
            continue
        cv2.rectangle(rod_bodies, (x0, y0), (x1, round(fy)), 1, -1)
        half = max(2, round(.012*rod['size']))
        cv2.rectangle(excluded, (round(fx)-half, y0), (round(fx)+half, round(fy)), 1, -1)
        cv2.circle(excluded, (round(fx), round(fy)), max(4, round(.035*rod['size'])), 1, -1)
    angles = np.repeat(np.deg2rad(np.arange(360, dtype=float)), 5)
    directions = np.c_[np.cos(angles), np.sin(angles)]
    normals = np.c_[-directions[:, 1], directions[:, 0]]
    # Resolve the dark line independently within the measured pedestal's
    # small center uncertainty. Do not force its pixels through one rounded
    # annotation point, nor align it to the expected light-source position.
    offsets = np.tile(np.linspace(-.004*np.hypot(w, h), .004*np.hypot(w, h), 5), 360)
    shadows, diagnostics = [], []
    for rod in rods:
        base, height = np.asarray(rod['foot']), rod['size']
        # Exclude the visible pedestal and its bright rim. Sampling its rim
        # as one flank reverses the contrast of an otherwise dark shadow.
        # The excluded neighborhood is recorded; it is never filled in.
        radii = np.arange(max(10., .15*height, 1.6*rod.get('pedestal_radius', 0.)),
                          min(3*height, .45*np.hypot(w, h)), 2.)
        origins = base[None, :] + normals*offsets[:, None]
        center = origins[:, None, :] + directions[:, None, :]*radii[None, :, None]
        widths = 4 + .06*radii
        left = center + normals[:, None, :]*widths[None, :, None]
        right = center - normals[:, None, :]*widths[None, :, None]
        valid = np.ones(center.shape[:2], bool)
        for pts in (center, left, right):
            valid &= (pts[..., 0] >= 1) & (pts[..., 0] < w-1) & (pts[..., 1] >= 1) & (pts[..., 1] < h-1)
            valid &= _read(excluded.astype(np.float32), pts) < .1
        # Dark outlines and glow fringes next to a bright shaft are object
        # appearance, not floor shadows. Only the CENTER uses this wider box;
        # widening the flank mask would also discard real oblique shadows.
        valid &= _read(rod_bodies.astype(np.float32), center) < .1
        deficit = np.minimum(_read(lum, left), _read(lum, right)) - _read(lum, center)
        supported = valid & (deficit >= 4.)
        profile = np.zeros(len(angles))
        runs = {}
        for j in range(len(angles)):
            # A cast shadow must begin just outside the foot, not at a distant
            # dark patch. Isolated texture pixels and long gaps remain absent.
            good = supported[j]
            if len(good) < 15 or np.count_nonzero(good[:8]) < 5:
                continue
            end = len(good)
            for k in range(3, len(good)):
                if not np.any(good[k-3:k+1]):
                    end = k-3
                    break
            if end < 15 or np.mean(good[:end]) < .75:
                continue
            observed = np.flatnonzero(good[:end])
            span = radii[observed[-1]]-radii[observed[0]]
            if span < max(28., .03*np.hypot(w, h)):
                continue
            profile[j] = float(np.median(deficit[j, observed])*np.sqrt(span))
            runs[j] = observed
        record = {'id': rod['id'], 'foot_xy': base.tolist(),
                  'excluded_pedestal_radius_px': float(radii[0]),
                  'candidate_directions': len(runs), 'accepted': False}
        diagnostics.append(record)
        if not runs:
            record['reason'] = 'no continuous dark ridge visibly attached to the measured rod foot'
            continue
        best = int(np.argmax(profile))
        observed = runs[best]
        theta = angles[best]
        contrast = float(np.median(deficit[best, observed]))
        # All values below are sampled from this frame. No hidden shadow tip,
        # expected lamp position or interpolated direction is supplied.
        record.update(accepted=True, theta_deg=float(np.rad2deg(theta)),
                      observed_line_origin_xy=origins[best].tolist(),
                      foot_center_offset_px=float(offsets[best]),
                      contrast_L=contrast, observed_length_px=float(radii[observed[-1]]),
                      observed_points_xy=center[best, observed].tolist(),
                      observed_deficit_L=deficit[best, observed].tolist())
        shadows.append({'cx': float(origins[best, 0]), 'cy': float(origins[best, 1]),
                        'rod_foot_xy': base.tolist(),
                        'axis': directions[best], 'theta_deg': float(np.rad2deg(theta)),
                        'contrast': contrast, 'level': 0., 'profile': -profile.reshape(360, 5).max(axis=1)[::2],
                        'half_len': float(radii[observed[-1]]), 'rod_size': height,
                        'observed_points_xy': record['observed_points_xy']})
    return shadows, diagnostics
