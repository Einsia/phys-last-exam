"""P2 ball identity from frame-zero appearance, independent of its motion.

Background subtraction can merge a visible ball with illumination changes.
When the initial frame contains one unambiguous chromatic circular object,
follow its observed colour component instead. Each accepted position comes
from that frame's pixels; no fall law, temporal interpolation or score is used.
The existing achromatic-object route remains available when no such identity
can be established.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..track import Track, chroma, find_color_targets, track_moving_blob


def _shape(labels, stats, index):
    x, y, w, h, area = map(int, stats[index])
    mask = (labels[y:y+h, x:x+w] == index).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float('inf'), 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * cv2.contourArea(contour) / max(perimeter**2, 1)
    return max(w, h) / max(min(w, h), 1), area / max(w*h, 1), float(circularity)


def initial_identity(frame):
    """Require one compact, round coloured object; never use future motion."""
    candidates = []
    c = chroma(frame)
    for target in find_color_targets(frame, k=8, max_area_frac=.04, sat_min=80):
        mask = cv2.morphologyEx(target.mask(frame), cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        _, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        index = int(labels[int(round(target.y)), int(round(target.x))])
        if not index:
            continue
        aspect, fill, circularity = _shape(labels, stats, index)
        if aspect > 1.45 or fill < .55 or circularity < .70 or np.median(c[labels == index]) < 25:
            continue
        if any(np.hypot(target.x-t.x, target.y-t.y) < target.radius for t in candidates):
            continue
        candidates.append(target)
    return candidates[0] if len(candidates) == 1 else None


def track_ball(clip):
    target = initial_identity(clip[0])
    if target is None:
        return track_moving_blob(clip)
    xs, ys, radii = [np.full(clip.n, np.nan) for _ in range(3)]
    previous = np.array([target.x, target.y], dtype=float)
    seed_area = np.pi * target.radius**2
    max_jump = max(90., 4. * target.radius)
    diagnostics = []
    kernel = np.ones((3,3), np.uint8)
    for i in range(clip.n):
        mask = cv2.morphologyEx(target.mask(clip[i]), cv2.MORPH_OPEN, kernel)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        for j in range(1, count):
            area = float(stats[j, cv2.CC_STAT_AREA])
            if not .35 * seed_area <= area <= 2.5 * seed_area:
                continue
            aspect, fill, circularity = _shape(labels, stats, j)
            if aspect > 2. or fill < .4 or circularity < .5:
                continue
            distance = float(np.linalg.norm(centers[j] - previous))
            candidates.append((distance, j, area, aspect, circularity))
        candidates.sort()
        # A unique appearance match is observable even after a large jump or
        # an occlusion. Never let a stale coordinate permanently lock it out.
        # With multiple matches, proximity may disambiguate only local ones.
        unique_match = len(candidates) == 1
        if not unique_match:
            candidates = [candidate for candidate in candidates if candidate[0] <= max_jump]
        if not candidates:
            diagnostics.append({'frame': i, 'observed': False, 'reason': 'No unique global or local size/shape/colour match'})
            continue
        if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < .5 * target.radius:
            diagnostics.append({'frame': i, 'observed': False, 'reason': 'Multiple appearance-matched components have ambiguous proximity'})
            continue
        distance, index, area, aspect, circularity = candidates[0]
        xs[i], ys[i] = centers[index]
        radii[i] = np.sqrt(area/np.pi)
        previous = centers[index].copy()
        diagnostics.append({'frame': i, 'observed': True, 'x': float(xs[i]), 'y': float(ys[i]),
                            'area_px': area, 'aspect': aspect, 'circularity': circularity,
                            'globally_unique_match': unique_match,
                            'reacquired_beyond_local_radius': distance > max_jump})
    return Track(x=xs, y=ys, r=radii, label='frame0_ball', quantities={
        'extractor': 'unique frame-zero chromatic ball; independent colour/shape observations',
        'seed_xy': [target.x, target.y], 'seed_hue': target.hue, 'seed_radius': target.radius,
        'hidden_frames_interpolated': False, 'physics_used_to_select_track': False,
        'frame_observations': diagnostics})
