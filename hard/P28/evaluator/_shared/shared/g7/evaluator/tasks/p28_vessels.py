"""Locate P28's open vessels from observed walls in one BGR image.

Only long vertical wall edges and their connecting bottom are used.  Image
halves name an observed vessel; they never provide substitute vessel bounds.
The detector deliberately leaves clipped, unconnected or ambiguous sides out.
"""
from __future__ import annotations

from itertools import combinations
import cv2
import numpy as np


def _wall_groups(vertical, minimum_height, merge_gap):
    _, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
    pieces = []
    for x, y, w, h, area in stats[1:]:
        if h >= minimum_height and w <= max(8, 3 * merge_gap):
            pieces.append(dict(x0=int(x), x1=int(x+w), y0=int(y), y1=int(y+h),
                               edge_pixels=int(area)))
    groups = []
    for piece in sorted(pieces, key=lambda p: p['x0']):
        candidates = [g for g in groups
                      if piece['x0']-g['x1'] <= merge_gap
                      and min(piece['y1'], g['y1'])-max(piece['y0'], g['y0'])
                      >= .45 * min(piece['y1']-piece['y0'], g['y1']-g['y0'])]
        if not candidates:
            groups.append(piece.copy())
            continue
        group = candidates[-1]
        for key in ('x0', 'y0'):
            group[key] = min(group[key], piece[key])
        for key in ('x1', 'y1'):
            group[key] = max(group[key], piece[key])
        group['edge_pixels'] += piece['edge_pixels']
    return groups


def _bottom(gy, left, right, threshold):
    """Require a broad horizontal edge close to both observed wall ends."""
    height = gy.shape[0]
    x0, x1 = left['x1'], right['x0']
    wall_height = min(left['y1']-left['y0'], right['y1']-right['y0'])
    lo = max(0, int(min(left['y1'], right['y1'])-.035*wall_height))
    hi = min(height, int(max(left['y1'], right['y1'])+.07*wall_height)+1)
    if hi <= lo or x1-x0 < 16:
        return None
    # Small local tolerance follows a rendered curved base without admitting
    # a distant table edge. No candidate extends the wall endpoints by much.
    local = cv2.dilate(gy, np.ones((5, 1), np.uint8))
    edge = local[lo:hi, x0:x1] >= threshold
    coverage = edge.mean(axis=1)
    flank = max(3, int(.15*(x1-x0)))
    joins = np.minimum(edge[:, :flank].mean(axis=1), edge[:, -flank:].mean(axis=1))
    strengths = np.median(local[lo:hi, x0:x1], axis=1)
    eligible = (coverage >= .55) & (joins >= .35)
    if not eligible.any():
        return None
    scores = coverage + .35*joins + .10*np.minimum(strengths/threshold, 4)
    scores[~eligible] = -1
    selected = int(np.argmax(scores))
    # Use the strongest actual gradient row, not the dilation's leading row.
    approximate = lo+selected
    ys = np.arange(max(lo, approximate-2), min(hi, approximate+3))
    row = int(ys[np.argmax(np.median(gy[ys, x0:x1], axis=1))])
    return {'y': row, 'coverage': float(coverage[selected]),
            'endpoint_support': float(joins[selected]), 'score': float(scores[selected])}


def locate_vessels(frame):
    """Return ({side: (x, y, width, height)}, JSON-safe geometry diagnostics).

    The half-open ROI includes up to five pixels of each observed inner-wall
    edge band so an independent surface detector can inspect the wall contact.
    It runs from the observed wall top to the connecting base, excluding the
    base row itself. Missing sides have a reason in diagnostics['sides'].
    """
    diagnostic = {'method': 'first_frame_vertical_walls_and_connected_bottom_v1',
                  'initialization_frame': 0, 'future_frames_used': False,
                  'sides': {}, 'wall_candidates': [], 'vessel_candidates': []}
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 48:
        diagnostic['reason'] = 'invalid_or_too_small_bgr_frame'
        diagnostic['sides'] = {s: {'reason': diagnostic['reason']} for s in ('left', 'right')}
        return {}, diagnostic
    h, w = array.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    signed_gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gx = np.abs(signed_gx)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.125))
    threshold = 6.0
    minimum_height = max(24, int(round(.23*h)))
    merge_gap = max(3, int(round(.012*w)))
    vertical = (gx >= threshold).astype(np.uint8)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE,
                               np.ones((max(3, round(.01*h)), 1), np.uint8))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN,
                               np.ones((minimum_height, 1), np.uint8))
    groups = _wall_groups(vertical, minimum_height, merge_gap)
    walls = []
    rejected = []
    for group in groups:
        band = signed_gx[group['y0']:group['y1'], group['x0']:group['x1']]
        # A thin opaque wall or glass highlight has both sides. A filled
        # rectangular object only gives one persistent gradient polarity at
        # each outside boundary and must not become a fabricated container.
        positive = float(np.mean(np.any(band >= threshold, axis=1)))
        negative = float(np.mean(np.any(band <= -threshold, axis=1)))
        group.update(positive_edge_coverage=positive, negative_edge_coverage=negative)
        if min(positive, negative) >= .35:
            walls.append(group)
        else:
            rejected.append({**group, 'reason': 'single_sided_outline_not_resolved_thin_wall'})
    diagnostic.update(frame_size_wh=[w, h], gradient_threshold=threshold,
                      minimum_wall_height_px=minimum_height, wall_candidates=walls,
                      rejected_wall_candidates=rejected)
    candidates = []
    for li, ri in combinations(range(len(walls)), 2):
        left, right = walls[li], walls[ri]
        x0, x1 = left['x1'], right['x0']
        width = x1-x0
        y0 = max(left['y0'], right['y0'])
        overlap = min(left['y1'], right['y1'])-y0
        if width < max(24, .075*w) or width > .46*w or overlap < minimum_height:
            continue
        if not .80 <= overlap/width <= 6.0:
            continue
        if abs(left['y0']-right['y0']) > .12*overlap:
            continue
        if abs(left['y1']-right['y1']) > .15*overlap:
            continue
        # Do not promote an ice silhouette to a vessel when a larger enclosing
        # pair of walls is present but clipped or otherwise unresolved.
        enclosing_left = [p for p in walls if p['x1'] <= left['x0']
                          and p['y0'] < y0-.15*overlap
                          and p['y1'] >= left['y1']-.10*overlap]
        enclosing_right = [p for p in walls if p['x0'] >= right['x1']
                           and p['y0'] < y0-.15*overlap
                           and p['y1'] >= right['y1']-.10*overlap]
        if enclosing_left and enclosing_right:
            continue
        if y0 <= 1 or max(left['y1'], right['y1']) >= h-1:
            continue
        bottom = _bottom(gy, left, right, threshold)
        if bottom is None:
            continue
        floor = bottom['y']
        if floor-y0 < minimum_height:
            continue
        side = 'left' if (x0+x1)/2 < w/2 else 'right'
        # A candidate spanning both image halves cannot safely establish a
        # left/right identity in this two-vessel task.
        if x0 < w/2 < x1:
            continue
        score = bottom['score'] + overlap/h
        roi_left = x0-min(5, left['x1']-left['x0'])
        roi_right = x1+min(5, right['x1']-right['x0'])
        candidate = {'side': side, 'roi': [roi_left, y0, roi_right-roi_left, floor-y0],
                     'inner_wall_bounds_x': [x0, x1],
                     'wall_indices': [li, ri], 'bottom': bottom,
                     'wall_overlap_px': overlap, 'score': float(score)}
        candidates.append(candidate)
    diagnostic['vessel_candidates'] = candidates
    rois = {}
    for side in ('left', 'right'):
        found = sorted((c for c in candidates if c['side'] == side),
                       key=lambda c: c['score'], reverse=True)
        if not found:
            diagnostic['sides'][side] = {'reason': 'no_connected_wall_pair_and_bottom'}
            continue
        # Competing non-overlapping vessels in one half are ambiguous; do not
        # silently select one or use an image-half rectangle as a replacement.
        best = found[0]
        def separate(candidate):
            x, y, width, height = best['roi']
            cx, cy, cw, ch = candidate['roi']
            intersection = max(0, min(x+width, cx+cw)-max(x, cx)) * max(
                0, min(y+height, cy+ch)-max(y, cy))
            # Contents can also have vertical edges and touch the same base.
            # Their smaller nested rectangle is not a second separate vessel.
            return intersection < .5*min(width*height, cw*ch)
        rivals = [c for c in found[1:] if c['score'] >= .85*best['score']
                  and not set(c['wall_indices']) & set(best['wall_indices'])
                  and separate(c)]
        if rivals:
            diagnostic['sides'][side] = {'reason': 'ambiguous_multiple_vessels',
                                        'candidate_count': len(found)}
            continue
        rois[side] = tuple(best['roi'])
        diagnostic['sides'][side] = {'reason': None, 'roi': best['roi'],
                                    'inner_wall_bounds_x': best['inner_wall_bounds_x'],
                                    'wall_indices': best['wall_indices'],
                                    'bottom': best['bottom'], 'score': best['score']}
    diagnostic['reason'] = None if len(rois) == 2 else 'vessel_geometry_incomplete'
    return rois, diagnostic
