"""Same-frame, vessel-normalized liquid-level lead for P28.

This is a qualitative phase lead, not a melt-completion or mass measurement.
No initial/final liquid level or interpolated observation is required.
All thresholds below are explicit observation settings, not physical constants.
"""
from __future__ import annotations

import math
import numpy as np

RULE_VERSION = 'p28_sustained_liquid_lead_v1'
SURFACE_ERROR_PX = 2.0
BASE_ERROR_PX = 2.0
MIN_LEAD_SECONDS = 0.25
MIN_NEGATIVE_OBSERVATION_SECONDS = 0.50


def _runs(mask):
    padded = np.r_[False, np.asarray(mask, bool), False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    ends = np.flatnonzero(np.diff(padded.astype(int)) == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def compare_surface_lead(series, fps, melting):
    """Return lead_observed / no_lead_observed / unobservable with evidence.

    Positive evidence needs one continuous 0.25-second common-time segment.
    A negative is explicitly limited to observed segments and needs at least
    0.50 continuous seconds; it never asserts absence during missing frames.
    Both ice-shrink prerequisites and a visible liquid layer on both sides
    are required. Evaluation begins after both shrink events are confirmed.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('P28 requires a positive finite frame rate')
    required = max(3, math.ceil(MIN_LEAD_SECONDS * fps) + 1)
    negative_required = max(3, math.ceil(MIN_NEGATIVE_OBSERVATION_SECONDS * fps) + 1)
    result = {
        'rule_version': RULE_VERSION, 'decision': 'unobservable',
        'reason_zh': "Indeterminate: no real liquid levels are simultaneously readable on both sides yet.",
        'usable': False, 'lead_observed': None,
        'normalization': '(observed_vessel_base_y - liquid_surface_y) / observed_vessel_height',
        'comparison': 'right_crushed_minus_left_block_at_the_same_frame',
        'surface_localization_error_px': SURFACE_ERROR_PX,
        'base_localization_error_px': BASE_ERROR_PX,
        'minimum_lead_seconds': MIN_LEAD_SECONDS, 'minimum_lead_frames': required,
        'minimum_negative_observation_seconds': MIN_NEGATIVE_OBSERVATION_SECONDS,
        'minimum_negative_observation_frames': negative_required,
        'initial_final_levels_required': False, 'missing_frames_interpolated': False,
        'interpretation': 'Visible-liquid phase lead only; not melt completion or melted mass.',
    }
    if not all(melting.get(side, {}).get('melting_observed') for side in ('left', 'right')):
        result['reason_zh'] = "Indeterminate: sustained ice shrinkage is unconfirmed on at least one side; level changes alone cannot establish a melting lead."
        return result
    starts = [(melting[s].get('melting_evidence') or {}).get('confirmed_frame') for s in ('left', 'right')]
    if any(frame is None for frame in starts):
        result['reason_zh'] = "Indeterminate: frames confirming sustained melting on both sides are missing."
        return result
    start = int(max(starts))
    levels = {}; valid = {}; uncertainty = {}; side_summary = {}
    for side in ('left', 'right'):
        item = series.get(side, {})
        y = np.asarray(item.get('surface_y_px', []), float)
        okay = np.asarray(item.get('surface_valid', np.zeros(len(y), bool)), bool)
        height = float(item.get('vessel_height_px', 0))
        base = np.asarray(item.get('surface_floor_px', np.full(len(y), np.nan)), float)
        if height <= 0 or y.ndim != 1 or okay.shape != y.shape or base.shape != y.shape:
            result['reason_zh'] = "Indeterminate: vessel geometry or liquid-level observation arrays are incomplete."
            return result
        okay = okay & np.isfinite(y) & np.isfinite(base)
        kinds = item.get('surface_observation_kind')
        if kinds is not None:
            if len(kinds) != len(y): raise ValueError('Surface observation kinds do not match frames')
            okay &= np.array([kind in ('bilateral_visible_interface', 'observed_bottom_connected_opaque_liquid_interface') for kind in kinds])
        depth = base - y
        # A dry base or sub-resolution film is not evidence of liquid formation.
        okay &= (depth > SURFACE_ERROR_PX + BASE_ERROR_PX) & (depth < height)
        okay[:start] = False
        levels[side] = depth / height
        valid[side] = okay
        uncertainty[side] = (SURFACE_ERROR_PX + BASE_ERROR_PX) / height
        side_summary[side] = {'vessel_height_px': height, 'valid_liquid_frames': int(okay.sum()),
                              'normalized_localization_bound': uncertainty[side]}
    if len(levels['left']) != len(levels['right']):
        raise ValueError('Left and right surface observations must share decoded frames')
    joint = valid['left'] & valid['right']
    delta = levels['right'] - levels['left']
    threshold = uncertainty['left'] + uncertainty['right']
    common_runs = _runs(joint)
    leading_runs = _runs(joint & (delta > threshold))
    reverse_runs = _runs(joint & (delta < -threshold))
    qualified = [(a, b) for a, b in leading_runs if b-a+1 >= required]
    def describe(run):
        a, b = run
        return {'start_frame': a, 'end_frame': b, 'start_time_s': a/fps, 'end_time_s': b/fps,
                'frames': b-a+1, 'duration_s': (b-a)/fps,
                'minimum_normalized_difference': float(np.min(delta[a:b+1])),
                'median_normalized_difference': float(np.median(delta[a:b+1]))}
    longest = max((b-a+1 for a, b in common_runs), default=0)
    result.update(
        evaluation_start_frame=start, evaluation_start_time_s=start/fps,
        frame_count=len(joint), common_valid_frames=int(joint.sum()),
        common_valid_fraction=float(np.mean(joint)) if len(joint) else 0.,
        longest_common_run_frames=longest, side_observations=side_summary,
        normalized_difference_threshold=threshold,
        lead_segments=[describe(run) for run in qualified],
        reverse_lead_segments=[describe(run) for run in reverse_runs if run[1]-run[0]+1 >= required],
        common_observation_segments=[describe(run) for run in common_runs],
        observation_scope='Only directly observed same-frame liquid levels after both melting confirmations.',
    )
    if qualified:
        result.update(usable=True, decision='lead_observed', lead_observed=True,
                      reason_zh="Lead observed: the crushed-ice vessel has a higher normalized liquid level than the intact-ice vessel, beyond localization error for at least 0.25 seconds.")
    elif longest >= negative_required:
        result.update(usable=True, decision='no_lead_observed', lead_observed=False,
                      reason_zh="No lead observed: both liquid levels are continuously clear for at least 0.50 seconds, but no reliable lead persists for 0.25 seconds within the visible intervals; missing intervals are not inferred.")
    else:
        result['reason_zh'] = "Indeterminate: continuous intervals with both levels readable are insufficient; occlusion, ambiguity, or missing tracks cannot be treated as absence of a lead."
    curves = {'left_normalized_depth': levels['left'], 'right_normalized_depth': levels['right'],
              'left_valid': valid['left'], 'right_valid': valid['right'], 'joint_valid': joint,
              'normalized_difference': delta, 'lead_frame': joint & (delta > threshold)}
    return result | {'_curves': curves}
