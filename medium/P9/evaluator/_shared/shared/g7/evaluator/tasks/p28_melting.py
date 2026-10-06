"""P28's independent, frame-zero ice identities and visible-melting gate."""
from pathlib import Path

from .p28_observed_melting import observe_melting
from .p28_vessels import locate_vessels


def initial_ice_region(frame_shape,vessel_roi):
    """Use observed apparatus scale, with room for ice touching the glass base.

    DINO's existing relative-size guard then rejects an entire vessel. Margins
    remain inside the visible image and also keep a resting ice body away
    from the segmentation observation boundary; the tight wall box is not an
    occlusion/exiting-frame test.
    """
    height,width=frame_shape[:2];x,y,w,h=vessel_roi
    px=max(8,round(.12*w));py=max(8,round(.05*h))
    return [max(0,x-px),max(0,y-py),min(width,x+w+px),min(height,y+h+py)]


def observe_pair_melting(frames, debug_dir):
    """Observe each declared side before interpreting any liquid-level change.

    Observed first-frame vessels constrain ice identity where available; the
    declared left/right regions are the fallback search space. No temporal
    difference, liquid-level trajectory, final reference or expected time ratio
    is supplied to the ice detector. Opaque bodies use observed color; realistic
    translucent ice uses the same local DINO/SAM2 route as P25/P27.
    """
    height, width = frames[0].shape[:2]
    vessels,geometry=locate_vessels(frames[0])
    observations = {}
    for side, lo, hi in (("left", .02, .475), ("right", .525, .98)):
        vessel = initial_ice_region(frames[0].shape,vessels[side]) if side in vessels else [int(lo * width), int(.04 * height), int(hi * width), int(.98 * height)]
        # The seed region is an identity search region, not a fabricated ice
        # box: the shared helper must resolve an actual first-frame body.
        seed = [vessel[0] + 4, vessel[1] + 4, vessel[2] - 4, vessel[3] - 4]
        observations[side] = observe_melting(
            frames, vessel, seed, debug_dir=Path(debug_dir) / ("ice_" + side))
        observations[side].update(initialization_vessel_roi=list(vessels[side]) if side in vessels else None,
                                  initialization_search_region_xyxy=vessel,
                                  initialization_geometry_source='observed_first_frame_vessel_with_visible_margin' if side in vessels else 'declared_side_search_region')
    failures = []
    for side, observation in observations.items():
        if not observation.get("initial_identity_resolved", False):
            failures.append("P28_" + side + "_initial_ice_identity_unresolved")
        elif not observation.get("melting_observed", False):
            failures.append("P28_" + side + "_" + (observation.get("reason") or "visible_melting_not_observed"))
    return observations, failures
