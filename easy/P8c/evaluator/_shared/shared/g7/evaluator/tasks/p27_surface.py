"""P27 liquid interfaces observed against independently located vessel walls.

No final-frame colour mask, desired rise direction, or expected side ordering
selects a surface. A dry origin is resolved only for an opaque flat schematic
whose exposed side columns are uniform down to the independently found base.
Photographic transparent containers require an actually observed initial level.
"""
from __future__ import annotations

import cv2
import numpy as np


def _side_columns(width):
    edge=max(3,round(.025*width));span=max(3,round(.045*width))
    return (slice(edge,edge+span),slice(width-edge-span,width-edge))


def _flat_dry_reference(roi,outside_background=None):
    """Expose a zero-depth origin only in resolved flat, opaque drawings."""
    # Uniform contents and dark walls alone cannot distinguish an empty cup
    # from an already filled opaque cup. Require independently visible exterior
    # background, rather than defining the contents' own colour to be empty.
    if outside_background is None:return None
    outside=np.asarray(outside_background,float).reshape(-1,3)
    if not len(outside):return None
    exterior=np.median(outside,axis=0)
    if np.percentile(np.linalg.norm(outside-exterior,axis=1),95)>24:return None
    h,w=roi.shape[:2];profiles=[]
    for columns in _side_columns(w):
        strip=roi[round(.25*h):h-4,columns].astype(float)
        if strip.size==0:return None
        color=np.median(strip.reshape(-1,3),axis=0)
        # A solid background down both sides of the ice is visible; reflective
        # glass and textured material do not satisfy this flat-colour identity.
        # Low-amplitude background/compression texture is allowed, while a
        # shaded/reflected column is not a resolved opaque empty background.
        if np.percentile(np.linalg.norm(strip-color,axis=2),95)>24:return None
        profiles.append(color)
    if np.linalg.norm(profiles[0]-profiles[1])>10:return None
    if any(np.linalg.norm(color-exterior)>12 for color in profiles):return None
    # Resolve opaque dark walls rather than assuming transparent liquid is
    # absent merely because a horizontal edge was not found.
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    interior=float(np.mean(profiles))
    walls=np.r_[gray[round(.2*h):h-8,:2].ravel(),gray[round(.2*h):h-8,-2:].ravel()]
    if interior-float(np.median(walls))<35:return None
    return np.mean(profiles,axis=0)


def _opaque_level(roi,dry_color,ice_mask=None):
    """Observe the top of a bottom-connected, matching liquid-colour layer.

    Only available after the exposed dry base was independently established.
    The liquid's current colour comes from this frame at both walls; no final
    frame supplies a prototype. Disconnected pixels cannot set its height.
    """
    h,w=roi.shape[:2];levels=[];colors=[]
    for columns in _side_columns(w):
        profile=np.median(roi[:,columns].astype(float),axis=1)
        if np.linalg.norm(np.median(profile[round(.15*h):round(.3*h)],axis=0)-dry_color)>15:
            return np.nan,'exposed_background_changed'
        bottom=h-4;color=profile[bottom]
        if np.linalg.norm(color-dry_color)<20:
            levels.append(float(h-3));colors.append(dry_color);continue
        same=np.linalg.norm(profile-color,axis=1)<18
        row=bottom
        while row>3 and same[row-1]:row-=1
        if bottom-row<1:return np.nan,'liquid_layer_below_pixel_resolution'
        levels.append(float(row));colors.append(color)
    if abs(levels[0]-levels[1])>3 or np.linalg.norm(colors[0]-colors[1])>20:
        return np.nan,'bilateral_liquid_layer_disagreement'
    level=float(np.mean(levels))
    if ice_mask is not None:
        for columns in _side_columns(w):
            row=int(round(level))
            contact=np.asarray(ice_mask,bool)[max(0,row-2):min(h,row+3),columns]
            if contact.size and np.mean(contact)>.15:return np.nan,'ice_occludes_liquid_contact'
    return level,'observed_dry_opaque_base' if level==h-3 else 'observed_bottom_connected_opaque_liquid_interface'


def _candidate_rows(roi,ice_mask=None,dry_color=None):
    h,w=roi.shape[:2]
    blurred=cv2.GaussianBlur(roi.astype(np.float32),(0,0),.7)
    gradient=np.linalg.norm(blurred[2:]-blurred[:-2],axis=2)/2.
    gradient=np.pad(gradient,((1,1),(0,0)))
    blocked=np.zeros((h,w),bool) if ice_mask is None else np.asarray(ice_mask,bool)
    blocked=cv2.dilate(blocked.astype(np.uint8),np.ones((3,3),np.uint8))>0
    tolerance=max(1,round(.012*w))
    available=~blocked
    response=np.where(available,gradient,0.)
    # Curved menisci meet the two walls at slightly different row heights.
    response=cv2.dilate(response,np.ones((2*tolerance+1,1),np.uint8))
    sides=[]
    for columns in _side_columns(w):
        strip=response[:,columns]
        sides.append(np.median(strip,axis=1))
    score=np.minimum(sides[0],sides[1])
    x0=max(3,round(.08*w));x1=min(w-3,round(.92*w))
    visible=available[:,x0:x1]
    # The evidence must span a meaningful fraction of the actual vessel,
    # not merely of the few pixels left after masking ice. Normalizing by
    # unmasked pixels made two narrow glass/ice reflections look like a full
    # liquid interface in transparent real clips.
    coherent=np.sum((response[:,x0:x1]>=4.)&visible,axis=1)/max(x1-x0,1)
    # At least both exposed side strips must support the same actual level.
    valid=(score>=4.)&(coherent>=.35)
    # A thick glass lip can have several horizontal edges below the top wall
    # endpoints; retain the entire upper apparatus band as rim, not water.
    valid[:max(6,round(.08*h)+tolerance+2)]=False
    base_margin=3 if dry_color is not None else max(8,round(.065*h))
    valid[h-base_margin:]=False
    peaks=[]
    for y in np.flatnonzero(valid):
        if peaks and y-peaks[-1][0]<=2*tolerance+2:
            if score[y]>peaks[-1][1]:peaks[-1]=(int(y),float(score[y]))
        else:peaks.append((int(y),float(score[y])))
    return peaks


def observe_surfaces(frames,rois,fps,ice_masks=None):
    """Return local-y observations and their explicit initial/final evidence."""
    output={}
    for side,(x,y,w,h) in rois.items():
        initial=frames[0][y:y+h,x:x+w]
        # Sample outside the two walls in the same vertical range as the
        # exposed interior. These are first-frame appearance observations.
        outside=[];margin=max(4,round(.035*w));span=max(4,round(.05*w))
        for lo,hi in ((x-margin-span,x-margin),(x+w+margin,x+w+margin+span)):
            if lo>=0 and hi<=frames[0].shape[1]:
                outside.append(frames[0][y+round(.25*h):y+h-4,lo:hi].reshape(-1,3))
        external=np.concatenate(outside) if len(outside)==2 else None
        dry_color=_flat_dry_reference(initial,external)
        values=[];valid=[];strengths=[];kinds=[];candidate_log=[];previous=None
        for i,frame in enumerate(frames):
            roi=frame[y:y+h,x:x+w]
            ice=None if not ice_masks or side not in ice_masks else ice_masks[side][i][y:y+h,x:x+w]
            candidates=_candidate_rows(roi,ice,dry_color)
            candidate_log.append([{'y_px':row,'strength':score} for row,score in candidates])
            occluded=False
            if dry_color is None:
                unmasked=_candidate_rows(roi,None,None) if ice is not None and np.any(ice) else candidates
                if unmasked:
                    upper=min(row for row,_ in unmasked)
                    # A liquid/air surface is the upper exposed phase boundary,
                    # not a deeper horizontal texture inside the ice/liquid.
                    # If this vessel-wide boundary is occluded, do not substitute
                    # any lower line, even a stronger/stable base reflection.
                    candidates=[(row,strength) for row,strength in candidates
                                if abs(row-upper)<=max(3.,.012*w)]
                    occluded=not candidates
            if dry_color is not None:
                chosen,kind=_opaque_level(roi,dry_color,ice);strength=0.
            elif candidates:
                if previous is None:
                    ranked=sorted(candidates,key=lambda item:-item[1])
                    unique=len(ranked)==1 or ranked[0][1]>=1.35*ranked[1][1]
                    chosen,strength=ranked[0] if unique else (np.nan,0.)
                else:
                    ranked=sorted(candidates,key=lambda item:abs(item[0]-previous))
                    chosen,strength=ranked[0]
                    # A weak nearby reflection must not keep the tracker stuck
                    # after a much stronger independent interface appears.
                    strongest=max(score for _,score in candidates)
                    if abs(chosen-previous)>max(6.,.025*h) or strength*1.35<strongest:
                        chosen=np.nan;strength=0.
                kind='bilateral_visible_interface' if np.isfinite(chosen) else 'ambiguous_or_discontinuous_interface'
            else:chosen=np.nan;strength=0.;kind='ice_occludes_strongest_interface' if occluded else 'interface_not_observed'
            okay=bool(np.isfinite(chosen))
            values.append(float(chosen));valid.append(okay);strengths.append(float(strength));kinds.append(kind)
            # Missing observations do not fill the curve. After a short loss,
            # independent acquisition can resume but the gap remains explicit.
            if okay:previous=chosen
            elif len(valid)>=max(3,round(.15*fps)) and not any(valid[-max(3,round(.15*fps)):]):previous=None
        output[side]={'surface_y_px':np.asarray(values),'surface_valid':np.asarray(valid),
                      'surface_edge_strength':np.asarray(strengths),'surface_floor_px':np.full(len(frames),h-3.),
                      'independent_surface_observation':True,'vessel_height_px':h,
                      'surface_observation_kind':kinds,'surface_candidates':candidate_log,
                      'initial_level_method':'exposed_flat_opaque_dry_base' if dry_color is not None else 'requires_observed_liquid_interface'}
    return output


def reliable_ice_masks(masks,melting_observations):
    """Exclude masks after a segmentation jump from liquid occlusion evidence."""
    trusted={};diagnostics={}
    for side,values in masks.items():
        observation=melting_observations[side];event=observation.get('melting_evidence') or {}
        first=event.get('start_frame');last=event.get('continuous_run_end_frame')
        use=np.zeros(len(values),bool);ambiguous_bands=[]
        vessel=observation.get('initialization_vessel_roi')
        if first is not None and last is not None:
            for i,row in enumerate(observation.get('observations',[])):
                frame=row.get('frame',i)
                if i<len(use):
                    use[i]=bool(first<=frame<=last and row.get('observed') and row.get('boundary_clear'))
                    box=row.get('box')
                    # SAM can migrate from shrinking ice to a wide thin liquid
                    # meniscus/reflection. Such a mask has ambiguous CURRENT
                    # identity and cannot hard-occlude surface candidates. This
                    # does not declare the ice gone or change the melting gate.
                    if use[i] and vessel and box:
                        width,height=box[2:];vw,vh=vessel[2:]
                        if width>=.70*vw and height<=.12*vh and width/max(height,1)>=5.:
                            use[i]=False;ambiguous_bands.append(int(frame))
        trusted[side]=np.asarray(values,bool)&use[:,None,None]
        diagnostics[side]={'trusted_frame_count':int(use.sum()),'untrusted_frame_count':int((~use).sum()),
                            'trusted_frame_indices':np.flatnonzero(use).tolist(),
                            'ambiguous_interface_band_frames':ambiguous_bands,
                            'ambiguous_band_policy':'Do not hard-occlude with a >=70%-width, <=12%-height, aspect>=5 band without independent current solid identity; do not infer disappearance.',
                           'policy':'Only the continuous boundary-clear ice segment supporting observed melting may occlude a liquid interface; later jumps/losses are excluded.'}
    return trusted,diagnostics


def rise_event(series,fps):
    """Half of an independently anchored full observed rise, never mask onset."""
    values=np.asarray(series['surface_y_px'],float);valid=np.asarray(series['surface_valid'],bool)&np.isfinite(values)
    n=len(values);window=max(3,round(.25*fps));flags=[];ids=np.flatnonzero(valid)
    initial=ids[ids<window];terminal=ids[ids>=n-window]
    initial_ok=len(initial)>=window-1;terminal_ok=len(terminal)>=window-1
    noise=2.
    if initial_ok and np.ptp(values[initial])>noise:initial_ok=False
    if terminal_ok and np.ptp(values[terminal])>noise:terminal_ok=False
    if not initial_ok:flags.append('initial_liquid_level_not_independently_resolved_and_stable')
    if not terminal_ok:flags.append('final_liquid_level_not_independently_resolved_and_stable')
    baseline=float(np.median(values[initial])) if initial_ok else None
    tail=float(np.median(values[terminal])) if terminal_ok else None
    dynamic=baseline-tail if baseline is not None and tail is not None else None
    if dynamic is not None and dynamic<8.:flags.append('resolved_liquid_rise_below_8px')
    maximum_gap=int(np.max(np.diff(ids)-1)) if len(ids)>1 else n
    gap_limit=max(2,round(.10*fps))
    unresolved_gaps=[];subresolution_gaps=[]
    for before,after in zip(ids[:-1],ids[1:]):
        if after-before-1<=gap_limit:continue
        half_level=(baseline+tail)/2 if baseline is not None and tail is not None else None
        crosses_half=half_level is None or min(values[before],values[after])<=half_level<=max(values[before],values[after])
        gap={'before_frame':int(before),'after_frame':int(after),'endpoint_displacement_px':float(abs(values[before]-values[after]))}
        # An unresolved one-pixel liquid film can precede a resolved rise in
        # a drawing. Keep that gap missing, but do not reject a later actually
        # observed half crossing when its bracketing change is below 2-pixel
        # localization noise at each endpoint and far from the half level.
        if gap['endpoint_displacement_px']<=2*noise and not crosses_half:subresolution_gaps.append(gap)
        else:unresolved_gaps.append(gap)
    if unresolved_gaps:flags.append('liquid_interface_internal_gap_too_long')
    if len(ids)<max(8,int(.65*n)):flags.append('liquid_interface_observation_coverage_insufficient')
    normalized=np.full(n,np.nan);rise=np.full(n,np.nan);half=None
    if dynamic is not None and dynamic>=8.:
        rise[valid]=baseline-values[valid];normalized[valid]=rise[valid]/dynamic
        steps=np.diff(values[ids])
        if len(steps) and np.max(np.abs(steps))>max(5.,.15*dynamic):flags.append('liquid_interface_abrupt_step_not_resolved')
        # A real bracket and sustained observations are required. No half-rise
        # can be manufactured inside an interpolated or unobserved interval.
        for i in range(1,n-2):
            if valid[i-1:i+3].all() and normalized[i-1]<.5<=normalized[i] and np.all(normalized[i:i+3]>=.5):
                half=i;break
        if half is None:flags.append('liquid_half_rise_crossing_not_observed')
    usable=not flags and half is not None
    return {'usable':usable,'signal_name':'surface_line','raw_area_px':np.asarray(series.get('area_px',np.zeros(n))),
            'raw_height_px':np.asarray(series.get('height_px',np.zeros(n))),
            'raw_surface_y_px':values,'surface_valid':valid,'surface_rise_px':rise,
            'surface_line_usable':usable,'surface_line_valid_fraction':float(np.mean(valid)) if n else 0.,
            'surface_line_baseline_y_px':baseline,'surface_line_tail_y_px':tail,
            'surface_line_dynamic_range_px':dynamic,'baseline_px':0. if baseline is not None else None,
            'tail_px':dynamic,'dynamic_range_px':dynamic,'minimum_dynamic_range':8.,
            'normalized_surface_rise':normalized,'isotonic_surface_rise':normalized.copy(),
            'half_rise_frame':half if usable else None,'half_rise_time_s':float(half/fps) if usable else None,
            'rise_rate_per_s':float(dynamic/((n-1)/fps)) if usable else None,
            'half_rise_definition':'Half the rise from independently observed stable initial level to independently observed stable final level.',
            'initial_level_resolved':initial_ok,'final_level_resolved':terminal_ok,
            'initial_level_method':series.get('initial_level_method'),
            'maximum_internal_missing_frames':maximum_gap,'maximum_allowed_internal_missing_frames':gap_limit,
            'unresolved_internal_gaps':unresolved_gaps,'unfilled_subresolution_gaps':subresolution_gaps,
            'observed_frames':len(ids),'observed_duration_s':float(len(ids)/fps),
            'first_observed_time_s':float(ids[0]/fps) if len(ids) else None,
            'last_observed_time_s':float(ids[-1]/fps) if len(ids) else None,
            'quality_flags':flags}
