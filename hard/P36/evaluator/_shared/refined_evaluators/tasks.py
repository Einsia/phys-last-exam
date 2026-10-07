"""Measurements use observed masks/tracks. Prompts are metadata only."""
from pathlib import Path
import csv, json, math
import cv2
import numpy as np
from scipy.signal import savgol_filter
from .common import ExtractionError,write_json
from .numeric import saturated_ratio,oscillation_count_score,error_score,soft_error_score,median_time,require_track,height_peak,count_cycles,match_events,linear_fit,fit_discharge_exponent,terminal_windows
from .vision import camera_motion

from .definitions import DEFAULTS


def csv_file(path, columns):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(list(columns));w.writerows(zip(*columns.values()))


def plot(path,t,lines,ylabel,hlines=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for name,y in lines.items():ax.plot(t,y,label=name)
    for y in hlines:ax.axhline(y,color='grey',linestyle=':',linewidth=.7)
    ax.set(xlabel='Source PTS (seconds)',ylabel=ylabel);ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def observed_track(c,j):
    obj=c.a['objects'][j];ids=c.groups[obj['name']]; xy=c.xy[:,ids].copy();valid=c.vis[:,ids].copy()
    h,w=c.frames[0].shape[:2]
    for i,p in enumerate(xy):
        x=np.rint(p[:,0]).astype(int);y=np.rint(p[:,1]).astype(int)
        inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        near=cv2.dilate(c.masks[i,j].astype('uint8'),np.ones((7,7),np.uint8))
        valid[i]&=inside&near[y.clip(0,h-1),x.clip(0,w-1)].astype(bool)
    return xy,valid


def p33(c):
    c.m['principle']="Fix ring identities using the first-frame gap; compare maximum rises relative to initial positions, normalized by fixed initial outer diameters."
    c.m['score_details']={'formula':'clip((1 - h_open/h_closed)/margin,0,1)','range':[0,1],'full_score_max_height_ratio':1-c.cfg['margin']}
    drift,res=camera_motion(c.xy,c.vis,c.groups);write_json(c.out/'camera_motion.json',{'offset_xy':drift,'reference_residual_px':res})
    heights=[]; peaks={};cal={};curves={}; columns={'frame':np.arange(len(c.t)),'time_sec':c.t}
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];xy,valid=observed_track(c,j);d=xy-xy[0];d[~valid]=np.nan
        displacement=np.nanmedian(d,axis=1)-drift
        good=valid.sum(1)>=6;require_track(good,c.t,c.cfg['max_track_gap_sec'],c.cfg['min_valid_track_fraction'])
        yy,xx=np.where(c.masks[0,j]);diam=float(xx.max()-xx.min()+1)
        u=-displacement[:,1]/diam;u[~good]=np.nan;u=np.interp(c.t,c.t[good],u[good])
        sm=median_time(u,c.t,c.cfg['smooth_window_sec']);p=height_peak(sm,c.t,c.cfg['height_noise_diameter'],c.cfg['peak_window_sec'])
        peaks[name]=p;heights.append(p['height']);curves[name]=sm
        columns[name+'_raw_diameter']=u;columns[name+'_smooth_diameter']=sm;columns[name+'_visible_points']=valid.sum(1)
        spread=np.nanmedian(abs(d[:,:,1]-np.nanmedian(d[:,:,1],axis=1)[:,None]),axis=1)
        cal[name]={'initial_outer_diameter_px':diam,'initial_reference':'verified first-frame image and corresponding video frame 0','initial_track_points':xy[0],'height_px':p['height']*diam,'track_dispersion_px_median':float(np.nanmedian(spread)),'valid_frame_fraction':float(good.mean()),'identity':obj['identity_evidence']}
        c.annotations[name]={'trajectory_xy':np.nanmedian(xy,axis=1),'curve':sm,'peak':p}
    csv_file(c.out/'heights.csv',columns);plot(c.out/'heights.png',c.t,curves,'Upward displacement / initial ring diameter')
    write_json(c.out/'peaks.json',peaks);write_json(c.out/'calibration.json',cal)
    hclosed,hopen=heights;ratio,score=saturated_ratio(hopen,hclosed,c.cfg['margin'])
    # Quantify the small-scale uncertainty rather than overinterpreting a near-zero score.
    hc_noise=max(c.cfg['height_noise_diameter'],cal['closed_ring']['track_dispersion_px_median']/cal['closed_ring']['initial_outer_diameter_px'])
    ho_noise=max(c.cfg['height_noise_diameter'],cal['open_ring']['track_dispersion_px_median']/cal['open_ring']['initial_outer_diameter_px'])
    if hclosed>hc_noise:
        ratio_bounds=[max(0.,(hopen-ho_noise)/(hclosed+hc_noise)),(hopen+ho_noise)/(hclosed-hc_noise)]
        c.m['uncertainty']={'height_noise_diameter':[hc_noise,ho_noise],'height_ratio_sensitivity_bounds':ratio_bounds,'metric_sensitivity_bounds':[float(np.clip((1-ratio_bounds[1])/c.cfg['margin'],0,1)),float(np.clip((1-ratio_bounds[0])/c.cfg['margin'],0,1))],'note':'Sensitivity to configured height noise, not a statistical confidence interval'}
    c.m['measurements']={'h_closed_diameter':hclosed,'h_open_diameter':hopen,'h_closed_px':cal['closed_ring']['height_px'],'h_open_px':cal['open_ring']['height_px'],'height_ratio':ratio,'closed_peak_time_sec':peaks['closed_ring']['time_sec'],'open_peak_time_sec':peaks['open_ring']['time_sec']}
    if any(p['right_censored'] for p in peaks.values()):raise ExtractionError('Highest ring position is right-censored: no descent or resolved plateau')
    if 0<hclosed<c.cfg['height_noise_diameter']*c.cfg['min_height_snr']:
        if hclosed<=c.cfg['height_noise_diameter']:
            c.m['measurements']['height_ratio']=None;score=0.;c.m['reason']='zero_closed_height; both continuous visibility and noise bound checked'
        else:raise ExtractionError('Closed height is comparable to measurement noise')
    else:c.m['reason']="Ring identities are fixed by the visible gap; both peaks have plateau or descent evidence. Score using the measured height ratio."
    c.calculation=[f'Closed ring h={hclosed:.8f} outer diameters ({cal["closed_ring"]["height_px"]:.4f} px), open ring h={hopen:.8f} outer diameters ({cal["open_ring"]["height_px"]:.4f} px).',f'Peaks: {peaks}',f'height_ratio={ratio}; margin={c.cfg["margin"]}; full-score boundary={1-c.cfg["margin"]}; score={score:.8f}.']
    return score


def p34(c):
    c.m['principle']="Compare valid complete-cycle counts for solid and slotted plates using the same observation window and angular-amplitude threshold."
    c.m['score_details']={'version':'p34_directional_v1','formula':'1.0 if N_solid < N_slotted else 0.0; zero reference remains 0','range':[0,1],'equal_positive_counts_score':0.,'zero_reference_score':0.,'interpretation':"Fewer complete cycles for the solid plate within the shared window support the target relationship; equal or greater counts do not. The score is not a probability that the physical relationship holds."}
    drift,res=camera_motion(c.xy,c.vis,c.groups);series=[];cal={};columns={'frame':np.arange(len(c.t)),'time_sec':c.t}
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];xy,valid=observed_track(c,j);source=xy[0];centroid=np.mean(source,axis=0);reference=[];fits=[]
        for i,p in enumerate(xy):
            good=valid[i];M=inlier=None
            if good.sum()>=4:M,inlier=cv2.estimateAffinePartial2D(source[good].astype('float32'),p[good].astype('float32'),method=cv2.RANSAC,ransacReprojThreshold=3.)
            if M is None:reference.append([np.nan,np.nan]);fits.append({'valid':False,'reason':'insufficient_consistent_material_points'});continue
            pred=source[good]@M[:,:2].T+M[:,2];err=np.linalg.norm(pred-p[good],axis=1);residual=float(np.median(err));scale=float(np.linalg.norm(M[:,0]));ok=bool(residual<5 and .85<scale<1.15 and inlier.sum()>=4)
            reference.append(M[:,:2]@centroid+M[:,2]-drift[i] if ok else [np.nan,np.nan]);fits.append({'valid':ok,'scale':scale,'residual_px':residual,'inliers':int(inlier.sum())})
        reference=np.array(reference);good=np.isfinite(reference[:,0]);require_track(good,c.t,c.cfg['max_track_gap_sec'],.90)
        for k in range(2):reference[:,k]=np.interp(c.t,c.t[good],reference[good,k])
        pivot=np.array(c.a['geometry']['pivot_points'][j]);v=reference-pivot;theta=np.degrees(np.arctan2(v[:,0],v[:,1]));sm=median_time(theta,c.t,c.cfg['smooth_window_sec']);series.append(sm)
        columns[name+'_angle_deg']=theta;columns[name+'_smoothed_deg']=sm
        cal[name]={'pivot_xy':pivot,'initial_material_reference_xy':centroid,'fits':fits,'method':'CoTracker visible plate material points; independent similarity fit; annotated physical suspension point; gravity vertical'}
        c.annotations[name]={'trajectory_xy':reference+drift,'pivot':pivot,'angles':sm}
    cutoff=max(c.cfg['amplitude_floor_deg'],c.cfg['amplitude_fraction']*float(np.mean(np.abs(np.array(series)[:,0]))))
    peaks={};cycles={};tails={};counts=[]
    for j,obj in enumerate(c.a['objects']):
        sm=series[j];initial=(c.t<=c.t[0]+.20)
        release=bool(np.ptp(sm[initial])<.5 and abs(sm[0])>=cutoff)
        p,cy,tail=count_cycles(sm,c.t,cutoff,c.cfg['peak_prominence_deg'],c.cfg['min_peak_separation_sec'],release)
        name=obj['name'];peaks[name]=p;cycles[name]=cy;tails[name]=tail;counts.append(len(cy));cal[name]['release_zero_speed_verified']=release
    csv_file(c.out/'angles.csv',columns);plot(c.out/'angles_peaks.png',c.t,{o['name']:series[j] for j,o in enumerate(c.a['objects'])},'Signed pendulum angle (degrees)',[-cutoff,0,cutoff])
    write_json(c.out/'calibration.json',cal);write_json(c.out/'peaks.json',peaks);write_json(c.out/'cycles.json',{'cycles':cycles,'partial_cycles':tails})
    ratio,score=oscillation_count_score(counts[0],counts[1])
    c.m['measurements']={'N_solid':counts[0],'N_slotted':counts[1],'oscillation_count_ratio':ratio,'amplitude_cutoff_deg':cutoff,'observation_window_sec':[float(c.t[0]),float(c.t[-1])],'right_censored':{name:x['right_censored'] for name,x in tails.items()}}
    if c.t[-1]-c.t[0]<c.cfg['min_observation_sec']:raise ExtractionError('Observation shorter than configured minimum')
    if counts[1]==0:
        # Need an observed scale for a full cycle, or continuous stationary reference.
        if np.ptp(series[1])>cutoff and len([p for p in peaks['slotted_plate'] if p['accepted']])<3:raise ExtractionError('Slotted trajectory is truncated before an observable full cycle')
        c.m['reason']='zero_reference_count; observed continuous reference trajectory'
    else:
        relation="The solid plate has fewer cycles, supporting the target relationship." if counts[0]<counts[1] else "Cycle counts are equal or the solid plate has more cycles, so the target relationship is not supported."
        c.m['reason']="Complete cycles pair boundaries of a fixed polarity; "+relation+"Count only within the shared window; the M2 decay rate is not computed."
    c.calculation=[f'W={c.m["measurements"]["observation_window_sec"]}; A_cut={cutoff:.6f} deg.',f'Individual valid-cycle timestamps: {cycles}',f'N_solid={counts[0]}, N_slotted={counts[1]}, ratio={ratio}; scoring_version=p34_directional_v1.',f'score = 1.0 if {counts[0]} < {counts[1]} else 0.0 = {score:.8f}.' if counts[1]>0 else "N_slotted=0: zero_reference_count, score=0; 0/0 is not assigned a neutral score.","Assess only the direction of complete-cycle counts: 1 when the solid plate has fewer, otherwise 0."]
    return score


def circle_observation(mask):
    from scipy.optimize import least_squares
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    points=max(contours,key=cv2.contourArea)[:,0,:].astype(float)
    if len(points)<12:return None
    center=points.mean(0);radius=np.median(np.linalg.norm(points-center,axis=1))
    fit=least_squares(lambda p:np.linalg.norm(points-p[:2],axis=1)-p[2],np.r_[center,radius],loss='soft_l1',f_scale=1.)
    x,y,r=fit.x;res=np.linalg.norm(points-fit.x[:2],axis=1)-r
    return {'center':[float(x),float(y)],'radius':float(r),'relative_residual':float(np.sqrt(np.mean(res**2))/r),'contour':points,'valid':bool(r>3 and np.sqrt(np.mean(res**2))/r<.10)}


def p38(c):
    c.m['principle']="Fix ball 1 as the large ball and ball 2 as the small ball; independently fit terminal windows and observed outer radii, then compute relative error against the Stokes square law."
    c.m['score_details']={'formula':'1 / (1 + abs((v1/v2)/(r1/r2)^2-1)/error_at_zero)','range':[0,1], 'half_score_error':c.cfg['error_at_zero'], 'finite_error_cutoff':False, 'parameter_note':'error_at_zero is retained as a legacy CLI name; its unchanged value is now the half-score scale a'}
    drift,res=camera_motion(c.xy,c.vis,c.groups);radii={};windows={};chosen=[];columns={'frame':np.arange(len(c.t)),'time_sec':c.t};curves={};obsall={}
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];obs=[circle_observation(m) for m in c.masks[:,j]];obsall[name]=obs
        centers=np.array([x['center'] if x else [np.nan,np.nan] for x in obs])-drift;r=np.array([x['radius'] if x else np.nan for x in obs]);valid=np.array([x is not None and x['valid'] for x in obs])
        y=centers[:,1];cols=c.a['geometry'];clearance=cols['bottom_y']-(y+r);boundary=clearance>=c.cfg['bottom_clearance_diameter']*2*r
        left,right=cols['tank_walls_x'];side_clearance=np.minimum(centers[:,0]-left-r,right-centers[:,0]-r)
        candidates=terminal_windows(y,r,c.t,valid&boundary,c.cfg);windows[name]={'selection_rule':'longest eligible contiguous window, earlier wins equal duration','candidates':candidates,'bottom_clearance_px':clearance,'side_clearance_px':side_clearance,'side_clearance_diameter':side_clearance/(2*r),'wall_proximity_warning':bool(np.any(side_clearance<2*r)),'fit_eligible':(valid&boundary),'chosen':candidates[0] if candidates else None}
        radii[name]={'per_frame_px':r,'relative_residual':[x['relative_residual'] if x else None for x in obs],'initial_radius_px':r[0]}
        columns[name+'_y_px']=y;columns[name+'_radius_px']=r;columns[name+'_speed_px_s']=np.gradient(y,c.t);curves[name]=y
        c.annotations[name]={'trajectory_xy':centers+drift,'radii':r};chosen.append(candidates[0] if candidates else None)
    csv_file(c.out/'position_velocity.csv',columns);plot(c.out/'position_velocity.png',c.t,curves,'Camera-corrected downward centre position (px)')
    write_json(c.out/'radii.json',radii);write_json(c.out/'circle_fits.json',obsall);write_json(c.out/'terminal_windows.json',windows)
    if any(w is None for w in chosen):raise ExtractionError('No sufficiently long nonzero terminal-speed window before bottom clearance cutoff; see terminal_windows.json')
    r1,r2=[w['radius_px'] for w in chosen];v1,v2=[w['velocity'] for w in chosen]
    if r1/r2<c.cfg['min_radius_ratio']:raise ExtractionError('Large/small radius identity not sufficiently separated')
    rr=r1/r2;vr=v1/v2;err=abs(vr/rr**2-1);score=soft_error_score(err,c.cfg['error_at_zero'])
    c.m['measurements']={'r_1_large_px':r1,'r_2_small_px':r2,'v_1_large_px_per_sec':v1,'v_2_small_px_per_sec':v2,'radius_ratio':rr,'velocity_ratio':vr,'normalized_velocity_ratio':vr/rr**2,'raw_m1_error':err,'large_terminal_window_sec':chosen[0]['window_sec'],'small_terminal_window_sec':chosen[1]['window_sec']}
    c.m['applicability']={'same_material_and_fluid':'task assumption, not image measurement','low_Reynolds_number':'not measured without physical length and fluid calibration','projection':'near frontal shared tank scale','wall_effects':'visible side clearance recorded; no fabricated 3D wall correction'}
    c.m['reason']="Outer ball outlines and nonzero constant-speed windows before bottom contact were independently fitted; deviations from theory receive lower scores as usual."
    c.calculation=[f'Ball 1=large ball, ball 2=small ball. r1={r1:.6f} px, r2={r2:.6f} px.',f'v1={v1:.6f} px/s @ {chosen[0]["window_sec"]}; v2={v2:.6f} px/s @ {chosen[1]["window_sec"]}.',f'r1/r2={rr:.8f}; v1/v2={vr:.8f}; E=abs({vr:.8f}/{rr:.8f}^2-1)={err:.8f}; score={score:.8f}.']
    return score


def refine_solid_region(frame, mask):
    """Refine a coarse proposal from observed interior appearance, without shape priors."""
    binary=mask.astype('uint8');kernel=np.ones((7,7),np.uint8)
    interior=cv2.erode(binary,kernel)>0
    if interior.sum()<50:return mask
    pixels=frame.astype(float);color=np.median(pixels[interior],axis=0)
    spread=float(np.median(np.linalg.norm(pixels[interior]-color,axis=1)))
    # Only locally uniform objects support this refinement. Textured proposals
    # retain their original segmentation instead of forcing a rectangle.
    if spread>25:return mask
    near=cv2.dilate(binary,np.ones((41,41),np.uint8))>0
    candidate=(near & (np.linalg.norm(pixels-color,axis=2)<max(20.,4*spread))).astype('uint8')
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate,8)
    if n<2:return mask
    k=max(range(1,n),key=lambda j:int(np.sum(interior & (labels==j))))
    result=labels==k;ratio=float(result.sum()/max(1,mask.sum()))
    if not .5<ratio<1.8 or np.sum(result & interior)<.65*interior.sum():return mask
    return result


def quad_observation(mask,previous=None):
    from itertools import permutations
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:return None
    contour=max(contours,key=cv2.contourArea);hull=cv2.convexHull(contour);p=None
    for frac in [.01,.015,.02,.025,.03,.04,.05]:
        q=cv2.approxPolyDP(hull,frac*cv2.arcLength(hull,True),True)
        if len(q)==4:p=q[:,0,:].astype(float);break
    if p is None:return None
    if previous is None:
        order=np.argsort(p[:,1]);top=p[order[:2]];bottom=p[order[2:]];top=top[np.argsort(top[:,0])];bottom=bottom[np.argsort(bottom[:,0])]
        p=np.array([top[0],top[1],bottom[1],bottom[0]])
    else:p=min((p[list(o)] for o in permutations(range(4))),key=lambda x:np.sum((x-previous)**2))
    # Colored material markers can occlude the actual corners. Fit each
    # independently observed straight side away from its endpoints, then
    # intersect adjacent lines. No right-angle, rigid-length or pivot prior.
    points=contour[:,0,:].astype(float);lines=[]
    for j in range(4):
        a,b=p[j],p[(j+1)%4];edge=b-a;length=np.linalg.norm(edge);u=edge/length;n=np.array([-u[1],u[0]])
        along=(points-a)@u;dist=np.abs((points-a)@n)
        selected=points[(along>.15*length)&(along<.85*length)&(dist<max(8.,.07*length))]
        if len(selected)<8:return None
        fitted=cv2.fitLine(selected.astype('float32'),cv2.DIST_HUBER,0,.01,.01).ravel()
        lines.append((fitted[2:].astype(float),fitted[:2].astype(float)))
    refined=[]
    for j in range(4):
        a,u=lines[(j-1)%4];b,v=lines[j];A=np.c_[u,-v]
        if abs(np.linalg.det(A))<.15:return None
        refined.append(a+np.linalg.solve(A,b-a)[0]*u)
    p=np.asarray(refined)
    rect_area=abs(cv2.contourArea(p.astype('float32')));filled=cv2.contourArea(contour)
    return p,float(abs(filled-rect_area)/max(rect_area,1))


def p10(c):
    """A continuously pushed block is checked for rigid rotation about its support."""
    c.m['principle']="Under continuous external pushing, measure support-contact drift, lift-off distance, and rigid-body shape changes; do not equate angular-motion onset with free-instability onset."
    c.m['score_details']={'version':'p10_forced_support_v4','formula':'1/(1+max(resolved_pivot_drift, resolved_contact_error, rigid_shape_error)/geometry_error_half_score)',
                          'range':[0,1],'half_score_relative_error':c.cfg['geometry_error_half_score']}
    drift,res=camera_motion(c.xy,c.vis,c.groups);quad=[];fiterrors=[];previous=None
    for frame, mask in zip(c.frames,c.masks[:,0]):
        observed=refine_solid_region(frame,mask)
        obs=quad_observation(observed,previous)
        if obs is None:quad.append(np.full((4,2),np.nan));fiterrors.append(None)
        else:previous=obs[0];quad.append(previous);fiterrors.append(obs[1])
    quad=np.array(quad);valid=np.all(np.isfinite(quad),axis=(1,2));require_track(valid,c.t,.1,.95)
    for j in range(4):
        for axis in range(2):quad[:,j,axis]=np.interp(c.t,c.t[valid],quad[valid,j,axis])
    q=quad-drift[:,None,:];support=q[:,2];diag=float(np.linalg.norm(q[0,0]-q[0,2]))
    if diag<10:raise ExtractionError('Block geometry too small for support measurement')
    ground=np.asarray(c.a['geometry']['ground_points'],float);axis=ground[-1]-ground[0]
    axis/=np.linalg.norm(axis);normal=np.array([-axis[1],axis[0]])
    displacement=np.linalg.norm(support-support[0],axis=1)
    contact=np.abs((support-ground[0])@normal)
    edges=np.linalg.norm(np.roll(q,-1,axis=1)-q,axis=2)
    rigid=np.max(np.abs(edges/edges[0]-1),axis=1)
    noise=max(float(c.cfg['geometry_noise_px']),float(np.nanpercentile(res,95)))
    drift_px=float(np.percentile(displacement,95));contact_px=float(np.percentile(contact,95))
    pivot_error=max(0.,drift_px-2*noise)/diag;contact_error=max(0.,contact_px-2*noise)/diag
    shape_error=max(0.,float(np.percentile(rigid,95))-2*noise/max(float(np.min(edges[0])),1.))
    error=max(pivot_error,contact_error,shape_error)
    top=(q[:,0]+q[:,1])/2;bottom=(q[:,2]+q[:,3])/2;v=top-bottom
    theta=np.degrees(np.unwrap(np.arctan2(v[:,0],-v[:,1])));rotation=float(np.ptp(theta))
    # Geometry can only support a pivot claim once actual rotation is observed.
    if rotation<5:raise ExtractionError('Insufficient observed rotation to identify the support pivot')
    score=soft_error_score(error,c.cfg['geometry_error_half_score'])
    c.m['measurements']={'pivot_drift_p95_px':drift_px,'contact_gap_p95_px':contact_px,
       'initial_block_diagonal_px':diag,'geometry_noise_px':noise,'pivot_drift_relative_resolved':pivot_error,
       'contact_gap_relative_resolved':contact_error,'rigid_shape_error':shape_error,'raw_m1_error':error,
       'observed_rotation_deg':rotation,'support_drift_max_px':float(displacement.max())}
    c.m['uncertainty']={'pixel_noise_floor':noise,'resolution_allowance_px':2*noise,'kind':'measurement resolution allowance, not fitted to score labels'}
    c.m['applicability']={'driving':'continuous external actuator as specified by the task',
       'free_instability_onset_required':False,'ground_geometry':'observed static first-frame support line',
       'limitations':'checks visible rigid support kinematics; actuator forces and friction are not measured'}
    c.annotations['block']={'quads':quad,'trajectory_xy':quad.mean(1),'angles':theta}
    csv_file(c.out/'pose_events.csv',{'frame':np.arange(len(c.t)),'time_sec':c.t,'theta_deg':theta,
       'pivot_drift_px':displacement,'contact_gap_px':contact,'rigid_shape_relative_change':rigid})
    plot(c.out/'pose_events.png',c.t,{'pivot drift / diagonal':displacement/diag,'contact gap / diagonal':contact/diag,'shape change':rigid},'Relative observed geometry error')
    write_json(c.out/'corners.json',{'corners_TL_TR_BR_BL':quad,'fit_area_relative_errors':fiterrors})
    write_json(c.out/'events.json',{'driving':'forced rotation','support_position_px':support,'rotation_deg':theta})
    c.m['reason']="Check fixed-support-edge and rigid-body constraints for the continuously pushed task; omit inapplicable center-of-mass crossing and free-toppling synchronization requirements."
    c.calculation=[f'Pivot drift={drift_px:.5f}px; contact gap={contact_px:.5f}px; block diagonal={diag:.5f}px.',
       f'Resolved geometry error={error:.8f}; half-score scale={c.cfg["geometry_error_half_score"]}; score={score:.8f}.']
    return score


def p30(c):
    """Measure two observed events: entry and light response."""
    import importlib.util
    from .resources import task_resource
    path = task_resource('P30') / 'evaluator' / 'entry_light.py'
    spec = importlib.util.spec_from_file_location('_packaged_p30_entry_light', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate(c)


def p36(c):
    c.m['principle']="Recover axisymmetric volume from the visible interior and free surface, independently fit Q proportional to h^beta, and compare abs(beta_water-.5)+abs(beta_sand)."
    c.m['score_details']={'version':'p36_soft_exponent_error_v3','formula':'1 / (1 + (abs(beta_water - 0.5) + abs(beta_sand)) / error_half_score)','range':[0,1],'half_score_error':c.cfg['error_half_score'],'finite_error_cutoff':False,'fitting_policy':'observed positive-flow data; no relative-height window or minimum height ratio','interpretation':"Apply a smooth penalty for exponent error; finite errors are not truncated to zero. Scores are not probabilities of physical correctness; fitting uncertainty is reported separately."}
    drift,res=camera_motion(c.xy,c.vis,c.groups);geometry={};surfaces={};fit_details={};columns={'frame':np.arange(len(c.t)),'time_sec':c.t};curves={};errors=[];betas=[];streams=[]
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];apex=np.array(c.a['geometry']['outlet_points'][j],float)
        # The annotated taper point is the cone-to-stem junction; the physical
        # open outlet is the stream-box top. Both are kept separately in geometry.
        outlet_y=float(c.a['geometry']['stream_boxes'][j][1]);yy,xx=np.where(c.masks[0,j]);initial_surface=float(np.percentile(yy,1));height_cone=apex[1]-initial_surface
        wall=[]
        for y in range(int(initial_surface+.2*height_cone),int(apex[1]-.15*height_cone)):
            x=np.flatnonzero(c.masks[0,j,y])
            if len(x)>5:wall.append([float(apex[1]-y),float((x[-1]-x[0])/2)])
        wall=np.array(wall);slope,intercept=np.polyfit(wall[:,0],wall[:,1],1);radius0=float(np.percentile(abs(xx-apex[0]),99));valid_geometry=bool(slope>0 and np.sqrt(np.mean((wall[:,1]-(slope*wall[:,0]+intercept))**2))<3)
        h=[];vol=[];rows=[]
        for i,mask in enumerate(c.masks[:,j]):
            axis_x=apex[0]+drift[i,0];apex_y=apex[1]+drift[i,1];points=[]
            for x in range(int(axis_x-.8*radius0),int(axis_x+.8*radius0)):
                y=np.flatnonzero(mask[:,x])
                if len(y)>5:points.append([float(x-axis_x),float(y[0]-drift[i,1])])
            points=np.array(points)
            if len(points)<12:h.append(np.nan);vol.append(np.nan);rows.append({'valid':False});continue
            # Symmetric surface fit can represent a centre crater; never silently
            # replace a measured sand crater by a horizontal water surface.
            x=points[:,0];z=apex[1]-points[:,1]
            if name=='water':coef=np.array([float(np.median(z)),0.])
            else:
                X=np.c_[np.ones(len(x)),x*x];coef=np.linalg.lstsq(X,z,rcond=None)[0]
                for _ in range(3):
                    residual=z-X@coef;ok=abs(residual)<max(2.,3*1.4826*np.median(abs(residual)))
                    if ok.sum()>=8:coef=np.linalg.lstsq(X[ok],z[ok],rcond=None)[0]
            height=outlet_y-apex[1]+coef[0]
            r=np.linspace(0,radius0*1.15,300);bottom=np.maximum((r-intercept)/slope,0);surface=coef[0]+coef[1]*r*r
            # Extrapolate only the gently varying symmetric observed surface to
            # its independently fitted wall intersection; integral is 3D volume.
            volume=float(np.trapezoid(2*np.pi*r*np.maximum(surface-bottom,0),r))
            h.append(height);vol.append(volume);rows.append({'valid':True,'surface_points_x_relative_y':points,'surface_z_coefficients':[float(x) for x in coef],'axis_surface_y':float(apex[1]-coef[0]),'surface_fit_rms_px':float(np.sqrt(np.mean((z-(coef[0]+coef[1]*x*x))**2)))})
        h=np.array(h);vol=np.array(vol);good=np.isfinite(h)&np.isfinite(vol)
        # A discharge exponent is identifiable over a measured interval; a
        # late empty/occluded funnel does not invalidate earlier observations.
        # Never extrapolate unobserved endpoints or bridge long missing gaps.
        ids=np.flatnonzero(good)
        if len(ids)<12:raise ExtractionError('Too few observed fluid surfaces')
        first,last=int(ids[0]),int(ids[-1])
        require_track(good[first:last+1],c.t[first:last+1],.1,.9)
        inner=np.arange(first,last+1)
        h[inner]=np.interp(c.t[inner],c.t[good],h[good]);vol[inner]=np.interp(c.t[inner],c.t[good],vol[good])
        # Actual independent regression windows; no framewise pixel-area derivative.
        q=[];qnoise=[]
        for t in c.t:
            ix=np.abs(c.t-t)<=c.cfg['derivative_window_sec']/2+1e-9
            if ix.sum()<3 or not np.isfinite(vol[ix]).all():q.append(np.nan);qnoise.append(np.nan);continue
            fit=linear_fit(c.t[ix],vol[ix]);q.append(-fit['velocity']);qnoise.append(fit['slope_se'])
        q=np.array(q);qnoise=np.array(qnoise);frac=h/h[0]
        details=fit_discharge_exponent(h,q,qnoise,c.t,c.cfg)
        height_ratio=float(np.nanmax(h)/np.nanmin(h));problem=details['failures']
        if not valid_geometry:problem.append('inner-wall geometry residual too large')
        if problem:details['fit']=None;betas.append(None);errors.extend(name+': '+p for p in problem)
        else:betas.append(details['fit']['velocity'])
        geometry[name]={'axis_x':apex[0],'cone_apex_y':apex[1],'open_outlet_y':outlet_y,'wall_R_of_z_slope':float(slope),'wall_R_of_z_intercept':float(intercept),'observed_wall_points_z_R':wall,'axisymmetric_assumption':True,'volume_formula':'2*pi*integral r*max(z_surface(r)-z_bottom(r),0) dr','valid':valid_geometry}
        surfaces[name]=rows;fit_details[name]={'height_px_initial_final':[float(h[first]),float(h[last])],'observed_surface_fraction':float(good.mean()),'observed_frame_span':[first,last],'unobserved_tail_extrapolated':False,'height_ratio':height_ratio,'height_fraction_range':[float(np.nanmin(frac)),float(np.nanmax(frac))],**details}
        columns[name+'_height_px']=h;columns[name+'_volume_px3']=vol;columns[name+'_flow_px3_s']=q;columns[name+'_flow_se']=qnoise;curves[name+'_h/h0']=frac
        c.annotations[name]={'surface_y':np.array([r.get('axis_surface_y',np.nan) for r in rows])+drift[:,1],'axis_x':apex[0],'curves':{'h':h,'v':vol,'q':q}}
        # Independently preserve actual outlet color evidence on every frame.
        sm=np.zeros(c.masks[:,j].shape,dtype=bool);x1,y1,x2,y2=c.a['geometry']['stream_boxes'][j]
        for i,f in enumerate(c.frames):
            hsv=cv2.cvtColor(f[y1:y2,x1:x2],cv2.COLOR_BGR2HSV)
            color=(hsv[:,:,0]>80)&(hsv[:,:,0]<120)&(hsv[:,:,1]>50) if name=='water' else (hsv[:,:,0]>8)&(hsv[:,:,0]<40)&(hsv[:,:,1]>40)
            sm[i,y1:y2,x1:x2]=color
        streams.append(sm)
    c.extra_masks={'water_stream':streams[0],'sand_stream':streams[1]}
    csv_file(c.out/'height_volume_flow.csv',columns);plot(c.out/'height_volume_flow.png',c.t,curves,'Relative remaining head above outlet')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['water','sand']:
        h=columns[name+'_height_px'];q=columns[name+'_flow_px3_s'];v=(q>0)&(h>0);ax.scatter(np.log(h[v]),np.log(q[v]),s=8,alpha=.2,label=name+' positive flow')
        ids=fit_details[name]['independent_fit_frames'];fit=fit_details[name]['fit']
        if ids:ax.scatter(np.log(h[ids]),np.log(q[ids]),s=18,label=name+' selected')
        if fit:
            x=np.linspace(np.log(h[ids]).min(),np.log(h[ids]).max(),100);ax.plot(x,fit['intercept_at_mean_time']+fit['velocity']*(x-fit['log_height_mean']),label=f'{name} beta={fit["velocity"]:.3f}')
    ax.set(xlabel='log(h)',ylabel='log(Q)',title='Observed-window fits; no relative-height gate');ax.legend();fig.tight_layout();fig.savefig(c.out/'log_q_log_h.png',dpi=150);plt.close(fig)
    write_json(c.out/'geometry.json',geometry);write_json(c.out/'surfaces.json',surfaces);write_json(c.out/'fit_details.json',fit_details)
    c.m['measurements']={'beta_water':betas[0],'beta_sand':betas[1],'raw_m1_error':None,'volume_reconstruction_method':'axisymmetric inner-wall and independently fitted surface radial integration','water_height_ratio':fit_details['water']['height_ratio'],'sand_height_ratio':fit_details['sand']['height_ratio']}
    c.m['uncertainty']={'note':'OLS 95% intervals conditional on selected observations and reconstructed geometry; wide intervals are reported, not rejected','warnings':{name:fit_details[name]['warnings'] for name in ['water','sand']}}
    for name in ['water','sand']:
        fit=fit_details[name]['fit'];c.m['measurements']['beta_'+name+'_ci']=fit['slope_ci95'] if fit else None;c.m['measurements'][name+'_fit_window_sec']=fit_details[name]['fit_window_sec']
    c.calculation=[f'Observed height ranges and fitting eligibility: {fit_details}',f'Volume-recovery geometry: axisymmetric surface/wall integration; cone apex and open outlet treated separately.',"Neither -dh/dt nor changes in two-dimensional mask area were treated as flow rate."]
    if errors:raise ExtractionError('; '.join(errors))
    err=abs(betas[0]-.5)+abs(betas[1]);score=soft_error_score(err,c.cfg['error_half_score']);c.m['measurements']['raw_m1_error']=err;c.m['reason']="Fit the observed discharge without a fixed height threshold; penalize exponent error smoothly rather than forcing zero beyond a cutoff, and report fitting uncertainty separately.";c.calculation.append(f'p36_soft_exponent_error_v3: beta_water={betas[0]}, beta_sand={betas[1]}; E={err}; half-score error={c.cfg["error_half_score"]}; score=1/(1+{err}/{c.cfg["error_half_score"]})={score}')
    return score

TASKS={'P33':p33,'P34':p34,'P30':p30,'P36':p36,'P10':p10,'P38':p38}
