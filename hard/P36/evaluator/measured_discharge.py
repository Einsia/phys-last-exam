"""P36: visible wall measurements independent of eroded material masks."""
from discharge_integral import fit_integrated_discharge
import cv2
import numpy as np
from refined_evaluators.tasks import csv_file,plot
from refined_evaluators.vision import camera_motion
from refined_evaluators.numeric import require_track,linear_fit,fit_discharge_exponent,soft_error_score
from refined_evaluators.common import ExtractionError,write_json

def visible_inner_edges(frame,y,left,right):
    gray=cv2.cvtColor(frame[max(0,y-1):y+2],cv2.COLOR_BGR2GRAY).astype(float).mean(axis=0)
    derivative=np.diff(gray);edges=[]
    for proposed,sign in [(left,1.),(right,-1.)]:
        lo=max(1,int(proposed)-7);hi=min(len(derivative)-2,int(proposed)+7)
        if hi<=lo:return float(left)-.5,float(right)+.5,False
        values=sign*derivative;peak=lo+int(np.argmax(values[lo:hi+1]))
        if values[peak]<8:return float(left)-.5,float(right)+.5,False
        center=float(peak)+.5;den=values[peak-1]-2*values[peak]+values[peak+1]
        if den<-1e-8:center+=float(np.clip(.5*(values[peak-1]-values[peak+1])/den,-.5,.5))
        edges.append(center)
    if edges[1]<=edges[0]:return float(left)-.5,float(right)+.5,False
    return edges[0],edges[1],True

def wall_geometry_quality(wall, height_cone, fallback_radius=None):
    """Keep observed wall quality separate from the known, annotated cone model.

    A thick glass rim can move an unconstrained line's intercept several pixels.
    For the known cone, the reviewed apex anchors R(0)=0. Its constant radius
    scale cancels from the discharge exponent; the apex and free surface do not.
    Without an independently supplied cone scale, retain the observed-fit check.
    """
    wall = np.asarray(wall, float).reshape(-1, 2)
    wall = wall[np.isfinite(wall).all(axis=1)]
    scale = float(height_cone)
    if not np.isfinite(scale) or scale <= 0:
        raise ExtractionError('Unresolved physical vessel height')
    slope = intercept = rms = relative = None
    if len(wall) >= 3 and np.ptp(wall[:, 0]) > 0:
        slope, intercept = map(float, np.polyfit(wall[:, 0], wall[:, 1], 1))
        rms = float(np.sqrt(np.mean((wall[:, 1] - slope*wall[:, 0] - intercept)**2)))
        relative = rms / scale
    observed_valid = bool(slope is not None and slope > 0 and relative < .02)
    known_cone = bool(fallback_radius is not None and np.isfinite(fallback_radius) and fallback_radius > 1.)
    quality = dict(residual_rms_px=rms, observed_cone_height_px=scale,
        relative_residual=relative, relative_residual_limit=.02,
        observed_slope=slope, observed_intercept=intercept,
        observed_fit_valid=observed_valid, valid=observed_valid or known_cone)
    if known_cone:
        ratios = wall[:, 1]/np.maximum(wall[:, 0], 1e-9)
        ratios = ratios[(wall[:, 0] > .1*scale) & (wall[:, 1] > 0)]
        cone_slope = float(np.median(ratios)) if len(ratios) >= 3 else float(fallback_radius)/scale
        quality.update(geometry_mode='fixed_cone_annotated_apex',
            fixed_intercept=0., fallback_reason=None if observed_valid else 'noisy_or_missing_wall_edges',
            policy='Task cone assumption plus reviewed frame-zero apex; noisy wall fit retained as a warning, never used to move the apex.')
        return cone_slope, 0., quality
    quality.update(geometry_mode='observed_cone_fit', fallback_reason=None,
                   policy='No fixed cone supplied; observed geometry must pass the residual check.')
    return slope if slope is not None else 0., intercept if intercept is not None else 0., quality


def visible_surface_y(frame, x, proposal, radius=7):
    """Refine a mask proposal on an independently observed local RGB step."""
    x = int(x)
    lo = max(0, int(round(proposal))-radius)
    hi = min(frame.shape[0], int(round(proposal))+radius+1)
    left, right = max(0, x-1), min(frame.shape[1], x+2)
    if hi-lo < 7 or right <= left:
        return float(proposal), False
    profile = np.median(frame[lo:hi, left:right].astype(float), axis=1)
    strength = np.linalg.norm(np.diff(profile, axis=0), axis=1)
    peak = int(np.argmax(strength))
    if strength[peak] < 8 or peak == 0 or peak == len(strength)-1:
        return float(proposal), False
    a, b = max(0, peak-1), min(len(strength), peak+2)
    weights = np.maximum(0., strength[a:b] - float(np.median(strength)))
    if weights.sum() <= 0:
        return float(proposal), False
    return float(lo + np.sum((np.arange(a, b)+.5)*weights)/weights.sum()), True


def cone_and_outlet(geometry, index):
    """Resolve independent reviewed geometric origins, never a stream crop."""
    apex = np.asarray(geometry.get('cone_apex_points', geometry['outlet_points'])[index], float)
    outlets = geometry.get('open_outlet_points', geometry['outlet_points'])
    outlet = np.asarray(outlets[index], float)
    if apex.shape != (2,) or outlet.shape != (2,) or not np.isfinite([apex, outlet]).all():
        raise ExtractionError('Invalid reviewed cone apex or physical outlet')
    reference = 'explicit_open_outlet' if 'open_outlet_points' in geometry else 'annotated_cone_outlet_assumption'
    return apex, float(outlet[1]), reference


def p36(c):
    c.m['principle']="Recover axisymmetric volume from the visible interior and free surface, independently fit Q proportional to h^beta, and compare abs(beta_water-.5)+abs(beta_sand)."
    c.m['score_details']={'version':'p36_fixed_cone_integrated_exponent_v5','formula':'1 / (1 + (abs(beta_water - 0.5) + abs(beta_sand)) / error_half_score)','range':[0,1],'half_score_error':c.cfg['error_half_score'],'finite_error_cutoff':False,'fitting_policy':'Integrated observed volume and head; beta free; no noisy derivative selection or target-exponent prior','interpretation':"Apply a smooth penalty for exponent error; finite errors are not truncated to zero. Scores are not probabilities of physical correctness; fitting uncertainty is reported separately."}
    drift,res=camera_motion(c.xy,c.vis,c.groups);geometry={};surfaces={};fit_details={};columns={'frame':np.arange(len(c.t)),'time_sec':c.t};curves={};errors=[];betas=[];streams=[]
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];apex,outlet_y,outlet_reference=cone_and_outlet(c.a['geometry'],j)
        # A stream ROI is a detection crop, not an outlet-height calibration.
        # Use a separately reviewed open outlet when provided; otherwise use
        # the task's annotated cone outlet and record this assumption.
        yy,xx=np.where(c.masks[0,j])
        if len(yy)<12:raise ExtractionError(name+': initial material region unresolved')
        initial_surface=float(np.percentile(yy,1));height_cone=apex[1]-initial_surface
        wall=[]
        for y in range(int(initial_surface+.2*height_cone),int(apex[1]-.15*height_cone)):
            x=np.flatnonzero(c.masks[0,j,y])
            if len(x)>5:
                left,right,edge_visible=visible_inner_edges(c.frames[0],y,x[0],x[-1])
                wall.append([float(apex[1]-y),float((right-left)/2)])
        wall=np.array(wall);radius0=float(np.percentile(abs(xx-apex[0]),99));slope,intercept,wall_quality=wall_geometry_quality(wall,height_cone,fallback_radius=radius0);valid_geometry=wall_quality["valid"]
        h=[];vol=[];rows=[]
        for i,mask in enumerate(c.masks[:,j]):
            axis_x=apex[0]+drift[i,0];apex_y=apex[1]+drift[i,1];points=[];refined_points=0
            for x in range(max(0,int(axis_x-.8*radius0)),min(mask.shape[1],int(axis_x+.8*radius0))):
                y=np.flatnonzero(mask[:,x])
                if len(y)>5:
                    surface_y, resolved = visible_surface_y(c.frames[i], x, float(y[0]))
                    refined_points+=int(resolved)
                    points.append([float(x-axis_x),float(surface_y-drift[i,1])])
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
            h.append(height);vol.append(volume);rows.append({'valid':True,'surface_points_x_relative_y':points,'surface_z_coefficients':[float(x) for x in coef],'axis_surface_y':float(apex[1]-coef[0]),'surface_fit_rms_px':float(np.sqrt(np.mean((z-(coef[0]+coef[1]*x*x))**2))),'local_rgb_edge_refined_points':refined_points,'mask_proposal_retained_points':len(points)-refined_points})
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
        details=fit_integrated_discharge(h,vol,c.t,c.cfg)
        height_ratio=float(np.nanmax(h)/np.nanmin(h));problem=details['failures']
        if not valid_geometry:problem.append('inner-wall geometry residual too large')
        if problem:details['fit']=None;betas.append(None);errors.extend(name+': '+p for p in problem)
        else:betas.append(details['fit']['velocity'])
        geometry[name]={'axis_x':apex[0],'cone_apex_y':apex[1],'open_outlet_y':outlet_y,'outlet_reference':outlet_reference,'cone_apex_reference':'explicit_cone_apex' if 'cone_apex_points' in c.a['geometry'] else 'annotated_outlet_as_apex_assumption','stream_roi_used_as_outlet':False,'wall_R_of_z_slope':float(slope),'wall_R_of_z_intercept':float(intercept),'observed_wall_points_z_R':wall,'axisymmetric_assumption':True,'wall_quality':wall_quality,'volume_formula':'2*pi*integral r*max(z_surface(r)-z_bottom(r),0) dr','valid':valid_geometry}
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
        positive_ids=[i for i in ids if h[i]>0 and q[i]>0]
        if positive_ids:ax.scatter(np.log(h[positive_ids]),np.log(q[positive_ids]),s=18,label=name+' selected')
        if fit:
            x=np.linspace(np.log(h[ids]).min(),np.log(h[ids]).max(),100);ax.plot(x,fit['intercept_at_mean_time']+fit['velocity']*(x-fit['log_height_mean']),label=f'{name} beta={fit["velocity"]:.3f}')
    ax.set(xlabel='log(h)',ylabel='log(Q)',title='Observed-window fits; no relative-height gate');ax.legend();fig.tight_layout();fig.savefig(c.out/'log_q_log_h.png',dpi=150);plt.close(fig)
    write_json(c.out/'geometry.json',geometry);write_json(c.out/'surfaces.json',surfaces);write_json(c.out/'fit_details.json',fit_details)
    c.m['measurements']={'beta_water':betas[0],'beta_sand':betas[1],'raw_m1_error':None,'volume_reconstruction_method':'axisymmetric conical-wall and independently fitted surface radial integration; fixed cone anchored at annotated apex; local RGB surface refinement','water_height_ratio':fit_details['water']['height_ratio'],'sand_height_ratio':fit_details['sand']['height_ratio']}
    c.m['uncertainty']={'note':'Linearized 95% intervals conditional on integrated volume/head fit and reconstructed geometry; wide intervals are reported, not rejected','warnings':{name:fit_details[name]['warnings'] for name in ['water','sand']}}
    for name in ['water','sand']:
        fit=fit_details[name]['fit'];c.m['measurements']['beta_'+name+'_ci']=fit['slope_ci95'] if fit else None;c.m['measurements'][name+'_fit_window_sec']=fit_details[name]['fit_window_sec']
    c.calculation=[f'Observed height ranges and fitting eligibility: {fit_details}',f'Volume-recovery geometry: axisymmetric surface/wall integration; cone apex and open outlet treated separately.',"Neither -dh/dt nor changes in two-dimensional mask area were treated as flow rate."]
    if errors:raise ExtractionError('; '.join(errors))
    err=abs(betas[0]-.5)+abs(betas[1]);score=soft_error_score(err,c.cfg['error_half_score']);c.m['measurements']['raw_m1_error']=err;c.m['reason']="Recover volume from reviewed cone geometry and freely fit the exponent over observed discharge; expand the numerical range adaptively, retain the original exponent-error penalty, and report geometric assumptions and fitting uncertainty separately.";c.calculation.append(f'p36_fixed_cone_integrated_exponent_v5: beta_water={betas[0]}, beta_sand={betas[1]}; E={err}; half-score error={c.cfg["error_half_score"]}; score=1/(1+{err}/{c.cfg["error_half_score"]})={score}')
    return score
