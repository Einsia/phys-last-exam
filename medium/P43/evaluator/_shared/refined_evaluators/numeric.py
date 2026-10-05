"""Task mathematics, independent of models and visual initialization."""
import math
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
from .common import ExtractionError


def saturated_ratio(numerator, denominator, margin):
    if not math.isfinite(margin) or not 0<margin<=1: raise ValueError('margin must satisfy 0 < margin <= 1')
    if any(not math.isfinite(x) or x<0 for x in (numerator,denominator)): raise ValueError('Measurements must be finite and nonnegative')
    if denominator==0: return None,0.0
    ratio=numerator/denominator
    return ratio,float(np.clip((1-ratio)/margin,0,1))


def oscillation_count_score(n_solid, n_slotted):
    """P38 directional v1: fewer solid cycles supports stronger damping."""
    if any(not math.isfinite(x) or x<0 or x!=int(x) for x in (n_solid,n_slotted)):
        raise ValueError('Cycle counts must be finite nonnegative integers')
    if n_slotted==0:return None,0.0
    ratio=n_solid/n_slotted
    return ratio,float(n_solid<n_slotted)


def error_score(error, scale=1.):
    if not math.isfinite(scale) or scale<=0: raise ValueError('Error scale must be positive and finite')
    return float(np.clip(1-error/scale,0,1))


def soft_error_score(error, half_score_error):
    """Continuous error penalty with no finite-error cutoff."""
    if not math.isfinite(half_score_error) or half_score_error<=0:
        raise ValueError('Half-score error must be positive and finite')
    if not math.isfinite(error) or error<0:
        raise ValueError('Error must be finite and nonnegative')
    return float(half_score_error/(half_score_error+error))


def median_time(y,t,width):
    return np.array([np.nanmedian(y[np.abs(t-u)<=width/2+1e-9]) for u in t])


def require_track(valid,t,max_gap=.1,min_fraction=.9):
    valid=np.asarray(valid,bool)
    if not valid[0]: raise ExtractionError('Initial object observation is unavailable')
    if valid.mean()<min_fraction: raise ExtractionError(f'Insufficient valid tracking frames: {valid.mean():.3f}')
    good=np.flatnonzero(valid)
    if len(good)<2 or np.max(np.diff(t[good]))>max_gap+np.median(np.diff(t))+1e-6:
        raise ExtractionError('Long tracking gap; missing observations cannot be treated as zero motion')
    if not valid[-1] and t[-1]-t[good[-1]]>max_gap: raise ExtractionError('Object lost at end of clip')


def height_peak(y,t,noise,window=.12):
    """Max upward displacement with actual descent or a resolved plateau."""
    k=int(np.nanargmax(y)); peak=float(max(0,y[k]))
    after=(t>=t[k])&(t<=t[k]+window+1e-9)
    plateau=(np.abs(y-y[k])<=noise)
    left=k
    while left>0 and plateau[left-1]:left-=1
    right=k
    while right+1<len(y) and plateau[right+1]:right+=1
    is_plateau=bool(t[right]-t[left]>=window)
    descent=bool(np.any((t>t[k])&(y<y[k]-3*noise)))
    clipped=not (is_plateau or descent)
    return {'frame':k,'time_sec':float(t[k]),'height':peak,'plateau_window_sec':[float(t[left]),float(t[right])],'plateau':is_plateau,'descent':descent,'right_censored':clipped}


def count_cycles(theta,t,cutoff,prominence=.5,separation=.1,release_boundary=False):
    """Anchor one polarity; never count overlapping positive-negative-positive triples."""
    dt=float(np.median(np.diff(t))); candidates=[]
    for sign in (1,-1):
        ids,prop=find_peaks(sign*theta,prominence=prominence,distance=max(1,math.ceil(separation/dt)))
        for k,p in zip(ids,prop['prominences']):candidates.append({'frame':int(k),'time_sec':float(t[k]),'angle_deg':float(theta[k]),'polarity':sign,'prominence_deg':float(p),'accepted':bool(sign*theta[k]>=cutoff),'reason':'above_shared_cutoff' if sign*theta[k]>=cutoff else 'below_shared_cutoff'})
    if release_boundary and abs(theta[0])>=cutoff:
        candidates.append({'frame':0,'time_sec':float(t[0]),'angle_deg':float(theta[0]),'polarity':int(np.sign(theta[0])),'prominence_deg':None,'accepted':True,'reason':'verified_release_zero_speed'})
    candidates.sort(key=lambda x:x['frame']); seq=[]
    for p in candidates:
        if not p['accepted']:continue
        if seq and seq[-1]['polarity']==p['polarity']:
            if abs(p['angle_deg'])>abs(seq[-1]['angle_deg']):
                seq[-1]['accepted']=False;seq[-1]['reason']='same_polarity_weaker_peak';seq[-1]=p
            else:p['accepted']=False;p['reason']='same_polarity_weaker_peak'
        else:seq.append(p)
    cycles=[]
    for i in range(0,len(seq)-2,2):
        triple=seq[i:i+3]
        cycles.append({'start_frame':triple[0]['frame'],'end_frame':triple[-1]['frame'],'start_sec':triple[0]['time_sec'],'end_sec':triple[-1]['time_sec'],'extrema_frames':[p['frame'] for p in triple],'accepted':True,'reason':'three_alternating_extrema_above_shared_cutoff'})
    tail=seq[2*len(cycles):]
    return candidates,cycles,{'extrema_frames':[p['frame'] for p in tail],'reason':'incomplete_terminal_cycle','right_censored':bool(abs(theta[-1])>=cutoff or (len(theta)>2 and abs(theta[-1]-theta[-3])>.1))}


def match_events(reference,onsets,tolerance=.5):
    if not math.isfinite(tolerance) or tolerance<0: raise ValueError('Invalid event tolerance')
    n,m=len(reference),len(onsets)
    if not n or not m: return [],0,m,n,0.0
    # A forbidden edge costs more than every possible feasible timing cost.
    diff=np.abs(np.array(reference)[:,None]-np.array(onsets)[None,:]);cost=np.where(diff<=tolerance+1e-10,diff,(n+m+1)*(tolerance+1))
    r,c=linear_sum_assignment(cost);pairs=[{'reference_index':int(i),'light_index':int(j),'delta_sec':float(onsets[j]-reference[i])} for i,j in zip(r,c) if diff[i,j]<=tolerance+1e-10]
    tp=len(pairs);fp=m-tp;fn=n-tp
    return pairs,tp,fp,fn,2*tp/(2*tp+fp+fn)


def linear_fit(t,y):
    x=t-t.mean(); X=np.c_[np.ones(len(t)),x];coef=np.linalg.lstsq(X,y,rcond=None)[0];res=y-X@coef
    se=float(np.sqrt(np.sum(res**2)/max(1,len(t)-2)/max(np.sum(x*x),1e-15)))
    return {'velocity':float(coef[1]),'intercept_at_mean_time':float(coef[0]),'residual_rms':float(np.sqrt(np.mean(res**2))),'slope_se':se,'slope_ci95':[float(coef[1]-1.96*se),float(coef[1]+1.96*se)]}


def fit_discharge_exponent(h,q,qnoise,t,cfg):
    """Fit all observed positive discharge, without a relative-height gate."""
    h,q,qnoise,t=[np.asarray(x,float) for x in (h,q,qnoise,t)]
    reasons=[];independent=[]
    for i in range(len(t)):
        if not np.all(np.isfinite([h[i],q[i],qnoise[i],t[i]])):reason='nonfinite_observation'
        elif h[i]<=0:reason='nonpositive_height'
        elif q[i]<=0:reason='nonpositive_discharge'
        elif qnoise[i]<0 or q[i]<=cfg['min_flow_snr']*qnoise[i]:reason='flow_below_noise_threshold'
        elif independent and t[i]-t[independent[-1]]<cfg['derivative_window_sec']-1e-9:reason='correlated_with_selected_window'
        else:reason='accepted';independent.append(i)
        reasons.append(reason)
    failures=[];warnings=[];fit=None
    if len(independent)<max(3,cfg['min_independent_fit_points']):
        failures.append(f'only {len(independent)} independent positive-flow observations')
    else:
        x=np.log(h[independent]);y=np.log(q[independent])
        if np.ptp(x)<=32*np.finfo(float).eps*max(1.,float(np.max(abs(x)))):
            failures.append('constant observed height: discharge exponent is undefined')
        else:
            fit=linear_fit(x,y)
            fit.update(log_height_mean=float(x.mean()),independent_points=len(independent))
            if fit['slope_ci95'][1]-fit['slope_ci95'][0]>1:
                warnings.append('wide_beta_interval: reported estimate has limited precision; not a rejection gate')
    return {'selection_rule':'all observed finite positive h and Q above flow noise; samples separated by derivative_window_sec; no relative-height or height-ratio gate','independent_fit_frames':independent,'frame_selection_reasons':reasons,'fit_window_sec':[float(t[independent[0]]),float(t[independent[-1]])] if independent else None,'fit':fit,'failures':failures,'warnings':warnings}


def terminal_windows(y,r,t,valid,cfg):
    candidates=[]
    pixel_sigma=float(cfg.get('position_resolution_sigma_px',.5))
    def resolved_fit(tt,yy):
        fit=linear_fit(tt,yy);x=tt-tt.mean()
        fit['slope_se']=max(fit['slope_se'],pixel_sigma/np.sqrt(np.sum(x*x)))
        fit['slope_ci95']=[fit['velocity']-1.96*fit['slope_se'],fit['velocity']+1.96*fit['slope_se']]
        return fit
    # A single long quadratic can hide acceleration followed by deceleration.
    # Retain local slopes too; their range must satisfy the same declared
    # relative-speed-change bound inside every candidate terminal window.
    local=[]
    for k in range(len(t)):
        for l in range(k+3,len(t)):
            if t[l]-t[k]>=cfg['terminal_window_sec']-1e-9:
                if not np.all(valid[k:l+1]):break
                fit=resolved_fit(t[k:l+1],y[k:l+1])
                if fit['velocity']>0 and 1.96*fit['slope_se']/fit['velocity']<=.5*cfg['max_relative_speed_change']:
                    local.append((k,l,fit));break
    for i in range(len(t)):
        for j in range(i+3,len(t)):
            duration=t[j]-t[i]
            if duration<cfg['terminal_window_sec'] or not np.all(valid[i:j+1]):continue
            tt=t[i:j+1]; yy=y[i:j+1]; radius=float(np.median(r[i:j+1])); fit=resolved_fit(tt,yy);v=fit['velocity']
            if v<=0 or v*duration<2*radius*cfg['min_terminal_displacement_diameter']:continue
            if v<cfg['min_speed_snr']*fit['slope_se']:continue
            x=tt-tt.mean();q=np.polyfit(x,yy,2);design=np.c_[x*x,x,np.ones(len(x))]
            sigma=max(pixel_sigma,float(np.sqrt(np.mean((yy-design@q)**2))))
            accel_se=2*sigma*np.sqrt(np.linalg.pinv(design.T@design)[0,0])
            # Equivalence requires the upper uncertainty bound on speed
            # change to fit inside the declared tolerance. Failure to detect
            # acceleration is not positive evidence for terminal motion.
            change=abs(2*q[0]*duration)/v
            change_upper=(abs(2*q[0]*duration)+1.96*accel_se*duration)/v
            if change>cfg['max_relative_speed_change']:continue
            slopes=[f for k,l,f in local if k>=i and l<=j]
            if fit['residual_rms']>max(pixel_sigma*2,.15*v*duration):continue
            local_change=max(0.,max(f['slope_ci95'][0] for f in slopes)-min(f['slope_ci95'][1] for f in slopes))/v if len(slopes)>1 else 0.
            if local_change>cfg['max_relative_speed_change']:continue
            candidates.append({'start_frame':i,'end_frame':j,'window_sec':[float(t[i]),float(t[j])],'duration_sec':float(duration),'radius_px':radius,'relative_speed_change':float(change),'relative_speed_change_upper95':float(change_upper),'local_relative_speed_range':float(local_change),'position_resolution_sigma_px':pixel_sigma,'speed_change_definition':'resolved positive speed and displacement, fitted speed-change tolerance; uncertainty retained as diagnostic, short windows allowed',**fit})
    candidates.sort(key=lambda x:(-x['duration_sec'],x['start_frame']))
    return candidates
