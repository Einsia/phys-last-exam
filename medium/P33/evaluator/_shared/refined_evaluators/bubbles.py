"""P39: independent observed outer arcs and internal partition, without a Laplace fit prior.

This contour backend uses no video-specific coordinates. Multiple resolved
partition boundaries are reported as projection/identity ambiguity; neither
boundary is chosen because its radius happens to agree with the expected law.
"""
from collections import Counter
from pathlib import Path
import json
import math
import time
import traceback
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from .common import ExtractionError, write_json
from .media import decode, VideoWriter, fingerprint
from .output import finalize
from .resources import ROOT, task_resource
from .scoring import residual_score


def circle_fit(points, min_noise_px=.5):
    points = np.asarray(points, dtype=float)
    if len(points) < 12:
        raise ExtractionError('Too few independently observed arc points')
    origin = points.mean(axis=0)
    local = points-origin
    a = np.c_[2*local, np.ones(len(points))]
    fit, _, rank, _ = np.linalg.lstsq(a, np.sum(local**2,axis=1), rcond=None)
    if rank < 3:
        raise ExtractionError('Circle curvature is unidentifiable for a straight or degenerate arc')
    center = fit[:2]
    radius = math.sqrt(max(1e-12, fit[2]+center@center))
    opt = least_squares(lambda x:np.linalg.norm(local-x[:2],axis=1)-x[2],
                        [*center,radius],loss='soft_l1',f_scale=1.0,
                        bounds=([-np.inf,-np.inf,1e-6],[np.inf,np.inf,np.inf]))
    resid = np.linalg.norm(local-opt.x[:2],axis=1)-opt.x[2]
    sigma = max(min_noise_px,float(np.sqrt(np.mean(resid**2))))
    # Decimate neighboring pixels when estimating effective independent evidence.
    covariance = np.linalg.pinv(opt.jac.T@opt.jac)*sigma**2*min(5,len(points)/12)
    uncertainty = float(np.sqrt(max(0,covariance[2,2])))
    return {'center':(opt.x[:2]+origin).tolist(),'radius':float(opt.x[2]),
            'rmse_px':float(np.sqrt(np.mean(resid**2))),
            'radius_std_px':uncertainty,'relative_radius_uncertainty':uncertainty/opt.x[2],
            'n_points':len(points)}


def locate_pair(frame):
    h,w = frame.shape[:2];scale=min(1.,640/w)
    grey=cv2.cvtColor(cv2.resize(frame,None,fx=scale,fy=scale),cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(cv2.GaussianBlur(grey,(5,5),1),cv2.HOUGH_GRADIENT,1.2,
                            grey.shape[0]*.15,param1=60,param2=35,
                            minRadius=int(grey.shape[0]*.12),maxRadius=int(grey.shape[0]*.49))
    if circles is None:raise ExtractionError('Two sufficiently visible circular bubble outlines were not located')
    candidates=[]
    for c in circles[0]/scale:
        x,y,r=map(float,c)
        if x-r < 0 or y-r < 0 or x+r >= w or y+r >= h:continue
        candidates.append(c)
    pairs=[]
    for i,a in enumerate(candidates):
        for b in candidates[i+1:]:
            d=np.linalg.norm(a[:2]-b[:2]);rsum=a[2]+b[2]
            # Connected bubbles need a resolved common chord. Requiring the
            # center spacing to exceed the larger radius incorrectly excludes
            # many legitimate unequal-bubble geometries.
            if abs(a[2]-b[2])+4 < d < rsum-2:
                pairs.append((float(a[2]+b[2]),a,b))
    if not pairs:raise ExtractionError('No bubble pair with resolved intersecting outer arcs')
    # A large Hough circle can be a short internal membrane arc. Select
    # independently observed outer-arc support, never the largest radius sum
    # or closeness to the Laplace relation being tested.
    supported=[]
    full_grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    for _,a,b in pairs:
        try:
            fa,pa=sample_outer(full_grey,a,b);fb,pb=sample_outer(full_grey,b,a)
        except (ExtractionError,ValueError):continue
        if min(len(pa),len(pb))<40:continue
        score=min(len(pa),len(pb))/(1+fa['rmse_px']+fb['rmse_px'])
        supported.append((score,fa,fb))
    if not supported:raise ExtractionError('No candidate pair has independently resolved outer arcs')
    _,fa,fb=max(supported,key=lambda row:row[0])
    return sorted([fa['center']+[fa['radius']],fb['center']+[fb['radius']]],key=lambda row:row[2])


def ridge_centers(profile, min_contrast=12.):
    """Resolve bright or dark finite-width ridges by their band centroid."""
    baseline=float(np.median(profile));contrast=np.abs(np.asarray(profile,float)-baseline)
    peak=float(contrast.max())
    if peak<min_contrast:return []
    selected=contrast>max(min_contrast,.45*peak)
    indices=np.flatnonzero(selected)
    runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    return [float(np.average(run,weights=contrast[run])) for run in runs if len(run)>=2]


def sample_outer(grey, approximate, other):
    center=np.asarray(approximate[:2],dtype=float);r=approximate[2]
    toward=np.asarray(other[:2],dtype=float)-center;toward/=np.linalg.norm(toward)
    angles=np.linspace(0,2*np.pi,360,endpoint=False)
    directions=np.c_[np.cos(angles),np.sin(angles)]
    # Use the outer side of each bubble, away from the deformed junction.
    directions=directions[directions@toward < .15]
    radii=np.linspace(.85*r,1.15*r,101)
    xy=center+directions[:,None,:]*radii[None,:,None]
    samples=cv2.remap(grey,xy[:,:,0].astype('float32'),xy[:,:,1].astype('float32'),cv2.INTER_LINEAR)
    points=[]
    for direction,profile in zip(directions,samples):
        peaks=ridge_centers(profile)
        if not peaks:continue
        peak=min(peaks,key=lambda index:abs(index-50))
        radius=float(np.interp(peak,np.arange(len(radii)),radii))
        points.append(center+direction*radius)
    points=np.asarray(points)
    result=circle_fit(points)
    residual=np.abs(np.linalg.norm(points-np.asarray(result['center']),axis=1)-result['radius'])
    points=points[residual<max(2.0,.025*result['radius'])]
    return circle_fit(points),points


def partition_geometry(small,large):
    cs=np.asarray(small['center'],dtype=float);cl=np.asarray(large['center'],dtype=float);rs=small['radius'];rl=large['radius']
    direction=cl-cs;distance=np.linalg.norm(direction);direction/=distance
    x=(rs*rs-rl*rl+distance*distance)/(2*distance)
    height2=rs*rs-x*x
    if height2<=0:raise ExtractionError('Contact partition has not formed a resolvable chord')
    normal=np.array([-direction[1],direction[0]])
    if normal[1]<0:normal=-normal
    mid=cs+x*direction;half=math.sqrt(height2)
    return mid, direction, normal, half


def partition_edges(grey,mid,direction,normal,half):
    z=np.linspace(-.87*half,.87*half,max(24,int(1.74*half)))
    # A one-pixel membrane is still a resolved image feature. Sample it at
    # half-pixel spacing so the two-sample ridge check is independent of the
    # fractional location of the chord grid; coverage/fitting checks remain.
    offsets=np.arange(-.38*half,.38*half+.5,.5)
    xy=mid+z[:,None,None]*normal+offsets[None,:,None]*direction
    samples=cv2.remap(grey,xy[:,:,0].astype('float32'),xy[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
    high=samples-gaussian_filter1d(samples,4,axis=1)
    noise=max(1.,float(np.median(np.abs(high-np.median(high)))*1.4826))
    left=[];right=[];support=[]
    for i,row in enumerate(samples):
        peaks=ridge_centers(row,min_contrast=max(12.,3*noise))
        if not len(peaks):continue
        # Keep observed extremal ridges; do not score-select a membrane curve.
        left.append(mid+z[i]*normal+np.interp(peaks[0],np.arange(len(offsets)),offsets)*direction)
        right.append(mid+z[i]*normal+np.interp(peaks[-1],np.arange(len(offsets)),offsets)*direction)
        support.append(i)
    return np.asarray(left).reshape(-1,2),np.asarray(right).reshape(-1,2),len(support)/len(z),noise


def signed_partition_fit(points,mid,direction,normal,half):
    local=np.asarray(points)-mid
    z=local@normal;x=local@direction
    polynomial=np.polyfit(z/half,x,2)
    # Chord runs between both endpoints. Midpoint bulge determines observed sign.
    sag=-polynomial[0]
    fit=circle_fit(points)
    fit['signed_radius']=math.copysign(fit['radius'],sag)
    fit['signed_curvature']=1/fit['signed_radius']
    fit['sagitta_px']=float(sag)
    return fit


def straight_partition_limit(points,mid,direction,normal,half,small,large):
    """Prove a near-zero curvature independently before assigning an unbounded limit."""
    local=np.asarray(points)-mid;z=(local@normal)/half;x=local@direction
    design=np.c_[np.ones(len(z)),z,z*z]
    parameters=np.linalg.lstsq(design,x,rcond=None)[0]
    residual=x-design@parameters
    sigma=max(.5,float(np.sqrt(np.mean(residual**2))))
    covariance=np.linalg.pinv(design.T@design)*sigma**2*min(5,len(z)/12)
    curvature=-2*parameters[2]/half**2
    standard_error=2*math.sqrt(max(0,covariance[2,2]))/half**2
    expected=1/small['radius']-1/large['radius']
    expected_error=math.hypot(small['radius_std_px']/small['radius']**2,large['radius_std_px']/large['radius']**2)
    upper=abs(curvature)+3*standard_error
    lower_expected=expected-3*expected_error
    proven=abs(curvature)<=3*standard_error and upper<.25*lower_expected and sigma<1.5
    return {'proven':bool(proven),'curvature_px_inv':float(curvature),
            'curvature_std_px_inv':float(standard_error),'curvature_abs_upper_3sigma':float(upper),
            'expected_curvature_lower_3sigma':float(lower_expected),
            'policy':'Measured near-zero curvature with uncertainty excludes the independently predicted curvature; no theoretical fit constraint'}


def inspect_frame(frame,approximate,cfg):
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    row={'valid':False,'formed':False,'reason':None,'outer':{},'contours':{}}
    updated=approximate
    try:
        observations=[]
        for j,name in enumerate(('small','large')):
            fit,points=sample_outer(grey,approximate[j],approximate[1-j])
            row['outer'][name]=fit;row['contours'][name]=points.tolist();observations.append(fit)
            if fit['rmse_px']/fit['radius']>cfg['max_circle_relative_residual'] or fit['relative_radius_uncertainty']>cfg['max_relative_radius_uncertainty']:
                raise ExtractionError('Outer arc radius is not reliably measured')
        small,large=observations
        updated=[[*x['center'],x['radius']] for x in observations]
        if large['radius']/small['radius']<cfg['min_radius_ratio']:
            raise ExtractionError('Bubble sizes are not sufficiently distinguishable')
        mid,direction,normal,half=partition_geometry(small,large)
        row['endpoints']=[(mid-half*normal).tolist(),(mid+half*normal).tolist()]
        if half/small['radius'] < cfg['min_partition_chord_small_diameter_ratio']:
            raise ExtractionError('Partition chord is too short for curvature measurement')
        row['formed']=True
        left,right,coverage,noise=partition_edges(grey,mid,direction,normal,half)
        row['contours'].update(partition_left=left.tolist(),partition_right=right.tolist())
        row.update(edge_coverage=coverage,edge_noise=noise,chord_length_px=2*half)
        if coverage<cfg['min_edge_coverage']:
            raise ExtractionError('Insufficient visible partition edge coverage')
        separation=np.linalg.norm(left-right,axis=1)
        row['partition_band_width_px']=float(np.median(separation))
        fits=[]
        for points in (left,right):
            try:fits.append(signed_partition_fit(points,mid,direction,normal,half))
            except (ExtractionError,ValueError):fits.append(None)
        row['partition_candidates']=fits
        separated=separation>max(3.,cfg['max_partition_band_chord_ratio']*2*half)
        if np.mean(separated)>.5:
            raise ExtractionError('Two resolved partition boundaries: a unique membrane meridian and its projection cannot be established')
        points=(left+right)/2
        straight=straight_partition_limit(points,mid,direction,normal,half,small,large)
        row['straight_partition_test']=straight
        if straight['proven']:
            row.update(valid=True,raw_m1_error=None,error_kind='unbounded_straight_partition',
                       reason='Resolved near-straight partition with curvature uncertainty excluding the unequal-bubble prediction')
            return row,updated
        fit=signed_partition_fit(points,mid,direction,normal,half)
        row['partition']=fit
        if fit['relative_radius_uncertainty']>cfg['max_relative_radius_uncertainty']:
            # Do not turn an unidentifiable, near-straight arc into an infinite error.
            raise ExtractionError('Partition curvature uncertainty is too large')
        if fit['rmse_px']/fit['radius']>cfg['max_circle_relative_residual']:
            raise ExtractionError('Partition is not adequately represented by an independent circular arc')
        row['raw_m1_error']=abs(fit['signed_radius']*(1/small['radius']-1/large['radius'])-1)
        row.update(valid=True,reason='Three independently observed circular arcs with unambiguous partition direction')
    except (ExtractionError,ValueError,np.linalg.LinAlgError) as exc:
        row['reason']=str(exc)
    return row,updated


def weighted_median(values,weights):
    values=np.asarray(values);weights=np.asarray(weights)
    order=np.argsort(values,kind='stable');return float(values[order][np.searchsorted(np.cumsum(weights[order]),weights.sum()/2)])


def summarize(rows,t,cfg):
    intervals=np.diff(t)
    durations=np.r_[intervals[0]/2,(intervals[:-1]+intervals[1:])/2,intervals[-1]/2]
    formed=np.array([row['formed'] for row in rows]);valid=np.array([row['valid'] for row in rows])
    duration=float(durations[valid].sum());candidate_duration=float(durations[formed].sum())
    measurements={'raw_m1_error':None,'valid_duration_sec':duration,'candidate_duration_sec':candidate_duration,
                  'valid_frame_fraction':float(np.mean(valid[formed])) if formed.any() else 0,
                  'valid_window_sec':None,'minimum_observations':3,'coverage_policy':'local invalid frames skipped; usable duration and three independent observations suffice','invalid_reasons':dict(Counter(row['reason'] for row in rows if not row['valid']))}
    if valid.sum()<3 or duration<cfg['min_valid_duration_sec']:
        return measurements,None
    values=np.array([np.inf if row.get('error_kind')=='unbounded_straight_partition' else row['raw_m1_error'] for row in rows if row['valid']]);weights=durations[valid]
    error=weighted_median(values,weights)
    measurements['raw_m1_error']=error if math.isfinite(error) else None
    measurements['error_kind']='finite' if math.isfinite(error) else 'unbounded_straight_partition'
    measurements['valid_window_sec']=[float(t[valid][0]),float(t[valid][-1])]
    for key,name in [('small','r_small_px_median'),('large','r_large_px_median')]:
        measurements[name]=weighted_median([row['outer'][key]['radius'] for row in rows if row['valid']],weights)
    partition_values=[row.get('partition',{}).get('signed_radius') for row in rows if row['valid']]
    measurements['r_partition_signed_px_median']=weighted_median(partition_values,weights) if all(x is not None for x in partition_values) else None
    return measurements,residual_score(error,cfg['error_at_zero']) if math.isfinite(error) else 0.0


def save_evidence(frames,t,pts,rows,directory):
    h,w=frames[0].shape[:2];sheet=[];picks=set(np.linspace(0,len(frames)-1,12).astype(int))
    masks=np.zeros((len(frames),3,h,w),dtype=bool)
    names=('small_bubble','large_bubble','partition')
    writers={name:VideoWriter(directory/(name+'.mp4'),w,h,t) for name in (*names,'segmentation_overlay','circle_fit_overlay')}
    colors={'small':(50,220,40),'large':(255,170,30),'partition_left':(30,220,255),'partition_right':(230,100,255)}
    try:
        for i,(frame,row) in enumerate(zip(frames,rows)):
            canvas=frame.copy()
            for key,points in row['contours'].items():
                points=np.asarray(points)
                if len(points)==0:continue
                points=points.round().astype('int32')
                plane=np.zeros((h,w),np.uint8)
                for point in points:cv2.circle(plane,tuple(point),2,1,-1)
                j=0 if key=='small' else 1 if key=='large' else 2;masks[i,j]|=plane.astype(bool)
                for point in points:cv2.circle(canvas,tuple(point),1,colors[key],-1)
            for key,fit in row['outer'].items():
                cv2.circle(canvas,tuple(np.rint(fit['center']).astype(int)),round(fit['radius']),colors[key],1)
                cv2.putText(canvas,f"{key} R={fit['radius']:.1f}px",tuple(np.rint(fit['center']).astype(int)),0,.6,colors[key],2)
            for point in row.get('endpoints',[]):cv2.circle(canvas,tuple(np.rint(point).astype(int)),4,(255,255,255),1)
            cv2.putText(canvas,f'P39 f{i} PTS={t[i]:.3f}s valid={row["valid"]}',(20,30),0,.65,(255,255,255),2)
            label='Unique partition' if row['valid'] else str(row['reason'])[:95]
            cv2.putText(canvas,label,(20,h-22),0,.52,(20,200,255),1)
            for j,name in enumerate(names):
                writers[name].write(np.where(masks[i,j,:,:,None],frame,0).astype('uint8'),t[i])
            writers['segmentation_overlay'].write(canvas,t[i]);writers['circle_fit_overlay'].write(canvas,t[i])
            if i in picks:
                cv2.imwrite(str(directory/f'keyframe_{i:04d}.png'),canvas);sheet.append(cv2.resize(canvas,(504,288)))
    finally:
        for writer in writers.values():writer.close()
    if len(sheet)==12:cv2.imwrite(str(directory/'review_contact_sheet.jpg'),np.vstack([np.hstack(sheet[k:k+3]) for k in range(0,12,3)]))
    np.savez_compressed(directory/'masks.npz',masks=masks,object_names=np.array(names))
    write_json(directory/'timestamps.json',[dict(p,frame_index=i) for i,p in enumerate(pts)])
    write_json(directory/'contours.json',{'method':'Observed intensity ridge pixels; circle models are overlays only','frames':[{'frame':i,'pts_sec':float(t[i]),**row['contours']} for i,row in enumerate(rows)]})
    write_json(directory/'circle_fits.json',[dict(frame=i,pts_sec=float(t[i]),**{key:value for key,value in row.items() if key!='contours'}) for i,row in enumerate(rows)])
    table=['frame,pts_sec,formed,valid,r_small_px,r_large_px,r_partition_signed_px,raw_error,partition_band_width_px']
    for i,row in enumerate(rows):
        values=[i,float(t[i]),int(row['formed']),int(row['valid']),row['outer'].get('small',{}).get('radius'),row['outer'].get('large',{}).get('radius'),row.get('partition',{}).get('signed_radius'),row.get('raw_m1_error'),row.get('partition_band_width_px')]
        table.append(','.join('' if x is None else str(x) for x in values))
    (directory/'radii_error.csv').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for key in ('small','large'):
        axes[0].plot(t,[row['outer'].get(key,{}).get('radius',np.nan) for row in rows],label=key)
    axes[0].set_ylabel('Outer arc radius (px)');axes[0].legend()
    axes[1].plot(t,[row.get('partition_band_width_px',np.nan) for row in rows]);axes[1].set_ylabel('Observed ridge separation (px)')
    axes[2].plot(t,[row.get('raw_m1_error',np.nan) for row in rows],'.');axes[2].set_ylabel('Valid-frame physical error');axes[2].set_xlabel('Source PTS (s)')
    fig.tight_layout();fig.savefig(directory/'radii_error.png',dpi=150);plt.close(fig)


def run(argv=None):
    from .runtime import parser,new_result
    from .definitions import DEFAULTS
    args=parser('P39').parse_args(argv);started=time.monotonic()
    sid=args.sample_id or Path(args.video_path).stem
    out=Path(args.debug_dir) if args.debug_dir else Path(args.output).parent/('debug_'+sid)
    out.mkdir(parents=True,exist_ok=True);result=new_result('P39',args.video_path,args.image_path,args.video_prompt,args.model,args.seed,sid)
    verbose=result['verbose']['M1'];code=0
    verbose['principle']="Independently fit both outer bubble arcs and the uniquely identifiable signed partition arc; aggregate abs(r_s*(1/r_small-1/r_large)-1) using the real-PTS-weighted median, without imposing theoretical curvature constraints."
    try:
        cfg=dict(DEFAULTS['P39'])
        if args.config:
            values=json.loads(Path(args.config).read_text())
            if not isinstance(values,dict) or set(values)-set(cfg):raise ValueError('Unknown P39 configuration keys')
            cfg.update(values)
        cfg.update({key:getattr(args,key) for key in cfg if getattr(args,key) is not None})
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0 for v in cfg.values()):raise ValueError('All P39 parameters must be finite and positive')
        verbose['thresholds']=cfg
        if args.video_prompt_file:result['video_prompt']=Path(args.video_prompt_file).read_text().strip()
        frames,t,pts=decode(args.video_path)
        if args.image_path:
            still=cv2.imread(args.image_path)
            if still is None:raise ExtractionError('Supplied first image is unreadable')
            still=cv2.resize(still,frames[0].shape[1::-1]);difference=float(np.mean(np.abs(cv2.GaussianBlur(still,(5,5),0).astype(float)-cv2.GaussianBlur(frames[0],(5,5),0).astype(float))))
            verbose['first_frame_check']={'mean_absolute_difference':difference,'accepted':difference<15}
            if difference>=15:raise ExtractionError('First image does not match this video')
        approximate=locate_pair(frames[0]);rows=[]
        for i,frame in enumerate(frames):
            row,approximate=inspect_frame(frame,approximate,cfg);rows.append(row)
            if i%30==0:print(f'P39: measured image contours in frame {i}/{len(frames)}',flush=True)
        save_evidence(frames,t,pts,rows,out)
        verbose['measurements'],q=summarize(rows,t,cfg)
        verbose['score_details']={'formula':'1 / (1 + E_M1 / error_at_zero)','half_score_error':cfg['error_at_zero'],'finite_error_cutoff':False,'aggregation':'PTS duration-weighted median of simultaneous-frame residuals'}
        verbose['extraction_run']={'video_decoded':True,'physical_measurements_recomputed':True,'backend':'OpenCV observed contours + independent SciPy circle fits','video_sha256':fingerprint(args.video_path),'model_inference_used':False}
        verbose['evidence']=[{'stage':'observed_contours','path':str(out/'contours.json')},{'stage':'independent_fits','path':str(out/'circle_fits.json')},{'stage':'visual_measurements','path':str(out/'circle_fit_overlay.mp4')}]
        verbose['applicability']={'projection':'A unique, sufficiently thin membrane contour is required for the side-view meridian approximation. Multiple resolved boundaries are not assigned a 3D curvature.','fit_prior':'No Laplace pressure or ideal radius constraint used','mask_semantics':'Observed contour pixel masks, not solid-object SAM segmentation'}
        if q is None:raise ExtractionError('Insufficient reliably measurable partition duration: '+json.dumps(verbose['measurements']['invalid_reasons'],ensure_ascii=False))
        result['metrics']['M1']={'extract_success':True,'metric':q}
        verbose.update(status='scored',reason='Three independent arcs measured; wrong sign or nonideal radius is scored rather than rejected')
    except ExtractionError as exc:verbose.update(status='extraction_failed',reason=str(exc));code=1
    except (ValueError,FileNotFoundError) as exc:verbose.update(status='configuration_error',reason=str(exc));code=2
    except Exception as exc:
        verbose.update(status='runtime_error',reason=f'{type(exc).__name__}: {exc}');code=2
        (out/'traceback.txt').write_text(traceback.format_exc())
    verbose['elapsed_sec']=time.monotonic()-started
    result=finalize(result,task_resource('P39'),out);write_json(args.output,result)
    print(f'P39: {verbose["status"]}: {verbose["reason"]}',flush=True)
    return code
