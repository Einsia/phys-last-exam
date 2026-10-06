"""P34: measure the visible plate center independently of feature sampling."""
import cv2
import numpy as np
from refined_evaluators.tasks import observed_track,csv_file,plot
from refined_evaluators.vision import camera_motion
from refined_evaluators.numeric import require_track,median_time,count_cycles,oscillation_count_score
from refined_evaluators.common import ExtractionError,write_json

def visible_plate_center(mask):
    contours,_=cv2.findContours(np.asarray(mask,dtype='uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:raise ExtractionError('Initial plate outline is unavailable')
    outline=max(contours,key=cv2.contourArea)
    if cv2.contourArea(outline)<30:raise ExtractionError('Initial plate outline is unresolved')
    return np.asarray(cv2.minAreaRect(outline)[0],float)

def p34(c):
    c.m['principle']='共同观察窗口、共同角振幅门槛下，比较实心板与开槽板的有效完整周期数。'
    c.m['score_details']={'version':'p34_directional_v1','formula':'1.0 if N_solid < N_slotted else 0.0; zero reference remains 0','range':[0,1],'equal_positive_counts_score':0.,'zero_reference_score':0.,'interpretation':'共同窗口内实心板完整周期数少于开槽板即支持目标关系；相等或更多不支持。分数不是物理成立概率。'}
    drift,res=camera_motion(c.xy,c.vis,c.groups);series=[];cal={};columns={'frame':np.arange(len(c.t)),'time_sec':c.t}
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];xy,valid=observed_track(c,j);source=xy[0]
        centroid=visible_plate_center(c.masks[0,j]);reference=[];fits=[]
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
        cal[name]={'pivot_xy':pivot,'initial_material_reference_xy':centroid,'initial_reference_method':'visible plate outer-rectangle center, independent of tracking-point sampling','fits':fits,'method':'CoTracker points estimate independent rigid transform; transform visible plate outline center, not the biased feature-point mean; annotated suspension point; gravity vertical'}
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
        relation='实心板次数较少，支持目标关系。' if counts[0]<counts[1] else '两板次数相同或实心板次数更多，不支持目标关系。'
        c.m['reason']='完整周期由固定极性边界配对；'+relation+'只计算共同窗口内次数，未计算 M2 衰减率。'
    c.calculation=[f'W={c.m["measurements"]["observation_window_sec"]}; A_cut={cutoff:.6f} deg.',f'有效周期逐项时刻：{cycles}',f'N_solid={counts[0]}, N_slotted={counts[1]}, ratio={ratio}; scoring_version=p34_directional_v1.',f'score = 1.0 if {counts[0]} < {counts[1]} else 0.0 = {score:.8f}.' if counts[1]>0 else 'N_slotted=0: zero_reference_count，score=0；不将 0/0 记为中性分。','只按完整周期数的方向判定：实心板更少为 1，相等或更多为 0。']
    return score
