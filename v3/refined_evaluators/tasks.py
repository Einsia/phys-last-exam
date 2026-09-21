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


def p37(c):
    c.m['principle']='由首帧缺口固定环身份，比较相对初始位置、固定初始外径归一化的最大升高量。'
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
    else:c.m['reason']='两环身份由可见缺口固定；两个峰值均有平台/下降证据。按实际高度比给分。'
    c.calculation=[f'闭合环 h={hclosed:.8f} 外径 ({cal["closed_ring"]["height_px"]:.4f} px)，开口环 h={hopen:.8f} 外径 ({cal["open_ring"]["height_px"]:.4f} px)。',f'峰值: {peaks}',f'height_ratio={ratio}; margin={c.cfg["margin"]}; full-score boundary={1-c.cfg["margin"]}; score={score:.8f}.']
    return score


def p38(c):
    c.m['principle']='共同观察窗口、共同角振幅门槛下，比较实心板与开槽板的有效完整周期数。'
    c.m['score_details']={'version':'p38_count_balance_v2','formula':'N_slotted / (N_solid + N_slotted) if N_slotted > 0 else 0','range':[0,1],'equal_positive_counts_score':.5,'zero_reference_score':0.,'interpretation':'0.5 表示共同窗口内次数相同；高于 0.5 表示实心板次数较少，低于 0.5 表示实心板次数较多。分数不是物理成立概率。'}
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
        relation='两板次数相同，得 0.5 中性分，本窗口未区分出阻尼差异。' if counts[0]==counts[1] else '实心板次数较少，得分高于 0.5。' if counts[0]<counts[1] else '实心板次数较多，得分低于 0.5。'
        c.m['reason']='完整周期由固定极性边界配对；'+relation+'只计算共同窗口内次数，未计算 M2 衰减率。'
    c.calculation=[f'W={c.m["measurements"]["observation_window_sec"]}; A_cut={cutoff:.6f} deg.',f'有效周期逐项时刻：{cycles}',f'N_solid={counts[0]}, N_slotted={counts[1]}, ratio={ratio}; scoring_version=p38_count_balance_v2.',f'score = {counts[1]} / ({counts[0]} + {counts[1]}) = {score:.8f}.' if counts[1]>0 else 'N_slotted=0: zero_reference_count，score=0；不将 0/0 记为中性分。','0.5 是相同正次数的中性分；使用次数比平滑映射，不再使用旧 margin 饱和评分。']
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


def p49(c):
    c.m['principle']='固定大球为球1、小球为球2；独立拟合各自终端窗口与真实外轮廓半径，计算 Stokes 平方律相对误差。'
    c.m['score_details']={'formula':'1 / (1 + abs((v1/v2)/(r1/r2)^2-1)/error_at_zero)','range':[0,1], 'half_score_error':c.cfg['error_at_zero'], 'finite_error_cutoff':False, 'parameter_note':'error_at_zero is retained as a legacy CLI name; its unchanged value is now the half-score scale a'}
    drift,res=camera_motion(c.xy,c.vis,c.groups);radii={};windows={};chosen=[];columns={'frame':np.arange(len(c.t)),'time_sec':c.t};curves={};obsall={}
    for j,obj in enumerate(c.a['objects']):
        name=obj['name'];obs=[circle_observation(m) for m in c.masks[:,j]];obsall[name]=obs
        centers=np.array([x['center'] if x else [np.nan,np.nan] for x in obs])-drift;r=np.array([x['radius'] if x else np.nan for x in obs]);valid=np.array([x is not None and x['valid'] for x in obs]);require_track(valid,c.t,.1,.9)
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
    c.m['reason']='外球轮廓与触底前非零匀速窗口已独立拟合；理论偏离正常计低分。'
    c.calculation=[f'球1=大球，球2=小球。r1={r1:.6f} px，r2={r2:.6f} px。',f'v1={v1:.6f} px/s @ {chosen[0]["window_sec"]}; v2={v2:.6f} px/s @ {chosen[1]["window_sec"]}.',f'r1/r2={rr:.8f}; v1/v2={vr:.8f}; E=abs({vr:.8f}/{rr:.8f}^2-1)={err:.8f}; score={score:.8f}.']
    return score


def quad_observation(mask,previous=None):
    from itertools import permutations
    contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    rect_area=abs(cv2.contourArea(p.astype('float32')));filled=cv2.contourArea(contour)
    return p,float(abs(filled-rect_area)/max(rect_area,1))


def p43(c):
    c.m['principle']='独立检测质心投影越过支撑边、以及角运动从准静态进入持续倾倒的起点；保留原始事件帧差。'
    c.m['score_details']={'version':'p43_soft_time_error_v2','formula':'1 / (1 + abs(delta_time_sec) / time_error_half_score_sec)','range':[0,1],'half_score_error_sec':c.cfg['time_error_half_score_sec'],'finite_error_cutoff':False}
    drift,res=camera_motion(c.xy,c.vis,c.groups);quad=[];fiterrors=[];previous=None
    for mask in c.masks[:,0]:
        obs=quad_observation(mask,previous)
        if obs is None:quad.append(np.full((4,2),np.nan));fiterrors.append(None)
        else:previous=obs[0];quad.append(previous);fiterrors.append(obs[1])
    quad=np.array(quad);valid=np.all(np.isfinite(quad),axis=(1,2));require_track(valid,c.t,.1,.95)
    for j in range(4):
        for axis in range(2):quad[:,j,axis]=np.interp(c.t,c.t[valid],quad[valid,j,axis])
    q=quad-drift[:,None,:];com=q.mean(1);support=q[:,2];d=com[:,0]-support[:,0]
    top=(q[:,0]+q[:,1])/2;bottom=(q[:,2]+q[:,3])/2;v=top-bottom;theta=np.degrees(np.arctan2(v[:,0],-v[:,1]));theta=median_time(theta,c.t,.08)
    omega=np.array([linear_fit(c.t[np.abs(c.t-t)<=c.cfg['onset_window_sec']/2],theta[np.abs(c.t-t)<=c.cfg['onset_window_sec']/2])['velocity'] for t in c.t])
    diag=float(np.linalg.norm(q[0,0]-q[0,2]));hyst=diag*c.cfg['com_hysteresis_diagonal_ratio']
    cross=[]
    for i in range(1,len(c.t)):
        if d[i-1]<=0<d[i]:
            later=(c.t>=c.t[i])&(c.t<=c.t[i]+c.cfg['onset_hold_sec']+1e-9)
            if np.any(d[later]>=hyst):cross.append(i)
    # Onset reads only angle/time, never COM distance or a theoretical critical angle.
    onset=[]
    for i in range(1,len(c.t)):
        before=(c.t>=c.t[i]-c.cfg['onset_window_sec'])&(c.t<c.t[i]);after=(c.t>=c.t[i])&(c.t<=c.t[i]+c.cfg['onset_hold_sec']+1e-9)
        if before.sum()<3 or after.sum()<3:continue
        pre=float(np.median(omega[before]));post=float(np.median(omega[after]));noise=max(.5,float(np.median(abs(omega[before]-pre))))
        ok=post>=c.cfg['min_tip_speed_deg_per_sec'] and post-pre>max(2*noise,1.) and pre<c.cfg['min_tip_speed_deg_per_sec'] and np.all(omega[after]>c.cfg['min_tip_speed_deg_per_sec']*.8)
        if ok:onset.append({'frame':i,'time_sec':float(c.t[i]),'pre_speed':pre,'post_speed':post,'noise_deg_s':noise})
    # Adjacent candidates represent one threshold-crossing interval, not competing distant onsets.
    groups=[]
    for e in onset:
        if not groups or e['frame']>groups[-1][-1]['frame']+1:groups.append([e])
        else:groups[-1].append(e)
    events={'com_crossing_frames':cross,'tip_candidates':onset,'tip_candidate_groups':groups,'onset_rule':'first sustained >= configured min tip speed transition from a below-threshold quasi-static state; independent of COM','prompt_benchmark_mismatch':True,'support_drift_px':np.linalg.norm(support-support[0],axis=1)}
    columns={'frame':np.arange(len(c.t)),'time_sec':c.t,'theta_deg':theta,'omega_deg_s':omega,'com_distance_px':d,'com_x':com[:,0],'support_x':support[:,0]}
    csv_file(c.out/'pose_events.csv',columns);plot(c.out/'pose_events.png',c.t,{'theta_deg':theta,'omega_deg_s':omega,'com_distance_px':d},'Angle / angular speed / signed COM distance',[0])
    write_json(c.out/'corners.json',{'corners_TL_TR_BR_BL':quad,'fit_area_relative_errors':fiterrors});write_json(c.out/'events.json',events)
    c.annotations['block']={'quads':quad,'trajectory_xy':quad.mean(1),'com_distance':d,'angles':theta}
    c.m['applicability']={'prompt_benchmark_mismatch':True,'reason':'Actuator continues pushing. Operational kinematic onset is not proof of free gravitational instability.','support_drift_max_px':float(np.max(np.linalg.norm(support-support[0],axis=1)))}
    if not cross:raise ExtractionError('No observed COM crossing before clip end')
    if not groups:raise ExtractionError('no_identifiable_tip_onset: no independent angular-speed transition')
    if len(groups)>1:raise ExtractionError('Multiple separated credible angular onset transitions')
    if c.t[groups[0][-1]['frame']]-c.t[groups[0][0]['frame']]>c.cfg['max_event_uncertainty_sec']:raise ExtractionError('Tip onset uncertainty exceeds configured limit')
    a=cross[0];b=groups[0][0]['frame'];dt=float(c.t[a]-c.t[b]);score=soft_error_score(abs(dt),c.cfg['time_error_half_score_sec'])
    c.m['measurements']={'N_COM_cross':a,'N_tip':b,'delta_frames':a-b,'t_COM_cross_sec':float(c.t[a]),'t_tip_sec':float(c.t[b]),'delta_time_sec':dt,'event_uncertainty_sec':[float(c.t[groups[0][0]['frame']]),float(c.t[groups[0][-1]['frame']])],'prompt_benchmark_mismatch':True}
    c.m['reason']='质心越界与角速度起点独立测量，按时间差平滑扣分；持续外力驱动下的运动起点不等同于自由失稳，定义差异已记录。'
    c.calculation=[f'N_COM_cross={a} ({c.t[a]:.6f}s); N_tip={b} ({c.t[b]:.6f}s).',f'Delta_N={a-b}; Delta_t={dt:.8f}s; half-score error={c.cfg["time_error_half_score_sec"]}s; score=1/(1+{abs(dt):.8f}/{c.cfg["time_error_half_score_sec"]})={score:.8f}.','Scoring version: p43_soft_time_error_v2. 有限时间差平滑扣分，取消旧0.5秒截零规则。','Onset operational definition and actuator/benchmark mismatch are saved in events.json.']
    return score


def p39(c):
    c.m['principle']='独立测量前端进入和从另一端露出的时刻，与灯亮起点按配置容差一对一匹配；不直接测磁通。'
    c.m['score_details']={'formula':'2*TP/(2*TP+FP+FN)','range':[0,1],'matching_rule':'abs(t_light_onset-t_crossing)<=onset_tolerance_sec'}
    drift,res=camera_motion(c.xy,c.vis,c.groups)
    entry=float(np.mean(np.array(c.a['geometry']['entry_plane'])[:,0]));exit=float(np.mean(np.array(c.a['geometry']['exit_plane'])[:,0]));front=[];rawbright=[];bg=[];tip_evidence=[]
    for i,f in enumerate(c.frames):
        hsv=cv2.cvtColor(f,cv2.COLOR_BGR2HSV);red=((hsv[:,:,0]<8)|(hsv[:,:,0]>170))&(hsv[:,:,1]>140)&(hsv[:,:,2]>75)&c.masks[i,0]
        n,labels,stats,cent=cv2.connectedComponentsWithStats(red.astype('uint8'));components=[s for s in stats[1:] if s[4]>=50]
        if components:
            s=max(components,key=lambda s:s[0]+s[2]);front.append(float(s[0]+s[2]-1-drift[i,0]));tip_evidence.append({'visible':True,'red_component_box':s[:4],'area':int(s[4])})
        else:front.append(np.nan);tip_evidence.append({'visible':False,'reason':'occluded_by_coil; no invisible point promoted to observation'})
        gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(float)
        x1,y1,x2,y2=c.a['geometry']['lamp_emission_box'];xoff,yoff=np.rint(drift[i]).astype(int);roi=gray[y1+yoff:y2+yoff,x1+xoff:x2+xoff];rawbright.append(float(np.mean(roi)))
        x1,y1,x2,y2=c.a['geometry']['lamp_background_box'];bg.append(float(np.mean(gray[y1+yoff:y2+yoff,x1+xoff:x2+xoff])))
    front=np.array(front);rawbright=np.array(rawbright);bg=np.array(bg);brightness=rawbright-bg
    threshold=(exit-entry)*c.cfg['crossing_hysteresis_coil_length_ratio'];crossings=[]
    for label,plane in [('enter',entry),('exit',exit)]:
        idx=np.flatnonzero(front>=plane+threshold)
        if not len(idx):continue
        for k in idx:
            window=(c.t>=c.t[k])&(c.t<=c.t[k]+c.cfg['crossing_hold_sec']+1e-9)
            if np.count_nonzero((front>=plane)&window)<2:continue
            first=k
            while first>0 and np.isfinite(front[first-1]) and front[first-1]>=plane:first-=1
            crossings.append({'id':label,'frame':int(first),'time_sec':float(c.t[first]),'plane_x':plane,'source':'visible red leading end at near mouth' if label=='enter' else 'visible red tip reacquired beyond far rim','time_uncertainty_sec':float(np.median(np.diff(c.t)))});break
    # Baseline uses observed stationary magnet frames at the start, with explicit
    # low-intensity filament evidence. The initial frame is never assumed dark blindly.
    speed=np.r_[0,np.abs(np.diff(front))/np.diff(c.t)];stationary=(speed<3)&np.isfinite(front)&(front<entry-10)
    ids=np.flatnonzero(stationary)
    baseline_ids=ids[c.t[ids]<c.t[ids[0]]+1.] if len(ids) else np.array([],int)
    if len(baseline_ids)<3:raise ExtractionError('No observed stationary dark calibration interval')
    base=float(np.median(brightness[baseline_ids]));noise=max(1.,1.4826*float(np.median(abs(brightness[baseline_ids]-base))))
    dark_verified=bool(np.median(rawbright[baseline_ids])<90 and np.percentile(rawbright[baseline_ids],95)<110)
    if not dark_verified:raise ExtractionError('Initial lamp is not demonstrably dark; cannot define onset baseline')
    on=base+c.cfg['brightness_on_sigma']*noise;off=base+c.cfg['brightness_off_sigma']*noise;events=[];start=None
    for i,value in enumerate(brightness):
        if start is None and value>on:start=i
        if start is not None and (value<off or i==len(c.t)-1):
            end=i
            if c.t[end]-c.t[start]>=c.cfg['min_event_duration_sec']:events.append({'start_frame':start,'end_frame':end,'t_on':float(c.t[start]),'t_off':float(c.t[end]),'left_censored':start==0,'right_censored':i==len(c.t)-1 and value>=off})
            start=None
    merged=[]
    for e in events:
        if merged and e['t_on']-merged[-1]['t_off']<=c.cfg['merge_gap_sec']:merged[-1].update(end_frame=e['end_frame'],t_off=e['t_off'],right_censored=e['right_censored'])
        else:merged.append(e)
    write_json(c.out/'brightness_calibration.json',{'dark_verified':dark_verified,'baseline_frames':baseline_ids,'baseline':base,'noise':noise,'on_threshold':on,'off_threshold':off,'lamp_emission_box':c.a['geometry']['lamp_emission_box'],'background_box':c.a['geometry']['lamp_background_box']})
    write_json(c.out/'events.json',{'geometric':crossings,'light':merged,'tip_evidence':tip_evidence})
    csv_file(c.out/'motion_brightness.csv',{'frame':np.arange(len(c.t)),'time_sec':c.t,'leading_end_x':front,'lamp_raw':rawbright,'background':bg,'lamp_corrected':brightness});plot(c.out/'motion_brightness.png',c.t,{'corrected lamp brightness':brightness,'on threshold':np.full(len(c.t),on),'off threshold':np.full(len(c.t),off)},'Background-corrected filament intensity')
    c.m['measurements']={'crossing_events':crossings,'light_events':merged,'t_enter_sec':next((x['time_sec'] for x in crossings if x['id']=='enter'),None),'t_exit_sec':next((x['time_sec'] for x in crossings if x['id']=='exit'),None)}
    c.annotations['events']={'crossings':crossings,'light':merged}
    if len(crossings)!=2:raise ExtractionError('Complete entry and far-side exit could not both be measured')
    pairs,tp,fp,fn,score=match_events([x['time_sec'] for x in crossings],[x['t_on'] for x in merged],c.cfg['onset_tolerance_sec'])
    c.m['measurements'].update(matches=pairs,TP=tp,FP=fp,FN=fn,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn))
    write_json(c.out/'event_matches.json',{'pairs':pairs,'TP':tp,'FP':fp,'FN':fn,'tolerance_sec':c.cfg['onset_tolerance_sec'],'F1':score})
    c.m['reason']='前端进入、远端露出均有可见证据；灯亮起点独立检测并匹配。全暗情况正常得0分。'
    c.calculation=[f'参考事件 {crossings}',f'亮段 {merged}; matches={pairs}; tolerance=±{c.cfg["onset_tolerance_sec"]} s.',f'TP={tp}, FP={fp}, FN={fn}; F1=2*{tp}/(2*{tp}+{fp}+{fn})={score}.','没有直接测量磁通量。']
    return score


def p41(c):
    c.m['principle']='由可见内腔与自由表面恢复轴对称体积，独立拟合 Q∝h^beta，比较 abs(beta_water-.5)+abs(beta_sand)。'
    c.m['score_details']={'version':'p41_soft_exponent_error_v3','formula':'1 / (1 + (abs(beta_water - 0.5) + abs(beta_sand)) / error_half_score)','range':[0,1],'half_score_error':c.cfg['error_half_score'],'finite_error_cutoff':False,'fitting_policy':'observed positive-flow data; no relative-height window or minimum height ratio','interpretation':'按拟合指数误差平滑扣分；有限误差不截零。分数不是物理成立概率，拟合不确定度单独报告。'}
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
        if not good.all():
            require_track(good,c.t,.1,.9);h=np.interp(c.t,c.t[good],h[good]);vol=np.interp(c.t,c.t[good],vol[good])
        # Actual independent regression windows; no framewise pixel-area derivative.
        q=[];qnoise=[]
        for t in c.t:
            ix=np.abs(c.t-t)<=c.cfg['derivative_window_sec']/2+1e-9;fit=linear_fit(c.t[ix],vol[ix]);q.append(-fit['velocity']);qnoise.append(fit['slope_se'])
        q=np.array(q);qnoise=np.array(qnoise);frac=h/h[0]
        details=fit_discharge_exponent(h,q,qnoise,c.t,c.cfg)
        height_ratio=float(np.max(h)/np.min(h));problem=details['failures']
        if not valid_geometry:problem.append('inner-wall geometry residual too large')
        if problem:details['fit']=None;betas.append(None);errors.extend(name+': '+p for p in problem)
        else:betas.append(details['fit']['velocity'])
        geometry[name]={'axis_x':apex[0],'cone_apex_y':apex[1],'open_outlet_y':outlet_y,'wall_R_of_z_slope':float(slope),'wall_R_of_z_intercept':float(intercept),'observed_wall_points_z_R':wall,'axisymmetric_assumption':True,'volume_formula':'2*pi*integral r*max(z_surface(r)-z_bottom(r),0) dr','valid':valid_geometry}
        surfaces[name]=rows;fit_details[name]={'height_px_initial_final':[float(h[0]),float(h[-1])],'height_ratio':height_ratio,'height_fraction_range':[float(frac.min()),float(frac.max())],**details}
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
    c.calculation=[f'真实测得的高度范围与拟合资格: {fit_details}',f'体积恢复几何: axisymmetric surface/wall integration; cone apex and open outlet treated separately.','未把 -dh/dt 或二维掩码面积变化当作流量。']
    if errors:raise ExtractionError('; '.join(errors))
    err=abs(betas[0]-.5)+abs(betas[1]);score=soft_error_score(err,c.cfg['error_half_score']);c.m['measurements']['raw_m1_error']=err;c.m['reason']='按实际可测排出数据拟合，取消固定高度门槛；指数误差平滑扣分，不再超过阈值直接归零，拟合不确定度单独报告。';c.calculation.append(f'p41_soft_exponent_error_v3: beta_water={betas[0]}, beta_sand={betas[1]}; E={err}; half-score error={c.cfg["error_half_score"]}; score=1/(1+{err}/{c.cfg["error_half_score"]})={score}')
    return score

TASKS={'P37':p37,'P38':p38,'P39':p39,'P41':p41,'P43':p43,'P49':p49}
