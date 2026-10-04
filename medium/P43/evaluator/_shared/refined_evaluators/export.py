"""All exported masks, cutouts and overlays represent actual intermediate data."""
import numpy as np
import cv2
from .media import VideoWriter
from .common import write_json
from .vision import COLORS


def render_geometry(canvas,c,i):
    for name,data in c.annotations.items():
        if 'quads' in data:
            q=np.rint(data['quads'][i]).astype(int);cv2.polylines(canvas,[q],True,(0,255,255),2)
            com=q.mean(0).astype(int);support=q[2];cv2.circle(canvas,tuple(com),5,(0,255,255),-1);cv2.line(canvas,tuple(com),(int(com[0]),int(support[1])),(0,255,255),1)
        if 'trajectory_xy' in data:
            pt=data['trajectory_xy'][i]
            if np.isfinite(pt).all():
                xy=tuple(np.rint(pt).astype(int));cv2.circle(canvas,xy,4,(0,255,255),-1)
                if 'pivot' in data:cv2.line(canvas,tuple(np.rint(data['pivot']).astype(int)),xy,(0,255,255),2)
                if 'radii' in data:cv2.circle(canvas,xy,int(round(data['radii'][i])),(0,255,255),1)
        if 'surface_y' in data:
            y=data['surface_y'][i];x=data['axis_x']
            if np.isfinite(y):cv2.line(canvas,(int(x-35),int(round(y))),(int(x+35),int(round(y))),(0,255,255),2)
    if c.task=='P39':
        for key in ['entry_plane','exit_plane']:
            line=c.a['geometry'][key];cv2.line(canvas,tuple(line[0]),tuple(line[1]),(0,255,255),2)
        x1,y1,x2,y2=c.a['geometry']['lamp_emission_box'];cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,255),1)
        for e in c.annotations.get('events',{}).get('crossings',[]):
            if abs(c.t[i]-e['time_sec'])<=c.cfg['onset_tolerance_sec']:
                cv2.putText(canvas,f"{e['id']} t={e['time_sec']:.3f}s (+/- {c.cfg['onset_tolerance_sec']:.2f}s)",(20,80 if e['id']=='enter' else 110),0,.7,(0,255,255),2)
    return canvas


def export(c):
    diagnostic_plots(c) if c.annotations else None
    h,w=c.frames[0].shape[:2];writers=[];specs=[];crop_data={};artifacts={};sheets=[]
    mask_names={'closed_ring':'closed_mask','open_ring':'open_mask','solid_plate':'solid_mask','slotted_plate':'slotted_mask','large_ball':'large_mask','small_ball':'small_mask'}
    try:
        items=[(o['name'],c.masks[:,j]) for j,o in enumerate(c.a['objects'])]+list(c.extra_masks.items())
        for name,seq in items:
            y,x=np.where(np.any(seq,axis=0));
            if len(x)==0:continue
            pad=12;x1=max(0,int(x.min())-pad);y1=max(0,int(y.min())-pad);x2=min(w,int(x.max())+pad+1);y2=min(h,int(y.max())+pad+1)
            crop_data[name]={'xyxy':[x1,y1,x2,y2],'coordinate_frame':'original source frame','invalid_frames':np.flatnonzero(~np.any(seq,axis=(1,2))).tolist()}
            for kind,suffix in [('cutout',name),('mask',mask_names.get(name,name+'_mask'))]:
                writer=VideoWriter(c.out/(suffix+'.mp4'),x2-x1,y2-y1,c.t);writers.append(writer);specs.append((writer,seq,[x1,y1,x2,y2],kind));artifacts[suffix]=str(writer.path)
        segmentation=VideoWriter(c.out/'segmentation_overlay.mp4',w,h,c.t);tracking=VideoWriter(c.out/'tracking_overlay.mp4',w,h,c.t);writers.extend([segmentation,tracking]);artifacts['segmentation_overlay']=str(segmentation.path);artifacts['tracking_overlay']=str(tracking.path)
        task_overlay=None
        overlay_names={'P39':'event_overlay','P41':'surface_tracking','P43':'geometry_overlay','P49':'circle_fit_overlay'}
        if c.task in overlay_names:
            task_overlay=VideoWriter(c.out/(overlay_names[c.task]+'.mp4'),w,h,c.t);writers.append(task_overlay);artifacts[overlay_names[c.task]]=str(task_overlay.path)
        picks=set(np.linspace(0,len(c.t)-1,12).round().astype(int))
        for i,(frame,t) in enumerate(zip(c.frames,c.t)):
            for writer,seq,(x1,y1,x2,y2),kind in specs:
                m=seq[i,y1:y2,x1:x2];image=np.repeat((m*255).astype('uint8')[:,:,None],3,2) if kind=='mask' else np.where(m[:,:,None],frame[y1:y2,x1:x2],0).astype('uint8');writer.write(image,t)
            seg=frame.copy()
            for j,o in enumerate(c.a['objects']):
                mask=c.masks[i,j];color=COLORS[j%len(COLORS)];seg[mask]=(seg[mask]*.72+np.array(color)*.28).astype('uint8');cs,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(seg,cs,-1,color,1)
                y,x=np.where(mask)
                if len(x):cv2.putText(seg,o['name'],(int(x.min()),max(25,int(y.min())-8)),0,.6,color,2)
            cv2.putText(seg,f'{c.task} frame={i} PTS={t:.3f}s',(20,30),0,.7,(255,255,255),2);segmentation.write(seg,t)
            tracked=seg.copy()
            if c.xy is not None:
                for j,(name,ids) in enumerate(c.groups.items()):
                    for k in ids:
                        if c.vis[i,k] and np.isfinite(c.xy[i,k]).all():cv2.circle(tracked,tuple(np.rint(c.xy[i,k]).astype(int)),2,COLORS[j%len(COLORS)],-1)
            tracked=render_geometry(tracked,c,i);tracking.write(tracked,t)
            if task_overlay:task_overlay.write(tracked,t)
            if i in picks:
                sheets.append(cv2.resize(tracked,(504,288)));cv2.imwrite(str(c.out/f'keyframe_{i:04d}.png'),tracked)
        if len(sheets)==12:cv2.imwrite(str(c.out/'review_contact_sheet.jpg'),np.vstack([np.hstack(sheets[k:k+3]) for k in range(0,12,3)]))
    finally:
        for writer in writers:writer.close()
    write_json(c.out/'crop_coordinates.json',{'source_size_wh':[w,h],'source_frame_count':len(c.t),'objects':crop_data})
    if c.xy is not None:write_json(c.out/'tracks.json',{'backend':'CoTracker3 scaled_offline','groups':c.groups,'source_frame_index':list(range(len(c.t))),'pts_sec':c.t,'tracks_xy':c.xy,'visibility':c.vis,'note':'Invisible model predictions are preserved for debugging but excluded from quantitative observations.'})
    return artifacts


def diagnostic_plots(c):
    """Plots include the actual score-defining extrema/windows, not just labels."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json
    needed={'P37':['heights.csv','peaks.json'],'P38':['angles.csv','peaks.json'],'P39':['motion_brightness.csv','brightness_calibration.json'],'P49':['position_velocity.csv','terminal_windows.json'],'P41':['height_volume_flow.csv'],'P43':['pose_events.csv']}
    if any(not (c.out/name).exists() for name in needed.get(c.task,[])):return
    if c.task=='P37':
        data=np.genfromtxt(c.out/'heights.csv',delimiter=',',names=True);peaks=json.loads((c.out/'peaks.json').read_text());fig,ax=plt.subplots(figsize=(10,4))
        for name in ['closed_ring','open_ring']:
            ax.plot(c.t,data[name+'_raw_diameter'],alpha=.3);line=ax.plot(c.t,data[name+'_smooth_diameter'],label=name)[0];p=peaks[name];ax.scatter([p['time_sec']],[p['height']],color=line.get_color());ax.axvspan(*p['plateau_window_sec'],color=line.get_color(),alpha=.08)
        ax.set(xlabel='Source PTS (s)',ylabel='Height / initial outer diameter');ax.legend();name='heights.png'
    elif c.task=='P38':
        data=np.genfromtxt(c.out/'angles.csv',delimiter=',',names=True);peaks=json.loads((c.out/'peaks.json').read_text());fig,ax=plt.subplots(figsize=(10,4))
        for obj in c.a['objects']:
            key=obj['name'];line=ax.plot(c.t,data[key+'_smoothed_deg'],label=key)[0]
            for p in peaks[key]:ax.scatter(p['time_sec'],p['angle_deg'],marker='o' if p['accepted'] else 'x',s=25,color=line.get_color())
        cut=c.m['measurements'].get('amplitude_cutoff_deg')
        if cut:
            for y in [-cut,cut]:ax.axhline(y,color='grey',linestyle='--')
        ax.set(xlabel='Source PTS (s)',ylabel='Signed angle (deg)');ax.legend();name='angles_peaks.png'
    elif c.task=='P39':
        data=np.genfromtxt(c.out/'motion_brightness.csv',delimiter=',',names=True);cal=json.loads((c.out/'brightness_calibration.json').read_text());fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
        axes[0].plot(c.t,data['magnet_center_x'],label='observed magnet center');axes[0].set_ylabel('Magnet center x (px)')
        for key in ['entry_plane','exit_plane']:axes[0].axhline(np.mean(np.array(c.a['geometry'][key])[:,0]),label=key,linestyle='--')
        axes[1].plot(c.t,data['lamp_corrected'],label='lamp minus background');axes[1].axhline(cal['baseline']+3*cal['noise'],label='baseline plus 3 noise',color='orange');axes[1].set_ylabel('Corrected lamp intensity')
        for a in axes:
            for key in ['emission_time_centroid_sec','reference_activity_time_centroid_sec']:
                value=c.m['measurements'].get(key)
                if value is not None:a.axvline(value,label=key,linestyle='--')
            a.legend()
        axes[-1].set_xlabel('Source PTS (s)');name='motion_brightness.png'
    elif c.task=='P49':
        data=np.genfromtxt(c.out/'position_velocity.csv',delimiter=',',names=True);windows=json.loads((c.out/'terminal_windows.json').read_text());fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for key in ['large_ball','small_ball']:
            for ax,suffix in zip(axes,['_y_px','_speed_px_s','_radius_px']):
                line=ax.plot(c.t,data[key+suffix],label=key)[0];win=windows[key]['chosen']
                if win:ax.axvspan(*win['window_sec'],color=line.get_color(),alpha=.12)
                ax.legend()
        for ax,label in zip(axes,['Downward centre y (px)','Local speed (px/s)','Contour radius (px)']):ax.set_ylabel(label)
        axes[-1].set_xlabel('Source PTS (s)');name='position_velocity.png'
    elif c.task=='P41':
        data=np.genfromtxt(c.out/'height_volume_flow.csv',delimiter=',',names=True);fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for key in ['water','sand']:
            for ax,suffix in zip(axes,['_height_px','_volume_px3','_flow_px3_s']):ax.plot(c.t,data[key+suffix],label=key);ax.legend()
        for ax,label in zip(axes,['Head above open outlet (px)','Reconstructed volume (px^3)','Volume outflow (px^3/s)']):ax.set_ylabel(label)
        axes[-1].set_xlabel('Source PTS (s)');name='height_volume_flow.png'
    elif c.task=='P43':
        data=np.genfromtxt(c.out/'pose_events.csv',delimiter=',',names=True);fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        axes[0].plot(c.t,data['theta_deg']);axes[0].set_ylabel('Observed angle (deg)')
        for key in ['pivot_drift_px','contact_gap_px']:axes[1].plot(c.t,data[key],label=key)
        axes[1].set_ylabel('Support geometry (px)');axes[1].legend()
        axes[2].plot(c.t,data['rigid_shape_relative_change']);axes[2].set_ylabel('Relative edge length change')
        axes[-1].set_xlabel('Source PTS (s)');name='pose_events.png'
    else:return
    fig.tight_layout();fig.savefig(c.out/name,dpi=150);plt.close(fig)
