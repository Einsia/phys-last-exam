#!/usr/bin/env python3
"""Independent P36 magnetic-braking evaluator with optical-flow visualization."""
from __future__ import annotations
import argparse,csv,json,math,os,re,sys
from pathlib import Path
from typing import Any
import cv2,numpy as np
TASK_ID="P36"; TASK_ROOT=Path(__file__).resolve().parents[1]; os.environ.setdefault("MPLCONFIGDIR","/tmp/vdmbench-matplotlib")
def finite(v:Any)->Any:
    if isinstance(v,dict):return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [finite(x) for x in v]
    if isinstance(v,np.ndarray):return finite(v.tolist())
    if isinstance(v,(np.floating,np.integer,np.bool_)):return finite(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def unit_score(value:float)->float:
    value=float(value)
    if not math.isfinite(value):raise ValueError("physics score must be finite")
    return float(np.clip(value,0.0,1.0))
def recognized_score(physics_score:float)->float:
    return 0.15+0.85*unit_score(physics_score)
def rel(v):
    if v is None:return None
    try:return Path(v).resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:return str(v)
def metadata(video,image,seed):
    stem=Path(video).stem;m=re.search(r"_seed(\d+)$",stem,re.I);seed=seed if seed is not None else(int(m.group(1)) if m else None);base=re.sub(r"_seed\d+$","",stem,flags=re.I)
    if image is None:
        p=TASK_ROOT/("first_frames/simulation/sim_first_frame.png" if base.lower().startswith("sim") else "first_frames/provided/first_frame.png")
        if p.is_file():image=str(p)
    return rel(video),rel(image),seed
def read_video(path,limit=None):
    c=cv2.VideoCapture(path)
    if not c.isOpened():raise RuntimeError(f"cannot open video: {path}")
    fps=float(c.get(cv2.CAP_PROP_FPS) or 24);a=[]
    while True:
        ok,b=c.read()
        if not ok:break
        a.append(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
        if limit and len(a)>=limit:break
    c.release()
    if not a:raise RuntimeError("video contains no frames")
    return np.stack(a),fps
def load_rois():
    path=TASK_ROOT/"evaluator"/"roi.json"
    if not path.is_file():return {},"auto"
    try:
        raw=json.loads(path.read_text());raw=raw.get("rois",raw);out={}
        for k,v in raw.items():
            if k in {"task_id","frame_size","source_video","source_frame"}:continue
            if isinstance(v,(list,tuple)) and len(v)==4:
                b=tuple(float(x) for x in v)
                if all(0<=x<=1 for x in b) and b[2]>0 and b[3]>0 and b[0]+b[2]<=1 and b[1]+b[3]<=1:out[k]=b
        return out,"annotated"
    except (OSError,ValueError,TypeError):return {},"auto_invalid"
def roi_pixels(box,w,h):
    if box is None:return None
    x,y,ww,hh=box;return max(0,int(x*w)),max(0,int(y*h)),min(w,int((x+ww)*w)),min(h,int((y+hh)*h))
def components(mask,h,w):
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask,8);out=[]
    for i in range(1,n):
        x,y,ww,hh,area=stats[i]
        if area<80 or area>.06*h*w or ww<10 or hh<8 or ww>180 or hh>140:continue
        if y<.05*h or y>.55*h:continue
        out.append({"x":int(x),"y":int(y),"w":int(ww),"h":int(hh),"area":int(area),"cx":float(cent[i][0]),"cy":float(cent[i][1])})
    return out
def find_pair(frame, roi=None):
    h,w=frame.shape[:2];hsv=cv2.cvtColor(frame,cv2.COLOR_RGB2HSV);s,v=hsv[:,:,1],hsv[:,:,2]
    white=((s<70)&(v>175)).astype(np.uint8)*255; gray=((s<80)&(v>55)&(v<190)).astype(np.uint8)*255
    if roi:
        hbox=roi_pixels(roi.get("magnet"),w,h); cbox=roi_pixels(roi.get("control"),w,h)
        if hbox:
            keep=np.zeros_like(gray);keep[hbox[1]:hbox[3],hbox[0]:hbox[2]]=255;gray=cv2.bitwise_and(gray,keep)
        if cbox:
            keep=np.zeros_like(white);keep[cbox[1]:cbox[3],cbox[0]:cbox[2]]=255;white=cv2.bitwise_and(white,keep)
    k=np.ones((5,5),np.uint8);white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,k);gray=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,k)
    wc,gc=components(white,h,w),components(gray,h,w); pairs=[]
    for wi in wc:
        for gi in gc:
            dx=abs(wi["cx"]-gi["cx"]);dy=abs(wi["cy"]-gi["cy"])
            if .025*w<=dx<=.22*w and dy<=.10*h:
                score=(wi["area"]+gi["area"])*(1-abs(dy)/(.10*h+1))*max(.1,1-abs(dx-.09*w)/(.18*w))
                pairs.append((score,wi,gi))
    if not pairs:return None,None,{"white_candidates":wc,"gray_candidates":gc}
    _,white_box,gray_box=max(pairs,key=lambda z:z[0]);return gray_box,white_box,{"white_candidates":wc,"gray_candidates":gc}
def track(frames,box,start):
    gray=[cv2.cvtColor(f,cv2.COLOR_RGB2GRAY) for f in frames];x,y,w,h=box; x0=max(0,x-4);y0=max(0,y-4);x1=min(frames.shape[2],x+w+4);y1=min(frames.shape[1],y+h+4)
    mask=np.zeros_like(gray[start],dtype=np.uint8); mask[y0:y1,x0:x1]=255
    pts=cv2.goodFeaturesToTrack(gray[start],mask=mask,maxCorners=120,qualityLevel=.001,minDistance=2,blockSize=3)
    if pts is not None:pts=pts.reshape(-1,2)
    if pts is None or len(pts)<3:return np.full((len(frames),2),np.nan)
    tr=np.full((len(frames),2),np.nan);tr[:start]=np.median(pts,axis=0);tr[start]=np.median(pts,axis=0);prev=pts.reshape(-1,1,2)
    for i in range(start+1,len(frames)):
        nxt,st,err=cv2.calcOpticalFlowPyrLK(gray[i-1],gray[i],prev,None,winSize=(31,31),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if nxt is None:break
        good=(st.reshape(-1)>0)&(err.reshape(-1)<40)&(nxt[:,0,0]>=0)&(nxt[:,0,0]<frames.shape[2])&(nxt[:,0,1]>=0)&(nxt[:,0,1]<frames.shape[1]);
        if good.sum()<3:break
        prev=nxt[good];tr[i]=np.median(prev[:,0,:],axis=0)
    valid=np.isfinite(tr[:,0]);
    if valid.sum()<.5*len(frames):return tr
    for d in range(2):tr[:,d]=np.interp(np.arange(len(frames)),np.flatnonzero(valid),tr[valid,d])
    return tr
def displacement(tr):
    p=tr[0];return np.linalg.norm(tr-p,axis=1)
def event_and_speed(d,fps):
    n=len(d);base=float(np.median(d[:max(5,n//10)]));total=max(float(np.percentile(d[-max(5,n//10):],50)-base),1.0); threshold=base+.10*total; hit=np.flatnonzero(d>=threshold); onset=int(hit[0]) if len(hit) else None
    if onset is None:return None,None,total,None,0
    end=max(onset+3,n-2);sl=slice(onset,end);times=np.arange(onset,end)/fps;coef,intercept=np.polyfit(times,d[sl],1);pred=coef*times+intercept;rmse=float(np.sqrt(np.mean((d[sl]-pred)**2)));return onset,max(float(coef),1e-6),total,rmse,int(end-onset)
def extract(frames,fps):
    idx=min(15,len(frames)-1); rois,roi_source=load_rois(); hh,ww=frames.shape[1:3]; rois={"board":(.05,.05,.75,.85),"magnet":(.08,.08,.42,.52),"control":(.14,.08,.42,.52),**rois}
    mag,ctrl,cand=find_pair(frames[idx],rois);debug={"initialization_frame":idx,"roi_source":roi_source,"roi":rois,"magnet_candidate":mag,"control_candidate":ctrl,**cand}
    if mag is None or ctrl is None:return None,debug,"P36_object_pair_missing"
    mt=track(frames,(mag["x"],mag["y"],mag["w"],mag["h"]),idx);ct=track(frames,(ctrl["x"],ctrl["y"],ctrl["w"],ctrl["h"]),idx);md=displacement(mt);cd=displacement(ct);mo,mv,mdist,merr,mcount=event_and_speed(md,fps);co,cv,cdist,cerr,ccount=event_and_speed(cd,fps);debug.update({"magnet_track":mt,"control_track":ct,"magnet_displacement_px":md,"control_displacement_px":cd,"motion_fit":{"magnet":{"rmse_px":merr,"sample_count":mcount},"control":{"rmse_px":cerr,"sample_count":ccount}}})
    if mo is None or co is None:return None,debug,"P36_motion_event_missing"
    m={"magnet_onset_frame":mo,"control_onset_frame":co,"magnet_onset_time_s":mo/fps,"control_onset_time_s":co/fps,"magnet_speed_px_s":mv,"control_speed_px_s":cv,"magnet_total_displacement_px":mdist,"control_total_displacement_px":cdist,"magnet_speed_fit_rmse_px":merr,"control_speed_fit_rmse_px":cerr,"magnet_speed_fit_sample_count":mcount,"control_speed_fit_sample_count":ccount,"time_ratio":(mo+1)/(co+1),"speed_ratio":mv/max(cv,1e-6),"track_coverage":float(min(np.isfinite(mt[:,0]).mean(),np.isfinite(ct[:,0]).mean()))}
    return m,debug,None
def artifacts(frames,d,m,root,sample,fps):
    folder=root/sample;folder.mkdir(parents=True,exist_ok=True);raw=folder/"measurements.json";raw.write_text(json.dumps(finite({"measurements":m or {},"detection":d}),indent=2));import matplotlib.pyplot as plt
    plot=folder/"plot.png";fig,ax=plt.subplots(figsize=(8,4));ax.plot(d.get("magnet_displacement_px",[]),label="magnet displacement");ax.plot(d.get("control_displacement_px",[]),label="control displacement");
    if m:
        ax.axvline(m.get("magnet_onset_frame"),color="tab:blue",linestyle="--",label="magnet onset"); ax.axvline(m.get("control_onset_frame"),color="tab:orange",linestyle="--",label="control onset")
    ax.set_xlabel("frame");ax.set_ylabel("displacement (px)");ax.legend();fig.tight_layout();fig.savefig(plot,dpi=120);plt.close(fig)
    ov=folder/"overlay.mp4";w=cv2.VideoWriter(str(ov),cv2.VideoWriter_fourcc(*"mp4v"),fps or 24,(frames.shape[2],frames.shape[1]));
    if not w.isOpened():return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":None}
    mt,ct=d.get("magnet_track",[]),d.get("control_track",[])
    for i,f in enumerate(frames):
        b=cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
        for name, color in (("board", (255, 180, 0)), ("magnet", (220, 60, 40)), ("control", (40, 60, 220))):
            box = roi_pixels(d.get("roi", {}).get(name), b.shape[1], b.shape[0])
            if box:
                x0, y0, x1, y1 = box
                cv2.rectangle(b, (x0, y0), (x1, y1), color, 1)
                cv2.putText(b, name, (x0 + 3, max(18, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        for tr,col,name in ((mt,(220,60,40),"mag"),(ct,(40,60,220),"ctrl")):
            if i<len(tr) and np.isfinite(tr[i][0]):cv2.circle(b,(int(tr[i][0]),int(tr[i][1])),9,col,2);cv2.putText(b,name,(int(tr[i][0])+8,int(tr[i][1])),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2)
        if m and i == m.get("magnet_onset_frame"): cv2.putText(b,"magnet onset",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,60,40),2)
        if m and i == m.get("control_onset_frame"): cv2.putText(b,"control onset",(20,62),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,60,220),2)
        w.write(b)
    w.release();return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":str(ov)}
def payload(video,image,seed,model,m,d,reason,sample):
    vp,ip,sd=metadata(video,image,seed);ok=m is not None and reason is None;warn=[]
    if m and m["time_ratio"]<=1:warn.append("P36_magnet_onset_not_later")
    if m and m["speed_ratio"]>=1:warn.append("P36_magnet_not_slower")
    q1=unit_score(1.0-1.0/max(m["time_ratio"],1e-9)) if ok else None;q2=unit_score(1.0-m["speed_ratio"]) if ok else None;m1=recognized_score(q1) if q1 is not None else None;m2=recognized_score(q2) if q2 is not None else None
    if m:m["score_normalization"]={"score_range":[0.0,1.0],"higher_is_better":True,"M1":{"raw_measurement":"time_ratio","physics_score":q1,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 * clip(1 - 1 / time_ratio, 0, 1)","score":m1},"M2":{"raw_measurement":"speed_ratio","physics_score":q2,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 * clip(1 - speed_ratio, 0, 1)","score":m2}}
    return finite({"task_id":TASK_ID,"video_path":vp,"image_path":ip,"seed":sd,"model":model or "unknown","metrics":{"M1":{"extract_success":ok,"metric":m1},"M2":{"extract_success":ok,"metric":m2},"M3":None},"verbose":{"sample_id":sample,"failure_reason":reason,"quality_warnings":warn,"measurements":m or {},"debug":d}})
def evaluate(video,out,sample,image,seed,model,limit=None):
    try:f,fps=read_video(video,limit);m,d,r=extract(f,fps);root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug";dbg=artifacts(f,d,m,root,sample,fps);p=payload(video,image,seed,model,m,dbg,r,sample);rc=0 if r is None else 1
    except Exception as e:
        root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug";folder=root/sample;folder.mkdir(parents=True,exist_ok=True);ep=folder/"error.txt";ep.write_text(f"{type(e).__name__}: {e}\n");p=payload(video,image,seed,model,None,{"directory":str(folder),"error":str(ep)},f"{type(e).__name__}: {e}",sample);rc=2
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False));return rc
def main():
    a=argparse.ArgumentParser();a.add_argument("--video");a.add_argument("--videos");a.add_argument("--output");a.add_argument("--outdir");a.add_argument("--task_id",default=TASK_ID);a.add_argument("--model",default="unknown");a.add_argument("--image-path","--image_path",dest="image");a.add_argument("--seed",type=int);a.add_argument("--sample-id");a.add_argument("--max-frames",type=int);x=a.parse_args()
    if x.task_id.upper()!=TASK_ID:a.error(f"this evaluator only supports {TASK_ID}")
    if x.video:
        if not x.output:a.error("--video requires --output")
        return evaluate(x.video,Path(x.output),x.sample_id or Path(x.video).stem,x.image,x.seed,x.model,x.max_frames)
    src=Path(x.videos or ".");fs=[src] if src.is_file() else sorted(src.glob("*.mp4"));out=Path(x.outdir or "eval_results");jd=out/"json";jd.mkdir(parents=True,exist_ok=True)
    if not fs:return 1
    for i,p in enumerate(fs,1):print(f"[{i}/{len(fs)}] {p.stem} {'OK' if evaluate(str(p),jd/(p.stem+'.json'),p.stem,x.image,x.seed,x.model,x.max_frames)==0 else 'CHECK'}")
    fields=["task_id","video_path","image_path","seed","model","M1_extract_success","M1","M2_extract_success","M2","M3","failure_reason"]
    with(out/"results.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader()
        for p in sorted(jd.glob("*.json")):
            q=json.loads(p.read_text());w.writerow({"task_id":q["task_id"],"video_path":q["video_path"],"image_path":q["image_path"],"seed":q["seed"],"model":q["model"],"M1_extract_success":q["metrics"]["M1"]["extract_success"],"M1":q["metrics"]["M1"]["metric"],"M2_extract_success":q["metrics"]["M2"]["extract_success"],"M2":q["metrics"]["M2"]["metric"],"M3":None,"failure_reason":q["verbose"]["failure_reason"]})
    return 0
if __name__=="__main__":raise SystemExit(main())
