#!/usr/bin/env python3
"""Independent P44 catenary evaluator with skeleton and fit visualization."""
from __future__ import annotations
import argparse,csv,json,math,os,re,sys
from pathlib import Path
from typing import Any
import cv2,numpy as np
TASK_ID="P44"; TASK_ROOT=Path(__file__).resolve().parents[1]; os.environ.setdefault("MPLCONFIGDIR","/tmp/vdmbench-matplotlib")
def finite(v:Any)->Any:
    if isinstance(v,dict):return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [finite(x) for x in v]
    if isinstance(v,np.ndarray):return finite(v.tolist())
    if isinstance(v,(np.floating,np.integer,np.bool_)):return finite(v.item())
    if isinstance(v,float) and not math.isfinite(v):return None
    return v
def residual_physics_score(error:float,scale:float=1.0)->float:
    """V2 residual normalization q(e; a) = 1 / (1 + |e| / a)."""
    error,scale=float(error),float(scale)
    if not math.isfinite(error) or not math.isfinite(scale) or scale<=0.0:raise ValueError("residual error must be finite and scale must be finite and positive")
    return 1.0/(1.0+abs(error)/scale)
def recognized_score(physics_score:float)->float:
    return 0.15+0.85*float(np.clip(physics_score,0.0,1.0))
def rel(v):
    if v is None:return None
    try:return Path(v).resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:return str(v)
def metadata(video,image,seed):
    stem=Path(video).stem;m=re.search(r"_seed(\d+)$",stem,re.I);seed=seed if seed is not None else(int(m.group(1)) if m else None);base=re.sub(r"_seed\d+$","",stem,flags=re.I)
    if image is None:
        p=TASK_ROOT/("sim_first_frame.png" if base.lower().startswith("sim") else "first_frame.png")
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
def contrast_mask(crop):
    """Foreground evidence relative to the observed local background, any hue."""
    if not crop.size:return np.zeros(crop.shape[:2],np.uint8),0.0
    pixels=crop.astype(np.float32)
    background=np.median(pixels.reshape(-1,3),axis=0)
    distance=np.linalg.norm(pixels-background,axis=2)
    noise=float(np.median(distance));mad=float(np.median(np.abs(distance-noise)))
    threshold=max(12.0,noise+6.0*max(mad,1.0))
    mask=(distance>threshold).astype(np.uint8)
    return mask,threshold


def skeleton(frame,chain_box=None):
    h,w=frame.shape[:2];ys=np.full(w,np.nan)
    xlo,ylo,xhi,yhi=chain_box or (int(.08*w),int(.10*h),int(.92*w),int(.88*h))
    crop=frame[ylo:yhi,xlo:xhi];mask,threshold=contrast_mask(crop)
    if not mask.size:return ys,threshold
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    candidates=[]
    for label in range(1,count):
        x,y,bw,bh,area=stats[label]
        # Require a long, thin connected object. A filled background/ROI edge
        # is not an observed chain. Selection never uses catenary fit quality.
        if bw<.45*(xhi-xlo) or area>.35*mask.size:continue
        ribbon=area/max(bw,1)
        if ribbon>max(18.0,.12*(yhi-ylo)):continue
        columns=np.flatnonzero(np.any(labels==label,axis=0))
        if len(columns)<.75*bw:continue
        candidates.append((bw,label,columns))
    if not candidates:return ys,threshold
    _,label,columns=max(candidates,key=lambda candidate:candidate[0])
    for x in columns:
        hit=np.flatnonzero(labels[:,x]==label)
        ys[xlo+x]=ylo+float(np.median(hit))
    return ys,threshold


def support_point(frame, box, threshold):
    """Locate a compact support by foreground thickness, independent of hue."""
    if box is None:return None
    x0,y0,x1,y1=box;crop=frame[y0:y1,x0:x1]
    mask,_=contrast_mask(crop)
    if not mask.size or mask.sum()<5 or mask.mean()>.60:return None
    distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    peak=float(distance.max())
    if peak<2.0:return None
    ys,xs=np.where(distance>=.8*peak)
    return {"x":float(x0+np.median(xs)),"y":float(y0+np.median(ys)),"confidence":float(min(1.,peak/5.))}
def fit_catenary(ys):
    valid=np.isfinite(ys);x=np.flatnonzero(valid).astype(float)
    if len(x)<30:return None
    # Ignore isolated highlights and retain the central connected span.
    gaps=np.flatnonzero(np.diff(x)>8); starts=np.r_[0,gaps+1];ends=np.r_[gaps+1,len(x)];j=int(np.argmax(x[ends-1]-x[starts]));x=x[starts[j]:ends[j]];y=ys[x.astype(int)]
    if len(x)<30:return None
    best=None;h=max(float(np.ptp(y)),1.)
    for sign in (-1.,1.):
        for a in np.geomspace(max(8.,.02*len(ys)),max(20.,1.8*len(ys)),30):
            for x0 in np.linspace(float(x.min()+.15*(x.max()-x.min())),float(x.max()-.15*(x.max()-x.min())),11):
                z=sign*a*np.cosh((x-x0)/a);c=float(np.median(y-z));pred=c+z;rmse=float(np.sqrt(np.mean((y-pred)**2)))
                if best is None or rmse<best[0]:best=(rmse,sign,a,x0,c,pred,x,y)
    if best is None:return None
    rmse,sign,a,x0,c,pred,x,y=best
    edge_count=max(3,int(.03*len(y))); observed_error=float(abs(np.median(y[:edge_count])-np.median(y[-edge_count:])))
    return {"rmse_px":rmse,"a_px":a,"x0_px":x0,"c_px":c,"sign":sign,"x_points_px":x,"y_points_px":y,"fit_points_px":pred,"x_min_px":float(x.min()),"x_max_px":float(x.max()),"endpoint_height_error_px":observed_error}
def extract(frames):
    rois,roi_source=load_rois(); h,w=frames.shape[1:3]; rois={"chain":(.07,.18,.86,.72),"left_support":(.07,.15,.12,.20),"right_support":(.82,.15,.12,.20),**rois}; chain_box=roi_pixels(rois["chain"],w,h); support_boxes={side:roi_pixels(rois.get(side+"_support"),w,h) for side in ("left","right")}; ys=[];ths=[];fits=[];supports={"left":[],"right":[]}
    tail=max(5,min(15,len(frames)//8))
    for i,f in enumerate(frames):
        y,t=skeleton(f,chain_box);ys.append(y);ths.append(t)
        for side in ("left","right"): supports[side].append(support_point(f,support_boxes[side],t))
        fits.append(fit_catenary(y) if (i % 4 == 0 or i >= len(frames)-tail) else None)
    n=len(frames);valid=[f for f in fits[-tail:] if f]
    if not valid:return None,{"skeleton_y_px":ys,"threshold":ths,"fits":fits,"support_points":supports},"P44_chain_skeleton_missing"
    # Fit the temporal median skeleton over the stable tail for a less noisy final measurement.
    arr=np.stack(ys[-tail:]); valid_count=np.isfinite(arr).sum(axis=0); med=np.full(arr.shape[1],np.nan)
    cols=valid_count>0
    if np.any(cols): med[cols]=np.nanmedian(arr[:,cols],axis=0)
    med[valid_count<max(2,tail//3)]=np.nan; final=fit_catenary(med)
    if final is None:final=max(valid,key=lambda z:z["x_points_px"].size)
    support_pairs=[]
    for left,right in zip(supports["left"][-tail:],supports["right"][-tail:]):
        if left and right: support_pairs.append(abs(left["y"]-right["y"]))
    support_error=float(np.median(support_pairs)) if len(support_pairs)>=3 else None
    support_coverage=float(len(support_pairs)/max(tail,1))
    endpoint_error=float(support_error if support_error is not None else final["endpoint_height_error_px"])
    h,w=frames.shape[1:3];m={"stable_tail_frames":tail,"skeleton_coverage":float(np.isfinite(med).mean()),"fit_rmse_px":float(final["rmse_px"]),"fit_rmse_norm":float(final["rmse_px"]/h),"catenary_a_px":float(final["a_px"]),"catenary_x0_px":float(final["x0_px"]),"catenary_c_px":float(final["c_px"]),"catenary_endpoint_height_error_px":float(final["endpoint_height_error_px"]),"support_height_error_px":support_error,"support_track_coverage":support_coverage,"endpoint_height_error_px":endpoint_error,"endpoint_height_error_norm":float(endpoint_error/h),"fit_x_min_px":final["x_min_px"],"fit_x_max_px":final["x_max_px"]}
    # Normalize by observed chain geometry, not empty image padding.
    shape_scale=max(float(np.ptp(final["y_points_px"])),3.0)
    span=max(float(final["x_max_px"]-final["x_min_px"]),1.0)
    m.update(chain_sag_px=shape_scale,chain_span_px=span,
             fit_rmse_relative_sag=float(final["rmse_px"]/shape_scale),
             support_height_error_relative_span=float(endpoint_error/span))
    d={"roi_source":roi_source,"roi":rois,"skeleton_y_px":ys,"threshold":ths,"fits":fits,"final_fit":final,"support_points":supports};return m,d,None
def artifacts(frames,d,m,root,sample,fps):
    folder=root/sample;folder.mkdir(parents=True,exist_ok=True);raw=folder/"measurements.json";raw.write_text(json.dumps(finite({"measurements":m or {},"detection":d}),indent=2));import matplotlib.pyplot as plt
    plot=folder/"plot.png";fig,ax=plt.subplots(2,1,figsize=(8,6));rm=[f["rmse_px"] if f else np.nan for f in d["fits"]];ep=[f["endpoint_height_error_px"] if f else np.nan for f in d["fits"]];ax[0].plot(rm,label="catenary RMSE (px)");ax[1].plot(ep,label="catenary endpoint error (px)");sp=d.get("support_points",{});support_err=[abs(a["y"]-b["y"]) if a and b else np.nan for a,b in zip(sp.get("left",[]),sp.get("right",[]))];ax[1].plot(support_err,label="support point height error (px)");ax[0].legend();ax[1].legend();ax[1].set_xlabel("frame");fig.suptitle(sample);fig.tight_layout();fig.savefig(plot,dpi=120);plt.close(fig)
    ov=folder/"overlay.mp4";w=cv2.VideoWriter(str(ov),cv2.VideoWriter_fourcc(*"mp4v"),fps or 24,(frames.shape[2],frames.shape[1]));
    if not w.isOpened():return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":None}
    for i,f in enumerate(frames):
        b=cv2.cvtColor(f,cv2.COLOR_RGB2BGR); y=d["skeleton_y_px"][i]
        for key,color in (("chain",(255,180,0)),("left_support",(0,220,0)),("right_support",(0,0,220))):
            if key in d.get("roi",{}):
                box=roi_pixels(tuple(d["roi"][key]),b.shape[1],b.shape[0]); cv2.rectangle(b,(box[0],box[1]),(box[2],box[3]),color,1)
        for x in np.flatnonzero(np.isfinite(y))[::3]:cv2.circle(b,(int(x),int(y[x])),1,(0,220,0),-1)
        for side,color in (("left",(0,220,255)),("right",(255,80,0))):
            point=d.get("support_points",{}).get(side,[])[i] if i<len(d.get("support_points",{}).get(side,[])) else None
            if point: cv2.circle(b,(int(point["x"]),int(point["y"])),5,color,-1); cv2.putText(b,side+" support",(int(point["x"])+5,int(point["y"])),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        fit=d["fits"][i]
        if fit:
            xx=np.linspace(fit["x_min_px"],fit["x_max_px"],200);yy=fit["c_px"]+fit["sign"]*fit["a_px"]*np.cosh((xx-fit["x0_px"])/fit["a_px"])
            pts=np.column_stack([xx,yy]).astype(np.int32).reshape(-1,1,2);cv2.polylines(b,[pts],False,(0,0,230),2)
        w.write(b)
    w.release();return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":str(ov)}
def payload(video,image,seed,model,m,d,reason,sample):
    vp,ip,sd=metadata(video,image,seed);ok=m is not None and reason is None;warn=[]
    if m and m["fit_rmse_norm"]>.05:warn.append("P44_catenary_fit_residual_high")
    if m and m["endpoint_height_error_norm"]>.05:warn.append("P44_support_heights_not_equal")
    if m and m.get("support_track_coverage",0.0)<.50:warn.append("P44_support_points_low_confidence")
    q1=residual_physics_score(m["fit_rmse_relative_sag"],.10) if ok else None;q2=residual_physics_score(m["support_height_error_relative_span"],.02) if ok else None;m1=recognized_score(q1) if q1 is not None else None;m2=recognized_score(q2) if q2 is not None else None
    if m:m["score_normalization"]={"score_range":[0.0,1.0],"higher_is_better":True,"M1":{"raw_measurement":"fit_rmse_relative_sag","scale":0.10,"physics_score":q1,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 / (1 + abs(fit_rmse_relative_sag)/0.10)","score":m1},"M2":{"raw_measurement":"support_height_error_relative_span","scale":0.02,"physics_score":q2,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 / (1 + abs(support_height_error_relative_span)/0.02)","score":m2}}
    return finite({"task_id":TASK_ID,"video_path":vp,"image_path":ip,"seed":sd,"model":model or "unknown","metrics":{"M1":{"extract_success":ok,"metric":m1},"M2":{"extract_success":ok,"metric":m2},"M3":None},"verbose":{"sample_id":sample,"failure_reason":reason,"quality_warnings":warn,"measurements":m or {},"debug":d}})
def evaluate(video,out,sample,image,seed,model,limit=None):
    try:f,fps=read_video(video,limit);m,d,r=extract(f);root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug";dbg=artifacts(f,d,m,root,sample,fps);p=payload(video,image,seed,model,m,dbg,r,sample);rc=0 if r is None else 1
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
