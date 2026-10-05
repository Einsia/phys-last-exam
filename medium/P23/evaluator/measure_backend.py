#!/usr/bin/env python3
"""Independent P23 freezing-water evaluator."""
from __future__ import annotations
import argparse,csv,json,math,os,re,sys
from pathlib import Path
from typing import Any
import cv2,numpy as np
TASK_ID="P23"; TASK_ROOT=Path(__file__).resolve().parents[1]; RHO_RATIO=1000.0/917.0; os.environ.setdefault("MPLCONFIGDIR","/tmp/vdmbench-matplotlib")
def finite(v:Any)->Any:
    if isinstance(v,dict): return {str(k):finite(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [finite(x) for x in v]
    if isinstance(v,np.ndarray): return finite(v.tolist())
    if isinstance(v,(np.floating,np.integer,np.bool_)): return finite(v.item())
    if isinstance(v,float) and not math.isfinite(v): return None
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
    p=Path(v)
    try:return p.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError:return str(v)
def metadata(video,image,seed):
    stem=Path(video).stem;m=re.search(r"_seed(\d+)$",stem,re.I);seed=seed if seed is not None else(int(m.group(1)) if m else None)
    if image is None:image=str(TASK_ROOT/"first_frame.png") if (TASK_ROOT/"first_frame.png").is_file() else None
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
def profile(f, vessel_box=None):
    g=cv2.cvtColor(f,cv2.COLOR_RGB2GRAY);h,w=g.shape;s=np.abs(cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3))
    if vessel_box:
        x0, _, x1, _ = vessel_box
        xa, xb = x0 + int(.15 * (x1 - x0)), x1 - int(.15 * (x1 - x0))
    else:
        xa, xb = int(.35*w), int(.65*w)
    return np.median(s[:,xa:xb],axis=1)


def detect_floor(frames, floor_box=None):
    h, w = frames.shape[1:3]
    if floor_box:
        x0, y0, x1, y1 = floor_box
        profiles = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
            profiles.append(np.median(grad[y0:y1, x0:x1], axis=1))
        profile_values = np.median(np.stack(profiles), axis=0)
        return int(y0 + np.argmax(profile_values)) if len(profile_values) else int(.88 * h)
    return int(.88 * h)
def wall_edges(frame,vessel_box=None):
    h,w=frame.shape[:2];x0,y0,x1,y1=vessel_box or (int(.3*w),int(.04*h),int(.7*w),int(.95*h))
    grad=cv2.Sobel(frame.astype(np.float32),cv2.CV_32F,1,0,ksize=3)
    profile=np.median(np.linalg.norm(grad,axis=2)[y0+5:y1-8],axis=0)
    window=max(8,int(.08*(x1-x0)));edges=[]
    for x in [x0,x1]:
        lo=max(1,x-window);hi=min(w-1,x+window+1)
        k=lo+int(np.argmax(profile[lo:hi]))
        if profile[k]<8:return np.nan,np.nan
        nearby=np.arange(max(lo,k-6),min(hi,k+7));strong=nearby[profile[nearby]>=.35*profile[k]]
        edges.append(float((strong[0]+strong[-1])/2))
    return edges[0],edges[1]

def widths(frames,y,vessel_box=None):
    measured=np.array([wall_edges(f,vessel_box) for f in frames]);left,right=np.nanmedian(measured,axis=0)
    return left,right,{'vessel_left_x_px':left,'vessel_right_x_px':right,'vessel_width_px':right-left,'wall_method':'color-vector edges near independently observed vessel boundaries','wall_coverage':float(np.isfinite(measured).all(1).mean())}

def width_one(frame,vessel_box=None):
    left,right=wall_edges(frame,vessel_box)
    return float(right-left)

def water_geometry(frames: np.ndarray) -> tuple[np.ndarray | None, int | None, dict[str, Any], str | None]:
    """Track a supported horizontal interface inside the observed vessel ROI.

    Side bands avoid the floating ice/stone. Color-vector gradients support
    different liquid hues; no target displacement or motion sign is supplied.
    """
    height,width=frames.shape[1:3];rois,_=load_rois()
    vessel=roi_pixels(rois.get("vessel_inner",(.30,.04,.40,.91)),width,height)
    x0,y0,x1,y1=vessel;vw=x1-x0;vh=y1-y0
    margin=max(5,int(.025*vw));band=max(5,int(.15*vw))
    xs=np.r_[np.arange(x0+margin,min(x0+margin+band,x1-margin)),
             np.arange(max(x1-margin-band,x0+margin),x1-margin)]
    profiles=[]
    for frame in frames:
        gradient=cv2.Sobel(frame.astype(np.float32),cv2.CV_32F,0,1,ksize=3)
        strengths=np.linalg.norm(gradient,axis=2)
        profile=np.median(strengths[:,xs],axis=1)
        profiles.append(np.convolve(profile,np.ones(3)/3.0,mode="same"))
    profiles=np.stack(profiles)
    reference=np.median(profiles[:max(3,min(8,len(frames)//10))],axis=0)
    floor_box=roi_pixels(rois.get("floor"),width,height)
    lower_lo,lower_hi=(floor_box[1],floor_box[3]) if floor_box else (y0+int(.78*vh),y1)
    lower=reference[lower_lo:lower_hi]
    debug={"vessel_box_px":vessel,"profile_columns_px":xs.tolist()}
    if not len(lower) or float(np.max(lower))<8:
        return None,None,debug,"P23_vessel_floor_missing"
    floor_peak=lower_lo+int(np.argmax(lower))
    # A thick vessel stroke has two gradient peaks; use its center rather
    # than choosing whichever edge happens to have higher contrast.
    near=np.arange(max(lower_lo,floor_peak-8),min(lower_hi,floor_peak+9))
    strong=near[reference[near]>=.30*reference[floor_peak]]
    floor_y=int(round(float((strong[0]+strong[-1])/2)))
    upper_lo=y0+max(5,int(.025*vh));upper_hi=floor_y-max(10,int(.05*vh))
    if upper_hi<=upper_lo:return None,floor_y,debug,"P23_vessel_interior_missing"
    baseline=upper_lo+int(np.argmax(reference[upper_lo:upper_hi]))
    if reference[baseline]<8:return None,floor_y,debug,"P23_waterline_edge_missing"
    levels=[];edge_strengths=[];previous=float(baseline)
    # Motion is tracked locally without imposing its expected direction.
    window=max(8,int(.06*vh))
    for profile in profiles:
        lo=max(upper_lo,int(previous)-window);hi=min(upper_hi,int(previous)+window+1)
        index=lo+int(np.argmax(profile[lo:hi]));strength=float(profile[index])
        if strength<8:
            debug.update(waterline_y_px=np.asarray(levels),vessel_floor_y_px=floor_y)
            return None,floor_y,debug,"P23_waterline_edge_lost"
        rows=np.arange(max(lo,index-2),min(hi,index+3))
        previous=float(np.average(rows,weights=profile[rows]+1e-6))
        levels.append(previous);edge_strengths.append(strength)
    levels_array=np.asarray(levels)
    debug.update(waterline_y_px=levels_array,vessel_floor_y_px=floor_y,
                 reference_surface_y_px=baseline,surface_search_range_px=[upper_lo,upper_hi],
                 waterline_edge_strength=edge_strengths,waterline_local_window_px=window)
    if float(np.std(levels_array))>.18*height:
        return None,floor_y,debug,"P23_waterline_track_unstable"
    return levels_array,floor_y,debug,None

def extract(frames):
    rois,roi_source=load_rois();hh,ww=frames.shape[1:3]
    rois={"vessel_inner":(.30,.04,.40,.91),"surface":(.30,.18,.40,.18),"floor":(.30,.84,.40,.08),**rois}
    vessel_box=roi_pixels(rois['vessel_inner'],ww,hh)
    ys,floor,surface_debug,reason=water_geometry(frames)
    if reason:return None,surface_debug,reason
    slo,shi=surface_debug['surface_search_range_px'];left,right,wd=widths(frames,ys,vessel_box)
    if wd['wall_coverage']<.8:return None,{'surface_y_px':ys,**wd},'P23_vessel_wall_observations_incomplete'
    tail=max(5,min(12,len(ys)//8)); before=float(np.median(ys[:tail])); after=float(np.median(ys[-tail:])); height_before=max(1.,floor-before); height_after=max(1.,floor-after); ratio=height_after/height_before
    widths_px=np.asarray([width_one(f,vessel_box) for f in frames]); wi=float(np.median(widths_px[:tail])); wf=float(np.median(widths_px[-tail:]));
    m={"initial_waterline_y_px":before,"final_ice_surface_y_px":after,"vessel_floor_y_px":floor,"initial_water_height_px":height_before,"final_ice_height_px":height_after,"height_ratio":ratio,"density_ratio_reference":RHO_RATIO,"density_ratio_error":ratio-RHO_RATIO,"initial_cross_section_width_px":wi,"final_cross_section_width_px":wf,"cross_section_consistency_error":abs(wf-wi)/max(wi,1.),"surface_track_coverage":float(np.isfinite(ys).mean()),**wd}
    d={"roi_source":roi_source,"roi":rois,"surface_y_px":ys,"vessel_floor_y_px":floor,"surface_search_range_px":[slo,shi],"vessel_width":wd}
    return m,d,None
def artifacts(frames,d,m,root,sample,fps):
    folder=root/sample;folder.mkdir(parents=True,exist_ok=True);raw=folder/"measurements.json";raw.write_text(json.dumps(finite({"measurements":m,"detection":d}),indent=2));import matplotlib.pyplot as plt
    if m is None:
        # Early geometry returns carry waterline_y_px; surface_y_px and wall
        # coordinates exist only after extraction succeeds.
        plot=folder/"plot.png";fig,ax=plt.subplots(figsize=(8,4))
        levels=d.get("surface_y_px",d.get("waterline_y_px",[]))
        if len(levels):
            ax.plot(levels,label="observed surface prefix");ax.invert_yaxis();ax.legend()
        ax.set_title("Surface/geometry incomplete; no physical value inferred")
        ax.set_xlabel("frame");fig.tight_layout();fig.savefig(plot,dpi=120);plt.close(fig)
        return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":None}
    plot=folder/"plot.png";fig,ax=plt.subplots(figsize=(8,4));ax.plot(d["surface_y_px"],label="water/ice surface y");ax.axhline(d["vessel_floor_y_px"],ls="--",label="floor");ax.invert_yaxis();ax.set_xlabel("frame");ax.legend();fig.tight_layout();fig.savefig(plot,dpi=120);plt.close(fig)
    ov=folder/"overlay.mp4";w=cv2.VideoWriter(str(ov),cv2.VideoWriter_fourcc(*"mp4v"),fps or 24,(frames.shape[2],frames.shape[1]));
    if not w.isOpened():return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":None}
    for i,f in enumerate(frames):
        b=cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
        for name, color in (("vessel_inner", (255, 180, 0)), ("surface", (0, 255, 0)), ("floor", (0, 220, 220))):
            box = roi_pixels(d.get("roi", {}).get(name), b.shape[1], b.shape[0])
            if box:
                x0, y0, x1, y1 = box
                cv2.rectangle(b, (x0, y0), (x1, y1), color, 1)
                cv2.putText(b, name, (x0 + 3, max(18, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        y=int(round(d["surface_y_px"][i]));cv2.line(b,(int(.18*b.shape[1]),y),(int(.82*b.shape[1]),y),(0,220,0),2);cv2.line(b,(int(.18*b.shape[1]),d["vessel_floor_y_px"]),(int(.82*b.shape[1]),d["vessel_floor_y_px"]),(0,220,220),2);cv2.line(b,(int(round(d["vessel_width"]["vessel_left_x_px"])),0),(int(round(d["vessel_width"]["vessel_left_x_px"])),b.shape[0]),(220,80,0),2);cv2.line(b,(int(round(d["vessel_width"]["vessel_right_x_px"])),0),(int(round(d["vessel_width"]["vessel_right_x_px"])),b.shape[0]),(220,80,0),2);w.write(b)
    w.release();return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":str(ov)}
def make_payload(video,image,seed,model,m,d,reason,sample):
    vp,ip,sd=metadata(video,image,seed);ok=reason is None;warn=[]
    if m and abs(m["density_ratio_error"])>.08:warn.append("P23_height_ratio_differs_from_density_prediction")
    q1=residual_physics_score(m["density_ratio_error"]) if ok and m else None;q2=residual_physics_score(m["cross_section_consistency_error"]) if ok and m else None;m1=recognized_score(q1) if q1 is not None else None;m2=recognized_score(q2) if q2 is not None else None
    if m:m["score_normalization"]={"score_range":[0.0,1.0],"higher_is_better":True,"M1":{"raw_measurement":"density_ratio_error","scale":1.0,"physics_score":q1,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 / (1 + abs(density_ratio_error))","score":m1},"M2":{"raw_measurement":"cross_section_consistency_error","scale":1.0,"physics_score":q2,"recognition_score":0.15 if ok else 0.0,"formula":"0.15 + 0.85 / (1 + abs(cross_section_consistency_error))","score":m2}}
    return finite({"task_id":TASK_ID,"video_path":vp,"image_path":ip,"seed":sd,"model":model or "unknown","metrics":{"M1":{"extract_success":ok,"metric":m1},"M2":{"extract_success":ok,"metric":m2},"M3":None},"verbose":{"sample_id":sample,"failure_reason":reason,"quality_warnings":warn,"measurements":m or {},"debug":d}})
def evaluate(video,out,sample,image,seed,model,limit=None):
    try:f,fps=read_video(video,limit);m,d,r=extract(f);root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug";dbg=artifacts(f,d,m,root,sample,fps);p=make_payload(video,image,seed,model,m,dbg,r,sample);rc=0
    except Exception as e:
        root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug";folder=root/sample;folder.mkdir(parents=True,exist_ok=True);ep=folder/"error.txt";ep.write_text(f"{type(e).__name__}: {e}\n");p=make_payload(video,image,seed,model,None,{"directory":str(folder),"error":str(ep)},f"{type(e).__name__}: {e}",sample);rc=2
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(p,indent=2,allow_nan=False));return rc
def main():
    a=argparse.ArgumentParser();a.add_argument("--video");a.add_argument("--videos");a.add_argument("--output");a.add_argument("--outdir");a.add_argument("--task_id",default=TASK_ID);a.add_argument("--model",default="unknown");a.add_argument("--image-path","--image_path",dest="image");a.add_argument("--seed",type=int);a.add_argument("--sample-id");a.add_argument("--max-frames",type=int);x=a.parse_args()
    if x.task_id.upper()!=TASK_ID: a.error(f"this evaluator only supports {TASK_ID}")
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
