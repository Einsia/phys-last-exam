#!/usr/bin/env python3
"""Independent P21c freshwater-ice-in-salt-water evaluator."""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np

TASK_ID = "P21c"
TASK_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vdmbench-matplotlib")

def finite(v: Any) -> Any:
    if isinstance(v, dict): return {str(k): finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [finite(x) for x in v]
    if isinstance(v, np.ndarray): return finite(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)): return finite(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def unit_score(value: float) -> float:
    value = float(value)
    if not math.isfinite(value): raise ValueError("physics score must be finite")
    return float(np.clip(value, 0.0, 1.0))

def direction_score(value: float, expected_sign: int) -> float:
    """Score an inequality-only metric without inventing a target magnitude."""
    value = float(value)
    if not math.isfinite(value): raise ValueError("direction measurement must be finite")
    return 1.0 if expected_sign * value > 0.0 else 0.0

def recognized_score(physics_score: float) -> float:
    return 0.15 + 0.85 * unit_score(physics_score)

def rel(v: str | Path | None) -> str | None:
    if v is None: return None
    p = Path(v).expanduser()
    try: return p.resolve().relative_to(TASK_ROOT.resolve()).as_posix()
    except ValueError: return str(p)

def metadata(video: str, image: str | None, seed: int | None):
    stem = Path(video).stem; m = re.search(r"_seed(\d+)$", stem, re.I)
    seed = seed if seed is not None else (int(m.group(1)) if m else None)
    base = re.sub(r"_seed\d+$", "", stem, flags=re.I)
    if image is None:
        for name in ("first_frames/provided/first_frame.png", "first_frames/simulation/sim_first_frame.png"):
            p = TASK_ROOT / name
            if p.is_file(): image = str(p); break
    return rel(video) or str(video), rel(image), seed

def read_video(path: str, limit: int | None = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise RuntimeError(f"cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0); out = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if limit and len(out) >= limit: break
    cap.release()
    if not out: raise RuntimeError("video contains no frames")
    return np.stack(out), fps

def load_rois():
    path = TASK_ROOT / "evaluator" / "roi.json"
    if not path.is_file(): return {}, "auto"
    try:
        raw = json.loads(path.read_text()); raw = raw.get("rois", raw); out = {}
        for key, value in raw.items():
            if key in {"task_id", "frame_size", "source_video", "source_frame"}: continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                box = tuple(float(v) for v in value)
                if all(0 <= v <= 1 for v in box) and box[2] > 0 and box[3] > 0 and box[0] + box[2] <= 1 and box[1] + box[3] <= 1: out[key] = box
        return out, "annotated"
    except (OSError, ValueError, TypeError): return {}, "auto_invalid"

def roi_pixels(box, width, height):
    if box is None: return None
    x, y, w, h = box; x0, y0 = max(0, int(x * width)), max(0, int(y * height)); x1, y1 = min(width, int((x + w) * width)), min(height, int((y + h) * height)); return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)

def edge_profile(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY); h, w = gray.shape
    sob = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    p = np.median(sob[:, int(.4*w):int(.6*w)], axis=1)
    return np.convolve(p, np.ones(3)/3, mode="same")

def geometry(frames: np.ndarray):
    h = frames.shape[1]; profiles = np.stack([edge_profile(f) for f in frames])
    ref = np.median(profiles[:max(3, min(8, len(frames)//10))], axis=0)
    lo, hi = int(.18*h), int(.55*h); base = lo + int(np.argmax(ref[lo:hi]))
    flo, fhi = int(.75*h), int(.93*h); floor = flo + int(np.argmax(ref[flo:fhi]))
    ys=[]; slo=max(lo, int(base-.12*h)); shi=min(hi, int(base+.12*h))
    for p in profiles:
        i=slo+int(np.argmax(p[slo:shi])); r=np.arange(max(slo,i-2), min(shi,i+3)); ys.append(float(np.average(r,weights=p[r]+1e-3)))
    ys=np.asarray(ys); return ys, floor, {"waterline_y_px":ys,"vessel_floor_y_px":floor,"reference_surface_y_px":base,"surface_search_range_px":[slo,shi]}

def solid_evidence(frame: np.ndarray, surface: float, floor: int, vessel_box=None, ice_box=None) -> tuple[float, tuple[int,int,int,int] | None]:
    gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY); h,w=gray.shape
    base = ice_box or vessel_box; x0,x1=(base[0],base[2]) if base else (int(.30*w),int(.70*w)); y0=max(0,int(surface+8),base[1] if base else 0); y1=min(h,int(floor-4),base[3] if base else h)
    if y1<=y0+10:return 0.0,None
    roi=gray[y0:y1,x0:x1]; grad=np.abs(cv2.Sobel(roi,cv2.CV_32F,1,1,ksize=3))
    # Ice has a compact textured/high-gradient footprint; suppress vessel borders.
    mask=((roi < np.percentile(roi,45)) | (grad > np.percentile(grad,78))).astype(np.uint8)
    mask[:max(1,int(.08*mask.shape[0]))]=0; mask[-max(1,int(.05*mask.shape[0])):]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask,8); best=None
    for i in range(1,n):
        x,y,ww,hh,a=stats[i]
        if a>0.001*roi.size and a<.80*roi.size and ww>8 and hh>8:
            if best is None or a>best[0]:best=(int(a),x,y,ww,hh)
    area=float(best[0]) if best else 0.0; denom=max(1,float(roi.size)); box=None
    if best: box=(x0+best[1],y0+best[2],best[3],best[4])
    return area/denom,box

def final_frame_ice_detection(frame: np.ndarray, surface: float, floor: int,
                              vessel_box) -> dict[str, Any]:
    """Independent end-frame residual-ice detector (no temporal averaging)."""
    x0, y0v, x1, y1v = vessel_box; h, w = frame.shape[:2]
    xa = max(x0 + int(.04 * (x1 - x0)), 0); xb = min(x1 - int(.04 * (x1 - x0)), w)
    ya = max(y0v, int(surface + 6)); yb = min(y1v, int(floor + .06 * h), h)
    out = {"detected": False, "confidence": 0.0, "box_px": None,
           "edge_density": 0.0, "candidate_count": 0,
           "method": "final_frame_irregular_edge"}
    if xb <= xa + 20 or yb <= ya + 20: return out
    gray = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 70)
    edges[:8] = 0; edges[-8:] = 0; edges[:, :8] = 0; edges[:, -8:] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = gray.shape; candidates = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < .15*cw or bw > .85*cw or bh < .12*ch or bh > .60*ch: continue
        if bx < .02*cw or bx+bw > .98*cw: continue
        if not .15 <= (bx + .5*bw)/cw <= .85: continue
        arc = float(cv2.arcLength(c, False)); complexity = arc/max(2.*(bw+bh), 1.)
        if not .45 <= complexity <= 3.0: continue
        density = float(np.mean(edges[by:by+bh, bx:bx+bw] > 0))
        if density < .035: continue
        brightness_p65 = float(np.percentile(gray[by:by+bh, bx:bx+bw], 65))
        if brightness_p65 < 165.0: continue
        score = min(1., density/.14)*min(1., arc/700.)
        candidates.append({"x":int(xa+bx),"y":int(ya+by),"width":int(bw),"height":int(bh),"arc_px":arc,"edge_density":density,"brightness_p65":brightness_p65,"complexity":complexity,"score":score})
    out["candidate_count"] = len(candidates)
    if candidates:
        best=max(candidates,key=lambda x:float(x["score"]))
        out.update({"detected":bool(best["score"] >= .20),"confidence":float(best["score"]),"box_px":{k:best[k] for k in ("x","y","width","height")},"edge_density":float(best["edge_density"]),"candidate":best})
    return out

def extract(frames: np.ndarray):
    levels,floor,debug=geometry(frames); rois, roi_source = load_rois(); hh, ww = frames.shape[1:3]
    rois={"vessel":(.30,.04,.40,.91),"ice_initial":(.34,.20,.32,.42),"surface":(.32,.18,.36,.18),**rois}
    vessel_box=roi_pixels(rois["vessel"],ww,hh); ice_box=roi_pixels(rois.get("ice_initial"),ww,hh); ev=[]; boxes=[]
    for f,y in zip(frames,levels):
        e,b=solid_evidence(f,y,floor,vessel_box,ice_box); ev.append(e); boxes.append(b)
    ev=np.asarray(ev); debug.update({"roi_source":roi_source,"roi":rois,"solid_evidence":ev,"solid_boxes":boxes})
    tail=max(5,min(12,len(levels)//8)); initial=float(np.median(levels[:tail])); final=float(np.median(levels[-tail:])); H=max(1.,floor-initial)
    final_ice = final_frame_ice_detection(frames[-1], final, floor, vessel_box)
    m={"initial_waterline_y_px":initial,"final_waterline_y_px":final,"vessel_floor_y_px":int(floor),"initial_water_height_px":H,"final_water_height_px":float(floor-final),"water_height_change_norm":float((initial-final)/H),"initial_solid_evidence":float(np.median(ev[:tail])),"final_solid_evidence":float(np.median(ev[-tail:])),"solid_evidence_ratio":float(np.median(ev[-tail:])/max(np.median(ev[:tail]),1e-6)),"solid_track_coverage":float(np.mean(ev>0)),"final_frame_ice_detected":bool(final_ice["detected"]),"final_frame_ice_detection":final_ice}
    reason=None
    if np.isfinite(levels).mean()<.8: reason="P21c_waterline_track_incomplete"
    debug["initial_ice_evidence_sufficient"] = bool(m["initial_solid_evidence"] >= 1e-4)
    return m,debug,reason

def artifacts(frames,debug,m,root,sample,fps):
    folder=root/sample; folder.mkdir(parents=True,exist_ok=True); raw=folder/"measurements.json"; raw.write_text(json.dumps(finite({"measurements":m,"detection":debug}),indent=2))
    import matplotlib.pyplot as plt
    plot=folder/"plot.png"; fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True); ax[0].plot(debug["waterline_y_px"],label="waterline y"); ax[0].invert_yaxis(); ax[0].legend(); ax[1].plot(debug["solid_evidence"],label="ice evidence"); ax[1].set_xlabel("frame"); ax[1].legend(); fig.suptitle(sample); fig.tight_layout(); fig.savefig(plot,dpi=120); plt.close(fig)
    overlay=folder/"overlay.mp4"; wr=cv2.VideoWriter(str(overlay),cv2.VideoWriter_fourcc(*"mp4v"),fps or 24,(frames.shape[2],frames.shape[1]));
    if not wr.isOpened(): return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":None}
    for i,f in enumerate(frames):
        bgr=cv2.cvtColor(f,cv2.COLOR_RGB2BGR)
        for name, color in (("vessel", (255, 180, 0)), ("ice_initial", (255, 0, 255)), ("surface", (0, 255, 0))):
            box = roi_pixels(debug.get("roi", {}).get(name), bgr.shape[1], bgr.shape[0])
            if box:
                x0, y0, x1, y1 = box
                cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 1)
                cv2.putText(bgr, name, (x0 + 3, max(18, y0 + 16)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
        y=int(round(debug["waterline_y_px"][i])); cv2.line(bgr,(int(.3*bgr.shape[1]),y),(int(.7*bgr.shape[1]),y),(0,220,0),2); cv2.line(bgr,(int(.3*bgr.shape[1]),debug["vessel_floor_y_px"]),(int(.7*bgr.shape[1]),debug["vessel_floor_y_px"]),(0,220,220),2); box=debug["solid_boxes"][i];
        if box: x,y,w,h=box; cv2.rectangle(bgr,(x,y),(x+w,y+h),(0,0,230),2)
        if i == len(frames) - 1:
            final_detection = m.get("final_frame_ice_detection", {}) if m else {}
            final_box = final_detection.get("box_px")
            if final_box:
                cv2.rectangle(bgr, (int(final_box["x"]), int(final_box["y"])),
                              (int(final_box["x"] + final_box["width"]), int(final_box["y"] + final_box["height"])),
                              (0, 165, 255), 3)
            cv2.putText(bgr, f"final-frame ice: {bool(final_detection.get('detected', False))}",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 165, 255), 2)
        cv2.putText(bgr,f"ice evidence={debug['solid_evidence'][i]:.3f}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(30,30,220),2); wr.write(bgr)
    wr.release(); return {"directory":str(folder),"plot":str(plot),"measurements":str(raw),"overlay_video":str(overlay)}

def payload(video,image,seed,model,m,debug,reason,sample,m1_ok=None,m2_ok=None):
    vp,ip,sd=metadata(video,image,seed); warnings=[]
    if m and m["water_height_change_norm"]<=0: warnings.append("P21c_waterline_did_not_increase")
    if m and m["solid_evidence_ratio"]>.25: warnings.append("P21c_ice_not_fully_disappeared")
    if m and m.get("final_frame_ice_detected",False): warnings.append("P21c_ice_detected_in_final_frame")
    if m1_ok is None: m1_ok = m is not None and reason is None
    if m2_ok is None:
        m2_ok = (m1_ok and m is not None and m.get("initial_solid_evidence", 0.0) >= 1e-4
                 and math.isfinite(float(m.get("solid_evidence_ratio", float("nan")))))
    q1 = direction_score(m["water_height_change_norm"], 1) if m1_ok and m else None
    q2 = unit_score(1.0 - m["solid_evidence_ratio"]) if m2_ok and m else None
    if m and m.get("final_frame_ice_detected",False):
        # Residual ice is a definitive failure of the completion criterion;
        # the end-frame rule overrides ratio confidence with a zero physics
        # score. V2 still awards this measurable M item its recognition share.
        m2_ok = True
        q2 = 0.0
    m1_score = recognized_score(q1) if q1 is not None else None
    m2_score = recognized_score(q2) if q2 is not None else None
    if m:
        m["score_normalization"] = {
            "score_range": [0.0, 1.0], "higher_is_better": True,
            "M1": {"raw_measurement": "water_height_change_norm",
                   "physics_score": q1, "recognition_score": 0.15 if m1_ok else 0.0,
                   "formula": "0.15 + 0.85 * (1 if water_height_change_norm > 0 else 0)", "score": m1_score},
            "M2": {"raw_measurement": "solid_evidence_ratio",
                   "physics_score": q2, "recognition_score": 0.15 if m2_ok else 0.0,
                   "formula": "0.15 + 0.85 * (0 if final_frame_ice_detected else clip(1 - solid_evidence_ratio, 0, 1))", "score": m2_score},
        }
    return finite({"task_id":TASK_ID,"video_path":vp,"image_path":ip,"seed":sd,"model":model or "unknown","metrics":{"M1":{"extract_success":bool(m1_ok),"metric":m1_score},"M2":{"extract_success":bool(m2_ok),"metric":m2_score},"M3":None},"verbose":{"sample_id":sample,"failure_reason":reason,"quality_warnings":warnings,"measurements":m or {},"debug":debug}})

def evaluate(video,out,sample,image,seed,model,limit=None):
    try:
        frames,fps=read_video(video,limit); m,d,reason=extract(frames); root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug"; dbg=artifacts(frames,d,m,root,sample,fps)
        m1_ok=m is not None and reason is None
        m2_ok=(m1_ok and m.get("initial_solid_evidence",0.0)>=1e-4
               and math.isfinite(float(m.get("solid_evidence_ratio",float("nan")))))
        if not m2_ok and reason is None:
            reason="P21c_initial_ice_missing"
        p=payload(video,image,seed,model,m,dbg,reason,sample,m1_ok,m2_ok); rc=0 if m1_ok else 1
    except Exception as e:
        root=out.parent.parent/"debug" if out.parent.name=="json" else out.parent/"debug"; folder=root/sample; folder.mkdir(parents=True,exist_ok=True); ep=folder/"error.txt"; ep.write_text(f"{type(e).__name__}: {e}\n"); p=payload(video,image,seed,model,None,{"directory":str(folder),"error":str(ep)},f"{type(e).__name__}: {e}",sample,False,False); rc=2
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(p,indent=2,allow_nan=False)); return rc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--videos"); ap.add_argument("--output"); ap.add_argument("--outdir"); ap.add_argument("--task_id",default=TASK_ID); ap.add_argument("--model",default="unknown"); ap.add_argument("--image-path","--image_path",dest="image"); ap.add_argument("--seed",type=int); ap.add_argument("--sample-id"); ap.add_argument("--max-frames",type=int); a=ap.parse_args()
    if a.task_id.lower()!=TASK_ID.lower(): ap.error(f"this evaluator only supports {TASK_ID}")
    if a.video:
        if not a.output: ap.error("--video requires --output")
        return evaluate(a.video,Path(a.output),a.sample_id or Path(a.video).stem,a.image,a.seed,a.model,a.max_frames)
    src=Path(a.videos or "."); files=[src] if src.is_file() else sorted(src.glob("*.mp4")); out=Path(a.outdir or "eval_results"); jd=out/"json"; jd.mkdir(parents=True,exist_ok=True)
    if not files: print(f"no MP4 videos found under {src}",file=sys.stderr); return 1
    for i,p in enumerate(files,1): print(f"[{i}/{len(files)}] {p.stem} {'OK' if evaluate(str(p),jd/(p.stem+'.json'),p.stem,a.image,a.seed,a.model,a.max_frames)==0 else 'CHECK'}")
    fields=["task_id","video_path","image_path","seed","model","M1_extract_success","M1","M2_extract_success","M2","M3","failure_reason"]
    with (out/"results.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields); w.writeheader()
        for p in sorted(jd.glob("*.json")):
            x=json.loads(p.read_text()); w.writerow({"task_id":x["task_id"],"video_path":x["video_path"],"image_path":x["image_path"],"seed":x["seed"],"model":x["model"],"M1_extract_success":x["metrics"]["M1"]["extract_success"],"M1":x["metrics"]["M1"]["metric"],"M2_extract_success":x["metrics"]["M2"]["extract_success"],"M2":x["metrics"]["M2"]["metric"],"M3":None,"failure_reason":x["verbose"]["failure_reason"]})
    return 0
if __name__=="__main__": raise SystemExit(main())
