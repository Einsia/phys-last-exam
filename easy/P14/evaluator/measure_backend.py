#!/usr/bin/env python3
"""Deterministic, non-LLM/non-VLM evaluator for P14 two-pendulum videos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import av
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks, savgol_filter

from runtime_paths import resolve_tracking_runtime
from scoring import score_record as continuous_score_record


HERE = Path(__file__).resolve().parent


def finite(v):
    return v is not None and np.isfinite(v)


def js(v):
    if isinstance(v, (np.floating, np.integer)): return v.item()
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, dict): return {str(k): js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [js(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def lerp_score(value, good, bad):
    if not finite(value): return None
    if value <= good: return 100.0
    if value >= bad: return 0.0
    return float(100.0 * (bad - value) / (bad - good))


def high_score(value, good, bad):
    if not finite(value): return None
    if value >= good: return 100.0
    if value <= bad: return 0.0
    return float(100.0 * (value - bad) / (good - bad))


def weighted(values):
    usable = [(float(v), float(w)) for v, w in values if finite(v) and w > 0]
    return None if not usable else sum(v*w for v, w in usable) / sum(w for _, w in usable)


def decode_video(path):
    frames, times = [], []
    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        rate = float(stream.average_rate or 24)
        for i, f in enumerate(c.decode(stream)):
            frames.append(f.to_ndarray(format="rgb24"))
            times.append(float(f.pts * f.time_base) if f.pts is not None else i / rate)
    if not frames: raise ValueError("video contains no decoded frames")
    times = np.asarray(times, np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        dt = np.median(np.diff(times)[np.diff(times) > 0]) if np.any(np.diff(times) > 0) else 1/24
        times = np.arange(len(frames)) * dt
    return np.asarray(frames, np.uint8), times


def sample_id_from_name(stem):
    m = re.match(r"(.+)_seed\d+$", stem)
    return m.group(1) if m else stem


def camera_audit(frames, stride=8):
    """SIFT/RANSAC similarity audit; values are maxima against frame zero."""
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.025)
    k0, d0 = sift.detectAndCompute(gray0, None)
    h, w = gray0.shape; diag = math.hypot(w, h)
    records = []
    if d0 is None or len(k0) < 20:
        return {"available": False, "translation_px_max": None,
                "translation_diagonal_fraction_max": None, "scale_change_max": None,
                "rotation_deg_max": None, "inliers_min": 0, "samples": []}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    for idx in list(range(stride, len(frames), stride)) + ([len(frames)-1] if (len(frames)-1) % stride else []):
        g = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        k, d = sift.detectAndCompute(g, None)
        if d is None or len(k) < 20: continue
        pairs = matcher.knnMatch(d0, d, k=2)
        good = [a for a, b in pairs if a.distance < .72 * b.distance]
        if len(good) < 12: continue
        src = np.float32([k0[m.queryIdx].pt for m in good])
        dst = np.float32([k[m.trainIdx].pt for m in good])
        M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=2.5,
                                              maxIters=3000, confidence=.995)
        if M is None: continue
        a, b = float(M[0,0]), float(M[1,0])
        scale = math.hypot(a, b); rot = abs(math.degrees(math.atan2(b, a)))
        trans = math.hypot(float(M[0,2]), float(M[1,2]))
        records.append({"frame": idx, "translation_px": trans,
                        "translation_diagonal_fraction": trans/diag,
                        "scale_change": abs(scale-1), "rotation_deg": rot,
                        "inliers": int(mask.sum()) if mask is not None else 0})
    if not records:
        return {"available": False, "translation_px_max": None,
                "translation_diagonal_fraction_max": None, "scale_change_max": None,
                "rotation_deg_max": None, "inliers_min": 0, "samples": []}
    return {"available": True,
            "translation_px_max": max(x["translation_px"] for x in records),
            "translation_diagonal_fraction_max": max(x["translation_diagonal_fraction"] for x in records),
            "scale_change_max": max(x["scale_change"] for x in records),
            "rotation_deg_max": max(x["rotation_deg"] for x in records),
            "inliers_min": min(x["inliers"] for x in records), "samples": records}


def interpolate_xy(xy, maximum_gap=5):
    out = np.asarray(xy, float).copy(); n = len(out)
    valid = np.isfinite(out).all(axis=1)
    for a in range(n):
        if valid[a]: continue
        b = a
        while b < n and not valid[b]: b += 1
        left = a-1
        if left >= 0 and b < n and b-a <= maximum_gap:
            for k in range(a, b): out[k] = out[left] + (out[b]-out[left]) * ((k-left)/(b-left))
            valid[a:b] = True
        a = b
    return out, valid


def release_index(theta):
    x = np.asarray(theta, float)
    good = np.isfinite(x)
    if good.sum() < 8: return None
    amp = np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
    ref = np.nanmedian(x[:min(4, len(x))]); threshold = max(.006, .02*amp)
    for i in range(1, len(x)-3):
        w = x[i:i+3]
        if np.isfinite(w).all() and np.all(np.abs(w-ref) > threshold): return i
    return 0


def robust_cv(x):
    x = np.asarray(x, float); x = x[np.isfinite(x) & (x > 0)]
    if len(x) < 2: return None
    med = np.median(x); mad = np.median(np.abs(x-med))
    return float(1.4826 * mad / med) if med > 0 else None


def period_measure(theta, times, cfg):
    theta = np.asarray(theta, float); times = np.asarray(times, float)
    valid = np.isfinite(theta)
    if valid.sum() < max(16, int(.55*len(theta))): return None
    filled = np.interp(np.arange(len(theta)), np.flatnonzero(valid), theta[valid])
    win = min(11, len(filled) if len(filled)%2 else len(filled)-1)
    smooth = savgol_filter(filled, win, 2) if win >= 5 else filled
    rel = release_index(smooth)
    if rel is None: return None
    t = times[rel:] - times[rel]; y = smooth[rel:]
    if len(y) < 12 or t[-1] <= .5: return None
    amplitude = .5*(np.percentile(y,95)-np.percentile(y,5))
    prominence = max(math.radians(1.2), .14*(np.percentile(y,95)-np.percentile(y,5)))
    dt = float(np.median(np.diff(t))); min_dist = max(4, int(.18/max(dt,1e-3)))
    pos, pp = find_peaks(y, prominence=prominence, distance=min_dist)
    neg, nprom = find_peaks(-y, prominence=prominence, distance=min_dist)
    ext = [(int(i), 1, float(y[i])) for i in pos] + [(int(i), -1, float(y[i])) for i in neg]
    # The configured initial state is release from rest at a turning point.
    if len(y) > 4:
        initial_sign = 1 if y[0] >= np.median(y) else -1
        ext.append((0, initial_sign, float(y[0])))
    ext.sort()
    collapsed = []
    for e in ext:
        if collapsed and e[1] == collapsed[-1][1]:
            better = e[2] > collapsed[-1][2] if e[1] > 0 else e[2] < collapsed[-1][2]
            if better: collapsed[-1] = e
        else: collapsed.append(e)
    ext = collapsed
    half = np.diff([t[e[0]] for e in ext]) if len(ext) >= 2 else np.asarray([])
    half = half[half > .12]
    cand = 2*np.median(half) if len(half) else min(2.0, t[-1])
    lo = max(.35, .55*cand); hi = min(max(lo+.1, 1.65*cand), max(.5, 1.25*t[-1]))

    def fit(T):
        w = 2*np.pi/T
        X = np.column_stack([np.ones(len(t)), t, np.sin(w*t), np.cos(w*t)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        pred = X @ beta
        return float(np.sum((y-pred)**2)), pred, beta
    # A many-cycle sinusoidal objective is not unimodal across the whole
    # plausible band. Bracket its best coarse solution before local fitting.
    grid=np.linspace(lo,hi,161);best=int(np.argmin([fit(T)[0] for T in grid]))
    bracket=(grid[max(0,best-1)],grid[min(len(grid)-1,best+1)])
    opt = minimize_scalar(lambda T: fit(T)[0], bounds=bracket, method="bounded",
                          options={"xatol": 1e-4, "maxiter": 120})
    T = float(opt.x); sse, pred, beta = fit(T)
    sst = float(np.sum((y-y.mean())**2)); r2 = 1-sse/sst if sst > 1e-9 else None
    full_intervals = []
    for sign in (-1,1):
        ts = [t[e[0]] for e in ext if e[1] == sign]
        full_intervals.extend(np.diff(ts).tolist())
    interval_period = float(np.median(full_intervals)) if full_intervals else None
    # Alternating turning points provide more samples than same-side peaks in a
    # five-second clip; all doubled half-cycle intervals should estimate the same T.
    interval_cv = robust_cv((2*half).tolist())
    # Integer-frame extrema are an independent diagnostic, not a weighted
    # replacement for a sub-frame period fitted to all observations.
    cycles = float((t[ext[-1][0]]-t[ext[0][0]])/T) if len(ext)>=2 and T>0 else 0.
    return {"period_s": T, "period_estimator":"continuous harmonic fit with coarse global bracket", "extrema_period_s":interval_period, "fit_r2": r2, "amplitude_deg": math.degrees(amplitude),
            "turning_points": int(len(ext)), "turning_frames": [int(rel+e[0]) for e in ext],
            "half_cycle_intervals_s": half.tolist(), "full_cycle_intervals_s": full_intervals,
            "period_cv": interval_cv, "observed_cycles": cycles, "release_frame": int(rel),
            "fit_t": (t+times[rel]).tolist(), "fit_theta": pred.tolist(),
            "smooth_theta": smooth.tolist()}


def track_summary(data, times, cfg):
    shift = np.asarray(data["camera_shift"], float)
    output = {}; max_gap = int(cfg["tracking"]["maximum_gap_frames"])
    for slot in ("short", "long"):
        cot = np.asarray(data[f"{slot}_bob_xy"],float)
        sam = np.asarray(data[f"sam2_{slot}_bob_xy"],float)
        cr = np.asarray(data[f"{slot}_bob_radius"],float)
        sr = np.asarray(data[f"sam2_{slot}_bob_radius"],float)
        nominal = float(cfg["samples"][cfg["_active_sample"]][slot]["radius"])
        cv = np.isfinite(cot).all(axis=1)
        seed_bob=np.asarray(cfg["samples"][cfg["_active_sample"]][slot]["bob"],float)
        # A mask can include the string as well as the bob. Validate object
        # identity against the observed frame-zero query, before trusting its
        # centroid for any frame; period/physics scores play no role here.
        sam_identity_ok=bool(np.isfinite(sam[0]).all() and np.linalg.norm(sam[0]-seed_bob)<.75*nominal
                             and np.isfinite(sr[0]) and .55*nominal<sr[0]<1.6*nominal)
        sv = np.isfinite(sam).all(axis=1) & np.isfinite(sr) & (sr>.35*nominal) & (sr<2.2*nominal) & sam_identity_ok
        braw = np.full_like(cot,np.nan)
        both=cv&sv; agree=both & (np.linalg.norm(cot-sam,axis=1)<1.6*nominal)
        braw[agree]=.35*cot[agree]+.65*sam[agree]
        braw[sv&~agree]=sam[sv&~agree]
        braw[cv&~sv]=cot[cv&~sv]
        praw = np.asarray(data[f"{slot}_pivot_xy"],float)
        b, bv = interpolate_xy(braw-shift, max_gap); p, pv = interpolate_xy(praw-shift, max_gap)
        valid = bv & pv
        vec = b-p; lengths = np.linalg.norm(vec,axis=1); lengths[~valid]=np.nan
        theta = np.arctan2(vec[:,0], vec[:,1]); theta[~valid]=np.nan
        p0 = np.nanmedian(p[:min(4,len(p))],axis=0); l0 = float(np.nanmedian(lengths[:min(4,len(lengths))]))
        pivot_drift = np.linalg.norm(p-p0,axis=1)
        rel_len = np.abs(lengths/l0-1) if l0>0 else np.full(len(lengths),np.nan)
        rad = np.where(sv,sr,cr)
        r0 = np.nanmedian(rad[:min(6,len(rad))]); rel_rad = np.abs(rad/r0-1) if finite(r0) and r0>0 else np.full(len(rad),np.nan)
        output[slot] = {"bob_xy":b,"pivot_xy":p,"valid":valid,"lengths":lengths,
                        "theta":theta,"initial_length_px":l0,
                        "coverage":float(valid.mean()),"sam2_coverage":float(sv.mean()),
                        "cotracker_coverage":float(cv.mean()),"sam2_initial_identity_verified":sam_identity_ok,
                        "pivot_drift_p95_px":float(np.nanpercentile(pivot_drift,95)),
                        "length_relative_change_p95":float(np.nanpercentile(rel_len,95)),
                        "radius_relative_change_p95":float(np.nanpercentile(rel_rad,95)) if np.isfinite(rel_rad).any() else None,
                        "period":period_measure(theta,times,cfg)}
    sep = np.linalg.norm(output["short"]["pivot_xy"]-output["long"]["pivot_xy"],axis=1)
    sep0=np.nanmedian(sep[:4]); sepchange=np.abs(sep/sep0-1) if sep0>0 else np.full(len(sep),np.nan)
    output["pivot_separation_change_p95"] = float(np.nanpercentile(sepchange,95))
    return output


def status_and_scores(tr, cam, meta, cfg):
    S,L=tr["short"],tr["long"]; st=cfg["structure"]; me=cfg["measurement"]; ph=cfg["physics"]
    errors=[]; warnings=[]
    extract = len(meta.get("errors",{})) < 2 and S["coverage"]>.5 and L["coverage"]>.5
    if not extract: errors.append("coordinate extraction failed or coverage <= 0.5")
    camera_ok = (not cam["available"] or
                 (cam["translation_diagonal_fraction_max"]<=st["camera_translation_diagonal_fraction_max"] and
                  cam["scale_change_max"]<=st["camera_scale_change_max"] and
                  cam["rotation_deg_max"]<=st["camera_rotation_deg_max"]))
    if not cam["available"]: warnings.append("SIFT camera audit unavailable; CoTracker background translation retained")
    pivot_norm=max(S["initial_length_px"],1e-6)
    pivot_drift=max(S["pivot_drift_p95_px"],L["pivot_drift_p95_px"])/pivot_norm
    string_change=max(S["length_relative_change_p95"],L["length_relative_change_p95"])
    structural = bool(extract and camera_ok and pivot_drift<=st["pivot_drift_short_length_fraction_max"] and
                      tr["pivot_separation_change_p95"]<=st["pivot_separation_change_fraction_max"] and
                      string_change<=st["string_length_p95_relative_change_max"])
    if not camera_ok: errors.append("camera motion exceeds frozen hard gate")
    if pivot_drift>st["pivot_drift_short_length_fraction_max"]: errors.append("pivot drift exceeds hard gate")
    if tr["pivot_separation_change_p95"]>st["pivot_separation_change_fraction_max"]: errors.append("pivot separation changes")
    if string_change>st["string_length_p95_relative_change_max"]: errors.append("string length changes")
    valid=True
    for slot,x in (("short",S),("long",L)):
        p=x["period"]
        if x["coverage"]<cfg["tracking"]["minimum_coverage"] or p is None or p["fit_r2"] is None or p["fit_r2"]<me["minimum_fit_r2"] or p["amplitude_deg"]<me["minimum_motion_amplitude_deg"] or p["turning_points"]<me["minimum_turning_points"] or p["observed_cycles"]<me["minimum_observed_cycles"]:
            valid=False; errors.append(f"{slot} pendulum lacks a valid complete-period measurement")
    measurement = bool(extract and valid)
    Ts=S["period"]["period_s"] if S["period"] else None; Tl=L["period"]["period_s"] if L["period"] else None
    lratio=S["initial_length_px"]/L["initial_length_px"] if L["initial_length_px"]>0 else None
    m1=abs((Ts/Tl)**2/lratio-1) if measurement and finite(lratio) and lratio>0 else None
    cvs=[x["period"]["period_cv"] for x in (S,L) if x["period"] and finite(x["period"]["period_cv"])]
    maxcv=max(cvs) if len(cvs)==2 else None
    release_delta=abs(S["period"]["release_frame"]-L["period"]["release_frame"])*meta["median_dt_s"] if S["period"] and L["period"] else None

    length_target=abs(lratio-ph["initial_length_ratio_target"]) if finite(lratio) else None
    setup=weighted([(lerp_score(length_target, .03, ph["initial_length_ratio_tolerance"]),.55),
                    (lerp_score(release_delta,ph["simultaneous_release_good_s"],ph["simultaneous_release_bad_s"]),.25),
                    (100*min(S["coverage"],L["coverage"]),.20)])
    struct_score=weighted([(lerp_score(pivot_drift,.008,st["pivot_drift_short_length_fraction_max"]),.35),
                           (lerp_score(string_change,.035,st["string_length_p95_relative_change_max"]),.40),
                           (lerp_score(tr["pivot_separation_change_p95"],.008,st["pivot_separation_change_fraction_max"]),.25)])
    r2min=min(S["period"]["fit_r2"],L["period"]["fit_r2"]) if measurement else None
    periodic=weighted([(high_score(r2min,ph["sinusoid_r2_good"],ph["sinusoid_r2_bad"]),.55),
                       (lerp_score(maxcv,ph["period_cv_good"],ph["period_cv_bad"]),.45)])
    law=lerp_score(m1,ph["m1_good"],ph["m1_bad"])
    dims={"setup":setup,"structural":struct_score,"periodicity":periodic,"pendulum_law":law}
    w=cfg["scores"]
    composite=weighted([(dims[k],w[k]) for k in dims]) if measurement else None
    overall=float(composite) if structural and measurement and finite(composite) else 0.0
    physics=bool(structural and measurement and finite(m1) and m1<=ph["physics_pass_m1_max"] and
                 finite(maxcv) and maxcv<=ph["physics_pass_period_cv_max"] and r2min>=ph["physics_pass_fit_r2_min"])
    return {"extract_success":bool(extract),"structural_ok":structural,
            "measurement_valid":measurement,"physics_pass":physics}, {
            "initial_length_ratio_short_over_long":lratio,"period_ratio_short_over_long":Ts/Tl if finite(Ts) and finite(Tl) and Tl>0 else None,
            "m1_abs_residual":m1,"maximum_period_cv":maxcv,"release_time_difference_s":release_delta,
            "pivot_drift_short_length_fraction_p95":pivot_drift,
            "string_length_relative_change_p95_max":string_change,
            "pivot_separation_relative_change_p95":tr["pivot_separation_change_p95"]}, dims, overall, errors, warnings


def render_plot(path,times,tr,metrics,status):
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for slot,color in (("short","tab:blue"),("long","tab:orange")):
        x=tr[slot]; ax[0,0].plot(times,np.degrees(x["theta"]),color=color,alpha=.35,label=f"{slot} measured")
        if x["period"]:
            ax[0,0].plot(x["period"]["fit_t"],np.degrees(x["period"]["fit_theta"]),color=color,lw=2,label=f"{slot} fit T={x['period']['period_s']:.2f}s")
        ax[0,1].plot(times,x["lengths"]/x["initial_length_px"],color=color,label=slot)
    ax[0,0].set(title="Angular motion and deterministic sinusoid fit",xlabel="time (s)",ylabel="angle (deg)"); ax[0,0].legend(fontsize=8)
    ax[0,1].axhline(1,color='k',lw=.8); ax[0,1].set(title="String length / frozen frame-zero length",xlabel="time (s)",ylabel="ratio"); ax[0,1].legend()
    labels=["M1 residual","period CV","length drift","pivot drift"]
    vals=[metrics.get("m1_abs_residual"),metrics.get("maximum_period_cv"),metrics.get("string_length_relative_change_p95_max"),metrics.get("pivot_drift_short_length_fraction_p95")]
    ax[1,0].bar(labels,[0 if v is None else v for v in vals],color=['tab:red','tab:purple','tab:green','tab:brown']); ax[1,0].tick_params(axis='x',rotation=20); ax[1,0].set_title("Normalized residual diagnostics")
    ax[1,1].axis('off'); ax[1,1].text(.02,.95,json.dumps(js(status),indent=2),va='top',family='monospace',fontsize=12)
    fig.savefig(path,dpi=130); plt.close(fig)


def render_overlay(path,frames,times,tr,status,metrics,rate):
    out=av.open(str(path),'w'); stream=out.add_stream('libx264',rate=max(1,int(round(rate))))
    stream.width=frames.shape[2]; stream.height=frames.shape[1]; stream.pix_fmt='yuv420p'; stream.options={'crf':'20','preset':'veryfast'}
    history={"short":[],"long":[]}; colors={"short":(0,180,255),"long":(255,160,0)}
    for i,rgb in enumerate(frames):
        im=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        for slot in ("short","long"):
            x=tr[slot]; b=x["bob_xy"][i]; p=x["pivot_xy"][i]
            if np.isfinite(b).all() and np.isfinite(p).all():
                history[slot].append(tuple(np.round(b).astype(int))); history[slot]=history[slot][-35:]
                cv2.line(im,tuple(np.round(p).astype(int)),tuple(np.round(b).astype(int)),colors[slot],2)
                cv2.circle(im,tuple(np.round(p).astype(int)),5,colors[slot],-1)
                cv2.circle(im,tuple(np.round(b).astype(int)),10,colors[slot],2)
                if len(history[slot])>1: cv2.polylines(im,[np.asarray(history[slot],np.int32)],False,colors[slot],1)
        cv2.rectangle(im,(0,0),(700,64),(0,0,0),-1)
        cv2.putText(im,f"P14 t={times[i]:.2f}s  structural={status['structural_ok']} measurement={status['measurement_valid']} physics={status['physics_pass']}",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        m=metrics.get('m1_abs_residual'); label='null' if m is None else f'{m:.4f}'
        cv2.putText(im,f"M1 abs residual={label}",(8,49),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
        vf=av.VideoFrame.from_ndarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),format='rgb24')
        for pkt in stream.encode(vf): out.mux(pkt)
    for pkt in stream.encode(): out.mux(pkt)
    out.close()


def tracking_metadata(data):
    meta=json.loads(str(np.asarray(data['meta_json']).item()))
    if meta.get('errors'):
        raise RuntimeError('Tracking backend failed: '+json.dumps(meta['errors'])[:3000])
    return meta


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True); ap.add_argument('--task_id',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--config',default=str(HERE/'config.yaml'))
    ap.add_argument('--debug-dir'); ap.add_argument('--device'); ap.add_argument('--force-tracking',action='store_true'); ap.add_argument('--no-debug',action='store_true')
    a=ap.parse_args()
    if a.task_id!='P14': raise SystemExit('this evaluator only supports --task_id P14')
    cfg=yaml.safe_load(Path(a.config).read_text())
    cfg['runtime']=resolve_tracking_runtime(cfg['runtime'])
    video=Path(a.video).resolve(); output=Path(a.output).resolve(); output.parent.mkdir(parents=True,exist_ok=True)
    sample=sample_id_from_name(video.stem)
    if sample not in cfg['samples']: raise SystemExit(f'no frozen first-frame configuration for {sample}')
    debug=Path(a.debug_dir).resolve() if a.debug_dir else output.parent/(output.stem+'_debug'); debug.mkdir(parents=True,exist_ok=True)
    frames,times=decode_video(video); dt=float(np.median(np.diff(times))) if len(times)>1 else 1/24
    cache=debug/'tracks_raw.npz'
    if a.force_tracking or not cache.exists():
        npy=debug/'frames_rgb.npy'; np.save(npy,frames)
        job={"frames":str(npy),"output":str(cache),"seeds":cfg['samples'][sample],
             "device":a.device or cfg['runtime']['default_device'],
             "cotracker_source":cfg['runtime']['cotracker_source'],"cotracker_checkpoint":cfg['runtime']['cotracker_checkpoint']}
        job.update({"sam2_path":cfg['runtime']['sam2_path'],"sam2_dtype":cfg['runtime']['sam2_dtype']})
        jobpath=debug/'track_job.json'; jobpath.write_text(json.dumps(job,indent=2))
        proc=subprocess.run([cfg['runtime']['python_track'],str(HERE/'track_worker_p14.py'),str(jobpath)],capture_output=True,text=True)
        try: npy.unlink()
        except OSError: pass
        if proc.returncode not in (0,2): raise RuntimeError(f'tracking process crashed: {proc.stderr[-3000:]}')
    data=np.load(cache,allow_pickle=False); meta=tracking_metadata(data); meta.update({"median_dt_s":dt,"frame_count":len(frames),"duration_s":float(times[-1]-times[0]),"track_cache":str(cache)})
    cfg['_active_sample']=sample
    cam=camera_audit(frames); tr=track_summary(data,times,cfg)
    status,metrics,dims,overall,errors,warnings=status_and_scores(tr,cam,meta,cfg)
    config_hash=hashlib.sha256(Path(a.config).read_bytes()).hexdigest()
    result={"task_id":"P14","video":str(video),"sample_id":sample,"evaluator_version":cfg['evaluator_version'],"config_sha256":config_hash,
            "status":status,"overall_score":overall,"dimension_scores":dims,"metrics":metrics,
            "pendulums":{s:{k:js(v) for k,v in tr[s].items() if k not in ('bob_xy','pivot_xy','valid','lengths','theta')} for s in ('short','long')},
            "camera_audit":cam,"extraction":{"backend":"SAM2 segmentation + CoTracker coordinates (no LLM/VLM)","meta":meta,
            "coverage":{"short":tr['short']['coverage'],"long":tr['long']['coverage']}},"errors":errors,"warnings":warnings,
            "debug":{"overlay":None,"plot":None}}
    # Retain the frozen status labels, then expose a separate continuous score.
    result=continuous_score_record(result)
    if not a.no_debug:
        plot=debug/'diagnostics.png'; overlay=debug/'overlay.mp4'
        render_plot(plot,times,tr,metrics,status)
        render_overlay(overlay,frames,times,tr,status,metrics,1/dt)
        result['debug']={"overlay":str(overlay),"plot":str(plot)}
    output.write_text(json.dumps(js(result),indent=2,ensure_ascii=False))
    print(json.dumps({"output":str(output),"status":status,"overall_score":overall,"m1":metrics['m1_abs_residual']},ensure_ascii=False))


if __name__=='__main__': main()
