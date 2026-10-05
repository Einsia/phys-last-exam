"""P5 - two identical steel balls in a head-on near-elastic collision.

Benchmark metrics, taken verbatim:
  M1  v'_1 + v'_2 - v_1: with equal masses and the right ball initially at rest,
      momentum conservation forces the two outgoing speeds to add up to the
      incoming one.
  The benchmark lists no auxiliary metric for this task, so M2 is reported as
  not defined rather than as a failed extraction.

The residual is normalised by the incoming speed. The raw difference is in
pixels per frame, which is not comparable between clips of different scale or
speed; dividing by v_1 makes it the fractional momentum defect and leaves the
zero of the metric where the benchmark puts it.
"""
from __future__ import annotations

import numpy as np

from .. import viz
from ..context import Context
from ..fitting import moving_phase, theil_sen
from ..schema import Result
from ..track import camera_drift, find_chroma_markers, track_target
from ..video import Clip

def velocity_quality(times, values, reference_speed=None):
    t=np.asarray(times,float); y=np.asarray(values,float)
    good=np.isfinite(t)&np.isfinite(y);t=t[good];y=y[good]
    info={'valid_points':int(len(t))}
    if len(t)<3:
        return None,dict(info,usable=False,reason='fewer than three valid points; no residual estimate')
    span=float(t[-1]-t[0]); info['time_span_sec']=span
    if span<=0:return None,dict(info,usable=False,reason='zero time span')
    slope,intercept=theil_sen(t,y)
    residual=y-(slope*t+intercept);rms=float(np.sqrt(np.mean(residual**2)))
    error=max(.35,rms);displacement=abs(float(slope)*span)
    scale=max(displacement,abs(reference_speed or 0)*span)
    usable=bool(np.isfinite(slope) and rms<=max(.75,.15*scale)
                and (reference_speed is not None or displacement>=3*error))
    info.update(usable=usable,displacement_px=displacement,localization_error_px=error,
                fit_residual_rms_px=rms,reason='resolved velocity' if usable else 'motion/residual signal insufficient')
    return float(slope),info


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "(v'_1 + v'_2) / v_1 - 1: momentum conservation",
                 principle=(
        "The two balls are identical and the right one starts at rest, so "
        "momentum conservation reads m*v_1 = m*v'_1 + m*v'_2 and the masses "
        "cancel: the outgoing speeds must sum to the incoming speed. Each speed "
        "is the slope of that ball's horizontal position against frame index, "
        "fitted separately before and after impact. The metric is the fractional "
        "defect (v'_1 + v'_2)/v_1 - 1, which is zero for a conserving collision "
        "whatever the pixel scale or frame rate."), tol=0.1)
    res.add("M2", defined=False)

    targets = find_chroma_markers(clip[0], k=2, chroma_min=28.0, min_area=60,
                                  max_area_frac=0.05)
    targets = sorted(targets, key=lambda t: t.x)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "balls_found_in_first_frame": len(targets),
                 "ball_seed_hues": [round(t.hue, 1) for t in targets]}
    if len(targets) < 2:
        res.fail_all(f"found {len(targets)} balls in the first frame, need 2")
        return res

    left, right = (track_target(clip, t) for t in targets)
    res.scene["ball_track_coverage"] = [round(left.coverage, 4),
                                        round(right.coverage, 4)]

    t = np.arange(clip.n, dtype=float) / clip.fps
    # Impact is where the initially resting ball starts to move. That is a
    # single unambiguous event in the clip and needs no contact-geometry model.
    impact, _ = moving_phase(_filled(right.x))
    res.scene["impact_frame"] = int(impact)

    M1.steps = [
        "segment each ball by its own colour, sampled from the first frame, and "
        "take its sub-pixel centroid every frame",
        "impact frame = the frame at which the initially resting right ball "
        "starts to move",
        "fit x(t) with a median-of-slopes line separately before and after "
        "impact for each ball; the slope is that ball's speed in px/s",
        "M1 = (v'_left + v'_right) / v_left - 1",
    ]

    pre = slice(0, max(impact, 0))
    post = slice(min(impact + 2, clip.n), clip.n)
    v1,q1=velocity_quality(t[pre],left.x[pre])
    if v1 is None or not q1['usable']:
        M1.fail('Incoming trajectory quality insufficient',trajectory_quality={'left_before':q1})
        return _finish(res,ctx,clip,left,right,impact,None,None,None)
    v2,q2=velocity_quality(t[pre],right.x[pre],v1)
    v1p,q1p=velocity_quality(t[post],left.x[post],v1)
    v2p,q2p=velocity_quality(t[post],right.x[post],v1)
    quality={'left_before':q1,'right_before':q2,'left_after':q1p,'right_after':q2p}
    res.scene['trajectory_quality']=quality
    if not all(q['usable'] for q in quality.values()):
        M1.fail('Collision trajectory quality insufficient',trajectory_quality=quality)
        return _finish(res,ctx,clip,left,right,impact,None,None,None)

    if not all(np.isfinite(value) for value in (v1, v2, v1p, v2p)):
        raise ArithmeticError('Non-finite collision velocity despite sufficient finite observations')
    if abs(v1) < 0.05:
        M1.fail(f"incoming speed {v1:.4f} px/s is too small to normalise "
                "the momentum residual", v_left_before=float(v1))
    else:
        M1.succeed((v1p + v2p) / v1 - 1.0,
                   v_left_before_px_per_sec=float(v1),
                   v_right_before_px_per_sec=float(v2),
                   v_left_after_px_per_sec=float(v1p),
                   v_right_after_px_per_sec=float(v2p),
                   outgoing_sum_px_per_sec=float(v1p + v2p),
                   trajectory_quality=quality,
                   impact_frame=int(impact),
                   frames_before=int(pre.stop - pre.start),
                   frames_after=int(post.stop - post.start))
    return _finish(res, ctx, clip, left, right, impact, v1, v1p, v2p)


def _filled(x: np.ndarray) -> np.ndarray:
    """Carry the last known value across gaps so a phase split can be found."""
    out = np.array(x, float)
    last = np.nan
    for i, v in enumerate(out):
        if np.isfinite(v):
            last = v
        else:
            out[i] = last
    return np.nan_to_num(out, nan=float(np.nanmin(out)) if np.isfinite(out).any()
                         else 0.0)


def _finish(res, ctx, clip, left, right, impact, v1, v1p, v2p):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, left, right, impact,
                                 v1, v1p, v2p, res)
    return res


def _debug(clip, ctx, left, right, impact, v1, v1p, v2p, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    t = np.arange(clip.n, dtype=float)

    viz.show_frame(ax[0], clip[min(impact, clip.n - 1)],
                   f"impact frame {impact}: tracked paths")
    ax[0].plot(left.x, left.y, "-", lw=1.4, color="#ff3b30", label="left ball")
    ax[0].plot(right.x, right.y, "-", lw=1.4, color="#0a84ff",
               label="right ball")
    ax[0].legend(loc="lower left", fontsize=7)

    ax[1].plot(t, left.x, ".-", ms=2, color="#ff3b30", label="left x(t)")
    ax[1].plot(t, right.x, ".-", ms=2, color="#0a84ff", label="right x(t)")
    ax[1].axvline(impact, color="#ffd400", lw=1.2, ls="--",
                  label=f"impact = frame {impact}")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("x  [px]")
    ax[1].set_title("horizontal positions; slopes are the speeds")
    ax[1].legend(fontsize=7)

    if v1 is not None and np.isfinite(v1):
        names = ["v_left before", "v_left after", "v_right after",
                 "sum after", "v_left before"]
        vals = [v1, v1p, v2p, v1p + v2p, v1]
        ax[2].bar(range(3), vals[:3],
                  color=["#ff3b30", "#ff9f0a", "#0a84ff"])
        ax[2].bar([3.4, 4.4], [v1p + v2p, v1], color=["#34c759", "#8e8e93"])
        ax[2].set_xticks([0, 1, 2, 3.4, 4.4])
        ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax[2].axhline(0, color="#000", lw=0.8)
        ax[2].set_ylabel("speed  [px/s]")
        ax[2].set_title("outgoing sum should equal incoming")
    m = res.metrics["M1"]
    cap = (f"{ctx.task_id}  M1 (v'_1+v'_2)/v_1 - 1 = "
           f"{'n/a' if m.value is None else f'{m.value:+.4f}'}")
    if v1 is not None and np.isfinite(v1):
        cap += (f"   (v_1 = {v1:+.2f}, v'_1 = {v1p:+.2f}, v'_2 = {v2p:+.2f}, "
                f"sum = {v1p + v2p:+.2f} px/s)")
    return viz.save(fig, ctx.debug(), cap)
