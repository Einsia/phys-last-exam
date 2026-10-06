"""P2 - a dense steel ball released from rest falls freely.

Benchmark metrics, taken verbatim:
  M1  CV(dv_y): the vertical velocity increment over equal time intervals should
      be near constant, so its coefficient of variation is the metric.
  M2  dy1 : dy2 : dy3 ~ 1 : 3 : 5, the displacement ratio over three consecutive
      equal intervals starting from rest.

Both come from one trajectory: the ball's sub-pixel centroid per frame. No
calibration is needed because both metrics are ratios of pixel displacements.
"""
from __future__ import annotations

import numpy as np

from .. import viz
from ..context import Context
from ..fitting import cv, longest_finite_run, moving_phase
from ..schema import Result
from ..track import camera_drift
from .observed_ball import track_ball
from ..video import Clip

# The ball must move at least this many pixels in total for the fall to be a
# fall rather than tracker noise.
MIN_FALL_PX = 6.0
MIN_FRAMES = 4
# Comparison intervals adapt to the complete observed flight; no fixed frame window.
N_INTERVALS = 6


def full_fall_measurement(values, fps):
    """Measured endpoint velocities over the whole observed motion, without law fitting."""
    y = np.asarray(values, float)
    if len(y) < 4 or not np.isfinite(y).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Need at least four finite motion observations with valid times')
    t = np.arange(len(y), dtype=float) / fps
    coef = np.polyfit(t, y, min(3, len(y)-2))
    residual = y - np.polyval(coef, t)
    noise = max(.25, float(1.4826*np.median(abs(residual-np.median(residual)))))
    displacement = float(y[-1]-y[0])
    if displacement < max(MIN_FALL_PX, 6*noise):
        raise ValueError('Motion displacement is not resolved relative to localization/fit residual')
    count = min(N_INTERVALS, max(3, (len(y)-1)//2))
    edges = np.linspace(t[0], t[-1], count+1)
    positions = np.interp(edges, t, y)
    velocities = np.diff(positions)/np.diff(edges)
    dv = np.diff(velocities)
    # A well observed constant velocity is a physical violation, not missing data.
    increment = float(np.mean(dv))
    velocity_noise = np.sqrt(2)*noise/(edges[1]-edges[0])
    if abs(increment) <= velocity_noise:
        m1 = max(1., float(np.std(dv))/max(velocity_noise,1e-9))
    else:
        m1 = float(np.std(dv,ddof=1)/abs(increment))
    thirds = np.interp(np.linspace(t[0],t[-1],4),t,y)
    d = np.diff(thirds)
    m2 = None if d[0] <= noise else float(np.sqrt(np.mean((d[1:]/d[0]/[3.,5.]-1.)**2)))
    details = dict(valid_points=len(y), duration_sec=float(t[-1]-t[0]),
                   displacement_px=displacement, localization_error_px=noise,
                   fit_residual_rms_px=float(np.sqrt(np.mean(residual**2))),
                   comparison_times_sec=edges.tolist(), velocities_px_per_sec=velocities.tolist(),
                   velocity_increments_px_per_sec=dv.tolist(), thirds_displacement_px=d.tolist(),
                   policy='complete_motion_span_adaptive_equal_time_intervals_observed_positions')
    return m1,m2,details


def first_contact_end(y, start):
    """End at first resolved reversal or landing plateau, otherwise last observation."""
    y=np.asarray(y,float)
    noise=.5
    for i in range(start+3,len(y)-2):
        if y[i]-y[start] < MIN_FALL_PX:continue
        before=float(np.median(np.diff(y[max(start,i-3):i+1])))
        after=y[i+1:min(len(y),i+4)]-y[i]
        if before > noise and (float(np.median(after)) < -noise or float(np.max(abs(after))) <= noise):
            return i+1
    return len(y)


def observed_descent_span(values):
    """Use the first continuous observed descent, never the highest score.

    A longer stationary segment after an occlusion must not replace an already
    observed fall. Missing frames are boundaries; no span bridges a gap. The
    existing duration/displacement limits define observability only.
    """
    y = np.asarray(values, dtype=float)
    edges = np.diff(np.r_[False, np.isfinite(y), False].astype(np.int8))
    runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    for start, stop in runs:
        if stop - start >= MIN_FRAMES and np.max(y[start:stop]) - y[start] >= MIN_FALL_PX:
            return int(start), int(stop)
    return longest_finite_run(y)



def fit_released_motion(values, motion_hint):
    """Infer release from observed pixels using a held-then-cubic trajectory.

    A cubic term remains free: nonconstant acceleration is not projected onto
    the correct quadratic law. Noise comes from high-frequency fit residuals,
    and non-polynomial systematic motion is retained as an error term.
    """
    y=np.asarray(values,float)
    if y.ndim!=1 or len(y)<MIN_FRAMES or not np.isfinite(y).all():raise ValueError('Release fit requires a sufficiently long continuously observed finite trajectory')
    t=np.arange(len(y),dtype=float)
    max_release=min(max(1,int(motion_hint)+4),len(y)//3)
    candidates=[]
    for release in np.arange(0,max_release+.01,.5):
        u=np.maximum(t-release,0)/max(len(y)-1-release,1)
        X=np.column_stack([np.ones(len(y)),u*u,u*u*u]);weights=np.ones(len(y))
        for _ in range(5):
            coef=np.linalg.lstsq(X*weights[:,None]**.5,y*weights**.5,rcond=None)[0]
            residual=y-X@coef;scale=max(.25,1.4826*np.median(abs(residual-np.median(residual))))
            weights=np.minimum(1,1.5*scale/np.maximum(abs(residual),1e-9))
        loss=float(np.mean(np.minimum(residual**2,(3*scale)**2)))
        candidates.append((loss,float(release),X@coef,coef,residual))
    _,release,fit,coef,residual=min(candidates,key=lambda z:z[0])
    noise=float(np.clip(1.4826*np.median(abs(np.diff(residual)-np.median(np.diff(residual))))/np.sqrt(2),.15,1.5))
    rms=float(np.sqrt(np.mean(residual**2)))
    return int(round(release)),fit,{'release_frame_relative':release,'position_fit_rms_px':rms,'estimated_observation_noise_px':noise,'cubic_coefficient_px':float(coef[2]),'fit_model':'held position then free quadratic+cubic; no ground-truth input','systematic_position_rms_px':float(np.sqrt(max(0,rms*rms-noise*noise)))}


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "CV(dv_y): equal-interval vertical velocity increments "
                       "should be constant", principle=(
        "Free fall has constant acceleration, so the change in vertical velocity "
        "over successive equal time intervals is the same every time. The metric "
        "is CV(dv_y) = std/mean of those increments; exact free fall gives 0."), tol=0.1)
    M2 = res.add("M2", "dy1:dy2:dy3 ~ 1:3:5", principle=(
        "Starting from rest, the distances fallen in three consecutive equal time "
        "intervals are in the ratio 1:3:5. The metric is the RMS relative "
        "deviation of the measured dy2/dy1 and dy3/dy1 from 3 and 5."), tol=0.1)

    tr = track_ball(clip)
    a0, b0 = observed_descent_span(tr.y)
    tr.quantities['span_selection'] = 'earliest continuously observed descent meeting existing duration/displacement limits; longest observed run only for failure diagnostics; no score-based selection'
    if b0-a0<MIN_FRAMES:
        message=f'Longest continuously observed ball trajectory has only {b0-a0} frames; need {MIN_FRAMES}'
        res.scene={'track_coverage':tr.coverage,'tracked_span_frames':[int(a0),int(b0)],'extractor':tr.quantities}
        M1.fail(message,tracked_frames=int(b0-a0));M2.fail(message,tracked_frames=int(b0-a0))
        return _finish(res,ctx,clip,tr,a0,b0,None,None,None)
    segment=tr.y[a0:b0]
    onset,_=moving_phase(segment)
    # Retain the observed position immediately preceding resolved motion.
    b=a0+first_contact_end(segment,max(0,int(onset)-1))
    # Estimate only the release boundary; speeds remain measured from pixels.
    release,_,release_details=fit_released_motion(tr.y[a0:b],onset)
    a=a0+release
    res.scene={'track_coverage':tr.coverage,'tracked_span_frames':[int(a0),int(b0)],
               'free_fall_phase_frames':[int(a),int(b)],'extractor':tr.quantities,
               'phase_note':'start of descent through first observed contact/reversal or final valid frame'}
    fall=tr.y[a:b]-tr.y[a] if b>a else None
    try:
        m1,m2,details=full_fall_measurement(tr.y[a:b],clip.fps)
    except ValueError as exc:
        M1.fail(str(exc));M2.fail(str(exc))
        return _finish(res,ctx,clip,tr,a,b,fall,None,None)
    res.scene['trajectory_quality']=details
    steps=['Track actual ball positions without bridging missing frames.',
           'Use the complete descent from release to first contact or last valid observation.',
           'Adapt equal-time comparison boundaries to the complete span; measure endpoint velocities in px/s.',
           'Assess observation count, duration, displacement relative to localization error and fit residual.']
    M1.steps=steps+['M1 is CV of measured interval velocity increments; resolved constant velocity is a violation.']
    M2.steps=steps+['M2 compares observed displacements across three equal parts of the complete span with 1:3:5.']
    M1.succeed(m1,**details)
    if m2 is None:
        M2.fail('First third displacement is below localization error',**details)
    else:
        M2.succeed(m2,**details)
    v=np.asarray(details['velocities_px_per_sec'])/clip.fps
    dv=np.diff(v)
    return _finish(res,ctx,clip,tr,a,b,fall,v,dv)


def _finish(res, ctx, clip, tr, a, b, fall, v, dv):
    # A figure is produced even when nothing could be measured: the reason a
    # sample failed is exactly what a reader needs to see, and an empty slot
    # tells them nothing.
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, tr, a, b, fall, v, dv, res)
    return res


def _debug(clip, ctx, tr, a, b, fall, v, dv, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)

    viz.show_frame(ax[0], clip[max(0, b - 1)],
                   f"trajectory on frame {max(0, b - 1)}")
    if b > a:
        ax[0].plot(tr.x[a:b], tr.y[a:b], "-", color="#00e5ff", lw=1.6,
                   label="tracked centroid")
        ax[0].scatter(tr.x[a:b:8], tr.y[a:b:8], s=14, color="#ff3b30", zorder=3)
        k = (b - a - 1) // 3
        for j, lab in ((0, "start"), (k, "1/3"), (2 * k, "2/3"), (3 * k, "end")):
            if a + j < b:
                ax[0].axhline(tr.y[a + j], color="#ffd400", lw=0.8, ls="--")
                ax[0].annotate(lab, (10, tr.y[a + j] - 6), color="#ffd400",
                               fontsize=7)
        ax[0].legend(loc="lower left", fontsize=7)
    else:
        ax[0].set_title(f"frame {max(0, b - 1)}: the ball was not tracked")

    if fall is None:
        for k in (1, 2):
            ax[k].set_axis_off()
        note = next((m.note for m in res.metrics.values() if m.note), "")
        ax[1].text(0.5, 0.5, note, ha="center", va="center", wrap=True,
                   fontsize=10, transform=ax[1].transAxes)
        ax[1].set_title("nothing measurable in this clip")
        return _save(fig, ctx, res)

    t = np.arange(fall.size)
    ax[1].plot(t, fall, ".-", ms=3, color="#0a84ff", label="measured fall  [px]")
    # A true from-rest free fall is a pure quadratic through the origin; drawing
    # it makes a constant-velocity fall obvious by eye.
    q = float(fall[-1]) / max(1.0, t[-1] ** 2)
    ax[1].plot(t, q * t ** 2, "--", color="#ff3b30", lw=1.2,
               label="from-rest free fall through same endpoint")
    ax[1].set_xlabel("frame (from first tracked)")
    ax[1].set_ylabel("distance fallen  [px]")
    ax[1].set_title("displacement vs the from-rest law")
    ax[1].legend(fontsize=7, loc="upper left")

    if v is not None and v.size:
        ax[2].step(np.arange(v.size), v, where="mid", color="#34c759",
                   marker="o", ms=4, label="interval velocity  [px/frame]")
        ax[2].plot(np.arange(dv.size) + 0.5, dv, "o-", ms=4, color="#ff9500",
                   label="dv_y between intervals")
        ax[2].axhline(0.0, color="#8e8e93", lw=0.8)
        if dv.size and np.isfinite(np.mean(dv)):
            ax[2].axhline(float(np.mean(dv)), color="#ff3b30", ls="--", lw=1.0,
                          label=f"mean dv_y = {np.mean(dv):.3f}")
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel("interval index")
    ax[2].set_title("velocity per interval and its increments")

    return _save(fig, ctx, res)


def _save(fig, ctx, res) -> str:
    M1, M2 = res.metrics["M1"], res.metrics["M2"]
    q = M2.quantities
    cap = (f"{ctx.task_id}  M1 CV(dv_y) = "
           f"{'n/a' if M1.value is None else f'{M1.value:.4f}'}"
           f"   |   M2 1:3:5 deviation = "
           f"{'n/a' if M2.value is None else f'{M2.value:.4f}'}")
    # Failed ratio extraction may retain only dy1_px. A partial measurement
    # must still produce its diagnostic figure and JSON result.
    if all(key in q for key in ("dy1_px", "dy2_px", "dy3_px",
                               "ratio_dy2_dy1", "ratio_dy3_dy1")):
        cap += (f"   (dy1:dy2:dy3 = {q['dy1_px']:.0f}:{q['dy2_px']:.0f}:"
                f"{q['dy3_px']:.0f} px, normalised 1:{q['ratio_dy2_dy1']:.2f}:"
                f"{q['ratio_dy3_dy1']:.2f} vs 1:3:5)")
    return viz.save(fig, ctx.debug(), cap)
