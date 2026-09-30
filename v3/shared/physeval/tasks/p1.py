"""P1 - a dense steel ball released from rest falls freely.

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
from ..track import camera_drift, track_moving_blob
from ..video import Clip

# The ball must move at least this many pixels in total for the fall to be a
# fall rather than tracker noise.
MIN_FALL_PX = 30.0
MIN_FRAMES = 12
# Velocity is measured over multi-frame intervals rather than frame pairs. A
# single-frame difference of a sub-pixel centroid is dominated by codec noise,
# and taking a second difference of it doubles that noise, so the interval is
# made long enough that the real velocity change is well above it.
MIN_INTERVAL_FRAMES = 10
N_INTERVALS = 6


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

    tr = track_moving_blob(clip)
    a0, b0 = longest_finite_run(tr.y)
    # The benchmark measures the free fall only: the clip may hold the ball at
    # rest before release and may show it resting after it lands, and both would
    # corrupt an acceleration measured across them.
    # Release is where motion begins; the fall ends at the lowest point the ball
    # reaches, which is the contact instant when it lands and the final frame
    # when it does not. Using the extremum rather than a speed threshold keeps a
    # small bounce after contact out of the measured interval.
    ma, _ = moving_phase(tr.y[a0:b0])
    seg = tr.y[a0:b0]
    landing = int(np.nanargmax(seg))
    a, b = a0 + ma, a0 + max(landing + 1, ma + 4)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "track_coverage": tr.coverage,
                 "tracked_span_frames": [int(a0), int(b0)],
                 "free_fall_phase_frames": [int(a), int(b)],
                 "phase_note": ("frames outside the free-fall phase (held at "
                                "rest, or resting after landing) are excluded"),
                 "extractor": tr.quantities}

    if b - a < MIN_FRAMES:
        msg = f"ball tracked for only {b - a} consecutive frames"
        M1.fail(msg, tracked_frames=int(b - a))
        M2.fail(msg, tracked_frames=int(b - a))
        return _finish(res, ctx, clip, tr, a, b, None, None, None)

    y = tr.y[a:b]
    # Image y grows downward, so falling is increasing y; flip to make the
    # measured displacement positive downward-as-positive-fall.
    fall = y - y[0]
    if float(fall[-1]) < MIN_FALL_PX:
        msg = (f"ball fell only {float(fall[-1]):.1f} px over the tracked span, "
               "no measurable fall")
        M1.fail(msg, total_fall_px=float(fall[-1]))
        M2.fail(msg, total_fall_px=float(fall[-1]))
        return _finish(res, ctx, clip, tr, a, b, fall, None, None)

    # --- M1: CV of velocity increments over equal intervals -----------------
    span = b - a
    k = max(MIN_INTERVAL_FRAMES, span // N_INTERVALS)
    edges = np.arange(0, span, k)
    if edges.size < 4:
        M1.fail(f"tracked span {span} frames only yields {edges.size - 1} "
                    f"intervals of {k} frames, need at least 3 to form "
                    "two velocity increments")
        v = dv = np.asarray([], float)
    else:
        # Mean velocity in each equal-length interval, from its endpoints.
        v = np.asarray([(fall[edges[i + 1]] - fall[edges[i]]) / k
                        for i in range(edges.size - 1)], float)
        dv = np.diff(v)
        M1.steps = [
            "take the static scene as the temporal median frame, then per frame "
            "keep the compact blob that differs from it and is nearest the "
            "previous position; its intensity-weighted centroid is the ball",
            f"use the longest continuous tracked span, frames {a}-{b - 1}",
            f"split it into equal intervals of {k} frames "
            f"({edges.size - 1} intervals)",
            "v_y[j] = (y at interval end - y at interval start) / interval length",
            "dv_y[j] = v_y[j+1] - v_y[j]",
            "M1 = std(dv_y) / |mean(dv_y)|; constant acceleration gives 0",
        ]
        m1 = cv(dv)
        if np.isfinite(m1):
            M1.succeed(m1, interval_frames=int(k),
                           n_intervals=int(v.size),
                           interval_velocities_px_per_frame=v.tolist(),
                           dv_y_px_per_frame=dv.tolist(),
                           mean_dv_y=float(np.mean(dv)),
                           std_dv_y=float(np.std(dv, ddof=1)),
                           total_fall_px=float(fall[-1]))
            M1.note = (
                "a large value means the velocity increments are not consistent, "
                "i.e. the fall is not uniformly accelerated; CV grows without "
                "bound as the mean increment approaches zero, which is what a "
                "constant-velocity fall produces")
        else:
            M1.fail("fewer than two velocity increments, CV undefined",
                        n_intervals=int(v.size))

    # --- M2: 1:3:5 displacement ratio ---------------------------------------
    # Three equal windows measured from the first tracked frame, which is the
    # release instant because the ball is at rest before it.
    # (span-1)//3 so the third window's end index is the last tracked frame.
    k = (span - 1) // 3
    if k >= 3:
        d1 = float(fall[k] - fall[0])
        d2 = float(fall[2 * k] - fall[k])
        d3 = float(fall[3 * k] - fall[2 * k])
        M2.steps = [
            f"split the tracked span into three equal windows of {k} frames",
            "dy1, dy2, dy3 = vertical displacement in each window",
            "compare dy2/dy1 and dy3/dy1 against the from-rest law 3 and 5",
            "M2 = sqrt(mean(((dy2/dy1)/3 - 1)^2, ((dy3/dy1)/5 - 1)^2))",
        ]
        if d1 > 1.0:
            r2, r3 = d2 / d1, d3 / d1
            m2 = float(np.sqrt(np.mean([(r2 / 3.0 - 1) ** 2,
                                        (r3 / 5.0 - 1) ** 2])))
            M2.succeed(m2, window_frames=int(k),
                           dy1_px=d1, dy2_px=d2, dy3_px=d3,
                           ratio_dy2_dy1=r2, ratio_dy3_dy1=r3,
                           expected_ratios=[3.0, 5.0])
        else:
            M2.fail(f"first window displacement {d1:.1f} px is too small "
                        "to normalise the ratio", dy1_px=d1)
    else:
        M2.fail(f"tracked span {span} frames gives windows of {k} frames, "
                    "too short for a three-window ratio")

    return _finish(res, ctx, clip, tr, a, b, fall, v, dv)


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
