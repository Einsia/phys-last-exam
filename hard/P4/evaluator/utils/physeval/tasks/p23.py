"""P23 - a still liquid surface beside a freely falling ball.

Benchmark metrics, taken verbatim:
  M1  angle(a_ball, t_surface) - 90 degrees: gravity is perpendicular to the
      free surface of a liquid at rest, so the ball's acceleration direction and
      the surface tangent must meet at a right angle.
  M2  the straight-line fit residual of the liquid surface,
  M3  the constant-acceleration fit residual of the ball trajectory.
      The benchmark lists these two auxiliary quantities, so each takes a slot.

Both directions are measured in the same image, so no calibration enters: the
angle between two image directions is what the metric asks for.
"""
from __future__ import annotations

import numpy as np

from .. import liquid, viz
from ..context import Context
from ..fitting import longest_finite_run, moving_phase, polyfit
from ..schema import Result
from ..track import camera_drift, track_moving_blob
from ..video import Clip

MIN_FALL_FRAMES = 12
MIN_FALL_PX = 25.0


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "angle(a_ball, surface tangent) - 90 deg", principle=(
        "A liquid at rest settles so that its free surface is perpendicular to "
        "gravity. A freely falling ball accelerates along gravity. The angle "
        "between the ball's acceleration direction and the surface tangent is "
        "therefore 90 degrees, and the metric is the signed departure from it. "
        "Both directions are image directions measured in the same frame, so "
        "the comparison needs no calibration."), tol=5, unit="deg")
    M2 = res.add("M2", "straight-line fit residual of the liquid surface",
                 principle=(
        "The surface is located inside the vessel and fitted with a straight "
        "line; the RMS distance of the per-column surface height from that line "
        "says how flat the rendered surface actually is."), tol=3, unit="px")
    M3 = res.add("M3", "constant-acceleration fit residual of the ball path",
                 principle=(
        "A free fall obeys y(t) = y0 + v t + a t^2/2. Fitting that quadratic to "
        "the tracked centroid and taking the RMS residual in pixels says how "
        "closely the motion is uniformly accelerated, which is the premise "
        "behind treating the path direction as the acceleration direction."), tol=8, unit="px")

    # --- liquid surface ------------------------------------------------------
    # Measured on the first frame: the benchmark specifies a liquid already at
    # rest, and the ball never enters it, so the surface does not evolve.
    cont = liquid.find_container(clip[0])
    if cont is None:
        res.fail_all("SAM2 returned no usable vessel in the first frame")
        return res
    surf = liquid.find_waterline(clip[0], cont)
    if surf is None:
        res.fail_all("no liquid surface found inside the vessel")
        return res

    res.scene = {
        "camera_drift_frac_diag": camera_drift(clip),
        "vessel_bbox_xyxy": [cont.x0, cont.y0, cont.x1, cont.y1],
        "surface_y_px": round(surf.y, 2),
        "surface_tilt_deg": round(surf.angle_deg, 4),
        "surface_columns": [surf.x0, surf.x1],
        "extractor": "SAM2 vessel mask, then the strongest horizontal CIELAB "
                     "colour step inside its upper half",
    }
    M2.steps = [
        "segment the vessel in the first frame with SAM2 from one point prompt",
        "inside the vessel, score each row by the CIELAB colour step between "
        "the band just below and the band just above it",
        "take the strongest row in the vessel's upper half, then refine to "
        "sub-pixel height per column by the centre of mass of that step",
        "fit a straight line to the per-column heights",
        "M2 = RMS residual of that fit, in pixels",
    ]
    M2.succeed(surf.rms_px, surface_tilt_deg=surf.angle_deg,
               surface_y_px=surf.y, columns_used=surf.x1 - surf.x0,
               colour_step=surf.step)

    # --- falling ball --------------------------------------------------------
    tr = track_moving_blob(clip)
    a0, b0 = longest_finite_run(tr.y)
    ball_steps = [
        "take the static scene as the temporal median frame and keep, per "
        "frame, the compact blob that differs from it nearest the previous "
        "position; its intensity-weighted centroid is the ball",
        "restrict to the free-fall phase: from the first moving frame to the "
        "lowest point reached",
    ]
    M1.steps = ball_steps + [
        "fit x(t) and y(t) with quadratics; the acceleration direction is "
        "(d2x/dt2, d2y/dt2), i.e. twice the quadratic coefficients",
        "surface tangent comes from the fitted surface line",
        "M1 = angle between the two directions, minus 90 degrees",
    ]
    M3.steps = ball_steps + [
        "fit y(t) = y0 + v t + a t^2 / 2",
        "M3 = RMS residual of that fit, in pixels",
    ]

    if b0 - a0 < MIN_FALL_FRAMES:
        msg = f"ball tracked for only {b0 - a0} consecutive frames"
        M1.fail(msg)
        M3.fail(msg)
        return _finish(res, ctx, clip, cont, surf, tr, None, None)

    seg_y = tr.y[a0:b0]
    ma, _ = moving_phase(seg_y)
    landing = int(np.nanargmax(seg_y))
    a, b = a0 + ma, a0 + max(landing + 1, ma + 4)
    fall = float(tr.y[b - 1] - tr.y[a])
    res.scene["free_fall_phase_frames"] = [int(a), int(b)]
    res.scene["ball_track_coverage"] = round(tr.coverage, 4)
    res.scene["total_fall_px"] = round(fall, 1)

    if b - a < MIN_FALL_FRAMES or fall < MIN_FALL_PX:
        msg = (f"the free-fall phase is {b - a} frames covering {fall:.1f} px, "
               "too short to fit an acceleration")
        M1.fail(msg)
        M3.fail(msg)
        return _finish(res, ctx, clip, cont, surf, tr, (a, b), None)

    t = np.arange(b - a, dtype=float)
    fx = polyfit(t, tr.x[a:b], 2)
    fy = polyfit(t, tr.y[a:b], 2)
    acc = np.array([2.0 * fx.coef[0], 2.0 * fy.coef[0]])
    norm = float(np.linalg.norm(acc))

    M3.succeed(fy.rms, quadratic_coefficients_y=fy.coef.tolist(),
               accel_y_px_per_frame2=float(2.0 * fy.coef[0]),
               accel_x_px_per_frame2=float(2.0 * fx.coef[0]),
               fall_frames=int(b - a), total_fall_px=fall)

    if norm < 1e-3:
        M1.fail(f"the fitted acceleration is {norm:.5f} px/frame^2, so the ball "
                "has no measurable acceleration direction",
                accel_magnitude_px_per_frame2=norm)
    else:
        u = acc / norm
        cosang = abs(float(u @ surf.tangent))
        between = float(np.degrees(np.arccos(min(1.0, cosang))))
        M1.succeed(between - 90.0,
                   angle_between_deg=between,
                   accel_direction_deg=float(np.degrees(np.arctan2(u[1], u[0]))),
                   surface_tangent_deg=surf.angle_deg,
                   accel_magnitude_px_per_frame2=norm)
    return _finish(res, ctx, clip, cont, surf, tr, (a, b), (fx, fy, acc))


def _finish(res, ctx, clip, cont, surf, tr, span, fits):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, cont, surf, tr, span, fits, res)
    return res


def _debug(clip, ctx, cont, surf, tr, span, fits, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    last = span[1] - 1 if span else clip.n - 1

    viz.show_frame(ax[0], clip[last], "vessel, surface line and ball path")
    xs = np.arange(surf.x0, surf.x1)
    ax[0].plot(xs, [surf.y_at(x) for x in xs], "-", color="#ffd400", lw=2.2,
               label=f"surface, tilt {surf.angle_deg:+.3f} deg")
    ax[0].add_patch(__import__("matplotlib").patches.Rectangle(
        (cont.x0, cont.y0), cont.width, cont.height, fill=False,
        edgecolor="#34c759", lw=1.2, label="SAM2 vessel"))
    if span:
        a, b = span
        ax[0].plot(tr.x[a:b], tr.y[a:b], "-", color="#00e5ff", lw=1.8,
                   label="ball path")
        if fits is not None:
            _, _, acc = fits
            u = acc / max(np.linalg.norm(acc), 1e-9)
            x0, y0 = tr.x[a], tr.y[a]
            ax[0].arrow(x0, y0, 130 * u[0], 130 * u[1], color="#ff3b30",
                        width=3, label="acceleration direction")
    ax[0].legend(loc="lower left", fontsize=7)

    prof = surf.profile
    ax[1].plot(prof, np.arange(prof.size), "-", color="#0a84ff", lw=1.0)
    ax[1].axhline(surf.y, color="#ffd400", lw=1.4,
                  label=f"surface y = {surf.y:.1f}")
    ax[1].axhspan(cont.y0, cont.y1, color="#34c759", alpha=0.08,
                  label="vessel extent")
    ax[1].invert_yaxis()
    ax[1].set_xlabel("horizontal colour step")
    ax[1].set_ylabel("row  [px]")
    ax[1].set_title("where the surface was found")
    ax[1].legend(fontsize=7)

    if span and fits is not None:
        a, b = span
        t = np.arange(b - a, dtype=float)
        fx, fy, _ = fits
        ax[2].plot(t, tr.y[a:b], ".", ms=3, color="#0a84ff", label="ball y(t)")
        ax[2].plot(t, fy(t), "-", color="#ff9500", lw=1.3,
                   label=f"constant-a fit, RMS {fy.rms:.2f} px")
        ax[2].set_xlabel("frame from release")
        ax[2].set_ylabel("y  [px]")
        ax[2].legend(fontsize=7)
    ax[2].set_title("free-fall fit")

    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:+.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 angle-90 = {fmt('M1')} deg   |   "
                    f"M2 surface fit RMS = {fmt('M2')} px   "
                    f"M3 free-fall fit RMS = {fmt('M3')} px")
