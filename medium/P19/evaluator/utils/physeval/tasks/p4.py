"""P4 - a ball launched at a stated angle, landing back at launch height.

Benchmark metrics, taken verbatim:
  M1  H/R - tan(theta)/4, where H is the apex height above the launch level and
      R the horizontal range. For ideal projectile motion H/R = tan(theta)/4
      exactly, independent of speed and of any pixel calibration.
  M2  three auxiliary quantities: CV(v_x), CV(dv_y) and the residual of a
      quadratic fit to the trajectory.

The launch angle is not measurable from a single frame, so it comes from the
prompt, which states it.
"""
from __future__ import annotations

import numpy as np

from .. import viz
from ..context import Context
from ..fitting import cv, longest_finite_run, moving_phase, polyfit
from ..schema import Result
from ..track import camera_drift, find_color_targets, track_target
from ..video import Clip

MIN_FLIGHT_FRAMES = 16
MIN_INTERVAL_FRAMES = 8
N_INTERVALS = 6
DEFAULT_THETA_DEG = 45.0


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    theta = float(ctx.params.get("theta_deg") or DEFAULT_THETA_DEG)
    target_ratio = np.tan(np.radians(theta)) / 4.0

    M1 = res.add("M1", "H/R - tan(theta)/4", principle=(
        f"For projectile motion launched at {theta:g} deg and landing at the "
        "launch height, the apex height over the range is H/R = tan(theta)/4 = "
        f"{target_ratio:.4f}. Both H and R are pixel lengths in the same frame, "
        "so the ratio needs no calibration. The metric is the signed difference "
        "H/R - tan(theta)/4."), tol=0.1)
    # The benchmark lists three auxiliary quantities for P4, so each takes its
    # own slot rather than being collapsed into one number.
    M2 = res.add("M2", "CV(v_x): horizontal velocity should be constant",
                 principle=("With no horizontal force the horizontal velocity "
                            "is constant, so the coefficient of variation of "
                            "the per-interval v_x is zero for ideal motion."), tol=0.1)
    M3 = res.add("M3", "CV(dv_y): equal-interval vertical velocity increments",
                 principle=("Gravity is constant, so the vertical velocity "
                            "changes by the same amount over each equal time "
                            "interval and CV(dv_y) is zero for ideal motion."), tol=0.1)
    M4 = res.add("M4", "quadratic-fit residual of the trajectory",
                 principle=("An ideal trajectory is a parabola y(x). The metric "
                            "is the RMS distance of the tracked points from the "
                            "least-squares parabola, divided by the range R so "
                            "it is scale-free."), tol=0.05)
    aux = (M2, M3, M4)

    # The clip paints a trail behind the ball, so the ball is segmented by its
    # own chroma rather than by motion: the trail is neutral grey, the ball is not.
    targets = find_color_targets(clip[0], k=1)
    if not targets:
        res.fail_all("no chromatic ball found in the first frame")
        return res

    tr = track_target(clip, targets[0])
    a0, b0 = longest_finite_run(tr.x)
    if b0 - a0 < MIN_FLIGHT_FRAMES:
        res.fail_all(f"ball tracked for only {b0 - a0} consecutive frames")
        return res

    x, y = tr.x[a0:b0], tr.y[a0:b0]
    # Flight starts when the ball leaves the ground and ends when it comes back
    # down to the launch level; frames before launch and after landing are
    # outside the arc the metric is defined on.
    start, _ = moving_phase(x)
    apex = int(np.nanargmin(y[start:])) + start
    y_launch = float(y[start])
    land = _landing_index(y, apex, y_launch)

    H = y_launch - float(y[apex])
    R = abs(float(x[land]) - float(x[start]))
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "track_coverage": tr.coverage,
                 "tracked_span_frames": [int(a0), int(b0)],
                 "flight_phase_frames": [int(a0 + start), int(a0 + land + 1)],
                 "launch_xy": [float(x[start]), y_launch],
                 "apex_xy": [float(x[apex]), float(y[apex])],
                 "landing_xy": [float(x[land]), float(y[land])],
                 "theta_deg_from_prompt": theta}

    M1.steps = [
        "segment the ball by its first-frame colour and take the sub-pixel "
        "centroid of the nearest connected component in every frame",
        "launch = first frame with horizontal motion; apex = frame of minimum "
        "image y; landing = frame where the descending path returns to the "
        "launch height (linearly interpolated between frames)",
        "H = y_launch - y_apex   (pixels, upward)",
        "R = |x_landing - x_launch|   (pixels)",
        f"M1 = H/R - tan({theta:g} deg)/4",
    ]
    if land - start < MIN_FLIGHT_FRAMES:
        M1.fail(f"flight phase is only {land - start} frames",
                    flight_frames=int(land - start))
    elif H <= 2.0 or R <= 2.0:
        M1.fail(f"degenerate arc: H={H:.1f} px, R={R:.1f} px",
                    apex_height_px=H, range_px=R)
    else:
        M1.succeed(H / R - target_ratio, apex_height_px=H, range_px=R,
                       measured_H_over_R=H / R,
                       expected_H_over_R=target_ratio,
                       flight_frames=int(land - start))

    _aux(aux, x[start:land + 1], y[start:land + 1], R)

    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, tr, a0, start, apex, land,
                                 x, y, H, R, target_ratio, res)
    return res


def _landing_index(y: np.ndarray, apex: int, y_launch: float) -> int:
    """First index after the apex where the ball is back at the launch level."""
    for i in range(apex + 1, y.size):
        if y[i] >= y_launch:
            return i
    return int(y.size - 1)


def _aux(aux, x: np.ndarray, y: np.ndarray, R: float) -> None:
    M2, M3, M4 = aux
    n = x.size
    k = max(MIN_INTERVAL_FRAMES, n // N_INTERVALS)
    edges = np.arange(0, n, k)
    common = [f"over the flight phase, split into equal intervals of {k} frames",
              "v_x, v_y = interval endpoint displacement / interval length"]
    M2.steps = common + ["M2 = std/|mean| of the interval v_x"]
    M3.steps = common + ["dv_y = differences of successive interval v_y",
                         "M3 = std/|mean| of dv_y"]
    M4.steps = ["fit y = a x^2 + b x + c to the tracked flight points",
                "M4 = RMS residual in pixels, divided by the range R"]

    if edges.size < 4 or R <= 2.0:
        msg = "flight phase too short to form three equal intervals"
        for m in aux:
            m.fail(msg)
        return

    vx = np.asarray([(x[edges[i + 1]] - x[edges[i]]) / k
                     for i in range(edges.size - 1)], float)
    vy = np.asarray([(y[edges[i + 1]] - y[edges[i]]) / k
                     for i in range(edges.size - 1)], float)
    fit = polyfit(x, y, 2)
    shared = {"interval_frames": int(k), "vx_px_per_frame": vx.tolist(),
              "vy_px_per_frame": vy.tolist()}

    for metric, value, extra in (
            (M2, cv(vx), {"mean_vx_px_per_frame": float(np.mean(vx))}),
            (M3, cv(np.diff(vy)),
             {"dvy_px_per_frame": np.diff(vy).tolist()}),
            (M4, fit.rms / R, {"parabola_rms_px": fit.rms, "range_px": R,
                               "parabola_coefficients": fit.coef.tolist()})):
        if np.isfinite(value):
            metric.succeed(value, **shared, **extra)
        else:
            metric.fail("quantity undefined: the mean it normalises by is zero",
                        **shared, **extra)


def _debug(clip, ctx, tr, a0, start, apex, land, x, y, H, R, target, res) -> str:
    M1 = res.metrics["M1"]
    fig, ax = viz.figure(ncols=3, width_each=5.0)

    viz.show_frame(ax[0], clip[min(a0 + land, clip.n - 1)],
                   "trajectory, H and R")
    xs, ys = x[start:land + 1], y[start:land + 1]
    ax[0].plot(xs, ys, "-", color="#00e5ff", lw=1.8, label="tracked arc")
    ax[0].plot([x[start], x[land]], [y[start], y[start]], "-",
               color="#ffd400", lw=1.6, label=f"R = {R:.0f} px")
    ax[0].plot([x[apex], x[apex]], [y[apex], y[start]], "-",
               color="#ff3b30", lw=1.6, label=f"H = {H:.0f} px")
    ax[0].scatter([x[start], x[apex], x[land]], [y[start], y[apex], y[land]],
                  s=26, color="#ffffff", edgecolor="k", zorder=4)
    ax[0].legend(loc="lower left", fontsize=7)

    ax[1].plot(xs, -ys, ".", ms=3, color="#0a84ff", label="tracked points")
    if xs.size > 3:
        fit = polyfit(xs, ys, 2)
        xg = np.linspace(xs.min(), xs.max(), 200)
        ax[1].plot(xg, -fit(xg), "-", color="#ff9500", lw=1.3,
                   label=f"parabola fit, RMS {fit.rms:.2f} px")
    ax[1].set_xlabel("x  [px]")
    ax[1].set_ylabel("height  [px, up]")
    ax[1].set_title("arc vs least-squares parabola")
    ax[1].legend(fontsize=7)

    q = res.metrics["M2"].quantities
    if q.get("vx_px_per_frame"):
        vx = np.asarray(q["vx_px_per_frame"], float)
        vy = np.asarray(q["vy_px_per_frame"], float)
        ax[2].plot(vx, "o-", ms=4, color="#34c759", label="v_x per interval")
        ax[2].plot(np.diff(vy), "o-", ms=4, color="#ff9500",
                   label="dv_y between intervals")
        ax[2].axhline(0, color="#8e8e93", lw=0.8)
        ax[2].legend(fontsize=7)
    ax[2].set_xlabel("interval index")
    ax[2].set_title("v_x flat and dv_y constant if projectile")

    m1 = "n/a" if M1.value is None else f"{M1.value:+.4f}"
    cap = (f"{ctx.task_id}  M1 = H/R - tan(theta)/4 = {m1}"
           f"   (H/R = {H / R:.4f} vs {target:.4f})")
    def fmt(key: str) -> str:
        m = res.metrics[key]
        return "n/a" if m.value is None else f"{m.value:.4f}"
    cap += (f"   |   M2 CV(v_x) = {fmt('M2')}   M3 CV(dv_y) = {fmt('M3')}   "
            f"M4 parabola resid = {fmt('M4')}")
    return viz.save(fig, ctx.debug(), cap)
