"""P15 - two identical pendulums released from about 5 and 15 degrees.

Benchmark metrics, taken verbatim:
  M1  T_5 / T_15 - 1, the small-angle isochronism: equal string lengths give
      equal periods regardless of amplitude, so the ratio is 1.
  M2  the sinusoidal-fit residual of theta(t) - one per pendulum, so it occupies
      two slots, M2 for the left and M3 for the right.
"""
from __future__ import annotations

import numpy as np

from .. import pendulum, viz
from ..context import Context
from ..schema import Result
from ..track import camera_drift
from ..video import Clip


def evaluate(clip: Clip, ctx: Context) -> Result:
    return run(clip, ctx, ratio_label="T_left / T_right - 1", isochronism=True)


def run(clip: Clip, ctx: Context, ratio_label: str, isochronism: bool) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", ratio_label, principle=(
        "For a pendulum at small amplitude the period depends only on the "
        "string length, so two pendulums of equal length must share a period "
        "whatever amplitude each was released from. The metric is the period "
        "ratio minus one, which is zero when they agree; periods are measured "
        "in frames and the ratio removes the frame rate."), tol=0.1)
    M2 = res.add("M2", "sinusoidal-fit residual of theta(t), left pendulum",
                 principle=(
        "A small-amplitude pendulum swings sinusoidally. Fitting "
        "A cos(wt) + B sin(wt) + C at the measured period and taking the RMS "
        "residual in radians says how harmonic the motion actually is."), tol=0.1)
    M3 = res.add("M3", "sinusoidal-fit residual of theta(t), right pendulum",
                 principle=M2.principle, tol=0.1)

    targets = pendulum.find_bobs(clip)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "bobs_found_in_first_frame": len(targets),
                 "bob_seed_hues": [round(t.hue, 1) for t in targets]}
    if len(targets) < 2:
        res.fail_all(f"found {len(targets)} pendulum bobs in the first frame, "
                     "need 2")
        return res

    bobs = pendulum.measure(clip, targets)
    left, right = bobs
    t = np.arange(clip.n, dtype=float)
    for bob in bobs:
        resolve_release_cycle(bob, t)
    res.scene.update({
        "bob_track_coverage": [round(b.track.coverage, 4) for b in bobs],
        "pivot_xy": [[round(v, 1) for v in b.pivot] for b in bobs],
        "string_length_px": [round(b.radius, 1) for b in bobs],
        "amplitude_deg": [round(np.degrees(b.amplitude), 2) for b in bobs],
        "period_frames": [round(b.period, 3) for b in bobs],
        "half_cycle_period_samples": [
            [round(v, 2) for v in b.period_samples] for b in bobs],
    })

    # A sine fit to an incomplete sweep (or a stationary bob) can have a very
    # small residual. It is not evidence of an oscillation. Both bobs must
    # show a complete observed cycle before any physical metric earns credit.
    cycle_checks = [cycle_evidence(b, t) for b in bobs]
    res.scene["observed_cycle_checks"] = cycle_checks
    if not all(check["passed"] for check in cycle_checks):
        reasons = [f"{side}: {check['reason']}" for side, check in
                   zip(("left", "right"), cycle_checks) if not check["passed"]]
        res.fail_all("P15_complete_period_required; " + "; ".join(reasons))
        if ctx.debug_path:
            res.debug_image = _debug(clip, ctx, bobs, t, res)
        return res

    steps = [
        "segment each bob by its own colour, sampled from the first frame, and "
        "take its sub-pixel centroid every frame",
        "fit a circle to each bob path; its centre is the pivot and its radius "
        "the string length",
        "theta(t) = atan2(x - x_pivot, y - y_pivot), the angle from vertical",
        "take the times where theta crosses its own mean, linearly interpolated; "
        "consecutive crossings are half a period apart",
        "period = median of 2 x (successive crossing gaps)",
    ]
    M1.steps = steps + [f"M1 = {ratio_label}"]
    for m, side in ((M2, "left"), (M3, "right")):
        m.steps = steps + [
            f"fit A cos(wt)+B sin(wt)+C to theta(t) of the {side} pendulum at "
            "its measured period",
            "metric = RMS residual in radians",
        ]

    for m, bob, side in ((M2, left, "left"), (M3, right, "right")):
        if bob.static:
            m.fail(f"{side} pendulum never leaves its starting position "
                   f"(path extent {bob.path_extent:.1f} px), so theta(t) is "
                   "constant and there is no oscillation to fit",
                   path_extent_px=bob.path_extent,
                   track_coverage=bob.track.coverage)
        elif np.isfinite(bob.sine_rms):
            m.succeed(bob.sine_rms, period_frames=bob.period,
                      path_extent_px=bob.path_extent,
                      amplitude_deg=float(np.degrees(bob.amplitude)),
                      sine_rms_deg=float(np.degrees(bob.sine_rms)))
        else:
            m.fail(f"{side} pendulum: too few tracked frames to fit a sinusoid",
                   track_coverage=bob.track.coverage)

    frozen = [s for b, s in ((left, "left"), (right, "right")) if b.static]
    bad = [s for b, s in ((left, "left"), (right, "right"))
           if not b.static and not cycle_evidence(b, t)["passed"]]
    if frozen:
        M1.fail(f"the {' and '.join(frozen)} pendulum never swings "
                f"(path extent "
                f"{', '.join(f'{b.path_extent:.1f}' for b in bobs if b.static)}"
                " px against a tracked centroid noise floor well under 1 px), "
                "so its period does not exist and the ratio cannot be formed",
                path_extent_px=[round(b.path_extent, 2) for b in bobs],
                track_coverage=[round(b.track.coverage, 3) for b in bobs])
    elif bad:
        M1.fail(f"{' and '.join(bad)} pendulum completes fewer than "
                f"{pendulum.MIN_CROSSINGS} half-swings, so its period is not "
                "resolved",
                crossings=[int(pendulum.zero_crossings(t, b.theta).size)
                           for b in bobs])
    elif not (np.isfinite(left.period) and np.isfinite(right.period)
              and right.period > 0):
        M1.fail("period undefined for one pendulum")
    else:
        ratio = left.period / right.period
        M1.succeed(ratio - 1.0,
                   period_left_frames=left.period,
                   period_right_frames=right.period,
                   period_ratio=ratio,
                   period_cv_left=_cv(left.period_samples),
                   period_cv_right=_cv(right.period_samples),
                   amplitude_left_deg=float(np.degrees(left.amplitude)),
                   amplitude_right_deg=float(np.degrees(right.amplitude)))
        if isochronism:
            M1.note = ("the two amplitudes are reported alongside: the point of "
                       "the task is that the ratio stays at 1 even though they "
                       "differ")

    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, bobs, t, res)
    return res


def release_cycle_return(theta, t, crossings):
    """Resolve a release-extreme return using observed samples only."""
    if len(crossings) != 2 or not np.isfinite(theta).all():
        return None
    approximate_period = 2 * (crossings[1] - crossings[0])
    if approximate_period <= 0:
        return None
    amplitude = float(np.ptp(theta))
    first = float(theta[0])
    center = float((np.min(theta) + np.max(theta)) / 2)
    candidates = np.flatnonzero((t-t[0] >= .98*approximate_period) &
                               (t-t[0] <= 1.25*approximate_period) &
                               (np.abs(theta-first) <= .02*amplitude))
    if amplitude <= 0 or abs(first-center) < .45*amplitude or not len(candidates):
        return None
    # Use the closest observed return, not the first sample entering its
    # tolerance band. No fitted or extrapolated endpoint is introduced.
    index = min(candidates, key=lambda i: (abs(theta[i]-first),
                                          abs(t[i]-t[0]-approximate_period)))
    return float(t[index]-t[0])


def resolve_release_cycle(bob, t):
    """The shared crossing estimator needs three crossings; one full cycle
    starting at an extreme has only two. Recover this observed boundary case.
    """
    if bob.static or (np.isfinite(bob.period) and bob.period > 0):
        return
    period = release_cycle_return(bob.theta, t, pendulum.zero_crossings(t, bob.theta))
    if period is not None:
        bob.period = period
        bob.period_samples = np.asarray([period], float)
        bob.sine_rms = pendulum._sine_rms(t, bob.theta, period)


def cycle_evidence(bob, t):
    """Require a crossing-bounded cycle or an observed release-extreme return."""
    crossings = pendulum.zero_crossings(t, bob.theta)
    variability_estimable = len(bob.period_samples) >= 2
    period_cv = _cv(bob.period_samples) if variability_estimable else None
    relative_rms = bob.sine_rms / max(bob.amplitude, 1e-9)
    complete=len(crossings)>=3
    cycle_method='three_observed_mean_crossings' if complete else None
    # A clip released at a turning point may contain exactly one complete
    # cycle but only two mean crossings. Verify its observed return to the
    # initial extreme instead of demanding an extra half swing.
    if not complete and np.isfinite(bob.period) and bob.period>0:
        if release_cycle_return(bob.theta,t,crossings) is not None:
            complete=True;cycle_method='observed_return_to_release_extreme'
    reason = None
    if bob.static:
        reason = "no visible swing"
    elif not complete:
        reason = "less than one complete observed period"
    elif not np.isfinite(bob.period) or bob.period <= 0:
        reason = "period undefined"
    elif (variability_estimable and (not np.isfinite(period_cv) or period_cv > .35)) or not np.isfinite(relative_rms) or relative_rms > .50:
        reason = "motion has no sufficiently repeatable period"
    return {"passed": reason is None, "reason": reason,
            "mean_crossings": int(len(crossings)), "period_cv": period_cv,
            "period_variability_estimable": variability_estimable,
            "sine_rms_over_amplitude": relative_rms,
            "complete_cycle_observed":complete,"cycle_detection_method":cycle_method,
            "required_crossings": 3, "max_period_cv": .35, "max_relative_sine_rms": .50}


def _cv(samples: np.ndarray) -> float:
    from ..fitting import cv
    return float(cv(samples))


def _debug(clip, ctx, bobs, t, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    cols = ["#ff3b30", "#0a84ff"]

    viz.show_frame(ax[0], clip[0], "bob paths, fitted pivots and strings")
    for b, c in zip(bobs, cols):
        ax[0].plot(b.track.x, b.track.y, "-", lw=1.2, color=c, alpha=0.8)
        ax[0].scatter(*b.pivot, s=40, marker="x", color=c, zorder=4)
        i0 = int(np.argmax(np.isfinite(b.track.x)))
        ax[0].plot([b.pivot[0], b.track.x[i0]], [b.pivot[1], b.track.y[i0]],
                   "--", lw=1.0, color=c)
    ax[0].legend(["left path", "left pivot", "left string",
                  "right path", "right pivot", "right string"], fontsize=6,
                 loc="lower left")

    for b, c, lab in zip(bobs, cols, ("left", "right")):
        th = np.degrees(b.theta)
        ax[1].plot(t, th, "-", lw=1.1, color=c,
                   label=f"{lab}: T = {b.period:.2f} frames, "
                         f"amp {np.degrees(b.amplitude):.1f} deg")
        for z in pendulum.zero_crossings(t, b.theta):
            ax[1].axvline(z, color=c, lw=0.5, alpha=0.35)
    ax[1].axhline(0, color="#8e8e93", lw=0.8)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("theta  [deg from vertical]")
    ax[1].set_title("angle signals with mean-crossings")
    ax[1].legend(fontsize=7)

    for b, c, lab in zip(bobs, cols, ("left", "right")):
        if b.period_samples.size:
            ax[2].plot(b.period_samples, "o-", ms=4, color=c,
                       label=f"{lab} per-half-cycle T")
    ax[2].set_xlabel("half-cycle index")
    ax[2].set_ylabel("period  [frames]")
    ax[2].set_title("period stability")
    ax[2].legend(fontsize=7)

    def fmt(key):
        m = res.metrics[key]
        return "n/a" if m.value is None else f"{m.value:+.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 period ratio - 1 = {fmt('M1')}   |   "
                    f"M2 sine resid L = {fmt('M2')}   M3 sine resid R = "
                    f"{fmt('M3')}")
