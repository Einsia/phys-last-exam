"""P12 - two pendulums of equal length but clearly different bob masses.

Benchmark metrics, taken verbatim:
  M1  T_H / T_L - 1: the period of a pendulum does not depend on the bob mass,
      so the heavy and light periods must agree.
  M2  the relative error of the two string lengths in the first frame, which is
      the scene precondition the period comparison rests on.
"""
from __future__ import annotations

import numpy as np

from .. import pendulum, viz
from ..context import Context
from ..fitting import cv
from ..schema import Result
from ..track import camera_drift
from ..video import Clip


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "T_heavy / T_light - 1", principle=(
        "The period of a simple pendulum is set by its length alone; the bob "
        "mass cancels out of the equation of motion. Two pendulums of equal "
        "length must therefore share a period however different their masses, "
        "so the metric is the period ratio minus one."), tol=0.1)
    M2 = res.add("M2", "relative error of the two string lengths in the first "
                       "frame", principle=(
        "The mass-independence claim only means something if the two lengths "
        "really are equal, so the first-frame lengths are measured and compared: "
        "|L_left - L_right| / mean(L). Each length is the distance from the bob "
        "in the first frame to that pendulum's pivot, and the pivot is the "
        "centre of the circle the bob traces over the clip."), tol=0.1)

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

    # First-frame length: bob position in the first tracked frame against the
    # pivot recovered from the whole arc.
    lengths, first_idx = [], []
    for b in bobs:
        i0 = int(np.argmax(np.isfinite(b.track.x)))
        first_idx.append(i0)
        lengths.append(float(np.hypot(b.track.x[i0] - b.pivot[0],
                                      b.track.y[i0] - b.pivot[1])))

    res.scene.update({
        "bob_track_coverage": [round(b.track.coverage, 4) for b in bobs],
        "pivot_xy": [[round(v, 1) for v in b.pivot] for b in bobs],
        "bob_radius_px": [round(float(np.nanmedian(b.track.r)), 1) for b in bobs],
        "first_tracked_frame": first_idx,
        "period_frames": [round(b.period, 3) for b in bobs],
        "amplitude_deg": [round(np.degrees(b.amplitude), 2) for b in bobs],
    })

    common = [
        "segment each bob by its own colour, sampled from the first frame, and "
        "take its sub-pixel centroid every frame",
        "fit a circle to each bob path; its centre is the pivot",
    ]
    M1.steps = common + [
        "theta(t) = atan2(x - x_pivot, y - y_pivot)",
        "period = median of 2 x (gaps between successive mean-crossings of "
        "theta), measured in frames",
        "M1 = T_heavy / T_light - 1, with heavy/light taken as the larger and "
        "smaller tracked bob radius",
    ]
    M2.steps = common + [
        "L = |bob position in its first tracked frame - pivot|",
        "M2 = |L_left - L_right| / mean(L_left, L_right)",
    ]

    mean_len = float(np.mean(lengths))
    if any(b.static for b in bobs):
        M2.fail("a pendulum never swings, so its pivot cannot be recovered from "
                "the arc and its string length is undefined",
                path_extent_px=[round(b.path_extent, 2) for b in bobs])
    elif mean_len > 1.0:
        M2.succeed(abs(lengths[0] - lengths[1]) / mean_len,
                   length_left_px=lengths[0], length_right_px=lengths[1],
                   mean_length_px=mean_len)
    else:
        M2.fail("fitted string lengths are degenerate",
                length_left_px=lengths[0], length_right_px=lengths[1])

    # Heavy vs light from the tracked bob size: the benchmark scene makes the
    # masses differ visibly, and the ratio is written heavy over light.
    r = [float(np.nanmedian(b.track.r)) for b in bobs]
    heavy, light = (0, 1) if r[0] >= r[1] else (1, 0)
    frozen = [s for b, s in ((left, "left"), (right, "right")) if b.static]
    bad = [s for b, s in ((left, "left"), (right, "right"))
           if not b.static and not pendulum.crossings_ok(b, t)]
    if frozen:
        M1.fail(f"the {' and '.join(frozen)} pendulum never swings, so its "
                "period does not exist and the ratio cannot be formed",
                path_extent_px=[round(b.path_extent, 2) for b in bobs])
    elif bad:
        M1.fail(f"{' and '.join(bad)} pendulum completes fewer than "
                f"{pendulum.MIN_CROSSINGS} half-swings, so its period is not "
                "resolved")
    elif not np.isfinite(bobs[light].period) or bobs[light].period <= 0:
        M1.fail("period undefined for the light pendulum")
    else:
        ratio = bobs[heavy].period / bobs[light].period
        M1.succeed(ratio - 1.0,
                   heavy_side="left" if heavy == 0 else "right",
                   period_heavy_frames=bobs[heavy].period,
                   period_light_frames=bobs[light].period,
                   period_ratio=ratio,
                   bob_radius_heavy_px=r[heavy],
                   bob_radius_light_px=r[light],
                   period_cv_heavy=float(cv(bobs[heavy].period_samples)),
                   period_cv_light=float(cv(bobs[light].period_samples)))

    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, bobs, t, lengths, res)
    return res


def _debug(clip, ctx, bobs, t, lengths, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    cols = ["#ff3b30", "#0a84ff"]

    viz.show_frame(ax[0], clip[0], "first frame: pivots and string lengths")
    for b, c, L in zip(bobs, cols, lengths):
        i0 = int(np.argmax(np.isfinite(b.track.x)))
        ax[0].plot(b.track.x, b.track.y, "-", lw=1.0, color=c, alpha=0.7)
        ax[0].scatter(*b.pivot, s=44, marker="x", color=c, zorder=4)
        ax[0].plot([b.pivot[0], b.track.x[i0]], [b.pivot[1], b.track.y[i0]],
                   "-", lw=1.6, color=c)
        ax[0].annotate(f"L = {L:.0f} px", (b.pivot[0] + 8, b.pivot[1] + 26),
                       color=c, fontsize=8)

    for b, c, lab in zip(bobs, cols, ("left", "right")):
        ax[1].plot(t, np.degrees(b.theta), "-", lw=1.1, color=c,
                   label=f"{lab}: T = {b.period:.2f} frames, "
                         f"r = {np.nanmedian(b.track.r):.1f} px")
    ax[1].axhline(0, color="#8e8e93", lw=0.8)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("theta  [deg from vertical]")
    ax[1].set_title("angle signals")
    ax[1].legend(fontsize=7)

    for b, c, lab in zip(bobs, cols, ("left", "right")):
        if b.period_samples.size:
            ax[2].plot(b.period_samples, "o-", ms=4, color=c, label=f"{lab}")
    ax[2].set_xlabel("half-cycle index")
    ax[2].set_ylabel("period  [frames]")
    ax[2].set_title("period stability")
    ax[2].legend(fontsize=7)

    def fmt(key):
        m = res.metrics[key]
        return "n/a" if m.value is None else f"{m.value:+.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 T_heavy/T_light - 1 = {fmt('M1')}   |   "
                    f"M2 string-length relative error = {fmt('M2')}   "
                    f"(L = {lengths[0]:.0f} / {lengths[1]:.0f} px)")
