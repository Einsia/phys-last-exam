"""P16 - four collinear markers on a rigid rod that translates and rotates.

Benchmark metrics, taken verbatim:
  M1  CV(chi_t), the cross-ratio drift. The cross-ratio of four collinear points
      is a projective invariant, so a genuinely rigid rod seen by a fixed camera
      keeps it constant no matter how the rod moves.
  M2  the collinearity residual of the four markers.

The four markers have four different hues, so each one is segmented by its own
colour. That makes marker identity exact for free, which is the part a generic
point tracker has to guess and can get wrong when the rod sweeps past itself.
"""
from __future__ import annotations

import numpy as np

from .. import viz
from ..context import Context
from ..fitting import cv
from ..schema import Result
from ..track import camera_drift, find_chroma_markers, track_target
from ..video import Clip

N_MARKERS = 4
MIN_FRAMES = 12
MIN_SPAN_PX = 60.0
# The cross-ratio is a ratio of differences between neighbouring projected
# coordinates, so its conditioning is set by the smallest adjacent gap, not by
# the overall span. Centroid noise is a few tenths of a pixel; requiring the
# smallest gap to be at least this wide keeps that noise below about 5% of the
# gap. Frames where the rod is nearly end-on collapse the gaps and are the only
# frames this excludes.
MIN_ADJACENT_GAP_PX = 8.0
# The rod is rigid and the camera is fixed, so the projected spacing of its
# markers changes smoothly. A frame where the smallest gap suddenly collapses
# to a fraction of what it is through the rest of the clip is not a pose, it is
# a frame where two marker centroids merged or one jumped, and the cross-ratio
# there is a reading of that failure rather than of the rod. Judging the gap
# against the clip's own median catches this; a fixed pixel floor cannot, since
# the collapse can still leave tens of pixels between the centroids.
MIN_GAP_FRAC_OF_MEDIAN = 0.6


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "CV(chi_t): cross-ratio drift", principle=(
        "The cross-ratio chi = (t3-t1)(t4-t2) / ((t3-t2)(t4-t1)) of four "
        "collinear points, with t the position along the rod axis, is invariant "
        "under any projective transform. A rigid rod filmed by a fixed camera "
        "therefore holds chi fixed while it translates and rotates, so the "
        "metric is CV(chi) over frames; a rod that bends or whose markers slide "
        "makes it grow."), tol=0.1)
    M2 = res.add("M2", "collinearity residual", principle=(
        "Collinearity residual: per frame, fit a line to the four marker "
        "centroids and take the RMS perpendicular distance from it, divided by "
        "the marker span so the number is scale-free. The metric is the mean "
        "over frames."), tol=0.05)

    targets = find_chroma_markers(clip[0], k=N_MARKERS)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "markers_found_in_first_frame": len(targets),
                 "marker_seed_hues": [round(t.hue, 1) for t in targets]}
    if len(targets) < N_MARKERS:
        msg = (f"found {len(targets)} coloured markers in the first frame, "
               f"need {N_MARKERS}")
        M1.fail(msg)
        M2.fail(msg)
        return res

    tracks = [track_target(clip, t) for t in targets]
    xs = np.stack([t.x for t in tracks])   # (4, n)
    ys = np.stack([t.y for t in tracks])
    ok = np.all(np.isfinite(xs) & np.isfinite(ys), axis=0)

    # One pass to see how the markers are normally spaced in this clip, so a
    # frame can be judged against the clip's own geometry.
    per_frame = {}
    for i in np.nonzero(ok)[0]:
        c = _cross_ratio(xs[:, i], ys[:, i])
        if c is not None:
            per_frame[int(i)] = c
    if not per_frame:
        msg = "the four markers were never all resolved in one frame"
        M1.fail(msg)
        M2.fail(msg)
        if ctx.debug_path:
            res.debug_image = _debug(clip, ctx, xs, ys, [], [], [], [], res)
        return res
    gap_median = float(np.median([c["min_gap"] for c in per_frame.values()]))
    gap_floor = max(MIN_ADJACENT_GAP_PX,
                    MIN_GAP_FRAC_OF_MEDIAN * gap_median)

    chi, resid, span, used = [], [], [], []
    gaps, skipped = [], []
    for i, c in per_frame.items():
        gaps.append((int(i), c["min_gap"]))
        if c["span"] < MIN_SPAN_PX or c["min_gap"] < gap_floor:
            skipped.append(int(i))
            continue
        chi.append(c["chi"])
        resid.append(c["resid_frac"])
        span.append(c["span"])
        used.append(int(i))

    res.scene["marker_coverage"] = [round(float(t.coverage), 4) for t in tracks]
    res.scene["frames_with_all_markers"] = int(ok.sum())
    res.scene["frames_used"] = len(used)
    res.scene["frames_skipped_ill_conditioned"] = len(skipped)
    res.scene["min_adjacent_gap_floor_px"] = round(gap_floor, 2)
    res.scene["min_adjacent_gap_median_px"] = round(gap_median, 2)
    if skipped:
        res.scene["skipped_reason"] = (
            f"smallest adjacent marker gap below {gap_floor:.1f} px - "
            f"{MIN_GAP_FRAC_OF_MEDIAN:g} of this clip's median gap of "
            f"{gap_median:.1f} px - or span below {MIN_SPAN_PX:g} px; the "
            "markers merged or one jumped, and the cross-ratio there reads "
            "that rather than the rod")

    M1.steps = [
        "segment each of the four markers by its own hue, sampled from the "
        "first frame, and take its sub-pixel centroid every frame",
        "per frame, fit the rod axis to the four centroids and project them "
        "onto it to get ordered coordinates t1<t2<t3<t4",
        "chi = (t3-t1)(t4-t2) / ((t3-t2)(t4-t1))",
        "drop frames where the smallest adjacent marker gap falls below "
        f"{MIN_GAP_FRAC_OF_MEDIAN:g} of this clip's median: the rod is rigid "
        "and the camera fixed, so a sudden collapse of the spacing is a "
        "detection failure and the cross-ratio there is unbounded",
        "M1 = std(chi)/|mean(chi)| over the frames that remain",
    ]
    M2.steps = [
        "per frame, RMS perpendicular distance of the four centroids from the "
        "fitted rod axis, divided by the marker span",
        "M2 = mean of that ratio over the same frames",
    ]

    if len(used) < MIN_FRAMES:
        msg = (f"only {len(used)} frames have all four markers visible with a "
               f"usable span, need {MIN_FRAMES}")
        M1.fail(msg, frames_used=len(used))
        M2.fail(msg, frames_used=len(used))
    else:
        chi_a = np.asarray(chi, float)
        m1 = cv(chi_a)
        gv = np.asarray([g[1] for g in gaps], float)
        if np.isfinite(m1):
            M1.succeed(m1, frames_used=len(used),
                           chi_mean=float(np.mean(chi_a)),
                           chi_std=float(np.std(chi_a, ddof=1)),
                           chi_min=float(chi_a.min()),
                           chi_max=float(chi_a.max()),
                           chi_series=chi_a.tolist(),
                           min_adjacent_gap_px_min=float(gv.min()),
                           min_adjacent_gap_px_median=float(np.median(gv)))
            M1.note = (
                "chi is a ratio of differences between neighbouring projected "
                "marker positions, so it is best conditioned when those gaps "
                "are wide; the smallest adjacent gap per frame is reported here "
                "and drawn in the debug figure so a chi excursion can be read "
                "against it. A rigid rod under a pinhole projection must hold "
                "chi constant regardless of pose, so a sustained excursion "
                "means the rendered rod is not rigid.")
        else:
            M1.fail("cross-ratio has zero mean, CV undefined")
        M2.succeed(float(np.mean(resid)), frames_used=len(used),
                       collinearity_residual_frac_series=list(map(float, resid)),
                       mean_marker_span_px=float(np.mean(span)))

    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, xs, ys, used, chi, resid, gaps, res)
    return res


def _cross_ratio(px: np.ndarray, py: np.ndarray) -> dict | None:
    """Cross-ratio and collinearity residual of four points, via the rod axis."""
    pts = np.stack([px, py], axis=1)
    c = pts.mean(axis=0)
    d = pts - c
    # Principal direction of the four points is the rod axis.
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    axis, normal = vt[0], vt[1]
    t = d @ axis
    perp = d @ normal
    order = np.argsort(t)
    t = t[order]
    span = float(t[-1] - t[0])
    if span < MIN_SPAN_PX:
        return None
    t1, t2, t3, t4 = t
    den = (t3 - t2) * (t4 - t1)
    if abs(den) < 1e-6:
        return None
    chi = float(((t3 - t1) * (t4 - t2)) / den)
    return {"chi": chi, "span": span, "min_gap": float(np.min(np.diff(t))),
            "resid_frac": float(np.sqrt(np.mean(perp ** 2)) / span)}


def _debug(clip, ctx, xs, ys, used, chi, resid, gaps, res) -> str:
    M1, M2 = res.metrics["M1"], res.metrics["M2"]
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    cols = ["#ff3b30", "#ffd400", "#0a84ff", "#34c759"]

    mid = used[len(used) // 2] if used else 0
    viz.show_frame(ax[0], clip[mid], f"marker tracks (frame {mid} shown)")
    for j in range(xs.shape[0]):
        ax[0].plot(xs[j], ys[j], "-", lw=1.0, color=cols[j % 4], alpha=0.55)
        ax[0].scatter(xs[j, mid], ys[j, mid], s=42, color=cols[j % 4],
                      edgecolor="k", zorder=4, label=f"marker {j + 1}")
    if np.all(np.isfinite(xs[:, mid])):
        ax[0].plot(xs[:, mid], ys[:, mid], "--", color="w", lw=1.0)
    ax[0].legend(loc="upper right", fontsize=7)

    if used:
        ax[1].plot(used, chi, ".-", ms=4, color="#0a84ff")
        ax[1].axhline(float(np.mean(chi)), color="#ff3b30", ls="--",
                      label=f"mean = {np.mean(chi):.4f}")
        ax[1].legend(fontsize=7)
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("cross-ratio chi")
    ax[1].set_title("cross-ratio over time (flat if rigid)")

    if used:
        ax[2].plot(used, resid, ".-", ms=4, color="#ff9500",
                   label="collinearity residual / span")
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("RMS perp. dist / span")
    ax[2].set_title("collinearity residual and conditioning")
    if gaps:
        gi = [g[0] for g in gaps]
        gv = [g[1] for g in gaps]
        tw = ax[2].twinx()
        tw.plot(gi, gv, "-", color="#5e5ce6", lw=1.0,
                label="smallest adjacent marker gap [px]")
        tw.axhline(MIN_ADJACENT_GAP_PX, color="#5e5ce6", ls=":", lw=1.0)
        tw.set_ylabel("gap [px]", color="#5e5ce6")
        tw.grid(False)
        tw.legend(loc="upper right", fontsize=6)
    ax[2].legend(loc="upper left", fontsize=6)

    m1 = "n/a" if M1.value is None else f"{M1.value:.4f}"
    m2 = "n/a" if M2.value is None else f"{M2.value:.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 CV(chi) = {m1}   |   "
                    f"M2 collinearity residual = {m2}   "
                    f"({len(used)} frames used)")
