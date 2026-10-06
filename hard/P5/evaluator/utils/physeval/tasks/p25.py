"""P25 - floating ice melts completely in a straight-walled glass.

Benchmark metrics, taken verbatim:
  M1  (h_after - h_before) / H: floating ice displaces exactly the volume its
      meltwater will occupy, so the level must not change as it melts.
  M2  consistency of the vessel boundary before and after, which is the check
      that nothing about the framing moved and the level comparison is between
      the same two things.

Level is measured on the columns the ice does not occupy, since the ice edge is
a stronger step than the surface it interrupts.
"""
from __future__ import annotations

import numpy as np

from .. import liquid, viz
from ..context import Context
from ..schema import Result
from ..track import background, camera_drift
from ..video import Clip

N_WINDOW = 5          # frames averaged at each end
EDGE_SKIP = 2         # first/last frames can carry codec artefacts


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "(h_after - h_before) / H", principle=(
        "A floating body displaces a volume of water whose weight equals its "
        "own. When ice melts, the meltwater weighs exactly what the ice weighed, "
        "so it exactly fills the volume the ice was displacing and the level is "
        "unchanged. The metric is the level change divided by the vessel height, "
        "which is zero for correct physics; a positive value means the level "
        "rose, in image coordinates that is the surface moving up."), tol=0.05)
    M2 = res.add("M2", "vessel boundary consistency before vs after",
                 principle=(
        "The before/after comparison is only meaningful if the vessel occupies "
        "the same pixels at both ends. The metric is the relative change of the "
        "vessel's segmented width, |w_after - w_before| / w_before."), tol=0.05)

    first = _mean_frame(clip, EDGE_SKIP, EDGE_SKIP + N_WINDOW)
    last = _mean_frame(clip, clip.n - EDGE_SKIP - N_WINDOW, clip.n - EDGE_SKIP)

    # One vessel for the whole clip: the camera and the glass are static, so
    # re-segmenting at each end would add SAM2's jitter to both metrics. It is
    # read from the opening frame rather than the temporal median - the glass
    # is pale on a pale ground and averaging the clip softens its walls, which
    # is enough to make the segmentation return the water body or a truncated
    # glass instead of the whole vessel.
    cont = liquid.find_container(clip[0])
    if cont is None:
        res.fail_all("the glass walls were not located on the temporal median "
                     "frame")
        return res
    c0 = c1 = cont

    M2.steps = [
        f"average the first and last {N_WINDOW} usable frames to suppress codec "
        "noise",
        "segment the vessel once, on the opening frame, and use that one "
        "outline at both ends of the clip; the glass is static, and averaging "
        "the clip first softens its pale walls enough that the segmentation "
        "returns the water body or a truncated glass instead",
        "in each averaged frame, locate the two vessel walls as the strongest "
        "vertical-gradient columns near the segmented box, to sub-pixel "
        "accuracy",
        "M2 = |width_after - width_before| / width_before",
    ]
    wl0, wr0 = liquid.wall_columns(first, cont)
    wl1, wr1 = liquid.wall_columns(last, cont)
    w0, w1 = wr0 - wl0, wr1 - wl1
    if np.isfinite(w0) and np.isfinite(w1) and w0 > 1:
        M2.succeed(abs(w1 - w0) / w0,
                   vessel_width_before_px=float(w0),
                   vessel_width_after_px=float(w1),
                   wall_x_before=[float(wl0), float(wr0)],
                   wall_x_after=[float(wl1), float(wr1)],
                   vessel_bbox=[cont.x0, cont.y0, cont.x1, cont.y1])
    else:
        M2.fail("vessel walls not resolved in one of the two windows")

    M1.steps = [
        "segment the vessel once, on the opening frame, and use that one "
        "outline at both ends of the clip",
        "inside the vessel, summarise each row by the median CIELAB a/b of its "
        "pixels - chroma only, with lightness discarded - and take the surface "
        "as the row where the contents gain the liquid's colour going "
        "downwards, line-fitted per column",
        "discarding lightness is what keeps the vessel base, the bench line "
        "behind the glass, the boundary between lit and shaded water and the "
        "face of the floating ice out of the answer: each is a step in "
        "brightness, none is a step in the liquid's hue",
        f"search only the middle of the vessel, from "
        f"{liquid.SURFACE_BAND[0]:.0%} to {liquid.SURFACE_BAND[1]:.0%} of its "
        "height, since neither the rim nor the base can be a surface, and "
        f"require the gain in chroma to exceed {liquid.MIN_TINT_STEP:g} Lab "
        "units so an untinted vessel reports nothing rather than the best of "
        "a set of meaningless steps",
        "M1 = (y_before - y_after) / vessel height; image y grows downward, so "
        "the sign is flipped to make a rising level positive",
    ]

    before = _level(first, c0, exclude_ice=True)
    after = _level(last, c1, exclude_ice=False)
    res.scene = {
        "camera_drift_frac_diag": camera_drift(clip),
        "window_frames": [[EDGE_SKIP, EDGE_SKIP + N_WINDOW],
                          [clip.n - EDGE_SKIP - N_WINDOW, clip.n - EDGE_SKIP]],
        "vessel_height_px": c0.height,
        "level_before_px": None if before is None else round(before[0], 2),
        "level_after_px": None if after is None else round(after[0], 2),
        "level_method_before": None if before is None else before[2],
        "level_method_after": None if after is None else after[2],
    }
    if before is None or after is None:
        which = "start" if before is None else "end"
        M1.fail(f"no liquid surface found at the {which} of the clip")
        return _finish(res, ctx, clip, c0, c1, before, after)

    H = float(c0.height)
    M1.succeed((before[0] - after[0]) / H,
               level_before_px=before[0], level_after_px=after[0],
               level_change_px=after[0] - before[0],
               vessel_height_px=H,
               surface_fit_rms_before_px=before[1],
               surface_fit_rms_after_px=after[1],
               direction=("rose" if after[0] < before[0] - 1 else
                          "fell" if after[0] > before[0] + 1 else "held"))
    return _finish(res, ctx, clip, c0, c1, before, after)


def _mean_frame(clip: Clip, a: int, b: int) -> np.ndarray:
    a, b = max(0, a), min(clip.n, b)
    return np.mean(np.stack([clip[i] for i in range(a, b)]),
                   axis=0).astype(np.uint8)


def _level(frame, cont, exclude_ice: bool):
    """Surface height at the vessel centre, read from the water's own colour.

    One measurement, both ends of the clip: the row inside the vessel where the
    contents gain the liquid's colour. The floating ice needs no special
    handling - it spans part of the vessel's width, and each row is summarised
    by the median across the width, so the water either side of the block
    carries the row.
    """
    wl = liquid.surface_by_tint(frame, cont)
    if wl is None:
        return None
    return (float(wl.y_at(0.5 * (cont.x0 + cont.x1))), float(wl.rms_px),
            f"chroma step of {wl.step:.1f} Lab units inside the vessel")


def _finish(res, ctx, clip, c0, c1, before, after):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, c0, c1, before, after, res)
    return res


def _debug(clip, ctx, c0, c1, before, after, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    f0 = _mean_frame(clip, EDGE_SKIP, EDGE_SKIP + N_WINDOW)
    f1 = _mean_frame(clip, clip.n - EDGE_SKIP - N_WINDOW, clip.n - EDGE_SKIP)

    for axis, frame, cont, lvl, title in (
            (ax[0], f0, c0, before, "before: ice floating"),
            (ax[1], f1, c1, after, "after: ice melted")):
        viz.show_frame(axis, frame, title)
        axis.plot([cont.x0, cont.x1], [cont.y0, cont.y0], "-",
                  color="#34c759", lw=1.0)
        axis.plot([cont.x0, cont.x1], [cont.y1, cont.y1], "-",
                  color="#34c759", lw=1.0)
        if lvl is not None:
            axis.axhline(lvl[0], color="#ffd400", lw=2.0,
                         label=f"level y = {lvl[0]:.1f} px")
            axis.legend(loc="lower left", fontsize=7)

    if before is not None and after is not None:
        H = float(c0.height)
        ax[2].bar(["before", "after"], [before[0], after[0]],
                  color=["#0a84ff", "#ff9500"])
        ax[2].set_ylabel("surface height in image  [px, down]")
        ax[2].invert_yaxis()
        ax[2].set_title(f"level change = {after[0] - before[0]:+.1f} px "
                        f"over H = {H:.0f} px")
        for i, v in enumerate([before[0], after[0]]):
            ax[2].annotate(f"{v:.1f}", (i, v), ha="center", va="bottom",
                           fontsize=8)
    else:
        ax[2].set_title("level not measured at both ends")

    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:+.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 (h_after-h_before)/H = {fmt('M1')}   |   "
                    f"M2 vessel width change = {fmt('M2')}")
