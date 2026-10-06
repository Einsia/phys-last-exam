"""P22 - a rectangular block of ice floating freely in fresh water.

Benchmark metrics, taken verbatim:
  M1  |h_sub / h_total - rho_ice / rho_water|: the submerged fraction of a
      freely floating body equals the density ratio, 0.917 for ice in fresh
      water.
  M2  left-right waterline height consistency: the free surface either side of
      the block must sit at the same height.

Both are ratios of lengths in the same image, so no calibration is involved.
The waterline is measured only on columns the block does not occupy, because
the block's own edges are stronger than the surface step it interrupts.
"""
from __future__ import annotations

import numpy as np

from .. import liquid, viz
from ..context import Context
from ..schema import Result
from ..track import background, camera_drift
from ..video import Clip

RHO_RATIO = 0.917
N_SAMPLES = 9
# The block bobs after release; the metric is about the floating equilibrium, so
# the measurement uses the later part of the clip.
SETTLE_FROM = 0.35


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "|h_sub/h_total - rho_ice/rho_water|", principle=(
        "A freely floating body displaces its own weight, so the fraction of "
        "its height below the waterline equals the density ratio of body to "
        "liquid. For pure ice in fresh water that is 0.917. The metric is the "
        "absolute difference between the measured submerged fraction and "
        f"{RHO_RATIO}."), tol=0.1)
    M2 = res.add("M2", "left-right waterline height consistency", principle=(
        "The free surface on the two sides of the floating block belongs to one "
        "connected body of still liquid, so both sides must sit at the same "
        "height. The metric is |y_left - y_right| divided by the vessel width, "
        "measured on the ice-free columns either side of the block."), tol=0.05)

    # The glass comes from its two vertical wall edges rather than from a
    # segmentation. Prompting SAM2 inside the vessel returns whichever of the
    # glass, the water below the surface or the ice block sits under the point,
    # and here the block fills most of the width, so the returned mask has no
    # ice-free columns left to read a waterline on. The walls are unambiguous
    # and, with the camera locked off, fixed for the clip.
    ref = background(clip).astype(np.uint8)
    cont = liquid.find_container_by_walls(ref)
    if cont is None:
        res.fail_all("the glass walls were not located on the temporal median "
                     "frame")
        return res

    idx = np.linspace(int(SETTLE_FROM * clip.n), clip.n - 1,
                      N_SAMPLES).round().astype(int)
    rows, per_frame = [], []
    for i in np.unique(idx):
        frame = clip[i]
        ice = liquid.find_floating_block(frame, cont)
        if ice is None:
            per_frame.append({"frame": int(i), "note": "no floating block found"})
            continue
        cols = liquid.columns_of(ice)
        wl = liquid.find_waterline(frame, cont, exclude_cols=cols,
                                   search=(0.02, 0.75))
        if wl is None:
            per_frame.append({"frame": int(i),
                              "note": (f"the block covers "
                                       f"{int(cols.sum())} px of the glass "
                                       "width, leaving too few ice-free "
                                       "columns to read a waterline")})
            continue
        ys = np.flatnonzero(ice.any(axis=1))
        top, bottom = float(ys.min()), float(ys.max())
        h_total = bottom - top
        if h_total < 8:
            per_frame.append({"frame": int(i), "note": "block too small"})
            continue
        surf = wl.y_at(0.5 * (cont.x0 + cont.x1))
        h_sub = bottom - surf
        # A floating body is partly in and partly out, so the submerged
        # fraction must lie strictly between 0 and 1. Anything else means the
        # block and the surface were not both located correctly, and the ratio
        # would not be a physical quantity. Report that instead of a number.
        if not (0.0 < h_sub < h_total):
            per_frame.append({
                "frame": int(i),
                "note": (f"block spans y={top:.0f}..{bottom:.0f} while the "
                         f"surface sits at y={surf:.0f}, giving a submerged "
                         f"height of {h_sub:.0f} px against a block height of "
                         f"{h_total:.0f} px; a floating body must be partly "
                         "above and partly below the surface, so the block and "
                         "the waterline were not both located correctly here")})
            continue
        left, right = _side_heights(frame, cont, cols, wl)
        rows.append({"frame": int(i), "top": top, "bottom": bottom,
                     "surface": surf, "h_total": h_total, "h_sub": h_sub,
                     "ratio": h_sub / h_total, "y_left": left,
                     "y_right": right, "tilt_deg": wl.angle_deg})
        per_frame.append({"frame": int(i), "submerged_fraction": h_sub / h_total,
                          "h_sub_px": h_sub, "h_total_px": h_total,
                          "surface_y_px": surf})

    res.scene = {
        "camera_drift_frac_diag": camera_drift(clip),
        "vessel_bbox_xyxy": [cont.x0, cont.y0, cont.x1, cont.y1],
        "frames_sampled": [int(i) for i in np.unique(idx)],
        "frames_measured": len(rows),
        "per_frame": per_frame,
        "extractor": "glass from its vertical wall edges, block from SAM2, "
                     "waterline from the strongest horizontal CIELAB colour "
                     "step on the block-free columns between the walls",
    }
    common = [
        "locate the glass walls as the best-supported pair of long vertical "
        "edges on the temporal median frame",
        "segment the floating block with SAM2, prompted at the middle of the "
        "glass where the opaque block sits",
        "read the waterline on the glass-interior columns the block does not "
        "occupy, as the strongest horizontal CIELAB colour step there",
    ]
    M1.steps = common + [
        "h_total = block mask height; h_sub = block bottom - waterline height "
        "at the vessel centre",
        f"M1 = |median(h_sub/h_total) over the settled part of the clip - "
        f"{RHO_RATIO}|",
    ]
    M2.steps = common + [
        "take the fitted surface height at the block's left and right sides",
        "M2 = |y_left - y_right| / vessel width",
    ]

    if not rows:
        res.fail_all("the floating block could not be separated from the water "
                     "in any sampled frame")
        return _finish(res, ctx, clip, cont, rows)

    ratios = np.array([r["ratio"] for r in rows], float)
    med = float(np.median(ratios))
    M1.succeed(abs(med - RHO_RATIO), submerged_fraction_median=med,
               submerged_fraction_series=ratios.tolist(),
               expected_fraction=RHO_RATIO, frames_measured=len(rows),
               h_total_px_median=float(np.median([r["h_total"] for r in rows])),
               h_sub_px_median=float(np.median([r["h_sub"] for r in rows])))

    lr = np.array([[r["y_left"], r["y_right"]] for r in rows], float)
    good = np.isfinite(lr).all(axis=1)
    if good.any():
        d = np.abs(lr[good, 0] - lr[good, 1]) / cont.width
        M2.succeed(float(np.median(d)),
                   left_minus_right_px_median=float(
                       np.median(lr[good, 0] - lr[good, 1])),
                   vessel_width_px=cont.width,
                   surface_tilt_deg_median=float(
                       np.median([r["tilt_deg"] for r in rows])))
    else:
        M2.fail("no frame has ice-free surface columns on both sides of the "
                "block")
    return _finish(res, ctx, clip, cont, rows)


def _side_heights(frame, cont, cols, wl):
    """Fitted surface height just outside the block on each side."""
    on = np.flatnonzero(cols)
    if on.size == 0:
        return float("nan"), float("nan")
    xl, xr = int(on.min()), int(on.max())
    left = wl.y_at(max(cont.x0 + 4, xl - 12))
    right = wl.y_at(min(cont.x1 - 4, xr + 12))
    return float(left), float(right)


def _finish(res, ctx, clip, cont, rows):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, cont, rows, res)
    return res


def _debug(clip, ctx, cont, rows, res) -> str:
    import matplotlib.patches as mpatches
    fig, ax = viz.figure(ncols=3, width_each=5.0)

    if rows:
        r = rows[len(rows) // 2]
        f = int(r["frame"])
        viz.show_frame(ax[0], clip[f],
                       f"frame {f}: block, waterline, submerged part")
        ax[0].axhline(r["surface"], color="#ffd400", lw=2.0,
                      xmin=0.02, xmax=0.98, label="waterline")
        ax[0].add_patch(mpatches.Rectangle(
            (cont.x0, r["top"]), cont.width, r["h_total"], fill=False,
            edgecolor="#00e5ff", lw=1.4, label="block extent"))
        ax[0].add_patch(mpatches.Rectangle(
            (cont.x0, r["surface"]), cont.width, r["h_sub"], color="#0a84ff",
            alpha=0.20, label=f"submerged {r['h_sub']:.0f} px"))
        ax[0].legend(loc="lower left", fontsize=7)
    else:
        viz.show_frame(ax[0], clip[0], "no block measured")

    if rows:
        fr = [r["frame"] for r in rows]
        ax[1].plot(fr, [r["ratio"] for r in rows], "o-", ms=4, color="#0a84ff",
                   label="h_sub / h_total")
        ax[1].axhline(RHO_RATIO, color="#ff3b30", ls="--",
                      label=f"ice/water density ratio {RHO_RATIO}")
        ax[1].set_ylim(0, 1.05)
        ax[1].legend(fontsize=7)
        ax[2].plot(fr, [r["h_total"] for r in rows], "o-", ms=4,
                   color="#34c759", label="h_total [px]")
        ax[2].plot(fr, [r["h_sub"] for r in rows], "o-", ms=4, color="#ff9500",
                   label="h_sub [px]")
        ax[2].legend(fontsize=7)
    ax[1].set_xlabel("frame")
    ax[1].set_title("submerged fraction over the settled clip")
    ax[2].set_xlabel("frame")
    ax[2].set_ylabel("px")
    ax[2].set_title("block height and submerged height")

    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 |h_sub/h_total - {RHO_RATIO}| = "
                    f"{fmt('M1')}   |   M2 left-right waterline = {fmt('M2')}")
