"""P45 - capillary rise in two clean tubes of different inner radius.

Benchmark metrics, taken verbatim:
  M1  |h_1 r_1 / (h_2 r_2) - 1|: Jurin's law makes the rise inversely
      proportional to the bore radius, so the product h*r is the same for both
      tubes.
  M2  tube-diameter detection confidence,
  M3  meniscus consistency.
      The benchmark lists these two auxiliary quantities, so each takes a slot.

The clip starts with both tubes dry above the reservoir and lowers them in, so
the rise is measured against the reservoir surface at the end of the clip and
the bores are measured on the first frame, where the tubes are empty and their
walls are unobstructed.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import liquid, viz
from ..context import Context
from ..fitting import cv as coeff_var
from ..schema import Result
from ..track import camera_drift
from ..video import Clip

TAIL_FRAC = 0.18          # final part of the clip taken as equilibrium
MIN_BORE_PX = 4.0
# The task specifies two clearly different bores, so a measured ratio below this
# means the measurement did not separate them rather than that they are equal.
MIN_BORE_RATIO = 1.25


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "|h_1 r_1 / (h_2 r_2) - 1|", principle=(
        "Jurin's law gives the equilibrium rise as h = 2*gamma*cos(theta)/"
        "(rho*g*r), so with the same liquid and the same glass the product h*r "
        "is a constant of the pair and the narrower tube must rise higher in "
        "exact inverse proportion to its bore. The metric is the absolute "
        "fractional departure of h_1 r_1 / (h_2 r_2) from 1, which is "
        "dimensionless and needs no pixel calibration."), tol=0.1)
    M2 = res.add("M2", "tube-diameter detection confidence", principle=(
        "The bore radii carry the metric, so their measurement is reported with "
        "a confidence: the fraction of the tube's height over which its two "
        "wall edges are both detected, averaged over the two tubes. 1 means "
        "each wall was traced over the full visible tube."), tol=0.1, ideal=1, non_residual=True)
    M3 = res.add("M3", "meniscus consistency", principle=(
        "The rise is only defined once the columns are at equilibrium, so the "
        "metric is the coefficient of variation of the measured rise over the "
        "final part of the clip, averaged over the two tubes. Near zero means "
        "the menisci had settled and the height read off them is stable."), tol=0.1)

    # The tubes hang still at the start, so averaging the opening frames removes
    # codec noise from the wall edges without blurring anything that moves.
    head = np.mean(np.stack([clip[i] for i in range(min(5, clip.n))]),
                   axis=0).astype(np.uint8)
    tailf = np.mean(np.stack([clip[i] for i in range(max(0, clip.n - 5),
                                                     clip.n)]),
                    axis=0).astype(np.uint8)
    # With a tinted liquid the risen columns give the bores directly; without
    # one there is nothing to see inside the glass and the tube walls are all
    # there is to go on.
    tubes = _find_tubes_by_column(tailf) or _find_tubes(head, tailf)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "tubes_found": len(tubes)}
    if len(tubes) < 2:
        res.fail_all(f"only {len(tubes)} liquid column(s) appear between the start and "
                     "the end of the clip, so two bores cannot be identified")
        return _finish(res, ctx, clip, tubes, None, None)

    narrow, wide = sorted(tubes, key=lambda t: t["bore"])[:2]
    bore_ratio = wide["bore"] / max(narrow["bore"], 1e-6)
    res.scene.update({
        "narrow_tube": {k: round(narrow[k], 2) for k in
                        ("x_left", "x_right", "bore", "support")},
        "wide_tube": {k: round(wide[k], 2) for k in
                      ("x_left", "x_right", "bore", "support")},
        "bore_ratio_wide_over_narrow": round(wide["bore"] / narrow["bore"], 3),
    })

    M2.steps = [
        "where the liquid carries a tint, the two risen columns above the tank "
        "are its own coloured regions and each bore is that column's width; "
        "where it is clear, the bores come from each tube's bundle of vertical "
        "wall edges on the dry opening frame",
        "M2 = mean over the two tubes of the fraction of tube height over which "
        "the band keeps that width",
    ]
    M2.succeed(0.5 * (narrow["support"] + wide["support"]),
               narrow_bore_px=narrow["bore"], wide_bore_px=wide["bore"],
               narrow_wall_support=narrow["support"],
               wide_wall_support=wide["support"])

    # Reservoir surface and the two columns, over the settled tail of the clip.
    tail = range(max(0, int((1 - TAIL_FRAC) * clip.n)), clip.n)
    series = {"narrow": [], "wide": [], "reservoir": []}
    method = None
    # The tank does not move, so it is segmented once; its surface is not
    # assumed to stay put, and is read on the same frame as the columns. A rise
    # is the height of a column above the reservoir at that moment, and where a
    # clip lets the reservoir itself move - one of these does, filling the tank
    # towards its brim - a reference frozen at the start stops being the water
    # line partway through and leaves the mark stranded inside the water.
    tank = liquid.find_container(clip[0])
    if tank is None:
        msg = "the tank was not segmented, so the reservoir surface is unknown"
        M1.fail(msg)
        M3.fail(msg)
        return _finish(res, ctx, clip, (narrow, wide), series, None)
    for i in tail:
        f = clip[i]
        b = _reservoir_y(f, tank, narrow, wide)
        if b is None:
            continue
        tint = _tinted_levels(f, narrow, wide)
        if tint is not None:
            method = "top edge of the tinted liquid inside each tube"
            hn, hw = tint["narrow"], tint["wide"]
        else:
            method = method or "strongest colour step inside each tube"
            hn = _column_top(f, narrow, b)
            hw = _column_top(f, wide, b)
            if hn is None or hw is None:
                continue
        series["reservoir"].append(b)
        series["narrow"].append(b - hn)
        series["wide"].append(b - hw)
    if not series["reservoir"]:
        msg = ("the reservoir surface is not readable, so there is no "
               "reference the rise could be measured from")
        M1.fail(msg)
        M3.fail(msg)
        return _finish(res, ctx, clip, (narrow, wide), series, None)
    base = float(np.median(series["reservoir"]))
    res.scene["reservoir_y_px"] = base
    res.scene["reservoir_y_px_range"] = [min(series["reservoir"]),
                                         max(series["reservoir"])]
    res.scene["level_method"] = method

    common = [
        "segment the tank once - it does not move - and read the reservoir "
        "surface inside it as the row that gains the water's colour going "
        "downwards, excluding the columns the tubes occupy; the same "
        "measurement, made the same way, as the water surface in the melting "
        "ice task",
        "read the reservoir on the same frame as the columns rather than "
        "freezing it at the start: a rise is a height above the reservoir at "
        "that moment, and in one of these clips the tank itself fills towards "
        "its brim, which leaves a frozen reference stranded inside the water",
        f"inside each tube, the meniscus is read by {method}: a tinted liquid "
        "is a coloured region whose top edge is the meniscus outright, while "
        "clear water in clear glass has no colour of its own and only leaves "
        "a step where it ends",
        "rise h = reservoir surface height - meniscus height, in pixels",
    ]
    M1.steps = common + ["M1 = |h_narrow * r_narrow / (h_wide * r_wide) - 1|, "
                         "with r taken as half the measured bore"]
    M3.steps = common + [f"over the final {int(TAIL_FRAC * 100)}% of the clip, "
                         "M3 = mean of CV(h_narrow) and CV(h_wide)"]

    res.scene["frames_measured"] = len(series["narrow"])
    if len(series["narrow"]) < 3:
        msg = ("the reservoir surface and both liquid columns are not readable "
               f"in the final part of the clip ({len(series['narrow'])} frames "
               "usable)")
        M1.fail(msg)
        M3.fail(msg)
        return _finish(res, ctx, clip, (narrow, wide), series, None)

    hn = float(np.median(series["narrow"]))
    hw = float(np.median(series["wide"]))
    rn, rw = narrow["bore"] / 2.0, wide["bore"] / 2.0
    M3.succeed(0.5 * (abs(coeff_var(np.asarray(series["narrow"])))
                      + abs(coeff_var(np.asarray(series["wide"])))),
               rise_narrow_px_series=series["narrow"],
               rise_wide_px_series=series["wide"],
               reservoir_y_px_median=float(np.median(series["reservoir"])))

    if bore_ratio < MIN_BORE_RATIO:
        M1.fail(
            f"the two bores measure {narrow['bore']:.1f} px and "
            f"{wide['bore']:.1f} px, a ratio of {bore_ratio:.2f}. The task "
            f"specifies clearly different radii, so a ratio below "
            f"{MIN_BORE_RATIO} means the two bores were not resolved as "
            "different and any h*r comparison built on them would be "
            "meaningless",
            bore_narrow_px=narrow["bore"], bore_wide_px=wide["bore"],
            bore_ratio=bore_ratio, rise_narrow_px=hn, rise_wide_px=hw)
    elif hn <= 1.0 or hw <= 1.0:
        M1.fail(f"measured rise is not positive in both tubes "
                f"(narrow {hn:.1f} px, wide {hw:.1f} px), so no capillary rise "
                "took place to compare",
                rise_narrow_px=hn, rise_wide_px=hw)
    else:
        M1.succeed(abs((hn * rn) / (hw * rw) - 1.0),
                   rise_narrow_px=hn, rise_wide_px=hw,
                   radius_narrow_px=rn, radius_wide_px=rw,
                   product_narrow=hn * rn, product_wide=hw * rw,
                   rise_ratio_narrow_over_wide=hn / hw,
                   expected_rise_ratio=rw / rn)
    return _finish(res, ctx, clip, (narrow, wide), series, (hn, hw))


TUBE_LINE_GAP = 34      # px: lines closer than this belong to the same tube
# Both tubes hang from one clamp directly over the tank, so they sit in the
# middle of the frame. The clamp stand's own pole is off to the side and has
# edges just as strong, so it is excluded by position rather than by strength.
TUBE_X_RANGE = (0.20, 0.88)
MAX_TUBE_WIDTH_FRAC = 0.09


def _find_tubes_by_column(tailf: np.ndarray) -> list[dict]:
    """The two bores, taken as the risen liquid columns themselves.

    With a tinted liquid this is the most direct measurement available: the
    column standing inside a tube is a coloured bar whose width is that tube's
    bore and whose top is that tube's meniscus, which are exactly the two
    quantities Jurin's law relates. Nothing has to be inferred from wall edges,
    which is what previously let a clamp stand pass as a tube and let a
    thick-walled narrow bore read as almost the same width as a wide one.
    """
    mask = liquid.liquid_mask(tailf)
    if mask is None:
        return []
    h, w = mask.shape
    # The reservoir fills the lower part of the frame; the columns are the parts
    # of the liquid that stand above it, so the tank body is cut away by taking
    # the rows where the liquid does not span most of the width.
    width_per_row = mask.sum(axis=1)
    tank_rows = width_per_row >= 0.35 * w
    top_of_tank = int(np.argmax(tank_rows)) if tank_rows.any() else h
    if top_of_tank < 20:
        return []
    upper = np.zeros_like(mask)
    upper[:top_of_tank] = mask[:top_of_tank]

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        upper.astype(np.uint8), 8)
    cands = []
    for i in range(1, n):
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if bw < MIN_BORE_PX or bw > MAX_TUBE_WIDTH_FRAC * w or bh < 8:
            continue
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        y0 = int(stats[i, cv2.CC_STAT_TOP])
        # How well the bore is resolved: the fraction of the column's own
        # height over which the coloured band actually holds the full width
        # the bore is quoted as. A clean column holds it all the way; a band
        # that is ragged, partly occluded or tapering does not, and the metric
        # is meant to say so rather than assert the bore is certain.
        rows = labels[y0:y0 + bh, x0:x0 + bw] == i
        held = (rows.sum(axis=1) >= 0.8 * bw)
        cands.append({"x_left": float(x0), "x_right": float(x0 + bw),
                      "bore": float(bw), "lines": 2,
                      "strength": float(stats[i, cv2.CC_STAT_AREA]),
                      "support": float(held.mean()),
                      "y_end": float(top_of_tank)})
    if len(cands) < 2:
        return []
    cands.sort(key=lambda r: -r["strength"])
    return sorted(cands[:2], key=lambda r: r["x_left"])


def _find_tubes(head: np.ndarray, tailf: np.ndarray) -> list[dict]:
    """The two tubes, as the two tight clusters of vertical lines above the tank.

    A glass tube in this view is not one edge but a tight bundle of two to four
    parallel vertical lines - its outer walls and its bore walls. That bundle
    structure is what separates a tube from the clamp stand's single-line pole
    and from the tank corners, and it is why the width has to be taken across
    the whole bundle: the two strongest lines of a thick-walled narrow tube are
    often an adjacent inner pair and span almost nothing, which is how the two
    bores came out equal when they plainly are not.
    """
    g = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ax = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    h, w = g.shape
    band = ax[int(0.05 * h):int(0.55 * h)]          # tube bodies, above the tank
    sup = cv2.blur(band.mean(axis=0).reshape(1, -1), (3, 1))[0]

    x_lo, x_hi = int(TUBE_X_RANGE[0] * w), int(TUBE_X_RANGE[1] * w)
    peaks = [x for x in range(max(2, x_lo), min(w - 2, x_hi))
             if sup[x] >= 0.20 * float(sup.max())
             and sup[x] == sup[max(0, x - 3):x + 4].max()]
    if len(peaks) < 2:
        return []

    clusters: list[list[int]] = [[peaks[0]]]
    for x in peaks[1:]:
        if x - clusters[-1][-1] <= TUBE_LINE_GAP:
            clusters[-1].append(x)
        else:
            clusters.append([x])

    cands = []
    for c in clusters:
        width = c[-1] - c[0]
        if not (MIN_BORE_PX <= width <= MAX_TUBE_WIDTH_FRAC * w):
            continue
        if len(c) < 2:
            continue
        cands.append({"x_left": float(c[0]), "x_right": float(c[-1]),
                      "bore": float(width), "lines": len(c),
                      "strength": float(sum(sup[x] for x in c) / len(c))})
    if len(cands) < 2:
        return []

    # The two tubes are the two strongest bundles that do not overlap: an
    # overlapping pair is one tube counted twice.
    cands.sort(key=lambda r: -r["strength"] * r["bore"])
    picked = [cands[0]]
    for c in cands[1:]:
        a, b = picked[0], c
        if min(a["x_right"], b["x_right"]) < max(a["x_left"], b["x_left"]):
            picked.append(c)
            break
    if len(picked) < 2:
        return []
    picked = sorted(picked, key=lambda r: r["x_left"])

    # Confidence: over what fraction of the tube's height its bundle keeps a
    # comparable edge response, which is what a trustworthy width means.
    for r in picked:
        lo, hi = int(r["x_left"]), int(r["x_right"]) + 1
        rows = band[:, lo:hi].max(axis=1)
        on = rows >= 0.20 * float(sup.max())
        r["support"] = float(on.mean())
        # Lower rim of the tube: the last row of its own edge bundle, searched
        # over the full height rather than the body band. In the first frame the
        # tubes hang clear of the water, so the reservoir surface has to lie
        # below this, which is what pins it down.
        full = ax[:, lo:hi].max(axis=1)
        lit = np.flatnonzero(full >= 0.20 * float(sup.max()))
        r["y_end"] = float(lit.max()) if lit.size else float(h)
    return picked


MIN_PROFILE_COLS = 6


def _step_profile(frame: np.ndarray, x0: int, x1: int, band: int = 5
                  ) -> np.ndarray | None:
    """Row-wise CIELAB colour step, averaged over a column range."""
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    if x1 - x0 < MIN_PROFILE_COLS:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi = lab[:, x0:x1]
    below = cv2.blur(np.roll(roi, -band - 1, axis=0), (1, band))
    above = cv2.blur(np.roll(roi, band + 1, axis=0), (1, band))
    prof = cv2.blur(np.linalg.norm(below - above, axis=2), (3, 1)).mean(axis=1)
    # np.roll wraps, so the rows within one band of either frame edge compare
    # the top of the image against its bottom and carry a huge false step. Left
    # in, that fake edge outranks the meniscus and the column top is reported at
    # row 0.
    dead = band + 2
    prof[:dead] = 0.0
    prof[-dead:] = 0.0
    return prof


def _tinted_levels(frame: np.ndarray, t1: dict, t2: dict):
    """Both meniscus heights, from the liquid's own tint.

    Where the liquid is tinted the meniscus is the top edge of the coloured
    region inside that tube's own columns, so nothing has to be scored, ranked
    or searched. That is what removes the failure mode where the tube's rim or
    the clamp above it carried a stronger edge than the liquid and was taken
    for the meniscus.
    """
    mask = liquid.liquid_mask(frame)
    if mask is None:
        return None
    h, w = mask.shape

    def top_edge(x0: int, x1: int) -> float | None:
        tops = []
        for x in range(max(0, x0), min(w, x1)):
            on = np.flatnonzero(mask[:, x])
            if on.size:
                tops.append(float(on.min()))
        return float(np.median(tops)) if len(tops) >= 4 else None

    out = {}
    for key, t in (("narrow", t1), ("wide", t2)):
        mid = 0.5 * (t["x_left"] + t["x_right"])
        half = max(2.0, 0.35 * t["bore"])
        v = top_edge(int(mid - half), int(mid + half) + 1)
        if v is None:
            return None
        out[key] = v
    return out


def _reservoir_y(frame: np.ndarray, cont, t1: dict, t2: dict) -> float | None:
    """Reservoir surface: the row inside the tank where the water's colour starts.

    The same measurement, made the same way, as the melting-ice task: SAM2
    segments the tank, each row inside it is summarised by its median CIELAB
    a/b, and the surface is the row that gains the water's colour going
    downwards. Reading chroma rather than brightness is what keeps the tank's
    upper rim, its base and the reflections down the glass out of the answer -
    an earlier version scored brightness steps and took the rim for the
    surface on the route where the tank is only part full.

    The columns the tubes occupy are excluded: a tube crossing the surface
    carries its own edges down the whole of its length.
    """
    w = frame.shape[1]
    exclude = np.zeros(w, bool)
    for t in (t1, t2):
        exclude[max(0, int(t["x_left"]) - 12):
                min(w, int(t["x_right"]) + 12)] = True
    surf = liquid.surface_by_tint(frame, cont, exclude_cols=exclude)
    if surf is None:
        return None
    return float(surf.y_at(0.5 * (cont.x0 + cont.x1)))


def _column_top(frame: np.ndarray, tube: dict, base: float) -> float | None:
    """Meniscus height inside one tube, searched above the reservoir surface."""
    # Widen a narrow bore symmetrically so the profile has columns to average,
    # while staying between the tube's own walls as far as possible.
    mid = 0.5 * (tube["x_left"] + tube["x_right"])
    half = max(0.5 * tube["bore"], 0.5 * MIN_PROFILE_COLS + 1)
    prof = _step_profile(frame, mid - half, mid + half)
    if prof is None:
        return None
    lo = int(max(0, base - 0.55 * frame.shape[0]))
    hi = int(base - 2)
    if hi - lo < 8:
        return None
    return float(lo + int(np.argmax(prof[lo:hi])))


def _finish(res, ctx, clip, tubes, series, heights):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, tubes, series, heights, res)
    return res


def _debug(clip, ctx, tubes, series, heights, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    last = clip.n - 1

    viz.show_frame(ax[0], clip[last], f"frame {last}: tubes, surface, columns")
    if tubes and len(tubes) == 2:
        # Measure the displayed frame itself. Drawing the tail median here would
        # put the lines where the levels were on average, which is not where
        # they are in the frame underneath whenever the columns are still moving.
        here = _tinted_levels(clip[last], tubes[0], tubes[1])
        # The last entry belongs to the frame shown underneath, which matters
        # now that the reservoir is read per frame rather than frozen.
        base = (float(series["reservoir"][-1])
                if series and series["reservoir"] else None)
        if base is not None:
            ax[0].axhline(base, color="#ffd400", lw=1.8,
                          label=f"reservoir surface y={base:.0f}")
        for t, col, name in zip(tubes, ("#ff3b30", "#0a84ff"),
                                ("narrow", "wide")):
            ax[0].axvline(t["x_left"], color=col, lw=0.9, alpha=0.8)
            ax[0].axvline(t["x_right"], color=col, lw=0.9, alpha=0.8)
            top = here[name] if here else None
            if top is None and base is not None and series[name]:
                top = base - float(np.median(series[name]))
            if top is not None and base is not None:
                ax[0].plot([t["x_left"] - 14, t["x_right"] + 14], [top, top],
                           "-", color=col, lw=2.2,
                           label=f"{name}: bore {t['bore']:.1f} px, "
                                 f"h {base - top:.0f} px in this frame")
        ax[0].legend(loc="lower left", fontsize=7)

    if series and series["narrow"]:
        n = len(series["narrow"])
        ax[1].plot(series["narrow"], "o-", ms=3, color="#ff3b30",
                   label="narrow rise [px]")
        ax[1].plot(series["wide"], "o-", ms=3, color="#0a84ff",
                   label="wide rise [px]")
        ax[1].set_xlabel(f"frame index within the final {n} frames")
        ax[1].set_ylabel("rise  [px]")
        ax[1].set_title("rise over the settled tail")
        ax[1].legend(fontsize=7)

        q = res.metrics["M1"].quantities
        if q.get("product_narrow"):
            ax[2].bar(["h*r narrow", "h*r wide"],
                      [q["product_narrow"], q["product_wide"]],
                      color=["#ff3b30", "#0a84ff"])
            ax[2].set_title("Jurin: the two products should match")
            for i, v in enumerate([q["product_narrow"], q["product_wide"]]):
                ax[2].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom",
                               fontsize=8)
    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 |h1r1/(h2r2)-1| = {fmt('M1')}   |   "
                    f"M2 bore confidence = {fmt('M2')}   "
                    f"M3 meniscus consistency = {fmt('M3')}")
