"""P37: compare independently measured rises above one shared reservoir.

Only M1 is scored: the narrower tube must rise higher than the wider tube.
Radius magnitude, Jurin-product residual and temporal variation remain diagnostic.
A connected tinted reservoir and both tube interiors are required; ambiguous
colour edges are not substituted for a missing reservoir or meniscus.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import replace

from .. import liquid, viz
from ..context import Context
from ..schema import Result
from ..track import camera_drift
from ..video import Clip

TAIL_FRAC = 0.18          # final part of the clip taken as equilibrium
MIN_BORE_PX = 4.0
# The task specifies two clearly different bores, so a measured ratio below this
# means the measurement did not separate them rather than that they are equal.
MIN_BORE_RATIO = 1.25


def _interface_center(frame, columns, proposal):
    """Fit an observed step, or a thin interfacial stroke, in RGB space.

    Three independently fitted color plateaus distinguish the center of a
    drawn/visible meniscus from the first tinted pixel below a thick stroke.
    No rise ratio or tube-radius relation enters this local measurement.
    """
    cols=np.asarray(columns,int);cols=cols[(cols>=0)&(cols<frame.shape[1])]
    if len(cols)<4 or proposal is None:return proposal
    lo=max(0,int(round(proposal))-10);hi=min(frame.shape[0],int(round(proposal))+11)
    profile=np.median(frame[lo:hi,cols,:].astype(float),axis=1);n=len(profile)
    if n<15:return proposal
    def loss(a,b):
        y=profile[a:b];return float(np.sum((y-y.mean(0))**2))
    two=min((loss(0,k)+loss(k,n),k) for k in range(4,n-3))
    three=min((loss(0,a)+loss(a,b)+loss(b,n),a,b) for a in range(4,n-4) for b in range(a+1,min(n-3,a+7)))
    if three[0]<.55*max(two[0],1.):return float(lo+(three[1]+three[2]-1)/2)
    return float(lo+two[1]-.5)


def _refine_bore_edges(frame,tube):
    # Use un-morphed color transitions. Opening/closing is only for identifying
    # columns; its eroded width is not a physical inner-diameter measurement.
    mask=liquid.liquid_mask(frame)
    if mask is None:return tube
    mid=int(round((tube['x_left']+tube['x_right'])/2));on=np.flatnonzero(mask[:,mid])
    if len(on)<15:return tube
    lo=int(on.min())+8;hi=int(tube['y_end'])-8
    if hi-lo<12:return tube
    profile=np.median(frame[lo:hi].astype(float),axis=0)
    # RGB retains luma resolution; isolated chroma edges shift by 1-2 pixels
    # after 4:2:0 video encoding and are unsuitable for thin bore metrology.
    gradient=np.linalg.norm(np.diff(profile,axis=0),axis=1)
    edges=[]
    for side,x in enumerate([tube['x_left'],tube['x_right']]):
        a=max(0,int(round(x))-5);b=min(len(gradient),int(round(x))+5)
        # Select the inner dark-wall transition: left dark-to-liquid,
        # right liquid-to-dark. Magnitude alone may select the OUTER wall.
        light=np.mean(np.diff(profile,axis=0),axis=1)
        support=gradient[a:b]*((light[a:b]>8) if side==0 else (light[a:b]<-8))
        if not np.any(support):return tube
        k=a+int(np.argmax(support));a=max(a,k-1);b=min(b,k+2)
        if gradient[k]<4:return tube
        edges.append(float(np.sum((np.arange(a,b)+.5)*gradient[a:b])/sum(gradient[a:b])))
    if edges[1]-edges[0]<MIN_BORE_PX:return tube
    return {**tube,'x_left':edges[0],'x_right':edges[1],'bore':edges[1]-edges[0],
            'edge_method':'subpixel full-color inner-wall transition before morphology'}


def evaluate(clip: Clip, ctx: Context) -> Result:
    if not ctx.params.get("p37_annotation"):
        return _evaluate(clip, ctx)
    # Measure colour and reviewed geometry independently. Cross-check positions
    # only when both are reliable; use geometry when colour cannot measure M1.
    colored = _evaluate(clip, replace(ctx, params={k:v for k,v in ctx.params.items()
                                                   if k != "p37_annotation"}))
    result = _evaluate(clip, ctx)
    if colored.metrics["M1"].extract_success:
        if result.metrics["M1"].extract_success:
            comparison = _route_agreement(colored.metrics["M1"].quantities,
                                          result.metrics["M1"].quantities)
            result.scene["independent_route_comparison"] = comparison
            if not comparison["agree"]:
                result.metrics["M1"].fail("independent colour and reviewed-geometry routes disagree on observed meniscus positions; neither score selected",
                                           route_comparison=comparison)
                _finish(result,ctx,clip,(result.scene["narrow_tube"],result.scene["wide_tube"]),None,None)
            else:
                colored.scene["independent_route_comparison"] = comparison
                colored.scene["geometry_annotation_used"] = False
                if ctx.debug_path:
                    # Restore the selected colour route's own measurement plot.
                    colored = _evaluate(clip, replace(ctx,params={k:v for k,v in ctx.params.items() if k != "p37_annotation"}))
                    colored.scene["independent_route_comparison"] = comparison
                    colored.scene["geometry_annotation_used"] = False
                return colored
        else:
            colored.scene["geometry_route_unavailable"] = result.metrics["M1"].note
            colored.scene["geometry_annotation_used"] = False
            if ctx.debug_path:
                colored = _evaluate(clip,replace(ctx,params={k:v for k,v in ctx.params.items() if k != "p37_annotation"}))
                colored.scene["geometry_route_unavailable"] = result.metrics["M1"].note
                colored.scene["geometry_annotation_used"] = False
            return colored
    result.scene["colored_route_failure_before_geometry_fallback"] = colored.metrics["M1"].note
    result.scene["geometry_annotation_used"] = True
    return result


def _route_agreement(first,second):
    errors={}
    for side in ("narrow","wide"):
        key="meniscus_"+side+"_px_series"
        a=dict(zip(first["frame_indices"],first[key]));b=dict(zip(second["frame_indices"],second[key]))
        common=sorted(set(a)&set(b));errors[side]={"paired_frames":len(common),"median_absolute_position_difference_px":float(np.median([abs(a[i]-b[i]) for i in common])) if common else None}
    return {"agree":all(d["paired_frames"]>=3 and d["median_absolute_position_difference_px"]<=6. for d in errors.values()),
            "maximum_allowed_median_position_difference_px":6.,"per_tube":errors,
            "policy":"compare independent measured positions, never scores or expected height order"}


def _stable_tail(series,tubes,height):
    failures=[]
    gaps=np.diff(np.asarray(series["frame_indices"],float))
    for side,tube in zip(("narrow","wide"),tubes):
        ys=np.asarray(series["meniscus_"+side],float);speed=np.abs(np.diff(ys))/np.maximum(gaps,1.)
        limit=max(12.,2.*tube["bore"],.03*height)
        bad=np.flatnonzero(speed>limit)
        if len(bad):failures.append({"tube":side,"limit_px_per_frame":limit,"candidate_switches":[{"from_frame":series["frame_indices"][int(i)],"to_frame":series["frame_indices"][int(i)+1],"displacement_px_per_frame":float(speed[i])} for i in bad]})
    return {"valid":not failures,"failures":failures,"policy":"reject unresolved distant interface-candidate switches; do not inflate uncertainty into an equal-height score"}


def _evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    m1 = res.add("M1", "narrow-tube rise exceeds wide-tube rise", principle=(
        "For the same wetting liquid and glass, capillary rise is larger in the "
        "narrower bore. Independently read each meniscus and the shared reservoir "
        "surface at that tube's position. Score 1 only when the measured narrow "
        "rise exceeds the wide rise by more than the image measurement resolution; "
        "equal or lower rises score 0. No inverse-radius magnitude is imposed."),
        ideal=1.0, non_residual=True)
    for key in ("M2", "M3"):
        res.add(key, defined=False).note = "removed from scoring by review; diagnostics only"
    head = np.mean(np.stack([clip[i] for i in range(min(5, clip.n))]),
                   axis=0).astype(np.uint8)
    tailf = np.mean(np.stack([clip[i] for i in range(max(0, clip.n - 5), clip.n)]),
                    axis=0).astype(np.uint8)
    geometry = None
    if ctx.params.get("p37_annotation"):
        from . import clear_liquid
        geometry = clear_liquid.parse_annotation(clip, ctx.params["p37_annotation"])
        tubes = clear_liquid.tubes_from_geometry(geometry)
    else:
        tubes = _find_tubes_by_column(tailf) or _find_tubes(head, tailf)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip), "tubes_found": len(tubes),
                 "measurement_module": __file__, "scoring_policy": "capillary_height_order_only"}
    if len(tubes) != 2:
        m1.fail("two independently identifiable liquid columns / bores are required")
        return _finish(res, ctx, clip, tubes, None, None)
    narrow, wide = sorted((t if geometry else _refine_bore_edges(tailf, t) for t in tubes),
                          key=lambda t: t["bore"])
    bore_ratio = wide["bore"] / max(narrow["bore"], 1e-6)
    res.scene.update({"narrow_tube": narrow, "wide_tube": wide,
                      "bore_ratio_wide_over_narrow": bore_ratio})
    if bore_ratio < MIN_BORE_RATIO:
        m1.fail("the two bore widths are not distinguishable", bore_ratio=bore_ratio)
        return _finish(res, ctx, clip, (narrow, wide), None, None)

    tail = range(max(0, int((1 - TAIL_FRAC) * clip.n)), clip.n)
    series = {"narrow": [], "wide": [], "reservoir": [], "frame_indices": [],
              "reservoir_narrow": [], "reservoir_wide": [], "surface_fit_rms_px": [],
              "meniscus_narrow": [], "meniscus_wide": []}
    if geometry:
        res.scene.update(geometry_annotation=geometry, clear_liquid_observations=[])
    for i in tail:
        frame = clip[i]
        if geometry:
            surface, levels, diagnostic = clear_liquid.observe(frame, geometry, narrow, wide)
            res.scene["clear_liquid_observations"].append({"frame_index":i, **diagnostic})
        else:
            surface = _shared_reservoir_surface(frame, narrow, wide)
            levels = _tinted_levels(frame, narrow, wide) if surface else None
        if surface is None:
            continue
        if levels is None:
            continue
        bn = surface["slope"] * (narrow["x_left"] + narrow["x_right"]) / 2 + surface["intercept"]
        bw = surface["slope"] * (wide["x_left"] + wide["x_right"]) / 2 + surface["intercept"]
        series["narrow"].append(bn - levels["narrow"])
        series["wide"].append(bw - levels["wide"])
        series["reservoir"].append((bn + bw) / 2)
        series["reservoir_narrow"].append(bn)
        series["reservoir_wide"].append(bw)
        series["surface_fit_rms_px"].append(surface["rms_px"])
        series["meniscus_narrow"].append(levels["narrow"])
        series["meniscus_wide"].append(levels["wide"])
        series["frame_indices"].append(i)
    res.scene.update(frames_measured=len(series["narrow"]), tail_frames=len(tail),
                     level_method=("reviewed frame-zero apparatus; independent grayscale meniscus edges and shared free-surface line each frame"
                                   if geometry else "observed tinted menisci and local fitted surface of one connected reservoir"))
    if len(series["narrow"]) < max(3, int(np.ceil(.5 * len(tail)))):
        m1.fail("the shared reservoir and both menisci are not readable in enough tail frames",
                frames_measured=len(series["narrow"]), tail_frames=len(tail))
        return _finish(res, ctx, clip, (narrow, wide), series, None)
    stability=_stable_tail(series,(narrow,wide),clip.h)
    res.scene["tail_observation_stability"]=stability
    if not stability["valid"]:
        m1.fail("tail interface candidates switch discontinuously; stable paired meniscus measurements are unresolved",tail_stability=stability)
        return _finish(res,ctx,clip,(narrow,wide),series,None)
    hn, hw = (float(np.median(series[key])) for key in ("narrow", "wide"))
    differences = np.asarray(series["narrow"]) - np.asarray(series["wide"])
    delta = float(np.median(differences))
    # Independent pixel quantization and observed temporal variation set the
    # resolution. Neither expected ordering nor the radius ratio enters it.
    uncertainty = max(1.0, 1.4826 * float(np.median(np.abs(differences - delta))),
                      float(np.median(series["surface_fit_rms_px"])))
    score = float(delta > uncertainty)
    m1.steps = [
        "identify both bores independently and order them by measured inner width",
        "require one connected tinted reservoir spanning both tubes and their gap",
        "fit its observed free surface outside the tube bands on each measured frame",
        "read each tinted meniscus independently; subtract from the local reservoir surface",
        "take the median paired rise difference over the tail; ties within measured resolution score 0",
        "M1 = 1 if h_narrow - h_wide > resolution_px else 0; M2/M3 excluded",
    ]
    if geometry:
        m1.steps[:4] = [
            "verify reviewed first-frame video/pixel hashes and the two separate inner-bore boxes",
            "read a broad shared free-surface edge outside the bores in each frame; geometry alone supplies no liquid level",
            "independently select each tube-local grayscale meniscus edge, rejecting exterior/background edges and ambiguity",
            "subtract each observed meniscus from the observed shared surface at that tube's own x coordinate",
        ]
    m1.succeed(score, rise_narrow_px=hn, rise_wide_px=hw,
               radius_narrow_px=narrow["bore"] / 2, radius_wide_px=wide["bore"] / 2,
               rise_difference_px=delta, comparison_resolution_px=uncertainty,
               comparison="narrow_higher" if score else "equal_or_lower_at_measurement_resolution",
               rise_narrow_px_series=series["narrow"], rise_wide_px_series=series["wide"],
               reservoir_narrow_px_series=series["reservoir_narrow"],
               reservoir_wide_px_series=series["reservoir_wide"],
               frame_indices=series["frame_indices"],
               meniscus_narrow_px_series=series["meniscus_narrow"],
               meniscus_wide_px_series=series["meniscus_wide"],
               shared_reservoir_verified=True,
               product_narrow=hn * narrow["bore"] / 2,
               product_wide=hw * wide["bore"] / 2,
               bore_support_diagnostic=[narrow["support"], wide["support"]])
    res.scene["reservoir_y_px"] = float(np.median(series["reservoir"]))
    return _finish(res, ctx, clip, (narrow, wide), series, (hn, hw))


def _shared_reservoir_surface(frame, narrow, wide):
    """Fit the free surface of the SAME visible reservoir, outside both bores.

    Requiring a broad continuous body across the inter-tube gap rejects separate
    vessels. Local line heights remove apparent order induced by surface slope.
    A missing tint/body is a missing measurement, never a guessed edge.
    """
    mask = liquid.liquid_mask(frame)
    if mask is None:
        return None
    h, w = mask.shape
    broad_rows = np.flatnonzero(mask.sum(axis=1) >= .35 * w)
    if len(broad_rows) < max(10, .025 * h):
        return None
    # Sample inside the broad liquid body, avoiding its bottom and surface.
    body_row = int(broad_rows[min(len(broad_rows) - 1, max(5, len(broad_rows) // 3))])
    on = np.flatnonzero(mask[body_row])
    if len(on) < .35 * w:
        return None
    x0, x1 = int(on.min()), int(on.max()) + 1
    centers = [(t["x_left"] + t["x_right"]) / 2 for t in (narrow, wide)]
    if not all(x0 < x < x1 for x in centers):
        return None
    exclude = np.zeros(w, bool)
    for t in (narrow, wide):
        margin = max(5, int(.2 * t["bore"]))
        exclude[max(0, int(t["x_left"]) - margin):min(w, int(t["x_right"]) + margin + 1)] = True
    left, right = sorted(centers)
    middle = np.arange(int(left), int(right) + 1)
    middle = middle[~exclude[middle]]
    if len(middle) < 8 or np.mean(mask[body_row, middle]) < .90:
        return None
    # Interior band excludes vessel-wall refraction. Do not throw away sloped
    # surface endpoints by trimming y quantiles: fit and trim residuals instead.
    margin = max(5, int(.05 * (x1 - x0)))
    xs, ys = [], []
    for x in range(x0 + margin, x1 - margin, max(1, w // 200)):
        if exclude[x] or not mask[body_row, x]:
            continue
        on = np.flatnonzero(mask[:body_row + 1, x])
        if len(on) < 5:
            continue
        xs.append(x); ys.append(float(on.min()))
    if len(xs) < 20:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    good = np.ones(len(xs), bool)
    for _ in range(3):
        slope, intercept = np.polyfit(xs[good], ys[good], 1)
        residual = np.abs(ys - (slope * xs + intercept))
        good = residual <= max(2., 3 * 1.4826 * np.median(residual[good]))
        if good.sum() < max(20, .7 * len(xs)):
            return None
    rms = float(np.sqrt(np.mean((ys[good] - slope * xs[good] - intercept) ** 2)))
    if rms > max(2., .005 * h):
        return None
    # Refine independently in short local bands. This reads thick surface
    # strokes at their center while retaining the observed surface tilt.
    refined_x, refined_y = [], []
    for chunk in np.array_split(xs[good], max(3, len(xs[good]) // 8)):
        if len(chunk) < 4:
            continue
        cx = float(np.mean(chunk)); proposal = float(slope * cx + intercept)
        refined_x.append(cx); refined_y.append(_interface_center(frame, chunk, proposal))
    if len(refined_x) >= 3:
        slope, intercept = np.polyfit(refined_x, refined_y, 1)
    return {"slope": float(slope), "intercept": float(intercept), "rms_px": rms,
            "support_columns": int(good.sum()), "connected_shared_body": True}


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
    # Estimate the reservoir edge independently before cutting away its body.
    # A single horizontal cut turns a tilted surface into a false broad column.
    body_rows = np.flatnonzero(tank_rows)
    if len(body_rows) < 10:
        return []
    body_y = int(body_rows[len(body_rows) // 3])
    xs = np.flatnonzero(mask[body_y])
    ys = np.asarray([np.flatnonzero(mask[:body_y + 1, x]).min() for x in xs])
    good = np.ones(len(xs), bool)
    for _ in range(6):
        slope, intercept = np.polyfit(xs[good], ys[good], 1)
        residual = ys - (slope * xs + intercept)
        center = float(np.median(residual[good]))
        tolerance = max(2., 3 * 1.4826 * np.median(np.abs(residual[good] - center)))
        good = np.abs(residual - center) <= tolerance
        if good.sum() < .6 * len(xs):
            return []
    surface_rows = slope * np.arange(w) + intercept
    upper = mask & (np.arange(h)[:, None] < surface_rows[None, :] - 4)

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
                      "y_end": float(slope * (x0 + bw / 2) + intercept)})
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

    # A resolved thick wall can itself have two peaks. Do not promote that
    # single wall to a fictitious narrow tube when no column has yet risen.
    # Join the two isolated wall bundles only when their whole bore fits the
    # declared maximum tube width; keep already complete multi-wall bundles.
    paired = []
    index = 0
    while index < len(clusters):
        group = clusters[index]
        if (len(group) <= 2 and index + 1 < len(clusters)
                and len(clusters[index + 1]) <= 2
                and clusters[index + 1][-1] - group[0] <= MAX_TUBE_WIDTH_FRAC * w):
            group = group + clusters[index + 1]
            index += 1
        paired.append(group)
        index += 1
    clusters = paired

    cands = []
    for c in clusters:
        width = c[-1] - c[0]
        if not (MIN_BORE_PX <= width <= MAX_TUBE_WIDTH_FRAC * w):
            continue
        if len(c) < 3:
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

    # The global mask identifies the liquid's hue/body, but its 5x5 opening and
    # largest-component filter can erase a faint narrow column. Read original
    # pixels in each tube instead of requiring that fragile global connection.
    lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB).astype(float)[...,1:]-128.
    body=np.median(lab[mask],axis=0);magnitude=float(np.linalg.norm(body))
    if magnitude<8:return None
    direction=body/magnitude;projection=np.sum(lab*direction,axis=2)
    norm=np.linalg.norm(lab,axis=2)
    local=((projection>=max(3.,.12*magnitude)) &
           (projection/np.maximum(norm,1.)>=.75) &
           (cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[...,2]>=40))

    def top_edge(x0: int, x1: int) -> float | None:
        tops = []
        for x in range(max(0, x0), min(w, x1)):
            on = np.flatnonzero(local[:, x])
            if on.size >= 8:
                # liquid_mask keeps one connected body: the tube interior must
                # be a sustained vertical column, not a lone chromatic pixel.
                starts = on[np.r_[True, np.diff(on) > 1]]
                ends = on[np.r_[np.diff(on) > 1, True]]
                runs = [(a, b) for a, b in zip(starts, ends) if b - a >= 7]
                if runs:
                    tops.append(float(runs[0][0]))
        if len(tops)<2:return None
        # Require several agreeing original columns. Reservoir-only columns
        # outside a narrow tinted core must not outvote its visible upper top.
        tops=np.asarray(tops,float);minimum=max(2,int(np.ceil(.2*(x1-x0))))
        for proposal in np.sort(tops):
            agreeing=np.abs(tops-proposal)<=4
            if int(agreeing.sum())>=minimum:return float(np.median(tops[agreeing]))
        return None

    out = {}
    for key, t in (("narrow", t1), ("wide", t2)):
        mid = 0.5 * (t["x_left"] + t["x_right"])
        half = max(2.0, 0.35 * t["bore"])
        v = top_edge(int(mid - half), int(mid + half) + 1)
        if v is None:
            return None
        columns=np.arange(int(mid-half),int(mid+half)+1)
        out[key] = _interface_center(frame,columns,v)
    return out


def _finish(res, ctx, clip, tubes, series, heights):
    if ctx.debug_path:
        res.debug_image = _debug(clip, ctx, tubes, series, heights, res)
    return res


def _debug(clip, ctx, tubes, series, heights, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.0)
    last = series["frame_indices"][-1] if series and series.get("frame_indices") else clip.n - 1

    viz.show_frame(ax[0], clip[last], f"frame {last}: tubes, surface, columns")
    if tubes and len(tubes) == 2:
        # Measure the displayed frame itself. Drawing the tail median here would
        # put the lines where the levels were on average, which is not where
        # they are in the frame underneath whenever the columns are still moving.
        here = ({"narrow": series["meniscus_narrow"][-1], "wide": series["meniscus_wide"][-1]}
                if res.scene.get("geometry_annotation") and series and series.get("meniscus_narrow")
                else _tinted_levels(clip[last], tubes[0], tubes[1]))
        # The last entry belongs to the frame shown underneath, which matters
        # now that the reservoir is read per frame rather than frozen.
        base = (float(series["reservoir"][-1])
                if series and series["reservoir"] else None)
        if base is not None:
            centers = [(t["x_left"] + t["x_right"]) / 2 for t in tubes]
            local_bases = [series["reservoir_narrow"][-1], series["reservoir_wide"][-1]]
            ax[0].plot(centers, local_bases, color="#ffd400", lw=1.8,
                       label="same reservoir: local surface reference")
        for t, col, name in zip(tubes, ("#ff3b30", "#0a84ff"),
                                ("narrow", "wide")):
            ax[0].axvline(t["x_left"], color=col, lw=0.9, alpha=0.8)
            ax[0].axvline(t["x_right"], color=col, lw=0.9, alpha=0.8)
            top = here[name] if here else None
            local_base = series["reservoir_" + name][-1] if base is not None else None
            if top is not None and local_base is not None:
                ax[0].plot([t["x_left"] - 14, t["x_right"] + 14], [top, top],
                           "-", color=col, lw=2.2,
                           label=f"{name}: bore {t['bore']:.1f} px, "
                                 f"h {local_base - top:.0f} px in this frame")
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
            ax[2].bar(["narrow rise", "wide rise"],
                      [q["rise_narrow_px"], q["rise_wide_px"]],
                      color=["#ff3b30", "#0a84ff"])
            ax[2].set_title("Narrow rise must exceed wide rise")
            for i, v in enumerate([q["rise_narrow_px"], q["rise_wide_px"]]):
                ax[2].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom",
                               fontsize=8)
    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:.4f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 narrow higher = {fmt('M1')} (M2/M3 not scored)")
