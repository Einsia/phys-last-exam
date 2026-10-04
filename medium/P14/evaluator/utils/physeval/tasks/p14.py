"""P14 - one point light source, four vertical rods, four shadows on the floor.

Benchmark metrics, taken verbatim:
  M1  the residual of the common intersection obtained by back-extending the
      object-shadow lines: a single point source puts every rod, its base and
      its shadow tip on one line through the source's ground projection, so all
      four lines meet at one point.
  M2  the variance of the independent light-source position estimates.

The colored-rod route uses the established angular lightness profile. For
photographic scenes, reviewed frame-zero identities allow color-independent
rod registration and two-sided local-darkness measurements. Both routes retain
their observations; neither uses the expected light-source location to choose
shadow directions. Unresolved near-parallel intersections remain unavailable.
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import viz
from ..context import Context
from ..schema import Result
from ..track import camera_drift, chroma
from ..video import Clip

MIN_LINES = 3
MIN_ROD_CHROMA = 22.0
# The angular scan around a rod's foot. The ring runs from just clear of the
# rod out to a few rod heights, which is where a cast shadow lies.
ANGLE_STEP_DEG = 2.0
RING_INNER_FACTOR = 0.6
RING_OUTER_FACTOR = 3.2
# How much darker the shadow direction has to be than a typical direction
# around the same rod, in L* units. A floor with no cast shadow on it varies by
# only a couple of units with angle, so this is what keeps the photographic
# route - which has no real shadows - from faking a measurement.
MIN_ANGULAR_CONTRAST = 6.0
# Two shadow lines have to actually cross for their crossing point to mean
# anything. Rods on opposite sides of the lamp give nearly parallel lines.
MIN_PAIR_ANGLE_DEG = 20.0


def evaluate(clip: Clip, ctx: Context) -> Result:
    res = Result(task_id=ctx.task_id, video_path=ctx.video_path,
                 image_path=ctx.image_path, video_prompt=ctx.video_prompt,
                 model=ctx.model, seed=ctx.seed)
    M1 = res.add("M1", "common-intersection residual of the back-extended "
                       "object-shadow lines", principle=(
        "A point source, a rod and that rod's shadow are coplanar, so in the "
        "image the rod and its shadow lie on one line that passes through the "
        "source's ground projection. All rods share the one source, so all such "
        "lines meet at one point. The metric is the RMS distance of those lines "
        "from their least-squares common point, divided by the frame diagonal "
        "so it is resolution-free."), tol=0.05)
    M2 = res.add("M2", "variance of the independent light-source position "
                       "estimates", principle=(
        "Every pair of shadow lines gives its own estimate of where the source "
        "projects. Under a true point source they all agree. The metric is the "
        "mean coordinate standard deviation of the pairwise intersections, "
        "divided by the frame diagonal."), tol=0.05)

    # Late in the clip the lamp is fully on and the shadows are strongest; the
    # geometry is static, so one frame carries the measurement.
    i = int(0.8 * (clip.n - 1))
    frame = clip[i]
    observation = None
    if ctx.params.get("annotation"):
        from .observed_shadows import load_initialization, relocate_rods, relocate_pedestals, shadow_rays, track_feet
        initialization = load_initialization(ctx.params["annotation"], clip)
        tracked = track_feet(clip.frames[:i+1], initialization["objects"])
        rods, registration = relocate_rods(clip[0], frame, initialization["objects"], tracked)
        if len(rods) < MIN_LINES:
            relocated, global_registration = relocate_pedestals(frame, len(initialization["objects"]))
            registration += global_registration
            if len(relocated) >= MIN_LINES:
                rods = relocated
        shadows, observed = shadow_rays(frame, rods)
        observation = {"registration": registration, "shadow_samples": observed,
                       "frame0_sha256": initialization["source_frame0_sha256"],
                       "initializer": "reviewed first-frame identities only",
                       "hidden_shadow_pixels_interpolated": False}
    else:
        shadows = _shadow_rays(frame)
    res.scene = {"camera_drift_frac_diag": camera_drift(clip),
                 "frame_measured": i,
                 "shadows_found": len(shadows),
                 "rod_feet": [s.get("rod_foot_xy", [round(s["cx"], 1), round(s["cy"], 1)])
                              for s in shadows],
                 "measured_line_origins": [[s["cx"], s["cy"]] for s in shadows],
                 "shadow_direction_deg": [round(s["theta_deg"], 1)
                                          for s in shadows],
                 "angular_contrast_L": [round(s["contrast"], 1)
                                        for s in shadows],
                 "extractor": "per rod, the floor's brightness is read on a "
                              "ring around the rod's foot as a function of "
                              "angle; the minimum of that profile is the "
                              "shadow direction, and the ray from the foot "
                              "along it is one object-shadow line"}

    common = [
        f"measure on frame {i}, where the lamp is fully on",
        "find the rods as the compact strongly coloured objects standing on "
        "the floor, and take each rod's foot as the bottom of its outline",
        f"around each foot, read the mean floor lightness along rays from "
        f"{RING_INNER_FACTOR:g} to {RING_OUTER_FACTOR:g} rod heights out, "
        f"every {ANGLE_STEP_DEG:g} degrees, skipping anything that is not "
        "bare floor",
        "the shadow is the minimum of that angular profile, refined to "
        "sub-step accuracy by a parabola through the minimum and its "
        "neighbours; scanning at fixed radius cancels the lamp's radial "
        "falloff, which is far stronger than the shadows themselves",
        f"accept the direction only if the minimum is at least "
        f"{MIN_ANGULAR_CONTRAST:g} L* units below the profile's median, so a "
        "floor with no shadow on it yields nothing",
        "the ray from the foot along that direction is the rod's "
        "object-shadow line",
    ]
    if observation is not None:
        res.scene["observations"] = observation
        res.scene["extractor"] = "registered first-frame rods; continuous dark ridge below both local flanks on the measured frame"
        common = [
            f"measure on the predetermined frame {i}",
            "verify video/frame hashes and reviewed frame-zero rod identities",
            "register each shaft and pedestal using observed image gradients; reject unresolved matches",
            "exclude each pedestal and read center-minus-both-flanks lightness along rays, with no floor-color filter",
            "require sustained local darkness near the pedestal and a continuous observed segment; do not fill hidden pixels",
            "select each ray by observed local contrast and support length, without fitting to the expected source",
        ]
    M1.steps = common + [
        "solve for the point of least squared distance to all the lines",
        "M1 = RMS distance of the lines from that point / frame diagonal",
    ]
    M2.steps = common + [
        f"intersect every pair of lines that meet at more than "
        f"{MIN_PAIR_ANGLE_DEG:g} degrees to get one source estimate per pair; "
        "a pair of nearly parallel lines - which is what two rods on opposite "
        "sides of the lamp give - has no well-defined crossing point",
        "M2 = mean of the x and y standard deviations of those estimates / "
        "frame diagonal",
    ]

    if len(shadows) < MIN_LINES:
        drift = res.scene["camera_drift_frac_diag"]
        res.fail_all(
            f"found {len(shadows)} cast shadows, need at least {MIN_LINES} to "
            "test whether their lines share a point. Background shift over the "
            f"clip is {drift:.3f} of the frame diagonal"
            + ("; actual registration and local-darkness evidence are retained in scene.observations"
               if observation is not None else
               "; the clip does not hold the locked-off framing the task "
               "specifies, so the four-rod scene is not present to measure"
               if drift > 0.02 else
               "; the floor around the rods shows no direction at least "
               f"{MIN_ANGULAR_CONTRAST:g} L* units darker than the rest, so "
               "there is no cast shadow to read"))
        return _finish(res, ctx, frame, shadows, None, None)

    lines = [(np.array([s["cx"], s["cy"]]), s["axis"]) for s in shadows]
    point, rms = _common_point(lines)
    diag = clip.diag
    M1.succeed(rms / diag, intersection_xy=[float(point[0]), float(point[1])],
               residual_px=float(rms), frame_diagonal_px=float(diag),
               n_lines=len(lines))

    ests = _pairwise(lines)
    if len(ests) < 2:
        M2.fail("fewer than two independent line pairs")
    else:
        e = np.asarray(ests, float)
        sd = e.std(axis=0, ddof=1)
        M2.succeed(float(sd.mean() / diag), n_estimates=len(ests),
                   estimate_std_px=[float(sd[0]), float(sd[1])],
                   estimate_mean_xy=[float(e[:, 0].mean()),
                                     float(e[:, 1].mean())])
    return _finish(res, ctx, frame, shadows, point, ests)


MIN_ROD_CHROMA = 22.0
# A shadow starts at its rod's foot, so one end of the patch has to be this
# close to a rod, as a multiple of the rod's own size.
MAX_ROD_GAP_FACTOR = 2.5


def _rods(frame: np.ndarray) -> list[dict]:
    """The rods, as compact strongly coloured objects standing on the floor."""
    C = chroma(frame)
    m = (C >= MIN_ROD_CHROMA).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    h, w = C.shape
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.0004 * h * w or area > 0.05 * h * w:
            continue
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if bh < 2 * bw:
            continue
        out.append({"cx": float(cents[i][0]), "cy": float(cents[i][1]),
                    "size": float(max(bw, bh)),
                    "x0": float(stats[i, cv2.CC_STAT_LEFT]),
                    "y0": float(stats[i, cv2.CC_STAT_TOP]),
                    "x1": float(stats[i, cv2.CC_STAT_LEFT] + bw),
                    "y1": float(stats[i, cv2.CC_STAT_TOP] + bh)})
    return out


def _near_rod(pts: np.ndarray, rods: list[dict]) -> bool:
    """Does this patch have an end close to some rod?

    This is what tells a cast shadow from the darkening along the edge of the
    lit plate: the plate edge is a long dark band too, but it runs around the
    boundary of the floor with no object at either end, while a shadow is
    anchored at the foot of the rod that casts it.
    """
    for r in rods:
        dx = np.maximum.reduce([r["x0"] - pts[:, 0], pts[:, 0] - r["x1"],
                                np.zeros(pts.shape[0])])
        dy = np.maximum.reduce([r["y0"] - pts[:, 1], pts[:, 1] - r["y1"],
                                np.zeros(pts.shape[0])])
        if float(np.min(np.hypot(dx, dy))) <= MAX_ROD_GAP_FACTOR * r["size"]:
            return True
    return False


def _lit_floor(lab: np.ndarray) -> np.ndarray:
    """The lit floor the shadows are cast on.

    In the rendered route the floor is a plate inset in a black surround. A
    shadow that runs off the edge of the plate merges with that surround into
    one huge dark region, and then no per-patch test can see it as a shadow.
    Restricting the search to the plate cuts each shadow off at the plate edge
    and leaves it as its own component. The plate is found as the largest
    bright region, filled in through its outline so that the shadows and rods
    sitting on it - which are darker - stay part of the floor.
    """
    h, w = lab.shape
    thr = max(12, int(0.22 * float(lab.max())))
    m = (lab > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones((h, w), np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < 0.15 * h * w:
        return np.ones((h, w), np.uint8)
    comp = (labels == big).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return out


def _shadow_rays(frame: np.ndarray) -> list[dict]:
    """For each rod, the direction its shadow leaves the rod's foot in."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    lum = cv2.GaussianBlur(lab, (5, 5), 0).astype(np.float32)
    h, w = lum.shape
    # Sample only bare floor: outside it there is no shadow to read, and the
    # rods themselves are dark objects that would masquerade as one.
    floor = (_lit_floor(lab) > 0) & (chroma(frame) < MIN_ROD_CHROMA)
    angles = np.deg2rad(np.arange(0.0, 360.0, ANGLE_STEP_DEG))
    out = []
    for rod in _rods(frame):
        base = np.array([rod["cx"], rod["y1"]], dtype=np.float64)
        radii = np.arange(RING_INNER_FACTOR * rod["size"],
                          RING_OUTER_FACTOR * rod["size"], 2.0)
        if radii.size < 8:
            continue
        prof = np.full(angles.size, np.nan)
        for j, a in enumerate(angles):
            xi = np.rint(base[0] + radii * np.cos(a)).astype(int)
            yi = np.rint(base[1] + radii * np.sin(a)).astype(int)
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            if np.count_nonzero(ok) < 0.6 * radii.size:
                continue
            xi, yi = xi[ok], yi[ok]
            on = floor[yi, xi]
            if np.count_nonzero(on) < 0.5 * radii.size:
                continue
            prof[j] = float(lum[yi[on], xi[on]].mean())
        if np.count_nonzero(~np.isnan(prof)) < angles.size // 3:
            continue
        level = float(np.nanmedian(prof))
        filled = np.where(np.isnan(prof), level, prof)
        # Smooth around the circle; a shadow subtends a wide angle, so this
        # only removes speckle.
        kern = np.ones(5) / 5.0
        sm = np.convolve(np.concatenate([filled[-4:], filled, filled[:4]]),
                         kern, mode="same")[4:4 + filled.size]
        j = int(np.argmin(sm))
        contrast = level - float(sm[j])
        if contrast < MIN_ANGULAR_CONTRAST:
            continue
        # Refine to below the scan step by fitting a parabola to the minimum
        # and its two neighbours around the circle.
        y0, y1, y2 = sm[(j - 1) % sm.size], sm[j], sm[(j + 1) % sm.size]
        den = y0 - 2.0 * y1 + y2
        shift = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
        theta = np.deg2rad((j + float(np.clip(shift, -1, 1))) * ANGLE_STEP_DEG)
        out.append({"cx": float(base[0]), "cy": float(base[1]),
                    "axis": np.array([np.cos(theta), np.sin(theta)]),
                    "theta_deg": float(np.rad2deg(theta) % 360.0),
                    "contrast": contrast, "level": level,
                    "profile": sm, "half_len": float(2.6 * rod["size"]),
                    "rod_size": rod["size"]})
    return out


def _line_form(p: np.ndarray, d: np.ndarray):
    """Line through point ``p`` with direction ``d`` as (unit normal, offset)."""
    nrm = np.linalg.norm(d)
    if nrm < 1e-9:
        return None
    d = d / nrm
    nvec = np.array([-d[1], d[0]])
    return nvec, float(nvec @ p)


def _common_point(lines):
    """Least-squares point closest to all lines, and the RMS distance to them."""
    forms = [f for f in (_line_form(p, d) for p, d in lines) if f is not None]
    A = np.asarray([n for n, _ in forms])
    b = np.asarray([c for _, c in forms])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = np.array([abs(n @ p - c) for n, c in forms], float)
    return p, float(np.sqrt(np.mean(d ** 2)))


def _pairwise(lines):
    """Where each pair of lines crosses, for pairs that cross definitely.

    Two rods on opposite sides of the lamp lie almost on one line through it,
    so their two shadow lines are nearly parallel and where they cross says
    almost nothing: a fraction of a degree of direction error moves the
    crossing point by thousands of pixels. Such a pair is not a second opinion
    about the source position, it is an ill-conditioned calculation, and
    including it makes the spread of the estimates a measure of the geometry's
    conditioning rather than of the scene's consistency.
    """
    out = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (p1, d1), (p2, d2) = lines[i], lines[j]
            sin_between = abs(float(d1[0] * d2[1] - d1[1] * d2[0]))
            if sin_between < np.sin(np.deg2rad(MIN_PAIR_ANGLE_DEG)):
                continue
            f1, f2 = _line_form(p1, d1), _line_form(p2, d2)
            if f1 is None or f2 is None:
                continue
            (n1, c1), (n2, c2) = f1, f2
            out.append(np.linalg.solve(np.stack([n1, n2]),
                                       np.array([c1, c2])))
    return out


def _finish(res, ctx, frame, shadows, point, ests):
    if ctx.debug_path:
        res.debug_image = _debug(ctx, frame, shadows, point, ests, res)
    return res


PALETTE = ("#ff3b30", "#0a84ff", "#ffd400", "#34c759")


def _debug(ctx, frame, shadows, point, ests, res) -> str:
    fig, ax = viz.figure(ncols=3, width_each=5.2, height=4.4)
    h, w = frame.shape[:2]

    viz.show_frame(ax[0], frame,
                   "rod feet, the shadow direction found at each, common point")
    for s, col in zip(shadows, PALETTE):
        c = np.array([s["cx"], s["cy"]])
        d = s["axis"] * s["half_len"]
        # Draw the ray the way it was measured: outward from the foot along the
        # shadow, and back the other way to the point it is tested against.
        ax[0].annotate("", xy=(c[0] + d[0], c[1] + d[1]), xytext=(c[0], c[1]),
                       arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax[0].scatter(*c, s=34, color=col, edgecolor="k", lw=0.5, zorder=4)
        if s.get("observed_points_xy"):
            measured = np.asarray(s["observed_points_xy"])
            ax[0].scatter(measured[:, 0], measured[:, 1], s=2, color=col)
        if point is not None:
            ax[0].plot([c[0], point[0]], [c[1], point[1]], ":", color=col,
                       lw=1.1)
    if point is not None:
        ax[0].scatter(*point, s=130, marker="x", color="#ffffff", zorder=5,
                      lw=2.5, label="common point = light source")
        ax[0].legend(loc="upper right", fontsize=7)
    ax[0].set_xlim(0, w)
    ax[0].set_ylim(h, 0)

    # The profile each direction was read off, so the reader can see the
    # shadow as the dip it is rather than take the arrow on trust.
    for s, col in zip(shadows, PALETTE):
        prof = s["profile"]
        deg = np.arange(prof.size) * ANGLE_STEP_DEG
        ax[1].plot(deg, prof, "-", color=col, lw=1.3)
        ax[1].axvline(s["theta_deg"], color=col, ls="--", lw=1.0)
        value = (prof[int(round(s["theta_deg"] / ANGLE_STEP_DEG)) % len(prof)]
                 if "observations" in res.scene else s["level"] - s["contrast"])
        ax[1].scatter([s["theta_deg"]], [value], s=30,
                      color=col, zorder=4)
    if shadows:
        ax[1].set_xlim(0, 360)
        ax[1].set_xticks([0, 90, 180, 270, 360])
    ax[1].set_xlabel("direction from the rod's foot  [deg]")
    local = "observations" in res.scene
    ax[1].set_ylabel("negative observed ridge strength" if local else "mean floor lightness on the ring  [L*]")
    ax[1].set_title("observed local darkness and support" if local else "the shadow is the dip in each profile")
    ax[1].grid(alpha=0.3)

    if ests:
        e = np.asarray(ests, float)
        ax[2].scatter(e[:, 0], e[:, 1], s=30, color="#0a84ff",
                      label="pairwise source estimates")
        if point is not None:
            ax[2].scatter(*point, s=110, marker="x", color="#ff3b30",
                          label="least-squares point")
        ax[2].invert_yaxis()
        ax[2].set_xlabel("x  [px]")
        ax[2].set_ylabel("y  [px]")
        ax[2].legend(fontsize=7)
    ax[2].set_title("agreement of the independent estimates")

    def fmt(k):
        m = res.metrics[k]
        return "n/a" if m.value is None else f"{m.value:.5f}"
    return viz.save(fig, ctx.debug(),
                    f"{ctx.task_id}  M1 intersection residual / diagonal = "
                    f"{fmt('M1')}   |   M2 estimate spread / diagonal = "
                    f"{fmt('M2')}   ({len(shadows)} shadows)")
