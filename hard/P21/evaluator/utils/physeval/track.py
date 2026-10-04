"""Trajectory extraction for the coloured-object tasks.

One method, used everywhere a ball or bob has to be followed: segment the object
by the colour it actually has in the first frame, then take the sub-pixel centroid
of the connected component nearest the previous position.

Why this and not a learned point tracker: these clips put one or two
high-chroma objects on a plain, static, low-saturation background, which is the
regime where colour segmentation is exact rather than approximate. It gives a
sub-pixel centroid of the whole object every frame, whereas a point tracker gives
a surface point that drifts as the object rotates, and it has no failure mode
that needs a second estimator to cover. The nearest-component rule is what keeps
two same-coloured objects from swapping identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .video import Clip


def chroma(frame_bgr: np.ndarray) -> np.ndarray:
    """CIELAB chroma sqrt(a*^2 + b*^2), i.e. colourfulness independent of
    lightness.

    Preferred over HSV saturation for picking out coloured markers: saturation
    rises on any mid-tone surface, so bare wood and a yellow sticker on wood can
    share a saturation band, whereas their chroma differs by a factor of two.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)


@dataclass
class ColorTarget:
    """A colour-defined object, seeded from one frame."""

    hue: float           # OpenCV hue 0-179
    sat_min: int
    val_min: int
    x: float
    y: float
    radius: float
    hue_tol: float = 14.0
    chroma_min: float = 0.0
    label: str = ""

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h = hsv[..., 0].astype(np.int16)
        d = np.abs(h - int(round(self.hue)))
        d = np.minimum(d, 180 - d)  # hue is circular
        m = ((d <= self.hue_tol) &
             (hsv[..., 1] >= self.sat_min) &
             (hsv[..., 2] >= self.val_min))
        if self.chroma_min > 0:
            m &= chroma(frame_bgr) >= self.chroma_min
        return m.astype(np.uint8)


@dataclass
class Track:
    x: np.ndarray
    y: np.ndarray
    r: np.ndarray
    label: str = ""
    quantities: dict = field(default_factory=dict)

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.x) & np.isfinite(self.y)

    @property
    def coverage(self) -> float:
        return float(self.valid.mean())

    def longest_run(self) -> tuple[int, int]:
        """Longest span of consecutive detected frames, as [start, stop)."""
        best = cur = (0, 0)
        v = self.valid
        i = 0
        while i < len(v):
            if v[i]:
                j = i
                while j < len(v) and v[j]:
                    j += 1
                cur = (i, j)
                if cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
                i = j
            else:
                i += 1
        return best


def find_color_targets(frame: np.ndarray, k: int = 1, min_area: int = 25,
                       max_area_frac: float = 0.08,
                       sat_min: int = 90) -> list[ColorTarget]:
    """Find the ``k`` most chromatic compact blobs, ordered left to right.

    Chromatic blobs are the tracked objects in every task that uses this: the
    scenes are otherwise grey glass, wood and plaster. Blobs larger than
    ``max_area_frac`` of the frame are scenery (a painted wall, a coloured floor)
    rather than the object, so they are excluded.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    mask = ((sat >= sat_min) & (val >= 50)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    area_cap = max_area_frac * frame.shape[0] * frame.shape[1]
    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > area_cap:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Compactness keeps thin coloured strips (a painted rail, a laser line)
        # out of the candidate list.
        if max(w, h) > 4.0 * max(1, min(w, h)):
            continue
        sel = labels == i
        cands.append({"area": float(area), "sat": float(sat[sel].mean()),
                      "hue": float(np.median(hsv[..., 0][sel])),
                      "val": float(val[sel].mean()),
                      "x": float(cents[i][0]), "y": float(cents[i][1]),
                      "r": float(np.sqrt(area / np.pi))})
    if not cands:
        return []
    cands.sort(key=lambda c: c["sat"] * np.sqrt(c["area"]), reverse=True)
    chosen = sorted(cands[:k], key=lambda c: c["x"])
    return [ColorTarget(hue=c["hue"], sat_min=max(60, int(0.55 * c["sat"])),
                        val_min=max(40, int(0.45 * c["val"])),
                        x=c["x"], y=c["y"], radius=c["r"],
                        label=f"obj{i + 1}")
            for i, c in enumerate(chosen)]


def track_target(clip: Clip, target: ColorTarget,
                 max_jump_radii: float = 12.0) -> Track:
    """Follow one colour target through the clip.

    Per frame, keep the component whose centroid is nearest the last known
    position and within ``max_jump_radii`` object radii of it. That single rule
    replaces both a re-detection heuristic and an identity-assignment step.
    """
    n = clip.n
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    prev = np.array([target.x, target.y], float)
    min_area = max(6.0, 0.12 * np.pi * target.radius ** 2)
    jump = max_jump_radii * max(3.0, target.radius)

    for i in range(n):
        m = target.mask(clip[i])
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnt, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        best, best_d = None, np.inf
        for j in range(1, cnt):
            if stats[j, cv2.CC_STAT_AREA] < min_area:
                continue
            d = float(np.hypot(cents[j][0] - prev[0], cents[j][1] - prev[1]))
            if d < best_d:
                best, best_d = j, d
        if best is None or best_d > jump:
            continue
        sel = (labels == best).astype(np.uint8)
        mom = cv2.moments(sel, binaryImage=True)
        if mom["m00"] <= 0:
            continue
        cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        xs[i], ys[i] = cx, cy
        rs[i] = float(np.sqrt(stats[best, cv2.CC_STAT_AREA] / np.pi))
        prev = np.array([cx, cy], float)

    return Track(x=xs, y=ys, r=rs, label=target.label,
                 quantities={"seed_hue": target.hue,
                             "seed_xy": [target.x, target.y],
                             "seed_radius": target.radius})


def find_chroma_markers(frame: np.ndarray, k: int,
                        chroma_min: float = 30.0, min_area: int = 40,
                        max_aspect: float = 3.0,
                        max_area_frac: float = 0.02) -> list[ColorTarget]:
    """Find ``k`` compact, strongly coloured markers, ordered along their own axis.

    One criterion does the work: a marker is a compact patch whose CIELAB chroma
    clears ``chroma_min``. On these scenes that separates the stickers from the
    wooden rod and the grey room in both the photographic and the rendered
    route, and each marker keeps its own hue for tracking.
    """
    C = chroma(frame)
    m = (C >= chroma_min).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    area_cap = max_area_frac * frame.shape[0] * frame.shape[1]

    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > area_cap:
            continue
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(w, h) > max_aspect * max(1, min(w, h)):
            continue
        sel = labels == i
        cands.append({"area": float(area), "chroma": float(C[sel].mean()),
                      "hue": float(np.median(hsv[..., 0][sel])),
                      "sat": float(np.median(hsv[..., 1][sel])),
                      "val": float(np.median(hsv[..., 2][sel])),
                      "x": float(cents[i][0]), "y": float(cents[i][1]),
                      "r": float(np.sqrt(area / np.pi))})
    if len(cands) < k:
        return []
    cands.sort(key=lambda c: c["chroma"] * np.sqrt(c["area"]), reverse=True)
    chosen = cands[:k]
    # Order along the markers' own principal axis so index 1..k runs down the rod.
    pts = np.array([[c["x"], c["y"]] for c in chosen], float)
    d = pts - pts.mean(axis=0)
    axis = np.linalg.svd(d, full_matrices=False)[2][0]
    chosen = [c for _, c in sorted(zip(d @ axis, chosen), key=lambda s: s[0])]

    return [ColorTarget(hue=c["hue"], sat_min=max(50, int(0.5 * c["sat"])),
                        val_min=max(30, int(0.4 * c["val"])),
                        chroma_min=0.7 * chroma_min,
                        x=c["x"], y=c["y"], radius=c["r"],
                        label=f"marker{i + 1}")
            for i, c in enumerate(chosen)]


def background(clip: Clip, sample: int = 25) -> np.ndarray:
    """Temporal median frame, i.e. the static scene.

    Valid because every task in this group specifies a locked-off camera: a pixel
    shows background in the majority of frames unless the object parks on it.
    """
    idx = np.linspace(0, clip.n - 1, min(sample, clip.n)).round().astype(int)
    stack = np.stack([clip[i] for i in np.unique(idx)]).astype(np.float32)
    return np.median(stack, axis=0)


def track_moving_blob(clip: Clip, min_area: int = 20,
                      max_area_frac: float = 0.05,
                      max_aspect: float = 3.0,
                      diff_thresh: float = 18.0,
                      max_jump_px: float = 90.0) -> Track:
    """Follow the one moving compact object against the static scene.

    Used where the object is not defined by colour - a polished steel ball on a
    pale wall has no chroma to segment. The static scene is the temporal median,
    so anything that moves stands out in the per-frame difference; among those
    blobs the object is the compact one, and continuity with the previous
    position separates it from its own shadow and from a second mover.
    """
    bg = background(clip)
    n = clip.n
    xs = np.full(n, np.nan)
    ys = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    area_cap = max_area_frac * clip.w * clip.h
    prev: np.ndarray | None = None
    kern = np.ones((3, 3), np.uint8)

    for i in range(n):
        d = np.abs(clip[i].astype(np.float32) - bg).max(axis=2)
        m = (d >= diff_thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
        cnt, labels, stats, cents = cv2.connectedComponentsWithStats(m, 8)

        cands = []
        for j in range(1, cnt):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < min_area or area > area_cap:
                continue
            w, h = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
            if max(w, h) > max_aspect * max(1, min(w, h)):
                continue
            cands.append((j, float(area), float(cents[j][0]), float(cents[j][1])))
        if not cands:
            continue

        if prev is None:
            # Seed on the strongest mover: the object carries more changed
            # pixels than its soft shadow does.
            j, area, cx, cy = max(cands, key=lambda c: c[1])
        else:
            scored = [(float(np.hypot(c[2] - prev[0], c[3] - prev[1])), c)
                      for c in cands]
            dist, (j, area, cx, cy) = min(scored, key=lambda s: s[0])
            if dist > max_jump_px:
                continue

        sel = (labels == j).astype(np.uint8)
        # Weight the centroid by how strongly each pixel changed, which puts it
        # on the object body rather than on the anti-aliased rim.
        wts = (d * sel).astype(np.float32)
        tot = float(wts.sum())
        if tot <= 0:
            continue
        yy, xx = np.nonzero(sel)
        xs[i] = float((xx * wts[yy, xx]).sum() / tot)
        ys[i] = float((yy * wts[yy, xx]).sum() / tot)
        rs[i] = float(np.sqrt(area / np.pi))
        prev = np.array([xs[i], ys[i]], float)

    return Track(x=xs, y=ys, r=rs, label="mover",
                 quantities={"extractor": "temporal-median background difference",
                             "diff_threshold": diff_thresh})


def camera_drift(clip: Clip, sample: int = 8) -> float:
    """Median background shift in pixels, as a fraction of the frame diagonal.

    Every task in this group is specified with a locked-off camera, so this is a
    scene-validity readout that goes into verbose rather than a gate.
    """
    idx = list(range(0, clip.n, max(1, clip.n // sample)))
    ref = clip.gray(idx[0])
    shifts = []
    for i in idx[1:]:
        cur = clip.gray(i)
        # Phase correlation on the whole frame: the moving object is a small
        # fraction of the pixels, so the peak follows the static background.
        (dx, dy), _ = cv2.phaseCorrelate(ref.astype(np.float32),
                                         cur.astype(np.float32))
        shifts.append(float(np.hypot(dx, dy)))
    return float(np.median(shifts) / clip.diag) if shifts else 0.0
