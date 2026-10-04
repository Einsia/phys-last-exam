"""Container and liquid-surface measurement, backed by SAM2.

The benchmark names "SAM2 + waterline detection" as the tool for the floating-ice
and melting tasks, and the same capability is what the free-surface task needs.
SAM2 supplies the one thing plain edge logic cannot: which pixels are the
container. Once the container is known the surface is unambiguous, because inside
it there is exactly one long horizontal colour step - the room's own horizontal
boundaries (bench edge, skirting, whiteboard rim) are all outside.

The model is loaded through transformers, which carries SAM2 natively, so the
evaluator needs no separate sam2 checkout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .track import chroma

import cv2
import numpy as np

import sys
from pathlib import Path
_PACKAGE_ROOT = next(candidate for parent in Path(__file__).resolve().parents for candidate in (parent, parent / "_shared") if (candidate / "task_catalog.json").is_file() and (candidate / "shared/model_paths.py").is_file())
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))
from shared.model_paths import model_path
DEFAULT_CKPT = os.environ.get('SAM2_CKPT') or str(model_path(
    'sam2.1-hiera-large-transformers', 'EVALUATOR_SAM2_LARGE_TRANSFORMERS_MODEL'))


@lru_cache(maxsize=1)
def _sam2(ckpt: str, device: str):
    import torch
    from transformers import Sam2Model, Sam2Processor
    proc = Sam2Processor.from_pretrained(ckpt)
    model = Sam2Model.from_pretrained(ckpt, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    return proc, model.to(device).eval(), torch


def segment(frame: np.ndarray, points: list[tuple[int, int]],
            labels: list[int] | None = None,
            ckpt: str = DEFAULT_CKPT, device: str = "auto") -> np.ndarray:
    """Boolean mask for the object indicated by the prompt points."""
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_num_threads(int(os.environ.get("EVALUATOR_TORCH_THREADS", "8")))
    proc, model, torch = _sam2(ckpt, device)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pts = [[[list(map(int, p)) for p in points]]]
    lbl = [[list(labels or [1] * len(points))]]
    inp = proc(images=rgb, input_points=pts, input_labels=lbl,
               return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inp, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.float().cpu(),
                                    inp["original_sizes"])
    return np.asarray(masks[0][0, 0].numpy(), bool)


@dataclass
class Container:
    mask: np.ndarray
    x0: int
    x1: int
    y0: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def interior(self, inset_frac: float = 0.10) -> tuple[int, int]:
        """Column range strictly inside the walls."""
        d = max(3, int(inset_frac * self.width))
        return self.x0 + d, self.x1 - d


def find_container_by_walls(frame: np.ndarray, min_support: float = 0.10,
                            gap_frac: tuple[float, float] = (0.15, 0.85)
                            ) -> Container | None:
    """Locate a straight-walled vessel from its two vertical wall edges.

    Every vessel in these tasks is specified as straight-walled and transparent,
    which makes its two sides the longest vertical edges in the frame. Taking the
    best-supported pair is deterministic and repeats frame to frame, where a
    point-prompted segmentation does not: prompting inside the vessel returns
    whichever of the glass, the water or a floating block happens to sit under
    the point, and that choice changes with the frame.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ax = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    h, w = g.shape
    strong = ax >= np.percentile(ax, 98.5)
    support = cv2.blur(strong.sum(axis=0).astype(np.float32).reshape(1, -1),
                       (5, 1))[0]

    peaks = []
    for x in range(3, w - 3):
        if support[x] >= min_support * h and \
                support[x] == support[max(0, x - 6):x + 7].max():
            peaks.append((x, float(support[x])))
    keep: list[tuple[int, float]] = []
    for x, s in sorted(peaks, key=lambda p: -p[1]):
        if all(abs(x - x2) > 15 for x2, _ in keep):
            keep.append((x, s))
    if len(keep) < 2:
        return None

    best = None
    for i, (xa, sa) in enumerate(keep):
        for xb, sb in keep[i + 1:]:
            lo, hi = min(xa, xb), max(xa, xb)
            if not (gap_frac[0] * w <= hi - lo <= gap_frac[1] * w):
                continue
            score = min(sa, sb)
            if best is None or score > best[0]:
                best = (score, lo, hi)
    if best is None:
        return None
    _, x0, x1 = best

    # Vertical extent: the rows over which either wall actually carries an edge.
    rows = np.flatnonzero(strong[:, max(0, x0 - 3):x0 + 4].any(axis=1) |
                          strong[:, max(0, x1 - 3):x1 + 4].any(axis=1))
    if rows.size < 20:
        return None
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1 + 1] = True
    return Container(mask=mask, x0=x0, x1=x1 + 1, y0=y0, y1=y1)


SEED_HEIGHTS = (0.45, 0.62, 0.78, 0.90)


def find_container(frame: np.ndarray, seed_x: float = 0.5,
                   seed_heights: tuple[float, ...] = SEED_HEIGHTS,
                   **kw) -> Container | None:
    """Segment the vessel by prompting down its vertical centreline.

    A single prompt is not enough: a floating block can sit exactly where the
    point lands, and SAM2 then returns the block instead of the glass holding
    it. The vessel is the object that contains the others, so every mask a
    centreline prompt returns is either the vessel itself or something inside
    it, and their union is bounded by the vessel. Taking the union rather than
    the largest single mask is what makes this stable: SAM2 quite often returns
    only part of a pale glass, and a partial vessel is still the largest of the
    candidates, so the maximum inherits the truncation while the union does
    not - the water and the block inside fill the part that was cut off.
    """
    h, w = frame.shape[:2]
    union = None
    for fy in seed_heights:
        m = segment(frame, [(int(seed_x * w), int(fy * h))], **kw)
        area = int(m.sum())
        if area < 0.01 * h * w or area > 0.85 * h * w:
            continue
        ys, xs = np.nonzero(m)
        # A vessel standing on a bench is fully inside the frame with headroom
        # above it, so a mask touching the top or side borders is the room, not
        # the vessel.
        if ys.min() <= 1 or xs.min() <= 1 or xs.max() >= w - 2:
            continue
        union = m if union is None else (union | m)
    if union is None:
        return None
    ys, xs = np.nonzero(union)
    return Container(mask=union, x0=int(xs.min()), x1=int(xs.max()) + 1,
                     y0=int(ys.min()), y1=int(ys.max()) + 1)


@dataclass
class Waterline:
    y: float              # height at the mid-column of the searched range
    slope: float
    x0: int
    x1: int
    rms_px: float
    step: float
    profile: np.ndarray   # per-row step score inside the container

    def y_at(self, x: float) -> float:
        return self.y + self.slope * (x - 0.5 * (self.x0 + self.x1))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.slope, 1.0)))

    @property
    def tangent(self) -> np.ndarray:
        v = np.array([1.0, self.slope])
        return v / np.linalg.norm(v)


# How far inside the vessel the surface is looked for, as fractions of the
# vessel's height. The rim and the base are both strong horizontal boundaries
# and neither can be the surface: none of these vessels is overflowing or
# empty.
SURFACE_BAND = (0.08, 0.80)
# How much more tinted the liquid has to be than what is above it, in Lab a/b
# units, for the boundary between them to count as a surface.
MIN_TINT_STEP = 2.0


def surface_by_tint(frame: np.ndarray, cont: Container, band: int = 9,
                    exclude_cols: np.ndarray | None = None
                    ) -> Waterline | None:
    """The liquid surface, as the row where the vessel's contents change colour.

    The surface is read from chroma alone - the Lab a/b plane, with lightness
    discarded. That is the whole point: a tinted liquid differs from the air
    above it in colour, whereas the boundaries that kept being mistaken for it
    differ only in brightness. The vessel's base, the bench line behind the
    glass, the edge between lit and shaded water, and the top of a submerged
    ice block are all steps in lightness; none of them is a step in colour of
    the liquid's own hue, so dropping L removes them rather than having to
    outrank them.

    Two things bound the answer. The search runs only over the middle of the
    vessel, since neither the rim nor the base can be a surface. And the row
    below the boundary has to be the more tinted of the two by a margin, so a
    vessel of untinted liquid reports nothing instead of returning the best of
    a set of meaningless steps.
    """
    h, w = frame.shape[:2]
    xa, xb = cont.interior()
    if xb - xa < 12:
        return None
    keep = np.ones(xb - xa, bool)
    if exclude_cols is not None:
        keep = ~exclude_cols[xa:xb]
        if keep.sum() < 12:
            return None

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, xa:xb, 1:][:, keep, :] - 128.0
    # One colour per row, taken as the median across the vessel so that a
    # floating block spanning part of the width cannot carry the row.
    rows = np.median(ab, axis=1)
    chroma_row = np.linalg.norm(rows, axis=1)

    # A vessel proposal may contain only the liquid, so its upper boundary
    # can itself be the surface. Search that boundary as well; the signed
    # chroma gain and multi-column support reject the bottom of the tank.
    y_lo = cont.y0 - band
    y_hi = cont.y1 - 2
    y_lo = max(y_lo, band + 1)
    y_hi = min(y_hi, h - band - 1)
    if y_hi - y_lo < 8:
        return None

    ys = np.arange(y_lo, y_hi)
    above = np.stack([rows[y - band:y - 1].mean(axis=0) for y in ys])
    below = np.stack([rows[y + 2:y + band + 1].mean(axis=0) for y in ys])
    # Signed: the liquid is the tinted side, so only a step that gains colour
    # going downwards can be a surface.
    gain = (np.linalg.norm(below - above, axis=1)
            * np.sign(np.linalg.norm(below, axis=1)
                      - np.linalg.norm(above, axis=1)))
    gain = cv2.blur(gain.astype(np.float32).reshape(-1, 1), (1, 5)).ravel()
    k = int(np.argmax(gain))
    if gain[k] < MIN_TINT_STEP:
        return None
    y_best = float(ys[k])

    # Per-column height around that row, which gives the tilt and the residual.
    col_ab = lab[:, xa:xb, 1:] - 128.0
    xs_, ys_ = [], []
    for j in np.flatnonzero(keep):
        prof = np.linalg.norm(col_ab[:, j, :], axis=1)
        lo, hi = int(y_best - band), int(y_best + band + 1)
        seg = np.diff(cv2.blur(prof[lo:hi + 1].astype(np.float32).reshape(-1, 1),
                               (1, 3)).ravel())
        if seg.size < 3 or seg.max() <= 1e-6:
            continue
        i = int(np.argmax(seg))
        xs_.append(float(xa + j))
        ys_.append(float(lo + i))
    if len(xs_) < 12:
        return Waterline(y=y_best, slope=0.0, x0=xa, x1=xb, rms_px=0.0,
                         step=float(gain[k]), profile=chroma_row)
    A = np.polyfit(np.asarray(xs_) - 0.5 * (xa + xb), ys_, 1)
    resid = np.asarray(ys_) - np.polyval(A, np.asarray(xs_) - 0.5 * (xa + xb))
    return Waterline(y=float(A[1]), slope=float(A[0]), x0=xa, x1=xb,
                     rms_px=float(np.sqrt(np.mean(resid ** 2))),
                     step=float(gain[k]), profile=chroma_row)


def find_ice(frame: np.ndarray, cont: Container,
             min_area_frac: float = 0.02) -> np.ndarray | None:
    """Segment the floating ice inside the vessel by its own lightness.

    Ice is the one thing in these vessels with an appearance of its own: it
    scatters light and comes out markedly lighter than the water around it. An
    Otsu split of the L* channel taken over the vessel interior alone therefore
    separates it, with no threshold carried between scenes.

    SAM2 is not used here even though it is used for the vessel. Prompting it
    inside the ice returns the whole glass: the block fills most of the vessel
    and shares its boundary, so a point prompt cannot tell them apart, whereas
    lightness can.
    """
    xa, xb = cont.interior(0.04)
    if xb - xa < 10:
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]
    roi = lab[cont.y0:cont.y1, xa:xb]
    if roi.size < 400:
        return None
    thr, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = np.zeros(lab.shape, np.uint8)
    m[cont.y0:cont.y1, xa:xb] = (roi > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area_frac * cont.mask.sum():
        return None
    return labels == k


MIN_LIQUID_CHROMA = 18.0
MIN_LIQUID_AREA_FRAC = 0.01


def liquid_mask(frame: np.ndarray, min_chroma: float = MIN_LIQUID_CHROMA
                ) -> np.ndarray | None:
    """Mask of a tinted liquid, taken as its own coloured region.

    Where the liquid carries a tint this is the whole measurement problem
    solved: the liquid is the one large chromatic region in an otherwise neutral
    scene of glass, bench and wall, so its top edge is the surface with no edge
    scoring, no search window and no way for a bench line behind the glass to
    be mistaken for it. Returns None when the scene has no such region, which
    is how a clear-water clip is told apart from a tinted one.
    """
    C = chroma(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = ((C >= min_chroma) & (hsv[..., 2] >= 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < MIN_LIQUID_AREA_FRAC * frame.shape[0] \
            * frame.shape[1]:
        return None
    return labels == k


def surface_from_mask(mask: np.ndarray, x0: int, x1: int,
                      exclude_cols: np.ndarray | None = None,
                      trim: float = 0.15) -> Waterline | None:
    """Fit a line to the top edge of a liquid mask over a column range.

    Columns where a floating body breaks the surface are excluded by the
    caller; the remaining per-column top edges are trimmed of their extremes
    before fitting so a stray column of foam or a bubble cannot tilt the line.
    """
    cols = []
    for x in range(max(0, x0), min(mask.shape[1], x1)):
        if exclude_cols is not None and exclude_cols[x]:
            continue
        on = np.flatnonzero(mask[:, x])
        if on.size:
            cols.append((float(x), float(on.min())))
    if len(cols) < 12:
        return None
    arr = np.asarray(cols, float)
    lo, hi = np.quantile(arr[:, 1], [trim, 1.0 - trim])
    keep = arr[(arr[:, 1] >= lo) & (arr[:, 1] <= hi)]
    if keep.shape[0] < 8:
        keep = arr
    mid = 0.5 * (x0 + x1)
    slope, intercept = np.polyfit(keep[:, 0] - mid, keep[:, 1], 1)
    rms = float(np.sqrt(np.mean((keep[:, 1]
                                 - (slope * (keep[:, 0] - mid)
                                    + intercept)) ** 2)))
    return Waterline(y=float(intercept), slope=float(slope), x0=int(x0),
                     x1=int(x1), rms_px=rms, step=float("nan"),
                     profile=np.zeros(mask.shape[0], np.float32))


def wall_columns(frame: np.ndarray, cont: Container) -> tuple[float, float]:
    """Sub-pixel x of the vessel's left and right walls in this frame.

    Used to compare the vessel between two moments without re-running SAM2:
    re-segmenting would fold the model's own frame-to-frame jitter into a metric
    that is supposed to report whether the scene moved. The walls are the two
    strongest vertical-gradient columns near the segmented bounding box.
    """
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prof = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))[cont.y0:cont.y1]
    col = cv2.blur(prof.mean(axis=0).reshape(1, -1), (5, 1))[0]
    pad = max(8, int(0.12 * cont.width))

    def peak(lo: int, hi: int) -> float:
        lo, hi = max(0, lo), min(col.size, hi)
        if hi - lo < 3:
            return float("nan")
        seg = col[lo:hi]
        k = int(np.argmax(seg))
        # Parabolic interpolation on the gradient peak for sub-pixel position.
        if 0 < k < seg.size - 1:
            a, b, c = seg[k - 1], seg[k], seg[k + 1]
            den = a - 2 * b + c
            k = k + (0.5 * (a - c) / den if abs(den) > 1e-6 else 0.0)
        return float(lo + k)

    return peak(cont.x0 - pad, cont.x0 + pad), peak(cont.x1 - pad, cont.x1 + pad)


def find_floating_block(frame: np.ndarray, cont: Container,
                        min_area_frac: float = 0.04, **kw
                        ) -> np.ndarray | None:
    """Segment the floating ice block with SAM2, prompted at the vessel centre.

    This is the one place a point prompt is reliable on these scenes: the block
    is opaque and sits in the middle of the glass, so a point there lands on it
    rather than on the water or the wall. The mask must stay clear of the
    glass's own width, otherwise it is the glass or the water body and not the
    block.
    """
    cx = (cont.x0 + cont.x1) // 2
    for fy in (0.45, 0.55, 0.35):
        y = int(cont.y0 + fy * cont.height)
        m = segment(frame, [(cx, y)], **kw)
        area = int(m.sum())
        if area < min_area_frac * cont.width * cont.height:
            continue
        xs = np.flatnonzero(m.any(axis=0))
        if xs.size == 0:
            continue
        # A block narrower than the glass leaves the columns the waterline is
        # read on; a mask as wide as the glass is the glass or the water.
        if xs.max() - xs.min() > 0.88 * cont.width:
            continue
        return m
    return None


def find_ice_by_texture(frame: np.ndarray, cont: Container, y_water: float,
                        win: int = 11, min_area_frac: float = 0.03
                        ) -> np.ndarray | None:
    """Segment a frosted ice body by its surface texture.

    Clear water and clear glass render smooth, while ice is frosted and carries
    fine structure at every scale, so the local standard deviation of L*
    separates them where brightness cannot: the wall behind the glass is just as
    bright as the ice but perfectly smooth. The body is required to straddle the
    waterline, which is what distinguishes the floating block from bubbles or
    reflections elsewhere in the vessel.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    mean = cv2.blur(lab, (win, win))
    sd = cv2.sqrt(cv2.blur((lab - mean) ** 2, (win, win)))

    # Search a band around the surface, wide enough to hold the whole block.
    y_lo = int(max(0, y_water - 0.9 * cont.height))
    y_hi = int(min(frame.shape[0], y_water + 1.1 * cont.height))
    xa, xb = cont.x0, cont.x1
    roi = sd[y_lo:y_hi, xa:xb]
    if roi.size < 400:
        return None
    thr, _ = cv2.threshold(np.clip(roi, 0, 255).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = np.zeros(lab.shape, np.uint8)
    m[y_lo:y_hi, xa:xb] = (roi > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    need = min_area_frac * cont.width * cont.height
    best = None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < need:
            continue
        top = stats[i, cv2.CC_STAT_TOP]
        bot = top + stats[i, cv2.CC_STAT_HEIGHT]
        if not (top <= y_water <= bot):
            continue
        if best is None or stats[i, cv2.CC_STAT_AREA] > stats[best,
                                                             cv2.CC_STAT_AREA]:
            best = i
    return None if best is None else (labels == best)


def columns_of(mask: np.ndarray, min_rows: int = 3) -> np.ndarray:
    """Columns where ``mask`` is present, as a boolean array over image width."""
    return mask.sum(axis=0) >= min_rows


def find_waterline(frame: np.ndarray, cont: Container, band: int = 6,
                   search: tuple[float, float] = (0.05, 0.55),
                   exclude_cols: np.ndarray | None = None,
                   margin: int = 25
                   ) -> Waterline | None:
    """Strongest horizontal colour step inside the container.

    Restricting to the container is what makes this unambiguous, so no ranking
    heuristics or line-type filters are needed beyond taking the maximum. The
    search window covers the container's upper half only. Every one of these
    other horizontal boundaries inside its bounding box; a liquid surface cannot
    coincide with either, since the vessel is neither empty nor overflowing in
    any of these tasks.

    Note that the liquid itself cannot be segmented directly: clear water in
    clear glass has no appearance of its own, and prompting SAM2 inside it
    returns the vessel base rather than the water column. What SAM2 contributes
    is the vessel, and that is enough to make the surface unique.
    """
    h, w = frame.shape[:2]
    xa, xb = cont.interior()
    if xb - xa < 12:
        return None
    keep = np.ones(xb - xa, bool)
    if exclude_cols is not None:
        # Columns occupied by a floating solid carry its own edges, which would
        # otherwise outscore the surface itself.
        keep = ~exclude_cols[xa:xb]
        if keep.sum() < 12:
            return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    roi = lab[:, xa:xb]
    below = cv2.blur(np.roll(roi, -band - 1, axis=0), (1, band))
    above = cv2.blur(np.roll(roi, band + 1, axis=0), (1, band))
    step = cv2.blur(np.linalg.norm(below - above, axis=2), (9, 1))
    # np.roll wraps, so rows within one band of either frame edge compare the
    # top of the image against its bottom and carry a large false step.
    dead = band + 2
    step[:dead] = 0.0
    step[-dead:] = 0.0

    y_lo = cont.y0 + int(search[0] * cont.height)
    y_hi = cont.y0 + int(search[1] * cont.height)
    prof = step[:, keep].mean(axis=1)
    if y_hi - y_lo < 8:
        return None

    # A liquid surface stops at the vessel walls. A bench edge or a skirting
    # line behind the glass crosses the vessel columns just as strongly but
    # carries on to the frame border, so each row is also scored outside the
    # vessel and rows that continue outside are demoted. Without this the
    # bench line behind a small beaker outscores the water it hides.
    out_cols = np.r_[np.arange(0, max(0, cont.x0 - margin)),
                     np.arange(min(w, cont.x1 + margin), w)]
    if out_cols.size >= 12:
        lab_all = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        b2 = cv2.blur(np.roll(lab_all, -band - 1, axis=0), (1, band))
        a2 = cv2.blur(np.roll(lab_all, band + 1, axis=0), (1, band))
        step_all = cv2.blur(np.linalg.norm(b2 - a2, axis=2), (9, 1))
        dead = band + 2
        step_all[:dead] = 0.0
        step_all[-dead:] = 0.0
        outside = step_all[:, out_cols].mean(axis=1)
        prof = np.maximum(prof - outside, 0.0)

    y_best = int(y_lo + np.argmax(prof[y_lo:y_hi]))

    # Sub-pixel height per column, which also gives the tilt and the residual.
    win = max(4, band)
    xs, ys = [], []
    for j in np.flatnonzero(keep):
        lo, hi = max(0, y_best - win), min(frame.shape[0], y_best + win + 1)
        col = step[lo:hi, j].astype(np.float64)
        col -= col.min()
        tot = col.sum()
        if tot <= 1e-6:
            continue
        ys.append(float((np.arange(lo, hi) * col).sum() / tot))
        xs.append(float(xa + j))
    if len(xs) < 12:
        return None
    xa_arr, ya_arr = np.asarray(xs), np.asarray(ys)
    mid = 0.5 * (xa + xb)
    slope, intercept = np.polyfit(xa_arr - mid, ya_arr, 1)
    rms = float(np.sqrt(np.mean((ya_arr - (slope * (xa_arr - mid)
                                           + intercept)) ** 2)))
    return Waterline(y=float(intercept), slope=float(slope), x0=xa, x1=xb,
                     rms_px=rms, step=float(prof[y_best]), profile=prof)
