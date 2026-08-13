"""Ball tracking. Five backends, three of them learned.

Default pair is `sam2` + `cotracker`, which is what proposal.md 3.3 asks for
directly ("SAM2/CoTracker disagreement" as the QC signal). Both are seeded with
nothing but the ball centre and radius the first frame's spec already records, so
there is no per-clip tuning anywhere on the default path:

  sam2       one positive click on frame 0, then video propagation. Position and
             apparent size come from the mask; presence comes from the model's own
             object-score logit.
  cotracker  a ring of points on the disc, tracked jointly; presence comes from the
             model's visibility head. A background grid rides along in the same call
             and yields camera translation as a by-product.
  sam3       Meta SAM3 video propagation from a text-and-point prompt. The worker
             converts its per-frame binary mask into a centroid, equivalent-disc
             radius and presence score. It can be paired with CoTracker as
             `backends: [sam3, cotracker]`.

The classic pair is kept and still selectable:

  color      Lab distance to the seeded ball colour, escalating thresholds.
  bgsub      deviation from a temporal-median background, MAD-scaled thresholds.

Both are hand-tuned to a flat scene with one high-chroma ball, and that tuning is the
reason to prefer the learned pair on real video: a VDM that recolours the ball, adds a
shadow, or drifts the background walks out of the regime those thresholds were set for,
and there is no seed we can hand them that puts it back.

Per-frame precision and residual precision are two different questions here, and only
the second one matters. Centroid scatter against exact synthetic truth, constant bias
removed because bias cancels in every ratio invariant (scripts/tracker_precision.py,
theta=45, r=16.9 px):

                clean render          motion blur (3 substeps)
                sd_x     sd_y         sd_x     sd_y
  sam2          0.138    0.067        0.408    1.664
  cotracker     0.267    0.196        0.421    1.469
  color         0.043    0.040        0.374    1.641
  bgsub         0.034    0.073        0.359    1.607

The classic pair is 3-6x tighter on the clean render; under blur all four land within
15% of each other. But neither ordering survives into the residuals, because H, R and
the per-half velocities are fits over ~60 frames and random scatter averages down.
Worst noise floor over the exact-physics renders (validate_metrics.py):

                        classic   neural
  M1_HR_nominal          0.0008   0.0013
  M2_time_symmetry       0.0041   0.0026
  M4_parabolicity        0.0014   0.0017
  M5_vx_conservation     0.0013   0.0007
  worst primary          0.0041   0.0026

So the learned pair costs nothing at the residual level -- it is marginally better on
the primaries -- and the per-frame scatter that looked decisive is averaged away. What it
does cost is a GPU, and less wall-clock than one might expect: 15.4 s/clip against 9.5 s
on a 124-frame clip (the classic pair is not cheap either -- both backends scan every
frame in Python). `classic` therefore stays the right choice when there is no GPU or when
a run must be reproducible on CPU, rather than for speed. validate_metrics.py reports the
floor for whichever pair ran, so the tolerances can be checked against the tracker in use.

What does not average away is error that correlates with speed, since that is
asymmetric between ascent and descent and so lands directly on M2 and M5. Both pairs
sit at r ~ 0.1-0.2 there, which is why tracker_precision.py reports it alongside sd.

Every backend detects (or propagates) independently per frame and then runs the same
model-agnostic temporal outlier filter. A causal greedy tracker can lock onto the
wrong blob and never recover; detect-then-clean cannot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .first_frame import BallSeed

# Ordered by preference: the first present backend leads unless coverage says
# otherwise. Kept here rather than in the task module so a second task reusing the
# tracker inherits the same default.
DEFAULT_BACKENDS: tuple[str, str] = ("sam2", "cotracker")
CLASSIC_BACKENDS: tuple[str, str] = ("color", "bgsub")
NEURAL_BACKENDS: frozenset[str] = frozenset({"sam2", "sam3", "cotracker"})


@dataclass
class Track:
    xy: np.ndarray       # (n, 2) float, NaN where not found
    radius: np.ndarray   # (n,) float, NaN where not found
    score: np.ndarray    # (n,) float detection score, NaN where not found
    backend: str

    @property
    def found(self) -> np.ndarray:
        return np.isfinite(self.xy[:, 0])

    @property
    def coverage(self) -> float:
        return float(self.found.mean())

    def masked_copy(self) -> "Track":
        return Track(self.xy.copy(), self.radius.copy(), self.score.copy(), self.backend)


def _components(hard: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
    return n, labels, stats


def _best_blob(weight: np.ndarray, expected_area: float,
               min_area: float) -> tuple[np.ndarray, float, float] | None:
    """Pick the component that best matches a disc of the expected area.

    Returns (centroid_xy, soft_radius, score) with score lower-is-better. The
    centroid is weighted by soft membership, which recovers subpixel position from
    the antialiased rim.
    """
    hard = (weight > 0.15).astype(np.uint8)
    if hard.sum() < min_area:
        return None
    n, labels, stats = _components(hard)
    best = None
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        bw, bh = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
        y0, x0 = int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_LEFT])
        sub = labels[y0:y0 + int(bh), x0:x0 + int(bw)] == i
        w = weight[y0:y0 + int(bh), x0:x0 + int(bw)] * sub
        soft = float(w.sum())
        if soft <= 0:
            continue
        ys, xs = np.nonzero(sub)
        wv = w[ys, xs]
        cx = float((xs * wv).sum() / wv.sum()) + x0
        cy = float((ys * wv).sum() / wv.sum()) + y0
        fill = area / max(bw * bh, 1.0)
        # Area dominates; shape is a weak cue because motion blur legitimately
        # stretches a fast ball into a streak.
        score = abs(np.log(max(soft, 1e-6) / expected_area)) + 0.6 * abs(fill - 0.785)
        if best is None or score < best[2]:
            best = (np.array([cx, cy]), float(np.sqrt(soft / np.pi)), score)
    return best


def _clean_temporal(track: Track, reject_px: float, iters: int = 2) -> Track:
    """Drop points a local low-order fit cannot explain.

    Deliberately local: the clip contains a rest phase and a flight phase, so no
    single global model applies, and using one would delete the very frames that
    decide the flight window.
    """
    out = track.masked_copy()
    n = len(out.xy)
    for _ in range(iters):
        found = np.isfinite(out.xy[:, 0])
        if found.sum() < 6:
            break
        drop = []
        idx = np.arange(n)
        for i in idx[found]:
            lo, hi = max(0, i - 3), min(n, i + 4)
            nb = [j for j in range(lo, hi) if j != i and found[j]]
            if len(nb) < 4:
                continue
            deg = 2 if len(nb) >= 5 else 1
            t = np.asarray(nb, float)
            pred = np.array([
                np.polyval(np.polyfit(t, out.xy[nb, k], deg), float(i)) for k in (0, 1)
            ])
            if np.linalg.norm(out.xy[i] - pred) > reject_px:
                drop.append(i)
        if not drop:
            break
        out.xy[drop] = np.nan
        out.radius[drop] = np.nan
        out.score[drop] = np.nan
    return out


def track_color(frames: np.ndarray, seed: BallSeed,
                reject_px: float | None = None) -> Track:
    """Classic: Lab distance to the seeded ball colour, adaptive threshold."""
    n = len(frames)
    xy = np.full((n, 2), np.nan)
    rad = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    seed_lab = cv2.cvtColor(
        np.asarray(seed.rgb, np.uint8).reshape(1, 1, 3), cv2.COLOR_RGB2LAB
    ).astype(np.float32).reshape(3)
    expected_area = float(np.pi * seed.radius ** 2)
    min_area = max(6.0, 0.06 * expected_area)

    for i in range(n):
        lab = cv2.cvtColor(frames[i], cv2.COLOR_RGB2LAB).astype(np.float32)
        dist = np.linalg.norm(lab - seed_lab, axis=2)
        # Escalating thresholds: the tight one is right when the ball keeps its
        # colour, the loose ones recover it after the VDM recolours or blurs it.
        for tau in (28.0, 40.0, 55.0, 75.0, 100.0):
            weight = np.clip(1.0 - dist / tau, 0.0, 1.0).astype(np.float32)
            hit = _best_blob(weight, expected_area, min_area)
            if hit is not None and hit[2] < 1.2:
                break
        if hit is not None:
            xy[i], rad[i], sc[i] = hit
    return _clean_temporal(Track(xy, rad, sc, "color"),
                           reject_px=reject_px or max(3.0 * seed.radius, 8.0))


def track_bgsub(frames: np.ndarray, seed: BallSeed, sample: int = 40,
                reject_px: float | None = None) -> Track:
    """Classic: deviation from a temporal-median background. Colour-agnostic."""
    n = len(frames)
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.float32)
    take = np.linspace(0, n - 1, min(n, sample)).astype(int)
    bg = np.median(gray[take], axis=0)

    xy = np.full((n, 2), np.nan)
    rad = np.full(n, np.nan)
    sc = np.full(n, np.nan)
    expected_area = float(np.pi * seed.radius ** 2)
    min_area = max(6.0, 0.06 * expected_area)

    for i in range(n):
        diff = np.abs(gray[i] - bg)
        # MAD-based floor so a globally brighter frame does not light up everywhere.
        mad = float(np.median(np.abs(diff - np.median(diff)))) + 1e-6
        for k in (10.0, 6.0, 4.0):
            tau = max(12.0, k * 1.4826 * mad)
            weight = np.clip(diff / tau, 0.0, 1.0).astype(np.float32)
            hit = _best_blob(weight, expected_area, min_area)
            if hit is not None and hit[2] < 1.2:
                break
        if hit is not None:
            xy[i], rad[i], sc[i] = hit
    return _clean_temporal(Track(xy, rad, sc, "bgsub"),
                           reject_px=reject_px or max(3.0 * seed.radius, 8.0))


@dataclass
class TrackSet:
    """The pair of tracks for one clip, plus whatever the backends gave us for free.

    `camera_shift` is present only when a backend estimated it directly (CoTracker's
    background grid does). When it is None the caller falls back to phase correlation,
    so the camera-drift gate behaves the same either way.
    """

    tracks: dict[str, Track]
    camera_shift: np.ndarray | None = None
    seconds: float = 0.0   # wall-clock for every backend that ran, neural or classic

    @property
    def pair(self) -> tuple[Track, Track]:
        a, b = list(self.tracks.values())[:2]
        return a, b

    @property
    def primary(self) -> Track:
        """Highest-coverage track; max() keeps the first on a tie, i.e. the preferred one."""
        return max(self.tracks.values(), key=lambda t: t.coverage)


def run_tracks(frames: np.ndarray, seed: BallSeed,
               backends: tuple[str, ...] = DEFAULT_BACKENDS,
               reject_px: float | None = None,
               backend_cfg=None) -> TrackSet:
    """Run a pair of backends and return both tracks under one interface.

    Neural backends are batched into a single subprocess call so the two model loads
    are paid once per clip rather than once per backend.
    """
    if len(backends) < 2:
        raise ValueError(f"need two backends for the disagreement gate, got {backends}")
    reject = reject_px if reject_px is not None else max(3.0 * seed.radius, 8.0)
    out: dict[str, Track] = {}
    shift = None
    secs = 0.0

    neural = tuple(b for b in backends if b in NEURAL_BACKENDS)
    if neural:
        from .neural_track import run_worker

        nt = run_worker(frames, seed, neural, cfg=backend_cfg)
        secs += nt.seconds
        shift = nt.camera_shift
        for b in neural:
            out[b] = _clean_temporal(
                Track(nt.xy[b], nt.radius[b], nt.score[b], b), reject_px=reject)

    for b in backends:
        if b in out:
            continue
        t0 = time.time()
        if b == "color":
            out[b] = track_color(frames, seed, reject_px=reject)
        elif b == "bgsub":
            out[b] = track_bgsub(frames, seed, reject_px=reject)
        else:
            raise ValueError(f"unknown tracking backend {b!r}; "
                             f"have {sorted(NEURAL_BACKENDS | {'color', 'bgsub'})}")
        secs += time.time() - t0

    # Preserve the requested order, so `primary` ties break toward the first asked for.
    return TrackSet(tracks={b: out[b] for b in backends}, camera_shift=shift,
                    seconds=secs)


def estimate_camera_shift(frames: np.ndarray, track: Track,
                          seed: BallSeed) -> np.ndarray:
    """Cumulative global translation per frame, ball region excluded.

    A panning camera adds a spurious drift to the track that would corrupt every
    kinematic fit, so it is both compensated and reported. Phase correlation gives
    subpixel translation; rotation and zoom are out of scope and instead surface as
    a large residual in the QC numbers.
    """
    n = len(frames)
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    h, w = gray[0].shape
    pad = int(3.0 * seed.radius) + 6
    filled = []
    for i in range(n):
        g = gray[i].copy()
        if np.isfinite(track.xy[i, 0]):
            cx, cy = track.xy[i]
            x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad)
            y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad)
            if x1 > x0 and y1 > y0:
                g[y0:y1, x0:x1] = float(np.median(g))
        filled.append(g)

    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift = np.zeros((n, 2))
    for i in range(1, n):
        (dx, dy), _ = cv2.phaseCorrelate(filled[i - 1], filled[i], win)
        shift[i] = shift[i - 1] + np.array([dx, dy])
    return shift


def track_disagreement(a: Track, b: Track, radius: float) -> float:
    """Median A-vs-B distance in ball radii over frames where both fired."""
    both = a.found & b.found
    if both.sum() < 5:
        return float("inf")
    d = np.linalg.norm(a.xy[both] - b.xy[both], axis=1)
    return float(np.median(d) / max(radius, 1e-6))
