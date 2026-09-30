"""Shared pendulum measurement for P8a and P8b.

Both tasks film two pendulums side by side and compare their periods, so the
extraction is identical and only the metric definition differs.

The angle signal is what gets measured, not the raw bob position: with the pivot
recovered from the bob's own arc, theta(t) = atan2(x - x_pivot, y - y_pivot) is
the quantity whose period is asked about, and it is immune to the bob being
tracked a pixel off centre.

Period comes from mean-crossings of theta(t) rather than from peak picking. A
sinusoid is steepest where it crosses the mean and flattest at its peaks, so the
crossing time is the best-conditioned feature in the signal; it is also
unaffected by the amplitude decay that flattens real peaks further.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fitting import period_from_crossings, zero_crossings
from .track import ColorTarget, Track, find_chroma_markers, track_target
from .video import Clip

MIN_CROSSINGS = 4
# A bob released from even the smallest angle these tasks use sweeps tens of
# pixels; centroid noise is a few tenths of a pixel. Anything below this never
# left its starting position, which is a physical outcome rather than a tracking
# problem, and a pendulum that does not swing has no period to measure.
MIN_SWING_PX = 4.0


@dataclass
class Bob:
    track: Track
    pivot: np.ndarray          # (x, y) of the suspension point
    radius: float              # string length in pixels
    theta: np.ndarray          # radians from vertical, NaN where untracked
    period: float              # frames
    period_samples: np.ndarray  # per-half-cycle period estimates
    sine_rms: float            # residual of a fitted sinusoid, in radians
    amplitude: float           # radians
    path_extent: float         # pixel span of the traced arc
    static: bool               # never left its starting position

    @property
    def n_crossings(self) -> int:
        return int(np.isfinite(self.theta).sum() and len(self.period_samples) + 1)


def _fit_pivot(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares circle through the bob path; its centre is the pivot.

    A pendulum bob moves on a circular arc about its suspension point, so the
    pivot needs no separate detection: it is the centre of the circle the bob
    itself traces. Solved in the linear form x^2+y^2 = 2ax + 2by + c.
    """
    good = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[good], y[good]
    A = np.stack([2 * xs, 2 * ys, np.ones_like(xs)], axis=1)
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 1e-6)))
    return np.array([cx, cy], float), r


def _sine_rms(t: np.ndarray, th: np.ndarray, period: float) -> float:
    """Residual of a fixed-period sinusoid plus offset fitted to theta(t).

    Linear in the unknowns once the period is known, so there is no optimiser
    and no starting guess: theta ~ A cos(wt) + B sin(wt) + C.
    """
    good = np.isfinite(th)
    if good.sum() < 8 or not np.isfinite(period) or period <= 0:
        return float("nan")
    w = 2 * np.pi / period
    tt, yy = t[good], th[good]
    M = np.stack([np.cos(w * tt), np.sin(w * tt), np.ones_like(tt)], axis=1)
    sol, *_ = np.linalg.lstsq(M, yy, rcond=None)
    return float(np.sqrt(np.mean((yy - M @ sol) ** 2)))


def measure(clip: Clip, targets: list[ColorTarget]) -> list[Bob]:
    """Track each bob and reduce it to a period and an angle signal."""
    bobs: list[Bob] = []
    t = np.arange(clip.n, dtype=float)
    nan = float("nan")
    for tg in targets:
        tr = track_target(clip, tg)
        g = np.isfinite(tr.x) & np.isfinite(tr.y)
        extent = (float(np.hypot(tr.x[g].max() - tr.x[g].min(),
                                 tr.y[g].max() - tr.y[g].min()))
                  if g.any() else 0.0)
        if extent < MIN_SWING_PX:
            # Fitting a circle to a stationary point puts the pivot anywhere,
            # so no angle, period or harmonicity is defined for this bob.
            bobs.append(Bob(track=tr, pivot=np.array([nan, nan]), radius=nan,
                            theta=np.full(clip.n, nan), period=nan,
                            period_samples=np.asarray([], float),
                            sine_rms=nan, amplitude=nan,
                            path_extent=extent, static=True))
            continue
        pivot, radius = _fit_pivot(tr.x, tr.y)
        # Angle from vertical, positive to the right of the pivot.
        theta = np.arctan2(tr.x - pivot[0], tr.y - pivot[1])
        period, samples = period_from_crossings(t, theta)
        amp = float(np.nanmax(np.abs(theta - np.nanmean(theta)))) \
            if np.isfinite(theta).any() else nan
        bobs.append(Bob(track=tr, pivot=pivot, radius=radius, theta=theta,
                        period=period, period_samples=samples,
                        sine_rms=_sine_rms(t, theta, period), amplitude=amp,
                        path_extent=extent, static=False))
    return bobs


def find_bobs(clip: Clip) -> list[ColorTarget]:
    """The two bobs, ordered left to right, seeded from the first frame."""
    tgs = find_chroma_markers(clip[0], k=2, chroma_min=28.0, min_area=60,
                              max_area_frac=0.05)
    return sorted(tgs, key=lambda t: t.x)


def crossings_ok(bob: Bob, t: np.ndarray) -> bool:
    return zero_crossings(t, bob.theta).size >= MIN_CROSSINGS
