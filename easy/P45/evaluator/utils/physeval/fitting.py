"""Fits and signal helpers shared by the tasks.

Each helper is the single chosen estimator for its job, so a task never has to
pick between two answers for the same quantity.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PolyFit:
    coef: np.ndarray      # highest power first, as returned by polyfit
    rms: float            # residual RMS in the units of y
    n: int

    def __call__(self, t: np.ndarray | float) -> np.ndarray | float:
        return np.polyval(self.coef, t)


def polyfit(t: np.ndarray, y: np.ndarray, deg: int) -> PolyFit:
    coef = np.polyfit(t, y, deg)
    rms = float(np.sqrt(np.mean((y - np.polyval(coef, t)) ** 2)))
    return PolyFit(coef=coef, rms=rms, n=len(t))


def cv(values: np.ndarray) -> float:
    """Coefficient of variation, std over |mean|.

    Used for every benchmark metric written as CV(x). Sample std (ddof=1) so a
    two-element series is not flattered by the population formula.
    """
    v = np.asarray([x for x in np.ravel(values) if np.isfinite(x)], float)
    if v.size < 2:
        return float("nan")
    m = float(np.mean(v))
    if abs(m) < 1e-12:
        return float("nan")
    return float(np.std(v, ddof=1) / abs(m))


def zero_crossings(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Times where ``y`` crosses its own mean, linearly interpolated.

    Preferred over peak picking for oscillation periods: a sinusoid is steepest
    at the crossing and flattest at the peak, so the crossing time is the
    best-conditioned feature in the signal, and its location is unaffected by
    amplitude decay.
    """
    yc = np.asarray(y, float) - float(np.nanmean(y))
    tt = np.asarray(t, float)
    good = np.isfinite(yc)
    yc, tt = yc[good], tt[good]
    out = []
    for i in range(len(yc) - 1):
        a, b = yc[i], yc[i + 1]
        if a == 0.0:
            out.append(tt[i])
        elif a * b < 0:
            out.append(tt[i] + (tt[i + 1] - tt[i]) * a / (a - b))
    return np.asarray(out, float)


def period_from_crossings(t: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Oscillation period and the per-half-cycle period estimates.

    Consecutive mean-crossings are half a period apart, so each gap times two is
    one period estimate; their spread is what the CV(T) diagnostics use.
    """
    zc = zero_crossings(t, y)
    if zc.size < 3:
        return float("nan"), np.asarray([], float)
    per = 2.0 * np.diff(zc)
    return float(np.median(per)), per


def theil_sen(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Median-of-pairwise-slopes line fit; returns (slope, intercept).

    Chosen over least squares where the series has occasional bad frames, since
    it needs no outlier-rejection pass of its own.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 2:
        return float("nan"), float("nan")
    slopes = []
    for i in range(t.size - 1):
        dt = t[i + 1:] - t[i]
        ok = np.abs(dt) > 1e-9
        slopes.append((y[i + 1:][ok] - y[i]) / dt[ok])
    s = float(np.median(np.concatenate(slopes)))
    return s, float(np.median(y - s * t))


def moving_phase(pos: np.ndarray, smooth: int = 5,
                 rel_thresh: float = 0.15) -> tuple[int, int]:
    """Indices [start, stop) over which ``pos`` is actually in motion.

    Several metrics are defined only on the moving phase - a free fall that has
    already landed, or a block that has not been released yet, is outside the
    measured interval by the benchmark's own wording. The phase is bounded by
    where the smoothed speed first and last exceeds ``rel_thresh`` of its peak,
    which needs no absolute pixel threshold and so carries over between scenes.
    """
    p = np.asarray(pos, float)
    if p.size < 4:
        return 0, p.size
    v = np.abs(np.diff(p))
    if smooth > 1 and v.size >= smooth:
        kern = np.ones(smooth) / smooth
        v = np.convolve(v, kern, mode="same")
    peak = float(np.nanmax(v))
    if not np.isfinite(peak) or peak <= 0:
        return 0, p.size
    live = np.nonzero(v > rel_thresh * peak)[0]
    if live.size == 0:
        return 0, p.size
    # +2 so the stop index includes the final moving sample's endpoint.
    return int(live[0]), int(min(p.size, live[-1] + 2))


def longest_finite_run(v: np.ndarray) -> tuple[int, int]:
    """Longest span of consecutive finite samples in ``v``, as [start, stop)."""
    ok = np.isfinite(np.asarray(v, float))
    best = (0, 0)
    i = 0
    while i < ok.size:
        if ok[i]:
            j = i
            while j < ok.size and ok[j]:
                j += 1
            if j - i > best[1] - best[0]:
                best = (i, j)
            i = j
        else:
            i += 1
    return best
