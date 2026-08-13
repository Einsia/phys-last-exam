"""Trajectory fits over an established flight window.

Two products:

`fit_parabola`  the calibration-free parabolicity test. Constant horizontal velocity
                and constant vertical acceleration are each scale-free statements:
                rescaling pixels or frames rescales the fitted coefficients but not
                the normalised residual. Each axis is normalised by its OWN extent,
                which is what makes the two numbers signal-to-noise ratios. A single
                isotropic scale would be dominated by the horizontal extent -- on a
                flat 45-degree arc the diagonal is 4x the vertical extent, so a
                vertical error worth 13% of the apex height reads as 3%, and a
                straight-line ascent stops looking any different from a parabola. The
                denominators are floored at a few ball radii so a nearly-vertical
                throw cannot divide by a vanishing horizontal extent.

`launch_slope`  the initial direction dy/dx at launch, from a *local* fit near
                launch. It must be local: the global fit's own coefficients satisfy
                H/R = tan(theta)/4 as an algebraic identity, so testing that ratio
                against a globally-fitted angle would always return zero. Fitted
                locally it becomes a real test -- does the arc's overall geometry
                agree with the direction it actually left the ground in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ParabolaFit:
    vx: float           # px per frame
    ay: float           # px per frame^2, positive = downward in image coords
    vy0: float          # px per frame at t_launch, positive = upward
    rms_x_norm: float   # dimensionless: rms residual / horizontal extent
    rms_y_norm: float   # dimensionless: rms residual / vertical extent
    n_samples: int
    scale_x: float      # px
    scale_y: float      # px


def fit_parabola(t: np.ndarray, x: np.ndarray, y: np.ndarray, t_launch: float,
                 radius: float = 1.0) -> ParabolaFit | None:
    """x linear in t, y quadratic in t. Each axis normalised by its own extent."""
    if len(t) < 6:
        return None
    px = np.polyfit(t, x, 1)
    py = np.polyfit(t, y, 2)
    rx = x - np.polyval(px, t)
    ry = y - np.polyval(py, t)
    floor = 4.0 * max(radius, 1e-6)
    scale_x = max(float(np.ptp(x)), floor)
    scale_y = max(float(np.ptp(y)), floor)
    return ParabolaFit(
        vx=float(px[0]),
        ay=float(2.0 * py[0]),
        vy0=float(-(2.0 * py[0] * t_launch + py[1])),
        rms_x_norm=float(np.sqrt(np.mean(rx ** 2)) / scale_x),
        rms_y_norm=float(np.sqrt(np.mean(ry ** 2)) / scale_y),
        n_samples=int(len(t)),
        scale_x=scale_x,
        scale_y=scale_y,
    )


@dataclass
class HalfFits:
    """Fits done separately on the ascent and the descent.

    Ratios between the two halves are the sharpest calibration-invariant tests in the
    task, because they are first-order in the quantity being tested. `fit_parabola`'s
    normalised RMS is not: horizontal drag turns x(t) from a line into a gentle
    parabola, and a parabola sits close to its own best-fit line in RMS terms, so a
    21% loss of horizontal speed shows up as a ~1.4% residual and slips under any
    tolerance loose enough to survive tracker noise. The velocity ratio shows the same
    clip as 18% off. Both numerator and denominator are px/frame, so s and tau cancel.
    """

    vx_ascent: float
    vx_descent: float
    ay_ascent: float
    ay_descent: float
    n_ascent: int
    n_descent: int

    @property
    def vx_ratio(self) -> float:
        return (self.vx_descent / self.vx_ascent
                if abs(self.vx_ascent) > 1e-9 else float("nan"))

    @property
    def ay_ratio(self) -> float:
        return (self.ay_descent / self.ay_ascent
                if abs(self.ay_ascent) > 1e-9 else float("nan"))


def half_fits(t: np.ndarray, x: np.ndarray, y: np.ndarray, t_apex: float,
              min_pts: int = 6) -> HalfFits | None:
    """Independent (v_x, a_y) on each side of the apex.

    v_x tests conservation of horizontal momentum; a_y tests that gravity is the same
    going up and coming down, which N_up/N_down only tests in the integrated sense --
    a clip can get the total times right while the two accelerations differ.
    """
    asc = t <= t_apex
    des = t >= t_apex
    if asc.sum() < min_pts or des.sum() < min_pts:
        return None
    out = []
    for sel in (asc, des):
        out.append((float(np.polyfit(t[sel], x[sel], 1)[0]),
                    float(2.0 * np.polyfit(t[sel], y[sel], 2)[0])))
    return HalfFits(vx_ascent=out[0][0], vx_descent=out[1][0],
                    ay_ascent=out[0][1], ay_descent=out[1][1],
                    n_ascent=int(asc.sum()), n_descent=int(des.sum()))


def launch_slope(t: np.ndarray, x: np.ndarray, y: np.ndarray, t_launch: float,
                 n_up: float, min_pts: int = 5) -> tuple[float, int] | None:
    """dy/dx immediately after launch, as (slope, n_points_used).

    Quadratic in t when there is room, so the gravity term does not bias the
    derivative at t_launch; linear only as a fallback. The first sample after launch
    is skipped when affordable -- it is the most motion-blurred and the most likely
    to still be partly in contact.
    """
    window = max(float(min_pts), 0.45 * n_up)
    sel = np.flatnonzero((t >= t_launch - 0.5) & (t <= t_launch + window))
    if len(sel) > min_pts + 1:
        sel = sel[1:]
    if len(sel) < min_pts:
        sel = np.arange(min(min_pts, len(t)))
    if len(sel) < 3:
        return None

    ts, xs, ys = t[sel], x[sel], y[sel]
    deg = 2 if len(sel) >= 5 else 1
    cx = np.polyfit(ts, xs, deg)
    cy = np.polyfit(ts, ys, deg)
    dx = np.polyval(np.polyder(cx), t_launch)
    dy = np.polyval(np.polyder(cy), t_launch)
    if abs(dx) < 1e-9:
        return None
    return float(-dy / dx), int(len(sel))
