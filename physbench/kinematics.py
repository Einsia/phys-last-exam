"""Flight-window extraction and subframe geometry, in pixels and frames only.

Coordinates are image coordinates, so y grows downward. Height above a reference
level is (y_ref - y). Under gravity the image-space vertical position is

    y(t) = y_ref - v0 t + (g/2) t^2

so the quadratic coefficient of y is +g/2 and must be positive; a non-positive
value means the clip has no downward vertical acceleration at all.

Launch and landing are both recovered as crossings of the *same* level y_ref, which
is what makes H and R commensurable: the textbook H/R = tan(theta)/4 holds only
between two points at equal height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Flight:
    n_launch: int          # integer frame where rising starts
    n_land: int            # integer frame where falling ends
    n_apex: int
    t_launch: float        # subframe level crossing, ascending
    t_apex: float          # subframe vertex of the local quadratic
    t_land: float          # subframe level crossing, descending
    x_launch: float
    x_apex: float
    x_land: float
    y_ref: float
    y_apex: float
    inner_coverage: float
    used_rest_level: bool
    launch_extrap_frames: float
    land_extrap_frames: float
    flags: list[str] = field(default_factory=list)

    @property
    def land_extrapolated(self) -> bool:
        return self.land_extrap_frames > 0.0

    @property
    def H(self) -> float:
        return self.y_ref - self.y_apex

    @property
    def R(self) -> float:
        return abs(self.x_land - self.x_launch)

    @property
    def n_up(self) -> float:
        return self.t_apex - self.t_launch

    @property
    def n_down(self) -> float:
        return self.t_land - self.t_apex


def compensate(xy: np.ndarray, shift: np.ndarray | None) -> np.ndarray:
    """Remove estimated global camera translation from the track."""
    if shift is None:
        return xy.copy()
    return xy - shift


def _cross(t: np.ndarray, y: np.ndarray, level: float,
           direction: str) -> tuple[float, float]:
    """Time at which y crosses `level`. Returns (time, frames_extrapolated).

    Direction matters and is not cosmetic. The ascending slice is fed a few
    pre-launch rest samples, which sit *at* the reference level, so a forward scan
    would return the first of those rather than the moment the ball left it. The
    launch is the LAST upward crossing before the apex, so ascent scans backward;
    the landing is the FIRST crossing after the apex, so descent scans forward.

    Every sample passed in must be a MOVING one. Sub-pixel noise routinely leaves the
    descent a fraction of a pixel short of the launch level, so no pair brackets it and
    the answer has to come from extrapolation -- and extrapolating off stationary
    samples divides by a zero slope and returns nonsense. Restricted to moving samples
    the extrapolation is a sub-frame nudge along a real slope, and the distance it
    travelled is returned so a long extrapolation can be treated as weak evidence.
    """
    pairs = range(len(t) - 2, -1, -1) if direction == "ascending" else range(len(t) - 1)
    for i in pairs:
        a, b = y[i], y[i + 1]
        if a == b:
            continue
        if (a - level) * (b - level) <= 0:
            frac = (level - a) / (b - a)
            return float(t[i] + frac * (t[i + 1] - t[i])), 0.0

    k = min(3, len(t))
    ts, ys = (t[:k], y[:k]) if direction == "ascending" else (t[-k:], y[-k:])
    anchor = float(ts[0]) if direction == "ascending" else float(ts[-1])
    if len(ts) < 2:
        return anchor, float("inf")
    slope, intercept = np.polyfit(ts, ys, 1)
    if abs(slope) < 1e-9:
        return anchor, float("inf")
    hit = float((level - intercept) / slope)
    return hit, abs(hit - anchor)


def _eval_at(t_f: np.ndarray, arr: np.ndarray, t: float, lo: int, hi: int,
             k: int = 4) -> float:
    """Linear-fit evaluation at t from the k nearest samples WITHIN [lo, hi].

    The window is mandatory, not a convenience. x is linear in time only while the ball
    is in flight; the samples just outside are stationary, and letting them into the fit
    flattens the slope. Unwindowed, this biased x_land back by ~5 px and x_launch forward
    by ~4 px, shortening R by 1.3% at every angle -- a constant systematic error in M1
    that looked like a noise floor until it failed to shrink with the angle.
    """
    lo, hi = max(0, lo), min(len(t_f) - 1, hi)
    if hi <= lo:
        return float(arr[max(lo, 0)])
    idx = np.arange(lo, hi + 1)
    near = idx[np.argsort(np.abs(t_f[idx] - t))[:max(2, k)]]
    sel = np.sort(near)
    slope, intercept = np.polyfit(t_f[sel], arr[sel], 1)
    return float(slope * t + intercept)


def _moving_run(speed: np.ndarray, k_apex: int, thr: float) -> tuple[int, int]:
    """Contiguous run of moving samples containing k_apex, as (k0, k1) inclusive."""
    k0 = k_apex
    while k0 > 0 and speed[k0 - 1] > thr:
        k0 -= 1
    k1 = k_apex
    while k1 < len(speed) - 1 and speed[k1 + 1] > thr:
        k1 += 1
    return k0, k1


def _first_contact(y: np.ndarray, k_apex: int, k_end: int,
                   tol: float = 1.0) -> int:
    """Index of the lowest point of the first descent -- where free flight ends.

    The moving run does not stop at the ground: a real ball bounces or rolls, so the
    run continues well past contact. Everything after contact has to be cut, for two
    separate reasons. It would enter the parabola fit, where post-bounce samples are a
    different trajectory entirely; and it would be the basis for extrapolating the
    landing time, where upward-moving rebound samples give the slope the wrong sign and
    send the estimate off by frames.

    Contact is the first local maximum of y (lowest point, image coords). `tol` is how
    far y must come back up to count as a rebound rather than tracker noise.
    """
    k_low = k_apex
    for k in range(k_apex, k_end + 1):
        if y[k] > y[k_low]:
            k_low = k
        elif y[k] < y[k_low] - tol:
            break
    return k_low


def extract_flight(xy: np.ndarray, radius: float,
                   min_flight: int = 12) -> tuple[Flight | None, str]:
    """Isolate the free-flight interval: rise into an apex, then fall.

    The airborne window is the run of *moving* samples containing the highest point.
    Moving is judged on total speed, not vertical speed: a projectile's vertical
    velocity passes through zero exactly at the apex, so a vertical-only test cuts the
    window in half at the one frame it most needs to keep. Total speed stays at |v_x|
    through the apex, which separates flight from the resting phases cleanly and needs
    no assumption about when the launch happens.
    """
    found = np.isfinite(xy[:, 0])
    idx = np.flatnonzero(found)
    if len(idx) < 8:
        return None, "too_few_detections"

    t_f = idx.astype(float)
    x_f, y_f = xy[idx, 0], xy[idx, 1]
    vx = np.gradient(x_f, t_f)
    vy = np.gradient(y_f, t_f)
    speed = np.hypot(vx, vy)
    k_apex = int(np.argmin(y_f))

    # Relative to the clip's own peak speed, with a floor tied to the ball size so a
    # genuinely slow-motion clip is not read as one long rest.
    peak = float(np.nanmax(speed))
    if not np.isfinite(peak) or peak <= 1e-6:
        return None, "no_motion_observed"
    thr = max(0.18 * peak, 0.015 * radius)
    k0, k_run_end = _moving_run(speed, k_apex, thr)

    if k_apex <= k0:
        return None, "no_ascent_observed"
    if k_run_end <= k_apex:
        return None, "no_descent_observed"

    # Free flight ends at first contact, not at the end of the moving run.
    k1 = _first_contact(y_f, k_apex, k_run_end, tol=max(1.0, 0.06 * radius))
    if k1 <= k_apex:
        return None, "no_descent_observed"

    n_launch, n_land, n_apex = int(idx[k0]), int(idx[k1]), int(idx[k_apex])
    if n_land - n_launch < min_flight:
        return None, f"flight_too_short({n_land - n_launch})"

    inner = float(found[n_launch:n_land + 1].mean())

    # Reference level: the resting height before launch when it is observed. Two or
    # more pre-launch samples give a median that beats using the first moving frame,
    # whose height is already above the ground by part of one frame of travel.
    pre = np.flatnonzero(found[:n_launch])
    if len(pre) >= 2:
        y_ref = float(np.median(xy[pre[-min(len(pre), 8):], 1]))
        used_rest = True
    else:
        y_ref = float(y_f[k0])
        used_rest = False

    # Subframe apex from a quadratic vertex; the window is a fraction of the flight
    # so it stays local while still averaging several samples.
    w = max(3, int(round(0.18 * (k1 - k0))))
    lo, hi = max(k0, k_apex - w), min(k1 + 1, k_apex + w + 1)
    if hi - lo >= 3:
        a, b, c = np.polyfit(t_f[lo:hi], y_f[lo:hi], 2)
        if a > 1e-9:
            t_apex = float(-b / (2 * a))
            y_apex = float(a * t_apex ** 2 + b * t_apex + c)
        else:
            t_apex, y_apex = float(t_f[k_apex]), float(y_f[k_apex])
    else:
        t_apex, y_apex = float(t_f[k_apex]), float(y_f[k_apex])
    t_apex = float(np.clip(t_apex, n_launch, n_land))

    # Both crossings see MOVING samples only, and for the same reason at each end.
    # y_ref is the median of the resting samples, so those samples straddle it by
    # fractions of a pixel and no clean bracket exists between rest and flight;
    # whichever side noise puts them on, extrapolating off a stationary slope is
    # meaningless. The rest phase's only job is to define y_ref. Ascent runs k0..apex,
    # descent apex..contact.
    t_launch, launch_ex = _cross(t_f[k0:k_apex + 1], y_f[k0:k_apex + 1],
                                 y_ref, "ascending")
    t_land, land_ex = _cross(t_f[k_apex:k1 + 1], y_f[k_apex:k1 + 1], y_ref, "descending")

    if not (t_launch < t_apex < t_land):
        return None, "degenerate_flight_ordering"

    flight = Flight(
        n_launch=n_launch, n_land=n_land, n_apex=n_apex,
        t_launch=t_launch, t_apex=t_apex, t_land=t_land,
        x_launch=_eval_at(t_f, x_f, t_launch, k0, k1),
        x_apex=_eval_at(t_f, x_f, t_apex, k0, k1),
        x_land=_eval_at(t_f, x_f, t_land, k0, k1),
        y_ref=y_ref, y_apex=y_apex, inner_coverage=inner,
        used_rest_level=used_rest,
        launch_extrap_frames=launch_ex, land_extrap_frames=land_ex,
    )
    if flight.H <= 0.5 * radius:
        return None, "apex_not_above_reference"
    if flight.R <= radius:
        return None, "range_below_ball_radius"
    # A long extrapolation means the descent never came near the launch level, so H and
    # R are not being measured between equal heights and the textbook ratio does not
    # apply to them. That is a measurement failure, not a physics verdict.
    if land_ex > 2.0:
        return None, f"landing_level_not_reached({land_ex:.1f}f)"
    return flight, "ok"
