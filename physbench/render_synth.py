"""Synthetic projectile clips with known ground truth.

Two jobs. First, an exact-physics render establishes the measurement noise floor:
whatever residual the chain reports on a clip that is correct by construction is
tracker and fitting error, not model error, and the tolerances have to sit well
above it. Second, deliberately-wrong renders check that each invariant fires on the
failure it is supposed to catch and stays quiet on the others -- without that, a
zero residual on a real clip proves nothing.

Parameterisation. Given range R (px), angle theta, and flight length N frames:

    v_x  = R / N                       constant
    v_y0 = v_x * tan(theta)            image-up positive
    g    = 2 * v_y0 / N                so the ball returns to y_ref exactly at N

which reproduces H/R = tan(theta)/4 and N_up/N_down = 1 by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .first_frame import BALL_RGB, BallSeed, draw_ball, render_scene


@dataclass
class Traj:
    x: np.ndarray
    y: np.ndarray
    launch: int
    land: int
    label: str
    truth: dict


def _pad(x: np.ndarray, y: np.ndarray, pre: int, post: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold the ball at rest before launch and after landing.

    The rest frames are not decoration: the pre-launch samples are what give the
    reference level y_ref its own estimate, instead of borrowing the first moving
    frame's height.
    """
    return (np.concatenate([np.full(pre, x[0]), x, np.full(post, x[-1])]),
            np.concatenate([np.full(pre, y[0]), y, np.full(post, y[-1])]))


def ideal(R: float, theta_deg: float, n_flight: int, x0: float, y_ref: float,
          pre: int = 10, post: int = 14, mode: str = "gt") -> Traj:
    """Exact projectile, or one of the ablations that breaks exactly one invariant."""
    th = np.radians(theta_deg)
    vx = R / n_flight
    vy0 = vx * np.tan(th)
    g = 2.0 * vy0 / n_flight
    H = vy0 ** 2 / (2.0 * g)
    t = np.arange(n_flight + 1, dtype=float)
    truth = {"R": R, "H": H, "H_over_R": H / R, "theta_deg": theta_deg,
             "n_up": n_flight / 2.0, "n_down": n_flight / 2.0, "vx": vx, "g": g}

    if mode == "gt":
        x = x0 + vx * t
        y = y_ref - vy0 * t + 0.5 * g * t ** 2

    elif mode == "no_gravity":
        # Straight up, straight down at constant speed: a triangle, not a parabola.
        # Same apex time, so N_up/N_down stays 1, but H/R becomes tan(theta)/2.
        half = n_flight / 2.0
        y = y_ref - np.where(t <= half, vy0 * t, vy0 * (n_flight - t))
        x = x0 + vx * t
        truth["H_over_R"] = np.tan(th) / 2.0

    elif mode == "time_warp":
        # Same geometric arc, traversed with the ascent stretched and the descent
        # compressed. Breaks the timing while leaving the shape alone.
        u = t / n_flight
        w = np.where(u <= 0.65, u / 0.65 * 0.5, 0.5 + (u - 0.65) / 0.35 * 0.5)
        tau = w * n_flight
        x = x0 + vx * tau
        y = y_ref - vy0 * tau + 0.5 * g * tau ** 2
        truth["n_up"], truth["n_down"] = 0.65 * n_flight, 0.35 * n_flight

    elif mode == "wrong_angle":
        # Physically perfect, but launched at 30 deg when the prompt said theta.
        th2 = np.radians(30.0)
        vy2 = vx * np.tan(th2)
        g2 = 2.0 * vy2 / n_flight
        x = x0 + vx * t
        y = y_ref - vy2 * t + 0.5 * g2 * t ** 2
        truth.update({"theta_deg": 30.0, "H_over_R": np.tan(th2) / 4.0,
                      "H": vy2 ** 2 / (2 * g2)})

    elif mode == "const_speed":
        # Correct parabolic *path*, traversed at uniform arc-length speed. Geometry
        # is perfect and both symmetries hold; only the parabolicity-in-time test
        # can see that the dynamics are wrong. This is the shape of failure the
        # proposal is most worried about.
        dense = np.linspace(0, n_flight, 4000)
        xd = x0 + vx * dense
        yd = y_ref - vy0 * dense + 0.5 * g * dense ** 2
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
        want = np.linspace(0, s[-1], n_flight + 1)
        x = np.interp(want, s, xd)
        y = np.interp(want, s, yd)

    else:
        raise ValueError(f"unknown mode {mode}")

    x, y = _pad(x, y, pre, post)
    return Traj(x=x, y=y, launch=pre, land=pre + n_flight, label=mode, truth=truth)


def render(traj: Traj, width: int, height: int, radius: float, ground_frac: float,
           blur_substeps: int = 1, noise: float = 1.4, seed: int = 0) -> np.ndarray:
    """Rasterise a trajectory over the flat side-view scene.

    `blur_substeps` > 1 averages sub-positions inside one frame's exposure, which is
    what a real capture (and a video model imitating one) does to a fast ball. The
    exact-physics render leaves it at 1 so the noise floor is not contaminated by
    blur; a separate blurred variant checks the tracker survives it.
    """
    base, _ = render_scene(width, height, ground_frac=ground_frac, seed=seed)
    rng = np.random.default_rng(seed + 1)
    frames = np.empty((len(traj.x), height, width, 3), np.uint8)

    for i in range(len(traj.x)):
        if blur_substeps <= 1:
            canvas = base.copy()
            draw_ball(canvas, traj.x[i], traj.y[i], radius, BALL_RGB)
        else:
            acc = np.zeros((height, width, 3), np.float64)
            nxt = min(i + 1, len(traj.x) - 1)
            for k in range(blur_substeps):
                f = k / blur_substeps * 0.6  # 0.6 shutter
                canvas = base.copy()
                draw_ball(canvas,
                          traj.x[i] + f * (traj.x[nxt] - traj.x[i]),
                          traj.y[i] + f * (traj.y[nxt] - traj.y[i]),
                          radius, BALL_RGB)
                acc += canvas
            canvas = (acc / blur_substeps).astype(np.uint8)
        if noise > 0:
            canvas = np.clip(canvas.astype(np.float64)
                             + rng.normal(0, noise, (height, width, 1)), 0, 255).astype(np.uint8)
        frames[i] = canvas
    return frames


def build(mode: str, width: int = 1344, height: int = 768, theta_deg: float = 45.0,
          n_flight: int = 60, radius_frac: float = 0.022, ground_frac: float = 0.86,
          x0_frac: float = 0.10, range_frac: float = 0.62, blur_substeps: int = 1,
          seed: int = 0) -> tuple[np.ndarray, BallSeed, Traj]:
    """One synthetic clip plus the tracker seed its first frame would have given."""
    radius = max(6.0, radius_frac * height)
    ground_y = height * ground_frac
    y_ref = ground_y - radius
    traj = ideal(R=range_frac * width, theta_deg=theta_deg, n_flight=n_flight,
                 x0=x0_frac * width, y_ref=y_ref, mode=mode)
    frames = render(traj, width, height, radius, ground_frac,
                    blur_substeps=blur_substeps, seed=seed)
    return frames, BallSeed(cx=traj.x[0], cy=traj.y[0], radius=radius, rgb=BALL_RGB), traj
