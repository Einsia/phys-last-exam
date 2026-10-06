"""Programmatic first frames for the projectile task.

Drawing the frame instead of sourcing it buys two things the benchmark needs:

1. The tracker seed (ball colour, radius, centre) is exact rather than estimated,
   so tracking failures are the video's fault and not the seed's.
2. The scene is deliberately plain -- flat background, one horizon line, one
   high-chroma ball -- which is the regime where "did we see the ball" is
   decidable and the measurability gate means something.

The launch angle is *not* drawn into the frame. It is only stated in the prompt,
so M1 measured against the nominal angle is a real instruction-following test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Ball colour is high-chroma and unlike anything else in the scene, which keeps the
# Lab colour distance in the tracker unambiguous. Shading is kept mild (see _shade)
# so a specular lobe can't drag the colour seed off the ball.
BALL_RGB = (222, 74, 38)
SKY_RGB = (196, 208, 216)
GROUND_RGB = (122, 128, 118)


@dataclass
class BallSeed:
    cx: float
    cy: float
    radius: float
    rgb: tuple[int, int, int]


@dataclass
class SampleSpec:
    """Everything downstream stages need, carried in a sidecar next to the image."""

    sample_id: str
    task: str
    width: int
    height: int
    prompt: str
    first_frame: str
    ball: BallSeed
    params: dict = field(default_factory=dict)
    ground_y: float | None = None
    notes: str = ""

    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["ball"]["rgb"] = list(payload["ball"]["rgb"])
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return str(path)

    @staticmethod
    def load(path: str | Path) -> "SampleSpec":
        raw = json.loads(Path(path).read_text())
        ball = raw.pop("ball")
        ball["rgb"] = tuple(ball["rgb"])
        return SampleSpec(ball=BallSeed(**ball), **raw)


def _shade(radius: int) -> np.ndarray:
    """Soft top-left lambertian ramp in [0.90, 1.06]. Enough to read as a sphere,
    too weak to split the ball into two colour clusters."""
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float64)
    r = np.hypot(xx, yy) / max(radius, 1)
    lit = np.clip(1.0 - 0.55 * (xx / max(radius, 1)) + 0.55 * (-yy / max(radius, 1)), 0.0, 2.0)
    ramp = 0.90 + 0.16 * (lit / 2.0)
    ramp[r > 1.0] = 1.0
    return ramp


def draw_ball(canvas: np.ndarray, cx: float, cy: float, radius: float,
              rgb: tuple[int, int, int], supersample: int = 4) -> None:
    """Alpha-composite an antialiased shaded disc. Supersampled coverage keeps the
    rendered centroid within ~0.05 px of (cx, cy), which matters because the
    synthetic renders are the noise floor the tolerances are calibrated against."""
    h, w = canvas.shape[:2]
    pad = int(np.ceil(radius)) + 2
    x0, x1 = max(0, int(cx) - pad), min(w, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(h, int(cy) + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return

    s = supersample
    ys = (np.arange(y0 * s, y1 * s) + 0.5) / s
    xs = (np.arange(x0 * s, x1 * s) + 0.5) / s
    dist = np.hypot(xs[None, :] - cx, ys[:, None] - cy)
    cover = (dist <= radius).astype(np.float64)
    alpha = cover.reshape(y1 - y0, s, x1 - x0, s).mean(axis=(1, 3))

    ramp = _shade(int(round(radius)))
    ry = np.clip(((np.arange(y0, y1) - cy) + radius).astype(int), 0, ramp.shape[0] - 1)
    rx = np.clip(((np.arange(x0, x1) - cx) + radius).astype(int), 0, ramp.shape[1] - 1)
    shade = ramp[np.ix_(ry, rx)]

    colour = np.clip(np.asarray(rgb, np.float64)[None, None, :] * shade[..., None], 0, 255)
    patch = canvas[y0:y1, x0:x1].astype(np.float64)
    a = alpha[..., None]
    canvas[y0:y1, x0:x1] = np.clip(patch * (1 - a) + colour * a, 0, 255).astype(np.uint8)


def render_scene(width: int, height: int, ground_frac: float = 0.86,
                 seed: int = 0) -> tuple[np.ndarray, float]:
    """Flat two-tone side view. Returns (canvas, ground_y)."""
    rng = np.random.default_rng(seed)
    ground_y = height * ground_frac
    canvas = np.zeros((height, width, 3), np.uint8)
    canvas[:, :] = SKY_RGB

    gy = int(round(ground_y))
    # Vertical value ramp on the ground reads as depth without adding texture that
    # would break the background-subtraction tracker.
    depth = np.linspace(1.06, 0.90, height - gy)[:, None, None]
    canvas[gy:, :] = np.clip(np.asarray(GROUND_RGB, np.float64) * depth, 0, 255).astype(np.uint8)
    canvas[max(0, gy - 2):gy + 1, :] = (96, 100, 92)

    # Very low-amplitude grain. Keeps a codec from posterising the flats into bands
    # that the frame-difference tracker would read as motion.
    grain = rng.normal(0.0, 1.4, (height, width, 1))
    return np.clip(canvas.astype(np.float64) + grain, 0, 255).astype(np.uint8), ground_y


def make_projectile_first_frame(width: int = 1344, height: int = 768,
                                radius_frac: float = 0.022,
                                start_x_frac: float = 0.10,
                                seed: int = 0) -> tuple[np.ndarray, BallSeed, float]:
    """Ball at rest on the ground near the left edge, leaving room for the arc."""
    canvas, ground_y = render_scene(width, height, seed=seed)
    radius = max(6.0, radius_frac * height)
    cx = start_x_frac * width
    cy = ground_y - radius
    draw_ball(canvas, cx, cy, radius, BALL_RGB)
    return canvas, BallSeed(cx=cx, cy=cy, radius=radius, rgb=BALL_RGB), ground_y
