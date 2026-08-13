"""Thresholds for the projectile task, overridable from YAML.

Tolerances are on the log-ratio residual, so tol=0.15 is "within about 15%". The
defaults are set well above the measurement noise floor observed on the synthetic
exact-physics renders (see scripts/validate_metrics.py, which prints that floor);
the point of calibrating against renders is that the floor is a property of the
tracker and the fitting, not of any video model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class GateConfig:
    """Observability only. Nothing here may depend on whether the physics is right."""

    min_coverage_overall: float = 0.80     # fraction of clip with a detection
    min_coverage_flight: float = 0.90      # fraction inside the flight window
    min_flight_frames: int = 12            # need enough samples to fit
    max_camera_drift: float = 0.02         # cumulative shift / frame diagonal
    max_tracker_disagree: float = 0.50     # median |A-B| in ball radii
    border_margin_radii: float = 1.5       # how close to an edge counts as exiting
    max_detection_gap: int = 3             # consecutive missing frames inside the span
    max_radius_cv: float = 0.45            # ball dissolving or deforming


@dataclass
class ToleranceConfig:
    hr_nominal: float = 0.15               # H/R vs tan(theta_nominal)/4
    time_symmetry: float = 0.10            # N_up / N_down vs 1
    hr_selfconsistent: float = 0.15        # H/R vs tan(theta_measured)/4
    space_symmetry: float = 0.12           # (x_apex-x_launch)/(x_land-x_apex) vs 1
    parabolicity: float = 0.030            # normalised RMS of the parabola fit
    vx_conservation: float = 0.10           # v_x(descent)/v_x(ascent) vs 1
    gravity_symmetry: float = 0.20          # a_y(descent)/a_y(ascent) vs 1
    accel_geometry: float = 0.12            # a_y*N^2/H vs 8


@dataclass
class ViolationConfig:
    """Hard bounds. Crossing one of these is qualitatively wrong, not just inaccurate."""

    time_symmetry_factor: float = 2.0      # N_up/N_down off by >= 2x either way
    hr_factor: float = 3.0                 # H/R off by >= 3x
    parabolicity_gross: float = 0.12       # trajectory is not a parabola at all


@dataclass
class TrackingConfig:
    """Which tracker pair runs, and how the camera shift is obtained.

    The default pair remains SAM2 + CoTracker for backwards-compatible runs. SAM3 +
    CoTracker is also supported; point `sam3_src` at a checkout of Meta's SAM3 repo
    and `sam3_checkpoint` at a local checkpoint, then select it in YAML. The classic
    pair reproduces the tighter synthetic noise floor and needs no GPU.
    """

    backends: tuple[str, str] = ("sam2", "cotracker")
    device: str = "6"                      # CUDA_VISIBLE_DEVICES for the worker
    dtype: str = "bfloat16"
    sam3_src: str = str(PROJECT_ROOT / "cache" / "sam3")
    sam3_checkpoint: str = str(PROJECT_ROOT / "cache" / "sam3" / "sam3.pt")
    sam3_prompt: str = "ball"
    # CoTracker's background grid gives translation directly. Off falls back to phase
    # correlation, which is what the classic pair uses anyway.
    camera_shift_from_tracker: bool = True

    @property
    def is_neural(self) -> bool:
        return any(b in ("sam2", "sam3", "cotracker") for b in self.backends)


@dataclass
class TaskConfig:
    task: str = "P2_projectile"
    gates: GateConfig = field(default_factory=GateConfig)
    tol: ToleranceConfig = field(default_factory=ToleranceConfig)
    viol: ViolationConfig = field(default_factory=ViolationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @staticmethod
    def load(path: str | Path | None) -> "TaskConfig":
        cfg = TaskConfig()
        if path is None:
            return cfg
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        if "task" in raw:
            cfg.task = raw["task"]
        for key, obj in (("gates", cfg.gates), ("tol", cfg.tol), ("viol", cfg.viol),
                         ("tracking", cfg.tracking)):
            for k, v in (raw.get(key) or {}).items():
                if not hasattr(obj, k):
                    raise KeyError(f"unknown {key} option: {k}")
                cur = getattr(obj, k)
                # backends is a tuple in the dataclass but a list in YAML.
                setattr(obj, k, tuple(v) if isinstance(cur, tuple) else type(cur)(v))
        if len(cfg.tracking.backends) != 2:
            raise ValueError("tracking.backends must name exactly two backends "
                             f"(got {cfg.tracking.backends})")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
