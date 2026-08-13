"""Residuals, hard violations, and the four aggregate numbers.

Residual convention. Every primary invariant is a quantity whose theoretical value
is a known ratio, so the residual is the symmetric log ratio

    e = |log(measured / theory)|

which is scale-free, treats 2x and 0.5x as equally wrong, and needs no arbitrary
choice of which side goes in the denominator. e = 0.10 is roughly "10% off".

Measurability vs physics. The gate is split in two on purpose:

  observability   Can the ball be seen and tracked at all -- coverage, camera drift,
                  tracker disagreement. Failing this makes the sample NOT measurable
                  and lowers PMR.
  structure       The ball was tracked cleanly but the clip has no rise-apex-fall at
                  all. This is measurable and counts as a HARD VIOLATION.

Keeping them apart is what stops a model from buying a good physics score by
producing clips that fail to measure: a ball that never comes down is not an
unmeasurable video, it is a wrong one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Residual:
    """One invariant on one sample.

    `primary` means it counts toward PPR. `in_pe` means it also enters the PE mean,
    which is restricted to log-ratio residuals so the average stays interpretable:
    a normalised-RMS of 0.1 and a log-ratio of 0.1 mean very different things, and
    averaging them together would produce a number with no unit at all.
    """

    name: str
    measured: float
    theory: float
    residual: float
    tol: float
    passed: bool
    primary: bool
    in_pe: bool = True
    kind: str = "log_ratio"
    note: str = ""


@dataclass
class Violation:
    name: str
    detail: str


@dataclass
class SampleResult:
    sample_id: str
    task: str
    model: str
    measurable: bool
    gate_reasons: list[str] = field(default_factory=list)
    residuals: list[Residual] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    qc: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> list[Residual]:
        return [r for r in self.residuals if r.primary]

    @property
    def scene_residual(self) -> float | None:
        """The e_i of the proposal: mean of the PE-eligible primary residuals."""
        vals = [r.residual for r in self.residuals
                if r.primary and r.in_pe and np.isfinite(r.residual)]
        return float(np.mean(vals)) if vals else None

    @property
    def passed_all(self) -> bool:
        prim = self.primary
        return bool(prim) and all(r.passed for r in prim) and not self.violations

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scene_residual"] = self.scene_residual
        d["passed_all"] = self.passed_all
        return d


def log_ratio_residual(measured: float, theory: float) -> float:
    """|log(measured/theory)|, +inf when either side is non-positive."""
    if not (np.isfinite(measured) and np.isfinite(theory)) or measured <= 0 or theory <= 0:
        return float("inf")
    return float(abs(np.log(measured / theory)))


def make_residual(name: str, measured: float, theory: float, tol: float,
                  primary: bool = True, note: str = "") -> Residual:
    """Ratio invariant: theory is the expected value of `measured`."""
    e = log_ratio_residual(measured, theory)
    return Residual(name=name, measured=float(measured), theory=float(theory),
                    residual=e, tol=float(tol), passed=bool(e < tol),
                    primary=primary, in_pe=True, kind="log_ratio", note=note)


def make_abs_residual(name: str, measured: float, tol: float,
                      primary: bool = True, note: str = "") -> Residual:
    """Deviation invariant: the target is 0, so the residual is the value itself.

    Excluded from PE (see Residual.in_pe) but still gates PPR.
    """
    e = float(abs(measured)) if np.isfinite(measured) else float("inf")
    return Residual(name=name, measured=float(measured), theory=0.0, residual=e,
                    tol=float(tol), passed=bool(e < tol), primary=primary,
                    in_pe=False, kind="absolute", note=note)


def aggregate(results: list[SampleResult]) -> dict:
    """PMR / PE / PPR / HVR plus the per-invariant breakdown.

    PE averages only over samples whose residuals are computable, and reports that
    count next to it -- averaging over samples where the residual is undefined would
    either silently drop them or poison the mean with +inf. PPR and HVR keep every
    measurable sample in the denominator, so a structurally broken clip still costs
    the model on both.
    """
    n_all = len(results)
    meas = [r for r in results if r.measurable]
    n_m = len(meas)

    out: dict[str, Any] = {
        "n_videos": n_all,
        "n_measurable": n_m,
        "PMR": (n_m / n_all) if n_all else float("nan"),
    }
    if not n_m:
        out.update({"PE": float("nan"), "PPR": float("nan"), "HVR": float("nan"),
                    "n_with_residuals": 0})
        return out

    scene = [(r, r.scene_residual) for r in meas]
    have = [v for _, v in scene if v is not None and np.isfinite(v)]
    out["PE"] = float(np.mean(have)) if have else float("nan")
    out["PE_median"] = float(np.median(have)) if have else float("nan")
    out["n_with_residuals"] = len(have)
    out["PPR"] = float(np.mean([r.passed_all for r in meas]))
    out["HVR"] = float(np.mean([bool(r.violations) for r in meas]))

    per: dict[str, dict] = {}
    for r in meas:
        for res in r.residuals:
            slot = per.setdefault(res.name, {"e": [], "pass": [], "primary": res.primary,
                                             "measured": [], "theory": []})
            if np.isfinite(res.residual):
                slot["e"].append(res.residual)
                slot["measured"].append(res.measured)
                slot["theory"].append(res.theory)
            slot["pass"].append(res.passed)
    out["per_invariant"] = {
        k: {
            "n": len(v["pass"]),
            "n_finite": len(v["e"]),
            "primary": v["primary"],
            "mean_residual": float(np.mean(v["e"])) if v["e"] else float("nan"),
            "median_residual": float(np.median(v["e"])) if v["e"] else float("nan"),
            "pass_rate": float(np.mean(v["pass"])) if v["pass"] else float("nan"),
            "mean_measured": float(np.mean(v["measured"])) if v["measured"] else float("nan"),
            "mean_theory": float(np.mean(v["theory"])) if v["theory"] else float("nan"),
        }
        for k, v in sorted(per.items())
    }

    vcount: dict[str, int] = {}
    for r in meas:
        for v in r.violations:
            vcount[v.name] = vcount.get(v.name, 0) + 1
    out["violation_counts"] = dict(sorted(vcount.items()))

    gcount: dict[str, int] = {}
    for r in results:
        if not r.measurable:
            for g in r.gate_reasons:
                key = g.split("(")[0]
                gcount[key] = gcount.get(key, 0) + 1
    out["gate_failure_counts"] = dict(sorted(gcount.items()))
    return out
