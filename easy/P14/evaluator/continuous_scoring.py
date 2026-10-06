#!/usr/bin/env python3
"""Shared helpers for continuous 0–1 physical-quality scores."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping


SCORE_VERSION = "continuous-0-1-v1"
SCORE_FLOOR = 0.01


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def measurable_floor(value: float, floor: float = SCORE_FLOOR) -> float:
    return max(float(floor), clamp01(value))


def residual_quality(residual: float | None, scale: float,
                     floor: float = SCORE_FLOOR) -> float:
    """Return q(r,s)=1/(1+r/s); missing evidence gets the measurable floor."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if residual is None or not math.isfinite(float(residual)):
        return float(floor)
    r = max(0.0, float(residual))
    return measurable_floor(1.0 / (1.0 + r / float(scale)), floor)


def high_quality(value: float | None, target: float, scale: float,
                 floor: float = SCORE_FLOOR) -> float:
    """Map a high-is-good value through its shortfall from target."""
    if value is None or not math.isfinite(float(value)):
        return float(floor)
    return residual_quality(max(0.0, float(target) - float(value)), scale, floor)


def evidence_fraction(observed: float | None, required: float,
                      floor: float = SCORE_FLOOR) -> float:
    if required <= 0:
        raise ValueError("required must be positive")
    if observed is None or not math.isfinite(float(observed)):
        return float(floor)
    return measurable_floor(float(observed) / float(required), floor)


def weighted_geometric(values: Mapping[str, float], weights: Mapping[str, float],
                       floor: float = SCORE_FLOOR) -> float:
    pairs: list[tuple[float, float]] = []
    for key, value in values.items():
        weight = float(weights.get(key, 0.0))
        if weight > 0:
            pairs.append((measurable_floor(float(value), floor), weight))
    if not pairs:
        return float(floor)
    total = sum(weight for _, weight in pairs)
    return measurable_floor(math.exp(sum(
        weight / total * math.log(value) for value, weight in pairs
    )), floor)


def assert_score_contract(scores: Iterable[float], measurement_valid: bool,
                          overall: float) -> None:
    values = [float(value) for value in scores]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise AssertionError(f"score outside [0,1]: {values}")
    if measurement_valid and not (0.0 < float(overall) <= 1.0):
        raise AssertionError(f"valid measurement must have non-zero score: {overall}")
    if not measurement_valid and float(overall) != 0.0:
        raise AssertionError(f"invalid measurement must score zero: {overall}")
