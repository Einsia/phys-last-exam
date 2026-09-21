#!/usr/bin/env python3
"""Deterministic regression tests for P4 event detection and metric directionality."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import evaluate as ev


HERE = Path(__file__).resolve().parent


def make_track(heights: list[float], durations: list[int], seed: int = 0,
               noise: float = 0.20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    plate = 300.0
    ys = [55.0] * 6
    # Initial clean release and free fall.
    u = np.linspace(0.0, 1.0, 18)
    ys.extend((55.0 + (plate - 55.0) * u ** 2).tolist()[1:])
    for height, duration in zip(heights, durations):
        u = np.linspace(0.0, 1.0, duration + 1)
        arc = plate - 4.0 * height * u * (1.0 - u)
        ys.extend(arc.tolist()[1:])
    y = np.asarray(ys)
    x = 250.0 + rng.normal(0.0, noise, len(y))
    y = y + rng.normal(0.0, noise, len(y))
    return np.c_[x, y]


def metric_case(heights: list[float], durations: list[int], cfg: dict) -> dict:
    xy = make_track(heights, durations)
    arcs, dbg = ev.detect_arcs(xy, radius=15.0, cfg=cfg)
    shift = np.zeros_like(xy)
    td = {"frame_diagonal_px": 800.0, "fused_coverage": 1.0,
          "confident_coverage": 1.0,
          "pairwise_median_disagreement_radii": {"a__b": 0.1}}
    metrics, scores = ev.calculate_metrics(arcs, xy, 15.0, shift, td, cfg)
    return {"detected_arcs": len(arcs), "event_debug": dbg,
            "metrics": metrics, "scores": scores}


def main() -> int:
    cfg = json.loads((HERE / "config.json").read_text())
    # T scales with sqrt(h), so M1 should sit at the tracker/noise floor.
    clean = metric_case([120, 76.8, 49.15, 31.46], [30, 24, 19, 15], cfg)
    nonmonotonic = metric_case([120, 76.8, 105.0, 67.2], [30, 24, 28, 22], cfg)
    equal_height = metric_case([90.0, 90.0, 90.0, 90.0], [26, 26, 26, 26], cfg)
    late_increase = metric_case([120.0, 90.0, 70.0, 76.0], [30, 26, 23, 24], cfg)
    incomplete = metric_case([120, 76.8, 49.15], [30, 24, 19], cfg)

    # Short-gap interpolation is allowed, long-gap fabrication is not.
    gapped = make_track([120, 76.8, 49.15, 31.46], [30, 24, 19, 15])
    gapped[40:43] = np.nan
    gapped[70:77] = np.nan
    filled, direct = ev.interpolate_short_gaps(gapped, max_gap=4)
    gap_test = {"short_gap_filled": bool(np.isfinite(filled[41]).all()),
                "long_gap_left_missing": bool(np.isnan(filled[73]).all()),
                "direct_fraction": float(direct.mean())}

    assertions = {
        "clean_four_arcs": clean["detected_arcs"] >= 4,
        "clean_low_M1": clean["metrics"]["M1_height_time_restitution_consistency"] < 0.10,
        "nonmonotonic_detected": nonmonotonic["metrics"]["height_monotonic_violation_count"] >= 1,
        "nonmonotonic_score_lower": (nonmonotonic["scores"]["monotonic_energy_loss"] <
                                      clean["scores"]["monotonic_energy_loss"]),
        "equal_height_not_accepted_as_decreasing":
            equal_height["scores"]["monotonic_energy_loss"] < 55.0,
        "single_late_increase_is_hard_failure": not ev.physics_decision(
            True,
            {**late_increase["scores"],
             "overall": ev.geometric_score(late_increase["scores"], cfg["dimension_weights"])},
            late_increase["metrics"], cfg)[0],
        "incomplete_below_four": incomplete["detected_arcs"] < 4,
        "short_gap_policy": gap_test["short_gap_filled"] and gap_test["long_gap_left_missing"],
    }
    report = {"all_passed": all(assertions.values()), "assertions": assertions,
              "clean": clean, "nonmonotonic": nonmonotonic,
              "equal_height": equal_height,
              "late_increase": late_increase,
              "incomplete": incomplete, "gap_test": gap_test}
    (HERE / "synthetic_validation.json").write_text(
        json.dumps(ev.finite(report), indent=2))
    print(json.dumps(ev.finite(report), indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
