#!/usr/bin/env python3
"""Deterministic regression tests for P3 event extraction and physics metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from evaluate import Track, _json_safe, extract_flight, load_config, measure_ball, score_results


HERE = Path(__file__).resolve().parent


def ballistic_track(seed: dict, timestamps: np.ndarray, speed: float,
                    gravity: float = 8.0, horizontal_accel: float = 0.0,
                    release_frame: int = 5) -> Track:
    diameter = 2.0 * seed["radius"]
    xy = np.repeat([[seed["cx"], seed["cy"]]], len(timestamps), axis=0).astype(float)
    angle = math.radians(seed["angle_deg"])
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)
    flight_time = 2.0 * vy / gravity
    t0 = timestamps[release_frame]
    tau = np.maximum(timestamps - t0, 0.0)
    active = (timestamps >= t0) & (tau <= flight_time)
    xy[active, 0] = seed["cx"] + diameter * (
        vx * tau[active] + 0.5 * horizontal_accel * tau[active] ** 2
    )
    xy[active, 1] = seed["cy"] - diameter * (
        vy * tau[active] - 0.5 * gravity * tau[active] ** 2
    )
    after = tau > flight_time
    x_land = seed["cx"] + diameter * (
        vx * flight_time + 0.5 * horizontal_accel * flight_time ** 2
    )
    xy[after] = [x_land, seed["cy"]]
    return Track(
        xy=xy,
        radius=np.full(len(timestamps), seed["radius"], dtype=float),
        score=np.ones(len(timestamps), dtype=float),
        name="synthetic",
    )


def evaluate_pair(speed30: float = 8.0, speed60: float = 8.0,
                  horizontal_accel30: float = 0.0,
                  horizontal_accel60: float = 0.0) -> dict:
    cfg, _ = load_config(HERE / "config.yaml")
    timestamps = np.arange(124, dtype=float) / 24.0
    seeds = {
        "lower": {"cx": 100.0, "cy": 500.0, "radius": 20.0, "angle_deg": 30.0},
        "upper": {"cx": 100.0, "cy": 350.0, "radius": 20.0, "angle_deg": 60.0},
    }
    tracks = {
        "lower": ballistic_track(seeds["lower"], timestamps, speed30,
                                 horizontal_accel=horizontal_accel30),
        "upper": ballistic_track(seeds["upper"], timestamps, speed60,
                                 horizontal_accel=horizontal_accel60),
    }
    flights, reasons, measured = {}, {}, {}
    for slot in ("lower", "upper"):
        flights[slot], reasons[slot] = extract_flight(tracks[slot], seeds[slot], timestamps, cfg)
        if flights[slot] is None:
            raise AssertionError(f"synthetic {slot} event extraction failed: {reasons[slot]}")
        measured[slot], _ = measure_ball(tracks[slot], seeds[slot], flights[slot], timestamps)
    qc = {
        "fps": 24.0,
        "balls": {
            slot: {"backend_disagreement_median_radii": 0.0, "fused_coverage": 1.0}
            for slot in seeds
        },
    }
    camera = {"max_translation_diag": 0.0}
    scores, metrics = score_results(measured, flights, qc, camera, cfg)
    return {"event_reasons": reasons, "scores": scores, "metrics": metrics}


def main() -> int:
    clean = evaluate_pair()
    unequal = evaluate_pair(speed30=6.0, speed60=8.0)
    accelerated = evaluate_pair(horizontal_accel30=3.0, horizontal_accel60=3.0)
    assertions = {
        "json_boolean_preserved": _json_safe(True) is True,
        "clean_events_resolved": all(value == "ok" for value in clean["event_reasons"].values()),
        "clean_range_error_low": clean["metrics"]["M1_abs"] < 0.02,
        "clean_speed_error_low": clean["metrics"]["initial_speed_error"] < 0.02,
        "clean_parabola_high": clean["scores"]["parabola"] > 95.0,
        "clean_horizontal_high": clean["scores"]["horizontal"] > 95.0,
        "unequal_speed_detected": unequal["metrics"]["initial_speed_error"] > 0.20,
        "unequal_range_detected": unequal["metrics"]["M1_abs"] > 0.20,
        "horizontal_acceleration_penalized": accelerated["scores"]["horizontal"] < 70.0,
    }
    report = {
        "all_passed": all(assertions.values()),
        "assertions": assertions,
        "clean": clean,
        "unequal_speed": unequal,
        "horizontal_acceleration": accelerated,
    }
    (HERE / "synthetic_validation.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"all_passed": report["all_passed"], "assertions": assertions},
                     ensure_ascii=False, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
