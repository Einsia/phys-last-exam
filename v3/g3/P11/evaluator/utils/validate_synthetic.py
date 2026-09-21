#!/usr/bin/env python3
"""Single-factor full-pipeline regression tests for the P11 evaluator."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from evaluate import evaluate_video


WIDTH, HEIGHT = 640, 360
SURFACE_Y = 180
INCIDENCE_X = 300
THETA_I_DEG = 50.0
EXPECTED_N = 1.333
THETA_T_DEG = math.degrees(math.asin(math.sin(math.radians(THETA_I_DEG)) / EXPECTED_N))


def point_on_ray(x: float, anchor_x: float, anchor_y: float, slope: float) -> tuple[int, int]:
    return int(round(x)), int(round(anchor_y + slope * (x - anchor_x)))


def background() -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (188, 188, 188), dtype=np.uint8)
    frame[SURFACE_Y:, :] = (125, 105, 86)
    cv2.line(frame, (50, SURFACE_Y), (590, SURFACE_Y), (85, 105, 110), 2, cv2.LINE_AA)
    cv2.line(frame, (50, 20), (50, 345), (90, 90, 90), 2, cv2.LINE_AA)
    cv2.line(frame, (590, 20), (590, 345), (90, 90, 90), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (84, 14), (110, 42), (30, 30, 30), -1)
    # Static non-red texture gives the camera-motion check unambiguous features.
    for x in range(145, 560, 70):
        cv2.circle(frame, (x, 75 + (x // 70) % 3 * 18), 4, (115, 115, 115), -1)
    return frame


def add_rays(
    frame: np.ndarray,
    theta_t_deg: float,
    water_intersection_shift_px: float = 0.0,
    include_water: bool = True,
    include_reflection: bool = True,
) -> np.ndarray:
    result = frame.copy()
    alpha_i = math.radians(90.0 - THETA_I_DEG)
    slope_i = math.tan(alpha_i)
    source = point_on_ray(100, INCIDENCE_X, SURFACE_Y, slope_i)
    incident = (INCIDENCE_X, SURFACE_Y)
    cv2.line(result, source, incident, (18, 25, 255), 4, cv2.LINE_AA)
    cv2.circle(result, source, 7, (15, 25, 255), -1, cv2.LINE_AA)
    if include_water:
        water_x = INCIDENCE_X + water_intersection_shift_px
        alpha_t = math.radians(90.0 - theta_t_deg)
        slope_t = math.tan(alpha_t)
        water_start = (int(round(water_x)), SURFACE_Y)
        water_end_x = min(575, water_x + (HEIGHT - 8 - SURFACE_Y) / max(0.08, slope_t))
        water_end = point_on_ray(water_end_x, water_x, SURFACE_Y, slope_t)
        cv2.line(result, water_start, water_end, (18, 25, 255), 4, cv2.LINE_AA)
    # A white normal must never be used as truth.  A negative-slope reflected
    # branch must never replace the positive-slope incident ray.
    cv2.line(result, (INCIDENCE_X, 90), (INCIDENCE_X, 270), (245, 245, 245), 1, cv2.LINE_AA)
    if include_reflection:
        reflected_end = point_on_ray(475, INCIDENCE_X, SURFACE_Y, -slope_i)
        cv2.line(result, incident, reflected_end, (30, 45, 190), 2, cv2.LINE_AA)
    return result


def write_case(
    path: Path,
    theta_t_deg: float = THETA_T_DEG,
    shift_px: float = 0.0,
    include_incident: bool = True,
    include_water: bool = True,
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (WIDTH, HEIGHT))
    assert writer.isOpened(), f"cannot create {path}"
    base = background()
    for index in range(60):
        frame = base.copy()
        if index >= 12 and include_incident:
            frame = add_rays(frame, theta_t_deg, shift_px, include_water, include_reflection=True)
        writer.write(frame)
    writer.release()


def main() -> int:
    config = Path(__file__).with_name("config.json")
    records = {}
    with tempfile.TemporaryDirectory(prefix="p11-regression-") as temporary:
        root = Path(temporary)
        cases = {
            "synthetic_perfect": dict(),
            "synthetic_disconnected": dict(shift_px=34.0),
            "synthetic_wrong_snell": dict(theta_t_deg=THETA_I_DEG),
            "synthetic_missing_water": dict(include_water=False),
            "synthetic_laser_off": dict(include_incident=False, include_water=False),
        }
        for name, options in cases.items():
            video = root / f"{name}.mp4"
            write_case(video, **options)
            records[name] = evaluate_video(video, "P11", config, write_debug=False)
    perfect = records["synthetic_perfect"]
    assert perfect["statuses"]["measurement_valid"] is True, perfect["statuses"]
    assert perfect["metrics"]["M1_snell_residual_abs"] < 0.04, perfect["metrics"]
    assert perfect["metrics"]["M2_intersection_disagreement_normalized"] < 0.012, perfect["metrics"]
    assert perfect["statuses"]["physics_pass"] is True, perfect["statuses"]
    disconnected = records["synthetic_disconnected"]
    assert disconnected["statuses"]["measurement_valid"] is True, disconnected["statuses"]
    assert disconnected["metrics"]["M2_intersection_disagreement_normalized"] > 0.035, disconnected["metrics"]
    assert disconnected["statuses"]["physics_pass"] is False, disconnected["statuses"]
    wrong = records["synthetic_wrong_snell"]
    assert wrong["statuses"]["measurement_valid"] is True, wrong["statuses"]
    assert wrong["metrics"]["M1_snell_residual_abs"] > 0.25, wrong["metrics"]
    assert wrong["statuses"]["physics_pass"] is False, wrong["statuses"]
    missing = records["synthetic_missing_water"]
    assert missing["statuses"]["measurement_valid"] is False, missing["statuses"]
    assert missing["metrics"]["M1_snell_residual_abs"] is None, missing["metrics"]
    assert missing["metrics"]["M2_intersection_disagreement_normalized"] is None, missing["metrics"]
    laser_off = records["synthetic_laser_off"]
    assert laser_off["statuses"]["structural_ok"] is False, laser_off["statuses"]
    assert laser_off["metrics"]["M1_snell_residual_abs"] is None, laser_off["metrics"]
    summary = {
        name: {
            "statuses": result["statuses"],
            "metrics": result["metrics"],
            "overall_continuous": result["scores"]["overall"],
            "overall_legacy_0_100": result["legacy_scores_0_100"]["overall_gated_score_0_100"],
        }
        for name, result in records.items()
    }
    print(json.dumps(summary, indent=2))
    print("P11 synthetic single-factor regressions: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
