#!/usr/bin/env python3
"""Temporary diagnostics for ball/pattern visibility; not part of formal scorer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

import evaluate


HERE = Path(__file__).resolve().parent
VIDEO_ROOT = HERE.parent / "minimax_h3" / "prompt_v1_20260825" / "videos"
RESULT_ROOT = HERE / "results_v1"
CONFIG = yaml.safe_load((HERE / "config_v1.yaml").read_text())


def longest_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return int(best)


def frame_visibility(frame: np.ndarray, center: np.ndarray, radius: float) -> tuple[float, float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)
    yy, xx = np.indices(gray.shape)
    distance = np.hypot(xx - center[0], yy - center[1]) / max(radius, 1e-6)
    inner = distance <= 0.70
    annulus = (distance >= 1.10) & (distance <= 1.35)
    # Median circumference gradient, normalized by nearby image texture.
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    boundary = (distance >= 0.90) & (distance <= 1.05)
    edge = float(np.percentile(magnitude[boundary], 60) / max(np.percentile(magnitude[annulus], 60), 3.0))
    colour = float(np.linalg.norm(np.median(lab[inner], axis=0) - np.median(lab[annulus], axis=0)))
    texture = float(np.median(np.abs(gray[inner].astype(float) - np.median(gray[inner]))))
    return edge, colour, texture


records = []
for video in sorted(VIDEO_ROOT.glob("P6_*_seed*.mp4")):
    source = evaluate.infer_source_id(video, CONFIG["sources"])
    frames, _, _ = evaluate.read_video(video)
    init = CONFIG["sources"][source]
    track = evaluate.track_ball(frames, tuple(init["center"]), float(init["radius"]))
    rows = list(csv.DictReader((RESULT_ROOT / video.stem / "frame_measurements.csv").open()))
    motion = np.array([row["motion_interval"].lower() == "true" for row in rows])
    hough = track["hough_confidence"]
    flow = track["flow_confidence"]
    edges, colours, textures = [], [], []
    for frame, center, radius in zip(frames, track["centers"], track["radii"]):
        edge, colour, texture = frame_visibility(frame, center, radius)
        edges.append(edge)
        colours.append(colour)
        textures.append(texture)
    edges = np.asarray(edges)
    colours = np.asarray(colours)
    textures = np.asarray(textures)
    idx = np.flatnonzero(motion)
    sl = slice(idx[0], idx[-1] + 1)
    rec = {
        "sample_id": video.stem,
        "hough_p10": float(np.percentile(hough[motion], 10)),
        "hough_low_run_0.10": longest_run(hough[sl] < 0.10),
        "hough_low_run_0.20": longest_run(hough[sl] < 0.20),
        "flow_p10": float(np.percentile(flow[motion], 10)),
        "edge_p10": float(np.percentile(edges[motion], 10)),
        "edge_low_run_0.8": longest_run(edges[sl] < 0.8),
        "colour_p10": float(np.percentile(colours[motion], 10)),
        "colour_low_run_8": longest_run(colours[sl] < 8),
        "texture_p10": float(np.percentile(textures[motion], 10)),
        "texture_low_run_2": longest_run(textures[sl] < 2),
    }
    records.append(rec)
    print(json.dumps(rec), flush=True)
(HERE / "visibility_probe.json").write_text(json.dumps(records, indent=2) + "\n")
