#!/usr/bin/env python3
"""Render every overlay frame for 24/24 human temporal sanity review."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent / "results_v1"
OUTPUT = ROOT / "dense_audit_sheets"
OUTPUT.mkdir(parents=True, exist_ok=True)
TILE = 96
COLUMNS = 16
ROWS = 8
HEADER = 26


for sample_dir in sorted(path for path in ROOT.iterdir() if path.is_dir() and path.name.startswith("P6_")):
    measurements = list(csv.DictReader((sample_dir / "frame_measurements.csv").open(encoding="utf-8")))
    capture = cv2.VideoCapture(str(sample_dir / "overlay.mp4"))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sheet = Image.new("RGB", (TILE * COLUMNS, HEADER + TILE * ROWS), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), f"{sample_dir.name} | ALL {frame_count} OVERLAY FRAMES | chronological left-to-right, top-to-bottom", fill="black")
    for frame_index in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            break
        cx = int(round(float(measurements[frame_index]["center_x"])))
        cy = int(round(float(measurements[frame_index]["center_y"])))
        half = 78
        x0, x1 = max(0, cx - half), min(frame.shape[1], cx + half)
        y0, y1 = max(0, cy - half), min(frame.shape[0], cy + half)
        crop = frame[y0:y1, x0:x1]
        crop = cv2.copyMakeBorder(
            crop,
            max(0, 2 * half - crop.shape[0]),
            0,
            max(0, 2 * half - crop.shape[1]),
            0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        tile = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((TILE, TILE))
        ImageDraw.Draw(tile).text((2, 2), str(frame_index), fill="white", stroke_fill="black", stroke_width=2)
        column = frame_index % COLUMNS
        row = frame_index // COLUMNS
        sheet.paste(tile, (column * TILE, HEADER + row * TILE))
    capture.release()
    sheet.save(OUTPUT / f"{sample_dir.name}.jpg", quality=82, optimize=True)
    print(sample_dir.name)
