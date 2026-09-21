#!/usr/bin/env python3
"""Artifact integrity audit and compact visual-review montages for the 24 P4 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent


def tile(image: np.ndarray, title: str, width: int = 600, height: int = 420) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, np.uint8)
    top = 44
    h, w = image.shape[:2]
    scale = min(width / w, (height - top) / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    y = top + (height - top - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, title, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def montage(items: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not items:
        return np.zeros((1, 1, 3), np.uint8)
    blank = np.full_like(items[0], 245)
    rows = []
    for i in range(0, len(items), columns):
        row = items[i:i + columns]
        row += [blank] * (columns - len(row))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=HERE / "results")
    args = p.parse_args()
    results = []
    plot_tiles, key_tiles = [], []
    missing: dict[str, list[str]] = {}
    for jp in sorted((args.results / "json").glob("*.json")):
        r = json.loads(jp.read_text())
        sid = r["sample_id"]
        d = args.results / "debug" / sid
        required = [jp, d / "overlay.mp4", d / "trajectory_plot.png",
                    d / "keyframes.jpg", d / "fused_track.csv"]
        bad = [str(x) for x in required if not x.exists() or x.stat().st_size == 0]
        if bad:
            missing[sid] = bad
        label = (f"{sid} | arcs={r.get('measurements', {}).get('complete_rebound_arc_count')} "
                 f"score={r.get('scores', {}).get('overall', 0):.1f} "
                 f"{'PASS' if r.get('physics_pass') else 'FAIL'}")
        plot = cv2.imread(str(d / "trajectory_plot.png"))
        keys = cv2.imread(str(d / "keyframes.jpg"))
        if plot is not None:
            plot_tiles.append(tile(plot, label))
        if keys is not None:
            key_tiles.append(tile(keys, label))
        results.append(r)

    if plot_tiles:
        cv2.imwrite(str(args.results / "audit_trajectory_montage.jpg"),
                    montage(plot_tiles), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if key_tiles:
        cv2.imwrite(str(args.results / "audit_keyframes_montage.jpg"),
                    montage(key_tiles), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    n = len(results)
    report = {
        "n_results": n,
        "all_24_present": n == 24,
        "all_artifacts_present": not missing,
        "missing_artifacts": missing,
        "all_no_llm_vlm": all(r.get("llm_or_vlm_used") is False for r in results),
        "all_video_hashes_unique": len({r.get("video_sha256") for r in results}) == n,
        "extract_success_count": sum(bool(r.get("extract_success")) for r in results),
        "structural_success_count": sum(bool(r.get("structural_ok")) for r in results),
        "physics_pass_count": sum(bool(r.get("physics_pass")) for r in results),
        "montages": [str(args.results / "audit_trajectory_montage.jpg"),
                     str(args.results / "audit_keyframes_montage.jpg")],
    }
    (args.results / "artifact_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["all_24_present"] and report["all_artifacts_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
