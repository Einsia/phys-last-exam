#!/usr/bin/env python3
"""Annotate normalized ROIs on the first-frame image only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2

TASK_DIR = Path(__file__).resolve().parents[1]
TASK_ID = TASK_DIR.name
FIELD_MAP = {
    "P13": ["mirror", "incident_ray", "reflected_ray"],
    "P19": ["left_arm", "right_arm", "junction_exclude", "floor"],
    "P28": ["support", "left_thread", "right_thread", "left_ball", "right_ball"],
    "P40": ["ground", "left_pile", "right_pile"],
    "P42": ["board", "block_left", "block_right"],
    "P21b": ["vessel", "ice", "stone_seed", "floor"],
    "P21c": ["vessel", "ice_initial", "surface"],
    "P23": ["vessel_inner", "surface", "floor"],
    "P36": ["board", "magnet", "control"],
    "P44": ["chain", "left_support", "right_support"],
}


def find_video(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if not path.is_file():
            raise SystemExit(f"video not found: {path}")
        return path
    candidates = []
    for directory in ("minimax_h3/videos", "minimax-h3/videos", "videos"):
        candidates.extend(sorted((TASK_DIR / directory).glob("*.mp4")))
    candidates.extend(sorted(TASK_DIR.glob("*.mp4")))
    if not candidates:
        raise SystemExit("no MP4 found; pass --video")
    return candidates[0]


def parse_coords(items: list[str], fields: list[str]) -> dict[str, tuple[float, float, float, float]]:
    """Parse normalized FIELD=x,y,w,h values for GUI-free annotation."""
    parsed = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"invalid --coords value {item!r}; use FIELD=x,y,w,h")
        field, raw = item.split("=", 1)
        if field not in fields:
            raise SystemExit(f"unknown ROI field {field!r}; choose from {', '.join(fields)}")
        if field in parsed:
            raise SystemExit(f"duplicate ROI field: {field}")
        parts = raw.split(",")
        if len(parts) != 4:
            raise SystemExit(f"invalid --coords value {item!r}; expected four numbers")
        try:
            box = tuple(float(value) for value in parts)
        except ValueError as exc:
            raise SystemExit(f"invalid --coords value {item!r}; expected numbers") from exc
        x, y, w, h = box
        if (not all(math.isfinite(value) for value in box) or
                min(x, y, w, h) < 0 or w <= 0 or h <= 0 or
                x + w > 1 or y + h > 1):
            raise SystemExit(f"ROI {field!r} must satisfy 0<=x,y,w,h and x+w,y+h<=1")
        parsed[field] = box
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Annotate normalized ROIs for {TASK_ID}")
    parser.add_argument("--video", help="optional source video; only frame 0 is read")
    parser.add_argument("--image", help="first-frame image to annotate (preferred)")
    parser.add_argument("--frame", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--output", default=str(TASK_DIR / "evaluator" / "roi.json"))
    parser.add_argument("--fields", nargs="+", default=FIELD_MAP.get(TASK_ID, ["roi"]))
    parser.add_argument("--coords", action="append", default=[], metavar="FIELD=X,Y,W,H",
                        help="normalized ROI; repeat for multiple fields")
    parser.add_argument("--no-gui", action="store_true",
                        help="fail if any requested ROI is missing instead of opening a window")
    args = parser.parse_args()

    if args.frame != 0:
        raise SystemExit("ROI annotation is first-frame-only; --frame must be 0")
    video = None
    if args.image:
        image = Path(args.image).expanduser()
        if not image.is_file():
            raise SystemExit(f"first-frame image not found: {image}")
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"cannot read first-frame image: {image}")
    else:
        video = find_video(args.video)
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open video: {video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("cannot read video frame 0")
    height, width = bgr.shape[:2]
    result = {
        "task_id": TASK_ID,
        "frame_size": [width, height],
        "source_video": str(video) if video else None,
        "source_frame": 0,
        "rois": {},
    }
    coords = parse_coords(args.coords, args.fields)
    missing = [field for field in args.fields if field not in coords]
    if args.no_gui and missing:
        raise SystemExit("--no-gui requires --coords for: " + ", ".join(missing))
    canvas = bgr.copy()
    for field in args.fields:
        if field in coords:
            nx, ny, nw, nh = coords[field]
            x, y = int(round(nx * width)), int(round(ny * height))
            w, h = max(1, int(round(nw * width))), max(1, int(round(nh * height)))
            x, y = min(x, width - 1), min(y, height - 1)
            w, h = min(w, width - x), min(h, height - y)
            result["rois"][field] = [float(nx), float(ny), float(nw), float(nh)]
        else:
            title = f"{TASK_ID}: {field} (Enter=accept, Esc=cancel)"
            x, y, w, h = cv2.selectROI(title, canvas, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(title)
            if w <= 0 or h <= 0:
                raise SystemExit(f"cancelled or empty ROI: {field}")
            result["rois"][field] = [
                float(x / width), float(y / height),
                float(w / width), float(h / height),
            ]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(canvas, field, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    cv2.destroyAllWindows()
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
