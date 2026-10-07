"""Command-line dispatcher for the five deterministic Group 7 evaluators.

Every task module consumes decoded BGR frames and applies explicit geometry or
physics equations. This entry point performs video I/O once and never calls a
VLM, OCR service, or learned detector.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .tasks import p11_mechanics, p20_optics, p28_thermal, p9_rotational, p13_pendulum
    from .physics import physics_decision
except ImportError:  # direct ``python evaluator/evaluate.py`` execution
    from evaluator.tasks import p11_mechanics, p20_optics, p28_thermal, p9_rotational, p13_pendulum
    from evaluator.physics import physics_decision

TASKS = {
    item["task_id"]: item
    for item in json.loads((ROOT / "tasks.json").read_text(encoding="utf-8"))["tasks"]
}
EVALUATORS = {
    "P9": p9_rotational.evaluate,
    "P13": p13_pendulum.evaluate,
    "P11": p11_mechanics.evaluate,
    "P20": p20_optics.evaluate,
    "P28": p28_thermal.evaluate,
}


def read_video(video: str | Path) -> tuple[float, list[np.ndarray], dict[str, Any]]:
    """Decode every frame and retain container metadata for traceability."""

    path = Path(video)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0, [], {"video": str(path), "decode_opened": False}
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not math.isfinite(fps) or fps <= 0.0:
        fps = 0.0
    return fps, frames, {
        "video": str(path.resolve()),
        "decode_opened": True,
        "width": width,
        "height": height,
        "fps": fps,
        "declared_frame_count": declared_count,
        "decoded_frame_count": len(frames),
        "duration_seconds": (len(frames) / fps) if fps else None,
    }


def _decode_failure(task_id: str, sample_id: int | str, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "extract_success": False,
        "measurements": {"video_metadata": metadata},
        "metrics": {"M1": None},
        "failure_reason": "video_decode_failed",
        "failure_reasons": ["video_decode_failed"],
        "debug_artifacts": [],
    }


def evaluate(task_id: str, video: str | Path, sample_id: int | str, debug_dir: str | Path) -> dict[str, Any]:
    """Evaluate one video through its task-specific deterministic module."""

    if task_id not in EVALUATORS:
        raise ValueError(f"unsupported task_id: {task_id}")
    # The CLI always supplies a debug directory, but the public Python API is
    # also used by batch callers. P11/P28 need a concrete path
    # for their plots; normalize ``None`` here instead of letting individual
    # task modules fail with ``Path(None)``.
    if debug_dir is None:
        debug_dir = Path.cwd() / ".vdm_debug" / task_id / str(sample_id)
    fps, frames, metadata = read_video(video)
    if not frames:
        output = _decode_failure(task_id, sample_id, metadata)
        decision, reasons = physics_decision(task_id, output)
        output["physics_pass"] = decision
        output["physics_failure_reasons"] = reasons
        return output
    output = EVALUATORS[task_id](frames, fps, debug_dir, sample_id=sample_id)
    output.setdefault("task_id", task_id)
    output.setdefault("sample_id", sample_id)
    output.setdefault("measurements", {}).setdefault("video_metadata", metadata)
    output.setdefault("failure_reason", None)
    output.setdefault(
        "failure_reasons",
        [] if output.get("extract_success") else [output["failure_reason"]],
    )
    output.setdefault("debug_artifacts", [])
    decision, reasons = physics_decision(task_id, output)
    output["physics_pass"] = decision
    output["physics_failure_reasons"] = reasons
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--task_id", required=True, choices=sorted(TASKS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample_id", default="0",
                        help="sample index or stable current-batch sample ID")
    parser.add_argument("--debug-dir", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = Path(args.output)
    debug_dir = Path(args.debug_dir) if args.debug_dir else output.parent / "debug" / str(args.sample_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.task_id, args.video, args.sample_id, debug_dir)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("extract_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
