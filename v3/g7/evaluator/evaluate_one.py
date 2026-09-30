"""Single-video adapter that emits the strict G7 public JSON contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V3_ROOT = HERE.parents[1]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from g7.evaluator.contract import TASK_METRICS, build_public_result, validate_public_result
from g7.evaluator.evaluate import evaluate as evaluate_raw


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_one(
    *,
    data_root: Path,
    task_id: str,
    sample_id: str,
    output: Path,
    debug_dir: Path,
    video: Path | None = None,
    image: Path | None = None,
    video_prompt: str | None = None,
    seed: int | None = None,
    model: str = "minimax-h3",
) -> dict[str, Any]:
    data_root = data_root.resolve()
    task_spec = {item["task_id"]: item for item in _load(data_root / "data" / "tasks.json")["tasks"]}[task_id]
    manifest = _load(data_root / "data" / "manifest.json")
    row = next((item for item in manifest.get("samples", []) if str(item.get("task_id")) == task_id and str(item.get("sample_id")) == sample_id), None)
    if row:
        video_rel = Path(str(row["video"]))
        image_rel = Path(str(row["first_frame"]))
        video = video or data_root / video_rel
        image = image or data_root / image_rel
        seed = seed if seed is not None else row.get("seed")
        video_prompt = video_prompt or str(task_spec.get("prompt_en", ""))
        source = row.get("source")
        image_variant = row.get("image_variant")
        image_id = f"{source}_{int(image_variant):02d}" if source and image_variant is not None else None
    else:
        if video is None or image is None or video_prompt is None:
            raise ValueError("unknown sample_id; provide --video, --image and --video-prompt")
        video_rel = Path(video)
        image_rel = Path(image)
        source = None
        image_variant = None
        image_id = None
    if task_id not in TASK_METRICS:
        raise ValueError(f"unsupported task_id: {task_id}")
    debug_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw = evaluate_raw(task_id, video, sample_id, debug_dir)
    except Exception as exc:
        raw = {
            "task_id": task_id,
            "sample_id": sample_id,
            "extract_success": False,
            "metrics": {},
            "measurements": {},
            "diagnostics": {},
            "failure_reason": "evaluator_exception",
            "failure_reasons": ["evaluator_exception"],
            "failure_details": {"exception": f"{type(exc).__name__}: {exc}"},
            "physics_pass": False,
            "physics_failure_reasons": ["evaluator_exception"],
            "debug_artifacts": [],
        }
    public = build_public_result(
        raw,
        task_id=task_id,
        sample_id=sample_id,
        video_path=video_rel,
        image_path=image_rel,
        video_prompt=str(video_prompt or ""),
        seed=seed,
        model=model,
        package_root=data_root,
        source=source,
        image_variant=image_variant,
        image_id=image_id,
    )
    errors = validate_public_result(public)
    if errors:
        raise ValueError(f"contract violation: {'; '.join(errors)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return public


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True, choices=sorted(TASK_METRICS))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video-prompt")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", default="minimax-h3")
    args = parser.parse_args(argv)
    debug_dir = args.debug_dir or args.output.parent / "debug" / args.sample_id
    evaluate_one(
        data_root=args.data_root,
        task_id=args.task_id,
        sample_id=args.sample_id,
        output=args.output,
        debug_dir=debug_dir,
        video=args.video,
        image=args.image,
        video_prompt=args.video_prompt,
        seed=args.seed,
        model=args.model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
