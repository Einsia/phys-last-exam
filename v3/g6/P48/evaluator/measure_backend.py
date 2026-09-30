#!/usr/bin/env python
"""Evaluate one generated video for this task and write one JSON verdict.

Self-contained: the measurement library is provided by the shared ``shared/physeval`` package,
so all tasks use one audited implementation in addition to the packages in
``requirements.txt``.

    python evaluator/evaluate.py \
        --video output_videos/minimax_h3/sample_00.mp4 \
        --image first_frames/gpt/gpt_01.png \
        --out eval_results/minimax_h3/result_sample_00.json \
        --debug eval_results/minimax_h3/debug/sample_00/plot.png

Which first-frame route and seed a sample number stands for is recorded in
``data/samples.csv``; this reads it from there so the JSON carries
the same ``sample_id``, ``route`` and ``seed`` whether the file was produced by
``run_eval.sh`` or shipped in the package. ``--route`` and ``--seed`` override
it. The continuation prompt comes from ``prompts/video.txt`` (``prompts/video_simulation.txt`` for
the simulation route, where the two routes were worded apart) unless
``--prompt`` is given.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
V3_ROOT = HERE.parents[3]
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from shared.physeval import Context, get_evaluator, read_clip  # noqa: E402

TASK_ID = "P48"


def _rel(path: str) -> str:
    """Path as recorded in the JSON: relative to the task root where possible."""
    try:
        return str(Path(path).resolve().relative_to(TASK_ROOT))
    except ValueError:
        return path


def _from_manifest(sample_id: str) -> dict:
    """What samples.csv says this sample number is."""
    f = TASK_ROOT / "data" / "samples.csv"
    if not f.is_file():
        return {}
    for line in f.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 3 and parts[0] == sample_id:
            return {"route": parts[1], "seed": int(parts[2])}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--image", required=True, help="first frame the video "
                                                   "was generated from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--debug", default=None,
                    help="path for the measurement figure")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--model", default="minimax-h3")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--route", default=None, choices=(None, "gpt", "sim"))
    ap.add_argument("--sample-id", default=None)
    ap.add_argument("--task", default=TASK_ID)
    args = ap.parse_args()

    sample_id = args.sample_id or Path(args.video).stem
    route, seed = args.route, args.seed
    known = _from_manifest(sample_id)
    route = route or known.get("route") or (
        "sim" if "sim" in Path(args.image).name else "gpt")
    if seed is None:
        seed = known.get("seed")
    if seed is None:
        m = re.search(r"(\d+)(?!.*\d)", Path(args.video).stem)
        seed = int(m.group(1)) if m else None

    if args.prompt:
        prompt_file = Path(args.prompt)
    else:
        alt = TASK_ROOT / "prompts" / "video_simulation.txt"
        prompt_file = (alt if route == "sim" and alt.is_file()
                       else TASK_ROOT / "prompts" / "video.txt")
    prompt = prompt_file.read_text(encoding="utf-8").strip() \
        if prompt_file.is_file() else ""

    ctx = Context(task_id=args.task, video_path=_rel(args.video),
                  image_path=_rel(args.image), video_prompt=prompt,
                  model=args.model, seed=seed, debug_path=args.debug)
    res = get_evaluator(args.task)(read_clip(args.video), ctx)
    if res.debug_image:
        res.debug_image = _rel(res.debug_image)
    out = res.write(args.out, sample_id=sample_id, route=route)
    print(res.summary())
    print(f"wrote {out}")
    if res.debug_image:
        print(f"figure {res.debug_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
