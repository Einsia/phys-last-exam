#!/usr/bin/env python3
"""Run the frozen P6 evaluator over all 24 formal videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from rescore_continuous import build_outputs, rescore_result, sha256


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEOS = SCRIPT_DIR.parent / "minimax_h3" / "prompt_v1_20260825" / "videos"
DEFAULT_CONTINUOUS_CONFIG = SCRIPT_DIR / "continuous_score_config_v1.yaml"


def evaluate_one(
    video: Path,
    output_root: Path,
    config: Path,
    continuous_config: Path,
) -> tuple[str, int, str]:
    sample_dir = output_root / video.stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "evaluate.py"),
        "--video",
        str(video),
        "--task_id",
        "P6",
        "--output",
        str(sample_dir / "result.json"),
        "--config",
        str(config),
        "--continuous-score-config",
        str(continuous_config),
        "--artifacts-dir",
        str(sample_dir),
    ]
    process = subprocess.run(command, text=True, capture_output=True)
    log = process.stdout + process.stderr
    (sample_dir / "evaluator.log").write_text(log, encoding="utf-8")
    return video.stem, process.returncode, log


def rebuild_continuous_batch_outputs(
    videos: list[Path],
    output_root: Path,
    continuous_config_path: Path,
    failures: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Idempotently enforce 0-1 JSONs and rebuild current CSV/summary.

    ``evaluate.py`` already emits continuous scores. Reapplying the mapper is
    deliberate: it also upgrades result JSONs left by interrupted/older batch
    workers, while ``preserve_legacy`` prevents double-wrapping.
    """
    continuous_config_path = continuous_config_path.resolve()
    continuous_config = yaml.safe_load(
        continuous_config_path.read_text(encoding="utf-8")
    )
    config_hash = sha256(continuous_config_path)
    results: list[dict[str, object]] = []
    for video in videos:
        result_path = output_root / video.stem / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result = rescore_result(result, continuous_config, config_hash)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    if results:
        build_outputs(
            output_root,
            results,
            continuous_config["schema_version"],
            evaluator_process_failures=failures or [],
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results_v1")
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config_v1.yaml")
    parser.add_argument(
        "--continuous-score-config",
        type=Path,
        default=DEFAULT_CONTINUOUS_CONFIG,
    )
    parser.add_argument("--pattern", default="P6_*_seed*.mp4")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    videos = sorted(args.videos.glob(args.pattern))
    if args.limit:
        videos = videos[: args.limit]
    if args.limit is None and args.pattern == "P6_*_seed*.mp4" and len(videos) != 24:
        raise SystemExit(f"expected exactly 24 formal P6 videos, found {len(videos)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                evaluate_one,
                video,
                args.output_root,
                args.config,
                args.continuous_score_config,
            ): video
            for video in videos
        }
        for future in as_completed(futures):
            sample_id, return_code, log = future.result()
            print(f"[{sample_id}] return_code={return_code}")
            if return_code:
                failures.append(
                    {
                        "sample_id": sample_id,
                        "return_code": return_code,
                        "log_tail": log[-2000:],
                    }
                )

    results = rebuild_continuous_batch_outputs(
        videos,
        args.output_root,
        args.continuous_score_config,
        failures,
    )
    summary_path = args.output_root / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {
            "formal_video_count": len(videos),
            "result_count": 0,
            "evaluator_process_failures": failures,
        }
    )
    # Requested/observed counts remain useful for limited smoke batches and
    # are separate from score statistics built from available result JSONs.
    summary["requested_video_count"] = len(videos)
    summary["result_count"] = len(results)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
