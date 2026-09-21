#!/usr/bin/env python3
"""Run and aggregate the frozen P3 evaluator without selecting videos or seeds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(data: dict, *keys, default=None):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def aggregate(results: list[dict], failures: list[dict], output_root: Path,
              manifest: dict) -> None:
    rows = []
    failure_codes: Counter[str] = Counter()
    for result in sorted(results, key=lambda item: (item["sample_id"], item.get("seed", -1))):
        for code in result.get("hard_fail_reasons", []):
            failure_codes[code] += 1
        rows.append({
            "task_id": result["task_id"],
            "sample_id": result["sample_id"],
            "seed": result.get("seed"),
            "first_frame_method": result.get("first_frame_method"),
            "video": result.get("video"),
            "extract_success": result.get("extract_success"),
            "structural_ok": result.get("structural_ok"),
            "measurement_valid": result.get("measurement_valid"),
            "M1_valid": nested(result, "metric_validity", "M1"),
            "M2_valid": nested(result, "metric_validity", "M2"),
            "physics_pass": result.get("physics_pass"),
            "score_end_to_end": nested(result, "scores", "end_to_end"),
            "score_conditional": nested(result, "scores", "overall_conditional"),
            "score_task": nested(result, "scores", "task"),
            "score_parabola": nested(result, "scores", "parabola"),
            "score_horizontal": nested(result, "scores", "horizontal"),
            "score_vertical": nested(result, "scores", "vertical"),
            "score_pair_consistency": nested(result, "scores", "pair_consistency"),
            "M1_abs_range_error": nested(result, "metrics", "M1_abs"),
            "initial_speed_error": nested(result, "metrics", "initial_speed_error"),
            "gravity_symmetric_error": nested(result, "metrics", "gravity_symmetric_error"),
            "failure_reason": result.get("failure_reason"),
        })

    csv_path = output_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)

    def count_true(field: str) -> int:
        return sum(bool(result.get(field)) for result in results)

    by_method = {}
    for method in sorted({str(result.get("first_frame_method")) for result in results}):
        group = [result for result in results if str(result.get("first_frame_method")) == method]
        by_method[method] = {
            "n": len(group),
            "extract_success": sum(bool(item.get("extract_success")) for item in group),
            "structural_ok": sum(bool(item.get("structural_ok")) for item in group),
            "measurement_valid": sum(bool(item.get("measurement_valid")) for item in group),
            "physics_pass": sum(bool(item.get("physics_pass")) for item in group),
            "mean_end_to_end": (sum(float(nested(item, "scores", "end_to_end", default=0) or 0)
                                       for item in group) / len(group)) if group else None,
        }
    summary = {
        "schema_version": "1.0",
        "task_id": "P3",
        "formal_video_count": manifest["video_count"],
        "result_count": len(results),
        "evaluator_process_failure_count": len(failures),
        "evaluator_process_failures": failures,
        "extract_success_count": count_true("extract_success"),
        "structural_ok_count": count_true("structural_ok"),
        "measurement_valid_count": count_true("measurement_valid"),
        "physics_pass_count": count_true("physics_pass"),
        "mean_end_to_end": (sum(float(nested(item, "scores", "end_to_end", default=0) or 0)
                                    for item in results) / len(results)) if results else None,
        "by_first_frame_method": by_method,
        "hard_failure_counts": dict(sorted(failure_codes.items())),
        "all_formal_videos_accounted_for": len(results) + len(failures) == manifest["video_count"],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--devices", nargs="+", default=["3", "6"])
    parser.add_argument("--expected-count", type=int, default=24)
    parser.add_argument("--force", action="store_true", help="rerun completed JSON records")
    parser.add_argument("--force-tracking", action="store_true")
    args = parser.parse_args()

    videos = sorted(args.videos.resolve().glob("P3_*_seed*.mp4"))
    if len(videos) != args.expected_count:
        raise RuntimeError(f"expected exactly {args.expected_count} formal videos, found {len(videos)}")
    stems = [video.stem for video in videos]
    if len(stems) != len(set(stems)):
        raise RuntimeError("duplicate formal video stems")

    output_root = args.output_root.resolve()
    json_dir = output_root / "json"
    debug_root = output_root / "debug"
    log_dir = output_root / "logs"
    for path in (json_dir, debug_root, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_id": "P3",
        "created_unix": time.time(),
        "video_count": len(videos),
        "videos": [str(path) for path in videos],
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config.resolve()),
        "evaluator_sha256": sha256(HERE / "evaluate.py"),
        "devices": list(args.devices),
        "selection_policy": "all matching formal videos; no filtering or best-of-seed",
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    devices: queue.Queue[str] = queue.Queue()
    for device in args.devices:
        devices.put(str(device))

    def run_one(video: Path) -> tuple[Path, dict | None]:
        result_path = json_dir / f"{video.stem}.json"
        overlay = debug_root / video.stem / "overlay.mp4"
        plot = debug_root / video.stem / "plot.png"
        if not args.force and result_path.exists() and overlay.exists() and plot.exists():
            return video, load_result(result_path)
        device = devices.get()
        try:
            command = [
                sys.executable, str(HERE / "evaluate.py"),
                "--video", str(video), "--task_id", "P3",
                "--output", str(result_path), "--config", str(args.config.resolve()),
                "--debug-dir", str(debug_root / video.stem), "--device", device,
            ]
            if args.force_tracking:
                command.append("--force-tracking")
            process = subprocess.run(command, text=True, capture_output=True)
            (log_dir / f"{video.stem}.log").write_text(
                process.stdout + "\n--- STDERR ---\n" + process.stderr,
                encoding="utf-8",
            )
            if process.returncode != 0 or not result_path.exists():
                raise RuntimeError(f"returncode={process.returncode}; see {log_dir / (video.stem + '.log')}")
            return video, load_result(result_path)
        finally:
            devices.put(device)

    results: list[dict] = []
    failures: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        future_map = {executor.submit(run_one, video): video for video in videos}
        for index, future in enumerate(as_completed(future_map), start=1):
            video = future_map[future]
            try:
                _, result = future.result()
                assert result is not None
                results.append(result)
                print(json.dumps({
                    "done": index, "total": len(videos), "video": video.name,
                    "extract_success": result.get("extract_success"),
                    "structural_ok": result.get("structural_ok"),
                    "measurement_valid": result.get("measurement_valid"),
                    "physics_pass": result.get("physics_pass"),
                }, ensure_ascii=False), flush=True)
            except Exception as error:
                failure = {"video": str(video), "error": str(error)}
                failures.append(failure)
                print(json.dumps({"done": index, "total": len(videos), **failure},
                                 ensure_ascii=False), flush=True)
            aggregate(results, failures, output_root, manifest)

    aggregate(results, failures, output_root, manifest)
    return 0 if not failures and len(results) == len(videos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
