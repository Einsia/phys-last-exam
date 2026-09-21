#!/usr/bin/env python3
"""Audit completeness and internal consistency of a P3 formal result bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EXPECTED_DEBUG_FILES = {
    "overlay.mp4", "plot.png", "tracks_raw.npz", "tracks_raw.meta.json",
    "upper_fusion_source.npy", "lower_fusion_source.npy",
}
EXPECTED_TRACK_KEYS = {
    f"{backend}_{slot}_{field}"
    for backend in ("sam2", "cotracker")
    for slot in ("upper", "lower")
    for field in ("xy", "radius", "score")
} | {"camera_shift", "meta_json"}
STATUS_FIELDS = ("extract_success", "structural_ok", "measurement_valid", "physics_pass")


def identity(result: dict[str, Any]) -> str:
    return f"{result.get('sample_id')}_seed{result.get('seed')}"


def nonfinite_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(nonfinite_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(nonfinite_paths(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(path)
    return found


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(root: Path, verify_video_hash: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path, summary_path, csv_path = (
        root / "run_manifest.json", root / "summary.json", root / "results.csv"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    json_paths = sorted((root / "json").glob("*.json"))
    debug_dirs = sorted(path for path in (root / "debug").glob("*") if path.is_dir())
    log_paths = sorted((root / "logs").glob("*.log"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in json_paths]
    result_ids = [identity(result) for result in results]
    expected_ids = [Path(video).stem for video in manifest.get("videos", [])]

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_ids = [f"{row['sample_id']}_seed{row['seed']}" for row in csv_rows]

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    config_hashes: set[str] = set()
    video_hashes: list[str] = []
    original_video_checks = {"available": 0, "missing": 0, "hash_verified": 0}

    def add_error(sample_id: str | None, check: str, detail: Any) -> None:
        errors.append({"sample_id": sample_id, "check": check, "detail": detail})

    for path, result in zip(json_paths, results):
        sample_id = identity(result)
        expected_frames = int(result.get("qc", {}).get("video", {}).get("n_frames", -1))
        item_errors_before = len(errors)
        config_hashes.add(str(result.get("config_sha256")))
        video_hash = str(result.get("video_sha256"))
        video_hashes.append(video_hash)
        if path.stem != sample_id:
            add_error(sample_id, "json_filename_identity", {"file": path.stem, "json": sample_id})
        if Path(str(result.get("video"))).stem != sample_id:
            add_error(sample_id, "video_path_identity", result.get("video"))
        if not re.fullmatch(r"[0-9a-f]{64}", video_hash):
            add_error(sample_id, "video_sha256_format", video_hash)
        nonfinite = nonfinite_paths(result)
        if nonfinite:
            add_error(sample_id, "json_nonfinite_values", nonfinite)

        video_path = Path(str(result.get("video")))
        if video_path.exists():
            original_video_checks["available"] += 1
            if video_path.stat().st_size <= 0:
                add_error(sample_id, "original_video_empty", str(video_path))
            if verify_video_hash:
                actual_hash = sha256_file(video_path)
                original_video_checks["hash_verified"] += 1
                if actual_hash != video_hash:
                    add_error(sample_id, "original_video_hash", {
                        "expected": video_hash, "actual": actual_hash,
                    })
        else:
            original_video_checks["missing"] += 1

        debug_dir = root / "debug" / sample_id
        if not debug_dir.is_dir():
            add_error(sample_id, "debug_directory_missing", str(debug_dir))
            continue
        present = {child.name for child in debug_dir.iterdir() if child.is_file()}
        if present != EXPECTED_DEBUG_FILES:
            add_error(sample_id, "debug_file_set", {
                "missing": sorted(EXPECTED_DEBUG_FILES - present),
                "unexpected": sorted(present - EXPECTED_DEBUG_FILES),
            })

        overlay = cv2.VideoCapture(str(debug_dir / "overlay.mp4"))
        overlay_opened = bool(overlay.isOpened())
        prop_frames = int(round(overlay.get(cv2.CAP_PROP_FRAME_COUNT))) if overlay_opened else -1
        overlay_fps = float(overlay.get(cv2.CAP_PROP_FPS)) if overlay_opened else 0.0
        overlay_width = int(round(overlay.get(cv2.CAP_PROP_FRAME_WIDTH))) if overlay_opened else -1
        overlay_height = int(round(overlay.get(cv2.CAP_PROP_FRAME_HEIGHT))) if overlay_opened else -1
        decoded_frames = 0
        while overlay_opened:
            ok, _frame = overlay.read()
            if not ok:
                break
            decoded_frames += 1
        overlay.release()
        if not overlay_opened or decoded_frames != expected_frames or prop_frames != expected_frames:
            add_error(sample_id, "overlay_decode_or_frame_count", {
                "opened": overlay_opened, "property_frames": prop_frames,
                "decoded_frames": decoded_frames, "expected_frames": expected_frames,
            })
        if overlay_fps <= 0.0 or overlay_width <= 0 or overlay_height <= 0:
            add_error(sample_id, "overlay_metadata", {
                "fps": overlay_fps, "width": overlay_width, "height": overlay_height,
            })

        plot = cv2.imread(str(debug_dir / "plot.png"), cv2.IMREAD_UNCHANGED)
        if plot is None or plot.size == 0:
            add_error(sample_id, "plot_decode", str(debug_dir / "plot.png"))
            plot_shape = None
        else:
            plot_shape = list(plot.shape)

        try:
            with np.load(debug_dir / "tracks_raw.npz", allow_pickle=False) as tracks:
                if set(tracks.files) != EXPECTED_TRACK_KEYS:
                    add_error(sample_id, "track_npz_keys", {
                        "missing": sorted(EXPECTED_TRACK_KEYS - set(tracks.files)),
                        "unexpected": sorted(set(tracks.files) - EXPECTED_TRACK_KEYS),
                    })
                lengths = {
                    key: int(tracks[key].shape[0])
                    for key in tracks.files if key != "meta_json" and tracks[key].ndim >= 1
                }
                wrong_lengths = {key: value for key, value in lengths.items()
                                 if value != expected_frames}
                if wrong_lengths:
                    add_error(sample_id, "track_npz_lengths", wrong_lengths)
        except Exception as exc:  # audit must report corruption, not crash
            add_error(sample_id, "track_npz_decode", repr(exc))

        fusion_lengths: dict[str, int] = {}
        for slot in ("upper", "lower"):
            try:
                source = np.load(debug_dir / f"{slot}_fusion_source.npy", allow_pickle=False)
                fusion_lengths[slot] = int(source.shape[0])
                if source.ndim != 1 or source.shape[0] != expected_frames:
                    add_error(sample_id, f"{slot}_fusion_source_shape", list(source.shape))
            except Exception as exc:
                add_error(sample_id, f"{slot}_fusion_source_decode", repr(exc))

        try:
            raw_meta = json.loads((debug_dir / "tracks_raw.meta.json").read_text(encoding="utf-8"))
            for key in ("video_sha256", "config_sha256", "evaluator_version"):
                if raw_meta.get(key) != result.get(key):
                    add_error(sample_id, f"raw_meta_{key}", {
                        "meta": raw_meta.get(key), "result": result.get(key),
                    })
        except Exception as exc:
            add_error(sample_id, "raw_meta_decode", repr(exc))

        per_sample.append({
            "sample_id": sample_id,
            "artifact_checks_pass": len(errors) == item_errors_before,
            "expected_frames": expected_frames,
            "overlay": {"decoded_frames": decoded_frames, "property_frames": prop_frames,
                        "fps": overlay_fps, "width": overlay_width, "height": overlay_height},
            "plot_shape": plot_shape,
            "fusion_lengths": fusion_lengths,
        })

    def exact_set_check(name: str, actual: list[str], expected: list[str]) -> None:
        if len(actual) != len(set(actual)):
            add_error(None, f"{name}_duplicate_identity", [key for key, count in Counter(actual).items() if count > 1])
        if set(actual) != set(expected):
            add_error(None, f"{name}_identity_set", {
                "missing": sorted(set(expected) - set(actual)),
                "unexpected": sorted(set(actual) - set(expected)),
            })

    exact_set_check("json", result_ids, expected_ids)
    exact_set_check("results_csv", csv_ids, expected_ids)
    exact_set_check("debug", [path.name for path in debug_dirs], expected_ids)
    exact_set_check("logs", [path.stem for path in log_paths], expected_ids)
    if any(path.stat().st_size <= 0 for path in log_paths):
        add_error(None, "empty_log", [path.name for path in log_paths if path.stat().st_size <= 0])

    if manifest.get("video_count") != len(expected_ids) or len(expected_ids) != 24:
        add_error(None, "manifest_count", {
            "declared": manifest.get("video_count"), "listed": len(expected_ids), "expected": 24,
        })
    if len(config_hashes) != 1 or manifest.get("config_sha256") not in config_hashes:
        add_error(None, "config_hash_consistency", {
            "result_hashes": sorted(config_hashes), "manifest": manifest.get("config_sha256"),
        })
    if len(set(video_hashes)) != len(video_hashes):
        add_error(None, "duplicate_video_hash", [
            value for value, count in Counter(video_hashes).items() if count > 1
        ])

    recomputed_status = {
        field: sum(bool(result.get(field)) for result in results) for field in STATUS_FIELDS
    }
    expected_summary_status = {
        "extract_success": summary.get("extract_success_count"),
        "structural_ok": summary.get("structural_ok_count"),
        "measurement_valid": summary.get("measurement_valid_count"),
        "physics_pass": summary.get("physics_pass_count"),
    }
    if recomputed_status != expected_summary_status:
        add_error(None, "summary_status_counts", {
            "recomputed": recomputed_status, "summary": expected_summary_status,
        })
    if summary.get("result_count") != len(results):
        add_error(None, "summary_result_count", {
            "summary": summary.get("result_count"), "actual": len(results),
        })

    if original_video_checks["missing"]:
        warnings.append({
            "check": "original_videos_not_in_review_bundle",
            "detail": "Run on the source server to verify video existence/hash; result/debug artifacts remain audited.",
            "counts": original_video_checks,
        })

    return {
        "schema_version": "1.0",
        "task_id": "P3",
        "audit_scope": "formal result/debug artifact completeness and internal consistency",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "expected_sample_count": 24,
        "counts": {
            "manifest_videos": len(expected_ids), "result_json": len(json_paths),
            "results_csv_rows": len(csv_rows), "debug_directories": len(debug_dirs),
            "debug_files": sum(len(list(path.glob("*"))) for path in debug_dirs),
            "logs": len(log_paths), "unique_video_hashes": len(set(video_hashes)),
        },
        "status_counts_recomputed": recomputed_status,
        "original_video_checks": original_video_checks,
        "bundle_integrity_pass": not errors,
        "error_count": len(errors), "errors": errors,
        "warning_count": len(warnings), "warnings": warnings,
        "per_sample": per_sample,
        "known_semantic_debug_limitations": [
            {
                "id": "COMPENSATED_OVERLAY_COORDINATE_MISMATCH",
                "severity": "high for moving-camera samples",
                "detail": "Fused tracks are camera-compensated but the overlay is drawn on raw frames. Static-camera overlays are effectively aligned; moving-camera overlays are not suitable for fit verification.",
            },
            {
                "id": "LATE_CROSSING_BEFORE_FIRST_CONTACT_FALLBACK",
                "severity": "high",
                "detail": "The landing extractor searches any later launch-height crossing before the local-contact fallback, so a post-bounce crossing can be selected instead of first contact.",
            },
            {
                "id": "NO_PROLONGED_OVERLAP_IDENTITY_GATE",
                "severity": "high",
                "detail": "Backend agreement alone does not reject two targets that visually merge/overlap for a prolonged interval; independent identity may be unobservable.",
            },
            {
                "id": "DERIVATIVE_EDGE_SPIKES_UNSHADED",
                "severity": "display only",
                "detail": "Derivative plots include Savitzky-Golay boundary spikes although edge samples are excluded from numerical metrics; the excluded window is not shaded.",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-video-hash", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, args.verify_video_hash)
    output = args.output or args.root / "artifact_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output), "pass": report["bundle_integrity_pass"],
        "errors": report["error_count"], "warnings": report["warning_count"],
        "counts": report["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
