#!/usr/bin/env python3
"""Write the completed, manually curated 24/24 P3 measurement sanity review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "sample_id", "seed", "first_frame_method", "overlay_reviewed", "plot_reviewed",
    "lane_assignment_ok", "identity_ok", "release_event_ok", "apex_event_ok",
    "first_landing_event_ok", "camera_gate_ok", "raw_track_alignment_ok",
    "overlay_coordinate_alignment_ok", "trajectory_fit_reliable",
    "reviewer_decision", "recommended_for_physics_interpretation", "review_notes",
    "machine_extract_success", "machine_structural_ok", "machine_measurement_valid",
    "machine_physics_pass", "machine_hard_fail_reasons",
]

EXPECTED_CONFIG_SHA256 = "b2ad0caa2de00568f293932436b82140e5bd62282f96c9b439ca8835dc39280e"
EXPECTED_MANIFEST_CREATED_UNIX = 1787755522.0933685


EXCEPTIONS: dict[str, dict[str, Any]] = {
    "P3_gpt_02_classic_seed45": {
        "overlay_coordinate_alignment_ok": False,
        "trajectory_fit_reliable": False,
        "reviewer_decision": "limited",
        "recommended_for_physics_interpretation": False,
        "review_notes": (
            "Camera translation is visually obvious and correctly hard-gated by the machine "
            "(max 270.55 px; CAMERA_TRANSLATION). Raw target tracking and event ordering are "
            "plausible, but the debug overlay draws camera-compensated fused coordinates on raw "
            "frames, so compensated trajectory/fit alignment cannot be manually validated from it."
        ),
    },
    "P3_gpt_03_industrial_seed45": {
        "identity_ok": False,
        "apex_event_ok": False,
        "first_landing_event_ok": False,
        "raw_track_alignment_ok": False,
        "overlay_coordinate_alignment_ok": False,
        "trajectory_fit_reliable": False,
        "reviewer_decision": "fail",
        "recommended_for_physics_interpretation": False,
        "review_notes": (
            "The two visually identical balls merge/overlap for a prolonged interval around "
            "frames 38-70, spanning both reported apices. Independent identity is not observable, "
            "so downstream per-ball apices, landings, and fits cannot be verified despite backend "
            "agreement and machine measurement_valid=true."
        ),
    },
    "P3_sim_02_seed44": {
        "first_landing_event_ok": False,
        "trajectory_fit_reliable": False,
        "reviewer_decision": "fail",
        "recommended_for_physics_interpretation": False,
        "review_notes": (
            "Upper/30-degree ball makes first contact and rebounds well before the reported "
            "landing_frame=108. The extractor selected a later launch-height crossing, so repeated "
            "post-contact rebounds enter the nominal free-flight fit. Lower/60-degree landing_frame=64 "
            "appears to be first contact."
        ),
    },
}


def write_review(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or manifest.get("created_unix") != EXPECTED_MANIFEST_CREATED_UNIX
    ):
        raise ValueError(
            "This curated manual review belongs only to the P3 formal_v1 batch "
            f"({EXPECTED_CONFIG_SHA256}, {EXPECTED_MANIFEST_CREATED_UNIX}); refusing a different batch."
        )
    json_paths = sorted((root / "json").glob("*.json"))
    if len(json_paths) != 24:
        raise ValueError(f"expected exactly 24 result JSON files, got {len(json_paths)}")
    rows: list[dict[str, Any]] = []
    for path in json_paths:
        machine = json.loads(path.read_text(encoding="utf-8"))
        row: dict[str, Any] = {
            "sample_id": machine["sample_id"], "seed": machine["seed"],
            "first_frame_method": machine["first_frame_method"],
            "overlay_reviewed": True, "plot_reviewed": True,
            "lane_assignment_ok": True, "identity_ok": True,
            "release_event_ok": True, "apex_event_ok": True,
            "first_landing_event_ok": True, "camera_gate_ok": True,
            "raw_track_alignment_ok": True, "overlay_coordinate_alignment_ok": True,
            "trajectory_fit_reliable": True, "reviewer_decision": "pass",
            "recommended_for_physics_interpretation": True,
            "review_notes": (
                "Manual overlay and plot review found stable frozen lane identity, plausible release, "
                "apex and first landing events, correct camera gate, and a fit interval restricted to "
                "the visually observed free flight."
            ),
            "machine_extract_success": machine.get("extract_success"),
            "machine_structural_ok": machine.get("structural_ok"),
            "machine_measurement_valid": machine.get("measurement_valid"),
            "machine_physics_pass": machine.get("physics_pass"),
            "machine_hard_fail_reasons": machine.get("hard_fail_reasons") or [],
        }
        row_id = f"{machine['sample_id']}_seed{machine['seed']}"
        row.update(EXCEPTIONS.get(row_id, {}))
        rows.append(row)

    decisions = Counter(row["reviewer_decision"] for row in rows)
    report = {
        "schema_version": "1.0",
        "task_id": "P3",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formal_batch_fingerprint": {
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "manifest_created_unix": EXPECTED_MANIFEST_CREATED_UNIX,
        },
        "review_type": "manual_human_measurement_sanity_review",
        "automated_llm_or_vlm_scoring_used": False,
        "scope": (
            "Judge whether tracker identity, release/apex/first-landing events, camera gate and fitted "
            "free-flight interval are measurable/reliable. This review does not rescore physics and "
            "does not override machine physics_pass or dimension scores."
        ),
        "artifacts_reviewed": ["24 overlay.mp4 videos", "24 plot.png diagnostic plots"],
        "reviewed_count": len(rows),
        "decision_counts": dict(sorted(decisions.items())),
        "reliable_for_physics_interpretation_count": sum(
            bool(row["recommended_for_physics_interpretation"]) for row in rows
        ),
        "machine_physics_conclusions_overridden": False,
        "rows": rows,
    }
    json_path = root / "human_sanity_review.json"
    csv_path = root / "human_sanity_review.csv"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            row["machine_hard_fail_reasons"] = ";".join(row["machine_hard_fail_reasons"])
            writer.writerow(row)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    report = write_review(args.root)
    print(json.dumps({
        "reviewed_count": report["reviewed_count"],
        "decision_counts": report["decision_counts"],
        "reliable_for_physics_interpretation_count": report["reliable_for_physics_interpretation_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
