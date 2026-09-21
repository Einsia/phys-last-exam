#!/usr/bin/env python3
"""Audit migrated JSON/CSV result invariants without opening any video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from continuous_scoring import SCORING_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    args = parser.parse_args()
    batches = ["prompt_v2_20260826", "prompt_v3_20260826", "prompt_v4_20260826"]
    audit = {}
    for batch in batches:
        root = args.evaluation_root / batch
        json_paths = sorted((root / "json").glob("*.json"))
        assert len(json_paths) == 24, (batch, len(json_paths))
        with (root / "summary.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            rows = {row["video_id"]: row for row in csv.DictReader(handle)}
        assert len(rows) == 24, (batch, len(rows))
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        assert report["score_version"] == SCORING_VERSION
        assert report["score_range"] == [0.0, 1.0]
        assert "legacy_mean_valid_overall_score_0_100" in report
        assert report["mean_valid_overall_score"] == report["continuous_scoring"]["overall"]["mean"]
        valid_count = 0
        positive_count = 0
        for path in json_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            video_id = result["video_id"]
            scores = result["scores"]
            assert result["score_version"] == SCORING_VERSION
            assert scores["scoring_version"] == SCORING_VERSION
            assert "legacy_scores_0_100" in result
            values = list(scores["dimensions"].values()) + [scores["derived"]["physics_pair_quality"], scores["overall"]]
            assert all(0.0 <= float(value) <= 1.0 for value in values), (video_id, values)
            valid = result["statuses"]["measurement_valid"] is True
            valid_count += int(valid)
            positive_count += int(float(scores["overall"]) > 0.0)
            assert (float(scores["overall"]) > 0.0) if valid else (float(scores["overall"]) == 0.0)
            row = rows[video_id]
            assert row["scoring_version"] == SCORING_VERSION
            assert abs(float(row["continuous_overall"]) - float(scores["overall"])) < 1.0e-12
            assert abs(float(row["M1_score"]) - float(row["continuous_M1_snell_quality"])) < 1.0e-12
            assert abs(float(row["M2_score"]) - float(row["continuous_M2_intersection_quality"])) < 1.0e-12
            assert abs(float(row["metric_score"]) - float(row["continuous_physics_pair_quality"])) < 1.0e-12
            assert abs(float(row["overall_gated_score"]) - float(row["continuous_overall"])) < 1.0e-12
            for legacy_name in (
                "legacy_M1_score_0_100",
                "legacy_M2_score_0_100",
                "legacy_metric_score_0_100",
                "legacy_overall_gated_score_0_100",
            ):
                assert row.get(legacy_name) not in (None, ""), (video_id, legacy_name)
        audit[batch] = {
            "json_count": len(json_paths),
            "csv_count": len(rows),
            "measurement_valid_count": valid_count,
            "positive_overall_count": positive_count,
        }
    print(json.dumps({"scoring_version": SCORING_VERSION, "status": "PASS", "batches": audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
