#!/usr/bin/env python3
"""No-video integration test for P6 future continuous scoring paths."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from batch_evaluate import rebuild_continuous_batch_outputs
from evaluate import apply_continuous_scoring


HERE = Path(__file__).resolve().parent
FORMAL_ROOT = HERE / "results_v1"
CONTINUOUS_CONFIG = HERE / "continuous_score_config_v1.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_legacy_input(current: dict[str, object]) -> dict[str, object]:
    """Recreate an evaluator-v1 result using the preserved frozen score."""
    legacy = copy.deepcopy(current)
    preserved = copy.deepcopy(legacy["legacy_score"]["scores"])
    for key in (
        "legacy_score",
        "score_version",
        "status",
        "measurement_valid",
        "score_components",
        "continuous_scoring",
    ):
        legacy.pop(key, None)
    legacy["scores"] = preserved
    return legacy


def main() -> None:
    result_paths = sorted(FORMAL_ROOT.glob("P6_*_seed*/result.json"))
    assert len(result_paths) == 24
    hashes_before = {path: sha256(path) for path in result_paths}
    current = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    valid_current = next(item for item in current if item["measurement_valid"])
    invalid_current = next(item for item in current if not item["measurement_valid"])

    synthetic = []
    for original in (valid_current, invalid_current):
        legacy_input = as_legacy_input(original)
        labels_before = (
            legacy_input.get("extract_success"),
            legacy_input.get("structural_ok"),
            legacy_input.get("physics_pass"),
        )
        old_scores = copy.deepcopy(legacy_input["scores"])
        scored = apply_continuous_scoring(legacy_input, CONTINUOUS_CONFIG)
        assert scored["score_version"] == "continuous-0-1-v1"
        assert labels_before == (
            scored.get("extract_success"),
            scored.get("structural_ok"),
            scored.get("physics_pass"),
        )
        assert scored["legacy_score"]["scores"] == old_scores
        assert all(0.0 <= float(value) <= 1.0 for value in scored["scores"].values())
        if scored["measurement_valid"]:
            assert float(scored["scores"]["overall"]) > 0.0
        else:
            assert float(scored["scores"]["overall"]) == 0.0
        # Idempotence: a batch finalizer may safely see an already-new JSON.
        rescored = apply_continuous_scoring(scored, CONTINUOUS_CONFIG)
        assert rescored["legacy_score"]["scores"] == old_scores
        assert rescored["scores"] == scored["scores"]
        synthetic.append(rescored)

    with tempfile.TemporaryDirectory(prefix="p6-continuous-integration-") as raw:
        root = Path(raw)
        videos = []
        for result in synthetic:
            sample_id = str(result["sample_id"])
            sample_dir = root / sample_id
            sample_dir.mkdir()
            (sample_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            videos.append(Path(sample_id + ".mp4"))
        failures = [{"sample_id": "synthetic_failure", "return_code": 7}]
        rebuilt = rebuild_continuous_batch_outputs(
            videos, root, CONTINUOUS_CONFIG, failures
        )
        assert len(rebuilt) == 2
        rows = list(csv.DictReader((root / "results.csv").open(encoding="utf-8")))
        assert len(rows) == 2
        assert all(0.0 <= float(row["overall"]) <= 1.0 for row in rows)
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        assert summary["score_version"] == "continuous-0-1-v1"
        assert summary["measurement_valid_count"] == 1
        assert summary["measurement_invalid_count"] == 1
        assert summary["evaluator_process_failures"] == failures

    hashes_after = {path: sha256(path) for path in result_paths}
    assert hashes_after == hashes_before
    print(
        json.dumps(
            {
                "single_video_mapping": "pass",
                "batch_auto_finalize": "pass",
                "legacy_score_preserved": "pass",
                "legacy_labels_preserved": "pass",
                "valid_overall_gt_zero": "pass",
                "invalid_overall_eq_zero": "pass",
                "formal_result_files_unchanged": 24,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
