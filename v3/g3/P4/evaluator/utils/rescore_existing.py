#!/usr/bin/env python3
"""Reapply frozen P4 gates to existing coordinate measurements without retracking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import evaluate as ev
from rescore_continuous import score_record as continuous_score_record
from run_all import flatten


HERE = Path(__file__).resolve().parent


def method_seed(stem: str) -> tuple[str | None, int | None]:
    match = re.search(r"_seed(\d+)$", stem)
    seed = int(match.group(1)) if match else None
    method = "gpt" if stem.startswith("P4_gpt_") else "simulation" if stem.startswith("P4_sim_") else None
    return method, seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    args = parser.parse_args()
    cfg_raw = args.config.read_bytes()
    cfg = json.loads(cfg_raw)
    config_hash = hashlib.sha256(cfg_raw).hexdigest()
    records = []
    for path in sorted((args.results / "json").glob("P4_*_seed*.json")):
        result = json.loads(path.read_text())
        method, seed = method_seed(result["sample_id"])
        result["evaluator_version"] = cfg["evaluator_version"]
        result["config_sha256"] = config_hash
        result["first_frame_method"] = method
        result["seed"] = seed
        result["measurement_valid"] = bool(result.get("extract_success"))
        arcs = int(result.get("measurements", {}).get("complete_rebound_arc_count") or 0)
        result["metric_validity"] = {
            "M1": bool(result["measurement_valid"] and arcs >= 2),
            "M2": bool(result["measurement_valid"] and arcs >= 3),
            "M3": False,
        }
        result.setdefault("metrics", {})["M3"] = None
        legacy_scores = dict(result.get("legacy_scores_0_100") or result.get("scores") or {})
        if result.get("structural_ok"):
            soft = float(legacy_scores.get("soft_diagnostic_overall", 0.0))
            candidate_scores = {**legacy_scores, "overall": soft}
            passed, reasons = ev.physics_decision(True, candidate_scores, result["metrics"], cfg)
            result["physics_pass"] = passed
            legacy_scores["overall"] = soft if passed else 0.0
            result["failure_reason"] = "; ".join(reasons) if reasons else None
        result["legacy_scores_0_100"] = legacy_scores
        result["scores"] = legacy_scores
        result = continuous_score_record(result)
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(ev.finite(result), ensure_ascii=False, indent=2))
        temp.replace(path)
        records.append(result)

    if len(records) != 24:
        raise RuntimeError(f"expected 24 existing P4 records, found {len(records)}")
    rows = [flatten(record) for record in records]
    with (args.results / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    failures = Counter(record.get("failure_reason") or "none" for record in records)
    current_values = [float(x.get("scores", {}).get("overall", 0.0)) for x in records]
    valid_values = [float(x.get("scores", {}).get("overall", 0.0)) for x in records
                    if x.get("measurement_valid")]
    legacy_values = [float(x.get("legacy_scores_0_100", {}).get("overall", 0.0))
                     for x in records]
    summary = {
        "task_id": "P4", "n_planned": 24, "n_results": 24,
        "score_version": "continuous-0-1-v1", "score_range": [0.0, 1.0],
        "zero_score_policy": "measurement_invalid_only",
        "evaluator_version": cfg["evaluator_version"],
        "config_sha256": config_hash, "llm_or_vlm_used": False,
        "extract_success_rate": sum(bool(x.get("extract_success")) for x in records) / 24,
        "structural_success_rate": sum(bool(x.get("structural_ok")) for x in records) / 24,
        "measurement_valid_rate": sum(bool(x.get("measurement_valid")) for x in records) / 24,
        "end_to_end_pass_rate": sum(bool(x.get("physics_pass")) for x in records) / 24,
        "end_to_end_mean_score": statistics.fmean(current_values),
        "legacy_end_to_end_mean_score_0_100": statistics.fmean(legacy_values),
        "continuous_score_mean_all": statistics.fmean(current_values),
        "continuous_score_mean_valid": statistics.fmean(valid_values) if valid_values else 0.0,
        "continuous_score_median_valid": statistics.median(valid_values) if valid_values else 0.0,
        "continuous_score_min_valid": min(valid_values) if valid_values else 0.0,
        "continuous_score_max_valid": max(valid_values) if valid_values else 0.0,
        "continuous_score_zero_count": sum(value == 0.0 for value in current_values),
        "failure_counts": dict(sorted(failures.items())),
        "results_csv": str((args.results / "results.csv").resolve()),
        "rescore_policy": "coordinates/events unchanged; p4-bounce-1.0.2 strict monotonic gate reapplied",
    }
    (args.results / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
