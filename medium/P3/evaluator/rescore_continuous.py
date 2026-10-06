#!/usr/bin/env python3
"""Non-destructively rescore a completed P3 result bundle.

The source JSON files are treated as immutable legacy records.  The output keeps
the original 0--100 scores under ``scores_legacy`` and replaces ``scores`` with
the continuous-0-1-v1 profile.  Status fields and ``physics_pass`` are copied
verbatim; only ``measurement_valid`` gates the new overall score to zero.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROFILE = "continuous-0-1-v1"
SCORE_FLOOR = 0.01


def load_score_config(path: Path) -> dict[str, Any]:
    """Load the two scoring sections, with no mandatory third-party dependency."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None
    if yaml is not None:
        loaded = yaml.safe_load(text)
        return {
            "score_weights": loaded["score_weights"],
            "thresholds": loaded["thresholds"],
        }

    # The evaluator configuration deliberately uses a small, versioned YAML
    # subset for these sections.  This fallback keeps the rescorer portable on
    # review machines without PyYAML; it fails closed on unfamiliar syntax.
    sections: dict[str, dict[str, Any]] = {"score_weights": {}, "thresholds": {}}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key = line[:-1] if line.endswith(":") else None
            current = key if key in sections else None
            continue
        if current is None or not line.startswith("  ") or line.startswith("    "):
            continue
        key, separator, raw_value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"unrecognized scoring config line: {raw_line!r}")
        raw_value = raw_value.strip()
        if current == "score_weights":
            sections[current][key] = float(raw_value)
        else:
            if not (raw_value.startswith("{") and raw_value.endswith("}")):
                raise ValueError(f"threshold must be an inline mapping: {raw_line!r}")
            entries: dict[str, float] = {}
            for item in raw_value[1:-1].split(","):
                subkey, subseparator, subvalue = item.strip().partition(":")
                if not subseparator:
                    raise ValueError(f"invalid threshold entry: {item!r}")
                entries[subkey.strip()] = float(subvalue.strip())
            sections[current][key] = entries
    if not sections["score_weights"] or not sections["thresholds"]:
        raise ValueError(f"could not parse scoring sections from {path}")
    return sections


def residual_quality(residual: Any, anchor: float) -> float:
    """Return q(r,s)=1/(1+|r|/s), or zero for unusable input."""
    try:
        value = abs(float(residual))
        scale = float(anchor)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
        return 0.0
    return float(1.0 / (1.0 + value / scale))


def count_quality(observed: Any, required: Any) -> float:
    """Continuous count evidence score required by continuous-0-1-v1."""
    try:
        obs, req = float(observed), float(required)
    except (TypeError, ValueError):
        return SCORE_FLOOR
    if not math.isfinite(obs) or not math.isfinite(req) or req <= 0.0:
        return SCORE_FLOOR
    return float(max(SCORE_FLOOR, min(max(obs, 0.0) / req, 1.0)))


def weighted_geometric_mean(
    values: Sequence[Any], weights: Sequence[float], floor: float = SCORE_FLOOR
) -> float:
    """Weighted GM with every input clamped to [floor, 1]."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must be non-empty and equal length")
    clean_weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in clean_weights):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = sum(clean_weights)
    if weight_sum <= 0.0:
        raise ValueError("at least one weight must be positive")
    clean_values: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        clean_values.append(max(floor, min(numeric, 1.0)))
    log_mean = sum(
        weight * math.log(value)
        for value, weight in zip(clean_values, clean_weights)
    ) / weight_sum
    return float(max(floor, min(math.exp(log_mean), 1.0)))


def _get(mapping: Mapping[str, Any], *path: str) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _bad_anchor(config: Mapping[str, Any], key: str) -> float:
    value = _get(config, "thresholds", key, "bad")
    anchor = float(value)
    if not math.isfinite(anchor) or anchor <= 0.0:
        raise ValueError(f"invalid bad/half-quality anchor for {key}: {value!r}")
    return anchor


def _metric_quality(value: Any, config: Mapping[str, Any], threshold_key: str) -> float:
    return residual_quality(value, _bad_anchor(config, threshold_key))


def score_continuous(
    result: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Compute all five dimensions and the overall continuous score."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    qc = result.get("qc") if isinstance(result.get("qc"), Mapping) else {}
    measurements = (
        result.get("measurements")
        if isinstance(result.get("measurements"), Mapping)
        else {}
    )

    integrity_values: list[float] = []
    integrity_labels: list[str] = []
    per_ball_metrics = _get(metrics, "M2", "per_ball") or {}
    ball_measurements = measurements.get("balls") or {}
    qc_balls = qc.get("balls") or {}
    for slot in ("upper", "lower"):
        radius_cv = _get(ball_measurements, slot, "radius_robust_cv")
        disagreement = _get(qc_balls, slot, "backend_disagreement_median_radii")
        coverage = _get(qc_balls, slot, "fused_coverage")
        try:
            coverage_error = abs(1.0 - float(coverage))
        except (TypeError, ValueError):
            coverage_error = None
        integrity_values.extend(
            [
                residual_quality(radius_cv, 0.45),
                residual_quality(disagreement, 0.75),
                residual_quality(coverage_error, 0.15),
            ]
        )
        integrity_labels.extend(
            [f"{slot}.radius_cv", f"{slot}.backend_disagreement", f"{slot}.coverage_error"]
        )
    integrity = weighted_geometric_mean(integrity_values, [1.0] * len(integrity_values))

    task_components = {
        "release_sync": _metric_quality(metrics.get("release_sync_frames"), config, "sync_frames"),
        "launch_angle": _metric_quality(metrics.get("max_angle_error_deg"), config, "angle_error_deg"),
        "landing_level": _metric_quality(
            metrics.get("max_landing_level_error_d"), config, "landing_level_d"
        ),
        "camera_translation": _metric_quality(
            _get(qc, "camera", "max_translation_diag"), config, "camera_translation_diag"
        ),
        "track_integrity": integrity,
    }
    task = weighted_geometric_mean(
        list(task_components.values()), [0.25, 0.25, 0.20, 0.15, 0.15]
    )

    per_ball_details: dict[str, Any] = {}
    parabola_balls: list[float] = []
    horizontal_balls: list[float] = []
    vertical_balls: list[float] = []
    for slot in ("upper", "lower"):
        values = per_ball_metrics.get(slot) if isinstance(per_ball_metrics, Mapping) else {}
        if not isinstance(values, Mapping):
            values = {}
        para_components = {
            "rmse": _metric_quality(values.get("parabola_rmse_d"), config, "parabola_rmse_d"),
            "p95": _metric_quality(values.get("parabola_p95_d"), config, "parabola_p95_d"),
        }
        horizontal_components = {
            "x_fit_rmse": _metric_quality(values.get("x_fit_rmse_d"), config, "x_fit_rmse_d"),
            "vx_robust_cv": _metric_quality(values.get("vx_robust_cv"), config, "vx_robust_cv"),
        }
        vertical_components = {
            "y_fit_rmse": _metric_quality(values.get("y_fit_rmse_d"), config, "y_fit_rmse_d"),
            "vy_line_residual": _metric_quality(
                values.get("vy_line_residual"), config, "vy_line_residual"
            ),
            "ay_robust_cv": _metric_quality(values.get("ay_robust_cv"), config, "ay_robust_cv"),
        }
        para_score = weighted_geometric_mean(list(para_components.values()), [0.65, 0.35])
        horizontal_score = weighted_geometric_mean(
            list(horizontal_components.values()), [0.60, 0.40]
        )
        vertical_score = weighted_geometric_mean(
            list(vertical_components.values()), [0.65, 0.20, 0.15]
        )
        parabola_balls.append(para_score)
        horizontal_balls.append(horizontal_score)
        vertical_balls.append(vertical_score)
        per_ball_details[slot] = {
            "parabola_components": para_components,
            "parabola": para_score,
            "horizontal_components": horizontal_components,
            "horizontal": horizontal_score,
            "vertical_components": vertical_components,
            "vertical": vertical_score,
        }

    # Each ball is an independent observation; neither is fit to or scored against
    # the other, and the dimension aggregate gives them equal weight.
    parabola = weighted_geometric_mean(parabola_balls, [0.5, 0.5])
    horizontal = weighted_geometric_mean(horizontal_balls, [0.5, 0.5])
    vertical = weighted_geometric_mean(vertical_balls, [0.5, 0.5])

    pair_components = {
        "equal_range": _metric_quality(metrics.get("M1_abs"), config, "range_ratio_error"),
        "equal_initial_speed": _metric_quality(
            metrics.get("initial_speed_error"), config, "initial_speed_error"
        ),
        "gravity_symmetry": _metric_quality(
            metrics.get("gravity_symmetric_error"), config, "gravity_error"
        ),
    }
    pair_consistency = weighted_geometric_mean(
        list(pair_components.values()), [0.45, 0.40, 0.15]
    )

    dimensions = {
        "task": task,
        "parabola": parabola,
        "horizontal": horizontal,
        "vertical": vertical,
        "pair_consistency": pair_consistency,
    }
    overall_weights = config.get("score_weights") or {}
    overall_candidate = weighted_geometric_mean(
        list(dimensions.values()),
        [float(overall_weights.get(name, 0.0)) for name in dimensions],
    )
    overall = overall_candidate if bool(result.get("measurement_valid")) else 0.0
    scores = {**dimensions, "overall": float(overall)}
    details = {
        "profile": PROFILE,
        "residual_quality_formula": "q(r,s)=1/(1+abs(r)/s)",
        "anchor_policy": "s is the legacy bad threshold (the q=0.5 half-quality anchor)",
        "count_quality_formula": "max(0.01,min(max(observed,0)/required,1))",
        "aggregation": "weighted geometric mean; every input is clamped to [0.01,1]",
        "measurement_valid_gate": "overall=0 iff measurement_valid=false; hard failures do not zero it",
        "task_components": task_components,
        "track_integrity_components": dict(zip(integrity_labels, integrity_values)),
        "per_ball": per_ball_details,
        "pair_components": pair_components,
        "overall_candidate_before_measurement_gate": overall_candidate,
    }
    return scores, details


def rescore_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    rescored = copy.deepcopy(dict(result))
    if "scores_legacy" in rescored:
        legacy = copy.deepcopy(rescored["scores_legacy"])
    else:
        legacy = copy.deepcopy(rescored.get("scores") or {})
    scores, details = score_continuous(rescored, config)
    rescored["scores_legacy"] = legacy
    rescored["scores"] = scores
    rescored["score_profile"] = PROFILE
    rescored["score_version"] = PROFILE
    rescored["score_details"] = details
    rescored["score_status_semantics"] = {
        "extract_success_preserved": bool(result.get("extract_success")),
        "structural_ok_preserved": bool(result.get("structural_ok")),
        "measurement_valid_preserved": bool(result.get("measurement_valid")),
        "physics_pass_preserved": bool(result.get("physics_pass")),
    }
    for name, value in scores.items():
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"continuous score out of range: {name}={value!r}")
    if bool(result.get("measurement_valid")) and not scores["overall"] > 0.0:
        raise AssertionError("measurement_valid=true must have overall>0")
    if not bool(result.get("measurement_valid")) and scores["overall"] != 0.0:
        raise AssertionError("measurement_valid=false must have overall=0")
    if rescored.get("physics_pass") != result.get("physics_pass"):
        raise AssertionError("physics_pass changed during rescore")
    return rescored


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    clean = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not clean:
        return {"n": 0, "zero_count": 0, "positive_count": 0, "min": None,
                "p26": None, "median": None, "mean": None, "p75": None,
                "max": None, "std_population": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(clean) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return clean[low]
        return clean[low] * (high - position) + clean[high] * (position - low)

    return {
        "n": len(clean),
        "zero_count": sum(value == 0.0 for value in clean),
        "positive_count": sum(value > 0.0 for value in clean),
        "min": clean[0],
        "p26": percentile(0.25),
        "median": statistics.median(clean),
        "mean": statistics.fmean(clean),
        "p75": percentile(0.75),
        "max": clean[-1],
        "std_population": statistics.pstdev(clean),
    }


def build_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_names = ["task", "parabola", "horizontal", "vertical", "pair_consistency", "overall"]
    continuous = {
        name: _distribution(result["scores"][name] for result in results)
        for name in score_names
    }
    legacy_names = [
        "task", "parabola", "horizontal", "vertical", "pair_consistency",
        "overall_conditional", "end_to_end",
    ]
    legacy = {
        name: _distribution(
            (result.get("scores_legacy") or {}).get(name, 0.0) or 0.0
            for result in results
        )
        for name in legacy_names
    }

    by_method: dict[str, Any] = {}
    for method in sorted({str(result.get("first_frame_method")) for result in results}):
        group = [result for result in results if str(result.get("first_frame_method")) == method]
        by_method[method] = {
            "n": len(group),
            "extract_success": sum(bool(item.get("extract_success")) for item in group),
            "structural_ok": sum(bool(item.get("structural_ok")) for item in group),
            "measurement_valid": sum(bool(item.get("measurement_valid")) for item in group),
            "physics_pass": sum(bool(item.get("physics_pass")) for item in group),
            "continuous_overall": _distribution(item["scores"]["overall"] for item in group),
            "legacy_end_to_end": _distribution(
                (item.get("scores_legacy") or {}).get("end_to_end", 0.0) for item in group
            ),
        }
    hard_fail_counts = Counter(
        reason
        for result in results
        for reason in (result.get("hard_fail_reasons") or [])
    )
    valid = [result for result in results if bool(result.get("measurement_valid"))]
    return {
        "schema_version": "1.0",
        "score_profile": PROFILE,
        "score_version": PROFILE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": "P3",
        "result_count": len(results),
        "status_counts": {
            "extract_success": sum(bool(item.get("extract_success")) for item in results),
            "structural_ok": sum(bool(item.get("structural_ok")) for item in results),
            "measurement_valid": len(valid),
            "physics_pass": sum(bool(item.get("physics_pass")) for item in results),
        },
        "score_contract_checks": {
            "all_dimension_and_overall_scores_in_0_1": all(
                0.0 <= float(item["scores"][name]) <= 1.0
                for item in results for name in score_names
            ),
            "measurement_valid_true_overall_positive": all(
                float(item["scores"]["overall"]) > 0.0 for item in valid
            ),
            "measurement_valid_false_overall_zero": all(
                float(item["scores"]["overall"]) == 0.0
                for item in results if not bool(item.get("measurement_valid"))
            ),
            "legacy_scores_present": all(bool(item.get("scores_legacy")) for item in results),
        },
        "continuous_distribution": continuous,
        "legacy_distribution": legacy,
        "by_first_frame_method": by_method,
        "hard_failure_counts": dict(sorted(hard_fail_counts.items())),
    }


def write_csv(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "task_id", "sample_id", "seed", "first_frame_method", "video",
        "extract_success", "structural_ok", "measurement_valid", "physics_pass",
        "score_profile", "score_version", "score_task", "score_parabola", "score_horizontal",
        "score_vertical", "score_pair_consistency", "score_overall",
        "legacy_score_task", "legacy_score_parabola", "legacy_score_horizontal",
        "legacy_score_vertical", "legacy_score_pair_consistency",
        "legacy_score_conditional", "legacy_score_end_to_end",
        "M1_abs_range_error", "initial_speed_error", "gravity_symmetric_error",
        "hard_fail_reasons", "failure_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            scores, legacy = result["scores"], result.get("scores_legacy") or {}
            metrics = result.get("metrics") or {}
            writer.writerow({
                "task_id": result.get("task_id"), "sample_id": result.get("sample_id"),
                "seed": result.get("seed"), "first_frame_method": result.get("first_frame_method"),
                "video": result.get("video"), "extract_success": result.get("extract_success"),
                "structural_ok": result.get("structural_ok"),
                "measurement_valid": result.get("measurement_valid"),
                "physics_pass": result.get("physics_pass"), "score_profile": PROFILE,
                "score_version": PROFILE,
                "score_task": scores["task"], "score_parabola": scores["parabola"],
                "score_horizontal": scores["horizontal"], "score_vertical": scores["vertical"],
                "score_pair_consistency": scores["pair_consistency"],
                "score_overall": scores["overall"], "legacy_score_task": legacy.get("task"),
                "legacy_score_parabola": legacy.get("parabola"),
                "legacy_score_horizontal": legacy.get("horizontal"),
                "legacy_score_vertical": legacy.get("vertical"),
                "legacy_score_pair_consistency": legacy.get("pair_consistency"),
                "legacy_score_conditional": legacy.get("overall_conditional"),
                "legacy_score_end_to_end": legacy.get("end_to_end"),
                "M1_abs_range_error": metrics.get("M1_abs"),
                "initial_speed_error": metrics.get("initial_speed_error"),
                "gravity_symmetric_error": metrics.get("gravity_symmetric_error"),
                "hard_fail_reasons": ";".join(result.get("hard_fail_reasons") or []),
                "failure_reason": result.get("failure_reason"),
            })


def run(source_root: Path, output_root: Path, config_path: Path) -> dict[str, Any]:
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if source_root == output_root:
        raise ValueError("source and output roots must differ; rescoring is non-destructive")
    config = load_score_config(config_path)
    json_paths = sorted((source_root / "json").glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no result JSON files under {source_root / 'json'}")
    (output_root / "json").mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    status_mismatches: dict[str, list[str]] = {
        key: [] for key in (
            "extract_success", "structural_ok", "measurement_valid", "physics_pass",
            "hard_fail_reasons",
        )
    }
    for path in json_paths:
        source_bytes = path.read_bytes()
        source = json.loads(source_bytes)
        rescored = rescore_result(source, config)
        identity = f"{source.get('sample_id')}_seed{source.get('seed')}"
        for key in status_mismatches:
            if rescored.get(key) != source.get(key):
                status_mismatches[key].append(identity)
        rescored["legacy_result_sha256"] = hashlib.sha256(source_bytes).hexdigest()
        rescored["legacy_result_path"] = str(path)
        (output_root / "json" / path.name).write_text(
            json.dumps(rescored, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        results.append(rescored)
    results.sort(key=lambda item: (str(item.get("sample_id")), int(item.get("seed", -1))))
    summary = build_summary(results)
    summary["status_preservation"] = {
        key: {"identical_for_all_24": not mismatches, "mismatch_ids": mismatches}
        for key, mismatches in status_mismatches.items()
    }
    summary["source_result_root"] = str(source_root)
    summary["config_path"] = str(config_path.resolve())
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(output_root / "results.csv", results)
    manifest = source_root / "run_manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, output_root / "run_manifest.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.source_root, args.output_root, args.config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
