#!/usr/bin/env python3
"""Read-only acceptance audit for a Feishu rev153 M1/M2-only delivery.

The formal result layer is intentionally kept separate from the optional
derived ranking layer.  This auditor independently recomputes every derived
score from the two documented metrics and rejects legacy/extra score fields in
the formal JSON and CSV artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TASK_COUNTS = {"P3": 24, "P4": 24, "P6": 24, "P9": 24, "P11": 72}
EXPECTED_RECORDS = sum(TASK_COUNTS.values())
EXPECTED_M1_VALID = 164
EXPECTED_M2_VALID = 163
FORMAL_METRICS = {"M1", "M2"}
TOL = 1e-12

FORMAL_JSON_FORBIDDEN = {
    "M3",
    "m3",
    "metric_scores",
    "overall",
    "overall_score",
    "overall_valid",
    "score_status",
    "dimensions",
    "physics_pass",
    "legacy_score",
    "legacy_dimensions",
    "score_delta_from_v1",
}
FORMAL_CSV_FORBIDDEN_FRAGMENTS = (
    "m3",
    "score",
    "overall",
    "dimension",
    "physics_pass",
    "legacy",
)

P3_M2_KEYS = {
    "trajectory_30deg_parabola_rmse_d",
    "trajectory_60deg_parabola_rmse_d",
    "initial_speed_magnitude_error",
}
P4_M2_KEYS = {
    "height_decrease_checks",
    "height_decrease_fraction",
    "adjacent_restitution_coefficient_cv",
}
P9_M2_KEYS = {
    "short_pendulum_period_cv",
    "long_pendulum_period_cv",
}


class Audit:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.checks: Counter[str] = Counter()

    def check(self, condition: bool, message: str, section: str) -> bool:
        self.checks[section] += 1
        if not condition:
            self.failures.append(f"[{section}] {message}")
            return False
        return True

    def fail(self, message: str, section: str) -> None:
        self.check(False, message, section)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def close(a: Any, b: Any, tol: float = TOL) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not finite(a) or not finite(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def quality(error: Any, scale: float) -> float | None:
    if not finite(error):
        return None
    ratio = abs(float(error)) / float(scale)
    return 1.0 / (1.0 + ratio * ratio)


def geometric(values: Iterable[float | None]) -> float | None:
    items = list(values)
    if not items or any(value is None for value in items):
        return None
    if any(float(value) < 0.0 or float(value) > 1.0 for value in items):
        return None
    if any(float(value) == 0.0 for value in items):
        return 0.0
    # Equal weighting is deliberate: the document defines no component weights.
    return math.exp(sum(math.log(float(value)) for value in items) / len(items))


def derived_scores(task: str, metrics: dict[str, Any], validity: dict[str, Any]) -> dict[str, Any]:
    """Independent implementation-derived ranking; consumes M1/M2 only."""
    m1 = metrics.get("M1")
    m2 = metrics.get("M2")
    if task == "P3":
        m1_score = quality(m1, 0.15)
        m2_score = geometric(
            [
                quality(m2.get("trajectory_30deg_parabola_rmse_d") if isinstance(m2, dict) else None, 0.30),
                quality(m2.get("trajectory_60deg_parabola_rmse_d") if isinstance(m2, dict) else None, 0.30),
                quality(m2.get("initial_speed_magnitude_error") if isinstance(m2, dict) else None, 0.20),
            ]
        )
    elif task == "P4":
        m1_score = quality(m1, 0.18)
        fraction = m2.get("height_decrease_fraction") if isinstance(m2, dict) else None
        fraction_score = float(fraction) if finite(fraction) and 0.0 <= float(fraction) <= 1.0 else None
        restitution_score = quality(
            m2.get("adjacent_restitution_coefficient_cv") if isinstance(m2, dict) else None,
            0.35,
        )
        m2_score = geometric([fraction_score, restitution_score])
    elif task == "P6":
        m1_score, m2_score = quality(m1, 0.25), quality(m2, 0.35)
    elif task == "P9":
        m1_score = quality(m1, 0.20)
        m2_score = geometric(
            [
                quality(m2.get("short_pendulum_period_cv") if isinstance(m2, dict) else None, 0.20),
                quality(m2.get("long_pendulum_period_cv") if isinstance(m2, dict) else None, 0.20),
            ]
        )
    elif task == "P11":
        m1_score, m2_score = quality(m1, 0.12), quality(m2, 0.025)
    else:
        raise ValueError(f"unknown task: {task}")

    # An evaluator may retain a finite diagnostic estimate while explicitly
    # marking the documented metric unmeasurable (for example, identity loss).
    # Such a value must not leak into the ranking layer.
    if validity.get("M1") is not True:
        m1_score = None
    if validity.get("M2") is not True:
        m2_score = None
    m1_score = m1_score if validity.get("M1") else None
    m2_score = m2_score if validity.get("M2") else None
    overall_valid = bool(validity.get("M1")) and bool(validity.get("M2"))
    if overall_valid and m1_score is not None and m2_score is not None:
        overall_score = math.sqrt(float(m1_score) * float(m2_score))
    else:
        overall_score = 0.0
        overall_valid = False
    return {
        "M1_score": m1_score,
        "M2_score": m2_score,
        "overall_score": overall_score,
        "overall_valid": overall_valid,
    }


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def parse_cell(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return float(text)
        except ValueError:
            return text


def deep_equal(a: Any, b: Any) -> bool:
    if finite(a) and finite(b):
        return close(a, b)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(deep_equal(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(deep_equal(x, y) for x, y in zip(a, b))
    return a == b


def safe_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def resolve_root_link(root: Path, value: str) -> Path:
    value = value.strip()
    if value.startswith("./"):
        value = value[2:]
    return (root / value).resolve()


def resolve_gallery_link(root: Path, value: str) -> Path:
    return (root / "video_gallery" / value).resolve()


def load_json(path: Path, audit: Audit, section: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit should collect all failures
        audit.fail(f"cannot parse {path.relative_to(audit.root)}: {exc}", section)
        return None


def validate_metric_shape(task: str, result: dict[str, Any], audit: Audit, label: str) -> None:
    section = "formal-json"
    metrics = result.get("metrics")
    validity = result.get("metric_validity")
    if not audit.check(isinstance(metrics, dict), f"{label}: metrics is not an object", section):
        return
    audit.check(set(metrics) == FORMAL_METRICS, f"{label}: metrics keys={sorted(metrics)}", section)
    if not audit.check(isinstance(validity, dict), f"{label}: metric_validity is not an object", section):
        return
    audit.check(set(validity) == FORMAL_METRICS, f"{label}: metric_validity keys={sorted(validity)}", section)
    audit.check(all(isinstance(validity.get(key), bool) for key in FORMAL_METRICS), f"{label}: validity must be boolean", section)

    m1, m2 = metrics.get("M1"), metrics.get("M2")
    audit.check(m1 is None or finite(m1), f"{label}: M1 must be finite or null", section)
    if task == "P3":
        if audit.check(isinstance(m2, dict), f"{label}: P3 M2 must be object", section):
            audit.check(set(m2) == P3_M2_KEYS, f"{label}: P3 M2 keys={sorted(m2)}", section)
            audit.check(all(value is None or finite(value) for value in m2.values()), f"{label}: P3 M2 values invalid", section)
            audit.check(not any("upper" in key or "lower" in key for key in m2), f"{label}: P3 must use 30deg/60deg identity", section)
    elif task == "P4":
        if audit.check(isinstance(m2, dict), f"{label}: P4 M2 must be object", section):
            audit.check(set(m2) == P4_M2_KEYS, f"{label}: P4 M2 keys={sorted(m2)}", section)
            checks = m2.get("height_decrease_checks")
            fraction = m2.get("height_decrease_fraction")
            audit.check(isinstance(checks, list), f"{label}: height_decrease_checks must be list", section)
            if isinstance(checks, list):
                audit.check(all(isinstance(value, bool) for value in checks), f"{label}: height_decrease_checks must be booleans", section)
                expected = sum(checks) / len(checks) if checks else None
                audit.check(close(fraction, expected), f"{label}: height_decrease_fraction={fraction!r}, expected={expected!r}", section)
            audit.check(fraction is None or (finite(fraction) and 0 <= float(fraction) <= 1), f"{label}: invalid height decrease fraction", section)
            cv = m2.get("adjacent_restitution_coefficient_cv")
            audit.check(cv is None or (finite(cv) and float(cv) >= 0), f"{label}: invalid adjacent restitution CV", section)
    elif task == "P9":
        if audit.check(isinstance(m2, dict), f"{label}: P9 M2 must be object", section):
            audit.check(set(m2) == P9_M2_KEYS, f"{label}: P9 M2 keys={sorted(m2)}", section)
            audit.check(all(value is None or (finite(value) and float(value) >= 0) for value in m2.values()), f"{label}: P9 M2 values invalid", section)
            audit.check(not any("max" in key.lower() for key in m2), f"{label}: maximum CV is not a document metric", section)
    else:
        audit.check(m2 is None or finite(m2), f"{label}: {task} M2 must be finite or null", section)

    # A declared-valid metric must be fully measurable.
    if validity.get("M1"):
        audit.check(finite(m1), f"{label}: M1_valid=true but M1 is not finite", section)
    if validity.get("M2"):
        if task == "P3":
            measurable = isinstance(m2, dict) and set(m2) == P3_M2_KEYS and all(finite(value) for value in m2.values())
        elif task == "P4":
            measurable = (
                isinstance(m2, dict)
                and set(m2) == P4_M2_KEYS
                and isinstance(m2.get("height_decrease_checks"), list)
                and len(m2.get("height_decrease_checks")) > 0
                and finite(m2.get("height_decrease_fraction"))
                and finite(m2.get("adjacent_restitution_coefficient_cv"))
            )
        elif task == "P9":
            measurable = isinstance(m2, dict) and set(m2) == P9_M2_KEYS and all(finite(value) for value in m2.values())
        else:
            measurable = finite(m2)
        audit.check(measurable, f"{label}: M2_valid=true but M2 is not fully measurable", section)


def audit_formal_json(audit: Audit) -> dict[str, dict[str, Any]]:
    files: list[Path] = []
    for task in TASK_COUNTS:
        files.extend(sorted((audit.root / task / "eval_results" / "json").glob("*.json")))
    audit.check(len(files) == EXPECTED_RECORDS, f"formal JSON count={len(files)}, expected={EXPECTED_RECORDS}", "formal-json")
    by_sample: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    m1_valid = m2_valid = 0
    for path in files:
        result = load_json(path, audit, "formal-json")
        if not isinstance(result, dict):
            continue
        rel = path.relative_to(audit.root).as_posix()
        task = str(result.get("task_id") or path.parts[-4])
        sample = str(result.get("record_id") or result.get("video_id") or result.get("sample_id") or path.stem)
        audit.check(task in TASK_COUNTS, f"{rel}: invalid task_id={task!r}", "formal-json")
        if task not in TASK_COUNTS:
            continue
        counts[task] += 1
        audit.check(sample not in by_sample, f"duplicate sample ID {sample}", "formal-json")
        by_sample[sample] = result
        forbidden = FORMAL_JSON_FORBIDDEN.intersection(result)
        audit.check(not forbidden, f"{rel}: forbidden formal fields={sorted(forbidden)}", "formal-json")
        validate_metric_shape(task, result, audit, sample)
        validity = result.get("metric_validity") or {}
        m1_valid += validity.get("M1") is True
        m2_valid += validity.get("M2") is True
    audit.check(dict(counts) == TASK_COUNTS, f"task counts={dict(counts)}, expected={TASK_COUNTS}", "formal-json")
    audit.check(len(by_sample) == EXPECTED_RECORDS, f"unique samples={len(by_sample)}", "formal-json")
    audit.check(m1_valid == EXPECTED_M1_VALID, f"M1 valid={m1_valid}, expected={EXPECTED_M1_VALID}", "formal-json")
    audit.check(m2_valid == EXPECTED_M2_VALID, f"M2 valid={m2_valid}, expected={EXPECTED_M2_VALID}", "formal-json")
    return by_sample


def read_csv(path: Path, audit: Audit, section: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)
    except Exception as exc:  # noqa: BLE001
        audit.fail(f"cannot parse {path.relative_to(audit.root)}: {exc}", section)
        return [], []


def sample_from_row(row: dict[str, str]) -> str:
    return str(row.get("Sample") or row.get("video_id") or row.get("sample_id") or "")


def audit_formal_csv(audit: Audit, formal: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    section = "formal-csv"
    root_csv = audit.root / "ALL_RESULTS_168.csv"
    audit.check(root_csv.is_file(), "ALL_RESULTS_168.csv missing", section)
    root_fields, root_rows = read_csv(root_csv, audit, section) if root_csv.is_file() else ([], [])
    lowered = [field.lower() for field in root_fields]
    forbidden = [field for field, low in zip(root_fields, lowered) if any(fragment in low for fragment in FORMAL_CSV_FORBIDDEN_FRAGMENTS)]
    audit.check(not forbidden, f"formal root CSV has forbidden columns={forbidden}", section)
    required = {"Sample", "task_id", "M1_valid", "M2_valid", "M1", "M2", "result_json", "raw_video", "overlay"}
    audit.check(required.issubset(root_fields), f"formal root CSV missing={sorted(required-set(root_fields))}", section)
    audit.check(len(root_rows) == EXPECTED_RECORDS, f"root CSV rows={len(root_rows)}", section)

    root_by_sample: dict[str, dict[str, str]] = {}
    for row in root_rows:
        sample = sample_from_row(row)
        audit.check(bool(sample), "formal root CSV row has no sample ID", section)
        audit.check(sample not in root_by_sample, f"duplicate root CSV sample={sample}", section)
        root_by_sample[sample] = row
        result = formal.get(sample)
        if result is None:
            audit.fail(f"root CSV sample={sample} has no formal JSON", section)
            continue
        audit.check(str(row.get("task_id")) == str(result.get("task_id")), f"{sample}: task mismatch CSV/JSON", section)
        audit.check(parse_bool(row.get("M1_valid")) is result["metric_validity"]["M1"], f"{sample}: M1 validity mismatch", section)
        audit.check(parse_bool(row.get("M2_valid")) is result["metric_validity"]["M2"], f"{sample}: M2 validity mismatch", section)
        audit.check(deep_equal(parse_cell(row.get("M1", "")), result["metrics"]["M1"]), f"{sample}: M1 mismatch CSV/JSON", section)
        audit.check(deep_equal(parse_cell(row.get("M2", "")), result["metrics"]["M2"]), f"{sample}: M2 mismatch CSV/JSON", section)
        if row.get("result_json"):
            result_path = resolve_root_link(audit.root, row["result_json"])
            audit.check(safe_inside(audit.root, result_path), f"{sample}: result_json escapes package", section)
            audit.check(result_path.is_file(), f"{sample}: result_json link missing", section)
            if result_path.is_file():
                linked = load_json(result_path, audit, section)
                if isinstance(linked, dict):
                    audit.check(str(linked.get("record_id")) == sample, f"{sample}: result_json points to another record", section)

    task_union: dict[str, dict[str, str]] = {}
    task_row_count = 0
    for task, expected in TASK_COUNTS.items():
        path = audit.root / task / "eval_results" / "results.csv"
        audit.check(path.is_file(), f"{task}/eval_results/results.csv missing", section)
        fields, rows = read_csv(path, audit, section) if path.is_file() else ([], [])
        audit.check(fields == root_fields, f"{task}: task/root CSV columns differ", section)
        audit.check(len(rows) == expected, f"{task}: rows={len(rows)}, expected={expected}", section)
        task_row_count += len(rows)
        for row in rows:
            sample = sample_from_row(row)
            audit.check(sample not in task_union, f"duplicate task CSV sample={sample}", section)
            task_union[sample] = row
            audit.check(row.get("task_id") == task, f"{sample}: appears under {task} but task_id={row.get('task_id')}", section)
    audit.check(task_row_count == EXPECTED_RECORDS, f"task CSV union rows={task_row_count}", section)
    audit.check(set(task_union) == set(root_by_sample), "task CSV/root CSV sample sets differ", section)
    for sample in set(task_union).intersection(root_by_sample):
        audit.check(task_union[sample] == root_by_sample[sample], f"{sample}: task/root CSV row differs", section)
    return root_by_sample


def audit_derived_csv(audit: Audit, formal: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    section = "derived-ranking"
    path = audit.root / "DERIVED_RANKING_168.csv"
    audit.check(path.is_file(), "DERIVED_RANKING_168.csv missing", section)
    fields, rows = read_csv(path, audit, section) if path.is_file() else ([], [])
    required = {"Sample", "task_id", "M1_score", "M2_score", "derived_overall", "derived_overall_valid"}
    audit.check(required.issubset(fields), f"derived ranking missing={sorted(required-set(fields))}", section)
    forbidden = [field for field in fields if any(x in field.lower() for x in ("m3", "dimension", "physics_pass", "legacy"))]
    audit.check(not forbidden, f"derived ranking has forbidden columns={forbidden}", section)
    audit.check(len(rows) == EXPECTED_RECORDS, f"derived rows={len(rows)}", section)
    by_sample: dict[str, dict[str, str]] = {}
    for row in rows:
        sample = sample_from_row(row)
        audit.check(sample not in by_sample, f"duplicate derived sample={sample}", section)
        by_sample[sample] = row
        result = formal.get(sample)
        if result is None:
            audit.fail(f"derived sample={sample} has no formal JSON", section)
            continue
        task = str(result["task_id"])
        audit.check(str(row.get("task_id")) == task, f"{sample}: derived task mismatch", section)
        expected = derived_scores(task, result["metrics"], result["metric_validity"])
        csv_to_expected = {
            "M1_score": "M1_score",
            "M2_score": "M2_score",
            "derived_overall": "overall_score",
        }
        for csv_key, expected_key in csv_to_expected.items():
            actual = parse_cell(row.get(csv_key, ""))
            audit.check(close(actual, expected[expected_key]), f"{sample}: {csv_key}={actual!r}, recomputed={expected[expected_key]!r}", section)
            if finite(actual):
                audit.check(0.0 <= float(actual) <= 1.0, f"{sample}: {csv_key} outside [0,1]", section)
        actual_valid = parse_bool(row.get("derived_overall_valid"))
        audit.check(actual_valid is expected["overall_valid"], f"{sample}: overall_valid mismatch", section)
        if not expected["overall_valid"]:
            audit.check(close(parse_cell(row.get("derived_overall", "")), 0.0), f"{sample}: invalid overall must be zero", section)

        # Counterfactual invariance: invented legacy/diagnostic values cannot enter
        # the function because its complete input is exactly metrics+validity.
        counterfactual = {
            **result,
            "dimensions": {"camera": 0.0, "tracking": 1.0, "appearance": 0.123},
            "legacy_score": 0.999,
            "diagnostic_status": {"extract_success": False, "structural_ok": False},
        }
        again = derived_scores(task, counterfactual["metrics"], counterfactual["metric_validity"])
        audit.check(deep_equal(expected, again), f"{sample}: derived ranking changed under counterfactual diagnostics", section)
    audit.check(set(by_sample) == set(formal), "derived/formal sample sets differ", section)
    return by_sample


def audit_gallery(
    audit: Audit,
    formal: dict[str, dict[str, Any]],
    derived_csv: dict[str, dict[str, str]],
) -> None:
    section = "gallery"
    pretty_path = audit.root / "video_gallery" / "full_evaluation_data.json"
    web_path = audit.root / "video_gallery" / "full_evaluation_data_web.json"
    html_path = audit.root / "video_gallery" / "evaluation.html"
    for path in (pretty_path, web_path, html_path):
        audit.check(path.is_file(), f"missing {path.relative_to(audit.root)}", section)
    pretty = load_json(pretty_path, audit, section) if pretty_path.is_file() else None
    web = load_json(web_path, audit, section) if web_path.is_file() else None
    audit.check(deep_equal(pretty, web), "pretty and web gallery JSON differ", section)
    if not isinstance(pretty, dict):
        return
    records = pretty.get("records")
    if not audit.check(isinstance(records, list), "gallery records is not a list", section):
        return
    audit.check(len(records) == EXPECTED_RECORDS, f"gallery records={len(records)}", section)
    gallery_samples: set[str] = set()
    raw_paths: set[Path] = set()
    overlay_paths: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            audit.fail("gallery record is not object", section)
            continue
        sample = str(record.get("record_id") or record.get("id") or record.get("video_id") or record.get("sample_id") or "")
        audit.check(bool(sample), "gallery record missing sample ID", section)
        audit.check(sample not in gallery_samples, f"duplicate gallery sample={sample}", section)
        gallery_samples.add(sample)
        result = formal.get(sample)
        if result is None:
            audit.fail(f"gallery sample={sample} has no formal JSON", section)
            continue
        forbidden = FORMAL_JSON_FORBIDDEN.intersection(record) - {"overall_valid"}
        # Any ranking value must be nested under the explicitly named derived layer.
        audit.check(not forbidden, f"{sample}: gallery top-level forbidden fields={sorted(forbidden)}", section)
        audit.check("derived_ranking" in record and isinstance(record.get("derived_ranking"), dict), f"{sample}: missing explicit derived_ranking", section)
        metrics = record.get("metrics")
        validity = record.get("metric_validity")
        audit.check(deep_equal(metrics, result.get("metrics")), f"{sample}: gallery/formal metrics differ", section)
        audit.check(deep_equal(validity, result.get("metric_validity")), f"{sample}: gallery/formal validity differs", section)
        expected = derived_scores(str(result["task_id"]), result["metrics"], result["metric_validity"])
        ranking = record.get("derived_ranking") or {}
        gallery_to_expected = {
            "M1_score": "M1_score",
            "M2_score": "M2_score",
            "overall": "overall_score",
            "valid": "overall_valid",
        }
        audit.check(set(gallery_to_expected).issubset(ranking), f"{sample}: derived_ranking keys missing={sorted(set(gallery_to_expected)-set(ranking))}", section)
        for gallery_key, expected_key in gallery_to_expected.items():
            if gallery_key == "valid":
                audit.check(ranking.get(gallery_key) is expected[expected_key], f"{sample}: gallery {gallery_key} mismatch", section)
            else:
                audit.check(close(ranking.get(gallery_key), expected[expected_key]), f"{sample}: gallery {gallery_key} mismatch", section)
        audit.check(ranking.get("official_benchmark_formula") is False, f"{sample}: derived ranking must be marked non-official", section)
        expected_status = "scored" if expected["overall_valid"] else "document_metric_unmeasurable"
        audit.check(ranking.get("status") == expected_status, f"{sample}: derived ranking status mismatch", section)
        csv_row = derived_csv.get(sample)
        if csv_row:
            comparisons = {
                "M1_score": ("M1_score", False),
                "M2_score": ("M2_score", False),
                "derived_overall": ("overall", False),
                "derived_overall_valid": ("valid", True),
            }
            for csv_key, (gallery_key, is_bool) in comparisons.items():
                actual = parse_bool(csv_row[csv_key]) if is_bool else parse_cell(csv_row[csv_key])
                audit.check(deep_equal(actual, ranking.get(gallery_key)), f"{sample}: gallery/derived CSV {csv_key} differs", section)

        links = record.get("links") or result.get("artifacts") or {}
        for key, container in (("raw", raw_paths), ("overlay", overlay_paths)):
            value = links.get(key)
            audit.check(isinstance(value, str) and bool(value), f"{sample}: missing {key} link", section)
            if not isinstance(value, str) or not value:
                continue
            path = resolve_gallery_link(audit.root, value)
            audit.check(safe_inside(audit.root, path), f"{sample}: {key} escapes package", section)
            audit.check(path.is_file() and path.stat().st_size > 0, f"{sample}: {key} artifact missing/empty", section)
            audit.check(path.suffix.lower() == ".mp4", f"{sample}: {key} is not MP4", section)
            container.add(path)
    audit.check(gallery_samples == set(formal), "gallery/formal sample sets differ", section)
    audit.check(len(raw_paths) == EXPECTED_RECORDS, f"unique raw MP4 links={len(raw_paths)}", section)
    audit.check(len(overlay_paths) == EXPECTED_RECORDS, f"unique overlay MP4 links={len(overlay_paths)}", section)
    if html_path.is_file():
        html = html_path.read_text(encoding="utf-8", errors="replace")
        audit.check("derived_ranking" in html, "gallery HTML does not explicitly use derived_ranking", section)
        for forbidden_text in ("legacy_score", "physics_pass", "score_delta_from_v1", ".dimensions"):
            audit.check(forbidden_text not in html, f"gallery HTML references {forbidden_text}", section)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_hashes(audit: Audit) -> None:
    section = "hashes"
    path = audit.root / "SHA256SUMS"
    if not audit.check(path.is_file(), "SHA256SUMS missing", section):
        return
    listed: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(?:\./)?(.+)$")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        if not match:
            audit.fail(f"SHA256SUMS:{line_no}: malformed line", section)
            continue
        expected, rel = match.group(1).lower(), match.group(2).replace("\\", "/")
        rel_path = Path(rel)
        audit.check(not rel_path.is_absolute() and ".." not in rel_path.parts, f"unsafe hash member={rel}", section)
        audit.check(rel not in listed, f"duplicate hash member={rel}", section)
        listed[rel] = expected
    actual_files = {
        file.relative_to(audit.root).as_posix()
        for file in audit.root.rglob("*")
        if file.is_file() and file.name != "SHA256SUMS"
    }
    audit.check(set(listed) == actual_files, f"hash coverage mismatch: missing={sorted(actual_files-set(listed))[:5]}, extra={sorted(set(listed)-actual_files)[:5]}", section)
    for rel, expected in sorted(listed.items()):
        file = (audit.root / rel).resolve()
        if not safe_inside(audit.root, file) or not file.is_file():
            audit.fail(f"hash member missing/unsafe={rel}", section)
            continue
        audit.check(not file.is_symlink(), f"hash member is symlink={rel}", section)
        actual = sha256_file(file)
        audit.check(actual == expected, f"SHA256 mismatch={rel}", section)


def audit_no_legacy(audit: Audit) -> None:
    section = "layout"
    bad_dirs = [path.relative_to(audit.root).as_posix() for path in audit.root.rglob("*") if path.is_dir() and path.name.lower() == "legacy_v1_json"]
    audit.check(not bad_dirs, f"legacy_v1_json directories remain={bad_dirs}", section)
    contract_path = audit.root / "DOCUMENT_METRIC_CONTRACT.json"
    audit.check(contract_path.is_file(), "DOCUMENT_METRIC_CONTRACT.json missing", section)
    if contract_path.is_file():
        contract = load_json(contract_path, audit, section)
        if isinstance(contract, dict):
            audit.check(contract.get("official_metrics") == ["M1", "M2"], f"official_metrics={contract.get('official_metrics')!r}", section)
            text = json.dumps(contract, ensure_ascii=False).lower()
            audit.check("m3" not in text, "scoring contract contains M3", section)


def run(root: Path) -> dict[str, Any]:
    audit = Audit(root)
    if not audit.check(audit.root.is_dir(), f"package root not found: {audit.root}", "layout"):
        return report(audit)
    audit_no_legacy(audit)
    formal = audit_formal_json(audit)
    audit_formal_csv(audit, formal)
    derived = audit_derived_csv(audit, formal)
    audit_gallery(audit, formal, derived)
    audit_hashes(audit)
    return report(audit)


def report(audit: Audit) -> dict[str, Any]:
    return {
        "audit": "feishu-rev153-M1-M2-only-independent-audit-v1",
        "package": str(audit.root),
        "passed": not audit.failures,
        "failure_count": len(audit.failures),
        "warning_count": len(audit.warnings),
        "checks_by_section": dict(sorted(audit.checks.items())),
        "failures": audit.failures,
        "warnings": audit.warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="extracted delivery package root")
    parser.add_argument("--report", type=Path, help="optional report path outside the frozen package")
    args = parser.parse_args()
    result = run(args.package)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
