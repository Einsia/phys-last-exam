"""Canonical per-video result schema for the Group 7 delivery.

The task evaluators intentionally expose a fairly rich, task-specific result
for debugging.  The public ``sample_xx.json`` files use a much smaller and
stable envelope instead.  This module is the single conversion point between
those two representations.

Only metrics declared by the benchmark task are promoted to ``metrics``.  All
other evaluator values remain available under ``verbose`` as diagnostic/raw
evidence and are never interpreted as additional benchmark scores.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Mapping


CANONICAL_SCHEMA_VERSION = "physical-vdm-sample-v1"
CANONICAL_METRIC_KEYS = ("M1", "M2", "M3")


# These are the metric declarations in the Feishu task packet.  The local
# tasks.json contains a few useful implementation diagnostics (for example,
# P8c sinusoid residual and P12 interface fit residual) that are deliberately
# *not* benchmark metrics.  Keep this table explicit so a future tasks.json
# edit cannot accidentally widen the scored surface.
TASK_METRIC_SPECS: dict[str, dict[str, Any]] = {
    "P7": {
        "m1": "abs((t_ring/t_sphere)/sqrt(10/7)-1)",
        "m2": (
            ("sphere_no_slip_error", "M2_sphere"),
            ("ring_no_slip_error", "M2_ring"),
        ),
    },
    "P8c": {
        "m1": "abs((T30/T15)-K(sin(15deg))/K(sin(7.5deg)))",
        "m2": (("T30_gt_T15", "T30_gt_T15"),),
    },
    "P10": {
        # ``theta_observed`` is read from the rendered ramp/trajectory for
        # each video; the nominal 30° task value is only a documented prior.
        "m1": "abs((a_up/a_down)/((sin(theta_observed)+mu*cos(theta_observed))/(sin(theta_observed)-mu*cos(theta_observed)))-1)",
        "m2": (
            ("constant_acceleration_fit_up", "M2_up_fit_relative_rmse"),
            ("constant_acceleration_fit_down", "M2_down_fit_relative_rmse"),
            ("t_down_gt_t_up", "t_down_gt_t_up"),
        ),
    },
    "P12": {
        "m1": "coefficient_of_variation(n_snell_estimates + n_critical_estimate)",
        # Interface-line fitting is useful diagnostics, but the task packet's
        # M2 is the independent per-ray Snell residual only.
        "m2": (("per_ray_snell_residual", "per_ray_snell_residual"),),
    },
    "P27": {
        "m1": "crushed_water_surface_rise_faster (binary)",
        # The ordering is the sole binary M1 decision.  M2 remains only the
        # initial equal-total-mass visual validity check; timing/area curves
        # stay in verbose measurements for audit and are never promoted as
        # extra benchmark scores.
        "m2": (("initial_mass_visual_validity", "initial_mass_visual_validity"),),
    },
}


def _finite(value: Any) -> float | None:
    """Return a finite number, preserving booleans as booleans elsewhere."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _present(value: Any) -> bool:
    """Whether a value is a usable scalar/object metric value."""

    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return _finite(value) is not None
    # A residual may be represented as a list (one value per ray).  Preserve
    # the list if it is finite and non-empty; an empty list is unavailable.
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_present(item) for item in value)
    if isinstance(value, Mapping):
        return bool(value)
    return bool(str(value).strip())


def _as_bool(value: Any) -> bool | None:
    """Parse evaluator condition flags without treating ``"False"`` as true."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and _finite(value) is not None:
        return bool(float(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "pass", "ok"}:
            return True
        if text in {"false", "no", "0", "fail", "bad"}:
            return False
    return None


def _jsonable(value: Any) -> Any:
    """Make copied evaluator data safe for strict JSON serialization."""

    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return _jsonable(value.tolist())
        except Exception:  # pragma: no cover - exotic array-like object
            pass
    # Numpy scalar support without importing numpy in the public converter.
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _jsonable(value.item())
        except Exception:  # pragma: no cover - exotic scalar object
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _safe_path(value: Any, *, kind: str, fallback: str | None = None) -> Any:
    """Make a public path relocatable while preserving useful suffixes."""

    if not isinstance(value, str):
        return value
    text = value.replace("\\", "/")
    if not text:
        return text
    # Already package-relative paths are left intact.
    if not re.match(r"^[A-Za-z]:[/]", text) and not text.startswith(("/", "//")):
        return text
    marker = "/first_frames/"
    if marker in text:
        return "first_frames/" + text.split(marker, 1)[1]
    marker = "/eval_results/debug/"
    if marker in text:
        return "eval_results/debug/" + text.split(marker, 1)[1]
    marker = "/eval_results/"
    if marker in text:
        return "eval_results/" + text.split(marker, 1)[1]
    if kind == "video" and text.lower().endswith(".mp4"):
        return fallback or f"minimax_h3/videos/{text.rsplit('/', 1)[-1]}"
    # Do not leak workstation/server roots in verbose diagnostics.  Preserve
    # the basename so an artifact can still be located in the debug folder.
    return f"source_artifacts/{text.rsplit('/', 1)[-1]}"


def _safe_verbose(value: Any, *, video_path: str | None, image_path: str | None) -> Any:
    """Recursively rewrite absolute diagnostic paths to package-relative ones."""

    if isinstance(value, Mapping):
        return {str(key): _safe_verbose(item, video_path=video_path, image_path=image_path) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_verbose(item, video_path=video_path, image_path=image_path) for item in value]
    if isinstance(value, str):
        text = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:[/]", text) or text.startswith(("/", "//")):
            lower = text.lower()
            if lower.endswith(".mp4"):
                return _safe_path(text, kind="video", fallback=video_path)
            if "/first_frames/" in text:
                return _safe_path(text, kind="image", fallback=image_path)
            return _safe_path(text, kind="artifact")
        return text
    return value


def _metric_value(result: Mapping[str, Any], key: str, *, task_id: str) -> Any:
    """Resolve one canonical component from old evaluator fields."""

    raw_metrics = result.get("metrics")
    metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
    measurements = result.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}

    # P27 M1 changed from a scalar disappearance-time ratio to a binary
    # water-surface-rise ordering.  Resolve that explicit decision before the
    # generic ``M1`` key so the old scalar cannot leak into the new schema.
    if task_id == "P27" and key == "M1":
        value = measurements.get("crushed_water_surface_rise_faster")
        if value is None:
            value = metrics.get("M1_crushed_water_surface_rise_faster")
        if value is None:
            value = metrics.get("M1_crushed_melts_faster")
        if value is None:
            ratio = measurements.get("t_crushed_over_t_block")
            if _finite(ratio) is not None:
                return bool(float(ratio) < 1.0)
        return _as_bool(value)

    if key in metrics:
        value = metrics.get(key)
        # Accepting an already-normalized wrapper makes conversion idempotent
        # and prevents a second pass from promoting the wrapper itself.
        if isinstance(value, Mapping) and "metric" in value:
            return value.get("metric")
        return value
    # Canonical files store composite M2 values under the wrapper.  Reading
    # that map here keeps direct normalizer calls idempotent as well as the
    # dedicated migration utility.
    m2_wrapper = metrics.get("M2")
    if isinstance(m2_wrapper, Mapping) and m2_wrapper.get("extract_success") is True:
        m2_value = m2_wrapper.get("metric")
        if isinstance(m2_value, Mapping) and key in m2_value:
            return m2_value.get(key)
        if isinstance(m2_value, Mapping):
            for pair in (TASK_METRIC_SPECS.get(task_id, {}).get("m2") or ()):
                if isinstance(pair, (list, tuple)) and len(pair) == 2 and str(pair[1]) == key:
                    return m2_value.get(str(pair[0]))
    if key in measurements:
        return measurements.get(key)

    # Condition metrics were historically encoded as penalties or numeric
    # flags.  Expose the task expression directly as a bool in the canonical
    # object while retaining the original numeric field in verbose.raw_metrics.
    if key == "T30_gt_T15":
        value = measurements.get("T30_greater_than_T15")
        if value is None:
            penalty = metrics.get("M2_T30_not_greater_penalty")
            if penalty is not None:
                try:
                    return bool(float(penalty) == 0.0)
                except (TypeError, ValueError):
                    return None
        return _as_bool(value)
    if key == "t_down_gt_t_up":
        value = metrics.get("M2_t_down_gt_t_up")
        if value is None:
            value = measurements.get("t_down_gt_t_up")
        return _as_bool(value)
    if key == "t_crushed_lt_t_block":
        value = metrics.get("M2_t_crushed_lt_t_block")
        if value is None:
            ratio = measurements.get("t_crushed_over_t_block")
            if _finite(ratio) is not None:
                return bool(float(ratio) < 1.0)
        return _as_bool(value)
    if key == "crushed_water_surface_rise_faster":
        # New P27 evaluator emits the binary comparison directly.  Keep a
        # compatibility fallback for legacy results whose only evidence was a
        # finite disappearance-time ratio.
        value = measurements.get("crushed_water_surface_rise_faster")
        if value is None:
            value = metrics.get("M1_crushed_water_surface_rise_faster")
        if value is None:
            value = metrics.get("M1_crushed_melts_faster")
        if value is None:
            ratio = measurements.get("t_crushed_over_t_block")
            if _finite(ratio) is not None:
                return bool(float(ratio) < 1.0)
        return _as_bool(value)
    if key == "per_ray_snell_residual":
        # The task asks for each independent ray.  Prefer the raw vector and
        # fall back to the historical aggregate only for old result files.
        raw_vector = measurements.get("per_ray_snell_residual")
        if _present(raw_vector):
            return raw_vector
        return metrics.get("per_ray_snell_residual_rmse")
    if key == "initial_mass_visual_validity":
        # This is a proxy error (lower is better), not a fabricated boolean.
        return metrics.get("M2_initial_projected_area_log_error")
    return None


def _canonical_metric(value: Any, *, success: bool) -> dict[str, Any]:
    """Wrap one metric according to the user-provided contract."""

    if not success or not _present(value):
        return {"extract_success": False, "metric": None}
    return {"extract_success": True, "metric": _jsonable(value)}


def _slug_expression(expression: Any, index: int) -> str:
    text = str(expression or "").strip()
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or f"submetric_{index:02d}"


def _spec_for(task_id: str, task_spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the authoritative metric spec, with safe generic fallback."""

    if task_id in TASK_METRIC_SPECS:
        return TASK_METRIC_SPECS[task_id]
    spec = task_spec if isinstance(task_spec, Mapping) else {}
    m2 = spec.get("m2")
    if isinstance(m2, (list, tuple)):
        pairs = tuple(
            (_slug_expression(item, index), _slug_expression(item, index))
            for index, item in enumerate(m2, start=1)
        )
    elif m2:
        pairs = ((_slug_expression(m2, 1), _slug_expression(m2, 1)),)
    else:
        pairs = ()
    return {"m1": spec.get("m1"), "m2": pairs}


def normalize_sample_result(
    result: Mapping[str, Any],
    *,
    task_id: str | None = None,
    sample_id: str | int | None = None,
    video_path: str | None = None,
    image_path: str | None = None,
    seed: int | None = None,
    model: str = "minimax-h3",
    task_spec: Mapping[str, Any] | None = None,
    extra_verbose: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one evaluator result to the canonical ``sample_xx.json``.

    ``result`` is not mutated.  ``M1`` and every declared ``M2`` component are
    all-or-nothing at metric level: an unavailable value is represented by
    ``extract_success=false, metric=null``.  If the entire video extraction
    failed, every declared metric is forced to the same null form.  An
    undeclared metric is represented by the explicit
    ``{extract_success: null, metric: null}`` wrapper for a stable three-slot
    envelope.
    """

    raw = _jsonable(copy.deepcopy(dict(result)))
    if not isinstance(raw, dict):  # defensive; Mapping input should be dict
        raw = {}
    resolved_task = str(task_id or raw.get("task_id") or "")
    nested_status = raw.get("verbose", {}).get("status", {}) if isinstance(raw.get("verbose"), Mapping) else {}
    resolved_sample = sample_id if sample_id is not None else raw.get("sample_id")
    if resolved_sample is None and isinstance(nested_status, Mapping):
        resolved_sample = nested_status.get("sample_id")
    resolved_video = video_path or raw.get("video_path") or raw.get("video")
    resolved_image = image_path or raw.get("image_path") or raw.get("first_frame")
    if isinstance(resolved_video, str):
        resolved_video = _safe_path(resolved_video, kind="video")
    if isinstance(resolved_image, str):
        resolved_image = _safe_path(resolved_image, kind="image")
    resolved_seed = seed if seed is not None else raw.get("seed")
    resolved_model = model or raw.get("model") or "minimax-h3"
    spec = _spec_for(resolved_task, task_spec)
    raw_status = raw.get("verbose", {}).get("status", {}) if isinstance(raw.get("verbose"), Mapping) else {}
    raw_extract = raw.get("extract_success")
    if raw_extract is None and isinstance(raw_status, Mapping):
        raw_extract = raw_status.get("extract_success")
    if raw_extract is None:
        raw_m1 = (raw.get("metrics") or {}).get("M1") if isinstance(raw.get("metrics"), Mapping) else None
        if isinstance(raw_m1, Mapping) and "extract_success" in raw_m1:
            raw_extract = raw_m1.get("extract_success")
    video_extract = bool(_as_bool(raw_extract))
    raw_physics = raw.get("physics_pass", raw_status.get("physics_pass") if isinstance(raw_status, Mapping) else None)
    raw_video_qa = raw.get("video_qa_ok", raw_status.get("video_qa_ok") if isinstance(raw_status, Mapping) else None)
    normalized_physics = None if raw_physics is None else _as_bool(raw_physics)
    normalized_video_qa = None if raw_video_qa is None else _as_bool(raw_video_qa)

    metrics: dict[str, Any] = {}
    if spec.get("m1"):
        m1 = _metric_value(raw, "M1", task_id=resolved_task)
        metrics["M1"] = _canonical_metric(m1, success=video_extract)
    else:
        metrics["M1"] = None

    m2_pairs = tuple(spec.get("m2") or ())
    if m2_pairs:
        values: dict[str, Any] = {}
        all_available = video_extract
        for index, pair in enumerate(m2_pairs, start=1):
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                label, lookup = str(pair[0]), str(pair[1])
            else:  # generic fallback for malformed custom specs
                label = _slug_expression(pair, index)
                lookup = label
            value = _metric_value(raw, lookup, task_id=resolved_task)
            values[label] = _jsonable(value)
            if not _present(value):
                all_available = False
        metrics["M2"] = (
            {"extract_success": True, "metric": values}
            if all_available
            else {"extract_success": False, "metric": None}
        )
    else:
        metrics["M2"] = {"extract_success": None, "metric": None}
    # No Group 7 task currently declares M3.  Keep the slot explicit and
    # unscored using the same two-field shape as every other metric slot.
    metrics["M3"] = {"extract_success": None, "metric": None}

    raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    raw_measurements = raw.get("measurements") if isinstance(raw.get("measurements"), Mapping) else {}
    raw_diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), Mapping) else {}
    verbose: dict[str, Any] = {
        "raw_metrics": _safe_verbose(_jsonable(raw_metrics), video_path=resolved_video, image_path=resolved_image),
        "measurements": _safe_verbose(_jsonable(raw_measurements), video_path=resolved_video, image_path=resolved_image),
        "diagnostics": _safe_verbose(_jsonable(raw_diagnostics), video_path=resolved_video, image_path=resolved_image),
        "status": {
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "sample_id": resolved_sample,
            "extract_success": None if raw_extract is None else video_extract,
            "physics_pass": normalized_physics,
            "failure_reason": _safe_verbose(raw.get("failure_reason"), video_path=resolved_video, image_path=resolved_image),
            "failure_reasons": _safe_verbose(raw.get("failure_reasons", []), video_path=resolved_video, image_path=resolved_image),
            "failure_details": _safe_verbose(raw.get("failure_details", {}), video_path=resolved_video, image_path=resolved_image),
            "metric_failures": _safe_verbose(raw.get("metric_failures", []), video_path=resolved_video, image_path=resolved_image),
            "physics_failure_reasons": _safe_verbose(raw.get("physics_failure_reasons", []), video_path=resolved_video, image_path=resolved_image),
            "video_qa_ok": normalized_video_qa,
        },
        "debug_artifacts": _safe_verbose(_jsonable(raw.get("debug_artifacts", [])), video_path=resolved_video, image_path=resolved_image),
        "metric_mapping": {
            "task_declarations": {
                "M1": spec.get("m1"),
                "M2": [str(pair[0]) for pair in m2_pairs],
                "M3": None,
            },
            "canonical_sources": {
                "M1": "raw_metrics.M1",
                "M2": {
                    str(pair[0]): str(pair[1])
                    for pair in m2_pairs
                    if isinstance(pair, (list, tuple)) and len(pair) == 2
                },
            },
            "notes": [
                "Only task-declared metrics are promoted to top-level metrics.",
                "All other evaluator values are diagnostic/raw evidence, not benchmark scores.",
            ],
        },
    }
    if resolved_task == "P27":
        verbose["metric_mapping"]["notes"].append(
            "P27 initial_mass_visual_validity uses M2_initial_projected_area_log_error as a visual equal-mass proxy; lower is better."
        )
    if extra_verbose:
        verbose.update(_safe_verbose(_jsonable(dict(extra_verbose)), video_path=resolved_video, image_path=resolved_image))

    canonical: dict[str, Any] = {
        "task_id": resolved_task,
        "video_path": resolved_video,
        "image_path": resolved_image,
        "seed": resolved_seed,
        "model": resolved_model,
        "metrics": metrics,
        "verbose": verbose,
    }
    # Keep source/batch metadata out of the scored envelope but make it easy to
    # recover in verbose when normalizing a delivery manifest row.
    for key in ("batch_id", "source", "image_variant", "image_id", "source_sample_id"):
        if key in raw:
            verbose.setdefault("status", {})[key] = raw[key]
    return _jsonable(canonical)


def validate_sample_result(
    sample: Mapping[str, Any],
    *,
    task_id: str | None = None,
    require_relative_paths: bool = False,
) -> list[str]:
    """Return structural errors for one public ``sample_xx.json``.

    The validator checks only the stable public envelope.  Task-specific
    intermediate arrays and quality diagnostics are intentionally permitted
    under ``verbose`` and are not interpreted as additional benchmark scores.
    An empty list means the sample conforms to the schema.
    """

    errors: list[str] = []
    if not isinstance(sample, Mapping):
        return ["result is not an object"]

    required = {"task_id", "video_path", "image_path", "seed", "model", "metrics", "verbose"}
    actual = set(sample)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        if missing:
            errors.append(f"missing top-level keys: {missing}")
        if extra:
            errors.append(f"unexpected top-level keys: {extra}")

    resolved_task = str(task_id or sample.get("task_id") or "")
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping):
        errors.append("metrics is not an object")
        metrics = {}
    if set(metrics) != set(CANONICAL_METRIC_KEYS):
        errors.append(f"metrics keys must be {list(CANONICAL_METRIC_KEYS)}")

    for name in CANONICAL_METRIC_KEYS:
        value = metrics.get(name)
        if value is None:
            # A declared metric is represented by an explicit wrapper even
            # when extraction failed; only an undeclared M1/M2 may be bare
            # null in a custom task.
            if resolved_task in TASK_METRIC_SPECS:
                errors.append(f"{name} must be an extract wrapper")
            continue
        if not isinstance(value, Mapping) or set(value) != {"extract_success", "metric"}:
            errors.append(f"{name} must contain exactly extract_success and metric")
            continue
        flag = value.get("extract_success")
        if flag not in (True, False, None):
            errors.append(f"{name}.extract_success must be true, false, or null")
        if flag in (False, None) and value.get("metric") is not None:
            errors.append(f"{name}.metric must be null when extract_success is {flag}")
        if flag is True and value.get("metric") is None:
            errors.append(f"{name}.metric cannot be null when extract_success is true")

    if metrics.get("M3") != {"extract_success": None, "metric": None}:
        errors.append("M3 is undeclared; use the explicit null wrapper")

    if resolved_task in TASK_METRIC_SPECS:
        spec = TASK_METRIC_SPECS[resolved_task]
        m2 = metrics.get("M2")
        pairs = tuple(spec.get("m2") or ())
        if pairs and isinstance(m2, Mapping) and m2.get("extract_success") is True:
            expected = {
                str(pair[0])
                for pair in pairs
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
            metric = m2.get("metric")
            if not isinstance(metric, Mapping) or set(metric) != expected:
                errors.append(f"M2.metric keys for {resolved_task} must be {sorted(expected)}")
        if not pairs and m2 != {"extract_success": None, "metric": None}:
            errors.append(f"{resolved_task} has no declared M2; use the explicit null wrapper")

    if require_relative_paths:
        for key in ("video_path", "image_path"):
            value = sample.get(key)
            if not isinstance(value, str) or not value:
                errors.append(f"{key} must be a non-empty relative path")
            elif value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", value):
                errors.append(f"{key} must be relative: {value}")
            elif "\\" in value:
                errors.append(f"{key} must use POSIX separators: {value}")

    if not isinstance(sample.get("verbose"), Mapping):
        errors.append("verbose is not an object")
    return errors


__all__ = [
    "CANONICAL_METRIC_KEYS",
    "CANONICAL_SCHEMA_VERSION",
    "TASK_METRIC_SPECS",
    "normalize_sample_result",
    "validate_sample_result",
]
