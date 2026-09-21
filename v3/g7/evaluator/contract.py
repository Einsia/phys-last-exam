"""Stable, auditable G7 result contract.

The task modules in :mod:`evaluator.tasks` intentionally return rich internal
measurements.  This module converts those measurements into the public
per-video envelope used by the refined Group 7 delivery.  It is deliberately
deterministic and imports no learned model, LLM, or VLM.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping


TASK_METRICS: dict[str, dict[str, Any]] = {
    "P7": {
        "title": "实心球与圆环从同一斜面同时释放",
        "m1_formula": "abs((t_ring/t_sphere)/sqrt(10/7)-1)",
        "m1_logic": "跟踪球和圆环到同一终点的共同距离，用到达时间比与无滑动滚动理论值 sqrt(10/7) 的相对误差作为 M1。",
        "m2_logic": "从物体上的旋转标记估计角速度，比较 v/(omega*R) 与 1 的偏差；球和圆环两个子项必须同时可提取。",
        "m2_keys": ("sphere_no_slip_error", "ring_no_slip_error"),
    },
    "P8c": {
        "title": "同长、不同初始角度的双摆同时释放",
        "m1_formula": "abs((T30/T15)-K(sin(15deg))/K(sin(7.5deg)))",
        "m1_logic": "分别跟踪 15° 与 30° 摆的摆锤，使用峰值间隔估计周期，并与有限振幅单摆的椭圆积分周期比比较。",
        "m2_logic": "检查大初始角度摆的周期是否严格长于小初始角度摆，即 T30 > T15。",
        "m2_keys": ("T30_gt_T15",),
    },
    "P10": {
        "title": "物块沿粗糙斜面上行后自行滑回",
        "m1_formula": "abs((a_up/a_down)/((sin(theta)+mu*cos(theta))/(sin(theta)-mu*cos(theta)))-1)",
        "m1_logic": "从视频中的斜面边缘拟合实际 theta，再对上行和下行轨迹分别做二次拟合，比较加速度比与含摩擦理论值。",
        "m2_logic": "报告上行/下行恒加速度拟合误差，并检查返回时间是否长于上行时间。",
        "m2_keys": ("constant_acceleration_fit_up", "constant_acceleration_fit_down", "t_down_gt_t_up"),
    },
    "P12": {
        "title": "多入射角光束在空气—水界面折射",
        "m1_formula": "coefficient_of_variation(n_snell_estimates + n_critical_estimate)",
        "m1_logic": "在同一采样帧用直线检测得到各入射/折射线角度，按 Snell 定律估计 n，并与临界角估计合并后计算变异系数。",
        "m2_logic": "保留每条独立光线的 Snell 定律残差；严格物理判定还要求三条配对和临界反射分支可验证。",
        "m2_keys": ("per_ray_snell_residual",),
    },
    "P27": {
        "title": "等质量碎冰和整冰并排融化",
        "m1_formula": "crushed_water_surface_rise_faster (binary)",
        "m1_logic": "分别分割两个烧杯中的水面信号，比较碎冰侧水面上升是否更早/更快。该任务的 M1 是明确的二值物理判定。",
        "m2_logic": "用首帧中两侧冰的投影面积和几何尺度给出等总质量的视觉有效性代理误差；低误差表示首帧比较可测。",
        "m2_keys": ("initial_mass_visual_validity",),
    },
}


# Fixed task-level anchors keep scores comparable across runs and models.  No
# anchor is estimated from the current batch.  Error metrics use the common
# rational mapping q(e,a)=1/(1+(abs(e)/a)^2); ordering metrics use a clipped
# continuous physical margin.  A missing measurement is represented by score
# zero and ``proxy_valid=false`` so it cannot be confused with a measured zero.
PROXY_VERSION = "physical-bench-proxy-v1"
CONTRACT_VERSION = "g7-proxy-v1"
PROXY_ANCHORS: dict[str, dict[str, float]] = {
    "P7": {"M1_half_error": 0.10, "M2_no_slip_half_error": 0.20},
    "P8c": {"M1_half_error": 0.05},
    "P10": {
        "M1_half_error": 0.10,
        "M2_fit_half_error": 0.10,
    },
    "P12": {"M1_half_error": 0.05, "M2_snell_residual_half_error": 0.06665},
    "P27": {
        "M2_log_area_half_error": math.log(1.25),
    },
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return _finite(value) is not None
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_present(item) for item in value)
    if isinstance(value, Mapping):
        return bool(value)
    return bool(str(value).strip())


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and _finite(value) is not None:
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "pass", "ok"}:
            return True
        if text in {"false", "no", "0", "fail", "bad"}:
            return False
    return None


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _error_proxy(value: Any, half_error: float) -> float | None:
    """Map an error to (0, 1], with ``half_error`` scoring exactly 0.5."""

    number = _finite(value)
    if number is None or half_error <= 0.0:
        return None
    return _clamp01(1.0 / (1.0 + (abs(number) / half_error) ** 2))


def _positive_margin_proxy(margin: Any, target_margin: Any) -> float | None:
    """Map zero/wrong order to 0 and the expected positive margin to 1."""

    number = _finite(margin)
    target = _finite(target_margin)
    if number is None or target is None or target <= 0.0:
        return None
    return _clamp01(number / target)


def _geometric_mean(values: list[float]) -> float:
    """Geometric mean on [0,1]; an unavailable/zero input is a hard zero."""

    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return _clamp01(math.exp(sum(math.log(value) for value in values) / len(values)))


def _component(raw: Any, score: float | None, formula: str, anchor: str) -> dict[str, Any]:
    return {
        "raw_metric": _jsonable(raw),
        "proxy_score": 0.0 if score is None else _clamp01(score),
        "formula": formula,
        "anchor": anchor,
        "measurable": score is not None,
    }


def _proxy_scores(
    task_id: str,
    m1: Any,
    m2: Mapping[str, Any],
    measurements: Mapping[str, Any],
    *,
    m1_available: bool,
    m2_available: bool,
) -> tuple[float, float, dict[str, Any], dict[str, Any]]:
    """Calculate fixed, deterministic proxies and auditable component data."""

    anchors = PROXY_ANCHORS[task_id]
    m1_components: dict[str, Any] = {}
    m2_components: dict[str, Any] = {}

    if task_id == "P27":
        crushed = _finite(measurements.get("crushed_water_surface_half_rise_time_s"))
        block = _finite(measurements.get("block_water_surface_half_rise_time_s"))
        ratio = crushed / block if crushed is not None and block is not None and block > 0.0 else None
        advantage = block - crushed if crushed is not None and block is not None else None
        event_scale = max(crushed, block) if crushed is not None and block is not None else None
        m1_score = (
            _clamp01(advantage / event_scale)
            if advantage is not None and event_scale is not None and event_scale > 0.0
            else None
        )
        m1_components["crushed_vs_block_half_rise"] = _component(
            {"declared_M1": m1, "t_crushed_over_t_block": ratio, "t_block_minus_t_crushed_s": advantage},
            m1_score,
            "clip((t_block - t_crushed) / max(t_block,t_crushed), 0, 1)",
            "relative event-time advantage; equal/slower crushed-side rise -> 0",
        )
    else:
        half_error = anchors["M1_half_error"]
        m1_score = _error_proxy(m1, half_error)
        m1_components["primary_error"] = _component(
            m1,
            m1_score,
            f"1/(1+(abs(error)/{half_error:g})^2)",
            f"error=0 -> 1; abs(error)={half_error:g} -> 0.5",
        )
    if not m1_available:
        m1_score = None

    if task_id == "P7":
        for name in ("sphere_no_slip_error", "ring_no_slip_error"):
            raw = m2.get(name)
            score = _error_proxy(raw, anchors["M2_no_slip_half_error"])
            m2_components[name] = _component(raw, score, "1/(1+(abs(no_slip_error)/0.20)^2)", "abs(error)=0.20 -> 0.5")
    elif task_id == "P8c":
        t15 = _finite(measurements.get("T15_s"))
        t30 = _finite(measurements.get("T30_s"))
        ratio = t30 / t15 if t15 is not None and t30 is not None and t15 > 0.0 else None
        expected_ratio = _finite(measurements.get("expected_period_ratio_exact"))
        score = _positive_margin_proxy(
            None if ratio is None else ratio - 1.0,
            None if expected_ratio is None else expected_ratio - 1.0,
        )
        m2_components["T30_gt_T15"] = _component(
            {"declared": m2.get("T30_gt_T15"), "T30_over_T15": ratio, "expected_T30_over_T15": expected_ratio},
            score,
            "clip((T30/T15 - 1)/(expected_ratio - 1), 0, 1)",
            "equal or reversed periods -> 0; finite-amplitude theoretical ratio -> 1",
        )
    elif task_id == "P10":
        for name in ("constant_acceleration_fit_up", "constant_acceleration_fit_down"):
            raw = m2.get(name)
            score = _error_proxy(raw, anchors["M2_fit_half_error"])
            m2_components[name] = _component(raw, score, "1/(1+(abs(relative_RMSE)/0.10)^2)", "relative RMSE=0.10 -> 0.5")
        t_up = _finite(measurements.get("t_up_s"))
        t_down = _finite(measurements.get("t_down_s"))
        ratio = t_down / t_up if t_up is not None and t_down is not None and t_up > 0.0 else None
        expected_acceleration_ratio = _finite(measurements.get("expected_acceleration_ratio"))
        expected_time_ratio = math.sqrt(expected_acceleration_ratio) if expected_acceleration_ratio is not None and expected_acceleration_ratio > 1.0 else None
        score = _positive_margin_proxy(
            None if ratio is None else ratio - 1.0,
            None if expected_time_ratio is None else expected_time_ratio - 1.0,
        )
        m2_components["t_down_gt_t_up"] = _component(
            {"declared": m2.get("t_down_gt_t_up"), "t_down_over_t_up": ratio, "expected_t_down_over_t_up": expected_time_ratio},
            score,
            "clip((t_down/t_up - 1)/(sqrt(expected_acceleration_ratio) - 1), 0, 1)",
            "equal/shorter return -> 0; theory-consistent time ratio -> 1",
        )
    elif task_id == "P12":
        residuals = m2.get("per_ray_snell_residual")
        values = residuals if isinstance(residuals, (list, tuple)) else []
        for index, raw in enumerate(values):
            score = _error_proxy(raw, anchors["M2_snell_residual_half_error"])
            m2_components[f"ray_{index + 1:02d}"] = _component(raw, score, "1/(1+(abs(Snell_residual)/0.06665)^2)", "5% of n_water≈1.333 (0.06665) -> 0.5")
        coverage = _clamp01(len(values) / 3.0)
        m2_components["three_ray_coverage"] = _component(
            {"measured_ray_count": len(values), "required_ray_count": 3},
            coverage,
            "clip(measured_ray_count/3, 0, 1)",
            "all three requested Snell pairs -> 1",
        )
    elif task_id == "P27":
        raw = m2.get("initial_mass_visual_validity")
        score = _error_proxy(raw, anchors["M2_log_area_half_error"])
        m2_components["initial_mass_visual_validity"] = _component(
            raw,
            score,
            "1/(1+(abs(log(projected_area_ratio))/log(1.25))^2)",
            "25% projected-area ratio error -> 0.5",
        )

    component_scores = [float(item["proxy_score"]) for item in m2_components.values()]
    all_measurable = bool(m2_components) and all(bool(item["measurable"]) for item in m2_components.values())
    if task_id == "P12" and all_measurable:
        ray_scores = [
            float(item["proxy_score"])
            for name, item in m2_components.items()
            if name.startswith("ray_")
        ]
        coverage_score = float(m2_components["three_ray_coverage"]["proxy_score"])
        m2_score = coverage_score * _geometric_mean(ray_scores) if ray_scores else None
    else:
        m2_score = _geometric_mean(component_scores) if all_measurable else None
    if not m2_available:
        m2_score = None
    return (
        0.0 if m1_score is None else _clamp01(m1_score),
        0.0 if m2_score is None else _clamp01(m2_score),
        m1_components,
        m2_components,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _relative_path(value: str | Path | None, *, package_root: Path, fallback: str) -> str:
    """Return a POSIX package-relative path without leaking local roots."""

    if value is None:
        return fallback
    path = Path(value)
    try:
        rel = path.resolve().relative_to(package_root.resolve())
        return rel.as_posix()
    except (ValueError, OSError):
        text = str(value).replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
            return fallback
        return text.lstrip("./") or fallback


def _safe_payload(value: Any, *, package_root: Path, task_id: str, sample_id: str) -> Any:
    """Recursively remove workstation roots from diagnostic payloads."""

    if isinstance(value, Mapping):
        return {
            str(key): _safe_payload(item, package_root=package_root, task_id=task_id, sample_id=sample_id)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_payload(item, package_root=package_root, task_id=task_id, sample_id=sample_id) for item in value]
    if isinstance(value, str):
        text = value.replace("\\", "/")
        if re.match(r"^[A-Za-z]:/", text) or text.startswith("/"):
            suffix = Path(text).name
            if text.lower().endswith(".mp4"):
                return _relative_path(text, package_root=package_root, fallback=f"{task_id}/minimax_h3/videos/{suffix}")
            if text.lower().endswith(('.png', '.jpg', '.jpeg', '.csv', '.json')):
                return _relative_path(text, package_root=package_root, fallback=f"{task_id}/eval_results/debug/{sample_id}/{suffix}")
            return suffix
        return text
    return _jsonable(value)


def _raw_metric(raw: Mapping[str, Any], key: str) -> Any:
    metrics = raw.get("metrics")
    if isinstance(metrics, Mapping) and key in metrics:
        value = metrics[key]
        if isinstance(value, Mapping) and "metric" in value:
            return value.get("metric")
        return value
    measurements = raw.get("measurements")
    if isinstance(measurements, Mapping):
        return measurements.get(key)
    return None


def _declared_values(task_id: str, raw: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Extract only the task-declared M1 and M2 values."""

    measurements = raw.get("measurements") if isinstance(raw.get("measurements"), Mapping) else {}
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    m1 = metrics.get("M1")
    if isinstance(m1, Mapping):
        m1 = m1.get("metric")

    if task_id == "P7":
        m2 = {
            "sphere_no_slip_error": metrics.get("M2_sphere"),
            "ring_no_slip_error": metrics.get("M2_ring"),
        }
    elif task_id == "P8c":
        m2 = {"T30_gt_T15": measurements.get("T30_greater_than_T15")}
        if m2["T30_gt_T15"] is None:
            m2["T30_gt_T15"] = metrics.get("M2_T30_gt_T15")
    elif task_id == "P10":
        m2 = {
            "constant_acceleration_fit_up": metrics.get("M2_up_fit_relative_rmse"),
            "constant_acceleration_fit_down": metrics.get("M2_down_fit_relative_rmse"),
            "t_down_gt_t_up": measurements.get("t_down_gt_t_up"),
        }
        if m2["t_down_gt_t_up"] is None:
            m2["t_down_gt_t_up"] = metrics.get("M2_t_down_gt_t_up")
    elif task_id == "P12":
        m2 = {"per_ray_snell_residual": measurements.get("per_ray_snell_residual")}
        if m2["per_ray_snell_residual"] is None:
            m2["per_ray_snell_residual"] = metrics.get("per_ray_snell_residual")
    elif task_id == "P27":
        m2 = {"initial_mass_visual_validity": metrics.get("M2_initial_projected_area_log_error")}
    else:
        raise ValueError(f"unsupported task_id: {task_id}")
    return m1, m2


def _wrapper(value: Any, *, available: bool, proxy_score: float, proxy_valid: bool) -> dict[str, Any]:
    if not available or not _present(value):
        return {"extract_success": False, "metric": None, "proxy_score": 0.0, "proxy_valid": False}
    return {
        "extract_success": True,
        "metric": _jsonable(value),
        "proxy_score": _clamp01(proxy_score) if proxy_valid else 0.0,
        "proxy_valid": bool(proxy_valid),
    }


def build_public_result(
    raw: Mapping[str, Any],
    *,
    task_id: str,
    sample_id: str,
    video_path: str | Path,
    image_path: str | Path,
    video_prompt: str,
    seed: int | None,
    model: str,
    package_root: str | Path,
    source: str | None = None,
    image_variant: int | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    """Build the strict per-video JSON envelope.

    The returned top-level keys are exactly the 9.2 contract plus the two
    declared metrics.  All task-specific curves, masks and diagnostics remain
    under ``verbose`` so they are auditable but cannot become extra scores.
    """

    if task_id not in TASK_METRICS:
        raise ValueError(f"unsupported task_id: {task_id}")
    root = Path(package_root).resolve()
    task = TASK_METRICS[task_id]
    raw_metrics = raw.get("metrics") if isinstance(raw.get("metrics"), Mapping) else {}
    raw_measurements = raw.get("measurements") if isinstance(raw.get("measurements"), Mapping) else {}
    raw_diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), Mapping) else {}
    extract = raw.get("extract_success")
    if extract is None and isinstance(raw.get("status"), Mapping):
        extract = raw["status"].get("extract_success")
    if extract is None and isinstance(raw.get("verbose"), Mapping):
        status = raw["verbose"].get("status")
        if isinstance(status, Mapping):
            extract = status.get("extract_success")
    video_extract = bool(_bool_value(extract))
    m1, m2 = _declared_values(task_id, raw)
    m1_available = video_extract and _present(m1)
    m2_available = video_extract and all(_present(m2.get(key)) for key in task["m2_keys"])
    m1_proxy, m2_proxy, m1_proxy_components, m2_proxy_components = _proxy_scores(
        task_id,
        m1,
        m2,
        raw_measurements,
        m1_available=m1_available,
        m2_available=m2_available,
    )
    m1_proxy_valid = m1_available and bool(m1_proxy_components) and all(
        bool(item.get("measurable")) for item in m1_proxy_components.values()
    )
    m2_proxy_valid = m2_available and bool(m2_proxy_components) and all(
        bool(item.get("measurable")) for item in m2_proxy_components.values()
    )
    m1_wrapper = _wrapper(
        m1, available=m1_available, proxy_score=m1_proxy, proxy_valid=m1_proxy_valid
    )
    m2_wrapper = _wrapper(
        m2, available=m2_available, proxy_score=m2_proxy, proxy_valid=m2_proxy_valid
    )
    overall_proxy_score = _geometric_mean([m1_wrapper["proxy_score"], m2_wrapper["proxy_score"]])

    rel_video = _relative_path(video_path, package_root=root, fallback=f"{task_id}/minimax_h3/videos/{Path(video_path).name}")
    rel_image = _relative_path(image_path, package_root=root, fallback=f"{task_id}/first_frames/unknown.png")
    debug_artifacts = raw.get("debug_artifacts", [])
    debug_artifacts = _jsonable(debug_artifacts if isinstance(debug_artifacts, list) else [])
    if isinstance(debug_artifacts, list):
        safe_artifacts: list[Any] = []
        for artifact in debug_artifacts:
            if isinstance(artifact, str):
                safe_artifacts.append(_relative_path(
                    artifact,
                    package_root=root,
                    fallback=f"{task_id}/eval_results/debug/{sample_id}/{Path(artifact).name}",
                ))
            else:
                safe_artifacts.append(artifact)
        debug_artifacts = safe_artifacts
    physics_pass = _bool_value(raw.get("physics_pass"))
    physics_reasons = _jsonable(raw.get("physics_failure_reasons", []))
    safe_raw_metrics = _safe_payload(raw_metrics, package_root=root, task_id=task_id, sample_id=sample_id)
    safe_measurements = _safe_payload(raw_measurements, package_root=root, task_id=task_id, sample_id=sample_id)
    safe_diagnostics = _safe_payload(raw_diagnostics, package_root=root, task_id=task_id, sample_id=sample_id)
    status = {
        "sample_id": sample_id,
        "extract_success": None if extract is None else video_extract,
        "physics_pass": physics_pass,
        "physics_failure_reasons": physics_reasons,
        "source": source,
        "image_variant": image_variant,
        "image_id": image_id,
        "video_metadata": _safe_payload(raw_measurements.get("video_metadata", {}), package_root=root, task_id=task_id, sample_id=sample_id),
    }
    raw_verbose = raw.get("verbose") if isinstance(raw.get("verbose"), Mapping) else {}
    # Keep pre-existing task diagnostics (for example video QA) without
    # allowing them to alter the public metric values.
    if isinstance(raw_verbose, Mapping):
        extra = {k: v for k, v in raw_verbose.items() if k not in {"raw_metrics", "measurements", "diagnostics", "status"}}
    else:
        extra = {}
    verbose: dict[str, Any] = {
        "M1": {
            "logic": task["m1_logic"],
            "formula": task["m1_formula"],
            "metric": m1_wrapper,
            "proxy": {
                "version": PROXY_VERSION,
                "aggregation": "single component (or task-specific continuous margin)",
                "components": m1_proxy_components,
            },
            "measurements": safe_measurements,
            "debug_artifacts": debug_artifacts,
        },
        "M2": {
            "logic": task["m2_logic"],
            "declared_components": list(task["m2_keys"]),
            "metric": m2_wrapper,
            "proxy": {
                "version": PROXY_VERSION,
                "aggregation": (
                    "three_ray_coverage * geometric_mean(per_ray_proxy_scores); unavailable -> 0"
                    if task_id == "P12"
                    else "geometric mean of declared component proxies; unavailable -> 0"
                ),
                "components": m2_proxy_components,
            },
            "measurements": safe_measurements,
            "debug_artifacts": debug_artifacts,
        },
        "raw_metrics": safe_raw_metrics,
        "measurements": safe_measurements,
        "diagnostics": safe_diagnostics,
        "status": status,
        "debug_artifacts": debug_artifacts,
    }
    verbose.update(_safe_payload(extra, package_root=root, task_id=task_id, sample_id=sample_id))
    return _jsonable({
        "task_id": task_id,
        "video_path": rel_video,
        "image_path": rel_image,
        "video_prompt": video_prompt,
        "model": model,
        "seed": seed,
        "metrics": {"M1": m1_wrapper, "M2": m2_wrapper},
        "proxy": {
            "proxy_version": PROXY_VERSION,
            "contract_version": CONTRACT_VERSION,
            "overall_proxy_score": overall_proxy_score,
            "proxy_valid": bool(m1_wrapper["proxy_valid"] and m2_wrapper["proxy_valid"]),
            "formula": "sqrt(M1_proxy_score*M2_proxy_score)",
        },
        "verbose": verbose,
    })


def validate_public_result(sample: Mapping[str, Any]) -> list[str]:
    """Return contract violations; no filesystem access is required."""

    errors: list[str] = []
    required = {"task_id", "video_path", "image_path", "video_prompt", "model", "seed", "metrics", "proxy", "verbose"}
    actual = set(sample) if isinstance(sample, Mapping) else set()
    if actual != required:
        errors.append(f"top-level keys must be {sorted(required)}; got {sorted(actual)}")
        return errors
    task_id = sample.get("task_id")
    if task_id not in TASK_METRICS:
        errors.append(f"unsupported task_id: {task_id}")
    for key in ("video_path", "image_path"):
        value = sample.get(key)
        if not isinstance(value, str) or not value or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[/\\]", value) or "\\" in value:
            errors.append(f"{key} must be a non-empty POSIX relative path")
    if not isinstance(sample.get("video_prompt"), str) or not sample["video_prompt"].strip():
        errors.append("video_prompt must be a non-empty string")
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"M1", "M2"}:
        errors.append("metrics must contain exactly M1 and M2")
    else:
        for key in ("M1", "M2"):
            wrapper = metrics[key]
            if not isinstance(wrapper, Mapping) or set(wrapper) != {"extract_success", "metric", "proxy_score", "proxy_valid"}:
                errors.append(f"metrics.{key} must contain extract_success, metric, proxy_score and proxy_valid")
                continue
            if wrapper["extract_success"] not in (True, False, None):
                errors.append(f"metrics.{key}.extract_success must be true, false, or null")
            if wrapper["extract_success"] is False and wrapper["metric"] is not None:
                errors.append(f"metrics.{key}.metric must be null when extraction fails")
            proxy_score = _finite(wrapper.get("proxy_score"))
            if proxy_score is None or not 0.0 <= proxy_score <= 1.0:
                errors.append(f"metrics.{key}.proxy_score must be finite and in [0,1]")
            if wrapper["extract_success"] is False and proxy_score != 0.0:
                errors.append(f"metrics.{key}.proxy_score must be 0 when extraction fails")
            if not isinstance(wrapper.get("proxy_valid"), bool):
                errors.append(f"metrics.{key}.proxy_valid must be boolean")
            if wrapper["extract_success"] is False and wrapper.get("proxy_valid") is not False:
                errors.append(f"metrics.{key}.proxy_valid must be false when extraction fails")
            if wrapper.get("proxy_valid") is False and proxy_score != 0.0:
                errors.append(f"metrics.{key}.proxy_score must be 0 when proxy_valid is false")
    proxy = sample.get("proxy")
    if not isinstance(proxy, Mapping) or set(proxy) != {"proxy_version", "contract_version", "overall_proxy_score", "proxy_valid", "formula"}:
        errors.append("proxy must contain proxy_version, contract_version, overall_proxy_score, proxy_valid and formula")
    else:
        overall = _finite(proxy.get("overall_proxy_score"))
        if proxy.get("proxy_version") != PROXY_VERSION:
            errors.append(f"proxy.proxy_version must be {PROXY_VERSION}")
        if proxy.get("contract_version") != CONTRACT_VERSION:
            errors.append(f"proxy.contract_version must be {CONTRACT_VERSION}")
        if proxy.get("formula") != "sqrt(M1_proxy_score*M2_proxy_score)":
            errors.append("proxy.formula must be sqrt(M1_proxy_score*M2_proxy_score)")
        if overall is None or not 0.0 <= overall <= 1.0:
            errors.append("proxy.overall_proxy_score must be finite and in [0,1]")
        if not isinstance(proxy.get("proxy_valid"), bool):
            errors.append("proxy.proxy_valid must be boolean")
        if proxy.get("proxy_valid") is False and overall != 0.0:
            errors.append("proxy.overall_proxy_score must be 0 when proxy_valid is false")
    verbose = sample.get("verbose")
    if not isinstance(verbose, Mapping):
        errors.append("verbose must be an object")
    else:
        for key in ("M1", "M2"):
            block = verbose.get(key)
            if not isinstance(block, Mapping) or not isinstance(block.get("logic"), str) or not block.get("logic"):
                errors.append(f"verbose.{key} must include non-empty logic")
    return errors


__all__ = ["TASK_METRICS", "PROXY_VERSION", "CONTRACT_VERSION", "PROXY_ANCHORS", "build_public_result", "validate_public_result"]
