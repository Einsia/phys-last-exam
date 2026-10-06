"""Result container, the JSON shape every task emits, and the 0..1 proxy score.

The delivery contract fixes the output: numbered metric slots M1, M2, ..., each
carrying its raw measurement plus the scores derived from it, and a ``verbose``
block that spells out how the number was obtained. One slot holds one scalar,
so where the benchmark lists several auxiliary quantities for a task they
occupy successive slots (M2, M3, ...) instead of being packed into one value.

Two rules follow from the contract and are enforced here rather than left to
each task:

* ``metric`` is null whenever ``extract_success`` is false,
* a slot the benchmark does not define for this task reports
  ``extract_success: null``, which is different from a defined metric that could
  not be extracted (false).

Scoring follows SCORING_RULES_V2 (``physical-bench-proxy-v2-recognition015-
arithmetic``):

* pure physics score ``q(e; a) = 1 / (1 + |e| / a)`` for residual quantities,
  each slot declaring its own error scale ``a``; quantities that are already a
  0..1 fraction, coverage or confidence keep their own mapping;
* per metric ``S = 0.15 + 0.85 q`` when it was measured, ``0`` when it was not,
  the 0.15 being credit for having obtained a usable measurement at all;
* the auxiliary quantities are sub-items of the benchmark's M2 entry: their
  ``q`` values combine by equal-weight geometric mean and the recognition
  credit is added once, after that mean, not per sub-item;
* the video's score is ``0.5 S_M1 + 0.5 S_M2``, an unmeasured half
  contributing 0 without the other half being renormalised.

The one case V2 leaves open is a task for which the benchmark defines no
auxiliary quantity at all - P5 here. Halving such a video's score for an entry
that does not exist would measure our paperwork rather than the video, so its
total is M1 alone and ``score_status`` says ``m1_only``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# SCORING_RULES_V2: a measurable metric earns this much for being measurable at
# all, and the remaining 0.85 is scaled by the pure physics score. Added once
# per M slot, never per sub-item.
RECOGNITION = 0.15


@dataclass
class Metric:
    """One metric slot.

    ``defined=False`` means the benchmark lists no such metric for this task, so
    ``extract_success`` serialises as null instead of a boolean.
    """

    name: str = ""
    # Short benchmark wording for this slot, so the JSON says what it measures.
    label: str = ""
    defined: bool = True
    extract_success: bool = False
    value: float | None = None
    # --- the 0..1 proxy, SCORING_RULES_V2 -----------------------------------
    # The benchmark's metrics are deviations in whatever unit each one is
    # naturally written in, so their raw values cannot be compared or pooled
    # across tasks. Each slot declares where perfection sits and the error
    # scale ``a``, and the pure physics score follows from those two numbers:
    #     q(e; a) = 1 / (1 + |e| / a)
    ideal: float = 0.0
    # V2's error scale: the deviation from ``ideal`` at which q is 0.5. It is a
    # property of the physics being tested, not of any model's results, and V2
    # is explicit that it must not be re-tuned to the batch being scored - do
    # that and the score stops meaning "how right is this" and starts meaning
    # "how does this rank among the runs we happen to have".
    a: float = 0.10
    unit: str = ""
    # A few quantities are not residuals - a coverage, a fraction, a detection
    # confidence - and V2 keeps their own mapping rather than forcing them
    # through q(e; a). For those, the measured value is already a 0..1 score.
    non_residual: bool = False
    # Free-form record of how the number was produced; goes into verbose.
    principle: str = ""
    steps: list[str] = field(default_factory=list)
    quantities: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    @property
    def physics_score(self) -> float | None:
        """V2's ``q``: physics only, no recognition bonus. None if unmeasured."""
        if not self.defined or not self.extract_success or self.value is None:
            return None
        v = float(self.value)
        if self.non_residual:
            return min(1.0, max(0.0, v))
        return 1.0 / (1.0 + abs(v - self.ideal) / self.a)

    @property
    def score(self) -> float | None:
        """V2's ``S``: ``0.15 + 0.85 q`` when measured, 0 when not.

        The 0.15 is V2's recognition credit: getting a usable measurement out
        of a video is itself worth something, separately from whether the
        physics in it was right. A video can be physically wrong and still earn
        it; a video whose object is visible but whose period or angle cannot be
        measured does not.
        """
        if not self.defined:
            return None
        q = self.physics_score
        return 0.0 if q is None else RECOGNITION + (1.0 - RECOGNITION) * q

    def succeed(self, value: float, **quantities: Any) -> None:
        self.extract_success = True
        self.value = float(value)
        self.quantities.update(quantities)

    def fail(self, reason: str, **quantities: Any) -> None:
        self.extract_success = False
        self.value = None
        self.note = reason
        self.quantities.update(quantities)

    def to_metric_json(self) -> dict[str, Any]:
        if not self.defined:
            # Same keys as every other slot, so a consumer never has to branch
            # on which keys are present.
            return {"extract_success": None, "metric": None,
                    "physics_score": None, "recognition_score": None,
                    "proxy_score": None, "proxy_valid": None}
        ok = bool(self.extract_success)
        q = self.physics_score
        return {"extract_success": ok,
                "metric": None if not ok else float(self.value),
                "physics_score": None if q is None else q,
                "recognition_score": RECOGNITION if ok else 0.0,
                "proxy_score": float(self.score),
                "proxy_valid": ok}

    def _rule(self) -> str:
        if not self.extract_success:
            return ("提取失败：physics_score 记 null，不给识别分，proxy_score = 0"
                    "（SCORING_RULES_V2 §3.1）")
        if self.non_residual:
            return ("本量本身即 0~1 的比例/覆盖率/置信度，按 V2 §2.1 保留其原有"
                    f"映射，physics_score 直接取该值；proxy_score = "
                    f"{RECOGNITION:g} + {1 - RECOGNITION:g} × physics_score")
        return (f"physics_score = 1 / (1 + |metric − {self.ideal:g}| / "
                f"{self.a:g})，误差达 {self.a:g}"
                f"{self.unit and ' ' + self.unit} 时为 0.5；"
                f"proxy_score = {RECOGNITION:g} + {1 - RECOGNITION:g} × "
                "physics_score（SCORING_RULES_V2 §2.1、§2.2）")

    def to_verbose_json(self) -> dict[str, Any]:
        if not self.defined:
            return {"defined": False,
                    "reason": self.note or
                    f"the benchmark defines no {self.name or 'such'} metric "
                    "for this task"}
        out: dict[str, Any] = {
            "defined": True,
            "measures": self.label,
            "extract_success": bool(self.extract_success),
            "metric": None if self.value is None else float(self.value),
            "unit": self.unit or "dimensionless",
            "error_scale_a": self.a,
            "ideal": self.ideal,
            "non_residual": self.non_residual,
            "physics_score": (None if self.physics_score is None
                              else self.physics_score),
            "proxy_score": float(self.score),
            "score_rule": self._rule(),
            "principle": self.principle,
            "measurement_steps": self.steps,
            "quantities": _round(self.quantities),
        }
        if self.note:
            out["note"] = self.note
        return out


def _round(obj: Any) -> Any:
    """Round floats for readability; long series are summarised, not dumped."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_round(v) for v in obj]
        if len(vals) > 24 and all(isinstance(v, (int, float)) for v in vals):
            return {"n": len(vals), "first": vals[:6], "last": vals[-6:]}
        return vals
    return obj


@dataclass
class Result:
    task_id: str
    video_path: str
    image_path: str
    video_prompt: str
    model: str
    seed: int | None
    # Slots in benchmark order: M1 primary, M2.. auxiliary.
    metrics: dict[str, Metric] = field(default_factory=dict)
    # Path of the figure a human reads the measured quantity off.
    debug_image: str | None = None
    scene: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, label: str = "", principle: str = "",
            defined: bool = True, ideal: float = 0.0, tol: float = 0.10,
            unit: str = "", non_residual: bool = False) -> Metric:
        m = Metric(name=name, label=label, principle=principle,
                   defined=defined, ideal=ideal, a=tol, unit=unit,
                   non_residual=non_residual)
        self.metrics[name] = m
        return m

    # ---- SCORING_RULES_V2 aggregation --------------------------------------
    @property
    def aux(self) -> list[Metric]:
        """The auxiliary quantities, i.e. everything the benchmark hangs off M2.

        The benchmark lists one auxiliary entry per task, but for several tasks
        that entry names more than one quantity. Those are kept as separate
        slots so each is reported with its own error and scale; V2 treats them
        as sub-items of M2 and combines them, which is what happens here.
        """
        return [m for k, m in self.metrics.items() if k != "M1" and m.defined]

    @property
    def m2_physics(self) -> float | None:
        """``q_M2``: equal-weight geometric mean of the sub-items' q.

        None when the auxiliary entry is undefined for this task, or when any
        sub-item it needs could not be measured - V2 does not let a missing
        sub-item be dropped to raise the mean.
        """
        subs = self.aux
        if not subs:
            return None
        qs = [m.physics_score for m in subs]
        if any(q is None for q in qs):
            return None
        if any(q <= 0.0 for q in qs):
            return 0.0
        return math.exp(sum(math.log(q) for q in qs) / len(qs))

    @property
    def m2_score(self) -> float | None:
        """``S_M2``: the recognition bonus added once, after the mean."""
        if not self.aux:
            return None
        q = self.m2_physics
        return 0.0 if q is None else RECOGNITION + (1.0 - RECOGNITION) * q

    @property
    def score_status(self) -> str:
        m1_ok = (self.metrics.get("M1") or Metric(defined=False)).extract_success
        if not self.aux:
            # The benchmark names no auxiliary quantity for this task, so there
            # is no M2 to measure. Charging the video half its marks for an
            # entry the benchmark never defined would say nothing about the
            # video, so the total is M1 alone; V2 covers M2 unmeasured, not M2
            # undefined.
            return "m1_only" if m1_ok else "unavailable"
        m2_ok = self.m2_physics is not None
        if m1_ok and m2_ok:
            return "complete"
        if m1_ok or m2_ok:
            return "partial"
        return "unavailable"

    @property
    def score(self) -> float | None:
        """``S_total``: V2's equal-weight mean of the two metric scores.

        An unmeasured half contributes 0 and the other half keeps its 0.5
        weight rather than being renormalised, so a video that yields only one
        of the two metrics can score at most 0.5.
        """
        m1 = self.metrics.get("M1")
        if m1 is None:
            return None
        if not self.aux:
            return m1.score
        return 0.5 * (m1.score or 0.0) + 0.5 * (self.m2_score or 0.0)

    def fail_all(self, reason: str) -> None:
        """Mark every defined slot unextractable with the same reason."""
        for m in self.metrics.values():
            if m.defined:
                m.fail(reason)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "video_path": self.video_path,
            "image_path": self.image_path,
            "video_prompt": self.video_prompt,
            "model": self.model,
            "seed": self.seed,
            # The one number for this video, 0 worst to 1 best, per
            # SCORING_RULES_V2 §2.3.
            "score": None if self.score is None else round(self.score, 6),
            "proxy": self._proxy_json(),
            "metrics": {k: m.to_metric_json() for k, m in self.metrics.items()},
            "verbose": {k: m.to_verbose_json() for k, m in self.metrics.items()},
        }
        if self.debug_image:
            payload["verbose"]["debug_image"] = self.debug_image
        if self.scene:
            payload["verbose"]["scene"] = _round(self.scene)
        return payload

    def _proxy_json(self) -> dict[str, Any]:
        """The V2 scoring block: both halves, how they combined, and coverage."""
        m1 = self.metrics.get("M1")
        subs = self.aux
        status = self.score_status
        out: dict[str, Any] = {
            "rules": "SCORING_RULES_V2 (physical-bench-proxy-v2-"
                     "recognition015-arithmetic)",
            "recognition_score_weight": RECOGNITION,
            "M1_physics_score": (None if m1 is None or m1.physics_score is None
                                 else round(m1.physics_score, 6)),
            "M1_proxy_score": (None if m1 is None else
                               round(float(m1.score or 0.0), 6)),
            "M2_physics_score": (None if self.m2_physics is None
                                 else round(self.m2_physics, 6)),
            "M2_proxy_score": (None if self.m2_score is None
                               else round(self.m2_score, 6)),
            "score_status": status,
            "overall_proxy_valid": status == "complete",
            "total_rule": ("S_total = 0.5*S_M1 + 0.5*S_M2" if subs else
                           "S_total = S_M1（题包未给本题定义辅助指标，"
                           "不为一个不存在的 M2 扣掉一半分）"),
        }
        if subs:
            out["M2_sub_items"] = [
                {"slot": m.name, "measures": m.label,
                 "metric": None if m.value is None else round(float(m.value), 6),
                 "error_scale_a": m.a,
                 "physics_score": (None if m.physics_score is None
                                   else round(m.physics_score, 6))}
                for m in subs]
            out["M2_combination"] = (
                "q_M2 = 等权几何平均(各子项 physics_score)，识别分只在 M2 层加"
                "一次（SCORING_RULES_V2 §2.2、§3.2）" if len(subs) > 1 else
                "本题辅助指标只有一个子项，q_M2 即该子项的 physics_score")
        return out

    def write(self, path: str | Path, sample_id: str | None = None,
              route: str | None = None) -> Path:
        """Write the verdict, optionally stamped with which sample it is.

        The delivery names samples ``sample_00`` upwards, so the JSON has to
        say which first-frame route and seed that number stands for; passing
        them here keeps a file written by ``run_eval.sh`` identical in shape to
        the one shipped in the package.
        """
        payload = self.to_json()
        if sample_id is not None or route is not None:
            head = {"task_id": payload["task_id"]}
            if sample_id is not None:
                head["sample_id"] = sample_id
            if route is not None:
                head["route"] = route
            head.update({k: v for k, v in payload.items() if k != "task_id"})
            payload = head
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                     encoding="utf-8")
        return p

    def summary(self) -> str:
        def cell(m: Metric) -> str:
            if not m.defined:
                return "n/a"
            if not m.extract_success:
                return "FAIL"
            return f"{m.value:.4f}"
        cells = "  ".join(f"{k}={cell(m)}" for k, m in self.metrics.items())
        return f"{self.task_id:6} {cells}"

    def first_failure(self) -> str:
        for m in self.metrics.values():
            if m.defined and not m.extract_success and m.note:
                return m.note
        return ""
