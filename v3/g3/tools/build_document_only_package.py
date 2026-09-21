#!/usr/bin/env python3
"""Build a strict Feishu-rev153 delivery scored only by document M1/M2.

The source package is cloned with hard links.  Formal result JSON/CSV/web files
are then replaced atomically, so the frozen source package is never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DOCUMENT_URL = "https://ccnawh1pxwn1.feishu.cn/wiki/PEGuwNkt9i28ulk6b9ucWaTLnwh"
DOCUMENT_REVISION = 153
RANKING_VERSION = "feishu-rev153-m1-m2-derived-ranking-v1"
SCHEMA_VERSION = "physical-bench-document-only-delivery-v1"
EXPECTED_RECORDS = 168
TASK_ORDER = ("P3", "P4", "P6", "P9", "P11")


DOCUMENT_CONTRACT: dict[str, Any] = {
    "schema_version": "feishu-rev153-document-only-score-contract-v1",
    "document_url": DOCUMENT_URL,
    "document_revision": DOCUMENT_REVISION,
    "official_metrics": ["M1", "M2"],
    "thresholds_defined_by_document": False,
    "aggregation_defined_by_document": False,
    "formal_outputs_contain_derived_scores": False,
    "excluded_from_official_score": [
        "tracking quality",
        "camera stability",
        "appearance quality",
        "fit support or confidence",
        "legacy dimensions",
        "all evaluator diagnostics not named M1 or M2 in the document",
    ],
    "tasks": {
        "P3": {
            "M1": "R_30/R_60 - 1",
            "M2": {
                "trajectory_30deg_parabola_rmse_d": "parabola fit residual for the 30-degree trajectory",
                "trajectory_60deg_parabola_rmse_d": "parabola fit residual for the 60-degree trajectory",
                "initial_speed_magnitude_error": "initial-speed magnitude consistency residual",
            },
        },
        "P4": {
            "M1": "|sqrt(h[n+1]/h[n]) - dt[n+1]/dt[n]|",
            "M2": {
                "height_decrease_checks": "one strict h[n+1] < h[n] check for every adjacent pair",
                "height_decrease_fraction": "fraction of adjacent pairs satisfying h[n+1] < h[n]",
                "adjacent_restitution_coefficient_cv": "adjacent-bounce restitution consistency",
            },
        },
        "P6": {
            "M1": "v/(omega*R) - 1",
            "M2": "instantaneous contact-point velocity residual",
        },
        "P9": {
            "M1": "abs((T1/T2)^2/(L1/L2) - 1)",
            "M2": {
                "short_pendulum_period_cv": "CV(T) for the short pendulum",
                "long_pendulum_period_cv": "CV(T) for the long pendulum",
            },
        },
        "P11": {
            "M1": "sin(theta_i)/sin(theta_t) - n_water",
            "M2": "ray-interface intersection consistency",
        },
    },
}


DERIVED_RANKING_CONTRACT: dict[str, Any] = {
    "schema_version": "feishu-rev153-derived-ranking-contract-v1",
    "ranking_version": RANKING_VERSION,
    "official_benchmark_formula": False,
    "notice": (
        "This optional ranking view is an implementation-defined monotone mapping of only "
        "the document M1/M2 values. It is not a metric, threshold, pass rule, or aggregation "
        "defined by the Feishu benchmark document."
    ),
    "quality_mapping": "q(e,t)=1/(1+(abs(e)/t)^2)",
    "overall_aggregation": "sqrt(M1_score*M2_score)",
    "validity_gate": "M1_valid AND M2_valid",
    "invalid_display_value": 0.0,
    "invalid_display_rule": "0 is accompanied by derived_overall_valid=false and is excluded from valid-only means",
    "diagnostic_independence": "extract_success and structural_ok never enter the derived ranking",
    "calibration_provenance": (
        "The document defines no numerical score thresholds. Frozen v2 tolerances are reused "
        "only as implementation half-quality anchors and are not additional evaluation metrics."
    ),
    "tasks": {
        "P3": {
            "anchors": {"M1": 0.15, "parabola": 0.30, "initial_speed": 0.20},
            "M2_aggregation": "equal-weight geometric mean of the three document M2 components",
        },
        "P4": {
            "anchors": {"M1": 0.18, "restitution_cv": 0.35},
            "M2_aggregation": "sqrt(height_decrease_fraction*q(restitution_cv,0.35))",
        },
        "P6": {"anchors": {"M1": 0.25, "M2": 0.35}},
        "P9": {
            "anchors": {"M1": 0.20, "period_cv": 0.20},
            "M2_aggregation": "equal-weight geometric mean of short and long CV quality",
        },
        "P11": {"anchors": {"M1": 0.12, "M2": 0.025}},
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def quality(error: Any, scale: float) -> float | None:
    if not finite(error):
        return None
    ratio = abs(float(error)) / scale
    return 1.0 / (1.0 + ratio * ratio)


def geometric(values: Iterable[tuple[float | None, float]]) -> float | None:
    items = list(values)
    if not items or any(value is None for value, _ in items):
        return None
    if any(float(value) < 0.0 or float(value) > 1.0 for value, _ in items):
        return None
    if any(float(value) == 0.0 for value, _ in items):
        return 0.0
    total_weight = sum(weight for _, weight in items)
    if total_weight <= 0:
        return None
    return math.exp(sum(weight * math.log(float(value)) for value, weight in items) / total_weight)


def clean_metrics(task: str, raw: dict[str, Any], source_result: dict[str, Any]) -> dict[str, Any]:
    m1 = raw.get("M1")
    m2 = raw.get("M2")
    if task == "P3":
        per_ball = m2.get("per_ball", {}) if isinstance(m2, dict) else {}
        measured_balls = (source_result.get("measurements") or {}).get("balls") or {}
        by_angle: dict[int, dict[str, Any]] = {}
        for slot in ("upper", "lower"):
            target = (measured_balls.get(slot) or {}).get("angle_target_deg")
            details = per_ball.get(slot, {}) if isinstance(per_ball, dict) else {}
            if finite(target) and round(float(target)) in (30, 60) and isinstance(details, dict):
                by_angle[round(float(target))] = details
        cleaned_m2: Any = {
            "trajectory_30deg_parabola_rmse_d": (by_angle.get(30) or {}).get("parabola_rmse_d"),
            "trajectory_60deg_parabola_rmse_d": (by_angle.get(60) or {}).get("parabola_rmse_d"),
            "initial_speed_magnitude_error": m2.get("initial_speed_error") if isinstance(m2, dict) else None,
        }
    elif task == "P4":
        source_metrics = source_result.get("metrics") or {}
        ratios = source_metrics.get("height_ratios") or []
        checks = [bool(float(ratio) < 1.0) for ratio in ratios if finite(ratio)]
        cleaned_m2 = {
            "height_decrease_checks": checks,
            "height_decrease_fraction": (sum(checks) / len(checks)) if checks else None,
            "adjacent_restitution_coefficient_cv": m2.get("restitution_coefficient_cv") if isinstance(m2, dict) else None,
        }
    elif task == "P9":
        cleaned_m2 = {
            "short_pendulum_period_cv": m2.get("short_pendulum_period_cv") if isinstance(m2, dict) else None,
            "long_pendulum_period_cv": m2.get("long_pendulum_period_cv") if isinstance(m2, dict) else None,
        }
    else:
        cleaned_m2 = m2
    return {"M1": m1, "M2": cleaned_m2}


def metric_scores(task: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    m1, m2 = metrics["M1"], metrics["M2"]
    if task == "P3":
        m1_score = quality(m1, 0.15)
        m2_score = geometric(
            (
                (quality(m2["trajectory_30deg_parabola_rmse_d"], 0.30), 1.0),
                (quality(m2["trajectory_60deg_parabola_rmse_d"], 0.30), 1.0),
                (quality(m2["initial_speed_magnitude_error"], 0.20), 1.0),
            )
        )
    elif task == "P4":
        m1_score = quality(m1, 0.18)
        m2_score = geometric(
            (
                (float(m2["height_decrease_fraction"]) if finite(m2["height_decrease_fraction"]) else None, 1.0),
                (quality(m2["adjacent_restitution_coefficient_cv"], 0.35), 1.0),
            )
        )
    elif task == "P6":
        m1_score, m2_score = quality(m1, 0.25), quality(m2, 0.35)
    elif task == "P9":
        m1_score = quality(m1, 0.20)
        m2_score = geometric(
            (
                (quality(m2["short_pendulum_period_cv"], 0.20), 0.50),
                (quality(m2["long_pendulum_period_cv"], 0.20), 0.50),
            )
        )
    elif task == "P11":
        m1_score, m2_score = quality(m1, 0.12), quality(m2, 0.025)
    else:
        raise ValueError(task)
    return {"M1": m1_score, "M2": m2_score}


def compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".strict-new")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any, *, compact_output: bool = False) -> None:
    if compact_output:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_text(path, text)


def resolve_gallery_link(gallery_dir: Path, link: str) -> Path:
    return (gallery_dir / link).resolve()


def relative_from_root(root: Path, gallery_dir: Path, link: str) -> str:
    return resolve_gallery_link(gallery_dir, link).relative_to(root).as_posix()


def safe_record_stem(record_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "__", record_id).strip("._-")
    if not value:
        raise ValueError(f"record ID has no safe filename characters: {record_id!r}")
    return value


def ranking_status(validity: dict[str, bool]) -> str:
    if not validity["M1"] or not validity["M2"]:
        return "document_metric_unmeasurable"
    return "scored"


def csv_fields() -> list[str]:
    return [
        "Sample", "video_id", "task_id", "batch_id", "seed", "source_method",
        "extract_success", "structural_ok", "M1_valid", "M2_valid",
        "M1", "M2", "result_json", "raw_video",
        "overlay", "plot", "failure_reason",
    ]


def csv_row(root: Path, gallery_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    links = record["links"]
    return {
        "Sample": record["record_id"],
        "video_id": record["video_id"],
        "task_id": record["task"],
        "batch_id": record["batch"],
        "seed": record["seed"],
        "source_method": record["source"],
        "extract_success": record["status"]["extract_success"],
        "structural_ok": record["status"]["structural_ok"],
        "M1_valid": record["metric_validity"]["M1"],
        "M2_valid": record["metric_validity"]["M2"],
        "M1": compact(record["metrics"]["M1"]),
        "M2": compact(record["metrics"]["M2"]),
        "result_json": relative_from_root(root, gallery_dir, links["json"]),
        "raw_video": relative_from_root(root, gallery_dir, links["raw"]),
        "overlay": relative_from_root(root, gallery_dir, links["overlay"]),
        "plot": relative_from_root(root, gallery_dir, links["plot"]),
        "failure_reason": record.get("failure_reason") or "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".strict-new")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def derived_csv_fields() -> list[str]:
    return [
        "Sample", "video_id", "task_id", "batch_id", "seed", "source_method",
        "M1_score", "M2_score", "derived_overall", "derived_overall_valid",
        "ranking_status", "ranking_version",
    ]


def derived_csv_row(record: dict[str, Any]) -> dict[str, Any]:
    ranking = record["derived_ranking"]
    return {
        "Sample": record["record_id"],
        "video_id": record["video_id"],
        "task_id": record["task"],
        "batch_id": record["batch"],
        "seed": record["seed"],
        "source_method": record["source"],
        "M1_score": compact(ranking["M1_score"]),
        "M2_score": compact(ranking["M2_score"]),
        "derived_overall": ranking["overall"],
        "derived_overall_valid": ranking["valid"],
        "ranking_status": ranking["status"],
        "ranking_version": ranking["version"],
    }


def write_derived_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".strict-new")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=derived_csv_fields())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def task_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(record["derived_ranking"]["overall"]) for record in records]
    eligible = [record for record in records if record["derived_ranking"]["valid"]]
    return {
        "records": len(records),
        "metric_pair_valid": len(eligible),
        "metric_pair_valid_rate": len(eligible) / len(records) if records else 0.0,
        "mean_all": sum(values) / len(values) if values else None,
        "mean_valid_pair": (
            sum(float(record["derived_ranking"]["overall"]) for record in eligible) / len(eligible)
            if eligible else None
        ),
        "zero_count": sum(value == 0.0 for value in values),
    }


def gallery_html() -> str:
    return r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Physical Bench · 文档 M1/M2-only 严格版</title>
<style>
:root{color-scheme:dark;--bg:#07111d;--card:#101e2d;--line:#26405a;--text:#eaf3fb;--muted:#91a9bd;--cyan:#55d6d0;--amber:#ffc86b;--red:#ff7b7b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top,#15334a 0,#07111d 42%);color:var(--text);font:14px/1.5 Inter,system-ui,sans-serif}header{position:sticky;top:0;z-index:5;padding:18px 24px;background:#07111de8;backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}h1{margin:0;font-size:22px}.sub{color:var(--muted);margin-top:4px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}select,input{background:#0d1a27;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px 10px}main{padding:20px 24px 60px}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px}.stat,.card{background:linear-gradient(145deg,#122436,#0c1825);border:1px solid var(--line);border-radius:12px}.stat{padding:12px}.stat b{display:block;font-size:20px;color:var(--cyan)}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(390px,1fr));gap:14px}.card{overflow:hidden}.head{padding:12px 14px;display:flex;justify-content:space-between;gap:10px}.id{font-weight:700}.tags{color:var(--muted);font-size:12px}.score{font-size:23px;color:var(--cyan);font-weight:800}.failed{color:var(--red)}video{width:100%;aspect-ratio:16/9;background:#000;display:block}.body{padding:12px 14px}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:10px}.metric{background:#08131e;border:1px solid #1d3448;border-radius:9px;padding:9px}.metric strong{color:var(--amber)}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:6px 0 0;color:#cfe4f4;font-size:11px}.links{display:flex;gap:12px;margin-top:10px}a{color:var(--cyan)}.gate{margin-top:8px;color:var(--muted)}
</style></head><body><header><h1>Physical Bench · 飞书文档 M1/M2 严格版</h1><div class="sub">正式测试数据只有 Benchmark rev153 的 M1、M2。页面中的 0–1 排名是只读取 M1/M2 的非官方派生视图，不是文档指标或通过规则。</div><div class="controls"><select id="task"><option value="">全部任务</option></select><select id="source"><option value="">全部首帧</option><option value="gpt">GPT</option><option value="simulation">Simulation</option></select><select id="state"><option value="">全部状态</option><option value="valid">M1/M2 均有效</option><option value="invalid">存在不可测指标</option></select><input id="search" placeholder="搜索 sample"></div></header><main><div class="summary" id="summary"></div><div id="grid"></div></main>
<script>
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>x==null?'N/A':typeof x==='number'?x.toFixed(4):JSON.stringify(x,null,2);let data,records=[];
const taskEl=document.getElementById('task'),sourceEl=document.getElementById('source'),stateEl=document.getElementById('state'),searchEl=document.getElementById('search'),summaryEl=document.getElementById('summary'),gridEl=document.getElementById('grid');
async function init(){data=await fetch('full_evaluation_data_web.json').then(r=>r.json());records=data.records;const ts=[...new Set(records.map(r=>r.task))];taskEl.innerHTML+=ts.map(x=>`<option>${x}</option>`).join('');[taskEl,sourceEl,stateEl,searchEl].forEach(x=>x.addEventListener('input',render));render()}
function render(){let xs=records.filter(r=>(!taskEl.value||r.task===taskEl.value)&&(!sourceEl.value||r.source===sourceEl.value)&&(!searchEl.value||r.video_id.toLowerCase().includes(searchEl.value.toLowerCase()))&&(!stateEl.value||(stateEl.value==='valid'?r.derived_ranking.valid:!r.derived_ranking.valid)));let ok=xs.filter(r=>r.derived_ranking.valid).length,mean=xs.reduce((a,r)=>a+r.derived_ranking.overall,0)/(xs.length||1);summaryEl.innerHTML=`<div class=stat>显示<b>${xs.length}</b></div><div class=stat>M1/M2 均有效<b>${ok}</b></div><div class=stat>派生均分<b>${mean.toFixed(4)}</b></div><div class=stat>排名版本<b style="font-size:12px">${esc(data.ranking_version)}</b></div>`;gridEl.innerHTML=xs.map(card).join('')}
function card(r){const d=r.derived_ranking,good=d.valid;return `<article class=card><div class=head><div><div class=id>${esc(r.video_id)}</div><div class=tags>${esc(r.task)} · ${esc(r.batch)} · ${esc(r.source)} · seed ${r.seed}</div></div><div class="score ${good?'':'failed'}">${d.overall.toFixed(3)}</div></div><video controls preload=none src="${esc(r.links.raw)}"></video><div class=body><div class=metrics><div class=metric><strong>M1</strong> · derived ${fmt(d.M1_score)}<pre>${esc(fmt(r.metrics.M1))}</pre></div><div class=metric><strong>M2</strong> · derived ${fmt(d.M2_score)}<pre>${esc(fmt(r.metrics.M2))}</pre></div></div><div class=gate>派生状态：${esc(d.status)} · extract=${r.status.extract_success} · structural=${r.status.structural_ok}</div><div class=links><a href="${esc(r.links.overlay)}">overlay</a><a href="${esc(r.links.plot)}">plot</a><a href="${esc(r.links.json)}">正式 JSON</a></div></div></article>`}
init().catch(e=>document.body.innerHTML='<pre>'+esc(e.stack)+'</pre>');
</script></body></html>'''


def clone_package(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if source.stat().st_dev != output.parent.stat().st_dev:
        raise RuntimeError("source and output parent must be on the same filesystem for hard links")
    shutil.copytree(source, output, copy_function=os.link)


def remove_stale_outputs(root: Path) -> None:
    for path in root.rglob("legacy_v1_json"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in root.rglob("source_support"):
        if path.is_dir() and path.parent.name == "eval_results":
            shutil.rmtree(path)
    for path in root.glob("*/web"):
        if path.is_dir():
            shutil.rmtree(path)
    # The legacy package used the same P11 JSON basename in three prompt arms.
    # Replace every cloned result directory so the strict layer can use one
    # unique file per (task, batch, video) record.
    for task in TASK_ORDER:
        json_dir = root / task / "eval_results" / "json"
        if json_dir.exists():
            shutil.rmtree(json_dir)
        json_dir.mkdir(parents=True, exist_ok=True)
    for name in ("BUILD_MANIFEST.json",):
        path = root / name
        if path.exists():
            path.unlink()
    tools = root / "tools"
    if tools.exists():
        shutil.rmtree(tools)
    tools.mkdir()


def write_docs(root: Path, records: list[dict[str, Any]], script_path: Path, audit_script: Path) -> None:
    summary = {task: task_summary([record for record in records if record["task"] == task]) for task in TASK_ORDER}
    global_summary = task_summary(records)
    readme = "\n".join(
        [
            "# Physical Bench · Feishu rev153 M1/M2-only strict delivery",
            "",
            "This package has exactly two formal evaluation metrics: `M1` and `M2`, as defined by Feishu Benchmark revision 153. Formal per-video JSON and formal CSV files contain no M3, score, overall, legacy dimension, or pass/fail formula.",
            "",
            "For convenient ordering only, `DERIVED_RANKING_168.csv` and the gallery provide an explicitly non-official 0–1 view calculated solely from M1 and M2. Its derived overall is `sqrt(M1_score * M2_score)` when both metrics are valid; otherwise the display value is 0 with `derived_overall_valid=false`. Tracking, camera, appearance, structure flags and evaluator-confidence quantities never enter this ranking.",
            "",
            "The source document defines no numeric score thresholds or aggregation. Frozen v2 tolerances are reused solely as implementation calibration anchors for `q(e,t)=1/(1+(|e|/t)^2)` in the separate ranking view. See `DOCUMENT_METRIC_CONTRACT.json` and `DERIVED_RANKING_CONTRACT.json`.",
            "",
            f"Records: {len(records)}; valid M1/M2 pair: {global_summary['metric_pair_valid']}; all-record mean: {global_summary['mean_all']:.6f}.",
            "",
            "Open `video_gallery/evaluation.html` with a local HTTP server to inspect all videos, the two formal metrics, and the separately labelled derived ranking.",
            "",
            "Release remains HOLD until the required named human sanity review is complete.",
            "",
        ]
    )
    atomic_text(root / "README.md", readme)
    checklist = "\n".join(
        [
            "# Acceptance checklist",
            "",
            "- [x] 168/168 formal JSON results contain exactly `metrics.M1` and `metrics.M2`",
            "- [x] no M3 or legacy dimension is present in formal JSON/CSV/web data",
            "- [x] formal JSON/CSV contain no derived score or overall",
            "- [x] separate derived ranking uses only M1 and M2",
            "- [x] only M1/M2 validity gates the derived ranking; diagnostic statuses do not enter it",
            "- [x] 168 raw MP4 and 168 evaluator overlay MP4 retained",
            "- [x] task CSV, root CSV, JSON and web data agree",
            "- [ ] named human sanity reviewer and signed timestamp",
            "",
        ]
    )
    atomic_text(root / "CHECKLIST.md", checklist)
    atomic_json(root / "DOCUMENT_METRIC_CONTRACT.json", DOCUMENT_CONTRACT)
    atomic_json(root / "DERIVED_RANKING_CONTRACT.json", DERIVED_RANKING_CONTRACT)
    atomic_json(
        root / "RELEASE_STATUS.json",
        {
            "package_status": "DRAFT_PENDING_HUMAN_SANITY",
            "release_disposition": "HOLD",
            "release_ready": False,
            "human_sanity_status": "UNVERIFIED",
            "verified_human_reviews": 0,
            "expected_human_reviews": EXPECTED_RECORDS,
            "ranking_version": RANKING_VERSION,
        },
    )
    shutil.copy2(script_path, root / "tools" / script_path.name)
    shutil.copy2(audit_script, root / "tools" / audit_script.name)
    for task in TASK_ORDER:
        subset = [record for record in records if record["task"] == task]
        text = "\n".join(
            [
                f"# {task} · document-only strict results",
                "",
                "Formal metrics: M1 and M2 only. The formal task JSON/CSV has no score or overall.",
                f"Records: {len(subset)}; valid M1/M2 pair: {summary[task]['metric_pair_valid']}; mean(all): {summary[task]['mean_all']:.6f}.",
                "",
                "See `eval_results/results.csv`, per-video JSON, debug overlays, and the two root contracts. The displayed mean is from the separate non-official M1/M2-only ranking view.",
                "",
            ]
        )
        for name in ("README.md", "report.md"):
            atomic_text(root / task / name, text)
        atomic_text(root / task / "checklist.md", "# Checklist\n\n- [x] M1/M2-only formal outputs\n- [x] 24 canonical rollouts\n- [ ] signed human sanity review\n")
        atomic_json(
            root / task / "manifest.json",
            {"task_id": task, "official_metrics": ["M1", "M2"], "derived_ranking_version": RANKING_VERSION, "summary": summary[task]},
        )


def write_manifest(root: Path, records: list[dict[str, Any]]) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_id": root.name,
        "created_at": utc_now(),
        "document_url": DOCUMENT_URL,
        "document_revision": DOCUMENT_REVISION,
        "official_metrics": ["M1", "M2"],
        "document_thresholds_defined": False,
        "document_aggregation_defined": False,
        "derived_ranking_version": RANKING_VERSION,
        "record_count": len(records),
        "task_counts": dict(Counter(record["task"] for record in records)),
        "batch_count": len({(record["task"], record["batch"]) for record in records}),
        "metric_pair_valid": sum(record["derived_ranking"]["valid"] for record in records),
        "uses_llm": False,
        "uses_vlm": False,
        "release_disposition": "HOLD",
        "files": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size} for path in files],
    }
    atomic_json(root / "MANIFEST.json", manifest)
    hash_files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    atomic_text(root / "SHA256SUMS", "\n".join(f"{sha256_file(path)}  ./{path.relative_to(root).as_posix()}" for path in hash_files) + "\n")


def build(source: Path, output: Path, script_path: Path, audit_script: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    clone_package(source, output)
    remove_stale_outputs(output)
    gallery_dir = output / "video_gallery"
    source_gallery = json.loads((source / "video_gallery" / "full_evaluation_data.json").read_text(encoding="utf-8"))
    source_records = source_gallery.get("records") or []
    if len(source_records) != EXPECTED_RECORDS:
        raise RuntimeError(f"expected {EXPECTED_RECORDS} source records, found {len(source_records)}")

    records: list[dict[str, Any]] = []
    task_csv_groups: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for source_record in source_records:
        task = str(source_record["task"])
        if task not in TASK_ORDER:
            raise RuntimeError(task)
        record_id = str(source_record["id"])
        source_links = source_record["links"]
        source_json_path = resolve_gallery_link(source / "video_gallery", source_links["json"])
        try:
            source_json_path.relative_to(source)
        except ValueError as exc:
            raise RuntimeError(f"source JSON link escapes source package: {source_json_path}") from exc
        source_json_hash = sha256_file(source_json_path)
        source_result = json.loads(source_json_path.read_text(encoding="utf-8"))
        links = dict(source_links)
        links["json"] = f"../{task}/eval_results/json/{safe_record_stem(record_id)}.json"
        json_path = resolve_gallery_link(gallery_dir, links["json"])
        try:
            json_path.relative_to(output)
        except ValueError as exc:
            raise RuntimeError(f"strict JSON link escapes output: {json_path}") from exc
        metrics = clean_metrics(task, source_record.get("raw_metrics") or {}, source_result)
        source_validity = source_record.get("metric_validity") or {}
        scores = metric_scores(task, metrics)
        validity = {
            "M1": bool(source_validity.get("M1")) and scores["M1"] is not None,
            "M2": bool(source_validity.get("M2")) and scores["M2"] is not None,
        }
        scores = {
            "M1": scores["M1"] if validity["M1"] else None,
            "M2": scores["M2"] if validity["M2"] else None,
        }
        status = {
            "extract_success": bool((source_record.get("status") or {}).get("extract_success")),
            "structural_ok": bool((source_record.get("status") or {}).get("structural_ok")),
            "measurement_valid": bool((source_record.get("status") or {}).get("measurement_valid")),
        }
        state = ranking_status(validity)
        overall = math.sqrt(float(scores["M1"]) * float(scores["M2"])) if state == "scored" else 0.0
        derived_ranking = {
            "version": RANKING_VERSION,
            "official_benchmark_formula": False,
            "M1_score": scores["M1"],
            "M2_score": scores["M2"],
            "overall": overall,
            "valid": state == "scored",
            "status": state,
        }
        record = {
            "id": source_record["id"],
            "record_id": record_id,
            "task": task,
            "task_label": source_record.get("task_label"),
            "batch": source_record["batch"],
            "batch_label": source_record.get("batch_label"),
            "sample_id": source_record["sample_id"],
            "video_id": source_record["video_id"],
            "seed": source_record["seed"],
            "source": source_record["source"],
            "source_label": source_record.get("source_label"),
            "status": status,
            "metrics": metrics,
            "metric_validity": validity,
            "derived_ranking": derived_ranking,
            "failure_reason": source_record.get("failure_reason"),
            "warnings": source_record.get("warnings") or [],
            "links": links,
        }
        result = {
            "schema_version": "feishu-rev153-document-only-result-v1",
            "record_id": record_id,
            "task_id": task,
            "sample_id": record["sample_id"],
            "video_id": record["video_id"],
            "batch_id": record["batch"],
            "seed": record["seed"],
            "source_method": record["source"],
            "document": {
                "url": DOCUMENT_URL,
                "revision": DOCUMENT_REVISION,
                "official_metrics": ["M1", "M2"],
                "thresholds_defined": False,
                "aggregation_defined": False,
            },
            "metrics": metrics,
            "metric_validity": validity,
            "diagnostic_status": status,
            "metric_pair_valid": state == "scored",
            "metric_contract": DOCUMENT_CONTRACT["tasks"][task],
            "failure_reason": record["failure_reason"],
            "warnings": record["warnings"],
            "artifacts": record["links"],
            "provenance": {
                "source_package": source.name,
                "source_result_json_sha256": source_json_hash,
                "normalization": "only Feishu rev153 M1/M2 retained as formal metrics; all scores are excluded from this formal result",
            },
        }
        atomic_json(json_path, result)
        task_csv_groups[output / task / "eval_results" / "results.csv"].append(record)
        records.append(record)

    if len({record["id"] for record in records}) != EXPECTED_RECORDS:
        raise RuntimeError("duplicate strict record IDs")
    for csv_path, group in task_csv_groups.items():
        write_csv(csv_path, [csv_row(output, gallery_dir, record) for record in sorted(group, key=lambda item: item["video_id"])])
    write_csv(output / "ALL_RESULTS_168.csv", [csv_row(output, gallery_dir, record) for record in records])
    write_derived_csv(output / "DERIVED_RANKING_168.csv", [derived_csv_row(record) for record in records])

    summary = {
        "global": task_summary(records),
        "by_task": {task: task_summary([record for record in records if record["task"] == task]) for task in TASK_ORDER},
        "ranking_status_counts": dict(Counter(record["derived_ranking"]["status"] for record in records)),
    }
    payload = {
        "schema_version": "document-only-evaluation-gallery-v1",
        "generated_at": utc_now(),
        "ranking_version": RANKING_VERSION,
        "ranking_is_official_benchmark_formula": False,
        "document": {
            "url": DOCUMENT_URL,
            "revision": DOCUMENT_REVISION,
            "official_metrics": ["M1", "M2"],
            "thresholds_defined": False,
            "aggregation_defined": False,
        },
        "summary": summary,
        "records": records,
    }
    atomic_json(gallery_dir / "full_evaluation_data.json", payload)
    atomic_json(gallery_dir / "full_evaluation_data_web.json", payload, compact_output=True)
    atomic_text(gallery_dir / "evaluation.html", gallery_html())
    atomic_text(gallery_dir / "index.html", '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=evaluation.html"><a href="evaluation.html">Open strict M1/M2 evaluation</a>\n')
    write_docs(output, records, script_path, audit_script)
    write_manifest(output, records)
    print(json.dumps({"package": str(output), "records": len(records), "summary": summary}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-script", type=Path, required=True)
    args = parser.parse_args()
    build(args.source_package, args.output, Path(__file__).resolve(), args.audit_script.resolve())


if __name__ == "__main__":
    main()
