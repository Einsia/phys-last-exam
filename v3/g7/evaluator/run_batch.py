"""Run the refined Group 7 evaluator over every manifest sample.

Usage::

    python evaluator/run_batch.py \
      --data-root ../g7_refined_20260829_delivery \
      --output-root ./eval_results

The runner never calls a remote model.  It decodes each MP4, invokes one
task-specific OpenCV/NumPy evaluator, writes one canonical JSON, and records a
CSV/summary.  Existing output is overwritten only when ``--force`` is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.contract import CONTRACT_VERSION, PROXY_VERSION, TASK_METRICS, build_public_result, validate_public_result
from evaluator.evaluate import evaluate as evaluate_raw


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_rows(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(data_root / "manifest.json")
    tasks = {item["task_id"]: item for item in _load_json(data_root / "tasks.json")["tasks"]}
    rows = manifest.get("samples")
    if not isinstance(rows, list):
        raise ValueError("manifest.json has no samples list")
    return {"manifest": manifest, "rows": rows}, tasks


def _safe_exception_result(task_id: str, sample_id: str, message: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "sample_id": sample_id,
        "extract_success": False,
        "metrics": {},
        "measurements": {},
        "diagnostics": {},
        "failure_reason": "evaluator_exception",
        "failure_reasons": ["evaluator_exception"],
        "failure_details": {"exception": message},
        "physics_pass": False,
        "physics_failure_reasons": ["evaluator_exception"],
        "debug_artifacts": [],
    }


def _flat_metric(wrapper: Any) -> Any:
    if not isinstance(wrapper, dict):
        return None
    return wrapper.get("metric")


def _proxy_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scores with unavailable samples retained as explicit zeros."""

    def describe(key: str, valid_key: str) -> dict[str, Any]:
        values = [float(row[key]) for row in rows]
        valid = sum(bool(row[valid_key]) for row in rows)
        return {
            "count": len(values),
            "valid_count": valid,
            "mean_including_unavailable_as_zero": (sum(values) / len(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }

    return {
        "M1": describe("M1_proxy_score", "M1_proxy_valid"),
        "M2": describe("M2_proxy_score", "M2_proxy_valid"),
        "overall": describe("overall_proxy_score", "overall_proxy_valid"),
    }


def run_batch(data_root: Path, output_root: Path, *, force: bool = False) -> dict[str, Any]:
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    source, tasks = _manifest_rows(data_root)
    rows = source["rows"]
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty: {output_root}; pass --force to replace results")
    output_root.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["task_id"])
        sample_id = str(row["sample_id"])
        if task_id not in TASK_METRICS or task_id not in tasks:
            raise ValueError(f"manifest task is not declared: {task_id}")
        video_rel = Path(str(row["video"]))
        image_rel = Path(str(row["first_frame"]))
        video = data_root / video_rel
        image = data_root / image_rel
        debug_dir = output_root / task_id / "eval_results" / "debug" / sample_id
        json_path = output_root / task_id / "eval_results" / "json" / f"{sample_id}.json"
        debug_dir.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            raw = evaluate_raw(task_id, video, sample_id, debug_dir)
        except Exception as exc:  # keep a single bad clip from hiding others
            raw = _safe_exception_result(task_id, sample_id, f"{type(exc).__name__}: {exc}")
        public = build_public_result(
            raw,
            task_id=task_id,
            sample_id=sample_id,
            video_path=video_rel,
            image_path=image_rel,
            video_prompt=str(tasks[task_id].get("prompt_en", "")),
            seed=row.get("seed"),
            model="minimax-h3",
            package_root=data_root,
            source=row.get("source"),
            image_variant=row.get("image_variant"),
            image_id=(f"{row.get('source')}_{int(row['image_variant']):02d}" if row.get("source") and row.get("image_variant") is not None else None),
        )
        errors = validate_public_result(public)
        if errors:
            raise ValueError(f"{task_id}/{sample_id} contract violation: {'; '.join(errors)}")
        json_path.write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        m1 = public["metrics"]["M1"]
        m2 = public["metrics"]["M2"]
        status = public["verbose"]["status"]
        result_rows.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "sample_index": row.get("sample_index"),
            "source": row.get("source"),
            "image_variant": row.get("image_variant"),
            "seed": row.get("seed"),
            "video_path": video_rel.as_posix(),
            "image_path": image_rel.as_posix(),
            "M1_extract_success": m1["extract_success"],
            "M1": json.dumps(_flat_metric(m1), ensure_ascii=False),
            "M1_proxy_score": m1["proxy_score"],
            "M1_proxy_valid": m1["proxy_valid"],
            "M2_extract_success": m2["extract_success"],
            "M2": json.dumps(_flat_metric(m2), ensure_ascii=False),
            "M2_proxy_score": m2["proxy_score"],
            "M2_proxy_valid": m2["proxy_valid"],
            "overall_proxy_score": public["proxy"]["overall_proxy_score"],
            "overall_proxy_valid": public["proxy"]["proxy_valid"],
            "physics_pass": status.get("physics_pass"),
            "physics_failure_reasons": ";".join(status.get("physics_failure_reasons") or []),
            "json_path": json_path.relative_to(output_root).as_posix(),
            "video_qa_ok": (public["verbose"].get("video_qa") or {}).get("video_qa_ok") if isinstance(public["verbose"].get("video_qa"), dict) else None,
        })

    fields = list(result_rows[0]) if result_rows else []
    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    by_task: dict[str, dict[str, Any]] = {}
    for task_id in TASK_METRICS:
        subset = [r for r in result_rows if r["task_id"] == task_id]
        by_task[task_id] = {
            "expected": sum(1 for r in rows if r["task_id"] == task_id),
            "evaluated": len(subset),
            "M1_extract_success": sum(bool(r["M1_extract_success"]) for r in subset),
            "M2_extract_success": sum(bool(r["M2_extract_success"]) for r in subset),
            "physics_pass": sum(bool(r["physics_pass"]) for r in subset),
            "proxy_statistics": _proxy_statistics(subset),
        }
    sources = sorted({str(row.get("source")) for row in result_rows})
    by_source = {
        source_name: _proxy_statistics([row for row in result_rows if str(row.get("source")) == source_name])
        for source_name in sources
    }
    by_task_source = {
        task_id: {
            source_name: _proxy_statistics([
                row for row in result_rows
                if row["task_id"] == task_id and str(row.get("source")) == source_name
            ])
            for source_name in sources
        }
        for task_id in TASK_METRICS
    }
    summary = {
        "schema_version": "g7-eval-summary-v3-proxy",
        "proxy_version": PROXY_VERSION,
        "contract_version": CONTRACT_VERSION,
        "batch_id": source["manifest"].get("batch_id"),
        "protocol": source["manifest"].get("protocol", {}),
        "evaluated": len(result_rows),
        "M1_extract_success": sum(bool(r["M1_extract_success"]) for r in result_rows),
        "M2_extract_success": sum(bool(r["M2_extract_success"]) for r in result_rows),
        "physics_pass": sum(bool(r["physics_pass"]) for r in result_rows),
        "runner_errors": sum(r["physics_failure_reasons"] == "evaluator_exception" for r in result_rows),
        "by_task": by_task,
        "proxy_statistics": {
            "all": _proxy_statistics(result_rows),
            "by_source": by_source,
            "by_task_source": by_task_source,
        },
        "evaluator": "evaluator/run_batch.py + evaluator/tasks (OpenCV/NumPy only)",
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="extracted refined delivery root")
    parser.add_argument("--output-root", type=Path, default=ROOT / "eval_results")
    parser.add_argument("--force", action="store_true", help="allow writing into a non-empty output directory")
    args = parser.parse_args(argv)
    summary = run_batch(args.data_root, args.output_root, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
