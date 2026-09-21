"""Rebuild G7 result JSONs with proxy-v1 without decoding videos again.

This migration consumes the complete raw metrics/measurements already stored
under ``verbose`` by the audited evaluator.  It never edits the source result
directory and it does not touch videos or first frames.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.contract import CONTRACT_VERSION, PROXY_VERSION, TASK_METRICS, build_public_result, validate_public_result
from evaluator.run_batch import _proxy_statistics


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_from_public(sample: dict[str, Any]) -> dict[str, Any]:
    verbose = sample.get("verbose") if isinstance(sample.get("verbose"), dict) else {}
    status = verbose.get("status") if isinstance(verbose.get("status"), dict) else {}
    reserved = {"M1", "M2", "raw_metrics", "measurements", "diagnostics", "status", "debug_artifacts"}
    raw_verbose = {key: value for key, value in verbose.items() if key not in reserved}
    return {
        "task_id": sample.get("task_id"),
        "sample_id": status.get("sample_id"),
        "extract_success": status.get("extract_success"),
        "metrics": verbose.get("raw_metrics") or {},
        "measurements": verbose.get("measurements") or {},
        "diagnostics": verbose.get("diagnostics") or {},
        "physics_pass": status.get("physics_pass"),
        "physics_failure_reasons": status.get("physics_failure_reasons") or [],
        "debug_artifacts": verbose.get("debug_artifacts") or [],
        "verbose": raw_verbose,
    }


def renormalize(source_root: Path, output_root: Path, data_root: Path, *, force: bool = False) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    data_root = data_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty: {output_root}; pass --force")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = sorted(source_root.glob("P*/eval_results/json/sample_*.json"))
    if not files:
        raise FileNotFoundError(f"no canonical sample JSONs under {source_root}")
    rows: list[dict[str, Any]] = []
    for path in files:
        old = _load(path)
        task_id = str(old["task_id"])
        raw = _raw_from_public(old)
        status = old["verbose"]["status"]
        sample_id = str(status.get("sample_id") or path.stem)
        new = build_public_result(
            raw,
            task_id=task_id,
            sample_id=sample_id,
            video_path=str(old["video_path"]),
            image_path=str(old["image_path"]),
            video_prompt=str(old["video_prompt"]),
            seed=old.get("seed"),
            model=str(old.get("model") or "minimax-h3"),
            package_root=data_root,
            source=status.get("source"),
            image_variant=status.get("image_variant"),
            image_id=status.get("image_id"),
        )
        errors = validate_public_result(new)
        if errors:
            raise ValueError(f"{task_id}/{sample_id}: {'; '.join(errors)}")
        destination = output_root / task_id / "eval_results" / "json" / f"{sample_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rows.append({
            "task_id": task_id,
            "sample_id": sample_id,
            "source": status.get("source"),
            "image_variant": status.get("image_variant"),
            "seed": old.get("seed"),
            "video_path": old["video_path"],
            "image_path": old["image_path"],
            "M1_extract_success": new["metrics"]["M1"]["extract_success"],
            "M1": json.dumps(new["metrics"]["M1"]["metric"], ensure_ascii=False),
            "M1_proxy_score": new["metrics"]["M1"]["proxy_score"],
            "M1_proxy_valid": new["metrics"]["M1"]["proxy_valid"],
            "M2_extract_success": new["metrics"]["M2"]["extract_success"],
            "M2": json.dumps(new["metrics"]["M2"]["metric"], ensure_ascii=False),
            "M2_proxy_score": new["metrics"]["M2"]["proxy_score"],
            "M2_proxy_valid": new["metrics"]["M2"]["proxy_valid"],
            "overall_proxy_score": new["proxy"]["overall_proxy_score"],
            "overall_proxy_valid": new["proxy"]["proxy_valid"],
            "physics_pass": status.get("physics_pass"),
            "physics_failure_reasons": ";".join(status.get("physics_failure_reasons") or []),
            "json_path": destination.relative_to(output_root).as_posix(),
        })

    with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sources = sorted({str(row.get("source")) for row in rows})
    by_task: dict[str, Any] = {}
    for task_id in TASK_METRICS:
        subset = [row for row in rows if row["task_id"] == task_id]
        by_task[task_id] = {
            "evaluated": len(subset),
            "M1_extract_success": sum(bool(row["M1_extract_success"]) for row in subset),
            "M2_extract_success": sum(bool(row["M2_extract_success"]) for row in subset),
            "physics_pass": sum(bool(row["physics_pass"]) for row in subset),
            "proxy_statistics": _proxy_statistics(subset),
        }
    summary = {
        "schema_version": "g7-eval-summary-v3-proxy",
        "proxy_version": PROXY_VERSION,
        "contract_version": CONTRACT_VERSION,
        "evaluated": len(rows),
        "source_results_root": source_root.as_posix(),
        "by_task": by_task,
        "proxy_statistics": {
            "all": _proxy_statistics(rows),
            "by_source": {
                source: _proxy_statistics([row for row in rows if str(row.get("source")) == source])
                for source in sources
            },
            "by_task_source": {
                task: {
                    source: _proxy_statistics([
                        row for row in rows if row["task_id"] == task and str(row.get("source")) == source
                    ])
                    for source in sources
                }
                for task in TASK_METRICS
            },
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = renormalize(args.source_results, args.output_root, args.data_root, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
