#!/usr/bin/env python3
"""Create a non-pooled comparison report for the three P11 prompt batches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LABELS = {
    "prompt_v2_20260826": "neutral incident-only prompt",
    "prompt_v3_20260826": "guided incident+reflection+refraction prompt",
    "prompt_v4_20260826": "current refraction+normal prompt (user-facing V3)",
}


def as_float(value: str) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def bootstrap_mean_ci(values: List[float], seed: int = 20260826) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "median": None, "ci95_low": None, "ci95_high": None}
    array = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_ci(values)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "ci95_low": low,
        "ci95_high": high,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    args = parser.parse_args(argv)
    batches: Dict[str, List[Dict[str, str]]] = {}
    for batch in LABELS:
        path = args.evaluation_root / batch / "summary.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            batches[batch] = list(csv.DictReader(handle))
    summaries = []
    by_sample = []
    detailed = {}
    for batch, rows in batches.items():
        valid = [row for row in rows if row.get("measurement_valid") == "True"]
        m1 = [value for row in valid if (value := as_float(row.get("M1_abs", ""))) is not None]
        m2 = [value for row in valid if (value := as_float(row.get("M2_normalized", ""))) is not None]
        signed = [value for row in valid if (value := as_float(row.get("M1_signed", ""))) is not None]
        legacy_scores = [value for row in valid if (value := as_float(row.get("overall_gated_score", ""))) is not None]
        continuous_scores = [value for row in valid if (value := as_float(row.get("continuous_overall", ""))) is not None]
        record = {
            "batch": batch,
            "prompt_label": LABELS[batch],
            "video_count": len(rows),
            "measurement_valid_count": len(valid),
            "physics_pass_count": sum(row.get("physics_pass") == "True" for row in rows),
            "physics_pass_rate": sum(row.get("physics_pass") == "True" for row in rows) / max(1, len(rows)),
            "M1_pass_count": sum(value <= 0.12 for value in m1),
            "M2_pass_count": sum(value <= 0.025 for value in m2),
            "M1_abs": summarize(m1),
            "M1_signed": summarize(signed),
            "M2": summarize(m2),
            "legacy_overall_score_0_100": summarize(legacy_scores),
            "continuous_overall_0_1": summarize(continuous_scores),
        }
        detailed[batch] = record
        summaries.append(
            {
                "batch": batch,
                "prompt_label": LABELS[batch],
                "n": len(rows),
                "valid_n": len(valid),
                "physics_pass_n": record["physics_pass_count"],
                "M1_pass_n": record["M1_pass_count"],
                "M2_pass_n": record["M2_pass_count"],
                "M1_abs_mean": record["M1_abs"]["mean"],
                "M1_abs_ci95_low": record["M1_abs"]["ci95_low"],
                "M1_abs_ci95_high": record["M1_abs"]["ci95_high"],
                "M2_mean": record["M2"]["mean"],
                "M2_ci95_low": record["M2"]["ci95_low"],
                "M2_ci95_high": record["M2"]["ci95_high"],
                "continuous_overall_mean": record["continuous_overall_0_1"]["mean"],
                "continuous_overall_median": record["continuous_overall_0_1"]["median"],
                "continuous_overall_min": min(continuous_scores) if continuous_scores else None,
                "continuous_overall_max": max(continuous_scores) if continuous_scores else None,
                "continuous_overall_ci95_low": record["continuous_overall_0_1"]["ci95_low"],
                "continuous_overall_ci95_high": record["continuous_overall_0_1"]["ci95_high"],
                "legacy_overall_mean_0_100": record["legacy_overall_score_0_100"]["mean"],
            }
        )
        for sample_id in sorted({row["sample_id"] for row in rows}):
            group = [row for row in rows if row["sample_id"] == sample_id and row.get("measurement_valid") == "True"]
            group_m1 = [value for row in group if (value := as_float(row.get("M1_abs", ""))) is not None]
            group_m2 = [value for row in group if (value := as_float(row.get("M2_normalized", ""))) is not None]
            group_score = [value for row in group if (value := as_float(row.get("overall_gated_score", ""))) is not None]
            group_continuous = [value for row in group if (value := as_float(row.get("continuous_overall", ""))) is not None]
            by_sample.append(
                {
                    "batch": batch,
                    "sample_id": sample_id,
                    "method": "gpt-image-2" if "_gpt_" in sample_id else "simulation",
                    "seed_count": len(group),
                    "physics_pass_count": sum(row.get("physics_pass") == "True" for row in group),
                    "M1_abs_mean": float(np.mean(group_m1)) if group_m1 else None,
                    "M1_abs_std": float(np.std(group_m1, ddof=1)) if len(group_m1) > 1 else 0.0,
                    "M2_mean": float(np.mean(group_m2)) if group_m2 else None,
                    "M2_std": float(np.std(group_m2, ddof=1)) if len(group_m2) > 1 else 0.0,
                    "overall_mean": float(np.mean(group_score)) if group_score else None,
                    "overall_std": float(np.std(group_score, ddof=1)) if len(group_score) > 1 else 0.0,
                    "continuous_overall_mean": float(np.mean(group_continuous)) if group_continuous else None,
                    "continuous_overall_std": float(np.std(group_continuous, ddof=1)) if len(group_continuous) > 1 else 0.0,
                    "continuous_overall_min": float(np.min(group_continuous)) if group_continuous else None,
                    "continuous_overall_max": float(np.max(group_continuous)) if group_continuous else None,
                }
            )
    out_json = args.evaluation_root / "comparison_report.json"
    out_csv = args.evaluation_root / "comparison_summary.csv"
    out_sample = args.evaluation_root / "comparison_by_sample.csv"
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task_id": "P11",
                "batches_are_independent": True,
                "best_seed_selection": False,
                "bootstrap_seed": 20260826,
                "bootstrap_resamples": 10000,
                "batches": detailed,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    with out_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    with out_sample.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(by_sample[0].keys()))
        writer.writeheader()
        writer.writerows(by_sample)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["neutral", "3-path guided", "refraction+normal"]
        figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
        for axis, key, title, threshold in (
            (axes[0], "M1_abs", "M1 |Snell residual|", 0.12),
            (axes[1], "M2_normalized", "M2 intersection / tank", 0.025),
            (axes[2], "continuous_overall", "Continuous overall (0-1)", None),
        ):
            data = []
            for batch in LABELS:
                values = [as_float(row.get(key, "")) for row in batches[batch] if row.get("measurement_valid") == "True"]
                data.append([value for value in values if value is not None])
            axis.boxplot(data, tick_labels=labels, showmeans=True)
            if threshold is not None:
                axis.axhline(threshold, color="#c62828", ls="--", lw=1.2, label="physics pass limit")
                axis.legend(loc="best", fontsize=8)
            axis.set_title(title)
            axis.grid(axis="y", alpha=0.2)
            axis.tick_params(axis="x", rotation=18)
        figure.suptitle("P11 prompt ablation, all 24 videos per batch (no best-seed filtering)")
        figure.savefig(args.evaluation_root / "comparison_plot.png", dpi=170)
        plt.close(figure)
    except Exception as exc:
        print(f"plot generation failed: {type(exc).__name__}: {exc}")
    print(json.dumps(detailed, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
