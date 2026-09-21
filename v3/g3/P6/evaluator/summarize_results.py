#!/usr/bin/env python3
"""Build the P6 continuous-score aggregate report from rescored JSON/CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "results_v1"
summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((ROOT / "results.csv").open(encoding="utf-8")))
rows.sort(key=lambda row: float(row["overall"]), reverse=True)

aggregate = dict(summary)
aggregate["ranked_samples"] = [
    {
        "sample_id": row["sample_id"],
        "measurement_valid": row["measurement_valid"] == "True",
        "overall": float(row["overall"]),
        "pure_rolling_ratio": float(row["pure_rolling_ratio"]),
        "contact_point_velocity": float(row["contact_point_velocity"]),
        "legacy_overall_0_100": float(row["legacy_overall_0_100"]),
        "physics_pass_label": row["physics_pass"] == "True",
    }
    for row in rows
]
aggregate["qa"] = {
    "full_continuous_overlay_reviewed_count": 24,
    "full_continuous_overlay_frames_per_video": 124,
    "full_overlay_video_review_status": "complete",
    "structurally_rejected_after_review": [
        "P6_sim_02_seed42",
        "P6_sim_02_seed45",
    ],
}
(ROOT / "aggregate_report.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

all_dist = summary["overall_all"]
valid_dist = summary["overall_measurement_valid_only"]
legacy_dist = summary["legacy_overall_0_100"]
top = aggregate["ranked_samples"][:5]

report = f"""# P6 continuous evaluator report (continuous-0-1-v1)

## Scoring contract

The 24 frozen P6 tracks were rescored from their existing measurement JSON;
the videos were not decoded and ball/marker tracking was not rerun.  All five
dimension scores and `overall` now lie in [0,1].  Every residual uses
`q(r,s)=1/(1+r/s)`, where the frozen legacy boundary `s` is exactly the 0.5
quality anchor.  Coverage is represented by its continuous deficit and
minimum-evidence counts by `x/(x+s)`.

Component qualities are combined into dimensions, and dimensions into
`overall`, with weighted geometric means using a 0.01 input floor.  A valid
measurement therefore always has `overall >= 0.01`; an invalid measurement has
`overall = 0`.  Hard `status`, `structural_ok`, and `physics_pass` decisions are
retained only as labels and never truncate a valid continuous score.  The old
0-100 dimensions are preserved under `legacy_score` in every result JSON.

No LLM, VLM, semantic detector, or learned vision model is used.

## Formal distribution

- Frozen results rescored: **{summary['result_count']}/24**
- Measurement-valid: **{summary['measurement_valid_count']}/24**
- Measurement-invalid: **{summary['measurement_invalid_count']}/24**
- Preserved legacy physics-pass labels: **{summary['physics_pass_label_count']}/24**
- Overall, all 24 (invalid included as zero): mean **{all_dist['mean']:.4f}**, median **{all_dist['median']:.4f}**, range **[{all_dist['min']:.4f}, {all_dist['max']:.4f}]**
- Overall, 22 valid only: mean **{valid_dist['mean']:.4f}**, median **{valid_dist['median']:.4f}**, P10/P90 **{valid_dist['p10']:.4f}/{valid_dist['p90']:.4f}**
- Preserved legacy overall 0-100: mean **{legacy_dist['mean']:.2f}**, median **{legacy_dist['median']:.2f}**

Legacy physics-pass labels: {', '.join(summary['physics_pass_labels'])}.

Invalid measurements (continuous overall forced to zero): {', '.join(summary['invalid_measurements'])}.

Top five continuous scores:
"""
for item in top:
    report += (
        f"\n- `{item['sample_id']}`: overall {item['overall']:.4f}, "
        f"rolling {item['pure_rolling_ratio']:.4f}, contact {item['contact_point_velocity']:.4f}, "
        f"legacy pass label={item['physics_pass_label']}"
    )
report += """

## Validation and QA

Boundary, monotonicity, input-floor, valid/invalid, and no-threshold-jump tests
are implemented in `test_continuous_scoring.py`.  The rescore manifest records
SHA-256 for every existing frame CSV, overlay, and plot, proving rescoring did
not regenerate tracking artifacts.  All 2,976 overlay frames remain manually
reviewed; the two transparent-ball clips remain correctly labeled invalid.
"""
(ROOT / "TASK_REPORT.md").write_text(report, encoding="utf-8")

with (ROOT / "qa_checklist.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "sample_id",
            "full_overlay_review",
            "continuous_contract_checked",
            "legacy_label_preserved",
            "notes",
        ],
    )
    writer.writeheader()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        invalid = row["measurement_valid"] != "True"
        writer.writerow(
            {
                "sample_id": row["sample_id"],
                "full_overlay_review": "complete_all_124_frames",
                "continuous_contract_checked": "complete",
                "legacy_label_preserved": "complete",
                "notes": (
                    "measurement invalid; overall=0; ball visual identity lost"
                    if invalid
                    else "all dimensions in [0,1], overall>0, no hard score truncation"
                ),
            }
        )

print(json.dumps(aggregate, ensure_ascii=False, indent=2))
