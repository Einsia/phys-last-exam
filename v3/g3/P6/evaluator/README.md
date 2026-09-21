# P6 deterministic pure-rolling evaluator

This evaluator is intentionally independent of LLMs and VLMs.  It also uses
no learned vision model.  Ball tracking, marker orientation, and physics
measurements are implemented with OpenCV, robust geometry, and SciPy.

Single-video CLI:

```bash
.venv/bin/python evaluate.py \
  --video ../minimax_h3/prompt_v1_20260825/videos/P6_sim_01_seed42.mp4 \
  --task_id P6 \
  --output results_v1/P6_sim_01_seed42/result.json \
  --artifacts-dir results_v1/P6_sim_01_seed42
```

Formal 24-video batch:

```bash
./run_batch.sh --workers 2
```

Each sample directory contains `result.json`, `overlay.mp4`, `plot.png`,
`frame_measurements.csv`, and `evaluator.log`.  The batch root contains
`results.csv` and `summary.json`.  Original videos are read-only inputs.

The structural gate rejects a continuous ball-boundary Hough dropout longer
than the frozen tolerance.  This prevents KLT extrapolation from silently
scoring clips in which the generated ball fades or becomes transparent; the
exact dropout interval is recorded in `result.json`.

`config_v1.yaml` freezes the six accepted-first-frame initialization ROIs,
metric thresholds, weights, and pass rules.  Robust fits never hide discarded
frames: all per-frame measurements and extraction confidences are exported for
human sanity checking.

Continuous rescoring of frozen measurements (no video decode or retracking):

```bash
.venv/bin/python rescore_continuous.py
.venv/bin/python -m unittest -v test_continuous_scoring.py
.venv/bin/python audit_results.py
.venv/bin/python summarize_results.py
```

`continuous_score_config_v1.yaml` defines `continuous-0-1-v1`.  Residuals use
`q(r,s)=1/(1+r/s)` and all component/dimension aggregation uses a weighted
geometric mean with a 0.01 input floor.  Invalid measurements have overall
zero; valid measurements remain strictly positive.  Legacy 0-100 scores and
the old hard physics-pass label are retained in every result JSON.
