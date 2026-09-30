# G5 evaluation report

## Delivery layout

Each task is self-contained under its own directory. The evaluator is in
`evaluator/evaluate.py`, dependencies are listed in
`evaluator/requirements.txt`, and `scripts/run_eval.sh` evaluates every MP4 in
the task directory. No evaluator imports code from another task.

The canonical delivery paths are `prompts/video.txt`,
`first_frames/{gpt,simulation,real}/`, `evaluator/evaluate.py`,
`evaluator/requirements.txt`, `evaluator/utils/`, `scripts/launch.sh`,
`scripts/run_eval.sh`, `output_videos/<model_name>/sample_XX.mp4`, and
`eval_results/<model_name>/result_sample_XX.json`. Legacy root-level videos and
`eval_results/json/` files are retained as compatibility copies only.

For P21b and P21c, the evaluator performs an independent detection on the last
video frame. Residual ice forces the ice-completion score M2 to `0`; the final
frame box, confidence, and warning are stored in the result JSON and drawn on
the final overlay frame.

Every evaluated sample has:

- `eval_results/json/<sample>.json` using the task, video, first-image, seed,
  model, nested metric, and verbose schema;
- `eval_results/debug/<sample>/measurements.json` with raw frame measurements;
- `eval_results/debug/<sample>/plot.png` and `overlay.mp4` for visual review;
- `eval_results/results.csv` as a flat summary.

## Metrics

All emitted metrics are scores in `[0,1]`, where `1` is best. Raw physical
measurements and the formula used for each score are retained under
`verbose.measurements`.

| Task | M1 score | M2 score |
| --- | --- | --- |
| P21b | `1` if water height decreases, else `0` | `clip(1-stone_bottom_gap_norm,0,1)` |
| P21c | `1` if water height increases, else `0` | `clip(1-solid_evidence_ratio,0,1)` |
| P23 | `clip(1-|h_ice/h_water-1000/917|,0,1)` | `clip(1-cross_section_error,0,1)` |
| P36 | `clip(1-1/(t_mag/t_ctrl),0,1)` | `clip(1-v_mag/v_ctrl,0,1)` |
| P44 | `clip(1-fit_rmse_norm,0,1)` | `clip(1-endpoint_height_error_norm,0,1)` |

The evaluators retain measured values even when a generated clip does not meet
the physical expectation. Such cases receive a low score and are marked in
`verbose.quality_warnings`; only unextractable measurements use
`extract_success: false` and `metric: null`.

## Current samples

The checked-in outputs cover one continuation video for P21b, P21c, and P23,
two videos (GPT and simulation) for P36, and two videos (GPT and simulation)
for P44. P36 simulation is intentionally reported as an object-pair extraction
failure because the two blocks overlap and cannot be separated reliably.
