# G2 Evaluators

Each task directory is self-contained and follows the delivery layout used by
`g5`: its own `evaluator/`, `scripts/`, first-frame image, input video, and
`eval_results/` output directory.

The canonical delivery layout for every task is:

```text
Pxx/
├── README.md
├── prompt.txt
├── first_frames/{gpt,simulation,real}/
├── evaluator/evaluate.py
├── evaluator/requirements.txt
├── evaluator/utils/
├── scripts/launch.sh
├── scripts/run_eval.sh
├── output_videos/<model_name>/sample_XX.mp4
└── eval_results/<model_name>/result_sample_XX.json
```

`bash scripts/launch.sh <model_name>` and `bash scripts/run_eval.sh
<model_name>` are the standard task-local interfaces. Existing legacy files
(`continuation.mp4`, `eval_results/json/`, etc.) are retained only for
backward-compatible audit and are not the canonical delivery paths.

Run a task from its directory:

```bash
bash scripts/run_eval.sh
```

The default input is the task directory itself. Only root-level MP4 files are
scanned, so `continuation.mp4` and (when present) `sim_continuation.mp4` are
evaluated independently. Results are written to `eval_results/json`, with
`eval_results/debug/<sample>/plot.png`, `measurements.json`, and
`overlay.mp4` for visual inspection.

All successfully extracted `metrics.M1/M2.metric` values are normalized scores
in `[0, 1]`, where `1` is best. The raw physical errors, ratios, and the exact
per-task formula remain in `verbose.measurements` under
`score_normalization`; extraction failures remain `metric: null`.

ROI annotation is first-frame-only. With a GUI, run `bash scripts/run_eval.sh
--annotate --image first_frame.png`; without a GUI, pass normalized boxes with
`--coords FIELD=x,y,w,h --no-gui`.
