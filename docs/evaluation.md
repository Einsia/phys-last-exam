# Evaluation configuration and input details

Run batch commands from the repository root with the evaluation environment active. The [main README](../README.md#evaluate-videos) contains the standard workflow; `python evaluate.py --help` lists all options.

## One-command evaluation

```bash
bash scripts/evaluate_all.sh videos
```

This evaluates all videos in `videos/`, using its `manifest.json` when available or scanning model subdirectories otherwise. A manifest path can be supplied directly. First use creates `.venv/eval/` and runs `setup.sh`; subsequent runs reuse the environment and weights. The launcher uses Python 3.12 (`PLE_PYTHON` can override its executable), one worker, and automatic resume. Pass `--output runs/my-run`, `--tasks P21`, or `--require-all-tasks` as needed; evaluator settings such as worker count, timeout, and gate options are also forwarded. A `--dry-run` uses only the lightweight controller environment and does not install evaluation models.

For an existing evaluation environment, set `PLE_EVAL_PYTHON=/absolute/path/to/its/bin/python`. The launcher then skips installation; that environment and the weights selected by `FINAL_MODELS_DIR` must already be ready. To reuse a lightweight controller, set `PLE_CONTROLLER_PYTHON` similarly.

```bash
export PLE_EVAL_PYTHON=/path/to/existing/env/bin/python
export FINAL_MODELS_DIR=/path/to/evaluation-models
CUDA_VISIBLE_DEVICES=1 bash scripts/evaluate_all.sh videos --output runs/gpu1
```

The low-level commands below remain available for manual setup and custom workflows.

## Evaluation models

`bash setup.sh` installs dependencies and downloads evaluator weights. To store those weights outside the repository, set this variable before setup and keep it set when evaluating:

```bash
export FINAL_MODELS_DIR=/absolute/path/to/evaluation-models
bash setup.sh
```

To reuse already provisioned weights, run `bash setup.sh --skip-models`. This installs dependencies and checks CUDA; it does not create or validate the model directory. The directory must contain:

```text
Qwen3.6-27B/
grounding-dino-tiny/
sam2.1-hiera-small/
sam2.1-hiera-small-transformers/
sam2.1-hiera-large-transformers/
cotracker3/scaled_offline.pth
cotracker3/source/                 # Pinned CoTracker checkout from setup.sh
```

These are evaluation weights. Video generators are prepared separately by the [generation launcher](../generation/README.md#one-command-generation), or connected through an existing local configuration.

## GPU selection

Batch evaluation uses one worker and the first visible GPU by default. Qwen runs locally; a model server is not required. Select another physical GPU with:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py --manifest videos/manifest.json --output runs/gpu1 --resume
```

When using an individual task’s entrypoint instead of the batch runner, change into that task’s directory and set both model paths:

```bash
export FINAL_MODELS_DIR=/absolute/path/to/evaluation-models
export VLM_MODEL="$FINAL_MODELS_DIR/Qwen3.6-27B"
```

Then follow that task’s `task.md` command.

## Existing benchmark videos

Directory scanning expects `videos/MODEL/VIDEO_STEM.mp4` and the current task ID in each filename. For a flat directory containing one model’s videos, add `--model my-model`. Scanning uses the bundled task image and prompt, unless a matching generation parameter record supplies the prompt. Use an explicit manifest for different inputs or filenames that do not identify a task.

The generator assigns the calibrated filenames below automatically. For existing videos, replace `N` with the sample seed:

```text
P3_gpt_01_modern_seedN.mp4
P8_gpt_01_modern_seedN.mp4
P14_gpt_01_modern_seedN.mp4
P17_gpt_01_30deg_seedN.mp4
```

P3/P14 require a **1344 × 768** evaluation canvas. The generator retains the native video and resize metadata. Follow each task’s `task.md` for scene geometry; a filename or resize alone does not establish calibration.

## Custom first frames and annotations

P33/P34/P30/P36/P10/P38 include `first_frame_annotations.json` for their fixed benchmark image. Standard benchmark inputs do not require a separate annotation directory. The evaluator checks the image hash, maps coordinates to the video resolution, and validates correspondence with the decoded first frame before tracking.

For a different image or layout, provide a manifest with the actual `image` and `prompt`, plus reviewed annotations matching the video and image hashes. Save an array such as this as `videos/custom-manifest.json`; paths are relative to that manifest:

```json
[
  {
    "model": "my-model",
    "task": "P33",
    "seed": 42,
    "sample_id": "g8_P33_seed42",
    "video": "my-model/g8_P33_seed42.mp4",
    "image": "inputs/P33/first_frame.png",
    "prompt": "inputs/P33/prompt.txt",
    "annotation": "inputs/P33/reviewed_annotations.json"
  }
]
```

Validate paths before evaluating:

```bash
python evaluate.py --manifest videos/custom-manifest.json --tasks P33 --output runs/custom_input_check --dry-run
```

Custom scenes must satisfy the task’s physical setup and calibration requirements. Annotations must follow that task’s schema. The dry run checks file availability; geometry and decoded-frame correspondence are checked during execution.

`--annotation-root annotations` selects optional overrides at `annotations/MODEL/TASK/VIDEO_STEM.json`. It does not replace the task image or prompt. Generation snapshots overrides into `videos/.inputs/`; move that hidden directory with the videos. Direct evaluation with `--annotation-root` reads the external directory, so keep it available.

## Inventory and result interpretation

Inspect the dry-run output and its `input_manifest.json` for input errors, video counts, model names, and task coverage. `--require-all-tasks` checks the selected tasks for each included model, not the number of models or samples. A generation dry run writes only a plan; it creates neither videos nor an evaluation manifest.

`by_model.csv` and `by_task.csv` contain summaries; `results.csv` and `results.json` contain per-video results. Per-video directories retain detailed measurements and debug evidence. Evaluator scores are normalized to **0–1**.

An exit code of zero means the batch completed without execution or input errors. A failed consistency gate or low physics score is a valid evaluation result. Missing measurement evidence does not establish a measured physical violation. Consult the per-video evidence when interpreting scores.

Dry runs validate input inventory without model inference. A successful single-task run checks that route; it does not validate every task evaluator or video generator. Failed generation samples remain in the manifest and become input errors when their videos are unavailable. Review generation logs under `runs/generation/OUTPUT_ID/` and evaluation logs under the selected output directory.
