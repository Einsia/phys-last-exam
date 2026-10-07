# PhysScope: A Multi-Domain Benchmark for Proxy-Based Evaluation of Video Physics

## Abstract

Visually convincing videos can still violate basic physical laws, making reliable physical evaluation essential for assessing video generation models. Existing evaluations often rely on learned judgments or reference videos, while direct physical tests largely focus on mechanics. We introduce PhysScope, a measurement-based benchmark comprising 40 controlled tasks spanning mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface-tension effects. Each task pairs an initial image and a generation prompt with observable quantities and predefined physical criteria. Our evaluation first screens for temporal consistency, excluding videos whose unstable object identities or structures make physical measurements unreliable. Videos that pass are then assessed through task-specific measurements, such as oscillation periods, reflection angles, and liquid levels. We distinguish failures of physical tests from cases with insufficient measurement evidence. Our protocol evaluates eight video generation models on 40 tasks with four seeds per task: 160 planned videos per model and 1,280 in total. Among the evaluated videos, 98.36% pass the temporal-consistency check, but only 159 of the 716 videos with sufficient measurement evidence (22.21%) pass all required physical checks. These findings reveal a substantial gap between temporal coherence and physical consistency, highlighting the need to report measurement coverage alongside physical performance. PhysScope provides an interpretable framework for evaluating diverse physical phenomena while making the limitations of measurement explicit.

## Task numbering

Tasks use **P1–P40**, following the nine physical categories. Within each category, tasks are grouped **Easy → Medium → Hard**, retaining the original results-table order within each difficulty, then numbered continuously. Original **P23 is Easy** (current **P24**) and original **P11 is Medium** (current **P17**). First frames, physical prompt descriptions, scoring rules, and frozen evaluation scores/ranks are unchanged. Commands, explicit prompt task tags, and generated filenames use the current IDs; existing experiment artifacts retain their legacy IDs. Each `easy/P*/task.md`, `medium/P*/task.md`, or `hard/P*/task.md` identifies its original ID and describes the scene and metrics. Use the current task IDs with the commands below.

## Setup

Use Linux and Bash with Python 3.12 and an NVIDIA GPU. The entrypoints use Linux file locking; native Windows is not supported. We recommend an 80 GB GPU for evaluation with the default [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) model and at least 100 GB of free disk space. Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and Git first; the installer uses CUDA 12.8 PyTorch wheels, so a compatible NVIDIA driver is required. Video generators have separate environments and GPU requirements; some of the presets require four GPUs.

Clone the repository and create the evaluation environment. After `cd phys-last-exam`, run subsequent commands from the repository root unless a task's own guide explicitly says to change directories. If you already have videos, complete Setup and go directly to [Evaluate videos](#evaluate-videos).

```bash
git clone https://github.com/Einsia/phys-last-exam.git
cd phys-last-exam
conda create -n physscope python=3.12 -y
conda activate physscope

# Optional: check all 1,280 generation plans before installing models.
python generate.py --models all --output videos --dry-run

bash setup.sh
```

`setup.sh` installs the Python packages and downloads Qwen, Grounding DINO, SAM 2.1, and CoTracker into `models/`. The evaluator finds these models automatically. The first setup downloads about 60 GB; rerun the same command if a download is interrupted. In a new terminal, run `conda activate physscope` again before evaluating.

The generation dry run checks the bundled task inputs and writes a plan under `runs/generation/`; it creates no videos or evaluation manifest and runs no inference. If evaluation weights are already provisioned, `bash setup.sh --skip-models` installs dependencies without downloading them, but still checks CUDA availability.

To download or reuse evaluation weights outside the repository, set `FINAL_MODELS_DIR` before setup and keep it set when evaluating:

```bash
export FINAL_MODELS_DIR=/absolute/path/to/evaluation-models
# Use bash setup.sh to download, or bash setup.sh --skip-models to reuse weights.
```

This directory must contain the same layout produced by `setup.sh`: `Qwen3.6-27B/`, `grounding-dino-tiny/`, `sam2.1-hiera-small/`, `sam2.1-hiera-small-transformers/`, `sam2.1-hiera-large-transformers/`, and `cotracker3/` (including `scaled_offline.pth` and the pinned `source/` checkout). `--skip-models` does not validate or create that layout. Generation checkpoints are configured separately with `--model-root` and `generation.local.json`.

## Prepare your videos

The benchmark has **40 tasks: 15 easy, 15 medium, and 10 hard**. The unified generator reads their `first_frame.png` and `prompt.txt` automatically and supports the eight models in the results table, plus custom models.

Connect your generation model environments once:

```bash
python generate.py --init-config --model-root /path/to/checkpoints
```

Check the Python, source, and weight paths in `generation.local.json`; see the short [model setup and custom model guide](generation/README.md). Generation models use their own environments; `setup.sh` prepares the evaluator and generation controller.

Replace `/path/to/checkpoints` with the root containing your generation weights and source checkouts. `--init-config` writes a path template; it does not install or download a generator, and refuses to overwrite an existing configuration. Edit that file directly when configuring an existing installation.

Start with one configured model, one task, and one seed:

```bash
python generate.py --models cogvideox1.5-5b-i2v --check
python generate.py --models cogvideox1.5-5b-i2v --tasks P21 --seeds 42 --output videos --resume
```

`--check` verifies the selected model's paths and required client configuration without inference or API submission. It does not check upstream package compatibility or available GPU memory. `setup.sh` does not install the video generation models. Before selecting `--models all`, configure all eight adapters, including the Seedance API credentials and each local model's environment, weights, and devices.

Once those environments are ready, generate all 40 tasks with all 8 models and seeds 42–45:

```bash
python generate.py --models all --output videos --resume
```

Add `--dry-run` to either generation command to preview the jobs without loading weights or calling an API.

Videos are written to `videos/MODEL/VIDEO_STEM.mp4`, with matching parameter records and an evaluation-ready `videos/manifest.json`. Special task filenames are handled automatically; P3/P14 receive the required 1344 × 768 evaluation copy while the native video and resize metadata are retained. Frozen images, prompts, and task annotations are stored in `videos/.inputs/`, so the manifest travels with the videos.

The fixed first-frame annotations for P33/P34/P30/P36/P10/P38 are included in their task packages and loaded automatically. You do not need to create an `annotations/` directory for standard benchmark inputs.

Generation returns a nonzero exit code if any sample fails. Check `runs/generation/OUTPUT_ID/summary.json` and the per-attempt logs before evaluating; failed samples remain in the manifest and are reported as input errors if their videos are missing.

<details>
<summary>Calibrated filenames and custom first frames</summary>

The generator assigns these calibrated filenames. If you bring existing videos, use the same names, replacing `N` with the seed:

```text
P3_gpt_01_modern_seedN.mp4
P8_gpt_01_modern_seedN.mp4
P14_gpt_01_modern_seedN.mp4
P17_gpt_01_30deg_seedN.mp4
```

P3/P14 require a 1344 × 768 video canvas. Follow each task's `task.md` for its scene geometry; renaming an incompatible video does not make it calibrated.

P33/P34/P30/P36/P10/P38 use the bundled `first_frame_annotations.json`. The evaluator verifies the input-image hash, scales coordinates to the video resolution, and checks the actual first frame before tracking. A layout that fails this correspondence check is reported as an initialization failure.

For a different input image or layout, use a manifest that explicitly supplies the actual `image` and `prompt` files, plus reviewed video annotations. `--annotation-root` selects an annotation override; it does not change the input image or prompt. These optional overrides use:

```text
annotations/MODEL/TASK/VIDEO_STEM.json
# Example: annotations/my-model/P33/g8_P33_seed42.json
```

Custom video annotations must match their video's image/video hashes. Standard task templates are bound to the fixed input image and reused automatically across models and seeds.

Generation copies supplied annotation overrides into `videos/.inputs/` and references those snapshots in the manifest. Move the complete `videos/` directory, including `.inputs/`, to keep the evaluation inputs together. When evaluating existing videos directly with `--annotation-root`, keep that external annotation directory available.

For example, save this array as `videos/custom-manifest.json`, with all paths relative to that file and pointing to your actual inputs:

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

Use `python evaluate.py --manifest videos/custom-manifest.json --tasks P33 --output runs/custom_input_check --dry-run` to check those paths before evaluating. The annotations must follow the task's annotation schema; the dry run checks file availability, not annotation geometry or hash correspondence with decoded video frames. Custom scenes must still satisfy the task's physical setup and calibration requirements.

</details>

## Evaluate videos

For the single-task generation above, check its inputs before running the evaluator:

```bash
python evaluate.py --manifest videos/manifest.json --tasks P21 --output runs/input_check --dry-run
python evaluate.py --manifest videos/manifest.json --tasks P21 --output runs/quick_test --resume
```

For the full benchmark, first validate the inventory, then evaluate (or use `--video-root videos` instead of `--manifest` for existing videos):

```bash
python evaluate.py --manifest videos/manifest.json \
  --output runs/input_check_all --require-all-tasks --dry-run

python evaluate.py --manifest videos/manifest.json \
  --output runs/all_tasks --require-all-tasks --resume
```

This evaluates the models present in the manifest, one video at a time, using the first visible GPU. It checks that every included model has all 40 tasks and resumes successful results whose inputs, code, and settings still match. Qwen runs locally; no separate model server is needed. For a different GPU, prefix the evaluation command with `CUDA_VISIBLE_DEVICES=1`.

If you already have benchmark videos and no manifest, arrange them as `videos/MODEL/VIDEO_STEM.mp4` with the current task ID in each filename, then use:

```bash
python evaluate.py --video-root videos --output runs/existing_input_check --dry-run
python evaluate.py --video-root videos --output runs/existing_videos --resume
```

For a flat directory containing one model's videos, add `--model my-model`. Directory scanning uses the bundled task images and prompts, unless a matching generation parameter record supplies the prompt; use a manifest when specifying different inputs or filenames that do not identify a task. Check `input_errors: 0` and the expected video/model/task counts in the dry-run output, and inspect `runs/existing_input_check/input_manifest.json` for details.

Results are saved under `runs/all_tasks/`: `by_model.csv` and `by_task.csv` contain summaries; `results.csv` and `results.json` contain per-video results. Per-video folders include the detailed result and debug evidence. Add `--dry-run` to check the input inventory without loading models. See `python evaluate.py --help` for other options.

`--require-all-tasks` checks the selected task coverage for each included model; without `--tasks`, that means all 40. It does not enforce all eight models or four seeds per task. For the full eight-model experiment, verify 1,280 planned inputs separately. Dry runs validate file inventory, not model execution or physical correctness. A successful single-task run is a useful first check, but does not establish that all 40 evaluators or all eight generators work in a fresh environment.

An evaluation command returns zero when the batch completes without execution/input errors; a low physics score or a failed consistency gate is still a valid evaluated result. Check the summaries and per-video evidence when interpreting performance. If the command returns a nonzero code, inspect `summary.json`, `results.json`, and the affected per-video logs before rerunning with `--resume`.

The individual `task.md` commands run from the task directory. When using those entrypoints directly, also set `FINAL_MODELS_DIR` to the absolute evaluation-model root and `VLM_MODEL` to its `Qwen3.6-27B` directory, so they use the same models as the batch evaluator.

Manifest portability regression checks can be run in the same Linux environment with `python -m unittest discover -s tests -v`. They use temporary fixtures and do not load models.

## Video generation model results

Original-prompt experiment, **2026-10-04**. The planned budget is **160 videos per model** (40 tasks × 4 seeds), or **1,280 videos across 8 models**. Scores range from 0 to 1; higher is better. This table uses the updated observability gate and evidence review, while the abstract retains the earlier paper snapshot. Enhanced-prompt videos are excluded.

**Physics** is the independent physical measurement score over the available evaluated videos; **coverage** is the mean fraction of defined metrics that were measured. The **automatic** score is `0.15 × consistency + 0.85 × physics` when the consistency gate passes (threshold 0.8), and `0.15 × consistency` otherwise. **Reviewed** uses the frozen evidence-review gate decisions. A fresh run produces automatic scores; it does not apply the historical review decisions.

**Direct VLM** is the Qwen3.6-27B visual physics baseline; its one unassessable sample is counted as zero in the available-video mean. Scores use completed evaluations, with unavailable samples and execution errors excluded from evaluator means. The **Planned videos** column shows the fixed benchmark budget.

| Video generation model | Planned videos | Gate pass rate | Coverage | Physics | Automatic | Reviewed | Direct VLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seedance-2.5 | 160 | 98.12% | 88.44% | 0.5111 | 0.5776 | 0.5790 | 0.6228 |
| minimax-h3 | 160 | 91.25% | 88.12% | 0.4912 | 0.5489 | 0.5489 | 0.5259 |
| cosmos3-super-image2video | 160 | 91.14% | 66.77% | 0.3570 | 0.4290 | 0.4290 | 0.4978 |
| vbvr-wan2.2 | 160 | 70.00% | 74.69% | 0.3753 | 0.3779 | 0.3835 | 0.3441 |
| wan2.2-i2v-a14b | 160 | 76.25% | 63.12% | 0.3311 | 0.3566 | 0.3656 | 0.4003 |
| lingbot-video-moe-30b-a3b | 160 | 78.12% | 53.12% | 0.2584 | 0.3225 | 0.3297 | 0.4072 |
| hunyuan-video-1.5-i2v | 160 | 75.62% | 53.44% | 0.2825 | 0.3197 | 0.3159 | 0.4153 |
| cogvideox1.5-5b-i2v | 160 | 51.25% | 38.44% | 0.1781 | 0.1881 | 0.1902 | 0.1541 |
| **Overall** | 1280 | 78.95% | 65.77% | 0.3480 | 0.3900 | 0.3927 | 0.4208 |
