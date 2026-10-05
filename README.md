# PhysScope: A Multi-Domain Benchmark for Proxy-Based Evaluation of Video Physics

## Abstract

Visually convincing videos can still violate basic physical laws, making reliable physical evaluation essential for assessing video generation models. Existing evaluations often rely on learned judgments or reference videos, while direct physical tests largely focus on mechanics. We introduce PhysScope, a measurement-based benchmark comprising 40 controlled tasks spanning mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface-tension effects. Each task pairs an initial image and a generation prompt with observable quantities and predefined physical criteria. Our evaluation first screens for temporal consistency, excluding videos whose unstable object identities or structures make physical measurements unreliable. Videos that pass are then assessed through task-specific measurements, such as oscillation periods, reflection angles, and liquid levels. We distinguish failures of physical tests from cases with insufficient measurement evidence. Our protocol evaluates eight video generation models on 40 tasks with four seeds per task: 160 planned videos per model and 1,280 in total. Among the evaluated videos, 98.36% pass the temporal-consistency check, but only 159 of the 716 videos with sufficient measurement evidence (22.21%) pass all required physical checks. These findings reveal a substantial gap between temporal coherence and physical consistency, highlighting the need to report measurement coverage alongside physical performance. PhysScope provides an interpretable framework for evaluating diverse physical phenomena while making the limitations of measurement explicit.

## Setup

Use a Linux machine with an NVIDIA GPU. We recommend an 80 GB GPU for the default [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) model and at least 100 GB of free disk space. Install [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) and Git first; the installer uses CUDA 12.8 PyTorch wheels, so a compatible NVIDIA driver is required.

From the repository root, run:

```bash
conda create -n physscope python=3.12 -y
conda activate physscope
bash setup.sh
```

`setup.sh` installs the Python packages and downloads Qwen, Grounding DINO, SAM 2.1, and CoTracker into `models/`. The evaluator finds these models automatically. The first setup downloads about 60 GB; rerun the same command if a download is interrupted. In a new terminal, run `conda activate physscope` again before evaluating.

## Prepare your videos

The benchmark has **40 tasks: 15 easy, 15 medium, and 10 hard**. The unified generator reads their `first_frame.png` and `prompt.txt` automatically and supports the eight models in the results table, plus custom models.

Connect your generation model environments once:

```bash
python generate.py --init-config --model-root /path/to/checkpoints
```

Check the Python, source, and weight paths in `generation.local.json`; see the short [model setup and custom model guide](generation/README.md). Generation models use their own environments; `setup.sh` prepares the evaluator and generation controller.

Generate all 40 tasks with all 8 models and seeds 42–45:

```bash
python generate.py --models all --output videos --resume
```

To start with one model and task, add `--models cogvideox1.5-5b-i2v --tasks P19 --seeds 42` in place of `--models all`. Add `--dry-run` to preview the jobs without loading weights or calling an API.

Videos are written to `videos/MODEL/VIDEO_STEM.mp4`, with matching parameter records and an evaluation-ready `videos/manifest.json`. Special task filenames are handled automatically; P3/P9 receive the required 1344 × 768 evaluation copy while the native video and resize metadata are retained. Frozen images, prompts, and task annotations are stored in `videos/.inputs/`, so the manifest travels with the videos.

The fixed first-frame annotations for P37/P38/P39/P41/P43/P49 are included in their task packages and loaded automatically. You do not need to create an `annotations/` directory for standard benchmark inputs.

<details>
<summary>Calibrated filenames and custom first frames</summary>

The generator assigns these calibrated filenames. If you bring existing videos, use the same names, replacing `N` with the seed:

```text
P3_gpt_01_modern_seedN.mp4
P6_gpt_01_modern_seedN.mp4
P9_gpt_01_modern_seedN.mp4
P11_gpt_01_30deg_seedN.mp4
```

P3/P9 require a 1344 × 768 video canvas. Follow each task's `task.md` for its scene geometry; renaming an incompatible video does not make it calibrated.

P37/P38/P39/P41/P43/P49 use the bundled `first_frame_annotations.json`. The evaluator verifies the input-image hash, scales coordinates to the video resolution, and checks the actual first frame before tracking. A layout that fails this correspondence check is reported as an initialization failure.

For a different input image or layout, provide reviewed video annotations with `--annotation-root annotations`. These optional overrides use:

```text
annotations/MODEL/TASK/VIDEO_STEM.json
# Example: annotations/my-model/P37/g8_P37_seed42.json
```

Custom video annotations must match their video's image/video hashes. Standard task templates are bound to the fixed input image and reused automatically across models and seeds.

</details>

## Evaluate all tasks

From the repository root, run (or use `--video-root videos` instead of `--manifest` for existing videos):

```bash
python evaluate.py --manifest videos/manifest.json \
  --output runs/all_tasks --require-all-tasks --resume
```

This evaluates all models and all 40 tasks, one video at a time, using the first visible GPU. It checks that every model has all 40 tasks and resumes completed videos when rerun. Qwen runs locally; no separate model server is needed. For a different GPU, prefix the command with `CUDA_VISIBLE_DEVICES=1`.

To try just P19 first:

```bash
python evaluate.py --manifest videos/manifest.json --tasks P19 --output runs/quick_test --resume
```

Results are saved under `runs/all_tasks/`: `by_model.csv` and `by_task.csv` contain summaries; `results.csv` and `results.json` contain per-video results. Per-video folders include the detailed result and debug evidence. Add `--dry-run` to check the input inventory without loading models. See `python evaluate.py --help` for other options.

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
