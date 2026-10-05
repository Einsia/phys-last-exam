# PhysScope: A Multi-Domain Benchmark for Proxy-Based Evaluation of Video Physics

## Abstract

Visually convincing videos can still violate basic physical laws, making reliable physical evaluation essential for assessing video generation models. Existing evaluations often rely on learned judgments or reference videos, while direct physical tests largely focus on mechanics. We introduce PhysScope, a measurement-based benchmark comprising 40 controlled tasks spanning mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface-tension effects. Each task pairs an initial image and a generation prompt with observable quantities and predefined physical criteria. Our evaluation first screens for temporal consistency, excluding videos whose unstable object identities or structures make physical measurements unreliable. Videos that pass are then assessed through task-specific measurements, such as oscillation periods, reflection angles, and liquid levels. We distinguish failures of physical tests from cases with insufficient measurement evidence. We evaluate eight video generation models on 1,278 generated videos. Although 98.36% pass the temporal-consistency check, only 159 of the 716 videos with sufficient measurement evidence (22.21%) pass all required physical checks. These findings reveal a substantial gap between temporal coherence and physical consistency, highlighting the need to report measurement coverage alongside physical performance. PhysScope provides an interpretable framework for evaluating diverse physical phenomena while making the limitations of measurement explicit.

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

The benchmark has **40 tasks: 15 easy, 15 medium, and 10 hard**. Each task folder contains `first_frame.png`, `prompt.txt`, `task.md`, and its evaluator. Generate videos using the task's image and prompt, then organize them by video generation model:

```text
videos/
├── my-model/
│   ├── g2_P19_seed42.mp4
│   ├── P3_gpt_01_modern_seed42.mp4
│   └── ...
└── another-model/
    └── ...
```

Keep the task ID in each filename. Videos are evaluated against the task's supplied image and prompt by default; a matching `VIDEO_STEM_config.json` supplies the actual generation prompt when present. Video generation itself is not included in this repository.

**Before running all 40 tasks:** P3/P6/P9/P11 use fixed scene calibrations, and P37/P38/P39/P41/P43/P49 require annotations for each video's first frame.

<details>
<summary>Required filenames and annotations for these tasks</summary>

Use these calibrated filenames, replacing `N` with the seed:

```text
P3_gpt_01_modern_seedN.mp4
P6_gpt_01_modern_seedN.mp4
P9_gpt_01_modern_seedN.mp4
P11_gpt_01_30deg_seedN.mp4
```

P3/P9 require a 1344 × 768 video canvas. Follow each task's `task.md` for its scene geometry; renaming an incompatible video does not make it calibrated.

For P37/P38/P39/P41/P43/P49, use the task's `first_frame_annotations.json` as a format reference. Annotate the current video's objects and update its source image/video hashes. Store each annotation at:

```text
annotations/MODEL/TASK/VIDEO_STEM.json
# Example: annotations/my-model/P37/g8_P37_seed42.json
```

The bundled annotation files refer to particular historical inputs and cannot be reused for arbitrary videos.

</details>

## Evaluate all tasks

From the repository root, run:

```bash
python evaluate.py --video-root videos --annotation-root annotations \
  --output runs/all_tasks --require-all-tasks --resume
```

This evaluates all models and all 40 tasks, one video at a time, using the first visible GPU. It checks that every model has all 40 tasks and resumes completed videos when rerun. Qwen runs locally; no separate model server is needed. For a different GPU, prefix the command with `CUDA_VISIBLE_DEVICES=1`.

To try just P19 first:

```bash
python evaluate.py --video-root videos --tasks P19 --output runs/quick_test --resume
```

Results are saved under `runs/all_tasks/`: `by_model.csv` and `by_task.csv` contain summaries; `results.csv` and `results.json` contain per-video results. Per-video folders include the detailed result and debug evidence. Add `--dry-run` to check the input inventory without loading models. See `python evaluate.py --help` for other options.

## Video generation model results

Original-prompt experiment, **2026-10-04**, with 8 models and 1,278 videos. Scores range from 0 to 1; higher is better. This table uses the updated observability gate and evidence review, while the abstract retains the earlier paper snapshot. Enhanced-prompt videos are excluded.

**Physics** is the independent physical measurement score over all videos; **coverage** is the mean fraction of defined metrics that were measured. The **automatic** score is `0.15 × consistency + 0.85 × physics` when the consistency gate passes (threshold 0.8), and `0.15 × consistency` otherwise. **Reviewed** uses the frozen evidence-review gate decisions. A fresh run produces automatic scores; it does not apply the historical review decisions.

**Direct VLM** is the Qwen3.6-27B visual physics baseline; its one unassessable sample is counted as zero in this all-video mean. Execution errors are excluded from evaluator means. Missing videos are not assigned zero: Cosmos has 158 videos (missing P13/seed42 and P5/seed45); the other models have 160 each.

| Video generation model | Videos | Gate pass rate | Coverage | Physics | Automatic | Reviewed | Direct VLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seedance-2.5 | 160 | 98.12% | 88.44% | 0.5111 | 0.5776 | 0.5790 | 0.6228 |
| minimax-h3 | 160 | 91.25% | 88.12% | 0.4912 | 0.5489 | 0.5489 | 0.5259 |
| cosmos3-super-image2video | 158 | 91.14% | 66.77% | 0.3570 | 0.4290 | 0.4290 | 0.4978 |
| vbvr-wan2.2 | 160 | 70.00% | 74.69% | 0.3753 | 0.3779 | 0.3835 | 0.3441 |
| wan2.2-i2v-a14b | 160 | 76.25% | 63.12% | 0.3311 | 0.3566 | 0.3656 | 0.4003 |
| lingbot-video-moe-30b-a3b | 160 | 78.12% | 53.12% | 0.2584 | 0.3225 | 0.3297 | 0.4072 |
| hunyuan-video-1.5-i2v | 160 | 75.62% | 53.44% | 0.2825 | 0.3197 | 0.3159 | 0.4153 |
| cogvideox1.5-5b-i2v | 160 | 51.25% | 38.44% | 0.1781 | 0.1881 | 0.1902 | 0.1541 |
| **Overall** | 1278 | 78.95% | 65.77% | 0.3480 | 0.3900 | 0.3927 | 0.4208 |
