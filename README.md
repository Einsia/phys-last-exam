# World Models’ Last Exam in Physics

[Paper](https://arxiv.org/abs/2610.08791) · [Project page](https://lab.einsia.ai/phys-last-exam)

## Abstract

Video world models can generate plausible scenes that violate physics, limiting their usefulness for prediction and planning in embodied AI. World Models’ Last Exam in Physics evaluates physical consistency through 40 controlled tasks covering mechanics, optics, fluids, thermal and phase-change phenomena, electromagnetism, and surface tension. Every task supplies an initial image, a generation prompt, and measurable physical criteria, allowing evaluation without reference videos. The evaluator screens task observability and temporal consistency, independently measures task-specific physical quantities, and uses the screening outcome to gate their contribution to the composite score. Across eight video generation models and 1,280 videos, results show persistent inconsistencies and substantial differences between tasks; the strongest model scores 57.76 out of 100. Tests on synthetic videos with known physical relationships support the measurement module under controlled conditions. The evaluator also agrees more closely with human judgments than direct VLM scoring in both within-task rankings and pairwise comparisons. Its measurements and explicit evidence limitations help diagnose model failures and track progress in physical consistency.

## Quick Start

On Linux with Python 3.12 and a compatible NVIDIA GPU/driver, run these commands from the repository root:

```bash
bash scripts/generate.sh cogvideox1.5-5b-i2v
bash scripts/evaluate_all.sh videos
```

Choose a model and copy its generation command below, then run `bash scripts/evaluate_all.sh videos`:

| Model | Generation command |
| --- | --- |
| Seedance 2.5 | `bash scripts/generate.sh seedance-2.5` |
| MiniMax H3 | `bash scripts/generate.sh minimax-h3` |
| Cosmos 3 Super | `bash scripts/generate.sh cosmos3-super-image2video` |
| VBVR Wan2.2 | `bash scripts/generate.sh vbvr-wan2.2` |
| Wan 2.2-A14B | `bash scripts/generate.sh wan2.2-i2v-a14b` |
| LingBot 30B-A3B | `bash scripts/generate.sh lingbot-video-moe-30b-a3b` |
| Hunyuan 1.5 | `bash scripts/generate.sh hunyuan-video-1.5-i2v` |
| CogVideoX 1.5-5B | `bash scripts/generate.sh cogvideox1.5-5b-i2v` |

- **Seedance:** the launcher prompts for your compatible Videos API base URL and API key; key input is hidden. For unattended runs, set `SEEDANCE_BASE_URL` and `SEEDANCE_API_KEY`. See [API setup](generation/README.md#setup) for provider formats.
- **Local models:** the launcher installs the selected model's environment and downloads its checkpoints on first use. To reuse an existing installation, add `--config generation.local.json` with [your Python, source, checkpoint, and GPU paths](generation/README.md#reuse-an-existing-installation). Model-specific prerequisites are listed in [setup](generation/README.md#setup).

[Environment and API setup](generation/README.md#setup) · [Generation options](generation/README.md#one-command-generation) · [Evaluation options](docs/evaluation.md)

## Evaluate your own model

Generate videos using each task's bundled `first_frame.png` and `prompt.txt`. Save them in `videos/my-model/` with the task ID in each filename, for example `P21_seed42.mp4`, then run:

```bash
bash scripts/evaluate_all.sh videos/my-model --model my-model
```

To run your own generator through the benchmark, create `custom-model.json` with [your inference command](generation/README.md#add-a-custom-model), then use the same two-step workflow:

```bash
bash scripts/generate.sh my-model --config custom-model.json
bash scripts/evaluate_all.sh videos
```

## Video generation model results

The following reproduces the full task-level results in [the paper’s Table 2](https://arxiv.org/pdf/2610.08791v1#page=8). Scores use a **0–100** scale; higher is better. **Bold** marks each row’s highest score, including ties.

For each video, `S = 0.15 × C + 0.85 × P × 1[C ≥ 80]`, where `C` is automatic consistency/observability and `P` is the independently measured physical score. Scores are averaged after applying the gate to each video. **✓** means every available video in that model–task pair has `C ≥ 80`; **✗** means at least one falls below 80. **E/M/H** denote the paper’s empirical Easy/Medium/Hard groups.

| ID | Task | Level | Seedance 2.5 | MiniMax H3 | Cosmos 3 Super | VBVR Wan2.2 | Wan 2.2-A14B | LingBot 30B-A3B | Hunyuan 1.5 | CogVideoX 1.5-5B |
| --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| | **1. Translational Motion and Collisions** |  |  |  |  |  |  |  |  |  |
| [P1](easy/P1/task.md) | Bounce-height decay | E | **90.76** ✓ | 88.19 ✓ | 57.46 ✓ | 0.00 ✗ | 85.66 ✓ | 48.67 ✓ | 83.03 ✓ | 3.75 ✗ |
| [P2](medium/P2/task.md) | Free fall | M | 29.50 ✓ | 25.84 ✓ | **34.54** ✓ | 24.27 ✓ | 22.72 ✓ | 34.41 ✓ | 23.60 ✓ | 18.03 ✓ |
| [P3](medium/P3/task.md) | Complementary-angle throws | M | 53.96 ✓ | **57.66** ✓ | 20.97 ✓ | 10.88 ✗ | 0.00 ✗ | 30.87 ✗ | 3.38 ✗ | 0.00 ✗ |
| [P4](hard/P4/task.md) | Projectile motion | H | 29.13 ✓ | **39.01** ✓ | 17.25 ✗ | 22.33 ✗ | 10.36 ✗ | 15.43 ✗ | 7.83 ✗ | 0.00 ✗ |
| [P5](hard/P5/task.md) | Equal-mass collision | H | 25.37 ✓ | **26.06** ✓ | 14.50 ✓ | 17.05 ✗ | 14.79 ✗ | 4.97 ✗ | 11.25 ✗ | 7.50 ✗ |
| | **2. Rolling, Friction, and Rigid-Body Statics** |  |  |  |  |  |  |  |  |  |
| [P6](easy/P6/task.md) | Mass-independent sliding | E | 66.00 ✓ | 97.89 ✓ | **98.46** ✓ | 51.12 ✓ | 50.98 ✓ | 57.50 ✓ | 36.06 ✓ | 43.62 ✗ |
| [P7](easy/P7/task.md) | Hanging-chain equilibrium | E | 68.85 ✓ | 70.29 ✓ | 71.09 ✓ | 69.71 ✓ | 71.26 ✓ | 69.11 ✓ | **76.95** ✓ | 69.60 ✓ |
| [P8](medium/P8/task.md) | Solid-sphere rolling | M | **75.56** ✓ | 45.82 ✓ | 28.79 ✓ | 31.53 ✓ | 35.39 ✓ | 18.77 ✓ | 37.11 ✓ | 14.78 ✗ |
| [P9](medium/P9/task.md) | Solid sphere vs. hoop | M | 32.01 ✓ | 40.83 ✓ | 30.36 ✓ | **41.63** ✓ | 17.28 ✓ | 22.08 ✓ | 39.36 ✓ | 3.75 ✗ |
| [P10](medium/P10/task.md) | Edge-pivot toppling | M | 34.27 ✓ | **66.53** ✓ | 34.03 ✓ | 4.76 ✗ | 43.23 ✓ | 14.38 ✗ | 18.99 ✓ | 0.00 ✗ |
| [P11](hard/P11/task.md) | Rough-incline round trip | H | 24.43 ✓ | **28.84** ✓ | 8.10 ✗ | 12.36 ✗ | 16.53 ✗ | 19.18 ✗ | 10.10 ✗ | 0.00 ✗ |
| | **3. Pendulum Motion and Oscillations** |  |  |  |  |  |  |  |  |  |
| [P12](easy/P12/task.md) | Pendulum period vs. mass | E | **80.41** ✓ | 72.58 ✓ | 68.16 ✓ | 56.28 ✗ | 35.56 ✓ | 19.25 ✗ | 52.08 ✓ | 40.38 ✓ |
| [P13](easy/P13/task.md) | Large-angle pendulum | E | 57.76 ✓ | **79.88** ✓ | 62.81 ✓ | 43.59 ✗ | 53.77 ✓ | 46.92 ✓ | 73.52 ✓ | 48.41 ✗ |
| [P14](easy/P14/task.md) | Pendulum period vs. length | E | 66.15 ✓ | 65.41 ✓ | 61.93 ✓ | 55.83 ✓ | **67.11** ✓ | 25.36 ✓ | 16.00 ✗ | 26.03 ✗ |
| [P15](medium/P15/task.md) | Small-angle isochronism | M | **82.60** ✓ | 51.35 ✓ | 66.77 ✓ | 21.90 ✗ | 29.69 ✓ | 23.40 ✗ | 32.52 ✗ | 25.64 ✓ |
| | **4. Optics and Projective Geometry** |  |  |  |  |  |  |  |  |  |
| [P16](easy/P16/task.md) | Collinear-point cross-ratio | E | 81.10 ✓ | 82.97 ✓ | 82.39 ✓ | 83.92 ✓ | 68.12 ✓ | 43.56 ✗ | **88.34** ✓ | 3.75 ✗ |
| [P17](medium/P17/task.md) | Light refraction | M | 65.59 ✓ | 65.68 ✓ | 56.38 ✓ | **66.18** ✓ | 57.94 ✓ | 14.62 ✓ | 15.00 ✓ | 15.00 ✓ |
| [P18](medium/P18/task.md) | Light reflection | M | **96.15** ✓ | 49.82 ✗ | 0.00 ✗ | 95.75 ✓ | 0.00 ✗ | 3.38 ✗ | 24.73 ✗ | 0.00 ✗ |
| [P19](medium/P19/task.md) | Projection concurrency | M | 67.14 ✓ | 45.62 ✗ | 48.69 ✓ | **69.64** ✓ | 32.45 ✓ | 11.62 ✗ | 7.12 ✗ | 3.38 ✗ |
| [P20](hard/P20/task.md) | Refraction and reflection | H | 9.09 ✗ | 31.67 ✓ | 13.69 ✗ | 22.86 ✗ | 14.50 ✗ | 14.88 ✗ | **39.66** ✓ | 7.87 ✗ |
| | **5. Hydrostatics and Buoyancy** |  |  |  |  |  |  |  |  |  |
| [P21](easy/P21/task.md) | Communicating vessels | E | 97.01 ✓ | 98.16 ✓ | 97.70 ✓ | **99.01** ✓ | 93.09 ✓ | 97.40 ✓ | 95.44 ✓ | 94.05 ✓ |
| [P22](easy/P22/task.md) | Floating-ice immersion | E | 66.56 ✓ | 83.22 ✓ | 78.82 ✓ | 85.29 ✓ | **85.86** ✓ | 68.96 ✓ | 67.32 ✓ | 73.38 ✓ |
| [P23](medium/P23/task.md) | Liquid-surface orientation | M | 57.52 ✗ | **72.68** ✓ | 8.32 ✗ | 0.00 ✗ | 21.06 ✗ | 48.36 ✗ | 34.72 ✗ | 0.00 ✗ |
| | **6. Phase Transitions and Melting** |  |  |  |  |  |  |  |  |  |
| [P24](easy/P24/task.md) | Freezing-induced expansion | E | 94.30 ✓ | **96.21** ✓ | 66.69 ✗ | 24.38 ✗ | 0.00 ✗ | 86.43 ✓ | 23.71 ✗ | 0.00 ✗ |
| [P25](hard/P25/task.md) | Ice melting: water level | H | **95.61** ✓ | 0.00 ✗ | 15.00 ✓ | 0.00 ✗ | 0.00 ✗ | 3.75 ✗ | 0.00 ✗ | 0.00 ✗ |
| [P26](hard/P26/task.md) | Ice with a stone: melting | H | **15.00** ✓ | 0.00 ✗ | 3.75 ✗ | 0.00 ✗ | 0.00 ✗ | 8.81 ✗ | 0.00 ✗ | 0.00 ✗ |
| [P27](hard/P27/task.md) | Freshwater ice in saltwater | H | 46.87 ✓ | 0.00 ✗ | **57.50** ✓ | 0.00 ✗ | 0.00 ✗ | 0.00 ✗ | 0.00 ✗ | 3.75 ✗ |
| [P28](hard/P28/task.md) | Crushed vs. intact ice | H | **57.50** ✓ | 36.25 ✓ | 15.00 ✓ | 15.00 ✓ | 15.00 ✓ | 7.50 ✗ | 7.50 ✗ | 0.00 ✗ |
| | **7. Electrostatics, Magnetism, and Electromagnetic Induction** |  |  |  |  |  |  |  |  |  |
| [P29](easy/P29/task.md) | Eddy-current braking | E | 36.25 ✓ | **89.38** ✓ | 78.75 ✓ | 39.38 ✗ | 53.75 ✗ | 57.50 ✓ | 32.50 ✗ | 3.75 ✗ |
| [P30](easy/P30/task.md) | Coil-induced light emission | E | 57.50 ✓ | 57.50 ✓ | **68.12** ✓ | 57.50 ✓ | 57.50 ✓ | **68.12** ✓ | 32.50 ✗ | 3.75 ✗ |
| [P31](medium/P31/task.md) | Charged-sphere equilibrium | M | **98.34** ✓ | 97.24 ✓ | 15.00 ✓ | 15.00 ✓ | 15.00 ✓ | 10.88 ✗ | 14.81 ✓ | 15.00 ✓ |
| [P32](medium/P32/task.md) | Final compass orientations | M | **36.43** ✓ | 15.05 ✓ | 15.00 ✓ | 18.01 ✓ | 36.32 ✓ | 16.41 ✓ | 15.00 ✓ | 15.95 ✓ |
| [P33](medium/P33/task.md) | Closed vs. open jumping rings | M | 23.00 ✓ | 20.84 ✓ | 15.00 ✓ | 15.00 ✓ | 36.25 ✓ | **57.50** ✓ | 15.00 ✓ | 15.00 ✓ |
| [P34](hard/P34/task.md) | Solid vs. slotted plate damping | H | **36.25** ✓ | 15.00 ✓ | 15.00 ✓ | 0.00 ✗ | 15.00 ✓ | 7.50 ✗ | 15.00 ✓ | 10.50 ✗ |
| | **8. Granular Media and Discharge Flow** |  |  |  |  |  |  |  |  |  |
| [P35](easy/P35/task.md) | Sandpile angle scaling | E | **99.08** ✓ | 94.21 ✓ | 73.64 ✓ | 98.55 ✓ | 98.08 ✓ | 30.11 ✓ | 97.18 ✓ | 93.18 ✓ |
| [P36](hard/P36/task.md) | Sand vs. water discharge | H | **38.39** ✓ | 21.89 ✓ | 15.25 ✓ | 7.12 ✗ | 16.44 ✗ | 10.69 ✗ | 7.88 ✗ | 7.81 ✗ |
| | **9. Surface Tension and Viscous Flow** |  |  |  |  |  |  |  |  |  |
| [P37](easy/P37/task.md) | Capillary rise vs. diameter | E | 78.75 ✓ | **100.00** ✓ | 57.12 ✓ | **100.00** ✓ | 36.25 ✓ | 11.44 ✗ | 15.00 ✓ | 10.88 ✗ |
| [P38](easy/P38/task.md) | Viscous settling speed | E | 66.88 ✓ | 67.25 ✓ | 68.50 ✓ | **69.85** ✓ | 50.86 ✗ | 65.91 ✓ | 51.39 ✗ | 58.97 ✓ |
| [P39](medium/P39/task.md) | Bubble-film curvature | M | 15.00 ✓ | **55.08** ✓ | 25.30 ✓ | 28.79 ✓ | 42.26 ✓ | 32.41 ✓ | 42.62 ✓ | 15.00 ✓ |
| [P40](medium/P40/task.md) | Droplet volume conservation | M | 58.30 ✓ | 43.52 ✓ | 32.46 ✓ | 37.07 ✗ | 26.31 ✗ | **62.00** ✓ | 14.62 ✓ | 0.00 ✗ |
| | **Average score** | | **57.76** | 54.89 | 42.46 | 37.79 | 35.66 | 32.25 | 31.97 | 18.81 |
