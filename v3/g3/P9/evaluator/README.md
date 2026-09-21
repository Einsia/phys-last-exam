# P9 deterministic evaluator

This scorer measures the period–length law for two pendulums without using an
LLM or VLM. SAM2 is used only for frame-zero point-prompted bob segmentation;
CoTracker supplies point coordinates for both bobs, pivots and static background.
All identity assignments are frozen from the input first frame.

The primary metric is

`M1 = abs((T_short / T_long)^2 / (L_short / L_long) - 1)`.

`L_short` and `L_long` are frozen from frame zero. Periods are estimated twice:
turning-point intervals and a robust continuous sinusoidal fit. The evaluator
also reports per-pendulum period CV, fit R², completed cycles, simultaneous
release error, pivot drift, string-length drift, pivot-separation drift and a
SIFT/RANSAC camera audit.

Hard structural gates (camera, pivot or string drift) cannot be averaged away.
The JSON always separates `extract_success`, `structural_ok`,
`measurement_valid`, and `physics_pass`. Missing measurements are JSON `null`,
never a fabricated zero residual.

## Single video

```bash
python evaluate.py --video VIDEO.mp4 --task_id P9 --output result.json
```

The output includes a diagnostic plot and an H.264 overlay with frozen identities,
tracked pivots, bob centres and trajectory tails.

## Formal batch

```bash
python run_batch.py --videos ../minimax_h3/prompt_v1_20260825/videos \
  --output-root ../evaluation/minimax_h3_prompt_v1_20260825 --device cuda:7 \
  --workers 2 --force
```

`validate_metrics.py` is the synthetic regression test. `config.yaml` is frozen
before the formal 24-video run and its SHA-256 is embedded in every result.
