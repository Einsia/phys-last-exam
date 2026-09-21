# P11 deterministic evaluator

This directory scores the air-to-water laser task without any LLM or VLM.
The frozen evaluator first fits the horizontal water interface from the
laser-off background. It then detects the laser-on interval using temporal
red-change ridges, and independently fits the incident ray in air and the
refracted ray in water with Hough proposals plus robust IRLS refinement.
The visible normal drawn by a video model is explicitly ignored; the normal is
computed as the perpendicular to the fitted interface.

## Metrics

- `M1 = abs(sin(theta_i) / sin(theta_t) - 1.333)`. The signed residual and
  measured ratio are also retained.
- `M2 = abs(x_incident - x_refracted) / frozen_tank_width`, where each x is the
  independent ray/interface intersection.

The JSON distinguishes extraction, structural, measurement, per-metric, and
physics validity. Missing metrics remain `null`; they are never replaced by a
zero residual. Structural or extraction failure gates the end-to-end score to
zero while leaving the geometric metric score `null`.

The primary score in the completed result set is `continuous-0-1-v1`.
Residual quality is `q(r,s)=1/(1+r/s)`, with the former M1/M2 pass limits used
as half-quality anchors. M1, M2, fit, temporal, and camera dimensions are
combined with a 0.01-floored weighted geometric mean. Hard pass/fail values are
retained only as labels. The original score is preserved as
`legacy_scores_0_100`.

## One video

```bash
python evaluate.py \
  --video /absolute/path/video.mp4 \
  --task_id P11 \
  --output /absolute/path/result.json
```

The run emits JSON plus an overlay video, overlay PNG, off/on keyframes, beam
mask, and temporal diagnostic plot.

## One complete prompt batch

```bash
python run_batch.py \
  --videos-dir /absolute/path/videos \
  --output-dir /absolute/path/evaluation \
  --batch-name prompt_v4_20260826 \
  --workers 4
```

Every MP4 is retained. Prompt versions are run into separate output folders
and are never pooled or filtered by seed.

## Regression tests

```bash
python validate_synthetic.py
```

The full-pipeline tests cover correct Snell geometry, a displaced interface
intersection, an incorrect no-bending ray, a missing refracted ray, a laser
that never switches on, a visible drawn normal, and an extra reflected branch.

Continuous-score regression tests and idempotent score-only migration:

```bash
python test_continuous_scoring.py
python rescore_continuous.py --evaluation-root /absolute/path/evaluation
python audit_continuous_results.py --evaluation-root /absolute/path/evaluation
```

The migration reads existing JSON geometry only; it does not open videos or
rerun extraction.
