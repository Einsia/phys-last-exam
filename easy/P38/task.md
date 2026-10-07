# P38 · Viscous settling speed

Category: 9. Surface Tension and Viscous Flow.

Difficulty: Easy; task rank: 6/40; mean final total score: 0.642551.

## Scene and objective

A fixed front-view camera observes two spheres of the same material but clearly different radii falling vertically in one deep glycerol vessel. Both remain visible long enough before bottom contact to form identifiable stable terminal-speed intervals, allowing comparison of terminal speeds against squared radii.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, front view. Two spherical balls made of exactly the same material but with clearly different radii are released into the same deep transparent tank filled with glycerin. Both balls fall vertically through the liquid and remain visible long enough to reach clear steady terminal-speed regimes before reaching the bottom. The camera does not move, pan, or zoom. Plain background, no other objects.
```

## M1 physical metric

M1 separately measures large and small sphere radii `r_large`, `r_small` (px) and terminal speeds `v_large`, `v_small` (px/s).

`E = |(v_large/v_small) / (r_large/r_small)² - 1|`

`M1 = 1 / (1 + E / a)`, with default `a=1.0`.

Ratios, error, and score are dimensionless. For each sphere, independently select the longest eligible nonzero terminal-speed interval before bottom contact, taking the earlier interval if tied. The default minimum radius ratio is 1.1. Identical material and fluid are task assumptions. Physical sizes and fluid parameters are not calibrated, so pixel speeds are not SI speeds and a low Reynolds number is not directly measured.

Implementation: `p38` in [tasks.py](evaluator/_shared/refined_evaluators/tasks.py).

## Evaluation entrypoint

Run from this task directory:

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use the configured evaluation environment. The evaluator automatically loads `first_frame_annotations.json` bundled with this task. It verifies the input-image hash, scales the coordinates to the video resolution, and checks correspondence with the decoded first frame before tracking. No per-video annotation is needed for the fixed task image when this check passes. Use `--annotation /absolute/path/to/video_annotations.json` only to override initialization for a different image or layout; custom video annotations retain their video/image hash checks.
