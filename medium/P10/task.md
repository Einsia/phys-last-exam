# P10 · Edge-pivot toppling

Category: 2. Rolling, Friction, and Rigid-Body Statics; legacy ID: `P43`.

Difficulty: Medium; task rank: 26/40; mean final total score: 0.270230.

## Scene and objective

A fixed side-view camera observes an actuator pushing a block. The left push rod continuously extends rightward; its pad stays in contact with the upper part of the left block face, rotating the block about its lower-right support edge until it lies down. The block neither slides along the floor nor leaves the pad to topple freely. The block, contact region, and supporting surface remain visible.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, side view. Continue from the supplied first frame. The linear actuator on the left slowly extends its ram to the right. The pad stays pressed against the upper part of the block's left face and keeps advancing with the ram, pushing the block. The block does not slide along the surface, does not leave the pad, and does not fall over by itself. Keep the ram extending until the block has rotated about its right-hand bottom edge and is lying on the surface. The actuator body stays fixed; only the ram lengthens. The complete block, contact region and supporting surface remain visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 physical metric

M1 checks support-point drift, ground clearance, and rigid-body shape changes during visible rotation.

`E = max(e_pivot, e_contact, e_shape)`

`M1 = I(visible rotation >= 5 degrees) / (1 + E / a)`, with default `a=0.02`.

For `e_pivot` and `e_contact`, subtract twice the pixel-noise floor from the 95th percentiles of support drift and contact gap, clip to nonnegative values, and divide by the initial block diagonal. `e_shape` is relative edge-length change after pixel-noise correction. Thus E and M1 are dimensionless; raw distances are in px and angles in degrees.

Use all continuous visible intervals of at least 4 frames and 0.3 seconds, taking the greatest observed rotation across them. Assign 0 if no visible rotation reaches 5 degrees; do not interpolate hidden poses. The metric does not use the frame difference between center-of-mass crossing and free-toppling onset.

Implementation: `measure` and `normalized_measurement` in [observed_support.py](evaluator/observed_support.py), version `p10_observed_support_v4`.

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
