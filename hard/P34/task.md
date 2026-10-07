# P34 · Solid vs. slotted plate damping

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction.

Difficulty: Hard; task rank: 39/40; mean final total score: 0.142813.

## Scene and objective

A fixed camera observes solid and slotted conducting plates matched in mass and moment of inertia. Released simultaneously from the same angle with zero initial speed, both swing through equivalent magnetic-field regions. Compare complete oscillations within the same observation window to examine the effect of slots on eddy-current damping.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. A solid conducting plate and a slotted conducting plate, matched in total mass and rotational inertia, are released simultaneously from the same initial angle with zero initial speed and swing through equivalent magnetic-field regions. Both oscillations remain fully visible for multiple cycles. The camera does not move, pan, or zoom. Plain background.
```

## M1 physical metric

M1 counts valid complete cycles `N_solid` and `N_slotted` (integers) within a shared observation window and using a shared angular-amplitude threshold.

`M1 = 1 if N_solid < N_slotted, otherwise 0`

When `N_slotted=0`, assign 0 and leave the count ratio null. The default minimum observation window is 2 seconds and the amplitude threshold is `max(1 degree, 0.1 * mean initial absolute angle of both plates)`. Complete cycles pair boundaries of the same polarity; truncated partial cycles are not counted. The metric does not estimate decay rate.

Implementation: `p34` in [measured_center.py](evaluator/measured_center.py) and [observed_zero.py](evaluator/observed_zero.py).

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
