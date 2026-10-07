# P33 · Closed vs. open jumping rings

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction.

Difficulty: Medium; task rank: 28/40; mean final total score: 0.246987.

## Scene and objective

A fixed camera observes two adjacent jumping-ring apparatuses, one with a closed aluminum ring and one with a gap. Both coils are energized simultaneously, and the rings move freely along their cores. Compare ring rise heights to observe induction in a closed circuit. Coils, cores, bases, and wires stay fixed; both rings remain distinguishable.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical jumping-ring apparatuses sit side by side. Each copper coil, iron core, stand and wiring stay bolted in place at the same compact size and height and do not stretch, lift, tilt or translate. Each aluminium ring is a separate loose part around its own core. Both coils are switched on at the same instant; thereafter each ring is free to slide along its core. Both rings remain clearly visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 physical metric

M1 compares maximum rises relative to initial ring positions. Each height is divided by that ring's initial outer diameter, giving dimensionless `h_closed` and `h_open`; raw pixel heights and peak times are retained separately.

`r = h_open / h_closed`

`M1 = clip((1 - r) / 0.2, 0, 1)`

The default `margin=0.2` gives 1 when the open ring relative rise is no more than 80% of the closed ring rise. Peaks require plateau or descent evidence; an incompletely observed peak is not a reliable height. A continuously observed zero reference height may receive 0, while an undefined height ratio remains null.

Implementation: `p33` in [tasks.py](evaluator/_shared/refined_evaluators/tasks.py) and [observed_zero.py](evaluator/observed_zero.py).

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
