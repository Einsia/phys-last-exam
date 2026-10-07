# P32 · Final compass orientations

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction.

Difficulty: Medium; task rank: 30/40; mean final total score: 0.210206.

## Scene and objective

A top-view camera observes a fixed wire and two compasses. After the wire is energized, both needles rotate about their centers and settle. Compare their final magnetic-pole directions on opposite sides of the wire. The wire, base, and compass cases remain fixed.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off top-down camera. Continue from the supplied first frame. A current begins to flow in the vertical wire. The only motion is the two compass needles turning on their own pivots on the compass faces; they then settle into stable orientations. The wire, the board, the compass housings and the rest of the apparatus stay exactly as in the first frame. The camera does not move, pan, or zoom. Plain background.
```

## M1 physical metric

M1 measures how opposite the final needle directions are. Use frames among the last 5 where both needles are readable, requiring at least 2. Compute the smaller angle between directions in each frame and take the median d (degrees, range 0-180).

`M1 = (1 - cos(d × π / 180)) / 2`

Opposite directions at 180 degrees receive 1; aligned directions at 0 degrees receive 0. M1 uses final directions only, without scoring first-frame-relative deflection, rotation trajectories, or stabilization duration.

Implementation: `summarize` in [measurement.py](evaluator/utils/measurement.py).

## Evaluation entrypoint

Run from this task directory:

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use a Python environment with evaluation dependencies and the consistency model configured.
