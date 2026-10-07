# P21 · Communicating vessels

Category: 5. Hydrostatics and Buoyancy; legacy ID: `P19`.

Difficulty: Easy; task rank: 1/40; mean final total score: 0.964831.

Task ID: `P21`.

A transparent U-shaped communicating vessel has arms of different widths containing the same continuous liquid. The liquid returns freely from a disturbed state to static equilibrium. Both waterlines remain visible and total liquid quantity stays fixed. At equilibrium, both levels should be equal.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Difference between the median final left and right water levels, divided by reference height; dimensionless.
- M2: Larger absolute linear water-level velocity in the final interval, divided by image height; units are inverse frames, frame^-1.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view. A transparent U-shaped tube has two vertical arms with clearly different diameters and contains the same continuous liquid. The liquid is initially disturbed and then freely settles back to static equilibrium. You need to display how the liquid in the U-shaped tube reaches a state of equilibrium. The total amount of liquid remains constant throughout the entire process, with no liquid added, removed, appearing, or disappearing. Both liquid surfaces remain clearly visible throughout the process. The camera does not move, pan, or zoom. Plain background, no other objects.
```

## Evaluation entrypoint

Run from this task directory; input and output paths may reside outside the task package:

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```

Metric definition references: `v4/g2/P19/evaluator/measure_backend.py:350`; `v4/unified_evaluators/contract.py:155`.
