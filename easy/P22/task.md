# P22 · Floating-ice immersion

Category: 5. Hydrostatics and Buoyancy.

Difficulty: Easy; task rank: 3/40; mean final total score: 0.761774.

Task ID: `P22`.

A uniform vertical freshwater ice column in a transparent straight-walled vessel is released from the first-frame immersion depth. It freely adjusts under gravity and buoyancy and reaches stable flotation. The column remains intact, does not melt or contact the vessel, and stays fully visible with the waterline. The equilibrium submerged-height ratio should approach the ice/water density ratio, 0.917.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: |median stable submerged-height ratio-0.917|; dimensionless.
- M2: Median absolute left-right waterline difference along the ice column divided by vessel width; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view, matching the input first frame.

A uniform vertical rectangular block of pure freshwater ice is initially motionless at its existing immersion depth in fresh water inside a transparent straight-walled container.

After a brief still moment, the ice is released from its existing position and moves freely under gravity and buoyancy until its motion naturally settles. No support, attachment, or external force acts on the ice after release.

The ice remains one intact upright rectangular block and does not melt, deform, or touch the container walls or bottom. Keep the complete ice block, waterline, and container visible throughout the shot.

The container, background, and camera remain stationary. No camera movement, cuts, people, new objects, labels, arrows, or annotations.

End after the motion has settled for a short moment.
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
