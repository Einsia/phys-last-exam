# P5 · Equal-mass collision

Category: 1. Translational Motion and Collisions.

Difficulty: Hard; task rank: 34/40; mean final total score: 0.157530.

Task ID: `P5`.

Two equal-size, equal-mass balls collide head-on on a horizontal surface. A red ball moves right at constant speed toward an initially stationary blue ball. Motion remains horizontal, with speed transfer expected after an equal-mass elastic collision.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: (v1_after+v2_after)/v1_before-1, assessing the equal-mass momentum relationship; dimensionless.
- This task defines only M1.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, side view. Two identical balls of equal size sit on a level horizontal surface. The left (red) ball slides to the right at a steady speed and strikes the right (blue) ball, which is initially at rest. After the head-on collision the balls behave as equal-mass elastic spheres. The motion is purely horizontal and stays inside the frame. The camera does not move, pan, or zoom. Plain flat background, only the two balls on the surface, nothing else enters the frame.
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
