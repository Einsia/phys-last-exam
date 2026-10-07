# P4 · Projectile motion

Category: 1. Translational Motion and Collisions.

Difficulty: Hard; task rank: 33/40; mean final total score: 0.183322.

Task ID: `P4`.

A fixed side-view camera observes a red ball launched from ground level at 45 degrees. It continuously rises, passes its apex, and returns to the same ground height. The complete trajectory should be parabolic with approximately constant horizontal speed and vertical acceleration.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: H/R-tan(theta)/4, where H is apex height, R is range, and theta=45 degrees; dimensionless.
- M2: Combine horizontal-speed CV(v_x), vertical-speed-increment CV(delta_v_y), and parabolic-fit RMS/R. All are dimensionless; normalized physics scores are combined by geometric mean.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, side view. A red ball on the ground launches immediately at 45 degrees above the horizontal, rises smoothly to the top of its arc, and falls back to the same ground level in one continuous trajectory. The whole arc stays inside the frame. The camera does not move, pan, or zoom. Plain flat background, no other objects, nothing enters the frame.
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
