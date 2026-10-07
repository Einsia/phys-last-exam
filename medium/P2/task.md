# P2 · Free fall

Category: 1. Translational Motion and Collisions.

Difficulty: Medium; task rank: 27/40; mean final total score: 0.266128.

Task ID: `P2`.

A fixed side-view camera observes a small ball released from rest into vertical free fall. The full measurable falling interval remains in frame and ends before ground contact. Speed should increase progressively under gravity.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Coefficient of variation of vertical-speed increments over equal time intervals, CV(delta_v_y); dimensionless.
- M2: RMS relative deviation of displacements over three successive equal-duration intervals from the ratio 1:3:5; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, side view. A small ball is released from rest and falls straight downward under gravity. The entire measured fall stays inside the frame, and the ball does not reach or touch the ground during the clip. The camera does not move, pan, or zoom. Plain flat background, no other objects, nothing enters the frame.
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
