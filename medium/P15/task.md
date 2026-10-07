# P15 · Small-angle isochronism

Category: 3. Pendulum Motion and Oscillations.

Difficulty: Medium; task rank: 17/40; mean final total score: 0.417340.

Task ID: `P15`.

Two equal-length pendulums hang from the same fixed beam. The red bob starts at a smaller angle and the blue bob at a larger angle. Both are released simultaneously from rest and complete multiple oscillations with fixed suspension points and lengths. Under the small-angle approximation, their full periods should be approximately equal.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: T_left/T_right-1, comparing both pendulum periods; dimensionless.
- M2: RMS residuals of sinusoidal fits to both theta(t) trajectories, each in radians; normalized left and right physics scores are combined by geometric mean.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame, which shows two pendulums hanging from the same horizontal bar: a red ball on the left and a blue ball on the right, on strings of exactly the same length, each already pulled aside and held at rest.

Both pendulums are let go at the very same instant. From the first moment of the clip the red ball and the blue ball are both already moving: each one immediately leaves the position it holds in the first frame, swings down through the lowest point, up to the far side, and back, and each keeps swinging back and forth for several complete cycles until the clip ends. Neither ball is ever stationary, neither ball stays parked at its starting position, and neither ball starts later than the other. The red ball swings through a smaller arc than the blue ball because it was released from a smaller angle, but both take the same time to complete one full swing, since their strings are the same length. Both strings stay straight and taut and keep their length; the bar and the two pivot points do not move.

Hard negative constraints: no frozen, still or motionless ball; no ball that stays hanging at its first-frame position while the other swings; no delayed or staggered release; no change of string length; no stretching, bending or slack string; no ball leaving its string; no collision between the two balls; no camera pan, zoom, shake or reframing; no hand, person, arrow, label, number, ruler, text or watermark; nothing else enters the frame.
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
