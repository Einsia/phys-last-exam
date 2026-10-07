# P23 · Liquid-surface orientation

Category: 5. Hydrostatics and Buoyancy.

Difficulty: Medium; task rank: 23/40; mean final total score: 0.323625.

Task ID: `P23`.

Beside a stationary liquid surface in a transparent vessel, a bare steel ball is released from a fixed electromagnetic support and falls vertically without a string. The full measurable fall and flat waterline remain visible, ending before ground contact. Gravitational acceleration should be perpendicular to the surface, with a constant-acceleration trajectory.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Angle between ball acceleration and the water-surface tangent minus 90 degrees; degrees.
- M2: Combine waterline straight-fit RMS and ball-trajectory constant-acceleration-fit RMS. Both raw quantities are in pixels; normalized physics scores are combined by geometric mean.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing transparent container, clear still liquid, flat horizontal free surface, steel ball, stationary release apparatus, camera view, framing, and background.

Exactly one small dense steel ball is initially motionless at its existing position beside the container, held by the existing stationary electromagnetic holder. The ball is one bare, smooth, rigid sphere with no string, hook, ring, cable, rod, clip, cap, or other object attached to it.

At the beginning of the shot, the existing holder releases the ball cleanly and remains completely fixed. The ball separates fully from the holder and falls vertically downward under gravity as one bare sphere. No part of the holder detaches, falls, or follows the ball. The ball does not remain tethered, swing like a pendulum, or drift noticeably sideways.

Keep the complete measured fall inside the frame and end before the ball reaches or touches the ground. The full ball and the flat liquid surface remain clearly visible simultaneously throughout the measured interval. The liquid surface remains still and horizontal.

The container, liquid, release apparatus, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, duplicated objects, motion trails, annotations, or newly appearing objects.
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
