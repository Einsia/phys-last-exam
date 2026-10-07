# P1 · Bounce-height decay

Category: 1. Translational Motion and Collisions.

Difficulty: Easy; task rank: 9/40; mean final total score: 0.574569.

Task ID: `P1`.

A ball is released from a fixed electromagnetic support, falls vertically onto a hard horizontal surface, and bounces repeatedly. Generation requires at least four clear bounces with successively decreasing peak heights. The release point, contacts, and all peaks remain in view; the surface and support stay fixed.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Consistency of restitution estimates from successive bounces, mean(|sqrt(h_after/h_before)-dt_after/dt_before|), with the score also accounting for restitution exceeding 1; dimensionless.
- M2: Mean relative increase of successive peak heights, mean(max(0,h_after/h_before-1)), combined with the coefficient of variation of successive restitution estimates when measurable; dimensionless.
- For a trackable single motion segment without repeated bounces, the implementation separately records that outcome and assigns both physics scores 0.1.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off, near-orthographic side camera, matching the input first frame.

Preserve the exact apparatus, ball, electromagnetic holder, hard horizontal impact plate, camera view, framing, and background from the input first frame. Exactly one ball is initially motionless at its existing release position, directly above the impact plate.

At the beginning of the shot, the existing electromagnetic holder releases the ball cleanly without moving, falling, or following the ball. The ball then falls vertically under gravity, strikes the hard plate, and makes at least four clear consecutive bounces, with each rebound reaching a lower height than the previous one. All impacts occur at approximately the same horizontal position, with no noticeable sideways drift.

The complete motion remains visible inside the frame: the initial release point, every impact, every rebound apex, and the full ball must never be cropped or occluded. The ball remains exactly one intact sphere throughout the shot, without duplication, disappearance, morphing, or permanent deformation.

The holder, support frame, impact plate, table, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, ghost images, motion trails, trajectory lines, arrows, measurements, annotations, or new objects. End shortly after the ball completes its fourth clearly visible rebound arc and returns to the plate.
```

## Evaluation entrypoint

The entrypoint is `evaluator/evaluate.py`. Supply an external video path, the first frame with `--image first_frame.png`, and the prompt file with `--prompt`. The measurement backend uses the bundled scene calibration, so the video must match the selected first-frame calibration.
