# P14 · Pendulum period vs. length

Category: 3. Pendulum Motion and Oscillations.

Difficulty: Easy; task rank: 14/40; mean final total score: 0.479792.

Task ID: `P14`.

Two identical bobs hang from fixed suspension points at the same height with a length ratio of 1:2. They are released simultaneously from approximately equal small angles on the same side. Each completes at least one full oscillation; lengths remain fixed and the full motion is visible. Their squared-period ratio should equal their length ratio.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: |(T_short/T_long)^2/(L_short/L_long)-1|; dimensionless.
- M2: Period coefficients of variation for the short and long pendulums; both are dimensionless, and their normalized physics scores are combined by geometric mean.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off, near-orthographic frontal camera, matching the input first frame.

Preserve the exact support frame, common overhead beam, two fixed pivot points, exactly two strings, exactly two identical spherical bobs, camera view, framing, materials, and background from the input first frame. Preserve each pendulum's existing position and lane assignment; do not swap or reorder the short and long pendulums.

Preserve the existing free-string length ratio of 1:2, the equal pivot height, the identical bob sizes and appearances, and the initial string directions shown in the input frame. Both pendulums are initially motionless on the same side of their equilibrium positions, at approximately the same initial angular displacement.

At the beginning of the shot, both pendulums are released simultaneously from rest without any visible hand, push, added impulse, or newly appearing release mechanism. Each bob swings freely back and forth under gravity about its own fixed pivot and remains within its own separate vertical swing plane.

Each string remains taut, straight, attached to its original pivot and bob, and unchanged in length throughout the shot. Each bob remains attached to the end of its original string. The strings must not stretch, bend, detach, cross, merge, switch bobs, or change length. The two bobs must not collide or pass into each other's swing lanes.

Continue the shot long enough for each pendulum to complete at least one full oscillation, from its initial side to the opposite turning point and back to its initial side. Keep every turning point and the complete swing arc of both pendulums visible. Do not skip, merge, or conceal any turning point.

The support frame, beam, pivots, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional pendulums, strings, bobs, or support parts. No auxiliary vertical lines, rods, catches, dots, guide marks, angle arcs, clocks, trajectory lines, arrows, measurements, annotations, ghost images, or motion trails. End shortly after both pendulums have each completed at least one full oscillation.
```

## Evaluation entrypoint

The entrypoint is `evaluator/evaluate.py`. Supply an external video path, the first frame with `--image first_frame.png`, and the prompt file with `--prompt`. The measurement backend uses the bundled scene calibration, so the video must match the selected first-frame calibration.


## Current scene calibration

The measurement backend selects an existing first-frame calibration by filename. Videos generated from this task's first frame should be named `P14_gpt_01_modern_seedN.mp4` (N is any integer) and use the configured 1344x768 measurement canvas. Videos may reside outside the task package. `--sample-id` does not replace backend filename matching.

The first-frame-to-measurement transform remains PIL LANCZOS resizing to 1344x768. The root `first_frame.png` was updated to a new scene on 2026-10-06. Earlier conclusions about pixel identity with the old calibrated input no longer apply; the backend calibration has not yet been updated for the new first frame and must be checked and updated before evaluating the new scene.
