# P3 · Complementary-angle throws

Category: 1. Translational Motion and Collisions.

Difficulty: Medium; task rank: 29/40; mean final total score: 0.222125.

Task ID: `P3`.

Two fixed launchers simultaneously fire balls rightward at 30 and 60 degrees with equal initial speed magnitudes. Each flies independently along its own horizontal lane and lands at its launch height. Complete flights and first landing points remain visible. Ideal gravitational projectiles should have equal ranges.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: R_30/R_60-1, comparing complementary-angle ranges; dimensionless.
- M2: Combine each trajectory parabolic-fit residual normalized by ball diameter with relative initial-speed-magnitude deviation. All components are dimensionless; normalized physics scores are combined by geometric mean.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off, slightly elevated near-orthographic side camera, matching the input first frame.

Preserve the exact two-lane parallel layout and all existing apparatus. Exactly two identical, rigidly mounted launchers release exactly two identical balls simultaneously toward the right with exactly the same initial speed. One launcher is oriented at 30 degrees above the horizontal, and the other is oriented at 60 degrees above the horizontal. Preserve each launcher's existing angle and lane assignment from the input first frame. Do not swap, rotate, move, or reorder either launcher.

After release, both balls move freely through the air under gravity. Each ball remains a distinct rigid sphere and lands on its own corresponding horizontal landing lane. At first landing contact, the center of each ball returns to the same vertical level as its center at launch. Both complete flights and both first landing points remain visible inside the frame.

The launchers, lanes, supports, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No new objects or people. No trajectory lines, annotations, motion trails, ghost images, duplicated balls, disappearing balls, or collisions between the two balls. End shortly after both balls make their first landing contact.
```

## Evaluation entrypoint

The entrypoint is `evaluator/evaluate.py`. Supply an external video path, the first frame with `--image first_frame.png`, and the prompt file with `--prompt`. The measurement backend uses the bundled scene calibration, so the video must match the selected first-frame calibration.


## Current scene calibration

The measurement backend selects an existing first-frame calibration by filename. Videos generated from this task's first frame should be named `P3_gpt_01_modern_seedN.mp4` (N is any integer) and use the configured 1344x768 measurement canvas. Videos may reside outside the task package. `--sample-id` does not replace backend filename matching.

The first-frame-to-measurement transform remains PIL LANCZOS resizing to 1344x768. The root `first_frame.png` was updated to a new scene on 2026-10-06. Earlier conclusions about pixel identity with the old calibrated input no longer apply; the backend calibration has not yet been updated for the new first frame and must be checked and updated before evaluating the new scene.
