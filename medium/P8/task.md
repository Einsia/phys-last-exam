# P8 · Solid-sphere rolling

Category: 2. Rolling, Friction, and Rigid-Body Statics.

Difficulty: Medium; task rank: 18/40; mean final total score: 0.359700.

Task ID: `P8`.

A sphere with asymmetric bands and an off-center marker is released by a stop without a push, rolls down an incline, and enters a horizontal extension. It stays in contact with the surface; markers are rigidly attached and rotate with it. Translation and rotation remain clearly visible and should satisfy pure rolling, v=omega*R.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Cumulative ball-center displacement/(ball radius * cumulative unwrapped rotation)-1, testing the integral pure-rolling relationship; dimensionless.
- M2: Normalized mean absolute tangential contact-point velocity residual |v-omega*R|; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off, near-orthographic side camera, matching the input first frame.

Preserve the exact inclined ramp, level runout, patterned ball, retractable gate, supports, camera view, framing, materials, and background from the input first frame. Exactly one ball is initially motionless at its existing position on the incline, in contact with the ramp and held by the existing gate. Preserve the ball's exact asymmetric surface band and offset marker.

At the beginning of the shot, the existing gate withdraws cleanly out of the ball's path without pushing, striking, or imparting an additional impulse to the ball. The gate then remains stationary outside the travel path. The ball moves downhill under gravity, visibly rotating as it travels along the incline, through the existing transition, and onto the level runout.

The asymmetric band and offset marker remain rigidly attached to the ball's surface and rotate continuously with the same ball. The pattern must not slide across the surface, remain fixed relative to the camera, swim, morph, mirror, disappear, or change design. The ball remains in contact with the ramp and follows the existing surface without floating, bouncing, sinking into the ramp, or passing through it.

Keep the complete ball and its entire travel path visible inside the frame, including the initial position, the ball–ramp contact region, the full incline, the transition, and the level runout. The ball's circular boundary, surface pattern, and contact region remain sharp enough to observe throughout the motion. Do not use motion blur that conceals the pattern or contact region.

The ramp, rails, supports, background, and camera remain stationary. Preserve the appearance and geometry of the input frame. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No additional balls, duplicated balls, ghost images, motion trails, rotation arrows, trajectory lines, measurements, annotations, or new objects. End while the ball is still completely visible on the level runout.
```

## Evaluation entrypoint

The entrypoint is `evaluator/evaluate.py`. Supply an external video path, the first frame with `--image first_frame.png`, and the prompt file with `--prompt`. The measurement backend uses the bundled scene calibration, so the video must match the selected first-frame calibration.


## Current scene calibration

The measurement backend selects an existing first-frame calibration by filename. Videos generated from this task's first frame should be named `P8_gpt_01_modern_seedN.mp4` (N is any integer) and may reside outside the package. Calibration uses 1344x768 as its reference; the backend applies only its existing scale and aspect-ratio rules. `--sample-id` does not replace filename matching.
