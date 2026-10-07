# P16 · Collinear-point cross-ratio

Category: 4. Optics and Projective Geometry.

Difficulty: Easy; task rank: 4/40; mean final total score: 0.710831.

Task ID: `P16`.

A straight rigid rod with four differently colored collinear markers rests against a vertical wall and horizontal floor. Released from rest, its upper end slides down the wall while its lower end moves outward along the floor, producing translation and rotation. Markers remain fixed and the rod stays straight. The four-point cross-ratio should remain invariant under perspective projection.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Coefficient of variation of the four-marker cross-ratio over valid frames, std(chi)/|mean(chi)|; dimensionless.
- M2: Per-frame residual of the collinear four-marker fit divided by marker span, averaged over valid frames; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing rigid straight rod, the four clearly separated coloured markers fixed along the same straight line, the vertical wall, the horizontal floor, the camera view, framing, and background.

The rod is initially motionless in its existing inclined position, with its upper end in contact with the vertical wall and its lower end in contact with the horizontal floor. After a brief still moment, the rod is released from rest and slides under gravity within the same vertical plane. Its upper end moves downward along the wall while its lower end moves horizontally away from the wall along the floor. Both ends remain in contact with their respective surfaces throughout the visible motion.

The rod translates and rotates smoothly as one rigid body. It remains perfectly straight and unchanged in length. All four markers remain permanently fixed at their original positions on the rod, remain collinear, and stay clearly visible throughout the motion. The markers must not slide, detach, swap positions, duplicate, disappear, or change shape or colour.

Keep the complete rod, both endpoints, all four markers, and the wall-floor contact regions visible inside the frame. End after the rod has undergone a clearly visible combination of translation and rotation, before either endpoint leaves its corresponding surface.

The wall, floor, background, and camera remain stationary. No camera movement, panning, zooming, cuts, slow motion, pauses, or time jumps. No hands or people. No new objects, trajectories, guide lines, arrows, measurements, labels, or annotations.
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
