# P18 · Light reflection

Category: 4. Optics and Projective Geometry; legacy ID: `P13`.

Difficulty: Medium; task rank: 22/40; mean final total score: 0.337269.

Task ID: `P18`.

A fixed camera observes a thin laser beam striking a flat mirror obliquely. The incident and reflected rays, mirror, incidence point, and normal are clearly visible. Specular reflection should occur at the same incidence point with equal incidence and reflection angles.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Absolute incidence/reflection angle difference; degrees.
- M2: Median reflected-ray contact-localization error at the mirror divided by image diagonal; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera. A thin clearly visible laser beam strikes a flat plane mirror at an oblique angle. The incident ray, reflected ray, mirror surface, point of incidence, and mirror normal are all clearly visible. The camera does not move, pan, or zoom. Plain dark background, no other objects.
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

Metric definition references: `v4/g2/P13/evaluator/measure_backend.py:652`; `v4/unified_evaluators/contract.py:155`; `v4/g2/P13/evaluator/measure_backend.py:551`.
