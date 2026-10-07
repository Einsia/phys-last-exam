# P19 · Projection concurrency

Category: 4. Optics and Projective Geometry; legacy ID: `P14`.

Difficulty: Medium; task rank: 19/40; mean final total score: 0.357083.

Task ID: `P19`.

A fixed camera observes a small point light source near the floor illuminating four separate vertical rods. Rods, bases, and clear long shadows are simultaneously visible; source and rods remain stationary. Backward extensions of the shadow axes should meet at one projected source position.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: RMS distance from backward-extended shadow lines to their shared intersection divided by image diagonal; dimensionless.
- M2: Mean x/y standard deviation of source positions estimated from independent line pairs divided by image diagonal; dimensionless. The implementation uses standard-deviation dispersion rather than variance.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Task: P19_point_source_four_shadows.
Start exactly from the supplied first frame: a pale matte floor panel, one small bare white bulb standing upright at the centre close to the floor as the only light, and four slender upright rods - red, blue, yellow, green - spaced well apart around the bulb, each casting one long crisp dark shadow radially outward away from the bulb.

Video action: Locked-off static camera with a clear view of the horizontal ground plane. A single compact point light source illuminates four vertical rods positioned at different locations on the ground, producing four distinct and clearly visible shadows. Every rod, rod base, and shadow tip is visible simultaneously. The camera does not move, pan, or zoom. Plain background, no other objects.
Hold this exact geometry. The bulb stays lit at constant brightness and does not move. All four rods stay standing upright at their starting positions and do not fall, slide, or wobble. All four shadows stay long, sharp, clearly darker than the pale floor beside them, separate from one another, and pointing radially away from the bulb, exactly as in the first frame.

Hard negative constraints: no second light source, no moving or flickering light, no additional or disappearing rods or shadows, no shadows fading out or losing contrast, no person or hand, no labels, numbers, arrows, watermark, logo, cartoon styling, no camera pan, zoom, shake or reframing, no cropped rods or shadow tips.
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

Metric definition references: `v4/g4/P14/evaluator/utils/physeval/tasks/p14.py:48`; `v4/unified_evaluators/contract.py:100`; `v4/g4/P14/evaluator/utils/physeval/tasks/p14.py:56`.
