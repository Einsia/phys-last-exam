# P31 · Charged-sphere equilibrium

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction; legacy ID: `P28`.

Difficulty: Medium; task rank: 20/40; mean final total score: 0.351582.

Task ID: `P31`.

Two identical charged balls hang symmetrically from equal-length insulating strings, repel one another, and freely reach static equilibrium. Both balls and complete strings remain clearly visible. The final configuration should be outward-separated and symmetric.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Absolute difference between the equilibrium suspension-line angles relative to the vertical; degrees.
- M2: Absolute logarithm of the horizontal-displacement ratio about the apparatus centerline, |ln(d_left/d_right)|; dimensionless.
- The implementation also checks visible outward separation, stationarity, and fixed equal-length strings. An observed configuration failing these conditions receives both physics scores 0.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view. Two identical charged balls hang from two insulating threads of exactly equal length, arranged symmetrically from the same support. The balls repel each other and freely settle into a stable static configuration. Both balls and both complete threads remain clearly visible. The camera does not move, pan, or zoom. Plain background, no other objects.
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

Metric definition references: `v4/g2/P28/evaluator/measure_backend.py:378`; `v4/unified_evaluators/contract.py:155`.
