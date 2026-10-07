# P35 · Sandpile angle scaling

Category: 8. Granular Media and Discharge Flow; legacy ID: `P40`.

Difficulty: Easy; task rank: 2/40; mean final total score: 0.855037.

Task ID: `P35`.

Two separate piles of different sizes made from the same dry, noncohesive sand rest on a horizontal surface. Two identical thin sand streams fall onto their peaks. Grains slide down the slopes and settle; the piles do not merge and their full outlines remain visible. Their angles of repose should be approximately equal.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: |small-pile angle of repose / large-pile angle of repose - 1|; dimensionless.
- M2: Compute each pile left-right slope-angle difference and take the larger value; degrees.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two piles of the same dry non-cohesive sand stand at rest on the same flat horizontal bench, the left pile clearly larger than the right one. A thin steady stream of the same dry sand begins to fall straight down onto the apex of each pile from above the top edge of the frame, at the same steady rate for both. Each pile grows as the sand lands, and the loose grains keep running down its slope faces and settling, so both piles stay conical and keep their slope faces clean and continuous down to the bench line. The two piles stay separate and never merge. The sand source stays outside the frame; no hopper, funnel, tube, container or hand ever enters the frame. Both complete pile profiles and the straight horizontal bench line remain clearly visible for the whole clip. The bench, the background and the camera stay exactly as in the first frame. The camera does not move, pan, or zoom. Plain background, no other objects.
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

Metric definition references: `v4/g2/P40/evaluator/measure_backend.py:315`; `v4/unified_evaluators/contract.py:155`.
