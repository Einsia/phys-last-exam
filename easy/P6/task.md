# P6 · Mass-independent sliding

Category: 2. Rolling, Friction, and Rigid-Body Statics; legacy ID: `P42`.

Difficulty: Easy; task rank: 7/40; mean final total score: 0.608458.

Task ID: `P6`.

Two blocks of different masses rest side by side on an approximately horizontal board. One end is hinged; a screw lift continuously raises the other end. Both blocks begin sliding freely at sufficient inclination, and filming continues until both visibly slide. With identical contact conditions, critical sliding inclination should be independent of mass.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: Absolute difference between the two sliding-onset frames divided by total video frames; dimensionless.
- M2: Absolute difference between board inclinations at the two sliding onsets; degrees.
- When both blocks are measurable but neither begins sliding, the implementation records a stationary outcome and assigns both physics scores 0.7.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, side view. Continue from the supplied first frame. The board starts lying almost flat with the two blocks resting side by side on it. The screw jack under the free end then extends steadily and the board is tilted up: its free end rises continuously and the tilt angle grows smoothly from nearly horizontal at the start of the clip to steeply inclined by the end. This lifting never stops, never pauses, never reverses and never jumps — the board is visibly at a larger angle in every later frame than in every earlier one. The hinged end stays fixed on the bench the whole time. While the board is still shallow both blocks stay exactly where they are on it, and each block starts to slide down the board on its own once the board has become steep enough; keep filming until both blocks have clearly broken away and are sliding. Nothing is added to or removed from either block and no hand ever enters the frame. Both blocks and the full board stay inside the frame. The camera does not move, pan, or zoom. Plain background, no other objects.
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

Metric definition references: `v4/g2/P42/evaluator/measure_backend.py:503`; `v4/unified_evaluators/contract.py:155`.
