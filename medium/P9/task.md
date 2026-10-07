# P9 · Solid sphere vs. hoop

Category: 2. Rolling, Friction, and Rigid-Body Statics; legacy ID: `P7`.

Difficulty: Medium; task rank: 25/40; mean final total score: 0.284114.

## Scene and objective

A uniform solid sphere and thin hoop with equal outer radii are released simultaneously from rest at the same height on the same incline and roll without slipping. Compare the times required to cover the same distance; the theoretical hoop/sphere travel-time ratio is sqrt(10/7).

## Inputs and evaluation

First frame: [first_frame.png](first_frame.png). Generation prompt: [prompt.txt](prompt.txt). Videos are external evaluation inputs and must correspond to the supplied first frame and prompt.

Run from this task directory using a Python environment with the project runtime dependencies installed:

```sh
python evaluator/evaluate.py \
  --video /absolute/path/video.mp4 \
  --image first_frame.png --prompt prompt.txt \
  --output /absolute/path/result.json
```

## Current metrics

In the table below, P is the pure physics score for each metric, ranging from 0 to 1. Raw quantities and physics scores are retained separately. With insufficient observations, raw quantities are null; the applicable unobserved-phenomenon policy may assign a physics score of 0. Full outputs also include consistency assessment.

| Metric | Definition and formula | Units |
| --- | --- | --- |
| M1: Equal-distance travel-time-ratio error | Take the median hoop/sphere time ratio r at multiple virtual equal-distance positions; e=abs(r/sqrt(10/7)-1); P=1/(1+e/0.10). | r, e, and P are dimensionless; time is in seconds. |

Formula references: `v4/g7/evaluator/tasks/p7_rotational.py:691`; `v4/unified_evaluators/physics.py:150`.

## Original generation prompt

```text
Locked-off static camera, side view. A uniform solid sphere and a thin circular ring have exactly the same outer radius. They are placed side by side at the same height on the same straight incline and released simultaneously from rest. Both objects roll down the incline without slipping. Their complete motions and the common finish position remain visible. The camera does not move, pan, or zoom. Plain background, no other objects.
```
