# P11 · Rough-incline round trip

Category: 2. Rolling, Friction, and Rigid-Body Statics; legacy ID: `P10`.

Difficulty: Hard; task rank: 37/40; mean final total score: 0.146558.

## Scene and objective

A block travels up a straight rough incline, slows to rest, then slides back down the same path. The kinetic-friction coefficient is 0.20 and the target incline angle is approximately 30 degrees. Use the visible incline and both trajectories to test the ascending/descending acceleration-magnitude ratio.

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
| M1: Ascending/descending acceleration-ratio error | R_theory=(sin(theta)+0.20*cos(theta))/(sin(theta)-0.20*cos(theta)); e=abs((a_up/a_down)/R_theory-1); P=1/(1+e/0.10). A separately confirmed ascent-only partial event receives P=0.1. | Ratios and e are dimensionless; acceleration is in pixels/second^2; theta is incline angle. |

Formula references: `v4/g7/evaluator/tasks/p10_mechanics.py:913`; `v4/unified_evaluators/physics.py:151`.

## Original generation prompt

```text
Locked-off static camera, side view. A rigid block is launched upward along a straight rough ramp whose rendered inclination is close to 30 degrees above horizontal; the kinetic friction coefficient is 0.20. The block slides upward, slows continuously, comes to a complete stop, and then slides back down along exactly the same path on its own. Keep the ramp edge and the full upward/downward motion visible so the rendered incline angle can be measured from pixels. The camera does not move, pan, or zoom. Plain background, only the ramp and block.
```
