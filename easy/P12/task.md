# P12 · Pendulum period vs. mass

Category: 3. Pendulum Motion and Oscillations; legacy ID: `P8b`.

Difficulty: Easy; task rank: 11/40; mean final total score: 0.530868.

## Scene and objective

Two equal-length pendulums hang from the same support. The left bob is small and light; the right bob is large and heavy. Both are released simultaneously from the same angle and oscillate for several cycles, testing whether period is independent of bob mass at equal length.

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
| M1: Period ratio of different-mass pendulums | e=T_heavy/T_light-1; P=1/(1+abs(e)/0.10). | e is dimensionless; T uses consistent time units. |
| M2: Relative first-frame length difference | e=abs(L_left-L_right)/mean(L_left,L_right); P=1/(1+e/0.10). Suspension points are fitted from video trajectories. | e is dimensionless; L is in pixels. |

Formula references: `v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:51`; `v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:148`; `v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:56`; `v4/g6/P8b/evaluator/utils/physeval/tasks/p8b.py:118`.

## Original generation prompt

```text
Locked-off static camera, side view. Two pendulums of equal rod length hang from the same horizontal support; the left bob is light and small, the right bob is heavy and large. They are released together from the same angle and swing back and forth for several cycles. The camera does not move, pan, or zoom. Plain flat background, only the two pendulums, nothing else enters the frame.
```
