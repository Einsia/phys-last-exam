# P13 · Large-angle pendulum

Category: 3. Pendulum Motion and Oscillations.

Difficulty: Easy; task rank: 8/40; mean final total score: 0.574597.

## Scene and objective

Two equal-length pendulums are released simultaneously from rest at the different amplitudes shown in the first frame. They complete multiple readable cycles with distinguishable bobs and suspension points. Test whether the larger-amplitude pendulum has a longer period and compare against the finite-amplitude theoretical period ratio used by the implementation.

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
| M1: Finite-amplitude period-ratio error | r=T_large_amplitude/T_small_amplitude; e=abs(r-r_theory), with r_theory from the finite-amplitude model; P=1/(1+e/0.05). | r, e, and P are dimensionless; periods are in seconds. |
| M2: Direction of period difference | P=clip((r-1)/(r_theory-1),0,1). | Ratios and P are dimensionless. |


## Original generation prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. The red pendulum on the left and the blue pendulum on the right are both released from rest at the same instant. After release, both bobs immediately leave their first-frame poses and each swings back and forth on its own string for several complete periods while remaining fully visible. The red bob is not frozen and does not hang still. String lengths stay the same, colours stay the same, and the two pendulums do not collide. The support frame stays fixed. The camera does not move, pan, or zoom. Plain background, only the two pendulums.
```
