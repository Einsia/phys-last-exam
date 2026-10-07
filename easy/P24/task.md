# P24 · Freezing-induced expansion

Category: 6. Phase Transitions and Melting.

Difficulty: Easy; task rank: 16/40; mean final total score: 0.438451.

## Scene and objective

Liquid water freezes completely in a transparent straight-walled vessel without material being added, removed, or spilled. Compare initial water and final ice levels to test volume expansion at unchanged vessel cross-section.

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
| M1: Freezing-height-ratio error | e=H_ice/H_water-1000/917; P=1/(1+abs(e)). | e is dimensionless; H is in pixels. |
| M2: Cross-sectional width consistency | e=abs(W_final-W_initial)/max(W_initial,1 pixel); P=1/(1+abs(e)). | e is dimensionless; W is in pixels. |


## Original generation prompt

```text
Locked-off static camera, front view. Pure liquid water fills part of a transparent straight-walled container and then freezes completely into solid ice. No water is added, removed, spilled, or visibly evaporated during the process. The initial liquid-water level and final top surface of the ice are both clearly visible. The camera does not move, pan, or zoom. Plain background.
```
