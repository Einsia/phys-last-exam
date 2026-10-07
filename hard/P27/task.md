# P27 · Freshwater ice in saltwater

Category: 6. Phase Transitions and Melting; legacy ID: `P21c`.

Difficulty: Hard; task rank: 36/40; mean final total score: 0.148438.

## Scene and objective

Freshwater ice floats on salt water in a transparent straight-walled vessel, gradually melting and mixing with the salt water. No liquid is added, removed, or spilled. Compare initial and final levels and check for residual solid ice in the final frame; the target behavior is a rising water level and complete melting.

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
| M1: Water-level rise after melting | e=(y_initial-y_final)/H_initial; P=1 when y_initial-y_final exceeds measurement uncertainty, otherwise P=0. | e is dimensionless; y and H are in pixels. |
| M2: Residual-solid evidence | r=final solid evidence/initial solid evidence; P=clip(1-r,0,1). P=0 if residual ice is detected in the final frame. | r and P are dimensionless. |

Formula references: `v4/g5/P21c/evaluator/measure_backend.py:292`; `v4/g5/P21c/evaluator/measure_backend.py:235`; `v4/g5/P21c/evaluator/measure_backend.py:293`; `v4/g5/P21c/evaluator/measure_backend.py:299`.

## Original generation prompt

```text
Locked-off static camera, front view, matching the input first frame.

A piece of freshwater ice initially floats freely in denser salt water inside a transparent straight-walled container.

The freshwater ice remains floating at the salt-water surface while it gradually melts completely, leaving no solid ice by the end of the shot. It does not sink as an intact solid block. The meltwater mixes naturally with the surrounding salt water.

No liquid is added, removed, spilled, or visibly evaporated. The liquid surface remains clearly visible throughout the shot.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```
