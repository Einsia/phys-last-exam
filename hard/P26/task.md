# P26 · Ice with a stone: melting

Category: 6. Phase Transitions and Melting.

Difficulty: Hard; task rank: 40/40; mean final total score: 0.034453.

## Scene and objective

Freshwater ice containing a dense stone floats in fresh water inside a transparent straight-walled vessel. It gradually melts, releasing the stone to sink. With no water or other material added or removed, compare initial and final water levels; the target behavior is a falling water level.

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
| M1: Water-level change after melting | e=(y_initial-y_final)/H_initial; P=1 when e<0, otherwise P=0. Melting must first be observed and the level change must exceed measurement resolution. | e is dimensionless; y and H are in pixels. |


## Original generation prompt

```text
Locked-off static camera, front view, matching the input first frame.

A dense stone is completely frozen inside a piece of freshwater ice that initially floats freely in fresh water inside a transparent straight-walled container.

The ice remains at the water surface while it gradually melts completely and releases the stone. No solid ice remains by the end of the shot. The ice does not sink as an intact solid block; only the released stone sinks to the bottom of the container.

No water or other material is added or removed. The waterline and the stone remain clearly visible throughout the shot.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```
