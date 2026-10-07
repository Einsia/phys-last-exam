# P28 · Crushed vs. intact ice

Category: 6. Phase Transitions and Melting.

Difficulty: Hard; task rank: 32/40; mean final total score: 0.192187.

## Scene and objective

Identical transparent straight-walled beakers sit side by side: intact ice on the left, the same mass of crushed ice on the right. Both melt simultaneously and form visible liquid layers. After confirming visible melting on both sides, assess whether the crushed-ice vessel sustains a lead in liquid level normalized by vessel height. Two-dimensional ice area is not treated as mass.

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
| M1: Sustained crushed-ice liquid-level lead | Confirm sustained melting on both sides; compare same-frame levels relative to each vessel base and normalized by height. P=1 if the crushed-ice vessel lead exceeds the sum of localization errors for at least 0.25 seconds. P=0 if no lead is observed within at least 0.50 seconds of continuously readable shared evidence. | Normalized level difference is dimensionless; duration is in seconds; localization error is normalized from pixels. |


## Original generation prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical transparent straight-walled beakers sit side by side; the left holds one compact ice block and the right holds crushed ice of the same total mass. Melting begins at the same time and continues until all visible ice has turned into water, and a liquid surface appears and rises in each beaker. Both beakers remain fully visible. The camera does not move, pan, or zoom. Plain dark background.
```
