# P20 · Total-reflection critical angle

Category: 4. Optics and Projective Geometry.

Difficulty: Hard; task rank: 31/40; mean final total score: 0.192774.

## Scene and objective

Four distinguishable colored light beams meet a fixed water surface at separate interface points. First complete the paths through the interface, then change incident directions while holding intersection points fixed. The target behavior follows refraction and, when applicable, total internal reflection. Scoring compares refractive indices inferred independently from the beams.

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
| M1: Refractive-index consistency | Estimate n from each measurable beam using Snell geometry; e=std(n)/mean(n); P=1/(1+e/0.05). Ordinary total internal reflection supplies only a lower bound, not a critical-angle equality. | n, e, and P are dimensionless. |


## Original generation prompt

```text
Locked-off exact front orthographic view. Continue the incident-only frame in two stages while keeping the horizontal water surface and all four surface hit points fixed. In stage one, complete each coloured beam across the interface with its physically correct Snell-law outcome. In stage two, smoothly swing each beam pair about its own fixed hit point, continuously preserving the refractive relationship between the air and water arms. Keep the tank, water level, dashed normals, colours, camera, scale and background fixed; do not slide a corner along the surface or add text or extra apparatus.
```
