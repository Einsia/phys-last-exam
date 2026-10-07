# P7 · Hanging-chain equilibrium

Category: 2. Rolling, Friction, and Rigid-Body Statics.

Difficulty: Easy; task rank: 5/40; mean final total score: 0.708567.

## Scene and objective

A continuous chain has both ends fixed at equal-height supports and is released from the nonequilibrium first-frame shape. It oscillates under gravity and gradually settles. Chain continuity, total length, and supports remain unchanged. Test whether the final outline follows a catenary and the supports remain at equal heights.

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
| M1: Catenary-fit error | e=outline-fit RMSE/measured sag; P=1/(1+abs(e)/0.10). | e is dimensionless; geometric lengths are in pixels. |
| M2: Equal-support-height error | e=support-height error/horizontal span; P=1/(1+abs(e)/0.02). | e is dimensionless; geometric lengths are in pixels. |


## Original generation prompt

```text
A single continuous real-time shot from a locked-off, exact front near-orthographic camera, matching the input first frame.

The input first frame captures the instant immediately after a temporary constraint has been removed. No hand, tool, magnet, or temporary support remains visible. The first and last links stay fixed to the two equal-height anchors, while the rest of the chain immediately begins moving freely under gravity from its existing non-equilibrium shape.

Preserve the existing support frame, the two equal-height fixed anchors, the complete chain, its material and link structure, the camera view, framing, and background. The first and last links remain fixed to their existing anchors throughout the shot.

The chain swings and oscillates naturally as the motion gradually damps, then settles into one smooth, stable, deep hanging shape. The chain remains one continuous flexible chain of rigid interlocked links, with unchanged total length. No link detaches, stretches, fuses, duplicates, disappears, or changes material.

Keep the complete chain, both anchors, and the full motion visible inside the frame. The support frame, anchors, background, and camera remain stationary. No camera movement, cuts, slow motion, time jumps, hands, people, added supports, external forces, text, formulas, plotted curves, arrows, or annotations.

End after the chain has settled and remained essentially motionless for a short moment.
```
