# P40 · Droplet volume conservation

Category: 9. Surface Tension and Viscous Flow.

Difficulty: Medium; task rank: 21/40; mean final total score: 0.342852.

## Scene and objective

Two free water droplets of different sizes with complete outlines approach in air, touch, and merge into one droplet. Liquid volume and complete boundaries remain observable. Test volume conservation and circularity changes across coalescence.

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
| M1: Droplet-volume residual | e=abs(r_final^3/(r1^3+r2^3)-1); P=1/(1+e/0.10). | e is dimensionless; radii r are in pixels. |
| M2: Circularity change | circularity=1-radial outline RMS error/fitted radius; e=abs(final circularity-mean initial circularity of both drops); P=1/(1+e/0.10). | Circularity, e, and P are dimensionless. |


## Original generation prompt

```text
Task: P40_free_droplet_coalescence.
A high-speed macro laboratory view freezes two clean spherical water droplets fully detached in air, separated by a very small gap and approaching one another along the same horizontal axis. Their radii are visibly different but both outlines are complete, sharply focused, and undeformed before contact. The release nozzles are outside the central measurement region, the background is dark and plain, and ample empty space surrounds the expected merged droplet.

Video action: use a locked-off short-exposure teaching-laboratory camera. Begin with the two complete, unequal, same-liquid droplets exactly as shown, with a small visible air gap. Then let them translate gently toward one another, touch, form one continuous liquid volume, and settle into one approximately spherical merged droplet while conserving volume. Keep every droplet fully inside frame before, during, and after coalescence. Preserve soft liquid refraction and natural diffuse highlights.

Hard negative constraints: no contact or liquid bridge in the first frame, no nozzle, support, surface, pool impact, splash crown, satellite droplets, extra droplets, material loss, hard glass-ball shell, hollow bubble, plastic bead, marble, frosted or milky material, motion blur, camera movement, labels, radius guides, arrows, trajectories, watermark, logo, or toy/illustration styling.
```
