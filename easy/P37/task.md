# P37 · Capillary rise vs. diameter

Category: 9. Surface Tension and Viscous Flow.

Difficulty: Easy; task rank: 12/40; mean final total score: 0.511797.

## Scene and objective

Two dry glass capillary tubes of different inner diameters initially hang above the same reservoir, then are simultaneously dipped shallowly into the liquid. Observe the common external waterline and both internal menisci. The narrower tube should have a greater capillary rise. Scoring assesses the direction of the height difference rather than an exact inverse-diameter magnitude.

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
| M1: Direction of capillary-rise difference | h=y_common_surface-y_tube_surface; d=median final paired difference h_narrow-h_wide. P=1 when d exceeds measurement resolution, otherwise P=0. Both tubes and the common surface must be jointly readable. | h, d, and resolution are in pixels; P is dimensionless. |


## Original generation prompt

```text
Task: P37_capillary_rise_two_radii.
Start exactly from the supplied dry first frame: two clean, vertical, parallel, open-ended glass capillary tubes are held side by side above one continuous transparent water reservoir. Their lower rims are completely above the sharp horizontal water surface, with a clear air gap below both rims. Both tube interiors are completely dry and empty at frame 0. One tube has a narrow inner radius and the other has an inner radius about twice as large.

Video action: use a locked-off, fixed, straight-on laboratory teaching-video camera with no pan, zoom, shake, or reframing. Hold the dry suspended state briefly, then lower both tubes together at the same speed into the same reservoir until their lower ends are immersed to the same shallow depth. After insertion, show the capillary phenomenon: water wets the identical glass walls and rises inside both open tubes from the common reservoir level, forming two attached concave menisci. The rise is modest and physically plausible, not a tall liquid column. The narrower tube rises higher than the wider tube because capillary rise height is inversely proportional to inner radius (approximately h_narrow*r_narrow = h_wide*r_wide). Keep both liquid columns clearly below the available tube length, with the narrow-tube column visibly higher but only moderately so. Keep the shared external water surface, tube bores, lower openings, menisci, clamp, and tank visible throughout, with realistic transparent-glass refraction and liquid reflections.

Hard negative constraints: no liquid or meniscus inside either tube at frame 0; no tube touching or crossing the water surface at frame 0; no exaggerated or near-top liquid columns; no equal final rise heights; no wider tube rising higher than the narrow tube; no closed or rounded tube ends; no separate reservoirs; no different immersion depths; no person or hand; no labels, numbers, ruler, equations, arrows, trajectories, watermark, logo, cartoon styling, camera motion, or cropped key objects.
```
