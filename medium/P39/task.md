# P39 · Bubble-film curvature

Category: 9. Surface Tension and Viscous Flow; legacy ID: `P47`.

Difficulty: Medium; task rank: 24/40; mean final total score: 0.320569.

## Scene and objective

In a fixed side-view close-up, two approximately spherical soap bubbles of clearly different sizes approach, touch, and join through a visible internal partition. They stay connected without bursting. Both outer outlines and the complete partition remain clearly visible to compare curvature and pressure difference.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, close-up side view. Continue from the supplied first frame. Two roughly spherical soap bubbles of clearly different sizes move toward each other and stay connected by one clearly visible internal partition. The outer boundaries of both bubbles and the complete partition remain sharply visible and inside the frame. The bubbles do not pop or detach. The camera does not move, pan, or zoom. Plain dark background.
```

## M1 physical metric

M1 independently fits the small outer-arc radius `r_small`, large outer-arc radius `r_large`, and signed partition radius `r_partition`, all in px.

`E_frame = |r_partition × (1/r_small - 1/r_large) - 1|`

`E = median of E_frame weighted by actual frame-time intervals`

`M1 = 1 / (1 + E / a)`, with default `a=1.0`.

Error and score are dimensionless. Use all continuous intervals of at least 3 frames and 0.08 seconds with a uniquely identifiable partition. Incorrect partition orientation or nonideal radii are penalized through the residual. If the partition between unequal bubbles is demonstrated to be approximately straight and curvature uncertainty excludes the theoretical prediction, separately record an unbounded-error limit and assign 0; ordinary unidentifiable curvature does not imply that limit.

Implementation: `inspect_frame` and `summarize` in [measured_bubbles.py](evaluator/measured_bubbles.py).

## Evaluation entrypoint

Run from this task directory:

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use a Python environment with evaluation dependencies and the consistency model configured.
