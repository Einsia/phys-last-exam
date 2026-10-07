# P36 · Sand vs. water discharge

Category: 8. Granular Media and Discharge Flow; legacy ID: `P41`.

Difficulty: Hard; task rank: 35/40; mean final total score: 0.156825.

## Scene and objective

A fixed camera observes two identical transparent conical funnels with matching outlet sizes and initial fill heights. Light-blue water is on the left and dry sand on the right. Both outlets open simultaneously and remain open until nearly empty. Observe material levels and streams to compare how water and sand flow rates depend on remaining height.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. Two identical transparent funnels sit side by side with identical outlet sizes and the same initial fill height; the left contains pale blue water and the right contains dry sand. Both outlets open simultaneously and remain fully open while each funnel discharges its own material until nearly empty. Both fill levels and both outlet streams remain clearly visible. The camera does not move, pan, or zoom. Plain background.
```

## M1 physical metric

M1 independently fits `Q proportional to h^beta` for water and sand, then compares the water exponent with 0.5 and the sand exponent with 0.

`E = |β_water - 0.5| + |β_sand|`

`M1 = 1 / (1 + E / a)`, with default `a=1.0`.

Exponents and errors are dimensionless. Heights are measured in pixels. Relative volume is recovered from reviewed conical-interior and free-surface geometry, and exponents are freely fitted through the integrated volume/head relationship. Relative volume is in px^3 and flow rate in px^3/s, not physically calibrated SI units. Finite exponent errors receive smooth penalties; fitting uncertainty is reported separately.

Implementation: `p36` in [measured_discharge.py](evaluator/measured_discharge.py) and [discharge_integral.py](evaluator/discharge_integral.py), version `p36_fixed_cone_integrated_exponent_v5`.

## Evaluation entrypoint

Run from this task directory:

```bash
python evaluator/evaluate.py \
  --video /absolute/path/to/input.mp4 \
  --image first_frame.png \
  --video_prompt_file prompt.txt \
  --output /absolute/path/to/result.json
```

Use the configured evaluation environment. The evaluator automatically loads `first_frame_annotations.json` bundled with this task. It verifies the input-image hash, scales the coordinates to the video resolution, and checks correspondence with the decoded first frame before tracking. No per-video annotation is needed for the fixed task image when this check passes. Use `--annotation /absolute/path/to/video_annotations.json` only to override initialization for a different image or layout; custom video annotations retain their video/image hash checks.
