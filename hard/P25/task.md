# P25 · Ice melting: water level

Category: 6. Phase Transitions and Melting.

Difficulty: Hard; task rank: 38/40; mean final total score: 0.142956.

Task ID: `P25`.

A piece of freshwater ice floating in a transparent straight-walled glass gradually melts completely. It remains at the surface, leaving no solid ice at the end. No overflow, evaporation, or liquid addition or removal occurs. Initial and final levels are clearly visible, with distinguishable light-cyan water and opaque white ice. Ideally, melting floating ice leaves the water level unchanged.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: (water-level height after melting - water-level height before melting)/vessel height; dimensionless.
- M2: |final vessel width - initial vessel width|/initial vessel width, checking vessel-boundary consistency; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

```text
Locked-off static camera, front view, matching the input first frame.

A piece of pure freshwater ice initially floats freely in fresh water inside a transparent straight-walled glass. The ice remains floating at the water surface while it gradually melts completely, leaving no solid ice by the end of the shot. It does not sink as an intact solid block.

There is no overflow, visible evaporation, or addition or removal of liquid or material. The initial and final water levels remain clearly visible.

The camera does not move, pan, or zoom. Plain background, no unrelated objects.

Throughout the clip the water keeps the same distinctly light cyan-blue tint it has in the first frame and the ice stays opaque white, so the waterline and the ice are never confusable. The wall behind the glass stays one flat tone, with no dark horizontal band appearing behind or across the beaker at any time. Nothing else in the scene takes on that blue.
```

## Evaluation entrypoint

Run from this task directory; input and output paths may reside outside the task package:

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```
