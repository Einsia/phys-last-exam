# P17 · Light refraction

Category: 4. Optics and Projective Geometry.

Difficulty: Medium; task rank: 15/40; mean final total score: 0.445497.

Task ID: `P17`.

A fixed laser source illuminates a stationary air-water interface in a transparent tank. The laser turns on and reaches steady illumination. The incident ray, interface, and refracted underwater ray directions and intersection points are clearly visible. Refraction should follow Snell's law, with straight ray segments within each uniform medium.

## Inputs

The only reference first frame is [first_frame.png](first_frame.png) in the task root. Videos are supplied externally for evaluation and are not included in the task package.

## Current physical metrics

- M1: sin(incidence angle)/sin(refraction angle)-n_water, with angles measured relative to the interface normal; dimensionless.
- M2: Distance between the incident and refracted ray intersections with the waterline, divided by the first-frame calibrated vessel width; dimensionless.

These are raw measured quantities; normalized physics scores range from 0 to 1.

## Original generation prompt

### Current generation prompt

The v4 prompt corresponding to the 30-degree first frame is used. The following is the only active generation prompt for this task.

```text
A single continuous real-time shot from a locked-off static side camera, matching the input first frame.

Preserve the existing transparent tank, clear still fresh water, laser source, support apparatus, framing, and background. The laser source and its mount remain completely stationary.

The laser is initially switched off. After a brief still moment, it switches on and emits a thin, clearly visible red beam along its existing optical axis. The beam travels through the air and enters the water obliquely at one fixed point on the flat air-water interface.

Exactly one incident ray in the air and one refracted ray in the water are clearly visible. They meet cleanly and continuously at the same point of incidence and remain straight within their respective media. The flat air-water interface, point of incidence, and a thin stationary interface normal perpendicular to the water surface at that point are all clearly visible.

The light remains thin and sharp without excessive bloom or overexposure. The water surface remains still and horizontal. Preserve the clean photographic appearance and existing uncluttered background of the input frame.

No camera movement, panning, zooming, cuts, slow motion, or time jumps. No people, hands, angle arcs, arrows, measurements, equations, numbers, labels, or annotations other than the interface normal.

End after the visible optical behavior has remained stable for a short moment.
```

## Evaluation entrypoint

From this task directory, explicitly supply the external video, root first frame, and retained full prompt file:

```sh
python evaluator/evaluate.py \
  --video /absolute/path/to/video.mp4 \
  --image first_frame.png \
  --prompt prompt.txt \
  --output /absolute/path/to/result.json
```

The measurement backend uses the bundled scene calibration; the input video must match the selected 30-degree first-frame calibration.


## Current scene calibration

To use the existing calibration for this task's 30-degree first frame, name the external video `P17_gpt_01_30deg_seedN.mp4` (N is any integer). Other filenames trigger the original generic backend calibration and must not be treated as the same scene calibration.
