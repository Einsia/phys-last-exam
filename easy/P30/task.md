# P30 · Coil-induced light emission

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction.

Difficulty: Easy; task rank: 13/40; mean final total score: 0.503125.

## Scene and objective

A fixed camera observes a bar magnet, fixed coil, and connected bidirectional indicator lamp. The magnet starts from rest, enters and passes through the coil, exits the far end, and stops again. Observe magnet motion and the light response while keeping the magnet, coil, and lamp clearly visible.

## Inputs

- First frame: [first_frame.png](first_frame.png).
- Video generation prompt: [prompt.txt](prompt.txt).
- Evaluation video: external input specified with `--video`.

## Original prompt

```text
Locked-off static camera, front view. Continue from the supplied first frame. A bar magnet first remains completely stationary near a fixed coil connected to a small bidirectional indicator lamp. The magnet then moves into the coil, passes completely through it, moves out the other side, and finally becomes stationary again. The magnet, coil and lamp remain clearly visible throughout the entire sequence. The camera does not move, pan, or zoom. Plain background, no unrelated objects.
```

## M1 physical metric

M1 consists of two directly observable events: magnet entry into the coil and confirmed lamp illumination during entry.

`M1 = 0.5 * I(entry observed) + 0.5 * I(lamp confirmed lit during entry)`

`I` is 1 when the event holds and 0 otherwise; M1 is dimensionless. A lamp already lit before magnet entry may count. The default event-matching tolerance is 0.5 seconds. If entry is confirmed but lamp state is indeterminate, M1=0.5 is an evidence lower bound and the uncertainty interval `[0.5, 1]` is reported; an unknown lamp state is not treated as unlit.

Complete passage and stopping again belong to the video task description. M1 does not measure magnetic flux, current, voltage, or the light temporal centroid.

Implementation: `evaluate` in [entry_light.py](evaluator/entry_light.py), version `p30_entry_light_v5`.

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
