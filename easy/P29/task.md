# P29 · Eddy-current braking

Category: 7. Electrostatics, Magnetism, and Electromagnetic Induction.

Difficulty: Easy; task rank: 10/40; mean final total score: 0.568750.

## Scene and objective

Two objects with matching shape, mass, and contact surfaces rest side by side on a copper incline. One contains a strong magnet; the other contains nonmagnetic ballast. Both are released simultaneously from rest. The target behavior is delayed, slower magnetic-object motion due to eddy-current braking.

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
| M1: Trajectory event-time ratio | r_t=(f_magnetic-f_start+1)/(f_control-f_start+1); P=1 when r_t>1, otherwise P=0. f is an event frame detected by the backend. | r_t is dimensionless; f is a frame index. |
| M2: Motion-speed ratio | r_v=v_magnetic/v_control; P=1 when r_v<1, otherwise P=0. | r_v is dimensionless; v is in pixels/second. |


## Original generation prompt

```text
Locked-off static camera, side view. Two externally identical blocks with the same total mass and identical flat contact surfaces are placed side by side at the same height on the same inclined copper plate. One block contains a strong neodymium magnet and the other contains an equal-mass non-magnetic insert. They are released simultaneously from rest and slide freely down the copper plate. Both complete motions remain visible. The camera does not move, pan, or zoom. Plain background.
```
