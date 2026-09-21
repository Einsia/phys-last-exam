# P4 deterministic repeated-bounce evaluator

This evaluator does **not** use an LLM or VLM. Its only optional learned component
is a locally deployed CoTracker point tracker. CoTracker is independently checked
against two classical CV tracks: Lab seed-colour segmentation and temporal median
background subtraction.

Formal batch:

```bash
python \
  run_all.py
```

SOP single-video interface:

```bash
python \
  evaluate.py --video VIDEO.mp4 --task_id P4 --output result.json \
  --debug-dir debug/sample_id
```

For CPU-only reproducibility, add `--no-cotracker`. The formal reported run uses all
three trackers and fixes `CUDA_VISIBLE_DEVICES` to physical GPU 4.

## Measurement definition

An eligible clip needs one well-measured chain of at least four complete rebound
arcs, i.e. five successive impact events and four intervening apexes. For arc `n`:

- `h_n`: apex height above the linearly interpolated level of its two adjacent
  impact centres;
- `Delta t_n`: number of frames between those impacts;
- `e_h,n = sqrt(h_(n+1)/h_n)`;
- `e_t,n = Delta t_(n+1)/Delta t_n`.

Primary M1 is the mean `abs(e_h,n - e_t,n)` across successive pairs. M2 records
height/time monotonic violations (including an explicit near-equality penalty, so
equal-height rebounds cannot pass as decreasing) and the coefficient-of-restitution consistency
(CV of `(e_h,n + e_t,n)/2`). Arc parabolic shape, apex timing, horizontal drift,
impact alignment, camera drift and tracker disagreement are independent dimensions.

An observable content failure is never converted into an extraction failure. A
reliably tracked clip with fewer than four arcs has `extract_success=true`,
`structural_ok=false`, and end-to-end score zero. Every run emits a JSON even if
decoding, seeding, or tracking fails.

## Debug artifacts

Each sample receives:

- `overlay.mp4`: all three tracks, fused trail, impacts and apexes;
- `trajectory_plot.png`: y(t), event locations, four heights and intervals;
- `keyframes.jpg`: release, impacts and apexes;
- `fused_track.csv`: stabilized coordinates and confidence flags;
- one strict JSON result.

The batch additionally writes `results.csv` and `summary.json`.
