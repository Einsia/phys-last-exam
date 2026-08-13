# physical-bench — Judge-Free Physics Evaluation

Measures whether a generated video *obeys* physics, instead of asking a model whether it
*looks* physical. No VLM judge, no reference video, no simulator ground truth.

The premise (`proposal.md` §3.1): a generated clip has no calibration. We do not know the
metres-per-pixel `s` or the seconds-per-frame `τ`, so anything needing absolute velocity,
force, or energy is unavailable. But a constraint whose truth value survives arbitrary `s`
and `τ` *can* be checked from pixels alone. So we do not estimate the calibration — we pick
observables that cancel it.

This repo implements the chain end to end for **P2** (`prompt.txt` row 3, ⭐): *a ball
launched at a stated angle*, whose textbook invariants are `H/R = tan θ / 4` and
`N_up / N_down = 1`.

## The chain

```
first frame  ->  VDM  ->  tracking  ->  calibration-invariant residuals  ->  PMR/PE/PPR/HVR
 (drawn)        (subproc)  (subproc)        (7 measurables)
                           SAM2 + CoTracker
```

Each stage is swappable. Both the VDM and the trackers run as subprocesses in their own
envs, so no two dependency sets have to agree — `envs/physbench` stays numpy/opencv-only
and torch lives exclusively in `envs/track`.

```bash
scripts/create_env.sh                                                   # measurement env
scripts/create_track_env.sh                                             # tracker env + weights
envs/physbench/bin/python scripts/make_first_frames.py                  # frames + specs
envs/physbench/bin/python scripts/generate.py --model minimax-h3        # ~370 s/clip
envs/physbench/bin/python scripts/evaluate.py --videos data/videos/minimax-h3 --viz
envs/physbench/bin/python scripts/validate_metrics.py                   # calibrate the chain
```

## What is measured, and why each survives unknown calibration

| | invariant | theory | cancels | role |
|---|---|---|---|---|
| **M1** | `H / R` | `tan θ / 4` | `s` (px/px), `τ` unused | primary |
| **M2** | `N_up / N_down` | `1` | `τ` (frame/frame), `s` unused | primary |
| M3 | `(x_apex−x_launch)/(x_land−x_apex)` | `1` | `s` | diagnostic |
| M4 | normalised RMS of (x linear, y quadratic in n) | `0` | both | primary |
| **M5** | `v_x(descent) / v_x(ascent)` | `1` | both (px/frame ÷ px/frame) | primary |
| M6 | `a_y(descent) / a_y(ascent)` | `1` | both | diagnostic |
| M7 | `a_y · N² / H` | `8` | both | diagnostic |

M7 is worth spelling out: with `H = v₀²/2g` and `N = 2v₀/g`, the combination is exactly 8
for any projectile — independent of `g`, `v₀`, and both calibrations. It is a *diagnostic*
rather than a gate because calibration showed it is empirically insensitive, for a reason
that will not go away. On a triangular path the least-squares quadratic returns
`a_y = 15v₀/4N` → `a_y N²/H_true = 7.5`, but the apex fit simultaneously rounds off the
sharp peak and shrinks measured `H` by ~6%. Both biases come from the same
non-parabolicity, so they cancel rather than compound, landing the observed value back at
7.97. Reported because it is informative when it *does* deviate; not allowed to gate PPR on
a weak signal.

**M1 is computed twice.** Against the prompt's angle it tests instruction-following and
shape together; against the angle measured from the launch direction it tests shape alone.
The gap between them *is* the instruction-following error. The measured angle must come
from a **local** fit at launch — against a globally-fitted angle, `H/R = tan θ/4` is an
algebraic identity and the test returns zero by construction.

## Residuals and the four metrics

Ratio invariants use the symmetric log residual `e = |log(measured/theory)|` — scale-free,
treats 2× and 0.5× as equally wrong, no arbitrary choice of denominator. `e = 0.10` ≈ "10%
off".

| metric | meaning |
|---|---|
| **PMR** | fraction of clips that are *measurable at all* |
| **PE** | mean scene residual `e_i` over samples with computable residuals |
| **PPR** | fraction of measurable clips passing every primary invariant |
| **HVR** | fraction of measurable clips breaking a hard bound |

PE averages only over log-ratio residuals. Mixing in M4's normalised RMS would produce a
number with no unit: an RMS of 0.1 and a log-ratio of 0.1 mean very different things.

## The measurability gate, and why it is split

A model must not be able to buy a good physics score by producing clips that cannot be
measured. So the gate admits **observability failures only** — low detection coverage,
camera drift, tracker disagreement, ball leaving frame. Those lower PMR.

A clip where the ball is tracked perfectly but never comes back down is *not*
unmeasurable — it is wrong. That is **measurable, with a hard violation**. Keeping these
apart is what makes PMR and PPR independently meaningful, and it is why `proposal.md` §3.3
insists on reporting PMR rather than silently dropping tracking failures.

`scripts/edge_cases.py` asserts the split on nine boundary clips. Three were misclassified
when first tested, and the fixes are the interesting part:

| clip | verdict | why |
|---|---|---|
| flight of 8 frames | not measurable | too short to fit is *our* sampling limit, not the video's error |
| flight of 12/14/20 | measurable | gate boundary holds, all 4 primaries present |
| truncated mid-descent | not measurable | ball still moving at last sighting → clip ran out |
| ball stops in mid-air | measurable + violation | at rest without landing is physics, not truncation |
| apex above canvas | not measurable | ball not continuously in view |
| no ball at all | not measurable | trackers disagree — nothing to agree about |
| **never launches** | **measurable + `ball_never_moved`** | **tracker-dependent; see below** |
| camera pan 0.55 px/frame | not measurable | `camera_drift(0.030)` |

The apex-above-canvas case is why the exit test is not border proximity. At 80° the ball
crosses the border band in one step — y=60 in one frame, y=−30 in the next — so it is never
*observed* near an edge, yet its apex is off-canvas and `H` is unmeasurable. The classic
pair catches this as `detection_gap(6)`; SAM2 simply stops returning the object, which
surfaces as `low_coverage(0.25)`. Different signals, same bucket, and the test is
deliberately agnostic about *why* the ball vanished: from pixels, "left the frame" and
"stopped existing" are the same observation, and neither is measurable.

**`never_launches` is the one verdict the tracker choice changes, and it exposed a flaw in
the old one.** A ball at rest for the whole clip used to be *not measurable* via
`tracker_disagreement(inf)` — but only because bgsub is blind to a stationary ball by
construction: it becomes its own temporal-median background, so coverage collapses to zero.
The right answer from the wrong mechanism.

Both learned backends track the static ball at full coverage and agree, so the clip now
lands where this repo's own rule puts it: tracked perfectly, in frame throughout, and still
not doing what was asked — **measurable, with a hard violation**. That is also the correct
incentive. Under the old behaviour a model emitting static clips lowered PMR instead of PPR,
which is precisely the trade `proposal.md` §3.3 warns against.

The violation now tests displacement (`centre moved < 1 radius over the clip`) rather than
matching `extract_flight`'s reason string, because a static ball does not reliably report
`no_motion_observed` — sub-pixel tracker noise keeps peak speed above the exact-zero test
and the run terminates as `no_ascent_observed` instead.

Distinguishing a truncated clip from a ball that halts in mid-air needs one extra bit —
whether the ball was still moving at its last sighting. Both present as the same extraction
failure, and they must not score the same way.

## Tracking: learned backends, and what switching to them proved

Two independent trackers run on every clip and their disagreement is a QC signal rather
than a second opinion from the same source. The default pair is learned, which is what
`proposal.md` §3.3 names directly:

| backend | seed | position from | presence from |
|---|---|---|---|
| **sam2** | one positive click at the ball centre, frame 0 | mask centroid, probability-weighted | the model's object-score logit |
| **sam3** | text `ball`; frame-0 seed selects the nearest returned instance | binary-mask centroid, equivalent-disc radius | masklet/object presence |
| **cotracker** | ring of 9 points on the disc, frame 0 | median of `point − frame-0 offset` | the model's visibility head |
| color | seeded RGB | Lab-distance blob centroid | escalating distance thresholds |
| bgsub | none | frame-difference blob centroid | MAD-scaled thresholds |

Nothing on the default path is tuned per clip: both learned backends take only the ball
centre and radius the spec already records. The classic pair's five escalating Lab
thresholds and MAD multipliers assume a flat scene with one high-chroma ball, and a VDM
that recolours the ball or drifts the background walks out of that regime with no seed
we can hand it to get back in.

`SAM3 + CoTracker` is available as an opt-in pair with
`--config configs/p2_projectile_sam3.yaml`. It uses Meta's video predictor on a temporary
numbered-JPEG view of the already-decoded frames, uses a `ball` text prompt and chooses
the returned instance nearest the exact frame-0 point from the spec, then converts the
selected masklet to the same `xy/radius/score` interface as SAM2. Install a local SAM3
checkout at `cache/sam3` and
place its approved checkpoint at `cache/sam3/sam3.pt` (or override the two paths in YAML)
before running it. The SAM3 source and checkpoint are deliberately not downloaded by the
benchmark because checkpoint access is gated and the worker runs offline for
reproducibility.

CoTracker's call also carries a background grid with the ball's neighbourhood excluded,
so camera translation comes from point correspondences instead of phase correlation —
the ball can no longer pull the estimate it is supposed to be excluded from. Points the
ball flies over are reported occluded by CoTracker's own visibility head and drop out of
the median, so the exclusion zone follows the ball without us predicting its path.

Two QC signals get materially better. Apparent-radius stability (`radius_cv`) drops from
0.024–0.039 to 0.010–0.014, because a segmentation mask measures the ball's extent
directly where a threshold on colour distance measures the extent of *the threshold*. And
`ball_never_moved` becomes detectable at all — see the edge cases below.

**The switch changed no conclusion, and that is the result.** Re-scoring the same 12
clips with two trackers that share no mechanism with the originals:

| | PMR | PE | PPR | HVR | M1 | M2 | M4 | M5 |
|---|---|---|---|---|---|---|---|---|
| classic | 1.000 | 0.3103 | 0.000 | 0.167 | 0.5686 | 0.0387 | 0.0243 | 0.3237 |
| learned | 1.000 | 0.3105 | 0.000 | 0.167 | 0.5682 | 0.0389 | 0.0244 | 0.3243 |

Every invariant agrees to within 0.002, every pass rate is identical, and both angle
regressions land on the same slope. The findings below are therefore properties of the
videos, not of a hand-tuned colour threshold.

Cost and precision, measured rather than assumed (`scripts/tracker_precision.py`,
`scripts/validate_metrics.py`):

- **Per-frame scatter** (bias removed, since bias cancels in every ratio invariant): on
  clean renders the classic pair is 3–6× tighter (0.043 px vs SAM2's 0.138). Under
  motion blur all four land within 15% of each other. The classic pair's edge comes from
  a soft colour centroid over an antialiased rim — exactly the cue a generated frame
  smears away.
- **Residual floor**: the ordering does not survive. H, R and the per-half velocities are
  fits over ~60 frames, so scatter averages down. Worst primary floor is **0.0026**
  learned against **0.0041** classic — the learned pair is marginally *better*.
- **Runtime**: 15.4 s/clip on one H100 against 9.5 s on CPU for a 124-frame clip. Only
  1.6×, because the classic backends also scan every frame in Python — the learned pair is
  not the expensive option it looks like.

So `backends: [color, bgsub]` in the config is the right choice when there is no GPU or a
run must be reproducible on CPU, not for speed. `validate_metrics.py` reports the floor for
whichever pair ran.

## Calibrating the chain

`scripts/validate_metrics.py` renders clips whose physics is known by construction and
checks that each invariant fires on the failure it targets and stays quiet otherwise.
Without this, a zero residual on a real clip proves nothing.

Numbers below are the default learned pair; every ablation behaves identically on the
classic pair (`--backends color,bgsub`), to within the floor.

| clip | M1 | M2 | M4 | M5 | what it breaks |
|---|---|---|---|---|---|
| `gt` | 0.0008 ok | 0.0010 ok | 0.0011 ok | 0.0000 ok | nothing — this is the noise floor |
| `no_gravity` | **0.619** | 0.001 ok | **0.074** | 0.001 ok | triangular path; `H/R` doubles |
| `time_warp` | 0.004 ok | **0.437** | **0.067** | **0.606** | same arc, ascent stretched |
| `wrong_angle` | **0.550** | 0.001 ok | 0.002 ok | 0.000 ok | perfect physics at 30° when asked 45° |
| `const_speed` | 0.002 ok | 0.001 ok | 0.020 (12× floor) | 0.006 ok | correct path, uniform speed |
| `gt+blur` | 0.001 ok | 0.003 ok | 0.002 ok | 0.001 ok | motion blur + mp4 round-trip |

**Noise floor ≤ 0.0037** across θ ∈ [20°, 70°] with the learned pair, **≤ 0.0014** with the
classic pair (`scripts/sweep_angles.py`). Either way the tolerances (0.10–0.15) sit
one and a half to two orders of magnitude above it. The floor is a property of the tracker
and the fits, not of any video model — which is why the script prints it for whichever pair
ran rather than hard-coding one number.

`wrong_angle` is the cleanest demonstration: prompt 45°, and the chain recovers the true
30° launch while M2/M4/M5 all stay at the floor. Its M1 residual is 0.5502 against the
exact `log(tan45°/tan30°) = 0.5493` — agreement to 9e-4 (0.5491, i.e. 2e-4, on the classic
pair). The instruction error is isolated from the physics, and measured to three decimals.

`sweep_angles.py` also checks the *fixture*: analytically, the renderer satisfies
`H/R = tan θ/4` to 1e-16 at every angle. Without that, the synthetic validation would be
resting on an unverified fixture and would prove nothing.

`const_speed` is a documented **sensitivity limit**: a correct parabolic *path* traversed
at uniform speed is only ~2% from parabolic in normalised RMS at 45°, because speed varies
by just `1/cos θ` over the arc. It is detected at 12× the floor (14× on the classic pair)
but sits under the tolerance. Reported rather than tuned away.

## Result: MiniMax-H3, 12 clips (3 angles × 2 phrasings × 2 seeds)

Scored with the default learned pair; the classic-pair numbers are in the tracking section
above and differ by ≤ 0.002 on every line.

```
PMR 1.000   PE 0.3105   PPR 0.000   HVR 0.167

invariant              mean e   pass   mean measured   theory
M1_HR_nominal          0.5682   0.25         0.4297   0.2758
M2_time_symmetry       0.0389   1.00         1.0303   1.0000
M4_parabolicity        0.0244   0.67         0.0244   0.0000
M5_vx_conservation     0.3243   0.00         0.7269   1.0000
M1_HR_selfconsistent   0.3456   0.00   (diag)
M3_space_symmetry      0.3704   0.00   (diag)
M6_gravity_symmetry    0.1226   0.75         1.0857   1.0000  (diag)
M7_accel_geometry      0.0100   1.00         8.0040   8.0000  (diag)
```

**PMR = 1.000.** Every clip was trackable — coverage 1.000 on both backends, disagreement
0.037–0.048 ball radii against a 0.50 gate — so none of the physics scores are hiding behind
unmeasurability. The failures below are real.

### The model ignores the angle instruction entirely

| prompted θ | H/R-implied θ, per seed and phrasing |
|---|---|
| 30° | 60.3°, 61.8°, 58.1°, 56.1° |
| 45° | 63.8°, 61.1°, 58.8°, 56.3° |
| 60° | 63.2°, 60.7°, 59.0°, 56.1° |

```
measured(H/R-implied)      =  0.023 * prompted + 58.6
measured(launch-direction) = -0.033 * prompted + 51.9
```

Slope ≈ 0 on both estimators: the launch angle is statistically independent of what was
asked. The model has one preferred trajectory (~56–64°) and produces it regardless. Both
slopes reproduce to two decimals under the classic tracker pair, so this is a property of
the videos and not of the measurement.

This is why M1 must be reported twice. M1-nominal's pass rate is 0.25 — and *all* the passes
are the θ=60° clips, which pass only because the model's fixed angle happens to sit near
60°. M1-selfconsistent passes **0 of 12**: even where instruction-following succeeds by
coincidence, the launch direction disagrees with the arc geometry. A scalar physics score
would have recorded θ=60° as a partial success.

### Gravity timing is right; horizontal momentum is not

M2 passes **12/12** (mean e = 0.039) — the up/down frame counts are symmetric everywhere. So
is M7 (mean measured 8.0040 vs theory 8, a 0.05% agreement): the fitted vertical
acceleration is consistent with the observed height and duration.

M5 fails **12/12** (mean e = 0.324): `v_x` drops ~27% over the flight, with no contact. At
θ=45° seed 42 it goes 13.84 → 11.58 px/frame. Drag-like, and wrong for free flight.

M4 rated that same clip 0.0150 — *passing*. That gap is the single most useful methodological
finding here: drag bends `x(t)` into a gentle parabola, a parabola sits close to its own
best-fit line in RMS terms, so an RMS-of-residual test is structurally weak against smooth
low-order error. The first-order velocity ratio reads it as 18% off. A residual test and a
ratio test are not interchangeable.

M3 (0/12) is the spatial shadow of the same defect — with `N_up ≈ N_down`, an asymmetric
horizontal extent *is* the `v_x` loss.

### Diagnostic profile

Exactly the per-domain breakdown `proposal.md` §4.3 asks for, from one task:

| capability | verdict |
|---|---|
| vertical dynamics (timing, accel–geometry consistency) | correct — M2 12/12, M7 12/12 |
| horizontal momentum conservation | fails systematically — M5 0/12 |
| angle instruction-following | absent — slope 0.02 |
| trajectory measurability | perfect — PMR 1.000 |

A single "physics score" would average these into one middling number. Separately measured,
they say something specific: the model has learned that thrown things rise and fall
symmetrically under gravity, has *not* learned that horizontal velocity is conserved, and
does not condition the launch angle on the prompt at all.

### Prompt phrasing

| variant | mean e_i | flight frames | mean launch angle |
|---|---|---|---|
| `plain` | 0.3504 | 96.8 | 51.7° |
| `slowmo` | 0.2702 | 101.3 | 49.0° |

Asking for slow motion bought 4.5 more flight frames out of 124 — essentially nothing. The
model fills the clip with the arc either way. `slowmo`'s lower e_i is not better physics: it
launches slightly less steeply (57.4° vs 61.8° H/R-implied), which happens to sit closer to
the prompted angles. M2 and M5 are unchanged between the two.

## Layout

```
physbench/       first_frame  render_synth  vdm       generation side
                 tracking  neural_track  kinematics  fits      measurement side
                 metrics  config  viz  io_video      scoring and reporting
                 tasks/projectile.py                 P2: the 7 measurables
scripts/         make_first_frames  generate  evaluate  report
                 track_worker.py                     SAM2/CoTracker, runs in envs/track
                 validate_metrics  sweep_angles  edge_cases    calibration + assertions
                 tracker_precision                   backend choice, measured
                 make_synth_dataset                            fixtures
                 debug_real  debug_track  dbg_edge             per-clip diagnosis
                 create_env  create_track_env  fetch_tracker_weights
configs/         p2_projectile.yaml                  tolerances, gates, backends, prompts
data/            first_frames  videos  results  synthetic
cache/           cotracker/  hf/  torch/  scratch/   vendored source, weights, frame handoff
envs/physbench/  numpy scipy opencv av matplotlib (no torch)
envs/track/      torch transformers scipy           the only env with torch
```

Three scripts assert correctness and should stay green — each takes `--backends` so both
tracker pairs can be checked:
`validate_metrics.py` (each invariant fires on its own failure mode),
`sweep_angles.py` (fixture is exact, chain unbiased over 20°–70°),
`edge_cases.py` (nine boundary clips land in the right bucket).
`tracker_precision.py` is not an assertion but the evidence for the default backend pair.

## Known limits

- **Angle is prompt-only, never drawn.** Intentional, so M1-nominal is a real
  instruction-following test — but it means M1-nominal conflates two errors, which is why
  M1-selfconsistent is reported alongside it.
- **Motion blur biases contact.** A ball descending 25 px/frame has its blurred centroid
  ~half a step above true contact, so the landing crossing is extrapolated a fraction of a
  frame. Bounded at 2 frames, beyond which the sample is not measurable.
- **Camera drift is translation only.** Rotation and zoom are not compensated; they surface
  as inflated QC residuals instead of being silently absorbed.
- **`const_speed` sensitivity** as above.
- **M4's tolerance is the loosest link.** Its synthetic floor is 0.0017 (learned pair;
  0.0014 classic), but real-video values run 0.015–0.032 against a 0.030 threshold, so the
  margin is ~2× rather than the ~100× the other invariants enjoy. Part of that spread is
  genuine non-parabolicity (M5 confirms a real 27% `v_x` loss on the same clips) and part is
  tracker noise on generated video, and the two cannot be separated without a known-good
  *real* recording. That both tracker pairs land on the same M4 values (0.0244 vs 0.0243
  mean) narrows the tracker-noise share but does not eliminate it — they share the fitting
  stage. Treat M4's pass/fail on real clips as the weakest verdict in the set; M5 is the
  load-bearing test for the same physics.
- **CoTracker runs at 384×512 internally** and SAM2's mask logits are 256×256 upsampled, so
  neither learned backend measures at native resolution. This does not show up in the
  residuals (see the tracking section) because the fits average over ~60 frames, but it
  would matter for any future measurable that depends on a single frame's geometry.
- **The learned pair needs a GPU and the vendored cache.** `cache/` must be populated by
  `scripts/fetch_tracker_weights.sh` and the worker runs with `HF_HUB_OFFLINE=1` on purpose:
  a silently re-downloaded checkpoint is a silently different tracker, and the noise floor
  above would no longer describe it. `cache/cotracker.rev` records the pinned revision.
- **CoTracker3's weights are CC-BY-NC-4.0.** Non-commercial only, unlike SAM2's Apache-2.0.
  Relevant if this benchmark is ever run inside a commercial pipeline; the classic pair has
  no such restriction.
- **Shared host.** Other tenants' memory spikes have SIGKILLed generations mid-run;
  `generate.py --retries` handles that, and finished samples are skipped on restart.
