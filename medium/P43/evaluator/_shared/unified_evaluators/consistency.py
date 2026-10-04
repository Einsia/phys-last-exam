"""Evidence-backed visual and task-observability gate before physical extraction."""
from dataclasses import dataclass, replace
from functools import lru_cache
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / '.models/Qwen3-VL-8B-Instruct'
RUBRIC_VERSION = 'task-observability-evidence-v26'
TASK_CONTRACT_VERSION = 'neutral_observable_contracts_v1'
TASK_SHARED_RULES = ['Judge whether the listed subjects, stages and comparisons are actually observable. A task description is not evidence that an event happened.', "Preserve identities and connections across time. A moving apparatus, a replacement object or an added duplicate does not establish the original subject's requested trajectory.", 'An incorrect but clearly observed physical response can be fully observable. Do not grade quantitative physical laws, predicted directions, timing ratios or conservation in this stage.', 'Do not infer hidden current, charge, mass, magnetism, friction, density or an unseen release mechanism from their expected physical effects.', 'Use the interval that provides the required comparison. Extra waiting or later contact does not erase already visible evidence unless the contract requires that terminal transformation itself.', 'Camera style, exact pauses, decorative colors and background appearance are not core requirements. They matter only when they hide subjects, break identity or prevent the listed comparison.']
TASK_CONTRACTS = json.loads(r'''{
  "P1": {
    "subjects": "One identifiable, visibly untethered ball and a stable spatial reference.",
    "required_observations": [
      "The same ball moves downward separately from its support over a readable interval; distinguish its movement from camera motion."
    ],
    "readable_regions": [
      "Ball boundary and its successive positions relative to the reference."
    ],
    "do_not_require": [
      "Impact, absence of impact, a particular final height, or continued motion until the last frame.",
      "A particular speed, acceleration or gravitational law."
    ]
  },
  "P2": {
    "subjects": "One identifiable projectile and a visible launch/landing level.",
    "required_observations": [
      "The same projectile leaves its initial location, rises, turns near its apex and descends; its return to the launch level is visible for a complete trajectory comparison."
    ],
    "readable_regions": [
      "The projectile throughout the trajectory, the apex and the endpoint at the reference level."
    ],
    "do_not_require": [
      "A particular launch angle, parabolic curve, acceleration, range or flight time."
    ]
  },
  "P3": {
    "subjects": "Two separately identifiable balls, their fixed launch apparatuses and a common table/reference level.",
    "required_observations": [
      "Both original balls detach and travel through the air as independently trackable subjects; moving or bending launcher rods do not count as ball flight.",
      "Observe both complete flight paths and both landing positions at the common reference level so their endpoints can be compared."
    ],
    "readable_regions": [
      "Each ball from launch through its endpoint, with enough separation to distinguish both trajectories and any extra same-frame objects."
    ],
    "do_not_require": [
      "Correct complementary angles, equal initial speeds, exact synchronization, particular arc heights, equal ranges or the predicted relative flight times."
    ]
  },
  "P4": {
    "subjects": "One identifiable ball and one stable rebound surface.",
    "required_observations": [
      "The original ball descends, contacts the surface and makes successive complete rebounds whose peaks can be compared; an additional ball cannot substitute for the unreleased original."
    ],
    "readable_regions": [
      "Ball-surface contacts and the successive peak positions of the same ball."
    ],
    "do_not_require": [
      "Correct decay of rebound height or a particular restitution coefficient.",
      "An exact decorative pause or exact total bounce count once the successive rebound comparison is available."
    ]
  },
  "P5": {
    "subjects": "Two identifiable balls on a common collision path.",
    "required_observations": [
      "Track both original balls before their encounter, through visible contact and into their post-contact motion."
    ],
    "readable_regions": [
      "Both centers and the contact region before and after collision."
    ],
    "do_not_require": [
      "Correct exchange of velocities, momentum or energy conservation, or visual proof of equal hidden masses."
    ]
  },
  "P6": {
    "subjects": "One patterned ball, an incline and its connected level runout.",
    "required_observations": [
      "The original ball moves along the incline toward its runout; falling off to the side or directly to the table is a different event.",
      "The pattern remains associated with the ball so translation and rotation can be compared during its movement along the ramp."
    ],
    "readable_regions": [
      "Ball boundary, attached pattern, ramp contact region and incline-to-runout transition."
    ],
    "do_not_require": [
      "The correct no-slip relation between angular and linear speed or the predicted acceleration.",
      "A particular release-gate withdrawal path after the ball is moving freely."
    ]
  },
  "P7": {
    "subjects": "A sphere and a ring, separately identifiable on the same inclined track.",
    "required_observations": [
      "Follow both subjects moving down the incline during a common observation window and reaching comparable track positions."
    ],
    "readable_regions": [
      "Both trajectories, outlines and visible rotation relative to the common track."
    ],
    "do_not_require": [
      "The correct winner, acceleration ratio, arrival-time ratio or no-slip relationship."
    ]
  },
  "P8a": {
    "subjects": "Two pendulums with equal visible lengths and distinguishable initial amplitudes.",
    "required_observations": [
      "Both original bobs swing back and forth for complete cycles that allow their periods to be compared."
    ],
    "readable_regions": [
      "Both bobs, strings, pivots and turning positions during the same time interval."
    ],
    "do_not_require": [
      "Equal periods, preserved phase synchronization or any predicted amplitude dependence."
    ]
  },
  "P8b": {
    "subjects": "Two pendulums with equal visible lengths and distinguishable bobs.",
    "required_observations": [
      "Both bobs undergo trackable back-and-forth cycles in a common observation window."
    ],
    "readable_regions": [
      "The bobs, strings, pivots and cycle timing."
    ],
    "do_not_require": [
      "Visual proof of hidden bob masses or correct mass independence of the period."
    ]
  },
  "P8c": {
    "subjects": "Two separately identifiable pendulums with equal visible string lengths and distinguishable initial swing amplitudes.",
    "required_observations": [
      "Both bobs leave their initial poses and complete back-and-forth cycles, allowing the periods at their amplitudes to be compared."
    ],
    "readable_regions": [
      "Both bobs and their turning positions through the common cycle interval."
    ],
    "do_not_require": [
      "The correct period ordering, exact release synchronization or predicted large-angle period correction."
    ]
  },
  "P9": {
    "subjects": "Two pendulums with visibly different string lengths.",
    "required_observations": [
      "Both original bobs execute complete oscillation cycles so the two periods can be compared."
    ],
    "readable_regions": [
      "Pivots, strings, bobs and turning times for both pendulums."
    ],
    "do_not_require": [
      "The correct square-root period-length relationship or period ordering."
    ]
  },
  "P10": {
    "subjects": "One identifiable block and an inclined track.",
    "required_observations": [
      "The same block travels up the incline, reverses direction and returns down it; the turn connecting outward and return motion must be observable."
    ],
    "readable_regions": [
      "The block, track and turning region across the outward and return segments."
    ],
    "do_not_require": [
      "Correct friction, speed or acceleration values, or symmetric outward and return durations."
    ]
  },
  "P11": {
    "subjects": "An incident light ray, a liquid interface, a normal/reference direction and the ray continuation at the interface.",
    "required_observations": [
      "A readable stationary ray/interface configuration is sufficient; connect the incident and outgoing segments to the same interface location."
    ],
    "readable_regions": [
      "Both ray segments, their interface junction and the interface/normal geometry."
    ],
    "do_not_require": [
      "Correct bending direction or Snell-law angles.",
      "Visible source switching or motion in an otherwise readable optical configuration."
    ]
  },
  "P12": {
    "subjects": "Four distinguishable beam configurations at separate air-water interface locations: three incident from air and one from water.",
    "required_observations": [
      "Keep each beam associated with its interface point and show its visible continuation or reflected branch.",
      "The incident angles vary about their interface points over the clip, providing multiple conditions for comparing the interface response."
    ],
    "readable_regions": [
      "The interface, each beam junction, normal/reference directions and the visible branches on the appropriate sides."
    ],
    "do_not_require": [
      "The correct refracted/reflected branch, critical angle or angle relationship; a straight or incorrectly bent ray is a physical result when its geometry is clear.",
      "A transmitted air branch for the water-side beam when the shown response is total reflection.",
      "Exact split into equal-duration stages or constant cosmetic interface styling."
    ]
  },
  "P13": {
    "subjects": "An incident ray, an outgoing ray, a mirror and a normal/reference at their junction.",
    "required_observations": [
      "A stable, clearly connected incident/outgoing ray configuration at the mirror is sufficient."
    ],
    "readable_regions": [
      "Both ray arms, mirror, reflection point and reference direction."
    ],
    "do_not_require": [
      "Equality of the two angles or visible motion/switching."
    ]
  },
  "P14": {
    "subjects": "One point-like light source, four distinguishable rods and their associated shadows.",
    "required_observations": [
      "Associate each rod base with its visible shadow and tip in a common scene; a static configuration is sufficient."
    ],
    "readable_regions": [
      "Rod bases, shadow paths and endpoints, and the source/reference geometry."
    ],
    "do_not_require": [
      "Correct radial convergence, shadow-length relationships or perspective geometry.",
      "Particular rod colors, background colors or decorative markings."
    ]
  },
  "P16": {
    "subjects": "One rigid straight rod with four distinct fixed markers, a wall and a floor.",
    "required_observations": [
      "The same marked rod translates and rotates through distinguishable poses while its endpoints can be related to the wall and floor.",
      "The four markers remain associated with the rod rather than sliding, swapping, appearing or disappearing."
    ],
    "readable_regions": [
      "The complete rod, all four marker centers and both endpoint/reference regions."
    ],
    "do_not_require": [
      "The correct cross-ratio or a particular trajectory or speed."
    ]
  },
  "P18": {
    "subjects": "One detached bare ball beside a liquid container, with a readable free surface.",
    "required_observations": [
      "Observe the same ball descending over a common interval in which the direction of the liquid free surface is also readable, allowing trajectory and surface directions to be compared."
    ],
    "readable_regions": [
      "Ball trajectory and the direction of the liquid surface simultaneously relative to stable references."
    ],
    "do_not_require": [
      "Exact perpendicularity or the correct acceleration.",
      "A fall into the liquid, impact, absence of impact or a particular last-frame position."
    ]
  },
  "P19": {
    "subjects": "A connected U-shaped liquid system with two distinguishable arms and liquid surfaces.",
    "required_observations": [
      "Both arms and their connection remain identifiable, with readable surface positions for a common equilibrium comparison; an already stable comparison is usable."
    ],
    "readable_regions": [
      "Both liquid surfaces, their common vertical reference and the connecting vessel."
    ],
    "do_not_require": [
      "Equal surface heights or the correct pressure relationship.",
      "A visible disturbance if the equilibrium geometry is already readable."
    ]
  },
  "P20": {
    "subjects": "One intact rectangular ice block in a transparent liquid container.",
    "required_observations": [
      "Track the block's position relative to the liquid surface and observe its settled immersion geometry; a readable stable state can provide this comparison."
    ],
    "readable_regions": [
      "The complete block outline, waterline and container reference."
    ],
    "do_not_require": [
      "A correct immersed fraction, correct buoyant response or visual proof of material density.",
      "A particular transient settling motion when the immersion geometry is already available."
    ]
  },
  "P21": {
    "subjects": "The original ice body, surrounding water and a readable waterline in one container.",
    "required_observations": [
      "Track the original solid ice actually reducing and melting completely into liquid, with the final absence of the original solid directly observable.",
      "Compare waterlines before and after the transformation. An intact body merely sinking, drifting or leaving view does not establish melting."
    ],
    "readable_regions": [
      "Original ice boundary and remaining solid, liquid surface and stable container reference."
    ],
    "do_not_require": [
      "A correct water-level change, prescribed melt rate or correct floating behavior while the transformation is readable."
    ]
  },
  "P21b": {
    "subjects": "An original ice body containing a stone, the surrounding water and container.",
    "required_observations": [
      "The original ice visibly melts completely and releases its original stone; observe the final absence of solid ice rather than substituting an added stone or translating the intact ice.",
      "Read the stone location and waterline before and after release."
    ],
    "readable_regions": [
      "Ice boundary, contained/released stone and waterline."
    ],
    "do_not_require": [
      "The predicted direction of water-level change, sink rate or correct buoyancy response."
    ]
  },
  "P21c": {
    "subjects": "One original ice body in a transparent liquid container with a readable surface.",
    "required_observations": [
      "The original solid ice visibly reduces and melts completely into liquid, with the final absence of the original solid directly observable; an intact body moving downward does not substitute for melting.",
      "Compare the liquid surface before and after the transformation."
    ],
    "readable_regions": [
      "Original ice and any remaining solid, liquid surface and container reference."
    ],
    "do_not_require": [
      "Correct salt-water density effects, mixing behavior, floating response or water-level change; salinity is not visually measurable here."
    ]
  },
  "P23": {
    "subjects": "The original water volume and its container.",
    "required_observations": [
      "The entire original liquid water volume transforms into a final solid-ice state, allowing its initial liquid and completely frozen states to be compared; merely adding ice or forming a separate patch while the original water remains liquid does not establish complete freezing."
    ],
    "readable_regions": [
      "Initial water surface, evolving solid boundary and final ice surface relative to the container."
    ],
    "do_not_require": [
      "A correct expansion ratio, freezing rate or prescribed surface-height change."
    ]
  },
  "P27": {
    "subjects": "Two distinguishable containers with compact ice and fragmented ice.",
    "required_observations": [
      "Follow the original solid material in both containers as it actually melts completely, with identifiable initial/intermediate states and final absence of solid ice for comparison."
    ],
    "readable_regions": [
      "Remaining solids and liquid surfaces in both containers throughout the common comparison."
    ],
    "do_not_require": [
      "The predicted relative melting rates, equal final water heights or a particular elapsed duration."
    ]
  },
  "P28": {
    "subjects": "Two separate suspended balls, their complete strings and the shared support arrangement.",
    "required_observations": [
      "The balls' stable positions and string geometry are readable for an equilibrium comparison; a static configuration is valid."
    ],
    "readable_regions": [
      "Both ball centers, strings and attachment points relative to the support."
    ],
    "do_not_require": [
      "Correct symmetry, separation angles, repulsion or visual proof of charge and mass."
    ]
  },
  "P34": {
    "subjects": "Two compass needles with identifiable pivots/housings and the central conductor arrangement.",
    "required_observations": [
      "Both original needles remain distinguishable, with their orientations readable during a common observation interval and at the comparison state; track any actual rotation without inventing it."
    ],
    "readable_regions": [
      "Each needle, its pivot and orientation relative to fixed housings and conductor."
    ],
    "do_not_require": [
      "The predicted opposite directions, correct angles, visible current switching or needle rotation when a stable orientation is already clearly shown."
    ]
  },
  "P36": {
    "subjects": "Two independently identifiable blocks side by side on the same inclined plate.",
    "required_observations": [
      "Observe both blocks' actual positions and responses over a common interval with the plate as reference; a clearly tracked block remaining still is an observable response rather than an invisible subject."
    ],
    "readable_regions": [
      "Both block boundaries, track contacts and positions along the common incline."
    ],
    "do_not_require": [
      "Which block is faster, a minimum speed difference or visual proof of internal magnets, equal masses or an unseen release mechanism."
    ]
  },
  "P37": {
    "subjects": "Two fixed coil/core assemblies, each with its own separately identifiable loose ring: one closed outline and one visibly gapped outline.",
    "required_observations": [
      "Observe each original ring's position relative to its own fixed core over the comparison interval, including an unchanged position when that is what the video shows.",
      "Distinguish movement of a separate ring from lifting, stretching or replacing the entire apparatus."
    ],
    "readable_regions": [
      "Both ring outlines, the visible closed-versus-gapped distinction, their relation to the cores and fixed coil/support geometry."
    ],
    "do_not_require": [
      "Either or both rings to jump, a predicted difference between rings, visible current activation or visual proof of hidden electrical continuity."
    ]
  },
  "P38": {
    "subjects": "One solid plate and one slotted plate, each attached to its own pendulum suspension.",
    "required_observations": [
      "Both original plates undergo readable back-and-forth motion over a common interval long enough to compare changes between cycles."
    ],
    "readable_regions": [
      "Complete plate outlines, slots, suspension connections and turning positions without identity loss."
    ],
    "do_not_require": [
      "The correct damping rate, which plate damps faster or matching periods."
    ]
  },
  "P39": {
    "subjects": "One identifiable magnet, a fixed coil, its leads and a visible indicator lamp.",
    "required_observations": [
      "Track the original magnet from an outside position into and through the coil and out the other side, with stationary comparison intervals before and after passage.",
      "The lamp remains readable throughout the magnet's passage, including when it stays dark."
    ],
    "readable_regions": [
      "Magnet, coil opening and lamp during the common timeline."
    ],
    "do_not_require": [
      "Correct illumination, polarity or brightness timing.",
      "A particular hand path, grip or withdrawal action if the magnet passage and lamp remain readable."
    ]
  },
  "P40": {
    "subjects": "Two separate piles of the same granular material with different visible sizes on a common supporting surface.",
    "required_observations": [
      "Observe each pile's formation or stable settled profile, with boundaries available for comparing their slopes."
    ],
    "readable_regions": [
      "Each pile face, base and outer outline relative to a surface reference."
    ],
    "do_not_require": [
      "Correct repose angles or scale invariance.",
      "A visible feeder or continued pouring once the stable comparison profiles are available."
    ]
  },
  "P41": {
    "subjects": "Two separate funnel/container systems, one with liquid and one with granular material.",
    "required_observations": [
      "Observe actual discharge and material-level evolution in each system over a common interval; a substituted material or unrelated moving object does not establish the requested discharge."
    ],
    "readable_regions": [
      "Both contents, surface levels, outlet regions and emitted streams."
    ],
    "do_not_require": [
      "Correct flow-rate dependence, which container empties first or an exact empty final frame once usable discharge/level comparisons exist."
    ]
  },
  "P42": {
    "subjects": "Two original blocks on one hinged tilting board.",
    "required_observations": [
      "Observe increasing board inclination and each block's transition from rest to sliding, so their onset conditions can be compared."
    ],
    "readable_regions": [
      "Both blocks relative to the board, the board angle and both onset intervals."
    ],
    "do_not_require": [
      "Equal onset angles, correct friction coefficients or mass independence.",
      "A perfectly monotonic actuator speed when the onset comparison is still available."
    ]
  },
  "P43": {
    "subjects": "One block, a contacting actuator pad and a supporting surface.",
    "required_observations": [
      "Track the same block from its supported upright state through visible rotation about a support region into a tipped state; translation alone is a different event."
    ],
    "readable_regions": [
      "The block outline, pad contact and lower support/pivot region during the tipping transition."
    ],
    "do_not_require": [
      "The correct tipping threshold, center-of-mass relation or exact angular trajectory.",
      "A decorative final pause after a clearly observable tipped state."
    ]
  },
  "P44": {
    "subjects": "One continuous chain attached to two distinct fixed endpoints.",
    "required_observations": [
      "The connected chain's settled hanging profile is visible for geometric comparison; a stable profile is sufficient."
    ],
    "readable_regions": [
      "The complete chain curve and both endpoint attachments."
    ],
    "do_not_require": [
      "A correct catenary equation, a particular settling transient or exact duration of the final pause."
    ]
  },
  "P45": {
    "subjects": "Two capillary tubes of distinguishable bore sizes with their lower ends accessing the same liquid reservoir.",
    "required_observations": [
      "Both tubes visibly access the shared liquid through their lower ends; isolated prefilled columns do not establish this setup.",
      "Both internal liquid columns or menisci and the external reservoir surface are readable over the common comparison interval, including a stationary or absent rise if clearly shown."
    ],
    "readable_regions": [
      "Tube bores, lower-end access to the reservoir, each internal meniscus and the shared external liquid level."
    ],
    "do_not_require": [
      "The smaller tube to rise higher, any nonzero rise, correct inverse-radius scaling or a prescribed ordering of menisci.",
      "Initially dry tubes, repeated insertion when the reference already shows immersion, visual proof of wetting chemistry or a particular liquid color."
    ]
  },
  "P47": {
    "subjects": "Two differently sized connected bubbles with one distinguishable internal partition.",
    "required_observations": [
      "Observe the original two bubble boundaries and their connected partition together; a stable connected state is sufficient."
    ],
    "readable_regions": [
      "Both outer contours and the full common partition without loss or substitution."
    ],
    "do_not_require": [
      "Correct partition curvature, its predicted direction or a curvature-radius ratio."
    ]
  },
  "P48": {
    "subjects": "Two original separate liquid drops with distinguishable outlines.",
    "required_observations": [
      "Follow the same two drops from separation through contact into one connected liquid body with a readable final single-drop state; approaching, touching while remaining two drops, or replacing them with another object is incomplete."
    ],
    "readable_regions": [
      "Both initial outlines, contact/connection region and final single-body boundary."
    ],
    "do_not_require": [
      "Correct volume conservation, final radius ratio, merger speed or photorealistic decorative styling."
    ]
  },
  "P49": {
    "subjects": "Two separately identifiable balls of different radii in the same transparent liquid tank.",
    "required_observations": [
      "Track both balls descending during a common interval with enough successive positions to compare their speed evolution; trajectories cut off before any usable comparison are incomplete."
    ],
    "readable_regions": [
      "Both ball boundaries and positions relative to the tank through the comparison interval."
    ],
    "do_not_require": [
      "Correct terminal speeds, attainment of an exact constant-speed regime, the predicted radius-squared ratio or visual proof of viscosity and material density."
    ]
  }
}''')

ATTENTION_POLICY = 'torch_sdpa_without_cudnn'
DIMENSIONS = ('identity_topology', 'shape_stability', 'scene_continuity', 'rendering_clarity')
SEVERITY_PENALTIES = {0: 0.0, 1: 0.05, 2: 0.15, 3: 0.30, 4: 0.50}
TASK_DIMENSIONS = ('setup_fidelity', 'event_coverage', 'observation_readability')
TASK_SCORE_CEILINGS = {0: 1.0, 1: 0.85, 2: 0.65, 3: 0.40, 4: 0.15}
OBSERVATION_RUBRIC = '''Inspect the actual visible objects across this video, without guessing its task.
The six overview panels show six different times, left to right then next row.
They are not six simultaneous scenes. Use the individual video samples as well.
For EACH of those six times make a concrete object inventory:
- Count the main bodies simultaneously visible, distinguishing their appearance.
- State their actual boundaries, positions relative to fixed apparatus, and
  whether they are connected to supports or separated by a visible gap.
- Include newly appearing bodies and bodies left behind. Two same-looking bodies
  visible at different positions in ONE instant are two visible bodies; do not
  narrate them as one moving body. A colored tip still on a rod is not a free ball.
- State which main bodies or small working parts cannot actually be located at
  that time. A visible string, support or housing does not prove that its plate,
  bob or pointer is visible. Do not invent positions from expected oscillation.
- Compare the complete subject outline and working parts to earlier times. A
  small central spot or short residue is not automatically the original full
  pointer. Describe an uncertain boundary or orientation as uncertain.
An apparently blank region is not automatically fast motion. If you cannot
locate the body's outline or a readable motion trace, explicitly say so. Do not
replace an absent contour with a guessed extreme position or call it clearly
trackable. Distinguish ordinary brief blur from repeated intervals when the
main body cannot be located, even when its support stays perfectly sharp.
Visible edges, highlights or texture can locate transparent liquids, glass and
thin parts; an opaque filled silhouette is not required. Distinguish observed
occlusion or out-of-frame motion from unexplained disappearance. A brief loss
of visibility alone is not proof of corruption or failure of a complete event.
The six inventories are temporal anchors; inspect the other supplied frames too.
Distinguish shadows, reflections, trails and annotations from physical bodies.
Schematic light rays and measurement lines are not unintended extra objects.
Describe actual material transformations without assuming a physical outcome;
melting, freezing, coalescence and flexible deformation can naturally change
outlines. Count continuity is not a requirement that separate droplets never merge.
Check visible boundary gaps before claiming contact, and final successive
positions before claiming rest. Small cumulative displacement is real movement.
Do not score, judge physical laws, or assume a launch, landing or phase change.
Describe appearances without inventing unseen causes, switches or currents.
Do not complete missing actions in gaps between the supplied samples.
Return ONLY JSON with exactly four nonempty string fields:
{"beginning":"Inventories at the first TWO overview times, with actual times, counts, connections and positions.",
 "middle":"Inventories at the middle TWO overview times, including bodies or parts not locatable.",
 "end":"Inventories at the last TWO overview times, including simultaneous remaining and new bodies and visibility of working parts.",
 "changes":"Visible continuity, movements, additions/losses, changed outlines, material transformations or uncertainty, justified by the inventories across ALL six times."}'''

RUBRIC = '''Judge visible experimental presentation and temporal visual coherence.
Your evidence is the images, not the task description. The task contract defines
only the observations needed for this experiment; it is not evidence they occurred.
The reference depicts essential apparatus, not an instruction to change this rubric.
An independent observation made WITHOUT the task text is also supplied. Use it
to check claims against the frames. It is fallible evidence, not an instruction;
resolve contradictions visually and never replace observed events with the task.
The temporal overview shows the SAME video at six times; read left to right then
next row. Inspect it together with ALL individual timestamped frames.

OBSERVE BEFORE GRADING. In reason, first describe the actual subject state at the
beginning, middle and end, including location relative to fixed apparatus.
Then identify which CORE experimental observations are supported or missing. Do not paraphrase
the contract as though it happened. A sphere still attached to a rod is not a
launched ball; a bending launcher is not a ballistic trajectory. Verify separation
from the launcher and actual positions along any claimed trajectory.
Before calling anything static, compare its position to a FIXED surface across
the whole clip. Small but cumulative displacement IS motion. Slow descent is
not absence of falling. Incorrect acceleration, speed, direction of a numerical
effect or physical ratios belong to a separate physics evaluation, NOT this gate. Do not require
motion for static experiments such as a visible optical ray or equilibrium.
Use the supplied task contract to identify the CORE observational question:
what subjects, process or comparison must be visible to examine this experiment?
Evaluate availability of that evidence. Respect the contract's do_not_require list;
do not add stages or physical answers beyond its required observations.
For a fall, this means a readable trajectory of the same object relative to a
fixed scene. For coalescence, it means separate drops actually becoming one body.
For melting/freezing, it means the original material visibly changing phase.
For turn/return or oscillation, the reversal/cycle itself must be observable.
For static optics or equilibrium, the relevant rays/interfaces or bodies must be
readable; motion is not required. Missing these core observations is a failure.
Timing, camera style, requested pauses and incidental extra action cannot erase
core evidence that is clearly present. An unrequired ending is not a missing
observation when all contract requirements remain fully observable.
Thus a clear readable fall is available even if it eventually reaches a surface;
contact is not a missing fall. Conversely, separate drops merely approaching
never supply evidence of merging. Decide from what IS visible, not a predicted
future. A shadow alone does not prove contact. Do not invent impacts or rest.
For any task severity 2+, name the unavailable CORE observation and explain why
the visible sequence is insufficient to examine it. An unrequired detail
by itself is not such an explanation.

TASK ASSESSMENT: rate these three dimensions using the TASK scale below:
setup_fidelity: essential subjects, count, material, connections and fixed
apparatus preserve the intended experiment. Cosmetic changes do not matter.
event_coverage: availability of the core phenomenon and its essential phases or
comparisons. A trajectory must actually move; a transformation must actually
transform. Smooth unrelated motion and plausible-looking objects do not establish
the phenomenon. Distinguish unavailable observations from visible behavior whose
quantitative physics is wrong. Only treat a final state as essential when it is
needed to observe the phenomenon itself, such as merged drops or a phase change.
observation_readability: the MAIN subject, interfaces, pointers, rays or contact
regions remain readable during critical phases and necessary before/after views.
Clean background and apparatus cannot substitute for unreadable measured objects.

TASK severity / meaning:
0: complete and clearly supported; cite positive timestamp evidence.
1: small omission/ambiguity, but all core subjects/stages remain available.
2: substantial partial presentation; some of the core phenomenon is visible but
   an essential comparison or phase needed to examine it is incomplete/ambiguous.
3: critical stage, main subject or observation absent/unreadable, or a major setup
   mismatch prevents following the experiment.
4: core event never happens or is replaced by another action, or experiment/main
   subject is absent or effectively unreadable throughout.
ALL THREE task dimensions require nonempty timestamp evidence EVEN at severity 0.
For event_coverage, describe the actual beginning, middle and end observations;
when claiming an event never occurs, specify what is seen instead over the clip.
Do not guess missing events in gaps between sampled frames. State uncertainty
and partial presentation instead of inventing completion or definite corruption.

Examples: two drops stay separate instead of merging; intact ice sinks instead
of melting; a block exits before a required turn/return; a ball falls off a ramp
instead of rolling along it. These are absent/replaced events, even if smooth.
A surface rising by the wrong amount after visible melting is a PHYSICS issue.
A clear complete flight at the wrong acceleration is still a presented flight.
Compass needles unreadable at the final state, or main bodies repeatedly lost
while supports stay sharp, fail observation_readability. Ordinary brief motion
blur is fine when identity and stages are still followable. A sharp stand with
no locatable suspended body is not readable motion. Repeatedly absent outlines
are not excused by calling them blur. For pointers, describe the actual pointer
body and orientation, not just the readable housing or a remaining central spot.
Never use extractor
failure codes, fitting thresholds or hidden physical quantities to grade images.

VISUAL ASSESSMENT: independently inspect four visual dimensions:
identity_topology: unexplained disappearance, duplication, identity/count change,
broken connectivity. Distinguish actual occlusion/out-of-frame motion from loss
of an exposed object. Reflections/shadows/isolated blur ghosts are not extra balls.
shape_stability: unintended rigid deformation, stretching or changing silhouette.
Natural flexible deformation, fluid merging/splitting or phase changes are not
artifacts by themselves; check whether they form the requested event instead.
Do not judge numerical conservation. Directly visible added or removed material
can still break identity or invalidate a before/after comparison of the original material.
scene_continuity: abrupt/repeated lighting/texture/layout jumps or continuity-
breaking cuts. Coherent camera motion and smooth illumination changes are fine.
rendering_clarity: persistent/repeated corruption, smearing, fragmented edges or
exposure/defocus that makes structure unreadable. Do not excuse persistent defects
because the main object remains recognizable.

VISUAL severity only: 0 no observable artifact; 1 slight/local/brief artifact;
2 clear repeated/local or moderate sustained defect; 3 strong/persistent defect
that disrupts continuity/readability; 4 gross loss or severe corruption.
For identity_topology, repeated complete disappearance of an exposed main body,
unexplained extra copies, or repeated loss/breakup of its essential working parts
is severity 3 or 4, not a minor local defect, even if the object later reappears.
This does not apply to directly observed occlusion, leaving the field of view,
ordinary brief blur or a visibly continuous natural material transformation.
Several isolated visible poses separated by missing subjects do not by themselves
establish a complete trackable cycle or trajectory. Rate essential missing
comparisons under event_coverage/readability instead of interpolating them.
For these FOUR VISUAL dimensions, severity 0 may have empty evidence or positive
evidence of continuity; nonzero severity requires timestamp evidence of a defect.
A core event being visible earlier does not excuse later visual duplication,
identity loss, deformation or unreadability. Inspect the WHOLE clip for those
visual dimensions, even when an earlier interval answers the core task.
Put each visual defect in ONE primary
visual dimension. Task dimensions may cite its impact; code takes the minimum
score ceiling instead of adding the same task failure several times.

quality_checks are true only if seen:
multiple_balls: multiple distinct physical balls, excluding shadows/reflections.
strong_background_flicker: repeated strong abrupt oscillation, not a smooth fade.
poor_visual_quality: sustained severe smearing/fragmentation/corruption.
Strong flicker is a failure for every task. P2 also requires one ball. P3 may
contain multiple balls; poor rendering is its additional failure condition.
Do not assume any of these problems exist without inspecting the images.

Use actual supplied frame times; evidence intervals must remain within the clip.
Never infer quality from a generator name, target score, prior score or ranking.
Return ONLY one flat JSON object. All seven dimensions and all three booleans
are top-level keys. Do not wrap dimensions in assessment or task_assessment.
Do not give a numerical final score; code applies fixed rules.
Structure (replace descriptions with actual observations):
{"reason":"First: actual state/location; middle: actual state/location; last: actual state/location; supported and missing stages.",
 "identity_topology":{"severity":0,"evidence":[]},
 "shape_stability":{"severity":0,"evidence":[]},
 "scene_continuity":{"severity":0,"evidence":[]},
 "rendering_clarity":{"severity":0,"evidence":[]},
 "setup_fidelity":{"severity":0,"evidence":[{"start_seconds":0.0,"end_seconds":0.0,"description":"observed setup"}]},
 "event_coverage":{"severity":0,"evidence":[{"start_seconds":0.0,"end_seconds":0.0,"description":"observed beginning, middle, end and supported stages"}]},
 "observation_readability":{"severity":0,"evidence":[{"start_seconds":0.0,"end_seconds":0.0,"description":"readable subjects and critical states"}]},
 "multiple_balls":false,"strong_background_flicker":false,"poor_visual_quality":false}
Every evidence item uses start_seconds, end_seconds, description as above.
'''
QUALITY_RUBRIC = RUBRIC


def rubric_for(task_id):
    return QUALITY_RUBRIC if task_id in ('P2', 'P3') else RUBRIC


def apply_quality_rules(judgment, task_id):
    if task_id not in ('P2', 'P3') and 'quality_checks' not in judgment:
        return judgment
    checks = judgment.get('quality_checks')
    required = ('multiple_balls', 'strong_background_flicker', 'poor_visual_quality')
    if not isinstance(checks, dict) or any(type(checks.get(k)) is not bool for k in required):
        raise ValueError('Gate requires explicit boolean quality_checks')
    rejected = [k for k in required if checks[k] and
                (k == 'strong_background_flicker' or (task_id == 'P2' and k == 'multiple_balls')
                 or (task_id == 'P3' and k == 'poor_visual_quality'))]
    if rejected:
        judgment = dict(judgment, raw_vlm_score=judgment['score'],
                        score=min(judgment['score'], .49), forced_rejection=True,
                        rejection_checks=rejected)
    return judgment


def _validate_assessment(value):
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('VLM response must include a nonempty reason')
    assessment = value.get('assessment')
    if not isinstance(assessment, dict) or set(assessment) != set(DIMENSIONS):
        raise ValueError('Assessment must include exactly the four visual dimensions')
    for name, dimension in assessment.items():
        if not isinstance(dimension, dict) or set(dimension) != {'severity', 'evidence'}:
            raise ValueError(name + ' must include severity and evidence')
        severity = dimension['severity']
        if type(severity) is not int or severity not in SEVERITY_PENALTIES:
            raise ValueError(name + ' severity must be an integer from 0 to 4')
        evidence = dimension['evidence']
        if not isinstance(evidence, list) or (severity > 0 and not evidence):
            raise ValueError(name + ' needs timestamp evidence when severity is nonzero')
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {'start_seconds', 'end_seconds', 'description'}:
                raise ValueError(name + ' evidence must include start_seconds, end_seconds and description')
            start, end = item['start_seconds'], item['end_seconds']
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
                raise ValueError(name + ' evidence must use finite ordered nonnegative timestamps')
            if not isinstance(item['description'], str) or not item['description'].strip():
                raise ValueError(name + ' evidence description cannot be empty')
    checks = value.get('quality_checks')
    keys = {'multiple_balls', 'strong_background_flicker', 'poor_visual_quality'}
    if not isinstance(checks, dict) or set(checks) != keys or any(type(v) is not bool for v in checks.values()):
        raise ValueError('All three quality_checks must be explicit booleans')


def _validate_task_assessment(value):
    task = value.get('task_assessment')
    if not isinstance(task, dict) or set(task) != set(TASK_DIMENSIONS):
        raise ValueError('Task assessment requires setup_fidelity, event_coverage and observation_readability')
    for name, dimension in task.items():
        if not isinstance(dimension, dict) or set(dimension) != {'severity', 'evidence'}:
            raise ValueError(name + ' must include severity and evidence')
        if type(dimension['severity']) is not int or dimension['severity'] not in TASK_SCORE_CEILINGS:
            raise ValueError(name + ' severity must be an integer from 0 to 4')
        if not isinstance(dimension['evidence'], list) or not dimension['evidence']:
            raise ValueError(name + ' requires positive or negative timestamp evidence, even at severity 0')
        for item in dimension['evidence']:
            if not isinstance(item, dict) or set(item) != {'start_seconds', 'end_seconds', 'description'}:
                raise ValueError(name + ' evidence requires timestamps and description')
            start, end = item['start_seconds'], item['end_seconds']
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end)) or not 0 <= start <= end:
                raise ValueError(name + ' evidence requires finite ordered nonnegative timestamps')
            if not isinstance(item['description'], str) or not item['description'].strip():
                raise ValueError(name + ' evidence description cannot be empty')


def score_assessment(judgment, task_id, items=None):
    """Apply fixed, published severity deductions, never cohort normalization."""
    _validate_assessment(judgment)
    _validate_task_assessment(judgment)
    assessment = judgment['assessment']
    video_items = [f for f in (items or []) if f.get('role') == 'video']
    if video_items:
        first = min(f['timestamp_seconds'] for f in video_items)
        last = max(f['timestamp_seconds'] for f in video_items)
        for dimension in list(assessment.values()) + list(judgment['task_assessment'].values()):
            for item in dimension['evidence']:
                # Frame labels are rounded to milliseconds in the VLM input.
                if item['start_seconds'] < first - .002 or item['end_seconds'] > last + .002:
                    raise ValueError('Defect evidence falls outside the sampled video timeline')
    penalties = {name: SEVERITY_PENALTIES[d['severity']] for name, d in assessment.items()}
    uncapped = round(max(0.0, 1.0 - sum(penalties.values())), 6)
    caps = []
    if any(d['severity'] == 4 for d in assessment.values()):
        caps.append('severe_visual_incoherence')
    if assessment['identity_topology']['severity'] >= 3:
        caps.append('major_identity_or_topology_defect')
    checks = judgment['quality_checks']
    # Repeated strong scene instability is a hard failure, including a boolean
    # accidentally missed by the VLM. Ordinary motion/lighting is severity zero.
    if checks['strong_background_flicker'] or assessment['scene_continuity']['severity'] >= 3:
        caps.append('strong_scene_discontinuity')
    if task_id == 'P2' and checks['multiple_balls']:
        caps.append('multiple_balls')
    if task_id == 'P3' and (checks['poor_visual_quality'] or assessment['rendering_clarity']['severity'] >= 3):
        caps.append('poor_visual_quality')
    visual_score = min(uncapped, .49) if caps else uncapped
    task_ceilings = {name: TASK_SCORE_CEILINGS[d['severity']]
                     for name, d in judgment['task_assessment'].items()}
    score = min(visual_score, *task_ceilings.values())
    caps.extend(name for name, d in judgment['task_assessment'].items() if d['severity'] >= 3)
    issues = [name + ': ' + e['description'] for name, d in assessment.items()
              if d['severity'] > 0 for e in d['evidence']]
    issues.extend(name + ': ' + e['description'] for name, d in judgment['task_assessment'].items()
                  if d['severity'] > 0 for e in d['evidence'])
    result = dict(judgment, score=score, issues=issues,
                  scoring={'method': 'visual_deductions_and_task_evidence_ceilings_v2',
                           'severity_penalties': SEVERITY_PENALTIES, 'dimension_deductions': penalties,
                           'score_before_cap': uncapped, 'visual_score': visual_score,
                           'task_score_ceilings': TASK_SCORE_CEILINGS, 'task_dimension_ceilings': task_ceilings,
                           'hard_rejection_cap': min(.49, *task_ceilings.values()) if caps else None,
                           'distribution_normalized': False})
    if caps:
        result.update(forced_rejection=True, rejection_checks=list(dict.fromkeys(caps)))
    return result


@dataclass(frozen=True)
class Settings:
    backend: str = 'local'
    model: str = str(DEFAULT_MODEL)
    device: str = 'cuda:0'
    threshold: float = 0.8
    frames: int = 24
    max_edge: int = 640
    max_new_tokens: int = 3072
    base_url: str = ''
    api_key_env: str = 'VLM_API_KEY'
    timeout: float = 120.0
    input_mode: str = 'images'

    def validate(self):
        if self.backend not in ('local', 'http', 'vllm'):
            raise ValueError('consistency backend must be local, http or vllm')
        if self.input_mode not in ('images', 'native_video', 'static_sheets', 'static_frames'):
            raise ValueError('consistency input mode must be images, native_video, static_sheets or static_frames')
        if self.input_mode == 'native_video' and self.backend not in ('local', 'vllm'):
            raise ValueError('Native video input requires a local or vllm backend')
        if self.backend == 'vllm' and (self.input_mode not in ('native_video', 'static_sheets', 'static_frames') or self.frames != 24
                                      or self.max_edge != 640 or self.device != 'cuda:0'):
            raise ValueError('Audited vllm backend requires native_video/static_sheets, 24 frames, max_edge 640 and cuda:0')
        if self.input_mode in ('static_sheets', 'static_frames') and (self.backend != 'vllm' or self.max_new_tokens != 3072):
            raise ValueError('Static sheets require vllm and the frozen 3072-token grade budget')
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('consistency threshold must be finite and in [0,1]')
        if not 2 <= self.frames <= 64 or not 112 <= self.max_edge <= 2048:
            raise ValueError('consistency frames must be 2..64 and max edge 112..2048')
        if not 32 <= self.max_new_tokens <= 4096 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('invalid consistency token limit or timeout')
        if self.backend == 'http' and not self.base_url.startswith(('http://', 'https://')):
            raise ValueError('HTTP consistency backend requires --consistency-base-url')


def add_arguments(parser):
    group = parser.add_argument_group('V3 consistency gate (always runs before physics)')
    group.add_argument('--consistency-backend', choices=['local', 'http', 'vllm'], default=os.getenv('VLM_BACKEND', 'local'))
    group.add_argument('--consistency-model', default=os.getenv('VLM_MODEL', str(DEFAULT_MODEL)))
    group.add_argument('--consistency-device', default=os.getenv('VLM_DEVICE', 'cuda:0'))
    group.add_argument('--consistency-threshold', type=float, default=0.8)
    group.add_argument('--consistency-frames', type=int, default=24)
    group.add_argument('--consistency-max-edge', type=int, default=640)
    group.add_argument('--consistency-max-new-tokens', type=int, default=3072)
    group.add_argument('--consistency-base-url', default=os.getenv('VLM_BASE_URL', ''))
    group.add_argument('--consistency-api-key-env', default='VLM_API_KEY')
    group.add_argument('--consistency-timeout', type=float, default=120.0)
    group.add_argument('--consistency-input-mode', choices=['images', 'native_video', 'static_sheets', 'static_frames'], default='images')


def settings_from_args(args):
    return Settings(**{name: getattr(args, 'consistency_' + name) for name in Settings.__dataclass_fields__})


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _image(path, max_edge):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('RGB')
        im.thumbnail((max_edge, max_edge))
        return im.copy()


def _legacy_sample_frames(video, reference, directory, settings):
    import cv2
    from PIL import Image, ImageDraw
    frame_dir = directory / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    items = []
    if reference:
        path = frame_dir / 'reference.jpg'
        _image(reference, settings.max_edge).save(path, quality=92)
        items.append({'path': str(path), 'label': 'Reference image of the intended setup', 'role': 'reference'})
    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot decode video frame count/FPS for consistency evaluation')
        indices = sorted({round(i * (count - 1) / (settings.frames - 1)) for i in range(settings.frames)})
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode requested consistency frame {index}')
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((settings.max_edge, settings.max_edge))
            path = frame_dir / f'frame_{index:06d}.jpg'
            im.save(path, quality=92)
            items.append({'path': str(path), 'role': 'video', 'frame_index': index,
                          'fps': fps, 'total_num_frames': count,
                          'timestamp_seconds': index / fps, 'label': f'Video frame at {index / fps:.3f} seconds'})
    finally:
        cap.release()
    # A simultaneous temporal overview complements individual chronological
    # frames. In particular, small cumulative motion is easily missed when
    # each similar-looking frame is considered in isolation.
    video_items = [item for item in items if item['role'] == 'video']
    selected = [video_items[round(i * (len(video_items) - 1) / 5)] for i in range(6)]
    # Retain each sampled frame's full available 640-pixel detail. Tiny working
    # parts must not be judged solely from the previous 360-pixel thumbnails.
    tiles = [_image(item['path'], 640) for item in selected]
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles) + 32
    overview = Image.new('RGB', (3 * tile_width, 2 * tile_height), 'white')
    draw = ImageDraw.Draw(overview)
    for index, (item, tile) in enumerate(zip(selected, tiles)):
        x, y = (index % 3) * tile_width, (index // 3) * tile_height
        overview.paste(tile, (x, y))
        draw.text((x + 4, y + tile_height - 24), f"{item['timestamp_seconds']:.3f} s", fill='black')
    overview_path = frame_dir / 'temporal_overview.jpg'
    overview.save(overview_path, quality=92)
    insert_at = 1 if reference else 0
    items.insert(insert_at, {'path': str(overview_path), 'role': 'overview', 'max_edge': max(overview.size),
                            'label': 'Same video, six chronological frames: left to right, then next row. Compare subject position relative to fixed apparatus.'})
    return items


def _legacy_observation_messages_for(items):
    content = [{'type': 'text', 'text': 'Describe the visible sequence before interpreting its purpose.'}]
    for item in items:
        if item.get('role') not in ('video', 'overview'):
            continue
        content.extend([{'type': 'text', 'text': item['label']},
                        {'type': 'image', 'image': item['path']}])
    return [{'role': 'system', 'content': OBSERVATION_RUBRIC}, {'role': 'user', 'content': content}]


def parse_observation(text):
    if not isinstance(text, str):
        raise ValueError('Observation must be JSON text')
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    keys = {'beginning', 'middle', 'end', 'changes'}
    if (not isinstance(value, dict) or set(value) != keys
            or any(not isinstance(v, str) or not v.strip() for v in value.values())):
        raise ValueError('Observation needs exactly beginning, middle, end, changes as nonempty strings')
    return value


def task_contract_for(task_id):
    """Return an independent copy; unknown tasks must never fall back to raw prompts."""
    if task_id not in TASK_CONTRACTS:
        raise ValueError('No audited observability contract for task ' + str(task_id))
    return json.loads(json.dumps({'shared_rules': TASK_SHARED_RULES, 'contract': TASK_CONTRACTS[task_id]}))


def task_contract_text(task_id):
    return json.dumps(task_contract_for(task_id), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def task_input_audit(task_id, prompt, reference):
    return {'generation_prompt': prompt, 'generation_prompt_sent_to_vlm': False,
            'task_contract_sent_to_vlm': True, 'task_contract_version': TASK_CONTRACT_VERSION,
            'task_contract_sha256': hashlib.sha256(task_contract_text(task_id).encode()).hexdigest(),
            'task_contract': task_contract_for(task_id), 'reference_sent_to_vlm': bool(reference)}


def _legacy_messages_for(items, task_id, prompt, observation=None):
    # The original generation prompt is preserved in request.json for association
    # audits, but never sent to the judge or independent observer.
    content = [{'type': 'text', 'text':
        'Inspect every supplied frame for temporal visual coherence, task presentation and readability. '
        'Compare adjacent samples and the first, middle and last frames. Track foreground '
        'identity, rigid structure, background/exposure continuity and the actual event stages. '
        'Task: ' + str(task_id) + '. Compare reference only for essential experimental setup. '
        'Physical correctness is not judged. Neutral task observability contract: ' + task_contract_text(task_id)}]
    if observation is not None:
        content.insert(0, {'type': 'text', 'text':
            'Independent visual observation recorded before the task was supplied '
            '(fallible evidence; verify against images): ' + json.dumps(observation, ensure_ascii=False)})
    for item in items:
        content.extend([{'type': 'text', 'text': item['label']},
                        {'type': 'image', 'image': item['path']}])
    content.append({'type': 'text', 'text':
        'Return one flat JSON object containing reason, the seven named dimensions and '
        'the three named booleans, with no grouping wrappers. Every task dimension needs '
        'timestamp evidence even at severity zero: identify the core experimental '
        'observations actually available, not just a recognizable stable scene. Look for '
        'unexplained count/identity changes, rigid deformation and unreadable rendering. '
        'Explain observed occlusion or camera motion when relevant. Do not excuse defects '
        'just because the main object is recognizable. Missing core phenomenon or unreadable '
        'main subject is a task failure even with perfect backgrounds. Do not invent '
        'defects or completion of unobserved stages. Base missing-stage claims on the actual '
        'sequence and acknowledge uncertainty from sampling gaps. Do not grade physical laws.'})
    return [{'role': 'system', 'content': rubric_for(task_id)}, {'role': 'user', 'content': content}]


@lru_cache(maxsize=1)
def _local_model(model_path, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    gpu = device.startswith('cuda') or device == 'auto'
    if gpu and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for the consistency model; configure a GPU or HTTP VLM backend')
    # On this torch/H100 stack cuDNN repeatedly builds attention plans for new
    # KV lengths (~0.4 s per new length). The audited probe selects native Flash
    # when cuDNN SDPA is disabled; efficient/math remain available for masks.
    # This is an inference policy change, recorded by the frozen source hash.
    if gpu:
        torch.backends.cuda.enable_cudnn_sdp(False)
    # Runtime never silently downloads a different checkpoint.
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    placement = {'device_map': device}
    if device == 'auto':
        placement['max_memory'] = {index: '36GiB' for index in range(torch.cuda.device_count())}
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        attn_implementation='sdpa', **placement).eval()
    if device == 'auto' and any(str(value) in ('cpu', 'disk') for value in model.hf_device_map.values()):
        raise RuntimeError('Model parallel inference requires all weights on the selected GPUs')
    return processor, model


def native_video_messages(messages, items):
    """Present the same audited frames as one video, retaining static images."""
    video_items = [item for item in items if item.get('role') == 'video']
    if not video_items:
        return messages
    paths = {item['path'] for item in video_items}
    labels = {item['label'] for item in video_items}
    converted = json.loads(json.dumps(messages))
    # Requests are stored in their actual form; the reply helper is idempotent.
    if any(part.get('type') == 'video' for message in converted
           if isinstance(message['content'], list) for part in message['content']):
        return converted
    inserted = False
    for message in converted:
        if not isinstance(message['content'], list):
            continue
        content = []
        for part in message['content']:
            if part.get('type') == 'text' and part.get('text') in labels:
                continue
            if part.get('type') == 'image' and part.get('image') in paths:
                if not inserted:
                    content.extend([
                        {'type': 'text', 'text': 'Chronological video samples at these exact seconds: ' +
                         ', '.join(f"{item['timestamp_seconds']:.3f}" for item in video_items)},
                        {'type': 'video', 'video': [item['path'] for item in video_items]}])
                    inserted = True
                continue
            content.append(part)
        message['content'] = content
    if not inserted:
        raise ValueError('Native video request has no associated frame placeholders')
    return converted


def _local_reply(messages, items, settings):
    import torch
    processor, model = _local_model(settings.model, settings.device)
    native = settings.input_mode == 'native_video' and any(item.get('role') == 'video' for item in items)
    if native:
        messages = native_video_messages(messages, items)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    image_items = [item for item in items if not native or item.get('role') != 'video']
    images = [_image(item['path'], item.get('max_edge', settings.max_edge)) for item in image_items]
    extra = {}
    if native:
        import numpy as np
        video_items = [item for item in items if item.get('role') == 'video']
        fps, count = video_items[0]['fps'], video_items[0]['total_num_frames']
        if any(item['fps'] != fps or item['total_num_frames'] != count for item in video_items):
            raise ValueError('Native video frame metadata is inconsistent')
        extra = {'videos': [np.stack([np.asarray(_image(item['path'], settings.max_edge)) for item in video_items])],
                 'video_metadata': [{'total_num_frames': count, 'fps': fps,
                                     'frames_indices': [item['frame_index'] for item in video_items]}],
                 'do_sample_frames': False}
    inputs = processor(text=[text], images=images or None, padding=True, return_tensors='pt', **extra).to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, do_sample=False)
    return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


VLLM_ENGINE_POLICY = {
    'dtype': 'bfloat16', 'trust_remote_code': False, 'tensor_parallel_size': 1,
    'gpu_memory_utilization': .90, 'max_model_len': 12288,
    'gdn_prefill_backend': 'triton', 'max_num_seqs': 4,
    'max_num_batched_tokens': 12288, 'enable_prefix_caching': True,
    'limit_mm_per_prompt': {'image': {'count': 2, 'width': 1920, 'height': 1344},
                            'video': {'count': 1, 'num_frames': 24, 'width': 640, 'height': 640}},
    'structured_outputs_config': {'backend': 'xgrammar'},
    'mm_processor_kwargs': {'do_sample_frames': False},
}


def structured_output_schemas():
    """Constrain JSON structure; evidence and final scoring are still validated separately."""
    text = {'type': 'string', 'minLength': 1}
    evidence = {'type': 'object', 'additionalProperties': False,
                'properties': {'start_seconds': {'type': 'number', 'minimum': 0},
                               'end_seconds': {'type': 'number', 'minimum': 0}, 'description': text},
                'required': ['start_seconds', 'end_seconds', 'description']}
    properties = {'reason': text}
    for name in DIMENSIONS + TASK_DIMENSIONS:
        properties[name] = {'type': 'object', 'additionalProperties': False,
                            'properties': {'severity': {'type': 'integer', 'enum': [0, 1, 2, 3, 4]},
                                           'evidence': {'type': 'array', 'items': evidence,
                                                        'minItems': 1 if name in TASK_DIMENSIONS else 0}},
                            'required': ['severity', 'evidence']}
    properties.update({name: {'type': 'boolean'} for name in
                       ('multiple_balls', 'strong_background_flicker', 'poor_visual_quality')})
    return {'observation': {'type': 'object', 'additionalProperties': False,
                            'properties': {name: text for name in ('beginning', 'middle', 'end', 'changes')},
                            'required': ['beginning', 'middle', 'end', 'changes']},
            'grade': {'type': 'object', 'additionalProperties': False,
                      'properties': properties, 'required': list(properties)}}


def attention_policy_for(settings):
    return {'local': ATTENTION_POLICY, 'http': 'remote_backend',
            'vllm': 'vllm_flash_attention_gdn_triton_structured_json'}[settings.backend]


@lru_cache(maxsize=1)
def _legacy_vllm_runtime_provenance(settings):
    import importlib.metadata
    files = {
        'vllm': ['vllm/model_executor/models/qwen3_5.py', 'vllm/model_executor/models/qwen3_vl.py',
                 'vllm/model_executor/models/qwen3_next.py',
                 'vllm/model_executor/layers/mamba/gdn_linear_attn.py',
                 'vllm/v1/structured_output/backend_xgrammar.py', 'vllm/sampling_params.py'],
        'transformers': ['transformers/models/qwen3_vl/processing_qwen3_vl.py',
                         'transformers/models/qwen3_vl/video_processing_qwen3_vl.py'],
    }
    implementations = {}
    for package, relative_paths in files.items():
        distribution = importlib.metadata.distribution(package)
        if package == 'vllm':
            relative_paths = relative_paths + [str(path) for path in distribution.files
                if str(path).startswith('vllm/model_executor/layers/fla/ops/') and str(path).endswith('.py')]
        for relative in relative_paths:
            path = Path(distribution.locate_file(relative))
            # Record absence too: implementations may move between package versions.
            implementations[relative] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                          'bytes': path.stat().st_size} if path.is_file() else {'present': False}
    schemas = structured_output_schemas()
    return {'engine_policy': VLLM_ENGINE_POLICY,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('vllm', 'torch', 'transformers', 'xgrammar')},
            'implementation_files': implementations, 'enable_thinking': False,
            'sampling': {'temperature': 0., 'seed': 0},
            'json_schema_sha256': {name: hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                                   for name, schema in schemas.items()}}


@lru_cache(maxsize=1)
def _vllm_engine(model_path):
    if not Path(model_path).is_dir():
        raise ValueError('vllm requires an existing audited local model directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    from vllm import LLM
    from vllm.transformers_utils.config import get_config
    from transformers import AutoProcessor
    get_config(model_path, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    engine = LLM(model=model_path, **VLLM_ENGINE_POLICY)
    return processor, engine


def vllm_request(processor, record, settings):
    import numpy as np
    items = record['frames']
    messages = native_video_messages(record['messages'], items) if items else record['messages']
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    images = [_image(item['path'], item.get('max_edge', settings.max_edge))
              for item in items if item.get('role') != 'video']
    video = [item for item in items if item.get('role') == 'video']
    multimodal = {}
    if images:
        multimodal['image'] = images
    if video:
        fps, count = video[0]['fps'], video[0]['total_num_frames']
        if len(video) != settings.frames or any(item['fps'] != fps or item['total_num_frames'] != count for item in video):
            raise ValueError('Native video frame count or original metadata is inconsistent')
        metadata = {'fps': fps, 'total_num_frames': count, 'duration': count / fps,
                    'frames_indices': [item['frame_index'] for item in video], 'do_sample_frames': False}
        pixels = np.stack([np.asarray(_image(item['path'], settings.max_edge)) for item in video])
        multimodal['video'] = [(pixels, metadata)]
    request = {'prompt': text}
    if multimodal:
        request.update(multi_modal_data=multimodal, mm_processor_kwargs={'do_sample_frames': False})
    return request


def _legacy_vllm_replies(prepared, settings):
    settings.validate()
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    if not prepared:
        raise ValueError('Cannot infer an empty batch')
    phases = ['observation' if record['messages'][0]['content'] == OBSERVATION_RUBRIC else 'grade'
              for record in prepared]
    if len(set(phases)) != 1:
        raise ValueError('A batch must contain one inference phase')
    schema = structured_output_schemas()[phases[0]]
    started = time.monotonic()
    processor, engine = _vllm_engine(settings.model)
    load_seconds = time.monotonic() - started
    requests = [vllm_request(processor, record, settings) for record in prepared]
    parameters = SamplingParams(temperature=0., seed=0, max_tokens=settings.max_new_tokens,
                                structured_outputs=StructuredOutputsParams(json=schema))
    request_audit = [{'prompt': request['prompt'], 'frame_metadata': record['frames'],
                      'mm_processor_kwargs': request.get('mm_processor_kwargs'), 'json_schema': schema,
                      'max_tokens': settings.max_new_tokens, 'temperature': 0., 'seed': 0}
                     for record, request in zip(prepared, requests)]
    started = time.monotonic()
    responses = engine.generate(requests, parameters, use_tqdm=False)
    values = [{'content': response.outputs[0].text, 'prompt_tokens': len(response.prompt_token_ids),
               'output_tokens': len(response.outputs[0].token_ids),
               'finish_reason': response.outputs[0].finish_reason} for response in responses]
    if len(values) != len(prepared):
        raise RuntimeError('vllm response count differs from request count')
    return [value['content'] for value in values], {
        'generate_seconds': time.monotonic() - started, 'engine_load_seconds': load_seconds,
        'phase': phases[0], 'request_audit': request_audit, 'response_audit': values,
        'output_tokens': [value['output_tokens'] for value in values]}


def _legacy_vllm_reply(messages, items, settings):
    return vllm_replies([{'messages': messages, 'frames': items}], settings)[0][0]


def reply_for(settings):
    return {'local': _local_reply, 'http': _http_reply, 'vllm': _vllm_reply}[settings.backend]


def _http_reply(messages, items, settings):
    messages = json.loads(json.dumps(messages))
    for message in messages:
        if not isinstance(message['content'], list):
            continue
        for i, part in enumerate(message['content']):
            if part['type'] == 'image':
                data = base64.b64encode(Path(part['image']).read_bytes()).decode()
                message['content'][i] = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}}
    payload = {'model': settings.model, 'messages': messages, 'temperature': 0,
               'max_tokens': settings.max_new_tokens}
    endpoint = settings.base_url.rstrip('/')
    if not endpoint.endswith('/chat/completions'):
        endpoint += '/chat/completions'
    headers = {'Content-Type': 'application/json'}
    api_key = os.getenv(settings.api_key_env)
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=settings.timeout) as response:
        result = json.load(response)
    return result['choices'][0]['message']['content']


def _unique_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def parse_judgment(text):
    if not isinstance(text, str):
        raise ValueError('VLM response content must be text')
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text, object_pairs_hook=_unique_json_keys)
    if not isinstance(value, dict):
        raise ValueError('VLM response must be a JSON object')
    checks = ('multiple_balls', 'strong_background_flicker', 'poor_visual_quality')
    if set(value) & {*DIMENSIONS, *TASK_DIMENSIONS, *checks}:
        expected = {'reason', *DIMENSIONS, *TASK_DIMENSIONS, *checks}
        if set(value) != expected:
            raise ValueError('Flat judgment must contain exactly reason, seven dimensions and three booleans')
        value = {'reason': value['reason'],
                 'assessment': {name: value[name] for name in DIMENSIONS},
                 'task_assessment': {name: value[name] for name in TASK_DIMENSIONS},
                 'quality_checks': {name: value[name] for name in checks}}
    if 'assessment' in value:
        _validate_assessment(value)
        # Historical four-dimension replies remain readable; fresh scoring
        # requires the expanded task evidence and never silently grants a pass.
        if 'task_assessment' in value:
            _validate_task_assessment(value)
        return {key: value[key] for key in ('reason', 'assessment', 'quality_checks', 'task_assessment') if key in value}
    # Old stored replies remain readable; fresh evaluation below requires the
    # current structured assessment and never falls back to the old score.
    score = value.get('score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('VLM score must be a finite number in [0,1]')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('VLM response must include a reason')
    if not isinstance(value.get('issues'), list) or not all(isinstance(v, str) for v in value['issues']):
        raise ValueError('VLM issues must be a list of strings')
    return {'score': float(score), 'reason': value['reason'], 'issues': value['issues'], **({'quality_checks':value['quality_checks']} if 'quality_checks' in value else {})}


def _json_scalar_tokens(text):
    """A format correction may move punctuation, but cannot change any values."""
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    pattern = re.compile(r'\s+|[{}\[\],:]|"(?:[^"\\]|\\.)*"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null')
    position, scalars = 0, []
    while position < len(text):
        match = pattern.match(text, position)
        if match is None:
            raise ValueError('Cannot verify scalar preservation of malformed JSON')
        token = match.group()
        position = match.end()
        if token.isspace() or token in '{}[],:':
            continue
        # Preserve numeric spellings and types; decode escaped strings only.
        scalars.append(('string', json.loads(token)) if token.startswith('"') else ('literal', token))
    return scalars


def final_judgment_with_retry(raw, messages, items, settings, directory, reply_fn=None):
    """One syntax-only retry; preserve every original key and scalar in order.

    Schema, evidence, transport and inference failures remain errors. This cannot
    add absent fields, revise severities or invent evidence from a truncated reply.
    """
    directory = Path(directory)
    try:
        return parse_judgment(raw), {'response_attempts': 1, 'accepted_response_artifact': 'response.json'}
    except json.JSONDecodeError as error:
        original_tokens = _json_scalar_tokens(raw)
        repair_messages = [
            {'role': 'system', 'content':
             'Repair JSON punctuation only. Preserve EVERY key and scalar value in exactly '
             'the original order. Do not change, add, delete or reinterpret any string, '
             'number, boolean or null. You may correct braces, brackets, commas, colons '
             'and whitespace. Return ONLY the corrected JSON. Do not grade anything.'},
            {'role': 'user', 'content': 'Malformed JSON to repair:\n' + raw}]
        _write(directory / 'format_retry_request.json', {'messages': repair_messages,
               'initial_parse_error': str(error), 'images_sent': False,
               'policy': 'one_syntax_only_retry_with_exact_scalar_preservation'})
        reply = reply_fn or reply_for(settings)
        repaired = reply(repair_messages, [], settings)
        _write(directory / 'format_retry_response.json', {'content': repaired})
        if _json_scalar_tokens(repaired) != original_tokens:
            raise ValueError('JSON format retry changed keys or values; response rejected')
        judgment = parse_judgment(repaired)
        return judgment, {'response_attempts': 2, 'initial_parse_error': str(error),
                          'accepted_response_artifact': 'format_retry_response.json',
                          'format_retry_scalar_preservation_verified': True}


def task_prompt_for_video(prompt, video):
    """Accept plain prompts or select exactly one audited generation manifest row.

    Some historical inputs use a whole manifest as prompt_source. Never send
    its unrelated tasks, model names or generation commands to the judge.
    """
    text = prompt.strip()
    if not text:
        raise ValueError('Task-observability evaluation requires a generation prompt')
    if text.startswith(('{', '[')):
        manifest = json.loads(text)
        sample_id = Path(video).stem
        entry = manifest.get(sample_id) if isinstance(manifest, dict) else None
        if (entry is None and isinstance(manifest, dict)
                and Path(str(manifest.get('output_path', ''))).stem == sample_id):
            entry = manifest.get('arguments')
        if not isinstance(entry, dict) or not isinstance(entry.get('prompt'), str) or not entry['prompt'].strip():
            raise ValueError('Cannot associate generation manifest prompt with video ' + sample_id)
        return entry['prompt'].strip()
    return text


def _legacy_evaluate(video, reference, prompt, task_id, debug, settings):
    """Return a fresh decision; transport/model errors are never passes or zero scores."""
    started = time.monotonic()
    directory = Path(debug) / 'consistency'
    directory.mkdir(parents=True, exist_ok=True)
    result = {'rubric_version': RUBRIC_VERSION, 'rubric_sha256': hashlib.sha256(rubric_for(task_id).encode()).hexdigest(),
              'backend': settings.backend, 'model': settings.model, 'device': settings.device,
              'input_mode': settings.input_mode,
              'attention_policy': attention_policy_for(settings),
              'threshold': settings.threshold, 'score': None, 'passed': None, 'status': 'error',
              'artifact_directory': str(directory.resolve())}
    try:
        settings.validate()
        prompt = task_prompt_for_video(prompt, video)
        task_audit = task_input_audit(task_id, prompt, reference)
        result.update({key: task_audit[key] for key in ('task_contract_version', 'task_contract_sha256')})
        items = sample_frames(video, reference, directory, settings)
        result['frames'] = items
        reply = reply_for(settings)
        if settings.backend == 'vllm':
            result['vllm_runtime'] = vllm_runtime_provenance(settings)
            call_number = 0
            def reply(messages, frames, selected_settings):
                nonlocal call_number
                values, timings = vllm_replies([{'messages': messages, 'frames': frames}], selected_settings)
                _write(directory / f'backend_call{call_number:02d}.json', timings)
                call_number += 1
                return values[0]
        observation_items = [item for item in items if item.get('role') in ('video', 'overview')]
        observation_messages = observation_messages_for(observation_items)
        if settings.input_mode == 'native_video':
            observation_messages = native_video_messages(observation_messages, observation_items)
        _write(directory / 'observation_request.json', {'messages': observation_messages,
               'generation_prompt_sent_to_vlm': False, 'reference_sent_to_vlm': False,
               'max_new_tokens': 1024})
        observation_raw = reply(observation_messages, observation_items, replace(settings, max_new_tokens=1024))
        _write(directory / 'observation_response.json', {'content': observation_raw})
        observation = parse_observation(observation_raw)
        result['independent_observation'] = observation
        result['observation_rubric_sha256'] = hashlib.sha256(OBSERVATION_RUBRIC.encode()).hexdigest()
        messages = messages_for(items, task_id, prompt, observation)
        if settings.input_mode == 'native_video':
            messages = native_video_messages(messages, items)
        # Store readable text and local frame paths, never credentials or data URLs.
        _write(directory / 'request.json', {'messages': messages, 'rubric_version': RUBRIC_VERSION,
            **task_audit})
        raw = reply(messages, items, settings)
        _write(directory / 'response.json', {'content': raw})
        parsed, retry_metadata = final_judgment_with_retry(raw, messages, items, settings, directory, reply)
        result.update(retry_metadata)
        judgment = score_assessment(parsed, task_id, items)
        result.update(judgment, passed=judgment['score'] >= settings.threshold and not judgment.get('forced_rejection', False), status='evaluated')
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}', issues=[])
    result['elapsed_seconds'] = time.monotonic() - started
    _write(directory / 'consistency.json', result)
    return result


import hashlib

import io

import json

import math

from pathlib import Path

V24_ROI_CONFIG = {'reference_sha256': 'f24300c837f959ebdcabe54f20bdac487d6cc80d02e4e761ca42210bfa180e9d',
 'reference_path': '/mnt/einsia/aws01-nvme/einsia-shared/homes/linxinjie/vdmbench/data/g1_g9/g8/P34/first_frame.png',
 'reference_dimensions_wh': [1659, 948],
 'coordinate_basis_wh': [640, 366],
 'regions': [{'name': 'Region A',
              'box_xyxy': [100, 96, 260, 256],
              'normalized_bbox_xyxy': [0.15625, 0.26229508196721313, 0.40625, 0.6994535519125683],
              'reference_subject': 'left dial including outer rim and surrounding board'},
             {'name': 'Region B',
              'box_xyxy': [384, 100, 544, 260],
              'normalized_bbox_xyxy': [0.6, 0.273224043715847, 0.85, 0.7103825136612022],
              'reference_subject': 'right dial including outer rim and surrounding board'}],
 'font_path': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
 'font_sha256': '690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648',
 'font_size_px': 24,
 'resize_policy': 'Crop original source frames at unchanged24 indices; floor(left/top),ceil(right/bottom); '
                  'LANCZOS aspect-preserving scale with white padding to320x320. No content-adaptive crop or '
                  'extra timestamps.',
 'reference_localization': 'Reference-only circle outer-rim proposals, visually confirmed before any '
                           'frame-specific crop; one shared fixed normalized ROI pair.',
 'circle_proposals': {'left': [179.5, 176.5, 43.6], 'right': [463.5, 179.5, 44.4]},
 'circle_proposal_parameters': 'OpenCV grayscale GaussianBlur5x5 sigma1, '
                               'HoughCircles(dp1,minDist120,param1=100,param2=30,minRadius25,maxRadius65). '
                               'Centers rounded to integers; common80px half-side, approximately1.8times '
                               'largest rim radius. Proposals unchanged at param2=25/35.',
 'scope_key': 'The shared reference SHA only; never generator name, predicted score, extracted physics, or '
              'defect timestamp.',
 'shared_reference_verification': {'manifest_path': '/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/evaluator/videos_all/v4_evaluator/opinion_v2_20261002/input_manifest.json',
                                   'manifest_sha256': '6d1a9c2ec66d37ea5c6c343b282bde1bd92b45ba1d6f810f1a40e37d99f0c708',
                                   'P34_input_count': 32,
                                   'model_count': 8,
                                   'unique_reference_hashes': ['f24300c837f959ebdcabe54f20bdac487d6cc80d02e4e761ca42210bfa180e9d']},
 'visual_boundary_review': {'videos_checked': 2,
                            'same24_sample_frames_per_video': 24,
                            'detail_crops_actually_viewed': 96,
                            'all_outer_rims_complete': True,
                            'all_crops_have_surrounding_board_context': True,
                            'camera_drift_or_geometry_change_exits_crop': False,
                            'scope_limit': 'Only these48 sampled source frames actually reviewed. This does '
                                           'not prove all P34 videos remain registered.',
                            'cases': [{'identity': {'model': 'cogvideox1.5-5b-i2v',
                                                    'task': 'P34',
                                                    'sample_id': 'g8_P34_seed43'},
                                       'source_dimensions_wh': [832, 480],
                                       'source_fps': 16.0,
                                       'source_video_sha256': '0fcaabd83528f5ebe829d90229a567631aaadeea6c544f2251e8b229c4273e03',
                                       'mapped_source_boxes': [[130, 125, 338, 336], [499, 131, 708, 341]],
                                       'sample_indices': [0,
                                                          3,
                                                          7,
                                                          10,
                                                          14,
                                                          17,
                                                          21,
                                                          24,
                                                          28,
                                                          31,
                                                          35,
                                                          38,
                                                          42,
                                                          45,
                                                          49,
                                                          52,
                                                          56,
                                                          59,
                                                          63,
                                                          66,
                                                          70,
                                                          73,
                                                          77,
                                                          80]},
                                      {'identity': {'model': 'minimax-h3',
                                                    'task': 'P34',
                                                    'sample_id': 'g8_P34_seed43'},
                                       'source_dimensions_wh': [1344, 768],
                                       'source_fps': 24.0,
                                       'source_video_sha256': '5749fe1dc5d01eabec55f26b284975bc5fbeef5ad0df7d4c3bc3090887812b0b',
                                       'mapped_source_boxes': [[210, 201, 546, 538], [806, 209, 1143, 546]],
                                       'sample_indices': [0,
                                                          5,
                                                          11,
                                                          16,
                                                          21,
                                                          27,
                                                          32,
                                                          37,
                                                          43,
                                                          48,
                                                          53,
                                                          59,
                                                          64,
                                                          70,
                                                          75,
                                                          80,
                                                          86,
                                                          91,
                                                          96,
                                                          102,
                                                          107,
                                                          112,
                                                          118,
                                                          123]}]},
 'interpretation_boundary': 'Always retain full-frame sheets/native context. Blank, clipped, off-screen, '
                            'shifted or unregistered crop is unavailable auxiliary evidence, not proof an '
                            'object/part disappeared. Cropping changes visibility only, never the scoring '
                            'rubric or thresholds.',
 'evidence_directory': '/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/evaluator/videos_all/v4_evaluator/consistency_observability_20261003/p34_reference_crop_study',
 'frozen_before_R18_results': True}

V24_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

V24_FONT_SHA256 = '690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648'

V24_LOCAL_PROMPT = ('Describe only the visible pixels in this single image. It contains two\n'
 'fixed-region crops from ONE source frame, Region A at left and Region B at right.\n'
 'For each region describe the central hub and any elongated colored or dark parts\n'
 'visibly extending from it, including their shape, color, connections and ends.\n'
 'Distinguish the rim/tick marks from parts attached to the hub. If only a spot or\n'
 'uncertain trace is visible, describe that actual appearance. Do not invent a\n'
 'direction, motion, history, function or score. This is a single instant with no\n'
 'earlier/later image. Return JSON with region_a and region_b; each has description\n'
 'and uncertainty as nonempty strings, and extended_parts as a list of strings\n'
 'describing ONLY the parts you can actually locate. An empty list is valid.')

V24_LOCAL_SCHEMA = {'type': 'object',
 'additionalProperties': False,
 'properties': {'region_a': {'type': 'object',
                             'additionalProperties': False,
                             'properties': {'description': {'type': 'string', 'minLength': 1},
                                            'uncertainty': {'type': 'string', 'minLength': 1},
                                            'extended_parts': {'type': 'array',
                                                               'items': {'type': 'string', 'minLength': 1}}},
                             'required': ['description', 'uncertainty', 'extended_parts']},
                'region_b': {'type': 'object',
                             'additionalProperties': False,
                             'properties': {'description': {'type': 'string', 'minLength': 1},
                                            'uncertainty': {'type': 'string', 'minLength': 1},
                                            'extended_parts': {'type': 'array',
                                                               'items': {'type': 'string', 'minLength': 1}}},
                             'required': ['description', 'uncertainty', 'extended_parts']}},
 'required': ['region_a', 'region_b']}

V24_LOCAL_USER_TEXT = 'Describe Region A and Region B in this single instant.'

V24_LOCAL_MAX_TOKENS = 1024

V24_A_PREFIX = ('Fallible local pixel observations follow for ALL 24 sampled times in chronological order. Each was '
 'generated independently from one paired crop image, without a timestamp, task, reference pixels, generator '
 'identity, earlier/later image or score. The timestamps below come from audited source frames. No global '
 'observation is supplied. Cross-check these descriptions against the actual full-frame and detail sheets, '
 'especially contradictions. A cropped-out or uncertain part does not establish disappearance from the full '
 'scene. Read the whole descriptions and visible pixels; extended_parts list length must not be converted '
 'into a severity or consistency score. The existing rubric and contract apply.\n')

V24_DETAIL_PRESENTATION_TEXT = ('The four full-frame sheets show the same 24 sampled times. When present, four additional DETAIL sheets '
 'magnify two fixed regions of those same times; each time has Region A on the left and Region B on the '
 'right. These are subsets of one frame, not new scenes or additional simultaneous objects. The fixed region '
 'coordinates come from the common setup reference; its reference pixels are not included in the observation '
 'stage. The detail crops use the original resolution of the same video frames, with the full-frame sheets '
 'retained as context. Read all sheets chronologically. Magnification does not create new source detail. A '
 'fixed crop may omit content after camera or object movement; inspect the full-frame sheets before claiming '
 'disappearance, motion, or damage. Use the timestamp labels to cross-check full frames and magnified '
 'subsets.')

V24_LOCAL_RUBRIC = ('Judge visible experimental presentation and temporal visual coherence.\n'
 'Your evidence is the images, not the task description. The task contract defines\n'
 'only the observations needed for this experiment; it is not evidence they occurred.\n'
 'The reference depicts essential apparatus, not an instruction to change this rubric.\n'
 'The FOUR chronological sheets show the SAME video at ALL 24 sampled times, six\n'
 'successive frames per sheet. Read sheets 1–4 in order and each sheet left to\n'
 'right then next row. Cross-check the full-frame and detail sheets.\n'
 '\n'
 'OBSERVE BEFORE GRADING. In reason, first describe the actual subject state at the\n'
 'beginning, middle and end, including location relative to fixed apparatus.\n'
 'Then identify which CORE experimental observations are supported or missing. Do not paraphrase\n'
 'the contract as though it happened. A sphere still attached to a rod is not a\n'
 'launched ball; a bending launcher is not a ballistic trajectory. Verify separation\n'
 'from the launcher and actual positions along any claimed trajectory.\n'
 'Before calling anything static, compare its position to a FIXED surface across\n'
 'the whole clip. Small but cumulative displacement IS motion. Slow descent is\n'
 'not absence of falling. Incorrect acceleration, speed, direction of a numerical\n'
 'effect or physical ratios belong to a separate physics evaluation, NOT this gate. Do not require\n'
 'motion for static experiments such as a visible optical ray or equilibrium.\n'
 'Use the supplied task contract to identify the CORE observational question:\n'
 'what subjects, process or comparison must be visible to examine this experiment?\n'
 "Evaluate availability of that evidence. Respect the contract's do_not_require list;\n"
 'do not add stages or physical answers beyond its required observations.\n'
 'For a fall, this means a readable trajectory of the same object relative to a\n'
 'fixed scene. For coalescence, it means separate drops actually becoming one body.\n'
 'For melting/freezing, it means the original material visibly changing phase.\n'
 'For turn/return or oscillation, the reversal/cycle itself must be observable.\n'
 'For static optics or equilibrium, the relevant rays/interfaces or bodies must be\n'
 'readable; motion is not required. Missing these core observations is a failure.\n'
 'Timing, camera style, requested pauses and incidental extra action cannot erase\n'
 'core evidence that is clearly present. An unrequired ending is not a missing\n'
 'observation when all contract requirements remain fully observable.\n'
 'Thus a clear readable fall is available even if it eventually reaches a surface;\n'
 'contact is not a missing fall. Conversely, separate drops merely approaching\n'
 'never supply evidence of merging. Decide from what IS visible, not a predicted\n'
 'future. A shadow alone does not prove contact. Do not invent impacts or rest.\n'
 'For any task severity 2+, name the unavailable CORE observation and explain why\n'
 'the visible sequence is insufficient to examine it. An unrequired detail\n'
 'by itself is not such an explanation.\n'
 '\n'
 'TASK ASSESSMENT: rate these three dimensions using the TASK scale below:\n'
 'setup_fidelity: essential subjects, count, material, connections and fixed\n'
 'apparatus preserve the intended experiment. Cosmetic changes do not matter.\n'
 'event_coverage: availability of the core phenomenon and its essential phases or\n'
 'comparisons. A trajectory must actually move; a transformation must actually\n'
 'transform. Smooth unrelated motion and plausible-looking objects do not establish\n'
 'the phenomenon. Distinguish unavailable observations from visible behavior whose\n'
 'quantitative physics is wrong. Only treat a final state as essential when it is\n'
 'needed to observe the phenomenon itself, such as merged drops or a phase change.\n'
 'observation_readability: the MAIN subject, interfaces, pointers, rays or contact\n'
 'regions remain readable during critical phases and necessary before/after views.\n'
 'Clean background and apparatus cannot substitute for unreadable measured objects.\n'
 '\n'
 'TASK severity / meaning:\n'
 '0: complete and clearly supported; cite positive timestamp evidence.\n'
 '1: small omission/ambiguity, but all core subjects/stages remain available.\n'
 '2: substantial partial presentation; some of the core phenomenon is visible but\n'
 '   an essential comparison or phase needed to examine it is incomplete/ambiguous.\n'
 '3: critical stage, main subject or observation absent/unreadable, or a major setup\n'
 '   mismatch prevents following the experiment.\n'
 '4: core event never happens or is replaced by another action, or experiment/main\n'
 '   subject is absent or effectively unreadable throughout.\n'
 'ALL THREE task dimensions require nonempty timestamp evidence EVEN at severity 0.\n'
 'For event_coverage, describe the actual beginning, middle and end observations;\n'
 'when claiming an event never occurs, specify what is seen instead over the clip.\n'
 'Do not guess missing events in gaps between sampled frames. State uncertainty\n'
 'and partial presentation instead of inventing completion or definite corruption.\n'
 '\n'
 'Examples: two drops stay separate instead of merging; intact ice sinks instead\n'
 'of melting; a block exits before a required turn/return; a ball falls off a ramp\n'
 'instead of rolling along it. These are absent/replaced events, even if smooth.\n'
 'A surface rising by the wrong amount after visible melting is a PHYSICS issue.\n'
 'A clear complete flight at the wrong acceleration is still a presented flight.\n'
 'Compass needles unreadable at the final state, or main bodies repeatedly lost\n'
 'while supports stay sharp, fail observation_readability. Ordinary brief motion\n'
 'blur is fine when identity and stages are still followable. A sharp stand with\n'
 'no locatable suspended body is not readable motion. Repeatedly absent outlines\n'
 'are not excused by calling them blur. For pointers, describe the actual pointer\n'
 'body and orientation, not just the readable housing or a remaining central spot.\n'
 'Never use extractor\n'
 'failure codes, fitting thresholds or hidden physical quantities to grade images.\n'
 '\n'
 'VISUAL ASSESSMENT: independently inspect four visual dimensions:\n'
 'identity_topology: unexplained disappearance, duplication, identity/count change,\n'
 'broken connectivity. Distinguish actual occlusion/out-of-frame motion from loss\n'
 'of an exposed object. Reflections/shadows/isolated blur ghosts are not extra balls.\n'
 'shape_stability: unintended rigid deformation, stretching or changing silhouette.\n'
 'Natural flexible deformation, fluid merging/splitting or phase changes are not\n'
 'artifacts by themselves; check whether they form the requested event instead.\n'
 'Do not judge numerical conservation. Directly visible added or removed material\n'
 'can still break identity or invalidate a before/after comparison of the original material.\n'
 'scene_continuity: abrupt/repeated lighting/texture/layout jumps or continuity-\n'
 'breaking cuts. Coherent camera motion and smooth illumination changes are fine.\n'
 'rendering_clarity: persistent/repeated corruption, smearing, fragmented edges or\n'
 'exposure/defocus that makes structure unreadable. Do not excuse persistent defects\n'
 'because the main object remains recognizable.\n'
 '\n'
 'VISUAL severity only: 0 no observable artifact; 1 slight/local/brief artifact;\n'
 '2 clear repeated/local or moderate sustained defect; 3 strong/persistent defect\n'
 'that disrupts continuity/readability; 4 gross loss or severe corruption.\n'
 'For identity_topology, repeated complete disappearance of an exposed main body,\n'
 'unexplained extra copies, or repeated loss/breakup of its essential working parts\n'
 'is severity 3 or 4, not a minor local defect, even if the object later reappears.\n'
 'This does not apply to directly observed occlusion, leaving the field of view,\n'
 'ordinary brief blur or a visibly continuous natural material transformation.\n'
 'Several isolated visible poses separated by missing subjects do not by themselves\n'
 'establish a complete trackable cycle or trajectory. Rate essential missing\n'
 'comparisons under event_coverage/readability instead of interpolating them.\n'
 'For these FOUR VISUAL dimensions, severity 0 may have empty evidence or positive\n'
 'evidence of continuity; nonzero severity requires timestamp evidence of a defect.\n'
 'A core event being visible earlier does not excuse later visual duplication,\n'
 'identity loss, deformation or unreadability. Inspect the WHOLE clip for those\n'
 'visual dimensions, even when an earlier interval answers the core task.\n'
 'Put each visual defect in ONE primary\n'
 'visual dimension. Task dimensions may cite its impact; code takes the minimum\n'
 'score ceiling instead of adding the same task failure several times.\n'
 '\n'
 'quality_checks are true only if seen:\n'
 'multiple_balls: multiple distinct physical balls, excluding shadows/reflections.\n'
 'strong_background_flicker: repeated strong abrupt oscillation, not a smooth fade.\n'
 'poor_visual_quality: sustained severe smearing/fragmentation/corruption.\n'
 'Strong flicker is a failure for every task. P2 also requires one ball. P3 may\n'
 'contain multiple balls; poor rendering is its additional failure condition.\n'
 'Do not assume any of these problems exist without inspecting the images.\n'
 '\n'
 'Use actual supplied frame times; evidence intervals must remain within the clip.\n'
 'Never infer quality from a generator name, target score, prior score or ranking.\n'
 'Return ONLY one flat JSON object. All seven dimensions and all three booleans\n'
 'are top-level keys. Do not wrap dimensions in assessment or task_assessment.\n'
 'Do not give a numerical final score; code applies fixed rules.\n'
 'Required field types; choose every severity and boolean from the evidence, using the grading definitions '
 'above. No example value is prescribed.\n'
 'reason: a nonempty string.\n'
 'identity_topology, shape_stability, scene_continuity, rendering_clarity, setup_fidelity, event_coverage, '
 'observation_readability: each is an object with severity (an integer from 0 through 4 according to its '
 'stated scale) and evidence (an array of evidence objects, with the existing requirements for empty or '
 'nonempty evidence unchanged).\n'
 'Each evidence object contains start_seconds (a number), end_seconds (a number), and description (a '
 'nonempty string).\n'
 'multiple_balls, strong_background_flicker, poor_visual_quality: each is a boolean selected independently '
 'from the actual evidence under its existing definition.\n'
 'Every evidence item uses start_seconds, end_seconds, description as above.\n')

def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _v24_make_sheets(items, directory):
    from PIL import Image, ImageDraw, ImageFont
    video = [item for item in items if item['role'] == 'video']
    if len(video) != 24:
        raise ValueError('Four-sheet experiment requires the original 24 distinct frames')
    font_path = Path(V24_FONT_PATH)
    if _file_sha256(font_path) != V24_FONT_SHA256:
        raise ValueError('Frozen sheet font changed')
    font = ImageFont.truetype(str(font_path), 24)
    sheets = []
    for sheet_index in range(4):
        selected = video[sheet_index * 6:(sheet_index + 1) * 6]
        tiles = [_image(item['path'], 640) for item in selected]
        width = max(tile.width for tile in tiles)
        height = max(tile.height for tile in tiles)
        canvas = Image.new('RGB', (3 * width, 40 + 2 * (height + 40)), 'white')
        if canvas.width > 1920 or canvas.height > 1344:
            raise ValueError('Four-sheet geometry exceeds frozen image profiling bounds; no silent shrinking')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f'Sheet {sheet_index + 1}/4 — frames {sheet_index * 6 + 1}–{sheet_index * 6 + 6} of 24', fill='black', font=font)
        for index, (item, tile) in enumerate(zip(selected, tiles)):
            x, y = (index % 3) * width, 40 + (index // 3) * (height + 40)
            canvas.paste(tile, (x, y))
            if canvas.crop((x, y, x + tile.width, y + tile.height)).tobytes() != tile.tobytes():
                raise AssertionError('Source tile pixels changed')
            draw.text((x + 6, y + height + 5), f"Frame {sheet_index * 6 + index + 1:02d} | {item['timestamp_seconds']:.3f} s", fill='black', font=font)
        path = Path(directory) / 'frames' / f'temporal_sheet_{sheet_index + 1:02d}.png'
        canvas.save(path)
        times = ', '.join(f"{item['timestamp_seconds']:.3f}" for item in selected)
        sheets.append({'path': str(path), 'role': 'overview', 'max_edge': max(canvas.size),
                       'label': f'Chronological sheet {sheet_index + 1}/4: six successive video frames, left to right then next row. Exact timestamps in seconds: {times}.',
                       'sheet_index': sheet_index + 1, 'frame_indices': [item['frame_index'] for item in selected],
                       'source_frame_paths': [item['path'] for item in selected],
                       'source_rgb_preserved_exactly': True})
    return [item for item in items if item['role'] not in ('video', 'overview')] + sheets + video

def _v24_mapped_box(box, basis, size):
    return [math.floor(box[0] * size[0] / basis[0]), math.floor(box[1] * size[1] / basis[1]),
            math.ceil(box[2] * size[0] / basis[0]), math.ceil(box[3] * size[1] / basis[1])]

def _v24_enlarge_crop(source, box):
    from PIL import Image, ImageOps
    crop = source.crop(box)
    fitted = ImageOps.contain(crop, (320, 320), Image.Resampling.LANCZOS)
    tile = Image.new('RGB', (320, 320), 'white')
    offset = ((320 - fitted.width) // 2, (320 - fitted.height) // 2)
    tile.paste(fitted, offset)
    return tile, {'source_crop_size': list(crop.size), 'resized_content_size': list(fitted.size),
                  'scale_xy': [fitted.width / crop.width, fitted.height / crop.height],
                  'padding_offset_xy': list(offset), 'tile_size': [320, 320],
                  'resize_filter': 'Pillow.Image.Resampling.LANCZOS', 'aspect_ratio_preserved': True}

def _v24_add_detail_sheets(items, video_path, reference_path, directory, config=None):
    if config is None:
        config = V24_ROI_CONFIG
    if reference_path is None:
        return items
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    # Reference identity is the only applicability condition; never generator,
    # predicted defect, score, or selected video content.
    if _file_sha256(reference_path) != config['reference_sha256']:
        return items
    frames = [f for f in items if f['role'] == 'video']
    if len(frames) != 24 or len({f['frame_index'] for f in frames}) != 24:
        raise ValueError('Detail protocol requires unchanged 24 distinct source indices')
    folder = Path(directory) / 'frames/detail_sources'
    folder.mkdir(parents=True, exist_ok=True)
    if config['font_path'] != V24_FONT_PATH or config['font_sha256'] != V24_FONT_SHA256:
        raise ValueError('Frozen detail font configuration changed')
    font_path = Path(config['font_path'])
    if _file_sha256(font_path) != config['font_sha256']:
        raise ValueError('Frozen detail font changed')
    font = ImageFont.truetype(str(font_path), 24)
    cap = cv2.VideoCapture(str(video_path))
    pairs, source_audits = [], []
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot decode original video for fixed-region detail')
        if abs(cap.get(cv2.CAP_PROP_FPS) - frames[0]['fps']) > 1e-6:
            raise ValueError('Detail source FPS differs from native sampled metadata')
        if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frames[0]['total_num_frames']:
            raise ValueError('Detail source frame count differs from native sampled metadata')
        for frame in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame['frame_index'])
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError('Original frame unavailable for fixed-region detail')
            source = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            thumbnail = source.copy()
            thumbnail.thumbnail((640, 640))
            encoded = io.BytesIO()
            thumbnail.save(encoded, format='JPEG', quality=92)
            if encoded.getvalue() != Path(frame['path']).read_bytes():
                raise ValueError('Original source replay does not reproduce the native sampled JPEG exactly')
            source_path = folder / f"source_{frame['frame_index']:06d}.png"
            source.save(source_path)
            pair = Image.new('RGB', (640, 320), 'white')
            regions = []
            for index, region in enumerate(config['regions']):
                box = _v24_mapped_box(region['box_xyxy'], config['coordinate_basis_wh'], source.size)
                if not (0 <= box[0] < box[2] <= source.width and 0 <= box[1] < box[3] <= source.height):
                    raise ValueError('Frozen region outside source; crop failure is a runtime error')
                tile, resize = _v24_enlarge_crop(source, box)
                pair.paste(tile, (index * 320, 0))
                regions.append({'name': region['name'], 'source_box_xyxy': box, **resize})
            pair_path = folder / f"pair_{frame['frame_index']:06d}.png"
            pair.save(pair_path)
            pairs.append(pair)
            source_audits.append({'frame_index': frame['frame_index'], 'timestamp_seconds': frame['timestamp_seconds'],
                                  'source_video': str(video_path), 'source_frame_path': str(source_path),
                                  'source_size': list(source.size), 'source_rgb_sha256': hashlib.sha256(source.tobytes()).hexdigest(),
                                  'native_frame_path': frame['path'], 'native_jpeg_sha256': _file_sha256(frame['path']),
                                  'native_thumbnail640_jpeg92_bytes_equal': True,
                                  'source_png_sha256': _file_sha256(source_path), 'pair_path': str(pair_path),
                                  'pair_sha256': _file_sha256(pair_path), 'regions': regions})
    finally:
        cap.release()
    sheets = []
    for sheet_index in range(4):
        canvas = Image.new('RGB', (1920, 760), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f'DETAIL {sheet_index + 1}/4 — fixed Region A | Region B at each time', fill='black', font=font)
        for tile_index in range(6):
            frame_index = sheet_index * 6 + tile_index
            x, y = (tile_index % 3) * 640, 40 + (tile_index // 3) * 360
            canvas.paste(pairs[frame_index], (x, y))
            frame = frames[frame_index]
            draw.text((x + 6, y + 325), f"Frame {frame_index + 1:02d} | {frame['timestamp_seconds']:.3f} s", fill='black', font=font)
        path = Path(directory) / 'frames' / f'temporal_detail_{sheet_index + 1:02d}.png'
        canvas.save(path)
        sources = source_audits[sheet_index * 6:(sheet_index + 1) * 6]
        times = ', '.join(f"{source['timestamp_seconds']:.3f}" for source in sources)
        sheets.append({'path': str(path), 'role': 'overview', 'overview_kind': 'fixed_reference_detail',
                       'max_edge': 1920, 'sheet_index': sheet_index + 1,
                       'label': f'DETAIL sheet {sheet_index + 1}/4, six successive times, left to right then next row. Each time is the same frame magnified in fixed Region A (left) and B (right). Timestamps: {times} s. Cross-check full frames for context.',
                       'frame_indices': [source['frame_index'] for source in sources],
                       'source_crop_audit': sources, 'reference_sha256': config['reference_sha256']})
    audit = {'configuration': config, 'reference_sha256': _file_sha256(reference_path),
             'source_video_sha256': _file_sha256(video_path), 'source_frames': source_audits,
             'same_24_indices_as_native': True, 'content_or_score_adaptive': False,
             'original_resolution_source_detail_added': True,
             'native_video_thumbnail640_unchanged': True, 'synthesized_objects_or_labels_added': False}
    (Path(directory) / 'detail_crop_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return [item for item in items if item['role'] != 'video'] + sheets + frames


def vllm_engine_policy(settings):
    policy = json.loads(json.dumps(VLLM_ENGINE_POLICY))
    if settings.input_mode in ('static_sheets', 'static_frames'):
        policy.update(max_model_len=49152, enable_chunked_prefill=True,
                      limit_mm_per_prompt={'image': {'count': 9, 'width': 1920,
                                                    'height': 1408}, 'video': 0},
                      mm_processor_kwargs={},
                      structured_outputs_config={'backend': 'xgrammar',
                                                 'reasoning_parser': 'qwen3',
                                                 'enable_in_reasoning': False})
    if settings.input_mode == 'static_frames':
        policy['limit_mm_per_prompt'] = {'image': {'count': 49, 'width': 640, 'height': 640}, 'video': 0}
        policy['structured_outputs_config']['disable_any_whitespace'] = True
    return policy

def source_frame_items(items):
    return [item for item in items if item.get('role') == 'video']

def uses_local_observation(items):
    return any(item.get('overview_kind') == 'fixed_reference_detail' or item.get('role') == 'local_detail' for item in items)

def inference_frame_items(items, settings, phase='grade'):
    if phase not in ('grade', 'observation'):
        raise ValueError('Local requests are prepared independently')
    if settings.input_mode == 'static_frames':
        if phase != 'grade':
            raise ValueError('Static frames use independent evidence calls, not a global observer')
        return _static_frames_inference_items(items)
    selected = list(items)
    if settings.input_mode in ('static_sheets', 'static_frames'):
        selected = [item for item in selected if item.get('role') != 'video']
    if phase == 'observation':
        selected = [item for item in selected if item.get('role') != 'reference']
    return selected

def validate_inference_response(timings, index):
    """Call after saving the whole batch, independently for each parent."""
    responses = timings.get('response_audit')
    if not isinstance(responses, list) or not 0 <= index < len(responses):
        raise ValueError('Inference response audit is missing or misassociated')
    value = responses[index]
    if value.get('finish_reason') != 'stop':
        raise ValueError('Incomplete inference: finish_reason=' + str(value.get('finish_reason')))
    if value.get('validation_error'):
        raise ValueError(value['validation_error'])

def decode_static_response(response, tokenizer, eos_ids):
    """Record all raw output first; narrowly remove at most one known EOS."""
    value = {'content': response.outputs[0].text,
             'token_ids': list(response.outputs[0].token_ids),
             'prompt_token_ids': list(response.prompt_token_ids),
             'prompt_tokens': len(response.prompt_token_ids),
             'output_tokens': len(response.outputs[0].token_ids),
             'finish_reason': response.outputs[0].finish_reason,
             'stop_reason': response.outputs[0].stop_reason,
             'validation_error': None, 'final_content': None}
    ids = value['token_ids']
    terminal = ids[-1] if ids and ids[-1] in eos_ids else None
    final_ids = ids[:-1] if terminal is not None else ids
    full = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    final = tokenizer.decode(final_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    value['text_token_decode_audit'] = {
        'terminal_eos_token_id': terminal,
        'text_equals_full_decode': value['content'] == full,
        'text_equals_decode_without_terminal_eos': value['content'] == final,
        'text_decode_verified': value['content'] in (full, final)}
    value['final_token_ids'] = final_ids
    value['removed_terminal_eos_token_id'] = terminal
    if not value['text_token_decode_audit']['text_decode_verified']:
        value['validation_error'] = 'Raw text differs from token decoding beyond a single terminal EOS'
    elif value['finish_reason'] != 'stop':
        value['validation_error'] = 'Incomplete generation: finish_reason=' + str(value['finish_reason'])
    elif any(tokenizer.convert_tokens_to_ids(marker) in final_ids for marker in ('<think>', '</think>')):
        value['validation_error'] = 'Unexpected thinking token with thinking disabled'
    else:
        value['final_content'] = final
    return value

@lru_cache(maxsize=1)
def visual_runtime_provenance():
    import cv2
    import numpy as np
    import PIL
    from PIL import _imaging
    from numpy._core import _multiarray_umath

    def file_record(path):
        path = Path(path).resolve()
        return {'path': str(path), 'bytes': path.stat().st_size,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    # Pin the selected OpenCV distribution, including its bundled shared
    # libraries. Generated Python bytecode is not an implementation input.
    cv_root = Path(cv2.__file__).resolve().parent.parent
    if not cv_root.name == 'opencv4':
        raise ValueError('Static sheets require the audited OpenCV overlay')
    files = {str(path.relative_to(cv_root)): file_record(path)
             for path in sorted(cv_root.rglob('*'))
             if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc'}
    font = file_record('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    if font['sha256'] != '690243adfefe0ce154b547db6205794bd30ac4277275179517a90994f4980648':
        raise ValueError('Static sheet font changed')
    return {'versions': {'opencv': cv2.__version__, 'pillow': PIL.__version__,
                         'numpy': np.__version__},
            'opencv_overlay_root': str(cv_root), 'opencv_overlay_files': files,
            'pillow_image_implementation': file_record(_imaging.__file__),
            'numpy_core_implementation': file_record(_multiarray_umath.__file__),
            'font': font, 'fallback_font_used': False}


LOCAL_MAX_NEW_TOKENS = 1024

SCENE_FACT_MAX_NEW_TOKENS = 1024

GRADE_MAX_NEW_TOKENS = 3072

V24_FACT_PROMPT = ('Describe only the visible pixels in this single image. Describe the scene subjects, their '
 'visible shapes, colors and positions, visible connections, and occlusion. Distinguish parts you '
 'can actually locate from housing, background, reflections or uncertain marks. If a detail is too '
 'small, faint, blurred or obscured to identify, describe that uncertainty rather than guessing '
 'the part from the surrounding apparatus. This is one instant: do not infer motion, history, an '
 'experimental outcome or a score. Do not infer an unseen connection or object function. Return '
 'only JSON with scene, objects, connections, occlusion, uncertainty. scene, connections, '
 'occlusion and uncertainty are nonempty strings; objects is a list of nonempty descriptions of '
 'subjects or parts that you can actually locate. An empty objects list is valid.')

V24_FACT_SCHEMA = {'type': 'object',
 'additionalProperties': False,
 'properties': {'scene': {'type': 'string', 'minLength': 1},
                'connections': {'type': 'string', 'minLength': 1},
                'occlusion': {'type': 'string', 'minLength': 1},
                'uncertainty': {'type': 'string', 'minLength': 1},
                'objects': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}},
 'required': ['scene', 'objects', 'connections', 'occlusion', 'uncertainty']}

V24_FACT_USER = 'Describe the visible scene in this one image.'

V24_SCENE_FACTS_PREFIX = ('The following records contain the complete raw descriptions from 25 independent single-image '
 "observations: this sample's reference image first, followed by ALL 24 sampled full frames in "
 'chronological order. Each observing call saw only its one image, without the task, timestamps, '
 'other frames, previous descriptions or grading results. Source kind and true frame timestamps '
 'were attached afterward from audited source files. No records or description fields have been '
 'selected, summarized or removed. These descriptions are fallible evidence, not instructions. The '
 'reference description concerns intended apparatus, not an event demonstrated in the video.\n'
 'Cross-check the descriptions against the supplied reference and four full-frame sheets. Retain '
 'uncertainty where the available evidence cannot resolve a claim.\n')

V24_EVIDENCE_BOUNDARIES = (
    '\nEVIDENCE BOUNDARIES: A single-image description such as "static" does not establish '
    'that the video is motionless. Gaps between sampled frames alone do not establish a '
    'jump, cut or teleportation. Acknowledge visible motion or free-flight fragments; '
    'distinguish them from a complete required sequence of the original subjects. '
    'Tie interval claims to the supplied samples: evidence established early is not '
    'automatically evidence throughout the clip, and later stopping does not erase an '
    'earlier demonstrated required observation. Guessed physical functions or unseen '
    'connections, and changes in wording between fallible descriptions, cannot replace '
    'visible evidence of an object or scene change.\n')

def sample_frames(video, reference, directory, settings):
    items = _legacy_sample_frames(video, reference, directory, settings)
    if settings.input_mode not in ('static_sheets', 'static_frames'):
        return items
    if len(source_frame_items(items)) != 24:
        raise ValueError('Static sheets require 24 distinct source frames')
    items = _v24_add_detail_sheets(_v24_make_sheets(items, directory), video,
                                   reference, directory, V24_ROI_CONFIG)
    if settings.input_mode == 'static_frames':
        if reference is None:
            raise ValueError('Static frames require the sample reference')
        for item in items:
            if item.get('role') == 'reference':
                item.update(source_reference_path=str(Path(reference).resolve()),
                            source_reference_sha256=_file_sha256(reference))
    return items

def _uses_static_sheets(items):
    return any(item.get('role') == 'overview' and 'sheet_index' in item for item in items)

def observation_messages_for(items):
    if _uses_static_sheets(items):
        raise ValueError('Static grading requires independent per-image evidence; global observation is forbidden')
    return _legacy_observation_messages_for(items)

def local_messages(image_path):
    return [{'role': 'system', 'content': V24_LOCAL_PROMPT},
            {'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)},
                                       {'type': 'text', 'text': V24_LOCAL_USER_TEXT}]}]

def scene_fact_messages(image_path):
    return [{'role': 'system', 'content': V24_FACT_PROMPT},
            {'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)},
                                       {'type': 'text', 'text': V24_FACT_USER}]}]

def parse_evidence(raw, phase):
    if phase not in ('local', 'scene_fact'):
        raise ValueError('Unknown independent evidence phase')
    if not isinstance(raw, str):
        raise ValueError('Evidence must be an unmodified JSON string')
    value = json.loads(raw, object_pairs_hook=_unique_json_keys)
    if phase == 'local':
        if not isinstance(value, dict) or set(value) != {'region_a', 'region_b'}:
            raise ValueError('Local observation must contain exactly the two regions')
        for region in value.values():
            if not isinstance(region, dict) or set(region) != {'description', 'uncertainty', 'extended_parts'}:
                raise ValueError('Local region fields differ from the frozen schema')
            for key in ('description', 'uncertainty'):
                if not isinstance(region[key], str) or not region[key].strip():
                    raise ValueError('Local description and uncertainty must be nonempty strings')
            if not isinstance(region['extended_parts'], list) or any(
                    not isinstance(part, str) or not part.strip() for part in region['extended_parts']):
                raise ValueError('Local extended_parts must contain nonempty descriptions')
    else:
        if not isinstance(value, dict) or set(value) != set(V24_FACT_SCHEMA['required']):
            raise ValueError('Scene fact fields differ from the frozen schema')
        for key in ('scene', 'connections', 'occlusion', 'uncertainty'):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError('Scene fact strings must be nonempty')
        if not isinstance(value['objects'], list) or any(
                not isinstance(part, str) or not part.strip() for part in value['objects']):
            raise ValueError('Scene fact objects must contain nonempty descriptions')
    return value

def parse_local_observation(raw):
    return parse_evidence(raw, 'local')

def _v24b_parent_inputs(source_frames, actual_frames, directory=None):
    if _uses_static_frames(actual_frames):
        return _static_frames_parent_inputs(source_frames, actual_frames, directory)
    if not isinstance(source_frames, list) or len(source_frames) != 24 or any(
            frame.get('role') != 'video' for frame in source_frames):
        raise ValueError('Expected exactly the 24 original source video frames')
    previous_index, previous_time = -1, -1.
    for frame in source_frames:
        index, timestamp = frame.get('frame_index'), frame.get('timestamp_seconds')
        if (type(index) is not int or index <= previous_index or
                type(timestamp) not in (int, float) or not math.isfinite(timestamp) or
                timestamp < 0 or timestamp <= previous_time):
            raise ValueError('Source indices and timestamps must be distinct and chronological')
        previous_index, previous_time = index, timestamp
    parent = Path(source_frames[0]['path']).resolve().parent.parent
    if directory is not None and Path(directory).resolve() != parent:
        raise ValueError('Source frames belong to another parent directory')
    local = uses_local_observation(actual_frames)
    if not isinstance(actual_frames, list) or len(actual_frames) != (9 if local else 5):
        raise ValueError('Static grading requires one reference, four full sheets and applicable four detail sheets')
    references = [frame for frame in actual_frames if frame.get('role') == 'reference']
    full = [frame for frame in actual_frames if frame.get('role') == 'overview' and
            frame.get('overview_kind') != 'fixed_reference_detail']
    detail = [frame for frame in actual_frames if frame.get('overview_kind') == 'fixed_reference_detail']
    if len(references) != 1 or len(full) != 4 or len(detail) != (4 if local else 0):
        raise ValueError('Static actual image roles do not match the complete protocol')
    if actual_frames != references + full + detail:
        raise ValueError('Static image order must preserve reference, full sheets, then detail sheets')
    all_frames = source_frames + actual_frames
    if len({str(frame['path']) for frame in all_frames}) != len(all_frames):
        raise ValueError('Source and actual images must have distinct paths')
    for frame in all_frames:
        path = Path(frame['path'])
        if path.resolve().parent != parent / 'frames' or not path.is_file():
            raise ValueError('Static images must be the actual files in their own parent frames directory')
    for number, sheet in enumerate(full, 1):
        selected = source_frames[(number-1)*6:number*6]
        if (sheet.get('sheet_index') != number or
                sheet.get('frame_indices') != [f['frame_index'] for f in selected] or
                sheet.get('source_frame_paths') != [f['path'] for f in selected]):
            raise ValueError('Full sheets do not preserve the original source order and paths')
    for number, sheet in enumerate(detail, 1):
        selected = source_frames[(number-1)*6:number*6]
        if (sheet.get('sheet_index') != number or
                sheet.get('frame_indices') != [f['frame_index'] for f in selected] or
                sheet.get('reference_sha256') != V24_ROI_CONFIG['reference_sha256']):
            raise ValueError('Fixed detail sheets have changed source order or reference identity')
    return parent, local, references[0]

def _v24b_expected_associations(source_frames, actual_frames, directory=None):
    from PIL import Image
    parent, local, reference = _v24b_parent_inputs(source_frames, actual_frames, directory)
    phase = 'local' if local else 'scene_fact'
    associations = []
    if local:
        crop_path = parent / 'detail_crop_audit.json'
        crop = json.loads(crop_path.read_text(), object_pairs_hook=_unique_json_keys)
        if (crop.get('reference_sha256') != V24_ROI_CONFIG['reference_sha256'] or
                crop.get('configuration') != V24_ROI_CONFIG or
                len(crop.get('source_frames', [])) != 24):
            raise ValueError('Local crop audit differs from the fixed reference protocol')
        crop_sha = _file_sha256(crop_path)
        for frame, original in zip(crop['source_frames'], source_frames):
            if (frame['frame_index'] != original['frame_index'] or
                    frame['timestamp_seconds'] != original['timestamp_seconds'] or
                    frame['native_frame_path'] != original['path'] or
                    _file_sha256(original['path']) != frame['native_jpeg_sha256']):
                raise ValueError('Local crop original image association changed')
            pair = Path(frame['pair_path'])
            if pair.resolve().parent != parent / 'frames' / 'detail_sources':
                raise ValueError('Local crop belongs to another parent')
            if _file_sha256(pair) != frame['pair_sha256']:
                raise ValueError('Local paired image changed')
            with Image.open(pair) as image:
                if image.size != (640, 320):
                    raise ValueError('Local request requires the original 640x320 paired image')
            association = {'phase': phase, 'parent_directory': str(parent),
                'source_kind': 'video_frame', 'frame_index': original['frame_index'],
                'timestamp_seconds': original['timestamp_seconds'],
                'image': str(pair), 'image_sha256': frame['pair_sha256'],
                'source_frame_path': original['path'], 'source_frame_sha256': frame['native_jpeg_sha256'],
                'source_crop_audit': str(crop_path), 'source_crop_audit_sha256': crop_sha}
            associations.append(association)
    else:
        for position, frame in enumerate([reference] + source_frames):
            association = {'phase': phase, 'parent_directory': str(parent),
                'source_kind': 'reference' if position == 0 else 'video_frame',
                'image': frame['path'], 'image_sha256': _file_sha256(frame['path'])}
            if position:
                association.update(frame_index=frame['frame_index'], timestamp_seconds=frame['timestamp_seconds'])
            associations.append(association)
    return parent, phase, associations

def prepare_evidence_members(prepared):
    """Prepare one parent's entire fixed set; never select frames or reuse replies."""
    if not isinstance(prepared, list) or len(prepared) != 1:
        raise ValueError('Prepare independent evidence separately for each parent')
    record = prepared[0]
    parent, phase, associations = _v24b_expected_associations(
        record['source_frames'], record['frames'], record['directory'])
    members = []
    for association in associations:
        suffix = ('reference' if association['source_kind'] == 'reference' else
                  f"frame_{association['frame_index']:06d}")
        directory = parent / ('local_observations' if phase == 'local' else 'scene_facts') / suffix
        directory.mkdir(parents=True, exist_ok=True)
        image_path = association['image']
        member = {'directory': directory, 'parent_directory': str(parent), 'phase': phase,
                  'source_frames': record['source_frames'], 'source_association': association,
                  'reference_guided_coordinates_used': phase == 'local',
                  'frames': [{'path': image_path,
                              'role': 'local_detail' if phase == 'local' else 'independent_scene_image',
                              'max_edge': 640}],
                  'messages': local_messages(image_path) if phase == 'local' else scene_fact_messages(image_path)}
        if phase == 'local':
            member['frames'][0]['label'] = 'One independent paired crop image'
            member['local_source_association'] = dict(association)
        members.append(member)
    return members

def _v24b_validate_raw_entry(entry, phase):
    raw = entry.get('raw_json')
    parse_evidence(raw, phase)
    if entry.get('raw_json_sha256') != hashlib.sha256(raw.encode()).hexdigest():
        raise ValueError('Independent evidence raw JSON changed')
    suffix = ('reference' if entry['source_kind'] == 'reference' else
              f"frame_{entry['frame_index']:06d}")
    owner = (Path(entry['parent_directory']).resolve() /
             ('local_observations' if phase == 'local' else 'scene_facts') / suffix)
    for key in ('raw_response_artifact', 'raw_final_artifact'):
        path = entry.get(key)
        filename = 'response.json' if key == 'raw_response_artifact' else 'final_response.json'
        if not isinstance(path, str) or Path(path).resolve() != owner / filename:
            raise ValueError('Independent response artifact belongs to another parent, phase or instant')
        if not isinstance(path, str) or _file_sha256(path) != entry.get(key + '_sha256'):
            raise ValueError('Independent evidence raw artifact changed: ' + key)
    final = json.loads(Path(entry['raw_final_artifact']).read_text(), object_pairs_hook=_unique_json_keys)
    if not isinstance(final, dict) or final.get('content') != raw:
        raise ValueError('Evidence is not the verbatim saved final response')
    response = json.loads(Path(entry['raw_response_artifact']).read_text(), object_pairs_hook=_unique_json_keys)
    if not isinstance(response, dict) or not isinstance(response.get('content'), str):
        raise ValueError('Raw response artifact is invalid')
    if response.get('final_content') is not None and response['final_content'] != raw:
        raise ValueError('Validated raw response and evidence final content differ')
    return raw

def evidence_text(entries, source_frames, actual_frames):
    if not isinstance(entries, list):
        raise ValueError('Independent evidence must be a complete ordered list')
    parent, phase, expected = _v24b_expected_associations(source_frames, actual_frames)
    if len(entries) != len(expected):
        raise ValueError('Every static parent needs all 24 local or all 25 scene observations')
    wrappers = []
    for entry, association in zip(entries, expected):
        if not isinstance(entry, dict) or any(entry.get(key) != value for key, value in association.items()):
            raise ValueError('Independent evidence phase, parent, image path/hash, order or source timestamp changed')
        if association['source_kind'] == 'reference' and any(key in entry for key in ('frame_index', 'timestamp_seconds')):
            raise ValueError('Reference evidence must not pretend to be a video instant')
        raw = _v24b_validate_raw_entry(entry, phase)
        if phase == 'local':
            wrapper = ('{"frame_index":' + json.dumps(association['frame_index']) +
                       ',"timestamp_seconds":' + json.dumps(association['timestamp_seconds']) +
                       ',"observation":' + raw + '}')
        elif association['source_kind'] == 'reference':
            wrapper = '{"source_kind":"reference","observation":' + raw + '}'
        else:
            wrapper = ('{"source_kind":"video_frame","frame_index":' + json.dumps(association['frame_index']) +
                       ',"timestamp_seconds":' + json.dumps(association['timestamp_seconds']) +
                       ',"observation":' + raw + '}')
        wrappers.append(wrapper)
    if _uses_static_frames(actual_frames):
        prefix = V25_LOCAL_EVIDENCE_PREFIX if phase == 'local' else V25_SCENE_FACTS_PREFIX
    else:
        prefix = V24_A_PREFIX if phase == 'local' else V24_SCENE_FACTS_PREFIX
    return prefix + '[\n' + ',\n'.join(wrappers) + '\n]'

def messages_for(items, task_id, prompt, observation=None, evidence_entries=None, source_frames=None, temporal_entries=None):
    if _uses_static_frames(items):
        if observation is not None or evidence_entries is None or source_frames is None:
            raise ValueError('Static frames require complete independent evidence without a global observation')
        evidence = evidence_text(evidence_entries, source_frames, items)
        messages = _legacy_messages_for(items, task_id, '', observation=None)
        messages[0]['content'] = V25_GRADE_SYSTEM
        messages[-1]['content'].insert(0, {'type': 'text', 'text': V25_INDIVIDUAL_PRESENTATION})
        messages[-1]['content'].append({'type': 'text', 'text': evidence})
        messages[-1]['content'].append({'type': 'text', 'text': V25_TEMPORAL_GROUNDING})
        appendix = temporal_evidence_text(temporal_entries, source_frames, items)
        messages[-1]['content'].extend([{'type': 'text', 'text': appendix},
                                      {'type': 'text', 'text': V26_TEMPORAL_RESPONSE_INSTRUCTION}])
        return messages
    if not _uses_static_sheets(items):
        if evidence_entries is not None:
            raise ValueError('Independent evidence requires the static sheet protocol')
        return _legacy_messages_for(items, task_id, prompt, observation)
    if observation is not None:
        raise ValueError('Static grading must not receive a global temporal narrative')
    if evidence_entries is None or source_frames is None:
        raise ValueError('Static grading requires complete independent evidence and source associations')
    evidence = evidence_text(evidence_entries, source_frames, items)
    # Preserve the frozen task-contract/images wording; original generation prompt
    # remains association metadata only and is never sent to the model.
    messages = _legacy_messages_for(items, task_id, '', observation=None)
    messages[0]['content'] = V24_LOCAL_RUBRIC + V24_EVIDENCE_BOUNDARIES
    messages[-1]['content'].insert(0, {'type': 'text', 'text': V24_DETAIL_PRESENTATION_TEXT})
    messages[-1]['content'].append({'type': 'text', 'text': evidence})
    return messages


def _static_phase_schemas(settings=None):
    schemas = structured_output_schemas()
    result = {'scene_fact': V24_FACT_SCHEMA, 'local': V24_LOCAL_SCHEMA, 'grade': schemas['grade']}
    if settings is not None and settings.input_mode == 'static_frames':
        result['temporal_comparison'] = V26_TEMPORAL_SCHEMA
    return result

def _static_phase_budgets(settings=None):
    result = {'scene_fact': SCENE_FACT_MAX_NEW_TOKENS,
              'local': LOCAL_MAX_NEW_TOKENS, 'grade': GRADE_MAX_NEW_TOKENS}
    if settings is not None and settings.input_mode == 'static_frames':
        result['temporal_comparison'] = TEMPORAL_MAX_NEW_TOKENS
    return result

@lru_cache(maxsize=1)
def _static_vllm_engine(model_path, policy_json):
    """Separate static engine cache; phase changes reuse the same model/engine."""
    model = Path(model_path)
    if not model.is_dir():
        raise ValueError('vllm requires an existing audited local model directory')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    from vllm import LLM
    from vllm.transformers_utils.config import get_config
    from transformers import AutoProcessor
    get_config(model_path, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                              trust_remote_code=False)
    policy = json.loads(policy_json)
    engine = LLM(model=model_path, **policy)
    _ACTIVE_VLLM_ENGINES[id(engine)] = engine
    generation_path = model / 'generation_config.json'
    generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
    eos = generation.get('eos_token_id', processor.tokenizer.eos_token_id)
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(token) is not int for token in eos):
        raise ValueError('Static inference requires explicit integer EOS token IDs')
    return processor, engine, frozenset(eos)

@lru_cache(maxsize=1)
def vllm_runtime_provenance(settings):
    if settings.input_mode not in ('static_sheets', 'static_frames'):
        return _legacy_vllm_runtime_provenance(settings)
    import importlib.metadata
    # Do not mutate the legacy cached nested dict or engine policy.
    result = json.loads(json.dumps(_legacy_vllm_runtime_provenance(settings)))
    distribution = importlib.metadata.distribution('vllm')
    for relative in ('vllm/reasoning/qwen3_reasoning_parser.py',
                     'vllm/reasoning/basic_parsers.py',
                     'vllm/v1/structured_output/__init__.py',
                     'vllm/config/structured_outputs.py', 'vllm/outputs.py',
                     'vllm/v1/engine/core_client.py', 'vllm/v1/engine/llm_engine.py'):
        path = Path(distribution.locate_file(relative))
        if not path.is_file():
            raise ValueError('Required audited static vLLM implementation missing: ' + relative)
        result['implementation_files'][relative] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
    result.update(engine_policy=vllm_engine_policy(settings), actual_input_mode=settings.input_mode,
                  native_video_sent=False, enable_thinking=False,
                  phase_required=True, stage_max_new_tokens=_static_phase_budgets(settings),
                  sampling={'temperature': 0., 'seed': 0, 'skip_special_tokens': False},
                  response_validation='strict_stop_and_token_decode_single_terminal_eos',
                  rendering_runtime=visual_runtime_provenance(),
                  json_schema_sha256={name: hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
                                      for name, schema in _static_phase_schemas(settings).items()})
    if settings.input_mode == 'static_frames':
        result['actual_input_presentation'] = 'independent_timestamped_frames'
        result['inference_protocol'] = inference_protocol_for(settings)
        result['evidence_policy'] = evidence_policy_for(settings)
        distribution = importlib.metadata.distribution('xgrammar')
        grammar_files = [str(path) for path in distribution.files
                         if str(path).startswith('xgrammar/') and
                         (str(path).endswith(('.py', '.so', '.pyd', '.dll', '.dylib')))]
        if not any(path.endswith(('.so', '.pyd', '.dll', '.dylib')) for path in grammar_files):
            raise ValueError('Native xgrammar implementation library is missing')
        for relative in sorted(grammar_files):
            path = Path(distribution.locate_file(relative))
            result['implementation_files'][relative] = {
                'sha256': _file_sha256(path), 'bytes': path.stat().st_size}
        result['temporal_prompt_sha256'] = hashlib.sha256(V26_TEMPORAL_PROMPT.encode()).hexdigest()
        result['temporal_response_instruction_sha256'] = hashlib.sha256(V26_TEMPORAL_RESPONSE_INSTRUCTION.encode()).hexdigest()
    return result

@lru_cache(maxsize=1)
def _static_multimodal_processor(model_path, policy_json):
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    policy = json.loads(policy_json)
    config = ModelConfig(model=model_path, dtype=policy['dtype'],
        max_model_len=policy['max_model_len'],
        limit_mm_per_prompt=policy['limit_mm_per_prompt'], mm_processor_kwargs={})
    return MULTIMODAL_REGISTRY.create_processor(config)

def static_prompt_token_ids(request, settings):
    """Process the complete input on CPU; no shortening or text-only estimate."""
    from vllm.multimodal.processing import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs
    processor = _static_multimodal_processor(
        settings.model, json.dumps(vllm_engine_policy(settings), sort_keys=True))
    processed = processor.apply(ProcessorInputs(prompt=request['prompt'],
        mm_data_items=processor.info.data_parser.parse_mm_data(request.get('multi_modal_data', {})),
        hf_processor_mm_kwargs={}), timing_ctx=TimingContext(enabled=False))
    return list(processed['prompt_token_ids'])

def _static_raw_response(response):
    candidates = [{'content': output.text, 'token_ids': list(output.token_ids or []),
                   'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason}
                  for output in (getattr(response, 'outputs', None) or [])]
    primary = candidates[0] if candidates else {
        'content': '', 'token_ids': [], 'finish_reason': 'empty_output', 'stop_reason': None}
    prompt_ids = list(getattr(response, 'prompt_token_ids', None) or [])
    return {**primary, 'prompt_token_ids': prompt_ids, 'prompt_tokens': len(prompt_ids),
            'output_tokens': len(primary['token_ids']), 'final_content': None,
            'validation_error': None if len(candidates) == 1 else 'Expected exactly one output candidate',
            'raw_output_candidates': candidates}

def vllm_replies(prepared, settings):
    if settings.input_mode not in ('static_sheets', 'static_frames'):
        return _legacy_vllm_replies(prepared, settings)
    settings.validate()
    if not prepared:
        raise ValueError('Cannot infer an empty batch')
    phases = [record.get('phase') for record in prepared]
    budgets, schemas = _static_phase_budgets(settings), _static_phase_schemas(settings)
    if any(phase not in budgets for phase in phases) or len(set(phases)) != 1:
        raise ValueError('Every static request must explicitly declare the same valid phase')
    phase = phases[0]
    maximum, schema = budgets[phase], schemas[phase]
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    started = time.monotonic()
    processor, engine, eos_ids = _static_vllm_engine(
        settings.model, json.dumps(vllm_engine_policy(settings), sort_keys=True))
    load_seconds = time.monotonic() - started
    requests, request_audit, selected = [], [], []
    values = [None] * len(prepared)
    for index, record in enumerate(prepared):
        audit = {
            'phase': phase, 'messages': record['messages'],
            'frame_metadata': record['frames'],
            'source_frame_metadata_for_scoring_only': record.get('source_frames'),
            'local_source_association': record.get('local_source_association'),
            'source_association': record.get('source_association'),
            'source_association_sent_to_model': False,
            'parent_directory': str(record.get('parent_directory', record.get('directory', ''))),
            'member_id': record.get('member_id'), 'parent_id': record.get('parent_id'),
            'actual_input_mode': 'one_independent_image' if phase != 'grade' else settings.input_mode,
            'actual_image_count': len(record['frames']), 'actual_video_count': 0, 'native_video_sent': False,
            'reference_guided_coordinates_used': any(
                frame.get('role') == 'local_detail' or frame.get('overview_kind') == 'fixed_reference_detail'
                for frame in record['frames']),
            'enable_thinking': False, 'mm_processor_kwargs': None,
            'json_schema': schema, 'max_tokens': maximum, 'temperature': 0., 'seed': 0,
            'skip_special_tokens': False, 'context_budget': vllm_engine_policy(settings)['max_model_len']}
        if settings.input_mode == 'static_frames':
            audit['actual_input_presentation'] = ('independent_timestamped_frames' if phase == 'grade' else 'one_independent_image')
        if phase == 'temporal_comparison':
            audit.update(actual_input_mode='temporal_comparison_images',
                actual_input_presentation='first_video_frame_anchor_and_chronological_window',
                temporal_source_association=record.get('temporal_source_association'),
                window_scope=record.get('window_scope'),
                parent_inference_frames=record.get('parent_inference_frames'),
                reference_pixels_sent=False, task_sent=False, prior_descriptions_sent=False)
        request_audit.append(audit)
        try:
            if phase == 'temporal_comparison':
                validate_temporal_member(record)
            if settings.input_mode == 'static_frames' and phase == 'grade':
                _static_frames_parent_inputs(record['source_frames'], record['frames'], record['directory'])
            if any(frame.get('role') == 'video' for frame in record['frames']):
                raise ValueError('Static actual inference cannot contain source video frames')
            parts = [part for message in record['messages']
                     if isinstance(message.get('content'), list) for part in message['content']]
            if any(part.get('type') == 'video' for part in parts):
                raise ValueError('Static messages cannot contain a video placeholder')
            paths = [part.get('image') for part in parts if part.get('type') == 'image']
            if paths != [str(frame['path']) for frame in record['frames']]:
                raise ValueError('Message image paths/order differ from the actual image metadata')
            request = vllm_request(processor, record, settings)
            request.pop('mm_processor_kwargs', None)
            if '<|video_pad|>' in request['prompt'] or 'video' in request.get('multi_modal_data', {}):
                raise ValueError('Static processor request unexpectedly contains video')
            images = request.get('multi_modal_data', {}).get('image', [])
            if len(images) != len(record['frames']):
                raise ValueError('Static processor image count differs from audited frame metadata')
            expected_ids = static_prompt_token_ids(request, settings)
            audit.update(prompt=request['prompt'], actual_processor_prompt_token_ids=expected_ids,
                         actual_processor_prompt_tokens=len(expected_ids))
            if len(expected_ids) + maximum > audit['context_budget']:
                raise ValueError('Complete evidence exceeds context; truncation is forbidden')
            requests.append(request)
            selected.append(index)
        except Exception as error:
            # Input failures stay with this member and remain explicit errors.
            audit['preparation_error'] = f'{type(error).__name__}: {error}'
            values[index] = {'content': '', 'token_ids': [], 'prompt_token_ids': [],
                'prompt_tokens': 0, 'output_tokens': 0, 'finish_reason': 'input_error',
                'stop_reason': None, 'validation_error': audit['preparation_error'],
                'final_content': None, 'inference_attempted': False}
    parameters = SamplingParams(temperature=0., seed=0, max_tokens=maximum,
                                skip_special_tokens=False,
                                structured_outputs=StructuredOutputsParams(json=schema))
    started = time.monotonic()
    responses = engine.generate(requests, parameters, use_tqdm=False) if requests else []
    generate_seconds = time.monotonic() - started
    # Snapshot every raw response before attempting any tokenizer decoding.
    # A per-member decoder exception must not discard its sibling raw replies.
    raw_values = [_static_raw_response(response) for response in responses]
    if len(responses) != len(requests):
        # Without the complete positional return, do not guess ownership of
        # available replies. Preserve them at call scope and reject the group.
        error = 'vllm response count differs from the recorded request count'
        for index in selected:
            values[index] = {'content': '', 'token_ids': [], 'prompt_token_ids': [],
                'prompt_tokens': 0, 'output_tokens': 0, 'finish_reason': 'response_count_error',
                'stop_reason': None, 'validation_error': error, 'final_content': None}
        return [value['content'] for value in values], {
            'generate_seconds': generate_seconds, 'engine_load_seconds': load_seconds,
            'phase': phase, 'max_tokens': maximum, 'request_audit': request_audit,
            'response_audit': values, 'infrastructure_error': error,
            'unassigned_raw_responses': raw_values,
            'output_tokens': [value['output_tokens'] for value in values]}
    for index, raw in zip(selected, raw_values):
        values[index] = raw
    for index, response in zip(selected, responses):
        if values[index]['validation_error'] is not None:
            continue
        try:
            values[index] = decode_static_response(response, processor.tokenizer, eos_ids)
        except Exception as error:
            values[index]['validation_error'] = f'{type(error).__name__}: {error}'
        expected = request_audit[index]['actual_processor_prompt_token_ids']
        values[index]['complete_prompt_token_ids_match'] = values[index]['prompt_token_ids'] == expected
        if not values[index]['complete_prompt_token_ids_match']:
            values[index]['validation_error'] = 'GPU prompt token IDs differ from complete CPU input'
        decode_audit = values[index].get('text_token_decode_audit')
        if decode_audit is not None:
            decode_audit['raw_content_matches_decode_without_single_terminal_eos'] = (
                decode_audit.get('terminal_eos_token_id') is not None and
                decode_audit.get('text_equals_decode_without_terminal_eos') is True)
    strings = [value['final_content'] if value.get('validation_error') is None and
               value.get('final_content') is not None else value['content'] for value in values]
    return strings, {'generate_seconds': generate_seconds, 'engine_load_seconds': load_seconds,
                     'phase': phase, 'max_tokens': maximum,
                     'request_audit': request_audit, 'response_audit': values,
                     'output_tokens': [value['output_tokens'] for value in values]}

def _vllm_reply(messages, items, settings, phase='grade'):
    """Single grading/reformatting adapter; static observer callers pass phase."""
    if settings.input_mode not in ('static_sheets', 'static_frames'):
        return _legacy_vllm_reply(messages, items, settings)
    strings, timings = vllm_replies([{'messages': messages, 'frames': items, 'phase': phase}], settings)
    # Full pipeline callers use vllm_replies directly to persist timings first.
    # This adapter must never turn an invalid completion into an accepted string.
    validate_inference_response(timings, 0)
    return strings[0]


def evaluate(video, reference, prompt, task_id, debug, settings):
    if settings.input_mode not in ('static_sheets', 'static_frames'):
        return _legacy_evaluate(video, reference, prompt, task_id, debug, settings)
    started = time.monotonic()
    base = Path(debug) / 'consistency'
    base.mkdir(parents=True, exist_ok=True)
    directory = base
    if any(base.iterdir()):
        attempt = 1
        while True:
            directory = base / 'attempts' / f'attempt_{attempt:04d}'
            try:
                directory.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                attempt += 1
    result = {'rubric_version': RUBRIC_VERSION, 'backend': settings.backend,
              'model': settings.model, 'device': settings.device,
              'input_mode': settings.input_mode, 'attention_policy': attention_policy_for(settings),
              'threshold': settings.threshold, 'score': None, 'passed': None,
              'status': 'error', 'artifact_directory': str(directory.resolve()),
              'stage_max_new_tokens': _static_phase_budgets(settings),
              'actual_backend': 'vllm', 'actual_input_mode': settings.input_mode,
              'previous_evidence_responses_reused': False, 'global_observation_used': False}
    if settings.input_mode == 'static_frames':
        result['actual_input_presentation'] = 'independent_timestamped_frames'
    presentation = ({'actual_input_mode': 'static_frames',
                     'actual_input_presentation': 'independent_timestamped_frames'}
                    if settings.input_mode == 'static_frames' else {})
    call_number = 0
    identity = 'single/' + Path(video).stem
    call_root = directory / 'inference'
    call_root.mkdir(parents=True, exist_ok=True)
    result['inference_call_directory'] = str(call_root)

    def infer(records):
        nonlocal call_number
        call_id = f'call{call_number:03d}'
        phase = records[0]['phase']
        _write(call_root / (call_id + '_request.json'), {
            'call_id': call_id, 'phase': phase,
            'members': [record['member_id'] for record in records],
            'parent_ids': [record['parent_id'] for record in records],
            'max_new_tokens': _static_phase_budgets(settings)[phase],
            'settings_max_new_tokens': settings.max_new_tokens,
            'requests': [{'messages': record['messages'], 'frame_metadata': record['frames'],
                'source_frame_metadata_for_scoring_only': record['source_frames'],
                'source_association': record.get('source_association'),
                'temporal_source_association': record.get('temporal_source_association'),
                'parent_inference_frames': record.get('parent_inference_frames')} for record in records]})
        values, timings = vllm_replies(records, settings)
        _write(call_root / (call_id + '_backend.json'), {'returned_strings': values, 'timings': timings})
        _write(directory / f'backend_call{call_number:03d}.json', timings)
        call_number += 1
        # Persist the complete batch before any per-member validation can fail.
        for index, record in enumerate(records):
            _write(record['directory'] / 'response.json', timings['response_audit'][index])
            _write(record['directory'] / 'request.json', {
                'messages': record['messages'], 'phase': record['phase'],
                'frame_metadata': record['frames'], 'source_frames': record['source_frames'],
                'source_association': record.get('source_association'),
                'temporal_source_association': record.get('temporal_source_association'),
                'window_scope': record.get('window_scope'),
                'parent_inference_frames': record.get('parent_inference_frames'),
                'source_association_sent_to_model': False,
                'generation_prompt_sent_to_vlm': False,
                'task_contract_sent_to_vlm': phase == 'grade',
                'member_id': record['member_id'], 'parent_id': record['parent_id'],
                'reference_guided_coordinates_used': record['phase'] == 'local' or uses_local_observation(record['frames']),
                'max_new_tokens': _static_phase_budgets(settings)[record['phase']],
                **((presentation if phase == 'grade' else
                   ({'actual_input_mode': 'temporal_comparison_images',
                     'actual_input_presentation': 'first_video_frame_anchor_and_chronological_window'}
                    if phase == 'temporal_comparison' else
                    {'actual_input_mode': 'one_independent_image', 'actual_input_presentation': 'one_independent_image'}))
                   if presentation else {}),
                })
        return values, timings

    try:
        settings.validate()
        prompt = task_prompt_for_video(prompt, video)
        task_audit = task_input_audit(task_id, prompt, reference)
        result.update({key: task_audit[key] for key in ('task_contract_version', 'task_contract_sha256')})
        items = sample_frames(video, reference, directory, settings)
        result['vllm_runtime'] = vllm_runtime_provenance(settings)
        source_frames = source_frame_items(items)
        actual_frames = inference_frame_items(items, settings)
        parent = {'directory': directory, 'frames': actual_frames, 'source_frames': source_frames,
                  'member_id': identity, 'parent_id': identity}
        local = uses_local_observation(actual_frames)
        phase = 'local' if local else 'scene_fact'
        result.update(frames=source_frames, inference_frame_metadata=actual_frames,
                      evidence_phase=phase, reference_guided_coordinates_used=local)
        result['observation_strategy'] = 'independent_local' if local else 'independent_scene_facts'
        _write(directory / 'prepared_input.json', {'member_id': identity,
            'source_frame_metadata_for_scoring_only': source_frames,
            'inference_frame_metadata': actual_frames, 'evidence_phase': phase,
            'expected_evidence_count': 24 if local else 25, 'task_input_audit': task_audit,
            'global_observation_used': False, 'reference_guided_coordinates_used': local,
            **({'expected_temporal_count': 4} if settings.input_mode == 'static_frames' else {}), **presentation})
        members = prepare_evidence_members([parent])
        temporal_members = prepare_temporal_members([parent]) if settings.input_mode == 'static_frames' else []
        for ordinal, member in enumerate(temporal_members):
            member.update(parent_id=identity, member_id=identity + f'/temporal_comparison/{ordinal:02d}')
        for ordinal, member in enumerate(members):
            member.update(parent_id=identity, member_id=identity + f'/{phase}/{ordinal:02d}')
        _write(call_root / 'stage_plan.json', {'original_parent_group': [identity],
            'planned_evidence_calls': [{'phase': phase,
                'members': [m['member_id'] for m in members[offset:offset + 4]]}
                for offset in range(0, len(members), 4)],
            'planned_temporal_calls': ([{'phase': 'temporal_comparison',
                'members': [m['member_id'] for m in temporal_members]}] if temporal_members else []),
            'planned_grade_parents': [identity]})
        entries, temporal_entries, errors = [], [], []
        batches = [members[offset:offset + 4] for offset in range(0, len(members), 4)]
        if temporal_members:
            batches.append(temporal_members)
        for batch in batches:
            member_phase = batch[0]['phase']
            values, timings = infer(batch)
            for index, member in enumerate(batch):
                try:
                    validate_inference_response(timings, index)
                    parsed = (parse_temporal_comparison(values[index], member['expected_comparison_ordinals'])
                              if member_phase == 'temporal_comparison' else parse_evidence(values[index], phase))
                    final_path = member['directory'] / 'final_response.json'
                    _write(final_path, {'content': values[index],
                        'token_ids': timings['response_audit'][index]['final_token_ids'],
                        'removed_terminal_eos_token_id': timings['response_audit'][index].get('removed_terminal_eos_token_id')})
                    raw_path = member['directory'] / 'response.json'
                    association = member['temporal_source_association'] if member_phase == 'temporal_comparison' else member['source_association']
                    entry = {**association, 'phase': member_phase,
                        'raw_json': values[index], 'parsed': parsed,
                        'raw_json_sha256': hashlib.sha256(values[index].encode()).hexdigest(),
                        'raw_response_artifact': str(raw_path),
                        'raw_response_artifact_sha256': _file_sha256(raw_path),
                        'raw_final_artifact': str(final_path),
                        'raw_final_artifact_sha256': _file_sha256(final_path)}
                    _write(member['directory'] / 'parsed.json', entry)
                    (temporal_entries if member_phase == 'temporal_comparison' else entries).append(entry)
                except Exception as error:
                    errors.append({'phase': member_phase, 'directory': str(member['directory']),
                                   'reason': f'{type(error).__name__}: {error}'})
            # Continue the fixed set, retaining every actual response, but never
            # use an incomplete set to infer a grade or substitute a narrative.
        if temporal_members:
            temporal_errors = [error for error in errors if error['phase'] == 'temporal_comparison']
            _write(directory / 'temporal_entries.json', {'entries': temporal_entries,
                'expected_count': 4, 'complete': not temporal_errors and len(temporal_entries) == 4,
                'errors': temporal_errors, 'fresh_inference': True})
        _write(directory / 'evidence_entries.json', {'entries': entries, 'phase': phase,
            'complete': not errors, 'errors': errors, 'expected_count': len(members), 'fresh_inference': True,
            'reference_guided_coordinates_used': local})
        _write(call_root / 'grade_group.json', {'original_parent_group': [identity],
            'actual_members': [] if errors else [identity],
            'excluded_parents': {identity: errors} if errors else {}})
        if errors:
            raise ValueError(f'{len(errors)} independent evidence request(s) failed')
        evidence = evidence_text(entries, source_frames, actual_frames)
        (directory / 'evidence.txt').write_text(evidence)
        result.update(independent_evidence=entries,
                      evidence_text_sha256=hashlib.sha256(evidence.encode()).hexdigest())
        result['observation_rubric_sha256'] = hashlib.sha256(
            (V24_LOCAL_PROMPT if local else V24_FACT_PROMPT).encode()).hexdigest()
        if temporal_members:
            appendix = temporal_evidence_text(temporal_entries, source_frames, actual_frames, directory)
            (directory / 'temporal_evidence.txt').write_text(appendix)
            result.update(independent_temporal_evidence=temporal_entries, complete_temporal_count=4,
                temporal_evidence_text_sha256=hashlib.sha256(appendix.encode()).hexdigest(),
                temporal_rubric_sha256=hashlib.sha256(V26_TEMPORAL_PROMPT.encode()).hexdigest())
        messages = messages_for(actual_frames, task_id, prompt, evidence_entries=entries,
                                source_frames=source_frames, temporal_entries=temporal_entries)
        values, timings = infer([dict(parent, messages=messages, phase='grade')])
        request = json.loads((directory / 'request.json').read_text())
        request.update(task_audit, rubric_version=RUBRIC_VERSION, evidence_phase=phase,
            complete_evidence_count=len(entries), evidence_text_sha256=result['evidence_text_sha256'],
            inference_frame_metadata=actual_frames, global_observation_used=False)
        if temporal_members:
            request.update(complete_temporal_count=4, temporal_evidence_text_sha256=result['temporal_evidence_text_sha256'])
        _write(directory / 'request.json', request)
        validate_inference_response(timings, 0)
        _write(directory / 'final_response.json', {'content': values[0],
            'token_ids': timings['response_audit'][0]['final_token_ids'],
            'removed_terminal_eos_token_id': timings['response_audit'][0].get('removed_terminal_eos_token_id')})
        result['rubric_sha256'] = hashlib.sha256(messages[0]['content'].encode()).hexdigest()
        judgment = score_assessment(parse_judgment(values[0]), task_id, source_frames)
        result.update(judgment, passed=judgment['score'] >= settings.threshold and not judgment.get('forced_rejection', False),
                      status='evaluated', format_retry_used=False,
                      accepted_response_artifact='final_response.json', response_attempts=1)
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}', issues=[])
    result['elapsed_seconds'] = time.monotonic() - started
    _write(directory / 'consistency.json', result)
    if directory != base:
        _write(base / 'consistency.json', result)
    return result


# v25 independent-frame presentation; all scoring and evidence schemas are unchanged.
V25_SYSTEM_SHEETS = 'The FOUR chronological sheets show the SAME video at ALL 24 sampled times, six\nsuccessive frames per sheet. Read sheets 1–4 in order and each sheet left to\nright then next row. Cross-check the full-frame and detail sheets.'
V25_SYSTEM_IMAGES = 'The 24 independent full-frame images show the SAME video at ALL 24 sampled times, in chronological order. When present, a detail pair immediately follows its full frame at the SAME timestamp. Cross-check each full frame and its corresponding detail pair.'
V25_INDIVIDUAL_PRESENTATION = 'The independent full-frame images show all 24 sampled times in chronological order. When present, each full frame is immediately followed by one DETAIL pair from the same timestamp, with Region A on the left and Region B on the right. These are subsets of that same frame, not new times, scenes or additional simultaneous objects. The fixed region coordinates come from the common setup reference; its reference pixels were not included in the local observation calls. The detail crops use the original resolution of the same video frames, with the full-frame images retained as context. Magnification does not create new source detail. A fixed crop may omit content after camera or object movement; inspect the corresponding full frame before claiming disappearance, motion or damage.'
V25_TEMPORAL_GROUNDING = 'Before assigning severities, start reason with a compact chronological inventory from the actual images: at least three separated supplied timestamps (early, intermediate, late), locating each main subject relative to the same fixed landmarks and describing its visible attachment or orientation when relevant. Use positions within each full video frame. Compare those observations before deciding whether the required event is present. Repeated names, counts or coarse location words do not establish equal positions. If claiming whole-clip absence or stationarity, reconcile all 24 sampled full frames and any contrary positional evidence. Precise coordinates, speeds and guessed physical behavior are not required. Static experiments do not require motion.'

V25_GRADE_SYSTEM = V24_LOCAL_RUBRIC.replace(V25_SYSTEM_SHEETS, V25_SYSTEM_IMAGES) + V24_EVIDENCE_BOUNDARIES
V25_LOCAL_EVIDENCE_PREFIX = V24_A_PREFIX.replace('full-frame and detail sheets', 'full-frame images and corresponding detail pairs')
V25_SCENE_FACTS_PREFIX = V24_SCENE_FACTS_PREFIX.replace('four full-frame sheets', '24 independent full-frame images')
_ACTIVE_VLLM_ENGINES = {}


def _uses_static_frames(items):
    return isinstance(items, list) and any(item.get('role') == 'independent_full_frame' for item in items)


def _static_frames_inference_items(items):
    sources = source_frame_items(items)
    unused_sheets = [item for item in items if item.get('role') != 'video']
    _, phase, associations = _v24b_expected_associations(sources, unused_sheets)
    references = [item for item in unused_sheets if item.get('role') == 'reference']
    frames = [dict(references[0])]
    for ordinal, source in enumerate(sources):
        label = f"Full video frame {ordinal + 1:02d}/24 at {source['timestamp_seconds']:.3f} seconds."
        frames.append(dict(source, role='independent_full_frame', max_edge=640, label=label))
        if phase == 'local':
            association = associations[ordinal]
            label = (f"DETAIL pair for the SAME frame {ordinal + 1:02d}/24 at "
                     f"{source['timestamp_seconds']:.3f} seconds: Region A left, Region B right. "
                     'This is the same instant, not another frame.')
            frames.append({'path': association['image'], 'role': 'local_detail', 'max_edge': 640,
                'frame_index': source['frame_index'], 'timestamp_seconds': source['timestamp_seconds'],
                'source_full_frame_path': source['path'], 'image_sha256': association['image_sha256'],
                'reference_guided_coordinates_used': True, 'label': label})
    _static_frames_parent_inputs(sources, frames)
    return frames


def _static_frames_parent_inputs(source_frames, actual_frames, directory=None):
    """Only exact own source JPEG aliases and same-time detail pairs are allowed."""
    import io
    from PIL import Image
    if (not isinstance(source_frames, list) or len(source_frames) != 24 or
            any(frame.get('role') != 'video' for frame in source_frames)):
        raise ValueError('Static frames require exactly the 24 original source video frames')
    parent = Path(source_frames[0]['path']).resolve().parent.parent
    if directory is not None and Path(directory).resolve() != parent:
        raise ValueError('Static frames belong to another parent directory')
    last_index, last_time = -1, -1.
    for frame in source_frames:
        index, timestamp = frame.get('frame_index'), frame.get('timestamp_seconds')
        if (type(index) is not int or index <= last_index or type(timestamp) not in (int, float)
                or not math.isfinite(timestamp) or timestamp < 0 or timestamp <= last_time):
            raise ValueError('Source frame indices/timestamps must be distinct and chronological')
        if (Path(frame['path']).resolve() != parent / 'frames' / f'frame_{index:06d}.jpg'
                or not Path(frame['path']).is_file()):
            raise ValueError('Source frame path belongs to another parent or index')
        last_index, last_time = index, timestamp
    if not isinstance(actual_frames, list) or not actual_frames or actual_frames[0].get('role') != 'reference':
        raise ValueError('Static frames require their own reference first')
    reference = actual_frames[0]
    if (Path(reference['path']).resolve() != parent / 'frames' / 'reference.jpg'
            or 'frame_index' in reference or 'timestamp_seconds' in reference):
        raise ValueError('Reference image must be own reference.jpg without video time')
    original = Path(reference.get('source_reference_path', ''))
    if not original.is_file() or _file_sha256(original) != reference.get('source_reference_sha256'):
        raise ValueError('Original reference path/hash is missing or changed')
    encoded = io.BytesIO()
    _image(original, 640).save(encoded, format='JPEG', quality=92)
    if Path(reference['path']).read_bytes() != encoded.getvalue():
        raise ValueError('Actual reference JPEG differs from its original source')
    local = reference['source_reference_sha256'] == V24_ROI_CONFIG['reference_sha256']
    if len(actual_frames) != (49 if local else 25):
        raise ValueError('Reference policy requires all 25 independent images or all 49 with details')
    if len({str(Path(frame['path']).resolve()) for frame in actual_frames}) != len(actual_frames):
        raise ValueError('Actual independent images must not duplicate another image path')
    crop = None
    if local:
        crop = json.loads((parent / 'detail_crop_audit.json').read_text(), object_pairs_hook=_unique_json_keys)
        if (crop.get('reference_sha256') != reference['source_reference_sha256'] or
                crop.get('configuration') != V24_ROI_CONFIG or len(crop.get('source_frames', [])) != 24):
            raise ValueError('Local crop audit differs from its own reference/configuration')
    for ordinal, source in enumerate(source_frames):
        actual = actual_frames[1 + ordinal * (2 if local else 1)]
        if (actual.get('role') != 'independent_full_frame' or actual.get('max_edge') != 640 or
                any(actual.get(key) != source[key] for key in ('path', 'frame_index', 'timestamp_seconds', 'fps', 'total_num_frames'))):
            raise ValueError('Actual full frame must be the exact same-time source JPEG alias')
        if local:
            detail = actual_frames[2 + ordinal * 2]
            audit = crop['source_frames'][ordinal]
            pair = parent / 'frames' / 'detail_sources' / f"pair_{source['frame_index']:06d}.png"
            if (detail.get('role') != 'local_detail' or detail.get('max_edge') != 640
                    or detail.get('reference_guided_coordinates_used') is not True
                    or Path(detail['path']).resolve() != pair or Path(audit['pair_path']).resolve() != pair
                    or detail.get('source_full_frame_path') != source['path']
                    or audit.get('native_frame_path') != source['path']
                    or any(detail.get(key) != source[key] or audit.get(key) != source[key]
                           for key in ('frame_index', 'timestamp_seconds'))
                    or _file_sha256(pair) != detail.get('image_sha256')
                    or _file_sha256(pair) != audit.get('pair_sha256')
                    or _file_sha256(source['path']) != audit.get('native_jpeg_sha256')):
                raise ValueError('Detail pair path/hash/source/time differs from its own full frame')
            with Image.open(pair) as image:
                if image.size != (640, 320):
                    raise ValueError('Local detail pair must preserve the original 640x320 image')
    return parent, local, reference


def shutdown_vllm_engines(timeout=30.0):
    """Close only this worker's existing engines; never allocate one during cleanup."""
    import gc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Shutdown timeout must be positive and finite')
    started = time.monotonic()
    outcomes = []
    for identity, engine in list(_ACTIVE_VLLM_ENGINES.items()):
        try:
            engine.llm_engine.engine_core.shutdown(timeout=timeout)
            outcomes.append({'engine_id': identity, 'completed': True})
            del _ACTIVE_VLLM_ENGINES[identity]
        except Exception as error:
            outcomes.append({'engine_id': identity, 'completed': False,
                             'error': f'{type(error).__name__}: {error}'})
    if not _ACTIVE_VLLM_ENGINES:
        _static_vllm_engine.cache_clear()
        _static_multimodal_processor.cache_clear()
        gc.collect()
    return {'attempted': bool(outcomes), 'completed': not _ACTIVE_VLLM_ENGINES,
            'engines': outcomes, 'api': 'LLM.llm_engine.engine_core.shutdown',
            'timeout_seconds': timeout, 'elapsed_seconds': time.monotonic() - started}


# v26: original v25 grading plus four fresh temporal comparisons (R32 W).
V26_TEMPORAL_PROMPT = 'Compare only the visible pixels at the supplied video times. The first sampled VIDEO frame is a comparison anchor, not the intended reference image and not a claim that its state is correct or must remain unchanged. Describe the subjects and surrounding apparatus visible at the anchor, then for every requested later frame describe concrete visible state and differences from that anchor: position or orientation, outlines and identifiable parts, visible contacts, connections or supports. Each target record must cover both the main subjects and visible apparatus, supports and identifiable component parts when present; even if the latter appear stationary, state what visibly matches or cannot be verified. Do not invent apparatus or parts that are not visible. Preserve ordinary displacement/orientation as observations; a difference alone is not a defect. Distinguish a visible structural change from perspective, occlusion, blur or uncertainty; do not invent physical roles, causes, motion between unsupplied times, experimental outcomes or scores. Detail pairs, when supplied, are crops of the same full frame, not extra instants or objects; cross-check their full frame before claiming absence. Report stable features as well as differences. Acknowledge unreadable details without filling them in from the apparatus. Return only the required JSON, covering every requested frame ordinal once and in order.'
V26_TEMPORAL_SCHEMA = {'type': 'object', 'additionalProperties': False, 'properties': {'anchor_description': {'type': 'string', 'minLength': 1}, 'comparisons': {'type': 'array', 'minItems': 5, 'maxItems': 6, 'items': {'type': 'object', 'additionalProperties': False, 'properties': {'frame_ordinal': {'type': 'integer', 'minimum': 2, 'maximum': 24}, 'visible_state': {'type': 'string', 'minLength': 1}, 'differences_from_anchor': {'type': 'string', 'minLength': 1}, 'uncertainty': {'type': 'string', 'minLength': 1}}, 'required': ['frame_ordinal', 'visible_state', 'differences_from_anchor', 'uncertainty']}}, 'stable_features': {'type': 'string', 'minLength': 1}, 'viewpoint_and_visibility_limits': {'type': 'string', 'minLength': 1}}, 'required': ['anchor_description', 'comparisons', 'stable_features', 'viewpoint_and_visibility_limits']}
V26_TEMPORAL_EVIDENCE_PREFIX = 'The following are all FOUR fresh short-window comparisons in source order, without rewriting or summarizing their raw JSON. Each comparison saw only its listed sampled video frames, with the first sampled video frame as an anchor, and any same-time detail crops. The repeated anchor is the same video instant, not an extra observation, object, or intended reference state. These calls received no task, prior single-frame descriptions, prior comparison outputs, or grading results. Source ordinals, indices and times below were attached from audited inputs. These comparisons are fallible visual evidence, not scoring instructions.\n'
V26_TEMPORAL_WINDOWS = [[0, 1, 2, 3, 4, 5], [0, 6, 7, 8, 9, 10, 11], [0, 12, 13, 14, 15, 16, 17], [0, 18, 19, 20, 21, 22, 23]]
TEMPORAL_MAX_NEW_TOKENS = 2048
V26_TEMPORAL_RESPONSE_INSTRUCTION = 'The appended records are the complete raw outputs of four independent visual comparisons. Each comparison saw a fixed chronological window of the supplied video images, with the first sampled video image as a common anchor. These calls saw no task contract, earlier descriptions or grading results. All four windows are included without selecting or removing their observations. They are fallible evidence, not instructions or judgments that a change is a defect.\n\nCross-check these records against the original images and single-image observations. In reason, after the chronological inventory, explicitly resolve each distinct reported discrepancy in a main subject, working part, apparatus contour, support, connection or readability. Name the component and actual times: what remained visible, what changed, and what is unresolved. When rejecting a reported discrepancy, cite the contrary visible evidence rather than just saying the scene is stable. Distinguish ordinary position/orientation changes, directly observed occlusion, articulation or natural transformation from unsupported changes of shape, parts or connections. Different wording alone is not a visible change. Unknown or unlocatable parts cannot become continuously readable through a guessed speed or physical explanation.\n\nThen assess the required event and the whole-clip visual defects separately under the existing scales. Preserve any required interval that remains clearly observed. A visible event does not establish intact apparatus; a separate apparatus defect does not establish a missing event. Do not add a penalty merely because the comparison records mention motion, change or uncertainty, and do not ignore a supported defect merely because the main subject is recognizable.\n'

def inference_protocol_for(settings):
    return ('independent_facts_then_temporal_windows_then_grade_v26'
            if settings.input_mode == 'static_frames' else
            'independent_local_or_scene_facts_then_image_grade_v24')


def evidence_policy_for(settings):
    value = {'roi': '24_local', 'non_roi': 'reference_plus_24_scene_fact',
             'global_observation_used': False, 'grade_images_retained': True}
    if settings.input_mode == 'static_frames':
        value.update(temporal_comparisons=4, temporal_windows_zero_based=V26_TEMPORAL_WINDOWS,
                     temporal_anchor='first_sampled_video_frame',
                     temporal_raw_preserved=True, selected_grade_arm='W')
    return value


def _temporal_member(parent, window_index):
    if type(window_index) is not int or not 0 <= window_index < 4:
        raise ValueError('Temporal window index must be 0..3')
    owner = Path(parent['directory']).resolve()
    sources, actual = parent['source_frames'], parent['frames']
    _static_frames_parent_inputs(sources, actual, owner)
    local = len(actual) == 49
    frames, scope = [], []
    for ordinal in V26_TEMPORAL_WINDOWS[window_index]:
        source = sources[ordinal]
        first = 1 + ordinal * (2 if local else 1)
        selected = actual[first:first + (2 if local else 1)]
        frames.extend(json.loads(json.dumps(selected)))
        scope.append({'frame_ordinal': ordinal + 1, 'frame_index': source['frame_index'],
            'timestamp_seconds': source['timestamp_seconds'], 'is_anchor': ordinal == 0,
            'images': [{'path': image['path'], 'role': image['role'],
                        'sha256': _file_sha256(image['path'])} for image in selected]})
    targets = [ordinal + 1 for ordinal in V26_TEMPORAL_WINDOWS[window_index] if ordinal != 0]
    content = [{'type': 'text', 'text': (
        'Anchor is supplied video frame ordinal 1. Compare requested frame ordinals ' +
        ', '.join(map(str, targets)) + ' with that anchor. Return comparisons in this exact order. '
        'The pictures below are only this short window plus the anchor. '
        'When present, each detail pair comes from the immediately preceding full frame; '
        'its fixed crop coordinates are reference-guided, but no reference image pixels are supplied here.')}]
    for frame in frames:
        content.extend([{'type': 'text', 'text': frame['label']}, {'type': 'image', 'image': frame['path']}])
    association = {'phase': 'temporal_comparison', 'parent_directory': str(owner),
                   'window_index': window_index, 'window_scope': scope,
                   'expected_comparison_ordinals': targets,
                   'source_image_hashes': {frame['path']: _file_sha256(frame['path']) for frame in frames}}
    return {'directory': owner / 'temporal_comparisons' / f'window{window_index + 1:02d}',
            'parent_directory': str(owner), 'phase': 'temporal_comparison',
            'window_index': window_index, 'window_scope': scope,
            'expected_comparison_ordinals': targets, 'temporal_source_association': association,
            'source_frames': sources, 'parent_inference_frames': actual, 'frames': frames,
            'messages': [{'role': 'system', 'content': V26_TEMPORAL_PROMPT},
                         {'role': 'user', 'content': content}],
            'actual_input_mode': 'temporal_comparison_images',
            'actual_input_presentation': 'first_video_frame_anchor_and_chronological_window',
            'reference_pixels_sent': False, 'reference_guided_coordinates_used': local,
            'task_sent': False, 'prior_descriptions_sent': False}


def prepare_temporal_members(prepared):
    if not isinstance(prepared, list) or len(prepared) != 1:
        raise ValueError('Prepare temporal comparisons separately for each parent')
    records = [_temporal_member(prepared[0], index) for index in range(4)]
    for record in records:
        record['directory'].mkdir(parents=True, exist_ok=True)
    return records


def validate_temporal_member(record):
    parent = {'directory': record['parent_directory'], 'source_frames': record['source_frames'],
              'frames': record['parent_inference_frames']}
    expected = _temporal_member(parent, record['window_index'])
    for key, value in expected.items():
        actual = Path(record.get(key, '')) if key == 'directory' else record.get(key)
        if actual != value:
            raise ValueError('Temporal parent, images, order, hashes or messages changed: ' + key)
    return expected


def parse_temporal_comparison(raw, expected_ordinals):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    value = json.loads(raw, object_pairs_hook=_unique_json_keys, parse_constant=invalid_constant)
    if not isinstance(value, dict) or set(value) != set(V26_TEMPORAL_SCHEMA['required']):
        raise ValueError('Comparison fields differ from the frozen schema')
    for key in ('anchor_description', 'stable_features', 'viewpoint_and_visibility_limits'):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError('Comparison description must be a nonempty string')
    items = value['comparisons']
    if not isinstance(items, list) or len(items) != len(expected_ordinals):
        raise ValueError('Every requested target frame needs exactly one comparison')
    for item, ordinal in zip(items, expected_ordinals):
        if (not isinstance(item, dict) or set(item) !=
                {'frame_ordinal', 'visible_state', 'differences_from_anchor', 'uncertainty'} or
                type(item['frame_ordinal']) is not int or item['frame_ordinal'] != ordinal):
            raise ValueError('Comparison target frame order/identity changed')
        if any(not isinstance(item[key], str) or not item[key].strip()
               for key in ('visible_state', 'differences_from_anchor', 'uncertainty')):
            raise ValueError('Comparison frame descriptions must be nonempty')
    return value


def temporal_evidence_text(entries, source_frames, actual_frames, directory=None):
    owner = Path(source_frames[0]['path']).resolve().parent.parent
    if directory is not None and Path(directory).resolve() != owner:
        raise ValueError('Temporal evidence belongs to another parent')
    if not isinstance(entries, list) or len(entries) != 4:
        raise ValueError('Exactly all four ordered temporal windows are required')
    parent = {'directory': owner, 'source_frames': source_frames, 'frames': actual_frames}
    wrappers = []
    for index, entry in enumerate(entries):
        expected = _temporal_member(parent, index)
        association = expected['temporal_source_association']
        if any(entry.get(key) != value for key, value in association.items()):
            raise ValueError('Temporal evidence parent/window/source association changed')
        raw = entry.get('raw_json')
        parse_temporal_comparison(raw, expected['expected_comparison_ordinals'])
        if hashlib.sha256(raw.encode()).hexdigest() != entry.get('raw_json_sha256'):
            raise ValueError('Temporal raw JSON changed')
        for key, filename in [('raw_response_artifact', 'response.json'),
                              ('raw_final_artifact', 'final_response.json')]:
            path = Path(entry.get(key, ''))
            if (path.resolve() != expected['directory'] / filename or
                    _file_sha256(path) != entry.get(key + '_sha256')):
                raise ValueError('Temporal raw artifact belongs to another parent/window or changed')
        final = json.loads(Path(entry['raw_final_artifact']).read_text(), object_pairs_hook=_unique_json_keys)
        response = json.loads(Path(entry['raw_response_artifact']).read_text(), object_pairs_hook=_unique_json_keys)
        validate_inference_response({'response_audit': [response]}, 0)
        if final.get('content') != raw or response.get('final_content') != raw:
            raise ValueError('Temporal evidence is not the exact validated saved response')
        model_scope = [{key: instant[key] for key in
                        ('frame_ordinal', 'frame_index', 'timestamp_seconds', 'is_anchor')}
                       for instant in entry['window_scope']]
        wrappers.append('{"window_index":' + json.dumps(index) +
                        ',"scope":' + json.dumps(model_scope, ensure_ascii=False) +
                        ',"comparison":' + raw + '}')
    return V26_TEMPORAL_EVIDENCE_PREFIX + '[\n' + ',\n'.join(wrappers) + '\n]'
