"""Task-specific, score-blind observability prompts for the direct video gate.

These instructions are deliberately independent of the generation prompt and
the physical evaluator's output. The caller supplies timestamped video frames,
an optional separately labelled reference image, and its desired JSON schema.
No GPU/model imports or repository-relative resource loads are needed.
"""

from __future__ import annotations

import copy
import hashlib
import json


TASK_PROMPT_VERSION = "task_observability_direct_v4"

SHARED_INSTRUCTIONS = """You are inspecting whether this particular video provides the visible experiment needed for an independent physical measurement. Decide from the supplied timestamped VIDEO images. A separately labelled reference image describes the intended essential setup only; it is not an observed video frame or proof that an event occurred.

First inventory the actual subjects, supports and connections at the beginning, middle and end. Then apply ONLY this task's necessary observations below. Track the original subjects through the useful measurement interval. Preserve any readable interval that supplies the necessary comparison. A late unrelated action, decorative mismatch, small local rendering defect, lighting drift, camera motion with usable scene references, or unrequested pause/ending does not erase already observed evidence. Describe a defect's concrete impact on the required measurement before using it to reject the video.

This is an observability gate, not a physics score. An observed wrong speed, direction, angle, ratio, timing, equilibrium, damping, illumination or material response remains observable. Do not infer mass, charge, current, magnetism, friction, density or a hidden switch from a predicted response. Do not predict whether the physical evaluator will pass, and never use a generator name, existing score or cohort target. Being recognizable is insufficient when the actual measured object, required event or comparison is missing.

Check the whole sampled sequence before claiming that an event never happens. Distinguish an unsupported identity replacement from visible occlusion, out-of-frame motion, ordinary blur, reflection/shadow or a natural material transformation. A detail too small to read is uncertain, not proof that the part disappeared. Do not invent intermediate events across sampling gaps. A positive decision requires the necessary observations to be supported; unresolved sampling/readability limitations must be stated rather than silently converted into completion or a definite physical defect.

Support both positive and negative decisions with actual supplied frame times and concrete visible states. For a rejection, identify the specific missing subject, phase or comparison, what is seen instead, and why the remaining visible interval cannot supply that measurement. A statement such as 'the setup changes' or 'the motion looks wrong' is insufficient. If substantial visual corruption is present while the required measurement is still available, describe that separately and preserve the observable-event finding; do not call the event absent merely to penalize appearance. P4 and P3 additionally have explicit user-required visual rejection rules specified below: those rules can independently reject an otherwise measurable event, with concrete timestamped pixel evidence. Follow the response schema supplied by the caller. Do not introduce extra scoring criteria.
"""


# Each record is intentionally task-specific. Families are dispatch metadata,
# never a fallback prompt: unknown IDs raise rather than borrowing a neighbour.
TASK_PROMPTS = {
    "P2": {
        "family": "single_body_motion",
        "measurement": "An untethered ball's downward trajectory relative to a stable scene reference.",
        "inspect": [
            "Locate the original ball and any supporting rod, clamp or hand. Identify a readable interval in which the ball is visibly separate from that support.",
            "Compare successive ball centers with the same table, wall or other stable reference. Name the actual downward displacement, including slow cumulative descent.",
        ],
        "sufficient": "The same detached ball has multiple readable downward positions relative to the scene, so its trajectory can be measured.",
        "reject_only_if": "No independently moving original ball can be followed, the apparent descent is only camera/apparatus motion, or the ball/reference is unreadable throughout the necessary interval.",
        "guardrails": "Do not require ground impact, absence of impact, an exact last-frame height, motion until the final frame, a particular speed, or gravitational acceleration. Later contact cannot erase a measurable fall.",
    },
    "P4": {
        "family": "single_body_motion",
        "measurement": "One projectile's complete outward, rising and falling trajectory at a readable launch/landing level.",
        "inspect": [
            "Identify one original projectile, its launch location and the reference level; distinguish a shadow, reflection or brief trail from an additional physical ball.",
            "Follow the same projectile through ascent, the apex region, descent and return to the launch-level endpoint. Cite frames for the phases actually visible.",
            "Launch level means vertical height above the reference table/ground, not the same horizontal starting coordinate. Read actual ball centers against the table across time; a decorative direction arrow or drawn trajectory curve is not a physical tether/rail without a visible material connection. Preserve any earlier complete flight even if the ball later bounces or leaves the frame.",
        ],
        "sufficient": "One projectile remains identifiable through a complete readable trajectory and its endpoint at the reference level.",
        "reject_only_if": "A necessary trajectory phase or endpoint is missing/unreadable, or the projectile never independently flies. In addition, the user explicitly requires rejection for multiple actual physical balls or repeated strong background flicker, even if one projectile trajectory remains measurable. Cite the simultaneous distinct balls or specific times/background regions of strong abrupt oscillation; keep this visual rejection separate from the event-observability finding.",
        "guardrails": "Do not require a parabola, a specified launch angle, correct acceleration/range/flight time, a particular post-landing action, or decorative fidelity. An identifiable trajectory may have incorrect physics. Reflections, shadows and ordinary motion trails are not additional physical balls; mild rendering defects and smooth lighting changes are not strong background flicker.",
    },
    "P3": {
        "family": "paired_motion",
        "measurement": "Two original projectiles' independently observable flights and landing positions at a common reference level.",
        "inspect": [
            "Locate each original ball and its launcher. For each, verify actual ball separation; a bending muzzle or moving rod is not ball flight.",
            "Follow both identities across their ascent/apex/descent and locate both landing endpoints. Compare each center with the fixed table rather than with a deforming launcher.",
        ],
        "sufficient": "Both original balls have independently trackable complete flights and readable endpoints at the common level, allowing the two trajectories and ranges to be compared.",
        "reject_only_if": "One original ball stays attached, an extra/replacement object supplies its apparent flight, a needed trajectory or endpoint is unavailable, or rendering prevents distinguishing the two flights. In addition, the user explicitly requires rejection for repeated strong background flicker or clearly severe visual corruption, even if some trajectory measurement is possible. Name the actual times and affected pixel regions or corrupted structures, such as sustained fragmentation/smearing or grossly incoherent rendering; keep this visual rejection separate from the event-observability finding.",
        "guardrails": "Do not require exact synchronization, complementary launch angles, equal initial speeds/ranges, a particular arc-height ordering or the predicted flight-time relationship. Harmless apparatus blemishes, ordinary brief motion blur, mild local defects and smooth lighting changes do not meet the user's strong-flicker/severe-quality rejection rule. Two intended original balls are correct for this task.",
    },
    "P1": {
        "family": "single_body_motion",
        "measurement": "Successive rebound peaks of one original ball above one contact surface.",
        "inspect": [
            "Follow the original ball downward to the surface and check visible contact/reversal, not its shadow alone.",
            "Locate successive complete rebounds of that same ball, including distinguishable peak positions referenced to the same surface.",
            "Inspect any dense opening frames for small, fast low-high-low-high sequences. Continuous single-ball tracking can preserve identity across a color change; color alone is not proof of a replacement. Small early rebounds count when their centers and reference surface are readable, even if unrelated extra balls appear much later.",
        ],
        "sufficient": "The original ball has a readable first encounter with the surface followed by successive upward excursions and rebound peaks, supplying a height comparison relative to that surface.",
        "reject_only_if": "The ball never undergoes the needed rebounds, successive peaks cannot be located, or a second/replacement ball substitutes for the original's rebound sequence.",
        "guardrails": "Equal, increasing or otherwise incorrect rebound heights are measurable physical outcomes. After a readable initial surface encounter, subsequent upward reversals that occur above the surface are incorrect physical trajectories but still provide observable peaks; do not demand perfect surface contact at every later reversal to erase those measurements. Do not require a restitution value, an exact total bounce count beyond the needed comparison, a prescribed pause, or a perfectly still final ball. Compare the actual sequence of low positions and later peaks before claiming there was only one upward excursion.",
    },
    "P5": {
        "family": "paired_motion",
        "measurement": "The motions of two identifiable balls before and after their encounter.",
        "inspect": [
            "Identify both original balls and their common encounter path. Track their centers before the encounter.",
            "Locate the contact/closest encounter region and both balls afterward; verify identity through the encounter rather than assuming a textbook velocity exchange.",
        ],
        "sufficient": "Both original balls and their encounter are visible, with readable pre-contact and post-contact motion for the two-body comparison.",
        "reject_only_if": "The balls never encounter, a ball is substituted or merged beyond identification, or one necessary side of the before/after motion comparison is unavailable.",
        "guardrails": "One or both balls stopping, moving together, reversing, rebounding, or exchanging velocities incorrectly are all observable post-contact outcomes. Zero post-contact velocity is not a missing post-contact phase. Compare each original ball center against the fixed background at early, middle and late post-contact times before calling it stationary; unchanged distance between the balls does not mean their centers are stationary. Do not infer equal hidden masses or judge momentum, energy, speed ratios or perfect head-on alignment here.",
    },
    "P8": {
        "family": "rolling",
        "measurement": "Translation and visible rotation of one patterned ball along an incline.",
        "inspect": [
            "Identify the original patterned ball, ramp contact region and connected runout. Follow its movement along the inclined track rather than straight off its side.",
            "Track the ball center and an attached visible pattern through several positions. Distinguish a changing texture from a coherent marker rotating with the ball.",
        ],
        "sufficient": "A useful interval along the ramp has readable ball translation and an associated pattern/orientation, allowing rotation and translation to be compared.",
        "reject_only_if": "The original ball does not traverse the incline, falls directly off it instead, or no attached pattern/orientation can be followed during the movement.",
        "guardrails": "Do not enforce no-slip rolling, correct acceleration or a predicted angular/linear-speed ratio. A readable pattern that does not rotate is an observable response. Gate motion and a later runout blemish do not invalidate an earlier readable ramp interval.",
    },
    "P9": {
        "family": "rolling",
        "measurement": "The motions of a sphere and a ring down a common inclined track.",
        "inspect": [
            "Identify the sphere and the open ring separately, including their outlines and contact with the track.",
            "Follow both original subjects over a common movement interval and locate comparable track positions; read any visible rotation without predicting the winner.",
        ],
        "sufficient": "Both subject identities and their progress along the same incline are readable for a common motion/arrival comparison.",
        "reject_only_if": "One subject is missing or replaced, the two outlines cannot be distinguished, or a necessary trajectory/comparable track interval is unavailable.",
        "guardrails": "Do not require the expected winner, equal releases, a specific arrival-time ratio, the correct accelerations or no-slip kinematics. A clearly observed unexpected ordering belongs to physical evaluation.",
    },
    "P15": {
        "family": "oscillation",
        "measurement": "Periods of two equal-length pendulums with distinguishable small initial amplitudes.",
        "inspect": [
            "Locate both pivots, strings and original bobs; establish their visible lengths and distinguishable starting amplitudes from the video/reference geometry.",
            "Follow each bob through back-and-forth cycles, identifying corresponding turning or crossing positions at actual times in a common interval.",
        ],
        "sufficient": "Both original bobs remain trackable through complete cycles, so their periods can be compared across the distinguishable amplitudes.",
        "reject_only_if": "A bob/string/pivot needed for the comparison is missing, the intended paired setup is unavailable, or only isolated poses/incomplete swings are visible without a readable cycle.",
        "guardrails": "Do not require equal periods, phase locking, exact simultaneous release, textbook sinusoidal motion or an exact small-angle value. Ordinary smooth bob displacement and string rotation are the event.",
    },
    "P12": {
        "family": "oscillation",
        "measurement": "Periods of two distinguishable pendulum bobs on equal visible lengths.",
        "inspect": [
            "Identify the two bobs, their strings and pivots. Preserve each bob's identity even if their sizes or visual styles differ.",
            "Locate full back-and-forth cycles of each original bob and corresponding turning/crossing times within the same observation window.",
        ],
        "sufficient": "The two original pendulums have readable full cycles and visible string geometry for comparing their periods.",
        "reject_only_if": "One pendulum cannot be followed through a cycle, its essential connection is absent/unreadable, or the paired comparison is replaced by another motion.",
        "guardrails": "Mass cannot be verified from pixels. Do not require visual proof of mass, correct mass independence, equal periods, equal bob sizes or continuing synchronization. Compare each bob center against its fixed support at several different times before calling it static. A damped cycle need not return to the exact original amplitude. Preserve an earlier readable cycle even if the bob later stops or the image deteriorates.",
    },
    "P13": {
        "family": "oscillation",
        "measurement": "Periods of two equal-length pendulums with distinguishable initial swing amplitudes, including a large swing.",
        "inspect": [
            "Identify both original bobs, pivots and equal-length suspension geometry; locate their different initial displacement angles.",
            "Follow the small and large swings through complete returns to corresponding phase positions, distinguishing actual cycles from a single outward sweep.",
            "Before calling either pendulum static, compare its actual bob center against the fixed support posts/pivot at at least three separated supplied video times. Describe the center's position at each time and inspect additional times if the sampled phase happens to repeat. An unchanged support frame or similar string appearance does not establish that its bob is stationary.",
        ],
        "sufficient": "Both amplitudes and complete cycles remain readable over a common interval, allowing a period comparison.",
        "reject_only_if": "An amplitude or pendulum identity is unavailable, the paired setup is missing, or one motion never provides a complete readable cycle.",
        "guardrails": "Do not enforce a large-angle correction, predicted period ordering, exact release synchronization or matching decay. A wrong but measurable period relationship is still observable. A damped cycle need not return to the exact original amplitude; corresponding repeated turning or crossing phases identify a cycle even with amplitude decay. Earlier readable full cycles remain valid if the bobs stop or the scene degrades later.",
    },
    "P14": {
        "family": "oscillation",
        "measurement": "Periods of two pendulums with visibly different string lengths.",
        "inspect": [
            "Locate the two pivots, strings and bobs and establish the visible length difference without relying on bob height alone.",
            "Track complete cycles of each original bob and identify phase-matched positions at actual supplied times.",
        ],
        "sufficient": "The length difference and both full oscillation cycles are available for a period-versus-length comparison.",
        "reject_only_if": "The two suspension lengths cannot be distinguished, a bob/connection is missing, or one complete cycle cannot be followed.",
        "guardrails": "Do not require the longer pendulum to be slower or enforce a square-root relation. Unequal amplitudes, loss of synchronization or incorrect period ratios are not themselves missing observations.",
    },
    "P11": {
        "family": "single_body_motion",
        "measurement": "The outward and return motion of one block on the same incline.",
        "inspect": [
            "Identify the original block and fixed inclined track. Locate its movement up the track.",
            "Follow the same block through the turning region into motion back down the track; cite actual positions on both sides of the reversal.",
            "Compare the block center against the fixed lower and upper ends at early, middle and late times, including late frames after any pause. A reversal inside the track is valid; the block does not need to touch an endpoint or leave the frame to reverse. A clear return lasting several sampled positions is usable even before it reaches the original start.",
        ],
        "sufficient": "The up-ramp segment, connecting reversal and down-ramp segment are readable for the same block.",
        "reject_only_if": "The block only translates one way, leaves view before a usable turn/return, is replaced, or the block/track relationship is unreadable during a necessary segment.",
        "guardrails": "Do not require symmetric travel times, a particular friction value, correct accelerations or an exact stopping pause. The reversal need not coincide with one sampled frame when adjacent readable positions establish both connected branches. Readable upward and downward segments of the same block joined by the turn are sufficient: the return need not reach or pass the original starting position. Judge only the supplied clip; do not invent a loop, repetition or future continuation beyond its end, and do not erase an already visible turn/return because its last pose resembles its first.",
    },
    "P17": {
        "family": "optics",
        "measurement": "Connected incident and outgoing ray geometry at a liquid interface.",
        "inspect": [
            "Locate the liquid interface, normal or equivalent orientation reference and incident ray.",
            "Associate the visible outgoing continuation with the same interface junction; inspect the ray arms and geometry in a readable stationary or moving configuration.",
            "A visible straight ray that crosses the liquid surface still supplies an air-side arm, a water-side arm and their common crossing. Zero visible bending is a measurable angle relationship; it is not a missing outgoing arm. Locate the surface crossing before deciding whether both portions are present.",
        ],
        "sufficient": "Both ray arms and their common interface point are readable relative to the interface, providing the angle comparison.",
        "reject_only_if": "A necessary ray arm or interface junction is absent, disconnected beyond association or unreadable, so the angle comparison cannot be formed.",
        "guardrails": "A static configuration is sufficient. Do not require source switching, motion, the correct bending direction, refractive index or Snell-law angles. Schematic rays and normal lines are intentional graphics, not extra physical bodies.",
    },
    "P20": {
        "family": "optics",
        "measurement": "Four distinguishable beam/interface configurations across changing incident angles.",
        "inspect": [
            "Locate the four separate interface junctions: three incident paths from the air side and one from the water side. Keep each ray associated with its own junction.",
            "Compare earlier and later angles at those junctions and read the continuation/reflected branches actually present on either side; explicitly record an absent transmitted branch when the water-side response is reflection.",
        ],
        "sufficient": "The four beam identities, interface geometry and a readable variation of their incidence conditions allow the shown responses to be compared.",
        "reject_only_if": "The required beam configurations cannot be identified, changing incidence conditions are absent/unreadable, or branches cannot be associated with their interface points.",
        "guardrails": "Do not enforce the correct branch, critical angle or refraction/reflection law. A straight ray, wrong bend or fully reflected water-side beam is an observable response. Do not require a transmitted air branch in every condition, equal-duration stages or exact interface styling.",
    },
    "P18": {
        "family": "optics",
        "measurement": "Incident and outgoing ray angles at one mirror junction.",
        "inspect": [
            "Locate the mirror, common ray junction and the normal or equivalent mirror orientation reference.",
            "Follow both incident and outgoing ray arms to that junction, distinguishing a normal annotation from a third ray.",
        ],
        "sufficient": "A readable connected ray/mirror configuration supplies both angles, even if the video is stationary.",
        "reject_only_if": "An essential ray arm, mirror reference or connected junction is absent/unreadable, preventing the angle comparison.",
        "guardrails": "Do not require equality of angles, ray motion, source switching or photorealistic beam rendering. A visibly wrong reflection angle remains measurable.",
    },
    "P19": {
        "family": "optics",
        "measurement": "The common-scene geometry of four rods and their associated shadows from one point-like source.",
        "inspect": [
            "Identify the source/reference geometry and four rod bases. Locate each shadow's path and tip rather than counting a shadow as a new rod.",
            "Associate every needed shadow with its own rod base in a shared readable configuration.",
        ],
        "sufficient": "All four rod-shadow associations, bases and endpoints are readable for the projection comparison; a static scene suffices.",
        "reject_only_if": "A needed rod/shadow pair or its association cannot be read, or the shared scene/reference geometry is unavailable.",
        "guardrails": "Do not judge radial convergence, shadow-length ratios or correct perspective here. Rod colors, background colors, decorative markers and visible motion are not required.",
    },
    "P16": {
        "family": "geometry",
        "measurement": "Four fixed collinear markers on the same moving rigid rod relative to a wall and floor.",
        "inspect": [
            "Locate the original rod, its two ends and four distinguishable marker centers, keeping marker order and rod attachment clear.",
            "Compare distinguishable rod poses while relating the endpoints to wall/floor references. Check whether markers move with the rod or visibly slide, swap, appear or disappear.",
            "Before claiming the rod is static, compare both endpoint positions against the fixed wall corner and floor trim in the first, middle and last video frames or comparison panels. Slow cumulative endpoint displacement and small angle changes are real pose changes even when the overall room composition looks unchanged.",
        ],
        "sufficient": "Four original fixed markers and the rod geometry remain readable through distinguishable poses, enabling their geometric comparison.",
        "reject_only_if": "A marker or rod/reference geometry is unavailable, marker substitution prevents correspondence, or there is no readable pose comparison.",
        "guardrails": "Do not enforce a cross-ratio value, a particular speed or expected endpoint trajectory. Perspective changes are not marker sliding; explain the actual correspondence failure before rejecting.",
    },
    "P23": {
        "family": "single_body_motion",
        "measurement": "A geometric direction comparison: a detached ball falls BESIDE the liquid container, while the separate liquid surface supplies a simultaneously readable direction reference. Entry into the liquid is not part of this task.",
        "inspect": [
            "Identify the bare original ball beside the liquid container and separate it from any support.",
            "Track downward ball positions over a useful interval while also locating the liquid surface and stable scene references at those times.",
            "Inspect the entire gap from holder to ball for a visible string/chain and follow the original ball through release. If a tethered original and a newly appearing free ball coexist, the new ball does not prove that the original was released. Distinguish this actual connection/identity issue from the irrelevant question of whether any ball enters the water.",
        ],
        "sufficient": "The same detached ball has a readable downward trajectory and the liquid surface direction is simultaneously readable for the directional comparison.",
        "reject_only_if": "The ball's free trajectory is missing/unreadable, motion is only apparent camera/support motion, or the free-surface direction cannot be read during the usable fall.",
        "guardrails": "Do not require perpendicularity, correct acceleration, a fall into the liquid, impact, absence of impact or a prescribed endpoint. A crooked measurable trajectory remains a physical result.",
    },
    "P21": {
        "family": "static_liquid",
        "measurement": "The two surface heights in an identifiable connected U-shaped liquid vessel.",
        "inspect": [
            "Locate both arms and the visible connecting vessel geometry; distinguish a connected system from two unrelated liquid columns.",
            "Read the two free surfaces against a common vertical/scene reference in a settled comparison interval.",
        ],
        "sufficient": "Both connected arms and readable surface heights are available in one equilibrium configuration, including an already stationary scene.",
        "reject_only_if": "The connecting setup or one liquid surface is unavailable/unreadable, so the two surface levels cannot be compared.",
        "guardrails": "Do not require equal heights, a correct pressure relation, an initial disturbance or visible settling. A persistent height difference is an observable physical outcome.",
    },
    "P22": {
        "family": "static_liquid",
        "measurement": "The immersion geometry of one intact rectangular ice block relative to the liquid surface.",
        "inspect": [
            "Locate the original block's full outline and the waterline through/around it in the transparent container.",
            "Identify a readable settled position and distinguish the block boundary from glare, reflections and the back container wall.",
        ],
        "sufficient": "The complete block and liquid surface provide a readable immersion comparison in a stable interval; an initially stable state is valid.",
        "reject_only_if": "The block is missing/replaced or its complete outline/waterline cannot be read well enough to establish immersion geometry.",
        "guardrails": "Do not require a correct immersed fraction, visual proof of density, textbook floating behavior or a transient settling motion. A clearly visible unexpected position is still a response to measure.",
    },
    "P25": {
        "family": "phase_change",
        "measurement": "The before/after waterline when the original floating ice actually melts completely.",
        "inspect": [
            "Identify the original solid ice outline and initial waterline relative to the container.",
            "Follow progressive loss of the original solid into liquid and verify its final absence in the container; distinguish melting from an intact block sinking, drifting out of sight or being hidden by glare.",
            "Locate the final waterline using the same container reference.",
        ],
        "sufficient": "The original ice's complete solid-to-liquid transition and both waterlines are readable for the before/after comparison.",
        "reject_only_if": "The original solid remains intact or unaccounted for, the complete transformation is not observed, or one waterline needed for comparison is unreadable.",
        "guardrails": "Do not require unchanged water height, correct buoyancy while melting or a prescribed melt rate. Natural shrinking, fragmentation and changing outlines can be the transformation, not identity defects, when the original material remains accounted for.",
    },
    "P26": {
        "family": "phase_change",
        "measurement": "Waterline change across complete melting of an original ice body containing a stone and release of that stone.",
        "inspect": [
            "Identify the original ice boundary, embedded stone and initial waterline. Establish the stone's visible association with the original ice.",
            "Track actual complete melting of that ice and release of its original stone, then read the final stone position and waterline.",
        ],
        "sufficient": "The original contained stone, complete ice disappearance through melting, stone release and before/after waterlines are visibly associated.",
        "reject_only_if": "An intact ice body only translates/sinks, a new stone substitutes for the embedded stone, original solid remains/unaccounted for, or a necessary initial/final comparison is unreadable.",
        "guardrails": "Do not enforce a predicted water-level decrease, stone sink rate or correct buoyancy. A stone at an unexpected but readable final location is observable; the crucial issue is whether it is the original released stone.",
    },
    "P27": {
        "family": "phase_change",
        "measurement": "The initial and final liquid levels across complete melting of the original ice body in the container.",
        "inspect": [
            "Locate the original solid ice and initial liquid surface against a fixed container reference.",
            "Follow reduction of the original solid into liquid and check for its complete disappearance at the earliest clear completion interval; distinguish disappearance through sinking/occlusion from melting.",
            "Read the liquid surface at that completion interval with the same geometric reference. A later added block or new poured material does not retroactively turn already melted original ice back into remaining original solid.",
        ],
        "sufficient": "The complete original ice-to-liquid transformation and its before/after liquid surfaces can be followed.",
        "reject_only_if": "The solid remains intact/unaccounted for, complete melting is unavailable, or the initial/final level comparison is unreadable.",
        "guardrails": "Salinity and density are not visually measurable here. Do not require the expected salt-water level change, correct floating response, a particular mixing pattern or melting rate.",
    },
    "P24": {
        "family": "phase_change",
        "measurement": "The original water volume before and after its complete transition to solid ice.",
        "inspect": [
            "Locate the initial liquid volume and its surface relative to the container.",
            "Follow the evolving solid boundary through that original volume and verify a final fully solid state. Check for original liquid remaining below an added ice cap or separate patch.",
        ],
        "sufficient": "The same original water volume has readable initial liquid and completely frozen final states, with comparable surface/container geometry.",
        "reject_only_if": "The original water remains liquid, only a separate piece of ice appears, complete freezing is not established, or initial/final geometry cannot be associated.",
        "guardrails": "Do not require the correct expansion ratio, surface-height change, a prescribed freezing front or realistic elapsed freezing time. A measurable wrong final volume belongs to physics scoring. Do not invent a chemical reaction or added reagent solely because an internal front turns the water opaque or white. A visibly advancing internal transformation through the original volume into a stable solid-looking final body can establish the task without a particular crystal texture; actual external pouring or inserted preformed solids must be separately identified from frames.",
    },
    "P28": {
        "family": "phase_change",
        "measurement": "The relative melting progress of compact ice versus fragmented ice in two distinguishable containers, comparing meltwater levels over common times.",
        "inspect": [
            "Identify each original ice population and its container: compact body versus fragments. Track their remaining solids separately.",
            "Compare actual progression of solid loss and the two liquid levels at common video times. Locate readable intervals of relative meltwater-level evolution; some original solid may remain when the clip ends.",
        ],
        "sufficient": "Both original ice populations undergo identifiable melting with readable intermediate solids/liquid-level changes over common times, providing a relative-progress comparison. Complete melting of both is not necessary.",
        "reject_only_if": "One population has no identifiable melting/progress, apparent melting is only displacement/occlusion, the two identities mix, or the common meltwater-level comparison is unreadable throughout.",
        "guardrails": "Do not require fragments to melt faster, a predicted rate ratio, equal final heights or a particular video duration. Fragmentation, shrinking and coalescing meltwater are natural transformation observations when identifiable.",
    },
    "P31": {
        "family": "static_geometry",
        "measurement": "Two suspended balls' equilibrium positions and complete string geometry relative to their support.",
        "inspect": [
            "Locate each original ball, its own string and the string's attachment to the support.",
            "Read both ball centers and the full suspension geometry in a common stable configuration.",
        ],
        "sufficient": "The two balls, both suspension connections and their equilibrium geometry are readable; no motion is necessary.",
        "reject_only_if": "A ball/string/attachment required to establish the paired geometry is absent or unreadable, or substituted objects prevent the comparison.",
        "guardrails": "Do not infer charge or mass from appearance, require visible repulsion, enforce symmetry or require equal separation angles. An asymmetric readable equilibrium remains measurable.",
    },
    "P32": {
        "family": "electromagnetic",
        "measurement": "The orientations of two original compass needles relative to their fixed housings and central conductor arrangement.",
        "inspect": [
            "Locate each actual needle body and pivot; a readable circular housing or central dot alone does not establish a readable needle orientation.",
            "Read the two orientations in a shared comparison state and follow any visible rotation without inventing a current-switching event.",
        ],
        "sufficient": "Both original needles have distinguishable orientations relative to the conductor/housings in a common interval, including a stationary state.",
        "reject_only_if": "A needle is missing/unreadable through the needed comparison, the conductor/needle setup is unavailable, or substitution prevents associating the original orientations.",
        "guardrails": "Do not require visible current activation, any rotation, opposite directions or predicted deflection angles. Mere housing clarity is insufficient if the main needles cannot be located.",
    },
    "P29": {
        "family": "electromagnetic",
        "measurement": "The responses of two separately identifiable original bodies side by side on the same inclined plate. Match their actual shape to the reference: short cylindrical pucks/discs are valid subjects; rectangular blocks are not required.",
        "inspect": [
            "Identify the two reference-matched original bodies, including silver short cylinders/pucks where shown, their boundaries and contacts with the common incline.",
            "Compare successive original-body positions against the plate during the same time interval, explicitly recording actual displacement or clearly observed stationarity for each. Additional later objects do not replace original bodies that remain separately visible and trackable.",
        ],
        "sufficient": "Both original-body positions/responses are readable against the common plate over a usable comparison interval, including clearly trackable bodies that remain still.",
        "reject_only_if": "One original body is actually missing/replaced, the shared incline setup is lost, or its boundaries/positions cannot be followed during any usable comparison interval.",
        "guardrails": "Do not require the expected faster block, any minimum speed difference, a visible release mechanism or proof of hidden magnets/equal masses. Small cumulative displacement is motion; visible stationarity is not invisibility.",
    },
    "P33": {
        "family": "electromagnetic",
        "measurement": "The positions of separate closed and gapped loose rings relative to their respective fixed coil/core assemblies.",
        "inspect": [
            "Locate each loose ring's outline, the visible closed-versus-gapped distinction, and its own coil/core and support.",
            "Match the original rings to the reference: the silver annuli resting on top of the copper coils are the ring subjects, including a visible slit in one annulus. They need not float in midair to exist as distinct subjects. Compare their outlines and positions in the earliest readable common interval, including both remaining at rest.",
            "Track the original rings relative to those cores during a common comparison interval; distinguish a separate ring moving from the whole core/support lifting or stretching.",
        ],
        "sufficient": "The two original loose rings and their distinct outlines remain identifiable relative to their respective fixed cores, so each actual response can be compared.",
        "reject_only_if": "A separate original ring or the visible gap distinction cannot be located, an apparatus deformation/replacement substitutes for ring motion, or the ring/core relationships are unreadable.",
        "guardrails": "Neither ring is required to jump, and no predicted difference/current activation may be assumed. A clearly stationary ring is an observed response. Do not treat an unreadable gap as definite closure; report the actual visibility limitation.",
    },
    "P34": {
        "family": "electromagnetic",
        "measurement": "Changes between oscillation cycles of one solid plate and one slotted plate on distinct pendulum suspensions.",
        "inspect": [
            "Locate both plate outlines, the visible slots and the original suspension connections.",
            "Follow each plate through back-and-forth motion over a common interval; locate successive turning positions to compare change between cycles.",
        ],
        "sufficient": "Both original plates, their solid/slotted distinction and repeated turning positions remain readable for the cycle-to-cycle comparison.",
        "reject_only_if": "One plate or suspension cannot be followed, plate identity/slots are lost, or the sampled motion lacks the necessary readable oscillation-cycle comparison.",
        "guardrails": "Do not require the predicted faster damping plate, a correct decay rate or equal periods. Ordinary rotation and perspective-dependent silhouettes are not rigid deformation by themselves.",
    },
    "P30": {
        "family": "electromagnetic",
        "measurement": "The lamp's visible state during one original magnet's observed entry into and movement relative to a fixed coil.",
        "inspect": [
            "Identify the original magnet, fixed coil opening, leads and lamp. Locate original-magnet positions outside and entering/inside the coil.",
            "Follow the same magnet's actual path while reading the lamp over the common timeline, including a lamp that stays dark. Entry followed by withdrawal through the same side is a usable response; a full transit to the far side is not necessary.",
        ],
        "sufficient": "Original-magnet entry/movement at the coil and the simultaneous lamp state are readable and associated in time, allowing the actual response to be assessed.",
        "reject_only_if": "The original magnet never enters or moves at the coil opening, an unrelated replacement movement substitutes for it, or the original magnet/coil opening/lamp cannot be read during a usable response interval.",
        "guardrails": "Do not require correct illumination, polarity, brightness timing, a far-side exit, a particular hand grip/path or prescribed stationary waiting. Same-side withdrawal and a dark lamp are observed outcomes, not absent experiments.",
    },
    "P35": {
        "family": "granular",
        "measurement": "The settled slope profiles of two differently sized piles of the same granular material.",
        "inspect": [
            "Locate both separate piles, their bases and a common supporting-surface orientation.",
            "Read the face profiles and outer boundaries in a settled comparison, whether formed during the clip or already stable.",
        ],
        "sufficient": "Two distinct-size piles have readable settled profiles and supporting references for comparing slopes.",
        "reject_only_if": "One pile or its face/base boundary is unavailable, the piles are not separately identifiable, or no settled profile can be read.",
        "guardrails": "Do not require equal repose angles, scale invariance, visible feeders, continued pouring or a particular grain texture when both slopes are readable.",
    },
    "P36": {
        "family": "granular",
        "measurement": "The discharge and material-level evolution of a liquid system and a granular system over a common interval.",
        "inspect": [
            "Identify the two separate funnel/container systems, their original contents, outlet regions and emitted streams.",
            "Compare actual discharge and successive internal levels for each system over the same times; distinguish flow of original material from an unrelated moving object or cosmetic stream.",
        ],
        "sufficient": "Both systems show identifiable original material discharging with readable level evolution/outlets, enabling a common flow comparison.",
        "reject_only_if": "One system/material/outlet is unavailable, actual discharge is not observed, or the necessary level/stream comparison cannot be read.",
        "guardrails": "Do not require a correct head dependence, the expected system to empty first, equal flow rates or a perfectly empty final frame after usable discharge comparisons already exist.",
    },
    "P6": {
        "family": "contact",
        "measurement": "The responses of two original blocks to an actual increase in one hinged board's inclination: sliding onset where present, or clearly observed non-sliding where absent.",
        "inspect": [
            "Identify both blocks and their positions relative to the same board rather than to the image edge.",
            "Verify an actual change in board angle against a fixed scene reference. Then track both original blocks relative to that board: locate sliding onset if visible, or explicitly record that one/both remain stationary on the tilted board. Stationarity relative to a tilting board is a readable response.",
        ],
        "sufficient": "The board visibly tilts and both original blocks' responses relative to it are readable over a common interval, including one or both blocks never sliding.",
        "reject_only_if": "The board never visibly tilts, an original block is lost/replaced before a usable tilt-response comparison, or board/block geometry is unreadable throughout that comparison. Missing sliding alone is not a rejection.",
        "guardrails": "Do not enforce equal onset angles, a friction coefficient, mass independence, actual sliding by either block, or perfectly monotonic actuator speed. Distinct onset angles and both blocks remaining fixed on a visibly tilting board are observable outcomes. A late scene break cannot erase an earlier readable tilt-response interval.",
    },
    "P10": {
        "family": "contact",
        "measurement": "One block's tipping transition about a support region under a contacting actuator pad.",
        "inspect": [
            "Locate the original block outline, actuator-pad contact, supporting surface and lower support/pivot region.",
            "Follow the same block from a supported upright pose through actual rotation into a tipped pose; distinguish translation or pad motion from block tipping.",
        ],
        "sufficient": "The block, contact and support region remain readable through an identifiable upright-to-tipped transition.",
        "reject_only_if": "Only translation/pad motion occurs, the original block is substituted, or the support/contact/outline is unreadable during the necessary rotation.",
        "guardrails": "Do not enforce a tipping threshold, center-of-mass relation, correct angular trajectory or a decorative final pause. A readable unexpected pivot/trajectory is available to the physical evaluator.",
    },
    "P7": {
        "family": "static_geometry",
        "measurement": "The full hanging profile of one connected chain between two fixed endpoints.",
        "inspect": [
            "Locate both distinct endpoint attachments and follow the same continuous chain between them.",
            "Read the settled curve against the scene reference, checking that apparent gaps are not occlusion or small-scale link texture.",
        ],
        "sufficient": "Both attachments and the full connected settled chain profile are readable in a stable configuration.",
        "reject_only_if": "The chain or an endpoint is missing/unreadable, continuity cannot be established, or no usable settled profile is visible.",
        "guardrails": "Do not require a catenary equation, a particular sag, an initial settling transient or an exact final pause. The chain is flexible; natural changes in curvature are not rigid-body deformation.",
    },
    "P37": {
        "family": "static_liquid",
        "measurement": "Internal meniscus heights in two different-bore capillary tubes relative to one shared liquid reservoir.",
        "inspect": [
            "Identify both tube bores and establish that their lower ends access the shared reservoir; isolated prefilled columns are a different setup.",
            "Read each internal liquid column/meniscus and the external reservoir surface at common times, explicitly allowing a stationary column or clearly absent rise.",
        ],
        "sufficient": "The different-bore immersed tubes and both internal/external level relationships are readable for comparison.",
        "reject_only_if": "Shared liquid access cannot be established, tube identities/bore distinction are unavailable, or an internal/external surface needed for the comparison cannot be read.",
        "guardrails": "Do not require the narrow tube to rise higher, any nonzero rise, inverse-radius scaling, initially dry tubes or repeated insertion. Wetting chemistry and liquid composition cannot be inferred from the expected rise.",
    },
    "P39": {
        "family": "surface_geometry",
        "measurement": "The shared partition and outer contours of two differently sized connected bubbles.",
        "inspect": [
            "Locate both original outer bubble contours and their visible size difference.",
            "Follow the common internal partition between them and distinguish that interface from a reflection or an unrelated third contour.",
        ],
        "sufficient": "The original two bubbles' outer boundaries and full shared partition are readable together in a connected state; a stationary configuration suffices.",
        "reject_only_if": "A bubble or the common partition is absent/unreadable, the two connected identities cannot be associated, or an unrelated object replaces the intended partition geometry.",
        "guardrails": "Do not enforce partition curvature, its predicted bending direction, radius ratios or visible motion. An incorrectly curved but readable partition remains measurable.",
    },
    "P40": {
        "family": "surface_transformation",
        "measurement": "Two original separate liquid drops joining into one connected final liquid body.",
        "inspect": [
            "Locate both initial separate drops and their outlines; preserve their material identities as they approach.",
            "Follow actual contact and connection into one body, then verify a readable final single-drop boundary. Two drops touching with two persistent disconnected outlines do not establish completed coalescence.",
        ],
        "sufficient": "The original two-drop state, the joining transition and the final single connected drop are visibly associated.",
        "reject_only_if": "The drops merely approach or remain separate, one original is lost/replaced, or the final connected single body cannot be established.",
        "guardrails": "Do not require volume conservation, a radius ratio, expected merger speed or photorealistic styling. Count changing from two to one is the required natural transformation, not an identity defect by itself.",
    },
    "P38": {
        "family": "paired_motion",
        "measurement": "Speed evolution of two differently sized original balls descending in the same transparent liquid tank.",
        "inspect": [
            "Identify both balls, their radius distinction and the tank references; separate each actual ball from its reflection or trail.",
            "Track both original ball centers during a common descent interval with enough successive positions to compare how their speeds change.",
            "Inspect the earliest shared descent before either ball reaches the tank bottom. Later spawned objects, collisions or image corruption cannot erase that interval. Do not say the original balls were replaced at a time when their two distinguishable original outlines remain visible below the new objects.",
        ],
        "sufficient": "Both different-size balls have readable successive descending positions in the same tank over a useful shared speed-comparison interval.",
        "reject_only_if": "One ball/radius identity is unavailable, descent is not observed, or trajectories end/disappear before a usable common speed-evolution comparison.",
        "guardrails": "Do not require exact terminal-speed attainment, constant speed, a radius-squared speed ratio, the expected faster ball or visual proof of liquid viscosity/material density. Wrong measured speeds remain physical results.",
    },
}


def task_profile_for(task_id: str) -> dict:
    """Return an isolated profile; fail closed on unreviewed task identifiers."""
    if task_id not in TASK_PROMPTS:
        raise ValueError(f"No dedicated consistency prompt for task {task_id!r}")
    return copy.deepcopy(TASK_PROMPTS[task_id])


def task_prompt_for(task_id: str) -> str:
    """Return the complete score-blind instruction for one task, without schema."""
    profile = task_profile_for(task_id)
    steps = "\n".join(f"{index}. {step}" for index, step in enumerate(profile["inspect"], 1))
    return (
        SHARED_INSTRUCTIONS
        + f"\nTASK {task_id}: {profile['measurement']}\n"
        + "Specific inspection sequence:\n" + steps
        + "\nSufficient observable evidence: " + profile["sufficient"]
        + "\nGrounds for an observability rejection: " + profile["reject_only_if"]
        + "\nTask-specific boundaries: " + profile["guardrails"]
    )


def task_prompt_audit(task_id: str) -> dict:
    """Small serializable provenance record suitable for each request artifact."""
    prompt = task_prompt_for(task_id)
    return {
        "task_id": task_id,
        "task_prompt_version": TASK_PROMPT_VERSION,
        "task_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "task_prompt": prompt,
        "generation_prompt_sent_to_judge": False,
        "physical_score_sent_to_judge": False,
    }


def prompt_inventory() -> dict:
    """Export exact prompt text and profiles for independent review/replay."""
    return {
        "version": TASK_PROMPT_VERSION,
        "task_count": len(TASK_PROMPTS),
        "shared_instructions": SHARED_INSTRUCTIONS,
        "tasks": {
            task_id: {"profile": task_profile_for(task_id), **task_prompt_audit(task_id)}
            for task_id in TASK_PROMPTS
        },
    }


if __name__ == "__main__":
    print(json.dumps(prompt_inventory(), ensure_ascii=False, indent=2))
