"""Frozen task protocols used by reliability controls and VLM comparisons.

The protocol separates four questions that used to be mixed together:
whether the experimental conditions are valid, whether the requested event
actually happened, whether the quantities can be measured, and whether the
measured physical relation is satisfied.
"""

from copy import deepcopy


TASK_PROTOCOLS = {
    'P14': {
        'phenomenon': "Pendulum period and length",
        'domain': "Mechanics",
        'applicable_conditions': [
            "Both pendulums are visible from the same fixed viewpoint",
            "Pendulum lengths remain fixed throughout the video",
            "Amplitudes are small enough for the small-angle approximation",
        ],
        'required_event': "Both pendulums complete at least three distinguishable full oscillations",
        'quantities': ['L1', 'L2', 'T1', 'T2'],
        'constraint': '(T1/T2)^2 = L1/L2',
        'residual': 'abs(((T1/T2)^2)/(L1/L2)-1)',
        'insufficient_evidence': ["Insufficient oscillation cycles", "Severe occlusion", "Pendulum tracking failed", "The viewpoint does not allow comparison of pendulum lengths"],
        'pass_threshold': 0.80,
    },
    'P18': {
        'phenomenon': "Reflection of light",
        'domain': "Optics",
        'applicable_conditions': [
            "The mirror and incident and reflected rays lie in the same measurable two-dimensional plane",
            "The normal direction and ray endpoints are distinguishable",
        ],
        'required_event': "The incident ray reaches the mirror and produces a distinguishable reflected ray",
        'quantities': ['theta_incident', 'theta_reflected', 'normal_direction'],
        'constraint': 'theta_incident = theta_reflected',
        'residual': 'abs(theta_incident-theta_reflected) / 90deg',
        'insufficient_evidence': ["The mirror or normal is not visible", "Severely blurred rays", "The assumptions for two-dimensional measurement are not satisfied"],
        'pass_threshold': 0.80,
    },
    'P25': {
        'phenomenon': "Water-level change as floating ice melts",
        'domain': "Thermal physics / Buoyancy",
        'applicable_conditions': [
            "The vessel, liquid, floating ice, and waterline are visible for most of the video",
            "The vessel does not visibly tilt and no liquid is added externally",
        ],
        'required_event': "The floating ice continuously melts from the initial to the final state, with interpretable evidence of any remaining ice",
        'quantities': ['initial_ice_fraction', 'final_ice_fraction', 'initial_level', 'final_level'],
        'constraint': "The water level remains approximately unchanged as floating ice melts in fresh water",
        'residual': 'abs(final_level-initial_level) / container_height',
        'insufficient_evidence': ["Persistent occlusion of the ice or waterline", "The video is too short to determine whether melting occurred", "The vessel boundary is not visible"],
        # Pilot calibration: the correct control scores about .70 with the
        # current level extractor, so .65 accepts it while retaining a margin
        # for measurement noise.
        'pass_threshold': 0.65,
    },
    'P40': {
        'phenomenon': "Volume conservation during droplet coalescence",
        'domain': "Surface tension",
        'applicable_conditions': [
            "The outlines of both initial droplets and the approximately spherical merged droplet are visible",
            "The camera scale remains approximately fixed throughout the video",
        ],
        'required_event': "The two droplets touch and merge into one connected droplet",
        'quantities': ['r1', 'r2', 'rf', 'merge_frame'],
        'constraint': 'rf^3 = r1^3 + r2^3',
        'residual': 'abs(rf^3/(r1^3+r2^3)-1)',
        'insufficient_evidence': ["The droplet outline cannot be segmented", "The merging frame is ambiguous", "Severe occlusion or motion out of frame"],
        # Pilot calibration: the known correct control scores about .56 for
        # M1, while the 10% and 30% volume deviations score lower.
        'pass_threshold': 0.50,
    },
}


def get_protocol(task_id):
    """Return a defensive copy so callers cannot mutate the frozen protocol."""
    return deepcopy(TASK_PROTOCOLS.get(str(task_id), {
        'phenomenon': str(task_id),
        'domain': 'unknown',
        'applicable_conditions': [],
        'required_event': 'task-specific event',
        'quantities': [],
        'constraint': 'task-specific physical relation',
        'residual': None,
        'insufficient_evidence': ['required quantities cannot be measured reliably'],
        'pass_threshold': 0.80,
    }))
