"""Dependency-free public defaults, matching the task evaluator specifications."""
PRINCIPLES={
'P33':"Fix ring identities using the first-frame gap; compare maximum rises relative to initial positions, normalized by fixed initial outer diameters.",
'P34':"Independently count complete oscillations of solid and slotted plates using a shared visibility window and amplitude threshold.",
'P30':"Compare induction-active intervals associated with visible magnet motion against the light temporal centroid and check anomalous illumination while stationary; report model and photometric assumptions separately.",
'P36':"Recover axisymmetric volume from the visible interior and free surface, compute flow rate, and independently fit the Q versus h exponents for water and sand.",
'P10':"For the continuously forced task, measure support-contact drift, lift-off distance, and rigid-body shape changes; do not use free-instability timing.",
'P39':"Independently fit both outer bubble arcs and the signed partition arc; compute the same-frame three-radius residual and aggregate with PTS weighting.",
'P38':"Independently measure the outer radii of both balls and their pre-contact terminal-speed windows; compare the speed ratio with the squared-radius ratio."}
DEFAULTS={
'P39':dict(min_radius_ratio=1.03,min_partition_chord_small_diameter_ratio=.12,min_valid_duration_sec=.08,min_valid_frame_fraction=.10,max_relative_radius_uncertainty=.50,error_at_zero=1.,max_circle_relative_residual=.08,min_edge_coverage=.40,max_partition_band_chord_ratio=.08),
'P33':dict(margin=.2,smooth_window_sec=.08,peak_window_sec=.12,height_noise_diameter=.01,min_height_snr=3.,max_track_gap_sec=.1,min_valid_track_fraction=.9),
'P34':dict(amplitude_floor_deg=1.,amplitude_fraction=.1,peak_prominence_deg=.5,smooth_window_sec=.08,min_observation_sec=2.,min_peak_separation_sec=.1,max_track_gap_sec=.1),
'P30':dict(onset_tolerance_sec=.5,crossing_hysteresis_coil_length_ratio=.01,crossing_hold_sec=.08,brightness_on_sigma=5.,brightness_off_sigma=3.,min_event_duration_sec=.08,merge_gap_sec=.08),
'P36':dict(derivative_window_sec=.25,min_independent_fit_points=8,min_flow_snr=3.,error_half_score=1.),
'P10':dict(geometry_error_half_score=.02,geometry_noise_px=2.,onset_window_sec=.20,onset_hold_sec=.12,min_tip_speed_deg_per_sec=5.,onset_speed_ratio=3.,com_hysteresis_diagonal_ratio=.01,max_event_uncertainty_sec=.15,time_error_half_score_sec=.5),
'P38':dict(terminal_window_sec=.10,max_relative_speed_change=.50,min_terminal_displacement_diameter=.20,min_speed_snr=2.,bottom_clearance_diameter=.50,min_radius_ratio=1.1,error_at_zero=1.)}
