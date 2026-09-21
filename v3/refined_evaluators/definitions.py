"""Dependency-free public defaults, matching the task evaluator specifications."""
PRINCIPLES={
'P37':'由首帧缺口固定环身份，比较相对初始位置、固定初始外径归一化的最大升高量。',
'P38':'在共同可见时间窗口和共同幅度门槛下，独立统计实心板与开槽板的完整摆动次数。',
'P39':'独立检测磁铁穿过线圈的几何事件和灯亮起点，按固定时间容差一对一匹配并计算 F1。',
'P41':'由可见内腔及自由表面恢复轴对称体积，计算流量，独立拟合水与沙的 Q 对 h 指数。',
'P43':'独立测量质心投影越过支撑边与持续倾倒起点，保留帧差并按同一事件的秒差评分。',
'P47':'独立拟合两泡外圆弧和有符号隔膜圆弧，按同帧三半径关系计算误差并进行 PTS 加权聚合。',
'P49':'独立测量大小球外轮廓半径和触底前终端速度窗口，计算速度比相对半径平方比的误差。'}
DEFAULTS={
'P47':dict(min_radius_ratio=1.10,min_partition_chord_small_diameter_ratio=.20,min_valid_duration_sec=.30,min_valid_frame_fraction=.80,max_relative_radius_uncertainty=.20,error_at_zero=1.,max_circle_relative_residual=.03,min_edge_coverage=.65,max_partition_band_chord_ratio=.025),
'P37':dict(margin=.2,smooth_window_sec=.08,peak_window_sec=.12,height_noise_diameter=.01,min_height_snr=3.,max_track_gap_sec=.1,min_valid_track_fraction=.9),
'P38':dict(amplitude_floor_deg=1.,amplitude_fraction=.1,peak_prominence_deg=.5,smooth_window_sec=.08,min_observation_sec=2.,min_peak_separation_sec=.1,max_track_gap_sec=.1),
'P39':dict(onset_tolerance_sec=.5,crossing_hysteresis_coil_length_ratio=.01,crossing_hold_sec=.08,brightness_on_sigma=5.,brightness_off_sigma=3.,min_event_duration_sec=.08,merge_gap_sec=.08),
'P41':dict(derivative_window_sec=.25,min_independent_fit_points=8,min_flow_snr=3.,error_half_score=1.),
'P43':dict(onset_window_sec=.20,onset_hold_sec=.12,min_tip_speed_deg_per_sec=5.,onset_speed_ratio=3.,com_hysteresis_diagonal_ratio=.01,max_event_uncertainty_sec=.15,time_error_half_score_sec=.5),
'P49':dict(terminal_window_sec=.50,max_relative_speed_change=.1,min_terminal_displacement_diameter=1.,min_speed_snr=3.,bottom_clearance_diameter=2.,min_radius_ratio=1.1,error_at_zero=1.)}
