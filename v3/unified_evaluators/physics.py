"""G3/G7 V2 mappings, with the frozen task-specific scales and validity gates."""
import math
from .contract import KEYS, clip, gm, measurement_block, metric, present, q

G3_SCALES = {'P3':0.15, 'P4':0.18, 'P6':0.25, 'P9':0.20, 'P11':0.12}
G3_LOGIC = {
    'P3': ('分别测量两条完整抛物轨迹的射程，计算 R30/R60 - 1。', '分别拟合 30°、60° 轨迹，按球直径归一化残差，并比较初速度。'),
    'P4': ('从轨迹识别碰撞、反弹最高点和相邻碰撞时间，用 |sqrt(h[n+1]/h[n])-dt[n+1]/dt[n]| 的原聚合误差衡量一致性。', '保留相邻反弹高度严格下降的比例，以及恢复系数变异系数。'),
    'P6': ('测量球心累计位移、标记转角和球半径，计算积分形式的 v/(omega*R)-1。', '由球心速度与 omega*R 的差估计瞬时接触点速度残差。'),
    'P9': ('跟踪两个摆锤与悬点，以峰值间隔估计周期，测量摆长，计算 abs((T1/T2)^2/(L1/L2)-1)。', '分别计算短摆和长摆周期的变异系数，再汇总纯物理分。'),
    'P11': ('从时序亮度差分提取光束，独立拟合液面，测量光线与法线夹角，计算 sin(theta_i)/sin(theta_t)-n_water。', '计算入射光、折射光与液面交点的距离差，以冻结的容器宽度归一化。'),
}


def component(value, scale=None, *, physics=None, formula=None):
    score = q(value, scale) if scale is not None else physics
    return {'raw_measurement':value, 'scale':scale, 'physics_score':score,
            'formula':formula or '1/(1+abs(e)/a)'}


def score_g3(task, values, validity, measurements=None, reason=None):
    components = {'M1':{'error':component(values['M1'], G3_SCALES[task])}}
    aux = values['M2']
    if task == 'P3':
        scales = {'trajectory_30deg_parabola_rmse_d':0.30, 'trajectory_60deg_parabola_rmse_d':0.30, 'initial_speed_magnitude_error':0.20}
        components['M2'] = {k:component((aux or {}).get(k), a) for k,a in scales.items()}
    elif task == 'P4':
        aux = aux or {}
        components['M2'] = {
            'height_decrease_fraction': component(aux.get('height_decrease_fraction'), physics=clip(aux.get('height_decrease_fraction')), formula='clip(height_decrease_fraction)'),
            'adjacent_restitution_coefficient_cv': component(aux.get('adjacent_restitution_coefficient_cv'),0.35)}
    elif task == 'P9':
        components['M2'] = {k:component((aux or {}).get(k),0.20) for k in ('short_pendulum_period_cv','long_pendulum_period_cv')}
    else:
        components['M2'] = {'error':component(aux,0.35 if task == 'P6' else 0.025)}
    metrics, blocks = {}, {}
    for i,key in enumerate(KEYS):
        physics = gm(c['physics_score'] for c in components[key].values())
        metrics[key] = metric(values[key], physics, validity.get(key, False))
        blocks[key] = measurement_block(G3_LOGIC[task][i], values[key],
            steps=['解码该视频，并取得经过有效性检查的轨迹或几何。', '按本题冻结的事件选择、拟合和尺度定义重新计算物理量。', '残差归一化后，在指标层加入一次识别分。'],
            measurements=measurements, normalization={'components':components[key], 'aggregation':'geometric mean' if len(components[key]) > 1 else 'single component'}, reason=reason)
    return metrics, blocks


def g3_result(raw):
    task, m = raw['task_id'], raw.get('metrics', {})
    measurements = raw.get('measurements', {})
    status = raw.get('status') if isinstance(raw.get('status'),dict) else raw.get('statuses', raw)
    valid = status.get('metric_validity', raw.get('metric_validity'))
    if task == 'P11' and valid is not None:
        valid = {'M1':valid.get('M1_snell_residual',False), 'M2':valid.get('M2_intersection_consistency',False)}
    if valid is None:
        valid = {key: bool(status.get('extract_success', raw.get('extract_success')) and status.get('measurement_valid', raw.get('measurement_valid',False))) for key in KEYS}
    if task == 'P3':
        aux = {}
        for slot, ball in measurements.get('balls',{}).items():
            angle = int(ball['angle_target_deg'])
            aux[f'trajectory_{angle}deg_parabola_rmse_d'] = m.get('M2',{}).get('per_ball',{}).get(slot,{}).get('parabola_rmse_d')
        aux['initial_speed_magnitude_error'] = m.get('initial_speed_error')
        values = {'M1':m.get('M1'), 'M2':aux}
    elif task == 'P4':
        heights = measurements.get('rebound_heights_px', [])
        checks = [right < left for left,right in zip(heights,heights[1:])]
        values = {'M1':m.get('M1_height_time_restitution_consistency'), 'M2':{
            'height_decrease_checks':checks, 'height_decrease_fraction':sum(checks)/len(checks) if checks else None,
            'adjacent_restitution_coefficient_cv':m.get('M2_restitution_coefficient_cv')}}
    elif task == 'P6':
        values = {'M1':m.get('M1_rolling_ratio_signed'), 'M2':m.get('M2_contact_velocity_nmae')}
    elif task == 'P9':
        pendulums = raw.get('pendulums', {})
        values = {'M1':m.get('m1_abs_residual'), 'M2':{f'{name}_pendulum_period_cv':(pendulums.get(name,{}).get('period') or {}).get('period_cv') for name in ('short','long')}}
        measurements = {'pendulums':pendulums, 'camera_audit':raw.get('camera_audit')}
    else:
        values = {'M1':m.get('M1_snell_residual_signed'), 'M2':m.get('M2_intersection_disagreement_normalized')}
        measurements = {'geometry':raw.get('geometry'), 'temporal':raw.get('temporal'), 'structure':raw.get('structure')}
    return score_g3(task, values, valid, measurements,
        raw.get('failure_reason') or status.get('measurement_invalid_reasons') or raw.get('errors'))


G7_LOGIC = {
    'P7': ('跟踪实心球与圆环到共同终点的时间，比较时间比与 sqrt(10/7)。', '分别从可见旋转标记估计角速度，测量两物体 v/(omega*R) 与 1 的偏差。'),
    'P8c': ('以转折点间隔估计两个摆的周期，比较周期比与已有有限振幅理论周期比。', '用 (T30/T15-1)/(r_theory-1) 的截断值衡量周期方向关系。'),
    'P10': ('拟合斜板方向和上行、下行轨迹的加速度，比较实测加速度比与含摩擦理论值。', '汇总上下行拟合相对 RMSE 与时间顺序的连续优势。'),
    'P12': ('拟合各光束和介质界面，用 Snell/临界角几何估计折射率并计算变异系数。', '计算所有可测光线的 Snell 残差，乘以 min(N/3,1) 的光线覆盖率。'),
    'P27': ('从两侧水面信号测量半上升事件时间，按 (t_block-t_crushed)/max(t_block,t_crushed) 计算连续优势；保留原二值判断。', '测量首帧冰的投影面积比，并计算面积比对数误差；这是二维面积代理。'),
}


def ratio(a,b):
    return float(a)/float(b) if present(a) and present(b) and float(b) > 0 else None


def g7_result(raw):
    task, m, measures = raw['task_id'], raw.get('metrics',{}), raw.get('measurements',{})
    first = m.get('M1')
    base_ok = bool(raw.get('extract_success'))
    if task == 'P7':
        aux = {'sphere_no_slip_error':m.get('M2_sphere'), 'ring_no_slip_error':m.get('M2_ring')}
        c2 = {k:component(v,0.20) for k,v in aux.items()}
    elif task == 'P8c':
        declared = measures.get('T30_greater_than_T15', m.get('M2_T30_gt_T15'))
        aux = {'T30_gt_T15':declared}
        r = ratio(measures.get('T30_s'),measures.get('T15_s'))
        expected = measures.get('expected_period_ratio_exact')
        score = clip((r-1)/(expected-1)) if r is not None and present(expected) and expected > 1 else None
        c2 = {'period_direction':component({'measured_ratio':r,'expected_ratio':expected},physics=score,formula='clip((T30/T15-1)/(r_theory-1))')}
    elif task == 'P10':
        aux = {'constant_acceleration_fit_up':m.get('M2_up_fit_relative_rmse'), 'constant_acceleration_fit_down':m.get('M2_down_fit_relative_rmse'), 't_down_gt_t_up':measures.get('t_down_gt_t_up',m.get('M2_t_down_gt_t_up'))}
        r = ratio(measures.get('t_down_s'),measures.get('t_up_s'))
        expected = measures.get('expected_acceleration_ratio')
        margin = clip((r-1)/(math.sqrt(expected)-1)) if r is not None and present(expected) and expected > 1 else None
        c2 = {'up_fit':component(aux['constant_acceleration_fit_up'],0.10), 'down_fit':component(aux['constant_acceleration_fit_down'],0.10),
              'time_margin':component({'time_ratio':r,'expected_acceleration_ratio':expected},physics=margin,formula='clip((t_down/t_up-1)/(sqrt(expected_acceleration_ratio)-1))')}
    elif task == 'P12':
        residuals = measures.get('per_ray_snell_residual',m.get('per_ray_snell_residual'))
        aux = {'per_ray_snell_residual':residuals}
        c2 = {f'ray_{i}':component(v,0.06665) for i,v in enumerate(residuals or [],1)}
    else:
        aux = {'initial_mass_visual_validity':m.get('M2_initial_projected_area_log_error')}
        c2 = {'log_area_error':component(aux['initial_mass_visual_validity'],math.log(1.25))}
    if task == 'P27':
        block, crushed = measures.get('block_water_surface_half_rise_time_s'), measures.get('crushed_water_surface_half_rise_time_s')
        advantage = clip((block-crushed)/max(block,crushed)) if present(block) and present(crushed) and max(block,crushed) > 0 else None
        c1 = {'time_advantage':component({'t_block':block,'t_crushed':crushed},physics=advantage,formula='clip((t_block-t_crushed)/max(t_block,t_crushed))')}
    else:
        c1 = {'error':component(first, {'P7':0.10,'P8c':0.05,'P10':0.10,'P12':0.05}[task])}
    metrics, blocks = {}, {}
    for i,(key,value,components) in enumerate((('M1',first,c1),('M2',aux,c2))):
        physics = gm(c['physics_score'] for c in components.values())
        normalization = {'components':components,'aggregation':'geometric mean' if len(components) > 1 else 'single component'}
        if task == 'P12' and key == 'M2':
            coverage = min(len(components)/3,1)
            physics = physics*coverage if physics is not None else None
            normalization.update(coverage=coverage, aggregation='coverage * geometric mean of measured Snell ray scores')
        metrics[key] = metric(value,physics,base_ok)
        blocks[key] = measurement_block(G7_LOGIC[task][i],value,
            steps=['逐帧解码并执行该题原有几何或运动提取。', '保留原有测量有效性检查；物理错误本身不取消识别分。', '使用冻结参数映射纯物理分，按指标计一次识别分。'],
            measurements=measures,normalization=normalization,reason=raw.get('failure_reason'))
    return metrics, blocks
