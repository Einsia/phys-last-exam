"""G3/G7 V2 mappings, with the frozen task-specific scales and validity gates."""
import math
from .contract import KEYS, clip, gm, measurement_block, metric, present, q

G3_SCALES = {'P3':0.15, 'P4':0.18, 'P6':0.25, 'P9':0.20, 'P11':0.12}
G3_LOGIC = {
    'P3': ('分别测量两条完整抛物轨迹的射程，计算 R30/R60 - 1。', '分别拟合 30°、60° 轨迹，按球直径归一化残差，并比较初速度。'),
    'P4': ('从独立观测的反弹高度与时间估计恢复系数；同时检查两种估计的一致性和被动碰撞恢复系数不超过1。', '用相邻反弹高度增加的相对幅度连续衡量无外部能量输入条件下的能量违例，并报告恢复系数变异。'),
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
        components['M1']['passive_restitution_excess']=component(aux.get('passive_restitution_excess'),0.15)
        components['M2'] = {
            'height_increase_excess': component(aux.get('height_increase_excess'),0.15),
            **({'adjacent_restitution_coefficient_cv': component(aux['adjacent_restitution_coefficient_cv'],0.35)}
               if present(aux.get('adjacent_restitution_coefficient_cv')) else {})}
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
            'height_increase_excess':sum(max(0.0,right/left-1.0) for left,right in zip(heights,heights[1:]))/len(checks) if checks and all(h>0 for h in heights) else None,
            'passive_restitution_excess':sum(max(0.0,math.sqrt(right/left)-1.0) for left,right in zip(heights,heights[1:]))/len(checks) if checks and all(h>0 for h in heights) else None,
            **({'adjacent_restitution_coefficient_cv':m['M2_restitution_coefficient_cv']} if present(m.get('M2_restitution_coefficient_cv')) else {})}}
    elif task == 'P6':
        values = {'M1':m.get('M1_rolling_ratio_signed'), 'M2':m.get('M2_contact_velocity_nmae')}
    elif task == 'P9':
        pendulums = raw.get('pendulums', {})
        values = {'M1':m.get('m1_abs_residual'), 'M2':{f'{name}_pendulum_period_cv':(pendulums.get(name,{}).get('period') or {}).get('period_cv') for name in ('short','long')}}
        measurements = {'pendulums':pendulums, 'camera_audit':raw.get('camera_audit')}
    else:
        values = {'M1':m.get('M1_snell_residual_signed'), 'M2':m.get('M2_intersection_disagreement_normalized')}
        measurements = {'geometry':raw.get('geometry'), 'temporal':raw.get('temporal'), 'structure':raw.get('structure')}
    if task == 'P4' and measurements.get('single_motion_no_repeated_bounce') is True:
        metrics={key:metric({'observed_outcome':'single_motion_no_repeated_bounce'},.1,True) for key in KEYS}
        blocks={key:measurement_block(G3_LOGIC[task][i],metrics[key]['raw_value'] if 'raw_value' in metrics[key] else {'observed_outcome':'single_motion_no_repeated_bounce'},
            measurements=measurements,steps=['持续跟踪确认只有单段运动，未形成可比较的多次回跳。'],
            normalization={'policy':'opinion_v2_observed_single_motion','physics_score':.1},reason='已测得单段运动，按意见给物理分0.1。') for i,key in enumerate(KEYS)}
        return metrics,blocks
    return score_g3(task, values, valid, measurements,
        raw.get('failure_reason') or status.get('measurement_invalid_reasons') or raw.get('errors'))


G7_LOGIC = {
    'P7': ('由两物体轨迹估计共同运动方向，在多个虚拟参考位置比较相同路程的到达时间比与 sqrt(10/7)。', '按当前评测意见取消 M2；不测旋转或无滑动，不参与分母。'),
    'P8c': ('以转折点间隔估计两个摆的周期，比较周期比与已有有限振幅理论周期比。', '用 (T30/T15-1)/(r_theory-1) 的截断值衡量周期方向关系。'),
    'P10': ('拟合斜板方向和上行、下行轨迹的加速度，比较实测加速度比与含摩擦理论值。', '按意见_v2取消M2，不参与评分分母。'),
    'P12': ('独立拟合各透射光束和介质界面，用 Snell 几何估计折射率并计算变异系数；普通全反射仅提供下界，不视为临界角。', '按意见_v2取消M2，不参与评分分母。'),
    'P27': ('先确认两侧冰持续缩小及液体形成；同帧比较以各自杯底为基准、按杯高归一化的液位。碎冰杯超过定位误差并持续至少0.25秒则记录液面阶段性领先；可见片段未领先与无法判断分开。本指标不等同完全融化时间或融化质量。', '二维投影面积不能辨识三维质量，面积比只保留为诊断；不纳入物理评分。'),
}


def ratio(a,b):
    return float(a)/float(b) if present(a) and present(b) and float(b) > 0 else None


def g7_result(raw):
    task, m, measures = raw['task_id'], raw.get('metrics',{}), raw.get('measurements',{})
    first = m.get('M1')
    base_ok = bool(raw.get('extract_success'))
    if task == 'P7':
        aux = {}
        c2 = {}
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
        c2 = {}  # Projected area cannot establish equal three-dimensional mass.
    if task == 'P27':
        lead = measures.get('surface_lead')
        if lead is not None:
            observed = lead.get('lead_observed')
            base_ok = base_ok and bool(lead.get('usable')) and isinstance(observed, bool)
            first = {'lead_observed': observed, 'decision': lead.get('decision'),
                     'rule_version': lead.get('rule_version'),
                     'lead_segment_count': len(lead.get('lead_segments', [])),
                     'common_valid_frames': lead.get('common_valid_frames', 0)}
            c1 = {'sustained_liquid_level_lead':component(first, physics=float(observed) if base_ok else None,
                    formula='1 if sustained normalized right-minus-left lead exceeds localization error; 0 if adequately observed without lead; null if unobservable')}
        else:
            # Explicit backward compatibility for archived raw records only.
            block, crushed = measures.get('block_water_surface_half_rise_time_s'), measures.get('crushed_water_surface_half_rise_time_s')
            violation = max(0.0, crushed/block-1.0) if present(block) and present(crushed) and block > 0 and crushed > 0 else None
            first = {'t_block':block,'t_crushed':crushed,'relative_order_violation':violation}
            c1 = {'relative_order_violation':component(violation,0.20,formula='legacy archived rule: 1/(1+max(0,t_crushed/t_block-1)/0.20)')}
    else:
        c1 = {'error':component(first, {'P7':0.10,'P8c':0.05,'P10':0.10,'P12':0.05}[task])}
    if task == 'P10' and measures.get('observed_outcome') == 'ascent_only':
        base_ok=True;first={'observed_outcome':'ascent_only'}
        c1={'observed_ascent_only':component(first,physics=.1,formula='opinion_v2_ascent_only_score_0.1')}
    metrics, blocks = {}, {}
    for i,(key,value,components) in enumerate((('M1',first,c1),('M2',aux,c2))):
        if task in ('P7','P10','P12') and key == 'M2':
            metrics[key] = metric(defined=False)
            blocks[key] = {'defined':False,'principle':G7_LOGIC[task][i],
                'measurement_steps':[], 'measurements':{},
                'normalization':{'policy':'removed_by_review; excluded_from_denominator'},'evidence':[]}
            continue
        if task == 'P27' and key == 'M2':
            metrics[key] = metric(defined=False)
            blocks[key] = {'defined':False,'principle':G7_LOGIC[task][i],
                'measurement_steps':['保留投影面积诊断，但不从二维面积推断三维质量。'],
                'measurements':{'projected_area_diagnostic':value,'equal_initial_mass':'declared task prerequisite; not identifiable from this view'},
                'normalization':{'policy':'not_applicable; excluded from denominator'},'evidence':[]}
            continue
        physics = gm(c['physics_score'] for c in components.values())
        normalization = {'components':components,'aggregation':'geometric mean' if len(components) > 1 else 'single component'}
        if task == 'P12' and key == 'M2':
            coverage = min(len(components)/3,1)
            physics = physics*coverage if physics is not None else None
            normalization.update(coverage=coverage, aggregation='coverage * geometric mean of measured Snell ray scores')
        metrics[key] = metric(value,physics,base_ok)
        blocks[key] = measurement_block(G7_LOGIC[task][i],value,
            steps=['逐帧解码并执行该题原有几何或运动提取。', '保留原有测量有效性检查；物理错误本身不取消识别分。', '使用冻结参数映射纯物理分，按指标计一次识别分。'],
            measurements=measures,normalization=normalization,reason=('已识别但测量不稳定：可靠公共区间不足或轨迹不稳定。' if task=='P7' and measures.get('identification_status')=='identified_measurement_unstable' else raw.get('failure_reason')))
        if task == 'P27' and key == 'M1' and measures.get('surface_lead') is not None:
            blocks[key]['failure_reason'] = None
            blocks[key]['reason'] = measures['surface_lead']['reason_zh']
            blocks[key]['measurement_steps'] = [
                '分别确认两侧冰持续缩小；仅比较形成可见液层的时段。',
                '同帧以各自杯底为基准并按杯高归一化液位。',
                '领先超出两侧定位误差之和并持续至少0.25秒，记录液面阶段性领先。',
                '连续可读至少0.50秒却未观察到领先，与观测不足分开；缺口不插值。']
    return metrics, blocks
