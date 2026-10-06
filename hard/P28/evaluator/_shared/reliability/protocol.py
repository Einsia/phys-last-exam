"""Frozen task protocols used by reliability controls and VLM comparisons.

The protocol separates four questions that used to be mixed together:
whether the experimental conditions are valid, whether the requested event
actually happened, whether the quantities can be measured, and whether the
measured physical relation is satisfied.
"""

from copy import deepcopy


TASK_PROTOCOLS = {
    'P14': {
        'phenomenon': '单摆周期与摆长关系',
        'domain': '力学',
        'applicable_conditions': [
            '两个摆在同一固定视角下可见',
            '摆长在视频内保持固定',
            '振幅足够小，允许使用小角度近似',
        ],
        'required_event': '两个摆都完成至少三个可辨认的完整振荡周期',
        'quantities': ['L1', 'L2', 'T1', 'T2'],
        'constraint': '(T1/T2)^2 = L1/L2',
        'residual': 'abs(((T1/T2)^2)/(L1/L2)-1)',
        'insufficient_evidence': ['周期不足', '严重遮挡', '摆点跟踪失败', '视角不允许比较摆长'],
        'pass_threshold': 0.80,
    },
    'P18': {
        'phenomenon': '光的反射',
        'domain': '光学',
        'applicable_conditions': [
            '镜面、入射光和反射光在同一可测二维平面内',
            '法线方向和光线端点可辨认',
        ],
        'required_event': '入射光到达镜面并产生可辨认的反射光',
        'quantities': ['theta_incident', 'theta_reflected', 'normal_direction'],
        'constraint': 'theta_incident = theta_reflected',
        'residual': 'abs(theta_incident-theta_reflected) / 90deg',
        'insufficient_evidence': ['镜面或法线不可见', '光线严重模糊', '二维测量前提不成立'],
        'pass_threshold': 0.80,
    },
    'P25': {
        'phenomenon': '浮冰融化液面变化',
        'domain': '热学 / 浮力',
        'applicable_conditions': [
            '容器、液体、浮冰和液面在大部分时间内可见',
            '容器没有明显倾斜或外部加液',
        ],
        'required_event': '浮冰从初始状态持续融化到末态，并且末态残冰证据可判断',
        'quantities': ['initial_ice_fraction', 'final_ice_fraction', 'initial_level', 'final_level'],
        'constraint': '纯水中浮冰融化后液面近似保持不变',
        'residual': 'abs(final_level-initial_level) / container_height',
        'insufficient_evidence': ['冰块或液面持续遮挡', '视频太短无法判断融化事件', '容器边界不可见'],
        # Pilot calibration: the correct control scores about .70 with the
        # current level extractor, so .65 accepts it while retaining a margin
        # for measurement noise.
        'pass_threshold': 0.65,
    },
    'P40': {
        'phenomenon': '液滴合并体积守恒',
        'domain': '表面张力',
        'applicable_conditions': [
            '两个初始液滴和合并后的近球形液滴轮廓可见',
            '相机尺度在视频内基本固定',
        ],
        'required_event': '两个液滴接触并合并为一个连续液滴',
        'quantities': ['r1', 'r2', 'rf', 'merge_frame'],
        'constraint': 'rf^3 = r1^3 + r2^3',
        'residual': 'abs(rf^3/(r1^3+r2^3)-1)',
        'insufficient_evidence': ['液滴轮廓不可分割', '合并帧不明确', '严重遮挡或出画'],
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
