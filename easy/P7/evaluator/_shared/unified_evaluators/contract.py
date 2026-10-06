"""One output contract. Measurements and extraction eligibility stay task-specific."""
from copy import deepcopy
import math
from pathlib import Path

try:
    from reliability.protocol import get_protocol
    from reliability.status import classify_result_status, status_summary
except ImportError:  # direct module execution from the unified_evaluators folder
    from ..reliability.protocol import get_protocol
    from ..reliability.status import classify_result_status, status_summary

SCHEMA = 'physical-bench-result-v2-g1-g9'
VERSION = 'physical-bench-proxy-v2-recognition015-arithmetic-defined-metrics'
V3_VERSION = 'physical-bench-v3-consistency015-physics085'
KEYS = ('M1', 'M2')


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(v) for v in value]
    if hasattr(value, 'tolist'):
        return finite(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def present(value):
    if value is None:
        return False
    if isinstance(value, (dict, list, tuple)):
        return bool(value) and all(present(v) for v in (value.values() if isinstance(value, dict) else value))
    return not isinstance(value, float) or math.isfinite(value)


def q(error, scale):
    if error is None or isinstance(error, bool):
        return None
    if not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError('Residual scale must be finite and positive')
    error = float(error)
    return 1 / (1 + abs(error)/scale) if math.isfinite(error) else None


def gm(values):
    values = list(values)
    if not values or any(v is None for v in values):
        return None
    if any(v == 0 for v in values):
        return 0.0
    if any(not math.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError('Geometric-mean components must be finite scores in [0,1]')
    return math.exp(sum(math.log(v) for v in values)/len(values))


def clip(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return min(1.0, max(0.0, float(value)))


def metric(value=None, physics=None, success=False, defined=True):
    if not defined:
        return dict(extract_success=None, metric=None, physics_score=None,
                    recognition_score=None, proxy_score=None, proxy_valid=None)
    ok = bool(success and present(value) and physics is not None and math.isfinite(physics) and 0 <= physics <= 1)
    return dict(extract_success=ok, metric=finite(value) if ok else None,
                physics_score=physics if ok else None, recognition_score=0.15 if ok else 0.0,
                proxy_score=0.15+0.85*physics if ok else 0.0, proxy_valid=ok)


def aggregate(metrics):
    defined = [key for key in KEYS if metrics[key]['extract_success'] is not None]
    successful = [key for key in defined if metrics[key]['extract_success']]
    weights = {key: (1/len(defined) if key in defined else 0.0) for key in KEYS}
    complete = bool(defined) and len(successful) == len(defined)
    total = sum(weights[key]*(metrics[key]['proxy_score'] or 0) for key in defined)
    return dict(version=VERSION, score=total, overall_proxy_score=total,
                weights=weights, defined_metrics=defined,
                not_applicable_metrics=[k for k in KEYS if k not in defined],
                recognized_metric_count=len(successful),
                overall_proxy_valid=complete,
                score_status='complete' if complete else ('partial' if successful else 'unavailable'),
                coverage_scope='defined_metrics',
                recognition_weight=0.15, physics_weight=0.85,
                formula=' + '.join(f'{weights[k]:g} * S_{k}' for k in defined) or '0',
                missing_metric_policy='defined_metric_failure_zero_no_redistribution',
                not_applicable_metric_policy='excluded_from_denominator')


def measurement_block(principle, value, *, steps=None, measurements=None, normalization=None, reason=None):
    return dict(defined=True, principle=principle, measurement_steps=steps or [],
                raw_measurement=finite(value), measurements=finite(measurements or {}),
                normalization=normalization or {}, failure_reason=reason, evidence=[])


def physeval_result(raw):
    """M2/M3/... are auxiliary subitems, grouped once into public M2."""
    old_metrics, old_verbose = raw['metrics'], raw['verbose']
    slots, blocks = {}, {}
    auxiliary = []
    for key, original in old_metrics.items():
        v = deepcopy(old_verbose.get(key, {}))
        ok = original['extract_success']
        value = original['metric']
        physics = None
        if ok and present(value):
            # Recompute from the original measurement/anchor, avoiding score rounding.
            rule = v.get('score_rule', '')
            physics = clip(value) if v.get('non_residual', '比例/覆盖率/置信度' in rule) else q(float(value)-v.get('ideal', 0.0), v['error_scale_a'])
        out = metric(value, physics, ok, defined=ok is not None)
        if key == 'M1':
            slots[key] = out
            blocks[key] = measurement_block(v.get('principle') or v.get('measures', key), value,
                steps=v.get('measurement_steps'), measurements=v.get('quantities'),
                normalization={'scale': v.get('error_scale_a'), 'rule': v.get('score_rule')}, reason=v.get('note'))
        elif ok is not None:
            auxiliary.append((key, out, v))
    if auxiliary:
        values = {key: m['metric'] for key, m, _ in auxiliary}
        physics = gm(m['physics_score'] for _, m, _ in auxiliary)
        success = all(m['extract_success'] for _, m, _ in auxiliary)
        value = next(iter(values.values())) if len(values) == 1 else values
        slots['M2'] = metric(value, physics, success)
        blocks['M2'] = measurement_block('；'.join(v.get('principle') or v.get('measures', k) for k, _, v in auxiliary),
            value, steps=[step for _, _, v in auxiliary for step in v.get('measurement_steps', [])],
            measurements={k: v.get('quantities', {}) for k, _, v in auxiliary},
            normalization={'combination': 'geometric mean of all required auxiliary physics scores; recognition once',
                'components': {k: {'raw_measurement':m['metric'], 'physics_score':m['physics_score'], 'scale':v.get('error_scale_a')} for k,m,v in auxiliary}},
            reason='; '.join(v.get('note','') for _,m,v in auxiliary if not m['extract_success']) or None)
        blocks['M2']['submetrics'] = {k: {'metric':m, 'measurement':v} for k,m,v in auxiliary}
    else:
        slots['M2'] = metric(defined=False)
        blocks['M2'] = {'defined':False, 'principle':'本题未定义 M2。', 'measurement_steps':[], 'evidence':[]}
    return slots, blocks


CV_LOGIC = {
    'P18': ('拟合入射光、反射光与镜面，沿法线计算两条光线夹角差。', '分别求两条光线与镜面的交点，以几何尺度归一化交点距离。'),
    'P21': ('逐帧分割两臂液体，测量末段左右液面高度差并按容器高度归一化。', '对末段液面位置随时间的变化求速度，报告残余液面运动。'),
    'P31': ('测量两条悬线相对竖直方向的角度，比较稳态角度差。', '比较两球相对装置中轴的水平位移，计算归一化对称误差。'),
    'P35': ('分割大小沙堆轮廓，拟合左右坡面，比较两堆休止角之比。', '计算每堆左右坡面的角度差，取两堆中较大的不对称误差。'),
    'P6': ('相对板面跟踪两物块，检测持续滑动起点，计算起滑帧差占视频帧数的比例。', '在各自起滑时刻拟合板面倾角，计算两物块临界角差。'),
    'P26': ('确认可见融化后分割容器液面，比较初末液面高度的归一化变化及方向。', '按当前评测意见取消 M2，不参与分母。'),
    'P27': ('分割容器液面，比较初末液面高度的归一化变化及方向。', '测量末帧固体证据比例，同时保留残余冰检测结果。'),
    'P24': ('分割界面和液柱，按液柱高度关系估计密度比，计算与设定值的偏差。', '测量两侧截面几何，计算归一化截面一致性误差。'),
    'P29': ('从两物体运动轨迹检测到达事件，比较运动时间比。', '由轨迹估计运动速度，计算对应速度比。'),
    'P7': ('提取真实悬链轮廓，按实测下垂高度归一化拟合RMSE，10%相对误差对应半分。', '比较独立支点高度，按悬链水平跨度归一化，2%相对误差对应半分。'),
}


def classical_result(raw):
    measurements = raw.get('verbose', {}).get('measurements', {})
    normalization = measurements.get('score_normalization', {})
    metrics, blocks = {}, {}
    for i, key in enumerate(KEYS):
        if raw['task_id'] == 'P26' and key == 'M2':
            metrics[key] = metric(defined=False)
            blocks[key] = {'defined':False, 'principle':CV_LOGIC['P26'][1],
                'measurement_steps':[], 'measurements':{}, 'evidence':[],
                'normalization':{'policy':'removed_by_review; excluded_from_denominator'}}
            continue
        scoring = normalization.get(key, {})
        name = scoring.get('raw_measurement')
        if name == 'max(small_side_asymmetry_deg, large_side_asymmetry_deg)':
            vals = [measurements.get(k) for k in ('small_side_asymmetry_deg','large_side_asymmetry_deg')]
            value = max(vals) if all(present(v) for v in vals) else None
        else:
            value = measurements.get(name)
        original = raw['metrics'][key]
        physics = scoring.get('physics_score')
        metrics[key] = metric(value, physics, original['extract_success'])
        blocks[key] = measurement_block(CV_LOGIC[raw['task_id']][i], value,
            steps=['逐帧解码视频并提取任务所需几何。', '按原提取器的有效性检查选择可用测量。', '由测量值计算该指标，原误差尺度保持不变。'],
            measurements=measurements, normalization=scoring,
            reason=raw.get('verbose',{}).get('failure_reason'))
    return metrics, blocks


def already_scored_result(raw):
    metrics = {}
    blocks = deepcopy(raw['verbose'])
    for key in KEYS:
        m = raw['metrics'][key]
        metrics[key] = metric(m['metric'], m.get('physics_score'), m['extract_success'], m['extract_success'] is not None)
        if not isinstance(blocks.get(key),dict):
            blocks[key] = {}
        b = blocks[key]
        b.setdefault('defined', m['extract_success'] is not None)
        b.setdefault('principle', b.get('measurement_principle') or b.get('logic') or ('本题未定义 M2。' if not b['defined'] else '从所列测量证据提取物理量并按冻结定义计算指标。'))
        b.setdefault('measurement_steps', b.get('measurement_process') or [])
        b.setdefault('raw_measurement', m['metric'])
    return metrics, {k:blocks[k] for k in KEYS}


def _public_video_path(metadata):
    """Expose the stable package path instead of the evaluator's disk path."""
    value = metadata.get('video_path')
    if not value:
        return value
    parts = Path(str(value)).as_posix().split('/')
    if 'output_videos' in parts:
        index = parts.index('output_videos')
        folder = metadata.get('model_folder') or (parts[index + 1] if len(parts) > index + 1 else None)
        if folder:
            return f'{folder}/videos/{parts[-1]}'
    return Path(str(value)).as_posix()


def _public_image_path(metadata):
    value = metadata.get('image_path')
    if not value:
        return None
    parts = Path(str(value)).as_posix().split('/')
    return '/'.join(parts[parts.index('first_frames'):]) if 'first_frames' in parts else Path(str(value)).as_posix()


def _public_model(metadata):
    value = metadata.get('model')
    return 'minimax-h3' if isinstance(value, str) and value.lower() == 'minimax_h3' else value


def public_metric(value):
    """Expose the normalized final indicator score, preserving null states."""
    flag = value.get('extract_success')
    return {'extract_success': flag,
            'metric': value['proxy_score'] if flag is True else None}


def _public_blocks(blocks, metrics, proxy, provenance, sample_id):
    out = deepcopy(blocks)
    for key in KEYS:
        block = out.setdefault(key, {})
        m = metrics[key]
        block['defined'] = m['extract_success'] is not None
        block['extract_success'] = m['extract_success']
        block['raw_metric'] = deepcopy(m.get('metric')) if m['extract_success'] is True else None
        block.setdefault('raw_measurement', deepcopy(m.get('metric')))
        block['metric'] = public_metric(m)['metric']
        # Preserve the measurement, pure physics score and recognition reward
        # so the public normalized score can be traced back to its inputs.
        block['scoring'] = {
            'physics_score': m.get('physics_score'),
            'recognition_score': m.get('recognition_score'),
            'proxy_score': m.get('proxy_score'),
            'proxy_valid': m.get('proxy_valid'),
        }
    # Reserved details live inside M1 so verbose still has the requested M1/M2
    # entries and no scoring field leaks into metrics.*.
    out['M1']['_scoring_summary'] = deepcopy(proxy)
    out['M1']['_scoring_summary']['metric_semantics'] = 'normalized_final_score'
    out['M1']['_scoring_summary']['metric_definition'] = (
        'metrics.M1/M2.metric = proxy_score（含识别奖励，范围 [0,1]）；'
        'extract_success=false 或 null 时 metric=null。原始量保存在 verbose.M*.raw_metric，'
        '完整测量与诊断保存在 verbose.M*.raw_measurement。'
    )
    out['M1']['_provenance'] = deepcopy(provenance)
    out['M1']['_sample_id'] = sample_id
    return out


def add_proxy_formulas(result):
    """Describe stored scores without recalculating measurements or scores."""
    for key in KEYS:
        flag = result['metrics'][key]['extract_success']
        scoring = result['verbose'][key]['scoring']
        scoring['proxy_formula'] = (
            'proxy_score = 0.15 + 0.85 * physics_score（extract_success=true）；'
            'extract_success=false 时为 0；extract_success=null 时为 null。'
        )
        if flag is True:
            scoring['proxy_calculation'] = (
                f"proxy_score = 0.15 + 0.85 * {scoring['physics_score']} = {scoring['proxy_score']}"
            )
        elif flag is False:
            scoring['proxy_calculation'] = (
                'proxy_score = 0；extract_success=false，未通过测量有效性检查，不加识别分。'
            )
        else:
            scoring['proxy_calculation'] = (
                'proxy_score = null；extract_success=null，指标未定义，不参与总分。'
            )
    summary = result['verbose']['M1']['_scoring_summary']
    terms = [
        f"{summary['weights'][key]:g} * {result['verbose'][key]['scoring']['proxy_score']}"
        for key in summary['defined_metrics']
    ]
    summary['calculation'] = f"score = {' + '.join(terms) or '0'} = {summary['score']}"
    return result


def finalize(task_id, metadata, metrics, blocks, evidence, provenance, *, consistency=None, physics_attempted=None):
    """Create the strict public envelope while retaining auditable verbose data."""
    if consistency is None:
        consistency = blocks.get('M1', {}).get('_consistency')
    if consistency is not None:
        if physics_attempted is None:
            physics_attempted = provenance.get('physics_attempted', consistency.get('passed') is True)
        return finalize_v3(task_id, metadata, metrics, blocks, evidence, provenance,
                           consistency, physics_attempted)
    proxy = aggregate(metrics)
    blocks = deepcopy(blocks)
    for key in KEYS:
        block = blocks.setdefault(key, {})
        block['evidence'] = evidence if block.get('defined', True) else []
        if block.get('defined', True) and not block.get('measurement_steps'):
            block['measurement_steps'] = ['解码视频并按上述任务原理提取所需几何或轨迹。', '检查所需物理量是否可靠可测；失败细节与中间观测保存在原始结果和证据文件中。']
        if metrics[key]['extract_success'] is False and not block.get('failure_reason'):
            block['failure_reason'] = 'Required physical quantities did not pass the existing measurement validity checks.'
    return add_proxy_formulas(finite({
        'task_id': task_id,
        'video_path': _public_video_path(metadata),
        'image_path': _public_image_path(metadata),
        'video_prompt': metadata.get('video_prompt'),
        'model': _public_model(metadata),
        'seed': metadata.get('seed'),
        'metrics': {key: public_metric(metrics[key]) for key in KEYS},
        'verbose': _public_blocks(blocks, metrics, proxy, provenance, metadata.get('sample_id')),
    }))


def finalize_v3(task_id, metadata, metrics, blocks, evidence, provenance, consistency, physics_attempted):
    """Add consistency exactly once; physics uses pure scores, not V2 bonuses."""
    consistency = deepcopy(consistency)
    evaluated = consistency.get('status') == 'evaluated'
    c = consistency.get('score')
    if evaluated and (isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c) or not 0 <= c <= 1):
        raise ValueError('Evaluated consistency requires a finite score in [0,1]')
    passed = evaluated and consistency.get('passed') is True
    if physics_attempted and not passed:
        raise ValueError('Physics cannot run without passing consistency')
    defined = [key for key in KEYS if metrics[key]['extract_success'] is not None]
    weights = {key: 1 / len(defined) if key in defined else 0.0 for key in KEYS}
    successful = [key for key in defined if physics_attempted and metrics[key]['extract_success'] is True]
    for key in successful:
        score = metrics[key].get('physics_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'{key}: invalid pure physics score')
    physics = sum(weights[key] * metrics[key]['physics_score'] for key in successful) if physics_attempted else None
    contribution = 0.15 * c if evaluated else None
    physics_contribution = 0.85 * physics if physics_attempted else 0.0
    total = contribution + physics_contribution if evaluated else None
    complete = bool(defined) and len(successful) == len(defined)
    physics_status = ('complete' if complete else 'partial' if successful else 'unavailable') if physics_attempted else 'not_run'
    if not evaluated:
        status = 'consistency_error'
    elif not passed:
        status = 'consistency_rejected'
    elif provenance.get('runtime_error'):
        status = 'execution_error'
    else:
        status = physics_status
    summary = {
        'version': V3_VERSION, 'score': total, 'overall_proxy_score': total,
        'consistency_score': c if evaluated else None, 'consistency_passed': passed if evaluated else None,
        'consistency_weight': 0.15, 'consistency_contribution': contribution,
        'physics_score': physics, 'physics_weight': 0.85, 'physics_contribution': physics_contribution,
        'physics_attempted': bool(physics_attempted), 'physics_status': physics_status,
        'weights': weights, 'defined_metrics': defined,
        'not_applicable_metrics': [key for key in KEYS if key not in defined],
        'recognized_metric_count': len(successful), 'overall_proxy_valid': passed and complete,
        'score_status': status, 'coverage_scope': 'defined_physics_metrics', 'recognition_weight': 0.0,
        'metric_semantics': 'pure_physics_score',
        'metric_definition': 'metrics.M1/M2.metric is the pure physics score in [0,1]; consistency is awarded once at video level.',
        'formula': '0.15 * consistency_score + 0.85 * mean_defined_physics_score' if physics_attempted else '0.15 * consistency_score',
        'calculation': f'score = {contribution} + {physics_contribution} = {total}' if evaluated else 'VLM evaluation failed; no valid total score.',
        'missing_metric_policy': 'defined_physics_failure_zero_no_redistribution',
        'not_applicable_metric_policy': 'excluded_from_physics_denominator',
    }
    out = deepcopy(blocks)
    public = {}
    for key in KEYS:
        block = out.setdefault(key, {})
        m = metrics[key]
        block['defined'] = key in defined
        flag = m['extract_success'] if physics_attempted else (False if key in defined else None)
        physical = m.get('physics_score') if flag is True else None
        public[key] = {'extract_success': flag, 'metric': physical}
        block.update(extract_success=flag, metric=physical,
                     raw_metric=deepcopy(m.get('metric')) if flag is True else None,
                     measurement_attempted=bool(physics_attempted and key in defined))
        block['evidence'] = evidence if key in defined else []
        block['scoring'] = {
            'physics_score': physical, 'recognition_score': 0.0 if key in defined else None,
            'proxy_score': physical if flag is True else 0.0 if key in defined else None,
            'proxy_valid': flag is True if key in defined else None,
            'proxy_formula': 'S_M = pure physics score; consistency is added once at video level.',
        }
        if key in defined and not physics_attempted:
            block.update(status='not_run', failure_reason='Consistency gate rejected the video.' if evaluated else 'Consistency evaluation failed.')
            block['measurement_steps'] = []
        elif flag is False:
            if not block.get('failure_reason'):
                block['failure_reason']=block.get('reason') or 'Required physical quantities could not be measured.'
    # Reliability state is intentionally separate from score_status.  In
    # particular, a false extraction flag becomes evidence_insufficient unless
    # the backend supplied explicit event/condition evidence for task failure.
    protocol = get_protocol(task_id)
    try:
        threshold = float(provenance.get('measurement_pass_threshold', protocol.get('pass_threshold', 0.80)))
    except (TypeError, ValueError):
        threshold = float(protocol.get('pass_threshold', 0.80))
    threshold = min(1.0, max(0.0, threshold))
    measurement_status, measurement_reason, metric_statuses = classify_result_status(
        metrics, out, physics_attempted=bool(physics_attempted), threshold=threshold)
    defined_count = len(defined)
    reliably_measured_count = sum(1 for key in defined if metrics[key].get('extract_success') is True)
    reliability = status_summary(measurement_status, metric_statuses,
                                 defined_count=defined_count,
                                 reliable_count=reliably_measured_count)
    reliability.update({
        'protocol_version': 'measurement-protocol-v1',
        'task_id': task_id,
        'pass_threshold': threshold,
        'metric_statuses': {
            key: {'measurement_status': value[0], 'reason': value[1]}
            for key, value in metric_statuses.items()
        },
        'event_failure_inference_rule': 'never infer task failure from extraction failure alone',
    })
    summary.update({
        'measurement_status': measurement_status,
        'measurement_status_reason': measurement_reason,
        'measurement_coverage': reliability['measurement_coverage'],
        'reliably_judged': reliability['reliably_judged'],
        'measurement_pass_threshold': threshold,
        'measurement_decision_policy': 'task_failed only with explicit event/condition evidence; otherwise score threshold or evidence_insufficient',
    })
    for key in KEYS:
        block = out.setdefault(key, {})
        status_item = metric_statuses.get(key)
        if status_item:
            block['measurement_status'] = status_item[0]
            block['measurement_status_reason'] = status_item[1]
        elif block.get('defined') is False:
            block['measurement_status'] = 'evidence_insufficient'
            block['measurement_status_reason'] = '指标不适用于本题，不参与判断。'
    out['M1'].update(_scoring_summary=summary, _consistency=consistency,
                     _reliability=reliability,
                     _provenance=deepcopy(provenance), _sample_id=metadata.get('sample_id'))
    return finite({'task_id': task_id, 'video_path': _public_video_path(metadata),
                   'image_path': _public_image_path(metadata), 'video_prompt': metadata.get('video_prompt'),
                   'model': _public_model(metadata), 'seed': metadata.get('seed'), 'metrics': public,
                   'measurement_status': measurement_status,
                   'measurement_status_reason': measurement_reason,
                   'measurement_coverage': reliability['measurement_coverage'],
                   'reliably_judged': reliability['reliably_judged'],
                   'reliability': reliability,
                   'verbose': out})
