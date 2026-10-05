"""One public V4 schema for all task, gate, missingness and error outcomes.

Legacy backend dictionaries are internal evidence only. Raw measurements never
share a field with normalized scores. This module does not infer physics from
filenames, source labels, or missing measurements.
"""
from __future__ import annotations
import copy,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='vdmbench.evaluation.v4'
METRIC_KEYS=('M1','M2')
RUN_STATUSES=('complete','partial','unavailable','scored_with_zero','consistency_rejected','consistency_error','execution_error')
METRIC_STATUSES=('measured','unmeasurable','not_observed','not_run','not_applicable','error')


def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):return None
    return float(value)


def invalidate_runtime_result(data, error):
    """An exception supersedes stale success states, measurements and scores."""
    attempted=data['physics']['attempted']
    data['status']='execution_error' if attempted else 'consistency_error'
    reason=str(error)
    data['physics'].update(status='unavailable' if attempted else 'not_run',
                           score=None, judgment='evidence_insufficient',
                           measured_metrics=0, coverage=0.0 if data['physics']['defined_metrics'] else None,
                           reason=reason)
    if 'scored_metrics' in data['physics']: data['physics']['scored_metrics']=0
    if 'zero_policy_metrics' in data['physics']: data['physics']['zero_policy_metrics']=[]
    for metric in data['metrics'].values():
        metric.update(status=('error' if attempted else 'not_run') if metric['defined'] else 'not_applicable',
                      measurement_attempted=bool(metric['defined'] and attempted),
                      judgment='evidence_insufficient' if metric['defined'] else 'not_applicable',
                      raw_value=None, physics_score=None, measurements={}, normalization={},
                      uncertainty=None, evidence=[], reason=reason if metric['defined'] else None)
        metric.pop('evaluation_source',None)
    data['score'].update(total=None,consistency_contribution=None,physics_contribution=None)
    if not attempted:
        data['consistency'].update(status='error',score=None,passed=None,reason=reason,details={})
    failure={'stage':'physics' if attempted else 'consistency','metric_id':None,
             'code':'runtime_error','message':reason}
    if failure not in data['failures']:data['failures'].append(failure)
    return data


def to_v4(result, *, task_root=None, runtime_error=None):
    """Publish the existing internal scoring decision with unambiguous names."""
    if result.get('schema_version')==VERSION:
        if runtime_error is not None:
            result=invalidate_runtime_result(copy.deepcopy(result),runtime_error)
        validate(result);return result
    blocks=result.get('verbose',{});first=blocks.get('M1',{})
    summary=first.get('_scoring_summary',{});gate=first.get('_consistency',{})
    provenance=copy.deepcopy(first.get('_provenance',{}));attempted=summary.get('physics_attempted',provenance.get('physics_attempted',False)) is True
    catalog=json.loads((ROOT/'task_catalog.json').read_text());task_id=result['task_id'];info=catalog[task_id]
    group=provenance.get('group') or info.get('original_group') or (Path(task_root).parent.name if task_root else None)
    status=summary.get('score_status','execution_error')
    if status not in RUN_STATUSES:raise ValueError(f'Unsupported run state: {status}')
    metrics={};failures=[]
    for key in METRIC_KEYS:
        source=result['metrics'][key];b=blocks.get(key,{}) or {};flag=source.get('extract_success');defined=flag is not None
        state='not_applicable' if not defined else 'error' if runtime_error and attempted else 'not_run' if not attempted else 'measured' if flag is True else 'unmeasurable'
        score=number(source.get('metric')) if state=='measured' else None
        reason=b.get('failure_reason') or b.get('reason') or b.get('measurement_status_reason')
        value=copy.deepcopy(b.get('raw_metric')) if state=='measured' else None
        details=copy.deepcopy(b.get('raw_measurement',b.get('measurements',b.get('quantities',{}))))
        if task_id=='P7':
            # G7 raw_measurement is the scalar residual; identity and tracking
            # diagnostics are in measurements, including recognized failures.
            details=copy.deepcopy(b.get('measurements') or {})
        normalization=copy.deepcopy(b.get('normalization',{}))
        metric={
            'id':key,'defined':defined,'status':state,'measurement_attempted':bool(defined and attempted),
            'judgment':b.get('measurement_status','evidence_insufficient') if defined else 'not_applicable',
            'raw_value':value,'unit':b.get('unit') or normalization.get('error_unit') or normalization.get('metric_unit'),
            'physics_score':score,'pass_threshold':summary.get('measurement_pass_threshold',.8) if defined else None,
            'reason':str(reason) if reason is not None else None,
            'principle':b.get('principle',''),'measurement_steps':b.get('measurement_steps',[]),
            'measurements':details,'normalization':normalization,
            'uncertainty':copy.deepcopy(b.get('uncertainty')),
            'evidence':copy.deepcopy(b.get('evidence',[])) if defined else [],
        }
        metrics[key]=metric
        if state in ('unmeasurable','error'):
            failures.append({'stage':'physics','metric_id':key,'code':state,'message':metric['reason'] or 'Required quantities could not be measured reliably.'})
    if runtime_error:failures.append({'stage':'physics' if attempted else 'consistency','metric_id':None,'code':'runtime_error','message':str(runtime_error)})
    defined=sum(m['defined'] for m in metrics.values());measured=sum(m['status']=='measured' for m in metrics.values())
    physics_score=number(summary.get('physics_score')) if attempted else None
    total=number(summary.get('score'))
    # Runtime failure is not an estimate of video quality.
    if status in ('consistency_error','execution_error') or runtime_error:total=None
    data={
        'schema_version':VERSION,
        'task':{'id':task_id,'group':group,'difficulty':info['difficulty'],'phenomenon':info['phenomenon'],'domain':info['domain']},
        'sample':{'id':first.get('_sample_id'),'model':result.get('model'),'seed':result.get('seed'),
                  'video_path':provenance.get('input_video_path',result.get('video_path')),
                  'image_path':provenance.get('input_image_path',result.get('image_path')),
                  'video_prompt':result.get('video_prompt'),'video_sha256':provenance.get('video_sha256'),'image_sha256':provenance.get('image_sha256')},
        'status':status,
        'consistency':{'status':gate.get('status','not_run'),'score':number(gate.get('score')),'threshold':number(gate.get('threshold')),
                       'passed':gate.get('passed'),'reason':gate.get('reason'),'details':copy.deepcopy(gate)},
        'physics':{'attempted':attempted,'status':'not_run' if not attempted else 'complete' if measured==defined else 'partial' if measured else 'unavailable',
                   'score':physics_score,'judgment':summary.get('measurement_status',result.get('measurement_status','evidence_insufficient')),
                   'defined_metrics':defined,'measured_metrics':measured,'coverage':measured/defined if defined else None,
                   'missing_metric_policy':'defined_unmeasurable_contributes_zero; not_applicable_excluded',
                   'reason':summary.get('measurement_status_reason')},
        'metrics':metrics,
        'score':{'total':total,'consistency_weight':.15,'physics_weight':.85,
                 'consistency_contribution':number(summary.get('consistency_contribution')),
                 'physics_contribution':number(summary.get('physics_contribution')),
                 'formula':'0.15*C + 0.85*P if gate passes; 0.15*C if rejected; null on execution errors'},
        'failures':failures,
        'provenance':provenance,
    }
    data['provenance']['evaluator_version']='v4'
    if runtime_error is not None or status in ('consistency_error','execution_error'):
        invalidate_runtime_result(data,runtime_error if runtime_error is not None else summary.get('measurement_status_reason') or status)
    from .zero_policy import apply
    data=apply(data)
    if task_id=='P7' and (metrics['M1'].get('measurements') or {}).get('identification_status')=='identified_measurement_unstable':
        data['physics']['observation_status']='identified_measurement_unstable'
        data['physics']['reason']='已识别但测量不稳定；可靠公共区间不足，不归为完全未测。'
    validate(data);return data


def validate(data):
    required={'schema_version','task','sample','status','consistency','physics','metrics','score','failures','provenance'}
    if set(data)!=required:raise ValueError(f'V4 top-level fields differ: {set(data)^required}')
    if data['schema_version']!=VERSION or data['status'] not in RUN_STATUSES:raise ValueError('Invalid V4 version/status')
    if set(data['metrics'])!=set(METRIC_KEYS):raise ValueError('Every task must expose M1 and M2')
    for key,m in data['metrics'].items():
        if m['id']!=key or m['status'] not in METRIC_STATUSES:raise ValueError('Invalid metric identity/status')
        q=m['physics_score']
        if q is not None and (number(q) is None or not 0<=q<=1):raise ValueError('Physics score must be finite in [0,1]')
        if m['status']=='measured':
            if not m['defined'] or not m['measurement_attempted'] or q is None or m['raw_value'] is None:raise ValueError('Measured metric lacks actual value/score')
        elif m['status']=='not_observed':
            if (not m['defined'] or not m['measurement_attempted'] or q!=0.0
                    or m['raw_value'] is not None or not m['evidence']
                    or m.get('evaluation_source')!='unobserved_zero_policy'):
                raise ValueError('Unobserved zero must retain evidence and cannot fabricate a measured value')
            if data['provenance'].get('unobserved_zero_policy')!='not_observed_zero_v1':
                raise ValueError('Unobserved zero requires the explicit scoring policy')
        elif q is not None or m['raw_value'] is not None:raise ValueError('Unmeasured metric must not fabricate raw values or scores')
        if not m['defined'] and (m['status']!='not_applicable' or m['measurement_attempted']):raise ValueError('Undefined metric must be not_applicable')
    c=data['consistency'];p=data['physics'];score=data['score']
    if c['status']=='evaluated':
        if number(c['score']) is None or not 0<=c['score']<=1 or number(c['threshold']) is None:raise ValueError('Invalid consistency values')
        if c['passed'] != (c['score']>=c['threshold'] and not c.get('details',{}).get('forced_rejection',False)):raise ValueError('Consistency decision differs from threshold')
    if c.get('passed') is not True and p['attempted']:raise ValueError('Physics ran without gate approval')
    measured=sum(m['status']=='measured' for m in data['metrics'].values())
    if p['measured_metrics']!=measured:raise ValueError('Measured coverage includes an unmeasured score')
    if data['status']=='scored_with_zero':
        zeros=[k for k,m in data['metrics'].items() if m['status']=='not_observed']
        if (not zeros or c.get('passed') is not True or not p['attempted']
                or p['status']!='scored_with_zero' or p.get('zero_policy_metrics')!=zeros
                or p.get('scored_metrics')!=p['defined_metrics']):
            raise ValueError('Invalid completed assessment with unobserved zeros')
    if data['status'] in ('execution_error','consistency_error'):
        if score['total'] is not None:raise ValueError('Runtime errors are not video-quality scores')
        if p['score'] is not None or p['judgment']!='evidence_insufficient' or p['measured_metrics']!=0:
            raise ValueError('Runtime errors must invalidate previous physical judgments')
        if score['physics_contribution'] is not None or score['consistency_contribution'] is not None:
            raise ValueError('Runtime errors cannot retain score contributions')
        if any(m['status'] in ('measured','not_observed') for m in data['metrics'].values()):
            raise ValueError('Runtime errors cannot retain successful measurements')
    elif score['total'] is not None:
        if not 0<=score['total']<=1:raise ValueError('Invalid total score')
        expected=.15*c['score']+(.85*p['score'] if p['attempted'] else 0)
        if abs(expected-score['total'])>1e-9:raise ValueError('Total violates the 15%/85% scoring equation')
    json.dumps(data,allow_nan=False)
    return data


def summary_fields(data):
    """Shared consumer API; new callers never reach through verbose.M1."""
    validate(data)
    return {'task_id':data['task']['id'],'sample_id':data['sample']['id'],'model':data['sample']['model'],
            'seed':data['sample']['seed'],'score':data['score']['total'],'score_status':data['status'],
            'consistency_score':data['consistency']['score'],'consistency_passed':data['consistency']['passed'],
            'physics_score':data['physics']['score'],'physics_attempted':data['physics']['attempted'],
            'measurement_status':data['physics']['judgment'],'measurement_coverage':data['physics']['coverage'],
            'zero_policy_metrics':data['physics'].get('zero_policy_metrics',[])}
