#!/usr/bin/env python3
"""Finalize the public envelope and reconcile CLI measurement status, without rescoring."""
from copy import deepcopy
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unified_evaluators.contract import finalize, metric
from unified_evaluators.runtime import ensure_visual_evidence, scoring_text, write_json
RUN=ROOT/'work/g1_g9_v2_20260911'


def rich_metrics(result):
    """Read either a pre-contract rich result or the strict public envelope."""
    if all(set(result.get('metrics',{}).get(k,{})) >= {'physics_score','proxy_score'} for k in ('M1','M2')):
        return result['metrics']
    out={}
    for key in ('M1','M2'):
        public=result['metrics'][key]
        flag=public.get('extract_success')
        if flag is None:
            out[key]=metric(defined=False)
        else:
            detail=result.get('verbose',{}).get(key,{})
            scoring=detail.get('scoring',{})
            # The strict public metric is a final score now.  Its original
            # measurement must be recovered separately, never normalized again.
            value=detail.get('raw_metric',detail.get('metric',public.get('metric')))
            out[key]={
                'extract_success':flag, 'metric':deepcopy(value) if flag else None,
                **{name:scoring[name] for name in ('physics_score','recognition_score','proxy_score','proxy_valid')},
            }
    return out


def main():
    manifest_path=RUN/'run/run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest.get('finished_at_utc'), 'Wait for all measurements before finalizing.'
    inputs=json.loads((RUN/'input_manifest.json').read_text())
    rows={(r['group'],r['task_id'],r['model_folder'],r['sample_id']):r for r in inputs['samples']}
    original=RUN/'run/run_manifest_before_contract_finalization.json'
    if not original.exists():
        write_json(original,manifest)
    for invocation in manifest['completed']:
        key=(invocation['group'],invocation['task_id'],invocation['model_folder'],invocation['sample_id'])
        metadata=rows[key]
        path=Path(invocation['output']);result=json.loads(path.read_text());before=deepcopy(result['metrics'])
        rich=rich_metrics(result)
        task=ROOT/metadata['group']/metadata['task_id']
        old_provenance=result.get('provenance') or result.get('verbose',{}).get('M1',{}).get('_provenance',{})
        debug=task/old_provenance['debug_dir'] if old_provenance.get('debug_dir') else (task/old_provenance['raw_result_path']).parent
        evidence=ensure_visual_evidence(result['verbose']['M1'].get('evidence',[]),task/metadata['video_path'],debug,task)
        result=finalize(result['task_id'],metadata,rich,result['verbose'],evidence,old_provenance)
        for key in ('M1','M2'):
            assert result['metrics'][key]['extract_success']==before[key]['extract_success']
            assert result['verbose'][key]['raw_metric']==rich[key]['metric']
            expected=rich[key]['proxy_score'] if rich[key]['extract_success'] is True else None
            assert result['metrics'][key]['metric']==expected
            for name in ('physics_score','recognition_score','proxy_score','proxy_valid'):
                assert result['verbose'][key]['scoring'][name]==rich[key][name]
        result['verbose']['M1']['_provenance']['contract_finalized_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        if metadata['group']=='g3':
            for detail in (result['verbose']['M1'],result['verbose']['M2']):
                detail['evidence_score_labels_note']='原后端图像或视频中的诊断分沿用后端旧定义；本版最终分以 verbose.M*.scoring.proxy_score 和 scoring_calculation.md 为准。图中的物理轨迹与几何测量用于复核。'
        write_json(path,result)
        for detail in (result['verbose']['M1'],result['verbose']['M2']):
            if detail.get('scoring_calculation_path'):
                task=ROOT/metadata['group']/metadata['task_id']
                (task/detail['scoring_calculation_path']).write_text(scoring_text(result))
                break
        if invocation['exit_code']==1 and result['verbose']['M1']['_provenance'].get('runtime_error') is None:
            invocation['measurement_exit_code']=1
            invocation['exit_code']=0
            invocation['exit_status_note']='CLI 1 means physical measurement unavailable; a current structured result was produced successfully.'
        summary=result['verbose']['M1']['_scoring_summary']
        invocation.update(metrics=result['metrics'],score=summary['score'],score_status=summary['score_status'])
    manifest['contract_finalized_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(manifest_path,manifest)
    for directory in {Path(r['output']).parent for r in manifest['completed']}:
        entries=[r for r in manifest['completed'] if Path(r['output']).parent==directory]
        write_json(directory/'batch_summary.json',{'schema_version':'physical-bench-batch-v2-g1-g9','samples':entries})
    print(json.dumps({'finalized':len(manifest['completed']),'runtime_errors':sum(bool(r.get('runtime_error') or r.get('error')) for r in manifest['completed'])}))


if __name__=='__main__':
    main()
