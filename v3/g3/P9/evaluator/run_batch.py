#!/usr/bin/env python3
"""Run the frozen P9 evaluator over the complete 24-video formal batch."""
from __future__ import annotations
import argparse, csv, json, statistics, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE=Path(__file__).resolve().parent

def run_one(video,outroot,device,force):
    stem=video.stem; out=outroot/'json'/f'{stem}.json'; debug=outroot/'debug'/stem
    out.parent.mkdir(parents=True,exist_ok=True); debug.mkdir(parents=True,exist_ok=True)
    cmd=[str(Path(__import__('sys').executable)),str(HERE/'evaluate.py'),'--video',str(video),'--task_id','P9','--output',str(out),'--debug-dir',str(debug),'--device',device]
    if force: cmd.append('--force-tracking')
    log=outroot/'logs'/f'{stem}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    start=time.time(); p=subprocess.run(cmd,capture_output=True,text=True)
    log.write_text(p.stdout+'\n'+p.stderr)
    return stem,p.returncode,time.time()-start,str(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--videos',required=True); p.add_argument('--output-root',required=True); p.add_argument('--device',default='cuda:7'); p.add_argument('--workers',type=int,default=2); p.add_argument('--force',action='store_true'); a=p.parse_args()
    videos=sorted(Path(a.videos).glob('*.mp4')); root=Path(a.output_root); root.mkdir(parents=True,exist_ok=True)
    if len(videos)!=24: raise SystemExit(f'formal batch must contain 24 videos, found {len(videos)}')
    completed=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(run_one,v,root,a.device,a.force):v for v in videos}
        for f in as_completed(futures):
            item=f.result(); completed.append(item); print(item,flush=True)
    results=[]; failures=[]
    for v in videos:
        path=root/'json'/f'{v.stem}.json'
        if not path.exists(): failures.append({'video':v.name,'reason':'missing JSON'}); continue
        try: r=json.loads(path.read_text())
        except Exception as e: failures.append({'video':v.name,'reason':f'invalid JSON: {e}'}); continue
        required=('extract_success','structural_ok','measurement_valid','physics_pass')
        if not all(isinstance(r.get('status',{}).get(k),bool) for k in required): failures.append({'video':v.name,'reason':'invalid status schema'})
        if not (Path(r['debug']['overlay']).exists() and Path(r['debug']['plot']).exists()): failures.append({'video':v.name,'reason':'missing debug artifact'})
        if not r['status']['measurement_valid'] and r['metrics']['m1_abs_residual'] is not None: failures.append({'video':v.name,'reason':'invalid video has non-null M1'})
        results.append(r)
    fields=['video','sample_id','seed','extract_success','structural_ok','measurement_valid','physics_pass','score_version','overall_score','setup_score','structural_score','periodicity_score','pendulum_law_score','legacy_overall_score_0_100','legacy_setup_score_0_100','legacy_structural_score_0_100','legacy_periodicity_score_0_100','legacy_pendulum_law_score_0_100','length_ratio','period_ratio','m1_abs_residual','max_period_cv','short_period_s','long_period_s','short_fit_r2','long_fit_r2','string_drift_p95','pivot_drift_fraction_p95']
    with (root/'summary.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
        for r in results:
            name=Path(r['video']).name; seed=name.rsplit('_seed',1)[-1].split('.')[0]
            legacy=r.get('legacy_dimension_scores_0_100',{})
            w.writerow({'video':name,'sample_id':r['sample_id'],'seed':seed,**r['status'],'score_version':r.get('score_version'),'overall_score':r['overall_score'],
              'setup_score':r['dimension_scores']['setup'],'structural_score':r['dimension_scores']['structural'],'periodicity_score':r['dimension_scores']['periodicity'],'pendulum_law_score':r['dimension_scores']['pendulum_law'],
              'legacy_overall_score_0_100':r.get('legacy_overall_score_0_100'),'legacy_setup_score_0_100':legacy.get('setup'),'legacy_structural_score_0_100':legacy.get('structural'),'legacy_periodicity_score_0_100':legacy.get('periodicity'),'legacy_pendulum_law_score_0_100':legacy.get('pendulum_law'),
              'length_ratio':r['metrics']['initial_length_ratio_short_over_long'],'period_ratio':r['metrics']['period_ratio_short_over_long'],'m1_abs_residual':r['metrics']['m1_abs_residual'],'max_period_cv':r['metrics']['maximum_period_cv'],
              'short_period_s':r['pendulums']['short']['period']['period_s'] if r['pendulums']['short']['period'] else None,'long_period_s':r['pendulums']['long']['period']['period_s'] if r['pendulums']['long']['period'] else None,
              'short_fit_r2':r['pendulums']['short']['period']['fit_r2'] if r['pendulums']['short']['period'] else None,'long_fit_r2':r['pendulums']['long']['period']['fit_r2'] if r['pendulums']['long']['period'] else None,
              'string_drift_p95':r['metrics']['string_length_relative_change_p95_max'],'pivot_drift_fraction_p95':r['metrics']['pivot_drift_short_length_fraction_p95']})
    current_values=[float(r['overall_score']) for r in results]
    valid_values=[float(r['overall_score']) for r in results if r['status']['measurement_valid']]
    legacy_values=[float(r.get('legacy_overall_score_0_100',0.0)) for r in results]
    if any(r.get('score_version')!='continuous-0-1-v1' for r in results): raise AssertionError('batch contains a non-continuous score schema')
    if any(not 0.0<=value<=1.0 for value in current_values): raise AssertionError('current P9 score outside [0,1]')
    report={'expected':24,'json_count':len(results),'failures':failures,
            'score_version':'continuous-0-1-v1','score_range':[0.0,1.0],'zero_score_policy':'measurement_invalid_only',
            'status_counts':{k:sum(bool(r['status'][k]) for r in results) for k in ('extract_success','structural_ok','measurement_valid','physics_pass')},
            'overall_mean':statistics.fmean(current_values) if current_values else None,
            'legacy_overall_mean_0_100':statistics.fmean(legacy_values) if legacy_values else None,
            'continuous_score_mean_all':statistics.fmean(current_values) if current_values else 0.0,
            'continuous_score_mean_valid':statistics.fmean(valid_values) if valid_values else 0.0,
            'continuous_score_median_valid':statistics.median(valid_values) if valid_values else 0.0,
            'continuous_score_min_valid':min(valid_values) if valid_values else 0.0,
            'continuous_score_max_valid':max(valid_values) if valid_values else 0.0,
            'continuous_score_zero_count':sum(value==0.0 for value in current_values),
            'completed_processes':completed}
    (root/'batch_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
    raise SystemExit(1 if failures or len(results)!=24 else 0)
if __name__=='__main__': main()
