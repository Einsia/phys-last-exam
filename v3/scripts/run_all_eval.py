#!/usr/bin/env python3
"""Rerun G1--G9 videos through the public single-video entrypoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unified_evaluators.runtime import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups',default='g1,g2,g3,g4,g5,g6,g7,g8,g9')
    parser.add_argument('--tasks',default='')
    parser.add_argument('--samples',default='')
    parser.add_argument('--model',default=None,help='Folder/model alias; omitted means all recorded videos.')
    parser.add_argument('--workers',type=int,default=1,help='Local VLM workers each load model weights; default one GPU worker.')
    parser.add_argument('--log-dir',type=Path,default=ROOT/'work/v3_runs')
    parser.add_argument('--output-root',type=Path,default=ROOT/'results',help='V3 result root, separate from frozen task fixtures.')
    args,forward = parser.parse_known_args(argv)
    args.log_dir = args.log_dir.resolve()
    args.log_dir.mkdir(parents=True,exist_ok=True)
    groups = set(args.groups.lower().split(','))
    selected_tasks = set(args.tasks.split(',')) if args.tasks else None
    selected_samples = set(args.samples.split(',')) if args.samples else None
    jobs = []
    for path in sorted(ROOT.glob('g[1-9]/P*/metadata_v2.json')):
        task,group = path.parent,path.parent.parent.name
        if group not in groups or selected_tasks and task.name not in selected_tasks:
            continue
        for record in json.loads(path.read_text())['samples']:
            model_folder = record.get('model_folder',record.get('model') or 'unknown_model')
            if args.model and model_folder.lower().replace('-','_') != args.model.lower().replace('-','_'):
                continue
            if selected_samples and record['sample_id'] not in selected_samples:
                continue
            result_dir = (args.output_root.resolve()/group/task.name if args.output_root else task)/'eval_results'/model_folder
            output = result_dir/f"result_{record['sample_id']}.json"
            log = args.log_dir/f"{group}_{task.name}_{model_folder}_{record['sample_id']}.log"
            cmd = [sys.executable,str(task/'evaluator/evaluate.py'),'--video',str(task/record['video_path']), '--output',str(output),*forward]
            jobs.append({'group':group,'task_id':task.name,'sample_id':record['sample_id'],'model_folder':model_folder,
                'video_path':str((task/record['video_path']).relative_to(ROOT)), 'output':str(output),'log':str(log),'command':cmd})
    if not jobs:
        parser.error('No matching videos in metadata_v2.json')
    manifest = {'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'requested_samples':len(jobs),'completed':[]}
    write_json(args.log_dir/'run_manifest.json',manifest)
    def execute(job):
        job = dict(job)
        job['started_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with Path(job['log']).open('w') as log:
            process = subprocess.run(job['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,MPLCONFIGDIR='/tmp/g1_g9_matplotlib',PYTHONDONTWRITEBYTECODE='1'))
        job.update(exit_code=process.returncode,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        output = Path(job['output'])
        if output.exists():
            result = json.loads(output.read_text())
            provenance = result.get('provenance',{}) or result.get('verbose',{}).get('M1',{}).get('_provenance',{})
            # A stale output never counts as a completed rerun.
            if provenance.get('run_started_at_utc','') < job['started_at_utc']:
                job['error'] = 'Output predates this invocation'
                job['exit_code'] = 1
            else:
                summary = result.get('verbose',{}).get('M1',{}).get('_scoring_summary',{})
                run_provenance = result.get('verbose',{}).get('M1',{}).get('_provenance',{})
                job.update(score=summary.get('score'),score_status=summary.get('score_status'),
                    consistency_score=summary.get('consistency_score'),consistency_passed=summary.get('consistency_passed'),
                    physics_score=summary.get('physics_score'),physics_attempted=summary.get('physics_attempted'),
                    metrics=result['metrics'],execution_mode=run_provenance.get('execution_mode'),runtime_error=run_provenance.get('runtime_error'))
                # CLI 1 is a completed rejection/unavailable measurement, not an execution error.
                if process.returncode == 1 and not run_provenance.get('runtime_error'):
                    job['evaluation_exit_code'] = 1
                    job['exit_code'] = 0
        else:
            job['error'] = 'Missing output'
            job['exit_code'] = 1
        return job
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute,job) for job in jobs]
        for future in as_completed(futures):
            job = future.result()
            manifest['completed'].append(job)
            write_json(args.log_dir/'run_manifest.json',manifest)
            print(f"{len(manifest['completed'])}/{len(jobs)} {job['group']}/{job['task_id']}/{job['model_folder']}/{job['sample_id']} rc={job['exit_code']} {job.get('score_status','missing')}",flush=True)
    manifest['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(args.log_dir/'run_manifest.json',manifest)
    # Task-level batch summaries use the same metric/total semantics as per-video JSON.
    dirs = {Path(row['output']).parent for row in manifest['completed']}
    for directory in dirs:
        subset = [row for row in manifest['completed'] if Path(row['output']).parent == directory]
        write_json(directory/'batch_summary.json',{'schema_version':'physical-bench-batch-v3-g1-g9','samples':subset})
    return int(any(row['exit_code'] for row in manifest['completed']))


if __name__ == '__main__':
    raise SystemExit(main())
