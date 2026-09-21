"""Single-video runner shared by all forty public evaluator entrypoints."""
import argparse
from copy import deepcopy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from .contract import (KEYS, already_scored_result, classical_result, finalize,
                       finite, measurement_block, metric, physeval_result)
from .physics import g3_result, g7_result
from . import consistency as consistency_gate

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(finite(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def relative(path, task):
    return os.path.relpath(Path(path).resolve(), task).replace(os.sep,'/')


def resolve(value, task):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return path.resolve() if path.exists() else (task/path).resolve()


def load_metadata(task, video, args):
    manifest = task/'metadata_v2.json'
    records = json.loads(manifest.read_text())['samples'] if manifest.exists() else []
    matches = [row for row in records if any(row.get(k) and (task/row[k]).resolve() == video for k in ('video_path','source_video_path'))]
    if len(matches) > 1:
        raise ValueError('Ambiguous video association in metadata_v2.json')
    row = deepcopy(matches[0]) if matches else {
        'sample_id':video.stem, 'video_path':relative(video,task), 'source_video_path':relative(video,task),
        'image_path':None, 'video_prompt':None, 'model':None, 'seed':None,
    }
    for key in ('sample_id','model','seed','video_prompt','route'):
        value = getattr(args,key,None)
        if value is not None:
            row[key] = value
    if args.image:
        row['image_path'] = relative(resolve(args.image,task),task)
    if args.prompt_file:
        row['video_prompt'] = resolve(args.prompt_file,task).read_text().strip()
    # Record the file actually requested, even if a byte-identical source alias was used.
    row['video_path'] = relative(video,task)
    current_hash = digest(video)
    if matches and current_hash != row['video_sha256']:
        raise ValueError('Video hash differs from the frozen sample association')
    row['video_sha256'] = current_hash
    if row.get('image_path'):
        image = task/row['image_path']
        image_hash = digest(image)
        if matches and not args.image and row.get('image_sha256') != image_hash:
            raise ValueError('First-frame hash differs from frozen association')
        row['image_sha256'] = image_hash
    return row


def prepare_track_cache(task, metadata, debug):
    """Copy immutable, associated coordinate caches; never copy prior physical scores."""
    import yaml
    source = task/metadata.get('source_video_path',metadata['video_path'])
    original = task/'eval_results/debug'/source.stem
    source_cache = original/'tracks_raw.npz'
    config_path = task/'evaluator/config.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    expected = {'video_sha256':metadata['video_sha256'],'config_sha256':digest(config_path),'evaluator_version':cfg['evaluator_version']}
    if task.name == 'P3':
        cache_meta = original/'tracks_raw.meta.json'
        if not cache_meta.exists() or json.loads(cache_meta.read_text()) != expected:
            raise ValueError('P3 cached tracking video/config hash mismatch')
        shutil.copy2(cache_meta,debug/cache_meta.name)
        association = 'original cache metadata matches video SHA-256, frozen config SHA-256 and evaluator version'
    else:
        job = json.loads((original/'track_job.json').read_text())
        sample = source.stem.rsplit('_seed',1)[0]
        if job['seeds'] != cfg['samples'][sample]:
            raise ValueError('P9 cached track-job seeds differ from frozen first-frame config')
        association = 'original per-video cache directory and track-job seeds; current video/config SHA-256 frozen in rerun manifest'
    if not source_cache.exists():
        raise FileNotFoundError(source_cache)
    shutil.copy2(source_cache,debug/'tracks_raw.npz')
    info = {**expected, 'cache_source':relative(source_cache,task),'cache_sha256':digest(source_cache),
            'association_validation':association,'neural_inference_rerun':False}
    write_json(debug/'track_cache_provenance.json',info)
    return info


def backend_command(task, row, raw_path, debug, forward):
    group = task.parent.name
    video = task/row['video_path']
    command = [sys.executable, str(task/'evaluator/measure_backend.py')]
    if group in ('g1','g4','g6'):
        command += ['--video',str(video),'--image',str(task/row['image_path']), '--out',str(raw_path),
                    '--debug',str(debug/'measurement.png'),'--sample-id',row['sample_id']]
        prompt_path = debug/'video_prompt.txt'
        prompt_path.write_text(row.get('video_prompt') or '')
        command += ['--prompt',str(prompt_path)]
        if row.get('route'):
            command += ['--route',row['route']]
    elif group in ('g2','g5'):
        command += ['--video',str(video),'--output',str(raw_path),'--image-path',str(task/row['image_path']), '--sample-id',row['sample_id']]
    elif group == 'g3':
        source_video = task/row.get('source_video_path',row['video_path'])
        if digest(source_video) != row['video_sha256']:
            raise ValueError('Original-filename input is not byte-identical to canonical video')
        command = [sys.executable,str(task/'evaluator/evaluate_raw_legacy.py'), '--video',str(source_video), '--task_id',task.name, '--output',str(raw_path)]
        command += ['--artifacts-dir' if task.name == 'P6' else '--debug-dir',str(debug)]
        if task.name == 'P4':
            command += ['--no-cotracker']
        return command + forward
    elif group == 'g7':
        return [sys.executable,str(task.parent/'evaluator/evaluate.py'),'--video',str(video),'--task_id',task.name,'--sample_id',row['sample_id'],'--output',str(raw_path),'--debug-dir',str(debug)] + forward
    else:
        command += ['--video_path',str(video),'--output',str(raw_path),'--debug_dir',str(debug),'--sample_id',row['sample_id']]
        if row.get('image_path'):
            command += ['--image_path',str(task/row['image_path'])]
        command += ['--video_prompt',row.get('video_prompt') or '']
        if row.get('annotation'):
            command += ['--annotation',str(task/row['annotation'])]
        if task.name != 'P47':
            command += ['--reuse_masks']
            if task.name != 'P34':
                command += ['--reuse_tracks']
        command += ['--device','cpu','--threads','2']
    if row.get('model'):
        command += ['--model',row['model']]
    if row.get('seed') is not None:
        command += ['--seed',str(row['seed'])]
    return command + forward


def artifacts(debug, task, started):
    kinds = {'.png':'image','.jpg':'image','.jpeg':'image','.mp4':'video','.csv':'measurement_data','.npz':'coordinate_cache','.json':'measurement_data'}
    found = []
    for path in sorted(debug.rglob('*')):
        if path.is_file() and path.suffix.lower() in kinds and (path.stat().st_mtime >= started-1 or path.name == 'tracks_raw.npz'):
            found.append({'path':relative(path,task),'kind':kinds[path.suffix.lower()]})
    return found


def ensure_visual_evidence(evidence, video, debug, task):
    """Show actual input frames when extraction stopped before a geometry plot existed."""
    if any(e['kind'] in ('image','video') for e in evidence):
        return evidence
    import cv2
    import numpy as np
    capture=cv2.VideoCapture(str(video))
    count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    panels=[]
    for index in sorted({0,max(0,count//2),max(0,count-1)}):
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=capture.read()
        if not ok:
            continue
        scale=min(480/frame.shape[1],340/frame.shape[0])
        frame=cv2.resize(frame,(int(frame.shape[1]*scale),int(frame.shape[0]*scale)))
        panel=np.full((390,500,3),245,dtype=np.uint8)
        cv2.putText(panel,f'Input frame {index}; measurement unavailable',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(30,30,30),1,cv2.LINE_AA)
        panel[35:35+frame.shape[0],10:10+frame.shape[1]]=frame
        panels.append(panel)
    capture.release()
    if panels:
        path=debug/'extraction_diagnostic.png'
        cv2.imwrite(str(path),np.concatenate(panels,axis=1))
        evidence.append({'path':relative(path,task),'kind':'image','description':'Decoded source frames for extraction-failure review; no measured geometry is asserted.'})
    return evidence


def scoring_text(result):
    summary = result['verbose']['M1'].get('_scoring_summary', {})
    sample_id = result['verbose']['M1'].get('_sample_id')
    lines = ['# Measurement and scoring','',f"Task: {result['task_id']}; sample: {sample_id}", '']
    if '_consistency' in result['verbose']['M1']:
        gate = result['verbose']['M1']['_consistency']
        lines += ['## V3 consistency gate', '',
                  f"Consistency = {gate.get('score')}; threshold = {gate.get('threshold')}; passed = {gate.get('passed')}",
                  str(gate.get('reason', '')), f"Physics attempted = {summary.get('physics_attempted')}",
                  'Total = 0.15 × consistency + 0.85 × mean defined pure physics score.',
                  'When consistency fails, physics is skipped and only 0.15 × consistency is retained.',
                  'V2 recognition bonuses are not added to V3 totals.', '']
    for key in KEYS:
        m,v = result['metrics'][key],result['verbose'][key]
        scoring = v.get('scoring', {})
        lines += [f'## {key}', '', str(v.get('principle','')), '', f"extract_success = {m['extract_success']}",
                  f"raw metric = {json.dumps(v.get('raw_metric',v.get('metric')),ensure_ascii=False)}",
                  f"metric (normalized final score) = {json.dumps(m['metric'])}", f"physics_score = {scoring.get('physics_score')}",
                  f"recognition_score = {scoring.get('recognition_score')}", f"proxy_score = {scoring.get('proxy_score')}",
                  scoring.get('proxy_formula', ''), scoring.get('proxy_calculation', ''), '']
    lines += ['Only defined indicators enter the denominator. A failed defined indicator contributes zero without redistributing its weight.',
              f"Total: {summary.get('formula')} = {summary.get('score')}", summary.get('calculation', ''), '']
    return '\n'.join(lines)


def publish(result, output, debug, task, raw_path, error):
    calculation = debug/'scoring_calculation.md'
    calculation.write_text(scoring_text(result))
    for block in result['verbose'].values():
        if block.get('defined'):
            block['scoring_calculation_path'] = relative(calculation,task)
            block['raw_result_path'] = relative(raw_path,task) if raw_path is not None else None
    write_json(output,result)
    summary = result['verbose']['M1']['_scoring_summary']
    print(json.dumps({'task_id':result['task_id'],'sample_id':result['verbose']['M1']['_sample_id'],
        'score':summary.get('score'),'consistency_score':summary.get('consistency_score'),
        'consistency_passed':summary.get('consistency_passed'),'physics_attempted':summary.get('physics_attempted'),
        'status':summary.get('score_status'),'runtime_error':error,'output':str(output)},ensure_ascii=False),flush=True)
    return 2 if error else (0 if any(m['extract_success'] is True for m in result['metrics'].values()) else 1)


def main(task_id, task_root, argv=None):
    task = Path(task_root).resolve()
    parser = argparse.ArgumentParser(description='V3: VLM consistency gate (15%), then physical evaluation (85%).')
    parser.add_argument('--video','--video_path',dest='video',required=True)
    parser.add_argument('--output','--out',dest='output',required=True)
    parser.add_argument('--image','--image_path','--image-path',dest='image')
    parser.add_argument('--video_prompt')
    parser.add_argument('--video_prompt_file','--prompt',dest='prompt_file')
    parser.add_argument('--debug-dir','--debug_dir','--debug',dest='debug')
    parser.add_argument('--sample-id','--sample_id',dest='sample_id')
    parser.add_argument('--model')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--route')
    parser.add_argument('--task_id','--task',choices=[task_id])
    consistency_gate.add_arguments(parser)
    args,forward = parser.parse_known_args(argv)
    gate_settings = consistency_gate.settings_from_args(args)
    try:
        gate_settings.validate()
    except ValueError as exc:
        parser.error(str(exc))
    video = resolve(args.video,task)
    output = resolve(args.output,task)
    row = load_metadata(task,video,args)
    default_debug = output.parent/('debug_'+row['sample_id']) if task.parent.name in ('g8','g9') else output.parent/'debug'/row['sample_id']
    debug = resolve(args.debug,task) if args.debug else default_debug
    if debug.suffix.lower() in ('.png','.jpg'):
        debug = debug.parent
    debug.mkdir(parents=True,exist_ok=True)
    raw_path = debug/'raw_result.json'
    # No old raw_result is accepted if the current backend fails to produce one.
    staged_raw = debug/'current_raw_result.json'
    if staged_raw.exists():
        staged_raw.replace(debug/f'previous_incomplete_raw_{time.time_ns()}.json')
    started = time.time()
    provenance = {'run_started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'video_sha256':row['video_sha256'],'image_sha256':row.get('image_sha256'),
                  'metadata_manifest':'metadata_v2.json','raw_result_path':relative(raw_path,task),
                  'group':task.parent.name,'execution_mode':'fresh_video_measurement',
                  'score_replay_only':False,'neural_inference_rerun':None,
                  'debug_dir':relative(debug,task),'physics_attempted':False}
    gate = consistency_gate.evaluate(video, task/row['image_path'] if row.get('image_path') else None,
                                     row.get('video_prompt'), task_id, debug, gate_settings)
    provenance['consistency_result_path'] = relative(debug/'consistency/consistency.json',task)
    if gate.get('status') != 'evaluated' or gate.get('passed') is not True:
        error = gate.get('reason') if gate.get('status') != 'evaluated' else None
        provenance.update(execution_mode='consistency_only', raw_result_path=None, backend_exit_code=None,
                          finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          elapsed_seconds=time.time()-started)
        if error:
            provenance['runtime_error'] = error
        metrics = {k:metric(defined=not (k == 'M2' and (task.parent.name in ('g8','g9') or task_id == 'P5'))) for k in KEYS}
        blocks = {k:{'defined':m['extract_success'] is not None,
                     'principle':'一致性检查未通过或未完成，物理测量未执行。'} for k,m in metrics.items()}
        evidence = ensure_visual_evidence(artifacts(debug,task,started),video,debug,task)
        result = finalize(task_id,row,metrics,blocks,evidence,provenance,
                          consistency=gate,physics_attempted=False)
        return publish(result,output,debug,task,None,error)
    provenance['physics_attempted'] = True
    raw = None
    error = None
    try:
        if task.parent.name == 'g3' and task_id in ('P3','P9'):
            provenance['track_cache'] = prepare_track_cache(task,row,debug)
            provenance['neural_inference_rerun'] = bool(provenance['track_cache'].get('neural_inference_rerun'))
            provenance['execution_mode'] = ('fresh_video_tracking_then_physics' if provenance['neural_inference_rerun']
                                            else 'fresh_physics_from_cached_coordinates_and_decoded_video')
        if task_id == 'P4':
            provenance['tracking_backends'] = ['color','bgsub']
            provenance['backend_note'] = 'Fresh CPU extraction with the existing --no-cotracker option; two independent backends and unchanged validity thresholds.'
        command = backend_command(task,row,staged_raw,debug,forward)
        command[0] = os.environ.get('EVALUATOR_MEASUREMENT_PYTHON',sys.executable)
        command = [command[0],str(ROOT/'unified_evaluators/execute_backend.py'),*command[1:]]
        provenance['command'] = command
        env = dict(os.environ, MPLCONFIGDIR='/tmp/g1_g9_matplotlib', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   EVALUATOR_USE_OPENCV4='1' if task.parent.name not in ('g8','g9') else '0')
        with (debug/'backend.log').open('w') as log:
            process = subprocess.run(command, cwd=task,env=env,stdout=log,stderr=subprocess.STDOUT)
        provenance['backend_exit_code'] = process.returncode
        if not staged_raw.exists():
            raise RuntimeError(f'Backend produced no result (exit {process.returncode}); see {relative(debug/"backend.log",task)}')
        if process.returncode not in (0,1,2):
            raise RuntimeError(f'Backend execution failed with exit code {process.returncode}')
        raw = json.loads(staged_raw.read_text())
        staged_raw.replace(raw_path)
        failure_text = json.dumps([raw.get('failure_reason'),raw.get('errors'),raw.get('statuses'),raw.get('verbose',{}).get('failure_reason')])
        if any(token in failure_text for token in ('TypeError:','IndexError:','fatal:','ModuleNotFoundError:','ImportError:','FileNotFoundError:')):
            raise RuntimeError(f'Measurement backend caught a program error: {failure_text[:1800]}')
        group = task.parent.name
        if group in ('g1','g4','g6'):
            metrics,blocks = physeval_result(raw)
        elif group in ('g2','g5'):
            metrics,blocks = classical_result(raw)
        elif group == 'g3':
            metrics,blocks = g3_result(raw)
        elif group == 'g7':
            metrics,blocks = g7_result(raw)
        else:
            metrics,blocks = already_scored_result(raw)
            provenance['extraction_run'] = raw.get('verbose',{}).get('M1',{}).get('extraction_run')
            provenance['execution_mode'] = 'fresh_video_measurement_with_validated_segmentation_cache' if task_id != 'P47' else 'fresh_video_measurement'
            extraction = provenance['extraction_run'] or {}
            provenance['neural_inference_rerun'] = bool(extraction) and not (
                extraction.get('sam2_cache_reused') is True and extraction.get('cotracker_cache_reused') is True)
            if raw.get('verbose',{}).get('M1',{}).get('status') in ('environment_error','configuration_error','runtime_error'):
                error = str(raw['verbose']['M1'].get('reason','Measurement environment failed'))
                provenance['runtime_error'] = error
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        provenance['runtime_error'] = error
        (debug/'runtime_error.txt').write_text(traceback.format_exc())
        metrics = {k:metric(defined=not (k == 'M2' and (task.parent.name in ('g8','g9') or task_id == 'P5'))) for k in KEYS}
        blocks = {k:measurement_block('测量运行未完成，详见运行日志。',None,reason=error) for k in KEYS}
        for k in KEYS:
            blocks[k]['defined'] = metrics[k]['extract_success'] is not None
    provenance['finished_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance['elapsed_seconds'] = time.time()-started
    evidence = artifacts(debug,task,started)
    evidence = ensure_visual_evidence(evidence,video,debug,task)
    result = finalize(task_id,row,metrics,blocks,evidence,provenance,consistency=gate,physics_attempted=True)
    return publish(result,output,debug,task,raw_path if raw is not None else None,error)
