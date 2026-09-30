#!/usr/bin/env python3
"""Run every external all_test video with persistent local VLM GPU workers.

Each worker retains only model weights between videos. Input-specific physics,
tracks, segmentation and consistency decisions are freshly evaluated. Completed
V3 results can be resumed only when the input signature and result hash match.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from all_test_common import ROOT, V3, SOURCE, ANNOTATED, digest, now, write_json

sys.path.insert(0, str(V3))
from unified_evaluators import runtime
from unified_evaluators.consistency import Settings, RUBRIC, RUBRIC_VERSION

PATTERN = re.compile(r'^(g[1-9])_(P[^_]+)_seed(\d+)$')
GOOD_STATUSES = {'complete', 'partial', 'unavailable', 'consistency_rejected'}
DEVICE = 'cuda:0'


def snapshot():
    files = {}
    excluded = {'.venv', '__pycache__', '.runtime', '.models', 'models', 'work', 'eval_results', 'results', 'tests'}
    for parent, dirs, names in os.walk(V3):
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(names):
            p = Path(parent) / name
            if p.suffix in {'.py', '.yaml', '.yml'}:
                files[str(p.relative_to(V3))] = digest(p)
    # Reporting/preparation code is recorded but does not invalidate finished measurements.
    measured = {p: h for p, h in files.items() if not p.startswith('scripts/')}
    measured['scripts/run_all_test.py'] = files['scripts/run_all_test.py']
    return {'files': files, 'measurement_sha256': hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()}


def make_job(video, frozen, threshold):
    m = PATTERN.fullmatch(video.stem)
    if not m:
        raise ValueError(f'Unexpected video filename: {video}')
    source_group, task_id, seed = m.groups()
    model = video.parent.parent.name
    entries = list(V3.glob(f'g[1-9]/{task_id}/evaluator/evaluate.py'))
    if len(entries) != 1:
        raise ValueError(f'Expected one evaluator for {task_id}')
    task = entries[0].parent.parent
    group = task.parent.name
    out = ROOT / ('v3_' + model)
    image = task / 'first_frames' / 'gpt' / 'gpt_01.png'
    prompt_source = task / 'prompts' / 'video.txt'
    generation_manifest = video.parent / 'manifest.json'
    record = json.loads(generation_manifest.read_text()).get(video.stem, {}) if generation_manifest.exists() else {}
    generation_config = video.with_name(video.stem + '_config.json')
    config_record = json.loads(generation_config.read_text()) if generation_config.exists() else {}
    config_prompt = config_record.get('arguments', {}).get('prompt')
    prompt = config_prompt or record.get('prompt') or prompt_source.read_text()
    if config_prompt:
        prompt_source = generation_config
    elif record.get('prompt'):
        prompt_source = generation_manifest
    stored_prompt = out / 'inputs/prompts' / (video.stem + '.txt')
    stored_prompt.parent.mkdir(parents=True, exist_ok=True)
    stored_prompt.write_text(prompt.strip() + '\n')
    result_dir = out / group / task_id / 'eval_results' / model
    output = result_dir / ('result_' + video.stem + '.json')
    debug = result_dir / 'debug' / video.stem
    argv = ['--video', str(video), '--output', str(output), '--image', str(image),
            '--video_prompt_file', str(stored_prompt), '--debug-dir', str(debug),
            '--sample-id', video.stem, '--model', model, '--seed', seed, '--route', 'gpt',
            '--consistency-threshold', str(threshold), '--consistency-backend', 'local']
    inputs = {'video_sha256': digest(video), 'image_sha256': digest(image),
              'prompt_sha256': digest(stored_prompt), 'evaluator_sha256': frozen,
              'vlm_revision': '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
              'rubric_sha256': hashlib.sha256(RUBRIC.encode()).hexdigest()}
    if task_id in ANNOTATED:
        annotation = out / 'inputs/annotations' / (video.stem + '.json')
        argv += ['--annotation', str(annotation)]
        inputs['annotation_sha256'] = digest(annotation)
    config = out / 'inputs/configs' / (video.stem + '.yaml')
    if group == 'g3' and config.exists():
        argv += ['--config', str(config)]
        inputs['config_sha256'] = digest(config)
    signature = hashlib.sha256(json.dumps({'argv': argv, 'inputs': inputs}, sort_keys=True).encode()).hexdigest()
    return {'model': model, 'group': group, 'source_group': source_group, 'task': task_id,
            'task_root': str(task), 'seed': int(seed), 'sample_id': video.stem,
            'video': str(video), 'image': str(image), 'prompt_source': str(prompt_source),
            'output': str(output), 'debug': str(debug), 'argv': argv, 'inputs': inputs,
            'config': str(config) if config.exists() else None,
            'log': str(out / 'logs' / (video.stem + '.log')),
            'execution_record': str(out / 'execution' / (video.stem + '.json')),
            'signature': signature}


def fresh_tracking(job):
    import yaml
    config = Path(job['config'])
    task = Path(job['task_root'])
    cfg = yaml.safe_load(config.read_text())
    base = yaml.safe_load((task / 'evaluator/config.yaml').read_text())
    if {k: v for k, v in cfg.items() if k != 'samples'} != {k: v for k, v in base.items() if k != 'samples'}:
        raise ValueError('Input profile modified physical thresholds or runtime settings')
    if any(cfg['samples'].get(k) != v for k, v in base['samples'].items()):
        raise ValueError('Input profile modified existing fixtures')
    association = json.loads(config.with_suffix('.association.json').read_text())
    for path, key in [(job['video'], 'video_sha256'), (config, 'config_sha256'), (job['image'], 'source_image_sha256')]:
        if digest(path) != association[key]:
            raise ValueError(f'Initialization association mismatch: {key}')

    def create(task_root, metadata, debug):
        started = time.monotonic()
        device = DEVICE.rsplit(':', 1)[-1] if job['task'] == 'P3' else DEVICE
        raw = debug / 'fresh_tracking_measurement.json'
        command = [sys.executable, str(V3 / 'unified_evaluators/execute_backend.py'),
                   str(task / 'evaluator/evaluate_raw_legacy.py'), '--video', job['video'],
                   '--task_id', job['task'], '--config', str(config), '--output', str(raw),
                   '--debug-dir', str(debug), '--device', device, '--force-tracking']
        env = dict(os.environ, EVALUATOR_USE_OPENCV4='1')
        with (debug / 'fresh_tracking.log').open('w') as log:
            proc = subprocess.run(command, cwd=task, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        cache = debug / 'tracks_raw.npz'
        if proc.returncode not in (0, 1, 2) or not cache.exists() or not raw.exists():
            raise RuntimeError(f'Fresh tracking failed (exit {proc.returncode}); see {debug / "fresh_tracking.log"}')
        if json.loads(raw.read_text()).get('config_sha256') != digest(config):
            raise ValueError('Fresh tracking used an unexpected configuration')
        info = {'video_sha256': metadata['video_sha256'], 'config_sha256': digest(config),
                'cache_source': str(cache), 'cache_sha256': digest(cache),
                'evaluator_version': cfg['evaluator_version'], 'neural_inference_rerun': True,
                'association_validation': 'Fresh inference on this exact video, initialized from the reviewed input image.',
                'fresh_tracking_command': command, 'fresh_tracking_exit_code': proc.returncode,
                'fresh_tracking_seconds': time.monotonic() - started}
        write_json(debug / 'track_cache_provenance.json', info)
        return info
    return create


def initialize(device):
    global DEVICE
    DEVICE = device
    import cv2
    import torch
    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)


def inspect(job):
    row = dict(job)
    result = json.loads(Path(row['output']).read_text())
    block = result['verbose']['M1']
    provenance = block['_provenance']
    summary = block['_scoring_summary']
    gate = block['_consistency']
    if provenance['video_sha256'] != row['inputs']['video_sha256'] or result['model'] != row['model'] or result['seed'] != row['seed']:
        raise ValueError('Output association mismatch')
    passed = gate['passed'] is True
    if not passed and (summary['physics_attempted'] or provenance.get('backend_exit_code') is not None):
        raise ValueError('Physics ran after a failed consistency gate')
    if gate['status'] == 'evaluated':
        expected = .15 * gate['score'] + (.85 * (summary.get('physics_score') or 0) if passed else 0)
        if abs(summary['score'] - expected) > 1e-10:
            raise ValueError('V3 weighted score mismatch')
    elif summary['score'] is not None:
        raise ValueError('VLM error must not become a numeric score')
    row.update(score=summary['score'], score_status=summary['score_status'],
               consistency_score=gate['score'], consistency_passed=gate['passed'],
               physics_score=summary.get('physics_score'), physics_attempted=summary['physics_attempted'],
               physics_status=summary.get('physics_status'), reason=provenance.get('runtime_error') or gate.get('reason'),
               result_sha256=digest(row['output']),
               extraction_successes=sum(v['extract_success'] is True for v in result['metrics'].values()))
    return row


def run_one(job):
    job = dict(job, device=DEVICE, started_at_utc=now())
    record = Path(job['execution_record'])
    start = time.monotonic()
    output, debug = Path(job['output']), Path(job['debug'])
    if output.exists() or debug.exists():
        archive = output.parent / 'previous_attempts' / (job['sample_id'] + '_' + str(time.time_ns()))
        archive.mkdir(parents=True, exist_ok=True)
        for old in [output, debug, record, Path(job['log'])]:
            if old.exists():
                old.replace(archive / old.name)
    Path(job['log']).parent.mkdir(parents=True, exist_ok=True)
    write_json(record, job)
    argv = job['argv'] + ['--consistency-device', DEVICE]
    if job['group'] in {'g8', 'g9'}:
        argv += ['--device', DEVICE, '--threads', '2']
    old_prepare = runtime.prepare_track_cache
    try:
        with Path(job['log']).open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            if job['task'] in {'P3', 'P9'}:
                runtime.prepare_track_cache = fresh_tracking(job)
            job['returncode'] = runtime.main(job['task'], job['task_root'], argv)
        job = inspect(job)
    except BaseException as exc:
        job.update(score_status='execution_error', score=None, reason=f'{type(exc).__name__}: {exc}')
        with Path(job['log']).open('a') as log:
            log.write(traceback.format_exc())
    finally:
        runtime.prepare_track_cache = old_prepare
    job.update(finished_at_utc=now(), elapsed_seconds=time.monotonic() - start)
    write_json(record, job)
    return job


def resume(job):
    record = Path(job['execution_record'])
    if not record.exists():
        return None
    old = json.loads(record.read_text())
    if (old.get('signature') == job['signature'] and old.get('finished_at_utc')
            and old.get('score_status') in GOOD_STATUSES and Path(job['output']).exists()
            and old.get('result_sha256') == digest(job['output'])):
        return inspect(dict(old, resumed=True))
    return None


def progress(rows, total, start):
    info = {'updated_at_utc': now(), 'completed': len(rows), 'total': total,
            'remaining': total - len(rows), 'elapsed_seconds': time.monotonic() - start,
            'statuses': dict(Counter(r.get('score_status') for r in rows)),
            'models_completed': dict(Counter(r['model'] for r in rows))}
    write_json(ROOT / 'progress.json', info)
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', default='0,1,2,3,5,6,7')
    p.add_argument('--workers-per-gpu', type=int, default=1)
    p.add_argument('--threshold', type=float, default=.8)
    p.add_argument('--models', nargs='*')
    p.add_argument('--tasks', nargs='*')
    p.add_argument('--seeds', nargs='*', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    Settings(threshold=args.threshold).validate()
    for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        os.environ[k] = '2'
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      TOKENIZERS_PARALLELISM='false', MPLBACKEND='Agg')
    frozen = snapshot()
    write_json(ROOT / 'evaluator_snapshot.json', frozen)
    videos = sorted((SOURCE / 'data/videos/all_test').glob('*/gpt/*.mp4'))
    jobs = []
    blocked = []
    for video in videos:
        m = PATTERN.fullmatch(video.stem)
        if args.models and video.parent.parent.name not in args.models:
            continue
        if args.tasks and m.group(2) not in args.tasks:
            continue
        if args.seeds and int(m.group(3)) not in args.seeds:
            continue
        if not os.access(video, os.R_OK):
            blocked.append({'model': video.parent.parent.name, 'video': str(video),
                            'reason': 'permission_denied', 'mode': oct(video.stat().st_mode & 0o777)})
            continue
        try:
            jobs.append(make_job(video, frozen['measurement_sha256'], args.threshold))
        except PermissionError as exc:
            blocked.append({'model': video.parent.parent.name, 'video': str(video),
                            'reason': 'permission_denied_generation_metadata', 'file': exc.filename})
    write_json(ROOT / 'blocked_inputs.json', {'updated_at_utc': now(), 'count': len(blocked), 'videos': blocked})
    # Interleave generators for steady, directly comparable progress.
    jobs.sort(key=lambda j: (int(re.search(r'\d+', j['task']).group()), j['task'], j['seed'], j['model']))
    if args.limit:
        jobs = jobs[:args.limit]
    write_json(ROOT / 'run_manifest.json', {'created_at_utc': now(), 'arguments': vars(args),
        'evaluator': str(V3), 'threshold': args.threshold, 'weights': {'consistency': .15, 'physics': .85},
        'rubric_version': RUBRIC_VERSION, 'evaluator_sha256': frozen['measurement_sha256'], 'jobs': jobs})
    if args.prepare_only:
        print(json.dumps({'prepared': len(jobs)}), flush=True)
        return
    started = time.monotonic()
    rows, pending = [], []
    for job in jobs:
        old = resume(job)
        (rows if old else pending).append(old or job)
    print(json.dumps(progress(rows, len(jobs), started)), flush=True)
    pools = [ProcessPoolExecutor(max_workers=args.workers_per_gpu,
             mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
             initargs=(f'cuda:{gpu}',)) for gpu in args.gpus.split(',')]
    try:
        futures = {pools[i % len(pools)].submit(run_one, job): job for i, job in enumerate(pending)}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            info = progress(rows, len(jobs), started)
            print(json.dumps({**{k: info[k] for k in ['completed', 'total', 'statuses']},
                'last': [row['model'], row['sample_id'], row.get('score'), row.get('score_status')],
                'seconds': round(row.get('elapsed_seconds', 0), 2)}, ensure_ascii=False), flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    write_json(ROOT / 'results.json', rows)
    print(json.dumps(progress(rows, len(jobs), started)), flush=True)


if __name__ == '__main__':
    main()
