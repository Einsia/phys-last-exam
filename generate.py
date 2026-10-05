#!/usr/bin/env python3
"""Generate PhysScope videos with a common interface for eight models or a custom command."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evaluate as batch
from generation.config import ANNOTATED, MODELS, SCENES, check_profile, default_config, resolve_profile

ROOT = Path(__file__).resolve().parent


def sample_id(task, group, seed):
    return f'{SCENES.get(task, group + "_" + task)}_seed{seed}'


def code_signature():
    files = [Path(__file__), *sorted((ROOT / 'generation').rglob('*.py'))]
    return batch.canonical({str(p.relative_to(ROOT)): batch.sha(p) for p in files})


def plan_jobs(tasks, models, profiles, seeds, output, signature):
    jobs = []
    for model in models:
        for task, info in tasks.items():
            folder = Path(info['path'])
            image, prompt = folder / 'first_frame.png', folder / 'prompt.txt'
            inputs = dict(image=batch.sha(image), prompt=batch.sha(prompt))
            annotation = folder / 'first_frame_annotations.json'
            if task in ANNOTATED:
                inputs['annotation'] = batch.sha(annotation)
            for seed in seeds:
                sid = sample_id(task, info['group'], seed)
                job = dict(model=model, task=task, seed=seed, sample_id=sid,
                           image=str(image), prompt_file=str(prompt), prompt=prompt.read_text().strip(),
                           annotation_template=str(annotation) if task in ANNOTATED else None,
                           input_hashes=inputs, profile=profiles[model], num_frames=profiles[model]['num_frames'],
                           video=str(output / model / (sid + '.mp4')),
                           requires_annotation=task in ANNOTATED)
                job['signature'] = batch.canonical(dict(model=model, task=task, seed=seed, inputs=inputs,
                                                        profile=profiles[model], code=signature))
                jobs.append(job)
    return jobs


def frozen_inputs(job):
    # Keep the evaluator manifest portable together with the videos directory.
    folder = Path(job['video']).parents[1] / '.inputs' / job['task']
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    sources = [('image', job['image'], '.png'), ('prompt', job['prompt_file'], '.txt')]
    if job.get('annotation_template'):
        sources.append(('annotation', job['annotation_template'], '.json'))
    for key, source, suffix in sources:
        if batch.sha(source) != job['input_hashes'][key]:
            raise ValueError(f'Task {key} changed after planning: {source}')
        target = folder / (job['input_hashes'][key] + suffix)
        if not target.exists():
            shutil.copyfile(source, target)
        if batch.sha(target) != job['input_hashes'][key]:
            raise ValueError(f'Generation input snapshot is corrupt: {target}')
        paths[key] = target
    return paths


def check_existing(job, resume):
    output = Path(job['video'])
    if not output.exists():
        return None
    if not resume:
        raise ValueError(f'Output exists: {output}; use --resume or a new --output directory')
    sidecar = output.with_name(output.stem + '_config.json')
    previous = batch.read(sidecar) if sidecar.exists() else {}
    if previous.get('generation_signature') != job['signature'] or previous.get('video_sha256') != batch.sha(output):
        raise ValueError(f'Existing output does not match these inputs/settings: {output}; use a new --output directory')
    return previous


def generate_job(job, work, resume, timeout):
    from generation.media import prepare_video
    output = Path(job['video'])
    sidecar = output.with_name(output.stem + '_config.json')
    directory = work / job['model'] / job['sample_id']
    directory.mkdir(parents=True, exist_ok=True)
    basic = {k: job[k] for k in ('model', 'task', 'seed', 'sample_id', 'video', 'signature', 'requires_annotation')}
    previous = check_existing(job, resume)
    if previous is not None:
        return {**basic, 'status': 'complete', 'resumed': True, 'media': previous['media']}
    inputs = frozen_inputs(job)
    attempt = directory / f'attempt{len(list(directory.glob("attempt*"))):03d}'
    attempt.mkdir()
    raw = attempt / 'raw.mp4'
    request = {**job, 'image': str(inputs['image']), 'prompt_file': str(inputs['prompt']),
               'raw_video': str(raw), 'output': str(raw), 'remote_state': str(directory / 'remote.json')}
    batch.save(attempt / 'request.json', request)
    command = [sys.executable, str(ROOT / 'generation/worker.py'),
               '--request', str(attempt / 'request.json'), '--response', str(attempt / 'response.json')]
    started = time.monotonic()
    with (attempt / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        try:
            rc = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            batch.stop_process(process)
            raise
    if rc:
        raise RuntimeError(f'Generation failed (exit {rc}); see {attempt / "worker.log"}')
    response = batch.read(attempt / 'response.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('.' + output.stem + '.partial.mp4')
    try:
        media = prepare_video(raw, temporary, job['task'])
        metadata = dict(schema_version='physscope.generation.v1', model=job['model'], task=job['task'],
                        generation_signature=job['signature'], video_sha256=batch.sha(temporary),
                        input_hashes=job['input_hashes'], profile=job['profile'], media=media,
                        arguments=dict(prompt=job['prompt'], image=str(inputs['image']), seed=job['seed'],
                                       num_frames=job['num_frames']), backend=response,
                        annotation_status='bundled_template_checked_during_evaluation' if job['requires_annotation'] else 'not_required')
        batch.save(sidecar, metadata)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    record = {**basic, 'status': 'complete', 'resumed': False, 'media': media,
              'elapsed_seconds': round(time.monotonic() - started, 3)}
    batch.save(directory / 'record.json', record)
    return record


def update_manifest(output, work, jobs, annotation_root=None):
    path = output / 'manifest.json'
    previous = batch.read(path) if path.exists() else []
    if not isinstance(previous, list):
        raise ValueError(f'Existing manifest is not a JSON array: {path}')
    entries = {(row['model'], row['task'], row['sample_id']): row for row in previous}
    for job in jobs:
        # Include every scheduled sample, including failures. Missing generations
        # become explicit evaluator input errors rather than vanishing from a cohort.
        inputs = frozen_inputs(job)
        row = dict(model=job['model'], task=job['task'], seed=job['seed'], sample_id=job['sample_id'],
                   video=os.path.relpath(job['video'], output),
                   image=os.path.relpath(inputs['image'], output), prompt=os.path.relpath(inputs['prompt'], output))
        if 'annotation' in inputs:
            row['annotation'] = os.path.relpath(inputs['annotation'], output)
        if annotation_root:
            annotation = annotation_root / job['model'] / job['task'] / (job['sample_id'] + '.json')
            if annotation.is_file():
                row['annotation'] = os.path.relpath(annotation, output)
        entries[(row['model'], row['task'], row['sample_id'])] = row
    batch.save(path, [entries[key] for key in sorted(entries)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', '--model', nargs='+', help='One or more model names, or all for the eight built-ins')
    parser.add_argument('--tasks', nargs='+', help='Task IDs; default: all 40')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45])
    parser.add_argument('--output', type=Path, default=ROOT / 'videos')
    parser.add_argument('--work-dir', type=Path, help='Raw videos and logs; default: runs/generation/OUTPUT_ID')
    parser.add_argument('--config', type=Path, default=ROOT / 'generation.local.json')
    parser.add_argument('--model-root', type=Path, default=ROOT / 'models/generation')
    parser.add_argument('--init-config', action='store_true', help='Write generation.local.json with paths inferred from --model-root')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check selected model paths/API configuration without generating')
    parser.add_argument('--dry-run', action='store_true', help='Write the task/seed plan without loading models or calling APIs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--annotation-root', type=Path, help='Optional MODEL/TASK/VIDEO_STEM.json overrides of bundled annotations')
    parser.add_argument('--timeout', type=float, default=3600, help='Seconds per video, including model startup')
    args = parser.parse_args(argv)
    try:
        defaults = default_config(args.model_root)
        if args.init_config:
            args.config.parent.mkdir(parents=True, exist_ok=True)
            with args.config.open('x') as stream:
                json.dump(defaults, stream, indent=2)
                stream.write('\n')
            print(f'Created {args.config}. Check the Python/source/checkpoint paths, then run --check.')
            return 0
        config = batch.read(args.config) if args.config.exists() else defaults
        definitions = config['models']
        if not isinstance(definitions, dict):
            raise ValueError('The config must contain a models object')
        if args.list_models:
            for model in dict.fromkeys([*MODELS, *definitions]):
                print(model)
            return 0
        if not args.models:
            raise ValueError('Choose --models MODEL [MODEL ...] or --models all')
        models = list(MODELS) if args.models == ['all'] else args.models
        if len(set(models)) != len(models) or any(m not in definitions for m in models):
            raise ValueError('Model names must be unique and configured; use --list-models')
        if not args.seeds or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds):
            raise ValueError('Seeds must be unique non-negative integers')
        if not 0 < args.timeout < float('inf'):
            raise ValueError('Timeout must be finite and positive')
        base = args.config.resolve().parent
        profiles = {name: resolve_profile(name, definitions[name], base) for name in models}
        if args.check or not args.dry_run:
            errors = []
            for name, profile in profiles.items():
                try:
                    check_profile(profile)
                except (OSError, ValueError, TypeError) as error:
                    errors.append(f'{name}: {error}')
            if errors:
                raise ValueError('\n'.join(errors))
            if importlib.util.find_spec('av') is None:
                raise ValueError('Activate the evaluation environment from setup.sh (PyAV is required for export validation)')
            if args.check:
                print(f'Configuration checks passed for {len(models)} model(s). No inference or API submission was performed.')
                return 0
        tasks = batch.discover_tasks()
        if args.tasks:
            if not set(args.tasks).issubset(tasks):
                raise ValueError('Unknown task ID')
            tasks = {key: value for key, value in tasks.items() if key in args.tasks}
        output = args.output.expanduser().resolve()
        work = (args.work_dir or ROOT / 'runs/generation' / batch.canonical(str(output))[:12]).expanduser().resolve()
        if work == output or output in work.parents or work in output.parents:
            raise ValueError('--work-dir and --output must be separate directories so raw videos are not evaluated twice')
        jobs = plan_jobs(tasks, models, profiles, args.seeds, output, code_signature())
        work.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            batch.save(work / 'plan.json', dict(jobs=jobs, output=str(output)))
            print(f'{len(models)} models × {len(tasks)} tasks × {len(args.seeds)} seeds = {len(jobs)} videos. Plan: {work / "plan.json"}')
            return 0
        output.mkdir(parents=True, exist_ok=True)
        with (output / '.generation.lock').open('a') as lock, (work / '.lock').open('a') as work_lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(work_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            binding = work / 'output.json'
            if binding.exists() and batch.read(binding)['output'] != str(output):
                raise ValueError('This --work-dir belongs to another output directory')
            batch.save(binding, dict(output=str(output)))
            for job in jobs:
                check_existing(job, args.resume)
            batch.save(work / 'plan.json', dict(jobs=jobs, output=str(output)))
            annotations = args.annotation_root.expanduser().resolve() if args.annotation_root else None
            update_manifest(output, work, jobs, annotations)
            records = []
            interrupted = False
            for job in jobs:
                try:
                    record = generate_job(job, work, args.resume, args.timeout)
                except KeyboardInterrupt:
                    interrupted = True
                    break
                except Exception as error:
                    record = {k: job[k] for k in ('model', 'task', 'sample_id')}
                    record.update(status='generation_error', error=f'{type(error).__name__}: {error}')
                records.append(record)
                batch.save(work / 'progress.json', records)
                print(json.dumps({k: record[k] for k in ('model', 'task', 'sample_id', 'status')}), flush=True)
            summary = dict(planned=len(jobs), finished=len(records), statuses=dict(Counter(r['status'] for r in records)),
                           annotation_tasks=sorted(ANNOTATED.intersection(tasks)), records=records,
                           evaluation_manifest=str(output / 'manifest.json'))
            batch.save(work / 'summary.json', summary)
            print(f'Evaluation manifest: {output / "manifest.json"}; generation logs: {work}')
            return 130 if interrupted else 2 if any(r['status'] != 'complete' for r in records) else 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
