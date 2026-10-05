#!/usr/bin/env python3
"""Run the forty standalone PhysScope task evaluators over external videos."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
TASK_PATTERN = re.compile(r'(?:^|[_-])(P\d+[a-z]?)(?=[_.-]|$)')
SEED_PATTERN = re.compile(r'(?:^|[_-])seed(\d+)(?=[_.-]|$)')
SAFE_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
ERROR_STATES = {'input_error', 'execution_error', 'consistency_error'}
ENVIRONMENT_KEYS = (
    'FINAL_MODELS_DIR', 'FINAL_TRACKING_PYTHON', 'EVALUATOR_MEASUREMENT_PYTHON',
    'EVALUATOR_GROUNDING_DINO_MODEL', 'EVALUATOR_SAM2_SMALL_CHECKPOINT',
    'EVALUATOR_SAM2_TRANSFORMERS_MODEL', 'EVALUATOR_SAM2_LARGE_TRANSFORMERS_MODEL',
    'EVALUATOR_COTRACKER_CHECKPOINT', 'EVALUATOR_COTRACKER_CODE', 'SAM2_CKPT',
    'CUDA_VISIBLE_DEVICES',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def discover_tasks(root=ROOT):
    tasks = {}
    for difficulty in ('easy', 'medium', 'hard'):
        for folder in sorted((root / difficulty).glob('P*')):
            if not folder.is_dir():
                continue
            for required in ('task.md', 'first_frame.png', 'prompt.txt', 'evaluator/evaluate.py'):
                if not (folder / required).is_file():
                    raise ValueError(f'Incomplete task package: {folder / required}')
            if folder.name in tasks:
                raise ValueError(f'Duplicate task: {folder.name}')
            info = read(folder / 'evaluator/_shared/task_catalog.json')[folder.name]
            tasks[folder.name] = dict(path=str(folder), difficulty=difficulty, group=info['original_group'])
    if not tasks:
        raise ValueError('No task packages found beside evaluate.py')
    return tasks


def checked_name(value, label):
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f'{label} must contain only letters, numbers, dots, underscores or hyphens: {value!r}')
    return value


def make_job(row, base, tasks, default_model=None):
    task = row.get('task')
    if task not in tasks:
        raise ValueError(f'Unknown or missing task: {task!r}')
    folder = Path(tasks[task]['path'])

    def path(value):
        p = Path(value).expanduser()
        # Keep the original filename: calibrated backends use it to select a scene.
        return str(p.absolute() if p.is_absolute() else (base / p).absolute())

    video = path(row['video'])
    model = checked_name(row.get('model') or default_model, 'model')
    sample = checked_name(row.get('sample_id') or Path(video).stem, 'sample_id')
    seed = row.get('seed')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f'{sample}: seed must be an integer or null')
    backend_args = row.get('backend_args', [])
    if not isinstance(backend_args, list) or not all(isinstance(x, str) for x in backend_args):
        raise ValueError(f'{sample}: backend_args must be an array of strings')
    reserved = {'--video', '--video_path', '--output', '--out', '--image', '--image_path', '--image-path',
                '--prompt', '--video_prompt', '--video_prompt_file', '--debug', '--debug-dir', '--debug_dir',
                '--sample-id', '--sample_id', '--model', '--seed', '--task', '--task_id', '--annotation'}
    if any(x.split('=')[0] in reserved or x.startswith('--consistency-') for x in backend_args):
        raise ValueError(f'{sample}: backend_args cannot override batch input/output or gate settings')
    return dict(task=task, model=model, sample_id=sample, seed=seed, video=video,
                image=path(row['image']) if row.get('image') else str(folder / 'first_frame.png'),
                prompt=path(row.get('prompt') or row.get('prompt_source')) if row.get('prompt') or row.get('prompt_source') else str(folder / 'prompt.txt'),
                annotation=path(row['annotation']) if row.get('annotation') else None,
                backend_args=backend_args, **tasks[task])


def scan_videos(video_root, tasks, model=None):
    rows = []
    for video in sorted(video_root.rglob('*')):
        if not video.is_file() or video.suffix.lower() not in {'.mp4', '.mov', '.mkv', '.avi', '.webm'}:
            continue
        relative = video.relative_to(video_root)
        sample = video.parent.name if video.stem == 'base' and TASK_PATTERN.search(video.parent.name) else video.stem
        candidates = set(TASK_PATTERN.findall(sample))
        if not candidates:
            candidates = {p for p in relative.parts[:-1] if p in tasks}
        if len(candidates) != 1 or not candidates.issubset(tasks):
            raise ValueError(f'Cannot identify one supported task for {video}; provide --manifest')
        generator = model or (relative.parts[0] if len(relative.parts) > 1 and relative.parts[0] not in tasks else None)
        match = SEED_PATTERN.search(sample)
        row = dict(video=str(video), task=next(iter(candidates)), model=generator,
                   sample_id=sample, seed=int(match.group(1)) if match else None)
        generation_config = video.with_name(video.stem + '_config.json')
        if generation_config.is_file():
            row['prompt'] = str(generation_config)
        rows.append(row)
    return rows


def prepare_jobs(rows, base, tasks, default_model=None, selected=None, annotation_root=None):
    jobs, duplicate_files, seen = [], [], {}
    for row in rows:
        if selected and row.get('task') not in selected:
            continue
        job = make_job(row, base, tasks, default_model)
        if annotation_root:
            candidate = annotation_root / job['model'] / job['task'] / (job['sample_id'] + '.json')
            if candidate.is_file():
                job['annotation'] = str(candidate.absolute())
        if not job['annotation']:
            template = Path(job['path']) / 'first_frame_annotations.json'
            if template.is_file():
                job['annotation'] = str(template)
        job['files'], job['input_errors'] = {}, []
        for name in ('video', 'image', 'prompt', 'annotation'):
            if job[name] is None:
                continue
            try:
                job['files'][name] = dict(path=job[name], sha256=sha(job[name]))
            except OSError as exc:
                job['input_errors'].append(f'{name}: {type(exc).__name__}: {exc}')
        key = (job['model'], job['task'], job['sample_id'])
        if key in seen:
            previous = seen[key]
            same_inputs = all(job['files'].get(n, {}).get('sha256') == previous['files'].get(n, {}).get('sha256')
                              for n in ('video', 'image', 'prompt', 'annotation'))
            if job['input_errors'] or previous['input_errors'] or not same_inputs or job['backend_args'] != previous['backend_args'] or job['seed'] != previous['seed']:
                raise ValueError(f'Conflicting duplicate sample {key}; use distinct sample_id values in --manifest')
            duplicate_files.append(dict(video=job['video'], same_as=previous['video'], video_sha256=job['files']['video']['sha256']))
            continue
        seen[key] = job
        jobs.append(job)
    if not jobs:
        raise ValueError('No input videos selected')
    return sorted(jobs, key=lambda j: (j['model'], j['task'], j['sample_id'])), duplicate_files


def task_fingerprint(folder):
    folder = Path(folder)
    return canonical({str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
                      if p.is_file() and p.suffix in {'.py', '.json', '.yaml', '.yml', '.txt'}
                      and not {'__pycache__', '.venv', 'models', '.models', '.cache'}.intersection(p.relative_to(folder).parts)})


def command_for(job, directory, python, gate, measurement_device=None):
    command = [python, str(Path(job['path']) / 'evaluator/evaluate.py'),
               '--video', job['video'], '--image', job['image'], '--prompt', job['prompt'],
               '--output', str(directory / 'result.json'), '--debug-dir', str(directory / 'debug'),
               '--sample-id', job['sample_id'], '--model', job['model']]
    if job['seed'] is not None:
        command += ['--seed', str(job['seed'])]
    if job['annotation']:
        command += ['--annotation', job['annotation']]
    for name, value in gate.items():
        command += ['--consistency-' + name.replace('_', '-'), str(value)]
    if measurement_device and (job['group'] in ('g8', 'g9') or job['task'] in ('P3', 'P9')):
        command += ['--device', measurement_device]
    command += job['backend_args']
    return command


def validate_result(result, job, validator):
    validator(result)
    if result['task']['id'] != job['task'] or result['sample']['id'] != job['sample_id'] or result['sample']['model'] != job['model']:
        raise ValueError('Evaluator returned a different task/model/sample')
    if job['seed'] is not None and result['sample'].get('seed') != job['seed']:
        raise ValueError('Evaluator returned a different seed')
    for name in ('video', 'image'):
        if result['sample'].get(name + '_sha256') != job['files'][name]['sha256']:
            raise ValueError(f'Evaluator returned a different {name} hash')


def stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            pass


def evaluate_job(job, output, config, resume, validator, active, active_lock, stopped):
    member = output / job['model'] / job['task'] / job['sample_id']
    record_path = member / 'batch_record.json'
    signature = canonical(dict(job=job, configuration=config))
    if resume and record_path.exists() and not job['input_errors']:
        previous = read(record_path)
        if previous.get('signature') == signature and previous.get('status') not in ERROR_STATES:
            try:
                if sha(previous['result_path']) != previous['result_sha256']:
                    raise ValueError('Stored result changed')
                validate_result(read(previous['result_path']), job, validator)
                return {**previous, 'resumed': True}
            except (OSError, ValueError, KeyError):
                pass
    record = {k: job[k] for k in ('model', 'task', 'sample_id', 'seed', 'video', 'difficulty')}
    record.update(signature=signature, status='input_error' if job['input_errors'] else 'execution_error',
                  consistency_score=None, consistency_passed=None, physics_score=None, physics_judgment=None,
                  measurement_coverage=None, total=None, error='; '.join(job['input_errors']),
                  result_path=None, result_sha256=None, resumed=False, exit_code=None)
    if job['input_errors']:
        save(record_path, record)
        return record
    member.mkdir(parents=True, exist_ok=True)
    attempt = max([int(p.name[7:]) for p in member.glob('attempt[0-9]*') if p.name[7:].isdigit()] or [-1]) + 1
    directory = member / f'attempt{attempt:03d}'
    directory.mkdir()
    command = command_for(job, directory, config['python'], config['gate'], config['measurement_device'])
    save(directory / 'request.json', dict(job=job, signature=signature, command=command, configuration=config))
    began = time.monotonic()
    process = None
    try:
        for name, info in job['files'].items():
            if sha(info['path']) != info['sha256']:
                raise ValueError(f'{name} changed after input preparation')
        with active_lock:
            if stopped.is_set():
                raise RuntimeError('Batch interrupted')
            log = (directory / 'evaluator.log').open('w')
            try:
                process = subprocess.Popen(command, cwd=job['path'], stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            finally:
                log.close()
            active.add(process)
        try:
            record['exit_code'] = process.wait(timeout=config['timeout'])
        except subprocess.TimeoutExpired:
            stop_process(process)
            raise RuntimeError(f'Evaluation exceeded {config["timeout"]} seconds')
        result_path = directory / 'result.json'
        if record['exit_code'] not in (0, 1, 2) or not result_path.exists():
            raise RuntimeError(f'Evaluator exited {record["exit_code"]}; inspect evaluator.log')
        result = read(result_path)
        validate_result(result, job, validator)
        if record['exit_code'] == 2 and result['status'] not in ERROR_STATES:
            raise ValueError('Evaluator reported an error exit code with a successful result')
        record.update(status=result['status'], consistency_score=result['consistency']['score'],
                      consistency_passed=result['consistency']['passed'], physics_score=result['physics']['score'],
                      physics_judgment=result['physics']['judgment'], measurement_coverage=result['physics']['coverage'],
                      total=result['score']['total'], result_path=str(result_path), result_sha256=sha(result_path),
                      error='; '.join(f['message'] for f in result['failures']) if result['status'] in ERROR_STATES else '')
    except Exception as exc:
        record.update(status='execution_error', error=f'{type(exc).__name__}: {exc}', total=None)
    finally:
        if process is not None:
            stop_process(process)
            with active_lock:
                active.discard(process)
    record['elapsed_seconds'] = time.monotonic() - began
    save(directory / 'batch_record.json', record)
    save(record_path, record)
    return record


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows):
    valid = [r for r in rows if r['status'] not in ERROR_STATES]
    return dict(videos=len(rows), evaluated=len(valid), errors=len(rows) - len(valid),
                scored=sum(r['total'] is not None for r in valid), total_mean=mean([r['total'] for r in valid]),
                consistency_passed=sum(r['consistency_passed'] is True for r in valid),
                consistency_mean=mean([r['consistency_score'] for r in valid]),
                physics_evaluated=sum(r['physics_score'] is not None for r in valid),
                physics_mean_attempted=mean([r['physics_score'] for r in valid]),
                measurement_coverage_mean=mean([r['measurement_coverage'] for r in valid]),
                status_counts=dict(Counter(r['status'] for r in rows)))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(output, records, planned, missing_tasks):
    records.sort(key=lambda r: (r['model'], r['task'], r['sample_id']))
    summary = dict(planned=planned, finished=len(records), complete=len(records) == planned and not any(r['status'] in ERROR_STATES for r in records),
                   overall=aggregate(records), missing_tasks_by_model=missing_tasks,
                   formula='0.15*C + 0.85*P if gate passes; 0.15*C if rejected; null on execution errors',
                   physics_mean_policy='Only attempted non-error physical evaluations; rejected videos have no independent physical score.')
    for field in ('model', 'task'):
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row)
        summary['by_' + field] = {k: aggregate(v) for k, v in sorted(groups.items())}
        write_csv(output / ('by_' + field + '.csv'), [{field: k, **v} for k, v in summary['by_' + field].items()])
    save(output / 'summary.json', summary)
    save(output / 'results.json', records)
    write_csv(output / 'results.csv', records)
    return summary


def main(argv=None):
    # setup.sh uses this layout; child evaluators inherit the resolved model root.
    model_root = Path(os.environ.get('FINAL_MODELS_DIR', ROOT / 'models')).expanduser().resolve()
    os.environ['FINAL_MODELS_DIR'] = str(model_root)
    tasks = discover_tasks()
    shared = Path(next(iter(tasks.values()))['path']) / 'evaluator/_shared/unified_evaluators'
    gate_module = load_module('physcope_batch_gate', shared / 'consistency.py')
    validator = load_module('physcope_batch_schema', shared / 'schema_v4.py').validate
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--video-root', type=Path, help='Scan MODEL/.../TASK-containing-filename.mp4 recursively')
    inputs.add_argument('--manifest', type=Path, help='JSON array of task/model/video rows; paths are relative to this file')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', help='Generator name for a flat or single-model input directory')
    parser.add_argument('--tasks', nargs='+', choices=sorted(tasks))
    parser.add_argument('--annotation-root', type=Path, help='Optional per-video overrides of bundled annotations: MODEL/TASK/SAMPLE_ID.json')
    parser.add_argument('--require-all-tasks', action='store_true', help='Require every selected task for every input model')
    parser.add_argument('--workers', type=int, default=1, help='Concurrent evaluator processes; default 1 to bound GPU use')
    parser.add_argument('--timeout', type=float, default=1800, help='Maximum seconds per evaluator process and its children')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter for the per-task entrypoints')
    parser.add_argument('--measurement-device', default='cuda:0', help='P3/P9/G8/G9 measurement device (default: cuda:0)')
    parser.add_argument('--resume', action='store_true', help='Reuse successful results only when input/code/settings signatures match')
    parser.add_argument('--dry-run', action='store_true', help='Validate inventory and write the plan without loading models')
    gate_module.add_arguments(parser)
    parser.set_defaults(consistency_model=os.environ.get('VLM_MODEL', str(model_root / 'Qwen3.6-27B')))
    for group in parser._action_groups:
        if group.title.startswith('V3 consistency gate'):
            group.title = 'Consistency gate (runs before physics)'
    args = parser.parse_args(argv)
    if args.workers < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('workers and timeout must be positive')
    try:
        gate = gate_module.settings_from_args(args)
        gate.validate()
        if args.manifest:
            source = args.manifest.expanduser().resolve()
            rows = read(source)
            if isinstance(rows, dict):
                rows = rows.get('items', rows.get('samples'))
            if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
                raise ValueError('Manifest must be a JSON array of objects (or an object with items/samples)')
            base = source.parent
        else:
            base = args.video_root.expanduser().resolve()
            if not base.is_dir():
                raise ValueError(f'Video root does not exist: {base}')
            rows = scan_videos(base, tasks, args.model)
        jobs, duplicates = prepare_jobs(rows, base, tasks, args.model, set(args.tasks or []), args.annotation_root)
        selected = set(args.tasks or tasks)
        missing = {model: sorted(selected - {j['task'] for j in jobs if j['model'] == model}) for model in sorted({j['model'] for j in jobs})}
        if args.require_all_tasks and any(missing.values()):
            raise ValueError('Missing tasks by model: ' + json.dumps(missing))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another batch process is using this output directory')
        if not args.resume and not args.dry_run and any(output.glob('*/*/*/batch_record.json')):
            parser.error('Output already contains results; use --resume or a new --output directory')
        plan = dict(videos=len(jobs), tasks=len({j['task'] for j in jobs}), models=len(missing),
                    missing_tasks_by_model=missing, duplicates=duplicates, input_errors=sum(bool(j['input_errors']) for j in jobs), jobs=jobs)
        save(output / 'input_manifest.json', plan)
        if args.dry_run:
            print(json.dumps({k: v for k, v in plan.items() if k not in ('jobs', 'duplicates')}, ensure_ascii=False))
            return 2 if plan['input_errors'] else 0
        fingerprints = {task: task_fingerprint(tasks[task]['path']) for task in {j['task'] for j in jobs}}
        config = dict(batch_version=1, batch_sha256=sha(__file__), python=args.python, gate=asdict(gate),
                      measurement_device=args.measurement_device, timeout=args.timeout,
                      environment={k: os.environ.get(k) for k in ENVIRONMENT_KEYS}, task_fingerprints=fingerprints)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        save(output / 'configurations' / (stamp + '.json'), config)
        active, active_lock, stopped = set(), threading.Lock(), threading.Event()
        records = []
        executor = ThreadPoolExecutor(max_workers=args.workers)
        futures = []
        interrupted = False
        try:
            futures = [executor.submit(evaluate_job, job, output, config, args.resume, validator, active, active_lock, stopped) for job in jobs]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(json.dumps(dict(finished=len(records), planned=len(jobs), **{k: record[k] for k in ('model', 'task', 'sample_id', 'status', 'total', 'resumed')}), ensure_ascii=False), flush=True)
                save(output / 'progress.json', dict(finished=len(records), planned=len(jobs), errors=sum(r['status'] in ERROR_STATES for r in records)))
        except KeyboardInterrupt:
            interrupted = True
            stopped.set()
            for future in futures:
                future.cancel()
            with active_lock:
                for process in list(active):
                    stop_process(process)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        summary = summarize(output, records, len(jobs), missing)
        print(json.dumps({k: summary[k] for k in ('planned', 'finished', 'complete', 'overall')}, ensure_ascii=False))
        return 130 if interrupted else 0 if summary['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
