"""Isolate a single generation job from the batch controller."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import read, save
from generation.backends import REGISTRY, GenResult


def seedance_generate(model, request, output, state_path):
    """Resume the same remote job after polling/download failures, without resubmitting."""
    started = time.monotonic()
    state = read(state_path) if state_path.exists() else {}
    if state and state.get('signature') != request['signature']:
        raise ValueError('Remote job belongs to different generation inputs; choose a new output directory')
    if state.get('status') == 'submitting' and not state.get('job_id'):
        raise RuntimeError('API submission outcome is unknown. Inspect remote.json and provider history before retrying; no duplicate job was submitted.')
    if not state.get('job_id'):
        save(state_path, dict(signature=request['signature'], status='submitting'))
        created, job_id, _ = model._create_job(first_frame=request['image'], prompt=request['prompt'],
                                              num_frames=request['num_frames'], seed=request['seed'])
        state = dict(signature=request['signature'], status='submitted', job_id=job_id, created=created)
        save(state_path, state)
    job_id = state['job_id']
    final, _ = model._poll_job(job_id)
    final['id'] = job_id
    model._download_result(final, str(output))
    save(state_path, {**state, 'status': 'downloaded'})
    return GenResult(str(output), time.monotonic() - started, f'Videos API model={model.model}', '',
                     dict(job_id=job_id, remote_model=model.model, requested_seconds=model.seconds,
                          seed_sent=False, seed_role='sample_label', effective_num_frames=final.get('num_frames')))


def run(request, request_path):
    profile = request['profile']
    raw = Path(request['raw_video'])
    raw.parent.mkdir(parents=True, exist_ok=True)
    backend = profile['backend']
    if backend == 'command':
        substitutions = {**request, 'request': str(request_path), 'output': str(raw)}
        command = [argument.format_map(substitutions) for argument in profile['command']]
        started = time.monotonic()
        subprocess.run(command, cwd=profile['cwd'], check=True)
        return dict(path=str(raw), seconds=time.monotonic() - started,
                    meta=dict(backend='command', seed_sent=True))
    model = REGISTRY[backend](**profile['options'])
    if backend == 'seedance-2.5':
        result = seedance_generate(model, request, raw, Path(request['remote_state']))
    else:
        result = model.generate(request['image'], request['prompt'], str(raw),
                                num_frames=request['num_frames'], seed=request['seed'],
                                log_path=str(raw.parent / 'model.log'))
    response = asdict(result)
    response.pop('log', None)
    response.setdefault('meta', {}).setdefault('seed_sent', backend != 'seedance-2.5')
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--response', type=Path, required=True)
    args = parser.parse_args()
    request = read(args.request)
    result = run(request, args.request)
    save(args.response, result)


if __name__ == '__main__':
    main()
