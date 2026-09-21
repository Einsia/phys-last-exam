#!/usr/bin/env python3
"""Run the existing task extractors on the analytical reliability controls."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import yaml


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOTS = {
    'P9': ROOT / 'v3/g3/P9',
    'P13': ROOT / 'v3/g2/P13',
    'P21': ROOT / 'v3/g4/P21',
    'P48': ROOT / 'v3/g6/P48',
}


def first_frame(video: Path, image: Path):
    cap = cv2.VideoCapture(str(video)); ok, frame = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(f'Cannot decode first frame: {video}')
    image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), frame)


def command(task_id, video, image, output, debug, item_id, python):
    task = TASK_ROOTS[task_id]
    if task_id == 'P9':
        return [python, str(task/'evaluator/evaluate_raw_legacy.py'), '--video', str(video), '--task_id', task_id,
                '--output', str(output), '--debug-dir', str(debug)]
    if task_id == 'P13':
        return [python, str(task/'evaluator/measure_backend.py'), '--video', str(video), '--image-path', str(image),
                '--output', str(output), '--sample-id', item_id]
    return [python, str(task/'evaluator/measure_backend.py'), '--video', str(video), '--image', str(image),
            '--out', str(output), '--debug', str(debug/'measurement.png'), '--sample-id', item_id]


def summarize(task_id, raw, truth):
    status = raw.get('status', {}) if isinstance(raw, dict) else {}
    metrics = raw.get('metrics', {}) if isinstance(raw, dict) else {}
    if task_id == 'P9':
        measured = bool(status.get('extract_success') and status.get('measurement_valid'))
        physical_pass = status.get('physics_pass')
        observed = raw.get('metrics', {}).get('m1_abs_residual') if isinstance(raw.get('metrics'), dict) else None
    else:
        flags = [m.get('extract_success') for m in metrics.values() if isinstance(m, dict) and m.get('extract_success') is not None]
        measured = bool(flags) and all(flags)
        physical_pass = all((m.get('physics_score') or 0) >= .8 for m in metrics.values() if isinstance(m, dict) and m.get('extract_success') is True) if measured else None
        observed = {key: value.get('metric') for key, value in metrics.items() if isinstance(value, dict)}
    return {'extractor_measurable': measured, 'extractor_physics_pass': physical_pass,
            'observed_measurement': observed, 'expected_status': truth.get('expected_status'),
            'truth_physics_residual': truth.get('physics_residual'), 'truth_event_completed': truth.get('event_completed')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('v3_evaluator/reliability_controls/manifest.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('v3_evaluator/reliability_controls/extractor_runs'))
    parser.add_argument('--python', default=os.environ.get('EVALUATOR_MEASUREMENT_PYTHON', os.sys.executable))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # P9 intentionally requires a frozen first-frame configuration.  Add the
    # analytical control geometry to a throw-away copy of that configuration;
    # the source evaluator config remains untouched.
    p9_config = args.output_dir / 'p9_control_config.yaml'
    base_config = yaml.safe_load((TASK_ROOTS['P9'] / 'evaluator/config.yaml').read_text())
    for item in manifest['items']:
        if item['task_id'] != 'P9':
            continue
        base_config['samples'][item['id']] = {
            'first_frame': str((args.manifest.parent / 'P9' / f'{item["variant"]}.png').resolve()),
            'short': {'pivot': [180, 92], 'bob': [180, 207], 'radius': 18},
            'long': {'pivot': [460, 92], 'bob': [460, 247], 'radius': 18},
        }
    p9_config.write_text(yaml.safe_dump(base_config, sort_keys=False))
    records = []
    for item in manifest['items']:
        task_id = item['task_id']; video = Path(item['video']).resolve(); item_id = item['id']
        video_eval = video
        if task_id == 'P9':
            # The frozen P9 entrypoint derives its sample key from the file
            # stem.  Give each control the corresponding configured stem while
            # preserving the original control path in the report.
            video_eval = (args.output_dir / 'P9_inputs' / f'{item_id}.mp4').resolve()
            video_eval.parent.mkdir(parents=True, exist_ok=True)
            if not video_eval.exists():
                shutil.copy2(video, video_eval)
        image = (args.manifest.parent / task_id / f'{item["variant"]}.png').resolve()
        if not image.exists(): first_frame(video, image)
        out = (args.output_dir / task_id / item_id / 'raw_result.json').resolve()
        debug = (args.output_dir / task_id / item_id / 'debug').resolve()
        out.parent.mkdir(parents=True, exist_ok=True); debug.mkdir(parents=True, exist_ok=True)
        cmd = command(task_id, video_eval, image, out, debug, item_id, args.python)
        if task_id == 'P9':
            cmd += ['--config', str(p9_config)]
        log = out.parent / 'backend.log'
        env = dict(os.environ, PYTHONUNBUFFERED='1', MPLCONFIGDIR='/tmp/vdmbench_reliability_mpl', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        proc = subprocess.run(cmd, cwd=TASK_ROOTS[task_id], env=env, stdout=log.open('w'), stderr=subprocess.STDOUT)
        raw = json.loads(out.read_text()) if out.exists() else {'runtime_error': f'exit {proc.returncode}; no result'}
        summary = summarize(task_id, raw, item['truth'])
        records.append({'id': item_id, 'task_id': task_id, 'variant': item['variant'], 'video': str(video),
                        'returncode': proc.returncode, 'raw_result': str(out), 'debug': str(debug), **summary})
    report = {'schema': 'vdmbench-control-extractor-run-v1', 'manifest': str(args.manifest),
              'items': len(records), 'records': records}
    (args.output_dir / 'extractor_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(args.output_dir/'extractor_report.json'), 'items': len(records),
                      'returncode_counts': {str(k): sum(r['returncode'] == k for r in records) for k in sorted(set(r['returncode'] for r in records))},
                      'measurable': sum(r['extractor_measurable'] for r in records)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
