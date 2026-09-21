#!/usr/bin/env python3
"""Decode every reliability control and verify its stored analytical truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    checks = []
    for item in manifest['items']:
        path = Path(item['video'])
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
        first_ok, last_ok = False, False
        if opened:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            first_ok, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - 1))
            last_ok, _ = cap.read()
        cap.release()
        truth = item['truth']
        finite_residual = truth['physics_residual'] is None or math.isfinite(float(truth['physics_residual']))
        ok = bool(opened and first_ok and last_ok and count == int(item['frames']) and abs(fps - float(item['fps'])) < 0.5 and finite_residual)
        checks.append({'id': item['id'], 'ok': ok, 'opened': opened, 'decoded_first': bool(first_ok),
                       'decoded_last': bool(last_ok), 'frames': count, 'fps': fps,
                       'expected_frames': item['frames'], 'expected_status': truth['expected_status'],
                       'physics_residual': truth['physics_residual'], 'video': str(path)})
    result = {'schema': 'vdmbench-reliability-controls-validation-v1', 'manifest': str(args.manifest),
              'items': len(checks), 'passed': sum(c['ok'] for c in checks), 'failed': sum(not c['ok'] for c in checks),
              'checks': checks}
    report = args.report or args.manifest.with_name('validation.json')
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'report': str(report), 'items': result['items'], 'passed': result['passed'], 'failed': result['failed']}, ensure_ascii=False))
    raise SystemExit(0 if result['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
