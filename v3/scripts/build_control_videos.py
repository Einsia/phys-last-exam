#!/usr/bin/env python3
"""Render a small, deterministic reliability control set with frozen truth.

The controls are deliberately simple analytical scenes.  They are not used to
inflate the model leaderboard; they calibrate measurement and failure-state
logic.  Every MP4 has a sidecar entry in manifest.json containing the exact
geometry, event label, perturbation level, and expected judgment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


W, H, FPS, FRAMES = 640, 480, 24, 96


def _writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open MP4 writer: {path}')
    return writer


def _palette(appearance):
    if appearance == 'dark':
        return {
            'bg': (22, 25, 30), 'ink': (225, 230, 235), 'muted': (130, 145, 160),
            'red': (70, 90, 240), 'blue': (230, 150, 55), 'green': (90, 210, 120),
            'water': (185, 110, 55), 'ice': (245, 220, 140),
        }
    return {
        'bg': (248, 248, 245), 'ink': (35, 35, 35), 'muted': (120, 120, 120),
        'red': (45, 60, 220), 'blue': (210, 100, 30), 'green': (45, 170, 75),
        'water': (235, 180, 110), 'ice': (240, 220, 145),
    }


def _base(appearance, title):
    colors = _palette(appearance)
    frame = np.full((H, W, 3), colors['bg'], dtype=np.uint8)
    cv2.putText(frame, title, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                colors['ink'], 2, cv2.LINE_AA)
    return frame, colors


def _p9_frame(index, variant, appearance):
    frame, c = _base(appearance, 'P9 double pendulum control')
    t = index / FPS
    data = [(180, 115, 1.35, c['red']), (460, 155, 1.35 * math.sqrt(155 / 115), c['blue'])]
    error = {'correct': 1.0, 'deviation_5pct': 1.05, 'deviation_25pct': 1.25,
             'appearance_dark': 1.0}[variant]
    for n, (pivot_x, length, period, color) in enumerate(data):
        if n == 1:
            period *= error
        angle = 0.30 * math.sin(2 * math.pi * t / period + n * 0.8)
        pivot = (pivot_x, 92)
        bob = (int(pivot_x + length * math.sin(angle)), int(pivot[1] + length * math.cos(angle)))
        cv2.line(frame, pivot, bob, c['ink'], 4, cv2.LINE_AA)
        cv2.circle(frame, pivot, 8, c['green'], -1, cv2.LINE_AA)
        cv2.circle(frame, bob, 18, color, -1, cv2.LINE_AA)
        cv2.circle(frame, bob, 18, c['ink'], 2, cv2.LINE_AA)
        cv2.putText(frame, f'L{n+1}={length:.0f}px  T{n+1}={period:.2f}s',
                    (35 + n * 310, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c['ink'], 1, cv2.LINE_AA)
    return frame


def _p13_frame(index, variant, appearance):
    frame, c = _base(appearance, 'P13 reflection control')
    mirror_y = 300
    cv2.line(frame, (60, mirror_y), (580, mirror_y), c['ink'], 7, cv2.LINE_AA)
    cv2.line(frame, (320, mirror_y - 115), (320, mirror_y + 80), c['muted'], 2, cv2.LINE_AA)
    cv2.putText(frame, 'mirror', (485, mirror_y + 32), cv2.FONT_HERSHEY_SIMPLEX, .5, c['ink'], 1, cv2.LINE_AA)
    src, hit = (120, 120), (320, mirror_y)
    endpoint = {'correct': (520, 120), 'deviation_5deg': (520, 148),
                'deviation_25deg': (520, 245), 'appearance_dark': (520, 120)}[variant]
    progress = min(1.0, max(0.0, (index - 6) / 30.0))
    def partial(a, b):
        return (int(a[0] + progress * (b[0] - a[0])), int(a[1] + progress * (b[1] - a[1])))
    cv2.line(frame, src, partial(src, hit), c['red'], 5, cv2.LINE_AA)
    if progress >= 0.9:
        cv2.line(frame, hit, partial(hit, endpoint), c['blue'], 5, cv2.LINE_AA)
    cv2.circle(frame, hit, 8, c['green'], -1, cv2.LINE_AA)
    cv2.putText(frame, 'incident', (115, 105), cv2.FONT_HERSHEY_SIMPLEX, .48, c['red'], 1, cv2.LINE_AA)
    cv2.putText(frame, 'reflected', (490, 105), cv2.FONT_HERSHEY_SIMPLEX, .48, c['blue'], 1, cv2.LINE_AA)
    return frame


def _p21_frame(index, variant, appearance):
    frame, c = _base(appearance, 'P21 floating ice melt control')
    x0, x1, top, bottom = 180, 460, 120, 405
    water_y = 250
    cv2.rectangle(frame, (x0, top), (x1, bottom), c['ink'], 5)
    cv2.rectangle(frame, (x0 + 5, water_y), (x1 - 5, bottom - 5), c['water'], -1)
    if variant in ('correct', 'appearance_dark'):
        fraction = max(0.0, 1.0 - max(0, index - 10) / 78.0)
    elif variant == 'event_not_completed':
        fraction = 1.0
    elif variant == 'event_incomplete':
        fraction = max(0.42, 1.0 - max(0, index - 10) / 180.0)
    else:
        fraction = 1.0
    side = max(3, int(82 * math.sqrt(fraction)))
    cx, cy = 320, 214
    if side > 4:
        cv2.rectangle(frame, (cx - side, cy - side // 2), (cx + side, cy + side // 2), c['ice'], -1)
        cv2.rectangle(frame, (cx - side, cy - side // 2), (cx + side, cy + side // 2), c['ink'], 2)
    cv2.putText(frame, f'ice fraction={fraction:.2f}', (205, 445), cv2.FONT_HERSHEY_SIMPLEX, .56, c['ink'], 1, cv2.LINE_AA)
    return frame


def _p48_frame(index, variant, appearance):
    frame, c = _base(appearance, 'P48 droplet merger control')
    r1, r2 = 42.0, 32.0
    target = (r1 ** 3 + r2 ** 3) ** (1 / 3)
    factor = {'correct': 1.0, 'deviation_10pct': 1.10, 'deviation_30pct': 1.30,
              'appearance_dark': 1.0}[variant]
    merge_at = 52
    if index < merge_at:
        p = index / merge_at
        c1 = (int(215 + 65 * p), 255)
        c2 = (int(425 - 65 * p), 255)
        cv2.circle(frame, (c1[0], c1[1]), int(r1), c['red'], -1, cv2.LINE_AA)
        cv2.circle(frame, (c2[0], c2[1]), int(r2), c['blue'], -1, cv2.LINE_AA)
        cv2.circle(frame, (c1[0], c1[1]), int(r1), c['ink'], 2, cv2.LINE_AA)
        cv2.circle(frame, (c2[0], c2[1]), int(r2), c['ink'], 2, cv2.LINE_AA)
    else:
        radius = target * factor
        cv2.circle(frame, (320, 255), int(radius), c['green'], -1, cv2.LINE_AA)
        cv2.circle(frame, (320, 255), int(radius), c['ink'], 2, cv2.LINE_AA)
    cv2.putText(frame, f'volume residual target={target:.2f}px', (150, 445), cv2.FONT_HERSHEY_SIMPLEX, .54, c['ink'], 1, cv2.LINE_AA)
    return frame


def _truth(task_id, variant):
    appearance = variant == 'appearance_dark'
    if task_id == 'P9':
        L1, L2, T1 = 115.0, 155.0, 1.35
        error = {'correct': 1.0, 'deviation_5pct': 1.05, 'deviation_25pct': 1.25, 'appearance_dark': 1.0}[variant]
        T2 = T1 * math.sqrt(L2 / L1) * error
        residual = abs(((T1 / T2) ** 2) / (L1 / L2) - 1)
        expected = 'physics_pass' if residual <= .05 else 'physics_fail'
        return {'conditions_valid': True, 'event_completed': True, 'measurements': {'L1': L1, 'L2': L2, 'T1': T1, 'T2': T2},
                'constraint': '(T1/T2)^2=L1/L2', 'physics_residual': residual, 'expected_status': expected,
                'appearance_perturbation': appearance}
    if task_id == 'P13':
        deviation = {'correct': 0.0, 'deviation_5deg': 5.0, 'deviation_25deg': 25.0, 'appearance_dark': 0.0}[variant]
        expected = 'physics_pass' if deviation <= 2.0 else 'physics_fail'
        return {'conditions_valid': True, 'event_completed': True, 'measurements': {'angle_deviation_deg': deviation},
                'constraint': 'incident angle equals reflected angle', 'physics_residual': deviation / 90.0,
                'expected_status': expected, 'appearance_perturbation': appearance}
    if task_id == 'P21':
        final = {'correct': 0.0, 'event_not_completed': 1.0, 'event_incomplete': .42, 'appearance_dark': 0.0}[variant]
        complete = final <= .05
        return {'conditions_valid': True, 'event_completed': complete, 'measurements': {'initial_ice_fraction': 1.0, 'final_ice_fraction': final, 'level_change': 0.0},
                'constraint': 'pure floating ice melt keeps liquid level approximately constant', 'physics_residual': 0.0 if complete else None,
                'expected_status': 'physics_pass' if complete else 'task_failed', 'appearance_perturbation': appearance,
                'event_evidence': 'ice area reaches zero' if complete else 'ice remains in the final frames'}
    if task_id == 'P48':
        r1, r2 = 42.0, 32.0
        target = (r1 ** 3 + r2 ** 3) ** (1 / 3)
        factor = {'correct': 1.0, 'deviation_10pct': 1.10, 'deviation_30pct': 1.30, 'appearance_dark': 1.0}[variant]
        rf = target * factor
        residual = abs(rf ** 3 / (r1 ** 3 + r2 ** 3) - 1)
        return {'conditions_valid': True, 'event_completed': True, 'measurements': {'r1': r1, 'r2': r2, 'rf': rf, 'merge_frame': 52},
                'constraint': 'rf^3=r1^3+r2^3', 'physics_residual': residual,
                'expected_status': 'physics_pass' if residual <= .05 else 'physics_fail', 'appearance_perturbation': appearance}
    raise KeyError(task_id)


RENDERERS = {'P9': _p9_frame, 'P13': _p13_frame, 'P21': _p21_frame, 'P48': _p48_frame}
VARIANTS = {
    'P9': ['correct', 'deviation_5pct', 'deviation_25pct', 'appearance_dark'],
    'P13': ['correct', 'deviation_5deg', 'deviation_25deg', 'appearance_dark'],
    'P21': ['correct', 'event_not_completed', 'event_incomplete', 'appearance_dark'],
    'P48': ['correct', 'deviation_10pct', 'deviation_30pct', 'appearance_dark'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'v3_evaluator/reliability_controls')
    parser.add_argument('--tasks', nargs='+', choices=sorted(RENDERERS), default=sorted(RENDERERS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 'vdmbench-reliability-controls-v1', 'width': W, 'height': H, 'fps': FPS, 'frames': FRAMES, 'items': []}
    for task_id in args.tasks:
        for variant in VARIANTS[task_id]:
            path = args.output / task_id / f'{variant}.mp4'
            writer = _writer(path)
            try:
                for index in range(FRAMES):
                    writer.write(RENDERERS[task_id](index, variant, 'dark' if variant == 'appearance_dark' else 'light'))
            finally:
                writer.release()
            truth = _truth(task_id, variant)
            manifest['items'].append({'id': f'{task_id}_{variant}', 'task_id': task_id, 'variant': variant,
                                     'video': str(path), 'fps': FPS, 'frames': FRAMES, 'truth': truth,
                                     'control_class': ('appearance_only' if truth['appearance_perturbation'] else
                                                       'physics_correct' if truth['expected_status'] == 'physics_pass' else
                                                       'event_failure' if truth['expected_status'] == 'task_failed' else 'graded_physics_deviation')})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'items': len(manifest['items']), 'manifest': str(args.output / 'manifest.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
