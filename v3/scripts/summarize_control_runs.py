#!/usr/bin/env python3
"""Turn control truth and extractor outputs into a calibration report."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def observed_residual(task, raw):
    if task == 'P9':
        return finite(raw.get('metrics', {}).get('m1_abs_residual'))
    metrics = raw.get('metrics', {})
    m1 = metrics.get('M1', {}) if isinstance(metrics, dict) else {}
    return finite(m1.get('metric')) if isinstance(m1, dict) else None


def observed_physics(task, raw):
    metrics = raw.get('metrics', {})
    if task == 'P9':
        return raw.get('status', {}).get('physics_pass') if isinstance(raw.get('status'), dict) else None
    scores = []
    for value in metrics.values() if isinstance(metrics, dict) else []:
        if isinstance(value, dict) and value.get('extract_success') is True:
            score = finite(value.get('physics_score'))
            if score is not None: scores.append(score)
    return bool(scores) and all(score >= .8 for score in scores) if scores else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=Path('v3_evaluator/reliability_controls/manifest.json'))
    parser.add_argument('--extractor-report', type=Path, default=Path('v3_evaluator/reliability_controls/extractor_runs3/extractor_report.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('v3_evaluator/reliability_controls'))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.extractor_report.read_text())
    by_id = {item['id']: item for item in manifest['items']}
    records = []
    for run in report['records']:
        truth = by_id[run['id']]['truth']
        raw = json.loads(Path(run['raw_result']).read_text()) if Path(run['raw_result']).exists() else {}
        observed = observed_residual(run['task_id'], raw)
        truth_residual = finite(truth.get('physics_residual'))
        records.append({**run, 'truth': truth, 'observed_residual': observed,
                        'truth_residual_error': abs(observed-truth_residual) if observed is not None and truth_residual is not None else None,
                        'observed_physics_pass': observed_physics(run['task_id'], raw),
                        'expected_physics_label': truth.get('expected_status') == 'physics_pass'})
    confusion = Counter()
    for row in records:
        if row['truth']['expected_status'] == 'task_failed':
            continue
        expected = row['expected_physics_label']
        predicted = row['observed_physics_pass']
        confusion[('expected_pass' if expected else 'expected_fail', 'predicted_pass' if predicted else 'predicted_fail_or_unmeasurable')] += 1
    appearance = []
    for task in sorted({r['task_id'] for r in records}):
        a = next((r for r in records if r['task_id'] == task and r['variant'] == 'correct'), None)
        b = next((r for r in records if r['task_id'] == task and r['variant'] == 'appearance_dark'), None)
        if a and b:
            appearance.append({'task_id': task, 'correct_residual': a['observed_residual'], 'appearance_residual': b['observed_residual'],
                               'absolute_change': abs(a['observed_residual']-b['observed_residual']) if a['observed_residual'] is not None and b['observed_residual'] is not None else None,
                               'both_measurable': a['extractor_measurable'] and b['extractor_measurable']})
    errors = [r['truth_residual_error'] for r in records if r['truth_residual_error'] is not None]
    summary = {'schema': 'vdmbench-control-calibration-v1', 'controls': len(records),
               'measurable': sum(r['extractor_measurable'] for r in records),
               'measurable_rate': sum(r['extractor_measurable'] for r in records)/len(records),
               'truth_status_counts': dict(Counter(r['truth']['expected_status'] for r in records)),
               'extractor_returncode_counts': dict(Counter(str(r['returncode']) for r in records)),
               'residual_error_mean': sum(errors)/len(errors) if errors else None,
               'residual_error_max': max(errors) if errors else None,
               'physics_confusion': {f'{a}/{b}': n for (a,b),n in sorted(confusion.items())},
               'appearance_pairs': appearance,
               'records': records,
               'limitations': ['16 pilot controls only', 'P9 and P13 simplified scenes expose evidence-insufficient behavior and are not used as leaderboard samples', 'thresholds are provisional until a larger independent control set is collected']}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'calibration_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Reliability control calibration', '',
             '控制视频具有解析真值；本报告同时记录现有提取器的可测性和真值残差误差。控制集不计入模型排行榜。', '',
             f"- 控制视频：{summary['controls']}", f"- 提取器可测：{summary['measurable']} ({summary['measurable_rate']:.2%})",
             f"- 可计算真值残差的平均绝对误差：{summary['residual_error_mean']:.4f}" if summary['residual_error_mean'] is not None else '- 可计算真值残差的平均绝对误差：N/A',
             f"- 最大绝对误差：{summary['residual_error_max']:.4f}" if summary['residual_error_max'] is not None else '- 最大绝对误差：N/A', '',
             '## 控制样本', '', '| Task | Variant | 真值状态 | 提取可测 | 提取通过 | 真值残差 | 提取残差 | 残差绝对误差 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
        lines.append(f"| {r['task_id']} | {r['variant']} | `{r['truth']['expected_status']}` | {'是' if r['extractor_measurable'] else '否'} | {'是' if r['observed_physics_pass'] else '否'} | {fmt(finite(r['truth'].get('physics_residual')))} | {fmt(r['observed_residual'])} | {fmt(r['truth_residual_error'])} |")
    lines += ['', '## 外观扰动一致性', '', '| Task | 正确残差 | 深色外观残差 | 绝对变化 | 两者均可测 |', '|---|---:|---:|---:|---:|']
    for p in appearance:
        fmt = lambda x: 'N/A' if x is None else f'{x:.4f}'
        lines.append(f"| {p['task_id']} | {fmt(p['correct_residual'])} | {fmt(p['appearance_residual'])} | {fmt(p['absolute_change'])} | {'是' if p['both_measurable'] else '否'} |")
    lines += ['', '## 说明', '', '“提取不可测”只说明当前测量程序缺少证据，不等价于物理失败。P21 的未融化控制保留了明确事件真值，供后续事件检测器识别 `task_failed`。阈值采用当前控制集的试运行校准，并标记为 provisional，不能被解释为已完成大规模计量学验证。', '']
    (args.output_dir/'calibration_report.md').write_text('\n'.join(lines))
    print(json.dumps({'summary': str(args.output_dir/'calibration_summary.json'), 'report': str(args.output_dir/'calibration_report.md'), 'controls': len(records), 'measurable': summary['measurable']}, ensure_ascii=False))


if __name__ == '__main__': main()
