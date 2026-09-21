#!/usr/bin/env python3
"""Add the explainable four-way physical status to an existing V3 run.

This is a score-preserving migration: original result JSON and leaderboard
scores are left untouched.  The augmented index and the report are written to
a separate reliability directory so the change is auditable and reversible.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from reliability.protocol import get_protocol
from reliability.status import classify_result_status


def _metric(result, key):
    public = result.get('metrics', {}).get(key, {})
    block = result.get('verbose', {}).get(key, {})
    scoring = block.get('scoring', {}) if isinstance(block, dict) else {}
    return {
        'extract_success': public.get('extract_success'),
        'physics_score': scoring.get('physics_score'),
    }


def _row_status(row, result):
    metrics = {key: _metric(result, key) for key in ('M1', 'M2')}
    blocks = {key: result.get('verbose', {}).get(key, {}) for key in ('M1', 'M2')}
    attempted = bool(row.get('physics_attempted', result.get('verbose', {}).get('M1', {}).get('_scoring_summary', {}).get('physics_attempted', False)))
    threshold = result.get('verbose', {}).get('M1', {}).get('_scoring_summary', {}).get(
        'measurement_pass_threshold', get_protocol(row.get('task')).get('pass_threshold', .80))
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = .80
    status, reason, details = classify_result_status(metrics, blocks, physics_attempted=attempted, threshold=threshold)
    reliable = status in {'task_failed', 'physics_pass', 'physics_fail'}
    defined = [key for key, value in metrics.items() if value['extract_success'] is not None]
    measured = [key for key in defined if metrics[key]['extract_success'] is True]
    return {
        'measurement_status': status,
        'measurement_status_reason': reason,
        'measurement_pass_threshold': threshold,
        'metric_measurement_statuses': {key: value[0] for key, value in details.items()},
        'metric_measurement_reasons': {key: value[1] for key, value in details.items()},
        'measurement_coverage': len(measured) / len(defined) if defined else 0.0,
        'reliably_judged': reliable,
        'defined_metric_count': len(defined),
        'reliably_measured_metric_count': len(measured),
        'event_failure_inference_rule': 'never infer task failure from extraction failure alone',
    }


def _rate(n, d):
    return n / d if d else None


def _stats(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get(field) or 'unknown'].append(row)
    out = {}
    for name, items in sorted(grouped.items()):
        counts = Counter(item['measurement_status'] for item in items)
        judged = sum(item['reliably_judged'] for item in items)
        passed = sum(item['measurement_status'] == 'physics_pass' for item in items)
        out[name] = {
            'videos': len(items), 'status_counts': dict(sorted(counts.items())),
            'reliably_judged': judged, 'coverage_rate': _rate(judged, len(items)),
            'physics_pass_rate_judged': _rate(passed, judged), 'verified_pass_all_videos': _rate(passed, len(items)),
            'old_score_status_counts': dict(sorted(Counter(item.get('score_status') for item in items).items())),
        }
    return out


def _markdown(summary):
    lines = [
        '# V3 reliability status migration', '',
        '本报告只新增可解释的测量状态，不改变原有连续分数。提取失败不会自动解释为任务失败；只有结果中有明确的事件/条件证据时才使用 `task_failed`。', '',
        f"- 总视频：{summary['videos']}",
        f"- 可可靠判断：{summary['reliably_judged']} ({summary['coverage_rate']:.2%})",
        f"- 可可靠判断样本中的物理通过率：{summary['physics_pass_rate_judged']:.2%}" if summary['physics_pass_rate_judged'] is not None else '- 可可靠判断样本中的物理通过率：N/A',
        f"- 全部视频中经验证通过比例：{summary['verified_pass_all_videos']:.2%}",
        f"- 物理判断阈值（按任务校准）：{summary['pass_thresholds_by_task']}", '',
        '## 状态定义', '',
        '| 状态 | 含义 |', '|---|---|',
        '| `task_failed` | 有明确证据表明事件未完成或条件无效 |',
        '| `physics_pass` | 关键量可测且达到冻结物理阈值 |',
        '| `physics_fail` | 关键量可测但违反冻结物理阈值 |',
        '| `evidence_insufficient` | 证据不足，不能支持物理判断 |', '',
        '## 总状态计数', '', '| 状态 | 数量 |', '|---|---:|',
    ]
    for status, count in summary['status_counts'].items():
        lines.append(f'| `{status}` | {count} |')
    lines += ['', '## 按模型', '', '| Model | 视频 | 可判断覆盖率 | 可判断通过率 | 全部视频通过比例 | 状态计数 |', '|---|---:|---:|---:|---:|---|']
    for name, stats in summary['by_model'].items():
        lines.append(f"| {name} | {stats['videos']} | {stats['coverage_rate']:.2%} | {stats['physics_pass_rate_judged']:.2%} | {stats['verified_pass_all_videos']:.2%} | {stats['status_counts']} |" if stats['physics_pass_rate_judged'] is not None else f"| {name} | {stats['videos']} | {stats['coverage_rate']:.2%} | N/A | {stats['verified_pass_all_videos']:.2%} | {stats['status_counts']} |")
    lines += ['', '## 按物理领域/任务组', '', '| Group | 视频 | 可判断覆盖率 | 可判断通过率 | 全部视频通过比例 | 状态计数 |', '|---|---:|---:|---:|---:|---|']
    for name, stats in summary['by_group'].items():
        judged = f"{stats['physics_pass_rate_judged']:.2%}" if stats['physics_pass_rate_judged'] is not None else 'N/A'
        lines.append(f"| {name} | {stats['videos']} | {stats['coverage_rate']:.2%} | {judged} | {stats['verified_pass_all_videos']:.2%} | {stats['status_counts']} |")
    lines += ['', '## 与旧状态的关系', '', '旧 `score_status` 继续保留用于兼容；新 `measurement_status` 专门描述物理判断证据。`consistency_rejected`、`consistency_error` 和 `execution_error` 仍属于运行/门控状态，不被伪装成物理任务失败。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('v3_evaluator/results.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('v3_evaluator/reliability_status'))
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    augmented = []
    missing = []
    for row in rows:
        path = Path(row.get('output', ''))
        if not path.exists():
            missing.append(row.get('sample_id'))
            continue
        try:
            result = json.loads(path.read_text())
            update = _row_status(row, result)
        except Exception as exc:
            update = {'measurement_status': 'evidence_insufficient', 'measurement_status_reason': f'无法读取结果证据：{type(exc).__name__}: {exc}', 'measurement_coverage': 0.0, 'reliably_judged': False, 'defined_metric_count': 0, 'reliably_measured_metric_count': 0, 'metric_measurement_statuses': {}, 'metric_measurement_reasons': {}, 'measurement_pass_threshold': get_protocol(row.get('task')).get('pass_threshold', .80), 'event_failure_inference_rule': 'never infer task failure from extraction failure alone'}
        augmented.append({**row, **update})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_counts = Counter(row['measurement_status'] for row in augmented)
    judged = sum(row['reliably_judged'] for row in augmented)
    passed = sum(row['measurement_status'] == 'physics_pass' for row in augmented)
    summary = {
        'schema': 'vdmbench-reliability-status-v1', 'source_results': str(args.results),
        'videos': len(augmented), 'missing_result_files': missing,
        'status_counts': dict(sorted(summary_counts.items())), 'reliably_judged': judged,
        'coverage_rate': _rate(judged, len(augmented)), 'physics_pass_rate_judged': _rate(passed, judged),
        'verified_pass_all_videos': _rate(passed, len(augmented)),
        'pass_thresholds_by_task': {task: get_protocol(task).get('pass_threshold', .80) for task in sorted({row.get('task') for row in augmented})},
        'by_model': _stats(augmented, 'model'), 'by_group': _stats(augmented, 'group'),
        'policy': 'task_failed requires explicit event/condition evidence; extraction failure alone is evidence_insufficient',
    }
    (args.output_dir / 'status_augmented_results.json').write_text(json.dumps(augmented, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'status_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir / 'status_report.md').write_text(_markdown(summary))
    with (args.output_dir / 'status_augmented_results.csv').open('w', newline='') as f:
        fields = ['model', 'group', 'task', 'sample_id', 'score_status', 'measurement_status', 'measurement_coverage', 'reliably_judged', 'score', 'physics_score']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in augmented: writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({'output_dir': str(args.output_dir), 'videos': len(augmented), 'status_counts': dict(summary_counts), 'coverage_rate': summary['coverage_rate'], 'physics_pass_rate_judged': summary['physics_pass_rate_judged']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
