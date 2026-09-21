#!/usr/bin/env python3
"""Compare direct Qwen-VL, protocol-guided Qwen-VL, and V3 results.

The sample is stratified by model, physics group, and old extraction status.
The run is resumable: each completed sample is appended to results.jsonl and
can survive an interrupted GPU process.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from reliability.vlm_compare import direct_messages, local_reply, parse_reply, rule_messages, sample_video


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'v3_evaluator/reliability_vlm'
DEFAULT_MODEL = ROOT / 'v3/.models/Qwen3-VL-8B-Instruct'


def _load(path):
    return json.loads(Path(path).read_text())


def _catalog():
    path = ROOT / 'v3_evaluator/task_catalog.json'
    if path.exists(): return _load(path)
    return {}


def _sample(rows, count, seed, status_source=None):
    rng = random.Random(seed)
    eligible = [row for row in rows if Path(row.get('video', '')).is_file()]
    groups = defaultdict(list)
    for row in eligible:
        groups[(row.get('model', 'unknown'), row.get('group', 'unknown'), row.get('score_status', 'unknown'))].append(row)
    for values in groups.values(): rng.shuffle(values)
    # Round-robin over strata gives every model/domain/status combination a
    # chance before larger strata fill the requested budget.
    keys = sorted(groups)
    selected = []
    cursor = {key: 0 for key in keys}
    while len(selected) < min(count, len(eligible)):
        progressed = False
        for key in keys:
            if cursor[key] < len(groups[key]) and len(selected) < count:
                selected.append(groups[key][cursor[key]])
                cursor[key] += 1; progressed = True
        if not progressed: break
    return selected


def _key(row):
    return f"{row.get('model','unknown')}::{row.get('sample_id') or Path(row.get('video','')).stem}"


def _spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 2: return None
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i]); out = [0.0] * len(values)
        i = 0
        while i < len(values):
            j = i
            while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]: j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1): out[order[k]] = rank
            i = j + 1
        return out
    a, b = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    da, db = [v-ma for v in a], [v-mb for v in b]
    den = math.sqrt(sum(v*v for v in da) * sum(v*v for v in db))
    return sum(x*y for x,y in zip(da,db))/den if den else 0.0


def _binary_agreement(a, b):
    values = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return sum(x == y for x, y in values) / len(values) if values else None


def _effective_vlm(item):
    """Only expose a physical pass/score when the VLM called it measurable."""
    if not isinstance(item, dict) or not item.get('measurable') or not isinstance(item.get('physics_pass'), bool):
        return None, None
    return item.get('physics_pass') if isinstance(item.get('physics_pass'), bool) else None, item.get('score')


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _pair_count(a, b):
    return sum(x is not None and y is not None for x, y in zip(a, b))


def summarize(records):
    parsed = [r for r in records if not r.get('error')]
    direct, guided = [r.get('direct', {}) for r in parsed], [r.get('rule', {}) for r in parsed]
    evaluator = [r.get('evaluator', {}) for r in parsed]
    direct_pass = [_effective_vlm(x)[0] for x in direct]
    guided_pass = [_effective_vlm(x)[0] for x in guided]
    eval_pass = [x.get('measurement_status') == 'physics_pass' if x.get('reliably_judged') else None for x in evaluator]
    eval_reliable = [x.get('reliably_judged') for x in evaluator]
    out = {
        'videos': len(records), 'completed_inference': len(parsed),
        'parse_errors_direct': sum(x.get('parse_error') for x in direct),
        'parse_errors_rule': sum(x.get('parse_error') for x in guided),
        'direct_rule_pass_agreement': _binary_agreement(direct_pass, guided_pass),
        'direct_rule_pass_pairs': _pair_count(direct_pass, guided_pass),
        'direct_rule_score_spearman': _spearman([_effective_vlm(x)[1] for x in direct], [_effective_vlm(x)[1] for x in guided]),
        'direct_evaluator_pass_agreement_on_reliable': _binary_agreement(direct_pass, eval_pass),
        'rule_evaluator_pass_agreement_on_reliable': _binary_agreement(guided_pass, eval_pass),
        'direct_evaluator_pass_pairs': _pair_count(direct_pass, eval_pass),
        'rule_evaluator_pass_pairs': _pair_count(guided_pass, eval_pass),
        'evaluator_reliable_coverage': sum(bool(x) for x in eval_reliable) / len(eval_reliable) if eval_reliable else None,
        'direct_measurable_rate': sum(bool(x.get('measurable')) for x in direct)/len(direct) if direct else None,
        'rule_measurable_rate': sum(bool(x.get('measurable')) for x in guided)/len(guided) if guided else None,
        'status_counts': dict(Counter(r.get('evaluator', {}).get('measurement_status', 'missing') for r in records)),
    }
    by_model = defaultdict(list)
    for r in records: by_model[r.get('model', 'unknown')].append(r)
    out['by_model'] = {}
    for model, values in sorted(by_model.items()):
        d = [_effective_vlm(r.get('direct', {}))[1] for r in values]
        g = [_effective_vlm(r.get('rule', {}))[1] for r in values]
        e = [r.get('evaluator', {}).get('physics_score') for r in values]
        out['by_model'][model] = {
            'videos': len(values),
            'direct_judged': sum(x is not None for x in d),
            'rule_judged': sum(x is not None for x in g),
            'evaluator_scored': sum(x is not None for x in e),
            'direct_score_mean': _mean(d), 'rule_score_mean': _mean(g), 'evaluator_physics_score_mean': _mean(e),
            'direct_measurable_rate': sum(bool(r.get('direct', {}).get('measurable')) for r in values)/len(values),
            'rule_measurable_rate': sum(bool(r.get('rule', {}).get('measurable')) for r in values)/len(values),
            'evaluator_reliable_rate': sum(bool(r.get('evaluator', {}).get('reliably_judged')) for r in values)/len(values),
            'evaluator_status_counts': dict(Counter(r.get('evaluator', {}).get('measurement_status', 'missing') for r in values)),
        }
    out['rankings'] = {
        'direct': sorted(((m, v['direct_score_mean']) for m,v in out['by_model'].items() if v['direct_score_mean'] is not None), key=lambda x:x[1], reverse=True),
        'rule': sorted(((m, v['rule_score_mean']) for m,v in out['by_model'].items() if v['rule_score_mean'] is not None), key=lambda x:x[1], reverse=True),
        'evaluator': sorted(((m, v['evaluator_physics_score_mean']) for m,v in out['by_model'].items() if v['evaluator_physics_score_mean'] is not None), key=lambda x:x[1], reverse=True),
    }
    return out


def markdown(summary):
    lines = ['# V3 real-video VLM reliability comparison', '',
             '同一批分层抽样视频上比较当前 Qwen3-VL 的直接判断、提供物理协议后的判断，以及 V3 的提取结果。这里的 VLM 只作比较对象；没有把 VLM 输出当成人工真值。', '',
             f"- 样本数：{summary['videos']}（完成推理：{summary['completed_inference']}）",
             f"- Direct 与 rule 的通过/不通过一致率：{summary['direct_rule_pass_agreement']:.2%}（共同有物理判断 {summary['direct_rule_pass_pairs']} 条）" if summary['direct_rule_pass_agreement'] is not None else f"- Direct 与 rule 的一致率：N/A（共同有物理判断 {summary['direct_rule_pass_pairs']} 条）",
             f"- Direct 与 rule 的分数 Spearman：{summary['direct_rule_score_spearman']:.4f}" if summary['direct_rule_score_spearman'] is not None else '- Direct 与 rule 的分数 Spearman：N/A',
             f"- V3 可可靠判断覆盖率：{summary['evaluator_reliable_coverage']:.2%}" if summary['evaluator_reliable_coverage'] is not None else '- V3 可可靠判断覆盖率：N/A',
             f"- Direct 可测率：{summary['direct_measurable_rate']:.2%}" if summary['direct_measurable_rate'] is not None else '- Direct 可测率：N/A',
             f"- Rule 可测率：{summary['rule_measurable_rate']:.2%}" if summary['rule_measurable_rate'] is not None else '- Rule 可测率：N/A', '',
             '## 评测器状态', '', '| 状态 | 数量 |', '|---|---:|']
    for status, count in sorted(summary['status_counts'].items()): lines.append(f'| `{status}` | {count} |')
    lines += ['', '## 按模型', '', '| Model | N | Direct 均分 (judged/N) | Rule 均分 (judged/N) | V3 物理均分 | Direct 可测率 | Rule 可测率 | V3 可判断率 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model, v in summary['by_model'].items():
        fmt = lambda score, judged: 'N/A' if score is None else f'{score:.3f} ({judged}/{v["videos"]})'
        ev = 'N/A' if v['evaluator_physics_score_mean'] is None else f'{v["evaluator_physics_score_mean"]:.3f}'
        lines.append(f"| {model} | {v['videos']} | {fmt(v['direct_score_mean'], v['direct_judged'])} | {fmt(v['rule_score_mean'], v['rule_judged'])} | {ev} | {v['direct_measurable_rate']:.2%} | {v['rule_measurable_rate']:.2%} | {v['evaluator_reliable_rate']:.2%} |")
    lines += ['', '## 排名对照', '']
    for name in ('direct','rule','evaluator'):
        lines.append(f'**{name}**：' + '；'.join(f'{i+1}. {m} ({s:.3f})' for i,(m,s) in enumerate(summary['rankings'][name])))
    lines += ['', '## 解读边界', '', 'VLM 的“可测/通过”是当前 Qwen3-VL 的代理判断，不等同于人工标注。可靠性结论应结合控制视频的真值、V3 的测量覆盖率、物理残差和失败证据；尤其不能因为 VLM 给出低分就把证据不足的视频强行归为物理失败。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=ROOT/'v3_evaluator/results.json')
    parser.add_argument('--status-results', type=Path, default=ROOT/'v3_evaluator/reliability_status/status_augmented_results.json')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--count', type=int, default=240)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--frames', type=int, default=6)
    parser.add_argument('--max-edge', type=int, default=448)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=160)
    parser.add_argument('--smoke', action='store_true', help='run only the first two sampled videos')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load(args.results)
    sidecar = _load(args.status_results) if args.status_results.exists() else []
    status_map = {(_key(row)): row for row in sidecar}
    sample = _sample(rows, 2 if args.smoke else args.count, args.seed, status_map)
    (args.output_dir/'sample_manifest.json').write_text(json.dumps({'seed': args.seed, 'requested_count': args.count, 'selected_count': len(sample), 'stratification': 'model × group × score_status', 'samples': sample}, ensure_ascii=False, indent=2) + '\n')
    jsonl = args.output_dir/'results.jsonl'
    completed = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            try:
                row = json.loads(line); completed[row['sample_key']] = row
            except (ValueError, KeyError): pass
    catalog = _catalog()
    with jsonl.open('a') as handle:
        for index, row in enumerate(sample, 1):
            key = _key(row)
            if key in completed: continue
            video = Path(row['video'])
            sample_id = row.get('sample_id') or video.stem
            frame_dir = args.output_dir/'frames'/row.get('model','unknown')/sample_id
            record = {'sample_key': key, 'index': index, 'model': row.get('model'), 'group': row.get('group'), 'task': row.get('task'), 'sample_id': sample_id, 'video': str(video), 'old_score_status': row.get('score_status'), 'evaluator': {**status_map.get(key, {}), 'physics_score': row.get('physics_score'), 'score': row.get('score'), 'consistency_score': row.get('consistency_score')}}
            try:
                frames = sample_video(video, frame_dir, args.frames, args.max_edge)
                record['frames'] = frames
                info = catalog.get(row.get('task'), {})
                prompt = ''
                p = row.get('prompt_source')
                if p and Path(p).exists(): prompt = Path(p).read_text(errors='replace')
                elif row.get('video_prompt'): prompt = row.get('video_prompt')
                phenomenon = f"{info.get('phenomenon','task '+str(row.get('task')))} ({info.get('domain','unknown')})"
                raw_direct = local_reply(direct_messages(row.get('task'), phenomenon, prompt, frames), frames, args.model, args.device, args.max_edge, args.max_new_tokens)
                record['direct'] = parse_reply(raw_direct)
                raw_rule = local_reply(rule_messages(row.get('task'), phenomenon, prompt, frames), frames, args.model, args.device, args.max_edge, args.max_new_tokens)
                record['rule'] = parse_reply(raw_rule)
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
            handle.write(json.dumps(record, ensure_ascii=False) + '\n'); handle.flush()
            print(json.dumps({'index': index, 'sample_key': key, 'error': record.get('error'), 'direct_score': record.get('direct',{}).get('score'), 'rule_score': record.get('rule',{}).get('score')}, ensure_ascii=False), flush=True)
            completed[key] = record
    records = list(completed.values())
    records.sort(key=lambda x: x.get('index', 0))
    summary = summarize(records)
    summary.update({
        'sample_seed': args.seed,
        'sample_stratification': 'model × group × score_status, deterministic round-robin',
        'vlm_model': str(args.model), 'vlm_device': args.device,
        'frames_per_video': args.frames, 'max_edge': args.max_edge,
        'max_new_tokens': args.max_new_tokens,
        'comparison_prompt_version': 'direct-and-protocol-v1',
        'human_labels_available': False,
    })
    (args.output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    (args.output_dir/'report.md').write_text(markdown(summary))
    print(json.dumps({'output_dir': str(args.output_dir), 'videos': len(records), 'completed_inference': summary['completed_inference'], 'report': str(args.output_dir/'report.md')}, ensure_ascii=False))


if __name__ == '__main__': main()
