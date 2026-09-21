#!/usr/bin/env python3
"""Summarize durable V3 all_test results, including missing and blocked inputs."""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import re
import statistics

from all_test_common import ROOT, SOURCE, V3, now, write_json, digest

NAMES = {
    'P1':'自由落体', 'P2':'斜抛运动', 'P3':'互余角抛射', 'P4':'弹跳高度衰减',
    'P5':'等质量钢球正碰', 'P6':'实心球纯滚动', 'P7':'实心球与圆环滚动对比',
    'P8a':'小角度单摆等时性', 'P8b':'单摆周期与质量无关', 'P8c':'大角度单摆周期',
    'P9':'单摆周期与摆长关系', 'P10':'粗糙斜面往返滑动', 'P11':'光的折射',
    'P12':'折射与全反射临界角', 'P13':'光的反射', 'P14':'点光源投影共点',
    'P16':'刚体共线点交比', 'P18':'静止液面与重力方向垂直', 'P19':'连通器液面平衡',
    'P20':'冰柱漂浮吃水比', 'P21':'浮冰融化液面变化', 'P21b':'含石浮冰融化',
    'P21c':'淡水冰在盐水中融化', 'P23':'水冻结体积膨胀', 'P27':'碎冰与整冰融化对比',
    'P28':'带电小球对称平衡', 'P34':'通电导线与指南针', 'P36':'磁性物块涡流制动',
    'P37':'闭合环与开口环电磁跳环', 'P38':'实心与开槽导体板电磁阻尼',
    'P39':'磁铁穿线圈与感应发光', 'P40':'沙堆休止角尺度一致性',
    'P41':'沙子与水漏斗排出', 'P42':'摩擦起滑与质量无关', 'P43':'刚体倾倒与支撑边',
    'P44':'链条悬垂静态形状', 'P45':'毛细上升与管径', 'P47':'皂泡隔膜曲率',
    'P48':'液滴合并体积守恒', 'P49':'黏性流体中小球终端速度',
}
VALID = {'complete', 'partial', 'unavailable', 'consistency_rejected'}


def catalog():
    source = V3.parent.parent / 'bench.md'
    result = {}
    for line in source.read_text().splitlines():
        if not re.match(r'^\|\s*P', line):
            continue
        cells = [re.sub(r'<br\s*/?>', ' ', c).strip() for c in re.split(r'(?<!\\)\|', line.strip().strip('|'))]
        if cells[0] not in NAMES:
            continue
        result[cells[0]] = {'difficulty': re.search(r'\b(Easy|Medium|Hard)\b', cells[4]).group(),
                            'phenomenon': NAMES[cells[0]], 'domain': cells[3]}
    if set(result) != set(NAMES):
        raise ValueError('Benchmark catalog is incomplete')
    return result, source


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows]])


def mean(rows, key='score'):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(vals) if vals else None


def fmt(value):
    return f'{value * 100:.2f}' if value is not None else '—'


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    definitions, definition_file = catalog()
    manifest_file = ROOT / 'run_manifest.json'
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {'jobs': []}
    signatures = {(j['model'], j['sample_id']): j['signature'] for j in manifest['jobs']}
    blocked_file = ROOT / 'blocked_inputs.json'
    blocked = {r['video']: r['reason'] for r in json.loads(blocked_file.read_text())['videos']} if blocked_file.exists() else {}
    models = sorted(p.name for p in (SOURCE / 'data/videos/all_test').iterdir() if p.is_dir())
    rows = []
    for model in models:
        for video in sorted((SOURCE / 'data/videos/all_test' / model / 'gpt').glob('*.mp4')):
            task = video.stem.split('_')[1]
            execution = ROOT / ('v3_' + model) / 'execution' / (video.stem + '.json')
            row = json.loads(execution.read_text()) if execution.exists() else {}
            row.update(model=model, sample_id=video.stem, task=task, seed=int(video.stem.rsplit('seed', 1)[1]),
                       video=str(video), **definitions[task])
            if not row.get('finished_at_utc'):
                import os
                row['score_status'] = ('running' if row.get('started_at_utc') else 'pending') if os.access(video, os.R_OK) else 'permission_denied'
                if str(video) in blocked:
                    row['score_status'] = blocked[str(video)]
            if row.get('signature') and signatures.get((model, video.stem)) not in (None, row.get('signature')):
                row['score_status'] = 'superseded_pending_rerun'
            row['valid_score'] = row.get('score_status') in VALID and row.get('score') is not None
            rows.append(row)
    stats = []
    for model in models:
        rr = [r for r in rows if r['model'] == model]
        valid = [r for r in rr if r['valid_score']]
        counts = Counter(r['score_status'] for r in rr)
        stat = {'model': model, 'source_videos': len(rr), 'valid_results': len(valid),
                'total_score': mean(valid), 'consistency_score': mean(valid, 'consistency_score'),
                'consistency_passed': sum(r.get('consistency_passed') is True for r in valid),
                'consistency_rejected': counts['consistency_rejected'], 'statuses': dict(counts)}
        for difficulty in ['Easy', 'Medium', 'Hard']:
            subset = [r for r in valid if r['difficulty'] == difficulty]
            stat[difficulty] = mean(subset)
            stat[difficulty + '_n'] = len(subset)
        stats.append(stat)
    stats.sort(key=lambda s: -(s['total_score'] if s['total_score'] is not None else -1))
    models = [s['model'] for s in stats]
    valid_rows = [r for r in rows if r['valid_score']]
    counts = Counter(r['score_status'] for r in rows)
    complete = len(valid_rows) == len(rows)
    threshold = manifest.get('threshold')
    if threshold is None:
        raise ValueError('Run manifest must record the consistency threshold')
    rejected = [r for r in valid_rows if r['score_status'] == 'consistency_rejected']
    consistency_bands = {
        label: sum(low <= r['consistency_score'] < high for r in valid_rows)
        for label, low, high in [('0.90–1.00', .9, 1.01), ('0.70–0.89', .7, .9),
                                 ('0.50–0.69', .5, .7), ('0.00–0.49', 0, .5)]
    }
    summary = {'updated_at_utc': now(), 'source_videos': len(rows), 'valid_results': len(valid_rows),
               'complete': complete, 'statuses': dict(counts), 'models': stats,
               'benchmark_source': str(definition_file), 'benchmark_sha256': digest(definition_file),
               'score_scale': [0, 1], 'threshold': threshold,
               'rubric_version': manifest.get('rubric_version'), 'consistency_score_bands': consistency_bands,
               'weights': {'consistency': .15, 'physics': .85}}
    write_json(ROOT / 'summary.json', summary)
    write_json(ROOT / 'task_catalog.json', definitions)
    save_csv(ROOT / 'per_video.csv', rows, ['model', 'task', 'difficulty', 'phenomenon', 'seed', 'score_status',
        'valid_score', 'score', 'consistency_score', 'consistency_passed', 'physics_score', 'physics_attempted',
        'extraction_successes', 'reason', 'video', 'output', 'result_sha256'])
    save_csv(ROOT / 'model_summary.csv', stats, ['model', 'source_videos', 'valid_results', 'total_score',
        'Easy', 'Easy_n', 'Medium', 'Medium_n', 'Hard', 'Hard_n', 'consistency_score', 'consistency_passed', 'consistency_rejected'])
    lines = ['# V3 视频物理评测报告', '', f'更新时间：{summary["updated_at_utc"]}。状态：' + ('全部完成。' if complete else '评测尚未全部完成；以下为当前已完成样本的阶段性统计。'), '',
        f'共 {len(models)} 个模型、{len(rows)} 个现有视频，当前有效结果 {len(valid_rows)} 个。Cosmos 缺少 g1_P5_seed45 和 g2_P13_seed42，未补零。', '',
        f'一致性模型：本地 Qwen3-VL-8B-Instruct；通过阈值 {threshold:.2f}，规则版本 `{summary["rubric_version"]}`。每个视频均匀采样 8 帧，仅评估视频内部时序连贯性；参考图和生成提示词只存档，不发送给 VLM。', '',
        '本轮对持续的主体形变、身份变化、部件复制/融合/分裂及连接结构变化扣分；场景、背景、布局、颜色、镜头变化和物理规律错误本身不构成一致性失败。纯运动模糊但主体结构稳定时通过。', '',
        '通过一致性检查后，总分 = 0.15 × 一致性分数 + 0.85 × 物理分数；不通过时跳过物理评测，总分仅保留 0.15 × 一致性分数。物理分数是已定义 M1/M2 的等权平均；提取失败的已定义指标计零，不重新分配权重。', '',
        '表中分数为百分制；CSV/JSON 保留 0–1 原始分数。总分按现有有效视频等权平均。权限问题、VLM 错误和程序执行错误不算作零分，也不纳入均分。难度来自 bench.md：Easy 10 项、Medium 17 项、Hard 13 项。', '',
        '## 1. 不同模型、难度及总评分', '',
        table(['Model', '有效/现有视频', 'Easy', 'Medium', 'Hard', '总分', '一致性均分', '一致性通过'],
              [[s['model'], f"{s['valid_results']}/{s['source_videos']}", *[fmt(s[k]) for k in ['Easy', 'Medium', 'Hard', 'total_score', 'consistency_score']],
                f"{s['consistency_passed']}/{s['valid_results']}"] for s in stats]), '',
        '## 2. 不同物理现象的评分', '', '每个单元格为该物理现象已有有效视频的 V3 总分均值，包含一致性贡献；“—”表示暂无有效结果。', '']
    phenomena = []
    for task in sorted(definitions, key=lambda t: (int(re.search(r'\d+', t).group()), t)):
        rr = [r for r in valid_rows if r['task'] == task]
        phenomena.append([task, definitions[task]['phenomenon'], definitions[task]['difficulty'],
                          *[fmt(mean([r for r in rr if r['model'] == model])) for model in models]])
    lines += [table(['任务', '物理现象', '难度', *models], phenomena), '', '## 3. 覆盖与运行状态', '',
              table(['状态', '数量'], sorted(counts.items())), '',
              table(['Model', '物理完整', '物理部分可用', '物理不可测', '一致性拒绝', '尚无有效结果'],
                    [[s['model'], *[s['statuses'].get(k, 0) for k in ['complete', 'partial', 'unavailable', 'consistency_rejected']],
                      s['source_videos'] - s['valid_results']] for s in stats]), '',
              '物理提取不完整或不可用与一致性拒绝分开记录。一致性通过仅说明场景和主体足够连贯，不代表物理正确。', '',
              '逐视频明细：[per_video.csv](per_video.csv)；模型汇总：[model_summary.csv](model_summary.csv)；结构化汇总：[summary.json](summary.json)。', '',
              '各模型的 `v3_<model>/` 内包含结果 JSON、逐视频运行记录、VLM 原始回答、采样帧、物理测量数据与调试日志。', '']
    lines += ['## 4. 一致性分布与未通过样本', '',
              f'通过 {sum(r["consistency_passed"] is True for r in valid_rows)}/{len(valid_rows)}，未通过 {len(rejected)}。以下分段为一致性原始分 C；通过条件统一为 C ≥ {threshold:.2f}。', '',
              table(['一致性分段', '视频数'], list(consistency_bands.items())), '',
              '以下为 VLM 的原始判定理由，供人工复核。每个视频只抽取 8 帧，因此这一门控不是逐帧检测。视频链接使用绝对路径；结果 JSON 内可查看采样帧和完整回答。', '']
    if rejected:
        lines += [table(['Model', '视频（绝对路径）', '一致性分 C', '总分', '判定理由', '结果'],
                        [[r['model'], f'[{r["sample_id"]}]({r["video"]})', f'{r["consistency_score"]:.2f}',
                          fmt(r['score']), ' '.join(str(r.get('reason') or '').split()),
                          f'[JSON]({r["output"]})'] for r in sorted(rejected, key=lambda r: (r['model'], r['sample_id']))]), '']
    else:
        lines += ['当前有效结果中没有一致性未通过样本。', '']
    # Keep the alternate HTML table inside this Markdown file, as requested.
    lines += ['<details>', '<summary>同一总表的 HTML 版本（嵌入本 Markdown）</summary>', '',
              '<table><thead><tr><th>Model</th><th>有效/现有</th><th>Easy</th><th>Medium</th><th>Hard</th><th>总分</th></tr></thead><tbody>']
    for s in stats:
        cells = [f'<td>{html.escape(s["model"])}</td>', f'<td>{s["valid_results"]}/{s["source_videos"]}</td>']
        for key in ['Easy', 'Medium', 'Hard', 'total_score']:
            value = s[key]
            shade = f'rgba(30,120,210,{0.05 + 0.5 * value:.3f})' if value is not None else 'transparent'
            cells.append(f'<td style="background:{shade};text-align:right">{fmt(value)}</td>')
        lines.append('<tr>' + ''.join(cells) + '</tr>')
    lines += ['</tbody></table>', '', '</details>', '']
    (ROOT / '评测报告.md').write_text('\n'.join(lines))
    print(json.dumps({k: summary[k] for k in ['source_videos', 'valid_results', 'complete', 'statuses']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
