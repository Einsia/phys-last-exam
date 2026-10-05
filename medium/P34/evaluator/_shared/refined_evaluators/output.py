"""Publish reviewable evidence and task-relative result paths."""
from pathlib import Path
import json
from .scoring import score_result


def write_scoring_calculation(result, debug_dir):
    """Refresh scoring text without modifying measurements or their path base."""
    debug_dir = Path(debug_dir)
    verbose = result['verbose']['M1']
    debug_dir.mkdir(parents=True, exist_ok=True)
    score_path = debug_dir / 'scoring_calculation.md'
    terms = [f'{weight:g} × {result["metrics"][key]["proxy_score"]}'
             for key, weight in result['proxy']['weights'].items() if weight]
    score_path.write_text('\n\n'.join([
        f'# {result["task_id"]} V2 测量与评分',
        str(verbose.get('principle') or '详见 measurement_process 和提取失败原因。'),
        '原始量与测量过程：\n```json\n' + json.dumps(verbose.get('measurements', {}), ensure_ascii=False, indent=2, default=str) + '\n```',
        '归一化：\n```json\n' + json.dumps(verbose.get('normalization', {}), ensure_ascii=False, indent=2) + '\n```',
        '指标分：\n```json\n' + json.dumps(result['metrics'], ensure_ascii=False, indent=2) + '\n```',
        f'综合分 = {" + ".join(terms) or "0"} = {result["proxy"]["score"]}。',
        '总分只对已定义指标等权平均。M2 未设置时，extract_success 和 metric 保留 null，且不计入分母，总分等于 M1 最终分。已定义但不可测的指标仍占权重，贡献为 0。',
        f'测量状态：{verbose.get("status")}。原因：{verbose.get("reason")}',
    ]) + '\n', encoding='utf-8')
    return score_path


def finalize(result, task_dir, debug_dir):
    task_dir, debug_dir = Path(task_dir).resolve(), Path(debug_dir).resolve()
    score_result(result)
    verbose = result['verbose']['M1']
    artifacts = verbose.setdefault('artifacts', {})
    # Include measurement tables, figures and source-aligned videos, not only masks.
    for path in sorted(debug_dir.iterdir()) if debug_dir.is_dir() else []:
        if path.suffix in ('.png', '.jpg', '.mp4', '.csv', '.json', '.npz', '.md'):
            artifacts.setdefault(path.stem, str(path))
    artifacts['scoring_calculation'] = str(write_scoring_calculation(result, debug_dir))
    verbose.setdefault('measurement_process', [
        '解码原始视频，保存每帧 PTS；核验对应首帧和物体身份。',
        '从实际图像分割区域、边缘或轨迹；测量窗口和可靠性检查保存在原始量与证据文件中。',
        verbose.get('principle'),
        '分别保存原始物理量、纯物理分和每项一次的识别奖励；总分仅对已定义指标等权平均。',
    ])
    result['path_base'] = 'task_directory'
    # CLI input paths are relative to the caller's cwd, while output paths are
    # explicitly relative to the task directory. Resolve before changing bases.
    for key in ('video_path','image_path'):
        if result.get(key):result[key]=str(Path(result[key]).resolve())

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str):
            if value.startswith(str(task_dir) + '/'):
                return str(Path(value).relative_to(task_dir))
            if '\n' not in value and '/' in value and len(value)<4096:
                try:
                    path=Path(value)
                    if not path.is_absolute() and path.exists():
                        path=path.resolve()
                        return str(path.relative_to(task_dir)) if path.is_relative_to(task_dir) else str(path)
                except OSError:
                    pass
        return value

    return portable(result)
