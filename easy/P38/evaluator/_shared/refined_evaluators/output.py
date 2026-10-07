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
        f'# {result["task_id"]} V2 measurement and scoring',
        str(verbose.get('principle') or "See measurement_process and the extraction failure reason."),
        "Raw quantities and measurement process:\n```json\n" + json.dumps(verbose.get('measurements', {}), ensure_ascii=False, indent=2, default=str) + '\n```',
        "Normalization:\n```json\n" + json.dumps(verbose.get('normalization', {}), ensure_ascii=False, indent=2) + '\n```',
        "Metric scores:\n```json\n" + json.dumps(result['metrics'], ensure_ascii=False, indent=2) + '\n```',
        f'Total score = {" + ".join(terms) or "0"} = {result["proxy"]["score"]}.',
        "The total is the equal-weight mean of defined metrics only. When M2 is not defined, extract_success and metric remain null and M2 is excluded from the denominator; the total equals the final M1 score. Defined but unmeasurable metrics retain their weight and contribute zero.",
        f'Measurement status: {verbose.get("status")}. Reason: {verbose.get("reason")}',
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
        "Decode the original video, retain every frame PTS, and verify the corresponding first frame and object identities.",
        "Segment regions, edges, or trajectories from observed images; retain measurement windows and reliability checks in the raw quantities and evidence files.",
        verbose.get('principle'),
        "Store raw physical quantities, pure physics scores, and one recognition reward per metric separately; average only defined metrics for the total.",
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
