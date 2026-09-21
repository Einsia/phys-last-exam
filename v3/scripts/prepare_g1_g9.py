#!/usr/bin/env python3
"""Freeze input associations and archive pre-rerun code/results, without changing videos."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'work/g1_g9_v2_20260911'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def resolve(task, value):
    if not value:
        return None
    path = Path(value)
    candidates = []
    marker = task.name + '/'
    if marker in str(value):
        candidates.append(task/str(value).split(marker, 1)[1])
    candidates.extend([task/path, task.parent/path, ROOT/path, path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'{task}: {value}')


def main():
    if (RUN/'input_manifest.json').exists():
        raise SystemExit('Prepared already; refusing to replace the frozen input manifest.')
    RUN.mkdir(parents=True, exist_ok=True)
    archived = []
    rows = []
    for group in sorted(ROOT.glob('g[1-9]')):
        for task in sorted(group.glob('P*')):
            if not task.is_dir():
                continue
            old_results = sorted(task.glob('eval_results/minimax_h3/result*.json'))
            if not old_results:
                continue
            source_files = set(old_results)
            source_files.update(task.glob('*.md'))
            source_files.update(task.glob('*.csv'))
            source_files.update(task.glob('metadata*.json'))
            source_files.update(task.glob('scripts/*'))
            source_files.update(task.glob('evaluator/**/*.py'))
            source_files.update(task.glob('evaluator/*.json'))
            source_files.update(task.glob('evaluator/*.yaml'))
            for src in sorted(source_files):
                if not src.is_file() or '__pycache__' in src.parts:
                    continue
                relative = src.relative_to(ROOT)
                dst = RUN/'before'/relative
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                archived.append({'path': str(relative), 'sha256': digest(dst)})
            samples = []
            for index, result_path in enumerate(old_results):
                old = json.loads(result_path.read_text())
                if group.name == 'g7':
                    manifest = json.loads((group/'manifest.json').read_text())['samples']
                    source = next(x for x in manifest if x['task_id'] == task.name and x['sample_id'] == result_path.stem.removeprefix('result_'))
                    # Five historical sample_00 JSONs refer to a separate continuation.
                    # The generation manifest and its SHA identify the formal samples.
                    old['video_path'] = source['video']
                    old['image_path'] = source['first_frame']
                    old['seed'] = source['seed']
                    old['video_prompt'] = (task/'prompt.txt').read_text().strip()
                    assert digest(resolve(task, source['video'])) == source['sha256']
                video = resolve(task, old['video_path'])
                image = resolve(task, old.get('image_path'))
                sample = f'sample_{index:02d}' if group.name == 'g3' else result_path.stem.removeprefix('result_')
                canonical = task/'output_videos/minimax_h3'/f'{sample}.mp4'
                canonical.parent.mkdir(parents=True, exist_ok=True)
                if not canonical.exists():
                    os.link(video, canonical)
                assert digest(video) == digest(canonical), (video, canonical)
                prompt = old.get('video_prompt')
                prompt_file = None
                if not prompt:
                    route_sim = image and 'simulation' in image.parts
                    prompt_file = task/('prompt_sim.txt' if route_sim and (task/'prompt_sim.txt').exists() else 'prompt.txt')
                    prompt = prompt_file.read_text().strip()
                record = {
                    'sample_id': sample,
                    'video_path': str(canonical.relative_to(task)),
                    'source_video_path': str(video.relative_to(task)),
                    'image_path': str(image.relative_to(task)) if image else None,
                    'video_prompt': prompt,
                    'model': old.get('model') or 'minimax_h3',
                    'model_folder': 'minimax_h3',
                    'seed': old.get('seed'),
                    'source_result': str(result_path.relative_to(task)),
                    'previous_result': str((RUN/'before'/result_path.relative_to(ROOT)).relative_to(ROOT)),
                    'video_sha256': digest(video),
                    'image_sha256': digest(image) if image else None,
                }
                if old.get('route'):
                    record['route'] = old['route']
                if old.get('source_metadata'):
                    record['source_metadata'] = old['source_metadata']
                # Preserve the explicit first-frame annotations used by G8/G9.
                if (task/'metadata_v2.json').exists() and group.name in ('g8','g9'):
                    previous = json.loads((task/'metadata_v2.json').read_text())['samples']
                    match = next(x for x in previous if x['sample_id'] == sample)
                    if match.get('annotation'):
                        record['annotation'] = match['annotation']
                samples.append(record)
            if group.name == 'g7' and (task/'continuation.mp4').exists():
                video = task/'continuation.mp4'
                if digest(video) not in {row['video_sha256'] for row in samples}:
                    image = task/'first_frame.png'
                    # No generation model or seed was recorded for these extra videos.
                    sample = 'sample_00'
                    canonical = task/'output_videos/unknown_model'/f'{sample}.mp4'
                    canonical.parent.mkdir(parents=True, exist_ok=True)
                    if not canonical.exists():
                        os.link(video, canonical)
                    samples.append({
                        'sample_id': sample, 'video_path': str(canonical.relative_to(task)),
                        'source_video_path': 'continuation.mp4',
                        'image_path': 'first_frame.png',
                        'video_prompt': (task/'video.txt').read_text().strip(),
                        'model': 'minimax_h3', 'model_folder': 'unknown_model', 'seed': None,
                        'video_sha256': digest(video), 'image_sha256': digest(image),
                        'metadata_note': 'Supplied continuation; model from historical continuation result, seed not recorded. Separate folder avoids collision with the formal sample_00.',
                    })
            (task/'metadata_v2.json').write_text(json.dumps({'samples': samples}, ensure_ascii=False, indent=2)+'\n')
            for record in samples:
                rows.append({'group': group.name, 'task_id': task.name, **record})
        for src in group.glob('evaluator/**/*.py'):
            relative = src.relative_to(ROOT)
            dst = RUN/'before'/relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            archived.append({'path': str(relative), 'sha256': digest(src)})
    manifest = {'version': 'g1-g9-inputs-20260911', 'samples': rows, 'archived_files': archived}
    (RUN/'input_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'samples': len(rows), 'tasks': len({(x['group'],x['task_id']) for x in rows}), 'archived_files': len(archived)}))


if __name__ == '__main__':
    main()
