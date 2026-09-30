#!/usr/bin/env python3
"""Stable metadata association and failure-isolated batch execution."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile
from utils.common import new_result, write_json, path_component

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(TASK_DIR.parents[1]))
from refined_evaluators.output import finalize


def output_location(root, video, record, model_override=None):
    """A model/sample identity, never the enumeration index, determines output paths."""
    model = model_override if model_override is not None else record.get('model')
    model_folder = path_component(model if model is not None else 'unknown_model','model')
    sample_id = path_component(record.get('sample_id') or Path(video).stem,'sample_id')
    directory = Path(root)/model_folder
    return model, sample_id, directory/f'result_{sample_id}.json', directory/f'debug_{sample_id}'


def load_metadata(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        entries = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        entries = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(entries, dict) and 'samples' in entries:
        entries = entries['samples']
    if isinstance(entries, dict):
        entries = [dict(value, sample_id=key) for key,value in entries.items()]
    if not isinstance(entries,list) or any(not isinstance(item,dict) for item in entries):
        raise ValueError('Metadata must be a list, {samples: [...]}, or {sample_id: {...}}')
    return entries


def resolve_metadata_path(value, manifest_dir):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (manifest_dir/p).resolve()


def associate(video, entries, manifest_dir, task_dir):
    """Canonical task metadata paths are relative to the task root; external manifests remain relative to their own file."""
    video = Path(video).resolve()
    matches = []
    for entry in entries:
        has_path = bool(entry.get('video_path'))
        same_path = has_path and resolve_metadata_path(entry['video_path'],manifest_dir) == video
        same_id = str(entry.get('sample_id','')) == video.stem
        if same_id and has_path and not same_path:
            raise ValueError(f'Metadata sample_id/path conflict for {video.name}')
        if same_path or (same_id and not has_path):
            matches.append(entry)
    if len(matches) > 1:
        raise ValueError(f'Multiple metadata records match {video.name}')
    result = dict(matches[0]) if matches else {}
    for key in ('image_path','video_prompt_file'):
        if result.get(key):
            resolved = resolve_metadata_path(result[key],manifest_dir)
            if not resolved.is_file():
                raise ValueError(f'Metadata {key} does not exist: {resolved}')
            result[key] = str(resolved)
    # Canonical task videos live below output_videos; use the standard assets
    # only when an external caller omits the canonical metadata row.
    try:
        canonical_video = video.is_relative_to((task_dir/'output_videos').resolve())
    except AttributeError:
        canonical_video = str(video).startswith(str((task_dir/'output_videos').resolve()))
    if not matches and canonical_video:
        for key,file in (('image_path','first_frame.png'),('video_prompt_file','prompts/video.txt')):
            if (task_dir/file).is_file():
                result[key] = str((task_dir/file).resolve())
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input_dir',type=Path)
    p.add_argument('--output_dir',type=Path)
    p.add_argument('--metadata',type=Path)
    p.add_argument('--model',help='Video generator name; overrides metadata model when explicitly supplied')
    args,forward = p.parse_known_args(argv)
    standard = TASK_DIR/'output_videos'/path_component(args.model,'model') if args.model else None
    use_standard = args.input_dir is None and standard is not None and standard.is_dir()
    input_dir = (args.input_dir or (standard if use_standard else TASK_DIR)).resolve()
    output_dir = (args.output_dir or (TASK_DIR if use_standard else input_dir)/'eval_results').resolve()
    # Disallow forwarded options that could make every sample overwrite the same file.
    forbidden = {'--video_path','--image_path','--output','--debug_dir','--video_prompt','--video_prompt_file','--sample_id','--seed'}
    if any(arg.split('=')[0] in forbidden for arg in forward):
        p.error('Sample-specific paths/metadata must come from the manifest, not batch overrides')
    metadata = args.metadata
    if metadata is None and use_standard and (TASK_DIR/'data'/'metadata.json').is_file():
        metadata = TASK_DIR/'data'/'metadata.json'
    if metadata is None:
        found = [input_dir/name for name in ('metadata.json','metadata.jsonl','manifest.json') if (input_dir/name).is_file()]
        canonical = TASK_DIR/'data'/'metadata.json'
        if canonical.is_file() and canonical not in found:
            found.insert(0, canonical)
        if len(found) > 1:
            p.error('Multiple metadata files; select one with --metadata')
        metadata = found[0] if found else None
    entries = load_metadata(metadata) if metadata else []
    canonical = (TASK_DIR/'data'/'metadata.json').resolve()
    manifest_dir = TASK_DIR if metadata and metadata.resolve() == canonical else (metadata.resolve().parent if metadata else input_dir)
    videos = sorted(v for v in input_dir.iterdir() if v.is_file() and v.suffix.lower()=='.mp4')
    if not videos:
        p.error(f'No MP4 videos in {input_dir}')
    if args.model is not None:
        path_component(args.model,'model')
    output_dir.mkdir(parents=True,exist_ok=True)
    planned = []
    for video in videos:
        try:
            record = associate(video,entries,manifest_dir,TASK_DIR)
            location = output_location(output_dir,video,record,args.model)
            error = None
        except Exception as e:
            record = {}
            location = output_location(output_dir,video,record,args.model)
            error = e
        planned.append((video,record,location,error))
    targets = [str(item[2][2]) for item in planned]
    if len(set(targets)) != len(targets):
        p.error('Duplicate model/sample output targets; give each video a unique sample_id in metadata')
    samples = []
    for video,record,(model,sample_id,output,debug_dir),error in planned:
        try:
            if error is not None:
                raise error
            output.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='p34_result_',dir=output.parent) as staging:
                staged_output = Path(staging)/output.name
                command = [sys.executable,str(Path(__file__).with_name('evaluate.py')),
                           '--video_path',str(video),'--output',str(staged_output),
                           '--debug_dir',str(debug_dir),'--sample_id',sample_id,*forward]
                if model is not None:
                    command.extend(['--model',str(model)])
                for key in ('image_path','video_prompt','video_prompt_file','seed'):
                    if record.get(key) is not None:
                        if key=='video_prompt_file' and record.get('video_prompt') is not None:
                            continue
                        command.extend(['--'+key,str(record[key])])
                completed = subprocess.run(command,check=False)
                code = completed.returncode
                # Never mistake a result from an earlier run for the current sample.
                if not staged_output.exists() or code not in (0,1,2):
                    raise RuntimeError(f'Evaluator process exited with code {code}')
                data = json.loads(staged_output.read_text(encoding='utf-8'))
            write_json(output,data)
        except Exception as e:
            code = 2
            data = new_result(video,model=model)
            data['sample_id'] = sample_id
            data['verbose']['M1'].update(status='batch_error',reason=f'{type(e).__name__}: {e}')
            data = finalize(data,TASK_DIR,debug_dir)
            write_json(output,data)
        samples.append({'model':model,'sample_id':sample_id,'video_path':str(video),'result_path':str(output),'exit_code':code,
                        'status':data['verbose']['M1']['status'],'M1':data['metrics']['M1'],
                        'M2':data['metrics']['M2'],'proxy':data['proxy']})
        print(f'Saved result: {output}',flush=True)
    for directory in sorted({Path(s['result_path']).parent for s in samples}):
        grouped = [s for s in samples if Path(s['result_path']).parent==directory]
        write_json(directory/'batch_summary.json',{'task_id':'P34','model':grouped[0]['model'],'samples':grouped})
    return int(any(s['exit_code'] != 0 for s in samples))


if __name__ == '__main__':
    sys.exit(main())
