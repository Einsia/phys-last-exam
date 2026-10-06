#!/usr/bin/env python3
"""P32 evaluator. Use --help without any ML dependencies installed."""
from dataclasses import asdict, fields
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback
from utils.common import EnvironmentError, ExtractionError, Thresholds, new_result, write_json

TASK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR / 'evaluator' / '_shared'))
from refined_evaluators.output import finalize
from refined_evaluators.resources import model_resource


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video_path', required=True)
    p.add_argument('--image_path')
    prompt = p.add_mutually_exclusive_group()
    prompt.add_argument('--video_prompt')
    prompt.add_argument('--video_prompt_file', help='Read the actual generation prompt from a UTF-8 file')
    p.add_argument('--model', default=None, help='Video generator name, not detector name')
    p.add_argument('--sample_id', default=None, help='Stable sample identifier, e.g. sample_00')
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--output', required=True)
    p.add_argument('--debug_dir', help='Default: sibling debug_<sample_id or video_stem>; intermediate videos are always saved')
    p.add_argument('--device', default='auto', help='auto selects the GPU with most free memory, cpu, or cuda:N')
    p.add_argument('--dino_model', default=str(model_resource('grounding-dino-tiny')))
    p.add_argument('--sam2_checkpoint', default=str(model_resource('sam2.1-hiera-small/sam2.1_hiera_small.pt')))
    p.add_argument('--sam2_config', default='configs/sam2.1/sam2.1_hiera_s.yaml')
    p.add_argument('--cache_dir', default=None)
    p.add_argument('--allow_download', action='store_true', help='Permit Transformers to download missing DINO files')
    p.add_argument('--reuse_masks', action='store_true', help='Reuse only a matching cached video/model segmentation')
    p.add_argument('--config', help='JSON object containing threshold overrides')
    p.add_argument('--annotation', help='Reviewed boxes and optional physical hub points on this actual video frame zero')
    p.add_argument('--threads', type=int, default=4)
    for f in fields(Thresholds):
        p.add_argument('--'+f.name, type=type(getattr(Thresholds(),f.name)), default=None)
    return p


def thresholds_from_args(args):
    values = asdict(Thresholds())
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding='utf-8'))
        if not isinstance(overrides, dict) or set(overrides)-set(values):
            raise ValueError('Threshold config must be an object with known threshold keys')
        values.update(overrides)
    values.update({k:getattr(args,k) for k in values if getattr(args,k) is not None})
    cfg = Thresholds(**values)
    cfg.validate()
    return cfg


def evaluate(args):
    started = time.monotonic()
    result = new_result(args.video_path,args.image_path,args.video_prompt,args.model,args.seed)
    result['sample_id'] = args.sample_id
    verbose = result['verbose']['M1']
    directory = Path(args.debug_dir) if args.debug_dir else Path(args.output).parent/('debug_'+(args.sample_id or Path(args.video_path).stem))
    artifacts = {}
    exit_code = 0
    try:
        cfg = thresholds_from_args(args)
        verbose['thresholds'] = asdict(cfg)
        if args.video_prompt_file:
            result['video_prompt'] = Path(args.video_prompt_file).read_text(encoding='utf-8').strip()
        if not Path(args.video_path).is_file():
            raise ExtractionError(f'Video does not exist: {args.video_path}')
        try:
            import torch
            import cv2
            import numpy as np
            from utils.media import decode, save_segments
            from utils.models import segment
            from utils.measurement import match_static, track_needles, summarize, measure_image_reference
            from utils.debug import save_measurements
        except Exception as e:
            raise EnvironmentError(f'Install evaluator/requirements.txt and official SAM 2: {e}') from e
        if args.threads < 1:
            raise ValueError('--threads must be positive')
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        if args.device == 'auto':
            args.device = ('cuda:'+str(max(range(torch.cuda.device_count()), key=lambda i:torch.cuda.mem_get_info(i)[0]))) if torch.cuda.is_available() else 'cpu'
        if args.device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise EnvironmentError('CUDA was requested but is unavailable')
            torch.cuda.set_device(torch.device(args.device))
        if not Path(args.sam2_checkpoint).is_file():
            raise EnvironmentError(f'SAM 2 checkpoint missing: {args.sam2_checkpoint}; see README.md')
        verbose['runtime'] = {'device': args.device, 'torch': torch.__version__, 'opencv': cv2.__version__,
                              'numpy': np.__version__, 'threads': args.threads,
                              'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'device_name':torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU'}
        print(f'Evaluating {args.video_path} on {args.device}', flush=True)
        frames, times, pts = decode(args.video_path)
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'timestamps.json', pts)
        verbose['video'] = {'frames':len(frames),'width':frames[0].shape[1],'height':frames[0].shape[0],
                            'first_pts_sec':float(times[0]),'last_pts_sec':float(times[-1])}
        match = None
        resized = None
        if args.image_path:
            reference = cv2.imread(str(args.image_path))
            if reference is None:
                match = {'reliable':False,'reason':'First-frame image missing or unreadable'}
            else:
                resized = cv2.resize(reference,frames[0].shape[1::-1])
                match = match_static(resized,frames[0])
                match['corresponds'] = bool(match['reliable'] and abs(match.get('scale',0)-1)<.03
                                            and abs(match.get('rotation_deg',180))<2
                                            and match.get('shift_px',1e6)<.02*frames[0].shape[1])
        verbose['first_frame_check'] = match
        masks, boxes, scores, signature = segment(frames,args,cfg,directory)
        verbose['extraction_run'] = {'video_decoded':True,'physical_measurements_recomputed':True,
            'sam2_cache_reused':getattr(args,'sam2_cache_used',False),
            'cache_validation':'SHA-256 of video and model files plus segmentation configuration'}
        artifacts.update(save_segments(frames,times,masks,boxes,directory))
        artifacts['binary_masks'] = str(directory/'masks.npz')
        artifacts['timestamps'] = str(directory/'timestamps.json')
        initialization=getattr(args,'initialization_annotation',None)
        verbose['evidence'].append({'stage':'segmentation','method':('Reviewed actual-frame object boxes + official SAM 2.1 Hiera Small' if initialization else 'Grounding DINO Tiny + official SAM 2.1 Hiera Small'),
                                    'ids':{'1':'left','2':'right'},'boxes_xyxy':boxes,'scores':scores,'signature':signature})
        print('Intermediate segmentation videos saved. Measuring red poles...',flush=True)
        pivots=[obj.get('pivot') for obj in initialization['objects']] if initialization else None
        rows, calibrations, errors = track_needles(frames,times,masks,cfg,initial_pivots=pivots)
        verbose['evidence'].append({'stage':'needle_measurement','calibration_errors':errors,
            'angle_convention':'CCW positive, physical pivot toward the red pole',
            'perspective_method':'Circular dial conic + independently observed physical centre',
            'perspective_correction_applied':[g is not None for g in calibrations]})
        from collections import Counter
        verbose['invalid_observations'] = {name:dict(Counter(r[name]['reason'] for r in rows if not r[name]['valid'])) for name in ('left','right')}
        image_reference = None
        verbose['evidence'].append({'stage':'final_orientation','policy':'Only final five frames contribute; no initial needle direction or stable-duration gate.'})
        try:
            metric = summarize(rows,times,cfg,verbose,image_reference)
            result['metrics']['M1'] = {'extract_success':True,'metric':metric}
        finally:
            artifacts.update(save_measurements(frames,times,rows,calibrations,verbose,directory))
    except EnvironmentError as e:
        verbose['status'],verbose['reason'],exit_code = 'environment_error',str(e),2
    except (ExtractionError, FileNotFoundError) as e:
        verbose['status'],verbose['reason'],exit_code = 'extraction_failed',str(e),1
    except ValueError as e:
        verbose['status'],verbose['reason'],exit_code = 'configuration_error',str(e),2
    except Exception as e:
        verbose['status'],verbose['reason'],exit_code = 'runtime_error',f'{type(e).__name__}: {e}',2
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        artifacts['traceback'] = str(directory/'traceback.txt')
    if exit_code:
        result['metrics']['M1'] = {'extract_success':False,'metric':None}
    verbose['artifacts'] = artifacts
    verbose['elapsed_sec'] = round(time.monotonic()-started,3)
    result = finalize(result,TASK_DIR,directory)
    if exit_code == 0 and result['metrics']['M1']['extract_success'] is not True:
        exit_code = 1
    write_json(args.output,result)
    print(f"{verbose['status']}: {verbose['reason']}\nResult: {args.output}",flush=True)
    return exit_code


def main(argv=None):
    return evaluate(parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
