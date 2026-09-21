"""Official Grounding DINO + SAM 2.1 video inference. No heuristic fallback."""
from contextlib import nullcontext
from pathlib import Path
import gc
import inspect
import tempfile
import numpy as np
import cv2
from .common import EnvironmentError, ExtractionError, write_json
from .media import fingerprint


def detect(frame, args, thresholds):
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        from PIL import Image
        options = {'local_files_only': not args.allow_download, 'cache_dir': args.cache_dir}
        processor = AutoProcessor.from_pretrained(args.dino_model, **options)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.dino_model, **options).to(args.device).eval()
    except Exception as e:
        raise EnvironmentError(f'Grounding DINO dependencies/weights/device: {e}') from e
    try:
        with torch.inference_mode():
            inputs = processor(images=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                               text='compass.', return_tensors='pt').to(args.device)
            outputs = model(**inputs)
            method = processor.post_process_grounded_object_detection
            threshold_arg = 'threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
            result = method(outputs, inputs.input_ids, **{threshold_arg: thresholds.detection_threshold},
                            text_threshold=thresholds.text_threshold, target_sizes=[frame.shape[:2]])[0]
            candidates = sorted(zip(result['boxes'].cpu().tolist(), result['scores'].cpu().tolist()),
                                key=lambda v: -v[1])
    finally:
        del model
        gc.collect()
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    # NMS removes duplicate boxes, not ambiguous extra objects.
    kept = []
    for box, score in candidates:
        x1, y1, x2, y2 = box
        bw, bh = x2-x1, y2-y1
        if bw < 24 or bh < 24 or not .5 < bw/bh < 2:
            continue
        duplicate = False
        for other, _ in kept:
            ox1, oy1, ox2, oy2 = other
            inter = max(0, min(x2, ox2)-max(x1, ox1)) * max(0, min(y2, oy2)-max(y1, oy1))
            union = bw*bh + (ox2-ox1)*(oy2-oy1) - inter
            duplicate |= inter / max(union, 1) > .5
        if not duplicate:
            kept.append((box, score))
    if len(kept) != 2:
        raise ExtractionError(f'Expected exactly two separate compass detections, found {len(kept)}')
    kept.sort(key=lambda v: (v[0][0]+v[0][2])/2)
    return np.array([v[0] for v in kept], dtype=np.float32), [v[1] for v in kept]


def segment(frames, args, thresholds, directory):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / 'masks.npz'
    dino_location = str(Path(args.dino_model).resolve()) if Path(args.dino_model).exists() else args.dino_model
    dino_files = {}
    if Path(args.dino_model).is_dir():
        for path in sorted(Path(args.dino_model).iterdir()):
            if path.is_file() and path.suffix in ('.safetensors','.json','.txt'):
                dino_files[path.name] = fingerprint(path)
    signature = {'video_sha256': fingerprint(args.video_path), 'dino_model': dino_location,
                 'dino_files_sha256':dino_files,
                 'sam2_config': args.sam2_config, 'sam2_checkpoint_sha256': fingerprint(args.sam2_checkpoint),
                 'detection_threshold': thresholds.detection_threshold, 'text_threshold': thresholds.text_threshold,
                 'implementation': 2}
    if Path(args.sam2_config).is_file():
        signature['sam2_config_sha256'] = fingerprint(args.sam2_config)
    import json
    signature_text = json.dumps(signature, sort_keys=True)
    if args.reuse_masks and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            saved_signature = json.loads(str(data['signature']))
            if Path(saved_signature['dino_model']).exists():
                saved_signature['dino_model'] = str(Path(saved_signature['dino_model']).resolve())
            # Relocating an identical model directory does not change inference.
            # All model-file hashes and every other cache key must still match.
            if saved_signature.get('dino_files_sha256') and saved_signature.get('dino_files_sha256') == signature['dino_files_sha256']:
                saved_signature['dino_model'] = signature['dino_model']
            if saved_signature != signature:
                raise ExtractionError('Mask cache signature differs; rerun without --reuse_masks')
            masks, boxes, scores = data['masks'], data['boxes'], data['scores'].tolist()
        if masks.shape != (len(frames), 2, *frames[0].shape[:2]):
            raise ExtractionError('Invalid mask cache shape')
        args.sam2_cache_used = True
        return masks, boxes, scores, signature
    print('Detecting left/right compasses with Grounding DINO...', flush=True)
    boxes, scores = detect(frames[0], args, thresholds)
    write_json(directory / 'detections.json', {'boxes_xyxy': boxes, 'scores': scores, 'ids': [1, 2]})
    print(f'Detected boxes: {boxes.tolist()}', flush=True)
    try:
        from sam2.build_sam import build_sam2_video_predictor
        config = args.sam2_config
        if Path(config).is_file():
            # Official builder uses Hydra package configs; support an absolute custom YAML as well.
            from hydra.core.global_hydra import GlobalHydra
            from hydra import initialize_config_dir
            GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=str(Path(config).resolve().parent), version_base='1.2')
            config = Path(config).name
        predictor = build_sam2_video_predictor(config, args.sam2_checkpoint, device=args.device,
                     hydra_overrides_extra=['++model.fill_hole_area=0'])
        # The official builder appends a hole-filling override of its own.
        # Disable only that optional CUDA extension; neural inference still uses the GPU.
        predictor.fill_hole_area = 0
    except Exception as e:
        raise EnvironmentError(f'SAM 2 dependencies/config/checkpoint: {e}') from e
    masks = np.zeros((len(frames), 2, *frames[0].shape[:2]), dtype=bool)
    # JPEGs are an official SAM 2 input, avoiding a second decoder with a different frame order.
    with tempfile.TemporaryDirectory(prefix='p34_sam2_') as staging:
        for i, frame in enumerate(frames):
            if not cv2.imwrite(str(Path(staging) / f'{i:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 98]):
                raise EnvironmentError('Failed to stage video frames for SAM 2')
        amp = torch.autocast('cuda', dtype=torch.bfloat16) if str(args.device).startswith('cuda') else nullcontext()
        with torch.inference_mode(), amp:
            state = predictor.init_state(staging, offload_video_to_cpu=True, offload_state_to_cpu=True)
            for j, box in enumerate(boxes):
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=j+1, box=box)
            seen = set()
            for i, ids, logits in predictor.propagate_in_video(state):
                if set(ids) != {1, 2}:
                    raise ExtractionError('SAM 2 lost object identities')
                for j, obj_id in enumerate(ids):
                    masks[i, obj_id-1] = (logits[j, 0] > 0).cpu().numpy()
                seen.add(i)
            if len(seen) != len(frames):
                raise ExtractionError('SAM 2 returned a truncated mask sequence')
    np.savez_compressed(cache, masks=masks, boxes=boxes, scores=scores, signature=signature_text)
    del predictor, state
    gc.collect()
    if str(args.device).startswith('cuda'):
        torch.cuda.empty_cache()
    return masks, boxes, scores, signature
