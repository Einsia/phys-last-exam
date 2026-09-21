"""Run direct and physics-rule Qwen-VL judgments on the same video frames."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .protocol import get_protocol


DIRECT_SYSTEM = '''You are a careful video evaluator. Judge the requested physical experiment from the chronological video frames. Do not reward visual polish, prompt compliance alone, or temporal smoothness. First decide whether the requested event is visibly completed and whether the key quantities are measurable. Then give a moderate physical-correctness score. You may answer unknown when the frames do not support a reliable physical judgment. Return ONLY JSON with exactly these fields: {"task_completed": true/false/null, "measurable": true/false, "physics_pass": true/false/null, "score": 0.0, "confidence": 0.0, "reason": "brief evidence with frame times"}.'''

RULE_SYSTEM = '''You are a physics video auditor. Follow the supplied measurement protocol rather than relying on visual plausibility. Separate event completion, measurement validity, and the physical constraint. If an essential quantity cannot be measured, set measurable=false and physics_pass=null; do not call it a physics failure. Return ONLY JSON with exactly these fields: {"task_completed": true/false/null, "measurable": true/false, "physics_pass": true/false/null, "score": 0.0, "confidence": 0.0, "reason": "brief evidence with frame times and the relevant constraint"}.'''


def sample_video(video: Path, out_dir: Path, count: int = 6, max_edge: int = 448):
    import cv2
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
    if not cap.isOpened() or total < 2:
        cap.release()
        raise ValueError(f'Cannot decode video: {video}')
    indices = sorted({round(i * (total - 1) / max(1, count - 1)) for i in range(count)})
    items = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {index} from {video}')
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((max_edge, max_edge))
            path = out_dir / f'frame_{index:06d}.jpg'
            image.save(path, quality=88)
            items.append({'path': str(path), 'frame_index': index,
                          'timestamp_seconds': index / fps,
                          'label': f'video frame at {index / fps:.3f}s'})
    finally:
        cap.release()
    return items


def _messages(system, content):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': content}]


def direct_messages(task_id, phenomenon, prompt, frames):
    content = [{'type': 'text', 'text': (
        f'Task {task_id}: {phenomenon}.\n'
        'This is a direct visual judgment. Use the task description only to identify the event; do not invent a formula or assume invisible experimental conditions.\n'
        f'Generation description (possibly incomplete): {prompt[:1800]}\n\n'
        'Inspect the chronological frames below. Look for event completion, visible key quantities, and obvious violations of the intended physical behavior.')}]
    for item in frames:
        content += [{'type': 'text', 'text': item['label']}, {'type': 'image', 'image': item['path']}]
    content.append({'type': 'text', 'text': 'Return the exact JSON schema now. Keep reason under 25 words. A low-confidence or unmeasurable case must use measurable=false and physics_pass=null.'})
    return _messages(DIRECT_SYSTEM, content)


def rule_messages(task_id, phenomenon, prompt, frames):
    protocol = get_protocol(task_id)
    rule = json.dumps(protocol, ensure_ascii=False, indent=2)
    content = [{'type': 'text', 'text': (
        f'Task {task_id}: {phenomenon}.\n'
        'Use this frozen measurement protocol:\n' + rule + '\n\n'
        f'Generation description: {prompt[:1200]}\n\n'
        'Audit in this order: (1) are the applicable conditions visible and valid? '
        '(2) did the required event happen? (3) are every listed quantity measurable? '
        '(4) only then judge the stated constraint. Do not turn an extraction or visibility problem into physics_fail.')}]
    for item in frames:
        content += [{'type': 'text', 'text': item['label']}, {'type': 'image', 'image': item['path']}]
    content.append({'type': 'text', 'text': 'Return the exact JSON schema now. Keep reason under 25 words and include the measured relation or missing evidence.'})
    return _messages(RULE_SYSTEM, content)


def local_reply(messages, frames, model_path, device='cuda:0', max_edge=448, max_new_tokens=160):
    """Use the same local Qwen-VL loader as the production consistency gate."""
    import torch
    from PIL import Image
    from unified_evaluators.consistency import _image, _local_model

    processor, model = _local_model(str(model_path), device)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [_image(item['path'], max_edge) for item in frames]
    inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


def parse_reply(raw):
    """Parse JSON despite occasional fenced output or a short leading sentence."""
    raw = str(raw or '').strip()
    candidates = [raw]
    candidates.extend(re.findall(r'\{.*?\}', raw, flags=re.S))
    parsed = None
    for candidate in candidates:
        candidate = re.sub(r'^```(?:json)?\s*|\s*```$', '', candidate.strip(), flags=re.I)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                break
        except json.JSONDecodeError:
            continue
    if not isinstance(parsed, dict):
        # A generation can stop immediately after a long reason. Recover the
        # scalar fields so truncation is counted as a warning, not as a zero.
        def partial_bool(name):
            match = re.search(rf'"{name}"\s*:\s*(true|false|null)', raw, re.I)
            if not match:
                return None
            value = match.group(1).lower()
            return None if value == 'null' else value == 'true'
        def partial_number(name):
            match = re.search(rf'"{name}"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))', raw, re.I)
            if not match:
                return None
            try:
                value = float(match.group(1))
                return value if math.isfinite(value) and 0 <= value <= 1 else None
            except ValueError:
                return None
        reason_match = re.search(r'"reason"\s*:\s*"(.*)$', raw, re.S)
        return {'parse_error': True, 'raw': raw, 'task_completed': partial_bool('task_completed'),
                'measurable': bool(partial_bool('measurable')), 'physics_pass': partial_bool('physics_pass'),
                'score': partial_number('score'), 'confidence': partial_number('confidence') or 0.0,
                'reason': (reason_match.group(1)[:2000] if reason_match else 'VLM response was not valid JSON.')}
    def boolean_or_none(value):
        if isinstance(value, bool) or value is None: return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {'true', 'yes', 'pass', 'passed', 'complete'}: return True
            if lower in {'false', 'no', 'fail', 'failed', 'incomplete'}: return False
        return None
    def score(value):
        try:
            value = float(value)
            return value if math.isfinite(value) and 0 <= value <= 1 else None
        except (TypeError, ValueError): return None
    return {
        'parse_error': False,
        'task_completed': boolean_or_none(parsed.get('task_completed')),
        'measurable': bool(parsed.get('measurable')),
        'physics_pass': boolean_or_none(parsed.get('physics_pass')),
        'score': score(parsed.get('score')),
        'confidence': score(parsed.get('confidence')),
        'reason': str(parsed.get('reason', ''))[:2000],
        'raw': raw,
    }
