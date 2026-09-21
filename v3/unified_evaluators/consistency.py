"""VLM temporal-coherence gate, evaluated before physical extraction."""
from dataclasses import dataclass
from functools import lru_cache
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / '.models/Qwen3-VL-8B-Instruct'
RUBRIC_VERSION = 'temporal-coherence-moderate-v7'
RUBRIC = '''You judge ONLY the temporal visual coherence of a generated video.
This is a MODERATELY PERMISSIVE screening gate before a separate quantitative
physics evaluator. Inspect the chronological video frames as a sequence and ask
whether the visible imagery evolves coherently from one frame to the next.

The reference image, generation prompt, task identity, background, floor, scene
layout, camera angle, camera motion, lighting, color, texture, object styling,
and changes in what is visible are NOT criteria. Do not compare the video to the
reference image. A video may use a different scene, viewpoint, color scheme,
object arrangement, or background and still pass this gate. Do not judge whether
the requested action happened, whether the video is static, or whether any
physical law is obeyed; those belong to the later physics evaluator.

PASS normal motion, camera motion, zoom, blur, temporary occlusion, objects
entering or leaving the frame, a fully static video, and unusual but coherent
physical states. Do not penalize a scene cut, background/floor/layout change,
camera/viewpoint change, lighting or color change, or an object simply becoming
partly hidden. Do not judge whether the requested action happened or whether a
physical law was obeyed. If a defect is isolated to one frame or can be explained
by blur, transparency, occlusion, perspective, or an object moving out of the
visible field, favor passing. Blur or defocus alone is not a temporal identity
failure: if the object's major silhouette, identity, and connectivity remain
stable, score it in the 0.90-1.0 band even when several frames are blurred.

Assess every prominent persistent object, not only laboratory apparatus. Give a
meaningful deduction when the same visible object repeatedly changes its major
silhouette, category, part count, attachment, or identity across exposed frames;
when a connected part appears, disappears, duplicates, or fuses without
occlusion or a coherent interaction; when the object repeatedly rubber-stretches
or jitters between incompatible shapes; or when an object abruptly teleports
while the surrounding imagery is stable. These defects must persist across more
than one sampled transition or occupy a substantial part of the sequence. Small
deformations, natural motion of flexible objects, perspective changes, and brief
generation glitches are limited artifacts, not failures.

Pay special attention to rigid topology: a tube, rod, vessel, or other rigid
structure visibly growing, shrinking, fusing, splitting, or turning into an
unrelated structure (for example, a U-shaped tube becoming a rectangular tank)
is a clear failure even when individual frames are sharp. A scene cut or a
different background alone is never a failure.

Score ONLY temporal visual coherence on [0,1], and do not default to 1.0 merely
because there is no catastrophic topology change:
0.90-1.0: stable identity, silhouette, and connectivity; at most tiny transient
artifacts;
0.70-0.89: minor or localized repeated instability, while the sequence remains
clearly coherent;
0.50-0.69: noticeable sustained shape/identity/connectivity instability, but
the main objects remain interpretable;
0.00-0.49: clear major or prolonged distortion, topology change, duplication,
fusion/splitting, or frame corruption.
Scores below 0.50 must name an observable defect and its frame time. Scores in
the 0.50-0.89 range must also mention the repeated or sustained artifact that
caused the deduction, if one is visible.
The evaluator applies its configurable threshold; your job is to supply the score.
Return ONLY a JSON object with exactly these fields:
{"score": 0.0, "reason": "brief explanation including frame times", "issues": ["observed major issues, or an empty list"]}
Do not calculate a physical-law score. Do not output markdown or a discussion.
'''


@dataclass(frozen=True)
class Settings:
    backend: str = 'local'
    model: str = str(DEFAULT_MODEL)
    device: str = 'cuda:0'
    threshold: float = 0.8
    frames: int = 8
    max_edge: int = 640
    max_new_tokens: int = 384
    base_url: str = ''
    api_key_env: str = 'VLM_API_KEY'
    timeout: float = 120.0

    def validate(self):
        if self.backend not in ('local', 'http'):
            raise ValueError('consistency backend must be local or http')
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError('consistency threshold must be finite and in [0,1]')
        if not 2 <= self.frames <= 64 or not 112 <= self.max_edge <= 2048:
            raise ValueError('consistency frames must be 2..64 and max edge 112..2048')
        if not 32 <= self.max_new_tokens <= 4096 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('invalid consistency token limit or timeout')
        if self.backend == 'http' and not self.base_url.startswith(('http://', 'https://')):
            raise ValueError('HTTP consistency backend requires --consistency-base-url')


def add_arguments(parser):
    group = parser.add_argument_group('V3 consistency gate (always runs before physics)')
    group.add_argument('--consistency-backend', choices=['local', 'http'], default=os.getenv('VLM_BACKEND', 'local'))
    group.add_argument('--consistency-model', default=os.getenv('VLM_MODEL', str(DEFAULT_MODEL)))
    group.add_argument('--consistency-device', default=os.getenv('VLM_DEVICE', 'cuda:0'))
    group.add_argument('--consistency-threshold', type=float, default=0.8)
    group.add_argument('--consistency-frames', type=int, default=8)
    group.add_argument('--consistency-max-edge', type=int, default=640)
    group.add_argument('--consistency-max-new-tokens', type=int, default=384)
    group.add_argument('--consistency-base-url', default=os.getenv('VLM_BASE_URL', ''))
    group.add_argument('--consistency-api-key-env', default='VLM_API_KEY')
    group.add_argument('--consistency-timeout', type=float, default=120.0)


def settings_from_args(args):
    return Settings(**{name: getattr(args, 'consistency_' + name) for name in Settings.__dataclass_fields__})


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def _image(path, max_edge):
    from PIL import Image, ImageOps
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert('RGB')
        im.thumbnail((max_edge, max_edge))
        return im.copy()


def sample_frames(video, reference, directory, settings):
    import cv2
    from PIL import Image
    frame_dir = directory / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    items = []
    if reference:
        path = frame_dir / 'reference.jpg'
        _image(reference, settings.max_edge).save(path, quality=92)
        items.append({'path': str(path), 'label': 'Reference image of the intended setup', 'role': 'reference'})
    cap = cv2.VideoCapture(str(video))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not cap.isOpened() or count < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('Cannot decode video frame count/FPS for consistency evaluation')
        indices = sorted({round(i * (count - 1) / (settings.frames - 1)) for i in range(settings.frames)})
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode requested consistency frame {index}')
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((settings.max_edge, settings.max_edge))
            path = frame_dir / f'frame_{index:06d}.jpg'
            im.save(path, quality=92)
            items.append({'path': str(path), 'role': 'video', 'frame_index': index,
                          'timestamp_seconds': index / fps, 'label': f'Video frame at {index / fps:.3f} seconds'})
    finally:
        cap.release()
    return items


def messages_for(items, task_id, prompt):
    # This gate deliberately does not send the reference or generation prompt to
    # the VLM: comparing against either one turns harmless scene/style changes
    # into false consistency failures. Only chronological video frames matter.
    video_items = [item for item in items if item.get('role') == 'video']
    content = [{'type': 'text', 'text': 'Inspect this video timeline for temporal coherence at moderate strictness. Compare adjacent frames as well as the first, middle, and last frames. '
                'Track every prominent persistent object: its major silhouette, identity, part count, and connected structure. '
                'A rigid U-shaped tube that becomes a rectangular tank is a failure. Repeated incompatible shape or identity changes '
                'should lower the score even when there is no catastrophic topology change. Evaluate only temporal visual coherence '
                'in these chronological video frames. Evidence metadata: ' +
                json.dumps({'task_id': task_id,
                    'reference_comparison': 'disabled',
                    'generation_prompt': 'not supplied',
                    'physical_correctness': 'not judged'}, ensure_ascii=False)}]
    for item in video_items:
        content.extend([{'type': 'text', 'text': item['label']}, {'type': 'image', 'image': item['path']}])
    content.append({'type': 'text', 'text':
        'Now compare adjacent transitions and the first, middle, and last frames. Judge only whether the imagery itself '
        'has repeated or sustained unreasonable temporal instability. Explicitly inspect the silhouette, identity, part count, '
        'and connectivity of prominent objects and rigid tubes, rods, vessels, and other apparatus across time. A U-shaped tube '
        'turning into a rectangular tank is a failure, but one brief blur or an occluded part is not. Ignore scene, layout, '
        'background, color, camera, whether the requested motion succeeded, and physical laws. Do not default to 1.0 when a '
        'visible repeated artifact deserves a 0.70-0.89 or 0.50-0.69 score. Return the required JSON.'})
    return [{'role': 'system', 'content': RUBRIC}, {'role': 'user', 'content': content}]


@lru_cache(maxsize=1)
def _local_model(model_path, device):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for the consistency model; configure a GPU or HTTP VLM backend')
    # Runtime never silently downloads a different checkpoint.
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32 if device == 'cpu' else torch.bfloat16,
        device_map=device, attn_implementation='sdpa').eval()
    return processor, model


def _local_reply(messages, items, settings):
    import torch
    processor, model = _local_model(settings.model, settings.device)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [_image(item['path'], settings.max_edge) for item in items if item.get('role') == 'video']
    inputs = processor(text=[text], images=images, padding=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, do_sample=False)
    return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


def _http_reply(messages, items, settings):
    messages = json.loads(json.dumps(messages))
    for message in messages:
        if not isinstance(message['content'], list):
            continue
        for i, part in enumerate(message['content']):
            if part['type'] == 'image':
                data = base64.b64encode(Path(part['image']).read_bytes()).decode()
                message['content'][i] = {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}}
    payload = {'model': settings.model, 'messages': messages, 'temperature': 0,
               'max_tokens': settings.max_new_tokens}
    endpoint = settings.base_url.rstrip('/')
    if not endpoint.endswith('/chat/completions'):
        endpoint += '/chat/completions'
    headers = {'Content-Type': 'application/json'}
    api_key = os.getenv(settings.api_key_env)
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    request = urllib.request.Request(endpoint, json.dumps(payload).encode(), headers=headers, method='POST')
    with urllib.request.urlopen(request, timeout=settings.timeout) as response:
        result = json.load(response)
    return result['choices'][0]['message']['content']


def parse_judgment(text):
    if not isinstance(text, str):
        raise ValueError('VLM response content must be text')
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('VLM response must be a JSON object')
    score = value.get('score')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('VLM score must be a finite number in [0,1]')
    if not isinstance(value.get('reason'), str) or not value['reason'].strip():
        raise ValueError('VLM response must include a reason')
    if not isinstance(value.get('issues'), list) or not all(isinstance(v, str) for v in value['issues']):
        raise ValueError('VLM issues must be a list of strings')
    return {'score': float(score), 'reason': value['reason'], 'issues': value['issues']}


def evaluate(video, reference, prompt, task_id, debug, settings):
    """Return a fresh decision; transport/model errors are never passes or zero scores."""
    started = time.monotonic()
    directory = Path(debug) / 'consistency'
    directory.mkdir(parents=True, exist_ok=True)
    result = {'rubric_version': RUBRIC_VERSION, 'rubric_sha256': hashlib.sha256(RUBRIC.encode()).hexdigest(),
              'backend': settings.backend, 'model': settings.model, 'device': settings.device,
              'threshold': settings.threshold, 'score': None, 'passed': None, 'status': 'error',
              'artifact_directory': str(directory.resolve())}
    try:
        settings.validate()
        items = sample_frames(video, reference, directory, settings)
        result['frames'] = items
        messages = messages_for(items, task_id, prompt)
        # Store readable text and local frame paths, never credentials or data URLs.
        _write(directory / 'request.json', {'messages': messages, 'rubric_version': RUBRIC_VERSION,
            'generation_prompt_for_audit_only': prompt, 'generation_prompt_sent_to_vlm': False})
        reply = _local_reply if settings.backend == 'local' else _http_reply
        # The reference is retained in the audit artifacts but is intentionally
        # excluded from the VLM input for this temporal-only gate.
        raw = reply(messages, items, settings)
        _write(directory / 'response.json', {'content': raw})
        judgment = parse_judgment(raw)
        result.update(judgment, passed=judgment['score'] >= settings.threshold, status='evaluated')
    except Exception as exc:
        result.update(reason=f'{type(exc).__name__}: {exc}', issues=[])
    result['elapsed_seconds'] = time.monotonic() - started
    _write(directory / 'consistency.json', result)
    return result
