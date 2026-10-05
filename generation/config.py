"""Portable configuration for the eight released generation backends."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import string
from urllib.parse import urlsplit

MODELS = ('seedance-2.5', 'minimax-h3', 'cosmos3-super-image2video', 'vbvr-wan2.2',
          'wan2.2-i2v-a14b', 'lingbot-video-moe-30b-a3b',
          'hunyuan-video-1.5-i2v', 'cogvideox1.5-5b-i2v')
ANNOTATED = {'P37', 'P38', 'P39', 'P41', 'P43', 'P49'}
SCENES = {'P3': 'P3_gpt_01_modern', 'P6': 'P6_gpt_01_modern',
          'P9': 'P9_gpt_01_modern', 'P11': 'P11_gpt_01_30deg'}


def default_config(root):
    root = Path(root).expanduser().resolve()
    source = root / 'source'

    def python_for(project, *extra):
        candidates = [*(Path(p) for p in extra), project / '.venv/bin/python']
        return str(next((p for p in candidates if p.is_file()), candidates[-1]))

    diffusers_python = python_for(source / 'Wan2.1')
    definitions = {
        'seedance-2.5': dict(num_frames=120, options=dict(
            base_url=os.environ.get('SEEDANCE_BASE_URL', ''),
            model=os.environ.get('SEEDANCE_MODEL', 'doubao-seedance-2-5-260628'),
            api_key_env='SEEDANCE_API_KEY', seconds=5, request_format='aihubmix_json')),
        'minimax-h3': dict(num_frames=124, options=dict(
            proj=str(source / 'minimax-h3'), model_dir=str(root / 'MiniMax-H3'),
            python_bin=python_for(source / 'minimax-h3', source / 'minimax-h3/envs/mh3/bin/python'),
            devices='0,1,2,3', height=480, width=832, flash3=False)),
        'cosmos3-super-image2video': dict(num_frames=81, options=dict(
            proj=str(root), model_dir=str(root / 'Cosmos3-Super-Image2Video'),
            python_bin=diffusers_python, devices='0,1,2,3', height=480, width=832, fps=16)),
        'vbvr-wan2.2': dict(num_frames=81, options=dict(
            proj=str(root), model_dir=str(root / 'VBVR-Wan2.2'), python_bin=diffusers_python,
            devices='0', height=480, width=832, fps=16, guidance_scale=5.0)),
        'wan2.2-i2v-a14b': dict(num_frames=81, options=dict(
            proj=str(source / 'Wan2.2'), ckpt_dir=str(root / 'Wan2.2-I2V-A14B'),
            python_bin=python_for(source / 'Wan2.2', root.parent / 'envs/wan2.2/bin/python'),
            devices='0', size='480*832', steps=25, offload_model=True)),
        'lingbot-video-moe-30b-a3b': dict(num_frames=81, options=dict(
            proj=str(source / 'lingbot-video'), model_dir=str(root / 'Lingbot-Video-MoE-30B-A3B'),
            python_bin=python_for(source / 'lingbot-video'), devices='0,1,2,3',
            distributed=True, nproc_per_node=4, enable_fsdp_inference=True,
            enable_vlm_fsdp_inference=True, context_parallel_degree=4,
            use_prompt_rewriter=False, run_refiner=False, height=480, width=832, fps=16, steps=25)),
        'hunyuan-video-1.5-i2v': dict(num_frames=81, options=dict(
            proj=str(source / 'HunyuanVideo-1.5'), model_dir=str(root / 'HunyuanVideo-1.5-I2V'),
            python_bin=python_for(source / 'HunyuanVideo-1.5', root.parent / 'envs/hunyuan-video15/bin/python'),
            devices='0', resolution='480p', steps=25, rewrite=False, sr=False)),
        'cogvideox1.5-5b-i2v': dict(num_frames=81, options=dict(
            proj=str(root), model_dir=str(root / 'CogVideoX1.5-5B-I2V'),
            python_bin=python_for(source / 'CogVideo', source / 'minimax-h3/envs/mh3/bin/python'),
            devices='0', height=480, width=832, fps=16)),
    }
    return {'models': {name: {'backend': name, **definitions[name]} for name in MODELS}}


def resolve_profile(name, profile, base):
    import copy
    from .backends import REGISTRY
    profile = copy.deepcopy(profile)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', name):
        raise ValueError(f'Invalid model name: {name}')
    backend = profile.setdefault('backend', name)
    if backend not in {*REGISTRY, 'command'}:
        raise ValueError(f'Unknown backend {backend}; use backend=command for a custom model')
    frames = profile.setdefault('num_frames', 81)
    if type(frames) is not int or frames < 2:
        raise ValueError(f'{name}: num_frames must be an integer >= 2')
    if backend == 'command':
        command = profile.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError(f'{name}: command must be a JSON array of arguments')
        joined = '\n'.join(command)
        if '{request}' not in joined and not all('{' + k + '}' in joined for k in ('image', 'prompt', 'seed', 'output')):
            raise ValueError(f'{name}: pass {{request}} or all of {{image}}, {{prompt}}, {{seed}}, {{output}}')
        allowed = {'request', 'image', 'prompt', 'seed', 'output', 'model', 'task', 'sample_id', 'num_frames'}
        for argument in command:
            for _, field, spec, conversion in string.Formatter().parse(argument):
                if field is not None and (field not in allowed or spec or conversion):
                    raise ValueError(f'{name}: unsupported command placeholder {{{field}}}')
        cwd = Path(profile.get('cwd', base)).expanduser()
        profile['cwd'] = str((cwd if cwd.is_absolute() else Path(base) / cwd).resolve())
        return profile
    options = profile.setdefault('options', {})
    for key in ('proj', 'model_dir', 'ckpt_dir', 'python_bin', 'processor_dir', 'text_encoder_dir', 'refiner_dir'):
        if options.get(key):
            path = Path(options[key]).expanduser()
            path = path if path.is_absolute() else Path(base) / path
            # Resolving bin/python's symlink can escape a virtual environment.
            options[key] = str(path.absolute() if key == 'python_bin' else path.resolve())
    if backend == 'seedance-2.5':
        options['base_url'] = os.environ.get('SEEDANCE_BASE_URL', options.get('base_url', ''))
        options['model'] = os.environ.get('SEEDANCE_MODEL', options.get('model', 'doubao-seedance-2-5-260628'))
        if options.get('request_format', 'aihubmix_json') not in ('aihubmix_json', 'multipart'):
            raise ValueError('Seedance supports aihubmix_json or multipart Videos API requests')
        if options.get('headers') or options.get('extra_body'):
            raise ValueError('Use api_key_env for authentication; custom headers/body are not supported by this adapter')
    elif options.get('use_prompt_rewriter') or options.get('rewrite') or options.get('use_prompt_extend'):
        raise ValueError('Prompt rewriting is disabled: evaluation must use the actual supplied task prompt')
    model = REGISTRY[backend](**options)  # Reject unsupported options before scheduling work.
    if any(getattr(model, flag, False) for flag in ('use_prompt_rewriter', 'rewrite', 'use_prompt_extend')):
        raise ValueError('Disable automatic prompt rewriting in the model options')
    return profile


def check_profile(profile):
    from .backends import REGISTRY, requests
    backend = profile['backend']
    if backend == 'command':
        if not Path(profile['cwd']).is_dir():
            raise ValueError(f'Custom model working directory is missing: {profile["cwd"]}')
        executable = profile['command'][0]
        if '{' in executable or not (shutil.which(executable) or (Path(profile['cwd']) / executable).is_file()):
            raise ValueError(f'Custom model executable is missing: {executable}')
        return
    options = profile['options']
    if backend == 'seedance-2.5':
        url = urlsplit(options['base_url'])
        if url.scheme not in ('https', 'http') or not url.netloc or url.username or url.password or url.query or url.fragment:
            raise ValueError('Set SEEDANCE_BASE_URL to your Videos API endpoint (without credentials or query parameters)')
        if not os.environ.get(options.get('api_key_env', 'SEEDANCE_API_KEY')):
            raise ValueError('Set the API key environment variable: ' + options.get('api_key_env', 'SEEDANCE_API_KEY'))
        if requests is None:
            raise ValueError('Install API client dependency: python -m pip install "requests>=2.32,<3"')
        return
    model = REGISTRY[backend](**options)
    for path in (model._python(), model.proj, options.get('model_dir', options.get('ckpt_dir'))):
        if path and not Path(path).exists():
            raise ValueError(f'Missing model environment/source/checkpoint: {path}; update generation.local.json')
    script = model._runner() if hasattr(model, '_runner') else model._script()
    if not Path(script).is_file():
        raise ValueError(f'Missing model entrypoint: {script}')
