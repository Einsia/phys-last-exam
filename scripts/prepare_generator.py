"""Install the selected generator only; write its ready-to-use local profile."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv

from run import ROOT, run, save_profile

DIFFUSERS = 'git+https://github.com/huggingface/diffusers.git@c6df88a511a98740646ee55577b590c9852650ce'
RECIPES = {
    'minimax-h3': ('MiniMax-H3', 'MiniMaxAI/MiniMax-H3', '42ed227ee7df40d41602854ae760620d6eb651fe'),
    'cosmos3-super-image2video': ('Cosmos3-Super-Image2Video', 'nvidia/Cosmos3-Super-Image2Video', '580f3f28e33ba93c8d464768876da8c322619aad'),
    'vbvr-wan2.2': ('VBVR-Wan2.2', 'vinesnt/VBVR-Wan2.2', 'cdb2fa248cb93d8b258cb9de76e1a5ea58b99eca'),
    'wan2.2-i2v-a14b': ('Wan2.2-I2V-A14B', 'Wan-AI/Wan2.2-I2V-A14B', '206a9ee1b7bfaaf8f7e4d81335650533490646a3'),
    'lingbot-video-moe-30b-a3b': ('Lingbot-Video-MoE-30B-A3B', 'robbyant/lingbot-video-moe-30b-a3b', 'f2e538f64afe00cc4ae674db2aeb52e2945edfd5'),
    'hunyuan-video-1.5-i2v': ('HunyuanVideo-1.5-I2V', 'tencent/HunyuanVideo-1.5', '9b49404b3f5df2a8f0b31df27a0c7ab872e7b038'),
    'cogvideox1.5-5b-i2v': ('CogVideoX1.5-5B-I2V', 'zai-org/CogVideoX1.5-5B-I2V', '46c90528707aebbe69066390b4fe7e7d24c9c2a4'),
}
SOURCES = {
    'wan2.2-i2v-a14b': ('Wan2.2', 'https://github.com/Wan-Video/Wan2.2.git', '1ea34ff48f87168174e12956e200b1d908b1c5ff'),
    'lingbot-video-moe-30b-a3b': ('lingbot-video', 'https://github.com/Robbyant/lingbot-video.git', 'dd5c231e793406c6a8893e9e4307008a9c2adfe4'),
    'hunyuan-video-1.5-i2v': ('HunyuanVideo-1.5', 'https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5.git', '60783e704160023913bee78f0b47036d393d4dfa'),
}


def checkout(directory, url, revision):
    if directory.exists():
        head = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain'], text=True)
        if head != revision or dirty:
            raise ValueError(f'Source checkout differs from the pinned recipe: {directory}. '
                             'Use --config to reuse a custom installation, or choose a new --model-root.')
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    # A failed fetch must not leave a source directory with no HEAD that would
    # prevent the next invocation from retrying installation.
    with tempfile.TemporaryDirectory(prefix=f'.{directory.name}-', dir=directory.parent) as temporary:
        staged = Path(temporary) / 'source'
        run(['git', 'init', '-q', staged])
        run(['git', '-C', staged, 'remote', 'add', 'origin', url])
        run(['git', '-C', staged, 'fetch', '--depth=1', 'origin', revision])
        run(['git', '-C', staged, 'checkout', '--detach', 'FETCH_HEAD'])
        staged.rename(directory)


def download(python, repo, revision, directory):
    # Tokens are picked up by huggingface_hub from HF_TOKEN or its login store.
    # Never put a credential in a command argument or generated configuration.
    run([python, '-c', (
        'import sys; from huggingface_hub import snapshot_download; '
        'snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3])'),
        repo, revision, directory])


def prepare(controller, model, model_root, devices, *, plan_only=False):
    model_root = Path(model_root).expanduser().resolve()
    # Read defaults using the controller so the launcher stays dependency-free.
    config = json.loads(subprocess.check_output([controller, '-c',
        'import json,sys; from generation.config import default_config; '
        'print(json.dumps(default_config(sys.argv[1])))', str(model_root)], cwd=ROOT, text=True))
    profile = config['models'][model]
    options = profile['options']
    directory = ROOT / '.venv' / f'generator-{model}'
    python = directory / 'bin/python'
    options['python_bin'] = str(python)
    if model not in SOURCES:
        options['proj'] = str(model_root)
    if devices:
        options['devices'] = devices
        if model == 'lingbot-video-moe-30b-a3b':
            count = len(devices.split(','))
            options.update(distributed=count > 1, nproc_per_node=count,
                           context_parallel_degree=count,
                           enable_fsdp_inference=count > 1, enable_vlm_fsdp_inference=count > 1)
    if plan_only:
        return save_profile(model, profile)
    if model == 'wan2.2-i2v-a14b':
        cuda_home = os.environ.get('CUDA_HOME')
        nvcc = shutil.which('nvcc') or (cuda_home and (Path(cuda_home) / 'bin/nvcc').is_file())
        if not nvcc or not (shutil.which('c++') or shutil.which('g++')):
            raise ValueError('Wan2.2 builds FlashAttention: install a CUDA toolkit (nvcc) '
                             'and a C++ compiler, then rerun this command.')
    model_root.mkdir(parents=True, exist_ok=True)
    if model in SOURCES:
        name, url, revision = SOURCES[model]
        checkout(model_root / 'source' / name, url, revision)
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(directory)
    marker = directory / '.generator-ready'
    signature = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if not marker.is_file() or marker.read_text() != signature:
        run([python, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'])
        if model == 'lingbot-video-moe-30b-a3b':
            source = model_root / 'source/lingbot-video'
            # The upstream February nightly wheels have expired from the index.
            # Use a stable CUDA pair; retain all other upstream runtime pins.
            run([python, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0',
                 'torchaudio==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
            requirements = directory / 'lingbot-requirements.txt'
            lines = (source / 'requirements.txt').read_text().splitlines()
            lines = [line for line in lines if not line.startswith(
                ('--extra-index-url', 'torch==', 'torchvision==', 'decord>='))]
            requirements.write_text('\n'.join(lines) + '\n')
            run([python, '-m', 'pip', 'install', '-r', requirements])
            # The official script adds its source root to sys.path. Avoid the
            # optional refiner decoder's obsolete wheel/metadata on Python 3.12;
            # the base I2V path does not use it and needs no editable package.
        elif model in {'wan2.2-i2v-a14b', 'hunyuan-video-1.5-i2v'}:
            run([python, '-m', 'pip', 'install', 'torch==2.6.0', 'torchvision==0.21.0',
                 'torchaudio==2.6.0', '--index-url', 'https://download.pytorch.org/whl/cu124'])
            source = Path(options['proj'])
            if model == 'wan2.2-i2v-a14b':
                # Install build prerequisites before FlashAttention's isolated build.
                run([python, '-m', 'pip', 'install', 'packaging', 'ninja', 'psutil'])
            run([python, '-m', 'pip', 'install', '--no-build-isolation', '-r', source / 'requirements.txt'])
        else:
            run([python, '-m', 'pip', 'install', 'torch==2.11.0', 'torchvision==0.26.0',
                 'torchaudio==2.11.0', '--index-url', 'https://download.pytorch.org/whl/cu128'])
            run([python, '-m', 'pip', 'install', DIFFUSERS, 'transformers==5.14.1',
                 'accelerate==1.14.0', 'huggingface-hub==1.32.0', 'av==18.0.0',
                 'imageio[ffmpeg]', 'einops', 'scipy', 'sentencepiece', 'protobuf',
                 'ftfy', 'soundfile', 'librosa'])
            if model == 'cosmos3-super-image2video':
                run([python, '-m', 'pip', 'install', 'cosmos-guardrail==0.3.1'])
        run([python, '-m', 'pip', 'check'])
        run([python, '-c', 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'])
        marker.write_text(signature)
    name, repo, revision = RECIPES[model]
    weights = model_root / name
    download(python, repo, revision, weights)
    if model == 'cosmos3-super-image2video':
        # The default checker has its own HF cache dependencies. Prepare them
        # online, then verify the offline constructor used by the adapter.
        warm = 'from cosmos_guardrail import CosmosSafetyChecker; CosmosSafetyChecker()'
        online = dict(os.environ)
        online.pop('HF_HUB_OFFLINE', None)
        online.pop('TRANSFORMERS_OFFLINE', None)
        run([python, '-c', warm], env=online)
        offline = {**online, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        run([python, '-c', warm], env=offline)
    if model == 'hunyuan-video-1.5-i2v':
        download(python, 'Qwen/Qwen2.5-VL-7B-Instruct', 'cc594898137f460bfe9f0759e9844b3ce807cfb5', weights / 'text_encoder/llm')
        download(python, 'google/byt5-small', '68377bdc18a2ffec8a0533fef03b1c513a4dd49d', weights / 'text_encoder/byt5-small')
        download(python, 'black-forest-labs/FLUX.1-Redux-dev', 'c95859fbf7703ca4d6824b4da4407d7cd0434f81', weights / 'vision_encoder/siglip')
        run([python, '-m', 'modelscope.cli.cli', 'download', '--model',
             'AI-ModelScope/Glyph-SDXL-v2', '--local_dir', weights / 'text_encoder/Glyph-SDXL-v2'])
    return save_profile(model, profile)
