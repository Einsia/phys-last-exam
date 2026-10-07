#!/usr/bin/env python3
"""Provision isolated runtimes and launch the benchmark's standard workflows."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER_PACKAGES = ('av==18.0.0', 'Pillow==12.2.0', 'requests>=2.32,<3')
LOCAL_MODELS = ('minimax-h3', 'cosmos3-super-image2video', 'vbvr-wan2.2',
                'wan2.2-i2v-a14b', 'lingbot-video-moe-30b-a3b',
                'hunyuan-video-1.5-i2v', 'cogvideox1.5-5b-i2v')
MODELS = ('seedance-2.5', *LOCAL_MODELS)


def run(command, **kwargs):
    subprocess.run([str(value) for value in command], check=True, **kwargs)


def environment(kind, packages=()):
    override = os.environ.get(f'PLE_{kind.upper()}_PYTHON')
    if override:
        python = Path(override).expanduser().absolute()
        if not python.is_file():
            raise ValueError(f'Python executable not found: {python}')
        return python
    directory = ROOT / '.venv' / kind
    python = directory / 'bin/python'
    if not python.is_file():
        print(f'Creating {kind} environment...', flush=True)
        venv.EnvBuilder(with_pip=True).create(directory)
    if packages:
        signature = hashlib.sha256('\n'.join(packages).encode()).hexdigest()
        marker = directory / '.packages-ready'
        if not marker.is_file() or marker.read_text() != signature:
            run([python, '-m', 'pip', 'install', *packages])
            run([python, '-m', 'pip', 'check'])
            marker.write_text(signature)
    return python


def save_profile(model, profile):
    path = ROOT / 'runs' / 'launch' / f'{model}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({'models': {model: profile}}, indent=2) + '\n'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content)
    temporary.replace(path)
    return path


def generate_command(python, model, config, forwarded):
    return [python, ROOT / 'generate.py', '--models', model,
            '--config', config, '--output', 'videos', '--resume', *forwarded]


def configured_backend(path, model):
    if not path.is_file():
        raise ValueError(f'Generator configuration not found: {path}')
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot read generator configuration {path}: {error}') from error
    definitions = config.get('models') if isinstance(config, dict) else None
    if not isinstance(definitions, dict) or not isinstance(definitions.get(model), dict):
        raise ValueError(f'Configuration must contain a models object with a profile for {model}')
    return definitions[model].get('backend', model)


def evaluation_input(value):
    path = Path(value).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == '.json':
        return ['--manifest', path]
    if not path.is_dir():
        raise ValueError(f'Video directory or manifest not found: {path}')
    manifest = path / 'manifest.json'
    return ['--manifest', manifest] if manifest.is_file() else ['--video-root', path]


def evaluation_python(dry_run):
    if dry_run:
        return environment('controller', CONTROLLER_PACKAGES)
    python = environment('eval')
    if os.environ.get('PLE_EVAL_PYTHON'):
        return python
    models = Path(os.environ.get('FINAL_MODELS_DIR', ROOT / 'models')).expanduser().resolve()
    needed = ('Qwen3.6-27B/config.json', 'grounding-dino-tiny/config.json',
              'sam2.1-hiera-small/sam2.1_hiera_small.pt',
              'sam2.1-hiera-small-transformers/config.json',
              'sam2.1-hiera-large-transformers/config.json',
              'cotracker3/scaled_offline.pth', 'cotracker3/source/hubconf.py')
    signature = hashlib.sha256((ROOT / 'setup.sh').read_bytes()
                               + (ROOT / 'requirements.txt').read_bytes()
                               + str(models).encode()).hexdigest()
    marker = python.parent.parent / '.evaluation-ready'
    if (not marker.is_file() or marker.read_text() != signature
            or any(not (models / name).is_file() for name in needed)):
        env = dict(os.environ)
        env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
        run(['bash', ROOT / 'setup.sh'], env=env, cwd=ROOT)
        marker.write_text(signature)
    return python


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # A positional with parse_known_args can consume a forwarded flag's value.
    # Only the first argument after evaluate may be the optional input path.
    if len(argv) > 1 and argv[0] == 'evaluate' and not argv[1].startswith('-'):
        argv[1:2] = ['--input', argv[1]]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog=(
        'Extra flags are forwarded to generate.py or evaluate.py. '
        'Examples: --tasks P21 --seeds 42; --dry-run; --output PATH.'))
    commands = parser.add_subparsers(dest='action', required=True)
    generation = commands.add_parser('generate', allow_abbrev=False, help='Generate with any built-in or configured custom model',
                                    epilog='Available models: ' + ', '.join(MODELS))
    generation.add_argument('model', help='Built-in model name, or custom name with --config')
    generation.add_argument('--config', type=Path, help='Use an existing or custom generator configuration')
    generation.add_argument('--model-root', type=Path, default=ROOT / 'models/generation')
    generation.add_argument('--devices', help='Physical GPU IDs for local generation, e.g. 1 or 0,1,2,3')
    seedance = commands.add_parser('seedance', allow_abbrev=False, help='Generate using a Seedance Videos API')
    seedance.add_argument('--config', type=Path, help='Optional existing generator configuration')
    local = commands.add_parser('open', allow_abbrev=False, help='Install and run one open-source generator')
    local.add_argument('model', choices=LOCAL_MODELS)
    local.add_argument('--config', type=Path, help='Reuse an existing installation; skip provisioning')
    local.add_argument('--model-root', type=Path, default=ROOT / 'models/generation')
    local.add_argument('--devices', help='Visible physical GPU IDs, e.g. 1 or 0,1,2,3')
    evaluation = commands.add_parser('evaluate', allow_abbrev=False, help='Evaluate every video in a directory or manifest')
    evaluation.add_argument('--input', default='videos', help='Video directory or manifest (default: videos)')
    args, forwarded = parser.parse_known_args(argv)
    if args.action == 'generate':
        if args.model not in MODELS and not args.config:
            parser.error('Unknown model. Choose ' + ', '.join(MODELS)
                         + ', or supply --config for your custom model.')
        args.action = 'seedance' if args.model == 'seedance-2.5' else 'open'
    if sys.platform != 'linux' or sys.version_info[:2] != (3, 12):
        parser.error('Use Linux and Python 3.12. Set PLE_PYTHON to its executable if needed.')
    os.chdir(ROOT)
    if args.action == 'evaluate':
        inputs = evaluation_input(args.input)  # Fail before installing anything for invalid inputs.
        python = evaluation_python('--dry-run' in forwarded)
        command = [python, ROOT / 'evaluate.py', *inputs,
                   '--output', 'runs/evaluation', '--resume', *forwarded]
    else:
        model = 'seedance-2.5' if args.action == 'seedance' else args.model
        backend = configured_backend(args.config, model) if args.config else model
        devices = getattr(args, 'devices', None)
        if devices is not None:
            ids = devices.split(',')
            if not all(value.isascii() and value.isdecimal() for value in ids) or len(set(ids)) != len(ids):
                raise ValueError('--devices must contain unique GPU IDs, e.g. 1 or 0,1,2,3')
            if backend not in LOCAL_MODELS:
                raise ValueError('--devices applies only to local model backends')
            if args.config:
                forwarded = ['--devices', devices, *forwarded]
        if args.action == 'seedance' and not args.config and '--dry-run' not in forwarded:
            if not os.environ.get('SEEDANCE_BASE_URL'):
                if sys.stdin.isatty():
                    os.environ['SEEDANCE_BASE_URL'] = input('Seedance Videos API base URL: ').strip()
                if not os.environ.get('SEEDANCE_BASE_URL'):
                    parser.error('Set SEEDANCE_BASE_URL in the environment, or run interactively to enter it.')
            if not os.environ.get('SEEDANCE_API_KEY'):
                if sys.stdin.isatty():
                    os.environ['SEEDANCE_API_KEY'] = getpass.getpass('Seedance API key: ').strip()
                if not os.environ.get('SEEDANCE_API_KEY'):
                    parser.error('Set SEEDANCE_API_KEY in the environment, or run interactively to enter it.')
        python = environment('controller', CONTROLLER_PACKAGES)
        if args.action == 'seedance':
            model = 'seedance-2.5'
            config = args.config
            if not config:
                config = save_profile(model, {'backend': model, 'num_frames': 120, 'options': {
                    'base_url': os.environ.get('SEEDANCE_BASE_URL', ''),
                    'model': os.environ.get('SEEDANCE_MODEL', 'doubao-seedance-2-5-260628'),
                    'api_key_env': 'SEEDANCE_API_KEY', 'seconds': 5,
                    'request_format': os.environ.get('SEEDANCE_REQUEST_FORMAT', 'litellm_json')}})
        else:
            model = args.model
            config = args.config
            if not config:
                # Keep heavyweight imports and generator packages out of this launcher.
                from prepare_generator import prepare
                config = prepare(python, model, args.model_root, args.devices,
                                 plan_only='--dry-run' in forwarded)
        command = generate_command(python, model, config, forwarded)
    # Do not print command arguments: custom arguments may contain sensitive information.
    run(command, cwd=ROOT)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            raise SystemExit(exc.returncode)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
