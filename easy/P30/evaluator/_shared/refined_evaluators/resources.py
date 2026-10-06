"""Resolve packaged tasks and explicitly configured external models."""
import json
from functools import lru_cache
from pathlib import Path
from shared.model_paths import final_root, model_path

ROOT = final_root()


def resource(relative):
    if relative == 'shared_vendor/co-tracker':
        return model_path('cotracker3/source', 'EVALUATOR_COTRACKER_CODE')
    return ROOT / relative


def model_resource(relative):
    variables = {
        'grounding-dino-tiny': 'EVALUATOR_GROUNDING_DINO_MODEL',
        'sam2.1-hiera-small/sam2.1_hiera_small.pt': 'EVALUATOR_SAM2_SMALL_CHECKPOINT',
        'cotracker3/scaled_offline.pth': 'EVALUATOR_COTRACKER_CHECKPOINT',
    }
    return model_path(relative, variables.get(relative))


@lru_cache(maxsize=None)
def task_resource(task_id):
    catalog = json.loads((ROOT / 'task_catalog.json').read_text(encoding='utf-8'))
    path = ROOT.parents[1]
    if task_id not in catalog or path.name != task_id or not path.is_dir():
        raise ValueError(f'Invalid packaged task path for {task_id}: {path}')
    return path
