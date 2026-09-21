"""Resolve local resources, allowing explicit CLI paths and legacy installations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def resource(relative):
    candidates = [ROOT / relative, ROOT.parents[1] / relative]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def model_resource(relative):
    return ROOT / 'g8/P34/models' / relative
