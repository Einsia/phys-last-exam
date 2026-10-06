"""Explicit external model locations for the standalone evaluation package."""
import os
from pathlib import Path


def final_root():
    return Path(__file__).resolve().parents[1]


def models_root():
    return Path(os.environ.get("FINAL_MODELS_DIR", str(final_root() / "models"))).expanduser().resolve()


def model_path(relative, env=None):
    value = os.environ.get(env) if env else None
    return Path(value).expanduser().resolve() if value else models_root() / relative
