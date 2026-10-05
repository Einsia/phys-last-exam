"""Resolve installation paths without changing tracker settings or measurements."""
from pathlib import Path
import os
import sys
_ROOT = next(candidate for parent in Path(__file__).resolve().parents for candidate in (parent, parent / "_shared") if (candidate / "task_catalog.json").is_file() and (candidate / "shared/model_paths.py").is_file())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from shared.model_paths import model_path


def resolve_tracking_runtime(runtime):
    runtime = dict(runtime)
    python_key = "tracking_python" if "tracking_python" in runtime else "python_track"
    runtime[python_key] = os.environ.get("FINAL_TRACKING_PYTHON") or runtime.get(python_key) or sys.executable
    sam_key = "sam2_snapshot" if "sam2_snapshot" in runtime else "sam2_path"
    runtime[sam_key] = str(model_path(runtime[sam_key], "EVALUATOR_SAM2_TRANSFORMERS_MODEL"))
    runtime["cotracker_source"] = str(model_path(runtime["cotracker_source"], "EVALUATOR_COTRACKER_CODE"))
    runtime["cotracker_checkpoint"] = str(model_path(runtime["cotracker_checkpoint"], "EVALUATOR_COTRACKER_CHECKPOINT"))
    return runtime
