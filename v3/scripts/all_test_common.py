"""Paths and durable records for the external all_test V3 evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path

V3 = Path(__file__).resolve().parents[1]
WORKSPACE = V3.parent
# Keep the repository usable outside the original evaluation workstation. The
# video dataset and output directory are external and configurable per run.
ROOT = Path(os.environ.get('V3_OUTPUT_ROOT', str(WORKSPACE / 'runs' / 'v3_evaluator'))).expanduser().resolve()
SOURCE = Path(os.environ.get('VDMBENCH_DATA_ROOT', str(WORKSPACE))).expanduser().resolve()
ANNOTATED = {'P37', 'P38', 'P39', 'P41', 'P43', 'P49'}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
