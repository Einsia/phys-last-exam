"""Parent side of the neural trackers: build a job, run the worker, read the npz.

Torch never enters this env. The worker runs in envs/track as a subprocess, exactly
as the VDM does in physbench/vdm.py, so the measurement env stays numpy/opencv-only
and the two dependency sets never have to agree.

Frames are handed over as a .npy the caller already decoded rather than as a path to
the mp4. Two reasons: the synthetic clips never touch disk as video at all, and
re-decoding in the child would put a second decoder on the measurement path, where a
different libav build could shift pixel values and make the tracks incomparable
between backends.

Weights are local and the child runs with HF_HUB_OFFLINE=1 -- see
scripts/fetch_tracker_weights.sh for why that is a correctness requirement and not
just a convenience.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .first_frame import BallSeed

PROJ = Path(__file__).resolve().parents[1]
CACHE = PROJ / "cache"


def _sam2_snapshot(repo: str = "facebook/sam2.1-hiera-small") -> str:
    """Resolve the local HF snapshot dir without importing huggingface_hub."""
    slug = "models--" + repo.replace("/", "--")
    snaps = CACHE / "hf" / "hub" / slug / "snapshots"
    if snaps.is_dir():
        cand = sorted(p for p in snaps.iterdir() if p.is_dir())
        if cand:
            return str(cand[-1])
    raise FileNotFoundError(
        f"no local snapshot for {repo} under {snaps}. "
        f"Run scripts/fetch_tracker_weights.sh first.")


@dataclass
class TrackerBackendConfig:
    """Where the worker and the weights live, and which GPU to use."""

    python: str = str(PROJ / "envs" / "track" / "bin" / "python")
    worker: str = str(PROJ / "scripts" / "track_worker.py")
    sam2_repo: str = "facebook/sam2.1-hiera-small"
    sam3_src: str = str(CACHE / "sam3")
    sam3_checkpoint: str = str(CACHE / "sam3" / "sam3.pt")
    sam3_prompt: str = "ball"
    sam3_detection_thresh: float = 0.05
    cotracker_ckpt: str = str(CACHE / "torch" / "hub" / "checkpoints" / "scaled_offline.pth")
    cotracker_src: str = str(CACHE / "cotracker")
    devices: str = "6"
    dtype: str = "bfloat16"
    bg_grid: bool = True
    # Frame handoff is ~380 MB for a 124-frame 1344x768 clip, so it must not land on
    # whatever /tmp happens to be -- on this host / is a 969 GB root that other tenants
    # keep at 100%. Defaults into the project tree, which is on the big volume and is
    # the same filesystem the weights and videos already live on. An explicit TMPDIR
    # still wins, so a caller can point this at faster storage.
    scratch_dir: str = str(PROJ / "cache" / "scratch")
    extra_env: dict = field(default_factory=dict)


@dataclass
class NeuralTracks:
    """Raw per-frame arrays for each backend that ran, plus the worker's own timing."""

    xy: dict[str, np.ndarray]
    radius: dict[str, np.ndarray]
    score: dict[str, np.ndarray]
    camera_shift: np.ndarray | None
    seconds: float
    cmd: str


def run_worker(frames: np.ndarray, seed: BallSeed, backends: tuple[str, ...],
               cfg: TrackerBackendConfig | None = None) -> NeuralTracks:
    """Run the requested neural backends over `frames` in one subprocess call.

    Both backends in one call on purpose: loading SAM2 and CoTracker costs a few
    seconds each, and a single process amortises that over the pair while keeping only
    one clip's frames resident.
    """
    cfg = cfg or TrackerBackendConfig()
    for p in (cfg.python, cfg.worker):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"{p} missing. Create envs/track with "
                f"`uv pip install --python envs/track/bin/python -r envs/track-requirements.txt`.")

    # Honour an explicit TMPDIR; otherwise use the project scratch rather than /tmp.
    parent = None
    if not os.environ.get("TMPDIR"):
        parent = cfg.scratch_dir
        Path(parent).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="physbench-track-", dir=parent) as td:
        td = Path(td)
        fr = td / "frames.npy"
        np.save(fr, np.ascontiguousarray(frames))
        sam3_src = Path(cfg.sam3_src).expanduser()
        if not sam3_src.is_absolute():
            sam3_src = PROJ / sam3_src
        sam3_checkpoint = Path(cfg.sam3_checkpoint).expanduser()
        if not sam3_checkpoint.is_absolute():
            sam3_checkpoint = PROJ / sam3_checkpoint
        job = {
            "frames": str(fr),
            "seed": {"cx": float(seed.cx), "cy": float(seed.cy),
                     "radius": float(seed.radius)},
            "backends": list(backends),
            "device": "cuda",
            "dtype": cfg.dtype,
            "bg_grid": bool(cfg.bg_grid),
            "sam2_path": (_sam2_snapshot(cfg.sam2_repo)
                          if "sam2" in backends else None),
            "sam3_src": str(sam3_src),
            "sam3_checkpoint": str(sam3_checkpoint),
            "sam3_prompt": cfg.sam3_prompt,
            "sam3_detection_thresh": float(cfg.sam3_detection_thresh),
            "sam3_frames": str(td / "sam3-frames"),
            "cotracker_ckpt": cfg.cotracker_ckpt,
            "cotracker_src": cfg.cotracker_src,
            "out": str(td / "track.npz"),
        }
        jp = td / "job.json"
        jp.write_text(json.dumps(job))

        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": cfg.devices,
            "HF_HOME": str(CACHE / "hf"),
            "HF_HUB_CACHE": str(CACHE / "hf" / "hub"),
            "TORCH_HOME": str(CACHE / "torch"),
            # Offline is a correctness setting: a silently re-downloaded checkpoint is
            # a silently different tracker, and the noise floor no longer describes it.
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        })
        env.update({k: str(v) for k, v in cfg.extra_env.items()})

        cmd = [cfg.python, cfg.worker, str(jp)]
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        secs = time.time() - t0
        if proc.returncode != 0 or not Path(job["out"]).exists():
            raise RuntimeError(
                f"tracker worker failed (rc={proc.returncode})\n"
                f"cmd: {shlex.join(cmd)}\n{(proc.stdout + proc.stderr)[-4000:]}")

        d = np.load(job["out"])
        xy = {b: d[f"{b}_xy"] for b in backends}
        rad = {b: d[f"{b}_radius"] for b in backends}
        sc = {b: d[f"{b}_score"] for b in backends}
        shift = d["cotracker_camera_shift"] if "cotracker_camera_shift" in d.files else None
        return NeuralTracks(xy=xy, radius=rad, score=sc, camera_shift=shift,
                            seconds=secs, cmd=shlex.join(cmd))
