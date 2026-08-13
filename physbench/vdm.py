"""Video-model adapters.

A VDM is called as a subprocess in its own conda env, not imported. The two
dependency sets (diffusers git main + torch cu128 vs numpy/opencv) then never have
to agree, and adding a second model means adding a class here rather than
renegotiating the environment.

The only contract an adapter has to honour: given a first frame and a prompt, write
an mp4 at `out` and return its path.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GenResult:
    path: str
    seconds: float
    cmd: str
    log: str


class VideoModel:
    name = "base"

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int, seed: int,
                 log_path: str | None = None) -> GenResult:
        raise NotImplementedError


@dataclass
class MiniMaxH3(VideoModel):
    """MiniMax-H3 i2va, four-card component split, driven through scripts/i2va.py.

    Model constraints that the caller has to live with (all enforced by the model, not
    by this wrapper): 24 fps fixed, num_frames snapped up to the next 17n+5 in
    [124, 345], height/width multiples of 32, and no negative prompt or guidance scale
    because guidance is distilled into the weights.
    """

    proj: str = "/mnt/einsia/aws01-nvme/einsia-shared/homes/gaomingju/workspace/minimax-h3"
    devices: str = "4,5,6,7"
    flash3: bool = True
    height: int | None = 768
    width: int | None = 1344
    steps: int | None = None
    extra_env: dict = field(default_factory=dict)
    name: str = "minimax-h3"

    def _python(self) -> str:
        return str(Path(self.proj) / "envs" / "mh3" / "bin" / "python")

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 124, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        script = str(Path(self.proj) / "scripts" / "i2va.py")
        # Absolute, because the child runs with cwd inside the model project -- a
        # relative path resolves against the wrong root and i2va.py reports it as a
        # malformed URL rather than a missing file.
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        for p in (self._python(), script, first_frame):
            if not Path(p).exists():
                raise FileNotFoundError(p)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        cmd = [self._python(), script, "--image", first_frame, "--prompt", prompt,
               "--out", out, "--num-frames", str(num_frames), "--seed", str(seed)]
        if self.flash3:
            cmd.append("--flash3")
        if self.height and self.width:
            cmd += ["--height", str(self.height), "--width", str(self.width)]
        if self.steps:
            cmd += ["--steps", str(self.steps)]

        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": self.devices,
            "HF_HOME": f"{self.proj}/cache/hf",
            "HF_HUB_CACHE": f"{self.proj}/cache/hf/hub",
            "HF_ASSETS_CACHE": f"{self.proj}/cache/hf/assets",
            "TORCH_HOME": f"{self.proj}/cache/torch",
            "TRITON_CACHE_DIR": f"{self.proj}/cache/triton",
            # The local model index was repointed at on-disk weights, so nothing needs
            # the network. Offline also stops a silent 135 GB re-download into the hub
            # cache. Note this makes --flash3 fall back unless i2va.py's cache preload
            # works, which it handles itself.
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        })
        env.update({k: str(v) for k, v in self.extra_env.items()})

        t0 = time.time()
        # Streamed to a file rather than captured, so a 6-minute denoise can be watched
        # with tail instead of only being readable once it is over (or lost when the
        # host OOM killer takes the process mid-run).
        if log_path is None:
            proc = subprocess.run(cmd, env=env, cwd=self.proj, capture_output=True, text=True)
            log = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
            rc = proc.returncode
        else:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as fh:
                rc = subprocess.run(cmd, env=env, cwd=self.proj,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            log = Path(log_path).read_text()
        secs = time.time() - t0
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(
                f"{self.name} failed (rc={rc})\n"
                f"cmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(path=str(out), seconds=secs, cmd=shlex.join(cmd), log=log)


REGISTRY: dict[str, type[VideoModel]] = {"minimax-h3": MiniMaxH3}


def build_model(name: str, **kw) -> VideoModel:
    if name not in REGISTRY:
        raise KeyError(f"unknown model {name}; have {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)
