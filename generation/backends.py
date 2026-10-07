"""Video-model adapters.

A VDM is called as a subprocess in its own conda env, not imported. The two
dependency sets (diffusers git main + torch cu128 vs numpy/opencv) then never have
to agree, and adding a second model means adding a class here rather than
renegotiating the environment.

The only contract an adapter has to honour: given a first frame and a prompt, write
an mp4 at `out` and return its path.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

try:
    import requests
except ImportError:
    requests = None  # Only the Seedance API adapter needs this package.


@dataclass
class GenResult:
    path: str
    seconds: float
    cmd: str
    log: str
    meta: dict[str, Any] = field(default_factory=dict)


class VideoAPIConfigurationError(RuntimeError):
    """A shared API protocol error that should stop further submissions."""


class VideoRemoteJobError(RuntimeError):
    """A terminal remote failure, distinct from an uncertain HTTP outcome."""

    def __init__(self, message: str, *, status: str, error: Any):
        super().__init__(message)
        self.status = status.lower()
        self.error = error
        self.code = error.get("code") if isinstance(error, dict) else None


class VideoModel:
    name = "base"

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int, seed: int,
                 log_path: str | None = None) -> GenResult:
        raise NotImplementedError


def _resolve_path(base: str | Path, path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(base) / p


@dataclass(kw_only=True)
class _BaseSubprocessModel(VideoModel):
    proj: str
    devices: str = "0"
    steps: int | None = None
    height: int | None = None
    width: int | None = None
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "base"

    def _python(self) -> str:
        raise NotImplementedError

    def _script(self) -> str:
        raise NotImplementedError

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(self.devices),
            "PYTHONUNBUFFERED": "1",
        })
        env.update({k: str(v) for k, v in self.extra_env.items()})
        return env

    def _exists(self, *paths: str) -> None:
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(p)

    def _run(self, cmd: list[str], *, log_path: str | None = None) -> tuple[int, str, float]:
        t0 = time.time()
        if log_path is None:
            proc = subprocess.run(cmd, env=self._env(), cwd=self.proj, capture_output=True, text=True)
            log = proc.stdout + ("\n[stderr]\n" + proc.stderr if proc.stderr.strip() else "")
            rc = proc.returncode
        else:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as fh:
                rc = subprocess.run(cmd, env=self._env(), cwd=self.proj,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            log = Path(log_path).read_text()
        return rc, log, time.time() - t0


@dataclass(kw_only=True)
class MiniMaxH3(_BaseSubprocessModel):
    """MiniMax-H3 i2va, four-card component split, driven through the bundled runner.

    Model constraints that the caller has to live with (all enforced by the model, not
    by this wrapper): 24 fps fixed, num_frames snapped up to the next 17n+5 in
    [124, 345], height/width multiples of 32, and no negative prompt or guidance scale
    because guidance is distilled into the weights.
    """

    proj: str
    model_dir: str
    python_bin: str | None = None
    devices: str = "0,1,2,3"
    flash3: bool = True
    height: int | None = 768
    width: int | None = 1344
    steps: int | None = None
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "minimax-h3"

    def _python(self) -> str:
        return self.python_bin or str(Path(self.proj) / ".venv" / "bin" / "python")

    def _script(self) -> str:
        return str(Path(__file__).resolve().parent / "runners" / "run_minimax_h3.py")

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 124, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        script = self._script()
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        self._exists(self._python(), script, first_frame)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        cmd = [self._python(), script, "--image", first_frame, "--prompt", prompt,
               "--out", out, "--num-frames", str(num_frames), "--seed", str(seed)]
        cmd += ["--model", str(_resolve_path(self.proj, self.model_dir).resolve())]
        if self.height and self.width:
            cmd += ["--height", str(self.height), "--width", str(self.width)]
        if self.flash3:
            cmd.append("--flash3")
        if self.steps:
            cmd += ["--steps", str(self.steps)]

        rc, log, secs = self._run(cmd, log_path=log_path)
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(path=str(out), seconds=secs, cmd=shlex.join(cmd), log=log)


@dataclass(kw_only=True)
class WanI2V(_BaseSubprocessModel):
    proj: str
    ckpt_dir: str
    python_bin: str | None = None
    devices: str = "0"
    task: str = "i2v-14B"
    size: str = "1280*720"
    use_prompt_extend: bool = False
    prompt_extend_model: str | None = None
    prompt_extend_method: str | None = None
    input_mode: str = "image"
    distributed: bool = False
    nproc_per_node: int = 1
    dit_fsdp: bool = False
    t5_fsdp: bool = False
    ulysses_size: int = 1
    ring_size: int = 1
    offload_model: bool | None = None
    height: int | None = 768
    width: int | None = 1344
    steps: int | None = None
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "wan"

    def _legal_num_frames(self, requested: int) -> int:
        # Wan's I2V path reshapes the time axis in groups of 4 and expects frame_num=4n+1.
        # The benchmark default (124) comes from MiniMax-H3, so adapt here rather than
        # forcing the shared config to satisfy incompatible model constraints.
        adjusted = requested - ((requested - 1) % 4)
        if adjusted < 5:
            raise ValueError(f"{self.name} requires frame_num=4n+1 with n>=1, got {requested}")
        return adjusted

    def _python(self) -> str:
        if self.python_bin:
            return self.python_bin
        cand = Path(self.proj) / ".venv" / "bin" / "python"
        return str(cand if cand.exists() else Path(self.proj) / "envs" / "wan" / "bin" / "python")

    def _script(self) -> str:
        return str(Path(self.proj) / "generate.py")

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 124, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        script = self._script()
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        ckpt_dir = str(_resolve_path(self.proj, self.ckpt_dir).resolve())
        self._exists(self._python(), script, first_frame, ckpt_dir)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        effective_num_frames = self._legal_num_frames(num_frames)

        task_flag = self.task
        if self.distributed:
            torchrun = [
                self._python(),
                "-m", "torch.distributed.run",
                f"--nproc_per_node={self.nproc_per_node}",
                script,
            ]
            cmd = torchrun + ["--task", task_flag, "--size", self.size,
                              "--ckpt_dir", ckpt_dir, "--image", first_frame,
                              "--prompt", prompt, "--save_file", out]
        else:
            cmd = [self._python(), script, "--task", task_flag, "--size", self.size,
                   "--ckpt_dir", ckpt_dir, "--image", first_frame, "--prompt", prompt,
                   "--save_file", out]
        if self.steps:
            cmd += ["--sample_steps", str(self.steps)]
        cmd += ["--frame_num", str(effective_num_frames)]
        if self.use_prompt_extend:
            cmd.append("--use_prompt_extend")
            if self.prompt_extend_model:
                cmd += ["--prompt_extend_model", self.prompt_extend_model]
            if self.prompt_extend_method:
                cmd += ["--prompt_extend_method", self.prompt_extend_method]
        if self.dit_fsdp:
            cmd.append("--dit_fsdp")
        if self.t5_fsdp:
            cmd.append("--t5_fsdp")
        if self.ulysses_size > 1:
            cmd += ["--ulysses_size", str(self.ulysses_size)]
        if self.ring_size > 1:
            cmd += ["--ring_size", str(self.ring_size)]
        if self.offload_model is not None:
            cmd += ["--offload_model", "True" if self.offload_model else "False"]
        if self.input_mode == "video":
            cmd += ["--last_frame", first_frame]
        cmd += ["--base_seed", str(seed)]

        rc, log, secs = self._run(cmd, log_path=log_path)
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(
            path=str(out),
            seconds=secs,
            cmd=shlex.join(cmd),
            log=log,
            meta={
                "requested_num_frames": num_frames,
                "effective_num_frames": effective_num_frames,
            },
        )


@dataclass(kw_only=True)
class Wan22I2V(WanI2V):
    """Wan2.2 MoE image-to-video through the official Wan2.2 launcher."""

    task: str = "i2v-A14B"
    name: str = "wan2.2-i2v-a14b"

    def _script(self) -> str:
        return str(Path(self.proj) / "generate.py")


@dataclass(kw_only=True)
class _DiffusersImageToVideo(_BaseSubprocessModel):
    """Common subprocess adapter for local Diffusers image-to-video checkpoints."""

    model_dir: str
    python_bin: str | None = None
    devices: str = "0"
    device_map: str | None = None
    gpu_only: bool = False
    height: int = 480
    width: int = 832
    fps: int = 16
    steps: int = 50
    guidance_scale: float = 6.0
    extra_env: dict[str, Any] = field(default_factory=dict)

    def _python(self) -> str:
        return self.python_bin or str(Path(self.proj) / ".venv" / "bin" / "python")

    def _runner(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _legal_num_frames(requested: int) -> int:
        adjusted = requested - ((requested - 1) % 4)
        if adjusted < 5:
            raise ValueError(f"{requested} is too short; expected 4k+1 frames with k>=1")
        return adjusted

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 81, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        model_dir = str(_resolve_path(self.proj, self.model_dir).resolve())
        self._exists(self._python(), self._runner(), first_frame, model_dir)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        effective_num_frames = self._legal_num_frames(num_frames)
        cmd = [self._python(), self._runner(), "--model-path", model_dir,
               "--image", first_frame, "--prompt", prompt, "--output", out,
               "--num-frames", str(effective_num_frames), "--height", str(self.height),
               "--width", str(self.width), "--fps", str(self.fps), "--steps", str(self.steps),
               "--guidance-scale", str(self.guidance_scale), "--seed", str(seed)]
        if self.device_map:
            cmd += ["--device-map", self.device_map]
        if self.gpu_only:
            cmd.append("--gpu-only")
        rc, log, secs = self._run(cmd, log_path=log_path)
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(path=out, seconds=secs, cmd=shlex.join(cmd), log=log,
                         meta={"requested_num_frames": num_frames,
                               "effective_num_frames": effective_num_frames,
                               "height": self.height, "width": self.width,
                               "fps": self.fps, "steps": self.steps})


@dataclass(kw_only=True)
class CogVideoX15I2V(_DiffusersImageToVideo):
    """CogVideoX1.5-5B-I2V from its local Diffusers checkpoint."""

    height: int = 768
    width: int = 1360
    fps: int = 16
    name: str = "cogvideox1.5-5b-i2v"

    @staticmethod
    def _legal_num_frames(requested: int) -> int:
        adjusted = requested - ((requested - 1) % 16)
        if adjusted < 17:
            raise ValueError(f"cogvideox1.5-5b-i2v requires 16k+1 frames, got {requested}")
        return adjusted

    def _runner(self) -> str:
        return str(Path(__file__).resolve().parent / "runners" / "run_cogvideox_i2v.py")


@dataclass(kw_only=True)
class VBVRWan22(_DiffusersImageToVideo):
    """VBVR-Wan2.2 local Diffusers image-to-video checkpoint."""

    height: int = 480
    width: int = 832
    fps: int = 16
    name: str = "vbvr-wan2.2"

    def _runner(self) -> str:
        return str(Path(__file__).resolve().parent / "runners" / "run_vbvr_wan22.py")


@dataclass(kw_only=True)
class HunyuanVideo15I2V(_BaseSubprocessModel):
    """HunyuanVideo-1.5 I2V through the official source entry point.

    The main Hugging Face snapshot contains the VAE, scheduler, and all
    resolution-specific transformer weights. The official repository expects
    the text and vision encoders to be downloaded separately into this same
    model root; ``generate`` validates those components before launching.
    """

    proj: str
    model_dir: str
    python_bin: str | None = None
    devices: str = "0"
    distributed: bool = False
    nproc_per_node: int = 1
    resolution: str = "720p"
    steps: int = 50
    sr: bool = False
    rewrite: bool = False
    offloading: bool = True
    group_offloading: bool | None = None
    overlap_group_offloading: bool = True
    dtype: str = "bf16"
    aspect_ratio: str = "16:9"
    height: int | None = None
    width: int | None = None
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "hunyuan-video-1.5-i2v-720p"

    def _python(self) -> str:
        return self.python_bin or str(Path(self.proj) / ".venv" / "bin" / "python")

    def _script(self) -> str:
        return str(Path(__file__).resolve().parent / "runners" / "run_hunyuan_video15_i2v.py")

    def _env(self) -> dict[str, str]:
        env = super()._env()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        return env

    @staticmethod
    def _legal_num_frames(requested: int) -> int:
        adjusted = requested - ((requested - 1) % 4)
        if adjusted < 5:
            raise ValueError(f"hunyuan-video-1.5 requires video_length=4n+1, got {requested}")
        return adjusted

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 121, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        model_dir = str(_resolve_path(self.proj, self.model_dir).resolve())
        self._exists(self._python(), self._script(), first_frame, model_dir)
        required = [model_dir + "/vae", model_dir + "/scheduler",
                    model_dir + "/text_encoder/llm", model_dir + "/text_encoder/byt5-small",
                    model_dir + "/text_encoder/Glyph-SDXL-v2",
                    model_dir + "/vision_encoder/siglip",
                    model_dir + f"/transformer/{self.resolution}_i2v"]
        missing = [path for path in required if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "HunyuanVideo-1.5 checkpoint is missing local components: "
                + ", ".join(missing)
                + ". The official snapshot requires the encoders listed in "
                "checkpoints/source/HunyuanVideo-1.5/checkpoints-download.md "
                "to be placed under the model root."
            )
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        effective_num_frames = self._legal_num_frames(num_frames)
        cmd = [self._python(), self._script(), "--repo", str(Path(self.proj).resolve()),
               "--model-path", model_dir, "--image", first_frame, "--prompt", prompt,
               "--output", out, "--resolution", self.resolution,
               "--aspect-ratio", self.aspect_ratio, "--num-frames", str(effective_num_frames),
               "--steps", str(self.steps), "--seed", str(seed), "--sr", str(self.sr).lower(),
               "--rewrite", str(self.rewrite).lower(), "--offloading", str(self.offloading).lower(),
               "--dtype", self.dtype, "--overlap-group-offloading", str(self.overlap_group_offloading).lower()]
        if self.distributed:
            cmd += ["--distributed", "--nproc-per-node", str(self.nproc_per_node)]
        if self.group_offloading is not None:
            cmd += ["--group_offloading", str(self.group_offloading).lower()]
        rc, log, secs = self._run(cmd, log_path=log_path)
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(path=out, seconds=secs, cmd=shlex.join(cmd), log=log,
                         meta={"requested_num_frames": num_frames,
                               "effective_num_frames": effective_num_frames,
                               "resolution": self.resolution, "conditioning": "image+text"})


@dataclass(kw_only=True)
class LingBotMoE(_BaseSubprocessModel):
    python_bin: str | None = None
    proj: str
    model_dir: str
    processor_dir: str | None = None
    text_encoder_dir: str | None = None
    refiner_dir: str | None = None
    devices: str = "0"
    distributed: bool = False
    nproc_per_node: int = 1
    backend: str = "diffusers"
    engine: str | None = None
    mode: str = "ti2v"
    run_refiner: bool = True
    use_prompt_rewriter: bool = True
    prompt_rewriter_dir: str | None = None
    prompt_rewriter_base_model: str | None = None
    cfg_parallel_degree: int = 1
    context_parallel_degree: int = 1
    context_parallel_ulysses_anything: bool = False
    enable_fsdp_inference: bool = False
    enable_vlm_fsdp_inference: bool = False
    batch_cfg: bool = False
    refiner_batch_cfg: bool = False
    release_base_before_refiner: bool = False
    reuse_condition_features: bool = False
    allow_tf32: bool = True
    diffusers_attn_backend: str | None = None
    default_dtype: str | None = None
    transformer_dtype: str | None = None
    text_encoder_dtype: str | None = None
    vae_dtype: str | None = None
    vae_tiling: bool | None = None
    guidance_scale: float | None = None
    shift: float | None = None
    fps: int | None = None
    resolution: str | None = None
    ratio: str | None = None
    duration: float | None = None
    quiet_progress: bool = False
    disable_run_refiner: bool = False
    height: int | None = 768
    width: int | None = 1344
    steps: int | None = None
    refiner_steps: int | None = None
    refiner_guidance_scale: float | None = None
    refiner_shift: int | None = None
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "lingbot-video-moe-30b-a3b"

    def _python(self) -> str:
        if self.python_bin:
            return self.python_bin
        cand = Path(self.proj) / ".venv" / "bin" / "python"
        return str(cand if cand.exists() else Path(self.proj) / "envs" / "lingbot" / "bin" / "python")

    def _script(self) -> str:
        return str(Path(self.proj) / "scripts" / "inference.py")

    def _rewriter(self) -> str:
        return str(Path(self.proj) / "rewriter" / "inference.py")

    def _auto_negative(self) -> str:
        return str(Path(self.proj) / "rewriter" / "auto_negative.py")

    def _maybe_append(self, cmd: list[str], flag: str, value: object | None) -> None:
        if value is None:
            return
        cmd += [flag, str(value)]

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 124, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        script = self._script()
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        model_dir = str(_resolve_path(self.proj, self.model_dir).resolve())
        processor_dir = str(_resolve_path(self.proj, self.processor_dir or self.model_dir).resolve())
        text_encoder_dir = str(_resolve_path(self.proj, self.text_encoder_dir or processor_dir).resolve())
        refiner_dir = str(_resolve_path(self.proj, self.refiner_dir or model_dir).resolve())
        self._exists(self._python(), script, first_frame, model_dir, processor_dir, text_encoder_dir, refiner_dir, self._rewriter(), self._auto_negative())
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        work = Path(out).with_suffix("")
        work.mkdir(parents=True, exist_ok=True)
        prompt_json = work / "prompt.json"
        negative_json = work / "negative.json"
        base_out = work / "base.mp4"
        refiner_out = work / "refined.mp4"
        prompt_json.write_text(json.dumps({"caption": prompt, "image": first_frame, "seed": seed}, indent=2))

        if self.use_prompt_rewriter:
            rewriter_cmd = [self._python(), self._rewriter(), "--backend", "transformers",
                            "--mode", self.mode, "--prompt", prompt, "--duration", "5",
                            "--output", str(prompt_json)]
            if self.mode == "ti2v":
                rewriter_cmd += ["--first-frame", first_frame]
            if self.prompt_rewriter_dir:
                rewriter_cmd += ["--adapter", self.prompt_rewriter_dir]
            if self.prompt_rewriter_base_model:
                rewriter_cmd += ["--base", self.prompt_rewriter_base_model]
            rc, log1, secs1 = self._run(rewriter_cmd)
            if rc != 0:
                raise RuntimeError(f"{self.name} rewriter failed (rc={rc})\ncmd: {shlex.join(rewriter_cmd)}\n{log1[-4000:]}")
            neg_cmd = [self._python(), self._auto_negative(), "--backend", "transformers",
                       "--mode", self.mode, "--caption", str(prompt_json),
                       "--output", str(negative_json), "--base", self.prompt_rewriter_base_model or "",
                       "--adapter", self.prompt_rewriter_dir or ""]
            if self.mode == "ti2v":
                neg_cmd += ["--first-frame", first_frame]
            rc, log2, secs2 = self._run(neg_cmd)
            if rc != 0:
                raise RuntimeError(f"{self.name} auto_negative failed (rc={rc})\ncmd: {shlex.join(neg_cmd)}\n{log2[-4000:]}")
            log_prefix = log1 + "\n" + log2
            elapsed_prefix = secs1 + secs2
        else:
            prompt_json.write_text(json.dumps({"caption": prompt, "image": first_frame, "seed": seed}, indent=2))
            negative_json.write_text(json.dumps({}, indent=2))
            log_prefix = ""
            elapsed_prefix = 0.0

        infer_cmd = [script, "--backend", self.backend, "--model_dir", model_dir,
                     "--mode", self.mode, "--prompt_json", str(prompt_json),
                     "--negative_prompt_json", str(negative_json), "--output", str(base_out),
                     "--refiner_output", str(refiner_out)]
        if self.mode == "ti2v":
            infer_cmd += ["--image", first_frame]
        if self.run_refiner and not self.disable_run_refiner:
            infer_cmd.append("--run_refiner")
        if self.context_parallel_ulysses_anything:
            infer_cmd.append("--context_parallel_ulysses_anything")
        if self.enable_fsdp_inference:
            infer_cmd.append("--enable_fsdp_inference")
        if self.enable_vlm_fsdp_inference:
            infer_cmd.append("--enable_vlm_fsdp_inference")
        if self.vae_tiling is not None:
            infer_cmd.append("--vae_tiling" if self.vae_tiling else "--no-vae_tiling")
        if self.batch_cfg:
            infer_cmd.append("--batch_cfg")
        if self.refiner_batch_cfg:
            infer_cmd.append("--refiner_batch_cfg")
        if self.release_base_before_refiner:
            infer_cmd.append("--release_base_before_refiner")
        if self.reuse_condition_features:
            infer_cmd.append("--reuse_condition_features")
        if not self.allow_tf32:
            infer_cmd.append("--no-allow_tf32")
        if self.quiet_progress:
            infer_cmd.append("--quiet_progress")

        self._maybe_append(infer_cmd, "--engine", self.engine)
        self._maybe_append(infer_cmd, "--height", self.height)
        self._maybe_append(infer_cmd, "--width", self.width)
        self._maybe_append(infer_cmd, "--num_frames", num_frames)
        self._maybe_append(infer_cmd, "--steps", self.steps)
        self._maybe_append(infer_cmd, "--guidance_scale", self.guidance_scale)
        self._maybe_append(infer_cmd, "--shift", self.shift)
        self._maybe_append(infer_cmd, "--fps", self.fps)
        self._maybe_append(infer_cmd, "--duration", self.duration)
        self._maybe_append(infer_cmd, "--resolution", self.resolution)
        self._maybe_append(infer_cmd, "--ratio", self.ratio)
        self._maybe_append(infer_cmd, "--cfg_parallel_degree", self.cfg_parallel_degree if self.cfg_parallel_degree > 1 else None)
        self._maybe_append(infer_cmd, "--context_parallel_degree", self.context_parallel_degree if self.context_parallel_degree > 1 else None)
        self._maybe_append(infer_cmd, "--diffusers_attn_backend", self.diffusers_attn_backend)
        self._maybe_append(infer_cmd, "--default_dtype", self.default_dtype)
        self._maybe_append(infer_cmd, "--transformer_dtype", self.transformer_dtype)
        self._maybe_append(infer_cmd, "--text_encoder_dtype", self.text_encoder_dtype)
        self._maybe_append(infer_cmd, "--vae_dtype", self.vae_dtype)
        self._maybe_append(infer_cmd, "--refiner_steps", self.refiner_steps)
        self._maybe_append(infer_cmd, "--refiner_guidance_scale", self.refiner_guidance_scale)
        self._maybe_append(infer_cmd, "--refiner_shift", self.refiner_shift)
        self._maybe_append(infer_cmd, "--seed", seed)

        env = self._env()
        if self.mode == "ti2v":
            env.setdefault("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "sdpa")
        if self.distributed:
            cmd = [self._python(), "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={self.nproc_per_node}", *infer_cmd]
        else:
            cmd = [self._python(), *infer_cmd]

        orig_env = self._env
        self._env = lambda: {**orig_env(), **env}
        try:
            rc, log3, secs3 = self._run(cmd, log_path=log_path)
        finally:
            self._env = orig_env
        if rc != 0:
            # The elastic launcher summary can push the original worker error
            # out of a short tail, hiding the actual failure (e.g. VAE OOM).
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log3}")
        chosen = refiner_out if self.run_refiner and refiner_out.exists() else base_out
        if not chosen.exists():
            raise RuntimeError(f"{self.name} produced no output mp4\ncmd: {shlex.join(cmd)}\n{log3[-4000:]}")
        if chosen != Path(out):
            Path(out).write_bytes(chosen.read_bytes())
        return GenResult(path=str(out), seconds=elapsed_prefix + secs3, cmd=shlex.join(cmd), log=log_prefix + ("\n" if log_prefix else "") + log3)


@dataclass(kw_only=True)
class Cosmos3SuperImage2Video(_BaseSubprocessModel):
    """Cosmos3-Super-Image2Video through the local Diffusers checkpoint.

    Cosmos3's released I2V checkpoint is a Diffusers pipeline.  Keeping the
    pipeline in a subprocess is important here because its torch/diffusers
    stack is not compatible with the benchmark's numpy-only environment.
    """

    proj: str
    model_dir: str
    python_bin: str | None = None
    devices: str = "0,1,2,3"
    device_map: str = "transformer-balanced"
    gpu_memory_gib: int = 74
    attention_backend: str = "auto"
    steps: int = 50
    fps: int = 24
    height: int = 480
    width: int = 832
    guidance_scale: float = 6.0
    flow_shift: float = 5.0
    add_resolution_template: bool = False
    add_duration_template: bool = False
    enable_safety_checker: bool = True
    extra_env: dict[str, Any] = field(default_factory=dict)
    name: str = "cosmos3-super-image2video"

    def _python(self) -> str:
        return self.python_bin or str(Path(__file__).resolve().parents[1] / "envs" / "physbench" / "bin" / "python3.12")

    def _script(self) -> str:
        return str(Path(__file__).resolve().parent / "runners" / "run_cosmos3_i2v.py")

    def _env(self) -> dict[str, str]:
        env = super()._env()
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        return env

    @staticmethod
    def _legal_num_frames(requested: int) -> int:
        # Cosmos3 uses the same causal VAE cadence as Wan: 4k+1 frames.
        adjusted = requested - ((requested - 1) % 4)
        if adjusted < 5:
            raise ValueError(f"cosmos3-super-image2video requires at least 5 frames, got {requested}")
        return adjusted

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int = 189, seed: int = 42,
                 log_path: str | None = None) -> GenResult:
        script = self._script()
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        model_dir = str(_resolve_path(self.proj, self.model_dir).resolve())
        self._exists(self._python(), script, first_frame, model_dir)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        effective_num_frames = self._legal_num_frames(num_frames)
        cmd = [self._python(), script, "--model-path", model_dir,
               "--image", first_frame, "--prompt", prompt, "--output", out,
               "--num-frames", str(effective_num_frames), "--height", str(self.height),
               "--width", str(self.width), "--fps", str(self.fps), "--seed", str(seed),
               "--device-map", self.device_map, "--steps", str(self.steps),
               "--gpu-memory-gib", str(self.gpu_memory_gib),
               "--attention-backend", self.attention_backend,
               "--guidance-scale", str(self.guidance_scale),
               "--flow-shift", str(self.flow_shift)]
        if self.add_resolution_template:
            cmd.append("--add-resolution-template")
        if self.add_duration_template:
            cmd.append("--add-duration-template")
        if not self.enable_safety_checker:
            cmd.append("--no-safety-checker")

        rc, log, secs = self._run(cmd, log_path=log_path)
        if rc != 0 or not Path(out).exists():
            raise RuntimeError(f"{self.name} failed (rc={rc})\ncmd: {shlex.join(cmd)}\n{log[-4000:]}")
        return GenResult(path=str(out), seconds=secs, cmd=shlex.join(cmd), log=log,
                         meta={"requested_num_frames": num_frames,
                               "effective_num_frames": effective_num_frames,
                               "height": self.height, "width": self.width, "fps": self.fps,
                               "steps": self.steps, "device_map": self.device_map,
                               "devices": self.devices, "parallelism": "decoder-layer-sharding",
                               "attention_backend": self.attention_backend,
                               "cpu_offload": False})


@dataclass(kw_only=True)
class OpenAIStyleVideo(VideoModel):
    model: str
    base_url: str = "https://api.openai.com"
    api_key_env: str = "OPENAI_API_KEY"
    create_path: str = "/v1/responses"
    status_path_template: str = "/v1/responses/{id}"
    download_path_template: str | None = "/v1/files/{file_id}/content"
    prompt_field: str = "input"
    image_field: str = "image"
    image_transport: str = "data_url"
    request_format: str = "json"
    seconds: int = 5
    response_input_role: str = "user"
    response_image_detail: str = "high"
    response_text_hint: str = "Generate a video from this first frame and prompt."
    result_url_field: str = "output[0].url"
    result_file_id_field: str = "output[0].file_id"
    status_field: str = "status"
    terminal_success: tuple[str, ...] = ("succeeded", "completed", "done")
    terminal_failure: tuple[str, ...] = ("failed", "error", "cancelled", "canceled", "rejected", "expired")
    poll_interval_s: float = 5.0
    timeout_s: float = 60.0
    max_wait_s: float = 1800.0
    failed_job_retries: int = 0
    failed_job_retry_wait_s: float = 60.0
    poll_retries: int = 2
    verify_tls: bool = True
    fps: int | None = None
    size: str | None = None
    quality: str | None = None
    background: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    name: str = "openai-video"

    def _api_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.name} missing API key env var {self.api_key_env}")
        return key

    def _join_url(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        path = path.lstrip("/")
        if base.endswith("/v1") and path.startswith("v1/"):
            path = path[3:]
        return base + "/" + path

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Accept": "application/json",
        }
        headers.update({str(k): str(v) for k, v in self.headers.items()})
        return headers

    def _json_headers(self) -> dict[str, str]:
        return {**self._auth_headers(), "Content-Type": "application/json"}

    def _encode_first_frame(self, first_frame: str) -> tuple[str, str]:
        suffix = Path(first_frame).suffix.lower().lstrip(".") or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        data = base64.b64encode(Path(first_frame).read_bytes()).decode("ascii")
        return mime, data

    def _image_payload(self, first_frame: str) -> Any:
        mime, data = self._encode_first_frame(first_frame)
        if self.image_transport == "base64":
            return data
        if self.image_transport == "data_url":
            return f"data:image/{mime};base64,{data}"
        raise ValueError(f"{self.name} unsupported image_transport={self.image_transport!r}")

    def _response_api_body(self, *, first_frame: str, prompt: str,
                           num_frames: int, seed: int) -> dict[str, Any]:
        text = self.response_text_hint if self.response_text_hint else prompt
        body = {
            "model": self.model,
            self.prompt_field: [{
                "role": self.response_input_role,
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", self.image_field: self._image_payload(first_frame), "detail": self.response_image_detail},
                ],
            }],
        }
        if num_frames:
            body.setdefault("metadata", {})["requested_num_frames"] = num_frames
        if seed is not None:
            body["seed"] = seed
        if self.fps is not None:
            body["fps"] = self.fps
        if self.size is not None:
            body["size"] = self.size
        if self.quality is not None:
            body["quality"] = self.quality
        if self.background is not None:
            body["background"] = self.background
        body.update(self.extra_body)
        return body

    def _legacy_video_body(self, *, first_frame: str, prompt: str,
                           num_frames: int, seed: int) -> dict[str, Any]:
        body = {
            "model": self.model,
            "prompt": prompt,
            self.image_field: self._image_payload(first_frame),
        }
        if num_frames:
            body["num_frames"] = num_frames
        if seed is not None:
            body["seed"] = seed
        if self.fps is not None:
            body["fps"] = self.fps
        if self.size is not None:
            body["size"] = self.size
        if self.quality is not None:
            body["quality"] = self.quality
        if self.background is not None:
            body["background"] = self.background
        body.update(self.extra_body)
        return body

    def _request_body(self, *, first_frame: str, prompt: str,
                      num_frames: int, seed: int) -> dict[str, Any]:
        if self.request_format == "litellm_json":
            # Provider-specific JSON belongs in extra_body on this video route.
            # Keep the exact first frame as a data URL rather than a binary upload.
            if self.seconds <= 0:
                raise ValueError("video duration seconds must be positive")
            return {
                "model": self.model, "prompt": prompt,
                "extra_body": {
                    "duration": self.seconds, "aspect_ratio": "adaptive",
                    "frame_images": [{"frame_type": "first_frame",
                                      "image_url": {"url": self._image_payload(first_frame)}}],
                },
            }
        if self.request_format == "aihubmix_json":
            if self.seconds <= 0:
                raise ValueError("video duration seconds must be positive")
            # AiHubMix's JSON contract. A gateway must preserve this format;
            # LiteLLM's OpenAI video route may re-encode it as multipart.
            mime, image = self._encode_first_frame(first_frame)
            body = {
                "model": self.model,
                "prompt": prompt,
                "duration": self.seconds,
                "aspect_ratio": "adaptive",
                "frame_images": [{
                    "frame_type": "first_frame",
                    "image_url": {"url": f"data:image/{mime};base64,{image}"},
                }],
            }
            body.update(self.extra_body)
            return body
        if self.request_format == "multipart":
            if self.seconds <= 0:
                raise ValueError("video duration seconds must be positive")
            # The Videos API uses seconds and an uploaded input_reference. GPU
            # settings, frame counts and seeds are not standard request fields.
            body = {"model": self.model, "prompt": prompt, "seconds": str(self.seconds)}
            if self.size is not None:
                body["size"] = self.size
            body.update(self.extra_body)
            return body
        if self.request_format != "json":
            raise ValueError(f"unsupported request_format={self.request_format!r}")
        if self.create_path.rstrip("/").endswith("/responses"):
            return self._response_api_body(first_frame=first_frame, prompt=prompt,
                                           num_frames=num_frames, seed=seed)
        return self._legacy_video_body(first_frame=first_frame, prompt=prompt,
                                       num_frames=num_frames, seed=seed)

    def _extract_field(self, payload: Any, dotted: str) -> Any:
        cur = payload
        for token in dotted.split("."):
            if "[" in token and token.endswith("]"):
                key, idx_s = token[:-1].split("[", 1)
                if key:
                    if not isinstance(cur, dict) or key not in cur:
                        raise KeyError(key)
                    cur = cur[key]
                if not isinstance(cur, list):
                    raise KeyError(token)
                cur = cur[int(idx_s)]
                continue
            if isinstance(cur, dict):
                if token not in cur:
                    raise KeyError(token)
                cur = cur[token]
                continue
            if isinstance(cur, list):
                cur = cur[int(token)]
                continue
            raise KeyError(token)
        return cur

    def _response_json(self, resp: requests.Response) -> dict[str, Any]:
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(f"{self.name} expected JSON response from {resp.url}, got non-JSON body") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{self.name} expected JSON object from {resp.url}, got {type(data).__name__}")
        return data

    def _summarize(self, payload: dict[str, Any]) -> str:
        keep = {k: payload.get(k) for k in ("id", self.status_field, "model", "error") if k in payload}
        return json.dumps(keep, ensure_ascii=False, sort_keys=True)

    def _create_job(self, *, first_frame: str, prompt: str,
                    num_frames: int, seed: int) -> tuple[dict[str, Any], str, str]:
        url = self._join_url(self.create_path)
        body = self._request_body(first_frame=first_frame, prompt=prompt,
                                  num_frames=num_frames, seed=seed)
        if self.request_format == "multipart":
            form = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    for k, v in body.items()}
            mime = mimetypes.guess_type(first_frame)[0] or "application/octet-stream"
            with open(first_frame, "rb") as image:
                # requests supplies Content-Type with the multipart boundary.
                resp = requests.post(
                    url, headers=self._auth_headers(), data=form,
                    files={self.image_field: (Path(first_frame).name, image, mime)},
                    timeout=self.timeout_s, verify=self.verify_tls,
                )
        else:
            resp = requests.post(url, headers=self._json_headers(), json=body,
                                 timeout=self.timeout_s, verify=self.verify_tls)
        if resp.status_code >= 400:
            message = f"{self.name} create failed ({resp.status_code}) at {url}: {resp.text[:1000]}"
            if (self.request_format == "aihubmix_json" and resp.status_code == 400
                    and "request body must be a valid json object" in resp.text.lower()):
                raise VideoAPIConfigurationError(
                    message + "\nThe client sent application/json with duration and frame_images. "
                    "Check whether the gateway converts it to multipart. Use a confirmed JSON "
                    "video route in generation.local.json, or fix the gateway adapter. "
                    "Changing only Content-Type or retrying samples will not fix this protocol error."
                )
            raise RuntimeError(message)
        data = self._response_json(resp)
        job_id = str(data.get("id") or data.get("job_id") or "")
        if not job_id:
            raise RuntimeError(f"{self.name} create response missing job id: {self._summarize(data)}")
        log = [
            f"create_url={url}",
            f"create_status={resp.status_code}",
            f"create_response={self._summarize(data)}",
        ]
        return data, job_id, "\n".join(log)

    def _raise_job_failure(self, job_id: str, data: dict[str, Any]) -> None:
        status = str(data.get(self.status_field) or data.get("state") or "unknown")
        raise VideoRemoteJobError(
            f"{self.name} remote job {job_id} ended with status={status}: {self._summarize(data)}",
            status=status, error=data.get("error"),
        )

    def _poll_job(self, job_id: str, *, on_update: Callable[[str], None] | None = None
                  ) -> tuple[dict[str, Any], str]:
        deadline = time.monotonic() + self.max_wait_s
        logs: list[str] = []
        ok = {s.lower() for s in self.terminal_success}
        bad = {s.lower() for s in self.terminal_failure}
        failures = 0
        last_status = None
        last_print = 0.0

        def record(line: str) -> None:
            logs.append(line)
            if on_update:
                on_update(line)

        while True:
            url = self._join_url(self.status_path_template.format(id=quote(job_id, safe="")))
            try:
                resp = requests.get(url, headers=self._auth_headers(), timeout=self.timeout_s,
                                    verify=self.verify_tls)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    raise requests.HTTPError(f"poll HTTP {resp.status_code}: {resp.text[:1000]}")
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                record(f"poll_transient_error job_id={job_id} error={exc}")
                if failures >= self.poll_retries or time.monotonic() >= deadline:
                    raise
                failures += 1
                delay = min(self.poll_interval_s * 2 ** (failures - 1), 60.0,
                            max(0.0, deadline - time.monotonic()))
                print(f"    {self.name}: poll interrupted; retry {failures}/{self.poll_retries} "
                      f"in {delay:g}s using the same job_id={job_id}", flush=True)
                time.sleep(delay)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} poll failed ({resp.status_code}) at {url}: {resp.text[:1000]}")
            failures = 0
            data = self._response_json(resp)
            status = str(data.get(self.status_field) or data.get("state") or "unknown")
            record(f"poll status={status} response={self._summarize(data)}")
            now = time.monotonic()
            if status != last_status or now - last_print >= 60:
                print(f"    {self.name}: status={status} job_id={job_id}", flush=True)
                last_status, last_print = status, now
            low = status.lower()
            if low in ok:
                return data, "\n".join(logs)
            if low in bad:
                self._raise_job_failure(job_id, data)
            if time.monotonic() >= deadline:
                raise RuntimeError(f"{self.name} timed out waiting for remote job {job_id}; last={self._summarize(data)}")
            time.sleep(self.poll_interval_s)

    def _download_via_file_id(self, file_id: str, out: str) -> tuple[int, str]:
        if not self.download_path_template:
            raise RuntimeError(f"{self.name} download_path_template is required for file-id downloads")
        url = self._join_url(self.download_path_template.format(file_id=quote(file_id, safe="")))
        partial: Path | None = None
        try:
            with requests.get(url, headers=self._auth_headers(), timeout=self.timeout_s,
                              verify=self.verify_tls, stream=True) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"{self.name} file download failed ({resp.status_code}) from {url}: {resp.text[:1000]}")
                total = 0
                with tempfile.NamedTemporaryFile(dir=Path(out).parent, suffix=".part", delete=False) as fh:
                    partial = Path(fh.name)
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            total += len(chunk)
                if total <= 0:
                    raise RuntimeError(f"{self.name} downloaded empty output from {url}")
                # Some gateways wrap upstream error bodies in HTTP 200 and even
                # label them video/mp4. Do not save such a response as a video.
                with partial.open("rb") as fh:
                    prefix = fh.read(256).lstrip()
                if prefix.startswith((b"{", b"[", b"<")):
                    raise RuntimeError(f"{self.name} download returned JSON/HTML instead of video from {url}")
                partial.replace(out)
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)
        return total, url

    def _download_result(self, payload: dict[str, Any], out: str) -> tuple[int, str]:
        if self.request_format in {"multipart", "aihubmix_json", "litellm_json"}:
            job_id = payload.get("id") or payload.get("job_id")
            if not job_id:
                raise RuntimeError(f"{self.name} video response missing job id")
            return self._download_via_file_id(str(job_id), out)
        try:
            url = str(self._extract_field(payload, self.result_url_field))
        except Exception:  # noqa: BLE001
            url = ""
        if url:
            resp = requests.get(url, headers=self._auth_headers(), timeout=self.timeout_s,
                                verify=self.verify_tls, stream=True)
            if resp.status_code >= 400:
                raise RuntimeError(f"{self.name} download failed ({resp.status_code}) from {url}: {resp.text[:1000]}")
            total = 0
            with open(out, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    total += len(chunk)
            if total <= 0 or not Path(out).exists():
                raise RuntimeError(f"{self.name} downloaded empty output from {url}")
            return total, url
        try:
            file_id = str(self._extract_field(payload, self.result_file_id_field))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"{self.name} could not find result url via {self.result_url_field!r} "
                f"or file id via {self.result_file_id_field!r}: {self._summarize(payload)}"
            ) from exc
        return self._download_via_file_id(file_id, out)

    def generate(self, first_frame: str, prompt: str, out: str, *,
                 num_frames: int, seed: int,
                 log_path: str | None = None) -> GenResult:
        first_frame = str(Path(first_frame).resolve())
        out = str(Path(out).resolve())
        if not Path(first_frame).exists():
            raise FileNotFoundError(first_frame)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

        if self.failed_job_retries < 0 or self.failed_job_retry_wait_s < 0 or self.poll_retries < 0:
            raise ValueError("API retry counts and delay must be non-negative")
        t0 = time.time()
        # Keep earlier task IDs when the user reruns the same sample.
        history = Path(log_path).read_text() if log_path and Path(log_path).exists() else ""
        logs: list[str] = [history.rstrip()] if history else []

        def record(line: str) -> None:
            logs.append(line)
            if log_path:
                with Path(log_path).open("a") as fh:
                    fh.write(line + "\n")

        record(f"\ngeneration_start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.failed_job_retries + 1):
            record(f"attempt={attempt + 1}")
            # Never retry creation exceptions: a timeout can hide an accepted job.
            create_data, job_id, create_log = self._create_job(
                first_frame=first_frame, prompt=prompt, num_frames=num_frames, seed=seed,
            )
            record(f"{create_log}\njob_id={job_id}")
            print(f"    {self.name}: created attempt={attempt + 1} job_id={job_id}", flush=True)
            entry: dict[str, Any] = {"attempt": attempt + 1, "job_id": job_id}
            attempts.append(entry)
            try:
                status = str(create_data.get(self.status_field) or create_data.get("state") or "").lower()
                if status in {s.lower() for s in self.terminal_failure}:
                    self._raise_job_failure(job_id, create_data)
                if status in {s.lower() for s in self.terminal_success}:
                    final_data = create_data
                else:
                    final_data, _ = self._poll_job(job_id, on_update=record)
                entry["status"] = str(final_data.get(self.status_field) or final_data.get("state"))
                break
            except VideoRemoteJobError as exc:
                entry.update(status=exc.status, error=exc.error)
                record(f"terminal_failure={json.dumps(entry, ensure_ascii=False)}")
                if (exc.status != "failed" or exc.code != "upstream_unreachable"
                        or attempt >= self.failed_job_retries):
                    raise
                delay = min(self.failed_job_retry_wait_s * 2 ** attempt, 300.0)
                message = (f"confirmed failed job: upstream_unreachable; retrying generation "
                           f"{attempt + 1}/{self.failed_job_retries} in {delay:g}s")
                record(message)
                print(f"    {self.name}: {message}", flush=True)
                time.sleep(delay)
        videos_api = self.request_format in {"multipart", "aihubmix_json", "litellm_json"}
        if videos_api:
            # LiteLLM may return a re-encoded ID without model routing data when
            # polling. Always download using the original ID from creation.
            final_data["id"] = job_id
        else:
            final_data.setdefault("id", job_id)
        output_bytes, result_url = self._download_result(final_data, out)
        final_status = str(final_data.get(self.status_field) or final_data.get("state") or "completed")
        record(f"download_source={result_url}\noutput_bytes={output_bytes}")
        log = "\n".join(logs)
        effective_num_frames = final_data.get("num_frames") or final_data.get("frames")
        if effective_num_frames is not None:
            effective_num_frames = int(effective_num_frames)
        elif not videos_api:
            effective_num_frames = num_frames
        generation_args = f"seconds={self.seconds}" if videos_api else f"frames={num_frames} seed={seed}"
        return GenResult(
            path=out,
            seconds=time.time() - t0,
            cmd=f"POST {self._join_url(self.create_path)} model={self.model} {generation_args}",
            log=log,
            meta={
                "provider": "openai-compatible",
                "remote_model": self.model,
                "job_id": job_id,
                "attempts": attempts,
                "status": final_status,
                "requested_num_frames": None if videos_api else num_frames,
                "effective_num_frames": effective_num_frames,
                "base_url": self.base_url,
                "create_path": self.create_path,
                "output_bytes": output_bytes,
                "create_response": {k: v for k, v in create_data.items() if k in {"id", self.status_field, "model"}},
                **({"requested_seconds": self.seconds,
                    "seconds": final_data.get("duration") or final_data.get("seconds") or self.seconds,
                    "seed_sent": False} if videos_api else {}),
            },
        )


@dataclass(kw_only=True)
class Seedance20Video(OpenAIStyleVideo):
    """Seedance 2.0 via LiteLLM and AiHubMix's JSON video API."""

    model: str = "doubao-seedance-2-0-260128"
    base_url: str = ""
    create_path: str = "/v1/videos"
    status_path_template: str = "/v1/videos/{id}"
    download_path_template: str | None = "/v1/videos/{file_id}/content"
    image_field: str = "input_reference"
    request_format: str = "aihubmix_json"
    name: str = "seedance2.0"


@dataclass(kw_only=True)
class Seedance25Video(Seedance20Video):
    """Seedance 2.5 using the same AiHubMix JSON video contract."""

    model: str = "doubao-seedance-2-5-260628"
    name: str = "seedance2.5"


REGISTRY = {
    "minimax-h3": MiniMaxH3,
    "wan2.2-i2v-a14b": Wan22I2V,
    "cogvideox1.5-5b-i2v": CogVideoX15I2V,
    "vbvr-wan2.2": VBVRWan22,
    "hunyuan-video-1.5-i2v": HunyuanVideo15I2V,
    "lingbot-video-moe-30b-a3b": LingBotMoE,
    "cosmos3-super-image2video": Cosmos3SuperImage2Video,
    "seedance-2.5": Seedance25Video,
}
