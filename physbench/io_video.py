"""Frame decoding. PyAV first, OpenCV as fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Clip:
    """Decoded frames plus what little metadata the container gives us.

    `fps` is recorded but never used in any residual -- the whole point is that the
    true seconds-per-frame is unknown, so metrics use frame indices only.
    """

    frames: np.ndarray  # (n, H, W, 3) uint8 RGB
    fps: float
    path: str

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def height(self) -> int:
        return self.frames.shape[1]

    @property
    def width(self) -> int:
        return self.frames.shape[2]

    @property
    def diag(self) -> float:
        return float(np.hypot(self.width, self.height))


def read_clip(path: str | Path, max_frames: int | None = None) -> Clip:
    path = str(path)
    try:
        return _read_av(path, max_frames)
    except Exception as exc:  # noqa: BLE001 - fall back on any decode trouble
        print(f"[io_video] PyAV failed on {path} ({type(exc).__name__}: {exc}); trying OpenCV")
        return _read_cv2(path, max_frames)


def _read_av(path: str, max_frames: int | None) -> Clip:
    import av

    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if max_frames and len(frames) >= max_frames:
                break
    if not frames:
        raise RuntimeError("no video frames decoded")
    return Clip(frames=np.stack(frames), fps=fps, path=path)


def _read_cv2(path: str, max_frames: int | None) -> Clip:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr[:, :, ::-1].copy())
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError("no video frames decoded")
    return Clip(frames=np.stack(frames), fps=fps, path=path)


def write_clip(path: str | Path, frames: np.ndarray, fps: float = 24.0) -> str:
    """Write RGB uint8 frames to mp4 (yuv420p, even dims required by libx264)."""
    import imageio.v2 as imageio

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    if h % 2 or w % 2:
        frames = frames[: h - h % 2, : w - w % 2]
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                                macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    return path
