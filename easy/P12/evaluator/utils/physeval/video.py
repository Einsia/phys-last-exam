"""Clip loading. Frames are held in memory: 124 frames at 1344x768 is ~370 MB
as uint8, which is fine for one clip at a time and lets every task index frames
freely instead of re-decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Clip:
    frames: list[np.ndarray]
    fps: float
    path: str

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def h(self) -> int:
        return self.frames[0].shape[0]

    @property
    def w(self) -> int:
        return self.frames[0].shape[1]

    @property
    def diag(self) -> float:
        return float(np.hypot(self.w, self.h))

    def __getitem__(self, i: int) -> np.ndarray:
        return self.frames[i]

    def gray(self, i: int) -> np.ndarray:
        return cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)

    def hsv(self, i: int) -> np.ndarray:
        return cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2HSV)


def read_clip(path: str | Path) -> Clip:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: list[np.ndarray] = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    if not frames:
        raise IOError(f"no frames decoded from {path}")
    return Clip(frames=frames, fps=float(fps), path=str(path))
