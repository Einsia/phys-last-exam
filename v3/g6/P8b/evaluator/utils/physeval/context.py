"""What a task needs to know about the sample it is evaluating."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Context:
    task_id: str
    video_path: str
    image_path: str
    video_prompt: str
    model: str = "minimax-h3"
    seed: int | None = None
    debug_path: str | None = None
    # Angle-like scene constants a task cannot measure from pixels and that the
    # benchmark states in the prompt (e.g. P2's launch angle).
    params: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.params is None:
            self.params = {}

    def debug(self, suffix: str = ".png") -> str | None:
        if not self.debug_path:
            return None
        p = Path(self.debug_path)
        return str(p.with_suffix(suffix))
