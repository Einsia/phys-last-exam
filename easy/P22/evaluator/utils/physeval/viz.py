"""Debug figure helpers.

Every sample gets one figure whose purpose is that a reader can recover the
measured physical quantity from it without trusting the code: the extracted
feature is drawn on the actual video frame, and the series the metric is computed
from is plotted beside it with its fit.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.titlesize": 10, "axes.grid": True,
                     "grid.alpha": 0.25})


def figure(ncols: int = 2, height: float = 4.2, width_each: float = 5.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width_each * ncols, height))
    if ncols == 1:
        axes = [axes]
    return fig, list(np.ravel(axes))


def show_frame(ax, frame_bgr: np.ndarray, title: str = "") -> None:
    ax.imshow(frame_bgr[..., ::-1])
    ax.set_title(title)
    ax.set_axis_off()
    ax.grid(False)


def save(fig, path: str | Path, caption: str) -> str:
    """Write the figure with the measurement summary printed underneath it."""
    fig.suptitle(caption, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return str(p)
