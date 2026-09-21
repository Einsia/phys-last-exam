#!/usr/bin/env python3
"""Task-package entry point for the standalone Blender first-frame renderer."""

from __future__ import annotations

import sys
import types
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
TASK_ID = "P19"

# The shared renderer is written as the ``simulation`` Blender package.  Keep
# the package flat in this delivery folder while making ``render.py`` runnable
# directly with Blender's Python interpreter.
simulation = types.ModuleType("simulation")
simulation.__path__ = [str(CODE_DIR)]
simulation.__package__ = "simulation"
sys.modules["simulation"] = simulation

from simulation.render_all import main as _render_main  # noqa: E402


def main() -> int:
    """Render this task with package-local defaults unless explicitly overridden."""
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    if "--tasks" not in argv:
        argv += ["--tasks", str(CODE_DIR / "tasks.json")]
    if "--task" not in argv:
        argv += ["--task", TASK_ID]
    if "--output-root" not in argv:
        argv += ["--output-root", str(CODE_DIR.parent)]
    if "--no-reuse-pilot" not in argv:
        argv += ["--no-reuse-pilot"]
    sys.argv = [sys.argv[0], *argv]
    return _render_main()


if __name__ == "__main__":
    raise SystemExit(main())
