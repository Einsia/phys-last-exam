"""Evaluators for VDM-Bench Group 1 / 4 / 6.

One measurement path per physical quantity, metrics exactly as listed in
``题包Benchmark.md``, and one readable debug figure per sample.
"""
from .context import Context
from .registry import get_evaluator
from .schema import Metric, Result
from .video import Clip, read_clip

__all__ = ["Context", "Metric", "Result", "Clip", "read_clip", "get_evaluator"]
